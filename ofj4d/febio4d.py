"""FEBio backend for one staging solve: OFJ_SOLVER=febio.  EXPERIMENTAL, being verified
against solve4d (docs/FINDINGS.md section 31).

Drop-in replacement for solve4d.solve_config.  Each call writes the CURRENT
configuration (actual tooth positions, seated aligner) as a FEBio 4 model, runs FEBio
with the NVIDIA cuDSS linear solver on the GPU (pipeline/febio_cudss, falls back to
PARDISO), and returns the tooth rigid motions in the form run4d.py expects.

What FEBio solves, against the custom solver (solve4d):
  * finite-deformation kinematics everywhere (custom: small strain per solve);
  * aligner neo-Hookean E 1500 / nu 0.30 tet4 (custom: linear elastic tet4);
  * PDL `pdl bilinear` (pipeline/febio_pdl): the same bilinear secant law on the
    equivalent Green-Lagrange strain, neo-Hookean kinematics;
  * teeth + attachments rigid bodies (crown/attachment surfaces as rigid shells, the
    inner PDL layer tied to them), outer PDL layer fixed -- as custom;
  * aligner-crown contact: solve4d's own law as an FEBio constraint (pipeline/febio_pairs):
    node-to-surface penalty K_N, secant Coulomb friction K_T / mu, pairing and normals fixed per solve;
  * the manufactured standoff is solve4d's reference gref = gap_now - gap_pre per pair, ramped
    in from exactly passive; the aligner mesh is never moved;
  * the 150 gf elastic as nodal forces on the precision-cut nodes, direction from the
    current hook position; midsagittal symmetry as a zero normal displacement (the
    model is written in a frame whose x axis is the midsagittal normal).

Environment:
  OFJ_FEBIO            FEBio launcher (default ~/febio/febio.sh)
  OFJ_FEBIO_LINSOLVE   cudss | pardiso      (default cudss when the plug-in is built)
  OFJ_FEBIO_STEPS      load steps per solve (default 10, adaptive; interference and elastic ramp together)
  OFJ_FEBIO_WORK       working directory    (default build4d/febio_run)
  OFJ_FEBIO_KEEP=1     keep every model / log (default: overwritten per solve)
"""
import os, sys, re, time, subprocess
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spec
from solve4d import CLEARANCE, K_N, K_T, Operators, pdl_secant_modulus, vm_equiv_strain

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEBIO = os.environ.get('OFJ_FEBIO', os.path.expanduser('~/febio/febio.sh'))
PDL_PLUGIN = os.environ.get('OFJ_FEBIO_PDL', os.path.join(ROOT, 'pipeline/febio_pdl/build/libFEBioPDL.so'))
PAIRS_PLUGIN = os.environ.get('OFJ_FEBIO_PAIRS', os.path.join(ROOT, 'pipeline/febio_pairs/build/libFEBioAlignerPairs.so'))
CUDSS_PLUGIN = os.environ.get('OFJ_FEBIO_CUDSS', os.path.join(ROOT, 'pipeline/febio_cudss/build/libFEBioCuDSS.so'))
LINSOLVE = os.environ.get('OFJ_FEBIO_LINSOLVE', 'cudss' if os.path.exists(CUDSS_PLUGIN) else 'pardiso')
STEPS = int(os.environ.get('OFJ_FEBIO_STEPS', 10))
WORK = os.environ.get('OFJ_FEBIO_WORK', os.path.join(ROOT, 'build4d/febio_run'))
KEEP = os.environ.get('OFJ_FEBIO_KEEP', '0') == '1'
CUDSS_DIR = os.environ.get('OFJ_CUDSS_DIR', os.path.expanduser('~/.pylibs_cudss'))
_NV = os.path.join(os.path.dirname(np.__file__), '..', 'nvidia')        # torch's CUDA 12 runtime
_count = [0]


def _frame(S):
    """Rotation whose first row is the midsagittal normal, and a point on the plane."""
    n = S.n_sag/np.linalg.norm(S.n_sag)
    a = np.array([0, 0, 1.0]) if abs(n[2]) < 0.9 else np.array([0, 1.0, 0])
    e2 = np.cross(a, n); e2 /= np.linalg.norm(e2)
    return np.stack([n, e2, np.cross(n, e2)]), n*S.sag_offset


def _fmt(v):
    return ','.join(f'{x:.10g}' for x in v)


def _rotvec(R):
    th = np.arccos(np.clip((np.trace(R) - 1)/2, -1, 1))
    if th < 1e-14:
        return np.zeros(3)
    return th/(2*np.sin(th))*np.array([R[2, 1]-R[1, 2], R[0, 2]-R[2, 0], R[1, 0]-R[0, 1]])


def _kabsch(X, Y):
    cx, cy = X.mean(0), Y.mean(0)
    U, _, Vt = np.linalg.svd((X - cx).T @ (Y - cy))
    s = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1, 1, s]) @ U.T, cx, cy


def _oriented_inner_faces(tets, faces, X):
    """Aligner inner faces ordered so the normal points out of the aligner solid."""
    opp = {}
    for c, o in (([0, 1, 2], 3), ([0, 1, 3], 2), ([0, 2, 3], 1), ([1, 2, 3], 0)):
        for f, p in zip(np.sort(tets[:, c], 1), tets[:, o]):
            opp[tuple(f)] = p
    F = faces.copy()
    o = np.array([opp[tuple(f)] for f in np.sort(F, 1)])
    n = np.cross(X[F[:, 1]] - X[F[:, 0]], X[F[:, 2]] - X[F[:, 0]])
    inward = np.einsum('ij,ij->i', n, X[o] - X[F].mean(1)) > 0
    F[inward] = F[inward][:, [0, 2, 1]]
    return F


def _positive(tets, X):
    P = X[tets]
    v = np.einsum('ij,ij->i', np.cross(P[:, 1]-P[:, 0], P[:, 2]-P[:, 0]), P[:, 3]-P[:, 0])
    t = tets.copy(); t[v < 0] = t[v < 0][:, [0, 1, 3, 2]]
    return t


def _write(path, S, state, load, nodes, mu, R, p0, pairs_path):
    """The FEBio model of this configuration, in the symmetry frame."""
    # Full Newton (max_ups 0): BFGS updates on this non-smooth contact problem build a wrong
    # approximate stiffness and the next increment inverts elements -- with them every variant
    # stopped at 20 % of step 1, with full Newton the step proceeds (docs/FINDINGS.md section 31).
    # Convergence on energy and residual (1e-3 relative).  A displacement criterion is not used:
    # marginal contact pairs toggle between iterations (the active-set hunting solve4d also shows,
    # docs/FINDINGS.md section 9) and hold a global displacement norm above any tight tolerance
    # after the energy has converged (section 31).
    X = (nodes - p0) @ R.T
    L = ['<?xml version="1.0" encoding="ISO-8859-1"?>', '<febio_spec version="4.0">', '<Module type="solid"/>',
         '<Control>', '<analysis>STATIC</analysis>',
         f'<time_steps>{STEPS}</time_steps>', f'<step_size>{1.0/STEPS:.8g}</step_size>',
         '<plot_level>PLOT_NEVER</plot_level>', '<output_level>OUTPUT_FINAL</output_level>',
         '<time_stepper type="default">', '<max_retries>10</max_retries>', '<opt_iter>12</opt_iter>',
         f'<dtmin>{1e-3/STEPS:.6g}</dtmin>', f'<dtmax>{1.0/STEPS:.6g}</dtmax>', '</time_stepper>',
         '<solver type="solid">', '<dtol>0</dtol>', '<etol>1e-3</etol>', '<rtol>1e-3</rtol>',
         '<lstol>0.9</lstol>', '<max_refs>30</max_refs>',
         '<qn_method type="BFGS">', '<max_ups>0</max_ups>', '</qn_method>',
         '</solver>', '</Control>', '<Material>',
         f'<material id="1" name="CA" type="neo-Hookean"><density>1</density>'
         f'<E>{spec.E_ALIGNER}</E><v>{spec.NU_ALIGNER}</v></material>',
         f'<material id="2" name="PDL" type="pdl bilinear"><density>1</density><E1>{spec.PDL_E1}</E1>'
         f'<E2>{spec.PDL_E2}</E2><eps_knee>{spec.PDL_EPS_KNEE}</eps_knee><v>{spec.PDL_NU}</v></material>']
    for k, u in enumerate(S.keep):
        c = (state.centroid(u) - p0) @ R.T
        L.append(f'<material id="{k+3}" name="rb{u}" type="rigid body"><density>1</density>'
                 f'<center_of_mass>{_fmt(c)}</center_of_mass></material>')
    L += ['</Material>', '<Mesh>', '<Nodes name="all">']
    L += [f'<node id="{i+1}">{x[0]:.10g},{x[1]:.10g},{x[2]:.10g}</node>' for i, x in enumerate(X)]
    L.append('</Nodes>')
    eid = [1]

    def elems(kind, name, E):
        L.append(f'<Elements type="{kind}" name="{name}">')
        L.extend(f'<elem id="{eid[0]+i}">' + ','.join(map(str, e + 1)) + '</elem>' for i, e in enumerate(E))
        L.append('</Elements>'); eid[0] += len(E)

    elems('tet4', 'CA', _positive(S.aligner_tets, X))
    elems('tet4', 'PDL', _positive(S.pdl_tets, X))
    for u in S.keep:
        elems('tri3', f'crown{u}', S.tooth[u]['surf_tris'])
    outer = np.concatenate([S.tooth[u]['pdl_off'] + S.tooth[u]['pdl_layers'][-1] for u in S.keep])
    sets = {'outer': outer, 'sym': S.sym_nodes}
    for u in S.keep:
        sets[f'inner{u}'] = S.tooth[u]['pdl_off'] + S.tooth[u]['pdl_layers'][0]
    if load.enabled:
        sets['hook'] = load.hook_nodes
    for name, ids in sets.items():
        L.append(f'<NodeSet name="{name}">' + ','.join(map(str, np.asarray(ids) + 1)) + '</NodeSet>')
    ca = _oriented_inner_faces(S.aligner_tets, S.aligner_inner_faces, X)
    L.append('<Surface name="ca_inner">')
    L.extend(f'<tri3 id="{i+1}">' + ','.join(map(str, f + 1)) + '</tri3>' for i, f in enumerate(ca))
    L += ['</Surface>', '<Surface name="teeth">']
    L.extend(f'<tri3 id="{i+1}">' + ','.join(map(str, f + 1)) + '</tri3>' for i, f in enumerate(S.master_tris))
    L += ['</Surface>', '<SurfacePair name="ca_teeth"><primary>ca_inner</primary>'
          '<secondary>teeth</secondary></SurfacePair>', '</Mesh>', '<MeshDomains>',
          '<SolidDomain name="CA" mat="CA"/>', '<SolidDomain name="PDL" mat="PDL"/>']
    L += [f'<ShellDomain name="crown{u}" mat="rb{u}"><shell_thickness>0.1</shell_thickness></ShellDomain>'
          for u in S.keep]
    L += ['</MeshDomains>', '<Boundary>',
          '<bc name="bone" node_set="outer" type="zero displacement"><x_dof>1</x_dof><y_dof>1</y_dof><z_dof>1</z_dof></bc>',
          '<bc name="symmetry" node_set="sym" type="zero displacement"><x_dof>1</x_dof><y_dof>0</y_dof><z_dof>0</z_dof></bc>']
    L += [f'<bc name="tie{u}" node_set="inner{u}" type="rigid"><rb>rb{u}</rb></bc>' for u in S.keep]
    L.append('</Boundary>')
    pen = K_N
    L += ['<Constraints>', '<constraint name="aligner" type="aligner-pairs">',
          f'<pairs_file>{os.path.abspath(pairs_path)}</pairs_file>', f'<penalty>{K_N}</penalty>',
          f'<tangential_penalty>{K_T}</tangential_penalty>', f'<fric_coeff>{mu}</fric_coeff>',
          '<ramp lc="1">1</ramp>', '</constraint>', '</Constraints>']
    if load.enabled:
        d, _ = load.direction(state)
        f = (load.magnitude*d/len(load.hook_nodes)) @ R.T
        L += ['<Loads>', f'<nodal_load name="elastic" type="nodal_force" node_set="hook">'
              f'<value lc="1">{_fmt(f)}</value></nodal_load>', '</Loads>']
    L += ['<LoadData>', '<load_controller id="1" type="loadcurve"><interpolate>LINEAR</interpolate>'
          '<points><pt>0,0</pt><pt>1,1</pt></points></load_controller>', '</LoadData>',
          '<Output>', '<logfile>',
          f'<node_data data="ux;uy;uz" delim="," file="{os.path.basename(path)[:-4]}_u.txt"/>',
          '</logfile>', '</Output>', '</febio_spec>']
    with open(path, 'w') as fh:
        fh.write('\n'.join(L) + '\n')
    return pen


def _config(path):
    imports = [PDL_PLUGIN, PAIRS_PLUGIN] + ([CUDSS_PLUGIN] if LINSOLVE == 'cudss' else [])
    with open(path, 'w') as fh:
        fh.write('<?xml version="1.0" encoding="ISO-8859-1"?>\n<febio_config version="3.0">\n'
                 f'\t<default_linear_solver type="{LINSOLVE}"/>\n'
                 + ''.join(f'\t<import>{p}</import>\n' for p in imports) + '</febio_config>\n')


def _env():
    """Library path for FEBio + the cuDSS plug-in: cuDSS, and ONE cuBLAS / CUDA 12 runtime.

    libcudss resolves libcublas through its own RUNPATH (the copy installed next to it),
    so libcublasLt must come from that same directory, or the loader mixes two cuBLAS
    releases (undefined symbol cublasLtGetEnvironmentMode) and FEBio silently drops the
    plug-in.  Torch's copies are only a fallback when the cuDSS install has none."""
    e = dict(os.environ)
    pick = lambda sub: next((d for d in (os.path.join(CUDSS_DIR, 'nvidia', sub, 'lib'), os.path.join(_NV, sub, 'lib'))
                             if os.path.isdir(d)), None)
    libs = [os.path.join(CUDSS_DIR, 'nvidia/cu12/lib'), pick('cublas'), pick('cuda_runtime')]
    e['LD_LIBRARY_PATH'] = ':'.join([os.path.abspath(d) for d in libs if d] + [e.get('LD_LIBRARY_PATH', '')])
    return e


def _read_u(path, n):
    txt = open(path).read()
    block = txt[txt.rfind('*Data'):]
    block = block[block.find('\n', block.find('*Data')) + 1:]
    A = np.loadtxt(block.splitlines(), delimiter=',')
    U = np.zeros((n, 3)); U[A[:, 0].astype(int) - 1] = A[:, 1:4]
    return U


def _log_stats(path):
    txt = open(path, errors='ignore').read()
    sec = lambda lab: (lambda m: float(m.group(1)) if m else float('nan'))(
        re.search(r'\s' + lab + r'\s*\.*\s*:\s*[\d:]+\s*\(([\d.eE+-]+) sec\)', txt))
    it = re.search(r'Total number of equilibrium iterations\s*\.*\s*:\s*(\d+)', txt)
    return dict(ok='N O R M A L   T E R M I N A T I O N' in txt, iters=int(it.group(1)) if it else -1,
                total_s=sec('Total elapsed time'), linear_s=sec('time in linear solver'),
                stiffness_s=sec('evaluating stiffness'), residual_s=sec('evaluating residual'))


def solve_config(S, state, load, mu=spec.FRICTION, verbose=False, contact_ref=None, **_ignored):
    """One configuration, solved by FEBio.  Same return contract as solve4d.solve_config."""
    if getattr(S, 'deformable_bone', False):
        raise NotImplementedError('OFJ_SOLVER=febio supports the rigid-bone model only')
    t0 = time.perf_counter()
    os.makedirs(WORK, exist_ok=True)
    _count[0] += 1
    name = f'solve{_count[0]:04d}' if KEEP else 'solve'
    feb = os.path.join(WORK, name + '.feb')
    nodes = state.nodes()
    for u in S.keep:
        S.tooth[u]['centroid_cur'] = state.centroid(u)

    # contact: the pairing, normals and reference of solve4d, fixed for the solve
    # (pipeline/febio_pairs, `aligner-pairs`): slave node -> master triangle, barycentric weights,
    # master normal at the start, gref = gap_now - gap_pre + clearance, ramped in on load curve 1
    slave = contact_ref['slave']
    gap_now, nrm = S.gap_for_pairing(nodes, slave, contact_ref['tri'], contact_ref['bary'])
    gref = gap_now - contact_ref['gap_pre'] + CLEARANCE
    model = nodes
    R, p0 = _frame(S)
    pairs_path = feb[:-4] + '_pairs.csv'
    np.savetxt(pairs_path, np.column_stack([slave + 1, contact_ref['tri'] + 1, contact_ref['bary'], nrm @ R.T, gref]),
               fmt=['%d']*4 + ['%.12g']*7, delimiter=',')
    pen = _write(feb, S, state, load, model, mu, R, p0, pairs_path)
    _config(os.path.join(WORK, 'febio.cnf.xml'))
    t1 = time.perf_counter()
    p = subprocess.run([FEBIO, '-cnf', os.path.join(WORK, 'febio.cnf.xml'), '-i', feb, '-silent'],
                       cwd=WORK, env=_env(), capture_output=True, text=True)
    t2 = time.perf_counter()
    bad = [l for l in (p.stdout + p.stderr).splitlines() if 'Failed loading plugin' in l]
    if bad:
        raise RuntimeError('FEBio could not load a plug-in: ' + '; '.join(bad))
    st = _log_stats(feb[:-4] + '.log')
    if not st['ok']:
        tail = open(feb[:-4] + '.log', errors='ignore').read()[-3000:]
        raise RuntimeError(f'FEBio did not terminate normally ({feb}):\n{tail}\n{p.stdout[-1500:]}')
    U = _read_u(feb[:-4] + '_u.txt', S.n_global) @ R          # back to the model frame

    q = {}
    for u in S.keep:
        idx = S.rigid_idx[u]
        Rt, cx, cy = _kabsch(model[idx], model[idx] + U[idx])
        c = state.centroid(u)
        q[u] = np.concatenate([(Rt @ (c - cx) + cy) - c, _rotvec(Rt)])

    ndof = 3*S.n_global
    opP = Operators(nodes, S.pdl_tets, spec.PDL_NU)
    eps = opP.strain(U.ravel())
    E_pdl = pdl_secant_modulus(vm_equiv_strain(eps))
    # net load the aligner puts on the dentition = PDL reaction on the teeth (equilibrium)
    fr = (opP.K(E_pdl, ndof) @ U.ravel()).reshape(-1, 3)
    Fc = float(np.linalg.norm(sum(fr[S.rigid_idx[u]].sum(0) for u in S.keep)))
    gap_def, _ = S.gap_for_pairing(model + U, slave, contact_ref['tri'], contact_ref['bary'])
    t3 = time.perf_counter()
    hist = [dict(it=st['iters'], active=int((gap_def < 0).sum()), dA=0, cg=0, Fc=Fc, du=0.0, dF=0.0, dE=0.0,
                 du_max=0.0, dE_max=0.0, febio=st, write_s=t1 - t0, febio_wall_s=t2 - t1, post_s=t3 - t2,
                 penalty=pen, linsolve=LINSOLVE)]
    if verbose:
        print(f"    [febio {LINSOLVE}] {st['iters']} it  |Fc|~{Fc:.3f} N  write {t1-t0:.1f}s  "
              f"febio {t2-t1:.1f}s (linear {st['linear_s']:.1f}s)  post {t3-t2:.1f}s", flush=True)
    return dict(u=U.ravel(), x=None, q=q, lam=None, lam_relaxed=None, active=gap_def < 0, nrm=nrm, C=None,
                hist=hist, E_pdl=E_pdl, eps=eps, opP=opP, P=None, nodes=nodes)
