"""Stage E: the 70-step 4D staging simulation.

Per step, following Mao et al.:
  1. the CA is regenerated for the prescribed tooth positions of this step;
  2. it is seated on the actual crowns (wear-in best fit) before EACH iteration;
  3. two PDL iterations are run.  Each iteration is one bone-remodelling cycle of
     Hamanaka et al.: the tooth moves elastically under the appliance load, that
     position is retained, and the PDL is restored to its original thickness
     around the tooth's new position.

Because the PDL shell is a rigid offset of the root, step 3's 'restore' is just a
rigid transform of the shell -- the mesh is never rebuilt.
"""
import sys, os, pickle, time, argparse, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ofj'))
from model4d import Static, State
from arch4d import ArchForm
from morph4d import Morpher
from loads4d import Elastic
from solve4d import solve_config, vm_equiv_strain, MAX_GAP
# Staging solver: 'custom' (solve4d: GPU assembly + cuDSS/PARDISO) or 'febio'
# (febio4d: FEBio 4, finite deformation, cuDSS on the GPU).  Same inputs and outputs.
SOLVER = os.environ.get('OFJ_SOLVER', 'custom').lower()
if SOLVER == 'febio':
    from febio4d import solve_config
from contact import build_pairs
import spec


def crown_point(S, u):
    """Paper's crown point: incisal midpoint / cusp tip / occlusal-surface midpoint."""
    T = S.tooth[u]
    P = T['surf_rest'][:T['surf_n']]
    return P[np.argmax(P @ T['e_o'])]


def root_point(S, u):
    T = S.tooth[u]
    P = S.tooth[u]['pdl_rest']
    return P[np.argmin(P @ T['e_o'])]


def measure(S, state, ref):
    """3D displacement of the crown and root points, and long-axis rotation."""
    out = {}
    for u in S.keep:
        T = state.T[u]
        cp = ref['crown'][u] @ T[:3, :3].T + T[:3, 3]
        rp = ref['root'][u] @ T[:3, :3].T + T[:3, 3]
        st = S.tooth[u]
        dc = cp - ref['crown'][u]; dr = rp - ref['root'][u]
        R = T[:3, :3]
        la0 = ref['LA'][u]
        la1 = R @ la0
        # decompose the long-axis rotation on the paper's three axes
        ang = np.degrees(np.arccos(np.clip(la0 @ la1, -1, 1)))
        axis = np.cross(la0, la1)
        na = np.linalg.norm(axis)
        axis = axis/na if na > 1e-12 else np.zeros(3)
        rot = axis*ang
        out[u] = dict(
            crown=dc, root=dr,
            crown_distal=float(dc @ st['e_d']), crown_buccal=float(dc @ st['e_b']),
            crown_occlusal=float(dc @ st['e_o']),
            root_distal=float(dr @ st['e_d']), root_buccal=float(dr @ st['e_b']),
            root_occlusal=float(dr @ st['e_o']),
            rot_coronal=float(rot @ st['e_b']),     # tipping (mesiodistal plane)
            rot_sagittal=float(rot @ st['e_d']),    # torque / buccolingual crown-root
            rot_vertical=float(rot @ st['e_o']),    # rotation about the long axis
            rot_total=float(ang))
    return out


def run(group='buccal_tad', n_steps=spec.N_STEPS, out=None, resume=True, tag=None):
    tag = tag or group
    out = out or f'build4d/run_{group}.pkl'
    S = Static()
    state = State(S)
    arch = ArchForm(S)
    morph = Morpher(S, arch)
    for u in S.keep:
        S.tooth[u]['centroid_cur'] = state.centroid(u)
    load = Elastic(S, state, group=group)
    if os.environ.get('OFJ_ALIGNER_TOOTH_CONTACT', '0') == '1':
        import pickle as _pk
        from contact4d import ToothContact, TC_K
        S.tooth_contact = ToothContact(S, _pk.load(open(os.environ.get('OFJ_GEOM', 'build4d/geom.pkl'), 'rb')))
        print(f"tooth-tooth contact: {len(S.tooth_contact.pts)} crown points, K {TC_K:g} N/mm per point",
              flush=True)
    print(f"bone model: {'deformable' if getattr(S, 'deformable_bone', False) else 'rigid'}"
          f"   staging solver: {SOLVER}", flush=True)
    print(load.describe(S, state), flush=True)
    print(arch.report(), flush=True)

    ref = dict(crown={u: crown_point(S, u) for u in S.keep},
               root={u: root_point(S, u) for u in S.keep})
    ref['LA'] = {}
    for u in S.keep:
        v = ref['crown'][u] - ref['root'][u]
        ref['LA'][u] = v/np.linalg.norm(v)

    hist = []
    start = 1
    if resume and os.path.exists(out):
        D = pickle.load(open(out, 'rb'))
        if D.get('group') == group:
            hist = D['hist']; state.T = D['T']; start = D['step']+1
            print(f"resuming at step {start}", flush=True)

    cache = {}
    E_prev = None
    t_start = time.time()
    slave = np.unique(S.aligner_inner_faces)
    if len(getattr(load, 'cut_nodes', [])):
        # aligner trimmed around a bonded metal hook: that part of it touches nothing
        slave = np.setdiff1d(slave, load.cut_nodes)
        print(f"aligner cut-out: {len(load.cut_nodes)} inner nodes removed from contact", flush=True)
    for step in range(start, n_steps+1):
        shape = morph.shape(step)
        # the appliance is manufactured for the prescribed positions of this step;
        # pair against that configuration and keep the pairing for the whole step
        pre = S.prescribed_nodes(state, arch, step, shape)
        Cp = build_pairs(pre, slave, S.master_tris, k=16, max_gap=MAX_GAP)
        assert len(Cp['slave']) == len(slave), 'pairing lost nodes'
        gap_pre, _ = S.gap_for_pairing(pre, slave, Cp['tri'], Cp['bary'])
        cref = dict(slave=slave, tri=Cp['tri'], bary=Cp['bary'], gap_pre=gap_pre)
        t0 = time.time()
        info = []
        for k in range(spec.PDL_ITERATIONS_PER_STEP):
            state.aligner = morph.seat(shape, state, step)   # wear-in before each iteration
            r = solve_config(S, state, load, E_pdl0=E_prev,
                             x0=cache.get('x0'), cache=cache, contact_ref=cref)
            E_prev = r['E_pdl']
            # --- bone remodelling: retain the new tooth position, reset the PDL ---
            for u in S.keep:
                q = r['q'][u]
                state.apply_rigid(u, q[:3], q[3:])
            info.append(dict(nit=(r['hist'][-1]['it'] if SOLVER == 'febio' else len(r['hist'])),
                             Fc=r['hist'][-1]['Fc'],
                             act=r['hist'][-1]['active'],
                             du=r['hist'][-1]['du'],
                             emax=float(np.percentile(vm_equiv_strain(r['eps']), 99)),
                             emax_abs=float(vm_equiv_strain(r['eps']).max()),
                             converged=(SOLVER == 'febio' or len(r['hist']) < 35),
                             tie_inside=getattr(S, 'tie_inside', None),
                             tooth_contact=r.get('tooth_contact')))
        m = measure(S, state, ref)
        hist.append(dict(step=step, meas=m, info=info,
                         T={u: state.T[u].copy() for u in S.keep}))
        act = [u for u in spec.active_teeth(step)]
        print(f"[{tag}] step {step:3d}/{n_steps}  moving "
              f"{','.join(str(spec.FDI[u]) for u in act):<12} "
              f"Fc={info[-1]['Fc']:6.3f}N it={info[0]['nit']}+{info[1]['nit']} "
              f"eps99={info[-1]['emax']*100:4.1f}% "
              f"{'' if all(i['converged'] for i in info) else 'CAP '}"
              f"17d={m[2]['crown_distal']:+.3f} 11b={m[8]['crown_buccal']:+.3f} "
              f"{('tie=%.3f ' % S.tie_inside) if getattr(S, 'deformable_bone', False) else ''}"
              f"{('Ftt=%.2fN gap=%+.0fum ' % (info[-1]['tooth_contact']['Fsum'], info[-1]['tooth_contact']['gap_min']*1e3)) if info[-1].get('tooth_contact') else ''}"
              f"({time.time()-t0:.0f}s, total {(time.time()-t_start)/60:.0f}m)", flush=True)
        pickle.dump(dict(group=group, step=step, hist=hist,
                         T={u: state.T[u].copy() for u in S.keep},
                         ref=ref, keep=S.keep), open(out, 'wb'))
    return hist


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--group', default='buccal_tad')
    ap.add_argument('--steps', type=int, default=spec.N_STEPS)
    ap.add_argument('--out', default=None)
    ap.add_argument('--no-resume', action='store_true')
    a = ap.parse_args()
    run(group=a.group, n_steps=a.steps, out=a.out, resume=not a.no_resume)
