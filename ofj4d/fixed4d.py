"""HG group: fixed appliance + cervical headgear, 4D (bone-remodelling) simulation.

Model, on the same patient geometry, PDL, rigid teeth and remodelling scheme as the
aligner groups (so that the only differences are the appliance and the force system):

    teeth          rigid bodies, FDI 17..11, midsagittal symmetry
    PDL            0.30 mm shell, bilinear (spec.PDL_E1/E2), outer surface fixed (rigid bone)
    brackets       11-15 from the supplied Parasolid library (MBT, 0.022" slot), 16/17 tubes;
                   rigid with their teeth, placed at the FA point (brackets.py)
    archwire       0.019x0.025 SS, passive, 3D beams, bonded in every slot (wire4d.py)
    headgear       cervical pull, 250 gf at the centre of the 16 headgear tube, directed
                   distally and HG_ANGLE below the occlusal plane (extrusive), fixed in space
    no aligner, no contact, no miniscrew

4D scheme (Hamanaka et al., as for the aligners): each iteration the teeth move
elastically under the load with the PDL at its original thickness; that position is
retained and the PDL is reset around it.  The wire is NOT reset -- it is elastic and keeps
the energy of any differential tooth movement.  HG_ITER iterations (default 140) = the
aligner protocol's 70 steps x 2 PDL iterations, i.e. the same number of remodelling cycles;
results are also written per pair of iterations so they line up with aligner steps.

usage:  python ofj4d/fixed4d.py [--iters 140] [--out build4d/study/run_hg.pkl]
"""
import sys, os, pickle, time, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ofj'))
from scipy.sparse import diags, bmat, csr_matrix
import spec
from model4d import Static, State
from arch4d import ArchForm
from solve4d import (Operators, direct_solver, vm_equiv_strain, pdl_secant_modulus,
                     U_TOL, MAXIT)
from run4d import crown_point, root_point, measure
import brackets as BR
import wire4d as WR

EMPTY_ALIGNER = os.path.join('build4d', 'aligner_none.pkl')
FIXED_PKL = os.environ.get('OFJ_FIXED', os.path.join('build4d', 'fixed.pkl'))


def empty_aligner(path=EMPTY_ALIGNER):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        pickle.dump(dict(nodes=np.zeros((0, 3)), tets=np.zeros((0, 4), np.int64),
                         inner_faces=np.zeros((0, 3), np.int64), n_inner=0, n_layer_nodes=0,
                         vert_tooth=np.zeros(0, np.int64), thickness=0.0),
                    open(path, 'wb'))
    return path


def log_rot(R):
    """Rotation vector of a rotation matrix."""
    c = np.clip((np.trace(R) - 1)/2, -1, 1)
    th = np.arccos(c)
    if th < 1e-12:
        return 0.5*np.array([R[2, 1]-R[1, 2], R[0, 2]-R[2, 0], R[1, 0]-R[0, 1]])
    return th/(2*np.sin(th))*np.array([R[2, 1]-R[1, 2], R[0, 2]-R[2, 0], R[1, 0]-R[0, 1]])


def build(geom=None, xt=BR.XT_DEFAULT, out=FIXED_PKL):
    """Brackets, tubes and the passive wire for this patient -> build4d/fixed.pkl."""
    geom = geom or os.environ.get('OFJ_GEOM', 'build4d/geom.pkl')
    G = pickle.load(open(geom, 'rb'))
    S = Static(geom=geom, aligner=empty_aligner())
    fx = BR.build_fixed(G, xt=xt, keep=S.keep, fdi_of=spec.FDI)
    sec = WR.section()
    if BR.WIRE_SETUP in ('level', 'smooth'):
        A = ArchForm(S)
        flat = BR.WIRE_SETUP == 'level'
        fx, level = BR.level_on_wire(fx, G, A.origin, A.en, S.n_sag, S.sag_offset, sec['h'],
                                     level=float(BR.WIRE_LEVEL) if BR.WIRE_LEVEL else None, flat=flat)
        print(f"passive level wire at {level:+.2f} mm along the occlusal normal; slots seated on it" if flat
              else "passive smooth wire through the FA slot centres; brackets at the FA point")
    W = WR.build_path(fx, list(S.keep), S.n_sag, S.sag_offset, wire_h=sec['h'])
    D = dict(fixed=fx, wire=W, section=sec, wire_spec=WR.WIRE, geom=geom, xt=os.path.basename(xt))
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    pickle.dump(D, open(out, 'wb'))
    print(BR.report(fx))
    L = np.linalg.norm(np.diff(W['X'], axis=0), axis=1)
    print(f"archwire {WR.WIRE}\" SS: {len(W['X'])} nodes, {len(W['elems'])} beams, "
          f"length {L.sum():.1f} mm (elements {L.min():.2f}-{L.max():.2f} mm), "
          f"{(W['owner'] >= 0).sum()} nodes in slots")
    for u, r in fx.items():
        i = r['info']
        print(f"  FDI {r['fdi']}: FA height {i['h_fa']:.2f} (crown {i['crown_height']:.2f} mm)  "
              + (f"moved {i['moved_from_fa']:.2f} mm, turned {i['turned']:.1f} deg from FA, crown contact {i['depth']:+.3f} mm  "
                 if 'moved_from_fa' in i else '')
              + f"+x mesial: {i['mesial_is_plus_x']}"
              + (f"   HG tube {np.round(r['hg_tube'], 2)}" if 'hg_tube' in r else ''))
    print(f"-> {out}")
    return D


class Headgear:
    """Cervical headgear: constant force at the headgear tube of the first molar."""

    def __init__(self, S, fx, arch, gf=spec.HG_FORCE_GF, angle=spec.HG_ANGLE):
        u = spec.UNN[16]
        self.unn = u
        self.tube_rest = fx[u]['hg_tube']
        self.F = gf*spec.GF_TO_N*BR.unit(np.cos(np.radians(angle))*arch.ey
                                         + np.sin(np.radians(angle))*arch.en)
        self.gf, self.angle = gf, angle

    def point(self, state):
        T = state.T[self.unn]
        return self.tube_rest @ T[:3, :3].T + T[:3, 3]

    def reduced(self, S, state, n):
        r = np.zeros(n)
        q = S.q_col[self.unn]
        x = self.point(state)
        c = S.tooth[self.unn]['centroid_cur']
        r[q:q+3] = self.F
        r[q+3:q+6] = np.cross(x - c, self.F)
        return r

    def describe(self, S):
        t = S.tooth[self.unn]
        rel = self.tube_rest - t['crown_c']
        return (f"cervical headgear: {self.gf:.0f} gf = {np.linalg.norm(self.F):.4f} N at the 16 "
                f"headgear tube, {self.angle:.0f} deg below the occlusal plane\n"
                f"  tube vs 16 crown centre: {rel @ t['e_d']:+.2f} distal {rel @ t['e_b']:+.2f} buccal "
                f"{rel @ t['e_o']:+.2f} occlusal;  force {np.round(self.F, 3)}")


from contact4d import ToothContact, TOOTH_CONTACT, TC_K   # tooth-tooth contact (shared with solve4d)
STRESS_DUMP = {int(x) for x in os.environ.get('OFJ_STRESS_DUMP', '').split(',') if x.strip()}


class FixedSystem:
    def __init__(self, fixed=FIXED_PKL):
        geom = os.environ.get('OFJ_GEOM', 'build4d/geom.pkl')
        if not os.path.exists(fixed):
            build(geom)
        D = pickle.load(open(fixed, 'rb'))
        self.S = Static(geom=D['geom'] if os.path.exists(D['geom']) else geom,
                        aligner=empty_aligner())
        self.fx = D['fixed']; self.W = D['wire']; self.sec = D['section']
        self.wire = WR.CorotWire(self.W, self.sec)
        self.wmap = WR.WireMap(self.W, self.S, self.fx)
        self.sc = WR.SlotContact(self.W, self.S, self.fx, self.sec) if self.wmap.coupling == 'slot' else None
        self.tc = None
        if TOOTH_CONTACT:
            G = pickle.load(open(D['geom'] if os.path.exists(D['geom']) else geom, 'rb'))
            self.tc = ToothContact(self.S, G)
            print(f"tooth-tooth contact: {len(self.tc.pts)} crown points on {len(self.S.keep)-1} "
                  f"adjacent pairs, K {TC_K:g} N/mm per point")
        print(f"wire: {len(self.W['X'])} nodes, coupling '{self.wmap.coupling}', "
              f"{self.wmap.n_w} free wire dofs; section {WR.WIRE} "
              f"(A {self.sec['A']:.4f} mm2, I {self.sec['I_n']:.2e}/{self.sec['I_g']:.2e} mm4)")

    def wire_now(self, state):
        """Slaved wire nodes follow their teeth exactly (position and rotation)."""
        X0 = self.W['X']; own = self.W['owner']
        w = self.wire
        for u in self.S.keep:
            k = (own == u) & (self.wmap.kind == 1)
            if k.any():
                T = state.T[u]
                w.x[k] = X0[k] @ T[:3, :3].T + T[:3, 3]
                w.R[k] = T[:3, :3]
        return w.x

    def solve(self, state, hg, E_pdl0=None, maxit=MAXIT, verbose=False):
        S = self.S
        nodes = state.nodes()
        for u in S.keep:
            S.tooth[u]['centroid_cur'] = state.centroid(u)
        P = S.build_P(nodes)
        ndof = 3*S.n_global
        opP = Operators(nodes, S.pdl_tets, spec.PDL_NU, reuse=True)
        Xw = self.wire_now(state)
        Gm = self.wmap.build(Xw, {u: S.tooth[u]['centroid_cur'] for u in S.keep})
        f_int, Kt, energy0 = self.wire.forces()
        Kwr = (Gm.T @ Kt @ Gm).tocsr()
        n_ext = self.wmap.n_ext
        b = np.zeros(n_ext)
        b[:S.n_red] = hg.reduced(S, state, S.n_red)
        b -= Gm.T @ f_int                                # the wire's stored elastic force
        E_pdl = np.full(len(S.pdl_tets), spec.PDL_E1) if E_pdl0 is None else E_pdl0.copy()
        direct = direct_solver()
        hist = []; u_prev = None; x = None
        sc = self.sc; b0 = b
        if sc is not None:                               # wire-slot contact (play + friction)
            sc.select(state.T, self.wire.x, self.wire.R)
            Cm = sc.rows(self.wmap, S, {u: S.tooth[u]['centroid_cur'] for u in S.keep})
            dlt = np.zeros(Cm.shape[0])
        tc = self.tc
        if tc is not None:                               # tooth-tooth contact
            tc.select(state.T)
            Ct = tc.rows(S, n_ext, {u: S.tooth[u]['centroid_cur'] for u in S.keep})
            dlt_t = np.zeros(Ct.shape[0])
        for it in range(maxit):
            KP = opP.K(E_pdl, ndof)
            Kr = (P.T @ KP @ P).tocsr()
            Kr.resize((n_ext, n_ext))
            A = (Kr + Kwr).tocsr()
            if sc is not None:
                kd, f0, st, Nn = sc.law(dlt)
                A = (A + Cm.T @ diags(kd) @ Cm).tocsr()
                b = b0 - Cm.T @ f0
            if tc is not None:
                kt, ft = tc.law(dlt_t)
                A = (A + Ct.T @ diags(kt) @ Ct).tocsr()
                b = (b if sc is not None else b0) - Ct.T @ ft
            dg = np.sqrt(np.abs(A.diagonal())); dg[dg <= 0] = 1.0
            Di = diags(1.0/dg)
            As = (Di @ A @ Di).tocsr(); bs = b/dg
            if direct is not None:
                xs, _ = direct.solve(As, bs)
            else:
                from scipy.sparse.linalg import spsolve
                xs = spsolve(As.tocsc(), bs)
            res = np.linalg.norm(As @ xs - bs)/max(np.linalg.norm(bs), 1e-30)
            x = xs/dg
            u = P @ x[:S.n_red]
            eps = opP.strain(u)
            E_new = pdl_secant_modulus(vm_equiv_strain(eps))
            dE = np.percentile(np.abs(E_new - E_pdl), 99.9)/spec.PDL_E2
            E_pdl = 0.5*E_pdl + 0.5*E_new
            uscale = max(np.abs(u).max(), 1e-30)
            du = np.percentile(np.abs(u - u_prev), 99.9)/uscale if u_prev is not None else 1.0
            u_prev = u
            same = True
            if sc is not None:
                dn = Cm @ x
                same = np.abs(dn - dlt).max() < 2e-5            # contact increments settled (mm, rad)
                dlt = dn
            if tc is not None:
                dt = Ct @ x
                same = same and np.abs(dt - dlt_t).max() < 2e-5
                dlt_t = dt
            hist.append(dict(it=it, du=float(du), dE=float(dE), res=float(res), contact_same=same))
            if verbose:
                print(f"    it{it:2d} du={du:.2e} dE={dE:.2e} umax={uscale*1e3:.2f}um res={res:.1e}"
                      + ('' if sc is None else f" contact {'=' if same else 'changed'}"))
            if du < U_TOL and dE < 1e-2 and it >= 2 and same:
                break
        q = {u_: x[S.q_col[u_]:S.q_col[u_]+6] for u_ in S.keep}
        dU = Gm @ x
        out = dict(x=x, q=q, dU=dU, eps=eps, E_pdl=E_pdl, hist=hist, wire_energy=float(energy0))
        if tc is not None:
            gmin = float((np.array([c[1] for c in tc.cur]) + dlt_t).min()) if len(dlt_t) else 0.0
            out['tooth_contact'] = dict(Fsum=float(tc.N_last.sum()), gap_min=float(gmin),
                                        n_active=int((tc.N_last > 1e-3).sum()))
        if sc is not None:
            sc.N_lag = Nn.copy()
            out['contact'] = [dict(fdi=p['fdi'], state=s_, N=float(n_)) for p, s_, n_ in zip(sc.pts, st, Nn)]
        return out


def run(iters=spec.HG_ITER, out=None, resume=True, tag='hg'):
    out = out or os.path.join('build4d', 'study', 'run_hg.pkl')
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    FS = FixedSystem()
    S = FS.S
    state = State(S)
    for u in S.keep:
        S.tooth[u]['centroid_cur'] = state.centroid(u)
    arch = ArchForm(S)
    hg = Headgear(S, FS.fx, arch)
    print(f"group 'hg': fixed appliance + cervical headgear   [rigid bone, custom solver]", flush=True)
    print(hg.describe(S), flush=True)
    ref = dict(crown={u: crown_point(S, u) for u in S.keep},
               root={u: root_point(S, u) for u in S.keep})
    ref['LA'] = {u: (ref['crown'][u]-ref['root'][u])/np.linalg.norm(ref['crown'][u]-ref['root'][u])
                 for u in S.keep}
    hist, per_iter = [], []
    start = 1; E_prev = None
    if resume and os.path.exists(out):
        D = pickle.load(open(out, 'rb'))
        if D.get('group') == 'hg':
            hist, per_iter = D['hist'], D.get('per_iter', [])
            state.T = D['T']; FS.wire.x = D['wire_x']; FS.wire.R = D['wire_R']
            start = D['iter'] + 1; E_prev = D.get('E_pdl')
            if FS.sc is not None and D.get('N_lag') is not None:
                FS.sc.N_lag = D['N_lag']
            print(f"resuming at iteration {start}", flush=True)
    t_start = time.time()
    info = []
    for k in range(start, iters + 1):
        t0 = time.time()
        r = FS.solve(state, hg, E_pdl0=E_prev)
        E_prev = r['E_pdl']
        if k in STRESS_DUMP:                               # PDL stress of this loaded state (OFJ_STRESS_DUMP)
            import json, stress4d
            hyd, vm, cen = stress4d.fields(S, state.nodes(), r['eps'], r['E_pdl'])
            pre = os.environ.get('OFJ_STRESS_OUT', out[:-4] + '_pdlstress')
            np.savez_compressed(f"{pre}_iter{k}.npz", centroid=cen, owner=np.array([spec.FDI[u] for u in S.pdl_owner]),
                                hydrostatic_kPa=hyd, von_mises_kPa=vm)
            js = pre + '.json'
            summ = json.load(open(js)) if os.path.exists(js) else dict(run=out, group='hg', steps={})
            summ['steps'][str((k + 1) // spec.PDL_ITERATIONS_PER_STEP)] = dict(iteration=k, teeth=stress4d.per_tooth(S, state, ref, hyd, vm, cen))
            json.dump(summ, open(js, 'w'), indent=1)
        for u in S.keep:
            q = r['q'][u]
            state.apply_rigid(u, q[:3], q[3:])            # retain position, reset PDL
        FS.wire.advance(r['dU'], FS.wmap.kind != 1)      # free wire nodes
        FS.wire_now(state)                                # slaved nodes exact
        e99 = float(np.percentile(vm_equiv_strain(r['eps']), 99))
        m = measure(S, state, ref)
        rec = dict(iter=k, nit=len(r['hist']), eps99=e99, wire_energy=r['wire_energy'],
                   meas=m, T={u: state.T[u].copy() for u in S.keep})
        per_iter.append({kk: vv for kk, vv in rec.items() if kk != 'T'})
        info.append(dict(nit=len(r['hist']), Fc=float(np.linalg.norm(hg.F)), act=0,
                         du=r['hist'][-1]['du'], emax=e99, converged=len(r['hist']) < MAXIT))
        if k % spec.PDL_ITERATIONS_PER_STEP == 0:
            step = k // spec.PDL_ITERATIONS_PER_STEP
            hist.append(dict(step=step, meas=m, info=info, T=rec['T']))
            info = []
        print(f"[{tag}] iter {k:3d}/{iters}  it={len(r['hist'])} eps99={e99*100:4.1f}% "
              f"16d={m[3]['crown_distal']:+.3f} 17d={m[2]['crown_distal']:+.3f} "
              f"11b={m[8]['crown_buccal']:+.3f} 16o={m[3]['crown_occlusal']:+.3f} "
              f"Ew={r['wire_energy']:.4f}Nmm"
              + (f" Ftt={r['tooth_contact']['Fsum']:.2f}N gap={r['tooth_contact']['gap_min']*1e3:+.0f}um"
                 if 'tooth_contact' in r else '') + f" ({time.time()-t0:.0f}s, total "
              f"{(time.time()-t_start)/60:.0f}m)", flush=True)
        pickle.dump(dict(group='hg', iter=k, step=k // spec.PDL_ITERATIONS_PER_STEP,
                         hist=hist, per_iter=per_iter, T={u: state.T[u].copy() for u in S.keep},
                         wire_x=FS.wire.x.copy(), wire_R=FS.wire.R.copy(), E_pdl=E_prev, ref=ref, keep=S.keep,
                         N_lag=None if FS.sc is None else FS.sc.N_lag.copy(),
                         hg=dict(F=hg.F, gf=hg.gf, angle=hg.angle, tube=hg.tube_rest)),
                    open(out, 'wb'))
    return hist


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--build', action='store_true', help='(re)build brackets + wire only')
    ap.add_argument('--iters', type=int, default=spec.HG_ITER)
    ap.add_argument('--out', default=None)
    ap.add_argument('--no-resume', action='store_true')
    a = ap.parse_args()
    if a.build:
        build()
    else:
        run(iters=a.iters, out=a.out, resume=not a.no_resume)
