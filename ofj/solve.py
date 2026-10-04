"""Stage 6: nonlinear solve -- active-set unilateral contact + Coulomb friction.

Contact is node-to-surface, small-sliding (pairing fixed at the reference
configuration), regularised with a penalty. The aligner inner surface is the slave
side, the clinical crowns + composite attachments are the master side.
"""
import sys, os, pickle, time, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.linalg import cg
import pyamg
from fem import assemble, rigid_body_modes

K_N    = 2000.0   # N/mm  normal penalty per contact node
K_T    = 2000.0   # N/mm  tangential (stick) penalty
MU     = 0.20     # Coulomb friction coefficient, PET-G on enamel / composite
ADJUST = 0.20     # mm; initial gaps below this are closed (passive thermoformed fit)
MAXIT  = 30
RELAX  = 0.5      # relaxation on the friction normal force (explicit coupling)
U_TOL  = 2e-3     # convergence: relative change of the displacement field
F_TOL  = 5e-3     # convergence: relative change of the total contact force
RTOL   = 1e-8


def make_amg(A, B):
    return pyamg.smoothed_aggregation_solver(
        A, B=B, max_coarse=2000, smooth='energy',
        strength=('symmetric', {'theta': 0.02}),
        presmoother=('gauss_seidel', {'sweep': 'symmetric', 'iterations': 2}),
        postsmoother=('gauss_seidel', {'sweep': 'symmetric', 'iterations': 2}))


def build_P(M, ndof):
    """Prolongation solved-dof -> all-dof: drops fixed dofs and applies the MPCs."""
    nn = len(M['nodes'])
    fixed_node = np.zeros(nn, bool)
    fixed_node[M['fixed']] = True
    slave_node = np.zeros(nn, bool)
    for s in M['mpc']:
        slave_node[s] = True
    keep = (~fixed_node) & (~slave_node)
    kd = np.repeat(keep, 3)
    col = -np.ones(ndof, np.int64)
    col[kd] = np.arange(int(kd.sum()))
    r = np.where(kd)[0]
    rows = [r]; cols = [col[r]]; vals = [np.ones(len(r))]
    for s, w in M['mpc'].items():
        for m, wt in w.items():
            if fixed_node[m]:
                continue
            d = np.arange(3)
            rows.append(3*s+d); cols.append(col[3*m+d]); vals.append(np.full(3, wt))
    rows = np.concatenate(rows); cols = np.concatenate(cols); vals = np.concatenate(vals)
    ok = cols >= 0
    return csr_matrix((vals[ok], (rows[ok], cols[ok])), shape=(ndof, int(kd.sum())))


def contact_ops(C):
    """Per-pair 12-dof relative-displacement operator R and normal operator a."""
    s, tri, w, n = C['slave'], C['tri'], C['bary'], C['normal']
    npair = len(s)
    dofs = np.concatenate(
        [3*s[:, None] + np.arange(3)[None, :],
         (3*tri[:, :, None] + np.arange(3)[None, None, :]).reshape(npair, 9)], axis=1)
    R = np.zeros((npair, 3, 12))
    for d in range(3):
        R[:, d, d] = 1.0
        for j in range(3):
            R[:, d, 3+3*j+d] = -w[:, j]
    a = np.einsum('nij,ni->nj', R, n)
    return dofs, R, a


def solve(model='build/model.pkl', contact='build/contact.pkl',
          out='build/solution.pkl', load_scale=1.0, mu=MU, tag='base'):
    M = pickle.load(open(model, 'rb'))
    C = pickle.load(open(contact, 'rb'))
    nodes = M['nodes']; ndof = 3*len(nodes)

    t0 = time.time()
    mats = {m: (M['MAT'][m][1], M['MAT'][m][2]) for m in M['MAT']}
    K = assemble(nodes, M['elems'], M['mats'], mats, chunk=40000, verbose=False)
    print(f"[{tag}] K assembled: nnz={K.nnz/1e6:.1f}M in {time.time()-t0:.0f}s", flush=True)

    P = build_P(M, ndof)
    Kr = (P.T @ K @ P).tocsr()
    del K
    print(f"[{tag}] reduced system: {Kr.shape[0]} dof, nnz={Kr.nnz/1e6:.1f}M", flush=True)

    f = np.zeros(ndof)
    for L in M['loads'].values():
        nd = L['nodes']
        F = L['magnitude']*load_scale*L['dirn']/len(nd)
        for d in range(3):
            np.add.at(f, 3*nd+d, F[d])
    fr = P.T @ f
    tot = sum(L['magnitude']*load_scale for L in M['loads'].values())
    print(f"[{tag}] load: {tot:.4f} N total ({tot/2:.4f} N per side)", flush=True)

    dofs, R, a = contact_ops(C)
    nrm = C['normal']
    gref = np.maximum(0.0, C['gap0'] - ADJUST)
    npair = len(gref)
    # Initialise with the pairs that are genuinely touching (gref == 0). Forcing
    # stood-off pairs closed would inject a huge spurious pull-in force K_N*gref.
    active = (gref <= 0.0)
    stick = np.ones(npair, bool)
    tdir = np.zeros((npair, 3))
    lam = np.zeros(npair)
    tmag_prev = np.zeros(npair)

    Br = P.T @ rigid_body_modes(nodes)
    Pt_all = np.eye(3)[None, :, :] - np.einsum('ni,nj->nij', nrm, nrm)

    rebuild = False
    ml = None; x0 = None; u = np.zeros(ndof); hist = []
    u_prev = None; Fc_prev = None
    for it in range(MAXIT):
        act = np.where(active)[0]
        fc = np.zeros(ndof)
        Kn = K_N*np.einsum('ni,nj->nij', a[act], a[act])
        # Coulomb friction via a SECANT tangential stiffness: stick uses K_T, slip
        # uses k = mu*lambda/|s_t| so the tangential force is capped at mu*lambda.
        # This keeps friction implicit (in the matrix) instead of an explicit force,
        # which removes the lambda <-> friction feedback loop that diverges.
        ksec = np.where(K_T*tmag_prev[act] <= mu*lam[act], K_T,
                        mu*lam[act]/np.maximum(tmag_prev[act], 1e-12))
        ksec = np.clip(ksec, 0.0, K_T)
        stick = np.zeros(npair, bool); stick[act] = (ksec >= K_T - 1e-9)
        Ke = Kn + ksec[:, None, None]*np.einsum('nki,nkl,nlj->nij',
                                                R[act], Pt_all[act], R[act])
        d = dofs[act]
        Kc = coo_matrix((Ke.ravel(),
                         (np.repeat(d, 12, axis=1).ravel(), np.tile(d, (1, 12)).ravel())),
                        shape=(ndof, ndof)).tocsr()
        np.add.at(fc, d.ravel(), ((-K_N*gref[act])[:, None]*a[act]).ravel())

        A = (Kr + (P.T @ Kc @ P)).tocsr()
        b = fr + P.T @ fc
        if ml is None or rebuild:
            ts = time.time()
            ml = make_amg(A, Br)
            print(f"[{tag}]   AMG setup {time.time()-ts:.0f}s, levels={len(ml.levels)}", flush=True)
        rebuild = False
        nit = [0]
        x, info = cg(A, b, rtol=RTOL, maxiter=1500, x0=x0,
                     M=ml.aspreconditioner(cycle='V'),
                     callback=lambda xk: nit.__setitem__(0, nit[0]+1))
        res = np.linalg.norm(b - A@x)/max(np.linalg.norm(b), 1e-30)
        if res > 1e-6:
            print(f"[{tag}]   CG stalled (res={res:.1e}); rebuilding AMG", flush=True)
            ml = make_amg(A, Br)
            x, info = cg(A, b, rtol=RTOL, maxiter=1500, M=ml.aspreconditioner(cycle='V'))
            res = np.linalg.norm(b - A@x)/max(np.linalg.norm(b), 1e-30)
        if nit[0] > 250:
            rebuild = True            # preconditioner has gone stale as the set changed
        x0 = x
        u = P @ x

        rel = np.einsum('nij,nj->ni', R, u[dofs])
        cn = gref + np.einsum('ni,ni->n', nrm, rel)
        lam_new = np.where(cn < 0, -K_N*cn, 0.0)
        lam = lam_new if it == 0 else (1-RELAX)*lam + RELAX*lam_new
        new_active = cn < 0
        tang = rel - np.einsum('ni,ni->n', rel, nrm)[:, None]*nrm
        tmag = np.linalg.norm(tang, axis=1)
        tdir = np.where(tmag[:, None] > 1e-12, tang/np.maximum(tmag, 1e-12)[:, None], 0.0)
        tmag_prev = (1-RELAX)*tmag_prev + RELAX*tmag
        new_stick = stick
        Fc = float(np.linalg.norm((lam_new[:, None]*nrm).sum(0)))
        du_rel = (np.abs(u-u_prev).max()/max(np.abs(u).max(), 1e-30)) if u_prev is not None else 1.0
        dF_rel = (abs(Fc-Fc_prev)/max(Fc, 1e-30)) if Fc_prev is not None else 1.0
        u_prev = u.copy(); Fc_prev = Fc

        dA = int((new_active != active).sum())
        dS = 0
        hist.append(dict(it=it, active=int(new_active.sum()),
                         slip=int((new_active & ~new_stick).sum()),
                         dA=dA, dS=dS, res=float(res), cg=nit[0], Fc=Fc,
                         du_rel=float(du_rel), dF_rel=float(dF_rel),
                         umax=float(np.abs(u).max())))
        print(f"[{tag}] it{it:2d} active={new_active.sum():6d} slip={int((new_active&~stick).sum()):5d}"
              f" dA={dA:5d} cg={nit[0]:4d} |Fc|={Fc:.4f}N"
              f" du={du_rel:.2e} dF={dF_rel:.2e} umax={np.abs(u).max()*1e3:.3f} um", flush=True)
        conv = (du_rel < U_TOL and dF_rel < F_TOL and it >= 4)
        active = new_active
        if conv:
            print(f"[{tag}] CONVERGED: du={du_rel:.2e} < {U_TOL}, dF={dF_rel:.2e} < {F_TOL}", flush=True)
            break

    pickle.dump(dict(u=u, active=active, stick=stick, lam=lam, gref=gref, hist=hist,
                     load_scale=load_scale, mu=mu, gap0=C['gap0']), open(out, 'wb'))
    return u


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='build/solution.pkl')
    ap.add_argument('--load-scale', type=float, default=1.0)
    ap.add_argument('--mu', type=float, default=MU)
    ap.add_argument('--tag', default='250gf-per-side')
    A = ap.parse_args()
    solve(out=A.out, load_scale=A.load_scale, mu=A.mu, tag=A.tag)
