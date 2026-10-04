"""Validation: single-tooth loading against classical orthodontic benchmarks.

A 1 N horizontal force applied at the crown of an isolated tooth (aligner removed,
no contact) must reproduce the textbook initial-displacement behaviour of a tooth
suspended in a linear-elastic PDL:
  * crown displacement of the order of 0.05-0.15 mm per newton
  * centre of rotation apical to the centre of resistance, roughly in the apical
    third to half of the root for an uncontrolled tipping force
Also checks global force equilibrium of the assembled system.
"""
import sys, os, pickle, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy.sparse.linalg import cg
from fem import assemble, rigid_body_modes
from solve import build_P, make_amg
from postprocess import rigid_fit, centre_of_rotation


def run(unn=6, force=1.0, direction='buccal', model='build/model.pkl'):
    M = pickle.load(open(model, 'rb'))
    nodes = M['nodes']; ndof = 3*len(nodes)
    # drop the aligner: keep bone + pdl + teeth + attachments
    keep = M['pid'] < 4
    elems = M['elems'][keep]; mats = M['mats'][keep]
    matmap = {m: (M['MAT'][m][1], M['MAT'][m][2]) for m in M['MAT']}
    K = assemble(nodes, elems, mats, matmap, chunk=40000, verbose=False)
    P = build_P(M, ndof)
    Kr = (P.T @ K @ P).tocsr()

    t = M['teeth'][unn]
    d = {'buccal': t['e_b'], 'distal': t['e_d'], 'occlusal': t['e_o']}[direction]
    # apply at the bracket site: buccal crown surface, mid clinical crown
    cn = np.unique(t['crown_faces'])
    o_c = t['occl_tip'] - 0.5*(t['occl_tip'] - t['gingival_level'])
    seed = t['crown_c'] + t['e_b']*6.0 + t['e_o']*(o_c - t['crown_c'] @ t['e_o'])
    dd = np.linalg.norm(nodes[cn]-seed, axis=1)
    centre = nodes[cn][np.argmin(dd)]
    sel = cn[np.linalg.norm(nodes[cn]-centre, axis=1) <= 1.5]
    if len(sel) < 10:                       # fall back to the k nearest surface nodes
        sel = cn[np.argsort(np.linalg.norm(nodes[cn]-centre, axis=1))[:40]]
    f = np.zeros(ndof)
    for k in range(3):
        np.add.at(f, 3*sel+k, force*d[k]/len(sel))
    b = P.T @ f
    Br = P.T @ rigid_body_modes(nodes)
    ml = make_amg(Kr, Br)
    x, info = cg(Kr, b, rtol=1e-10, maxiter=2000, M=ml.aspreconditioner(cycle='V'))
    res = np.linalg.norm(b - Kr@x)/np.linalg.norm(b)
    u = P @ x; U = u.reshape(-1, 3)

    nd = np.unique(t['tets'])
    tr, w, c, rr = rigid_fit(nodes[nd], U[nd])
    crot, wn = centre_of_rotation(tr, w, c)
    eo = t['e_o']
    apex = nodes[nd][np.argmax(nodes[nd] @ (-eo))]
    root_len = float(t['gingival_level'] - apex @ eo)
    du = lambda p: tr + np.cross(w, p-c)
    dc, da = du(t['crown_c']), du(apex)
    # reaction check
    return dict(unn=unn, name=t['name'], direction=direction, force=force, res=res,
                nload=int(len(sel)),
                crown_disp_mm=float(np.linalg.norm(dc)), apex_disp_mm=float(np.linalg.norm(da)),
                crown_along=float(dc @ d), apex_along=float(da @ d),
                rot_deg=float(np.degrees(wn)), rigid_resid=rr, root_len=root_len,
                crot_above_apex=float((crot-apex) @ eo) if crot is not None else None,
                crot_frac=float((crot-apex) @ eo/root_len) if crot is not None else None,
                total_disp_max=float(np.linalg.norm(U, axis=1).max()))


if __name__ == '__main__':
    print("Single-tooth validation (aligner removed, no contact; 1 N at the crown)\n")
    print(f"{'tooth':<20}{'dir':<9}{'crown um':>10}{'apex um':>9}{'rot deg':>9}"
          f"{'CRot/root':>11}{'rootLen':>9}{'rigidRes':>10}")
    rows = []
    for unn, dr in [(6, 'buccal'), (6, 'distal'), (8, 'buccal'), (3, 'distal'), (11, 'distal')]:
        r = run(unn=unn, direction=dr)
        rows.append(r)
        cf = r['crot_frac'] if r['crot_frac'] is not None else float('nan')
        print(f"{r['name']:<20}{dr:<9}{r['crown_disp_mm']*1e3:10.1f}{r['apex_disp_mm']*1e3:9.1f}"
              f"{r['rot_deg']:9.4f}{cf:11.3f}{r['root_len']:9.2f}{r['rigid_resid']:10.2e}"
              f"  nload={r['nload']}")
    pickle.dump(rows, open('build/validation.pkl', 'wb'))
