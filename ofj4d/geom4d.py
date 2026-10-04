"""Stage A: half-arch geometry for the 4D staging model.

Teeth (and their bonded attachments) are RIGID bodies -- only their surfaces are
kept.  The PDL is rebuilt as a constant-thickness multi-layer shell offset from
each root surface, exactly as the paper specifies (0.30 mm).  Because the shell
is a rigid offset of the root, 'restoring the PDL to its original configuration'
after each remodelling iteration is just a rigid transform of the whole shell --
no remeshing is ever required.
"""
import sys, os, pickle
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ofj'))
from scipy.spatial import cKDTree
from mesh_io import boundary_faces
from extrude import tet_vol, extrude_faces_to_tets
from subdivide import subdivide_patch
import spec

PDL_LAYERS = 3           # elements through the 0.30 mm PDL thickness


def vertex_normals(verts, faces):
    fn = np.cross(verts[faces[:, 1]]-verts[faces[:, 0]],
                  verts[faces[:, 2]]-verts[faces[:, 0]])
    vn = np.zeros_like(verts)
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    return vn/np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)


def extrude_layers(verts, faces, direction, thickness, layers):
    """Multi-layer prism->tet extrusion.  Returns nodes, tets, layer node ids."""
    nv = len(verts)
    pts = [verts]
    for L in range(1, layers+1):
        pts.append(verts + direction*(thickness*L/layers))
    nodes = np.vstack(pts)
    lids = [np.arange(nv) + L*nv for L in range(layers+1)]
    f = np.sort(faces, axis=1)              # consistent diagonals -> conforming
    tets = []
    for L in range(layers):
        b, t = lids[L], lids[L+1]
        i, j, k = b[f[:, 0]], b[f[:, 1]], b[f[:, 2]]
        I, J, K = t[f[:, 0]], t[f[:, 1]], t[f[:, 2]]
        tets.append(np.vstack([np.stack([i, j, k, K], 1),
                               np.stack([i, j, J, K], 1),
                               np.stack([i, I, J, K], 1)]))
    tets = np.vstack(tets)
    v = tet_vol(nodes, tets)
    tets[v < 0] = tets[v < 0][:, [0, 1, 3, 2]]
    return nodes, tets, lids


def face_normals(nodes, faces):
    v = nodes[faces]
    n = np.cross(v[:, 1]-v[:, 0], v[:, 2]-v[:, 0])
    a = np.linalg.norm(n, axis=1, keepdims=True)
    return n/np.maximum(a, 1e-12)


BUCCAL_MIN_DOT = 0.15
PROTRUSION = 1.0
SUBDIV = {2: 2, 3: 2, 4: 1, 5: 1, 6: 1}      # molars need a finer footprint


def build_attachment(nodes, t, size, unn):
    """Rectangular attachment on the buccal crown, mid clinical crown height.

    Same footprint rule as the validated full-arch builder: face-centroid inside a
    box shrunk by half an element edge, buccal-facing, supragingival.  Returns the
    enlarged node array, the attachment tets and its outward contact faces.  The
    attachment is rigid with its tooth, so no MPC is needed for the refined nodes.
    """
    MD, OG, TH = size
    ed, eb, eo = t['e_d'], t['e_b'], t['e_o']
    cf = t['crown_faces']
    nrm = face_normals(nodes, cf)
    cent = nodes[cf].mean(1)
    supra = (nodes[cf] @ eo >= t['gingival_level']).all(1)
    cand = supra & ((nrm @ eb) > BUCCAL_MIN_DOT)
    if cand.sum() < 4:
        raise RuntimeError(f'no buccal patch on UNN {unn}')
    s_ = cent @ ed; o_ = cent @ eo
    md_c = 0.5*(s_[cand].min() + s_[cand].max())
    o_c = t['occl_tip'] - 0.5*(t['occl_tip'] - t['gingival_level'])
    ev = nodes[cf[cand]]
    h = np.mean([np.linalg.norm(ev[:, 1]-ev[:, 0], axis=1),
                 np.linalg.norm(ev[:, 2]-ev[:, 1], axis=1),
                 np.linalg.norm(ev[:, 0]-ev[:, 2], axis=1)])
    sel = (cand & (np.abs(s_ - md_c) <= max(MD-0.5*h, MD*0.5)/2)
                & (np.abs(o_ - o_c) <= max(OG-0.5*h, OG*0.5)/2))
    faces = cf[sel]
    nodes, faces, _ = subdivide_patch(nodes, faces, levels=SUBDIV[unn])
    base = faces.copy()
    used = np.unique(faces)
    plane = (nodes[used] @ eb).max() + PROTRUSION
    disp = np.zeros_like(nodes)
    disp[used] = np.outer(plane - nodes[used] @ eb, eb)
    NL, layer = 2, []
    cur = faces
    for L in range(NL):
        d = np.zeros_like(nodes)
        d[np.unique(cur)] = disp[np.unique(cur)]/(NL-L)
        nodes, tt, topf, _ = extrude_faces_to_tets(nodes, cur, d)
        layer.append(tt); cur = topf
        disp = np.vstack([disp, np.zeros((len(nodes)-len(disp), 3))])
        disp[np.unique(cur)] = np.outer(plane - nodes[np.unique(cur)] @ eb, eb)
    tets = np.vstack(layer)
    vol = float(np.abs(tet_vol(nodes, tets)).sum())
    # outward contact surface = boundary of the attachment minus its base
    bf = boundary_faces(tets)
    baseset = set(map(tuple, np.sort(base, axis=1)))
    out_f = bf[[tuple(x) not in baseset for x in np.sort(bf, axis=1)]]
    span = (float(np.ptp(nodes[np.unique(base)] @ ed)),
            float(np.ptp(nodes[np.unique(base)] @ eo)))
    return nodes, tets, out_f, vol, span


def build(stage2='build/stage2.pkl', out='build4d/geom.pkl'):
    S = pickle.load(open(stage2, 'rb'))
    nodes, teeth = S['nodes'], S['teeth']
    keep = spec.HALF_ARCH_UNN

    print(f"half arch: {[spec.FDI[u] for u in keep]}")
    out_teeth = {}
    pdl_nodes, pdl_tets = [], []
    surf_nodes, surf_tris = [], []
    n_acc = 0

    for u in keep:
        t = teeth[u]
        tt = t['tets']
        allf = boundary_faces(tt)
        crownset = set(map(tuple, np.sort(t['crown_faces'], axis=1)))
        is_crown = np.array([tuple(x) in crownset for x in np.sort(allf, axis=1)])
        root_f = allf[~is_crown]
        crown_f = allf[is_crown]

        # ---- attachment (rigid with the tooth; its outer shell is a contact surface)
        att_f = None
        if u in spec.ATTACHMENTS:
            kind, size = spec.ATTACHMENTS[u]
            nodes, a_tets, a_faces, a_vol, a_span = build_attachment(nodes, t, size, u)
            att_f = dict(tets=a_tets, faces=a_faces, volume=a_vol, span=a_span,
                         label=kind, nominal=size)

        # ---- PDL: constant-thickness outward shell off the root surface -------
        used = np.unique(root_f)
        rm = np.full(len(nodes), -1, np.int64); rm[used] = np.arange(len(used))
        rv = nodes[used]; rf = rm[root_f]
        n = vertex_normals(rv, rf)
        # outward = away from the tooth centroid
        if np.einsum('ij,ij->i', n, rv - nodes[np.unique(tt)].mean(0)).sum() < 0:
            n = -n
        pn, pt, lids = extrude_layers(rv, rf, n, spec.PDL_THICKNESS, PDL_LAYERS)
        vol = np.abs(tet_vol(pn, pt)).sum()
        area = 0.5*np.linalg.norm(np.cross(rv[rf[:, 1]]-rv[rf[:, 0]],
                                           rv[rf[:, 2]]-rv[rf[:, 0]]), axis=1).sum()
        out_teeth[u] = dict(
            unn=u, fdi=spec.FDI[u], name=t['name'],
            e_d=t['e_d'], e_b=t['e_b'], e_o=t['e_o'],
            crown_c=t['crown_c'], occl_tip=t['occl_tip'],
            gingival_level=t['gingival_level'],
            root_faces=root_f, crown_faces=crown_f, att=att_f,
            pdl_local=(pn, pt, lids), pdl_root_area=area, pdl_vol=vol,
            centroid=nodes[np.unique(tt)].mean(0))
        print(f"  FDI {spec.FDI[u]:2d}  root {len(rf):5d} tris  crown {len(crown_f):5d} tris"
              f"  PDL {len(pt):6d} tets  {vol:6.1f} mm3  (area {area:6.1f} mm2,"
              f" mean thk {vol/area:.3f} mm)"
              + (f"  att {len(att_f['faces']):4d} tris {att_f['volume']:5.2f} mm3"
                 f" (nom {np.prod(att_f['nominal']):.1f}) span "
                 f"{att_f['span'][0]:.2f}x{att_f['span'][1]:.2f}" if att_f else ""))

    tot_pdl = sum(len(v['pdl_local'][1]) for v in out_teeth.values())
    print(f"\ntotal PDL elements {tot_pdl}")
    pickle.dump(dict(nodes=nodes, teeth=out_teeth, keep=keep), open(out, 'wb'))
    return out_teeth


if __name__ == '__main__':
    build()
