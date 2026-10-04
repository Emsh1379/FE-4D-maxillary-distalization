"""Stage 2: composite attachments on canines (vertical) and first molars (horizontal)."""
import sys, os, pickle, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from extrude import extrude_faces_to_tets, tet_vol
from subdivide import subdivide_patch, resolve_mpc

# unn -> (mesiodistal mm, occlusogingival mm, thickness mm, subdiv levels, label)
ATTACHMENTS = {
    6:  (2.0, 3.0, 1.0, 1, 'canine UR3 vertical rectangular'),
    11: (2.0, 3.0, 1.0, 1, 'canine UL3 vertical rectangular'),
    3:  (3.0, 2.0, 1.0, 2, 'molar UR6 horizontal rectangular'),
    14: (3.0, 2.0, 1.0, 2, 'molar UL6 horizontal rectangular'),
}
BUCCAL_MIN_DOT = 0.15       # face must face buccally
PROTRUSION     = 1.0        # mm beyond the most prominent point of the patch


def face_normals(nodes, faces):
    v = nodes[faces]
    n = np.cross(v[:,1]-v[:,0], v[:,2]-v[:,0])
    a = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.maximum(a, 1e-12), a[:,0] * 0.5


def build(stage1='build/stage1.pkl', out='build/stage2.pkl'):
    S = pickle.load(open(stage1, 'rb'))
    nodes, teeth = S['nodes'], S['teeth']
    att = {}
    report = []
    n_orig = len(nodes)
    all_mpc = {}
    for unn, (MD, OG, TH, LV, label) in ATTACHMENTS.items():
        t = teeth[unn]
        cf = t['crown_faces']
        ed, eb, eo = t['e_d'], t['e_b'], t['e_o']
        nrm, area = face_normals(nodes, cf)
        cent = nodes[cf].mean(1)
        supra = (nodes[cf] @ eo >= t['gingival_level']).all(1)
        buccal = (nrm @ eb) > BUCCAL_MIN_DOT
        cand = supra & buccal
        s = cent @ ed
        o = cent @ eo
        md_c = 0.5 * (s[cand].min() + s[cand].max())          # mesiodistal centre of buccal face
        o_c = t['occl_tip'] - 0.5 * (t['occl_tip'] - t['gingival_level'])  # mid clinical crown
        # shrink the selection box by one element edge so the centroid criterion
        # yields a patch whose span matches the nominal attachment size
        ev = nodes[cf[cand]]
        h = np.mean([np.linalg.norm(ev[:,1]-ev[:,0],axis=1), np.linalg.norm(ev[:,2]-ev[:,1],axis=1),
                     np.linalg.norm(ev[:,0]-ev[:,2],axis=1)])
        sel = cand & (np.abs(s - md_c) <= max(MD-0.5*h, MD*0.5)/2) & (np.abs(o - o_c) <= max(OG-0.5*h, OG*0.5)/2)
        faces = cf[sel]
        # refine the footprint so the attachment is adequately meshed
        nodes, faces, mpc = subdivide_patch(nodes, faces, levels=LV)
        all_mpc.update(mpc)
        # flat outer face 1 mm beyond the most prominent patch point
        used = np.unique(faces)
        plane = (nodes[used] @ eb).max() + PROTRUSION
        disp = np.zeros_like(nodes)
        disp[used] = np.outer(plane - nodes[used] @ eb, eb)
        # two element layers through the attachment thickness (better bending)
        NL = 2
        layer_tets = []
        cur_faces = faces
        for L in range(NL):
            d = np.zeros_like(nodes); d[np.unique(cur_faces)] = disp[np.unique(cur_faces)] / (NL - L)
            nodes, tt, topf, _ = extrude_faces_to_tets(nodes, cur_faces, d)
            layer_tets.append(tt)
            cur_faces = topf
            disp = np.vstack([disp, np.zeros((len(nodes)-len(disp), 3))])
            disp[np.unique(cur_faces)] = np.outer(plane - nodes[np.unique(cur_faces)] @ eb, eb)
        tets = np.vstack(layer_tets)
        vol = np.abs(tet_vol(nodes, tets)).sum()
        att[unn] = dict(tets=tets, faces=faces, top_faces=topf, label=label,
                        nominal=(MD, OG, TH), volume=vol,
                        centre=nodes[used].mean(0) + eb*PROTRUSION/2)
        report.append((unn, label, len(faces), vol, MD*OG*TH,
                       np.ptp(nodes[used] @ ed), np.ptp(nodes[used] @ eo), len(tets)))
    S['nodes'] = nodes
    S['attachments'] = att
    S['att_mpc'] = resolve_mpc(all_mpc, n_orig)
    S['n_orig_nodes'] = n_orig
    pickle.dump(S, open(out, 'wb'))
    return report, nodes


if __name__ == '__main__':
    rep, nodes = build()
    print(f"total nodes after attachments: {len(nodes)}\n")
    print("UNN  label                              baseTri  tets  vol(mm3) nominal MDspan OGspan")
    for unn, label, nf, vol, nom, mds, ogs, nt in rep:
        print(f"{unn:3d}  {label:<34} {nf:6d} {nt:5d}  {vol:7.2f}  {nom:5.1f}  {mds:5.2f}  {ogs:5.2f}")
