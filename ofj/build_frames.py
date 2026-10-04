"""Stage 1: label teeth, build per-tooth clinical frames, define the gingival trim line."""
import sys, os, pickle, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from mesh_io import load_msh22, boundary_faces
from anatomy import split_bodies, load_axes, UNN_NAME, FDI, TAG_TEETH, TAG_PDL, TAG_BONE

# clinical (supragingival) crown height, mm -- standard maxillary values
CLIN_CROWN_H = {2:7.0, 3:7.5, 4:8.5, 5:8.5, 6:10.0, 7:9.0, 8:10.5,
                9:10.5, 10:9.0, 11:10.0, 12:8.5, 13:8.5, 14:7.5, 15:7.0}


def orthonormal_frame(ax_x, ax_y, ax_z):
    """Orthonormal clinical frame e_d = distal, e_b = buccal, e_o = occlusal.

    The anatomical triad is mirror-image between the right and left sides, so the
    frame is deliberately NOT forced right-handed: each axis keeps its own
    anatomical direction and is only Gram-Schmidt orthogonalised.
    """
    eo = ax_z / np.linalg.norm(ax_z)                          # occlusal
    ed = ax_x - (ax_x @ eo) * eo; ed /= np.linalg.norm(ed)    # distal
    eb = ax_y - (ax_y @ eo) * eo - (ax_y @ ed) * ed           # buccal
    eb /= np.linalg.norm(eb)
    return ed, eb, eo


def build(msh, axjson, cache='build/p8_maxilla.npz', out='build/stage1.pkl'):
    nodes, tets, tags = load_msh22(msh, cache=cache)
    axes = load_axes(axjson)
    mT = tags == TAG_TEETH
    tetsT = tets[mT]
    lab, nb = split_bodies(tetsT, len(nodes))
    unns = sorted(axes)
    C = np.array([axes[u]['c'] for u in unns])
    cen = np.array([nodes[np.unique(tetsT[lab == i])].mean(0) for i in range(nb)])
    D = np.linalg.norm(cen[:, None] - C[None], axis=2)
    assign = {}
    for i in np.argsort(D.min(1)):
        for j in np.argsort(D[i]):
            if unns[j] not in assign.values():
                assign[i] = unns[j]; break
    pdl_nodes = np.zeros(len(nodes), bool)
    pdl_nodes[np.unique(tets[tags == TAG_PDL])] = True

    teeth = {}
    for b, u in assign.items():
        tt = tetsT[lab == b]
        f = boundary_faces(tt)
        crown_f = f[~pdl_nodes[f].any(1)]
        ed, eb, eo = orthonormal_frame(axes[u]['ax_x'], axes[u]['ax_y'], axes[u]['ax_z'])
        pts = nodes[np.unique(crown_f)]
        tip = (pts @ eo).max()                       # occlusal/incisal extreme
        trim = tip - CLIN_CROWN_H[u]                 # gingival margin level
        teeth[u] = dict(unn=u, fdi=FDI[u], name=UNN_NAME[u], body=b,
                        tets=tt, crown_faces=crown_f, all_faces=f,
                        e_d=ed, e_b=eb, e_o=eo, crown_c=axes[u]['c'],
                        occl_tip=tip, gingival_level=trim,
                        nodes=np.unique(tt))
    pickle.dump(dict(nodes=nodes, tets=tets, tags=tags, teeth=teeth,
                     pdl_nodes=pdl_nodes), open(out, 'wb'))
    return nodes, tets, tags, teeth


if __name__ == '__main__':
    nodes, tets, tags, teeth = build('data/P8/maxilla_volumetric_mesh.msh',
                                     'data/P8/teeth_axes_maxilla.json')
    print(f"{len(teeth)} teeth labelled\n")
    print("UNN FDI  name                 clinCrownH  gingZ   e_distal            e_buccal")
    for u in sorted(teeth):
        t = teeth[u]
        supra = (nodes[np.unique(t['crown_faces'])] @ t['e_o'] >= t['gingival_level'])
        print(f"{u:3d} {t['fdi']:>3}  {t['name']:<20} {CLIN_CROWN_H[u]:5.1f}  "
              f"{t['gingival_level']:7.2f}  {np.round(t['e_d'],3)} {np.round(t['e_b'],3)}"
              f"  supragingival nodes {supra.sum()}/{len(supra)}")
