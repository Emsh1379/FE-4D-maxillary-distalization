"""Conforming prism->tet extrusion of a triangulated patch."""
import numpy as np


def tet_vol(nodes, tets):
    p = nodes[tets]
    return np.einsum('ij,ij->i', np.cross(p[:,1]-p[:,0], p[:,2]-p[:,0]), p[:,3]-p[:,0]) / 6.0


def extrude_faces_to_tets(nodes, faces, disp):
    """Extrude `faces` (indices into `nodes`) by per-vertex displacement `disp`.

    Returns (new_nodes, tets, top_faces, bottom_to_top) where new nodes are appended.
    Prisms are split into 3 tets using a sorted-index rule, which guarantees the
    quad-face diagonals agree between neighbouring prisms (conforming mesh).
    """
    used = np.unique(faces)
    remap = np.full(nodes.shape[0], -1, np.int64)
    remap[used] = np.arange(len(used))
    top_pts = nodes[used] + disp[used] if disp.shape[0] == nodes.shape[0] else nodes[used] + disp
    new_nodes = np.vstack([nodes, top_pts])
    top_id = len(nodes) + remap                    # top node id for each bottom node id

    f = np.sort(faces, axis=1)                     # i<j<k  -> consistent diagonals
    i, j, k = f[:, 0], f[:, 1], f[:, 2]
    I, J, K = top_id[i], top_id[j], top_id[k]
    tets = np.vstack([np.stack([i, j, k, K], 1),
                      np.stack([i, j, J, K], 1),
                      np.stack([i, I, J, K], 1)])
    v = tet_vol(new_nodes, tets)
    bad = v < 0
    tets[bad] = tets[bad][:, [0, 1, 3, 2]]         # fix orientation, face set unchanged
    top_faces = np.stack([top_id[faces[:,0]], top_id[faces[:,1]], top_id[faces[:,2]]], 1)
    return new_nodes, tets, top_faces, top_id
