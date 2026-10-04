"""Loop-free 1->4 triangle subdivision producing midpoint nodes tied by exact MPCs.

For linear (T3/T4) elements the displacement field along an edge is linear, so
constraining a new edge-midpoint node to the average of its two parents is exact:
the refined patch stays perfectly compatible with the coarse parent surface.
"""
import numpy as np


def subdivide_patch(nodes, faces, levels=1):
    """Return (nodes, faces, mpc) where mpc maps new node id -> (pa, pb) with w=0.5."""
    mpc = {}
    for _ in range(levels):
        e = np.vstack([faces[:, [0,1]], faces[:, [1,2]], faces[:, [2,0]]])
        e = np.sort(e, axis=1)
        uniq, inv = np.unique(e, axis=0, return_inverse=True)
        mid_id = len(nodes) + np.arange(len(uniq))
        nodes = np.vstack([nodes, 0.5 * (nodes[uniq[:,0]] + nodes[uniq[:,1]])])
        for m, (a, b) in zip(mid_id, uniq):
            mpc[int(m)] = (int(a), int(b))
        nf = len(faces)
        m01 = mid_id[inv[:nf]]; m12 = mid_id[inv[nf:2*nf]]; m20 = mid_id[inv[2*nf:]]
        faces = np.vstack([
            np.stack([faces[:,0], m01, m20], 1),
            np.stack([m01, faces[:,1], m12], 1),
            np.stack([m20, m12, faces[:,2]], 1),
            np.stack([m01, m12, m20], 1)])
    return nodes, faces, mpc


def resolve_mpc(mpc, n_orig):
    """Flatten recursive midpoint constraints to weights on original (master) nodes."""
    out = {}
    def rec(nid):
        if nid < n_orig:
            return {nid: 1.0}
        if nid in out:
            return out[nid]
        a, b = mpc[nid]
        w = {}
        for src, f in ((a, 0.5), (b, 0.5)):
            for k, v in rec(src).items():
                w[k] = w.get(k, 0.0) + f * v
        out[nid] = w
        return w
    import sys
    lim = sys.getrecursionlimit(); sys.setrecursionlimit(max(lim, 100000))
    for nid in mpc:
        rec(nid)
    sys.setrecursionlimit(lim)
    return out
