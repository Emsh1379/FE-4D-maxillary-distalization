"""Loader for Open-Full-Jaw gmsh 2.2 ASCII volumetric meshes, with npz caching."""
import os, numpy as np

def load_msh22(path, cache=None):
    if cache and os.path.exists(cache):
        d = np.load(cache)
        return d['nodes'], d['tets'], d['tags']
    with open(path) as f:
        txt = f.read()
    i = txt.index('$Nodes'); j = txt.index('$EndNodes')
    body = txt[i:j].split('\n')
    nn = int(body[1])
    arr = np.fromstring(' '.join(body[2:2+nn]), sep=' ').reshape(nn, 4)
    ids = arr[:, 0].astype(np.int64)
    nodes = np.ascontiguousarray(arr[:, 1:4])
    # build id -> index map (ids are 1..nn in this dataset but do not assume)
    maxid = ids.max()
    lut = np.full(maxid + 1, -1, dtype=np.int64)
    lut[ids] = np.arange(nn)

    i = txt.index('$Elements'); j = txt.index('$EndElements')
    body = txt[i:j].split('\n')
    ne = int(body[1])
    rows = body[2:2+ne]
    # all elements are type 4 (tet4) with 2 tags -> 9 ints per row
    a = np.fromstring(' '.join(rows), sep=' ', dtype=np.float64).reshape(ne, 9).astype(np.int64)
    assert (a[:, 1] == 4).all(), 'expected only tet4'
    tags = a[:, 4]                      # second tag = physical/material id
    tets = lut[a[:, 5:9]]
    assert tets.min() >= 0
    if cache:
        np.savez_compressed(cache, nodes=nodes, tets=tets, tags=tags)
    return nodes, tets, tags


def tet_volumes(nodes, tets):
    p = nodes[tets]
    return np.einsum('ij,ij->i', np.cross(p[:,1]-p[:,0], p[:,2]-p[:,0]), p[:,3]-p[:,0]) / 6.0


def boundary_faces(tets, return_owner=False):
    """Return faces that appear exactly once (the boundary of the given tet set)."""
    f = np.concatenate([tets[:, [0,2,1]], tets[:, [0,1,3]],
                        tets[:, [1,2,3]], tets[:, [0,3,2]]], axis=0)
    owner = np.tile(np.arange(len(tets)), 4)
    s = np.sort(f, axis=1)
    _, idx, cnt = np.unique(s, axis=0, return_index=True, return_counts=True)
    keep = idx[cnt == 1]
    return (f[keep], owner[keep]) if return_owner else f[keep]


def connected_components(tets, nnodes):
    """Label tets into connected components via shared nodes (union-find)."""
    parent = np.arange(nnodes)
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for t in tets:
        r = find(t[0])
        for k in (1, 2, 3):
            r2 = find(t[k])
            if r2 != r:
                parent[r2] = r
    roots = np.array([find(i) for i in range(nnodes)])
    return roots
