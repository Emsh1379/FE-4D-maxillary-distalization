"""Stage B: 0.70 mm clear aligner over the right half arch.

The paper describes the CA as 'an external offset of dental crowns with a
thickness of 0.7 mm'.  A literal offset of seven separated crowns is not a
connected appliance, so the offset is preceded by a morphological closing with a
rolling ball -- the operation a thermoformed sheet actually performs as it
bridges the interproximal embrasures.  Attachments are present before the
closing, so the aligner forms its own pockets around them.
"""
import sys, os, pickle
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ofj'))
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from skimage import measure
from extrude import tet_vol
from build_aligner import rasterize_tris, fill_interior
from geom4d import extrude_layers
import spec

VOX = 0.15
R_CLOSE = 0.75
GAP = 0.00
PAD = 3.0
STEP = int(os.environ.get('OFJ_ALIGNER_STEP', 3))        # marching-cubes subsampling -> ~0.45 mm triangles
SHELL_LAYERS = int(os.environ.get('OFJ_SHELL_LAYERS', 2)) # elements through the 0.70 mm sheet


def compact(verts, faces):
    used = np.unique(faces)
    rm = np.full(len(verts), -1, np.int64); rm[used] = np.arange(len(used))
    return verts[used], rm[faces]


def largest_component(verts, faces):
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    g = coo_matrix((np.ones(len(e), np.int8), (e[:, 0], e[:, 1])), shape=(len(verts),)*2)
    n, lab = connected_components(g, directed=False)
    cnt = np.bincount(lab[faces[:, 0]], minlength=n)
    return compact(verts, faces[lab[faces[:, 0]] == cnt.argmax()])


def build(geom='build4d/geom.pkl', out='build4d/aligner.pkl'):
    G = pickle.load(open(geom, 'rb'))
    nodes, teeth = G['nodes'], G['teeth']

    # the voxeliser needs a CLOSED surface per tooth, so use the whole tooth
    # boundary (root + crown) plus the attachment; the shell is trimmed to the
    # clinical crown further down.
    tri = []
    for u, t in teeth.items():
        tri.append(t['root_faces']); tri.append(t['crown_faces'])
        if t['att'] is not None:
            tri.append(t['att']['faces'])
    tri = np.vstack(tri)
    print(f"dental surface (whole teeth + attachments): {len(tri)} triangles")

    pts = nodes[np.unique(tri)]
    lo = pts.min(0) - PAD; hi = pts.max(0) + PAD
    shape = tuple(np.ceil((hi-lo)/VOX).astype(int) + 1)
    print(f"grid {shape} = {np.prod(shape)/1e6:.1f} M voxels")

    Sv = rasterize_tris(nodes[tri], lo, shape, VOX)
    solid = fill_interior(Sv); del Sv
    print(f"  solid {solid.sum()*VOX**3:.0f} mm3")
    d_out = ndimage.distance_transform_edt(~solid, sampling=VOX)
    D = d_out <= R_CLOSE; del d_out, solid
    d_in = ndimage.distance_transform_edt(D, sampling=VOX)
    closed = d_in > R_CLOSE; del d_in, D
    print(f"  closed {closed.sum()*VOX**3:.0f} mm3")
    phi = (ndimage.distance_transform_edt(~closed, sampling=VOX)
           - ndimage.distance_transform_edt(closed, sampling=VOX))
    del closed

    verts, faces, _, _ = measure.marching_cubes(phi, level=GAP, spacing=(VOX,)*3,
                                                step_size=STEP)
    verts = verts + lo
    print(f"wrap: {len(verts)} verts {len(faces)} tris")

    # ---- assign vertices to teeth, trim at the gingival margin ----------------
    pa, ti = [], []
    for u, t in teeth.items():
        cn = np.unique(t['crown_faces'])
        pa.append(nodes[cn]); ti.append(np.full(len(cn), u))
        if t['att'] is not None:
            an = np.unique(t['att']['faces'])
            pa.append(nodes[an]); ti.append(np.full(len(an), u))
    pa = np.vstack(pa); ti = np.concatenate(ti)
    _, nn = cKDTree(pa).query(verts)
    vt = ti[nn]
    EO = np.array([teeth[u]['e_o'] for u in vt])
    GG = np.array([teeth[u]['gingival_level'] for u in vt])
    supra = np.einsum('ij,ij->i', verts, EO) >= GG
    keep = supra[faces].all(1)
    print(f"  supragingival: {keep.sum()}/{len(faces)} tris")
    verts, faces = compact(verts, faces[keep])
    verts, faces = largest_component(verts, faces)
    # drop degenerate slivers - no offset scaling can rescue them
    v = verts[faces]
    a = 0.5*np.linalg.norm(np.cross(v[:, 1]-v[:, 0], v[:, 2]-v[:, 0]), axis=1)
    verts, faces = compact(verts, faces[a > 1e-6])
    _, nn = cKDTree(pa).query(verts)
    vt = ti[nn]
    print(f"trimmed: {len(verts)} verts {len(faces)} tris")

    # ---- outward offset along grad(phi) ---------------------------------------
    gz = np.gradient(phi, VOX)
    ijk = ((verts - lo)/VOX).T
    g = np.stack([ndimage.map_coordinates(a, ijk, order=1, mode='nearest') for a in gz], 1)
    del gz, phi
    g /= np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9)
    v = verts[faces]
    fn = np.cross(v[:, 1]-v[:, 0], v[:, 2]-v[:, 0])
    if np.einsum('ij,ij->i', fn, g[faces[:, 0]]).sum() < 0:
        faces = faces[:, [0, 2, 1]]

    scale = np.full(len(verts), spec.ALIGNER_THICKNESS)
    for it in range(14):
        an, at, lids = extrude_layers_var(verts, faces, g, scale, SHELL_LAYERS)
        bad = tet_vol(an, at) <= 1e-10
        if not bad.any():
            break
        touched = np.unique(at[bad]) % len(verts)
        scale[touched[touched < len(verts)]] *= 0.7
        print(f"  offset iter {it}: {bad.sum()} inverted -> relax {len(touched)} verts")
    vol = float(np.abs(tet_vol(an, at)).sum())
    area = 0.5*np.linalg.norm(np.cross(verts[faces[:, 1]]-verts[faces[:, 0]],
                                       verts[faces[:, 2]]-verts[faces[:, 0]]), axis=1).sum()
    print(f"aligner: {len(an)} nodes {len(at)} tets  vol {vol:.1f} mm3"
          f"  area {area:.0f} mm2  mean thickness {vol/area:.3f} mm")

    A = dict(nodes=an, tets=at, inner_faces=faces, n_inner=len(verts),
             layer_ids=lids, vert_tooth=vt, thickness=spec.ALIGNER_THICKNESS,
             inner_verts=verts, offset_dir=g, offset_scale=scale)
    pickle.dump(A, open(out, 'wb'))
    return A


def extrude_layers_var(verts, faces, direction, scale, layers):
    """Layered extrusion with a per-vertex offset distance."""
    nv = len(verts)
    pts = [verts]
    for L in range(1, layers+1):
        pts.append(verts + direction*(scale[:, None]*L/layers))
    nodes = np.vstack(pts)
    lids = [np.arange(nv) + L*nv for L in range(layers+1)]
    f = np.sort(faces, axis=1)
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


if __name__ == '__main__':
    build(out=sys.argv[1] if len(sys.argv) > 1 else 'build4d/aligner.pkl')
