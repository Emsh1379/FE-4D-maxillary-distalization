"""Stage 3: clear aligner geometry.

The aligner is generated as a thermoforming-realistic shrink-wrap of the dentition:
  1. voxelise the solid (teeth + bonded attachments)
  2. morphological CLOSING with a rolling ball of radius R -- bridges interproximal
     embrasures into one continuous appliance and rounds concavities the sheet
     cannot reproduce during thermoforming
  3. signed distance field of the closed solid; inner aligner surface = {phi = GAP}
  4. trim to the clinical crown (gingival margin per tooth)
  5. offset outward by the sheet THICKNESS along grad(phi) -> prisms -> tets
"""
import sys, os, pickle, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage import measure
from mesh_io import boundary_faces
from extrude import extrude_faces_to_tets, tet_vol

VOX       = 0.15   # mm, voxel size
R_CLOSE   = 0.75   # mm, thermoforming rolling-ball radius (bridges <=1.5 mm embrasures)
GAP       = 0.00   # mm, passive-fit clearance on the inner surface
THICK     = 0.50   # mm, aligner sheet thickness
PAD       = 3.0    # mm, grid padding


def rasterize_tris(tri_pts, origin, shape, vox):
    """Mark voxels crossed by triangles by dense barycentric sampling."""
    S = np.zeros(shape, bool)
    e1 = tri_pts[:,1]-tri_pts[:,0]; e2 = tri_pts[:,2]-tri_pts[:,0]
    L = max(np.linalg.norm(e1,axis=1).max(), np.linalg.norm(e2,axis=1).max())
    n = int(np.ceil(L / (0.35*vox))) + 1
    a, b = np.meshgrid(np.linspace(0,1,n), np.linspace(0,1,n))
    m = (a+b) <= 1.0
    a = a[m]; b = b[m]
    for k in range(0, len(tri_pts), 20000):
        P = (tri_pts[k:k+20000,0][:,None,:]
             + e1[k:k+20000][:,None,:]*a[None,:,None]
             + e2[k:k+20000][:,None,:]*b[None,:,None]).reshape(-1,3)
        idx = np.round((P-origin)/vox).astype(np.int32)
        ok = ((idx>=0)&(idx<np.array(shape))).all(1)
        idx = idx[ok]
        S[idx[:,0], idx[:,1], idx[:,2]] = True
    return S


def fill_interior(S):
    """Flood-fill from the grid border; everything unreached is inside a closed shell."""
    free = ~S
    lab, _ = ndimage.label(free, structure=ndimage.generate_binary_structure(3,1))
    border = set(np.unique(np.concatenate([
        lab[0].ravel(), lab[-1].ravel(), lab[:,0].ravel(), lab[:,-1].ravel(),
        lab[:,:,0].ravel(), lab[:,:,-1].ravel()])))
    border.discard(0)
    outside = np.isin(lab, list(border))
    return ~outside          # solid = surface + everything not connected to outside


def build(stage2='build/stage2.pkl', out='build/stage3.pkl'):
    S = pickle.load(open(stage2,'rb'))
    nodes, tets, tags, teeth, att = S['nodes'], S['tets'], S['tags'], S['teeth'], S['attachments']

    # ---- closed outer surface of each tooth WITH its bonded attachment ----
    tri = []
    for u, t in teeth.items():
        tt = t['tets']
        if u in att:
            tt = np.vstack([tt, att[u]['tets']])
        tri.append(boundary_faces(tt))
    tri = np.vstack(tri)
    print(f"dental surface: {len(tri)} triangles")

    pts = nodes[np.unique(tri)]
    lo = pts.min(0) - PAD; hi = pts.max(0) + PAD
    shape = tuple(np.ceil((hi-lo)/VOX).astype(int) + 1)
    print(f"grid {shape} = {np.prod(shape)/1e6:.1f} M voxels, extent {np.round(hi-lo,1)} mm")

    Sv = rasterize_tris(nodes[tri], lo, shape, VOX)
    print(f"  surface voxels {Sv.sum()/1e6:.2f} M")
    solid = fill_interior(Sv); del Sv
    print(f"  solid voxels   {solid.sum()/1e6:.2f} M  ({solid.sum()*VOX**3:.0f} mm3)")

    # ---- morphological closing ----
    d_out = ndimage.distance_transform_edt(~solid, sampling=VOX)
    D = d_out <= R_CLOSE; del d_out
    d_in = ndimage.distance_transform_edt(D, sampling=VOX)
    closed = d_in > R_CLOSE; del d_in, D
    print(f"  closed solid   {closed.sum()/1e6:.2f} M  ({closed.sum()*VOX**3:.0f} mm3)")

    # ---- signed distance of the closed solid (positive outside) ----
    phi = (ndimage.distance_transform_edt(~closed, sampling=VOX)
           - ndimage.distance_transform_edt(closed, sampling=VOX))
    del closed
    verts, faces, _, _ = measure.marching_cubes(phi, level=GAP, spacing=(VOX,VOX,VOX))
    verts = verts + lo
    print(f"  shrink-wrap surface: {len(verts)} verts, {len(faces)} tris")
    pickle.dump(dict(phi_shape=phi.shape, lo=lo, vox=VOX), open('build/grid.pkl','wb'))
    np.save('build/phi.npy', phi.astype(np.float32))
    np.save('build/wrap_verts.npy', verts); np.save('build/wrap_faces.npy', faces)
    S['wrap'] = (verts, faces); S['grid'] = (lo, VOX, phi.shape)
    pickle.dump(S, open(out,'wb'))
    return verts, faces


if __name__ == '__main__':
    build()
