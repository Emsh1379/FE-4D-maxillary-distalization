"""Stage 4: trim the shrink-wrap to the clinical crown and extrude the aligner shell."""
import sys, os, pickle, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from skimage import measure
from extrude import extrude_faces_to_tets, tet_vol

STEP   = 3      # marching-cubes subsampling -> ~0.45 mm triangles
THICK  = 0.50   # aligner sheet thickness, mm
GAP    = 0.00


def largest_component(verts, faces):
    e = np.vstack([faces[:,[0,1]], faces[:,[1,2]], faces[:,[2,0]]])
    g = coo_matrix((np.ones(len(e),np.int8),(e[:,0],e[:,1])), shape=(len(verts),)*2)
    n, lab = connected_components(g, directed=False)
    cnt = np.bincount(lab[faces[:,0]], minlength=n)
    keep = lab[faces[:,0]] == cnt.argmax()
    return compact(verts, faces[keep])


def compact(verts, faces):
    used = np.unique(faces)
    rm = np.full(len(verts), -1, np.int64); rm[used] = np.arange(len(used))
    return verts[used], rm[faces]


def build(stage2='build/stage2.pkl', out='build/stage4.pkl'):
    S = pickle.load(open(stage2,'rb'))
    nodes, teeth, att = S['nodes'], S['teeth'], S['attachments']
    G = pickle.load(open('build/grid.pkl','rb')); lo, vox = G['lo'], G['vox']
    phi = np.load('build/phi.npy')

    verts, faces, _, _ = measure.marching_cubes(phi, level=GAP, spacing=(vox,)*3,
                                                step_size=STEP)
    verts = verts + lo
    print(f"wrap @step{STEP}: {len(verts)} verts {len(faces)} tris")

    # ---- assign each vertex to the nearest tooth, then trim at the gingival margin ----
    tooth_ids, pts_all, eo_all, ging = [], [], {}, {}
    for u, t in teeth.items():
        cn = np.unique(t['crown_faces'])
        pts_all.append(nodes[cn]); tooth_ids.append(np.full(len(cn), u))
        eo_all[u] = t['e_o']; ging[u] = t['gingival_level']
    pts_all = np.vstack(pts_all); tooth_ids = np.concatenate(tooth_ids)
    _, nn = cKDTree(pts_all).query(verts)
    vt = tooth_ids[nn]
    EO = np.array([eo_all[u] for u in vt]); GG = np.array([ging[u] for u in vt])
    supra = np.einsum('ij,ij->i', verts, EO) >= GG
    keep = supra[faces].all(1)
    verts, faces = compact(verts, faces[keep])
    verts, faces = largest_component(verts, faces)
    print(f"trimmed to clinical crown: {len(verts)} verts {len(faces)} tris")

    # ---- outward offset along grad(phi) (|grad phi| = 1 for a signed distance) ----
    gz = np.gradient(phi, vox)
    ijk = ((verts - lo)/vox).T
    g = np.stack([ndimage.map_coordinates(a, ijk, order=1, mode='nearest') for a in gz], 1)
    del gz, phi
    g /= np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9)

    # orient faces outward (normal . grad phi > 0)
    v = verts[faces]
    fn = np.cross(v[:,1]-v[:,0], v[:,2]-v[:,0])
    if np.einsum('ij,ij->i', fn, g[faces[:,0]]).sum() < 0:
        faces = faces[:, [0,2,1]]

    scale = np.full(len(verts), THICK)
    for it in range(12):
        disp = g * scale[:,None]
        nv, tets, topf, _ = extrude_faces_to_tets(verts, faces, disp)
        bad = tet_vol(nv, tets) <= 1e-9
        if not bad.any():
            break
        badv = np.unique(tets[bad][:, :3] % len(verts))
        badv = np.unique(np.concatenate([faces[np.isin(faces, np.unique(tets[bad])).any(1)].ravel(),
                                         badv[badv < len(verts)]]))
        scale[badv] *= 0.7
        print(f"  offset iter {it}: {bad.sum()} inverted -> relax {len(badv)} verts")
    vol = np.abs(tet_vol(nv, tets)).sum()
    print(f"aligner: {len(nv)-len(verts)+len(verts)} nodes {len(tets)} tets "
          f"volume {vol:.1f} mm3  (mean thickness {vol/ (0.5*np.linalg.norm(np.cross(verts[faces][:,1]-verts[faces][:,0], verts[faces][:,2]-verts[faces][:,0]),axis=1).sum()):.3f} mm)")
    S['aligner'] = dict(nodes=nv, tets=tets, inner_faces=faces, outer_faces=topf,
                        n_inner=len(verts), vert_tooth=vt[:0], thickness=THICK)
    S['aligner_inner_verts'] = verts
    pickle.dump(S, open(out,'wb'))
    return nv, tets, faces


if __name__ == '__main__':
    build()
