"""Stage A': the right half of the alveolar bone, for the deformable-bone variant.

Source: the Open-Full-Jaw patient 8 maxilla volumetric mesh (fTetWild, conforming
teeth / PDL / bone).  The rigid-bone variant follows Mao et al. and does not mesh
bone at all; this stage supplies it so the same model can be solved with the bone
deforming, and the effect of the paper's rigid-bone assumption measured.

  domain      bone tets whose centroid lies on the right of the midsagittal plane
              (the plane through the two central incisor crown centres, the same
              one the aligner's symmetry constraint uses)
  materials   the segmentation has one bone label, so bone within CORTICAL_T of the
              outer bone surface is cortical and the rest cancellous -- the values
              and the 1.5 mm shell of the package's full-arch model (build_model.py)
  fixed       the dataset authors' own 'Bone_fixed_region' (their FEBio setup), on
              the right side -- the superior maxilla, far from the teeth
  symmetry    nodes on the midsagittal cut: normal displacement zero

The outer bone surface is the part of the boundary of the WHOLE jaw mesh that
belongs to bone, so socket walls (bone-PDL interfaces) do not count as outer.
"""
import sys, os, pickle
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ofj'))
from scipy.spatial import cKDTree
from mesh_io import load_msh22, boundary_faces
from anatomy import load_axes, TAG_TEETH, TAG_PDL, TAG_BONE
from extrude import tet_vol

MSH = 'data/P8/maxilla_volumetric_mesh.msh'
AXES = 'data/P8/teeth_axes_maxilla.json'
CACHE = os.environ.get('OFJ_MSH_CACHE', 'data/cache/p8_maxilla.npz')
FIXED_XYZ = os.environ.get('OFJ_BONE_FIXED', 'data/cache/bone_fixed_region_xyz.npy')
CORTICAL_T = 1.5                    # mm
E_CORTICAL, E_CANCELLOUS, NU_BONE = 13700.0, 1370.0, 0.30
SYM_BAND = 0.6                      # mm either side of the midsagittal plane


def unit(v):
    v = np.asarray(v, float)
    return v/np.linalg.norm(v)


def quality(nodes, tets):
    """Normalised tet quality 6*sqrt(2)*V / l_rms^3 (1 = regular)."""
    X = nodes[tets]
    V = np.abs(tet_vol(nodes, tets))
    e = [X[:, j]-X[:, i] for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))]
    lrms = np.sqrt(np.mean([np.einsum('ij,ij->i', x, x) for x in e], axis=0))
    return 6*np.sqrt(2)*V/lrms**3


def build(out='build4d/bone.pkl'):
    nodes, tets, tags = load_msh22(MSH, cache=CACHE if os.path.isdir(os.path.dirname(CACHE)) else None)
    axes = load_axes(AXES)
    n_sag = unit(np.asarray(axes[8]['c']) - np.asarray(axes[9]['c']))
    off = float(0.5*(np.asarray(axes[8]['c']) + np.asarray(axes[9]['c'])) @ n_sag)

    bt_all = tets[tags == TAG_BONE]
    right = (nodes[bt_all].mean(1) @ n_sag - off) > 0
    bt = bt_all[right]

    whole_f, own = boundary_faces(tets, return_owner=True)
    outer_f = whole_f[tags[own] == TAG_BONE]
    outer_pts = nodes[np.unique(outer_f)]
    d_surf, _ = cKDTree(outer_pts).query(nodes[bt].mean(1))
    cortical = d_surf <= CORTICAL_T
    E = np.where(cortical, E_CORTICAL, E_CANCELLOUS)

    used = np.unique(bt)
    rm = np.full(len(nodes), -1, np.int64); rm[used] = np.arange(len(used))
    bn, btt = nodes[used], rm[bt]

    # The authors' .feb was meshed separately -- its nodes sit a median 0.14 mm from
    # ours, so matching node-by-node recovers only 58 of their 22,142 and would leave
    # the bone on a few point supports.  Their region is a superior slab: every node
    # at or above its lowest point.  Our mesh has 22,257 nodes there against their
    # 22,142, so the slab is the region, reproduced on this mesh.
    FX = np.load(FIXED_XYZ)
    z_fix = float(FX[:, 2].min())
    fixed = bn[:, 2] >= z_fix
    if not fixed.any():
        raise RuntimeError('no fixed nodes on the right half -- the bone would float')

    bnd = np.zeros(len(bn), bool); bnd[np.unique(boundary_faces(btt))] = True
    sym = bnd & (np.abs(bn @ n_sag - off) < SYM_BAND) & ~fixed

    socket = np.zeros(len(nodes), bool); socket[np.unique(tets[tags == TAG_PDL])] = True
    socket = socket[used]

    V = np.abs(tet_vol(bn, btt)); q = quality(bn, btt)
    pdl_pts = nodes[np.unique(tets[tags == TAG_PDL])]
    pdl_right = pdl_pts[(pdl_pts @ n_sag - off) > 0]
    d_fix_pdl = cKDTree(bn[fixed]).query(pdl_right)[0].min()

    print(f"right-half bone: {len(btt)} tets, {len(bn)} nodes")
    print(f"  cortical (<= {CORTICAL_T} mm from the outer surface): "
          f"{cortical.sum()} tets, {100*V[cortical].sum()/V.sum():.1f}% of the volume")
    print(f"  fixed (dataset Bone_fixed_region slab, z >= {z_fix:.2f} mm, right side): {fixed.sum()} nodes;"
          f" nearest to any right-side PDL {d_fix_pdl:.1f} mm")
    print(f"  symmetry (on the midsagittal cut): {sym.sum()} nodes")
    print(f"  socket-wall nodes (shared with the PDL): {socket.sum()}")
    print(f"  tet quality: median {np.median(q):.3f}, p1 {np.percentile(q, 1):.3f}, "
          f"{(q < 0.05).sum()} below 0.05;  volume {V.sum():.0f} mm3")

    os.makedirs(os.path.dirname(out), exist_ok=True)
    pickle.dump(dict(nodes=bn, tets=btt, E=E, cortical=cortical, fixed=fixed, sym=sym,
                     socket=socket, n_sag=n_sag, sag_offset=off, src_ids=used, z_fixed=z_fix,
                     materials=dict(E_cortical=E_CORTICAL, E_cancellous=E_CANCELLOUS,
                                    nu=NU_BONE, cortical_t=CORTICAL_T),
                     source=MSH), open(out, 'wb'))
    print(f"wrote {out}")


if __name__ == '__main__':
    build()
