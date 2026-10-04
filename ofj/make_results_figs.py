"""Stage 9: result figures."""
import sys, os, pickle
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mesh_io import boundary_faces
from fem import element_strain_stress
from postprocess import CAPILLARY_MPA

OUT = 'out/results'
os.makedirs(OUT, exist_ok=True)
ORDER = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
SHORT = {2: '17', 3: '16', 4: '15', 5: '14', 6: '13', 7: '12', 8: '11',
         9: '21', 10: '22', 11: '23', 12: '24', 13: '25', 14: '26', 15: '27'}


def surf_plot(ax, pts, tris, val, cmap, norm, light=np.array([0.35, -0.55, -0.75])):
    v = pts[tris]
    n = np.cross(v[:, 1]-v[:, 0], v[:, 2]-v[:, 0])
    L = np.linalg.norm(n, axis=1)
    k = L > 1e-12
    v, n, val = v[k], n[k]/L[k, None], val[k]
    b = 0.45 + 0.55*np.clip(n @ (light/np.linalg.norm(light)), 0, 1)
    c = plt.get_cmap(cmap)(norm(val))[:, :3]*b[:, None]
    o = np.argsort(v[:, :, 1].mean(1))
    ax.add_collection3d(Poly3DCollection(v[o], facecolors=np.clip(c[o], 0, 1), edgecolors='none'))


def frame(ax, pts, zoom=0.72):
    c = pts.mean(0); r = np.ptp(pts, 0).max()/2*zoom
    ax.set_xlim(c[0]-r, c[0]+r); ax.set_ylim(c[1]-r, c[1]+r); ax.set_zlim(c[2]-r, c[2]+r)
    ax.set_axis_off(); ax.set_box_aspect((1, 1, 1))


def fig_bars(R, path):
    fig, axs = plt.subplots(2, 2, figsize=(14, 8.5), facecolor='white')
    x = np.arange(len(ORDER)); lab = [SHORT[u] for u in ORDER]
    cd = np.array([R['teeth'][u]['crown_distal']*1e3 for u in ORDER])
    ad = np.array([R['teeth'][u]['apex_distal']*1e3 for u in ORDER])
    cb = np.array([R['teeth'][u]['crown_buccal']*1e3 for u in ORDER])
    fd = np.array([R['teeth'][u]['contact_distal'] for u in ORDER])
    tp = np.array([R['teeth'][u]['tip_deg'] for u in ORDER])
    cr = np.array([R['teeth'][u]['crot_frac_root'] for u in ORDER])

    a = axs[0, 0]
    a.bar(x-0.2, cd, 0.4, label='crown', color='#2f7fd0')
    a.bar(x+0.2, ad, 0.4, label='root apex', color='#d98032')
    a.axhline(0, c='k', lw=.8); a.set_xticks(x); a.set_xticklabels(lab)
    a.set_ylabel('distal displacement (um)')
    a.set_title('Crown moves distally, apex moves mesially\n= distal crown tipping', fontsize=10)
    a.legend(fontsize=8); a.grid(axis='y', alpha=.3)

    a = axs[0, 1]
    a.bar(x, fd, .6, color='#3b8f5a')
    a.set_xticks(x); a.set_xticklabels(lab); a.grid(axis='y', alpha=.3)
    a.set_ylabel('distal force delivered (N)')
    a.set_title(f'Distribution of the {abs(fd).sum():.2f} N total distal force\n'
                f'across the arch (500 gf applied)', fontsize=10)

    a = axs[1, 0]
    a.bar(x, cb, .6, color='#9b59b6')
    a.axhline(0, c='k', lw=.8); a.set_xticks(x); a.set_xticklabels(lab)
    a.set_ylabel('buccal(+) / palatal(-) displacement (um)')
    a.set_title('Transverse side-effect: arch constriction', fontsize=10)
    a.grid(axis='y', alpha=.3)

    a = axs[1, 1]
    a.bar(x, cr, .6, color='#c0392b')
    a.set_xticks(x); a.set_xticklabels(lab); a.set_ylim(0, 1)
    a.axhline(.5, ls='--', c='k', lw=.8)
    a.text(0.2, .52, 'mid-root', fontsize=8)
    a.set_ylabel('CRot height / root length  (0 = apex, 1 = CEJ)')
    a.set_title('Centre of rotation: high CRot = more crown tipping', fontsize=10)
    a.grid(axis='y', alpha=.3)
    fig.suptitle('Maxillary en-masse distalization, clear aligner + 4 attachments, 250 gf per side',
                 fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, .95]); plt.savefig(path, dpi=125, facecolor='white')
    plt.close(fig)


def fig_fields(M, S, R, path):
    nodes = M['nodes']; tets = M['tets']; tags = M['tags']
    U = S['u'].reshape(-1, 3)
    matmap = {m: (M['MAT'][m][1], M['MAT'][m][2]) for m in M['MAT']}

    # ---- PDL hydrostatic stress on the PDL surface ----
    pdl = tets[tags == 2]
    rp = element_strain_stress(nodes, pdl, np.full(len(pdl), 2), matmap, S['u'])
    f, own = boundary_faces(pdl, return_owner=True)
    hyd = -rp['hyd'][own]                     # positive = compression
    # ---- aligner von Mises ----
    A = M['aligner']; nS = M['nS']
    al = A['tets'] + nS
    ra = element_strain_stress(nodes, al, np.full(len(al), 6), matmap, S['u'])
    fa, owna = boundary_faces(al, return_owner=True)
    vma = ra['vm'][owna]
    # ---- tooth displacement magnitude ----
    tt = np.vstack([t['tets'] for t in M['teeth'].values()])
    ft = boundary_faces(tt)
    dmag = np.linalg.norm(U, axis=1)[ft].mean(1)*1e3

    views = [('occlusal', (-88, -90)), ('right buccal', (8, 170)), ('left buccal', (8, -10))]
    fig = plt.figure(figsize=(16, 13), facecolor='white')
    rows = [
        ('PDL hydrostatic compression (MPa)', nodes, f, hyd, 'coolwarm',
         TwoSlopeNorm(vcenter=CAPILLARY_MPA, vmin=-0.01, vmax=0.03)),
        ('Aligner von Mises stress (MPa)', nodes, fa, vma, 'viridis',
         Normalize(0, np.percentile(vma, 99))),
        ('Tooth displacement magnitude (um)', nodes, ft, dmag, 'magma',
         Normalize(0, np.percentile(dmag, 99))),
    ]
    allp = nodes[np.unique(ft)]
    for i, (title, pts, tri, val, cmap, norm) in enumerate(rows):
        for j, (vn, (el, az)) in enumerate(views):
            ax = fig.add_subplot(3, 3, 3*i+j+1, projection='3d')
            ax.view_init(el, az)
            surf_plot(ax, pts, tri, val, cmap, norm)
            frame(ax, allp)
            if j == 0:
                ax.set_title(f'{title}\n{vn}', fontsize=9)
            else:
                ax.set_title(vn, fontsize=9)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cax = fig.add_axes([0.92, 0.70-0.31*i, 0.013, 0.20])
        fig.colorbar(sm, cax=cax)
    fig.suptitle('Maxillary total arch distalization -- 250 gf per side\n'
                 'PDL compression above the capillary threshold (white = 0.0047 MPa) '
                 'indicates ischaemia risk', fontsize=13)
    plt.savefig(path, dpi=115, facecolor='white', bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    R = pickle.load(open('build/res_250_per_side.pkl', 'rb'))
    fig_bars(R, f'{OUT}/tooth_movement.png')
    print('wrote tooth_movement.png')
    M = pickle.load(open('build/model.pkl', 'rb'))
    S = pickle.load(open('build/sol_250_per_side.pkl', 'rb'))
    fig_fields(M, S, R, f'{OUT}/stress_fields.png')
    print('wrote stress_fields.png')
