"""Render the 4D half-arch model, laid out like Fig. 1 of Mao et al. 2024."""
import sys, os, pickle
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ofj'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mesh_io import boundary_faces
import spec

C_PDL = '#c1848d'
C_TOOTH = '#eae5d2'
C_ATT = '#c8791f'
C_ALIGN = '#57c0cd'
C_TAD = '#4a4f57'
C_ELAS = '#e2571e'


def shade(pts, tris, color, light=np.array([0.35, -0.55, -0.75]), amb=0.42):
    v = pts[tris]
    n = np.cross(v[:, 1]-v[:, 0], v[:, 2]-v[:, 0])
    L = np.linalg.norm(n, axis=1)
    k = L > 1e-12
    v, n = v[k], n[k]/L[k, None]
    b = amb + (1-amb)*np.clip(n @ (light/np.linalg.norm(light)), 0, 1)
    c = np.array(matplotlib.colors.to_rgb(color))[None, :]*b[:, None]
    return v, np.clip(c, 0, 1)


def draw(ax, layers):
    V, C = [], []
    for pts, tris, col in layers:
        if len(tris) == 0:
            continue
        v, c = shade(pts, tris, col)
        V.append(v); C.append(c)
    V = np.vstack(V); C = np.vstack(C)
    o = np.argsort(V[:, :, 1].mean(1))
    ax.add_collection3d(Poly3DCollection(V[o], facecolors=C[o], edgecolors='none'))


def frame(ax, pts, zoom=0.60):
    c = pts.mean(0); r = np.ptp(pts, 0).max()/2*zoom
    ax.set_xlim(c[0]-r, c[0]+r); ax.set_ylim(c[1]-r, c[1]+r); ax.set_zlim(c[2]-r, c[2]+r)
    ax.set_axis_off(); ax.set_box_aspect((1, 1, 1))


def tube(p0, p1, radius, n=16, cap=True):
    """Triangulated cylinder from p0 to p1 -- used for the elastic."""
    ax_ = np.asarray(p1) - np.asarray(p0)
    L = np.linalg.norm(ax_); ax_ = ax_/L
    tmp = np.array([0, 0, 1.0])
    if abs(tmp @ ax_) > 0.9:
        tmp = np.array([1.0, 0, 0])
    e1 = np.cross(ax_, tmp); e1 /= np.linalg.norm(e1)
    e2 = np.cross(ax_, e1)
    th = np.linspace(0, 2*np.pi, n, endpoint=False)
    ring = radius*(np.cos(th)[:, None]*e1 + np.sin(th)[:, None]*e2)
    V = np.vstack([p0 + ring, p1 + ring])
    T = []
    for i in range(n):
        j = (i+1) % n
        T += [[i, j, n+j], [i, n+j, n+i]]
    if cap:
        c0 = len(V); V = np.vstack([V, p0[None], p1[None]])
        for i in range(n):
            j = (i+1) % n
            T += [[i, c0, j], [n+i, n+j, c0+1]]
    return V, np.array(T)


def tad_point(ax, p):
    """The miniscrew as the point it is in the model, drawn over the mesh."""
    ax.computed_zorder = False
    # bright fill: a dark marker vanishes against the shaded PDL and reads as a ring
    ax.scatter([p[0]], [p[1]], [p[2]], s=95, c='#ffd23f', edgecolors='#111111',
               linewidths=1.5, depthshade=False, zorder=10)


def main():
    G = pickle.load(open('build4d/geom.pkl', 'rb'))
    A = pickle.load(open('build4d/aligner.pkl', 'rb'))
    nodes, teeth = G['nodes'], G['teeth']

    tooth_tris = np.vstack([np.vstack([t['root_faces'], t['crown_faces']])
                            for t in teeth.values()])
    att_tris = np.vstack([t['att']['faces'] for t in teeth.values()
                          if t['att'] is not None])
    pdl_parts = [(t['pdl_local'][0], boundary_faces(t['pdl_local'][1]))
                 for t in teeth.values()]
    alg_tris = boundary_faces(A['tets']); alg_nodes = A['nodes']
    allp = np.vstack([nodes[np.unique(tooth_tris)], alg_nodes[np.unique(alg_tris)]])

    from model4d import Static, State
    from loads4d import Elastic
    S = Static(); state = State(S)
    for u in S.keep:
        S.tooth[u]['centroid_cur'] = state.centroid(u)
    L = Elastic(S, state, group='buccal_tad')
    hook = state.aligner[L.hook_nodes].mean(0)
    eo = S.tooth[3]['e_o']
    # The TAD is a point in the model -- the fixed end of the elastic, no screw body
    # is meshed -- so the schematic draws it as a point rather than a cylinder.
    el_v, el_t = tube(hook, L.tad, 0.30)

    BUCCAL = (12, 158)
    fig = plt.figure(figsize=(17.5, 10), facecolor='white')
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.35], width_ratios=[1,1,1,1.15],
                          wspace=0.06, hspace=0.12)

    comps = [('Periodontal ligament', [(pn, pf, C_PDL) for pn, pf in pdl_parts]),
             ('Teeth', [(nodes, tooth_tris, C_TOOTH)]),
             ('Attachments', [(nodes, tooth_tris, '#f0efe9'),
                              (nodes, att_tris, C_ATT)]),
             ('Clear aligner', [(alg_nodes, alg_tris, C_ALIGN)])]
    for i, (label, layers) in enumerate(comps):
        ax = fig.add_subplot(gs[0, i], projection='3d')
        ax.view_init(*BUCCAL)
        draw(ax, layers); frame(ax, allp, zoom=0.52)
        ax.set_title(label, fontsize=12.5, pad=-6)

    assembled = ([(nodes, tooth_tris, C_TOOTH), (nodes, att_tris, C_ATT)]
                 + [(pn, pf, C_PDL) for pn, pf in pdl_parts]
                 + [(alg_nodes, alg_tris, C_ALIGN)])

    ax = fig.add_subplot(gs[1, 0:2], projection='3d')
    ax.view_init(*BUCCAL)
    draw(ax, assembled + [(el_v, el_t, C_ELAS)])
    tad_point(ax, L.tad)
    frame(ax, allp, zoom=0.50)
    ax.set_title("Buccal TAD group — assembled\n"
                 "TAD in the 16/17 buccal interradicular space, 4 mm apical to the "
                 "alveolar crest;\n150 gf elastic to a precision cut at the canine "
                 "mesial cervical region", fontsize=11.5, pad=-4)

    ax = fig.add_subplot(gs[1, 2], projection='3d')
    ax.view_init(-86, -96)
    draw(ax, assembled + [(el_v, el_t, C_ELAS)])
    tad_point(ax, L.tad)
    frame(ax, allp, zoom=0.88)
    ax.set_title("occlusal", fontsize=12, pad=-30)

    ax = fig.add_subplot(gs[1, 3])
    order = spec.HALF_ARCH_UNN
    for i, u in enumerate(order):
        a, b = spec.V_PATTERN[u]
        post = u not in (7, 8)
        ax.barh(i, b-a+1, left=a-1, height=0.6,
                color='#9fd6a8' if post else '#a8c4e8',
                edgecolor='#5c7a63' if post else '#5a7296', lw=0.7)
        ax.text(b+2, i, f"{spec.total_prescribed(u):.1f} mm "
                        f"{'distal' if post else 'palatal'}",
                va='center', fontsize=9, color='#333')
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"FDI {spec.FDI[u]}" for u in order], fontsize=10)
    ax.invert_yaxis(); ax.set_xlim(0, 100)
    ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 70])
    ax.set_xlabel("staging step  (one aligner per step)", fontsize=10)
    ax.set_title("'V pattern' schedule, 0.1 mm/step", fontsize=12)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.grid(axis='x', alpha=0.3)

    fig.suptitle("Maxillary right half arch — 4D staging model, buccal TAD group\n"
                 "Open-Full-Jaw patient 8  ·  291k elements, 150,644 dof  ·  teeth rigid, "
                 "0.30 mm bilinear PDL, 0.70 mm / 1500 MPa aligner, 5 attachments",
                 fontsize=13.5)
    plt.savefig('out4d/model_overview.png', dpi=125, facecolor='white',
                bbox_inches='tight')
    print("wrote out4d/model_overview.png")


if __name__ == '__main__':
    main()
