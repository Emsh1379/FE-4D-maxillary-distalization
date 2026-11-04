"""Figure: line of action of the 200 gf force in each group (patient 8), lateral and occlusal views."""
import os, sys, pickle, numpy as np
ROOT = sys.argv[1]; W = f'{ROOT}/lws/P8'; os.chdir(W)
os.environ.update(OFJ_GEOM='build4d/geom_fe.pkl', OFJ_ALIGNER='build4d/aligner_fe.pkl', OFJ_PRESCRIBE='tip', OFJ_TIP_BLEND='0.40', OFJ_GPU='off')
sys.path.insert(0, f'{W}/ofj4d'); sys.path.insert(0, f'{W}/ofj')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import spec
from model4d import Static, State
from arch4d import ArchForm
from loads4d import Elastic
S = Static(); st = State(S); A = ArchForm(S)
for u in S.keep: S.tooth[u]['centroid_cur'] = st.centroid(u)
E = {g: Elastic(S, st, group=g) for g in ['ca_precision_cut', 'ca_aligner_hook', 'ca_metal_hook']}
fx = pickle.load(open('build4d/fixed.pkl', 'rb'))['fixed']
tube = fx[spec.UNN[16]]['hg_tube']; c16 = S.tooth[spec.UNN[16]]['centroid']
def lat(p): q = np.atleast_2d(p) - A.origin; return np.stack([q @ A.ey, -(q @ A.en)], 1)   # x = distal, y = apical (+)
def occ(p): q = np.atleast_2d(p) - A.origin; return np.stack([q @ A.ey, q @ A.ex], 1)
hook = {'ca_precision_cut': E['ca_precision_cut'].hook_rest.mean(0), 'ca_aligner_hook': E['ca_aligner_hook'].hook_tip_rest,
        'ca_metal_hook': E['ca_metal_hook'].hook_tip_rest}
tad = E['ca_precision_cut'].tad
INK, MUTED = '#2b2b29', '#8a8a85'
COL = {'hg': '#2a78d6', 'ca_precision_cut': '#eb6834', 'ca_aligner_hook': '#1baf7a', 'ca_metal_hook': '#eda100'}
NAME = {'hg': 'HG', 'ca_precision_cut': 'PC', 'ca_aligner_hook': 'CAH', 'ca_metal_hook': 'CMH'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED})
fig, axs = plt.subplots(2, 4, figsize=(11, 4.3), gridspec_kw=dict(height_ratios=[1, 1]))
def teeth(ax, f):
    for u in S.keep:
        t = S.tooth[u]; P = f(t['surf_rest'][:t['surf_n']][::7]); ax.scatter(P[:, 0], P[:, 1], s=0.4, c='#c9c8c0', lw=0, rasterized=True)
        R = f(t['pdl_rest'][::25]); ax.scatter(R[:, 0], R[:, 1], s=0.3, c='#e3e2db', lw=0, rasterized=True)
def ang_of(d):
    return np.degrees(np.arctan2(-(d @ A.en), d @ A.ey))
for j, g in enumerate(['hg', 'ca_precision_cut', 'ca_aligner_hook', 'ca_metal_hook']):
    for i, f in enumerate([lat, occ]):
        ax = axs[i, j]; teeth(ax, f)
        if g == 'hg':
            d = BRu = (np.cos(np.radians(15))*A.ey + np.sin(np.radians(15))*A.en)
            p0 = tube; p1 = tube + 14*d
            ax.annotate('', f(p1)[0], f(p0)[0], arrowprops=dict(arrowstyle='-|>', color=COL[g], lw=2, mutation_scale=10))
            ax.plot(*f(p0)[0], 'o', ms=5, color=COL[g], mec='white', mew=1)
            # revision variants: matched to the IZC vector (25 deg above), and through the 16 centroid
            dz = np.cos(np.radians(-25))*A.ey + np.sin(np.radians(-25))*A.en
            ax.annotate('', f(tube + 14*dz)[0], f(tube)[0], arrowprops=dict(arrowstyle='-|>', color=COL[g], lw=1.2, ls=(0, (3, 2)), mutation_scale=8))
            if i == 0:
                cc = f(c16)[0]; ax.plot(*cc, marker='x', ms=6, color=INK)
                ax.text(cc[0]-1.0, cc[1]+0.9, '16 centroid\n(CRes)', fontsize=7, color=INK, ha='right')
                ax.text(*(f(p1)[0] + [0.3, -1.4]), '15° below OP\n(cervical, main)', color=INK, fontsize=7)
                ax.text(*(f(tube + 14*dz)[0] + [0.3, 0.2]), '25° above OP\n(IZC-matched, revision)', color=INK, fontsize=7)
            ttl = 'HG (headgear tube)'
        else:
            h = hook[g]; dvec = (tad - h)/np.linalg.norm(tad - h)
            ax.plot(*np.vstack([f(h), f(tad)]).T, color=COL[g], lw=2)
            ax.plot(*f(tad)[0], marker='s', ms=6, color=INK); ax.plot(*f(h)[0], 'o', ms=5, color=COL[g], mec='white', mew=1)
            if i == 0:
                ax.text(*(f(tad)[0] + [0.6, 0.4]), 'IZC screw head', fontsize=7, color=INK)
                ax.text(*(f(h)[0] + [-0.5, -2.2]), f'{ang_of(dvec):.0f}° above OP', fontsize=7, color=INK, ha='center')
            ttl = {'ca_precision_cut': 'PC (precision cut)', 'ca_aligner_hook': 'CAH (aligner-material hook)',
                   'ca_metal_hook': 'CMH (bonded metal hook)'}[g]
        ax.set_aspect('equal'); ax.set_xlim(-6, 48)
        if i == 0:
            ax.set_ylim(-6, 24); ax.axhline(0, color=MUTED, lw=0.6, ls=':'); ax.set_title(ttl, fontsize=8.5, color=INK, loc='left')
            if j == 0: ax.set_ylabel('apical (+), mm')
        else:
            ax.set_ylim(-4, 34)
            if j == 0: ax.set_ylabel('transverse, mm')
            ax.set_xlabel('distal from incisor midpoint, mm')
        for s in ('top', 'right'): ax.spines[s].set_visible(False)
axs[0, 0].text(-5.5, 0.4, 'OP', fontsize=7, color=MUTED)
fig.text(0.01, 0.97, 'a  Lateral view', fontsize=9, weight='bold', color=INK); fig.text(0.01, 0.50, 'b  Occlusal view', fontsize=9, weight='bold', color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.96))
for ext in ('png', 'pdf'): fig.savefig(f'{ROOT}/figs/Fig_line_of_action_P8.{ext}', dpi=300)
print({NAME[g]: round(ang_of((tad - hook[g])/np.linalg.norm(tad - hook[g])), 1) for g in hook}, 'tube', lat(tube), 'c16', lat(c16))
