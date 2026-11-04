import sys, os, pickle, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
S = pickle.load(open(os.path.join(os.path.dirname(__file__), 'series_main.pkl'), 'rb'))
PAT = ['P8', 'P3', 'P4', 'P5', 'P6']
G = ['hg', 'ca_precision_cut', 'ca_aligner_hook', 'ca_metal_hook', 'ca_control']
NAME = dict(zip(G, ['HG', 'PC', 'CAH', 'CMH', 'Control']))
COL = dict(zip(G, ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#8a8a85']))
LS = dict(zip(G, ['-', '-', (0, (4, 2)), (0, (1, 1.5)), (0, (6, 2, 1, 2))]))
INK, MUTED = '#2b2b29', '#8a8a85'
grid = np.linspace(0, 1.25, 126)
def on_grid(s, key):
    x = s['16 crown AP']; i = np.argmax(x >= 1.25) if (x >= 1.25).any() else len(x)
    # first passage: use the rising part up to the first time each value is reached
    xm = np.maximum.accumulate(x[:i+1]); y = s[key][:i+1]
    keep = np.concatenate([[True], np.diff(xm) > 1e-9])
    return np.interp(grid, xm[keep], y[keep], right=np.nan)
PANELS = [('16 apex AP', '16 root apex, AP (mm, distal +)'), ('16 tipping', '16 distal crown tipping (mm, crown − apex)'),
          ('16 V', '16 vertical (mm, intrusion +)'), ('16 BP', '16 DB cusp, transverse (mm, buccal +)'),
          ('13 crown AP', '13 cusp, AP (mm, distal +)'), ('11 crown BP', '11 incisal edge (mm, palatal +)')]
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.edgecolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.labelcolor': INK})
fig, axs = plt.subplots(2, 3, figsize=(10, 6.2))
for ax, (k, lab) in zip(axs.ravel(), PANELS):
    for g in G:
        Y = np.array([on_grid(S[P][g], k) for P in PAT]); m = np.nanmean(Y, 0); sd = np.nanstd(Y, 0, ddof=1)
        ax.plot(grid, m, color=COL[g], lw=2 if g != 'ca_control' else 1.5, ls=LS[g], label=NAME[g])
        if g in ('hg', 'ca_precision_cut'): ax.fill_between(grid, m - sd, m + sd, color=COL[g], alpha=0.12, lw=0)
    ax.axvline(0.8, color=MUTED, lw=0.8, ls=':'); ax.axhline(0, color='#d6d5cf', lw=0.6)
    ax.set_title(lab, fontsize=8.5, color=INK, loc='left'); ax.set_xlim(0, 1.25)
    ax.grid(axis='y', color='#ecebe6', lw=0.6)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
for ax in axs[1]: ax.set_xlabel('16 MB cusp distalization (mm)')
axs[0, 0].text(0.81, axs[0, 0].get_ylim()[1]*0.92, '0.8 mm', fontsize=7, color=MUTED)
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=5, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.95))
for ext in ('png', 'pdf'): fig.savefig(os.path.join(os.path.dirname(__file__), '..', 'figs', f'Fig_trajectory_matched.{ext}'), dpi=300)
