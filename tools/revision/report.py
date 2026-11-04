"""Revision analyses -> out/revision/*.md (+ json). Uses the landmark paths rebuilt in paths.py."""
import sys, os, json, pickle, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats import *
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
AB = {'hg': 'HG', 'ca_precision_cut': 'PC', 'ca_aligner_hook': 'CAH', 'ca_metal_hook': 'CMH', 'ca_control': 'Control',
      'hg_pdl068': 'HG', 'pc_pdl068': 'PC', 'cah_pdl068': 'CAH', 'cmh_pdl068': 'CMH', 'ctl_pdl068': 'Control',
      'hg_izc': 'HG-IZC', 'hg_cres': 'HG-CRes', 'hg_level': 'HG-level'}
MAIN = ['hg', 'ca_precision_cut', 'ca_aligner_hook', 'ca_metal_hook', 'ca_control']
LIN = ['hg_pdl068', 'pc_pdl068', 'cah_pdl068', 'cmh_pdl068', 'ctl_pdl068']
def have(P, g): return os.path.exists(f'{DATA}/{P}/run_{g}.pkl')
S = {}
for P in PAT:
    S[P] = {}
    for g in MAIN + LIN + ['hg_izc', 'hg_cres', 'hg_level']:
        if have(P, g):
            R = pickle.load(open(f'{DATA}/{P}/run_{g}.pkl', 'rb'))
            n = R.get('iter', R.get('step')); full = (n >= 140) if 'iter' in R else (n >= 70)
            if full: S[P][g] = series(P, R)
def avail(g): return all(g in S[P] for P in PAT)
def fmt(x): return f'{x:+.2f}'
def ms(a): return f'{np.mean(a):+.2f} ± {np.std(a, ddof=1):.2f}'
METN = [m[0] for m in METRICS]
LABEL = {'16 crown AP': '16 MB cusp, AP', '16 apex AP': '16 MB root apex, AP', '16 V': '16 mesial pit, vertical',
         '16 BP': '16 DB cusp, transverse', '13 crown AP': '13 cusp, AP', '13 V': '13 cusp, vertical',
         '11 crown BP': '11 incisal edge, labiopalatal', '11 apex BP': '11 apex, labiopalatal', '11 V': '11 incisal edge, vertical',
         '16 tipping': '16 crown − apex, AP (tipping)'}
def vals(groups, where, th=0.8):
    out = {}
    for g in groups:
        if where == 'final': out[g] = np.array([S[P][g][k][-1] for P in PAT for k in [None]] if False else [0]) 
    return out
def get(g, metric, where, th=0.8):
    if where == 'final': return np.array([S[P][g][metric][-1] for P in PAT])
    return np.array([at_matched(S[P][g], th)[1][metric] for P in PAT])
def table(groups, where, title, th=0.8, tests=True, pairs_filter=None):
    L = [f'### {title}', '', '| Outcome | ' + ' | '.join(AB[g] for g in groups) + (' | Friedman p (HG, PC, CAH, CMH) |' if tests else ' |'),
         '|---|' + '---|'*len(groups) + ('---|' if tests else '')]
    allpairs = {}
    for m in METN:
        if where != 'final' and m == '16 crown AP': continue
        V = {g: get(g, m, where, th) for g in groups}
        row = f'| {LABEL[m]} | ' + ' | '.join(ms(V[g]) for g in groups)
        if tests:
            act = [g for g in groups if 'control' not in g and 'ctl' not in g]; ctl = [g for g in groups if g not in act]
            fr, pr = compare({g: V[g] for g in act}, act); row += f' | {fr.pvalue:.3f} |'
            if ctl:
                c = ctl[0]; pc = [stats.ttest_rel(V[g], V[c]).pvalue for g in act[1:]]; ph = holm(pc)
                pr = pr + [(g, c, float((V[g] - V[c]).mean()), float(ph[i]), int((V[g] > V[c]).sum())) for i, g in enumerate(act[1:])]
            allpairs[m] = pr
        else: row += ' |'
        L.append(row)
    L.append('')
    return L, allpairs
def pair_lines(allpairs, keep):
    L = ['Pairwise paired t-tests (mean difference A − B, mm; Holm-adjusted p; patients with A > B of 5). Holm correction within each outcome over the six pairs of active groups, and separately over the three elastic-versus-control contrasts (anchorage reinforcement):', '']
    for m, pr in allpairs.items():
        s = [f'{AB[a]}−{AB[b]} {md:+.2f} (p={p:.3f}, {n}/5)' for a, b, md, p, n in pr if keep(a, b)]
        L.append(f'- {LABEL[m]}: ' + '; '.join(s))
    L.append(''); return L
R = {}
# ---------- 1. primary: matched 0.8 mm, all five groups incl. control ----------
doc = ['# Revision analyses', '', 'mm; AP + distal, transverse + buccal (incisors: + palatal), vertical + intrusion; mean ± SD over 5 patients (P8, P3, P4, P5, P6).', '']
steps = {g: np.mean([at_matched(S[P][g], 0.8)[0] for P in PAT]) for g in MAIN}
doc += ['## 1. Primary analysis: equal first-molar distalization (16 MB cusp 0.8 mm)', '',
        'Step at which 0.8 mm was reached (mean over patients; HG in aligner-step units = 2 remodelling iterations): '
        + ', '.join(f'{AB[g]} {steps[g]:.1f}' for g in MAIN) + '.', '']
L, ap = table(MAIN, 'matched', 'Table A. Matched (0.8 mm), main model (bilinear PDL), control included in the tests')
doc += L + pair_lines(ap, lambda a, b: True)
# ---------- 2. secondary: end of protocol ----------
L, ap = table(MAIN, 'final', 'Table B. End of protocol (aligner step 70; HG iteration 140), main model, control included')
doc += ['## 2. Secondary (exploratory) analysis: end of protocol', ''] + L + pair_lines(ap, lambda a, b: True)
# ---------- 3. thresholds ----------
doc += ['## 3. Sensitivity of the matched comparison to the threshold', '',
        'Largest 16 MB cusp distalization reached by every group in every patient: '
        f"{min(S[P][g]['16 crown AP'].max() for P in PAT for g in MAIN):.2f} mm (control, P6).", '']
TH = [0.4, 0.6, 0.8, 1.0, 1.2]
hdr = '| Outcome | Contrast | ' + ' | '.join(f'{t} mm' for t in TH) + ' |'
doc += ['Mean difference (mm) and patients with the same sign (of 5); * Holm p < 0.05 (6 pairwise tests among the four active groups).', '', hdr, '|---|---|' + '---|'*len(TH)]
for m in METN[1:]:
    for b in ['ca_precision_cut', 'ca_aligner_hook']:
        cells = []
        for t in TH:
            fr, pr = compare({g: get(g, m, 'matched', t) for g in MAIN[:4]}, MAIN[:4])
            a_, b_, md, p, n = [x for x in pr if x[0] == 'hg' and x[1] == b][0]
            cells.append(f'{md:+.2f} ({max(n, 5-n)}/5){"*" if p < 0.05 else ""}')
        doc.append(f'| {LABEL[m]} | HG − {AB[b]} | ' + ' | '.join(cells) + ' |')
    cells = []
    for t in TH:
        fr, pr = compare({g: get(g, m, 'matched', t) for g in MAIN[:4]}, MAIN[:4])
        a_, b_, md, p, n = [x for x in pr if x[0] == 'ca_precision_cut' and x[1] == 'ca_aligner_hook'][0]
        cells.append(f'{md:+.2f} ({max(n, 5-n)}/5){"*" if p < 0.05 else ""}')
    doc.append(f'| {LABEL[m]} | PC − CAH | ' + ' | '.join(cells) + ' |')
doc.append('')
# ---------- 4. linear PDL incl. control ----------
if avail('ctl_pdl068'):
    doc += ['## 4. Linear PDL (E = 0.68 MPa), control included', '']
    L, ap = table(LIN, 'final', 'Table C. End of protocol, linear PDL')
    doc += L + pair_lines(ap, lambda a, b: True)
    L, ap = table(LIN, 'matched', 'Table D. Matched 0.5 mm (as in the original linear-PDL analysis; HG reaches only 0.68 mm in P5), linear PDL', th=0.5)
    doc += L + pair_lines(ap, lambda a, b: True)
    # anchorage reinforcement = elastic group − control, both PDL models
    doc += ['### Table E. Anchorage reinforcement by the elastic (elastic group − aligner-only control), end of protocol', '',
            '| Outcome | PDL | PC − control | CAH − control | CMH − control |', '|---|---|---|---|---|']
    for m in ['16 crown AP', '17 crown AP', '16 apex AP', '13 crown AP', '11 crown BP', '11 V', '16 V']:
        if m not in METN: continue
        for lab, grp, ctl in [('bilinear', MAIN[1:4], 'ca_control'), ('linear', LIN[1:4], 'ctl_pdl068')]:
            c0 = get(ctl, m, 'final'); cells = []; ps = []
            for g in grp: ps.append(stats.ttest_rel(get(g, m, 'final'), c0).pvalue)
            ph = holm(ps)
            for i, g in enumerate(grp):
                d = get(g, m, 'final') - c0
                cells.append(f'{d.mean():+.2f} ± {d.std(ddof=1):.2f} ({int((d > 0).sum())}/5 > 0, Holm p={ph[i]:.3f})')
            doc.append(f'| {LABEL[m]} | {lab} | ' + ' | '.join(cells) + ' |')
    doc.append('')
# ---------- 5. headgear variants ----------
HV = [g for g in ['hg_izc', 'hg_cres', 'hg_level'] if avail(g)]
if HV:
    doc += ['## 5. Headgear force-system variants (all patients)', '',
            'HG = main model (cervical, 15° below the occlusal plane, force at the 16 headgear tube, passive wire bent through FA-placed slots); '
            'HG-IZC = force direction matched to the IZC elastic (19.7–26.3° above the occlusal plane, per patient), at the tube; '
            'HG-CRes = 15° below the occlusal plane with the line of action through the 16 centroid (outer bow adjusted, no moment); '
            'HG-level = main headgear on a levelled set-up (slots seated on one flat passive archwire).', '']
    for where in ['matched', 'final']:
        G2 = ['hg'] + HV + ['ca_precision_cut', 'ca_aligner_hook']
        L, ap = table(G2, where, f"Table F{'1' if where == 'matched' else '2'}. Headgear variants, {'matched 0.8 mm' if where == 'matched' else 'end of protocol'} (aligner groups for reference)", tests=False)
        doc += L
        if where == 'matched':
            doc.append('Step at which 0.8 mm was reached (aligner-step units): ' + ', '.join(f'{AB[g]} {np.mean([at_matched(S[P][g], 0.8)[0] for P in PAT]):.1f}' for g in ['hg'] + HV) + '.')
            doc.append('')
    # direction of HG - PC differences retained?
    doc += ['### Table G. Does the HG − aligner (PC) difference keep its sign under each headgear variant? (patients with the same sign as the main HG − PC mean difference, matched 0.8 mm and end of protocol)', '',
            '| Outcome | ' + ' | '.join(f'{AB[g]} matched | {AB[g]} end' for g in HV) + ' |', '|---|' + '---|---|'*len(HV)]
    for m in METN:
        cells = []
        for g in HV:
            for where in ['matched', 'final']:
                if where == 'matched' and m == '16 crown AP': cells.append('—'); continue
                ref = np.sign(np.mean(get('hg', m, where) - get('ca_precision_cut', m, where)))
                d = get(g, m, where) - get('ca_precision_cut', m, where)
                cells.append(f'{int((np.sign(d) == ref).sum())}/5 ({d.mean():+.2f})')
        doc.append(f'| {LABEL[m]} | ' + ' | '.join(cells) + ' |')
    doc.append('')
open(f'{OUT}/revision_analyses.md', 'w').write('\n'.join(doc))
print('\n'.join(doc)[:6000])
