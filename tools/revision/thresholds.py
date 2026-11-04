import sys, os, json, pickle; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats import *
G = ['hg', 'ca_precision_cut', 'ca_aligner_hook', 'ca_metal_hook', 'ca_control']
AB = dict(zip(G, ['HG', 'PC', 'CAH', 'CMH', 'control']))
S = {P: {g: series(P, f'{DATA}/{P}/run_{g}.pkl') for g in G} for P in PAT}
pickle.dump(S, open(os.path.join(os.path.dirname(__file__), 'series_main.pkl'), 'wb'))
peak = {(P, g): S[P][g]['16 crown AP'].max() for P in PAT for g in G}
print('peak 16 crown AP over the protocol (min over patients):', {AB[g]: round(min(peak[(P, g)] for P in PAT), 3) for g in G})
print('largest threshold reached by all:', round(min(peak.values()), 3), min(peak, key=peak.get))
# step at which each group reaches thresholds
TH = [0.2, 0.4, 0.6, 0.8, 1.0]
res = {}
for th in TH:
    st = {g: [at_matched(S[P][g], th)[0] for P in PAT] for g in G}
    ok = all(s is not None for g in G for s in st[g])
    res[th] = dict(steps={AB[g]: (np.mean(st[g]) if all(s is not None for s in st[g]) else None) for g in G}, metrics={})
    for name, *_ in METRICS[1:]:
        vals = {g: np.array([at_matched(S[P][g], th)[1][name] if at_matched(S[P][g], th)[0] is not None else np.nan for P in PAT]) for g in G}
        fr, rows = compare({g: vals[g] for g in G[:4]}, G[:4])
        res[th]['metrics'][name] = dict(mean={AB[g]: float(np.nanmean(vals[g])) for g in G}, sd={AB[g]: float(np.nanstd(vals[g], ddof=1)) for g in G},
                                        friedman_p=float(fr.pvalue), pairs=[(AB[a], AB[b], float(md), float(p), n) for a, b, md, p, n in rows])
json.dump(res, open(os.path.join(os.path.dirname(__file__), 'thresholds.json'), 'w'), indent=1)
for th in TH:
    print(f'\n== {th} mm  steps', {k: (round(v, 1) if v else None) for k, v in res[th]['steps'].items()})
    for name, d in res[th]['metrics'].items():
        hp = [f"{a}-{b} {md:+.3f}({n}/5,p={p:.3f})" for a, b, md, p, n in d['pairs'] if a == 'HG']
        print(f"  {name:12s}", ' '.join(f"{k}{v:+.3f}" for k, v in d['mean'].items()), '| HG vs:', '; '.join(hp))
