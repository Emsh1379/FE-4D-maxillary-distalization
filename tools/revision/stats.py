import sys, os, pickle, numpy as np
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import Landmarks, DATA
PAT = ['P8', 'P3', 'P4', 'P5', 'P6']
METRICS = [  # name, path, landmark, comp, sign
    ('16 crown AP', 'C (16 MB)', 'Cusp tip', 'AP'), ('16 apex AP', 'C (16 MB)', 'Apex', 'AP'),
    ('16 V', 'D (16 occlusal)', 'Mesial pit', 'V'), ('16 BP', 'E (16 DB)', 'Cusp tip', 'BP'),
    ('13 crown AP', 'A (13)', 'Cusp tip', 'AP'), ('13 V', 'A (13)', 'Cusp tip', 'V'),
    ('11 crown BP', 'B (11)', 'Incisal edge', 'BP'), ('11 apex BP', 'B (11)', 'Apex', 'BP'),
    ('11 V', 'B (11)', 'Incisal edge', 'V'), ('16 tipping', None, None, None)]
CI = {'AP': 0, 'BP': 1, 'V': 2}
_LM = {}
def lm(P):
    if P not in _LM: _LM[P] = Landmarks(P)
    return _LM[P]
def series(P, run):
    """Per-step metric values (step 0 = 0) for a run pickle (dict or path)."""
    R = pickle.load(open(run, 'rb')) if isinstance(run, str) else run
    L = lm(P); out = {m[0]: [0.0] for m in METRICS}
    for h in R['hist']:
        v = L.measure(h['T'])
        for name, path, land, comp in METRICS:
            if name == '16 tipping':
                out[name].append(v[('C (16 MB)', 'Cusp tip')][0] - v[('C (16 MB)', 'Apex')][0])
            else:
                out[name].append(v[(path, land)][CI[comp]])
    return {k: np.array(v) for k, v in out.items()}
def at_matched(S, mm=0.8, key='16 crown AP'):
    x = S[key]; i = np.argmax(x >= mm)
    if x[i] < mm: return None, None
    f = (mm - x[i-1])/(x[i] - x[i-1]); st = i - 1 + f
    return st, {k: v[i-1] + f*(v[i] - v[i-1]) for k, v in S.items()}
def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o): run = max(run, min(1, (m - r)*p[i])); adj[i] = run
    return adj
def compare(vals, groups):
    """vals: {group: array over patients}. Friedman + paired t (Holm) + sign counts."""
    X = np.array([vals[g] for g in groups])
    fr = stats.friedmanchisquare(*X) if len(groups) > 2 else None
    pairs = [(a, b) for i, a in enumerate(groups) for b in groups[i+1:]]
    p = []; rows = []
    for a, b in pairs:
        d = vals[a] - vals[b]
        p.append(stats.ttest_rel(vals[a], vals[b]).pvalue if np.std(d) > 1e-12 else 1.0)
        rows.append((a, b, d.mean(), int((d > 0).sum())))
    ph = holm(p)
    return fr, [(a, b, md, ph[i], n) for i, (a, b, md, n) in enumerate(rows)]
