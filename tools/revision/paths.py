"""Landmark paths of the study (A 13, B 11, F 12, C 16 MB, E 16 DB, D 16 occlusal), rebuilt from the
published per-step tables: each landmark's rest point is solved exactly from the stored rigid-body
transforms, then applied to any run of the same patient."""
import pickle, re, os, glob, numpy as np
DATA = os.environ.get('REV_DATA', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'revision_data'))
FDI = {2: 17, 3: 16, 4: 15, 5: 14, 6: 13, 7: 12, 8: 11}; UNN = {v: k for k, v in FDI.items()}
PATH_TOOTH = {'A (13)': 13, 'B (11)': 11, 'F (12)': 12, 'C (16 MB)': 16, 'E (16 DB)': 16, 'D (16 occlusal)': 16}
COLS = {'Headgear (HG)': 'hg', 'Precision cut (PC)': 'ca_precision_cut', 'Aligner hook (CAH)': 'ca_aligner_hook',
        'Metal hook (CMH)': 'ca_metal_hook', 'Aligner, no elastic': 'ca_control'}

def parse_tables(fn):
    rows = []; head = None; path = comp = None
    for line in open(fn):
        m = re.match(r'\*\*Path (.+), (AP|BP|V)\*\*', line.strip())
        if m: path, comp = 'ABFCED'.find(m.group(1)[0]) >= 0 and m.group(1), m.group(2); head = None; continue
        if line.startswith('| Landmark'):
            head = []
            for c in line.strip().strip('|').split('|')[1:]:
                c = c.strip(); name = re.sub(r'\s*\(step [\d.]+\)\s*$', '', c); st = re.search(r'\(step ([\d.]+)\)', c)
                head.append((COLS[name], float(st.group(1))))
            continue
        if head and line.startswith('|') and not line.startswith('|---'):
            c = [x.strip() for x in line.strip().strip('|').split('|')]
            for (g, s), v in zip(head, c[1:]): rows.append((g, s, path, c[0], comp, float(v)))
    return rows

def axes(t, u):
    s = -1 if u in (7, 8) else 1
    return np.array([t['e_d'], s*t['e_b'], -t['e_o']])

class Landmarks:
    def __init__(self, P, tabdir='tab', runs=None):
        self.P = P
        self.G = pickle.load(open(f'{DATA}/{P}/geom_fe.pkl', 'rb'))['teeth']
        rows = []
        for fn in sorted(glob.glob(f'{DATA}/{P}/{tabdir}/tables_step*.md')): rows += parse_tables(fn)
        sfx = '' if tabdir == 'tab' else '_pdl068'
        runs = runs or {g: pickle.load(open(f'{DATA}/{P}/run_{g}.pkl' if not sfx else f'{DATA}/{P}/run_{ {"hg":"hg","ca_precision_cut":"pc","ca_aligner_hook":"cah","ca_metal_hook":"cmh"}[g]}{sfx}.pkl', 'rb'))
                        for g in set(r[0] for r in rows) if g != 'ca_control' or not sfx}
        D = {}
        for g, s, path, lm, comp, v in rows:
            if g in runs: D.setdefault((path, lm), {}).setdefault((g, int(s)), {})[comp] = v
        self.order = list(D.keys()); self.p = {}; self.res = 0
        for key, obs in D.items():
            u = UNN[PATH_TOOTH[key[0]]]; A = axes(self.G[u], u)
            lhs, rhs = [], []
            for (g, s), d in obs.items():
                T = runs[g]['hist'][s-1]['T'][u]; y = np.array([d['AP'], d['BP'], d['V']])
                lhs.append(A @ (T[:3, :3] - np.eye(3))); rhs.append(y - A @ T[:3, 3])
            lhs = np.vstack(lhs); rhs = np.concatenate(rhs)
            p = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
            self.res = max(self.res, np.abs(lhs @ p - rhs).max()); self.p[key] = (u, p)

    def measure(self, T):
        """T: {unn: 4x4}. -> {(path, landmark): (AP, BP, V)} (table precision is 1e-4 mm)."""
        out = {}
        for key, (u, p) in self.p.items():
            A = axes(self.G[u], u); M = T[u]
            out[key] = A @ (M[:3, :3] @ p + M[:3, 3] - p)
        return out
