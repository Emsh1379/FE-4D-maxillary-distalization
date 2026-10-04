"""Runtime inputs.

`build4d/slim.pkl` holds the handful of quantities ofj4d needs from the full
99.6 MB build/model.pkl (two mesiobuccal cusps, four crown centres, one axis pair
and the two alveolar crest points).  It is used when present so the model can be
shipped and run without the full dataset; otherwise the values are recomputed
from build/model.pkl and are identical.
"""
import os, pickle
import numpy as np

SLIM = os.path.join('build4d', 'slim.pkl')


def load(model='build/model.pkl'):
    if os.path.exists(SLIM):
        return pickle.load(open(SLIM, 'rb')), None
    M = pickle.load(open(model, 'rb'))
    teeth = M['teeth']

    def mb_cusp(unn):
        t = teeth[unn]
        cn = np.unique(t['crown_faces'])
        P = M['nodes'][cn]
        d = P - t['crown_c']
        q = (d @ t['e_b'] > 0) & (d @ t['e_d'] < 0)
        if q.sum() < 5:
            q = d @ t['e_b'] > 0
        return P[q][np.argmax(P[q] @ t['e_o'])]

    bone = M['nodes'][np.unique(M['tets'][M['tags'] == 3])]
    crest = {}
    for side in ('buccal', 'palatal'):
        ta, tb = teeth[3], teeth[2]
        e_o = ta['e_o']/np.linalg.norm(ta['e_o'])
        es = 0.5*(ta['e_b'] + tb['e_b']); es /= np.linalg.norm(es)
        if side == 'palatal':
            es = -es
        mid = 0.5*(M['nodes'][np.unique(ta['tets'])].mean(0)
                   + M['nodes'][np.unique(tb['tets'])].mean(0))
        d = bone - mid
        near = (np.abs(d @ ta['e_d']) < 2.5) & ((d @ es) > 0)
        sel = bone[near]
        crest[side] = sel[(sel @ e_o).argmax()]
    return dict(mb_cusp={16: mb_cusp(3), 26: mb_cusp(14)},
                crown_c={u: teeth[u]['crown_c'] for u in (3, 8, 9, 14)},
                e_o_16=teeth[3]['e_o'], e_b_16=teeth[3]['e_b'],
                crest=crest), M
