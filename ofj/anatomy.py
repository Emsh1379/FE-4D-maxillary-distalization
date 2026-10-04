"""Split the Open-Full-Jaw maxilla into individual teeth / PDLs and attach UNN labels."""
import json, numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components as _cc

TAG_TEETH, TAG_PDL, TAG_BONE = 1, 2, 3

# Universal Numbering System, maxillary arch
UNN_NAME = {1:'UR3M',2:'UR7 2nd molar',3:'UR6 1st molar',4:'UR5 2nd premolar',
            5:'UR4 1st premolar',6:'UR3 canine',7:'UR2 lateral',8:'UR1 central',
            9:'UL1 central',10:'UL2 lateral',11:'UL3 canine',12:'UL4 1st premolar',
            13:'UL5 2nd premolar',14:'UL6 1st molar',15:'UL7 2nd molar',16:'UL3M'}
FDI = {1:'18',2:'17',3:'16',4:'15',5:'14',6:'13',7:'12',8:'11',
       9:'21',10:'22',11:'23',12:'24',13:'25',14:'26',15:'27',16:'28'}


def split_bodies(tets_sub, nnodes):
    """Connected components of a tet subset, by shared nodes. Returns per-tet label."""
    e = np.concatenate([tets_sub[:, [0,1]], tets_sub[:, [0,2]], tets_sub[:, [0,3]],
                        tets_sub[:, [1,2]], tets_sub[:, [1,3]], tets_sub[:, [2,3]]])
    g = coo_matrix((np.ones(len(e), np.int8), (e[:,0], e[:,1])), shape=(nnodes, nnodes))
    n, lab = _cc(g, directed=False)
    tl = lab[tets_sub[:, 0]]
    uniq, tl = np.unique(tl, return_inverse=True)
    return tl, len(uniq)


def load_axes(path):
    """teeth_axes_maxilla.json -> {unn: {'c':centroid, 'mesial':u, 'labial':u, 'long':u}}"""
    raw = json.load(open(path))
    out = {}
    for k, v in raw.items():
        c = np.array(v['c'], float)
        out[int(k)] = dict(c=c,
                           ax_x=np.array(v['x'], float) - c,
                           ax_y=np.array(v['y'], float) - c,
                           ax_z=np.array(v['z'], float) - c)
    for d in out.values():
        for k in ('ax_x', 'ax_y', 'ax_z'):
            d[k] = d[k] / np.linalg.norm(d[k])
    return out
