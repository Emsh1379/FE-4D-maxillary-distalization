"""Node-to-surface contact pairing with Coulomb friction (small-sliding)."""
import numpy as np
from scipy.spatial import cKDTree


def closest_point_on_triangles(p, a, b, c):
    """Vectorised closest point of p on triangles (a,b,c). Returns (q, bary)."""
    ab, ac, ap = b-a, c-a, p-a
    d1 = np.einsum('ij,ij->i', ab, ap); d2 = np.einsum('ij,ij->i', ac, ap)
    bp = p-b; d3 = np.einsum('ij,ij->i', ab, bp); d4 = np.einsum('ij,ij->i', ac, bp)
    cp = p-c; d5 = np.einsum('ij,ij->i', ab, cp); d6 = np.einsum('ij,ij->i', ac, cp)
    va = d3*d6 - d5*d4; vb = d5*d2 - d1*d6; vc = d1*d4 - d3*d2
    denom = np.where(np.abs(va+vb+vc) < 1e-30, 1e-30, va+vb+vc)
    v = vb/denom; w = vc/denom; u = 1.0-v-w
    # region tests
    u = np.clip(u, 0, 1); v = np.clip(v, 0, 1); w = np.clip(w, 0, 1)
    m = (d1 <= 0) & (d2 <= 0);                 u[m], v[m], w[m] = 1, 0, 0
    m = (d3 >= 0) & (d4 <= d3);                u[m], v[m], w[m] = 0, 1, 0
    m = (d6 >= 0) & (d5 <= d6);                u[m], v[m], w[m] = 0, 0, 1
    m = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    t = np.where(m, d1/np.where((d1-d3)==0,1e-30,d1-d3), 0)
    u[m], v[m], w[m] = 1-t[m], t[m], 0
    m = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    t = np.where(m, d2/np.where((d2-d6)==0,1e-30,d2-d6), 0)
    u[m], v[m], w[m] = 1-t[m], 0, t[m]
    m = (va <= 0) & ((d4-d3) >= 0) & ((d5-d6) >= 0)
    dd = (d4-d3)+(d5-d6); t = np.where(m, (d4-d3)/np.where(dd==0,1e-30,dd), 0)
    u[m], v[m], w[m] = 0, 1-t[m], t[m]
    s = u+v+w; u, v, w = u/s, v/s, w/s
    q = a*u[:,None] + b*v[:,None] + c*w[:,None]
    return q, np.stack([u,v,w],1)


def build_pairs(nodes, slave, master_tris, k=16, max_gap=1.2):
    """Pair each slave node with the closest master triangle. Returns dict."""
    A = nodes[master_tris[:,0]]; B = nodes[master_tris[:,1]]; C = nodes[master_tris[:,2]]
    cen = (A+B+C)/3.0
    tree = cKDTree(cen)
    P = nodes[slave]
    _, cand = tree.query(P, k=k)
    best_d = np.full(len(P), np.inf); best_t = np.zeros(len(P), np.int64)
    best_b = np.zeros((len(P),3)); best_q = np.zeros((len(P),3))
    for j in range(k):
        t = cand[:,j]
        q, bary = closest_point_on_triangles(P, A[t], B[t], C[t])
        d = np.linalg.norm(P-q, axis=1)
        m = d < best_d
        best_d[m] = d[m]; best_t[m] = t[m]; best_b[m] = bary[m]; best_q[m] = q[m]
    n = np.cross(B[best_t]-A[best_t], C[best_t]-A[best_t])
    n /= np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-12)
    gap = np.einsum('ij,ij->i', P-best_q, n)       # >0 = outside the master body
    keep = best_d <= max_gap
    return dict(slave=slave[keep], tri=master_tris[best_t[keep]], bary=best_b[keep],
                normal=n[keep], gap0=gap[keep], dist=best_d[keep], proj=best_q[keep])
