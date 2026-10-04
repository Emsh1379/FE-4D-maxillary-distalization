"""Linear-elastic 4-node tetrahedron FE core (units: N, mm, MPa)."""
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix


def isotropic_D(E, nu):
    lam = E*nu/((1+nu)*(1-2*nu)); mu = E/(2*(1+nu))
    D = np.zeros((6,6))
    D[:3,:3] = lam
    D[0,0]=D[1,1]=D[2,2] = lam+2*mu
    D[3,3]=D[4,4]=D[5,5] = mu
    return D


def shape_grads(nodes, tets):
    """Return (grad N (n,3,4), volume (n,))."""
    p = nodes[tets]                                  # (n,4,3)
    M = np.concatenate([np.ones((len(tets),4,1)), p], axis=2)   # (n,4,4)
    detM = np.linalg.det(M)
    C = np.linalg.inv(M)
    return C[:,1:4,:], detM/6.0


def B_matrix(G):
    n = G.shape[0]
    B = np.zeros((n,6,12))
    for i in range(4):
        gx,gy,gz = G[:,0,i], G[:,1,i], G[:,2,i]
        B[:,0,3*i+0]=gx; B[:,1,3*i+1]=gy; B[:,2,3*i+2]=gz
        B[:,3,3*i+0]=gy; B[:,3,3*i+1]=gx
        B[:,4,3*i+1]=gz; B[:,4,3*i+2]=gy
        B[:,5,3*i+0]=gz; B[:,5,3*i+2]=gx
    return B


def assemble(nodes, tets, mat_id, materials, chunk=40000, verbose=True):
    """materials: {mat_id: (E, nu)}.  Returns CSR stiffness (ndof x ndof)."""
    nn = len(nodes); ndof = 3*nn
    Ds = {m: isotropic_D(*materials[m]) for m in materials}
    parts = []
    for s in range(0, len(tets), chunk):
        t = tets[s:s+chunk]; m = mat_id[s:s+chunk]
        G, V = shape_grads(nodes, t)
        if (V <= 0).any():
            t = t.copy(); bad = V <= 0
            t[bad] = t[bad][:, [0,1,3,2]]
            G, V = shape_grads(nodes, t)
        B = B_matrix(G)
        Ke = np.empty((len(t),12,12))
        for mm in np.unique(m):
            k = m == mm
            Ke[k] = np.einsum('nki,kl,nlj->nij', B[k], Ds[mm], B[k]) * V[k][:,None,None]
        dofs = (3*t[:,:,None] + np.arange(3)[None,None,:]).reshape(len(t),12)
        r = np.repeat(dofs, 12, axis=1).ravel()
        c = np.tile(dofs, (1,12)).ravel()
        parts.append(coo_matrix((Ke.ravel(), (r, c)), shape=(ndof,ndof)).tocsr())
        del B, Ke, r, c, dofs
        if verbose:
            print(f"    assembled {min(s+chunk,len(tets))}/{len(tets)} elements", flush=True)
        # tree-reduce to bound peak memory
        while len(parts) > 1 and parts[-1].nnz >= parts[-2].nnz:
            b = parts.pop(); a = parts.pop(); parts.append(a+b)
    K = parts[0]
    for p in parts[1:]:
        K = K + p
    return K.tocsr()


def rigid_body_modes(nodes):
    nn = len(nodes); B = np.zeros((3*nn, 6))
    for d in range(3):
        B[d::3, d] = 1.0
    x = nodes - nodes.mean(0)
    B[0::3,3] = -x[:,1]; B[1::3,3] =  x[:,0]
    B[1::3,4] = -x[:,2]; B[2::3,4] =  x[:,1]
    B[0::3,5] =  x[:,2]; B[2::3,5] = -x[:,0]
    return B


def element_strain_stress(nodes, tets, mat_id, materials, u):
    G, V = shape_grads(nodes, tets)
    neg = V <= 0
    if neg.any():
        tets = tets.copy(); tets[neg] = tets[neg][:, [0,1,3,2]]
        G, V = shape_grads(nodes, tets)
    B = B_matrix(G)
    ue = u.reshape(-1,3)[tets].reshape(len(tets),12)
    eps = np.einsum('nij,nj->ni', B, ue)
    sig = np.zeros_like(eps)
    for mm in np.unique(mat_id):
        k = mat_id == mm
        sig[k] = eps[k] @ isotropic_D(*materials[mm]).T
    s = sig
    vm = np.sqrt(0.5*((s[:,0]-s[:,1])**2 + (s[:,1]-s[:,2])**2 + (s[:,2]-s[:,0])**2)
                 + 3*(s[:,3]**2 + s[:,4]**2 + s[:,5]**2))
    hyd = (s[:,0]+s[:,1]+s[:,2])/3.0
    # principal stresses
    n = len(s)
    T = np.zeros((n,3,3))
    T[:,0,0]=s[:,0]; T[:,1,1]=s[:,1]; T[:,2,2]=s[:,2]
    T[:,0,1]=T[:,1,0]=s[:,3]; T[:,1,2]=T[:,2,1]=s[:,4]; T[:,0,2]=T[:,2,0]=s[:,5]
    pr = np.linalg.eigvalsh(T)
    return dict(eps=eps, sig=sig, vm=vm, hyd=hyd, p_min=pr[:,0], p_max=pr[:,2], V=V)
