"""Build the it=0 contact system once, cache it, and benchmark AMG configurations."""
import sys, os, pickle, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy.sparse import coo_matrix, save_npz, load_npz
from scipy.sparse.linalg import cg
import pyamg
from fem import assemble, rigid_body_modes
from solve import build_P, contact_ops, K_N, K_T, ADJUST

if not os.path.exists('build/sys_A.npz'):
    M = pickle.load(open('build/model.pkl','rb')); C = pickle.load(open('build/contact.pkl','rb'))
    nodes = M['nodes']; ndof = 3*len(nodes)
    mats = {m:(M['MAT'][m][1], M['MAT'][m][2]) for m in M['MAT']}
    K = assemble(nodes, M['elems'], M['mats'], mats, chunk=40000, verbose=False)
    P = build_P(M, ndof); Kr = (P.T@K@P).tocsr(); del K
    f = np.zeros(ndof)
    for L in M['loads'].values():
        nd=L['nodes']; F=L['magnitude']*L['dirn']/len(nd)
        for d in range(3): np.add.at(f, 3*nd+d, F[d])
    dofs,R,a = contact_ops(C)
    gref = np.maximum(0.0, C['gap0']-ADJUST)
    nrm=C['normal']
    Pt = np.eye(3)[None]-np.einsum('ni,nj->nij',nrm,nrm)
    Ke = K_N*np.einsum('ni,nj->nij',a,a) + K_T*np.einsum('nki,nkl,nlj->nij',R,Pt,R)
    Kc = coo_matrix((Ke.ravel(),(np.repeat(dofs,12,axis=1).ravel(),np.tile(dofs,(1,12)).ravel())),
                    shape=(ndof,ndof)).tocsr()
    fc = np.zeros(ndof); np.add.at(fc, dofs.ravel(), ((-K_N*gref)[:,None]*a).ravel())
    A = (Kr + P.T@Kc@P).tocsr(); b = P.T@f + P.T@fc
    save_npz('build/sys_A.npz', A); np.save('build/sys_rhs.npy', b)
    np.save('build/sys_nullspace.npy', P.T@rigid_body_modes(nodes))
    save_npz('build/sys_P.npz', P)
    print("cached system", A.shape, A.nnz/1e6,"M nnz")

A=load_npz('build/sys_A.npz'); b=np.load('build/sys_rhs.npy'); B=np.load('build/sys_nullspace.npy')
print("A:",A.shape,"nnz %.1fM"%(A.nnz/1e6),"  diag min %.3e max %.3e"%(A.diagonal().min(),A.diagonal().max()))
configs=[
 ("jacobi/theta0",    dict(smooth='jacobi', strength=('symmetric',{'theta':0.0}))),
 ("energy/theta0",    dict(smooth='energy', strength=('symmetric',{'theta':0.0}))),
 ("energy/theta.02",  dict(smooth='energy', strength=('symmetric',{'theta':0.02}))),
 ("energy/theta.25",  dict(smooth='energy', strength=('symmetric',{'theta':0.25}))),
]
for name,kw in configs:
    try:
        t=time.time()
        ml=pyamg.smoothed_aggregation_solver(A,B=B,max_coarse=2000,
              presmoother=('gauss_seidel',{'sweep':'symmetric','iterations':2}),
              postsmoother=('gauss_seidel',{'sweep':'symmetric','iterations':2}),**kw)
        ts=time.time()-t
        sizes=[l.A.shape[0] for l in ml.levels]
        it=[0]; t=time.time()
        x,info=cg(A,b,rtol=1e-8,maxiter=400,M=ml.aspreconditioner(cycle='V'),
                  callback=lambda xk: it.__setitem__(0,it[0]+1))
        r=np.linalg.norm(b-A@x)/np.linalg.norm(b)
        print(f"  {name:18s} setup {ts:5.1f}s levels={len(sizes)} {sizes[:6]} "
              f"cg={it[0]:4d} {time.time()-t:6.1f}s res={r:.2e}", flush=True)
    except Exception as e:
        print(f"  {name:18s} FAILED {type(e).__name__}: {e}", flush=True)
