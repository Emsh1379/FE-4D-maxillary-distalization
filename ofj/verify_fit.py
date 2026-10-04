import sys, os, pickle, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from scipy.spatial import cKDTree
from mesh_io import boundary_faces

def tri_sample(pts, tris, n=4):
    a,b = np.meshgrid(np.linspace(0,1,n), np.linspace(0,1,n)); m=(a+b)<=1
    a,b=a[m],b[m]; v=pts[tris]
    return (v[:,0][:,None,:]+(v[:,1]-v[:,0])[:,None,:]*a[None,:,None]
            +(v[:,2]-v[:,0])[:,None,:]*b[None,:,None]).reshape(-1,3)

S=pickle.load(open('build/stage4.pkl','rb'))
nodes,teeth,att,A = S['nodes'],S['teeth'],S['attachments'],S['aligner']
inner = S['aligner_inner_verts']
# dental surface incl. attachments
tri=[]
for u,t in teeth.items():
    tt=t['tets']
    if u in att: tt=np.vstack([tt,att[u]['tets']])
    tri.append(boundary_faces(tt))
tri=np.vstack(tri)
dent = tri_sample(nodes,tri,5)
d,_ = cKDTree(dent).query(inner)
print(f"aligner inner surface -> tooth/attachment distance (mm):")
print(f"  min {d.min():.3f}  p5 {np.percentile(d,5):.3f}  median {np.median(d):.3f} "
      f" p95 {np.percentile(d,95):.3f}  max {d.max():.3f}")
print(f"  within 0.15 mm of the dentition: {100*(d<0.15).mean():.1f}% of inner nodes")
# per-tooth coverage
pts=[];ids=[]
for u,t in teeth.items():
    cn=np.unique(t['crown_faces']); pts.append(nodes[cn]); ids.append(np.full(len(cn),u))
pts=np.vstack(pts); ids=np.concatenate(ids)
_,nn=cKDTree(pts).query(inner); vt=ids[nn]
print("\nper-tooth aligner inner nodes / contact-capable (<0.2mm):")
for u in sorted(teeth):
    m=vt==u; print(f"  UNN{u:3d} {teeth[u]['name']:<20} nodes {m.sum():5d}  close {int((d[m]<0.2).sum()):5d}")
# attachment engagement
print("\nattachment pocket engagement:")
for u,a in att.items():
    ap=tri_sample(nodes,a['top_faces'],4)
    dd,_=cKDTree(inner).query(ap)
    print(f"  UNN{u:3d} {a['label']:<34} aligner-to-attachment-face min {dd.min():.3f} median {np.median(dd):.3f} mm")
np.save('build/inner_gap.npy', d)
