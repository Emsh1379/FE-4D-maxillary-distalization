import sys, os, pickle, numpy as np
sys.path.insert(0,'ofj')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mesh_io import boundary_faces
M=pickle.load(open('build/model.pkl','rb'))
nodes=M['nodes']; A=M['aligner']; nS=M['nS']
tooth_tris=np.vstack([boundary_faces(t['tets']) for t in M['teeth'].values()])
att_tris=np.vstack([boundary_faces(a['tets']) for a in M['att'].values()])
alg_tris=boundary_faces(A['tets'])

hook=np.zeros(len(A['nodes']),bool)
for L in M['loads'].values(): hook[L['nodes']-nS]=True
is_hook=hook[alg_tris].any(1)
print("hook triangles on aligner:",is_hook.sum())

def shade(ax,pts,tris,color,alpha=1.0,light=np.array([0.35,-0.55,-0.75])):
    if len(tris)==0: return
    v=pts[tris]; n=np.cross(v[:,1]-v[:,0],v[:,2]-v[:,0]); L=np.linalg.norm(n,axis=1)
    k=L>1e-12; v=v[k]; n=n[k]/L[k,None]
    b=0.38+0.62*np.clip(n@(light/np.linalg.norm(light)),0,1)
    c=np.array(matplotlib.colors.to_rgb(color))[None,:]*b[:,None]
    return v,np.clip(c,0,1)

def draw(ax, layers):
    V=[];C=[]
    for pts,tris,col,al in layers:
        r=shade(ax,pts,tris,col)
        if r: V.append(r[0]); C.append(r[1])
    V=np.vstack(V); C=np.vstack(C)
    o=np.argsort(V[:,:,1].mean(1))
    ax.add_collection3d(Poly3DCollection(V[o],facecolors=C[o],edgecolors='none'))

fig=plt.figure(figsize=(16,11),facecolor='white')
views=[('RIGHT buccal',(8,170)),('LEFT buccal',(8,-10)),
       ('occlusal (from below)',(-88,-90)),('anterior-right oblique',(10,140))]
allp=np.vstack([nodes[np.unique(tooth_tris)],A['nodes'][np.unique(alg_tris)]])
for i,(title,(el,az)) in enumerate(views):
    ax=fig.add_subplot(2,2,i+1,projection='3d'); ax.view_init(el,az)
    draw(ax,[(nodes,tooth_tris,'#efe6d2',1),(nodes,att_tris,'#2f7fd0',1),
             (A['nodes'],alg_tris[~is_hook],'#a9dcea',1),
             (A['nodes'],alg_tris[is_hook],'#d92b2b',1)])
    ctr=allp.mean(0); r=np.ptp(allp,0).max()/2*0.70
    ax.set_xlim(ctr[0]-r,ctr[0]+r); ax.set_ylim(ctr[1]-r,ctr[1]+r); ax.set_zlim(ctr[2]-r,ctr[2]+r)
    ax.set_axis_off(); ax.set_box_aspect((1,1,1)); ax.set_title(title,fontsize=11)
fig.suptitle("Force application: 250 gf distal per side on the aligner buccal wall at each canine\n"
             "red = loaded patch (precision cut / button)   blue = composite attachments   pale blue = aligner",
             fontsize=12)
plt.tight_layout(rect=[0,0,1,0.94]); plt.savefig('out/geometry/hook_location.png',dpi=120,facecolor='white')
print("ok")
