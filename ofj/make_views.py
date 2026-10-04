import sys, os, pickle, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mesh_io import boundary_faces
from export_stl import write_stl

S=pickle.load(open('build/stage4.pkl','rb'))
nodes,tets,tags,teeth,att,A = S['nodes'],S['tets'],S['tags'],S['teeth'],S['attachments'],S['aligner']
os.makedirs('out/geometry',exist_ok=True)

tooth_tris=np.vstack([boundary_faces(t['tets']) for t in teeth.values()])
att_tris=np.vstack([boundary_faces(a['tets']) for a in att.values()])
alg_tris=boundary_faces(A['tets'])
write_stl('out/geometry/teeth.stl', nodes, tooth_tris)
write_stl('out/geometry/attachments.stl', nodes, att_tris)
write_stl('out/geometry/aligner.stl', A['nodes'], alg_tris)
bone_tris=boundary_faces(tets[tags==3]); write_stl('out/geometry/bone.stl', nodes, bone_tris)
pdl_tris=boundary_faces(tets[tags==2]); write_stl('out/geometry/pdl.stl', nodes, pdl_tris)
print("STLs written")

def shade(ax, pts, tris, color, alpha=1.0, light=np.array([0.3,-0.6,-0.75])):
    v=pts[tris]
    n=np.cross(v[:,1]-v[:,0],v[:,2]-v[:,0]); L=np.linalg.norm(n,axis=1)
    keep=L>1e-12; v=v[keep]; n=n[keep]/L[keep,None]
    b=0.35+0.65*np.clip(n@(light/np.linalg.norm(light)),0,1)
    c=np.array(matplotlib.colors.to_rgb(color))[None,:]*b[:,None]
    order=np.argsort(v[:,:,1].mean(1))
    pc=Poly3DCollection(v[order],facecolors=np.clip(c[order],0,1),edgecolors='none',alpha=alpha)
    ax.add_collection3d(pc)

def frame(ax, pts):
    c=pts.mean(0); r=np.ptp(pts,0).max()/2*1.05
    ax.set_xlim(c[0]-r,c[0]+r); ax.set_ylim(c[1]-r,c[1]+r); ax.set_zlim(c[2]-r,c[2]+r)
    ax.set_axis_off(); ax.set_box_aspect((1,1,1))

views=[('occlusal (from below)',(-89,-90)),('right buccal',(8,175)),
       ('left buccal',(8,-5)),('anterior',(6,90))]
allpts=np.vstack([nodes[np.unique(tooth_tris)],A['nodes'][np.unique(alg_tris)]])

fig=plt.figure(figsize=(17,9),facecolor='white')
for i,(name,(el,az)) in enumerate(views):
    ax=fig.add_subplot(2,4,i+1,projection='3d'); ax.view_init(el,az)
    shade(ax,nodes,tooth_tris,'#f2ead8'); shade(ax,nodes,att_tris,'#4da3ff')
    frame(ax,allpts); ax.set_title(f"teeth + attachments\n{name}",fontsize=9)
    ax2=fig.add_subplot(2,4,i+5,projection='3d'); ax2.view_init(el,az)
    shade(ax2,nodes,tooth_tris,'#f2ead8'); shade(ax2,nodes,att_tris,'#4da3ff')
    shade(ax2,A['nodes'],alg_tris,'#7fd4e8',alpha=0.55)
    frame(ax2,allpts); ax2.set_title(f"with clear aligner\n{name}",fontsize=9)
plt.tight_layout(); plt.savefig('out/geometry/model_views.png',dpi=125,facecolor='white')
print("wrote out/geometry/model_views.png")
