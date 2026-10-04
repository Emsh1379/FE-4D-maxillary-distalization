import numpy as np, struct

def write_stl(path, pts, tris):
    v = pts[tris].astype(np.float32)
    n = np.cross(v[:,1]-v[:,0], v[:,2]-v[:,0])
    n /= np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-12)
    with open(path,'wb') as f:
        f.write(b'\0'*80); f.write(struct.pack('<I', len(tris)))
        rec = np.zeros((len(tris),12), np.float32)
        rec[:,0:3]=n; rec[:,3:6]=v[:,0]; rec[:,6:9]=v[:,1]; rec[:,9:12]=v[:,2]
        buf=np.zeros(len(tris), dtype=[('d','<f4',12),('a','<u2')])
        buf['d']=rec
        f.write(buf.tobytes())
