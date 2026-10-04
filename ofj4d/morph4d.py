"""Aligner staging morph and wear-in seating.

The paper drives the aligner's shape change with the temperature changing method:
a 1 mm band either side of each interproximal midpoint is given a thermal strain
so the appliance lengthens between the teeth without remeshing.  The purpose is
to deform the CA to the next prescribed tooth positions while keeping the mesh.

Here the same end is reached exactly rather than approximately: each aligner node
carries smooth blend weights over the teeth (unity inside a tooth's territory,
blended across a 2 mm interproximal band -- the same band width the paper uses),
and the shell is deformed by the weighted prescribed tooth transforms.  The mesh
is never rebuilt, and the interdental stretch is precisely the prescribed one.

'Prior to each iteration, the CA from the previous step was removed and
 regenerated, followed by the application of a best-fit algorithm to match the
 inner surface of the CA with the dental crowns, simulating a wear-in process.'
-> seat() performs that rigid best fit onto the CURRENT (actual) crowns.
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy.spatial import cKDTree
import spec

BLEND = 2.0            # mm, interproximal blend band (paper: 1 mm mesial + 1 mm distal)


class Morpher:
    def __init__(self, S, arch):
        self.S = S; self.arch = arch
        A = S.aligner_rest
        # distance from every aligner node to each tooth's crown+attachment cloud
        D = np.zeros((len(A), len(S.keep)))
        self.trees = {}
        for i, u in enumerate(S.keep):
            T = S.tooth[u]
            pts = T['surf_rest']
            tree = cKDTree(pts); self.trees[u] = tree
            D[:, i], _ = tree.query(A)
        dmin = D.min(1, keepdims=True)
        w = np.clip(1.0 - (D - dmin)/BLEND, 0.0, 1.0)**2
        self.W = w/w.sum(1, keepdims=True)
        self.owner = np.array(S.keep)[D.argmin(1)]

        # for the wear-in fit: material target of each inner vertex on its tooth
        iv = np.unique(S.aligner_inner_faces)
        self.inner = iv
        self.inner_tooth = self.owner[iv]
        self.inner_target_rest = np.zeros((len(iv), 3))
        for u in S.keep:
            k = self.inner_tooth == u
            if k.any():
                _, j = self.trees[u].query(A[iv[k]])
                self.inner_target_rest[k] = S.tooth[u]['surf_rest'][j]

    # ------------------------------------------------------------------ morph --
    def shape(self, step):
        """Aligner nodes as manufactured for the prescribed positions at `step`."""
        A = self.S.aligner_rest
        out = np.zeros_like(A)
        for i, u in enumerate(self.S.keep):
            T = self.arch.prescribed_transform(u, step)
            out += self.W[:, i:i+1]*(A @ T[:3, :3].T + T[:3, 3])
        return out

    # ------------------------------------------------------------------- seat --
    def seat(self, aligner, state, step):
        """Rigid wear-in of the appliance onto the ACTUAL crowns.

        `aligner` has ALREADY been morphed to the prescribed positions of this step,
        so the seating transform must be fitted from the PRESCRIBED crown points to
        the ACTUAL ones.  Fitting from the rest points instead applies the staged
        motion a second time, which leaves residual interference even when a tooth
        sits exactly where the appliance was made for it.
        """
        pre = np.zeros_like(self.inner_target_rest)
        act = np.zeros_like(self.inner_target_rest)
        for u in self.S.keep:
            k = self.inner_tooth == u
            if not k.any():
                continue
            Tp = self.arch.prescribed_transform(u, step)
            Ta = state.T[u]
            pre[k] = self.inner_target_rest[k] @ Tp[:3, :3].T + Tp[:3, 3]
            act[k] = self.inner_target_rest[k] @ Ta[:3, :3].T + Ta[:3, 3]
        mode = os.environ.get('OFJ_SEAT', 'full')     # full | trans | none (diagnostic)
        if mode == 'none':
            return aligner.copy()
        if mode == 'trans':
            return aligner + (act.mean(0) - pre.mean(0))
        R, t = kabsch(pre, act)
        return aligner @ R.T + t

    def report(self, step=spec.N_STEPS):
        A0 = self.S.aligner_rest; A1 = self.shape(step)
        d = np.linalg.norm(A1-A0, axis=1)
        lines = [f"aligner morph at step {step}: node displacement "
                 f"max {d.max():.3f} mm, mean {d.mean():.3f} mm"]
        for u in self.S.keep:
            k = self.owner == u
            lines.append(f"   FDI {spec.FDI[u]:2d}  {k.sum():5d} nodes  "
                         f"moved {d[k].mean():.3f} mm (designed "
                         f"{spec.total_prescribed(u):.2f})")
        return "\n".join(lines)


def kabsch(P, Q):
    """Rigid transform R,t minimising |R P + t - Q|."""
    cp = P.mean(0); cq = Q.mean(0)
    H = (P-cp).T @ (Q-cq)
    U, s, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, cq - R @ cp
