"""Frictionless contact between adjacent crowns of the rigid teeth (shared by the HG solver,
fixed4d.py, and the aligner solver, solve4d.py).  See ToothContact."""
import os
import numpy as np
from scipy.sparse import csr_matrix

# ------------------------------------------------------------------ tooth-tooth contact --
TOOTH_CONTACT = os.environ.get('OFJ_TOOTH_CONTACT', '1') == '1'
TC_K = float(os.environ.get('OFJ_TC_K', 500.0))      # N/mm per contact point (penalty)
TC_EPS = 0.005                                        # mm, smoothing of the contact onset
TC_BAND = 1.5                                         # mm, candidate points: crown surface this close to the neighbour at rest
TC_TANG = 0.5                                         # mm, a point must project onto its nearest face (sideways offset below this)
TC_PMAX = 0.3                                         # mm, measured penetration deeper than this is a projection flip, not contact


class ToothContact:
    """Frictionless contact between adjacent crowns (teeth are rigid bodies).  Without it the
    headgear slid the first molar through the second (up to 3.5 mm overlap).  Candidate points:
    crown face centroids of each tooth within TC_BAND of its neighbour at rest, both ways.  The
    gap of a point is measured in the neighbour's own (rest) frame against its nearest crown face
    (signed along that face's outward normal); a smooth one-sided penalty
        F = TC_K * TC_EPS * softplus(-gap / TC_EPS)
    pushes the two teeth apart.  Rows are the relative normal motion of the two rigid bodies at
    the point, as for the wire-slot contact."""

    def __init__(self, S, G):
        from scipy.spatial import cKDTree
        V = G['nodes']
        self.surf = {}
        # the neighbour's CLOSED surface (crown + root): on the crown alone, points near its open
        # cervical edge found sideways-facing faces and read as millimetres inside (patient 4, HG)
        for u in S.keep:
            F = np.concatenate([G['teeth'][u]['crown_faces'], G['teeth'][u]['root_faces']])
            tri = V[F]; c = tri.mean(1)
            n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
            cen = V[np.unique(np.concatenate([F, G['teeth'][u]['root_faces']]))].mean(0)
            n[((c - cen)*n).sum(1) < 0] *= -1
            self.surf[u] = (c, n, cKDTree(c))
        keep = list(S.keep)
        self.pts = []                                  # (owner a, other b, rest point)
        crown_c = {}
        for u in S.keep:
            Fc = G['teeth'][u]['crown_faces']; crown_c[u] = V[Fc].mean(1)
        for a, b in zip(keep[:-1], keep[1:]):
            for x, y in ((a, b), (b, a)):
                cx = crown_c[x]
                d, _ = self.surf[y][2].query(cx)
                for p in cx[d < TC_BAND]:
                    self.pts.append((x, y, p))
        self.S = S
        self.N_last = np.zeros(len(self.pts))

    def select(self, T):
        """Current point positions, gaps and normals (fixed during one remodelling iteration)."""
        self.cur = []
        for a, b, pr in self.pts:
            p = T[a][:3, :3] @ pr + T[a][:3, 3]
            pb = T[b][:3, :3].T @ (p - T[b][:3, 3])        # in b's rest frame
            c, n, tree = self.surf[b]
            j = int(tree.query(pb)[1])
            r = pb - c[j]
            g = float(r @ n[j])
            tang = np.linalg.norm(r - g*n[j])
            if tang > TC_TANG or g < -TC_PMAX:
                g = 10.0                                  # not a valid contact reading: inactive
            self.cur.append((p, g, T[b][:3, :3] @ n[j]))

    def rows(self, S, n_ext, cen):
        from model4d import rigid_block
        R_, C_, V_ = [], [], []
        for i, ((a, b, _), (p, g, nv)) in enumerate(zip(self.pts, self.cur)):
            for tooth, sgn in ((a, 1.0), (b, -1.0)):
                B = rigid_block(p[None], cen[tooth])[0]
                R_ += [i]*6; C_ += list(S.q_col[tooth] + np.arange(6)); V_ += list(sgn*(nv @ B))
        return csr_matrix((V_, (R_, C_)), shape=(len(self.pts), n_ext))

    def law(self, dlt):
        """Tangent kd >= 0 and offset f0 so that the contact force on each relative normal dof,
        linearised at the prediction, is -(kd*dlt + f0)."""
        g = np.array([c[1] for c in self.cur]) + dlt
        z = -g/TC_EPS
        F = TC_K*TC_EPS*np.logaddexp(0.0, z)             # >= 0, pushes apart
        k = TC_K/(1.0 + np.exp(-z))                      # = -dF/dg
        self.N_last = F
        return k, -F - k*dlt


