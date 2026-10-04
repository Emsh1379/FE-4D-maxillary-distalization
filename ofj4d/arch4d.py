"""Arch form and the 'V pattern' staging schedule.

Coordinate system exactly as the paper defines it:
    'The occlusal plane was delineated based on specific reference points, namely,
     the mesial-buccal cusps of the upper first molars and the central incisors'
     midpoint.  The origin for the coordinates was set at the central incisors'
     midpoint.  The X-axis was established parallel to the line connecting the
     mesial-buccal cusps of the upper first molars, while the Y-axis,
     perpendicular to the X-axis, was also defined.'

    'Under the occlusion view, the center point of each dental crown was determined
     and recorded to draw the arch form with the polynomial regression algorithm ...
     fourth-order polynomials were selected for curve fitting.  During staging, all
     the teeth except the incisors move along the fitting curve, and the incisors
     move palatally.'
"""
import sys, os, pickle
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spec
from inputs import load as load_inputs

# how a tooth's socket in the appliance is displaced for its prescribed crown movement:
# 'translate' (default, bodily) or 'tip' (diagnostic, rotation about the apical third)
PRESCRIBE = os.environ.get('OFJ_PRESCRIBE', 'translate')
TIP_PIVOT = float(os.environ.get('OFJ_TIP_PIVOT', 0.83))
# blend between the two socket kinematics: the tipping rotation is scaled by
# TIP_BLEND while the crown centre still lands exactly on its prescription, so
# 0 = bodily translation, 1 = full tipping about TIP_PIVOT (equivalently the pivot
# moves apically to TIP_PIVOT-distance / TIP_BLEND)
TIP_BLEND = float(os.environ.get('OFJ_TIP_BLEND', 1.0))
# optional separate blend for the incisors (default: the same as the other teeth);
# 0 plans a bodily palatal movement of the incisors while canine..molars tip
TIP_BLEND_INCISOR = float(os.environ.get('OFJ_TIP_BLEND_INCISOR', TIP_BLEND))


class ArchForm:
    def __init__(self, S, full_model='build/model.pkl'):
        IN, _ = load_inputs(full_model)

        # ---- occlusal plane + axes -------------------------------------------
        mb_R = IN['mb_cusp'][16]
        mb_L = IN['mb_cusp'][26]
        inc = 0.5*(IN['crown_c'][8] + IN['crown_c'][9])         # 11 / 21 midpoint
        ex = mb_R - mb_L; ex /= np.linalg.norm(ex)
        n = np.cross(mb_R - inc, mb_L - inc); n /= np.linalg.norm(n)
        if n @ IN['e_o_16'] < 0:
            n = -n                                   # +n = occlusal
        ey = np.cross(n, ex); ey /= np.linalg.norm(ey)
        # +ey must point posteriorly (towards the molars)
        if (IN['crown_c'][3] - inc) @ ey < 0:
            ey = -ey; ex = -ex
        self.origin, self.ex, self.ey, self.en = inc, ex, ey, n

        # ---- crown centre points of the half arch, in the occlusal view ------
        self.S = S
        P = np.array([S.tooth[u]['crown_c'] for u in S.keep])
        self.xy = np.stack([(P-inc) @ ex, (P-inc) @ ey], 1)
        self.z = (P-inc) @ n

        # ---- 4th-order polynomial arch form ----------------------------------
        # the maxillary arch is single valued in y(x) only over half the arch; fit
        # x as a function of y (y runs anterior -> posterior) to stay single valued
        self.coef = np.polyfit(self.xy[:, 1], self.xy[:, 0], 4)
        self.dcoef = np.polyder(self.coef)
        yy = np.linspace(self.xy[:, 1].min()-4, self.xy[:, 1].max()+4, 4000)
        xx = np.polyval(self.coef, yy)
        ds = np.hypot(np.gradient(xx), np.gradient(yy))
        self.s_tab = np.concatenate([[0], np.cumsum(ds)[:-1]])
        self.y_tab, self.x_tab = yy, xx
        resid = self.xy[:, 0] - np.polyval(self.coef, self.xy[:, 1])
        self.fit_rms = float(np.sqrt((resid**2).mean()))

        # arc-length coordinate of each tooth
        self.s0 = {u: float(np.interp(self.xy[i, 1], self.y_tab, self.s_tab))
                   for i, u in enumerate(S.keep)}
        # offset of the crown centre from the curve, in the curve's own frame
        self.offset = {}
        for i, u in enumerate(S.keep):
            p, t, nrm = self.frame(self.s0[u])
            d = self.xy[i] - p
            self.offset[u] = (float(d @ t), float(d @ nrm), float(self.z[i]))

    @staticmethod
    def _mb_cusp(full, unn):
        """Mesial-buccal cusp = most occlusal point of the mesiobuccal quadrant."""
        t = full['teeth'][unn]
        cn = np.unique(t['crown_faces'])
        P = full['nodes'][cn]
        d = P - t['crown_c']
        q = (d @ t['e_b'] > 0) & (d @ t['e_d'] < 0)       # buccal + mesial
        if q.sum() < 5:
            q = d @ t['e_b'] > 0
        return P[q][np.argmax(P[q] @ t['e_o'])]

    def frame(self, s):
        """Point, unit tangent (towards +y = distal) and in-plane normal at arc length s."""
        y = np.interp(s, self.s_tab, self.y_tab)
        x = np.polyval(self.coef, y)
        dxdy = np.polyval(self.dcoef, y)
        t = np.array([dxdy, 1.0]); t /= np.linalg.norm(t)
        nrm = np.array([-t[1], t[0]])
        return np.array([x, y]), t, nrm

    def to3d(self, xy, z):
        return self.origin + self.ex*xy[0] + self.ey*xy[1] + self.en*z

    # --------------------------------------------------------- prescription --
    def prescribed_transform(self, u, step):
        """4x4 transform taking tooth u from step 0 to its PRESCRIBED position at `step`.

        Canine..second molar translate along the arch form; incisors translate
        palatally.
        """
        T = np.eye(4)
        a, b = spec.V_PATTERN[u]
        k = max(0, min(step, b) - a + 1)
        if k == 0:
            return T
        if u in (7, 8):                                  # incisors: palatal
            d = -self.en*0 - self._palatal(u)*(k*spec.step_mm(u))
            T[:3, 3] = d
            return self._tipped(u, T) if PRESCRIBE == 'tip' else T
        # Pure translation along the arch form.  The paper's TCM applies a LINEAR
        # expansion along the Ci-Cj line joining crown centres, i.e. the socket is
        # displaced, not rotated.  Rotating the socket to follow the tangent makes
        # aligner nodes far from the crown centre travel up to twice the prescribed
        # distance, which over-drives the interference (0.18 mm for a 0.10 mm step)
        # and lets the tooth overshoot its own prescribed position.
        s_new = self.s0[u] + k*spec.step_mm(u)
        p0, t0, n0 = self.frame(self.s0[u])
        p1, t1, n1 = self.frame(s_new)
        off = self.offset[u]
        old_xy = p0 + t0*off[0] + n0*off[1]
        new_xy = p1 + t1*off[0] + n1*off[1]
        T[:3, 3] = self.to3d(new_xy, off[2]) - self.to3d(old_xy, off[2])
        return self._tipped(u, T) if PRESCRIBE == 'tip' else T

    def _tipped(self, u, T):
        """The same crown-centre displacement, delivered as a rotation about a pivot
        on the long axis, TIP_PIVOT of the crown-apex distance below the crown point
        (0.83 = middle of the apical third, where Mao et al. report the centre of
        rotation).  Diagnostic: the translated socket holds the crown's orientation."""
        t = self.S.tooth[u]
        P = t['surf_rest'][:t['surf_n']]; cp = P[np.argmax(P @ t['e_o'])]
        Q = t['pdl_rest']; rp = Q[np.argmin(Q @ t['e_o'])]
        L = np.linalg.norm(cp - rp); LA = (cp - rp)/L
        c = t['crown_c']; dvec = T[:3, 3].copy()
        dirn = dvec - (dvec @ LA)*LA
        if np.linalg.norm(dirn) < 1e-12:
            return T
        d = np.linalg.norm(dirn); dirn /= d
        p = cp - TIP_PIVOT*L*LA
        blend = TIP_BLEND_INCISOR if u in (7, 8) else TIP_BLEND
        th = blend*d/((c - p) @ LA)
        ax = np.cross(LA, dirn); ax /= np.linalg.norm(ax)
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        R = np.eye(3) + np.sin(th)*K + (1 - np.cos(th))*K @ K
        out = np.eye(4); out[:3, :3] = R; out[:3, 3] = p - R @ p
        out[:3, 3] += dvec - (R @ c + out[:3, 3] - c)     # crown centre exactly as prescribed
        return out

    def _palatal(self, u):
        """Unit palatal direction for an incisor (in the occlusal plane)."""
        eb = self.S.tooth[u]['e_b']
        p = eb - (eb @ self.en)*self.en
        return p/np.linalg.norm(p)

    def report(self):
        out = [f"arch form: 4th-order fit, rms residual {self.fit_rms:.3f} mm",
               f"  origin (incisor midpoint) {np.round(self.origin,2)}",
               f"  ex {np.round(self.ex,3)}  ey {np.round(self.ey,3)}  "
               f"occlusal normal {np.round(self.en,3)}", "",
               f"{'tooth':<8}{'x':>8}{'y':>8}{'s':>8}   prescribed total"]
        for i, u in enumerate(self.S.keep):
            T = self.prescribed_transform(u, spec.N_STEPS)
            d = np.linalg.norm(T[:3, 3] + (T[:3, :3] - np.eye(3)) @ self.S.tooth[u]['crown_c'])
            out.append(f"FDI {spec.FDI[u]:<4}{self.xy[i,0]:8.2f}{self.xy[i,1]:8.2f}"
                       f"{self.s0[u]:8.2f}   {d:.3f} mm "
                       f"({'palatal' if u in (7,8) else 'distal'}, "
                       f"designed {spec.total_prescribed(u):.2f})")
        return "\n".join(out)


if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model4d import Static
    S = Static()
    A = ArchForm(S)
    print(A.report())
