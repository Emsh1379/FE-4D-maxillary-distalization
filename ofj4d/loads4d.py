"""Anchorage groups: the elastic from a TAD to a precision cut on the aligner.

Buccal TAD group (Mao et al. 2024):
    'the TAD was located in the buccal interradicular space between the first and
     second molars 4 mm above the alveolar crest ... The application of elastics on
     the CAs were all set at the buccal or lingual mesial cervical region of the
     canine to simulate the precision cut ... the elastic forces for all the test
     groups were set as 150 g'

The TAD is a fixed point in the bone.  The precision cut is a material patch on
the aligner, so it travels with the appliance; the line of action is recomputed
from the CURRENT hook position at every iteration, as a real elastic would.
"""
import sys, os, pickle
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy.spatial import cKDTree
import spec
from inputs import load as load_inputs

HOOK_RADIUS = 1.75        # mm, precision-cut patch radius
TAD_PATCH = 0.6           # mm, bone nodes around the TAD point that take the reaction


class Elastic:
    """150 gf elastic, TAD -> precision cut at the canine."""

    def __init__(self, S, state, group='buccal_tad', model='build/model.pkl'):
        self.group = group
        g = spec.GROUPS[group]
        self.enabled = g['elastic']
        self.kind = g.get('hook_kind', 'aligner')
        self.cut_nodes = np.array([], np.int64)
        if not self.enabled:
            self.hook_nodes = np.array([], np.int64)
            self.tad = None
            return

        side = g['side']
        ua, ub = g['tad_between']                 # (16, 17)
        ta, tb = S.tooth[ua], S.tooth[ub]
        e_o = ta['e_o']/np.linalg.norm(ta['e_o'])
        e_side = 0.5*(ta['e_b'] + tb['e_b']); e_side /= np.linalg.norm(e_side)
        if side == 'palatal':
            e_side = -e_side

        # ---- the miniscrew point -------------------------------------------------
        # Preferred: build4d/tad_bone.pkl from tools/tad_from_bone.py -- a point on
        # the cortical surface of the segmented bone, 4 mm apical to that side's
        # alveolar crest, centred in the 16/17 interradicular gap.  The legacy rule
        # (most occlusal bone node on each side of the root midline, offset straight
        # apically) lands on the septum ridge and put both screws in the middle of
        # the septum, 3.5 mm apart (FINDINGS 19-20).  OFJ_TAD_RULE=legacy restores it.
        tb = os.environ.get('OFJ_TAD_FILE', g.get('tad_file', os.path.join('build4d', 'tad_bone.pkl')))   # position variants: tools/tad_from_bone.py
        if g.get('tad_file') and not os.path.exists(tb):
            raise FileNotFoundError(f'{tb} missing: build the IZC screw point first (python tools/izc_point.py)')
        if os.path.exists(tb) and os.environ.get('OFJ_TAD_RULE', 'bone') == 'bone':
            T = pickle.load(open(tb, 'rb'))['sides'][side]
            self.tad = np.asarray(T['tad'], float)
            self.crest = self.tad + e_o*g['tad_above_crest']     # reference only
            self.tad_rule = f'bone surface ({os.path.basename(tb)})'
        else:
            IN, _ = load_inputs(model)
            crest = IN['crest'][side]
            # 'above the alveolar crest' = apical, i.e. away from the occlusal plane
            self.tad = crest - e_o*g['tad_above_crest']
            self.crest = crest
            self.tad_rule = 'legacy septum-ridge crest'

        self.kind = g.get('hook_kind', 'aligner')
        self.magnitude = g.get('force_gf', spec.ELASTIC_FORCE_GF)*spec.GF_TO_N
        self.force_gf = g.get('force_gf', spec.ELASTIC_FORCE_GF)
        self.bone_nodes = np.array([], np.int64)
        if self.kind == 'metal':
            self._metal_hook(S, state, g)
            return
        if self.kind == 'aligner_arm':
            self._metal_hook(S, state, g)                 # identical geometry to the metal hook
            self._aligner_arm(S, state, g)
            return
        # ---- precision cut: mesial cervical region of the canine, on the aligner --
        can = S.tooth[g['hook_tooth']]
        e_b = can['e_b']; e_d = can['e_d']; e_oc = can['e_o']
        if side == 'palatal':
            e_b = -e_b
        cerv = can['gingival_level'] + 0.35*(can['occl_tip'] - can['gingival_level'])
        seed = (can['crown_c'] - e_d*2.2          # mesial
                + e_b*6.0                          # push outside, snapped back below
                + e_oc*(cerv - can['crown_c'] @ e_oc))
        al = state.aligner
        # restrict to the outer surface layer (ids >= n_inner) ...
        cand = np.arange(S.n_inner, getattr(S, 'nA_layers', S.nA))   # layer nodes only, not prism centroids
        # ... lying over the hook tooth itself, on the requested side of it.  The
        # nearest outer node to the seed is not enough: on the palatal side the arch
        # curves, and the closest appliance surface to a point palatal of the
        # canine belongs to the lateral incisor, so the cut landed there.
        hook_u = g['hook_tooth']
        ids = list(S.keep)
        D = np.stack([cKDTree(S.tooth[u]['surf_rest']).query(S.aligner_rest[cand])[0]
                      for u in ids], axis=1)
        on_tooth = np.array(ids)[D.argmin(1)] == hook_u
        sgn = 1.0 if side == 'buccal' else -1.0
        side_ok = ((S.aligner_rest[cand] - can['crown_c']) @ can['e_b'])*sgn > 0
        cand = cand[on_tooth & side_ok]
        if len(cand) == 0:
            raise RuntimeError(f'no aligner surface over the {side} side of FDI {spec.FDI[hook_u]}')
        j = cand[np.argmin(np.linalg.norm(al[cand] - seed, axis=1))]
        centre = al[j]
        self.hook_nodes = cand[np.linalg.norm(al[cand] - centre, axis=1) <= HOOK_RADIUS]
        self.hook_rest = al[self.hook_nodes].copy()
        self.cut_nodes = np.array([], np.int64)
        # With deformable bone the elastic pulls on the bone as well: the equal and
        # opposite force goes into the bone at the TAD point.  With rigid bone the
        # reaction is absorbed by the fixed support, as before.
        self.bone_nodes = np.array([], np.int64)
        if getattr(S, 'deformable_bone', False):
            Bx = S.nodes_rest[S.bone_off:S.bone_off + S.nB]
            dist = np.linalg.norm(Bx - self.tad, axis=1)
            near = np.where(dist <= TAD_PATCH)[0]
            if len(near) < 3:
                near = np.argsort(dist)[:3]
            near = near + S.bone_off
            self.bone_nodes = near[~np.isin(near, S.bone_fixed)]

    # ---------------------------------------------------------------- metal hook --
    def _metal_hook(self, S, state, g):
        """A stainless-steel hook bonded to the canine (rigid with the tooth: E 200 GPa
        against a 0.05 MPa ligament).  Button on the buccal surface in the cervical third,
        mesiodistally centred; a 4 mm arm runs apically, standing METAL_HOOK_STANDOFF mm off
        the surface; the elastic is hooked at its tip.  The force enters the tooth's rigid
        dofs directly as a force at the tip plus its moment about the tooth centroid.  The
        aligner is trimmed around the button: its inner-surface nodes within
        METAL_HOOK_CUTOUT mm of the button are removed from contact (self.cut_nodes)."""
        u = g['hook_tooth']
        t = S.tooth[u]
        self.hook_unn = u
        P = t['surf_rest'][:t['surf_n']]
        eb, ed, eo = t['e_b'], t['e_d'], t['e_o']
        h = P @ eo
        hb = t['gingival_level'] + spec.METAL_HOOK_BASE_H*(t['occl_tip'] - t['gingival_level'])
        band = (np.abs(h - hb) < 0.5)
        Q = P[band]
        q = Q[(Q - t['crown_c']) @ eb > 0]
        md = q @ ed; mdc = 0.5*(md.min() + md.max())
        base = q[np.argmin((md - mdc)**2 + 0.01*((q @ eb).max() - q @ eb))]
        # outward buccal normal at the button
        near = P[np.linalg.norm(P - base, axis=1) < 1.5]
        nb = eb.copy()
        if len(near) >= 6:
            C = near - near.mean(0)
            w, V = np.linalg.eigh(C.T @ C)
            nb = V[:, 0] if V[:, 0] @ eb > 0 else -V[:, 0]
        arm_top = base + nb*spec.METAL_HOOK_STANDOFF
        if spec.METAL_HOOK_TIP == 'screw':
            L = max(0.0, (arm_top - self.tad) @ eo)          # down to the screw's height
        else:
            L = spec.METAL_HOOK_ARM
        tip = arm_top - eo*L                                  # apical = -e_o
        self.hook_base_rest = base
        self.hook_tip_rest = tip
        self.hook_arm = L
        self.hook_nodes = np.array([], np.int64)
        # aligner cut-out: inner-surface aligner nodes near the button
        al = S.aligner_rest
        iv = np.unique(S.aligner_inner_faces)
        self.cut_nodes = iv[np.linalg.norm(al[iv] - base, axis=1) <= spec.METAL_HOOK_CUTOUT]

    # ------------------------------------------------------------- aligner-material arm --
    def _aligner_arm(self, S, state, g):
        """The metal hook's arm (same button site at the CEJ, standoff, length and tip), made of
        aligner material and part of the aligner: its root is a patch of the aligner's outer
        surface at the button site, so the elastic loads the aligner (force + the arm's moment)
        and reaches the teeth through contact.  The arm is a cantilever of the aligner sheet
        (thickness ALIGNER_THICKNESS x width AH_WIDTH, E_ALIGNER) that bends under the elastic,
        which moves its tip (Euler-Bernoulli, small deflection)."""
        u = g['hook_tooth']; t = S.tooth[u]
        al = S.aligner_rest
        cand = np.arange(S.n_inner, getattr(S, 'nA_layers', S.nA))
        ids = list(S.keep)
        D = np.stack([cKDTree(S.tooth[k]['surf_rest']).query(al[cand])[0] for k in ids], axis=1)
        cand = cand[(np.array(ids)[D.argmin(1)] == u) & (((al[cand] - t['crown_c']) @ t['e_b']) > 0)]
        j = cand[np.argmin(np.linalg.norm(al[cand] - self.hook_base_rest, axis=1))]
        self.hook_nodes = cand[np.linalg.norm(al[cand] - al[j], axis=1) <= HOOK_RADIUS]
        self.hook_rest = al[self.hook_nodes].copy()
        self.cut_nodes = np.array([], np.int64)
        self.arm_root_rest = self.hook_rest.mean(0)
        L = self.hook_arm
        eo = t['e_o']
        self.arm_axis = -eo                                  # apical, as the metal arm
        nb = t['e_b'] - (t['e_b'] @ eo)*eo; nb /= np.linalg.norm(nb)
        self.arm_weak = nb                                   # bends buccolingually about the thin section
        self.arm_strong = np.cross(self.arm_axis, nb)        # mesiodistal bending, about the wide section
        th, w = spec.ALIGNER_THICKNESS, spec.AH_WIDTH
        self.arm_I = (w*th**3/12.0, th*w**3/12.0)           # (weak, strong) mm^4
        self.arm_len = max(L, 1e-6)
        self.bone_nodes = np.array([], np.int64)

    def _arm_tip(self, state, tip0):
        """Tip of the aligner arm under the elastic: rigid follow of its root + cantilever bending."""
        E = spec.E_ALIGNER; L = self.arm_len
        tip = tip0
        for _ in range(3):                                  # the pull direction depends on the tip
            v = self.tad - tip; F = self.magnitude*v/np.linalg.norm(v)
            d = (L**3/(3*E))*((F @ self.arm_weak)/self.arm_I[0]*self.arm_weak
                              + (F @ self.arm_strong)/self.arm_I[1]*self.arm_strong)
            tip = tip0 + d
        return tip

    def tip_now(self, state):
        if self.kind == 'aligner_arm':
            disp = (state.aligner[self.hook_nodes] - self.hook_rest).mean(0)
            return self._arm_tip(state, self.hook_tip_rest + disp)
        T = state.T[self.hook_unn]
        return self.hook_tip_rest @ T[:3, :3].T + T[:3, 3]

    def reduced_vector(self, S, state, n_red):
        """Generalised force on the canine's rigid dofs for the metal hook (else zero)."""
        r = np.zeros(n_red)
        if not self.enabled or self.kind != 'metal':
            return r
        dirn, x = self.direction(state)
        F = self.magnitude*dirn
        c = S.tooth[self.hook_unn]['centroid_cur']
        q = S.q_col[self.hook_unn]
        r[q:q+3] += F
        r[q+3:q+6] += np.cross(x - c, F)
        return r

    def direction(self, state):
        if getattr(self, 'kind', 'aligner') in ('metal', 'aligner_arm'):
            c = self.tip_now(state)
            v = self.tad - c
            return v/np.linalg.norm(v), c
        c = state.aligner[self.hook_nodes].mean(0)
        v = self.tad - c
        return v/np.linalg.norm(v), c

    def vector(self, S, state, ndof):
        f = np.zeros(ndof)
        if not self.enabled or self.kind == 'metal':
            return f
        dirn, c = self.direction(state)
        if self.kind == 'aligner_arm':
            # force + the arm's moment about its root, spread over the root patch as a rigid
            # pattern: f_i = F/n + a x r_i with J a = M, J = sum(|r|^2 I - r r^T)
            X = state.aligner[self.hook_nodes]; xc = X.mean(0); r = X - xc
            Fv = self.magnitude*dirn
            M = np.cross(c - xc, Fv)
            J = (r*r).sum()*np.eye(3) - r.T @ r
            av = np.linalg.solve(J, M)
            fn = Fv/len(self.hook_nodes) + np.cross(av, r)
            for d in range(3):
                np.add.at(f, 3*self.hook_nodes + d, fn[:, d])
            return f
        F = self.magnitude*dirn/len(self.hook_nodes)
        for d in range(3):
            np.add.at(f, 3*self.hook_nodes + d, F[d])
        if len(self.bone_nodes):
            Fb = -self.magnitude*dirn/len(self.bone_nodes)
            for d in range(3):
                np.add.at(f, 3*self.bone_nodes + d, Fb[d])
        return f

    def describe(self, S, state):
        if not self.enabled:
            return f"group '{self.group}': no elastic (control)"
        dirn, c = self.direction(state)
        can = S.tooth[spec.GROUPS[self.group]['hook_tooth']]
        rel = c - can['crown_c']
        if self.kind == 'aligner_arm':
            b = self.arm_root_rest - can['crown_c']
            d0 = c - (self.hook_tip_rest + (state.aligner[self.hook_nodes] - self.hook_rest).mean(0))
            return (f"group '{self.group}'   [screw: {self.tad_rule}]\n"
                    f"  screw head     {np.round(self.tad, 2)}\n"
                    f"  aligner arm    root patch {len(self.hook_nodes)} aligner nodes at {b @ can['e_d']:+.2f} distal "
                    f"{b @ can['e_b']:+.2f} buccal {b @ can['e_o']:+.2f} occlusal of the canine crown centre; "
                    f"arm {self.hook_arm:.2f} mm apical (as the metal hook), section {spec.ALIGNER_THICKNESS:.2f} x "
                    f"{spec.AH_WIDTH:.2f} mm, E {spec.E_ALIGNER:.0f} MPa\n"
                    f"  hook tip       {np.round(c, 2)}   (bending {np.linalg.norm(d0)*1e3:.0f} um under the elastic)\n"
                    f"  force          {self.magnitude:.4f} N ({self.force_gf:.0f} gf) along {np.round(dirn, 3)}\n"
                    f"  free length    {np.linalg.norm(self.tad - c):.2f} mm")
        if self.kind == 'metal':
            b = self.hook_base_rest - can['crown_c']
            return (f"group '{self.group}'   [screw: {self.tad_rule}]\n"
                    f"  screw head     {np.round(self.tad, 2)}\n"
                    f"  metal hook     button {b @ can['e_d']:+.2f} distal {b @ can['e_b']:+.2f} buccal "
                    f"{b @ can['e_o']:+.2f} occlusal of the canine crown centre; arm {self.hook_arm:.2f} mm apical\n"
                    f"  hook tip       {np.round(c, 2)}   ({rel @ can['e_o']:+.2f} mm occlusal of the crown centre)\n"
                    f"  aligner cut    {len(self.cut_nodes)} inner nodes out of contact (r {spec.METAL_HOOK_CUTOUT} mm)\n"
                    f"  force          {self.magnitude:.4f} N ({self.force_gf:.0f} gf) along {np.round(dirn, 3)}\n"
                    f"  free length    {np.linalg.norm(self.tad - c):.2f} mm")
        return (f"group '{self.group}'   [TAD rule: {self.tad_rule}]\n"
                f"  TAD            {np.round(self.tad, 2)}  "
                f"({spec.GROUPS[self.group]['tad_above_crest']:.0f} mm apical to the "
                f"crest at {np.round(self.crest, 2)})\n"
                f"  precision cut  {np.round(c, 2)}  on {len(self.hook_nodes)} aligner nodes\n"
                f"                 vs canine crown centre: "
                f"{rel @ can['e_d']:+.2f} mm distal, {rel @ can['e_b']:+.2f} mm buccal, "
                f"{rel @ can['e_o']:+.2f} mm occlusal\n"
                f"  force          {self.magnitude:.4f} N ({self.force_gf:.0f} gf) "
                f"along {np.round(dirn, 3)}\n"
                f"  free length    {np.linalg.norm(self.tad - c):.2f} mm")
