"""Stage C: the reduced 4D model.

Degrees of freedom
    aligner nodes                     free (minus the midsagittal symmetry constraint)
    PDL middle layers                 free
    PDL inner layer + tooth surfaces  driven by their tooth's 6 rigid-body dofs
    PDL outer layer                   fixed  (= rigid alveolar bone, not meshed)
    7 teeth                           6 rigid dofs each

The mesh TOPOLOGY never changes during the 70-step simulation -- only node
positions -- so index maps are built once and reused for all 140 solves.
"""
import sys, os, pickle
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ofj'))
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
import spec

# Bone: 'rigid' follows Mao et al. (outer PDL layer fixed, bone not meshed);
# 'deformable' meshes the right half of the alveolar bone (build4d/bone.pkl) and
# ties the outer PDL layer into it.  OFJ_BONE_E_SCALE multiplies the bone modulus
# (a large value is the rigid limit, used to verify the tie).
BONE = os.environ.get('OFJ_BONE', 'rigid').lower()
BONE_E_SCALE = float(os.environ.get('OFJ_BONE_E_SCALE', 1.0))
# diagnostic: drop the attachments from the contact masters (aligner unchanged)
NO_ATTACH = os.environ.get('OFJ_NO_ATTACH', '0') == '1'

SYM_BAND = 1.2        # mm; aligner nodes within this of the midsagittal plane are
                      # constrained normal to it ('front end of the CA ... x-direction')


def rigid_block(x, c):
    """d u / d (t, omega) for u = t + omega x (x - c).  Returns (n,3,6)."""
    r = x - c
    B = np.zeros((len(x), 3, 6))
    B[:, 0, 0] = B[:, 1, 1] = B[:, 2, 2] = 1.0
    # omega x r
    B[:, 0, 4] = r[:, 2]; B[:, 0, 5] = -r[:, 1]
    B[:, 1, 3] = -r[:, 2]; B[:, 1, 5] = r[:, 0]
    B[:, 2, 3] = r[:, 1]; B[:, 2, 4] = -r[:, 0]
    return B


class Static:
    """Position-independent topology and index maps."""

    def __init__(self, geom=os.environ.get('OFJ_GEOM', 'build4d/geom.pkl'),
                 aligner=os.environ.get('OFJ_ALIGNER', 'build4d/aligner.pkl'),
                 bone='build4d/bone.pkl'):
        G = pickle.load(open(geom, 'rb'))
        A = pickle.load(open(aligner, 'rb'))
        src = G['nodes']; teeth = G['teeth']
        self.keep = G['keep']

        self.nA = len(A['nodes'])
        # nodes on the extruded layers (inner surface, middle, outer surface); a refined aligner
        # (tools/refine_prisms.py) appends interior prism-centroid nodes after them
        self.nA_layers = int(A.get('n_layer_nodes', self.nA))
        self.aligner_rest = A['nodes'].copy()
        self.aligner_tets = A['tets']
        self.aligner_inner_faces = A['inner_faces']
        self.n_inner = A['n_inner']
        self.vert_tooth = A['vert_tooth']

        self.tooth = {}
        nodes_rest = [A['nodes']]
        off = self.nA
        pdl_tets, pdl_owner = [], []
        for u in self.keep:
            t = teeth[u]
            pn, pt, lids = t['pdl_local']
            nP = len(pn)
            # --- surface nodes: crown + attachment outer shell (contact masters) ---
            sf = [t['crown_faces']]
            if t['att'] is not None and not NO_ATTACH:
                sf.append(t['att']['faces'])
            sf = np.vstack(sf)
            su = np.unique(sf)
            rm = np.full(src.shape[0], -1, np.int64); rm[su] = np.arange(len(su))
            s_nodes = src[su]
            s_tris = rm[sf]
            nS = len(s_nodes)

            self.tooth[u] = dict(
                unn=u, fdi=spec.FDI[u], name=t['name'],
                e_d=t['e_d'], e_b=t['e_b'], e_o=t['e_o'],
                crown_c=t['crown_c'], gingival_level=t['gingival_level'],
                occl_tip=t['occl_tip'],
                pdl_off=off, pdl_n=nP, pdl_layers=lids, pdl_rest=pn,
                surf_off=off+nP, surf_n=nS, surf_rest=s_nodes,
                surf_tris=s_tris + (off+nP),
                centroid=t['centroid'],
                has_att=t['att'] is not None,
                n_crown_tris=len(t['crown_faces']))
            nodes_rest.append(pn); nodes_rest.append(s_nodes)
            pdl_tets.append(pt + off); pdl_owner.append(np.full(len(pt), u))
            off += nP + nS

        # ---- deformable bone (optional) -----------------------------------------
        self.deformable_bone = BONE == 'deformable'
        if self.deformable_bone:
            Bd = pickle.load(open(bone, 'rb'))
            self.bone_off = off
            self.nB = len(Bd['nodes'])
            nodes_rest.append(Bd['nodes'])
            self.bone_tets = Bd['tets'] + off
            self.bone_E = Bd['E']*BONE_E_SCALE
            self.bone_nu = Bd['materials']['nu']
            self.bone_fixed = np.where(Bd['fixed'])[0] + off
            self.bone_sym = np.where(Bd['sym'])[0] + off
            off += self.nB
        self.nodes_rest = np.vstack(nodes_rest)
        self.n_global = off
        self.pdl_tets = np.vstack(pdl_tets)
        self.pdl_owner = np.concatenate(pdl_owner)
        self.master_tris = np.vstack([self.tooth[u]['surf_tris'] for u in self.keep])
        self.master_owner = np.concatenate(
            [np.full(len(self.tooth[u]['surf_tris']), u) for u in self.keep])

        # ---- midsagittal plane, from the full-arch central incisors -------------
        from inputs import load as load_inputs
        IN, _ = load_inputs()
        cR = IN['crown_c'][8]; cL = IN['crown_c'][9]
        n_sag = cR - cL; n_sag /= np.linalg.norm(n_sag)
        self.n_sag = n_sag
        self.sag_offset = float(0.5*(cR + cL) @ n_sag)

        # aligner nodes near the midsagittal plane -> constrained normal to it
        d = self.aligner_rest @ n_sag - self.sag_offset
        self.sym_nodes = np.where(np.abs(d) <= SYM_BAND)[0]
        if self.deformable_bone:
            self.sym_nodes = np.concatenate([self.sym_nodes, self.bone_sym])

        # ---- dof bookkeeping ---------------------------------------------------
        kind = np.zeros(self.n_global, np.int8)      # 0 free, 1 rigid, 2 fixed
        owner = np.full(self.n_global, -1, np.int64)
        for u in self.keep:
            T = self.tooth[u]
            L = T['pdl_layers']
            kind[T['pdl_off'] + L[0]] = 1; owner[T['pdl_off'] + L[0]] = u
            kind[T['pdl_off'] + L[-1]] = 2
            s = slice(T['surf_off'], T['surf_off']+T['surf_n'])
            kind[s] = 1; owner[s] = u
        if self.deformable_bone:
            for u in self.keep:
                T = self.tooth[u]
                kind[T['pdl_off'] + T['pdl_layers'][-1]] = 3     # tied into the bone
            kind[self.bone_fixed] = 2
            self.tied_nodes = np.where(kind == 3)[0]
            X = self.nodes_rest[self.bone_tets]
            M = np.stack([X[:, 1]-X[:, 0], X[:, 2]-X[:, 0], X[:, 3]-X[:, 0]], -1)
            det = np.linalg.det(M); good = np.abs(det) > 1e-12
            self._bMinv = np.full_like(M, np.nan)
            self._bMinv[good] = np.linalg.inv(M[good])
            self._bT0 = X[:, 0]
            self._btree = cKDTree(X.mean(1))
            del X, M
        self.kind = kind; self.owner = owner
        self.free_nodes = np.where(kind == 0)[0]
        self.n_free = len(self.free_nodes)
        self.n_red = 3*self.n_free + 6*len(self.keep) - len(self.sym_nodes)
        print(f"global nodes {self.n_global}  free {self.n_free}  "
              f"rigid {(kind==1).sum()}  fixed {(kind==2).sum()}"
              + (f"  tied {(kind==3).sum()}  [deformable bone: {self.nB} nodes, "
                 f"{len(self.bone_tets)} tets, E x{BONE_E_SCALE:g}]"
                 if self.deformable_bone else "  [rigid bone]"))
        print(f"elements: aligner {len(self.aligner_tets)}  PDL {len(self.pdl_tets)}")
        print(f"reduced dofs {self.n_red}  (symmetry constrains {len(self.sym_nodes)} aligner nodes)")
        self._dof_layout()

    # -------------------------------------------------------------- projection --
    def _dof_layout(self):
        """Column layout for the free-node part of P (built once)."""
        sym = np.zeros(self.n_global, bool); sym[self.sym_nodes] = True
        self._sym_mask = sym
        fs = sym[self.free_nodes]
        ncol = np.where(fs, 2, 3)
        start = np.concatenate([[0], np.cumsum(ncol)[:-1]])
        self.free_sym = fs
        self.free_col0 = start
        self.free_pos = np.full(self.n_global, -1, np.int64)
        self.free_pos[self.free_nodes] = np.arange(len(self.free_nodes))
        self.n_free_col = int(ncol.sum())
        # in-plane basis for the symmetry-constrained nodes
        e1 = np.array([1.0, 0, 0])
        if abs(e1 @ self.n_sag) > 0.9:
            e1 = np.array([0, 1.0, 0])
        a1 = e1 - (e1 @ self.n_sag)*self.n_sag; a1 /= np.linalg.norm(a1)
        self.sym_basis = np.stack([a1, np.cross(self.n_sag, a1)])      # (2,3)
        self.rigid_idx = {u: np.where(self.owner == u)[0] for u in self.keep}
        self.q_col = {u: self.n_free_col + 6*i for i, u in enumerate(self.keep)}
        assert self.n_free_col + 6*len(self.keep) == self.n_red

    def build_P(self, nodes):
        """Sparse map  x_reduced -> u_global (3*n_global).  Fully vectorised."""
        rows, cols, vals = [], [], []
        fn = self.free_nodes
        # --- ordinary free nodes: identity ---------------------------------------
        o = fn[~self.free_sym]; c0 = self.free_col0[~self.free_sym]
        d = np.arange(3)
        rows.append((3*o[:, None] + d).ravel())
        cols.append((c0[:, None] + d).ravel())
        vals.append(np.ones(3*len(o)))
        # --- symmetry nodes: two in-plane directions only -------------------------
        sN = fn[self.free_sym]; sc = self.free_col0[self.free_sym]
        if len(sN):
            for k in range(2):
                rows.append((3*sN[:, None] + d).ravel())
                cols.append(np.repeat(sc + k, 3))
                vals.append(np.tile(self.sym_basis[k], len(sN)))
        # --- rigid teeth -----------------------------------------------------------
        for u in self.keep:
            idx = self.rigid_idx[u]
            B = rigid_block(nodes[idx], self.tooth[u]['centroid_cur'])
            c = self.q_col[u]
            rows.append(np.repeat((3*idx[:, None] + d).ravel(), 6))
            cols.append(np.tile(c + np.arange(6), 3*len(idx)))
            vals.append(B.reshape(-1, 6).ravel())
        # --- tied outer PDL layer: interpolated from the bone tet containing it --
        if self.deformable_bone:
            t = self.tied_nodes
            tet, W, inside = self.tie_weights(nodes[t])
            self.tie_inside = float(inside.mean())
            fp = self.free_pos
            for j in range(4):
                b = tet[:, j]; w = W[:, j]
                ok = (fp[b] >= 0) & (w > 0)
                ns = ok & ~self._sym_mask[b]
                if ns.any():
                    c0 = self.free_col0[fp[b[ns]]]
                    rows.append((3*t[ns][:, None] + d).ravel())
                    cols.append((c0[:, None] + d).ravel())
                    vals.append(np.repeat(w[ns], 3))
                sy = ok & self._sym_mask[b]
                if sy.any():
                    c0 = self.free_col0[fp[b[sy]]]
                    for k in range(2):
                        rows.append((3*t[sy][:, None] + d).ravel())
                        cols.append(np.repeat(c0 + k, 3))
                        vals.append((w[sy][:, None]*self.sym_basis[k][None, :]).ravel())
        rows = np.concatenate(rows); cols = np.concatenate(cols); vals = np.concatenate(vals)
        return csr_matrix((vals, (rows, cols)), shape=(3*self.n_global, self.n_red))


    def prescribed_nodes(self, state, arch, step, aligner_shape):
        """Node array for the configuration the aligner was MANUFACTURED for:
        every tooth at its prescribed position for this step, aligner morphed."""
        n = self.nodes_rest.copy()
        n[:self.nA] = aligner_shape
        for u in self.keep:
            T = arch.prescribed_transform(u, step)
            sl = slice(self.tooth[u]['pdl_off'],
                       self.tooth[u]['surf_off']+self.tooth[u]['surf_n'])
            n[sl] = self.nodes_rest[sl] @ T[:3, :3].T + T[:3, 3]
        return n

    def tie_weights(self, pts, k=24):
        """Containing bone tet and barycentric weights for each point, in the REST
        bone configuration.  The bone is stress-free at the start of every solve --
        the remodelling assumption -- so ties are re-formed as the teeth move.  A
        point in no candidate tet (e.g. PDL that has moved into the socket void)
        takes the least-violated candidate with its weights clamped to >= 0, i.e.
        it is pinned to the nearest bone."""
        _, idx = self._btree.query(pts, k=k)
        r = pts[:, None, :] - self._bT0[idx]
        lam = np.einsum('nkij,nkj->nki', self._bMinv[idx], r)
        w = np.concatenate([1.0 - lam.sum(-1, keepdims=True), lam], -1)
        mn = np.nan_to_num(w.min(-1), nan=-np.inf)
        best = mn.argmax(1)
        ar = np.arange(len(pts))
        W = np.clip(np.nan_to_num(w[ar, best], nan=0.25), 0.0, None)
        W /= W.sum(1, keepdims=True)
        return self.bone_tets[idx[ar, best]], W, mn[ar, best] >= -1e-9

    @staticmethod
    def gap_for_pairing(nodes, slave, tri, bary):
        """Signed gap and outward normal for a FIXED (slave, triangle, bary) pairing.

        Small sliding in the Abaqus sense: the pairing is established once for the
        step and only the positions move.  Re-deriving the pairing every solve makes
        the reference drift as the appliance slides, which manufactures force that
        grows with distance travelled rather than with interference.
        """
        v = nodes[tri]
        n = np.cross(v[:, 1]-v[:, 0], v[:, 2]-v[:, 0])
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        q = np.einsum('nj,njk->nk', bary, v)
        return np.einsum('ij,ij->i', nodes[slave] - q, n), n

    def rigid_modes(self, nodes):
        """Exact reduced vectors whose image under P is a global rigid motion.

        This is the near-nullspace smoothed-aggregation AMG needs.  P.T @ B is NOT
        it (transpose is not pseudo-inverse) and degrades the preconditioner by
        four orders of magnitude.
        """
        c0 = nodes.mean(0)
        B = np.zeros((self.n_red, 6))
        fn = self.free_nodes
        r = nodes[fn] - c0
        ns = ~self.free_sym
        o = self.free_col0[ns]
        sN = self.free_col0[self.free_sym]

        def fill(v, col):
            B[(o[:, None] + np.arange(3)).ravel(), col] = v[ns].ravel()
            if len(sN):
                for k in range(2):
                    B[sN + k, col] = v[self.free_sym] @ self.sym_basis[k]

        for k in range(3):
            v = np.zeros((len(fn), 3)); v[:, k] = 1.0
            fill(v, k)
        for k, ax in enumerate(np.eye(3)):
            fill(np.cross(ax, r), 3+k)
        for u in self.keep:
            cu = self.tooth[u]['centroid_cur']; q = self.q_col[u]
            for k in range(3):
                B[q+k, k] = 1.0
            for k, ax in enumerate(np.eye(3)):
                B[q:q+3, 3+k] = np.cross(ax, cu - c0)
                B[q+3:q+6, 3+k] = ax
        return B


class State:
    """Current configuration: one rigid transform per tooth + the aligner nodes."""

    def __init__(self, S):
        self.S = S
        self.T = {u: np.eye(4) for u in S.keep}          # accumulated tooth transform
        self.aligner = S.aligner_rest.copy()
        self.step = 0

    def tooth_nodes(self, u):
        T = self.T[u]; st = self.S.tooth[u]
        idx = np.where(self.S.owner == u)[0]
        return idx, self.S.nodes_rest[idx] @ T[:3, :3].T + T[:3, 3]

    def nodes(self):
        n = self.S.nodes_rest.copy()
        n[:self.S.nA] = self.aligner
        for u in self.S.keep:
            T = self.T[u]
            st = self.S.tooth[u]
            sl = slice(st['pdl_off'], st['surf_off']+st['surf_n'])
            n[sl] = self.S.nodes_rest[sl] @ T[:3, :3].T + T[:3, 3]
        return n

    def centroid(self, u):
        T = self.T[u]
        return self.S.tooth[u]['centroid'] @ T[:3, :3].T + T[:3, 3]

    def apply_rigid(self, u, t, w):
        """Advance tooth u by an infinitesimal rigid motion (t, omega)."""
        th = np.linalg.norm(w)
        if th < 1e-14:
            R = np.eye(3)
        else:
            k = w/th; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            R = np.eye(3) + np.sin(th)*K + (1-np.cos(th))*(K @ K)
        c = self.centroid(u)
        M = np.eye(4); M[:3, :3] = R; M[:3, 3] = c + t - R @ c
        self.T[u] = M @ self.T[u]


if __name__ == '__main__':
    S = Static()
    st = State(S)
    for u in S.keep:
        S.tooth[u]['centroid_cur'] = st.centroid(u)
    P = S.build_P(st.nodes())
    print(f"P: {P.shape}, nnz {P.nnz/1e6:.2f}M")
    pickle.dump(dict(n_red=S.n_red), open('build4d/static_info.pkl', 'wb'))
