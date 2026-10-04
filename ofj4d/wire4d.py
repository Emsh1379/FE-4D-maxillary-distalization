"""Archwire: a passive rectangular stainless-steel wire, 3D beam elements, bonded in the slots.

Geometry
    The wire is made PASSIVE, i.e. stress-free in the initial configuration, as a
    pre-formed wire engaged in well-aligned brackets: straight along each slot (in-slot
    length = the slot's mesiodistal length) and a cubic Hermite curve between adjacent
    slots whose end tangents are the two slot axes.  From the mesial end of the 11 bracket
    it continues to the midsagittal plane, which it crosses at right angles.

Coupling
    'bonded' (default, as the previous 3D study: 'the full-size archwire and the bracket
    slots were assumed to be perfectly bonded'): every wire node inside a slot follows its
    tooth rigidly (translation and rotation).  'sliding': the node follows the tooth in
    everything except translation along the slot axis (frictionless sliding mechanics);
    the molar tube (17) stays bonded as a stop.
    The wire is slaved exactly (master-slave elimination), not tied by penalty springs.

Symmetry
    The node on the midsagittal plane: no displacement normal to the plane, no rotation
    about in-plane axes (a mirror-symmetric deformation of a beam crossing the plane).

Mechanics
    3D Euler-Bernoulli beams in a CO-ROTATIONAL formulation: each element's deformation
    (axial stretch and the two end rotations) is measured in a frame that follows the
    element, so a rigid translation or rotation of the arch -- which the headgear produces
    over the protocol, several degrees of molar tipping -- stores no energy, however large.
    The wire's force is recomputed each solve from its TOTAL deformation since the passive
    start: it is elastic and does not remodel (the PDL does).  The tangent is the material
    stiffness in the current element frames.

Units: N, mm, MPa.
"""
import os
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

IN = 25.4
WIRE = os.environ.get('OFJ_WIRE', '0.019x0.025')           # inches, occluso-gingival x labio-lingual
WIRE_E = float(os.environ.get('OFJ_WIRE_E', 210000.0))     # MPa, stainless steel (previous 3D study)
WIRE_NU = float(os.environ.get('OFJ_WIRE_NU', 0.30))
COUPLING = os.environ.get('OFJ_WIRE_COUPLING', 'slot')      # slot | bonded | sliding
# 'slot' (default): wire-slot contact after Kojima/Kawamura (Kawamura & Tamaya, Prog Orthod
# 2019;20:3; Kawamura et al., AJOP 2021;160:259; Fares et al., BMC Oral Health 2025): the
# wire sits centred in each slot with a uniform clearance gap, the bracket turns freely within
# the play, and the wire slides along the slot with Coulomb friction MU.
MU = float(os.environ.get('OFJ_WIRE_MU', 0.15))              # wire-slot friction coefficient
K_PEN = float(os.environ.get('OFJ_SLOT_K', 1.0e4))           # N/mm, contact penalty (translation)
K_PEN_R = float(os.environ.get('OFJ_SLOT_KR', 1.0e4))        # N mm/rad, contact penalty (torsion)
GAP_N = float(os.environ.get('OFJ_SLOT_GAP_N', 0.0))         # mm, labio-lingual play (0: ligated to the floor)
K_FREE = 0.1          # N/mm (N mm/rad): regulariser inside the play, keeps the free wire non-singular
EPS = 0.002           # mm, smoothing of the slot walls
EPS_R = 0.002         # rad, smoothing of the torque play limit
EPS_A = 0.001         # mm, regularisation of Coulomb friction (behaves as stick below ~1 um of slip)
H_ELEM = float(os.environ.get('OFJ_WIRE_H', 0.30))          # mm, target element length
TAIL = float(os.environ.get('OFJ_WIRE_TAIL', 2.0))           # mm of wire distal to the last tube (free end)


def unit(v):
    return v/np.linalg.norm(v)


def section(spec=WIRE):
    b, h = [float(x)*IN for x in spec.lower().split('x')]  # b along gingival, h along labial
    A = b*h
    I_n = h*b**3/12.0          # bending about the labial axis (deflection occluso-gingival)
    I_g = b*h**3/12.0          # bending about the gingival axis (deflection labio-lingual)
    a, c = max(b, h)/2, min(b, h)/2
    J = a*c**3*(16/3 - 3.36*(c/a)*(1 - c**4/(12*a**4)))
    return dict(b=b, h=h, A=A, I_n=I_n, I_g=I_g, J=J)


def hermite(p0, t0, p1, t1, n):
    s = np.linspace(0, 1, n)[:, None]
    h00 = 2*s**3 - 3*s**2 + 1; h10 = s**3 - 2*s**2 + s
    h01 = -2*s**3 + 3*s**2;    h11 = s**3 - s**2
    return h00*p0 + h10*t0 + h01*p1 + h11*t1


def build_path(fx, order, n_sag, sag_offset, wire_h, h_elem=H_ELEM):
    """Wire nodes, element connectivity, per-node labial/gingival axes and slot owner.

    fx     bracket records (brackets.build_fixed), keyed by UNN
    order  UNNs from distal (17) to mesial (11)
    """
    X, owner, nlab = [], [], []

    def add(pts, own, nl):
        for p, o, n in zip(pts, own, nl):
            if X and np.linalg.norm(p - X[-1]) < 1e-6:
                continue
            X.append(p); owner.append(o); nlab.append(n)

    prev_end = None
    for i, u in enumerate(order):
        r = fx[u]
        a = r['slot_axis']                               # mesial
        # wire centre: on the slot floor + half the wire depth, centred across the slot
        c = r['slot_floor'] + r['slot_n']*(wire_h/2)
        L = r['slot_length']
        s0, s1 = c - a*L/2, c + a*L/2                     # distal and mesial slot ends
        if prev_end is None and TAIL > 0:                 # free distal end past the last tube
            k = max(2, int(np.ceil(TAIL/h_elem)) + 1)
            P = s0 - np.outer(np.linspace(1, 0, k), a*TAIL)
            add(P[:-1], [-1]*(k-1), [r['slot_n']]*(k-1))
        if prev_end is not None:
            pe, ae, ne = prev_end
            chord = np.linalg.norm(s0 - pe)
            k = max(3, int(np.ceil(chord/h_elem)) + 1)
            P = hermite(pe, ae*chord, s0, a*chord, k)
            nl = [unit((1-w)*ne + w*r['slot_n']) for w in np.linspace(0, 1, k)]
            add(P[1:-1], [-1]*(k-2), nl[1:-1])
        k = max(3, int(np.ceil(L/h_elem)) + 1)
        P = s0 + np.outer(np.linspace(0, 1, k), s1 - s0)
        add(P, [u]*k, [r['slot_n']]*k)
        prev_end = (s1, a, r['slot_n'])
    # ---- mesial end: from the 11 slot to the midsagittal plane ----------------------
    pe, ae, ne = prev_end
    dist = pe @ n_sag - sag_offset                        # signed distance to the plane
    toward = -np.sign(dist)*n_sag                         # direction that reaches the plane
    ca = ae @ toward
    reach = abs(dist)/max(ca, 0.2)
    M = pe + ae*reach
    M = M - (M @ n_sag - sag_offset)*n_sag                # exactly on the plane
    chord = np.linalg.norm(M - pe)
    k = max(3, int(np.ceil(chord/h_elem)) + 1)
    P = hermite(pe, ae*chord, M, toward*chord, k)
    add(P[1:], [-1]*(k-1), [ne]*(k-1))
    X = np.array(X); owner = np.array(owner); nlab = np.array(nlab)
    elems = np.stack([np.arange(len(X)-1), np.arange(1, len(X))], 1)
    sym_node = len(X) - 1
    return dict(X=X, owner=owner, nlab=nlab, elems=elems, sym_node=sym_node)


def element_frame(x1, x2, nlab):
    L = np.linalg.norm(x2 - x1)
    ex = (x2 - x1)/L
    n = nlab - (nlab @ ex)*ex; n = unit(n)                 # local labial axis
    g = np.cross(n, ex)                                    # local gingival-ish axis
    return L, np.stack([ex, g, n])                          # rows = local axes


def element_K(x1, x2, nlab, sec, E=WIRE_E, nu=WIRE_NU, local=False):
    """12x12 stiffness of a 3D Euler-Bernoulli beam, dofs [u1 th1 u2 th2] (global, or local)."""
    L, lam = element_frame(x1, x2, nlab)
    ex, g, n = lam
    # local axes: 1 = ex, 2 = g (width b), 3 = n (depth h)
    G = E/(2*(1 + nu))
    A, J = sec['A'], sec['J']
    I2 = sec['I_g']       # about local 2 (g): bending in the 1-3 plane (deflection along n)
    I3 = sec['I_n']       # about local 3 (n): bending in the 1-2 plane (deflection along g)
    k = np.zeros((12, 12))
    EA = E*A/L; GJ = G*J/L
    k[0, 0] = k[6, 6] = EA; k[0, 6] = k[6, 0] = -EA
    k[3, 3] = k[9, 9] = GJ; k[3, 9] = k[9, 3] = -GJ
    # bending in 1-2 plane: v (dof 1,7), theta_3 (dof 5,11), inertia I3
    a = E*I3/L**3
    idx = [1, 5, 7, 11]
    kb = a*np.array([[12, 6*L, -12, 6*L], [6*L, 4*L*L, -6*L, 2*L*L],
                     [-12, -6*L, 12, -6*L], [6*L, 2*L*L, -6*L, 4*L*L]])
    k[np.ix_(idx, idx)] += kb
    # bending in 1-3 plane: w (dof 2,8), theta_2 (dof 4,10), inertia I2  (sign flip)
    a = E*I2/L**3
    idx = [2, 4, 8, 10]
    kb = a*np.array([[12, -6*L, -12, -6*L], [-6*L, 4*L*L, 6*L, 2*L*L],
                     [-12, 6*L, 12, 6*L], [-6*L, 2*L*L, 6*L, 4*L*L]])
    k[np.ix_(idx, idx)] += kb
    if local:
        return k, lam, L
    T = np.zeros((12, 12))
    for i in range(4):
        T[3*i:3*i+3, 3*i:3*i+3] = lam
    return T.T @ k @ T


def assemble(W, sec, E=WIRE_E, nu=WIRE_NU):
    X, el, nl = W['X'], W['elems'], W['nlab']
    nd = 6*len(X)
    rows, cols, vals = [], [], []
    for a, b in el:
        Ke = element_K(X[a], X[b], 0.5*(nl[a] + nl[b]), sec, E, nu)
        d = np.r_[6*a:6*a+6, 6*b:6*b+6]
        rows.append(np.repeat(d, 12)); cols.append(np.tile(d, 12)); vals.append(Ke.ravel())
    K = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(nd, nd)).tocsr()
    return K


def rotvec(R):
    c = np.clip((np.trace(R) - 1)/2, -1, 1)
    th = np.arccos(c)
    v = np.array([R[2, 1]-R[1, 2], R[0, 2]-R[2, 0], R[1, 0]-R[0, 1]])
    if th < 1e-9:
        return 0.5*v
    return th/(2*np.sin(th))*v


def expmap(w):
    th = np.linalg.norm(w)
    if th < 1e-14:
        return np.eye(3)
    k = w/th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th)*K + (1 - np.cos(th))*K @ K


class CorotWire:
    """Co-rotational beam wire.  State: node positions x (n,3) and node rotations R (n,3,3)
    relative to the passive start."""

    def __init__(self, W, sec, E=WIRE_E, nu=WIRE_NU):
        self.W = W
        X, el, nl = W['X'], W['elems'], W['nlab']
        self.X0 = X.copy()
        self.el = el
        self.k, self.lam0, self.L0 = [], [], []
        for a, b in el:
            k, lam, L = element_K(X[a], X[b], 0.5*(nl[a] + nl[b]), sec, E, nu, local=True)
            self.k.append(k); self.lam0.append(lam); self.L0.append(L)
        self.k = np.array(self.k); self.lam0 = np.array(self.lam0); self.L0 = np.array(self.L0)
        self.x = X.copy()
        self.R = np.tile(np.eye(3), (len(X), 1, 1))

    def forces(self):
        """Internal force vector (6n), tangent stiffness (6n x 6n) and strain energy."""
        n = len(self.x); el = self.el
        f = np.zeros(6*n)
        rows, cols, vals = [], [], []
        energy = 0.0
        for e, (a, b) in enumerate(el):
            lam0 = self.lam0[e]
            Ea = self.R[a] @ lam0.T                  # columns: end frames, current
            Eb = self.R[b] @ lam0.T
            d = self.x[b] - self.x[a]
            l = np.linalg.norm(d)
            e1 = d/l
            t = Ea[:, 1] + Eb[:, 1]
            e2 = t - (t @ e1)*e1; e2 /= np.linalg.norm(e2)
            e3 = np.cross(e1, e2)
            C = np.stack([e1, e2, e3], 1)            # columns: co-rotated element frame
            ta = rotvec(C.T @ Ea); tb = rotvec(C.T @ Eb)
            dl = np.zeros(12)
            dl[3:6] = ta; dl[6] = l - self.L0[e]; dl[9:12] = tb
            fl = self.k[e] @ dl
            energy += 0.5*dl @ fl
            T = np.zeros((12, 12))
            for i in range(4):
                T[3*i:3*i+3, 3*i:3*i+3] = C.T
            dofs = np.r_[6*a:6*a+6, 6*b:6*b+6]
            f[dofs] += T.T @ fl
            Kg = T.T @ self.k[e] @ T
            rows.append(np.repeat(dofs, 12)); cols.append(np.tile(dofs, 12)); vals.append(Kg.ravel())
        K = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                       shape=(6*n, 6*n)).tocsr()
        return f, K, energy

    def advance(self, dU, mask):
        """Apply an increment (6n) to the nodes in `mask`."""
        d = dU.reshape(-1, 6)
        for i in np.where(mask)[0]:
            self.x[i] += d[i, :3]
            self.R[i] = expmap(d[i, 3:]) @ self.R[i]


class WireMap:
    """Master-slave map from the extended reduced vector [x_model ; w] to the wire dofs.

    Column blocks: the model's reduced dofs (n_red; only the tooth rigid dofs are used),
    then the free wire dofs: 6 per free node, 3 for the symmetry node, and 1 axial
    sliding dof per in-slot node when coupling is 'sliding'.
    """

    def __init__(self, W, S, fx, coupling=COUPLING, stop_unn=None):
        self.W = W; self.S = S
        X, own = W['X'], W['owner']
        n = len(X)
        self.coupling = coupling
        col = S.n_red
        self.kind = np.zeros(n, np.int8)       # 0 free, 1 slave, 2 sym, 3 slide
        self.col = np.full(n, -1, np.int64)
        # the distal-most attachment (second-molar tube) stays bonded as the wire stop
        stop_unn = stop_unn if stop_unn is not None else int(own[own >= 0][0])
        self.axis = np.zeros((n, 3))
        for i in range(n):
            if i == W['sym_node']:
                self.kind[i] = 2; self.col[i] = col; col += 3
            elif own[i] < 0 or coupling == 'slot':      # 'slot': coupled by SlotContact only
                self.kind[i] = 0; self.col[i] = col; col += 6
            elif coupling == 'sliding' and own[i] != stop_unn:
                self.kind[i] = 3; self.col[i] = col; col += 1
                self.axis[i] = fx[own[i]]['slot_axis']
            else:
                self.kind[i] = 1
        self.n_ext = col
        self.n_w = col - S.n_red
        a1, a2 = S.sym_basis
        self.sym_frame = (a1, a2, S.n_sag)

    def build(self, Xcur, centroid):
        """Sparse G (6n x n_ext) at the current wire node positions and tooth centroids."""
        from model4d import rigid_block
        S, W = self.S, self.W
        own = W['owner']
        rows, cols, vals = [], [], []
        d3 = np.arange(3)
        for i in range(len(Xcur)):
            k = self.kind[i]; r0 = 6*i
            if k == 0:
                rows += list(r0 + np.arange(6)); cols += list(self.col[i] + np.arange(6)); vals += [1.0]*6
            elif k == 2:
                a1, a2, nn = self.sym_frame
                for j, v in enumerate((a1, a2)):
                    rows += list(r0 + d3); cols += [self.col[i] + j]*3; vals += list(v)
                rows += list(r0 + 3 + d3); cols += [self.col[i] + 2]*3; vals += list(nn)
            else:
                u = own[i]; q = S.q_col[u]
                B = rigid_block(Xcur[i:i+1], centroid[u])[0]        # (3,6)
                for a in range(3):
                    for b in range(6):
                        if B[a, b] != 0.0:
                            rows.append(r0 + a); cols.append(q + b); vals.append(B[a, b])
                for a in range(3):
                    rows.append(r0 + 3 + a); cols.append(q + 3 + a); vals.append(1.0)
                if k == 3:                                          # free axial slide
                    rows += list(r0 + d3); cols += [self.col[i]]*3; vals += list(self.axis[i])
        return csr_matrix((vals, (rows, cols)), shape=(6*len(Xcur), self.n_ext))


def torque_play(b, h, width):
    """Free torsional rotation (rad) of a b x h rectangular wire (b across the slot width)
    in a slot of the given width, until diagonal corners touch both walls."""
    D = np.hypot(b, h)
    if width >= D:
        return np.pi/2
    return max(0.0, np.arctan2(h, b) - np.arccos(width/D))


class SlotContact:
    """Wire-slot contact, one contact point at each end of every slot/tube (Kojima's
    spring-element coupling).  Per point, in the slot frame (a axial, w across the slot
    width, n labio-lingual) of the tooth's current position:
        w       dead zone +-GW (half the clearance), penalty K_PEN beyond
        n       dead zone +-GAP_N (0 = ligated), penalty K_PEN beyond
        twist   dead zone +-play (torque_play), penalty K_PEN_R beyond
        a       Coulomb friction MU x normal force: stick (penalty) or slip (force)
    The wire point is the wire node nearest to the slot end, re-found every iteration so
    the wire can slide through the slot.  States are decided by active-set iteration inside
    FixedSystem.solve."""

    def __init__(self, W, S, fx, sec, mu=MU, k=K_PEN, kr=K_PEN_R, gap_n=GAP_N):
        self.mu, self.k, self.kr = mu, k, kr
        self.pts = []
        for u, r in fx.items():
            a = unit(r['slot_axis']); n = unit(r['slot_n'] - (r['slot_n'] @ a)*a)
            w = np.cross(n, a)
            c = r['slot_floor'] + r['slot_n']*(sec['h']/2)
            L = r['slot_length']
            gw = max(0.0, (r['slot_width'] - sec['b'])/2)
            play = torque_play(sec['b'], sec['h'], r['slot_width'])
            for e in (-0.5, 0.5):
                self.pts.append(dict(u=u, P=c + a*L*e, F=np.stack([a, w, n]), gw=gw, gn=gap_n,
                                     play=play, fdi=r['fdi']))
        self.lever = sec['h']                              # torsion couple -> wall forces
        self.kind0 = np.arange(len(W['X'])) != W['sym_node']
        self.N_lag = np.zeros(len(self.pts))                # normal force per point, previous iteration

    def select(self, T, x, R):
        """Geometry at the start of an iteration: nearest wire node and current offsets."""
        from scipy.spatial import cKDTree
        ok = np.where(self.kind0)[0]                       # 6-dof wire nodes only (not the midline node)
        tree = cKDTree(x[ok])
        self.cur = []
        for p in self.pts:
            Tu = T[p['u']]
            E = Tu[:3, :3] @ p['P'] + Tu[:3, 3]
            F = p['F'] @ Tu[:3, :3].T                      # rows: a, w, n (current)
            i = int(ok[tree.query(E)[1]])
            d = F @ (x[i] - E)
            tw = rotvec(Tu[:3, :3].T @ R[i]) @ p['F'][0]
            self.cur.append(dict(i=i, E=E, F=F, d=d, tw=tw))
        return self.cur

    def rows(self, wmap, S, cen):
        """Sparse rows (4 per point: a, w, n, twist) of the relative increment wire - tooth."""
        from model4d import rigid_block
        R_, C_, V_ = [], [], []
        for j, (p, c) in enumerate(zip(self.pts, self.cur)):
            i, u = c['i'], p['u']
            q = S.q_col[u]; ci = wmap.col[i]
            B = rigid_block(c['E'][None], cen[u])[0]       # (3,6), tooth point at the slot end
            for m in range(3):
                e = c['F'][m]
                r = 4*j + m
                R_ += [r]*3; C_ += list(ci + np.arange(3)); V_ += list(e)
                R_ += [r]*6; C_ += list(q + np.arange(6)); V_ += list(-(e @ B))
            e = c['F'][0]; r = 4*j + 3
            R_ += [r]*3; C_ += list(ci + 3 + np.arange(3)); V_ += list(e)
            R_ += [r]*3; C_ += list(q + 3 + np.arange(3)); V_ += list(-e)
        return csr_matrix((V_, (R_, C_)), shape=(4*len(self.pts), wmap.n_ext))

    def law(self, dlt):
        """Linearised contact at the predicted increments dlt (4 per point: a, w, n, twist).
        Returns kd (tangent) and f0 so that the contact force on each relative dof is
        kd*dlt + f0 (= F(d0 + dlt_pred) at the prediction), a state summary for the
        convergence check, and the normal force per point.
        Smooth laws (Newton converges; sharp dead zones and stick/slip switching chattered):
            walls   F = K sign(d) EPS softplus((|d| - gap)/EPS)      (gap 0: F = K d)
            axial   F = MU N_lag tanh(da/EPS_A)                       (regularised Coulomb)"""
        n = len(self.pts)
        kd = np.zeros(4*n); F = np.zeros(4*n)
        Nn = np.zeros(n); st = []
        for j, (p, c) in enumerate(zip(self.pts, self.cur)):
            sj = []
            for m, g, kk, d0, eps in ((1, p['gw'], self.k, c['d'][1], EPS),
                                      (2, p['gn'], self.k, c['d'][2], EPS),
                                      (3, p['play'], self.kr, c['tw'], EPS_R)):
                d = d0 + dlt[4*j + m]
                if g == 0.0:
                    f, t = kk*d, kk
                else:
                    z = (abs(d) - g)/eps
                    sp = eps*(np.logaddexp(0.0, z))
                    f, t = kk*np.sign(d)*sp, kk/(1.0 + np.exp(-z))
                F[4*j + m] = f; kd[4*j + m] = t
                Nn[j] += abs(f)*(2.0/self.lever if m == 3 else 1.0)
                sj.append(int(np.sign(d)) if (g == 0.0 or abs(d) > g) else 0)
            da = dlt[4*j]
            cap = self.mu*self.N_lag[j]
            if cap > 0:
                th = np.tanh(da/EPS_A)
                F[4*j] = cap*th; kd[4*j] = cap/EPS_A*(1.0 - th*th)
            sj.append(0 if abs(da) < EPS_A else int(np.sign(da)))
            st.append(tuple(sj))
        kd += K_FREE
        f0 = F - (kd - K_FREE)*dlt                        # K_FREE only regularises the increment
        return kd, f0, st, Nn
