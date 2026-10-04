"""Fixed appliance geometry: brackets from the Parasolid library, molar tubes, placement.

Source of the brackets
    'Orthodontics Ceramic Bracket System final.x_t' (SolidWorks 2020, Parasolid 31.1 text
    transmit file).  It is an assembly of 20 twin brackets, one body per tooth, named by FDI
    number and prescription, e.g. '13 - Tor0 - Ang8-1'.  The upper-right quadrant (11-15)
    is used here; there are no molar tubes in the file, so 16 and 17 get parametric tubes.

What is read from the file, and what is not
    The file is parsed directly (no CAD kernel, no Parasolid schema catalog is needed):
    every vertex of every body, every straight edge, the instance names and the
    prescription.  From those the pipeline takes
      * the SLOT: position, mesiodistal axis, width and the floor normal (torque),
        detected from the straight slot edges -- this is what the archwire is bonded to;
      * the BASE: the bonding surface (part z = 0) and its centre under the slot;
      * the ANGULATION reference: the tie-wing side planes;
      * a closed surface for rendering and export: the convex hull of the body's vertices
        with the slot channel cut out (hooks are left off).
    Curved faces are NOT tessellated exactly.  That is sufficient here because a ceramic
    bracket (E ~ 380 GPa) is rigid with its tooth in this model -- teeth are rigid bodies --
    so only the slot frame enters the mechanics.  If exact bracket meshes are wanted for
    figures, export each body from SolidWorks as STL in its part frame and put it at
    brackets/stl/<FDI>.stl: it replaces the hull for rendering.

Part frame of the library brackets (verified on the file, see `report()`):
    +z  labial (base at z = 0, the bracket stands on +z)
    +y  gingival (hooks are gingival), slot centre at y ~ 1.7 mm
    x   mesiodistal; the side planes are rotated by the angulation about z
The slot of the canine (torque 0) is exactly y 1.398..1.952 mm (0.554 mm = 0.022" slot)
with its floor at z = 1.25 mm; the incisor and premolar slots are the same channel rotated
about x by the bracket torque ('torque in face').
"""
import os, re, collections
import numpy as np

XT_DEFAULT = os.environ.get('OFJ_BRACKET_XT', os.path.join('brackets',
                            'Orthodontics Ceramic Bracket System final.x_t'))

NUM = r'[-+]?(?:\d+\.?\d*|\.\d+)(?:e[-+]?\d+)?'

# ---------------------------------------------------------------- molar tubes (parametric) --
# MBT upper molar tubes: torque -14, tip 0, 10 deg distal offset (rotation).  ASSUMED values
# of the common MBT prescription -- the bracket file has no tubes.
TUBES = {
    16: dict(torque=-14.0, tip=0.0, offset=10.0, length=4.0, height=3.2, depth=2.2,
             slot=(0.022, 0.028), hg_tube=True),
    17: dict(torque=-14.0, tip=0.0, offset=10.0, length=4.0, height=3.0, depth=2.0,
             slot=(0.022, 0.028), hg_tube=False),
}
IN = 25.4
# headgear tube: 0.045" round, beside the rectangular tube.  Offset of its axis from the
# archwire-tube axis in the tube frame (mm, +gingival, +labial).  ASSUMED: occlusal.
HG_TUBE_OFFSET = (float(os.environ.get('OFJ_HG_TUBE_G', -1.25)),
                  float(os.environ.get('OFJ_HG_TUBE_N', 0.25)))
HG_TUBE_D = 0.045*IN
BRACKET_HEIGHT_RULE = os.environ.get('OFJ_BRACKET_HEIGHT', 'fa')   # 'fa' | 'mbt' (initial seat)
# 'wire' (default): passive straight wire, level with the occlusal plane, every slot seated on
# it (level_on_wire), as in the 3D study's ANSYS model.  'fa': brackets at the FA point with
# the wire bent through the slots.
WIRE_SETUP = os.environ.get('OFJ_WIRE_SETUP', 'fa')                # 'fa' | 'smooth' | 'level'
# 'fa' (default): brackets at the FA point -- mid-crown, centred, flush, vertical axis on the
#          FACC -- and the passive wire straight in every slot, bent between them.
# 'smooth': brackets at FA height, wire one smooth curve through the slot centres, slots turned
#          onto it (up to 40 deg on this patient: 17, 13).  'level': one flat wire, slots moved
#          onto it (17 tube ends at 88 % of its crown).  On an unaligned arch a straight wire and
#          correctly placed brackets exclude each other; the 3D study's ANSYS model is aligned.
WIRE_LEVEL = os.environ.get('OFJ_WIRE_LEVEL')                       # mm above the incisor midpoint
# MBT bracket-height chart, middle ('average crown') row: mm from the incisal edge/cusp tip
MBT_HEIGHT = {11: 5.0, 12: 4.5, 13: 5.5, 14: 5.0, 15: 4.5, 16: 4.0, 17: 2.0}


def unit(v):
    v = np.asarray(v, float)
    return v/np.linalg.norm(v)


def rot(axis, deg):
    k = unit(axis); th = np.radians(deg)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th)*K + (1 - np.cos(th))*K @ K


# ========================================================================= XT reader ==
def read_xt(path=XT_DEFAULT):
    """Bodies of a Parasolid text transmit file: {name: dict(points, lines, planes, ...)}.

    Coordinates are returned in mm in each body's PART frame (the file stores metres).
    Record layouts used (verified on this file):
        POINT   29 id attr 0 owner(vertex) next prev x y z
        VERTEX  18 id attr 0 next prev fin point tol owner(body)
        LINE    30 id attr 0 owner next prev geom_owner sense  px py pz  dx dy dz
        PLANE   50 id attr 0 owner next prev geom_owner sense  p(3) n(3) x_axis(3)
        ATTRIB  81 len id . 5 owner 0 0 0 0 name_node      (the SolidWorks body name)
        NAME    84 len id <string>
    Nodes of one body occupy one contiguous id block, which starts at its name node.
    """
    s = open(path, 'r', errors='replace').read().replace('\r', '').replace('\n', '')
    b = s[s.find('**END_OF_HEADER'):]
    if 'PARASOLID' not in s[:400]:
        raise ValueError(f'{path}: not a Parasolid text file')

    names = {}
    for m in re.finditer(r'84 (?:255 )?(\d+) (\d+) (\d\d) - Tor(-?\d+) - Ang(-?\d+)', b):
        names[int(m.group(2))] = (int(m.group(3)), float(m.group(4)), float(m.group(5)))
    owner_name = {}
    for m in re.finditer(r'81 (?:255 )?\d+ (\d+) (\d+) 5 (\d+) 0 0 0 0 (\d+)', b):
        nid = int(m.group(4))
        if nid in names:
            owner_name[int(m.group(3))] = names[nid]
    if not owner_name:
        raise ValueError(f'{path}: no named bracket bodies found')

    P = re.compile(r'(?<=[ ?])29 (\d+) (\d+) 0 (\d+) (\d+) (\d+) (' + NUM + ') (' + NUM + ') (' + NUM + ')(?= )')
    V = re.compile(r'(?<=[ ?])18 (\d+) (\d+) 0 (\d+) (\d+) (\d+) (\d+) (?:\?|' + NUM + r' )(\d+)(?= )')
    L = re.compile(r'(?<=[ ?])30 (\d+) (\d+) 0 (\d+) (\d+) (\d+) (\d+) ([+-])' + ' '.join(['(' + NUM + ')']*6))
    PL = re.compile(r'(?<=[ ?])50 (\d+) (\d+) 0 (\d+) (\d+) (\d+) (\d+) ([+-])' + ' '.join(['(' + NUM + ')']*9))
    pts = {int(m.group(1)): (int(m.group(3)), [float(m.group(i)) for i in (6, 7, 8)])
           for m in P.finditer(b)}
    vtx_body = {int(m.group(1)): int(m.group(7)) for m in V.finditer(b)}

    # id blocks: [name id of this body, name id of the next body)
    name_ids = sorted(nid for nid in names)
    body_of_nameid = {}
    for m in re.finditer(r'81 (?:255 )?\d+ (\d+) (\d+) 5 (\d+) 0 0 0 0 (\d+)', b):
        body_of_nameid[int(m.group(4))] = int(m.group(3))
    blocks = []
    for i, nid in enumerate(name_ids):
        hi = name_ids[i+1] if i+1 < len(name_ids) else 1 << 40
        if nid in body_of_nameid:
            blocks.append((nid, hi, body_of_nameid[nid]))

    def block_of(node_id):
        for lo, hi, body in blocks:
            if lo <= node_id < hi:
                return body
        return None

    bodies = collections.defaultdict(lambda: dict(points=[], lines=[], planes=[]))
    for pid, (vid, xyz) in pts.items():
        body = vtx_body.get(vid)
        if body in owner_name:
            bodies[body]['points'].append(xyz)
    for m in L.finditer(b):
        body = block_of(int(m.group(1)))
        if body in owner_name:
            v = [float(m.group(i)) for i in range(8, 14)]
            bodies[body]['lines'].append(v)
    for m in PL.finditer(b):
        body = block_of(int(m.group(1)))
        if body in owner_name:
            v = [float(m.group(i)) for i in range(8, 17)]
            bodies[body]['planes'].append(v)

    out = {}
    for body, D in bodies.items():
        fdi, tor, ang = owner_name[body]
        X = np.array(D['points'])*1e3
        Ls = np.array(D['lines']) if D['lines'] else np.zeros((0, 6))
        Ls[:, :3] *= 1e3
        Pl = np.array(D['planes']) if D['planes'] else np.zeros((0, 9))
        Pl[:, :3] *= 1e3
        out[fdi] = dict(fdi=fdi, torque=tor, tip=ang, body=body, points=X, lines=Ls,
                        planes=Pl, source=os.path.basename(path))
    return out


# ================================================================== slot detection ==
def slot_frame(B, width_range=(0.50, 0.62)):
    """Slot of one library bracket, in its part frame.

    The slot is a channel along x.  Its four long edges are straight lines parallel to x;
    seen end-on (y, z) they are the corners of the slot cross-section, rotated about x by
    the bracket torque.  Derotate by the torque, then look for the pair of corner pairs
    one slot-width apart at the same height: the lower pair is the floor, the upper the top.
    """
    Ls = B['lines']
    d = Ls[:, 3:]
    sel = np.abs(d[:, 0]) > 0.95
    yz = Ls[sel][:, 1:3]
    best = None
    for tor_sign in (+1, -1):
        th = np.radians(B['torque'])*tor_sign
        # rotate (y,z) by -th about x: floor becomes horizontal
        c, s = np.cos(th), np.sin(th)
        Rm = np.array([[c, s], [-s, c]])
        q = yz @ Rm.T
        cand = []
        for i in range(len(q)):
            for j in range(len(q)):
                dy = q[j, 0] - q[i, 0]
                if abs(q[i, 1] - q[j, 1]) < 0.03 and width_range[0] <= dy <= width_range[1]:
                    # a floor pair has body material (the walls) rising above both ends
                    up_i = q[(np.abs(q[:, 0] - q[i, 0]) < 0.15) & (q[:, 1] > q[i, 1] + 0.3)]
                    up_j = q[(np.abs(q[:, 0] - q[j, 0]) < 0.15) & (q[:, 1] > q[j, 1] + 0.3)]
                    if len(up_i) and len(up_j):
                        h = min(up_i[:, 1].max(), up_j[:, 1].max()) - q[i, 1]
                        cand.append((abs(dy - 0.5588), q[i, 1], q[i, 0], q[j, 0], h))
        if cand:
            cand.sort()
            c0 = cand[0]
            if best is None or c0[0] < best[0]:
                best = (c0[0], tor_sign, c0, Rm)
    if best is None:
        raise RuntimeError(f"slot not found on bracket {B['fdi']}")
    _, tor_sign, c0, Rm = best
    fl = (c0[1], c0[2], c0[3])
    Rinv = Rm.T                                     # derotated -> part (y,z)
    y_c = 0.5*(fl[1] + fl[2])
    width = fl[2] - fl[1]
    depth = float(c0[4])
    floor_c = Rinv @ np.array([y_c, fl[0]])         # part (y,z) of the slot floor centre
    n_yz = Rinv @ np.array([0.0, 1.0])              # floor normal, out of the slot
    w_yz = Rinv @ np.array([1.0, 0.0])              # across the slot, towards +y
    X = B['points']
    # mesiodistal extent of the slot: body vertices on the slot walls
    q = (X[:, 1:3] - floor_c) @ np.stack([w_yz, n_yz], 1)
    wall = (np.abs(np.abs(q[:, 0]) - width/2) < 0.12) & (q[:, 1] > -0.1) & (q[:, 1] < depth + 0.1)
    xs = X[wall, 0] if wall.sum() >= 4 else X[:, 0]
    x0, x1 = float(xs.min()), float(xs.max())
    return dict(floor=np.array([0.5*(x0+x1), *floor_c]),
                n=np.array([0.0, *n_yz]), w=np.array([0.0, *w_yz]), x=np.array([1.0, 0, 0]),
                width=float(width), depth=float(depth), length=float(x1 - x0),
                torque_sign=tor_sign, x_range=(x0, x1))


def wing_axis(B):
    """Vertical reference of the bracket: the tie-wing side planes (normal in the xy plane,
    rotated by the angulation about z).  Returns a unit vector in the part xy plane,
    oriented to +y, and the angle it makes with +y (deg, + = towards +x)."""
    Pl = B['planes']
    if len(Pl):
        n = Pl[:, 3:6]
        k = (np.abs(n[:, 2]) < 0.05) & (np.abs(n[:, 0]) > 0.7)
        if k.any():
            nn = n[k]*np.sign(n[k][:, 0:1])
            m = unit(nn.mean(0))
            v = np.array([-m[1], m[0], 0.0])
            v = v if v[1] > 0 else -v
            return v, float(np.degrees(np.arctan2(v[0], v[1])))
    return np.array([0, 1.0, 0]), 0.0


def base_point(B, S):
    """Bonding point: on the base plane (z = 0) straight under the slot centre."""
    return np.array([S['floor'][0], S['floor'][1], 0.0])


# ================================================================= render surfaces ==
def hull_mesh(points, cut=None, drop_above_y=None):
    from scipy.spatial import ConvexHull
    X = points
    if drop_above_y is not None:
        X = X[X[:, 1] <= drop_above_y]
    H = ConvexHull(X)
    V = X; F = H.simplices.copy()
    # orient outward
    c = V.mean(0)
    nrm = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    flip = np.einsum('ij,ij->i', nrm, V[F].mean(1) - c) < 0
    F[flip] = F[flip][:, [0, 2, 1]]
    used = np.unique(F); rm = -np.ones(len(V), int); rm[used] = np.arange(len(used))
    V, F = V[used], rm[F]
    if cut is not None:
        try:
            V, F = subtract_box(V, F, *cut)
        except Exception as e:                       # boolean is cosmetic only
            print(f"    [brackets] slot cut skipped ({type(e).__name__}: {e})")
    return V, F


def box(center, axes, half):
    """8 vertices + 12 triangles of an oriented box."""
    c = np.asarray(center); A = np.asarray(axes)
    s = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], float)
    V = c + (s*np.asarray(half)) @ A
    F = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                  [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    # orient outward
    nrm = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    flip = np.einsum('ij,ij->i', nrm, V[F].mean(1) - c) < 0
    F[flip] = F[flip][:, [0, 2, 1]]
    return V, F


def subtract_box(V, F, center, axes, half):
    import manifold3d as m3
    def man(V, F):
        return m3.Manifold(m3.Mesh(vert_properties=np.asarray(V, np.float32),
                                   tri_verts=np.asarray(F, np.uint32)))
    Vb, Fb = box(center, axes, half)
    r = (man(V, F) - man(Vb, Fb)).to_mesh()
    return np.asarray(r.vert_properties[:, :3], float), np.asarray(r.tri_verts, int)


# ================================================================ library brackets ==
def library(path=XT_DEFAULT, fdis=(11, 12, 13, 14, 15)):
    """Bracket definitions in PART coordinates for the requested teeth."""
    allb = read_xt(path)
    out = {}
    for f in fdis:
        if f not in allb:
            raise KeyError(f'bracket {f} not in {path}; bodies found: {sorted(allb)}')
        B = allb[f]
        S = slot_frame(B)
        v, ang = wing_axis(B)
        base = base_point(B, S)
        top_y = S['floor'][1] + 2.4        # drop gingival hooks from the hull
        cut = (S['floor'] + S['n']*(S['depth'] + 2.0)/2 + np.array([0, 0, 0]),
               np.stack([S['x'], S['w'], S['n']]),
               (S['length']/2 + 2.0, S['width']/2, (S['depth'] + 2.0)/2))
        stl = os.path.join(os.path.dirname(path), 'stl', f'{f}.stl')
        if os.path.exists(stl):
            import trimesh
            Mm = trimesh.load(stl, force='mesh')
            Vm, Fm = np.asarray(Mm.vertices, float), np.asarray(Mm.faces, int)
            mesh_src = stl
        else:
            Vm, Fm = hull_mesh(B['points'], cut=cut, drop_above_y=top_y)
            mesh_src = 'convex hull of the XT vertices, slot cut'
        out[f] = dict(B, slot=S, wing=v, wing_angle=ang, base=base,
                      mesh=(Vm, Fm), mesh_src=mesh_src)
    return out


def tube_part(fdi):
    """Parametric buccal tube in the same part-frame convention as the library brackets."""
    T = TUBES[fdi]
    L, H, D = T['length'], T['height'], T['depth']
    sw, sd = T['slot'][0]*IN, T['slot'][1]*IN
    floor_z = D/2 - sd/2                       # lumen centred in the tube body
    S = dict(floor=np.array([0.0, 0.0, floor_z]), n=np.array([0, 0, 1.0]),
             w=np.array([0, 1.0, 0]), x=np.array([1.0, 0, 0]), width=sw, depth=sd,
             length=L, torque_sign=1, x_range=(-L/2, L/2))
    Vb, Fb = box([0, 0, D/2], np.eye(3), (L/2, H/2, D/2))
    try:
        Vb, Fb = subtract_box(Vb, Fb, [0, 0, floor_z + sd/2], np.eye(3), (L/2 + 1, sw/2, sd/2))
    except Exception:
        pass
    hg = None
    if T['hg_tube']:
        g, n = HG_TUBE_OFFSET
        hg = np.array([0.0, g, floor_z + sd/2 + n])
    return dict(fdi=fdi, torque=T['torque'], tip=T['tip'], offset=T['offset'],
                slot=S, wing=np.array([0, 1.0, 0]), wing_angle=0.0,
                base=np.array([0.0, 0.0, 0.0]), mesh=(Vb, Fb),
                mesh_src='parametric tube', hg_tube=hg, points=Vb)


# ====================================================================== placement ==
def crown_surface(G, u):
    """Crown triangle soup of tooth u (no attachment), with face normals and centroids."""
    t = G['teeth'][u]
    F = t['crown_faces']
    V = G['nodes']
    tri = V[F]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    a = np.linalg.norm(n, axis=1)
    n = n/np.maximum(a[:, None], 1e-12)
    c = tri.mean(1)
    # outward: away from the crown centre
    flip = np.einsum('ij,ij->i', n, c - t['crown_c']) < 0
    n[flip] = -n[flip]
    return c, n, 0.5*a


def fa_point(G, u, fdi):
    """Facial-axis (FA) point: mid clinical crown on the buccal/labial surface
    (Andrews), or the MBT bracket-height chart when OFJ_BRACKET_HEIGHT=mbt."""
    t = G['teeth'][u]
    c, n, a = crown_surface(G, u)
    eo, ed, eb = unit(t['e_o']), unit(t['e_d']), unit(t['e_b'])
    h = c @ eo
    top, gum = float(t['occl_tip']), float(t['gingival_level'])
    buccal = (n @ eb > 0.35) & (h > gum) & (h < top)
    if BRACKET_HEIGHT_RULE == 'mbt':
        h_fa = top - MBT_HEIGHT[fdi]
    else:
        h_fa = gum + 0.5*(top - gum)
    band = buccal & (np.abs(h - h_fa) < 0.6)
    if band.sum() < 5:
        band = buccal
    md = c[band] @ ed
    md_c = 0.5*(md.min() + md.max())
    score = (c[buccal] @ ed - md_c)**2 + (h[buccal] - h_fa)**2
    k = np.where(buccal)[0][np.argsort(score)[:12]]
    fa = c[k[0]]
    near = np.linalg.norm(c - fa, axis=1) < 1.5
    nrm = unit((n[near]*a[near, None]).sum(0))
    return fa, nrm, dict(h_fa=h_fa, crown_height=top - gum)


def place(G, u, fdi, part):
    """Rigid transform (R, t) taking the PART frame of a bracket/tube onto tooth u.

    Base point -> FA point; base normal (+z) -> crown surface normal at FA;
    vertical reference (the tie wings) -> the FACC direction (gingival);
    x -> mesial.  The prescription (torque, tip) is then the one built into the bracket.
    Tubes get their prescription applied here (torque about x, offset about the normal).
    """
    t = G['teeth'][u]
    fa, nrm, info = fa_point(G, u, fdi)
    eo, ed = unit(t['e_o']), unit(t['e_d'])
    g = -eo - (-eo @ nrm)*nrm; g = unit(g)               # FACC, gingival
    m = np.cross(g, nrm)                                  # right-handed (m, g, n)
    mesial_is_plus = (m @ -ed) > 0
    # part frame of the bracket: (x', y'=wing axis, z)
    z = np.array([0, 0, 1.0]); yv = unit(part['wing']); xv = np.cross(yv, z)
    Rp = np.stack([xv, yv, z], 1)                         # columns: part axes
    Rt = np.stack([m, g, nrm], 1)                         # columns: tooth axes
    R = Rt @ Rp.T
    if 'offset' in part:                                  # parametric tube prescription
        # torque: rotate the tube about its mesiodistal axis; offset: about the normal
        R = R @ rot([1, 0, 0], part['torque'])
        sgn = 1.0 if mesial_is_plus else -1.0
        R = rot(nrm, sgn*part['offset']) @ R
    tvec = fa - R @ part['base']
    return R, tvec, dict(fa=fa, normal=nrm, facc=g, mesial_is_plus_x=bool(mesial_is_plus), **info)


def build_fixed(G, xt=XT_DEFAULT, keep=None, fdi_of=None):
    """Brackets + tubes placed on the half arch.  Everything in the model's REST frame."""
    lib = library(xt)
    out = {}
    for u in keep:
        f = fdi_of[u]
        part = lib[f] if f in lib else tube_part(f)
        R, tv, info = place(G, u, f, part)
        S = part['slot']
        mesial = R @ S['x'] if info['mesial_is_plus_x'] else -(R @ S['x'])
        c_floor = R @ S['floor'] + tv
        n = R @ S['n']; w = R @ S['w']
        Vm, Fm = part['mesh']
        rec = dict(fdi=f, unn=u, R=R, t=tv, info=info,
                   slot_floor=c_floor, slot_n=n, slot_w=w, slot_axis=unit(mesial),
                   slot_width=S['width'], slot_depth=S['depth'], slot_length=S['length'],
                   torque=part['torque'], tip=part['tip'],
                   wing_angle=part.get('wing_angle', 0.0),
                   mesh_V=Vm @ R.T + tv, mesh_F=Fm, mesh_src=part['mesh_src'],
                   kind='tube' if 'offset' in part else 'bracket')
        if part.get('hg_tube') is not None:
            rec['hg_tube'] = R @ part['hg_tube'] + tv
        out[u] = rec
    return out


def _turn(a, b):
    """Smallest rotation taking unit vector a onto unit vector b."""
    v = np.cross(a, b); c = float(a @ b)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K/(1 + c)


def _crown_depth(G, u, P):
    """Deepest penetration (mm, > 0 = inside) of points P into the crown of tooth u."""
    from scipy.spatial import cKDTree
    c, n, _ = crown_surface(G, u)
    _, j = cKDTree(c).query(P)
    return float(np.max(-np.einsum('ij,ij->i', P - c[j], n[j])))


def level_on_wire(fx, G, origin, en, n_sag, sag_offset, wire_h, level=None, iters=12, flat=True):
    """Passive straight-wire set-up (as in the 3D study's ANSYS model): one smooth archwire,
    level with the occlusal plane (normal en), and every slot seated on it -- slot centre on
    the wire, slot axis along the wire tangent, slot normal horizontal.  Each bracket/tube is
    turned rigidly from its FA placement onto that frame (its prescription stays in the body),
    then moved along the slot normal until it just touches the crown.  The wire runs through
    the slot centres as an interpolating spline mirrored about the midsagittal plane, so it
    is smooth and crosses the midline at a right angle.

    level  height of the wire above origin along en (mm); default: the mean FA slot height,
           lowered where needed to keep 1.8 mm between the wire and the occlusal tip of every
           tooth (the second-molar tube otherwise rides over the cusps)."""
    from scipy.interpolate import splprep, splev
    order = list(fx)                                      # distal (17) -> mesial (11)
    c0 = {u: fx[u]['slot_floor'] + fx[u]['slot_n']*(wire_h/2) for u in order}
    if not flat:
        level = None
    elif level is None:
        top = min(float(((G['nodes'][G['teeth'][u]['crown_faces']].reshape(-1, 3) - origin) @ en).max())
                  for u in order)
        level = min(float(np.mean([(c0[u] - origin) @ en for u in order])), top - 1.8)
    P = {u: (c0[u] - ((c0[u] - origin) @ en - level)*en) if flat else c0[u].copy() for u in order}

    def mirror(p):
        return p - 2*(p @ n_sag - sag_offset)*n_sag

    def spline():
        pts = [P[u] for u in order] + [mirror(P[u]) for u in reversed(order)]
        tck, s = splprep(np.array(pts).T, s=0, k=3)
        return tck, s[:len(order)]

    for _ in range(iters):
        tck, s = spline()
        new, worst = {}, 0.0
        for i, u in enumerate(order):
            r = fx[u]
            t = np.array(splev(s[i], tck, der=1))
            if flat:
                t -= (t @ en)*en
            t = unit(t)
            if t @ r['slot_axis'] < 0:
                t = -t                                    # keep the slot's mesial sense
            if flat:
                nn = unit(np.cross(en, t))
                if nn @ r['slot_n'] < 0:
                    nn = -nn
                F0 = np.stack([r['slot_axis'], unit(np.cross(r['slot_n'], r['slot_axis'])), r['slot_n']], 1)
                F0[:, 2] = unit(np.cross(F0[:, 0], F0[:, 1]))
                F1 = np.stack([t, unit(np.cross(nn, t)), nn], 1)
                Q = F1 @ F0.T
            else:                                         # smallest turn taking the slot onto the wire
                Q = _turn(r['slot_axis'], t)
                nn = Q @ r['slot_n']
            new[u] = (Q, P[u])
            d = _crown_depth(G, u, (r['mesh_V'] - c0[u]) @ Q.T + P[u])
            P[u] = P[u] + nn*d                            # push out (d > 0) or pull in to touch
            worst = max(worst, abs(d))
        if worst < 0.01:
            break
    tck, s = spline()
    out = {}
    for u in order:
        r = dict(fx[u])
        Q, c1 = new[u][0], P[u]
        tr = lambda p: (np.asarray(p) - c0[u]) @ Q.T + c1
        r['R'] = Q @ r['R']; r['t'] = tr(r['t'])
        r['mesh_V'] = tr(r['mesh_V'])
        r['slot_floor'] = tr(r['slot_floor'])
        r['slot_n'] = Q @ r['slot_n']; r['slot_w'] = Q @ r['slot_w']; r['slot_axis'] = Q @ r['slot_axis']
        if 'hg_tube' in r:
            r['hg_tube'] = tr(r['hg_tube'])
        r['info'] = dict(r['info'], wire_level=level, moved_from_fa=float(np.linalg.norm(c1 - c0[u])),
                         turned=float(np.degrees(np.arccos(np.clip((np.trace(Q) - 1)/2, -1, 1)))),
                         depth=_crown_depth(G, u, r['mesh_V']))
        out[u] = r
    return out, level


def report(fx):
    lines = [f"{'FDI':>4} {'kind':<8}{'tor':>6}{'tip':>6}{'slot w':>8}{'depth':>7}{'len':>6}"
             f"{'wing':>7}  mesh"]
    for u, r in fx.items():
        lines.append(f"{r['fdi']:>4} {r['kind']:<8}{r['torque']:6.0f}{r['tip']:6.0f}"
                     f"{r['slot_width']:8.3f}{r['slot_depth']:7.3f}{r['slot_length']:6.2f}"
                     f"{r['wing_angle']:7.1f}  {r['mesh_src']}")
    return "\n".join(lines)


if __name__ == '__main__':
    import sys
    lib = library(sys.argv[1] if len(sys.argv) > 1 else XT_DEFAULT)
    for f, B in lib.items():
        S = B['slot']
        print(f"FDI {f}: tor {B['torque']:+.0f} tip {B['tip']:+.0f}  slot width {S['width']:.3f} mm "
              f"({S['width']/IN*1000:.1f} thou) depth {S['depth']:.3f} length {S['length']:.2f}  "
              f"floor {np.round(S['floor'], 3)} normal {np.round(S['n'], 3)} "
              f"(torque from normal {np.degrees(np.arctan2(-S['n'][1], S['n'][2])):+.1f})  "
              f"wing axis {B['wing_angle']:+.1f} deg  mesh {len(B['mesh'][1])} tris")
