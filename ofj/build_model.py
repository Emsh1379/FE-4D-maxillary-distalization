"""Stage 5: assemble the complete FE model (parts, materials, BCs, contact, load)."""
import sys, os, pickle, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from scipy.spatial import cKDTree
from mesh_io import boundary_faces

# ---------------- material set (N, mm, MPa) ----------------
MAT = {
    1: ('tooth (enamel+dentin, lumped)', 19600.0, 0.30),
    2: ('periodontal ligament',              0.68, 0.45),
    3: ('cortical bone',                 13700.0, 0.30),
    4: ('cancellous bone',                1370.0, 0.30),
    5: ('composite attachment',          12500.0, 0.30),
    6: ('aligner sheet (PET-G, 0.5 mm)',  2030.0, 0.30),
}
CORTICAL_T = 1.5      # mm cortical shell thickness measured from the outer bone surface
FIX_Z      = 79.0     # mm, bone at/above this level is fully fixed (far field)
FORCE_GF   = 250.0    # gram-force per side
G_ACC      = 9.80665e-3   # N per gram-force
HOOK_TOOTH = {'right': 6, 'left': 11}   # precision-cut hook at the canines
HOOK_R     = 1.75     # mm, radius of the hook loading patch


def build(stage4='build/stage4.pkl', out='build/model.pkl'):
    S = pickle.load(open(stage4,'rb'))
    nodes, tets, tags, teeth, att, A = (S['nodes'], S['tets'], S['tags'],
                                        S['teeth'], S['attachments'], S['aligner'])
    nS = len(nodes)
    allnodes = np.vstack([nodes, A['nodes']])
    E_list, M_list, part = [], [], []

    bt = tets[tags==3]
    # cortical vs cancellous by distance of the element centroid to the outer bone surface
    bf = boundary_faces(bt)
    pdln = np.zeros(len(nodes), bool); pdln[np.unique(tets[tags==2])] = True
    outer = bf[~pdln[bf].any(1)]
    ocent = nodes[outer].mean(1)
    bcent = nodes[bt].mean(1)
    d,_ = cKDTree(ocent).query(bcent)
    bmat = np.where(d <= CORTICAL_T, 3, 4)
    E_list.append(bt);            M_list.append(bmat);                       part.append('bone')
    E_list.append(tets[tags==2]); M_list.append(np.full((tags==2).sum(),2));  part.append('pdl')
    E_list.append(tets[tags==1]); M_list.append(np.full((tags==1).sum(),1));  part.append('teeth')
    at = np.vstack([a['tets'] for a in att.values()])
    E_list.append(at);            M_list.append(np.full(len(at),5));          part.append('attachments')
    E_list.append(A['tets']+nS);  M_list.append(np.full(len(A['tets']),6));   part.append('aligner')
    elems = np.vstack(E_list); mats = np.concatenate(M_list)
    pid = np.concatenate([np.full(len(e), i) for i,e in enumerate(E_list)])

    # ---------------- boundary conditions ----------------
    bone_nodes = np.unique(bt)
    fixed = bone_nodes[allnodes[bone_nodes][:,2] >= FIX_Z]

    # ---------------- MPC (attachment refinement) ----------------
    mpc = S['att_mpc']

    # ---------------- contact: aligner inner surface -> crown + attachment ----------------
    master = []
    for u,t in teeth.items():
        cf = t['crown_faces']
        supra = (allnodes[cf] @ t['e_o'] >= t['gingival_level']).all(1)
        master.append(cf[supra])
    for u,a in att.items():
        master.append(boundary_faces(a['tets']))
    master = np.vstack(master)
    slave = np.arange(A['n_inner']) + nS          # aligner inner-surface nodes

    # ---------------- global occlusal plane ----------------
    # normal points occlusally (out of the crowns); the distalising force is applied
    # PARALLEL to this plane so it is a pure distalising force with no intrusive /
    # extrusive component.
    n_occ = np.mean([t['e_o'] for t in teeth.values()], axis=0)
    n_occ /= np.linalg.norm(n_occ)

    # ---------------- load: 250 gf per side at the canine precision cut ----------------
    loads = {}
    outer_al = np.unique(A['outer_faces']) + nS
    for side, u in HOOK_TOOTH.items():
        t = teeth[u]
        # hook sits on the buccal aligner wall, at mid clinical crown, distal half of the canine
        cn = np.unique(t['crown_faces'])
        cpts = allnodes[cn]
        o_c = t['occl_tip'] - 0.5*(t['occl_tip'] - t['gingival_level'])
        md  = (cpts @ t['e_d']).max() - 0.8            # 0.8 mm mesial of the distal edge
        seed = t['crown_c'] + t['e_d']*(md - t['crown_c']@t['e_d']) \
                            + t['e_o']*(o_c - t['crown_c']@t['e_o']) + t['e_b']*4.0
        d,_ = cKDTree(allnodes[outer_al]).query(seed)
        _, i0 = cKDTree(allnodes[outer_al]).query(seed)
        centre = allnodes[outer_al][i0]
        sel = outer_al[np.linalg.norm(allnodes[outer_al]-centre, axis=1) <= HOOK_R]
        dirn = t['e_d'] - (t['e_d'] @ n_occ) * n_occ      # project into the occlusal plane
        dirn /= np.linalg.norm(dirn)
        loads[side] = dict(nodes=sel, centre=centre, dirn=dirn, raw_dirn=t['e_d'],
                           magnitude=FORCE_GF*G_ACC, tooth=u)

    M = dict(n_occ=n_occ, nodes=allnodes, elems=elems, mats=mats, pid=pid, parts=part,
             fixed=fixed, mpc=mpc, master=master, slave=slave,
             loads=loads, teeth=teeth, att=att, nS=nS, aligner=A,
             tags=tags, tets=tets, MAT=MAT)
    pickle.dump(M, open(out,'wb'))
    return M


if __name__ == '__main__':
    M = build()
    n = len(M['nodes'])
    print(f"MODEL: {n} nodes, {3*n} dof, {len(M['elems'])} tet4 elements\n")
    print("part            elements   material")
    for i,p in enumerate(M['parts']):
        k = M['pid']==i
        for m in np.unique(M['mats'][k]):
            nm,E,nu = M['MAT'][m]
            print(f"  {p:<13} {int((M['mats'][k]==m).sum()):8d}   {nm}  E={E} MPa nu={nu}")
    print(f"\nfixed nodes (bone, z>={M and 79.0}): {len(M['fixed'])}")
    print(f"MPC constraints (attachment refinement): {len(M['mpc'])}")
    print(f"contact: {len(M['slave'])} slave nodes -> {len(M['master'])} master triangles")
    print(f"occlusal plane normal (occlusal +): {np.round(M['n_occ'],4)}")
    for s,L in M['loads'].items():
        ang=np.degrees(np.arcsin(abs(L['dirn']@M['n_occ'])))
        print(f"load {s}: {L['magnitude']:.4f} N ({FORCE_GF:.0f} gf) on {len(L['nodes'])} aligner nodes")
        print(f"     dir={np.round(L['dirn'],3)}  out-of-occlusal-plane angle {ang:.2f} deg"
              f"  centre={np.round(L['centre'],1)} (canine UNN{L['tooth']})")
