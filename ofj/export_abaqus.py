"""Stage 8: write an Abaqus/CalculiX .inp deck for the distalization model.

Units: N, mm, MPa.  Element type C3D4 (linear tetrahedron).
The deck reproduces the same model solved in-session, but declares true
surface-to-surface finite-sliding contact so it can be re-run in a commercial
solver without simplification.
"""
import sys, os, pickle
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mesh_io import boundary_faces

FACE_OF = {(0, 1, 2): 'S1', (0, 1, 3): 'S2', (1, 2, 3): 'S3', (0, 2, 3): 'S4'}


def tet_face_ids(tets, faces):
    """Map each triangle to (element index, Abaqus face label) for a *SURFACE."""
    key = {}
    for e, t in enumerate(tets):
        for combo, lab in FACE_OF.items():
            key[tuple(sorted(t[list(combo)]))] = (e, lab)
    out = []
    for f in faces:
        k = tuple(sorted(f))
        if k in key:
            out.append(key[k])
    return out


def w_nodes(fh, nodes):
    fh.write('*NODE\n')
    for i, p in enumerate(nodes):
        fh.write('%d, %.6f, %.6f, %.6f\n' % (i+1, p[0], p[1], p[2]))


def w_elems(fh, elems, start, name):
    fh.write('*ELEMENT, TYPE=C3D4, ELSET=%s\n' % name)
    for i, e in enumerate(elems):
        fh.write('%d, %d, %d, %d, %d\n' % (start+i, e[0]+1, e[1]+1, e[2]+1, e[3]+1))
    return start + len(elems)


def w_nset(fh, name, ids, per=8):
    fh.write('*NSET, NSET=%s\n' % name)
    for i in range(0, len(ids), per):
        fh.write(', '.join(str(int(x)+1) for x in ids[i:i+per]) + ',\n')


def export(model='build/model.pkl', path='out/fem/maxilla_distalization.inp',
           force_gf=250.0, mu=0.20):
    M = pickle.load(open(model, 'rb'))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nodes, tets, tags = M['nodes'], M['tets'], M['tags']
    teeth, att, A, nS = M['teeth'], M['att'], M['aligner'], M['nS']
    MAT = M['MAT']

    groups = []
    bt = tets[tags == 3]
    bmat = M['mats'][M['pid'] == 0]
    groups.append(('BONE_CORTICAL', bt[bmat == 3], 3))
    groups.append(('BONE_CANCELLOUS', bt[bmat == 4], 4))
    groups.append(('PDL', tets[tags == 2], 2))
    groups.append(('TEETH', tets[tags == 1], 1))
    for unn, a in att.items():
        groups.append(('ATTACH_%d' % unn, a['tets'], 5))
    groups.append(('ALIGNER', A['tets'] + nS, 6))

    with open(path, 'w') as f:
        f.write('** Open-Full-Jaw patient 8 maxilla - total arch distalization with a clear aligner\n')
        f.write('** teeth + PDL + maxillary bone + 0.5 mm PET-G aligner + 4 composite attachments\n')
        f.write('** units: N, mm, MPa, s\n*HEADING\nMaxillary en-masse distalization, %g gf per side\n'
                % force_gf)
        w_nodes(f, M['nodes'])
        s = 1
        spans = {}
        for name, el, mid in groups:
            spans[name] = (s, s+len(el)-1)
            s = w_elems(f, el, s, name)

        for name, el, mid in groups:
            label, E, nu = MAT[mid]
            f.write('*SOLID SECTION, ELSET=%s, MATERIAL=MAT%d\n' % (name, mid))
        for mid, (label, E, nu) in sorted(MAT.items()):
            f.write('*MATERIAL, NAME=MAT%d\n** %s\n*ELASTIC\n%g, %g\n' % (mid, label, E, nu))

        # ---- node sets ----
        w_nset(f, 'FIXED_BONE', M['fixed'])
        for side, L in M['loads'].items():
            w_nset(f, 'HOOK_%s' % side.upper(), L['nodes'])

        # ---- contact surfaces ----
        alg = A['tets'] + nS
        inner = A['inner_faces'] + nS
        fl = tet_face_ids(alg, inner)
        f.write('*SURFACE, NAME=ALIGNER_INNER, TYPE=ELEMENT\n')
        for e, lab in fl:
            f.write('%d, %s\n' % (spans['ALIGNER'][0]+e, lab))
        crown = []
        for unn, t in teeth.items():
            cf = t['crown_faces']
            supra = (M['nodes'][cf] @ t['e_o'] >= t['gingival_level']).all(1)
            crown.append(cf[supra])
        crown = np.vstack(crown)
        tt = tets[tags == 1]
        fl = tet_face_ids(tt, crown)
        f.write('*SURFACE, NAME=CROWNS, TYPE=ELEMENT\n')
        for e, lab in fl:
            f.write('%d, %s\n' % (spans['TEETH'][0]+e, lab))
        for unn, a in att.items():
            fl = tet_face_ids(a['tets'], boundary_faces(a['tets']))
            f.write('*SURFACE, NAME=ATT%d_SURF, TYPE=ELEMENT\n' % unn)
            for e, lab in fl:
                f.write('%d, %s\n' % (spans['ATTACH_%d' % unn][0]+e, lab))

        f.write('*SURFACE INTERACTION, NAME=ALIGNER_TOOTH\n*FRICTION\n%g,\n' % mu)
        f.write('*SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE=HARD\n')
        f.write('*CONTACT PAIR, INTERACTION=ALIGNER_TOOTH, TYPE=SURFACE TO SURFACE, ADJUST=0.2\n')
        f.write('ALIGNER_INNER, CROWNS\n')
        for unn in att:
            f.write('*CONTACT PAIR, INTERACTION=ALIGNER_TOOTH, TYPE=SURFACE TO SURFACE, ADJUST=0.2\n')
            f.write('ALIGNER_INNER, ATT%d_SURF\n' % unn)

        # ---- MPCs tying the refined attachment base to the tooth surface ----
        if M['mpc']:
            f.write('** refined attachment base nodes tied to the parent tooth surface\n')
            for sdof, w in M['mpc'].items():
                for d in range(1, 4):
                    terms = [(sdof, d, 1.0)] + [(m, d, -wt) for m, wt in w.items()]
                    f.write('*EQUATION\n%d\n' % len(terms))
                    f.write(', '.join('%d, %d, %.10g' % (n+1, dd, c) for n, dd, c in terms) + '\n')

        f.write('*BOUNDARY\nFIXED_BONE, 1, 3, 0.0\n')
        f.write('*STEP, NLGEOM=NO\n*STATIC\n0.1, 1.0, 1e-5, 1.0\n')
        F = force_gf*9.80665e-3
        for side, L in M['loads'].items():
            per = F/len(L['nodes'])
            for d in range(3):
                f.write('*CLOAD\nHOOK_%s, %d, %.10g\n' % (side.upper(), d+1, per*L['dirn'][d]))
        f.write('*OUTPUT, FIELD\n*NODE OUTPUT\nU, RF\n*ELEMENT OUTPUT\nS, E\n')
        f.write('*OUTPUT, FIELD\n*CONTACT OUTPUT\nCSTRESS, CDISP\n')
        f.write('*END STEP\n')
    return path, spans


if __name__ == '__main__':
    p, spans = export()
    print('wrote', p, '%.1f MB' % (os.path.getsize(p)/1e6))
    for k, (a, b) in spans.items():
        print('  %-18s elements %d..%d' % (k, a, b))
