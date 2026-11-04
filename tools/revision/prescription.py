"""Prescribed (planned) tooth movement at step 70 for every patient: crown-centre displacement
(distal along the arch / palatal), its components, and the planned crown angulation change
(the 0.40 tipping blend). No overcorrection is programmed."""
import os, sys, json, numpy as np
ROOT = sys.argv[1]
out = {}
for P in ['P8', 'P3', 'P4', 'P5', 'P6']:
    W = f'{ROOT}/lws/{P}'; os.chdir(W)
    os.environ.update(OFJ_GEOM='build4d/geom_fe.pkl', OFJ_ALIGNER='build4d/aligner_none.pkl', OFJ_PRESCRIBE='tip',
                      OFJ_TIP_BLEND='0.40', OFJ_PLAN_SCALE=open('build4d/plan_scale.txt').read().strip())
    for m in [k for k in list(sys.modules) if k in ('spec', 'arch4d', 'model4d', 'inputs')]: del sys.modules[m]
    sys.path.insert(0, f'{W}/ofj4d'); sys.path.insert(0, f'{W}/ofj')
    import spec, arch4d
    from model4d import Static
    S = Static(); A = arch4d.ArchForm(S)
    rows = {}
    for u in S.keep:
        t = S.tooth[u]; T = A.prescribed_transform(u, spec.N_STEPS)
        c = t['crown_c']; d = T[:3, :3] @ c + T[:3, 3] - c
        R = T[:3, :3]; ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1)/2, -1, 1)))
        Pc = t['surf_rest'][:t['surf_n']]; cp = Pc[np.argmax(Pc @ t['e_o'])]
        Q = t['pdl_rest']; rp = Q[np.argmin(Q @ t['e_o'])]
        la0 = (cp - rp)/np.linalg.norm(cp - rp); la1 = R @ la0
        # crown tip of the long axis in the tooth's mesiodistal plane (+ = crown distal) and buccolingual plane
        dl = la1 - la0
        tip_md = np.degrees(np.arctan2(dl @ t['e_d'], 1)); tip_bl = np.degrees(np.arctan2(dl @ t['e_b'], 1))
        s = -1 if u in (7, 8) else 1
        rows[spec.FDI[u]] = dict(total=float(np.linalg.norm(d)), distal=float(d @ t['e_d']), buccal=float(s*(d @ t['e_b'])),
                                 intrusion=float(-(d @ t['e_o'])), angle=float(ang), tip_md=float(tip_md), tip_bl=float(tip_bl),
                                 steps=spec.V_PATTERN[u], designed=spec.total_prescribed(u))
    out[P] = rows
    sys.path.pop(0); sys.path.pop(0)
json.dump(out, open(f'{ROOT}/ana/prescription.json', 'w'), indent=1)
for P, rows in out.items():
    print(P)
    for f, r in rows.items(): print(f"  {f}: steps {r['steps']} designed {r['designed']:.2f}  |d| {r['total']:.3f}  d {r['distal']:+.3f} b/p {r['buccal']:+.3f} int {r['intrusion']:+.3f}  rot {r['angle']:.2f} deg (md {r['tip_md']:+.2f}, bl {r['tip_bl']:+.2f})")
