"""Stage 7: clinical post-processing of the distalization solution."""
import sys, os, pickle
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fem import element_strain_stress
from anatomy import split_bodies

# capillary blood pressure in the PDL -- the classical threshold above which
# vascular occlusion and hyalinisation / undermining resorption are expected
CAPILLARY_MPA = 0.0047      # ~4.7 kPa ~= 26 g/cm^2


def rigid_fit(X, U):
    """Least-squares rigid motion u(x) = t + w x (x - c). Returns t, w, c, residual."""
    c = X.mean(0)
    r = X - c
    n = len(X)
    A = np.zeros((3*n, 6))
    A[0::3, 0] = 1; A[1::3, 1] = 1; A[2::3, 2] = 1
    A[0::3, 4] = r[:, 2]; A[0::3, 5] = -r[:, 1]
    A[1::3, 3] = -r[:, 2]; A[1::3, 5] = r[:, 0]
    A[2::3, 3] = r[:, 1]; A[2::3, 4] = -r[:, 0]
    sol, *_ = np.linalg.lstsq(A, U.ravel(), rcond=None)
    t, w = sol[:3], sol[3:]
    resid = np.linalg.norm(A@sol - U.ravel())/max(np.linalg.norm(U.ravel()), 1e-30)
    return t, w, c, resid


def centre_of_rotation(t, w, c):
    """Point on the screw axis closest to c (undefined if |w| ~ 0)."""
    wn = np.linalg.norm(w)
    if wn < 1e-14:
        return None, wn
    return c + np.cross(w, t)/wn**2, wn


def analyse(model='build/model.pkl', solution='build/solution.pkl',
            out='build/results.pkl'):
    M = pickle.load(open(model, 'rb'))
    S = pickle.load(open(solution, 'rb'))
    u = S['u']; nodes = M['nodes']; U = u.reshape(-1, 3)
    teeth = M['teeth']; tets = M['tets']; tags = M['tags']
    res = dict(load_scale=S['load_scale'], mu=S['mu'], hist=S['hist'])

    # ---------------- per-tooth rigid body motion ----------------
    pdl_lab, npdl = split_bodies(tets[tags == 2], len(nodes))
    pdl_tets = tets[tags == 2]
    pdl_cent = np.array([nodes[np.unique(pdl_tets[pdl_lab == i])].mean(0)
                         for i in range(npdl)])
    tooth_rows = {}
    for unn, t in teeth.items():
        nd = np.unique(t['tets'])
        X = nodes[nd]; Uu = U[nd]
        tr, w, c, rr = rigid_fit(X, Uu)
        crot, wn = centre_of_rotation(tr, w, c)
        ed, eb, eo = t['e_d'], t['e_b'], t['e_o']
        crown = t['crown_c']
        apex = X[np.argmax(X @ (-eo))]
        du = lambda p: tr + np.cross(w, p - c)
        d_crown, d_apex = du(crown), du(apex)
        # clinical root length: apex -> gingival margin (CEJ) along the long axis
        root_len = float(t['gingival_level'] - apex @ eo)
        row = dict(
            unn=unn, name=t['name'], fdi=t['fdi'],
            crown_disp=d_crown, apex_disp=d_apex,
            crown_mag=float(np.linalg.norm(d_crown)), apex_mag=float(np.linalg.norm(d_apex)),
            crown_distal=float(d_crown @ ed), crown_buccal=float(d_crown @ eb),
            crown_occlusal=float(d_crown @ eo),
            apex_distal=float(d_apex @ ed), apex_buccal=float(d_apex @ eb),
            apex_occlusal=float(d_apex @ eo),
            rot_deg=float(np.degrees(wn)),
            tip_deg=float(np.degrees(w @ eb)),      # about the buccal axis  = tipping
            torque_deg=float(np.degrees(w @ ed)),   # about the mesiodistal axis = torque
            yaw_deg=float(np.degrees(w @ eo)),      # about the long axis   = rotation
            rigid_residual=float(rr), root_len=root_len,
            e_d=ed, e_b=eb, e_o=eo, crown_c=crown, apex=apex)
        if crot is not None:
            row['crot'] = crot
            row['crot_above_apex'] = float((crot - apex) @ eo)
            row['crot_frac_root'] = float((crot - apex) @ eo / root_len)
        tooth_rows[unn] = row

    # ---------------- PDL stress, per tooth ----------------
    matmap = {m: (M['MAT'][m][1], M['MAT'][m][2]) for m in M['MAT']}
    pm = np.full(len(pdl_tets), 2)
    r = element_strain_stress(nodes, pdl_tets, pm, matmap, u)
    vol = np.abs(r['V'])
    for unn, row in tooth_rows.items():
        i = int(np.argmin(np.linalg.norm(
            pdl_cent - nodes[np.unique(teeth[unn]['tets'])].mean(0), axis=1)))
        k = pdl_lab == i
        vm, hyd, pmin, pmax, v = r['vm'][k], r['hyd'][k], r['p_min'][k], r['p_max'][k], vol[k]
        o = np.argsort(vm)
        cw = np.cumsum(v[o])/v.sum()
        row.update(
            pdl_body=i, pdl_vol=float(v.sum()), pdl_n=int(k.sum()),
            pdl_vm_mean=float((vm*v).sum()/v.sum()),
            pdl_vm_p95=float(vm[o][np.searchsorted(cw, 0.95)]),
            pdl_vm_max=float(vm.max()),
            pdl_hyd_min=float(hyd.min()), pdl_hyd_max=float(hyd.max()),
            pdl_p_min=float(pmin.min()), pdl_p_max=float(pmax.max()),
            pdl_vol_over_capillary=float(v[(-hyd) > CAPILLARY_MPA].sum()/v.sum()))
    res['teeth'] = tooth_rows
    res['pdl_elem'] = dict(vm=r['vm'], hyd=r['hyd'], lab=pdl_lab, V=vol)

    # ---------------- contact force per tooth ----------------
    C = pickle.load(open('build/contact.pkl', 'rb'))
    lam = S['lam']; nrm = C['normal']
    # lam*n is the force the tooth exerts ON THE ALIGNER; the reaction on the tooth
    # is its negative -- report the force actually delivered TO the dentition.
    Fn = -lam[:, None]*nrm
    owner = {}
    for unn, t in teeth.items():
        for nd in np.unique(t['tets']):
            owner[int(nd)] = unn
    for unn, a in M['att'].items():
        for nd in np.unique(a['tets']):
            owner.setdefault(int(nd), unn)
    tri0 = C['tri'][:, 0]
    pair_tooth = np.array([owner.get(int(x), -1) for x in tri0])
    att_nodes = set()
    for a in M['att'].values():
        att_nodes |= set(np.unique(a['tets']).tolist())
    on_att = np.array([int(x) in att_nodes for x in tri0])
    for unn, row in tooth_rows.items():
        k = pair_tooth == unn
        row['contact_force'] = Fn[k].sum(0)
        row['contact_force_mag'] = float(np.linalg.norm(Fn[k].sum(0)))
        row['contact_distal'] = float(Fn[k].sum(0) @ row['e_d'])
        row['n_contact_active'] = int((lam[k] > 0).sum())
        ka = k & on_att
        row['via_attachment_N'] = float(np.linalg.norm(Fn[ka].sum(0))) if ka.any() else 0.0
        row['via_attachment_distal'] = float(Fn[ka].sum(0) @ row['e_d']) if ka.any() else 0.0
        row['n_att_active'] = int((lam[ka] > 0).sum()) if ka.any() else 0
    res['total_contact_force'] = Fn.sum(0)
    res['applied_force'] = sum(L['magnitude']*S['load_scale']*L['dirn']
                               for L in M['loads'].values())

    # ---------------- aligner + bone ----------------
    A = M['aligner']; nS = M['nS']
    al = A['tets'] + nS
    ra = element_strain_stress(nodes, al, np.full(len(al), 6), matmap, u)
    res['aligner'] = dict(vm_max=float(ra['vm'].max()),
                          vm_p99=float(np.percentile(ra['vm'], 99)),
                          vm_mean=float(np.average(ra['vm'], weights=np.abs(ra['V']))),
                          disp_max=float(np.linalg.norm(U[np.unique(al)], axis=1).max()))
    bt = tets[tags == 3]
    bmat = M['mats'][M['pid'] == 0]
    rb = element_strain_stress(nodes, bt, bmat, matmap, u)
    res['bone'] = dict(vm_max=float(rb['vm'].max()),
                       vm_p999=float(np.percentile(rb['vm'], 99.9)),
                       vm_p99=float(np.percentile(rb['vm'], 99)),
                       disp_max=float(np.linalg.norm(U[np.unique(bt)], axis=1).max()))
    res['tooth_disp_max'] = float(max(v['crown_mag'] for v in tooth_rows.values()))
    pickle.dump(res, open(out, 'wb'))
    return res


def report(res):
    W = 112
    nl = chr(10)
    print(nl + "="*W)
    print("INITIAL TOOTH DISPLACEMENT (um)   sign: + distal / + buccal / + occlusal")
    print("="*W)
    print(f"{'tooth':<21}{'crown':>7}{'distal':>8}{'buccal':>8}{'occl':>7}"
          f"{'apex':>7}{'ap.dist':>8}{'tip deg':>9}{'CRot/root':>10}{'Fdist N':>9}{'%F':>6}")
    tot = sum(abs(r['contact_distal']) for r in res['teeth'].values())
    for unn in sorted(res['teeth']):
        r = res['teeth'][unn]
        cf = r.get('crot_frac_root', float('nan'))
        if not np.isfinite(cf) or abs(cf) > 99:
            cf = float('nan')
        print(f"{r['name']:<21}{r['crown_mag']*1e3:7.2f}{r['crown_distal']*1e3:8.2f}"
              f"{r['crown_buccal']*1e3:8.2f}{r['crown_occlusal']*1e3:7.2f}"
              f"{r['apex_mag']*1e3:7.2f}{r['apex_distal']*1e3:8.2f}"
              f"{r['tip_deg']:9.4f}{cf:10.2f}{r['contact_distal']:9.3f}"
              f"{100*abs(r['contact_distal'])/tot:6.1f}")

    print(nl + "="*W)
    print(f"PDL STRESS (MPa)    capillary occlusion threshold = {CAPILLARY_MPA:.4f} MPa")
    print("="*W)
    print(f"{'tooth':<21}{'vm mean':>10}{'vm p95':>10}{'vm max':>10}"
          f"{'max compr':>11}{'max tens':>10}{'%vol>cap':>10}")
    for unn in sorted(res['teeth']):
        r = res['teeth'][unn]
        print(f"{r['name']:<21}{r['pdl_vm_mean']:10.5f}{r['pdl_vm_p95']:10.5f}"
              f"{r['pdl_vm_max']:10.5f}{-r['pdl_hyd_min']:11.5f}{r['pdl_hyd_max']:10.5f}"
              f"{100*r['pdl_vol_over_capillary']:10.1f}")

    print(nl + "="*W)
    print("FORCE DELIVERY THROUGH THE ATTACHMENTS")
    print("="*W)
    for unn in sorted(res['teeth']):
        r = res['teeth'][unn]
        if r['via_attachment_N'] > 0:
            print(f"  {r['name']:<20} total {r['contact_force_mag']:6.3f} N   "
                  f"via attachment {r['via_attachment_N']:6.3f} N "
                  f"({100*r['via_attachment_N']/max(r['contact_force_mag'],1e-12):5.1f}%)   "
                  f"distal component via attachment {r['via_attachment_distal']:+.3f} N")

    print(nl + "="*W)
    print("GLOBAL")
    print("="*W)
    print("aligner : von Mises max %.2f MPa (p99 %.2f), max displacement %.4f mm"
          % (res['aligner']['vm_max'], res['aligner']['vm_p99'], res['aligner']['disp_max']))
    print("bone    : von Mises max %.2f MPa (p99.9 %.3f), max displacement %.5f mm"
          % (res['bone']['vm_max'], res['bone']['vm_p999'], res['bone']['disp_max']))
    ap = res['applied_force']; ct = res['total_contact_force']
    print("applied force on aligner       : %s N  |F| = %.3f" % (np.round(ap, 3), np.linalg.norm(ap)))
    print("normal contact force to teeth  : %s N  |F| = %.3f" % (np.round(ct, 3), np.linalg.norm(ct)))
    # equilibrium of the aligner: applied load = normal contact reaction + friction.
    # ct is the force ON THE TEETH, so the force on the aligner is -ct.
    rsd = ap - ct
    print("residual (carried by friction) : %s N  |F| = %.3f  (%.1f%% of applied)"
          % (np.round(rsd, 3), np.linalg.norm(rsd),
             100*np.linalg.norm(rsd)/np.linalg.norm(ap)))


if __name__ == '__main__':
    import argparse
    ap_ = argparse.ArgumentParser()
    ap_.add_argument('--solution', default='build/solution.pkl')
    ap_.add_argument('--out', default='build/results.pkl')
    ap_.add_argument('--report-only', action='store_true')
    a = ap_.parse_args()
    if a.report_only:
        report(pickle.load(open(a.out, 'rb')))
    else:
        report(analyse(solution=a.solution, out=a.out))
