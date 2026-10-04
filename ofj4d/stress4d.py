"""PDL stress and per-tooth load summaries, shared by tools/pdl_stress.py (aligner groups, replayed
steps) and fixed4d.py (HG group, dumped during the run with OFJ_STRESS_DUMP).

Stress = D(E_s, nu) eps per PDL element, E_s the converged secant modulus, in kPa.  Hydrostatic =
mean normal stress (negative = compression); von Mises.  Per tooth: 5th/95th percentile hydrostatic
and 95th percentile von Mises over the whole PDL and its cervical / apical thirds (depth along the
tooth axis).  Loads are resolved in the arch frame: distal (+ posterior), buccal/labial (+ outward,
per tooth), occlusal (+).
"""
import numpy as np
import spec
from fem import isotropic_D

_DHAT = None


def fields(S, nodes, eps, E_pdl):
    global _DHAT
    if _DHAT is None:
        _DHAT = isotropic_D(1.0, spec.PDL_NU)
    sig = np.einsum('kl,nl->nk', _DHAT, eps)*E_pdl[:, None]*1e3            # kPa
    hyd = sig[:, :3].mean(1)
    sx, sy, sz, txy, tyz, tzx = sig.T
    vm = np.sqrt(0.5*((sx-sy)**2 + (sy-sz)**2 + (sz-sx)**2) + 3*(txy**2 + tyz**2 + tzx**2))
    cen = nodes[S.pdl_tets].mean(1)
    return hyd, vm, cen


def per_tooth(S, state, ref, hyd, vm, cen):
    teeth = {}
    for u in S.keep:
        m = S.pdl_owner == u
        cp = ref['crown'][u] @ state.T[u][:3, :3].T + state.T[u][:3, 3]
        rp = ref['root'][u] @ state.T[u][:3, :3].T + state.T[u][:3, 3]
        ax = (cp - rp)/np.linalg.norm(cp - rp)
        depth = (cp - cen[m]) @ ax
        rel = (depth - depth.min())/max(np.ptp(depth), 1e-9)

        def st(sel):
            h_, v_ = hyd[m][sel], vm[m][sel]
            return dict(hyd_p5=float(np.percentile(h_, 5)), hyd_p95=float(np.percentile(h_, 95)),
                        vm_p95=float(np.percentile(v_, 95)))
        teeth[int(spec.FDI[u])] = dict(all=st(slice(None)), cervical=st(rel <= 1/3), apical=st(rel >= 2/3))
    return teeth


def resolve(S, arch, u, F):
    """Force vector -> (distal, buccal, occlusal) components for tooth u, in N."""
    eb = S.tooth[u]['e_b']; eb = eb - (eb @ arch.en)*arch.en; eb /= np.linalg.norm(eb)
    return dict(distal=float(F @ arch.ey), buccal=float(F @ eb), occlusal=float(F @ arch.en),
                magnitude=float(np.linalg.norm(F)))
