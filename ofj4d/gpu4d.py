"""GPU assembly of the reduced system.

Why this exists, and why it is not the GPU route the README warned about:

The README's judgement -- "GPU via CuPy can only accelerate CG -- the 14%, not the
75%" -- was right about the AMG-era cost split and right about arithmetic.  A Tesla
T4 measured here does 0.25 TFLOP/s of FP64 against the host's 0.19, i.e. nothing.
But it moves 154 GB/s against the host's 7.1 GB/s, a factor of 22, and *every*
per-iteration assembly kernel in this model is memory-bound, not FLOP-bound:

    PDL stiffness scatter-add (24.2M entries)   234 ms -> 5 ms
    triple product P.T K P (SpGEMM)             165 ms -> 17 ms
    diagonal scaling                             50 ms -> 1 ms

The factorisation itself stays on the host: sparse Cholesky is supernodal dense
GEMM, which is exactly the FP64-bound work a T4 is bad at.  So the division of
labour is assembly on the device, factorisation on the host, ~17 MB across PCIe
per iteration (7 ms).

The second, larger gain is structural.  Assembling on the device means assembling
into a pattern *we* choose instead of one scipy derives from the values:

  * every contact pair is assembled every iteration, inactive ones contributing a
    genuine zero, so the pattern no longer tracks the active set;
  * scipy prunes those zeros (from the triple product and from the sparse
    addition, which is why FINDINGS 8 records the fixed-pattern idea as a dead
    end on the host); cuSPARSE's SpGEMM does not, because it derives the output
    pattern from the input patterns alone -- verified stable across active sets.

The matrix therefore keeps one pattern for a whole staging step, and PARDISO's
symbolic analysis -- 0.59 s of its 0.86 s -- is computed once per step instead of
once per iteration.  The explicit zeros are structural zeros, the ordinary FEM
kind; no epsilon is substituted for any zero.

Units and conventions are those of solve4d; this module only rearranges the same
arithmetic onto the device.
"""
import os
import numpy as np
from scipy.sparse import csr_matrix

# Reductions default to atomics.  The bit-reproducible alternative (sorted segment
# sums) is exact but measured 143 ms per nonlinear iteration slower -- 30% of the
# whole iteration -- because `segment_reduce` costs O(number of segments) and this
# matrix has 8.84M of them averaging 2.7 contributions each.  What it buys is
# bit-identical reruns; what it costs is a third of the runtime, to remove a
# variation of ~1e-16 in a model whose nonlinear tolerance is 1e-3.  Set
# OFJ_GPU_DETERMINISTIC=1 when bit-identical reruns matter more than speed.
DETERMINISTIC = os.environ.get('OFJ_GPU_DETERMINISTIC', '0') != '0'


def available():
    """True when a usable CUDA device is present.  Never raises."""
    if os.environ.get('OFJ_GPU', 'auto').lower() in ('0', 'off', 'no'):
        return False
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def device_name():
    import torch
    p = torch.cuda.get_device_properties(0)
    return f"{p.name} (sm_{p.major}{p.minor}, {p.total_memory/2**30:.1f} GiB)"


class Scatter:
    """A fixed COO -> flat-index reduction, applied to changing values."""

    def __init__(self, inv, n_out, torch):
        self.torch = torch
        self.n_out = n_out
        self.inv = inv
        if DETERMINISTIC:
            self.order = torch.argsort(inv)
            # segment_reduce costs O(segments), so list only the outputs this
            # scatter touches: the contact block reaches 1.5M of 8.84M entries and
            # scanning the empty rest cost 45 of its 48 ms.
            self.touched, counts = torch.unique_consecutive(inv[self.order],
                                                            return_counts=True)
            self.lengths = counts

    def add_into(self, out, vals):
        t = self.torch
        if DETERMINISTIC:
            out.index_add_(0, self.touched,
                           t.segment_reduce(vals[self.order], 'sum',
                                            lengths=self.lengths, unsafe=True))
        else:
            out.index_add_(0, self.inv, vals)
        return out


class ElementScatter(Scatter):
    """The same reduction for element stiffness, with the permutation baked in.

    `K(E) = sum over elements of Ke_unit[e] * E[e]`, and only E changes between
    iterations.  The deterministic path was costing 143 ms per iteration -- not in
    the reduction but in gathering the 194 MB value array in scattered order every
    time.  Permuting the *constant* element operator once at build time turns that
    into a contiguous read plus a gather from the 1.3 MB modulus vector, which
    stays in cache.
    """

    def __init__(self, inv, n_out, Ke_unit, torch):
        super().__init__(inv, n_out, torch)
        self.per_elem = Ke_unit.shape[1]*Ke_unit.shape[2]          # 144
        if DETERMINISTIC:
            flat = Ke_unit.reshape(-1)
            self.Ke = flat[self.order].contiguous()
            self.elem = (self.order // self.per_elem).contiguous()
            del self.order                    # the permutation is now baked in
        else:
            self.Ke_unit = Ke_unit

    def add_modulus(self, out, E):
        """out += the assembled contribution of per-element modulus E."""
        t = self.torch
        if DETERMINISTIC:
            out.index_add_(0, self.touched,
                           t.segment_reduce(self.Ke*E[self.elem], 'sum',
                                            lengths=self.lengths, unsafe=True))
        else:
            out.index_add_(0, self.inv,
                           (self.Ke_unit*E[:, None, None]).reshape(-1))
        return out


class GpuSystem:
    """Builds the scaled reduced system on the device, in a pattern fixed for the
    whole staging step.

    The pattern depends on the element connectivity and on the contact pairing,
    both of which are constant for a step; the values depend on the nodal
    positions, which change between the step's two solves.  So `__init__` takes
    the pattern and `refresh` takes the values.
    """

    def __init__(self, ndof, dofs_contact, aligner_dofs, pdl_dofs, bone_dofs=None):
        import torch
        self.torch = torch
        self.dev = torch.device('cuda')
        self.ndof = ndof
        t = torch

        def coo(d):
            """(nelem, 12) element dofs -> the 144 (row, col) pairs each contributes."""
            d = t.as_tensor(d.astype(np.int64), device=self.dev)
            rows = d.repeat_interleave(12, dim=1).reshape(-1)
            cols = d.repeat(1, 12).reshape(-1)
            return rows*ndof + cols

        kA, kP, kC = coo(aligner_dofs), coo(pdl_dofs), coo(dofs_contact)
        kB = coo(bone_dofs) if bone_dofs is not None else kA[:0]
        nA, nP, nC = kA.numel(), kP.numel(), kC.numel()
        keys = t.cat([kA, kP, kC, kB])
        del kA, kP, kC, kB
        uk, inv = t.unique(keys, return_inverse=True)
        del keys
        self.nnz_g = uk.numel()
        self._invA = inv[:nA].contiguous()
        self._invP = inv[nA:nA+nP].contiguous()
        self.scC = Scatter(inv[nA+nP:nA+nP+nC].contiguous(), self.nnz_g, t)
        self._invB = inv[nA+nP+nC:].contiguous() if bone_dofs is not None else None
        self.scA = self.scP = self.scB = None
        del inv

        g_rows, g_cols = uk // ndof, uk % ndof
        del uk
        self.g_ptr = t.zeros(ndof+1, dtype=t.int64, device=self.dev)
        t.cumsum(t.bincount(g_rows, minlength=ndof), 0, out=self.g_ptr[1:])
        self.g_cols = g_cols
        del g_rows, g_cols

        # contact force scatter: 12 dofs per pair, values are per-pair vectors
        self.c_dofs = t.as_tensor(dofs_contact.astype(np.int64), device=self.dev)
        self.n_pair = dofs_contact.shape[0]
        self._A_pattern = None

    # ----------------------------------------------------------------- values --
    def pdl_update(self, x_host, E1, E2, ek):
        """Post-solve chain, on the device: the reduced solution -> global
        displacement -> PDL element strains -> von Mises equivalent strain ->
        secant modulus of the bilinear curve.

        This is the other half of the per-iteration host work: the strain einsum
        alone runs over 167,751 elements x (6 x 12), and it is the same
        bandwidth-bound shape as the assembly.  Returns the global displacement on
        the host (the contact kinematics still need it) and leaves the strains on
        the device until the loop ends.
        """
        t = self.torch
        x = t.as_tensor(np.ascontiguousarray(x_host), device=self.dev)
        u = t.mv(self.P, x)
        ue = u.reshape(-1, 3)[self.pdl_tets].reshape(-1, 12)
        eps = t.einsum('nij,nj->ni', self.Bp, ue)
        ex, ey, ez, gxy, gyz, gzx = eps.unbind(1)
        tr = (ex + ey + ez)/3.0
        dx, dy, dz = ex - tr, ey - tr, ez - tr
        ee = dx*dx + dy*dy + dz*dz + 0.5*(gxy*gxy + gyz*gyz + gzx*gzx)
        e = t.clamp(t.sqrt(t.clamp(2.0/3.0*ee, min=0.0)), min=1e-12)
        E_new = t.where(e <= ek, t.full_like(e, E1), (E1*ek + E2*(e - ek))/e)
        return u.cpu().numpy(), eps, E_new

    def refresh(self, P, opA, opP, E_aligner, R, a, Pt_tan, opB=None, E_bone=None):
        """Adopt the operators of the current configuration (once per solve)."""
        t = self.torch
        self.P = self._csr(P)
        self.Pt = self._csr(P.T.tocsr())
        KeA = t.as_tensor(opA.Ke_unit, device=self.dev)
        KeP = t.as_tensor(opP.Ke_unit, device=self.dev)
        self.scA = ElementScatter(self._invA, self.nnz_g, KeA, t)
        self.scP = ElementScatter(self._invP, self.nnz_g, KeP, t)
        self.n_pdl = KeP.shape[0]
        del KeA, KeP
        self.Bp = t.as_tensor(opP.B, device=self.dev)
        self.pdl_tets = t.as_tensor(opP.tets.astype(np.int64), device=self.dev)
        self.R = t.as_tensor(R, device=self.dev)
        self.a = t.as_tensor(a, device=self.dev)
        self.Pt_tan = t.as_tensor(Pt_tan, device=self.dev)
        # the aligner is linear elastic and its modulus never changes, so its
        # contribution to the global pattern is computed once per solve
        EA = t.full((opA.Ke_unit.shape[0],), float(E_aligner),
                    dtype=t.float64, device=self.dev)
        base = t.zeros(self.nnz_g, dtype=t.float64, device=self.dev)
        self.base = self.scA.add_modulus(base, EA)
        if opB is not None:
            KeB = t.as_tensor(opB.Ke_unit, device=self.dev)
            self.scB = ElementScatter(self._invB, self.nnz_g, KeB, t)
            del KeB
            self.scB.add_modulus(self.base, t.as_tensor(np.ascontiguousarray(E_bone),
                                                        dtype=t.float64, device=self.dev))
        # The reduced pattern depends on P.  With tied bone P is re-formed every
        # solve, so the pattern is rebuilt here; PARDISO still keeps its analysis
        # whenever the rebuilt pattern turns out identical.
        self._A_pattern = None
        # R^T Pt R, the tangential shape, is configuration- not iteration-dependent
        self.Kt_unit = t.einsum('nki,nkl,nlj->nij', self.R, self.Pt_tan, self.R)
        self.Kn_unit = t.einsum('ni,nj->nij', self.a, self.a)

    def _csr(self, A):
        t = self.torch
        return t.sparse_csr_tensor(
            t.as_tensor(A.indptr.astype(np.int64), device=self.dev),
            t.as_tensor(A.indices.astype(np.int64), device=self.dev),
            t.as_tensor(np.ascontiguousarray(A.data), device=self.dev),
            size=A.shape)

    # ----------------------------------------------------------------- system --
    def system(self, E_pdl, active, ksec, K_N, gref, f_ext_reduced):
        """One nonlinear iteration's scaled system.

        Returns (U, bs, dg): U the upper triangle as a host CSR whose index arrays
        are the SAME objects every call (so PARDISO sees an unchanged pattern and
        keeps its symbolic factorisation), bs the scaled rhs, dg the scaling.
        """
        t = self.torch
        dev = self.dev
        E = t.as_tensor(np.ascontiguousarray(E_pdl), device=dev)
        m = t.as_tensor(active.astype(np.float64), device=dev)
        ks = t.as_tensor(np.ascontiguousarray(ksec), device=dev)

        data = self.base.clone()
        self.scP.add_modulus(data, E)
        Kc = (K_N*self.Kn_unit + ks[:, None, None]*self.Kt_unit)*m[:, None, None]
        self.scC.add_into(data, Kc.reshape(-1))

        Kg = t.sparse_csr_tensor(self.g_ptr, self.g_cols, data, size=(self.ndof, self.ndof))
        A = t.sparse.mm(t.sparse.mm(self.Pt, Kg), self.P)
        vals = A.values()

        if self._A_pattern is None:
            self._build_A_pattern(A)
        rows, cols, up, diag = self._A_pattern

        # contact force, then the reduced rhs
        fc = t.zeros(self.ndof, dtype=t.float64, device=dev)
        gr = t.as_tensor(np.ascontiguousarray(gref), device=dev)
        contrib = ((-K_N*gr*m)[:, None]*self.a).reshape(-1)
        fc.index_add_(0, self.c_dofs.reshape(-1), contrib)
        b = t.as_tensor(np.ascontiguousarray(f_ext_reduced), device=dev) \
            + t.mv(self.Pt, fc)

        # symmetric Jacobi scaling, then the upper triangle PARDISO wants
        dg = t.sqrt(t.clamp(vals[diag], min=0.0))
        dg = t.where(dg > 0, dg, t.ones_like(dg))
        scaled = vals/(dg[rows]*dg[cols])
        self._U.data[:] = scaled[up].cpu().numpy()
        dg_h = dg.cpu().numpy()
        return self._U, (b/dg).cpu().numpy(), dg_h

    def _build_A_pattern(self, A):
        """Row indices, upper-triangle selection and diagonal positions of the
        reduced matrix -- all fixed for the step, so computed on first use."""
        t = self.torch
        crow, cols = A.crow_indices(), A.col_indices()
        n = A.shape[0]
        rows = t.repeat_interleave(t.arange(n, device=self.dev), crow[1:]-crow[:-1])
        up = t.nonzero(cols >= rows, as_tuple=True)[0]
        diag = t.nonzero(cols == rows, as_tuple=True)[0]
        if diag.numel() != n:
            raise RuntimeError(f'reduced matrix is missing {n-diag.numel()} diagonal '
                               f'entries; cannot scale')
        u_rows, u_cols = rows[up], cols[up]
        indptr = np.zeros(n+1, dtype=np.int32)
        np.cumsum(t.bincount(u_rows, minlength=n).cpu().numpy(), out=indptr[1:])
        self._U = csr_matrix((np.zeros(up.numel()), u_cols.cpu().numpy().astype(np.int32),
                              indptr), shape=A.shape)
        self._A_pattern = (rows, cols, up, diag)

    def close(self):
        for k in ('base', 'Bp', 'pdl_tets', 'R', 'a', 'Pt_tan',
                  'Kt_unit', 'Kn_unit'):
            if hasattr(self, k):
                delattr(self, k)
        self._A_pattern = None
        try:
            self.torch.cuda.empty_cache()
        except Exception:
            pass


# The pattern is fixed for a staging step, so the assembler is cached and reused
# across the step's two solves.  `token` is the pairing array the step was built
# for; a new step brings a new array and rebuilds.
_CACHE = {'token': None, 'sys': None}


def for_step(token, ndof, dofs_contact, aligner_dofs, pdl_dofs, bone_dofs=None):
    c = _CACHE
    if c['sys'] is not None and c['token'] is token:
        return c['sys']
    if c['sys'] is not None:
        # Release the previous system entirely first: while it is still referenced its
        # projections and scatter maps stay on the device, so a step change needs twice the memory.
        import gc
        old, t = c['sys'], c['sys'].torch
        c['sys'] = c['token'] = None
        old.close()
        del old
        gc.collect()
        t.cuda.empty_cache()
    c['sys'] = GpuSystem(ndof, dofs_contact, aligner_dofs, pdl_dofs, bone_dofs)
    c['token'] = token
    return c['sys']
