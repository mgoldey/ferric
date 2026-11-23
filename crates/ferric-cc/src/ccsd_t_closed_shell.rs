//! Spin-adapted (closed-shell, RHF-reference) CCSD(T) perturbative triples.
//!
//! A **spatial-orbital** reformulation of [`crate::ccsd_t::ccsd_t`]. Same
//! physics, same non-iterative O(N^7) triples energy, but every index runs
//! over `no`/`nv` spatial orbitals instead of `2no`/`2nv` spin orbitals — the
//! spin structure is summed analytically into the `4W + W(bca) + W(cab)
//! − 2W(cba) − 2W(acb) − 2W(bac)` coefficient pattern below.
//!
//! # Formulation and source
//!
//! Transcribed from the restricted (T) algorithm of **Rendell, Lee &
//! Komornicki, Chem. Phys. Lett. 178, 462 (1991)** / **Lee & Rendell, JCP 94,
//! 6229 (1991)**, in the concrete index convention of PySCF's
//! `pyscf/cc/ccsd_t_slow.py` (`kernel` + `r3`), which cites JCP 94, 442
//! (1991). Like PySCF's production kernel (`lib/cc/ccsd_t.c`), the loop bands
//! over *virtual* triples `a<=b<=c` and evaluates every occupied `(i,j,k)` of
//! a triple at once, so the per-triple working set is a `[no,no,no]` block
//! that stays in cache, and the six pair permutations, `V`, `W̃` and the
//! denominator are applied in passes over that small block. The equations
//! below are written for a fixed occupied triple; the virtual banding is their
//! mirror image (see "Combinatorial factor"). The occupied and virtual
//! bandings were proven numerically equivalent in numpy against PySCF's own
//! `ccsd_t_slow.kernel` before any Rust was written (residual 7.8e-16 on a
//! random 4-occupied / 6-virtual system with fully 8-fold-symmetric RI
//! integrals), and the Rust is proven against the spin-orbital `(T)` on real
//! molecules by the tests at the bottom of this file.
//!
//! ## The equations
//!
//! All integrals are **chemist notation** `(pq|rs) = Σ_P B^P_pq B^P_rs`, the
//! same dressed RI blocks [`crate::ccsd_closed_shell`] already builds. For a
//! fixed ordered occupied triple `(i,j,k)` the *raw* (unsymmetrized) connected
//! block is
//!
//! ```text
//! w0[a,b,c] = Σ_d (ia|bd) · t2[k,j,c,d]  −  Σ_l (ia|lj) · t2[l,k,b,c]
//! ```
//!
//! (both BLAS3 GEMMs; for a fixed virtual triple they run over all occupied
//! indices at once — see `raw_w_pair`).
//! The physical `W` is the sum of `w0` over the **6 simultaneous permutations
//! of the three (occupied, virtual) PAIRS** `(i,a) (j,b) (k,c)`:
//!
//! ```text
//! W[a,b,c] = Σ_{π ∈ S3} w0(i_π, j_π, k_π)[a_π, b_π, c_π]
//! ```
//!
//! `W` is therefore invariant under any simultaneous pair relabeling — the
//! property the triple banding rests on. With the virtual triple fixed, each
//! raw block is computed for the permuted VIRTUAL labels over all occupied
//! indices, and its occupied axes are permuted back by the inverse
//! permutation before accumulation (see `triple_energy`).
//!
//! The disconnected (T1) piece adds the singles×integral terms:
//!
//! ```text
//! V[a,b,c] = W[a,b,c] + (jb|kc)·t1[i,a] + (ia|kc)·t1[j,b] + (ia|jb)·t1[k,c]
//! ```
//!
//! (The `fock[v,o]·t2` piece of PySCF's `get_v` vanishes for canonical HF
//! orbitals, exactly as in the spin-orbital sibling.) The spin-summed
//! contraction weight is
//!
//! ```text
//! W̃[a,b,c] = 4·W[a,b,c] + W[b,c,a] + W[c,a,b]
//!            − 2·W[c,b,a] − 2·W[a,c,b] − 2·W[b,a,c]
//! ```
//!
//! (PySCF's `r3`; by pair invariance the same pattern acts on the occupied
//! axes of a fixed virtual triple, which is how the kernel applies it) and
//! finally
//!
//! ```text
//! D[a,b,c] = ε_i + ε_j + ε_k − ε_a − ε_b − ε_c
//! E_(T)    = (1/3) Σ_{a<=b<=c} m(a,b,c) · Σ_{i,j,k} W̃·V / D
//! ```
//!
//! # Combinatorial factor — DERIVED, not assumed
//!
//! This is the trap the spin-orbital module's long comment warns about, and
//! the closed-shell answer is **different from its `/6`** in two independent
//! ways. Write `f(i,j,k,a,b,c) = W̃·V/D`.
//!
//! 1. **The overall divisor is 3, not 6 and not 36.** The unrestricted
//!    reference sum — every one of `no³·nv³` index combinations, no
//!    restriction anywhere — equals `3 · E_(T)`. This was measured directly:
//!    `F.sum()/3` reproduced PySCF's `ccsd_t_slow.kernel` to 7.8e-16
//!    (scratch `step4.py`). There is no 1/36: the spin summation that produced
//!    `W̃`'s `4/1/1/−2/−2/−2` coefficients has already absorbed most of the
//!    spin-orbital antisymmetrizer's redundancy, leaving a factor 3.
//!
//! 2. **Repeated occupied indices contribute and MUST be included.** In the
//!    spin-orbital formulation `i=j` makes the antisymmetrized amplitude
//!    vanish, so `i<j<k` loses nothing. Here `i`, `j`, `k` label *spatial*
//!    orbitals, each holding two electrons, so `i=j` and even `i=j=k` are
//!    physically allowed and numerically large: on the scratch test system the
//!    repeated-index terms were `−0.988` of a `−1.717` total — **58% of the
//!    energy**. A naive copy of the spin-orbital `i<j<k` banding would
//!    silently drop them.
//!
//! Since `f` *is* invariant under simultaneous relabeling of `(i,j,k)` with
//! `(a,b,c)` (verified elementwise in scratch: all 6 orderings of a sample
//! index set agreed to ~1e-18), the sum over the orderings of a virtual triple
//! equals that ordering count times the sum at `a<=b<=c` with `i,j,k` over the
//! FULL axes. This module therefore bands over `a<=b<=c` with an explicit
//! multiplicity weight
//!
//! ```text
//! m(a,b,c) = 1  if a==b==c        (1 ordering)
//!          = 3  if exactly two are equal
//!          = 6  if all three distinct
//! ```
//!
//! reproducing the unrestricted sum exactly. See `triple_multiplicity`. The
//! same argument with the roles swapped gives the `i<=j<=k` banding; both
//! banded weighted sums divided by 3 reproduced PySCF to 7.8e-16 in scratch
//! (`step5.py`, `step6.py`) before this file existed. Repeated VIRTUAL indices
//! are physical for the same reason repeated occupied ones are (spatial
//! orbitals), so `a==b` and `a==b==c` are in the band with weights 3 and 1.
//!
//! # Cost vs the spin-orbital path
//!
//! Leading GEMM cost, in multiply-adds:
//!
//! - spin-orbital: `C(2no,3) ≈ (2no)³/6` occupied triples × 3 occupied
//!   permutations × `(2nv)⁴` ⇒ `~ 32 no³nv⁴`
//! - closed-shell: `~ nv³/6` virtual triples × 6 pair permutations ×
//!   `(no³·nv + no⁴)` ⇒ `~ no³nv⁴ + no⁴nv³`
//!
//! so the *ideal* FLOP ratio is ~32×. Measured times are in
//! `site/src/methods/cc.md` and the `perf_*` tests below; do not trust this
//! estimate in their place.
//!
//! # Memory and determinism
//!
//! Peak resident is three `[no,no,no]` f64 buffers (plus a `(2·no, nv)`
//! operand and three `[no,no]` slices) per in-flight `(a,b)` job
//! (`peak_job_bytes`) plus the precomputed `O(no·nv³)`-class chemist blocks. The parallel unit is a virtual pair
//! `a<=b`, which runs its `c = b..nv` triples serially; the reduction is the
//! spin-orbital sibling's chunk-then-collect-then-serial-fold idiom: band
//! width is a pure function of `(no, nv)` and the byte budget (never
//! `rayon::current_num_threads()`), every GEMM runs on one BLAS thread, and job
//! partials are folded serially in ascending `(a,b)` order, so the result is
//! bit-identical at any thread count (asserted by
//! `thread_count_bit_identical_h2o_ccpvdz` below).

use super::{CcConfig, CcResult};
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::blas_threads::with_blas_threads;
use ferric_integrals::operator::Operator;
use ferric_mp2::mo_transform::{transform_3center_oo, transform_3center_ov, transform_3center_vv};
use ferric_mp2::rimp2::{active_occ, cholesky_inverse_sqrt};
use ferric_mp2::spinorbital::build_b;
use ferric_scf::ScfResult;
use ferric_tensors::{einsum, Axis};
use ndarray::linalg::general_mat_mul;
use ndarray::{s, Array2, Array3, Array4};
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// The 6 permutations of `(0,1,2)` in a fixed, deterministic order, grouped
/// in pairs that agree in their LAST position. Both orderings of a pair put
/// the same virtual label third, so their raw blocks share the `t2x` operand
/// and are computed by one stacked GEMM (see `raw_w_pair`).
const PERM_PAIRS: [[[usize; 3]; 2]; 3] = [
    [[0, 1, 2], [1, 0, 2]],
    [[0, 2, 1], [2, 0, 1]],
    [[1, 2, 0], [2, 1, 0]],
];

/// Number of distinct orderings of the virtual triple `(a,b,c)` given
/// `a <= b <= c`.
///
/// This is the weight that turns the banded `a<=b<=c` loop back into the
/// unrestricted `Σ_{a,b,c}` sum the `1/3` divisor is defined against.
/// Repeated spatial indices are physical (see the module doc) — dropping
/// them, or weighting every triple by 6, is a silent large error.
fn triple_multiplicity(a: usize, b: usize, c: usize) -> f64 {
    debug_assert!(a <= b && b <= c);
    if a == c {
        1.0 // a == b == c
    } else if a == b || b == c {
        3.0
    } else {
        6.0
    }
}

/// Every `a <= b` virtual pair in ascending lexicographic order: one parallel
/// job each, which then runs `c = b..nv` serially. Pure function of `nv` — the
/// determinism anchor for the banded reduction (job boundaries and the fold
/// order never depend on the thread count).
fn virtual_pairs(nv: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::with_capacity(nv * (nv + 1) / 2);
    for a in 0..nv {
        for b in a..nv {
            out.push((a, b));
        }
    }
    out
}

/// Peak bytes one in-flight `(a,b)` job holds ([`JobScratch`]): a raw block
/// pair and `W` (three `[no,no,no]`), the stacked `(2·no, nv)` term-1 operand,
/// and the three `[no,no]` singles slices.
fn peak_job_bytes(no: usize, nv: usize) -> usize {
    let no3 = no.saturating_pow(3);
    let no2 = no.saturating_pow(2);
    no3.saturating_mul(3)
        .saturating_add(no2.saturating_mul(3))
        .saturating_add(no.saturating_mul(nv).saturating_mul(2))
        .saturating_mul(8)
}

/// Bytes held by the precomputed chemist-notation blocks ([`TripleIntegrals`]),
/// all live for the whole loop, plus the one transient source block that is
/// alive while its re-layout is built.
///
/// The spatial analogue of `ccsd_t::precomputed_block_bytes`, and it closes the
/// same gap: a guard checking only the per-job working set while `ovvv` is
/// `no·nv³` would let a job pass pre-flight and OOM on the next allocation.
/// See `tests/mwe_t_guard_covers_precomputed.rs`.
///
/// Counted: `ovvv` `no·nv³`; `ovoo` `no³·nv` and its re-layout `vooo`
/// `no³·nv` (both alive while `vooo` is filled); `ovov` `no²·nv²`; the
/// re-laid amplitudes `t2x` `no²·nv²` and `t1t` `no·nv`.
fn precomputed_block_bytes(no: usize, nv: usize) -> usize {
    let ovvv = no.saturating_mul(nv.saturating_pow(3));
    let ovoo = no.saturating_pow(3).saturating_mul(nv);
    let oovv = no.saturating_pow(2).saturating_mul(nv.saturating_pow(2));
    let t1 = no.saturating_mul(nv);

    ovvv.saturating_add(ovoo.saturating_mul(2))
        .saturating_add(oovv.saturating_mul(2))
        .saturating_add(t1)
        .saturating_mul(8)
}

/// Thread-count-independent band width (in `(a,b)` jobs) from a byte budget,
/// floored at 1. Mirrors `ccsd_t::triple_chunk_len` (see that function's doc
/// and the `ferric_scf::reduce` banding pattern it cites).
fn job_chunk_len(no: usize, nv: usize, band_budget_bytes: usize) -> usize {
    let per_job = peak_job_bytes(no, nv).max(1);
    (band_budget_bytes / per_job).max(1)
}

/// The band width [`ccsd_t_closed_shell`] uses on the UNBUDGETED path, as one
/// named function of `(no, nv, budget_bytes)`.
///
/// Spatial analogue of `ccsd_t::t_band_width`; see that function for why the
/// composition is named rather than open-coded.
pub fn t_band_width(no: usize, nv: usize, budget_bytes: usize) -> usize {
    let remaining = budget_bytes.saturating_sub(precomputed_block_bytes(no, nv));
    job_chunk_len(
        no,
        nv,
        ferric_core::memory::transient_share(remaining, ferric_core::memory::Share::Half),
    )
}

/// The band width [`ccsd_t_closed_shell`] will use RIGHT NOW, given the
/// process-global pool's current state -- the exact expression the driver
/// evaluates. Spatial analogue of `ccsd_t::t_band_width_now`; see that
/// function for why the pooled branch reads the LEDGER and never live RSS,
/// and why the driver calls this instead of inlining it.
pub fn t_band_width_now(no: usize, nv: usize, budget_bytes: usize) -> usize {
    match ferric_core::memory::pool::global_available_bytes() {
        Some(free) => job_chunk_len(no, nv, free),
        None => t_band_width(no, nv, budget_bytes),
    }
}

/// The floor [`ccsd_t_closed_shell`] charges before it allocates.
pub fn t_floor_bytes(no: usize, nv: usize) -> usize {
    precomputed_block_bytes(no, nv).saturating_add(peak_job_bytes(no, nv))
}

/// Bytes one in-flight `(a,b)` job's working set holds.
pub fn t_per_job_bytes(no: usize, nv: usize) -> usize {
    peak_job_bytes(no, nv)
}

/// Precomputed chemist-notation integral blocks and re-laid amplitudes the
/// per-triple kernel slices.
///
/// All are `O(no·nv³)`-class or smaller — the same size class the CCSD driver
/// already carries, NOT the `O((no·nv)³)` dense-6D ceiling the streaming
/// formulation exists to avoid. Every operand a GEMM reads is a 2-D view with
/// unit inner stride, so it goes to BLAS without a copy.
struct TripleIntegrals {
    /// `(ia|bd)` at `[i, a, b, d]` (`ovvv`); for fixed `(a,b)` the `[i,d]`
    /// slice is a `(no, nv)` view with row stride `nv³`.
    ovvv: Array4<f64>,
    /// `(ia|lj)` at `[a, i, j, l]`; for fixed `a` a contiguous `(no·no, no)`.
    vooo: Array4<f64>,
    /// `(ia|jb)` at `[i, a, j, b]` (feeds `V`'s singles terms).
    ovov: Array4<f64>,
    /// `t2[k,j,c,d]` at `[c, d, j, k]`: for fixed `c` a contiguous
    /// `(nv, no·no)` (term 1), and for fixed `(b,c)` the `[k,l]` block whose
    /// transpose is `t2[l,k,b,c]` (term 2).
    t2x: Array4<f64>,
    /// `t1[i,a]` at `[a, i]`.
    t1t: Array2<f64>,
}

/// Kernel steps timed inside the triples loop, summed over rayon workers.
const KERNEL_STEPS: [&str; 3] = [
    "triples kernel [thread-s]: W raw GEMMs",
    "triples kernel [thread-s]: W pair-permutation sum",
    "triples kernel [thread-s]: V singles + r3 + energy",
];

/// Thread-seconds per kernel step for the `FERRIC_TIMING` table. Inert (no
/// clock read) when timing is off; the clock never touches the arithmetic.
struct KernelClock {
    on: bool,
    nanos: [AtomicU64; KERNEL_STEPS.len()],
}

impl KernelClock {
    fn new() -> Self {
        KernelClock {
            on: ferric_core::timing::enabled(),
            nanos: Default::default(),
        }
    }

    #[inline]
    fn time<T>(&self, step: usize, f: impl FnOnce() -> T) -> T {
        if !self.on {
            return f();
        }
        let t0 = Instant::now();
        let out = f();
        let dt = u64::try_from(t0.elapsed().as_nanos()).unwrap_or(u64::MAX);
        self.nanos[step].fetch_add(dt, Ordering::Relaxed);
        out
    }

    fn report_into(&self, phases: &mut ferric_core::timing::PhaseAccumulator) {
        for (label, n) in KERNEL_STEPS.iter().zip(&self.nanos) {
            phases.add_overlapped(label, Duration::from_nanos(n.load(Ordering::Relaxed)));
        }
    }
}

/// `rr[h][i,j,k] = w0(i,j,k)[x_h, y_h, z]` for the two ordered virtual triples
/// `(x0, y0, z)` and `(x1, y1, z)`, over every occupied `(i,j,k)`, where
/// `w0(i,j,k)[a,b,c] = Σ_d (ia|bd)·t2[k,j,c,d] − Σ_l (ia|lj)·t2[l,k,b,c]`.
///
/// Both terms are BLAS3 GEMMs writing straight into `rr`:
/// - term 1, both blocks at once: `[ovvv[:,x0,y0,:]; ovvv[:,x1,y1,:]]`
///   `(2·no, d)` times the shared `t2x[z]` `(d, jk)` gives `(h·i, jk)`;
/// - term 2, per block: `vooo[x_h]` `(ij, l)` times `t2x[y_h,z]ᵀ` `(l, k)`
///   gives `(ij, k)`.
fn raw_w_pair(
    ints: &TripleIntegrals,
    no: usize,
    nv: usize,
    xy: [(usize, usize); 2],
    z: usize,
    a_stack: &mut Array2<f64>,
    rr: &mut Array4<f64>,
) {
    for (h, &(x, y)) in xy.iter().enumerate() {
        a_stack
            .slice_mut(s![h * no..(h + 1) * no, ..])
            .assign(&ints.ovvv.slice(s![.., x, y, ..]));
    }
    let t = ints
        .t2x
        .index_axis(ndarray::Axis(0), z)
        .into_shape_with_order((nv, no * no))
        .expect("t2x[z] is contiguous");
    let mut r1 = rr
        .view_mut()
        .into_shape_with_order((2 * no, no * no))
        .expect("raw pair is contiguous");
    general_mat_mul(1.0, &*a_stack, &t, 0.0, &mut r1);

    for (h, &(x, y)) in xy.iter().enumerate() {
        let hm = ints
            .vooo
            .index_axis(ndarray::Axis(0), x)
            .into_shape_with_order((no * no, no))
            .expect("vooo[x] is contiguous");
        let u = ints.t2x.slice(s![y, z, .., ..]); // [k, l]
        let mut r2 = rr
            .index_axis_mut(ndarray::Axis(0), h)
            .into_shape_with_order((no * no, no))
            .expect("raw block is contiguous");
        general_mat_mul(-1.0, &hm, &u.t(), 1.0, &mut r2);
    }
}

/// Per-job scratch: a raw block pair, its stacked term-1 left operand, `W`,
/// and the three singles slices. Allocated once per `(a,b)` job and reused for
/// every `c`.
struct JobScratch {
    rr: Array4<f64>,
    a_stack: Array2<f64>,
    w: Array3<f64>,
    s_bc: Array2<f64>,
    s_ac: Array2<f64>,
    s_ab: Array2<f64>,
}

impl JobScratch {
    fn new(no: usize, nv: usize) -> Self {
        JobScratch {
            rr: Array4::zeros((2, no, no, no)),
            a_stack: Array2::zeros((2 * no, nv)),
            w: Array3::zeros((no, no, no)),
            s_bc: Array2::zeros((no, no)),
            s_ac: Array2::zeros((no, no)),
            s_ab: Array2::zeros((no, no)),
        }
    }
}

/// Shared read-only inputs of the triples kernel.
struct KernelInputs<'a> {
    ints: &'a TripleIntegrals,
    /// `ε_i + ε_j + ε_k` at flat `[i,j,k]`.
    eijk: &'a [f64],
    ev: &'a [f64],
    no: usize,
    nv: usize,
    clock: &'a KernelClock,
}

/// `Σ_{i,j,k} W̃[i,j,k]·V[i,j,k] / D[i,j,k]` for one virtual triple `(a,b,c)`.
///
/// `W[i,j,k] = Σ_{π ∈ S3} w0(π(i,j,k))[π(a,b,c)]` (pairs permuted together):
/// for permutation `π` the raw block of the permuted VIRTUAL triple is indexed
/// by the permuted occupied labels, so its axes are permuted back by `π⁻¹`
/// before accumulation. Then, in one fused pass over `(i,j,k)`,
///
/// ```text
/// V  = W + (jb|kc)·t1[i,a] + (ia|kc)·t1[j,b] + (ia|jb)·t1[k,c]
/// W̃  = 4·W[ijk] + W[kij] + W[jki] − 2·W[kji] − 2·W[ikj] − 2·W[jik]
/// ```
///
/// `W̃` is the module doc's `r3` pattern moved from the virtual axes to the
/// occupied ones: by pair invariance `W(ijk)[bca] = W(kij)[abc]`,
/// `W(ijk)[cab] = W(jki)[abc]`, and each transposition maps to itself.
fn triple_energy(
    k: &KernelInputs<'_>,
    sc: &mut JobScratch,
    (a, b, c): (usize, usize, usize),
) -> f64 {
    let (no, nv, ints) = (k.no, k.nv, k.ints);
    let virt = [a, b, c];
    sc.w.fill(0.0);
    for pair in PERM_PAIRS {
        let xy = pair.map(|p| (virt[p[0]], virt[p[1]]));
        let z = virt[pair[0][2]];
        k.clock.time(0, || {
            raw_w_pair(ints, no, nv, xy, z, &mut sc.a_stack, &mut sc.rr)
        });
        for (h, p) in pair.iter().enumerate() {
            // inverse permutation: inv[p[t]] = t
            let mut inv = [0usize; 3];
            for (t, &pt) in p.iter().enumerate() {
                inv[pt] = t;
            }
            k.clock.time(1, || {
                sc.w += &sc.rr.index_axis(ndarray::Axis(0), h).permuted_axes(inv)
            });
        }
    }

    k.clock.time(2, || {
        // (jb|kc), (ia|kc), (ia|jb) as [no,no] slices of ovov[i,a,j,b].
        sc.s_bc.assign(&ints.ovov.slice(s![.., b, .., c]));
        sc.s_ac.assign(&ints.ovov.slice(s![.., a, .., c]));
        sc.s_ab.assign(&ints.ovov.slice(s![.., a, .., b]));
        let w = sc.w.as_slice().expect("W is contiguous");
        let s_bc = sc.s_bc.as_slice().expect("contiguous");
        let s_ac = sc.s_ac.as_slice().expect("contiguous");
        let s_ab = sc.s_ab.as_slice().expect("contiguous");
        let t1a = ints.t1t.row(a);
        let t1b = ints.t1t.row(b);
        let t1c = ints.t1t.row(c);
        let e_abc = k.ev[a] + k.ev[b] + k.ev[c];
        let nn = no * no;
        let mut partial = 0.0f64;
        for i in 0..no {
            for j in 0..no {
                for kk in 0..no {
                    let n = i * nn + j * no + kk;
                    let w_ijk = w[n];
                    let wt = 4.0 * w_ijk + w[kk * nn + i * no + j] + w[j * nn + kk * no + i]
                        - 2.0
                            * (w[kk * nn + j * no + i]
                                + w[i * nn + kk * no + j]
                                + w[j * nn + i * no + kk]);
                    let v = w_ijk
                        + s_bc[j * no + kk] * t1a[i]
                        + s_ac[i * no + kk] * t1b[j]
                        + s_ab[i * no + j] * t1c[kk];
                    partial += wt * v / (k.eijk[n] - e_abc);
                }
            }
        }
        partial
    })
}

/// `Σ_{c=b}^{nv-1} m(a,b,c) · triple_energy(a,b,c)` for one `(a,b)` job, `c`
/// ascending — the unit of parallel work and of the deterministic fold.
fn pair_job_energy(k: &KernelInputs<'_>, (a, b): (usize, usize)) -> f64 {
    let mut sc = JobScratch::new(k.no, k.nv);
    let mut sum = 0.0f64;
    for c in b..k.nv {
        sum += triple_multiplicity(a, b, c) * triple_energy(k, &mut sc, (a, b, c));
    }
    sum
}

/// Compute the spin-adapted (closed-shell) CCSD(T) perturbative-triples
/// correction.
///
/// Returns `E_(T)`, to be added to the CCSD correlation energy. `cc` must
/// carry **spatial-orbital** amplitudes shaped `[no,nv]` / `[no,no,nv,nv]` —
/// i.e. the output of [`crate::ccsd_closed_shell::ccsd_closed_shell`], NOT the
/// spin-orbital [`crate::ccsd::ccsd`] (whose amplitudes are `[2no,...]` and
/// will be rejected by the shape check below).
///
/// Closed-shell RHF references only; open-shell needs the spin-orbital
/// [`crate::ccsd_t::ccsd_t`], which remains the reference oracle and is
/// untouched.
pub fn ccsd_t_closed_shell(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    cc: &CcResult,
    cfg: &CcConfig,
) -> Result<f64, FerricError> {
    let t1 = cc.t1.as_ref().ok_or_else(|| {
        FerricError::General("closed-shell CCSD(T) requires T1 amplitudes".into())
    })?;

    let (nocc_total, nmo) = crate::helpers::closed_shell_counts(mol, rhf)?;
    let first_occ = cfg.frozen_core;
    let no = active_occ(nocc_total, first_occ)?;
    let nv = nmo - nocc_total;

    // Reject spin-orbital amplitudes loudly rather than computing a wrong
    // number from a mis-shaped tensor. (t1 is [no,nv], t2 is [no,no,nv,nv].)
    if t1.shape() != [no, nv] || cc.t2.shape() != [no, no, nv, nv] {
        return Err(FerricError::General(format!(
            "closed-shell CCSD(T) expects SPATIAL amplitudes t1{:?} t2{:?}, got t1{:?} t2{:?} \
             — pass the ccsd_closed_shell result, not the spin-orbital ccsd one",
            [no, nv],
            [no, no, nv, nv],
            t1.shape(),
            cc.t2.shape()
        )));
    }

    // Fewer than 1 occupied or 1 virtual spatial orbital leaves no triple at
    // all. NOTE: unlike the spin-orbital path there is NO `no < 3` shortcut —
    // repeated spatial indices (i==j==k) are physical here, so a 1-occupied
    // system like H2 still has (i,i,i) triples. They evaluate to a
    // near-vanishing (T) on H2 for physical reasons, not by construction.
    if no == 0 || nv == 0 {
        return Ok(0.0);
    }

    let peak_job = peak_job_bytes(no, nv);
    let precomputed = precomputed_block_bytes(no, nv);
    let budget = ferric_core::memory::resolve_budget_bytes(cfg.memory_budget_bytes);
    // Precomputed blocks are allocated once and shared; the per-job set is
    // per rayon worker. Charge one per-job set here -- the band width below
    // is sized from the REMAINING budget, so concurrency is bounded separately
    // and must not be double-counted into this floor.
    let floor = precomputed.saturating_add(peak_job);
    ferric_core::memory::check_alloc(
        &format!(
            "closed-shell CCSD(T) precomputed blocks + one per-job block \
             (no={no}, nv={nv} spatial)"
        ),
        floor,
        budget,
    )?;
    // HARD charge, held to the end of the function -- which is where the
    // precomputed blocks die. See the spin-orbital sibling in ccsd_t.rs for
    // why this is hard rather than soft: nothing streams these blocks, so a
    // soft gate's None branch would have no fallback to take, and a soft gate
    // whose None branch does not stream is a lie.
    let _floor_charge = ferric_core::memory::pool::reserve_global(
        &format!("closed-shell CCSD(T) precomputed blocks (no={no}, nv={nv})"),
        floor,
    )?;

    let eps = rhf.eps_r();
    let c = rhf.mos_r();
    let c_occ = c.slice(s![.., first_occ..nocc_total]).to_owned();
    let c_vir = c.slice(s![.., nocc_total..nmo]).to_owned();

    // Dressed RI 3-index MO blocks — identical construction to the CCSD
    // drivers, so the RI error cancels between CCSD and (T).
    // FERRIC_TIMING phase table (inert otherwise); reports on every exit.
    let mut phases =
        ferric_core::timing::PhaseAccumulator::new().report_on_drop("closed-shell (T)");
    let tok = phases.begin("setup: metric + AO 3-index");
    let v2c = ferric_integrals::threeindex::coulomb_metric_2c(op, dfbs)?;
    let v_inv_sqrt = cholesky_inverse_sqrt(&v2c)?;
    let eri3_ao = ferric_integrals::threeindex::eri3_tensor(op, obs, dfbs)?;
    phases.stop(tok);
    use Axis::{O, V};
    let tok = phases.begin("setup: RI B blocks");
    let b_ov = build_b(
        &transform_3center_ov(&eri3_ao, &c_occ, &c_vir),
        &v_inv_sqrt,
        O,
        V,
    );
    let b_oo = build_b(&transform_3center_oo(&eri3_ao, &c_occ), &v_inv_sqrt, O, O);
    let b_vv = build_b(&transform_3center_vv(&eri3_ao, &c_vir), &v_inv_sqrt, V, V);
    drop(eri3_ao);
    phases.stop(tok);

    // Chemist blocks (pq|rs) = Σ_P B^P_pq B^P_rs, and the GEMM-ready
    // re-layouts of ovoo and t2.
    let tok = phases.begin("setup: chemist blocks + re-layouts");
    let ovvv: Array4<f64> = einsum!("Pia,Pbd->iabd", &b_ov, &b_vv)
        .into_dimensionality()
        .expect("ovvv 4D");
    let ovoo: Array4<f64> = einsum!("Pia,Plj->ialj", &b_ov, &b_oo)
        .into_dimensionality()
        .expect("ovoo 4D");
    let ovov: Array4<f64> = einsum!("Pia,Pjb->iajb", &b_ov, &b_ov)
        .into_dimensionality()
        .expect("ovov 4D");
    let vooo = Array4::from_shape_fn((nv, no, no, no), |(a, i, j, l)| ovoo[[i, a, l, j]]);
    drop(ovoo);
    let t2 = &cc.t2;
    let t2x = Array4::from_shape_fn((nv, nv, no, no), |(c, d, j, k)| t2[[k, j, c, d]]);
    let t1t = Array2::from_shape_fn((nv, no), |(a, i)| t1[[i, a]]);
    let ints = TripleIntegrals {
        ovvv,
        vooo,
        ovov,
        t2x,
        t1t,
    };
    phases.stop(tok);

    let eo: Vec<f64> = (0..no).map(|i| eps[first_occ + i]).collect();
    let ev: Vec<f64> = (0..nv).map(|a| eps[nocc_total + a]).collect();
    let mut eijk = Vec::with_capacity(no * no * no);
    for i in 0..no {
        for j in 0..no {
            for k in 0..no {
                eijk.push(eo[i] + eo[j] + eo[k]);
            }
        }
    }

    // Banded parallel reduction over (a,b) jobs, a<=b, each running c=b..nv.
    // Band width is a pure function of (no, nv) and the byte budget (never the
    // thread count); job partials are folded serially in ascending (a,b)
    // order, so `et` is bit-identical at any RAYON_NUM_THREADS. See the
    // module doc and the spin-orbital sibling's matching comment block.
    let jobs = virtual_pairs(nv);
    // Band from what is LEFT after the precomputed blocks, not the full budget
    // (they are already resident here). Mirrors the spin-orbital sibling.
    //
    // The byte number is the pool LEDGER when a pool is installed (capacity
    // minus live reservations, which already includes this function's own
    // `_floor_charge`), and the resolved ceiling minus the precomputed blocks
    // otherwise. A ledger quantity is deterministic given the pool capacity;
    // an RSS quantity would not be, and that distinction is the KS grid
    // batch-width defect. See the spin-orbital sibling for the full argument,
    // and `mwe_t_band_width_is_not_an_energy_knob.rs` for the pin that the
    // width does not move `et` either way.
    // The `Share::Half` appears only on the UNPOOLED branch. On the pooled
    // branch `available_bytes()` is already net of every live reservation,
    // including this function's own `_floor_charge`, so halving it would both
    // narrow the band for no reason AND make the soft gate below unreachable
    // by construction (a width derived from `available/2` can never fail to
    // fit in `available`). See the spin-orbital sibling.
    let remaining = budget.saturating_sub(precomputed);
    // The None branch calls [`t_band_width`], not an inline copy -- see the
    // spin-orbital sibling for why the single implementation matters. The
    // width is capped at the job count: a wider band holds no more jobs, so
    // charging for it would reserve bytes nothing can use.
    let chunk_len = t_band_width_now(no, nv, budget).min(jobs.len()).max(1);
    // SOFT charge: narrowing the band is a real fallback (the None branch
    // below actually re-sizes and runs), so this must never refuse a job.
    let (chunk_len, _band_charge) = match ferric_core::memory::pool::try_reserve_global(
        &format!("closed-shell CCSD(T) job band (width {chunk_len}, no={no})"),
        chunk_len.saturating_mul(peak_job),
    ) {
        Some(g) => (chunk_len, g),
        None => {
            let narrowed = match ferric_core::memory::pool::global() {
                Some(p) => p.fit_units(peak_job.max(1)).min(chunk_len).max(1),
                None => 1,
            };
            let g = ferric_core::memory::pool::try_reserve_global(
                &format!("closed-shell CCSD(T) job band (narrowed to {narrowed}, no={no})"),
                narrowed.saturating_mul(peak_job),
            )
            .unwrap_or_else(|| {
                ferric_core::memory::pool::Reservation::inert(
                    "closed-shell CCSD(T) job band (width 1)",
                )
            });
            (narrowed, g)
        }
    };
    // Same floored-band warning as the spin-orbital sibling: width 1 means the
    // budget could not fund a two-job band, so the loop is effectively
    // serial — actionable (raise [memory] budget_gb), not a hang.
    if chunk_len == 1 && jobs.len() > 1 {
        eprintln!(
            "ccsd_t_closed_shell: memory budget floors the triple band at 1 \
             (no={no}, {} (a,b) jobs, {:.2} GiB remaining after precomputed \
             blocks) — the (T) triples loop will run effectively serially. \
             Raise the memory budget to widen the band.",
            jobs.len(),
            remaining as f64 / (1024.0 * 1024.0 * 1024.0),
        );
    }
    let clock = KernelClock::new();
    let kin = KernelInputs {
        ints: &ints,
        eijk: &eijk,
        ev: &ev,
        no,
        nv,
        clock: &clock,
    };
    let mut et = 0.0f64;
    let tok = phases.begin("triples loop");
    for chunk in jobs.chunks(chunk_len) {
        // BLAS pinned to 1 EXPLICITLY: the kernel's GEMMs run inside this rayon
        // region, and `opt_in_blas_threads`'s self-guard only fires when called
        // FROM a worker — this wrap runs on the caller thread, so it would pass
        // the user's `FERRIC_BLAS_THREADS` straight through. Mirrors the
        // spin-orbital sibling in ccsd_t.rs.
        let partials: Vec<f64> = with_blas_threads(1, || {
            chunk
                .par_iter()
                .map(|&ab| pair_job_energy(&kin, ab))
                .collect()
        });
        for p in partials {
            et += p;
        }
    }
    phases.stop(tok);
    clock.report_into(&mut phases);

    // Divisor 3 (NOT 6, NOT 36) — derived and measured, see the module doc's
    // "Combinatorial factor" section. The weighted a<=b<=c band above equals
    // the unrestricted Σ_{i,j,k} Σ_{a,b,c} sum, which is exactly 3·E_(T).
    Ok(et / 3.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ccsd_closed_shell::{ccsd_closed_shell, expand_amplitudes_to_spin_orbital};
    use crate::ccsd_t::ccsd_t;
    use ferric_core::basis;
    use ferric_core::parallel::ParallelContext;
    use ferric_scf::rhf::{solve_rhf, RhfConfig};
    use ferric_scf::screening::SchwarzBounds;

    const H2O: &str = "3\n\nO 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692\n";
    const H2: &str = "2\nH2\nH 0 0 0\nH 0 0 0.74\n";

    struct Setup {
        mol: Molecule,
        obs: PreparedBasis,
        dfbs: PreparedBasis,
        op: Operator,
        rhf: ScfResult,
    }

    fn setup(xyz: &str, obs_name: &str) -> Setup {
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled(obs_name).unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("def2-qzvpp-rifit").unwrap()).unwrap();
        let op = Operator::coulomb();
        let ctx = ParallelContext::default();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(
            &ctx,
            &mol,
            &obs,
            op,
            &bounds,
            &RhfConfig::default().with_energy_conv(1e-11),
        )
        .unwrap();
        Setup {
            mol,
            obs,
            dfbs,
            op,
            rhf,
        }
    }

    fn cc_cfg() -> CcConfig {
        CcConfig {
            frozen_core: 0,
            max_iter: 100,
            energy_conv: 1e-10,
            ..Default::default()
        }
    }

    /// Run BOTH (T) implementations from the SAME converged spin-adapted CCSD
    /// amplitudes, so any difference is the (T) formulation itself and not the
    /// two CCSD solvers' differing RI accumulation.
    fn both_t(s: &Setup) -> (f64, f64) {
        both_t_with(s, cc_cfg())
    }

    /// [`both_t`] with a caller-supplied config, so the frozen-core path can be
    /// held against the same spin-orbital oracle.
    fn both_t_with(s: &Setup, cfg: CcConfig) -> (f64, f64) {
        let r_cs = ccsd_closed_shell(&s.mol, &s.obs, &s.dfbs, s.op, &s.rhf, &cfg).unwrap();
        let t_new =
            ccsd_t_closed_shell(&s.mol, &s.obs, &s.dfbs, s.op, &s.rhf, &r_cs, &cfg).unwrap();

        let (t1_so, t2_so) = expand_amplitudes_to_spin_orbital(
            &r_cs.t1.as_ref().unwrap().clone().into_dyn(),
            &r_cs.t2.clone().into_dyn(),
        );
        let cc_so = CcResult {
            correlation_energy: r_cs.correlation_energy,
            t1: Some(t1_so.into_dimensionality().unwrap()),
            t2: t2_so.into_dimensionality().unwrap(),
            ladder_route: None,
        };
        let t_old = ccsd_t(&s.mol, &s.obs, &s.dfbs, s.op, &s.rhf, &cc_so, &cfg).unwrap();
        (t_old, t_new)
    }

    /// **The decisive correctness gate.** H2O/cc-pVDZ (5 occupied, 19 virtual
    /// spatial orbitals) exercises every branch that matters: all three
    /// multiplicity classes (i==j==k, one pair equal, all distinct), the
    /// exchange/`ovoo` term, and the full `4/1/1/-2/-2/-2` `r3` pattern.
    ///
    /// Both implementations are fed the SAME spin-adapted amplitudes, so a
    /// disagreement is a (T)-formulation bug, not an amplitude difference.
    #[test]
    fn closed_shell_t_matches_spin_orbital_t_h2o_ccpvdz() {
        let s = setup(H2O, "cc-pvdz");
        let (t_old, t_new) = both_t(&s);
        let diff = (t_old - t_new).abs();
        println!(
            "H2O/cc-pVDZ (T): spin-orbital = {t_old:.12}, closed-shell = {t_new:.12}, diff = {diff:.3e}"
        );
        // Guard against a vacuous pass: (T) must be a real, negative number.
        assert!(
            t_new < -1e-4,
            "(T) = {t_new:.12} is not a plausible triples energy"
        );
        assert!(
            diff < 1e-9,
            "closed-shell (T) = {t_new:.12} disagrees with spin-orbital (T) = {t_old:.12} by {diff:.3e}"
        );
    }

    /// **Frozen core.** Production CCSD(T) almost always freezes core orbitals,
    /// and `first_occ = cfg.frozen_core` shifts every occupied index AND the
    /// multiplicity bookkeeping — an off-by-one there would be silent, since the
    /// energy would still look like a plausible triples correction.
    ///
    /// Held against the same spin-orbital oracle as the all-electron gate. Note
    /// the two implementations index the frozen window differently (spatial
    /// `first_occ..nocc_total` vs spin-orbital `2*first_occ..`), so this is a
    /// real cross-check, not a tautology.
    ///
    /// Added on review of the closed-shell (T) work: the original submission
    /// noted frozen_core as compiled-but-untested.
    #[test]
    fn closed_shell_t_matches_spin_orbital_t_with_frozen_core() {
        let s = setup(H2O, "cc-pvdz");
        // Water has 5 occupied spatial orbitals; freeze the O 1s.
        let cfg = CcConfig {
            frozen_core: 1,
            max_iter: 100,
            energy_conv: 1e-10,
            ..Default::default()
        };
        let (t_old, t_new) = both_t_with(&s, cfg);
        let diff = (t_old - t_new).abs();
        println!(
            "H2O/cc-pVDZ frozen_core=1 (T): spin-orbital = {t_old:.12}, \
             closed-shell = {t_new:.12}, diff = {diff:.3e}"
        );
        assert!(
            t_new < -1e-4,
            "(T) = {t_new:.12} is not a plausible triples energy"
        );
        assert!(
            diff < 1e-9,
            "frozen-core closed-shell (T) = {t_new:.12} disagrees with spin-orbital \
             (T) = {t_old:.12} by {diff:.3e}"
        );

        // Freezing a core orbital must actually CHANGE the answer — otherwise
        // this test would pass even if frozen_core were silently ignored.
        let (_, t_all) = both_t(&s);
        assert!(
            (t_all - t_new).abs() > 1e-6,
            "frozen-core (T) ({t_new:.12}) is indistinguishable from all-electron \
             ({t_all:.12}) — frozen_core may be ignored entirely"
        );
    }

    /// Second system, smaller basis — cheap enough to run routinely and still
    /// exercises repeated-index triples (H2O/STO-3G has 5 occupied orbitals).
    #[test]
    fn closed_shell_t_matches_spin_orbital_t_h2o_sto3g() {
        let s = setup(H2O, "sto-3g");
        let (t_old, t_new) = both_t(&s);
        let diff = (t_old - t_new).abs();
        println!(
            "H2O/STO-3G (T): spin-orbital = {t_old:.12}, closed-shell = {t_new:.12}, diff = {diff:.3e}"
        );
        assert!(
            t_new < -1e-6,
            "(T) = {t_new:.12} is not a plausible triples energy"
        );
        assert!(diff < 1e-10, "closed-shell (T) disagrees by {diff:.3e}");
    }

    /// Absolute anchor against the external PySCF value already pinned for the
    /// spin-orbital path: H2O/cc-pVDZ (T) = -0.0030587091. ferric's RI floor
    /// against exact-integral PySCF is ~1e-6 Ha here, so 1e-4 is the honest
    /// tolerance (same band the spin-orbital test uses).
    #[test]
    fn closed_shell_t_h2o_ccpvdz_matches_pyscf() {
        let s = setup(H2O, "cc-pvdz");
        let cfg = cc_cfg();
        let r_cs = ccsd_closed_shell(&s.mol, &s.obs, &s.dfbs, s.op, &s.rhf, &cfg).unwrap();
        let t = ccsd_t_closed_shell(&s.mol, &s.obs, &s.dfbs, s.op, &s.rhf, &r_cs, &cfg).unwrap();
        println!("closed-shell (T) H2O/cc-pVDZ = {t:.10} (PySCF -0.0030587091)");
        assert!(
            (t - (-0.0030587091)).abs() < 1e-4,
            "(T) = {t:.10}, expected -0.0030587091"
        );
    }

    /// H2 has a single occupied spatial orbital, so the ONLY triple is
    /// `(0,0,0)` with multiplicity 1. This is not a "no triples exist"
    /// shortcut (unlike the spin-orbital path's `no2<3` early return) — the
    /// kernel genuinely runs and must produce the physically correct
    /// near-zero. Kept as a degenerate-shape smoke test; it is explicitly NOT
    /// a correctness gate (it cannot distinguish a working kernel from a stub).
    #[test]
    fn closed_shell_t_h2_sto3g_is_small() {
        let s = setup(H2, "sto-3g");
        let cfg = cc_cfg();
        let r_cs = ccsd_closed_shell(&s.mol, &s.obs, &s.dfbs, s.op, &s.rhf, &cfg).unwrap();
        let t = ccsd_t_closed_shell(&s.mol, &s.obs, &s.dfbs, s.op, &s.rhf, &r_cs, &cfg).unwrap();
        println!("closed-shell (T) H2/STO-3G = {t:.12}");
        assert!(t.abs() < 1e-10, "H2/STO-3G (T) should be ~0, got {t}");
    }

    /// Spin-orbital amplitudes must be REJECTED, not silently mis-contracted.
    #[test]
    fn spin_orbital_amplitudes_are_rejected() {
        let s = setup(H2O, "sto-3g");
        let cfg = cc_cfg();
        let r_so = crate::ccsd::ccsd(&s.mol, &s.obs, &s.dfbs, s.op, &s.rhf, &cfg).unwrap();
        let err =
            ccsd_t_closed_shell(&s.mol, &s.obs, &s.dfbs, s.op, &s.rhf, &r_so, &cfg).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("SPATIAL"), "unexpected error: {msg}");
    }

    /// The banded parallel reduction must be bit-identical at any thread
    /// count — same requirement (and same idiom) as the spin-orbital path.
    #[test]
    fn thread_count_bit_identical_h2o_ccpvdz() {
        let s = setup(H2O, "cc-pvdz");
        let cfg = cc_cfg();
        let r_cs = ccsd_closed_shell(&s.mol, &s.obs, &s.dfbs, s.op, &s.rhf, &cfg).unwrap();
        let run = |n: usize| -> f64 {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .unwrap();
            pool.install(|| {
                ccsd_t_closed_shell(&s.mol, &s.obs, &s.dfbs, s.op, &s.rhf, &r_cs, &cfg).unwrap()
            })
        };
        let e1 = run(1);
        let e4 = run(4);
        println!("closed-shell (T) 1-thread={e1:.17e} 4-thread={e4:.17e}");
        assert_eq!(
            e1.to_bits(),
            e4.to_bits(),
            "(T) must be bit-identical across thread counts: {e1:.17e} vs {e4:.17e}"
        );
    }

    /// Multiplicity weights must reproduce the unrestricted ordering count, and
    /// the `(a,b)` jobs with their serial `c = b..nv` runs must visit every
    /// `a<=b<=c` triple exactly once. This pins the piece the module doc calls
    /// out as the closed-shell-only trap (repeated spatial indices are physical
    /// and carry >50% of the energy on the scratch test system).
    #[test]
    fn multiplicity_weights_sum_to_the_unrestricted_count() {
        for nv in 1..7usize {
            let total: f64 = band_triples(nv)
                .iter()
                .map(|&(a, b, c)| triple_multiplicity(a, b, c))
                .sum();
            assert_eq!(
                total,
                (nv * nv * nv) as f64,
                "weighted a<=b<=c band must cover all nv^3 ordered triples (nv={nv})"
            );
        }
    }

    /// Every a<=b<=c triple the driver's (a,b) jobs visit, in visiting order.
    fn band_triples(nv: usize) -> Vec<(usize, usize, usize)> {
        virtual_pairs(nv)
            .into_iter()
            .flat_map(|(a, b)| (b..nv).map(move |c| (a, b, c)))
            .collect()
    }

    #[test]
    fn band_visits_each_sorted_triple_once() {
        for nv in 1..7usize {
            let band = band_triples(nv);
            let distinct: std::collections::BTreeSet<_> = band.iter().collect();
            assert_eq!(distinct.len(), band.len());
            assert_eq!(band.len(), nv * (nv + 1) * (nv + 2) / 6);
        }
    }

    /// Interleaved A/B timing of the two (T) paths on the SAME amplitudes.
    /// Run-to-run noise on this box is ~10%, hence the repeats.
    fn perf_ab(s: &Setup, label: &str, reps: usize) {
        use std::time::Instant;
        let cfg = cc_cfg();
        let r_cs = ccsd_closed_shell(&s.mol, &s.obs, &s.dfbs, s.op, &s.rhf, &cfg).unwrap();
        let (t1_so, t2_so) = expand_amplitudes_to_spin_orbital(
            &r_cs.t1.as_ref().unwrap().clone().into_dyn(),
            &r_cs.t2.clone().into_dyn(),
        );
        let cc_so = CcResult {
            correlation_energy: r_cs.correlation_energy,
            t1: Some(t1_so.into_dimensionality().unwrap()),
            t2: t2_so.into_dimensionality().unwrap(),
            ladder_route: None,
        };
        for rep in 0..reps {
            let t0 = Instant::now();
            let e_old = ccsd_t(&s.mol, &s.obs, &s.dfbs, s.op, &s.rhf, &cc_so, &cfg).unwrap();
            let dt_old = t0.elapsed().as_secs_f64();
            let t1i = Instant::now();
            let e_new =
                ccsd_t_closed_shell(&s.mol, &s.obs, &s.dfbs, s.op, &s.rhf, &r_cs, &cfg).unwrap();
            let dt_new = t1i.elapsed().as_secs_f64();
            println!(
                "{label} rep {rep}: spin-orbital {dt_old:.2}s (E={e_old:.10}) | closed-shell {dt_new:.2}s (E={e_new:.10}) | speedup {:.2}x",
                dt_old / dt_new
            );
            assert!(
                (e_old - e_new).abs() < 1e-9,
                "{label}: the two (T) paths disagree ({e_old:.12} vs {e_new:.12})"
            );
        }
    }

    /// A/B timing demo on water/aug-cc-pVDZ. `--ignored`; run with
    /// `OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=12 cargo test -p ferric-cc
    ///  --release perf_closed_shell_t -- --ignored --nocapture`.
    #[test]
    #[ignore = "timing demo, run explicitly with --ignored --nocapture"]
    fn perf_closed_shell_t_vs_spin_orbital_h2o() {
        let s = setup(H2O, "aug-cc-pvdz");
        perf_ab(&s, "H2O/aug-cc-pVDZ", 3);
    }

    /// A/B timing demo on ethane/cc-pVDZ — the case where (T) was measured at
    /// 90% of the total CCSD(T) wall time, so the (T) speedup is very nearly
    /// the end-to-end speedup here.
    #[test]
    #[ignore = "timing demo, run explicitly with --ignored --nocapture"]
    fn perf_closed_shell_t_vs_spin_orbital_ethane() {
        let xyz_path = format!(
            "{}/../../testdata/molecules/alkane_2.xyz",
            env!("CARGO_MANIFEST_DIR")
        );
        let mol = Molecule::load_xyz(&xyz_path).unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("def2-qzvpp-rifit").unwrap()).unwrap();
        let op = Operator::coulomb();
        let ctx = ParallelContext::default();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(
            &ctx,
            &mol,
            &obs,
            op,
            &bounds,
            &RhfConfig::default().with_energy_conv(1e-11),
        )
        .unwrap();
        let s = Setup {
            mol,
            obs,
            dfbs,
            op,
            rhf,
        };
        perf_ab(&s, "ethane/cc-pVDZ", 2);
    }
}
