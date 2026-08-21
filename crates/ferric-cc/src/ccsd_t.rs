//! CCSD(T) perturbative triples correction (spin-orbital).
//!
//! The non-iterative O(N^7) triples energy from the converged CCSD T1/T2.
//! Formula and conventions transcribed from PySCF `gccsd_t_slow.kernel`
//! (JCP 98, 8718 (1993)) and cross-checked in numpy against PySCF's RHF
//! `ccsd_t()` / GCCSD(T) — interleaved spin-orbital convention (2k=α, 2k+1=β),
//! reproducing H2O/cc-pVDZ (T) = -0.0030587091 to 7e-11. Uses the canonical-HF
//! simplification `fock[v,o]=0` (so the `fock·t2` piece of the disconnected
//! term vanishes), matching [`crate::ccsd::ccsd`].
//!
//! ```text
//! W = P(i/jk) P(a/bc) [ Σ_e t2_jkae <bc||ei>  −  Σ_m t2_imbc <ma||jk> ]   (connected)
//! V = P(i/jk) P(a/bc) [ t1_ia <bc||jk> ]                                  (disconnected)
//! t3c = W / D,   t3d = V / D,   D_ijkabc = ε_i+ε_j+ε_k − ε_a−ε_b−ε_c
//! E_(T) = (1/36) Σ_ijkabc (t3c + t3d) · D_ijkabc · t3c
//!       = (1/36) Σ_ijkabc (W + V) · W / D
//! ```
//!
//! # Streaming per-triple-block formulation (DONE — 2026-07-19)
//!
//! The dense 6D formulation (materializing full `[2no,2no,2no,2nv,2nv,2nv]`
//! W/V/t3c/t3d/D tensors) is O((2no·2nv)³) memory: ~0.3-0.4 GB for
//! H2O/cc-pVDZ but ~1.35 TB for butane/cc-pVDZ — unusable past very small
//! systems. This module now loops over unique unordered occupied spin-orbital
//! triples `i<j<k` and, for each, forms only the `[2nv,2nv,2nv]` W/V/D blocks
//! (tens of MB even for a few hundred virtuals) instead of the full 6D
//! tensor. Peak resident memory is therefore bounded by that per-triple block
//! plus the precomputed intermediates below — those are genuinely
//! O(no·nv³) / O(no³·nv) / O(no²·nv²), NOT O((no·nv)³):
//!
//! - `bcei` (VVVO, `[2nv,2nv,2nv,2no]`), `majk` (OVOO, `[2no,2nv,2no,2no]`),
//!   `bcjk` (VVOO, `[2nv,2nv,2no,2no]`) are the same antisymmetrized
//!   spin-orbital integral blocks the dense path built — these are cheap to
//!   keep fully materialized (they scale as the 4th power of the system
//!   size, same class as T2 itself), and they are exactly what the
//!   per-triple contractions slice into.
//! - `t1` `[2no,2nv]`, `t2` `[2no,2no,2nv,2nv]` (the converged CCSD
//!   amplitudes) are likewise kept dense — same size class as the CCSD
//!   driver already carries.
//!
//! For a FIXED occupied triple `(i,j,k)` (any order, not just i<j<k — the
//! antisymmetrizer needs all 3 signed permutations), the per-triple raw
//! (pre-antisymmetrized) blocks are:
//!
//! ```text
//! raw_w[a,b,c] = Σ_e t2[j,k,a,e]·bcei[b,c,e,i]  −  Σ_m t2[i,m,b,c]·majk[m,a,j,k]
//! raw_v[a,b,c] = t1[i,a]·bcjk[b,c,j,k]
//! ```
//!
//! Both contractions are BLAS3 GEMMs on `[2nv,2nv]`-scale reshapes (the first
//! term reshapes `bcei[:,:,:,i]` to `(nv2·nv2, nv2)` and right-multiplies by
//! `t2[j,k,:,:]ᵀ`; the second reshapes `t2[i,:,:,:]` to `(no2, nv2·nv2)` and
//! left-multiplies by `majk[:,:,j,k]ᵀ`; `raw_v` is a plain outer product) —
//! see [`raw_w_block`] / [`raw_v_block`]. The full antisymmetrized `W`/`V`
//! blocks for a canonical triple `i0<j0<k0` are then the signed sum over the
//! 3 occupied permutations `{(i0,j0,k0):+, (j0,i0,k0):−, (k0,j0,i0):−}`
//! (exactly `P(i/jk)`, transcribed from [`p_i_jk`]) crossed with the 3
//! virtual permutations `{(a,b,c):+, (b,a,c):−, (c,b,a):−}` applied to the
//! *free* `a,b,c` axes of each raw block (exactly `P(a/bc)`, transcribed from
//! [`p_a_bc`]) — see [`triple_block`]. This is verified to reproduce the
//! OLD dense path bit-for-bit (to ~1e-10) on H2O/cc-pVDZ in
//! `streaming_matches_dense_h2o_ccpvdz` below before the dense code was
//! removed, and the two existing correctness-gate tests
//! (`test_ccsd_t_h2_sto3g_is_zero`, `test_ccsd_t_h2o_ccpvdz`) still assert
//! the identical PySCF-cross-checked values through the new streaming path.
//!
//! **Combinatorial factor (NOT the textbook 1/36 — see the worked-out comment
//! at the summation site in [`ccsd_t`]):** the streaming loop ranges `i<j<k`
//! over *unique unordered* occupied triples (visited once, not 6 times), but
//! `a,b,c` each range over the FULL `0..2nv` axis per triple, i.e. every
//! unique unordered virtual triple is visited 6 times (all its orderings).
//! Writing `f(i,j,k,a,b,c) = (t3c+t3d)·D·t3c`: `D` depends only on the index
//! *set*, not the ordering, and `t3c`/`t3d` each flip sign under any single
//! occ-swap or vir-swap — but `f` contains `t3c` twice (once directly, once
//! inside `t3c+t3d` as itself), so `f` is invariant (not sign-flipping) under
//! ANY simultaneous relabeling of `(i,j,k)` or of `(a,b,c)`. Hence every one
//! of the 6(occ)×6(vir)=36 orderings of a given unordered (occ-triple,
//! vir-triple) pair contributes the identical value `f`, so the textbook
//! fully-ordered sum equals `36 × Σ_{i0<j0<k0,a0<b0<c0} f`. My loop computes
//! `Σ_{i0<j0<k0} (6 × Σ_{a0<b0<c0} f) = 6 × Σ_{i0<j0<k0,a0<b0<c0} f` (occ ×1,
//! vir ×6 from ranging the full axis) — so the correct final divisor is 6,
//! NOT 36. This was caught by (and is now proven by) the direct
//! dense-vs-streaming regression test — an earlier draft used /36 and was
//! off by exactly 6× until this was fixed.
//!
//! Peak per-triple footprint is a handful of `[2nv,2nv,2nv]` f64 buffers
//! (`raw_w`, `raw_v`, `w_block`, `v_block`, `d_block`, one scratch) — see
//! [`peak_triple_block_bytes`] and the updated size guard below, which now
//! bounds that per-triple footprint (plus the O(no·nv³)-class precomputed
//! intermediates) instead of the old O((no2·nv2)³) dense ceiling.

use super::{CcConfig, CcResult};
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::blas_threads::with_blas_threads;
use ferric_integrals::operator::Operator;
use ferric_mp2::mo_transform::{transform_3center_oo, transform_3center_ov, transform_3center_vv};
use ferric_mp2::rimp2::{active_occ, cholesky_inverse_sqrt};
use ferric_mp2::spinorbital::{asym_phys, build_b, transpose_b};
use ferric_scf::ScfResult;
use ferric_tensors::{einsum, Axis};
use ndarray::{Array2, Array3, ArrayD, Axis as NdAxis};
#[cfg(test)]
use ndarray::IxDyn;
use rayon::prelude::*;

/// P(a/bc) on the (…,a,b,c) axes 3,4,5: x − x.swap(a,b) − x.swap(a,c).
///
/// Kept ONLY as the documented reference definition the streaming
/// per-triple sign structure in [`triple_block`] is transcribed from (see
/// the module doc comment); not on the hot path any more (no full 6D tensor
/// exists to permute).
#[cfg(test)]
fn p_a_bc(x: ArrayD<f64>) -> ArrayD<f64> {
    let s_ab = x.view().permuted_axes(IxDyn(&[0, 1, 2, 4, 3, 5])).as_standard_layout().into_owned();
    let s_ac = x.view().permuted_axes(IxDyn(&[0, 1, 2, 5, 4, 3])).as_standard_layout().into_owned();
    let mut out = x;
    out -= &s_ab;
    drop(s_ab);
    out -= &s_ac;
    out
}
/// P(i/jk) on the (i,j,k,…) axes 0,1,2: x − x.swap(i,j) − x.swap(i,k).
#[cfg(test)]
fn p_i_jk(x: ArrayD<f64>) -> ArrayD<f64> {
    let s_ij = x.view().permuted_axes(IxDyn(&[1, 0, 2, 3, 4, 5])).as_standard_layout().into_owned();
    let s_ik = x.view().permuted_axes(IxDyn(&[2, 1, 0, 3, 4, 5])).as_standard_layout().into_owned();
    let mut out = x;
    out -= &s_ij;
    drop(s_ij);
    out -= &s_ik;
    out
}

/// Peak bytes for the per-triple `[nv2,nv2,nv2]` working set (6 buffers:
/// raw_w, raw_v, w_block, v_block, d_block, one scratch — see module doc).
fn peak_triple_block_bytes(nv2: usize) -> usize {
    nv2.saturating_pow(3).saturating_mul(6).saturating_mul(8)
}

/// Bytes held by the PRECOMPUTED spin-orbital integral blocks that live for the
/// whole triples loop, plus the largest transient live while they are built.
///
/// These are what the old guard missed. It checked only
/// [`peak_triple_block_bytes`] — the per-triple working set — while `bcei` is
/// allocated ~25 lines later at `(2nv)³·(2no)`, i.e.
///
/// ```text
///     bcei / per_triple = 2·no / 6
/// ```
///
/// so the shortfall GREW with system size, which is the opposite of what a
/// safety margin should do (measured: 1.7x at H2O/cc-pVDZ, 3.0x at
/// ethane/cc-pVDZ, 7.0x at benzene/cc-pVDZ). A job could pass pre-flight and
/// then OOM on the very next allocation — the failure mode the guard exists to
/// prevent. See `tests/mwe_t_guard_covers_precomputed.rs`.
///
/// Counted here:
/// - `bcei` `(2nv)³(2no)`, `majk` `(2no)³(2nv)`, `bcjk` `(2nv)²(2no)²` — all
///   three persist through the loop.
/// - The `asym_phys` construction transient: each block's two spatial `einsum!`
///   outputs (`d` and `e`) are still live while the doubled-dimension result is
///   allocated. `bcei`'s is the largest at `2·nv³·no`.
/// - `t1` `(2no)(2nv)` and `t2` `(2no)²(2nv)²`, cloned from the `CcResult` and
///   held for the duration.
fn precomputed_block_bytes(no2: usize, nv2: usize) -> usize {
    let bcei = nv2.saturating_pow(3).saturating_mul(no2);
    let majk = no2.saturating_pow(3).saturating_mul(nv2);
    let bcjk = nv2.saturating_pow(2).saturating_mul(no2.saturating_pow(2));
    // asym_phys holds both spatial inputs live while allocating its output; the
    // spatial dims are half the spin-orbital ones, so 2·(nv2/2)³·(no2/2).
    let asym_transient = 2usize
        .saturating_mul((nv2 / 2).saturating_pow(3))
        .saturating_mul(no2 / 2);
    let t1 = no2.saturating_mul(nv2);
    let t2 = no2.saturating_pow(2).saturating_mul(nv2.saturating_pow(2));

    bcei.saturating_add(majk)
        .saturating_add(bcjk)
        .saturating_add(asym_transient)
        .saturating_add(t1)
        .saturating_add(t2)
        .saturating_mul(8)
}

/// Enumerate every unique unordered occupied spin-orbital triple `i0<j0<k0` in
/// `0..no2`, in ascending lexicographic order. This flattened list is what the
/// parallel-triples loop below iterates over — building it once up front (a
/// pure function of `no2` alone) means the chunk boundaries used for
/// memory-bounded banding are a pure function of the triple list, never of the
/// thread count, so the summation order (and hence the bit pattern of `et`)
/// cannot depend on `RAYON_NUM_THREADS`.
fn unique_occ_triples(no2: usize) -> Vec<(usize, usize, usize)> {
    let mut triples = Vec::new();
    for i0 in 0..no2 {
        for j0 in (i0 + 1)..no2 {
            for k0 in (j0 + 1)..no2 {
                triples.push((i0, j0, k0));
            }
        }
    }
    triples
}

/// Thread-count-independent chunk width (number of triples processed in one
/// parallel band) from a byte budget: `chunk_len * peak_triple_block_bytes(nv2)
/// <= band_budget_bytes`, floored at 1 so an oversized single triple still
/// makes progress (the existing `check_alloc` pre-flight guard in [`ccsd_t`]
/// already rejects a job whose single-triple footprint alone exceeds the full
/// resolved budget, so this floor is only ever exercised when one triple fits
/// but many concurrent copies don't).
///
/// This mirrors the `ferric_scf::reduce` banding pattern (see that module's
/// doc comment): partition a flattened work list into fixed-size bands sized
/// from a byte budget, run one band in parallel, fold serially across bands in
/// ascending order. The chunk width here is a pure function of `nv2` and the
/// band budget — NOT of `rayon::current_num_threads()` — so chunk boundaries
/// (and therefore the serial fold order across chunks) stay identical at any
/// thread count.
fn triple_chunk_len(nv2: usize, band_budget_bytes: usize) -> usize {
    let per_triple = peak_triple_block_bytes(nv2).max(1);
    (band_budget_bytes / per_triple).max(1)
}

/// `raw_w[a,b,c] = Σ_e t2[j,k,a,e]·bcei[b,c,e,i]  −  Σ_m t2[i,m,b,c]·majk[m,a,j,k]`
/// for a FIXED (possibly unordered) occupied triple `(i,j,k)`. Both
/// contractions are BLAS3 GEMMs on `nv2`-scale reshapes.
fn raw_w_block(
    t2: &ArrayD<f64>,
    bcei: &ArrayD<f64>,
    majk: &ArrayD<f64>,
    no2: usize,
    nv2: usize,
    i: usize,
    j: usize,
    k: usize,
) -> Array3<f64> {
    // Term 1: Σ_e t2[j,k,a,e]·bcei[b,c,e,i]
    //   bcei_i[b,c,e] = bcei[b,c,e,i], reshape to (nv2*nv2, nv2) as (bc, e)
    //   t2_jk[a,e] = t2[j,k,a,e], shape (nv2, nv2)
    //   term1[bc, a] = bcei_i_flat (bc,e) . t2_jk^T (e,a)  -> (nv2*nv2, nv2)
    let bcei_i = bcei.index_axis(NdAxis(3), i).to_owned(); // [nv2,nv2,nv2] (b,c,e)
    let bcei_i_flat = bcei_i
        .to_shape((nv2 * nv2, nv2))
        .expect("bcei_i reshape")
        .to_owned();
    let t2_jk = t2.index_axis(NdAxis(0), j).index_axis(NdAxis(0), k).to_owned(); // [nv2,nv2] (a,e)
    let t2_jk: Array2<f64> = t2_jk.into_dimensionality().expect("t2_jk 2D");
    let term1_flat = bcei_i_flat.dot(&t2_jk.t()); // (bc, a)
    // reshape (nv2*nv2, nv2) [bc,a] -> [b,c,a] -> permute to [a,b,c]
    let term1_bca = term1_flat
        .to_shape((nv2, nv2, nv2))
        .expect("term1 reshape")
        .to_owned(); // [b,c,a]
    let mut term1 = term1_bca.view().permuted_axes([2, 0, 1]).as_standard_layout().into_owned(); // [a,b,c]

    // Term 2: Σ_m t2[i,m,b,c]·majk[m,a,j,k]
    //   t2_i[m,(b,c)] = t2[i,m,b,c], shape (no2, nv2*nv2)
    //   majk_jk[m,a] = majk[m,a,j,k], shape (no2, nv2)
    //   term2[a,(b,c)] = majk_jk^T (a,m) . t2_i (m,bc) -> (nv2, nv2*nv2)
    let t2_i = t2.index_axis(NdAxis(0), i).to_owned(); // [no2,nv2,nv2] (m,b,c)
    let t2_i_flat = t2_i
        .to_shape((no2, nv2 * nv2))
        .expect("t2_i reshape")
        .to_owned();
    let majk_jk = majk
        .index_axis(NdAxis(3), k)
        .index_axis(NdAxis(2), j)
        .to_owned(); // [no2,nv2] (m,a)
    let majk_jk: Array2<f64> = majk_jk.into_dimensionality().expect("majk_jk 2D");
    let term2_flat = majk_jk.t().dot(&t2_i_flat); // (a, bc)
    let term2 = term2_flat
        .to_shape((nv2, nv2, nv2))
        .expect("term2 reshape")
        .to_owned(); // [a,b,c]

    term1 -= &term2;
    term1
}

/// `raw_v[a,b,c] = t1[i,a]·bcjk[b,c,j,k]` for a fixed occupied triple
/// `(i,j,k)` — a plain outer product.
fn raw_v_block(
    t1: &ArrayD<f64>,
    bcjk: &ArrayD<f64>,
    nv2: usize,
    i: usize,
    j: usize,
    k: usize,
) -> Array3<f64> {
    let t1_i = t1.index_axis(NdAxis(0), i); // [nv2] (a)
    let bcjk_jk = bcjk
        .index_axis(NdAxis(3), k)
        .index_axis(NdAxis(2), j)
        .to_owned(); // [nv2,nv2] (b,c)
    let mut out = Array3::<f64>::zeros((nv2, nv2, nv2));
    for a in 0..nv2 {
        let ta = t1_i[a];
        if ta == 0.0 {
            continue;
        }
        let mut slice = out.index_axis_mut(NdAxis(0), a);
        slice.assign(&bcjk_jk);
        slice *= ta;
    }
    out
}

/// Swap axes 0,1 of a `[nv2,nv2,nv2]` block: out[a,b,c] = x[b,a,c].
fn swap01(x: &Array3<f64>) -> Array3<f64> {
    x.view().permuted_axes([1, 0, 2]).as_standard_layout().into_owned()
}
/// Swap axes 0,2 of a `[nv2,nv2,nv2]` block: out[a,b,c] = x[c,b,a].
fn swap02(x: &Array3<f64>) -> Array3<f64> {
    x.view().permuted_axes([2, 1, 0]).as_standard_layout().into_owned()
}

/// P(a/bc) applied to an already-materialized `[nv2,nv2,nv2]` block.
fn p_a_bc_block(x: &Array3<f64>) -> Array3<f64> {
    let mut out = x.clone();
    out -= &swap01(x);
    out -= &swap02(x);
    out
}

/// Full antisymmetrized `W` (or `V`) `[nv2,nv2,nv2]` block for the canonical
/// occupied triple `i0<j0<k0`: `P(i/jk) P(a/bc) raw(i,j,k,a,b,c)`, i.e. the
/// signed sum over the 3 occupied permutations `{+(i0,j0,k0), −(j0,i0,k0),
/// −(k0,j0,i0)}` of `P(a/bc) raw_block(perm)`.
fn triple_block(
    raw_fn: impl Fn(usize, usize, usize) -> Array3<f64>,
    i0: usize,
    j0: usize,
    k0: usize,
) -> Array3<f64> {
    let raw_ijk = raw_fn(i0, j0, k0);
    let raw_jik = raw_fn(j0, i0, k0);
    let raw_kji = raw_fn(k0, j0, i0);
    let mut out = p_a_bc_block(&raw_ijk);
    out -= &p_a_bc_block(&raw_jik);
    out -= &p_a_bc_block(&raw_kji);
    out
}

/// Compute the (T) triples correction to CCSD.
///
/// Returns `E_(T)` (a negative number for typical closed-shell systems), to be
/// added to the CCSD correlation energy. Requires the CCSD T1 amplitudes.
///
/// Streams over unique occupied spin-orbital triples `i<j<k`, forming only a
/// `[2nv,2nv,2nv]` block per triple — see the module doc comment for the
/// full derivation and the peak-memory accounting.
#[must_use]
pub fn ccsd_t(
    _mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    cc: &CcResult,
    cfg: &CcConfig,
) -> Result<f64, FerricError> {
    let t1_spatial = cc
        .t1
        .as_ref()
        .ok_or_else(|| FerricError::General("CCSD(T) requires T1 amplitudes".into()))?;

    let nbas = obs.nbasis();
    let nocc_total = rhf.eps_r().iter().filter(|&&e| e < 0.0).count();
    let first_occ = cfg.frozen_core;
    let no = active_occ(nocc_total, first_occ)?;
    let nv = nbas - nocc_total; // spatial virtual
    let (no2, nv2) = (2 * no, 2 * nv);

    // No valid i<j<k triple unless there are ≥3 occupied spin-orbitals AND
    // ≥3 virtual spin-orbitals; (T) is identically zero otherwise (e.g. H2).
    if no2 < 3 || nv2 < 3 {
        return Ok(0.0);
    }

    // Fail-fast size guard: the STREAMING (T) only ever materializes O(6)
    // `[nv2,nv2,nv2]` buffers at a time (peak_triple_block_bytes), plus the
    // O(no·nv³)/O(no³·nv)/O(no²·nv²)-scale precomputed intermediates
    // (bcei/majk/bcjk/t1/t2/.. — same size class the CCSD driver already
    // carries, NOT O((no2·nv2)³)). This bounds the genuinely large piece:
    // the per-triple virtual-block footprint, which can still be large for
    // a huge virtual space even though it no longer scales with no at all.
    let peak_triple = peak_triple_block_bytes(nv2);
    let precomputed = precomputed_block_bytes(no2, nv2);
    let budget = ferric_core::memory::resolve_budget_bytes(cfg.memory_budget_bytes);
    // The precomputed blocks (bcei/majk/bcjk + t1/t2) are allocated ONCE and
    // shared; the per-triple working set is per rayon worker. Charge one
    // per-triple set here — the chunk width below is sized from the remaining
    // budget, so concurrency is bounded separately and must not be
    // double-counted into this floor.
    let floor = precomputed.saturating_add(peak_triple);
    ferric_core::memory::check_alloc(
        &format!(
            "CCSD(T) precomputed blocks + one per-triple block (no={no}, nv={nv} \
             spatial; no2={no2}, nv2={nv2} spin-orbitals)"
        ),
        floor,
        budget,
    )?;

    let eps = rhf.eps_r();
    let c = rhf.mos_r();
    let c_occ = c.slice(ndarray::s![.., first_occ..nocc_total]).to_owned();
    let c_vir = c.slice(ndarray::s![.., nocc_total..]).to_owned();

    // Dressed RI 3-index MO blocks (same construction as the CCSD driver).
    let v2c = ferric_integrals::threeindex::coulomb_metric_2c(op, dfbs)?;
    let v_inv_sqrt = cholesky_inverse_sqrt(&v2c)?;
    let eri3_ao = ferric_integrals::threeindex::eri3_tensor(op, obs, dfbs)?;
    let b_ov = build_b(&transform_3center_ov(&eri3_ao, &c_occ, &c_vir), &v_inv_sqrt, Axis::O, Axis::V);
    let b_oo = build_b(&transform_3center_oo(&eri3_ao, &c_occ), &v_inv_sqrt, Axis::O, Axis::O);
    let b_vv = build_b(&transform_3center_vv(&eri3_ao, &c_vir), &v_inv_sqrt, Axis::V, Axis::V);
    let b_vo = transpose_b(&b_ov);

    // --- Spin-orbital integral blocks needed by (T) — same construction as
    // the old dense path, still O(no·nv³)-class, not O((no·nv)³). ---
    // <bc||ei> (VVVO) indexed [b,c,e,i]: dir (be|ci)=b_vv,b_vo ; exc (bi|ce)=b_vo,b_vv
    let bcei: ArrayD<f64> = {
        let d: ArrayD<f64> = einsum!("Pbe,Pci->beci", &b_vv, &b_vo);
        let e: ArrayD<f64> = einsum!("Pbi,Pce->bice", &b_vo, &b_vv);
        asym_phys(&d, &e, nv, nv, nv, no)
    };
    // <ma||jk> (OVOO) indexed [m,a,j,k].
    let majk: ArrayD<f64> = {
        let d: ArrayD<f64> = einsum!("Pmj,Pak->mjak", &b_oo, &b_vo);
        let e: ArrayD<f64> = einsum!("Pmk,Paj->mkaj", &b_oo, &b_vo);
        asym_phys(&d, &e, no, nv, no, no)
    };
    // <bc||jk> (VVOO) indexed [b,c,j,k].
    let bcjk: ArrayD<f64> = {
        let d: ArrayD<f64> = einsum!("Pbj,Pck->bjck", &b_vo, &b_vo);
        let e: ArrayD<f64> = einsum!("Pbk,Pcj->bkcj", &b_vo, &b_vo);
        asym_phys(&d, &e, nv, nv, no, no)
    };

    // --- Spin-orbital energies (D is now formed per-triple, not as a 6D tensor). ---
    let mut eo = vec![0.0f64; no2];
    let mut ev = vec![0.0f64; nv2];
    for i in 0..no {
        eo[2 * i] = eps[first_occ + i];
        eo[2 * i + 1] = eps[first_occ + i];
    }
    for a in 0..nv {
        ev[2 * a] = eps[nocc_total + a];
        ev[2 * a + 1] = eps[nocc_total + a];
    }

    // --- T1 / T2 as plain ArrayD (dyn-dim) for the slicing helpers above. ---
    let t1: ArrayD<f64> = t1_spatial.clone().into_dyn();
    let t2: ArrayD<f64> = cc.t2.clone().into_dyn();

    // --- Stream over unique occupied triples i<j<k, one [nv2,nv2,nv2] block
    // at a time, now PARALLEL over triples via rayon. See module doc comment
    // for why summing (t3c+t3d)*D*t3c over unique occ triples (but ALL
    // nv2^3 virtual triples) and dividing by 36 exactly reproduces the
    // fully-ordered dense sum — proven directly by the dense-vs-streaming
    // regression test below.
    //
    // Parallelization strategy (thread-count-independent reduction):
    // each triple's contribution to `et` is fully independent of every other
    // triple, so this is an embarrassingly-parallel map-then-sum. Floating
    // point `+` is non-associative, so a rayon `reduce`/`sum` (whose binary
    // tree shape depends on the worker count) would make `et` drift with
    // `RAYON_NUM_THREADS` — forbidden here (bit-identity is asserted by
    // `thread_count_bit_identical_h2o_ccpvdz` below). Instead: flatten the
    // unique triples into a `Vec` (pure function of `no2`, see
    // `unique_occ_triples`), split it into fixed-size CHUNKS (pure function of
    // `nv2` and the byte budget, see `triple_chunk_len` — never of thread
    // count), and for each chunk: `par_iter().map(..).collect::<Vec<f64>>()`
    // (rayon's `collect` preserves index order regardless of worker count) to
    // get one partial `et` contribution per triple in the chunk, then fold
    // those chunk-local partials into `et` SERIALLY in ascending triple order.
    // The total addition order across all chunks and triples is therefore
    // exactly the same ascending (i0,j0,k0) lexicographic order the old
    // serial loop used, so `et` is bit-identical to the pre-parallelization
    // value and to any other thread count — this mirrors the
    // `ferric_scf::reduce` grouped-deterministic-sum banding pattern.
    //
    // Memory: each triple concurrently "in flight" inside one chunk holds its
    // own raw_w/raw_v/w_block/v_block (`peak_triple_block_bytes(nv2)` total,
    // 6 buffers) — so a chunk of width `W` has a live set of up to
    // `W * peak_triple_block_bytes(nv2)` (rayon may not run the full chunk
    // concurrently if there are fewer workers than `W`, but the byte budget
    // must assume the worst case). `triple_chunk_len` sizes `W` from
    // `budget/2` (half of the resolved memory budget — the other half is
    // headroom for the precomputed O(no·nv³)-class intermediates (bcei/majk/
    // bcjk/t1/t2) already live at this point), so worst-case peak stays
    // `<= budget/2`, on top of those intermediates. Chunk boundaries are
    // purely a function of the triple list + nv2 + budget (never thread
    // count), which is what keeps the fold order (and therefore `et`'s bit
    // pattern) deterministic above.
    //
    // BLAS threading: `raw_w_block`'s two GEMMs (`.dot()`) run inside this
    // rayon parallel region, so the count is pinned to 1 EXPLICITLY. The
    // runtime self-guard in `opt_in_blas_threads` only fires when it is called
    // FROM a rayon worker; the wrap below is evaluated on the caller thread, so
    // `opt_in_blas_threads()` here would return the user's `FERRIC_BLAS_THREADS`
    // and raise the global count for the whole `par_iter` — the rayon×OpenBLAS
    // oversubscription / worker-stack-overflow hazard documented in
    // ferric-core/src/blas_threads.rs.
    let triples = unique_occ_triples(no2);
    // Size the band from what is LEFT after the precomputed blocks, not from
    // the full budget: bcei/majk/bcjk/t1/t2 are already resident by this point,
    // so charging the band against the whole budget would let concurrency
    // reclaim memory that is not actually free. The pre-flight above proved
    // `precomputed + one per-triple set` fits, so the remainder is >= 0 and the
    // chunk length still floors at 1 (slow, never stuck).
    let remaining = budget.saturating_sub(precomputed);
    let chunk_len = triple_chunk_len(
        nv2,
        ferric_core::memory::transient_share(remaining, ferric_core::memory::Share::Half),
    );
    // A floored chunk (width 1) means the memory budget could not fund even a
    // two-triple band: correct but effectively serial across triples, so the
    // (T) step will be slow for a reason the user can act on (raise
    // [memory] budget_gb). Say so once rather than letting it look like a hang.
    if chunk_len == 1 && triples.len() > 1 {
        eprintln!(
            "ccsd_t: memory budget floors the triple band at 1 (nv2={nv2}, \
             {} triples, {:.2} GiB remaining after precomputed blocks) — the \
             (T) triples loop will run effectively serially. Raise the memory \
             budget to widen the band.",
            triples.len(),
            remaining as f64 / (1024.0 * 1024.0 * 1024.0),
        );
    }
    let mut et = 0.0f64;
    for chunk in triples.chunks(chunk_len) {
        let partials: Vec<f64> = with_blas_threads(1, || {
            chunk
                .par_iter()
                .map(|&(i0, j0, k0)| {
                    let w_block = triple_block(
                        |i, j, k| raw_w_block(&t2, &bcei, &majk, no2, nv2, i, j, k),
                        i0,
                        j0,
                        k0,
                    );
                    let v_block = triple_block(
                        |i, j, k| raw_v_block(&t1, &bcjk, nv2, i, j, k),
                        i0,
                        j0,
                        k0,
                    );
                    let e_i = eo[i0] + eo[j0] + eo[k0];
                    let mut partial = 0.0f64;
                    for a in 0..nv2 {
                        for b in 0..nv2 {
                            for c in 0..nv2 {
                                let d = e_i - ev[a] - ev[b] - ev[c];
                                let w = w_block[[a, b, c]];
                                let v = v_block[[a, b, c]];
                                let t3c = w / d;
                                let t3d = v / d;
                                partial += (t3c + t3d) * d * t3c;
                            }
                        }
                    }
                    partial
                })
                .collect()
        });
        // Serial fold in ascending triple order — the determinism anchor
        // (see the comment block above).
        for p in partials {
            et += p;
        }
    }
    // NOTE the divisor here is 6, not the textbook 36: `i0<j0<k0` already
    // ranges over each UNIQUE unordered occupied triple exactly once (not
    // all 6 orderings), while the inner a,b,c loop ranges over the FULL
    // `0..nv2` range for each axis, i.e. all 6 orderings of every unique
    // unordered virtual triple. Since `f(i,j,k,a,b,c) := (t3c+t3d)·D·t3c` is
    // invariant (not just sign-changing) under ANY simultaneous relabeling
    // of (i,j,k) or (a,b,c) — `t3c`/`t3d` flip sign under a single swap but
    // `D` does not depend on ordering, and `f` pairs an even number of
    // sign-flips (t3c appears twice) — every one of the 6×6=36 orderings of
    // a given unordered (occ-triple, vir-triple) pair contributes the SAME
    // value. The textbook fully-ordered sum therefore has value
    // `36 × Σ_{i0<j0<k0,a0<b0<c0} f`; my loop computes
    // `Σ_{i0<j0<k0} (6 × Σ_{a0<b0<c0} f) = 6 × Σ_{i0<j0<k0,a0<b0<c0} f`
    // (occ ranges over unique triples ×1, vir ranges over unique triples
    // ×6), so dividing by 6 (not 36) recovers `Σ_{i0<j0<k0,a0<b0<c0} f`,
    // which equals `(textbook sum)/36`. Verified directly against the dense
    // path to ~1e-10 in `streaming_matches_dense_h2o_ccpvdz`.
    Ok(et / 6.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ccsd::ccsd;
    use ferric_core::basis;
    use ferric_core::mol::Molecule;
    use ferric_core::parallel::ParallelContext;
    use ferric_integrals::basis_bridge::PreparedBasis;
    use ferric_integrals::operator::Operator;
    use ferric_scf::rhf::{solve_rhf, RhfConfig};
    use ferric_scf::screening::SchwarzBounds;
    use ferric_tensors::Tensor;

    #[test]
    fn test_ccsd_t_h2_sto3g_is_zero() {
        // H2/STO-3G has 1 occ + 1 vir spatial orbital, so there is no valid
        // i<j<k triple: E(T) is identically 0.0 (PySCF agrees to 1e-10). This
        // is the correct PHYSICAL answer, not stub behavior — it cannot
        // distinguish a working kernel from a stub, which is why the real gate
        // is the H2O test below.
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("sto-3g").unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("def2-qzvpp-rifit").unwrap()).unwrap();
        let op = Operator::coulomb();
        let ctx = ParallelContext::default();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
        let cc_cfg = CcConfig { frozen_core: 0, max_iter: 100, energy_conv: 1e-9, ..Default::default() };
        let ccsd_res = ccsd(&mol, &obs, &dfbs, op, &rhf, &cc_cfg).unwrap();
        let t_corr = ccsd_t(&mol, &obs, &dfbs, op, &rhf, &ccsd_res, &cc_cfg).unwrap();
        assert!(t_corr.abs() < 1e-10, "H2 (T) should be 0, got {t_corr}");
    }

    #[test]
    fn test_ccsd_t_h2o_ccpvdz() {
        // Real correctness gate: H2O/cc-pVDZ (T) = -0.0030587091 (PySCF RHF
        // ccsd_t() and GCCSD(T), agree to 1e-9). Verified-exact spin-orbital
        // recipe (interleaved convention, drop canonical fock[v,o]); see the
        // numpy cross-check that reproduced this to 7e-11. def2-qzvpp-rifit
        // keeps the RI error below 1e-4. This now exercises the STREAMING
        // per-triple-block path (the dense path is retired).
        let mol = Molecule::parse_xyz(
            "3\n\nO 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692\n",
            0,
            1,
        )
        .unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("def2-qzvpp-rifit").unwrap()).unwrap();
        let op = Operator::coulomb();
        let ctx = ParallelContext::default();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
        let cc_cfg = CcConfig { frozen_core: 0, max_iter: 100, energy_conv: 1e-9, ..Default::default() };
        let ccsd_res = ccsd(&mol, &obs, &dfbs, op, &rhf, &cc_cfg).unwrap();
        let t_corr = ccsd_t(&mol, &obs, &dfbs, op, &rhf, &ccsd_res, &cc_cfg).unwrap();
        println!("CCSD(T) H2O/cc-pVDZ (T) = {t_corr:.10}");
        assert!(
            (t_corr - (-0.0030587091)).abs() < 1e-4,
            "(T) = {t_corr:.10}, expected -0.0030587091"
        );
    }

    #[test]
    #[ignore = "scale demo, run explicitly: cargo test -p ferric-cc ccsd_t_scale_butane -- --ignored --nocapture (see docs/ccsdt-roadmap-decision.md)"]
    fn ccsd_t_scale_butane_sto3g() {
        // Butane (C4H10, testdata/molecules/alkane_4.xyz) / STO-3G: nbas=30,
        // no=17, nv=13 spatial -> no2=34, nv2=26 spin-orbitals. This is the
        // "genuinely too big for the OLD dense path" demonstration the
        // streaming rewrite exists for:
        //   dense peak  = (no2*nv2)^3 * 6 * 8 bytes  ~ 33.2 GB
        //   streaming per-triple peak = nv2^3 * 6 * 8 bytes ~ 0.84 MB
        // 33 GB dense would not fit in this box's ~12 GB available RAM; the
        // streaming path's peak per-triple block is under 1 MB regardless of
        // how many of the C(34,3)=5984 unique occupied triples get streamed
        // through. Run under `/usr/bin/time -v` (or an equivalent RSS
        // sampler) for the peak-RSS number quoted in the task report.
        use std::time::Instant;
        let xyz_path = format!("{}/../../testdata/molecules/alkane_4.xyz", env!("CARGO_MANIFEST_DIR"));
        let mol = Molecule::load_xyz(&xyz_path).unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("sto-3g").unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("def2-qzvpp-rifit").unwrap()).unwrap();
        let op = Operator::coulomb();
        let ctx = ParallelContext::default();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
        assert!(rhf.converged, "butane/STO-3G RHF must converge");
        let cc_cfg = CcConfig { frozen_core: 0, max_iter: 100, energy_conv: 1e-8, ..Default::default() };
        let t0 = Instant::now();
        let ccsd_res = ccsd(&mol, &obs, &dfbs, op, &rhf, &cc_cfg).unwrap();
        let dt_ccsd = t0.elapsed();
        println!("butane/STO-3G CCSD E_corr = {:.10} ({:.1}s)", ccsd_res.correlation_energy, dt_ccsd.as_secs_f64());
        let t1 = Instant::now();
        let t_corr = ccsd_t(&mol, &obs, &dfbs, op, &rhf, &ccsd_res, &cc_cfg).unwrap();
        let dt_t = t1.elapsed();
        println!(
            "butane/STO-3G (T) = {:.10} ({:.1}s, streaming per-triple-block path)",
            t_corr,
            dt_t.as_secs_f64()
        );
        assert!(t_corr.is_finite() && t_corr < 0.0, "(T) should be a finite negative correction, got {t_corr}");
    }

    #[test]
    fn ccsd_t_fails_fast_under_tiny_budget() {
        // The size guard must ERROR cleanly (not OOM-kill) before the
        // per-triple-block allocation when the budget is tiny. Uses an
        // EXPLICIT config budget so no process-global env var is touched
        // (explicit wins in resolve_budget_bytes). We build a valid RHF + a
        // dummy CcResult (t1/t2 shapes don't matter — the guard fires before
        // they are used numerically), then assert the error.
        let mol = Molecule::parse_xyz(
            "3\n\nO 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692\n",
            0,
            1,
        )
        .unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("def2-qzvpp-rifit").unwrap()).unwrap();
        let op = Operator::coulomb();
        let ctx = ParallelContext::default();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();

        // Dummy CCSD result — the guard runs before t1/t2 are consumed.
        let nbas = obs.nbasis();
        let nocc = rhf.eps_r().iter().filter(|&&e| e < 0.0).count();
        let nv = nbas - nocc;
        let dummy = CcResult {
            correlation_energy: 0.0,
            t1: Some(ndarray::Array2::<f64>::zeros((nocc, nv))),
            t2: ndarray::Array4::<f64>::zeros((nocc, nocc, nv, nv)),
        };
        // 1e-6 GiB ≈ 1 KB budget — far below the H2O/cc-pVDZ (T) per-triple peak.
        let cc_cfg = CcConfig {
            frozen_core: 0,
            memory_budget_bytes: Some(ferric_core::memory::gib_to_bytes(1e-6)),
            ..Default::default()
        };
        let err = ccsd_t(&mol, &obs, &dfbs, op, &rhf, &dummy, &cc_cfg).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("CCSD(T)"), "unexpected error: {msg}");
        assert!(msg.contains("budget is"), "unexpected error: {msg}");
    }

    /// Direct regression proof that the streaming per-triple-block rewrite
    /// reproduces the OLD dense 6D formulation bit-for-bit (to ~1e-10) on
    /// H2O/cc-pVDZ — the primary correctness gate for this refactor, more
    /// rigorous than the 1e-4 PySCF tolerance above. Reimplements the OLD
    /// dense path inline (not calling the removed dense `ccsd_t`) using the
    /// SAME `p_a_bc`/`p_i_jk` reference permutation functions kept in this
    /// module under `#[cfg(test)]`.
    #[test]
    fn streaming_matches_dense_h2o_ccpvdz() {
        let mol = Molecule::parse_xyz(
            "3\n\nO 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692\n",
            0,
            1,
        )
        .unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("def2-qzvpp-rifit").unwrap()).unwrap();
        let op = Operator::coulomb();
        let ctx = ParallelContext::default();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
        let cc_cfg = CcConfig { frozen_core: 0, max_iter: 100, energy_conv: 1e-9, ..Default::default() };
        let ccsd_res = ccsd(&mol, &obs, &dfbs, op, &rhf, &cc_cfg).unwrap();

        // Streaming result (the production path).
        let t_streaming = ccsd_t(&mol, &obs, &dfbs, op, &rhf, &ccsd_res, &cc_cfg).unwrap();

        // --- Inline OLD dense path, reconstructed from the same intermediates. ---
        let nbas = obs.nbasis();
        let nocc_total = rhf.eps_r().iter().filter(|&&e| e < 0.0).count();
        let first_occ = 0usize;
        let no = nocc_total - first_occ;
        let nv = nbas - nocc_total;
        let (no2, nv2) = (2 * no, 2 * nv);

        let eps = rhf.eps_r();
        let c = rhf.mos_r();
        let c_occ = c.slice(ndarray::s![.., first_occ..nocc_total]).to_owned();
        let c_vir = c.slice(ndarray::s![.., nocc_total..]).to_owned();
        let v2c = ferric_integrals::threeindex::coulomb_metric_2c(op, &dfbs).unwrap();
        let v_inv_sqrt = cholesky_inverse_sqrt(&v2c).unwrap();
        let eri3_ao = ferric_integrals::threeindex::eri3_tensor(op, &obs, &dfbs).unwrap();
        let b_ov = build_b(&transform_3center_ov(&eri3_ao, &c_occ, &c_vir), &v_inv_sqrt, Axis::O, Axis::V);
        let b_oo = build_b(&transform_3center_oo(&eri3_ao, &c_occ), &v_inv_sqrt, Axis::O, Axis::O);
        let b_vv = build_b(&transform_3center_vv(&eri3_ao, &c_vir), &v_inv_sqrt, Axis::V, Axis::V);
        let b_vo = transpose_b(&b_ov);
        use Axis::{O, V};

        let bcei = {
            let d: ArrayD<f64> = einsum!("Pbe,Pci->beci", &b_vv, &b_vo);
            let e: ArrayD<f64> = einsum!("Pbi,Pce->bice", &b_vo, &b_vv);
            asym_phys(&d, &e, nv, nv, nv, no)
        };
        let majk = {
            let d: ArrayD<f64> = einsum!("Pmj,Pak->mjak", &b_oo, &b_vo);
            let e: ArrayD<f64> = einsum!("Pmk,Paj->mkaj", &b_oo, &b_vo);
            asym_phys(&d, &e, no, nv, no, no)
        };
        let bcjk = {
            let d: ArrayD<f64> = einsum!("Pbj,Pck->bjck", &b_vo, &b_vo);
            let e: ArrayD<f64> = einsum!("Pbk,Pcj->bkcj", &b_vo, &b_vo);
            asym_phys(&d, &e, nv, nv, no, no)
        };
        let bcei_t = Tensor::new(bcei, [V, V, V, O]);
        let majk_t = Tensor::new(majk, [O, V, O, O]);
        let bcjk_t = Tensor::new(bcjk, [V, V, O, O]);

        let mut eo = vec![0.0f64; no2];
        let mut ev = vec![0.0f64; nv2];
        for i in 0..no {
            eo[2 * i] = eps[first_occ + i];
            eo[2 * i + 1] = eps[first_occ + i];
        }
        for a in 0..nv {
            ev[2 * a] = eps[nocc_total + a];
            ev[2 * a + 1] = eps[nocc_total + a];
        }
        let mut d3 = ArrayD::zeros(IxDyn(&[no2, no2, no2, nv2, nv2, nv2]));
        for i in 0..no2 {
            for j in 0..no2 {
                for k in 0..no2 {
                    for a in 0..nv2 {
                        for b in 0..nv2 {
                            for cc_ in 0..nv2 {
                                d3[[i, j, k, a, b, cc_]] = eo[i] + eo[j] + eo[k] - ev[a] - ev[b] - ev[cc_];
                            }
                        }
                    }
                }
            }
        }

        let t1_spatial = ccsd_res.t1.as_ref().unwrap();
        let t1 = Tensor::new(t1_spatial.clone().into_dyn(), [O, V]);
        let t2 = Tensor::new(ccsd_res.t2.clone().into_dyn(), [O, O, V, V]);

        let term1: ArrayD<f64> = einsum!("jkae,bcei->jkabci", &t2, &bcei_t);
        let term1 = term1.permuted_axes(IxDyn(&[5, 0, 1, 2, 3, 4])).as_standard_layout().into_owned();
        let term2: ArrayD<f64> = einsum!("imbc,majk->ibcajk", &t2, &majk_t);
        let term2 = term2.permuted_axes(IxDyn(&[0, 4, 5, 3, 1, 2])).as_standard_layout().into_owned();
        let mut t3c = term1;
        t3c -= &term2;
        drop(term2);
        let mut t3c = p_a_bc(t3c);
        t3c = p_i_jk(t3c);
        t3c /= &d3;

        let t3d: ArrayD<f64> = einsum!("ia,bcjk->iabcjk", &t1, &bcjk_t);
        let t3d = t3d.permuted_axes(IxDyn(&[0, 4, 5, 1, 2, 3])).as_standard_layout().into_owned();
        let mut t3d = p_a_bc(t3d);
        t3d = p_i_jk(t3d);
        t3d /= &d3;

        let sum = &t3c + &t3d;
        let weighted = &sum * &d3;
        let t_dense: f64 = (&weighted * &t3c).sum() / 36.0;

        println!("dense = {t_dense:.12}, streaming = {t_streaming:.12}, diff = {:.3e}", (t_dense - t_streaming).abs());
        assert!(
            (t_dense - t_streaming).abs() < 1e-10,
            "streaming (T) = {t_streaming:.12} disagrees with dense (T) = {t_dense:.12}"
        );
    }

    /// The parallel-triples rewrite MUST produce a bit-identical `et` at any
    /// `RAYON_NUM_THREADS` — this is the whole point of the collect-then-
    /// serial-sum idiom (never a rayon reduce/sum) documented at the loop
    /// site in [`ccsd_t`]. Runs the SAME (mol, basis, CCSD result) through
    /// `ccsd_t` under two different rayon thread pools (1 and 4 workers,
    /// installed via `ThreadPoolBuilder::install`) and asserts the two `f64`
    /// results have identical bit patterns (`to_bits()`), not just "close".
    #[test]
    fn thread_count_bit_identical_h2o_ccpvdz() {
        let mol = Molecule::parse_xyz(
            "3\n\nO 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692\n",
            0,
            1,
        )
        .unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("def2-qzvpp-rifit").unwrap()).unwrap();
        let op = Operator::coulomb();
        let ctx = ParallelContext::default();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
        let cc_cfg = CcConfig { frozen_core: 0, max_iter: 100, energy_conv: 1e-9, ..Default::default() };
        let ccsd_res = ccsd(&mol, &obs, &dfbs, op, &rhf, &cc_cfg).unwrap();

        let run_with_pool = |n_threads: usize| -> f64 {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(n_threads)
                .build()
                .unwrap();
            pool.install(|| ccsd_t(&mol, &obs, &dfbs, op, &rhf, &ccsd_res, &cc_cfg).unwrap())
        };

        let et_1 = run_with_pool(1);
        let et_4 = run_with_pool(4);
        println!("(T) 1-thread = {et_1:.15}, 4-thread = {et_4:.15}");
        assert_eq!(
            et_1.to_bits(),
            et_4.to_bits(),
            "(T) must be bit-identical across thread counts: 1-thread={et_1:.17e}, 4-thread={et_4:.17e}"
        );
    }
}
