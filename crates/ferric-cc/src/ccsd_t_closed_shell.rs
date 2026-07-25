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
//! (1991). PySCF's loop bands over *virtual* triples `a>=b>=c` with the `d3`
//! multiplicity trick; this module bands over *occupied* triples `i<=j<=k`
//! instead — the mirror-image restriction — so the per-triple working set is a
//! `[nv,nv,nv]` block, matching the memory structure of the spin-orbital
//! sibling and keeping the same `check_alloc` guard shape. The two bandings
//! were proven numerically equivalent in numpy against PySCF's own
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
//! (both BLAS3 GEMMs on `[nv*nv, nv]`-shaped reshapes — see [`raw_w_block`]).
//! The physical `W` is the sum of `w0` over the **6 simultaneous permutations
//! of the three (occupied, virtual) PAIRS** `(i,a) (j,b) (k,c)`:
//!
//! ```text
//! W[a,b,c] = Σ_{π ∈ S3} w0(i_π, j_π, k_π)[a_π, b_π, c_π]
//! ```
//!
//! i.e. each permuted raw block has its *virtual* axes permuted back by the
//! inverse permutation before accumulation (see [`w_block`]). `W` is therefore
//! invariant under any simultaneous pair relabeling — the property the
//! occupied-triple banding rests on.
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
//! (PySCF's `r3`, acting here on the virtual axes) and finally
//!
//! ```text
//! D[a,b,c] = ε_i + ε_j + ε_k − ε_a − ε_b − ε_c
//! E_(T)    = (1/3) Σ_{i<=j<=k} m(i,j,k) · Σ_{a,b,c} W̃[a,b,c]·V[a,b,c] / D[a,b,c]
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
//! index set agreed to ~1e-18) and `a,b,c` here range over the FULL axes, this
//! module bands over `i<=j<=k` with an explicit multiplicity weight
//!
//! ```text
//! m(i,j,k) = 1  if i==j==k        (1 ordering)
//!          = 3  if exactly two are equal
//!          = 6  if all three distinct
//! ```
//!
//! reproducing the unrestricted sum exactly. See [`occ_triple_multiplicity`].
//! The `i<=j<=k` banded weighted sum divided by 3 reproduced PySCF to 7.8e-16
//! in scratch (`step5.py`, `step6.py`) before this file existed.
//!
//! # Cost vs the spin-orbital path
//!
//! Per triple both paths do a fixed number of `raw_w`-shaped GEMMs whose cost
//! is `O(nv_axis³)`:
//!
//! - spin-orbital: `C(2no,3) ≈ (2no)³/6` triples × 3 occupied permutations ×
//!   `O((2nv)³)` ⇒ `~ 8no³ · 8nv³ / 2 = 32 no³nv³`
//! - closed-shell: `~ no³/6` triples (i<=j<=k) × 6 pair permutations ×
//!   `O(nv³)` ⇒ `~ no³nv³`
//!
//! so the *ideal* FLOP ratio is ~32×. The realized speedup is smaller (the
//! closed-shell block has more per-triple permutation/transpose bookkeeping
//! and a smaller GEMM that gets less BLAS3 reuse) — see the measured numbers
//! recorded in the module tests and in docs/VALIDATION.md rather than
//! trusting this estimate.
//!
//! # Memory and determinism
//!
//! Same structure as the spin-orbital sibling: peak resident is a handful of
//! `[nv,nv,nv]` f64 buffers per in-flight triple ([`peak_triple_block_bytes`],
//! `nv` not `2nv`, i.e. 8× smaller per buffer) plus the precomputed
//! `O(no·nv³)`-class chemist blocks. Parallel reduction is the same
//! chunk-then-collect-then-serial-fold idiom: chunk width is a pure function
//! of `nv` and the byte budget (never `rayon::current_num_threads()`), and
//! chunk partials are folded serially in ascending triple order, so the result
//! is bit-identical at any thread count (asserted by
//! `thread_count_bit_identical_h2o_ccpvdz` below).

use super::{CcConfig, CcResult};
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::blas_threads::{opt_in_blas_threads, with_blas_threads};
use ferric_integrals::operator::Operator;
use ferric_mp2::mo_transform::{transform_3center_oo, transform_3center_ov, transform_3center_vv};
use ferric_mp2::rimp2::{active_occ, cholesky_inverse_sqrt};
use ferric_mp2::spinorbital::build_b;
use ferric_scf::ScfResult;
use ferric_tensors::{einsum, Axis};
use ndarray::{Array2, Array3, Array4};
use rayon::prelude::*;

/// The 6 permutations of `(0,1,2)` in a fixed, deterministic order. Used both
/// to build the pair-symmetrized `W` and to enumerate the multiplicity classes.
const PERMS3: [[usize; 3]; 6] =
    [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];

/// Number of distinct orderings of the occupied triple `(i,j,k)` given
/// `i <= j <= k`.
///
/// This is the weight that turns the banded `i<=j<=k` loop back into the
/// unrestricted `Σ_{i,j,k}` sum the `1/3` divisor is defined against. Unlike
/// the spin-orbital path, repeated spatial occupied indices are physical and
/// carry a large part of the energy (see the module doc) — dropping them, or
/// weighting every triple by 6, is a silent >50% error.
fn occ_triple_multiplicity(i: usize, j: usize, k: usize) -> f64 {
    debug_assert!(i <= j && j <= k);
    if i == k {
        1.0 // i == j == k
    } else if i == j || j == k {
        3.0
    } else {
        6.0
    }
}

/// Enumerate every `i <= j <= k` occupied triple in ascending lexicographic
/// order. Pure function of `no` — the determinism anchor for the banded
/// parallel reduction (chunk boundaries must never depend on thread count).
fn occ_triples_with_repeats(no: usize) -> Vec<(usize, usize, usize)> {
    let mut out = Vec::new();
    for i in 0..no {
        for j in i..no {
            for k in j..no {
                out.push((i, j, k));
            }
        }
    }
    out
}

/// Peak bytes for the per-triple `[nv,nv,nv]` working set (raw block, `W`,
/// `V`, `W̃`, and two scratch transposes — 6 buffers, same accounting shape as
/// the spin-orbital sibling but over `nv` rather than `2nv`).
fn peak_triple_block_bytes(nv: usize) -> usize {
    nv.saturating_pow(3).saturating_mul(6).saturating_mul(8)
}

/// Bytes held by the precomputed chemist-notation blocks ([`TripleIntegrals`])
/// plus the amplitudes cloned alongside them, all live for the whole loop.
///
/// The spatial analogue of `ccsd_t::precomputed_block_bytes`, and it closes the
/// same defect: the guard used to check only [`peak_triple_block_bytes`] — the
/// per-triple working set — while `ovvv` is `no·nv³`, i.e. `no/6` times larger,
/// so the shortfall GREW with system size. A job could pass pre-flight and OOM
/// on the next allocation. See `tests/mwe_t_guard_covers_precomputed.rs`.
///
/// Counted: `ovvv` `no·nv³`, `ovoo` `no³·nv`, `ovov` `no²·nv²`, plus the cloned
/// `t1` `no·nv` and `t2` `no²·nv²`.
fn precomputed_block_bytes(no: usize, nv: usize) -> usize {
    let ovvv = no.saturating_mul(nv.saturating_pow(3));
    let ovoo = no.saturating_pow(3).saturating_mul(nv);
    let ovov = no.saturating_pow(2).saturating_mul(nv.saturating_pow(2));
    let t1 = no.saturating_mul(nv);
    let t2 = no.saturating_pow(2).saturating_mul(nv.saturating_pow(2));

    ovvv.saturating_add(ovoo)
        .saturating_add(ovov)
        .saturating_add(t1)
        .saturating_add(t2)
        .saturating_mul(8)
}

/// Thread-count-independent chunk width from a byte budget, floored at 1.
/// Mirrors `ccsd_t::triple_chunk_len` exactly (see that function's doc and the
/// `ferric_scf::reduce` banding pattern it cites).
fn triple_chunk_len(nv: usize, band_budget_bytes: usize) -> usize {
    let per_triple = peak_triple_block_bytes(nv).max(1);
    (band_budget_bytes / per_triple).max(1)
}

/// Precomputed chemist-notation integral blocks the per-triple kernel slices.
///
/// All are `O(no·nv³)`-class or smaller — the same size class the CCSD driver
/// already carries, NOT the `O((no·nv)³)` dense-6D ceiling the streaming
/// formulation exists to avoid.
struct TripleIntegrals {
    /// `(ia|bd)` at `[i, a, b, d]` — the VVVO-analogue (`ovvv`).
    ovvv: Array4<f64>,
    /// `(ia|lj)` at `[i, a, l, j]` — the OVOO block.
    ovoo: Array4<f64>,
    /// `(ia|jb)` at `[i, a, j, b]` — the OVOV block (feeds `V`'s singles terms).
    ovov: Array4<f64>,
}

/// `w0[a,b,c] = Σ_d (ia|bd)·t2[k,j,c,d] − Σ_l (ia|lj)·t2[l,k,b,c]` for a fixed
/// ORDERED occupied triple `(i,j,k)`.
///
/// Both terms are BLAS3 GEMMs:
/// - term 1 reshapes `ovvv[i]` to `(nv·nv, nv)` as `(ab, d)` and
///   right-multiplies by `t2[k,j]ᵀ` shaped `(d, c)`, giving `(ab, c)`;
/// - term 2 takes `ovoo[i,:,:,j]` shaped `(a, l)` and right-multiplies by
///   `t2[:,k]` reshaped `(l, bc)`, giving `(a, bc)`.
///
/// Both land directly in `[a,b,c]` order — no transpose needed, unlike the
/// spin-orbital sibling's `term1`.
fn raw_w_block(
    ints: &TripleIntegrals,
    t2: &Array4<f64>,
    no: usize,
    nv: usize,
    i: usize,
    j: usize,
    k: usize,
) -> Array3<f64> {
    // Term 1: (ab, d) · (d, c) -> (ab, c) == [a,b,c].
    let g = ints.ovvv.index_axis(ndarray::Axis(0), i); // [a,b,d]
    let g2 = g.to_shape((nv * nv, nv)).expect("ovvv[i] reshape");
    let t2_k = t2.index_axis(ndarray::Axis(0), k);
    let t2_kj = t2_k.index_axis(ndarray::Axis(0), j); // [c,d]
    let term1 = g2.dot(&t2_kj.t()); // (ab, c)
    let mut out = term1
        .to_shape((nv, nv, nv))
        .expect("term1 reshape")
        .to_owned();

    // Term 2: (a, l) · (l, bc) -> (a, bc) == [a,b,c].
    let ovoo_j = ints.ovoo.index_axis(ndarray::Axis(3), j);
    let h = ovoo_j.index_axis(ndarray::Axis(0), i); // [a,l]
    let t2_bk = t2.index_axis(ndarray::Axis(1), k); // [l,b,c]
    let t2_k2 = t2_bk.to_shape((no, nv * nv)).expect("t2[:,k] reshape");
    let term2 = h.dot(&t2_k2); // (a, bc)
    let term2 = term2
        .to_shape((nv, nv, nv))
        .expect("term2 reshape")
        .to_owned();

    out -= &term2;
    out
}

/// Permute the three axes of a `[nv,nv,nv]` block: `out[x0,x1,x2] = x[axes]`.
fn permute3(x: &Array3<f64>, axes: [usize; 3]) -> Array3<f64> {
    x.view().permuted_axes(axes).as_standard_layout().into_owned()
}

/// Fully pair-symmetrized `W[a,b,c]` for the occupied triple `(i,j,k)`:
/// the sum over the 6 simultaneous permutations of the pairs `(i,a) (j,b)
/// (k,c)`.
///
/// For permutation `π` we evaluate `raw_w(i_π, j_π, k_π)`, whose axes are
/// indexed by the *permuted* virtual labels `(a_π, b_π, c_π)`; permuting those
/// axes by `π⁻¹` brings them back to the canonical `(a,b,c)` order before
/// accumulation.
fn w_block(
    ints: &TripleIntegrals,
    t2: &Array4<f64>,
    no: usize,
    nv: usize,
    i: usize,
    j: usize,
    k: usize,
) -> Array3<f64> {
    let occ = [i, j, k];
    let mut out = Array3::<f64>::zeros((nv, nv, nv));
    for p in PERMS3 {
        let r = raw_w_block(ints, t2, no, nv, occ[p[0]], occ[p[1]], occ[p[2]]);
        // inverse permutation: inv[p[t]] = t
        let mut inv = [0usize; 3];
        for (t, &pt) in p.iter().enumerate() {
            inv[pt] = t;
        }
        out += &permute3(&r, inv);
    }
    out
}

/// `V[a,b,c] = W[a,b,c] + (jb|kc)·t1[i,a] + (ia|kc)·t1[j,b] + (ia|jb)·t1[k,c]`.
///
/// The `fock[v,o]·t2` term of the general formula is dropped: for canonical HF
/// orbitals `fock[v,o] = 0` (same simplification as [`crate::ccsd_t::ccsd_t`]
/// and [`crate::ccsd_closed_shell`]).
fn add_v_singles(
    v: &mut Array3<f64>,
    ints: &TripleIntegrals,
    t1: &Array2<f64>,
    nv: usize,
    i: usize,
    j: usize,
    k: usize,
) {
    let ovov_k = ints.ovov.index_axis(ndarray::Axis(2), k);
    let ovov_j = ints.ovov.index_axis(ndarray::Axis(2), j);
    let g_jk = ovov_k.index_axis(ndarray::Axis(0), j); // [b,c]
    let g_ik = ovov_k.index_axis(ndarray::Axis(0), i); // [a,c]
    let g_ij = ovov_j.index_axis(ndarray::Axis(0), i); // [a,b]
    for a in 0..nv {
        let t1ia = t1[[i, a]];
        for b in 0..nv {
            let t1jb = t1[[j, b]];
            let gij_ab = g_ij[[a, b]];
            for c in 0..nv {
                v[[a, b, c]] += g_jk[[b, c]] * t1ia
                    + g_ik[[a, c]] * t1jb
                    + gij_ab * t1[[k, c]];
            }
        }
    }
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
    _mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    cc: &CcResult,
    cfg: &CcConfig,
) -> Result<f64, FerricError> {
    let t1 = cc
        .t1
        .as_ref()
        .ok_or_else(|| FerricError::General("closed-shell CCSD(T) requires T1 amplitudes".into()))?;

    let nbas = obs.nbasis();
    let nocc_total = rhf.eps_r().iter().filter(|&&e| e < 0.0).count();
    let first_occ = cfg.frozen_core;
    let no = active_occ(nocc_total, first_occ)?;
    let nv = nbas - nocc_total;

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

    let peak_triple = peak_triple_block_bytes(nv);
    let precomputed = precomputed_block_bytes(no, nv);
    let budget = ferric_core::memory::resolve_budget_bytes(cfg.memory_budget_bytes);
    // Precomputed blocks are allocated once and shared; the per-triple set is
    // per rayon worker. Charge one per-triple set here -- the chunk width below
    // is sized from the REMAINING budget, so concurrency is bounded separately
    // and must not be double-counted into this floor.
    let floor = precomputed.saturating_add(peak_triple);
    ferric_core::memory::check_alloc(
        &format!(
            "closed-shell CCSD(T) precomputed blocks + one per-triple block \
             (no={no}, nv={nv} spatial)"
        ),
        floor,
        budget,
    )?;

    let eps = rhf.eps_r();
    let c = rhf.mos_r();
    let c_occ = c.slice(ndarray::s![.., first_occ..nocc_total]).to_owned();
    let c_vir = c.slice(ndarray::s![.., nocc_total..]).to_owned();

    // Dressed RI 3-index MO blocks — identical construction to the CCSD
    // drivers, so the RI error cancels between CCSD and (T).
    let v2c = ferric_integrals::threeindex::coulomb_metric_2c(op, dfbs)?;
    let v_inv_sqrt = cholesky_inverse_sqrt(&v2c)?;
    let eri3_ao = ferric_integrals::threeindex::eri3_tensor(op, obs, dfbs)?;
    use Axis::{O, V};
    let b_ov = build_b(&transform_3center_ov(&eri3_ao, &c_occ, &c_vir), &v_inv_sqrt, O, V);
    let b_oo = build_b(&transform_3center_oo(&eri3_ao, &c_occ), &v_inv_sqrt, O, O);
    let b_vv = build_b(&transform_3center_vv(&eri3_ao, &c_vir), &v_inv_sqrt, V, V);
    drop(eri3_ao);

    // Chemist blocks (pq|rs) = Σ_P B^P_pq B^P_rs.
    let ovvv: Array4<f64> = einsum!("Pia,Pbd->iabd", &b_ov, &b_vv)
        .into_dimensionality()
        .expect("ovvv 4D");
    let ovoo: Array4<f64> = einsum!("Pia,Plj->ialj", &b_ov, &b_oo)
        .into_dimensionality()
        .expect("ovoo 4D");
    let ovov: Array4<f64> = einsum!("Pia,Pjb->iajb", &b_ov, &b_ov)
        .into_dimensionality()
        .expect("ovov 4D");
    let ints = TripleIntegrals { ovvv, ovoo, ovov };

    let eo: Vec<f64> = (0..no).map(|i| eps[first_occ + i]).collect();
    let ev: Vec<f64> = (0..nv).map(|a| eps[nocc_total + a]).collect();

    let t1_owned: Array2<f64> = t1.clone();
    let t2_owned: Array4<f64> = cc.t2.clone();

    // Banded parallel reduction over i<=j<=k. Chunk width is a pure function
    // of nv and the byte budget (never the thread count); chunk partials are
    // folded serially in ascending triple order, so `et` is bit-identical at
    // any RAYON_NUM_THREADS. See the module doc and the spin-orbital sibling's
    // matching comment block.
    let triples = occ_triples_with_repeats(no);
    // Band from what is LEFT after the precomputed blocks, not the full budget
    // (they are already resident here). Mirrors the spin-orbital sibling.
    let remaining = budget.saturating_sub(precomputed);
    let chunk_len = triple_chunk_len(
        nv,
        ferric_core::memory::transient_share(remaining, ferric_core::memory::Share::Half),
    );
    let mut et = 0.0f64;
    for chunk in triples.chunks(chunk_len) {
        let partials: Vec<f64> = with_blas_threads(opt_in_blas_threads(), || {
            chunk
                .par_iter()
                .map(|&(i, j, k)| {
                    let w = w_block(&ints, &t2_owned, no, nv, i, j, k);
                    let mut v = w.clone();
                    add_v_singles(&mut v, &ints, &t1_owned, nv, i, j, k);

                    // W̃ = 4W + W(bca) + W(cab) - 2W(cba) - 2W(acb) - 2W(bac),
                    // permutations acting on the virtual axes (PySCF `r3`).
                    let mut wt = 4.0 * &w;
                    wt += &permute3(&w, [1, 2, 0]);
                    wt += &permute3(&w, [2, 0, 1]);
                    wt.scaled_add(-2.0, &permute3(&w, [2, 1, 0]));
                    wt.scaled_add(-2.0, &permute3(&w, [0, 2, 1]));
                    wt.scaled_add(-2.0, &permute3(&w, [1, 0, 2]));

                    let e_ijk = eo[i] + eo[j] + eo[k];
                    let mult = occ_triple_multiplicity(i, j, k);
                    let mut partial = 0.0f64;
                    for a in 0..nv {
                        for b in 0..nv {
                            let dab = e_ijk - ev[a] - ev[b];
                            for cx in 0..nv {
                                let d = dab - ev[cx];
                                partial += wt[[a, b, cx]] * v[[a, b, cx]] / d;
                            }
                        }
                    }
                    mult * partial
                })
                .collect()
        });
        for p in partials {
            et += p;
        }
    }

    // Divisor 3 (NOT 6, NOT 36) — derived and measured, see the module doc's
    // "Combinatorial factor" section. The weighted i<=j<=k band above equals
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
        let dfbs =
            PreparedBasis::new(&mol, &basis::bundled("def2-qzvpp-rifit").unwrap()).unwrap();
        let op = Operator::coulomb();
        let ctx = ParallelContext::default();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(
            &ctx,
            &mol,
            &obs,
            op,
            &bounds,
            &RhfConfig { energy_conv: 1e-11, ..Default::default() },
        )
        .unwrap();
        Setup { mol, obs, dfbs, op, rhf }
    }

    fn cc_cfg() -> CcConfig {
        CcConfig { frozen_core: 0, max_iter: 100, energy_conv: 1e-10, ..Default::default() }
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
        assert!(t_new < -1e-4, "(T) = {t_new:.12} is not a plausible triples energy");
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
        let cfg = CcConfig { frozen_core: 1, max_iter: 100, energy_conv: 1e-10, ..Default::default() };
        let (t_old, t_new) = both_t_with(&s, cfg);
        let diff = (t_old - t_new).abs();
        println!(
            "H2O/cc-pVDZ frozen_core=1 (T): spin-orbital = {t_old:.12}, \
             closed-shell = {t_new:.12}, diff = {diff:.3e}"
        );
        assert!(t_new < -1e-4, "(T) = {t_new:.12} is not a plausible triples energy");
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
        assert!(t_new < -1e-6, "(T) = {t_new:.12} is not a plausible triples energy");
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
            let pool = rayon::ThreadPoolBuilder::new().num_threads(n).build().unwrap();
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

    /// Multiplicity weights must reproduce the unrestricted ordering count.
    /// This pins the piece the module doc calls out as the closed-shell-only
    /// trap (repeated spatial occupied indices are physical and carry >50% of
    /// the energy on the scratch test system).
    #[test]
    fn multiplicity_weights_sum_to_the_unrestricted_count() {
        for no in 1..6usize {
            let total: f64 = occ_triples_with_repeats(no)
                .iter()
                .map(|&(i, j, k)| occ_triple_multiplicity(i, j, k))
                .sum();
            assert_eq!(
                total,
                (no * no * no) as f64,
                "weighted i<=j<=k band must cover all no^3 ordered triples (no={no})"
            );
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
        let xyz_path =
            format!("{}/../../testdata/molecules/alkane_2.xyz", env!("CARGO_MANIFEST_DIR"));
        let mol = Molecule::load_xyz(&xyz_path).unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        let dfbs =
            PreparedBasis::new(&mol, &basis::bundled("def2-qzvpp-rifit").unwrap()).unwrap();
        let op = Operator::coulomb();
        let ctx = ParallelContext::default();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(
            &ctx,
            &mol,
            &obs,
            op,
            &bounds,
            &RhfConfig { energy_conv: 1e-11, ..Default::default() },
        )
        .unwrap();
        let s = Setup { mol, obs, dfbs, op, rhf };
        perf_ab(&s, "ethane/cc-pVDZ", 2);
    }
}
