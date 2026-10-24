//! DLPNO-MP2 pair screening on LOCALIZED (non-canonical) occupieds.
//!
//! [`crate::dlpno_mp2::dlpno_mp2_from_b_ov`] takes a diagonal `eps`, so it is
//! only valid for canonical occupieds, where the estimated pair energies of
//! distant pairs are not small and the pair screen retains ~every pair
//! (issue #475). This module lifts that restriction for the PAIR screen.
//!
//! # Formulation
//!
//! Rotate the occupieds, `C_loc = C_occ U`. The RI tensor rotates exactly,
//! `B'^P_{i'a} = sum_i U_{i i'} B^P_{ia}` ([`crate::dlpno_localized::rotate_b_ov_occupied`]), and the
//! occupied Fock block becomes the NON-diagonal `f = U^T diag(eps_occ) U`
//! ([`crate::dlpno_localized::localized_occupied_fock`]). Virtuals stay canonical (`eps_vir`). The
//! first-order amplitudes then satisfy
//!
//! ```text
//! K_ij^ab + (e_a + e_b) T_ij^ab - sum_k f_ik T_kj^ab - sum_k f_kj T_ik^ab = 0
//! ```
//!
//! solved by Jacobi iteration on the off-diagonal of `f` over the RETAINED
//! pairs only, starting from the diagonal-`f` amplitudes. Amplitudes stay in
//! the localized basis throughout (a direct diagonalization of `f` would undo
//! the localization and silently reproduce the canonical result).
//! `E = sum_ij sum_ab K_ij^ab (2 T_ij^ab - T_ij^ba)`.
//!
//! The pair screen estimates `e_ij` from the diagonal of `f` (the semicanonical
//! diagonal-f full-virtual pair energy), then retains pairs with `|e_ij| >= t_cut_pairs`.
//!
//! # Scope (honest limits)
//!
//! * Pair screening only. There is NO per-pair PNO/PAO virtual truncation
//!   here: the coupling sums amplitude blocks of different pairs, which needs
//!   one shared virtual basis. Virtuals are the full canonical set.
//! * Memory: `K` and `T` blocks (`nvir^2` each) are held for every retained
//!   pair (plus a Jacobi copy and two energy temporaries);
//!   [`crate::dlpno_localized::localized_working_bytes`] is checked against
//!   `memory_budget_bytes` before anything is allocated. The add-back runs
//!   after `K`/`T` are freed, in budget-sized chunks of pairs (see "Add-back").
//! * The screen itself costs O(no^2 nv^2 naux), the same as canonical MP2.
//! * Closed shell, library only. No speedup is claimed.
//!
//! # Add-back
//!
//! [`crate::dlpno_localized::LocalizedMp2Config::add_back_truncation`] adds the screened pairs'
//! estimated energy (the same diagonal-f estimate the screen ranked them by)
//! to the result. This is similar in spirit to the screened-pair corrections
//! production DLPNO codes report; it is not a reimplementation of any of them,
//! and the diagonal-f estimate is not a bound, so it can in principle
//! over-correct. Opt-in: `false` returns the retained-pair sum. Cost: one
//! `(ia|jb)` block per discarded pair, at most one extra screen pass. Under a
//! budget the discarded pairs are processed `cap / block` at a time, so at most
//! that many blocks are in flight whatever the thread count; each chunk's
//! per-pair terms are folded serially in pair order, so the result is
//! bit-identical at every chunk width. Measured
//! accuracy gain and its limits are in `tests/validation_dlpno_localized.rs`.
//!
//! # Exactness anchor
//!
//! With `t_cut_pairs = 0` every pair is retained and the energy equals the
//! canonical RI-MP2 energy (the energy is invariant to the occupied rotation).

use crate::boys::boys_localize;
use crate::pair_energy_screen::build_pair_domains_by_energy;
use crate::rimp2::SpinComponents;
use ferric_core::FerricError;
use ndarray::{s, Array2};
use rayon::prelude::*;
use std::ops::Range;

/// Settings for [`dlpno_mp2_localized_from_b_ov`].
#[derive(Debug, Clone)]
pub struct LocalizedMp2Config {
    /// Estimated-pair-energy threshold (Eh); `0.0` keeps every pair.
    pub t_cut_pairs: f64,
    /// Converged when `max |T_new - T_old| < conv_thresh`.
    pub conv_thresh: f64,
    /// Iteration cap; non-convergence is a hard error.
    pub max_iter: usize,
    /// Optional cap on the working set ([`crate::dlpno_localized::localized_working_bytes`]).
    pub memory_budget_bytes: Option<usize>,
    /// Add the screened pairs' diagonal-f estimated energy back into the result
    /// (see the module docs, "Add-back"). `false` returns the retained-pair sum
    /// only. With no pair screened the correction is exactly zero, so the
    /// `t_cut_pairs = 0` anchor is unchanged either way.
    pub add_back_truncation: bool,
}

impl Default for LocalizedMp2Config {
    fn default() -> Self {
        Self {
            t_cut_pairs: 1e-5,
            conv_thresh: 1e-10,
            max_iter: 200,
            memory_budget_bytes: None,
            add_back_truncation: false,
        }
    }
}

/// What a localized run kept and how the solve went.
#[derive(Debug, Clone)]
pub struct LocalizedMp2Diagnostics {
    /// Retained fraction of the unique `i <= j` pairs.
    pub pair_retention: f64,
    /// Retained unique pair count.
    pub n_pairs: usize,
    /// Jacobi iterations used.
    pub iterations: usize,
    /// Final `max |T_new - T_old|`.
    pub max_amplitude_change: f64,
    /// Sum of estimated `e_ij` over DISCARDED pairs (diagonal-f estimate).
    pub discarded_estimated_energy: f64,
    /// Energy added back into the returned components: the discarded pairs'
    /// estimate when [`LocalizedMp2Config::add_back_truncation`] is set, else `0.0`.
    pub truncation_correction: f64,
}

/// Bytes held by the solve for `n_pairs` retained pairs: `K`, `T` and the
/// Jacobi copy, each `nvir^2` doubles per pair, plus the two serial `nvir^2`
/// temporaries of the retained-pair energy loop (only when it runs).
pub fn localized_working_bytes(n_pairs: usize, nvir: usize) -> usize {
    let temporaries = if n_pairs > 0 { 2 } else { 0 };
    (3 * n_pairs + temporaries) * pair_block_bytes(nvir)
}

/// Bytes of one `nvir x nvir` pair block.
fn pair_block_bytes(nvir: usize) -> usize {
    nvir * nvir * std::mem::size_of::<f64>()
}

/// Pairs per add-back chunk: `cap / block` under a budget (at most `n_disc`,
/// at least 1), all of them in one chunk without one. Derived from the budget
/// only — never from the thread count — so admission and chunking are the
/// same on every pool.
///
/// # Errors
/// [`FerricError::General`] if there is something to add back but the budget
/// cannot hold even one block.
fn add_back_chunk(cap: Option<usize>, n_disc: usize, nvir: usize) -> Result<usize, FerricError> {
    let blk = pair_block_bytes(nvir);
    let Some(cap) = cap else {
        return Ok(n_disc.max(1));
    };
    if n_disc > 0 && cap < blk {
        return Err(FerricError::General(format!(
            "dlpno_mp2_localized: add-back needs one {blk} B block (nvir={nvir}) for \
             {n_disc} discarded pairs, budget is {cap} B"
        )));
    }
    Ok((cap / blk.max(1)).min(n_disc).max(1))
}

/// Rotate the occupied index of `b_ov`: `B'^P_{i'a} = sum_i U_{i i'} B^P_{ia}`.
///
/// # Errors
/// [`FerricError::General`] on mismatched shapes.
pub fn rotate_b_ov_occupied(
    b_ov: &Array2<f64>,
    u: &Array2<f64>,
    nocc: usize,
    nvir: usize,
) -> Result<Array2<f64>, FerricError> {
    if b_ov.ncols() != nocc * nvir || u.dim() != (nocc, nocc) {
        return Err(FerricError::General(format!(
            "rotate_b_ov_occupied: b_ov {:?}, u {:?}, expected width {} and ({nocc}, {nocc})",
            b_ov.dim(),
            u.dim(),
            nocc * nvir
        )));
    }
    let mut out = Array2::<f64>::zeros((b_ov.nrows(), nocc * nvir));
    for ip in 0..nocc {
        for i in 0..nocc {
            let uii = u[(i, ip)];
            if uii == 0.0 {
                continue;
            }
            let src = b_ov.slice(s![.., i * nvir..(i + 1) * nvir]);
            out.slice_mut(s![.., ip * nvir..(ip + 1) * nvir])
                .scaled_add(uii, &src);
        }
    }
    Ok(out)
}

/// Occupied Fock block in the rotated basis: `f = U^T diag(eps_occ) U`.
pub fn localized_occupied_fock(eps_occ: &[f64], u: &Array2<f64>) -> Array2<f64> {
    let mut ue = u.clone();
    for (i, mut row) in ue.rows_mut().into_iter().enumerate() {
        row *= eps_occ[i];
    }
    u.t().dot(&ue)
}

/// Boys-localize the active occupieds block by block (e.g. core and valence
/// separately, so the core is not mixed into the valence).
///
/// `c_occ` is `(nbas, nocc)`, `s_ao` the AO overlap, `dip` the AO dipole
/// matrices, `blocks` a partition of `0..nocc`. Returns the orthogonal
/// block-diagonal `U` (`C_loc = C_occ U`) and the `(nocc, 3)` Boys centers.
///
/// # Errors
/// [`FerricError::General`] if a block fails to converge or `blocks` is not a
/// partition of `0..nocc`.
pub fn localize_occupied_boys_blocks(
    c_occ: &Array2<f64>,
    s_ao: &Array2<f64>,
    dip: &[Array2<f64>; 3],
    blocks: &[Range<usize>],
) -> Result<(Array2<f64>, Array2<f64>), FerricError> {
    let nocc = c_occ.ncols();
    let mut covered = vec![false; nocc];
    for r in blocks {
        for i in r.clone() {
            if i >= nocc || covered[i] {
                return Err(FerricError::General(
                    "localize_occupied_boys_blocks: blocks must partition 0..nocc".into(),
                ));
            }
            covered[i] = true;
        }
    }
    if covered.iter().any(|c| !c) {
        return Err(FerricError::General(
            "localize_occupied_boys_blocks: blocks must partition 0..nocc".into(),
        ));
    }
    let mut u = Array2::<f64>::zeros((nocc, nocc));
    let mut centers = Array2::<f64>::zeros((nocc, 3));
    for r in blocks {
        let cb = c_occ.slice(s![.., r.clone()]).to_owned();
        let res = boys_localize(&cb, dip, 400);
        if !res.converged {
            return Err(FerricError::General(format!(
                "Boys localization of block {r:?} did not converge in {} sweeps",
                res.iterations
            )));
        }
        let ub = cb.t().dot(s_ao).dot(&res.c_loc);
        u.slice_mut(s![r.clone(), r.clone()]).assign(&ub);
        centers.slice_mut(s![r.clone(), ..]).assign(&res.centers);
    }
    Ok((u, centers))
}

/// Index of a unique pair; `(a, b)` with `a > b` resolves to the transpose.
struct PairIndex {
    nocc: usize,
    slot: Vec<Option<usize>>,
}

impl PairIndex {
    fn new(nocc: usize, pairs: &[(usize, usize)]) -> Self {
        let mut slot = vec![None; nocc * nocc];
        for (p, &(i, j)) in pairs.iter().enumerate() {
            slot[i * nocc + j] = Some(p);
        }
        Self { nocc, slot }
    }
    /// `(pair slot, transposed)` for the ordered pair `(a, b)`, if retained.
    fn get(&self, a: usize, b: usize) -> Option<(usize, bool)> {
        if a <= b {
            self.slot[a * self.nocc + b].map(|p| (p, false))
        } else {
            self.slot[b * self.nocc + a].map(|p| (p, true))
        }
    }
}

/// `(ia|jb)` for the pair, from the rotated RI tensor.
fn k_block(b: &Array2<f64>, i: usize, j: usize, nvir: usize) -> Array2<f64> {
    let bi = b.slice(s![.., i * nvir..(i + 1) * nvir]);
    let bj = b.slice(s![.., j * nvir..(j + 1) * nvir]);
    bi.t().dot(&bj)
}

fn diag_denominator(f: &Array2<f64>, ev: &[f64], i: usize, j: usize, a: usize, b: usize) -> f64 {
    f[(i, i)] + f[(j, j)] - ev[a] - ev[b]
}

/// One Jacobi update of pair `p = (i, j)` from the old amplitudes.
fn jacobi_update(
    p: usize,
    pairs: &[(usize, usize)],
    idx: &PairIndex,
    f: &Array2<f64>,
    ev: &[f64],
    k: &[Array2<f64>],
    t: &[Array2<f64>],
) -> Array2<f64> {
    let (i, j) = pairs[p];
    let nocc = f.nrows();
    let nvir = ev.len();
    // One block per call: accumulate into the output and divide in place.
    // Built element-wise (negation is exact) so the layout is standard
    // whatever `k[p]`'s is; same operations, same order as a separate
    // accumulator, hence bit-identical.
    let kp = &k[p];
    let mut out = Array2::from_shape_fn((nvir, nvir), |(a, b)| -kp[(a, b)]);
    for kk in 0..nocc {
        if kk != i {
            if let Some((q, tr)) = idx.get(kk, j) {
                add_partner(&mut out, f[(i, kk)], &t[q], tr);
            }
        }
        if kk != j {
            if let Some((q, tr)) = idx.get(i, kk) {
                add_partner(&mut out, f[(kk, j)], &t[q], tr);
            }
        }
    }
    for ((a, b), x) in out.indexed_iter_mut() {
        *x /= -diag_denominator(f, ev, i, j, a, b);
    }
    out
}

/// `out += w * T_q` (or `T_q^T` when the partner is stored transposed).
fn add_partner(out: &mut Array2<f64>, w: f64, tq: &Array2<f64>, transposed: bool) {
    if transposed {
        out.scaled_add(w, &tq.t());
    } else {
        out.scaled_add(w, tq);
    }
}

fn max_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()))
}

/// Estimated pair energies from the diagonal of `f` (no coupling).
fn estimate_pairs(
    b_loc: &Array2<f64>,
    f: &Array2<f64>,
    ev: &[f64],
    nocc: usize,
    nvir: usize,
) -> Result<crate::pair_energy_screen::PairEnergies, FerricError> {
    let mut eps = Vec::with_capacity(nocc + nvir);
    eps.extend((0..nocc).map(|i| f[(i, i)]));
    eps.extend_from_slice(ev);
    crate::dlpno_mp2::estimate_pair_energies_from_b_ov(b_loc, &eps, nocc, nvir, 0, nocc)
}

/// DLPNO-MP2 pair-screened MP2 on localized occupieds (module docs).
///
/// `b_loc` is the occupied-rotated `(naux, nocc*nvir)` RI tensor, `f_occ` the
/// `(nocc, nocc)` occupied Fock block in that basis, `eps_vir` the canonical
/// virtual energies and `centers` the `(nocc, 3)` orbital centers.
///
/// # Errors
/// [`FerricError::General`] on shape mismatch, a non-negative denominator,
/// exceeding `memory_budget_bytes`, or an unconverged solve.
pub fn dlpno_mp2_localized_from_b_ov(
    b_loc: &Array2<f64>,
    f_occ: &Array2<f64>,
    eps_vir: &[f64],
    centers: &Array2<f64>,
    cfg: &LocalizedMp2Config,
) -> Result<(SpinComponents, LocalizedMp2Diagnostics), FerricError> {
    let nocc = f_occ.nrows();
    let nvir = eps_vir.len();
    if f_occ.ncols() != nocc || b_loc.ncols() != nocc * nvir || centers.nrows() != nocc {
        return Err(FerricError::General(format!(
            "dlpno_mp2_localized: b_loc {:?}, f_occ {:?}, centers {:?} inconsistent with \
             nocc={nocc}, nvir={nvir}",
            b_loc.dim(),
            f_occ.dim(),
            centers.dim()
        )));
    }
    let pe = estimate_pairs(b_loc, f_occ, eps_vir, nocc, nvir)?;
    let domains = build_pair_domains_by_energy(centers, &pe, cfg.t_cut_pairs, f64::INFINITY)?;
    let pairs = domains.pairs.clone();
    let idx = PairIndex::new(nocc, &pairs);
    let mut discarded = 0.0;
    for i in 0..nocc {
        for j in i..nocc {
            if idx.get(i, j).is_none() {
                discarded += pe.e_ij(i, j);
            }
        }
    }
    let n_disc = nocc * (nocc + 1) / 2 - pairs.len();
    let chunk = if cfg.add_back_truncation {
        add_back_chunk(cfg.memory_budget_bytes, n_disc, nvir)?
    } else {
        1
    };
    if let Some(cap) = cfg.memory_budget_bytes {
        let need = localized_working_bytes(pairs.len(), nvir);
        if need > cap {
            return Err(FerricError::General(format!(
                "dlpno_mp2_localized: working set {need} B exceeds budget {cap} B \
                 ({} pairs, nvir={nvir})",
                pairs.len()
            )));
        }
    }
    let k: Vec<Array2<f64>> = pairs
        .par_iter()
        .map(|&(i, j)| k_block(b_loc, i, j, nvir))
        .collect();
    // Start: diagonal-f amplitudes.
    let mut t: Vec<Array2<f64>> = pairs
        .par_iter()
        .enumerate()
        .map(|(p, &(i, j))| {
            Array2::from_shape_fn((nvir, nvir), |(a, b)| {
                let d = diag_denominator(f_occ, eps_vir, i, j, a, b);
                k[p][(a, b)] / d
            })
        })
        .collect();
    let mut iterations = 0;
    let mut change = f64::INFINITY;
    while iterations < cfg.max_iter {
        let tn: Vec<Array2<f64>> = (0..pairs.len())
            .into_par_iter()
            .map(|p| jacobi_update(p, &pairs, &idx, f_occ, eps_vir, &k, &t))
            .collect();
        change = tn
            .iter()
            .zip(t.iter())
            .map(|(a, b)| max_abs_diff(a, b))
            .fold(0.0, f64::max);
        t = tn;
        iterations += 1;
        if change < cfg.conv_thresh {
            break;
        }
    }
    if change >= cfg.conv_thresh {
        return Err(FerricError::General(format!(
            "dlpno_mp2_localized: amplitudes unconverged after {iterations} iterations \
             (max change {change:.3e} >= {:.1e})",
            cfg.conv_thresh
        )));
    }
    let mut e_os = 0.0;
    let mut e_ss = 0.0;
    for (p, &(i, j)) in pairs.iter().enumerate() {
        let fac = if i == j { 1.0 } else { 2.0 };
        let kt = &k[p] * &t[p];
        let ks = &k[p] * &t[p].t();
        e_os += fac * kt.sum();
        e_ss += fac * (kt.sum() - ks.sum());
    }
    // Nothing below reads the retained blocks: free them before the add-back
    // so its chunks get the whole budget.
    drop(k);
    drop(t);
    let mut truncation_correction = 0.0;
    if cfg.add_back_truncation {
        let (c_os, c_ss) = discarded_spin_components(b_loc, f_occ, eps_vir, &idx, chunk);
        e_os += c_os;
        e_ss += c_ss;
        truncation_correction = c_os + c_ss;
    }
    Ok((
        SpinComponents {
            e_os,
            e_ss,
            e_total: e_os + e_ss,
        },
        LocalizedMp2Diagnostics {
            pair_retention: domains.pair_retention(),
            n_pairs: pairs.len(),
            iterations,
            max_amplitude_change: change,
            discarded_estimated_energy: discarded,
            truncation_correction,
        },
    ))
}

/// Opposite-spin / same-spin split of the diagonal-f estimate over the pairs
/// the screen DISCARDED (every unique `i <= j` absent from `idx`).
///
/// Same estimator, same mirror weight and same denominators as the screen
/// (`estimate_pairs`), so `os + ss` equals
/// [`LocalizedMp2Diagnostics::discarded_estimated_energy`] up to summation
/// order. Split into components so the add-back lands in `e_os`/`e_ss` and
/// `e_total` stays their sum. Cost: one `(ia|jb)` block per discarded pair,
/// built and dropped inside its closure; pairs run `chunk` at a time, so at
/// most `chunk` blocks are live. The per-pair terms are folded serially in
/// pair order, ONE left fold across all chunks (never per-chunk subtotals), so
/// the sum is bit-identical for every `chunk`.
fn discarded_spin_components(
    b_loc: &Array2<f64>,
    f: &Array2<f64>,
    ev: &[f64],
    idx: &PairIndex,
    chunk: usize,
) -> (f64, f64) {
    let nocc = f.nrows();
    let discarded: Vec<(usize, usize)> = (0..nocc)
        .flat_map(|i| (i..nocc).map(move |j| (i, j)))
        .filter(|&(i, j)| idx.get(i, j).is_none())
        .collect();
    let mut acc = (0.0, 0.0);
    for ch in discarded.chunks(chunk.max(1)) {
        let terms: Vec<(f64, f64)> = ch
            .par_iter()
            .map(|&(i, j)| discarded_pair_components(b_loc, f, ev, i, j))
            .collect();
        acc = terms.into_iter().fold(acc, |(a, b), (x, y)| (a + x, b + y));
    }
    acc
}

/// Mirror-weighted `(os, ss)` diagonal-f estimate of one discarded pair.
fn discarded_pair_components(
    b_loc: &Array2<f64>,
    f: &Array2<f64>,
    ev: &[f64],
    i: usize,
    j: usize,
) -> (f64, f64) {
    let nvir = ev.len();
    let fac = if i == j { 1.0 } else { 2.0 };
    let kb = k_block(b_loc, i, j, nvir);
    let (mut os, mut ss) = (0.0, 0.0);
    for a in 0..nvir {
        for b in 0..nvir {
            let d = diag_denominator(f, ev, i, j, a, b);
            let (kab, kba) = (kb[(a, b)], kb[(b, a)]);
            os += kab * kab / d;
            ss += kab * (kab - kba) / d;
        }
    }
    (fac * os, fac * ss)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::ShapeBuilder;

    /// The pre-change `jacobi_update`, verbatim: a separate `-&k[p]`
    /// accumulator plus a freshly built output block.
    fn jacobi_update_reference(
        p: usize,
        pairs: &[(usize, usize)],
        idx: &PairIndex,
        f: &Array2<f64>,
        ev: &[f64],
        k: &[Array2<f64>],
        t: &[Array2<f64>],
    ) -> Array2<f64> {
        let (i, j) = pairs[p];
        let nocc = f.nrows();
        let mut acc = -&k[p];
        for kk in 0..nocc {
            if kk != i {
                if let Some((q, tr)) = idx.get(kk, j) {
                    if tr {
                        acc.scaled_add(f[(i, kk)], &t[q].t());
                    } else {
                        acc.scaled_add(f[(i, kk)], &t[q]);
                    }
                }
            }
            if kk != j {
                if let Some((q, tr)) = idx.get(i, kk) {
                    if tr {
                        acc.scaled_add(f[(kk, j)], &t[q].t());
                    } else {
                        acc.scaled_add(f[(kk, j)], &t[q]);
                    }
                }
            }
        }
        let nvir = ev.len();
        Array2::from_shape_fn((nvir, nvir), |(a, b)| {
            acc[(a, b)] / -diag_denominator(f, ev, i, j, a, b)
        })
    }

    /// Deterministic values in (-1, 1) with full mantissas.
    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }

    #[test]
    fn jacobi_update_in_place_is_bit_identical_to_the_reference() {
        let (nocc, nvir) = (5, 7);
        let mut s = 0x5eed_u64;
        let mut f = Array2::<f64>::zeros((nocc, nocc));
        for i in 0..nocc {
            f[(i, i)] = -1.0 - 0.3 * i as f64 + 0.01 * lcg(&mut s);
            for j in 0..i {
                let x = 0.05 * lcg(&mut s);
                f[(i, j)] = x;
                f[(j, i)] = x;
            }
        }
        let ev: Vec<f64> = (0..nvir)
            .map(|a| 0.5 + 0.2 * a as f64 + 0.01 * lcg(&mut s))
            .collect();
        // A screened (incomplete) pair list, so both coupling branches see
        // present, absent and transposed partners.
        let pairs: Vec<(usize, usize)> = (0..nocc)
            .flat_map(|i| (i..nocc).map(move |j| (i, j)))
            .filter(|&(i, j)| !(i == 0 && j == 3) && !(i == 1 && j == 4))
            .collect();
        let idx = PairIndex::new(nocc, &pairs);
        let mut block = |fortran: bool| {
            let v: Vec<f64> = (0..nvir * nvir).map(|_| lcg(&mut s)).collect();
            if fortran {
                Array2::from_shape_vec((nvir, nvir).f(), v).unwrap()
            } else {
                Array2::from_shape_vec((nvir, nvir), v).unwrap()
            }
        };
        // Mixed layouts: the output must not inherit `k[p]`'s.
        let k: Vec<Array2<f64>> = (0..pairs.len()).map(|p| block(p % 2 == 1)).collect();
        let t: Vec<Array2<f64>> = (0..pairs.len()).map(|p| block(p % 3 == 0)).collect();
        let mut compared = 0;
        for p in 0..pairs.len() {
            let want = jacobi_update_reference(p, &pairs, &idx, &f, &ev, &k, &t);
            let got = jacobi_update(p, &pairs, &idx, &f, &ev, &k, &t);
            assert!(got.is_standard_layout(), "pair {p}: layout changed");
            for (x, y) in got.iter().zip(want.iter()) {
                assert_eq!(x.to_bits(), y.to_bits(), "pair {p}: {x:e} vs {y:e}");
                compared += 1;
            }
        }
        assert_eq!(compared, pairs.len() * nvir * nvir);
    }

    #[test]
    fn add_back_chunk_is_budget_derived_and_refuses_below_one_block() {
        let nvir = 6;
        let blk = pair_block_bytes(nvir);
        // No budget: one chunk (the pre-chunking behaviour).
        assert_eq!(add_back_chunk(None, 10, nvir).unwrap(), 10);
        // Below one block: refused, naming the sizes.
        let e = add_back_chunk(Some(blk - 1), 10, nvir)
            .unwrap_err()
            .to_string();
        assert!(
            e.contains(&blk.to_string()) && e.contains(&(blk - 1).to_string()),
            "{e}"
        );
        // Nothing to add back: never refused.
        assert_eq!(add_back_chunk(Some(blk - 1), 0, nvir).unwrap(), 1);
        // Exactly one block, a few blocks, and more than needed.
        assert_eq!(add_back_chunk(Some(blk), 10, nvir).unwrap(), 1);
        assert_eq!(add_back_chunk(Some(3 * blk + 1), 10, nvir).unwrap(), 3);
        assert_eq!(add_back_chunk(Some(100 * blk), 10, nvir).unwrap(), 10);
        // Thread-count independence of the width itself.
        let on = |n: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .unwrap()
                .install(|| add_back_chunk(Some(4 * blk), 10, nvir).unwrap())
        };
        assert_eq!(on(1), on(4));
    }
}
