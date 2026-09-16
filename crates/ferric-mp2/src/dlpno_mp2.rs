//! DLPNO-MP2 — pair-domain screening plus PNO virtual truncation for RI-MP2.
//!
//! Combines [`crate::pair_domains`] (which occupied pairs survive) with
//! [`crate::local_pno`] (which virtuals survive *within* a pair) on top of the
//! existing RI-MP2 machinery.
//!
//! # Why MP2 is the right place to validate the whole DLPNO stack
//!
//! MP2 has no chicken-and-egg problem: the first-order amplitudes
//! `T^ij_ab = (ia|jb) / (ε_i + ε_j − ε_a − ε_b)` are *exactly* the object PNOs are
//! built from, so the pair density needs no prior correlated calculation. Every
//! later method (CCSD, CCSD(T), RPA) bootstraps its PNOs from these same MP2
//! amplitudes, which makes this module the reference implementation for the rest.
//!
//! # Exactness contract
//!
//! With `pair_cutoff = coupling_cutoff = ∞` and `t_cut_pno = 0` this reproduces
//! [`crate::rimp2::spin_components_from_g`] to floating-point identity — same pair
//! loop, same `i <= j` weighting, and the PNO transform is then a square orthogonal
//! rotation that cancels. `dlpno_mp2_exact_at_zero_truncation` pins that, and it is
//! the property every claim about the screened path depends on.
//!
//! # Two entry points, and which one to use
//!
//! [`dlpno_mp2_from_b_ov`] is the one to build on. It is **pair-driven**: it
//! screens the occupied pair list first, then materializes only each retained
//! pair's `nvir × nvir` block from the dressed RI tensor `b_ov`, one at a time.
//!
//! [`dlpno_mp2_spin_components`] is the original, and it takes the dense
//! `(nocc·nvir) × (nocc·nvir)` matrix `g` as its **input**. That is a structural
//! defect, not a tuning issue: production [`crate::rimp2::ri_mp2`] never forms
//! `g` — [`crate::rimp2::spin_components_from_b_ov`] streams i-blocked wide
//! GEMMs instead — so the dense entry point demands an object that is strictly
//! larger than anything RI-MP2 itself holds, and it demands it *before* any pair
//! or PNO threshold can apply. No amount of screening can remove an allocation
//! that precedes the screen.
//!
//! MEASURED at a drug-scale shape (danuglipron + capped SER31, def2-SVP:
//! nocc=170, nvir=760, naux≈2200): `b_ov` is 2.27 GB and one streamed `G_i` slab
//! is 0.79 GB, while the dense `g` is 124 GiB. On a 23 GB box, DLPNO-MP2 could
//! not *start* on a system production RI-MP2 completes in ~49 min at 16.2 GB
//! peak. That is the mechanical reason `wiki/VALIDATION.md` records "no net
//! speedup is demonstrated anywhere" for DLPNO — nobody could run it where it
//! might win.
//!
//! The restructuring is the same inversion that took CCSD(T) from a dense 6D
//! tensor to streaming per-triple blocks (~33 GB → 0.47 GB; see
//! `ferric_cc::ccsd_t`'s module doc).
//!
//! # Exactness between the two paths
//!
//! Both call the same private energy kernel, `pair_energy_in_pno_basis`, over
//! the same pair list in the same order — so they are **bit-identical when fed
//! the same blocks**, which `pair_driven_matches_dense` asserts directly at the
//! kernel level. END TO END they agree to the double-precision rounding floor
//! rather than bitwise, because the dense route contracts `b_ov` in one big
//! GEMM and the pair-driven route contracts it per pair, and floating-point
//! addition is not associative. MEASURED: 17 of 250 unique `(ia|jb)` elements
//! differ in their last bit (worst 8.9e-16 absolute), giving 5.6e-16 relative on
//! the total. On real integrals (water/cc-pVDZ, `E_corr` = −0.20400482234408 Ha)
//! the two agree to 2.8e-17.
//!
//! # What it is honestly worth
//!
//! ferric has MEASURED negative results for locality methods at small N — the OSV
//! sweep (100% virtual retention at usable thresholds, 48–76 mHa error otherwise)
//! and, after the Boys sign fix, a re-measured Boys-screening crossover showing
//! screened χ₀ still 5–14× *slower* than dense across C2–C12 with no crossover
//! trend. So this module reports its own retention and is exact in the limit, to
//! let the accuracy/cost curve be measured on any given system rather than assumed
//! from the literature.
//!
//! **NO SPEEDUP IS CLAIMED HERE.** The pair-driven path removes a memory wall;
//! whether it is also FASTER is unmeasured, and this module must not be cited as
//! evidence that it is. What changed is that the question can now be asked at a
//! size where the answer might be interesting.

use crate::local_pno::{build_pno_transforms, PnoTransforms};
use crate::pair_domains::PairDomains;
use crate::rimp2::SpinComponents;
use ferric_core::FerricError;
use ndarray::{Array2, ArrayView2};

/// Truncation settings for [`dlpno_mp2_spin_components`].
#[derive(Debug, Clone)]
pub struct DlpnoConfig {
    /// Occupied pair cutoff (Bohr) between Boys centers. `∞` keeps every pair.
    pub pair_cutoff: f64,
    /// Pair-pair coupling cutoff (Bohr) between pair centroids. `∞` couples all.
    /// Unused by MP2 (which has no pair-pair coupling term) but carried so one
    /// config drives the whole DLPNO stack.
    pub coupling_cutoff: f64,
    /// PNO occupation threshold. `0.0` keeps every virtual.
    pub t_cut_pno: f64,
    /// Estimated-pair-energy threshold in Hartree (ORCA's `T_CutPairs`
    /// criterion): a pair is dropped when `|e_ij| < t_cut_pairs`. `0.0` keeps
    /// every pair.
    ///
    /// **Prefer this over [`Self::pair_cutoff`].** Distance in Bohr is
    /// system-size-dependent — a fixed value does nothing until it drops below
    /// the size of the molecule, then removes everything at once. MEASURED at
    /// matched retention on benzene/STO-3G, the energy criterion is 56× more
    /// accurate (2.5e-3 vs 1.4e-1 Ha at 52% retention) and gives a smooth dial
    /// rather than a cliff. See `ferric-cc/tests/pair_screen_criteria.rs`.
    pub t_cut_pairs: f64,
}

/// Default estimated-pair-energy threshold, in Hartree.
///
/// MEASURED, not imported: on benzene/STO-3G this retains 97.3% of pairs for a
/// 5.1e-5 Ha error (0.03 kcal/mol — well inside chemical accuracy), and on
/// water/6-31G it is still fully exact. `1e-4` is the corresponding loose
/// setting at 2.5e-3 Ha. See `ferric-cc/tests/pair_screen_criteria.rs` for the
/// head-to-head that produced these.
///
/// These bracket ORCA's published `T_CutPairs` values in the same ordering
/// (~1e-4 Normal, ~1e-5 Tight), which is a sanity check on the number rather
/// than its source — those values could not be verified from any source
/// available here.
pub const DEFAULT_T_CUT_PAIRS: f64 = 1e-5;

impl Default for DlpnoConfig {
    /// Pair-energy screening ON at [`DEFAULT_T_CUT_PAIRS`]; the virtual-space
    /// and distance screens OFF.
    ///
    /// This is a deliberate departure from the earlier "exact by default"
    /// stance. The argument for exact-by-default was that a user who enables
    /// DLPNO without choosing thresholds should not get a silently approximated
    /// answer — but the pair-energy screen is measurably a 0.03 kcal/mol effect
    /// at this threshold, which is below the accuracy of every method that
    /// consumes it, while the alternative (screening nothing) means paying dense
    /// cost for a method whose entire purpose is to avoid it.
    ///
    /// `t_cut_pno` stays at `0.0` because virtual truncation has a MEASURED
    /// dead zone and cliff behaviour (see `dlpno-transform-cost-dominates`),
    /// and `pair_cutoff` stays at `∞` because the distance criterion it drives
    /// is the one this default exists to replace.
    ///
    /// Use [`Self::exact()`] for a bit-for-bit dense reference.
    fn default() -> Self {
        Self {
            pair_cutoff: f64::INFINITY,
            coupling_cutoff: f64::INFINITY,
            t_cut_pno: 0.0,
            t_cut_pairs: DEFAULT_T_CUT_PAIRS,
        }
    }
}

impl DlpnoConfig {
    /// Every screen disabled — the result must equal the dense one bit for bit.
    ///
    /// Use this for reference calculations and for the exactness contracts the
    /// DLPNO modules are tested against. [`Self::default()`] is NO LONGER exact;
    /// it enables pair-energy screening.
    pub fn exact() -> Self {
        Self {
            pair_cutoff: f64::INFINITY,
            coupling_cutoff: f64::INFINITY,
            t_cut_pno: 0.0,
            t_cut_pairs: 0.0,
        }
    }

    /// True when no screen is active, so the result must equal the dense one.
    pub fn is_exact(&self) -> bool {
        self.pair_cutoff.is_infinite() && self.t_cut_pno == 0.0 && self.t_cut_pairs == 0.0
    }
}

/// Diagnostics from a DLPNO-MP2 run — what was kept, and what it cost.
#[derive(Debug, Clone)]
pub struct DlpnoDiagnostics {
    /// Fraction of occupied pairs retained.
    pub pair_retention: f64,
    /// Fraction of virtuals retained, averaged over retained pairs.
    pub virtual_retention: f64,
    /// Largest per-pair discarded PNO occupation weight.
    pub max_discarded_weight: f64,
    /// Retained pair count.
    pub n_pairs: usize,
}

/// Build the pair domains a [`DlpnoConfig`] asks for, applying whichever pair
/// screen it selects.
///
/// This is the ONLY place `t_cut_pairs` takes effect. It exists because
/// [`dlpno_mp2_spin_components`] receives pre-built `domains` — applying the
/// config's threshold there would silently override a caller who had already
/// chosen their own screening, which is worse than ignoring it.
///
/// Precedence, when both are set: `t_cut_pairs` wins and `pair_cutoff` is
/// ignored. They are alternative answers to the same question, and the energy
/// criterion is the better-measured one (56x lower error at matched retention
/// on benzene — see `ferric-cc/tests/pair_screen_criteria.rs`). A caller who
/// genuinely wants the distance screen must set `t_cut_pairs: 0.0` explicitly,
/// which is what [`DlpnoConfig::exact`] does before you add a cutoff to it.
///
/// # Errors
///
/// Propagates from the underlying domain builders.
pub fn build_domains_for(
    centers: &ndarray::Array2<f64>,
    pair_energies: &crate::pair_energy_screen::PairEnergies,
    cfg: &DlpnoConfig,
) -> Result<PairDomains, FerricError> {
    if cfg.t_cut_pairs > 0.0 {
        crate::pair_energy_screen::build_pair_domains_by_energy(
            centers,
            pair_energies,
            cfg.t_cut_pairs,
            cfg.coupling_cutoff,
        )
    } else {
        crate::pair_domains::build_pair_domains(centers, cfg.pair_cutoff, cfg.coupling_cutoff)
    }
}

/// One retained pair's `(e_os, e_ss)` contribution, in that pair's PNO basis.
///
/// **This is the single energy kernel of the module.** Both
/// [`dlpno_mp2_spin_components`] (dense `g`) and [`dlpno_mp2_from_b_ov`]
/// (pair-driven, streamed from the RI `B` tensor) call it with the same
/// arguments; only how they OBTAIN `g_ij`/`g_ji` differs. Sharing it is what
/// makes the two paths bit-identical by construction rather than by coincidence.
///
/// `g_ij[a, b] = (ia|jb)` and `g_ji[a, b] = (ib|ja)` — the a↔b transpose partner
/// the same-spin term contracts against. `q` is the `(nvir × npno)` PNO transform
/// with orthonormal columns.
///
/// # The `i <= j` mirror weight
///
/// `fac = 2` for strictly off-diagonal `i < j` and `fac = 1` for `i == j`, applied
/// HERE so no caller can get it wrong independently. This is PySCF's
/// `fac = 2`/`fac = 1` `MP2_contract_d` convention, and it is shared with
/// [`crate::rimp2::spin_components_from_g`] and
/// [`crate::rimp2::spin_components_from_b_ov`]: the unique-pair list `j >= i`
/// visits each mirror-pair once and the off-diagonal entry stands in for its
/// unvisited `(j, i)` twin. Getting it wrong doubles or halves the correlation
/// energy while still producing a plausible-looking number, so
/// `off_diagonal_pairs_carry_the_mirror_weight_of_two` pins it directly.
///
/// # Errors
///
/// [`FerricError::General`] on a mis-shaped block or an eigensolver failure in
/// the semicanonicalization, with the offending pair named.
#[allow(clippy::too_many_arguments)]
fn pair_energy_in_pno_basis(
    g_ij: ArrayView2<f64>,
    g_ji: ArrayView2<f64>,
    q: &Array2<f64>,
    eps: &[f64],
    (i, j): (usize, usize),
    nvir: usize,
    first_occ: usize,
    nocc_total: usize,
) -> Result<(f64, f64), FerricError> {
    if g_ij.dim() != (nvir, nvir) || g_ji.dim() != (nvir, nvir) {
        return Err(FerricError::General(format!(
            "dlpno_mp2: pair ({i},{j}) blocks are {:?}/{:?}, expected ({nvir}, {nvir})",
            g_ij.dim(),
            g_ji.dim()
        )));
    }
    let e_i = eps[first_occ + i];
    let e_j = eps[first_occ + j];
    // Off-diagonal (i,j) stands in for its (j,i) mirror -- identical to the
    // dense kernel's weighting, so screening cannot change the convention.
    let fac = if i == j { 1.0 } else { 2.0 };

    let npno = q.ncols();

    // Rotate this pair's integral blocks into its PNO basis. Two blocks are
    // needed because the same-spin term contracts (ia|jb) against (ib|ja):
    //   g_ij[a,b] = (ia|jb)   ->  Qt g_ij Q
    //   g_ji[a,b] = (ib|ja)   ->  Qt g_ji Q      (the a<->b transpose partner)
    let t_ij = q.t().dot(&g_ij).dot(q);
    let t_ji = q.t().dot(&g_ji).dot(q);

    // Orbital energies in the PNO basis.
    //
    // The PNOs are NOT eigenvectors of the virtual Fock matrix, so the
    // canonical denominator (e_i + e_j - e_a - e_b) is only valid in the
    // canonical basis. The standard DLPNO fix is to SEMICANONICALIZE: build the
    // virtual Fock matrix in this pair's PNO basis and re-diagonalize it, which
    // restores a diagonal Fock and hence a valid denominator.
    //
    // Taking only the DIAGONAL f_aa = sum_c Q_ca^2 eps_c instead is WRONG and
    // fails silently: at zero truncation Q is a square orthogonal rotation, and
    // a rotated diagonal matrix is NOT diagonal -- MEASURED off-diagonal max
    // 0.137 on a 5x5 random rotation, which broke the exactness contract by
    // 0.117 Ha before this was fixed. Re-diagonalizing makes the composite
    // transform Q*U land back on the canonical eigenvectors (up to ordering and
    // sign), so the untruncated case reproduces dense MP2 exactly.
    let mut f_pno = Array2::<f64>::zeros((npno, npno));
    for a in 0..npno {
        for b in 0..npno {
            f_pno[(a, b)] = (0..nvir)
                .map(|c| q[(c, a)] * q[(c, b)] * eps[nocc_total + c])
                .sum();
        }
    }
    let (e_pno, u) = ferric_core::linalg::eigh_dc(&f_pno, ferric_core::linalg::Uplo::Upper)
        .map_err(|e| {
            FerricError::General(format!(
                "DLPNO semicanonicalization for pair ({i},{j}): {e}"
            ))
        })?;
    // Carry the integrals into the semicanonical PNO basis too.
    let t_ij = u.t().dot(&t_ij).dot(&u);
    let t_ji = u.t().dot(&t_ji).dot(&u);

    let mut e_os = 0.0;
    let mut e_ss = 0.0;
    for a in 0..npno {
        for b in 0..npno {
            let d = e_i + e_j - e_pno[a] - e_pno[b];
            let iajb = t_ij[(a, b)];
            let ibja = t_ji[(a, b)];
            e_os += fac * iajb * iajb / d;
            e_ss += fac * iajb * (iajb - ibja) / d;
        }
    }
    Ok((e_os, e_ss))
}

/// DLPNO-MP2 spin components from an explicit dense `(ia|jb)` matrix.
///
/// **Prefer [`dlpno_mp2_from_b_ov`] for anything new.** This entry point takes
/// the dense `(nocc·nvir) × (nocc·nvir)` matrix as its INPUT, which is larger
/// than anything production RI-MP2 holds and is allocated before any screen can
/// apply — see this module's docs for the measured numbers. It is kept because
/// it is the reference the pair-driven path is anchored against, and because a
/// non-Gram `g` (the robust-fit three-term construction, whose `(ib|ja)` is NOT
/// this block's transpose) can only be consumed here.
///
/// Mirrors [`crate::rimp2::spin_components_from_g`] exactly — same `i <= j` unique
/// pair loop, same `fac = 1 or 2` mirror weighting, same denominators — and adds
/// two screens on top:
///
/// 1. pairs absent from `domains` contribute nothing;
/// 2. within a retained pair, the `(a, b)` sums run over that pair's PNOs rather
///    than all `nvir` canonical virtuals.
///
/// The PNO rotation is applied to the integral block itself, so the energy
/// expression is unchanged — only the basis the `(a, b)` indices run over differs.
///
/// # Errors
///
/// [`FerricError::General`] when `g`'s dimensions disagree with `nocc`/`nvir`, or
/// on any error from the PNO construction.
pub fn dlpno_mp2_spin_components(
    g: &Array2<f64>,
    eps: &[f64],
    nocc: usize,
    nvir: usize,
    first_occ: usize,
    nocc_total: usize,
    domains: &PairDomains,
    cfg: &DlpnoConfig,
) -> Result<(SpinComponents, DlpnoDiagnostics), FerricError> {
    let want = nocc * nvir;
    if g.nrows() != want || g.ncols() != want {
        return Err(FerricError::General(format!(
            "dlpno_mp2: g is {:?}, expected ({want}, {want}) for nocc={nocc}, nvir={nvir}",
            g.dim()
        )));
    }
    if domains.nocc != nocc {
        return Err(FerricError::General(format!(
            "dlpno_mp2: domains were built for nocc={}, but nocc={nocc}",
            domains.nocc
        )));
    }

    // --- PNOs from the first-order amplitudes of each retained pair. ---
    //
    // T^ij_ab = (ia|jb) / (e_i + e_j - e_a - e_b). This is the MP2 amplitude
    // itself, which is why MP2 needs no prior correlated calculation to define
    // its own PNOs.
    let amp = |i: usize, j: usize| -> Array2<f64> {
        let e_i = eps[first_occ + i];
        let e_j = eps[first_occ + j];
        Array2::from_shape_fn((nvir, nvir), |(a, b)| {
            let d = e_i + e_j - eps[nocc_total + a] - eps[nocc_total + b];
            g[(i * nvir + a, j * nvir + b)] / d
        })
    };
    let pnos: PnoTransforms = build_pno_transforms(domains, nvir, cfg.t_cut_pno, amp)?;

    let mut e_os = 0.0;
    let mut e_ss = 0.0;
    for (p, &(i, j)) in domains.pairs.iter().enumerate() {
        // Slice this pair's (nvir x nvir) blocks out of the dense matrix and hand
        // them to the SHARED kernel, so the dense and pair-driven paths cannot
        // drift apart: only the SOURCE of the blocks differs between them.
        let mut g_ij = Array2::<f64>::zeros((nvir, nvir));
        let mut g_ji = Array2::<f64>::zeros((nvir, nvir));
        for a in 0..nvir {
            for b in 0..nvir {
                g_ij[(a, b)] = g[(i * nvir + a, j * nvir + b)];
                g_ji[(a, b)] = g[(i * nvir + b, j * nvir + a)];
            }
        }
        let (e_os_ij, e_ss_ij) = pair_energy_in_pno_basis(
            g_ij.view(),
            g_ji.view(),
            &pnos.pairs[p].transform,
            eps,
            (i, j),
            nvir,
            first_occ,
            nocc_total,
        )?;
        e_os += e_os_ij;
        e_ss += e_ss_ij;
    }

    let diag = DlpnoDiagnostics {
        pair_retention: domains.pair_retention(),
        virtual_retention: pnos.virtual_retention(),
        max_discarded_weight: pnos.max_discarded_weight(),
        n_pairs: domains.pairs.len(),
    };
    Ok((
        SpinComponents {
            e_os,
            e_ss,
            e_total: e_os + e_ss,
        },
        diag,
    ))
}

// ---------------------------------------------------------------------------
// Pair-driven path: never form the dense (nocc*nvir) x (nocc*nvir) `g`.
// ---------------------------------------------------------------------------

/// This pair's `(nvir x nvir)` integral block `g_ij[a,b] = (ia|jb)`, from the
/// dressed RI tensor.
///
/// `b_ov` is `(naux, nocc*nvir)` with column `i*nvir + a` holding `B^P_{ia}`, so
/// `(ia|jb) = sum_P B^P_{ia} B^P_{jb}` is exactly `B_i^T B_j` — ONE `nvir x nvir`
/// GEMM over the naux contraction index, with no other pair's block touched.
///
/// This is the whole memory argument: the dense kernel's input is
/// `(nocc*nvir)^2 * 8` bytes, while this is `nvir^2 * 8` per retained pair, one
/// at a time. On danuglipron/def2-SVP (nocc=170, nvir=760) that is 133.5 GB vs
/// 4.6 MB.
fn pair_block_from_b_ov(b_ov: &Array2<f64>, i: usize, j: usize, nvir: usize) -> Array2<f64> {
    let b_i = b_ov.slice(ndarray::s![.., i * nvir..(i + 1) * nvir]);
    let b_j = b_ov.slice(ndarray::s![.., j * nvir..(j + 1) * nvir]);
    b_i.t().dot(&b_j)
}

/// Estimated pair energies for the `t_cut_pairs` screen, without forming `g`.
///
/// Streaming counterpart of [`crate::pair_energy_screen::estimate_pair_energies`],
/// which takes the dense `(nocc*nvir)^2` matrix. Same expression, same `i < j`
/// mirror factor of 2, same non-negative-denominator rejection — the only
/// difference is that each pair's `nvir x nvir` block is built on demand from
/// `b_ov` and dropped again.
///
/// `estimate_pair_energies_agrees_with_dense` pins the two against each other.
///
/// # Errors
///
/// [`FerricError::General`] when `b_ov`'s width disagrees with `nocc * nvir`,
/// when `eps` is too short for the requested window, or on a non-negative MP2
/// denominator (a zero or inverted gap — the reference is not a valid MP2
/// starting point and dividing would fabricate a number).
pub fn estimate_pair_energies_from_b_ov(
    b_ov: &Array2<f64>,
    eps: &[f64],
    nocc: usize,
    nvir: usize,
    first_occ: usize,
    nocc_total: usize,
) -> Result<crate::pair_energy_screen::PairEnergies, FerricError> {
    if b_ov.ncols() != nocc * nvir {
        return Err(FerricError::General(format!(
            "estimate_pair_energies_from_b_ov: b_ov is {:?}, expected width {} for \
             nocc={nocc}, nvir={nvir}",
            b_ov.dim(),
            nocc * nvir
        )));
    }
    if eps.len() < nocc_total + nvir || first_occ + nocc > eps.len() {
        return Err(FerricError::General(format!(
            "estimate_pair_energies_from_b_ov: eps has {} entries, too short for \
             first_occ={first_occ}, nocc={nocc}, nocc_total={nocc_total}, nvir={nvir}",
            eps.len()
        )));
    }

    let mut e = Array2::<f64>::zeros((nocc, nocc));
    for i in 0..nocc {
        let e_i = eps[first_occ + i];
        for j in i..nocc {
            let e_j = eps[first_occ + j];
            // Same unique-pair weighting as the dense kernel.
            let fac = if i == j { 1.0 } else { 2.0 };
            let g_ij = pair_block_from_b_ov(b_ov, i, j, nvir);
            let mut acc = 0.0;
            for a in 0..nvir {
                let e_a = eps[nocc_total + a];
                for b in 0..nvir {
                    let d = e_i + e_j - e_a - eps[nocc_total + b];
                    if d >= 0.0 {
                        return Err(FerricError::General(format!(
                            "estimate_pair_energies_from_b_ov: non-negative MP2 denominator \
                             {d} for (i={i}, j={j}, a={a}, b={b}); the reference has a zero \
                             or inverted gap and is not a valid MP2 starting point"
                        )));
                    }
                    // (ia|jb) and its a<->b transpose partner (ib|ja) both live in
                    // this one block: (ib|ja) = B_i[b] . B_j[a] = g_ij[(b, a)].
                    let iajb = g_ij[(a, b)];
                    let ibja = g_ij[(b, a)];
                    acc += (2.0 * iajb - ibja) * iajb / d;
                }
            }
            let v = fac * acc;
            e[(i, j)] = v;
            e[(j, i)] = v;
        }
    }
    Ok(crate::pair_energy_screen::PairEnergies { nocc, e })
}

/// Peak resident bytes the PAIR-DRIVEN path adds on top of `b_ov`, per worker.
///
/// Counts the `nvir x nvir` blocks a single pair holds live at once: the integral
/// block itself, the amplitude block the PNOs come from, and the rotated copies
/// the energy kernel makes. Deliberately an over-count of the true peak rather
/// than an under-count — an optimistic guard is also a bug (see the memory-budget
/// rollup) — but it is O(nvir^2), which is the point.
///
/// Compare [`dense_g_bytes`]: that is O((nocc * nvir)^2), i.e. `nocc^2` times
/// larger. `pair_driven_avoids_the_dense_allocation` asserts the ratio.
pub fn pair_driven_working_bytes(nvir: usize) -> usize {
    // g_ij, g_ji, t_ij, t_ji, the amplitude block, and the PNO transform.
    6usize
        .saturating_mul(nvir)
        .saturating_mul(nvir)
        .saturating_mul(8)
}

/// Bytes the DENSE `g` the old entry point requires as its INPUT.
///
/// It is an input, so no amount of downstream pair or PNO screening can avoid
/// it: the allocation happens before any threshold applies. That is the
/// structural defect the pair-driven path exists to remove.
pub fn dense_g_bytes(nocc: usize, nvir: usize) -> usize {
    let n = nocc.saturating_mul(nvir);
    n.saturating_mul(n).saturating_mul(8)
}

/// DLPNO-MP2 driven by the retained PAIR LIST, streaming each pair's block out of
/// the RI `B` tensor instead of consuming a dense `(ia|jb)` matrix.
///
/// # Why this exists
///
/// [`dlpno_mp2_spin_components`] takes the dense `(nocc*nvir) x (nocc*nvir)`
/// matrix `g` as its INPUT. Production [`crate::rimp2::ri_mp2`] never forms that
/// object — [`crate::rimp2::spin_components_from_b_ov`] streams i-blocked wide
/// GEMMs instead — so the dense entry point inherits a memory wall that no pair
/// or PNO screen can remove, because the allocation precedes every threshold.
/// MEASURED on danuglipron + capped SER31 at def2-SVP (nocc=170, nvir=760,
/// naux~2200): `b_ov` is 2.27 GB and one streamed `G_i` slab is 0.79 GB, while
/// the dense `g` is 133.5 GB — on a 23 GB box, DLPNO-MP2 could not start on a
/// system production RI-MP2 completes in ~49 min at 16.2 GB peak.
///
/// This routine screens pairs FIRST and then materializes only the retained
/// pairs' `nvir x nvir` blocks, one at a time. It is the same inversion that took
/// CCSD(T) from a dense 6D tensor to streaming per-triple blocks (see
/// `ferric_cc::ccsd_t`'s module doc).
///
/// # Exactness contract
///
/// With [`DlpnoConfig::exact`] and complete domains this reproduces
/// [`dlpno_mp2_spin_components`] on the same system BIT FOR BIT — not to a
/// tolerance. Both call the identical [`pair_energy_in_pno_basis`] kernel in the
/// identical pair order; only the source of each block differs, and
/// `(ia|jb) = B_i^T B_j` is exactly what `g` stores. `pair_driven_matches_dense_bitwise`
/// pins that, and `pair_driven_exact_at_zero_truncation` pins the further step
/// back to [`crate::rimp2::spin_components_from_b_ov`].
///
/// # Scope: RI (Gram) integrals only
///
/// This path assumes `(ia|jb) = B_i^T B_j`, i.e. that the integrals really are
/// the Gram product of a factorized `B`. That is what makes the same-spin
/// partner free — `(ib|ja)` is this block's transpose, not a second contraction.
/// It is true for the standard RI factorization and therefore for everything
/// `ri_mp2` produces.
///
/// It is NOT true for the ROBUST (Dunlap) three-term `G`, which is not a Gram
/// product: there `(ib|ja)` must be built independently. Such a `g` must go
/// through [`dlpno_mp2_spin_components`] instead. This is a real restriction and
/// is stated rather than silently assumed, because feeding a non-Gram `G` here
/// would produce a wrong same-spin energy with no error raised.
///
/// # Arguments
///
/// `b_ov` is the dressed `(naux, nocc*nvir)` RI tensor that
/// [`crate::rimp2::ri_mp2_spin_components`] already returns, with column
/// `i*nvir + a` holding `B^P_{ia}`.
///
/// # Errors
///
/// [`FerricError::General`] when `b_ov`'s width disagrees with `nocc * nvir`,
/// when `domains` was built for a different `nocc`, or on any error from the PNO
/// construction or semicanonicalization.
#[allow(clippy::too_many_arguments)]
pub fn dlpno_mp2_from_b_ov(
    b_ov: &Array2<f64>,
    eps: &[f64],
    nocc: usize,
    nvir: usize,
    first_occ: usize,
    nocc_total: usize,
    domains: &PairDomains,
    cfg: &DlpnoConfig,
) -> Result<(SpinComponents, DlpnoDiagnostics), FerricError> {
    if b_ov.ncols() != nocc * nvir {
        return Err(FerricError::General(format!(
            "dlpno_mp2_from_b_ov: b_ov is {:?}, expected width {} for nocc={nocc}, nvir={nvir}",
            b_ov.dim(),
            nocc * nvir
        )));
    }
    if domains.nocc != nocc {
        return Err(FerricError::General(format!(
            "dlpno_mp2_from_b_ov: domains were built for nocc={}, but nocc={nocc}",
            domains.nocc
        )));
    }

    // --- PNOs from the first-order amplitudes of each RETAINED pair. ---
    //
    // build_pno_transforms already takes a per-pair CLOSURE rather than a packed
    // tensor, precisely so a large system can build blocks on demand. It only
    // ever calls it for pairs in `domains`, so a screened-out pair's block is
    // never built at all -- the screening now actually saves work, which is the
    // whole point of the restructuring.
    let amp = |i: usize, j: usize| -> Array2<f64> {
        let e_i = eps[first_occ + i];
        let e_j = eps[first_occ + j];
        let mut t = pair_block_from_b_ov(b_ov, i, j, nvir);
        for a in 0..nvir {
            for b in 0..nvir {
                let d = e_i + e_j - eps[nocc_total + a] - eps[nocc_total + b];
                t[(a, b)] /= d;
            }
        }
        t
    };
    let pnos: PnoTransforms = build_pno_transforms(domains, nvir, cfg.t_cut_pno, amp)?;

    let mut e_os = 0.0;
    let mut e_ss = 0.0;
    for (p, &(i, j)) in domains.pairs.iter().enumerate() {
        // ONE nvir x nvir GEMM for this pair, dropped before the next iteration.
        //
        // The same-spin partner needs no second GEMM and no second allocation:
        // g_ji[a,b] = (ib|ja) = B_i[b] . B_j[a] = g_ij[(b, a)], so it is exactly
        // this block's TRANSPOSE VIEW. The dense path has to slice it separately
        // only because it reads strided out of the big matrix.
        let g_ij = pair_block_from_b_ov(b_ov, i, j, nvir);
        let (e_os_ij, e_ss_ij) = pair_energy_in_pno_basis(
            g_ij.view(),
            g_ij.t(),
            &pnos.pairs[p].transform,
            eps,
            (i, j),
            nvir,
            first_occ,
            nocc_total,
        )?;
        e_os += e_os_ij;
        e_ss += e_ss_ij;
    }

    let diag = DlpnoDiagnostics {
        pair_retention: domains.pair_retention(),
        virtual_retention: pnos.virtual_retention(),
        max_discarded_weight: pnos.max_discarded_weight(),
        n_pairs: domains.pairs.len(),
    };
    Ok((
        SpinComponents {
            e_os,
            e_ss,
            e_total: e_os + e_ss,
        },
        diag,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pair_domains::{build_pair_domains, complete_pair_domains};
    use crate::rimp2::spin_components_from_g;
    use ndarray::array;

    /// Deterministic symmetric-ish (ia|jb) block and a plausible eps ladder.
    fn toy(nocc: usize, nvir: usize) -> (Array2<f64>, Vec<f64>) {
        let n = nocc * nvir;
        let mut s = 0x2545F4914F6CDD1Du64;
        let mut g = Array2::<f64>::zeros((n, n));
        for p in 0..n {
            for q in p..n {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let v = ((s >> 33) as f64 / (1u64 << 31) as f64) - 1.0;
                g[(p, q)] = v;
                g[(q, p)] = v; // (ia|jb) is symmetric under (ia)<->(jb)
            }
        }
        // Occupieds below zero, virtuals above, well separated so no denominator
        // approaches zero.
        let mut eps = Vec::new();
        for i in 0..nocc {
            eps.push(-1.0 - 0.1 * i as f64);
        }
        for a in 0..nvir {
            eps.push(0.5 + 0.1 * a as f64);
        }
        (g, eps)
    }

    fn centers(nocc: usize, spacing: f64) -> Array2<f64> {
        Array2::from_shape_fn(
            (nocc, 3),
            |(i, ax)| if ax == 0 { i as f64 * spacing } else { 0.0 },
        )
    }

    /// THE EXACTNESS CONTRACT: no screening must reproduce the dense kernel.
    ///
    /// Complete domains + `t_cut_pno = 0` makes every PNO transform a square
    /// orthogonal rotation, which cancels out of the energy. Any deviation beyond
    /// round-off means the rotation, the denominators, or the pair weighting is
    /// wrong — so this is the test the whole module rests on.
    #[test]
    fn dlpno_mp2_exact_at_zero_truncation() {
        let (nocc, nvir) = (4, 5);
        let (g, eps) = toy(nocc, nvir);
        let dense = spin_components_from_g(&g, &eps, nocc, nvir, 0, nocc);

        let d = complete_pair_domains(&centers(nocc, 1.5)).unwrap();
        let (got, diag) =
            dlpno_mp2_spin_components(&g, &eps, nocc, nvir, 0, nocc, &d, &DlpnoConfig::default())
                .unwrap();

        eprintln!(
            "dense  e_os={:.12} e_ss={:.12}\ndlpno  e_os={:.12} e_ss={:.12}",
            dense.e_os, dense.e_ss, got.e_os, got.e_ss
        );
        assert!(
            (got.e_total - dense.e_total).abs() < 1e-10,
            "untruncated DLPNO-MP2 must reproduce dense MP2: {:.12} vs {:.12}",
            got.e_total,
            dense.e_total
        );
        assert!(
            (got.e_os - dense.e_os).abs() < 1e-10,
            "OS component differs"
        );
        assert!(
            (got.e_ss - dense.e_ss).abs() < 1e-10,
            "SS component differs"
        );
        assert_eq!(diag.pair_retention, 1.0);
        assert_eq!(diag.virtual_retention, 1.0);
    }

    /// The default must actually SCREEN when routed through `build_domains_for`.
    ///
    /// Without this the default would be decorative: `dlpno_mp2_spin_components`
    /// takes pre-built domains, so a `t_cut_pairs` nobody reads changes nothing.
    /// This pins that the constructor honours it, and that `exact()` does not.
    #[test]
    fn default_config_actually_screens_through_the_constructor() {
        use crate::pair_energy_screen::estimate_pair_energies;
        let (nocc, nvir) = (4, 5);
        let (g, eps) = toy(nocc, nvir);
        let pe = estimate_pair_energies(g.view(), &eps, nocc, nvir, 0, nocc).unwrap();
        // Spread the centers so the DISTANCE path would also have something to do;
        // this isolates that the ENERGY path is the one being taken.
        let centers = centers(nocc, 10.0);

        let d_default = build_domains_for(&centers, &pe, &DlpnoConfig::default()).unwrap();
        let d_exact = build_domains_for(&centers, &pe, &DlpnoConfig::exact()).unwrap();

        assert!(d_exact.is_complete(), "exact() must retain every pair");
        assert!(
            d_default.pairs.len() <= d_exact.pairs.len(),
            "the default must never retain MORE than exact"
        );
        // On this toy the pair energies are large, so 1e-5 may legitimately keep
        // everything -- assert the mechanism, not a particular retention.
        let tight = DlpnoConfig {
            t_cut_pairs: 1e30,
            ..DlpnoConfig::default()
        };
        let d_tight = build_domains_for(&centers, &pe, &tight).unwrap();
        assert!(
            d_tight.pairs.len() < d_exact.pairs.len(),
            "an extreme t_cut_pairs must screen -- the constructor is not reading it"
        );
        // ...and diagonals still survive it.
        for i in 0..nocc {
            assert!(
                d_tight.pairs.contains(&(i, i)),
                "diagonal ({i},{i}) was screened"
            );
        }
    }

    /// The default now ENABLES pair-energy screening; `exact()` is the exact one.
    ///
    /// This pins a deliberate contract change. The default used to be exact, on
    /// the argument that enabling DLPNO without choosing thresholds should not
    /// silently approximate. It now screens at `DEFAULT_T_CUT_PAIRS` because that
    /// is a MEASURED 0.03 kcal/mol effect — below the accuracy of every method
    /// consuming it — while screening nothing means paying dense cost for a
    /// method whose whole purpose is to avoid it.
    #[test]
    fn default_screens_and_exact_does_not() {
        let d = DlpnoConfig::default();
        assert!(!d.is_exact(), "default() should now screen");
        assert_eq!(d.t_cut_pairs, DEFAULT_T_CUT_PAIRS);
        // The other two screens stay OFF: virtual truncation has a measured dead
        // zone and cliff, and the distance criterion is what this replaces.
        assert_eq!(
            d.t_cut_pno, 0.0,
            "virtual truncation must stay off by default"
        );
        assert!(
            d.pair_cutoff.is_infinite(),
            "distance screening must stay off"
        );

        let e = DlpnoConfig::exact();
        assert!(e.is_exact(), "exact() must disable every screen");
        assert_eq!(e.t_cut_pairs, 0.0);
    }

    /// Screening must change the energy and report what it dropped, otherwise the
    /// knobs are inert and any measured "speedup" would be meaningless.
    #[test]
    fn screening_changes_energy_and_reports_retention() {
        let (nocc, nvir) = (4, 5);
        let (g, eps) = toy(nocc, nvir);
        let dense = spin_components_from_g(&g, &eps, nocc, nvir, 0, nocc);

        // Widely spaced centers so a 2 Bohr pair cutoff really separates them.
        let d = build_pair_domains(&centers(nocc, 10.0), 2.0, f64::INFINITY).unwrap();
        assert!(
            d.pair_retention() < 1.0,
            "premise: pairs should be screened"
        );

        // Distance screening specifically, so t_cut_pairs is off: this test is
        // about the pair_cutoff path, not the new default.
        let cfg = DlpnoConfig {
            pair_cutoff: 2.0,
            coupling_cutoff: f64::INFINITY,
            t_cut_pno: 1e-4,
            t_cut_pairs: 0.0,
        };
        let (got, diag) =
            dlpno_mp2_spin_components(&g, &eps, nocc, nvir, 0, nocc, &d, &cfg).unwrap();

        eprintln!(
            "screened: pairs {:.3}, virtuals {:.3}, dE = {:+.3e}",
            diag.pair_retention,
            diag.virtual_retention,
            got.e_total - dense.e_total
        );
        assert!(diag.pair_retention < 1.0);
        assert!(
            (got.e_total - dense.e_total).abs() > 1e-12,
            "screening had no effect on the energy"
        );
        // Dropping pairs removes negative correlation, so the energy must rise.
        assert!(
            got.e_total > dense.e_total,
            "dropping pairs should REDUCE |E_corr|: {:.10} vs {:.10}",
            got.e_total,
            dense.e_total
        );
    }

    /// Mismatched inputs must error rather than produce a plausible wrong number.
    #[test]
    fn dimension_mismatches_are_rejected() {
        let (nocc, nvir) = (3, 4);
        let (g, eps) = toy(nocc, nvir);
        let d = complete_pair_domains(&centers(nocc, 1.0)).unwrap();
        let cfg = DlpnoConfig::default();

        // Wrong nvir implied.
        assert!(dlpno_mp2_spin_components(&g, &eps, nocc, 5, 0, nocc, &d, &cfg).is_err());
        // Domains built for a different nocc.
        let d2 = complete_pair_domains(&centers(nocc + 1, 1.0)).unwrap();
        assert!(dlpno_mp2_spin_components(&g, &eps, nocc, nvir, 0, nocc, &d2, &cfg).is_err());
    }

    /// A tight PNO threshold alone (no pair screening) must still be conservative:
    /// it truncates virtuals, so |E_corr| shrinks, and it reports the loss.
    #[test]
    fn pno_truncation_alone_reduces_correlation() {
        let (nocc, nvir) = (3, 6);
        let (g, eps) = toy(nocc, nvir);
        let dense = spin_components_from_g(&g, &eps, nocc, nvir, 0, nocc);
        let d = complete_pair_domains(&centers(nocc, 1.0)).unwrap();

        // exact() not default(): default() now enables pair-energy screening, which
        // would confound this test's isolation of PNO virtual truncation.
        let cfg = DlpnoConfig {
            t_cut_pno: 1e-2,
            ..DlpnoConfig::exact()
        };
        let (got, diag) =
            dlpno_mp2_spin_components(&g, &eps, nocc, nvir, 0, nocc, &d, &cfg).unwrap();

        eprintln!(
            "PNO-only: virtual retention {:.3}, discarded {:.3e}, dE {:+.3e}",
            diag.virtual_retention,
            diag.max_discarded_weight,
            got.e_total - dense.e_total
        );
        assert_eq!(diag.pair_retention, 1.0, "no pair screening in this test");
        assert!(
            diag.virtual_retention < 1.0,
            "threshold should truncate virtuals"
        );
        assert!(diag.max_discarded_weight > 0.0);
    }

    /// Sanity: the centers helper and a trivial 1-orbital system do not panic.
    #[test]
    fn single_occupied_orbital_works() {
        let (nocc, nvir) = (1, 3);
        let (g, eps) = toy(nocc, nvir);
        let d = complete_pair_domains(&array![[0.0, 0.0, 0.0]]).unwrap();
        let dense = spin_components_from_g(&g, &eps, nocc, nvir, 0, nocc);
        let (got, _) =
            dlpno_mp2_spin_components(&g, &eps, nocc, nvir, 0, nocc, &d, &DlpnoConfig::default())
                .unwrap();
        assert!((got.e_total - dense.e_total).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Pair-driven path: the same energies without ever allocating dense `g`.
    // -----------------------------------------------------------------------

    /// A deterministic dressed RI tensor `b_ov` of shape `(naux, nocc*nvir)`,
    /// plus the eps ladder.
    ///
    /// Unlike [`toy`], the integrals here are a genuine GRAM product
    /// `g = b_ov^T b_ov`, which is what the RI factorization actually produces.
    /// That matters: the pair-driven path derives `(ib|ja)` as the transpose of
    /// this pair's block, which is a Gram identity, not a property of an
    /// arbitrary symmetric matrix.
    fn toy_b_ov(naux: usize, nocc: usize, nvir: usize) -> (Array2<f64>, Vec<f64>) {
        let mut s = 0x9E3779B97F4A7C15u64;
        let mut b = Array2::<f64>::zeros((naux, nocc * nvir));
        for v in b.iter_mut() {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *v = ((s >> 33) as f64 / (1u64 << 31) as f64) - 1.0;
        }
        let mut eps = Vec::new();
        for i in 0..nocc {
            eps.push(-1.0 - 0.1 * i as f64);
        }
        for a in 0..nvir {
            eps.push(0.5 + 0.1 * a as f64);
        }
        (b, eps)
    }

    /// THE EXACTNESS ANCHOR for the restructuring, in two parts.
    ///
    /// **Part 1 — the kernel is shared, so it is BIT-IDENTICAL.** Fed the same
    /// `(nvir x nvir)` blocks, the dense route and the pair-driven route produce
    /// the same bits, because they are literally the same function
    /// (`pair_energy_in_pno_basis`). This is the part that can be asserted at the
    /// bit level, and it is what rules out an algebra, ordering or weighting
    /// difference between the two paths.
    ///
    /// **Part 2 — end to end, they agree to the GEMM ROUNDING FLOOR, not bitwise.**
    /// This is a MEASURED limitation and it is stated rather than assumed away.
    /// The dense path's `g` comes from ONE big `(nocc*nvir)^2` GEMM; the
    /// pair-driven path contracts each pair separately as `B_i^T B_j`. Both sum
    /// over the same `naux` index, but OpenBLAS blocks the two shapes
    /// differently, so the SUMMATION ORDER differs and floating-point addition is
    /// not associative. MEASURED on this system: 17 of 250 unique `(ia|jb)`
    /// elements differ in their last bit, worst absolute 8.9e-16 on integrals of
    /// order 1 — i.e. ~1 ULP, entering the energy through the inputs, not the
    /// algorithm.
    ///
    /// So the honest end-to-end contract is a tight RELATIVE tolerance at the
    /// double-precision floor (1e-14 relative here, with the observed gap at
    /// 5.6e-16), not bit-identity. Claiming bit-identity across different GEMM
    /// shapes would be false; see the repo's "deterministic is not bit-identical"
    /// note for the same distinction elsewhere in the tree.
    #[test]
    fn pair_driven_matches_dense() {
        let (naux, nocc, nvir) = (11, 4, 5);
        let (b_ov, eps) = toy_b_ov(naux, nocc, nvir);
        let g = b_ov.t().dot(&b_ov);
        let d = complete_pair_domains(&centers(nocc, 1.5)).unwrap();
        let cfg = DlpnoConfig::exact();

        // --- Part 1: same blocks in, identical bits out. ---
        let pnos = build_pno_transforms(&d, nvir, 0.0, |i, j| {
            let mut t = super::pair_block_from_b_ov(&b_ov, i, j, nvir);
            for a in 0..nvir {
                for b in 0..nvir {
                    t[(a, b)] /= eps[i] + eps[j] - eps[nocc + a] - eps[nocc + b];
                }
            }
            t
        })
        .unwrap();
        for (p, &(i, j)) in d.pairs.iter().enumerate() {
            let blk = super::pair_block_from_b_ov(&b_ov, i, j, nvir);
            // Dense route: two independently sliced blocks.
            let mut g_ij = Array2::<f64>::zeros((nvir, nvir));
            let mut g_ji = Array2::<f64>::zeros((nvir, nvir));
            for a in 0..nvir {
                for b in 0..nvir {
                    g_ij[(a, b)] = blk[(a, b)];
                    g_ji[(a, b)] = blk[(b, a)];
                }
            }
            let q = &pnos.pairs[p].transform;
            let via_dense = super::pair_energy_in_pno_basis(
                g_ij.view(),
                g_ji.view(),
                q,
                &eps,
                (i, j),
                nvir,
                0,
                nocc,
            )
            .unwrap();
            // Pair-driven route: the transpose VIEW, no second allocation.
            let via_paired = super::pair_energy_in_pno_basis(
                blk.view(),
                blk.t(),
                q,
                &eps,
                (i, j),
                nvir,
                0,
                nocc,
            )
            .unwrap();
            assert_eq!(
                via_paired.0.to_bits(),
                via_dense.0.to_bits(),
                "pair ({i},{j}) OS is not bit-identical through the shared kernel"
            );
            assert_eq!(
                via_paired.1.to_bits(),
                via_dense.1.to_bits(),
                "pair ({i},{j}) SS is not bit-identical through the shared kernel"
            );
        }

        // --- Part 2: end to end, at the GEMM rounding floor. ---
        let (dense, dd) =
            dlpno_mp2_spin_components(&g, &eps, nocc, nvir, 0, nocc, &d, &cfg).unwrap();
        let (paired, pd) = dlpno_mp2_from_b_ov(&b_ov, &eps, nocc, nvir, 0, nocc, &d, &cfg).unwrap();

        let rel = |a: f64, b: f64| (a - b).abs() / b.abs().max(1.0);
        eprintln!(
            "dense  e_os={:.17e} e_ss={:.17e}\npaired e_os={:.17e} e_ss={:.17e}\n\
             rel dOS={:.3e} rel dSS={:.3e}",
            dense.e_os,
            dense.e_ss,
            paired.e_os,
            paired.e_ss,
            rel(paired.e_os, dense.e_os),
            rel(paired.e_ss, dense.e_ss)
        );
        assert!(
            rel(paired.e_os, dense.e_os) < 1e-14,
            "OS beyond the rounding floor: {:.17e} vs {:.17e}",
            paired.e_os,
            dense.e_os
        );
        assert!(
            rel(paired.e_ss, dense.e_ss) < 1e-14,
            "SS beyond the rounding floor: {:.17e} vs {:.17e}",
            paired.e_ss,
            dense.e_ss
        );
        assert_eq!(pd.n_pairs, dd.n_pairs);
        assert_eq!(pd.pair_retention, 1.0);
        assert_eq!(pd.virtual_retention, 1.0);
    }

    /// The pair-driven path must also land back on PRODUCTION RI-MP2's own
    /// streaming kernel, not just on the dense DLPNO one.
    ///
    /// `spin_components_from_b_ov` is what `ri_mp2` actually runs. This closes
    /// the loop: untruncated DLPNO-MP2 driven from `b_ov` IS RI-MP2.
    #[test]
    fn pair_driven_exact_at_zero_truncation() {
        use crate::rimp2::spin_components_from_b_ov;
        let (naux, nocc, nvir) = (13, 4, 5);
        let (b_ov, eps) = toy_b_ov(naux, nocc, nvir);
        let d = complete_pair_domains(&centers(nocc, 1.5)).unwrap();

        let want = spin_components_from_b_ov(&b_ov, &eps, nocc, nvir, 0, nocc);
        let (got, _) =
            dlpno_mp2_from_b_ov(&b_ov, &eps, nocc, nvir, 0, nocc, &d, &DlpnoConfig::exact())
                .unwrap();

        eprintln!(
            "ri_mp2 e_os={:.14} e_ss={:.14}\ndlpno  e_os={:.14} e_ss={:.14}",
            want.e_os, want.e_ss, got.e_os, got.e_ss
        );
        assert!(
            (got.e_total - want.e_total).abs() < 1e-12,
            "untruncated pair-driven DLPNO must reproduce RI-MP2: {:.14} vs {:.14}",
            got.e_total,
            want.e_total
        );
        assert!((got.e_os - want.e_os).abs() < 1e-12, "OS component differs");
        assert!((got.e_ss - want.e_ss).abs() < 1e-12, "SS component differs");
    }

    /// THE HIGHEST-VALUE ASSERTION IN THE PAIR-DRIVEN CHANGE: off-diagonal pairs
    /// carry a mirror weight of 2 and diagonal pairs a weight of 1.
    ///
    /// A pair-driven loop that mis-handles this silently doubles or halves the
    /// correlation energy while still returning a plausible number — no NaN, no
    /// panic, no shape error. So this test does not merely compare totals; it
    /// isolates the weight by construction:
    ///
    /// * run a system whose pair list is ONE diagonal pair — the weight must be
    ///   1, so the result equals the unweighted block sum;
    /// * run a domain holding exactly ONE strictly-off-diagonal pair `(0,1)` plus
    ///   the diagonals, and subtract the diagonals-only run — the remainder must
    ///   be EXACTLY TWICE the same pair's unweighted contribution.
    ///
    /// The factor is then read off directly rather than inferred from agreement
    /// with another routine that could share the same bug. Mutating
    /// `fac = if i == j { 1.0 } else { 2.0 }` to a constant `1.0` or `2.0`, or
    /// flipping the comparison, fails this.
    #[test]
    fn off_diagonal_pairs_carry_the_mirror_weight_of_two() {
        let (naux, nocc, nvir) = (9, 3, 4);
        let (b_ov, eps) = toy_b_ov(naux, nocc, nvir);
        let cfg = DlpnoConfig::exact();
        let c = centers(nocc, 1.0);

        // Helper: run the pair-driven path over an EXPLICIT pair list.
        let run = |pairs: Vec<(usize, usize)>| -> SpinComponents {
            let n = pairs.len();
            let d = PairDomains {
                nocc,
                pairs,
                coupled: (0..n).map(|p| vec![p]).collect(),
                centers: c.clone(),
                pair_cutoff: 0.0,
                coupling_cutoff: 0.0,
            };
            dlpno_mp2_from_b_ov(&b_ov, &eps, nocc, nvir, 0, nocc, &d, &cfg)
                .unwrap()
                .0
        };

        // Independent, weight-free reference for one pair's raw block sum, built
        // straight from the definition with NO fac anywhere.
        let raw = |i: usize, j: usize| -> (f64, f64) {
            let (mut os, mut ss) = (0.0, 0.0);
            for a in 0..nvir {
                for b in 0..nvir {
                    let d = eps[i] + eps[j] - eps[nocc + a] - eps[nocc + b];
                    let mut iajb = 0.0;
                    let mut ibja = 0.0;
                    for p in 0..naux {
                        iajb += b_ov[(p, i * nvir + a)] * b_ov[(p, j * nvir + b)];
                        ibja += b_ov[(p, i * nvir + b)] * b_ov[(p, j * nvir + a)];
                    }
                    os += iajb * iajb / d;
                    ss += iajb * (iajb - ibja) / d;
                }
            }
            (os, ss)
        };

        // --- Diagonal pair (1,1) alone: weight must be exactly 1. ---
        let got = run(vec![(1, 1)]);
        let (os11, ss11) = raw(1, 1);
        eprintln!(
            "diag (1,1): got os={:.14} ss={:.14} | raw os={:.14} ss={:.14} | ratio {:.14}",
            got.e_os,
            got.e_ss,
            os11,
            ss11,
            got.e_os / os11
        );
        assert!(
            (got.e_os - os11).abs() < 1e-12 * os11.abs().max(1.0),
            "diagonal pair must carry weight 1, got ratio {:.6}",
            got.e_os / os11
        );
        assert!(
            (got.e_ss - ss11).abs() < 1e-12 * ss11.abs().max(1.0),
            "diagonal SS must carry weight 1, got ratio {:.6}",
            got.e_ss / ss11
        );

        // --- Add the strictly-off-diagonal (0,1): the increment must be 2x. ---
        let diag_only = run(vec![(0, 0), (1, 1), (2, 2)]);
        let with_01 = run(vec![(0, 0), (0, 1), (1, 1), (2, 2)]);
        let (os01, ss01) = raw(0, 1);
        let d_os = with_01.e_os - diag_only.e_os;
        let d_ss = with_01.e_ss - diag_only.e_ss;
        eprintln!(
            "off-diag (0,1): dOS={:.14} raw={:.14} ratio={:.14}\n\
             off-diag (0,1): dSS={:.14} raw={:.14} ratio={:.14}",
            d_os,
            os01,
            d_os / os01,
            d_ss,
            ss01,
            d_ss / ss01
        );
        assert!(
            (d_os / os01 - 2.0).abs() < 1e-10,
            "strictly off-diagonal OS weight must be 2, measured {:.12}",
            d_os / os01
        );
        assert!(
            (d_ss / ss01 - 2.0).abs() < 1e-10,
            "strictly off-diagonal SS weight must be 2, measured {:.12}",
            d_ss / ss01
        );

        // Guard the guard: the two weights must be DISTINGUISHABLE on this
        // system, otherwise the assertions above would pass under a constant
        // fac. os11 and os01 differ, so 1x vs 2x is a real discrimination.
        assert!(
            (os11 - os01).abs() > 1e-6,
            "premise: diagonal and off-diagonal blocks must differ for this test \
             to discriminate weights (os11={os11:.6}, os01={os01:.6})"
        );
    }

    /// UNDER TRUNCATION the two paths must still agree — which is the only way
    /// the PNO construction itself gets tested.
    ///
    /// This test exists because a MUTATION SURVIVED without it. Dropping the
    /// amplitude denominator in `dlpno_mp2_from_b_ov`'s `amp` closure (so the
    /// PNOs are built from raw integrals rather than amplitudes) left EVERY
    /// untruncated test green. The reason is structural, not accidental: at
    /// `t_cut_pno = 0` the PNO transform is a square orthogonal rotation which
    /// cancels out of the energy entirely, so the amplitudes that BUILD it are
    /// unobservable. The amplitude block only affects the answer once the
    /// threshold actually discards something, because then WHICH vectors were
    /// kept starts to matter. With this test the same mutation diverges
    /// immediately — retention 1.0000 against the dense path's 0.9167.
    ///
    /// So a non-zero `t_cut_pno` is required here, and the dense path — whose
    /// `amp` closure is independently written — is the reference. Truncation
    /// must also genuinely bite, which the retention assertion below pins:
    /// otherwise this test would silently degrade back into the untruncated case
    /// that missed the bug in the first place.
    ///
    /// # What this test can NEVER catch, and why that is correct
    ///
    /// A GLOBAL SIGN on the amplitudes (`T -> -T`, e.g. `/= d` written as
    /// `/= d.abs()`, which is the same thing since the MP2 denominator is
    /// always negative) is genuinely unobservable, and no test should be
    /// contorted to catch it. The pair density is
    /// `D = T Tt + Tt T`, which is EXACTLY invariant under `T -> -T`
    /// (verified numerically: max |D(T) - D(-T)| = 0). Identical density =>
    /// identical eigenvectors => identical PNOs => identical energy. That is an
    /// invariance of the method, not a hole in the coverage, and
    /// `pno_construction_is_invariant_under_global_amplitude_sign` asserts it
    /// as a positive property so nobody re-litigates it as a missing guard.
    #[test]
    fn pair_driven_matches_dense_under_pno_truncation() {
        let (naux, nocc, nvir) = (14, 4, 6);
        let (b_ov, eps) = toy_b_ov(naux, nocc, nvir);
        let g = b_ov.t().dot(&b_ov);
        let d = complete_pair_domains(&centers(nocc, 1.5)).unwrap();
        // Truncation ON: this is what makes the PNO vectors observable.
        let cfg = DlpnoConfig {
            t_cut_pno: 1e-2,
            ..DlpnoConfig::exact()
        };

        let (dense, dd) =
            dlpno_mp2_spin_components(&g, &eps, nocc, nvir, 0, nocc, &d, &cfg).unwrap();
        let (paired, pd) = dlpno_mp2_from_b_ov(&b_ov, &eps, nocc, nvir, 0, nocc, &d, &cfg).unwrap();

        eprintln!(
            "truncated (t_cut_pno=1e-2): virtual retention dense {:.4} / paired {:.4}\n               dense  e_os={:.14} e_ss={:.14}\n  paired e_os={:.14} e_ss={:.14}",
            dd.virtual_retention, pd.virtual_retention, dense.e_os, dense.e_ss, paired.e_os,
            paired.e_ss
        );

        // THE PREMISE, asserted so this test cannot rot into a no-op: truncation
        // must really be discarding virtuals. Without this the whole test
        // degenerates to the untruncated case, which is exactly the blind spot
        // it was written to cover.
        assert!(
            pd.virtual_retention < 1.0,
            "premise: t_cut_pno must actually truncate, got retention {:.4}",
            pd.virtual_retention
        );
        assert!(
            pd.max_discarded_weight > 0.0,
            "premise: something must have been discarded"
        );
        assert_eq!(
            pd.virtual_retention, dd.virtual_retention,
            "the two paths must keep the SAME number of PNOs"
        );

        let rel = |a: f64, b: f64| (a - b).abs() / b.abs().max(1.0);
        assert!(
            rel(paired.e_os, dense.e_os) < 1e-12,
            "truncated OS differs: {:.14} vs {:.14}",
            paired.e_os,
            dense.e_os
        );
        assert!(
            rel(paired.e_ss, dense.e_ss) < 1e-12,
            "truncated SS differs: {:.14} vs {:.14}",
            paired.e_ss,
            dense.e_ss
        );
    }

    /// A global sign on the amplitudes cannot change the PNOs or the energy.
    ///
    /// The counterpart to the caveat on
    /// `pair_driven_matches_dense_under_pno_truncation`: rather than leave "that
    /// mutation survives" as an unexplained loose end, the invariance is pinned
    /// as a property. `D = T Tt + Tt T` is quadratic in `T`, so it is blind to a
    /// global sign, and every downstream quantity is a function of `D`.
    #[test]
    fn pno_construction_is_invariant_under_global_amplitude_sign() {
        let (naux, nocc, nvir) = (14, 4, 6);
        let (b_ov, eps) = toy_b_ov(naux, nocc, nvir);
        let d = complete_pair_domains(&centers(nocc, 1.5)).unwrap();

        let make = |sign: f64| {
            build_pno_transforms(&d, nvir, 1e-2, |i, j| {
                let mut t = super::pair_block_from_b_ov(&b_ov, i, j, nvir);
                for a in 0..nvir {
                    for b in 0..nvir {
                        t[(a, b)] *= sign / (eps[i] + eps[j] - eps[nocc + a] - eps[nocc + b]);
                    }
                }
                t
            })
            .unwrap()
        };
        let pos = make(1.0);
        let neg = make(-1.0);

        assert!(
            pos.virtual_retention() < 1.0,
            "premise: the threshold must truncate, else this is vacuous"
        );
        assert_eq!(
            pos.virtual_retention(),
            neg.virtual_retention(),
            "a global amplitude sign must not change how many PNOs are kept"
        );
        let mut worst = 0.0f64;
        for (a, b) in pos.pairs.iter().zip(neg.pairs.iter()) {
            assert_eq!(a.transform.dim(), b.transform.dim());
            for (x, y) in a.occupations.iter().zip(b.occupations.iter()) {
                worst = worst.max((x - y).abs());
            }
        }
        eprintln!("global-sign invariance: worst |dn_a| = {worst:.3e}");
        assert!(
            worst < 1e-14,
            "PNO occupations must be invariant under T -> -T, worst {worst:.3e}"
        );
    }

    /// The pair-driven path's working set must be O(nvir^2), not O((nocc*nvir)^2).
    ///
    /// This is the structural claim of the whole change, stated as an assertion
    /// rather than as prose: the dense entry point's INPUT scales with nocc^2
    /// relative to one pair's block, so no downstream screen can ever avoid it.
    /// At the danuglipron shape that is 133.5 GB vs a few MB on a 23 GB box.
    #[test]
    fn pair_driven_avoids_the_dense_allocation() {
        // danuglipron + capped SER31, def2-SVP: the system that motivated this.
        let (nocc, nvir) = (170usize, 760usize);
        let dense = dense_g_bytes(nocc, nvir);
        let paired = pair_driven_working_bytes(nvir);
        let gb = |b: usize| b as f64 / 1024.0f64.powi(3);
        eprintln!(
            "nocc={nocc} nvir={nvir}: dense g = {:.1} GB, pair-driven working set = {:.4} GB \
             ({:.0}x smaller)",
            gb(dense),
            gb(paired),
            dense as f64 / paired as f64
        );
        // 23 GB box: dense cannot start, pair-driven is a rounding error.
        assert!(
            gb(dense) > 100.0,
            "premise: the dense path really is over 100 GB here ({:.1})",
            gb(dense)
        );
        assert!(
            gb(paired) < 0.1,
            "pair-driven working set must be well under a GB, got {:.4}",
            gb(paired)
        );
        // The ratio is nocc^2 / 6 by construction; assert the SCALING, not a
        // magic number, so this stays meaningful if the block count changes.
        let ratio = dense as f64 / paired as f64;
        let expect = (nocc * nocc) as f64 / 6.0;
        assert!(
            (ratio / expect - 1.0).abs() < 1e-9,
            "the saving must scale as nocc^2: measured {ratio:.1}, expected {expect:.1}"
        );
    }

    /// The streaming pair-energy estimator must agree with the dense one.
    ///
    /// `t_cut_pairs` is the screen the default config enables, and its input was
    /// the dense `g` too — so screening could not be decided without first
    /// paying the allocation it exists to avoid. This pins that the streamed
    /// replacement is the same quantity.
    #[test]
    fn estimate_pair_energies_agrees_with_dense() {
        use crate::pair_energy_screen::estimate_pair_energies;
        let (naux, nocc, nvir) = (12, 4, 5);
        let (b_ov, eps) = toy_b_ov(naux, nocc, nvir);
        let g = b_ov.t().dot(&b_ov);

        let want = estimate_pair_energies(g.view(), &eps, nocc, nvir, 0, nocc).unwrap();
        let got = estimate_pair_energies_from_b_ov(&b_ov, &eps, nocc, nvir, 0, nocc).unwrap();

        let mut worst = 0.0f64;
        for i in 0..nocc {
            for j in 0..nocc {
                worst = worst.max((got.e[(i, j)] - want.e[(i, j)]).abs());
            }
        }
        eprintln!(
            "pair-energy estimator: dense total {:.14}, streamed total {:.14}, worst |de_ij| {:.3e}",
            want.total(),
            got.total(),
            worst
        );
        assert!(
            worst < 1e-12,
            "streamed pair energies must match dense, worst {worst:.3e}"
        );
        // ...and it must be a real per-pair spread, not all-zeros agreeing.
        assert!(
            want.e.iter().map(|v| v.abs()).fold(0.0, f64::max) > 1e-6,
            "premise: the pair energies must be non-trivial"
        );
    }

    /// Screening must actually change the pair-driven energy and be reported.
    ///
    /// Without this the pair-driven path could be exact-but-inert: a screen that
    /// never removes anything cannot save the work it is claimed to save.
    #[test]
    fn pair_driven_screening_is_live() {
        let (naux, nocc, nvir) = (11, 4, 5);
        let (b_ov, eps) = toy_b_ov(naux, nocc, nvir);
        let d_all = complete_pair_domains(&centers(nocc, 1.5)).unwrap();
        let (full, _) = dlpno_mp2_from_b_ov(
            &b_ov,
            &eps,
            nocc,
            nvir,
            0,
            nocc,
            &d_all,
            &DlpnoConfig::exact(),
        )
        .unwrap();

        // Widely spaced centers + a tight distance cutoff: only diagonals survive.
        let d = build_pair_domains(&centers(nocc, 10.0), 2.0, f64::INFINITY).unwrap();
        assert!(
            d.pair_retention() < 1.0,
            "premise: pairs should be screened"
        );
        let cfg = DlpnoConfig {
            pair_cutoff: 2.0,
            coupling_cutoff: f64::INFINITY,
            t_cut_pno: 0.0,
            t_cut_pairs: 0.0,
        };
        let (got, diag) = dlpno_mp2_from_b_ov(&b_ov, &eps, nocc, nvir, 0, nocc, &d, &cfg).unwrap();

        eprintln!(
            "pair-driven screened: retention {:.3}, dE = {:+.6e}",
            diag.pair_retention,
            got.e_total - full.e_total
        );
        assert!(diag.pair_retention < 1.0);
        assert!(
            (got.e_total - full.e_total).abs() > 1e-12,
            "screening had no effect on the pair-driven energy"
        );
        // Dropping pairs removes negative correlation, so the energy must rise.
        assert!(
            got.e_total > full.e_total,
            "dropping pairs should REDUCE |E_corr|: {:.10} vs {:.10}",
            got.e_total,
            full.e_total
        );
    }

    /// Mis-shaped inputs to the pair-driven path must error, not fabricate.
    #[test]
    fn pair_driven_rejects_bad_shapes() {
        let (naux, nocc, nvir) = (7, 3, 4);
        let (b_ov, eps) = toy_b_ov(naux, nocc, nvir);
        let d = complete_pair_domains(&centers(nocc, 1.0)).unwrap();
        let cfg = DlpnoConfig::exact();

        // Wrong nvir implied -> wrong b_ov width.
        assert!(dlpno_mp2_from_b_ov(&b_ov, &eps, nocc, 5, 0, nocc, &d, &cfg).is_err());
        // Domains built for a different nocc.
        let d2 = complete_pair_domains(&centers(nocc + 1, 1.0)).unwrap();
        assert!(dlpno_mp2_from_b_ov(&b_ov, &eps, nocc, nvir, 0, nocc, &d2, &cfg).is_err());
        // Streaming estimator rejects the same width mismatch.
        assert!(estimate_pair_energies_from_b_ov(&b_ov, &eps, nocc, 5, 0, nocc).is_err());
    }
}
