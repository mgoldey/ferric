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
//! # What it is honestly worth
//!
//! ferric has MEASURED negative results for locality methods at small N — the OSV
//! sweep (100% virtual retention at usable thresholds, 48–76 mHa error otherwise)
//! and, after the Boys sign fix, a re-measured Boys-screening crossover showing
//! screened χ₀ still 5–14× *slower* than dense across C2–C12 with no crossover
//! trend. So this module reports its own retention and is exact in the limit, to
//! let the accuracy/cost curve be measured on any given system rather than assumed
//! from the literature.

use crate::local_pno::{build_pno_transforms, PnoTransforms};
use crate::pair_domains::PairDomains;
use crate::rimp2::SpinComponents;
use ferric_core::FerricError;
use ndarray::Array2;

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

/// DLPNO-MP2 spin components from an explicit `(ia|jb)` matrix.
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

/// Compute DLPNO-MP2 spin-component energies from dressed MO integrals.
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
        let e_i = eps[first_occ + i];
        let e_j = eps[first_occ + j];
        // Off-diagonal (i,j) stands in for its (j,i) mirror -- identical to the
        // dense kernel's weighting, so screening cannot change the convention.
        let fac = if i == j { 1.0 } else { 2.0 };

        let q = &pnos.pairs[p].transform; // (nvir x npno), orthonormal columns
        let npno = q.ncols();

        // Rotate this pair's integral blocks into its PNO basis. Two blocks are
        // needed because the same-spin term contracts (ia|jb) against (ib|ja):
        //   g_ij[a,b] = (ia|jb)   ->  Qt g_ij Q
        //   g_ji[a,b] = (ib|ja)   ->  Qt g_ji Q      (the a<->b transpose partner)
        let mut g_ij = Array2::<f64>::zeros((nvir, nvir));
        let mut g_ji = Array2::<f64>::zeros((nvir, nvir));
        for a in 0..nvir {
            for b in 0..nvir {
                g_ij[(a, b)] = g[(i * nvir + a, j * nvir + b)];
                g_ji[(a, b)] = g[(i * nvir + b, j * nvir + a)];
            }
        }
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
                f_pno[(a, b)] =
                    (0..nvir).map(|c| q[(c, a)] * q[(c, b)] * eps[nocc_total + c]).sum();
            }
        }
        let (e_pno, u) = ferric_core::linalg::eigh_dc(&f_pno, ferric_core::linalg::Uplo::Upper)
            .map_err(|e| {
                FerricError::General(format!("DLPNO semicanonicalization for pair ({i},{j}): {e}"))
            })?;
        // Carry the integrals into the semicanonical PNO basis too.
        let t_ij = u.t().dot(&t_ij).dot(&u);
        let t_ji = u.t().dot(&t_ji).dot(&u);

        for a in 0..npno {
            for b in 0..npno {
                let d = e_i + e_j - e_pno[a] - e_pno[b];
                let iajb = t_ij[(a, b)];
                let ibja = t_ji[(a, b)];
                e_os += fac * iajb * iajb / d;
                e_ss += fac * iajb * (iajb - ibja) / d;
            }
        }
    }

    let diag = DlpnoDiagnostics {
        pair_retention: domains.pair_retention(),
        virtual_retention: pnos.virtual_retention(),
        max_discarded_weight: pnos.max_discarded_weight(),
        n_pairs: domains.pairs.len(),
    };
    Ok((SpinComponents { e_os, e_ss, e_total: e_os + e_ss }, diag))
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
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
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
        Array2::from_shape_fn((nocc, 3), |(i, ax)| if ax == 0 { i as f64 * spacing } else { 0.0 })
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
        let (got, diag) = dlpno_mp2_spin_components(
            &g,
            &eps,
            nocc,
            nvir,
            0,
            nocc,
            &d,
            &DlpnoConfig::default(),
        )
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
        assert!((got.e_os - dense.e_os).abs() < 1e-10, "OS component differs");
        assert!((got.e_ss - dense.e_ss).abs() < 1e-10, "SS component differs");
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
        let tight = DlpnoConfig { t_cut_pairs: 1e30, ..DlpnoConfig::default() };
        let d_tight = build_domains_for(&centers, &pe, &tight).unwrap();
        assert!(
            d_tight.pairs.len() < d_exact.pairs.len(),
            "an extreme t_cut_pairs must screen -- the constructor is not reading it"
        );
        // ...and diagonals still survive it.
        for i in 0..nocc {
            assert!(d_tight.pairs.contains(&(i, i)), "diagonal ({i},{i}) was screened");
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
        assert_eq!(d.t_cut_pno, 0.0, "virtual truncation must stay off by default");
        assert!(d.pair_cutoff.is_infinite(), "distance screening must stay off");

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
        assert!(d.pair_retention() < 1.0, "premise: pairs should be screened");

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
        let cfg = DlpnoConfig { t_cut_pno: 1e-2, ..DlpnoConfig::exact() };
        let (got, diag) =
            dlpno_mp2_spin_components(&g, &eps, nocc, nvir, 0, nocc, &d, &cfg).unwrap();

        eprintln!(
            "PNO-only: virtual retention {:.3}, discarded {:.3e}, dE {:+.3e}",
            diag.virtual_retention,
            diag.max_discarded_weight,
            got.e_total - dense.e_total
        );
        assert_eq!(diag.pair_retention, 1.0, "no pair screening in this test");
        assert!(diag.virtual_retention < 1.0, "threshold should truncate virtuals");
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
}
