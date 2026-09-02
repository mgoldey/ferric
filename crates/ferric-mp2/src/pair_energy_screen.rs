//! ORCA-style *estimated pair energy* screening for occupied pair domains.
//!
//! # The problem with the distance criterion
//!
//! [`crate::pair_domains::build_pair_domains`] screens on the raw Boys-center
//! separation, in **Bohr**. That is a length, and a length is not comparable across
//! systems: a fixed Bohr cutoff is inert until it drops below the molecule's own
//! extent, at which point it removes a large block of pairs at once. MEASURED on
//! water/cc-pVDZ CCSD, the distance knob is a cliff rather than a dial:
//!
//! ```text
//!   4.0 Bohr -> retention 1.000, dE 0.000e0
//!   2.0 Bohr -> retention 1.000, dE 0.000e0
//!   1.0 Bohr -> retention 0.440, dE 2.277e-2   (~14 kcal/mol)
//! ```
//!
//! There is no setting between "no screening at all" and "14 kcal/mol of error".
//!
//! # The energy criterion
//!
//! ORCA's DLPNO codes screen instead on an *estimated pair correlation energy*
//! (`T_CutPairs`): a pair `(i, j)` is dropped when its estimated `|e_ij|` falls
//! below a threshold **in Hartree**. Because the criterion is an energy, the
//! threshold means the same thing on water as on a protein, and the discarded
//! energy is bounded by (number of dropped pairs) x threshold — the knob is
//! directly interpretable as an error budget rather than as a geometric radius.
//!
//! The estimator used here is the semicanonical MP2 pair energy
//!
//! ```text
//! e_ij = sum_ab [ 2 (ia|jb) - (ib|ja) ] (ia|jb) / (e_i + e_j - e_a - e_b)
//! ```
//!
//! which is the same expression [`crate::rimp2::spin_components_from_g`] sums over
//! `i <= j`. That is deliberate: with the `i < j` mirror factor of 2 applied, the
//! pair energies produced by [`crate::pair_energy_screen::estimate_pair_energies`] sum **exactly** to the
//! RI-MP2 correlation energy, so the threshold is denominated in the same units as
//! the quantity being approximated. `pair_energies_sum_to_mp2_correlation` pins it.
//!
//! # Exactness
//!
//! `t_cut_pairs = 0.0` retains every pair (the test is `|e_ij| >= t`, and
//! `|e_ij| >= 0` always holds), so the resulting [`crate::pair_domains::PairDomains`] is
//! [`crate::pair_domains::PairDomains::is_complete`] and every downstream consumer — DLPNO-MP2,
//! DLPNO-CCSD, DLPNO-CCSD(T), LinLCCD — is unchanged. That is the load-bearing
//! guarantee and `zero_threshold_retains_every_pair` tests it first.
//!
//! # What this module does NOT claim
//!
//! It builds the same [`crate::pair_domains::PairDomains`] type the distance screen builds, so the
//! *coupling* screen is still geometric (pair-centroid distance). Only the pair
//! list itself is energy-screened. See `crates/ferric-mp2/tests/pair_screen_criteria.rs`
//! for the head-to-head measurement of the two pair criteria.

use crate::pair_domains::PairDomains;
use ferric_core::FerricError;
use ndarray::{Array2, ArrayView2};

/// Per-pair estimated correlation energies over the *unique* `i <= j` pair list.
///
/// The entries already carry the `i < j` mirror factor of 2, so
/// [`PairEnergies::total`] equals the MP2 correlation energy. `e_ij(i, j)` is
/// symmetric in its arguments.
#[derive(Debug, Clone)]
pub struct PairEnergies {
    /// Number of occupied spatial orbitals.
    pub nocc: usize,
    /// Row-major `nocc x nocc` estimated pair energies, symmetric, mirror factor
    /// already applied to the off-diagonal entries.
    pub e: Array2<f64>,
}

impl PairEnergies {
    /// Estimated pair energy of `(i, j)`; symmetric under exchange.
    pub fn e_ij(&self, i: usize, j: usize) -> f64 {
        self.e[(i.min(j), i.max(j))]
    }

    /// Sum over unique `i <= j` pairs — the full MP2 correlation energy.
    pub fn total(&self) -> f64 {
        let mut s = 0.0;
        for i in 0..self.nocc {
            for j in i..self.nocc {
                s += self.e[(i, j)];
            }
        }
        s
    }

    /// Total estimated energy carried by the pairs a threshold would DISCARD.
    ///
    /// This is the a-priori error budget of `t_cut_pairs`: dropping those pairs
    /// removes approximately this much correlation energy. Diagonal pairs are never
    /// discarded, so they never contribute.
    pub fn discarded_energy(&self, t_cut_pairs: f64) -> f64 {
        let mut s = 0.0;
        for i in 0..self.nocc {
            for j in (i + 1)..self.nocc {
                if self.e[(i, j)].abs() < t_cut_pairs {
                    s += self.e[(i, j)];
                }
            }
        }
        s
    }
}

/// Estimate every occupied pair's MP2 correlation energy from the MO integral matrix.
///
/// `g` is the `(nocc*nvir, nocc*nvir)` matrix of `(ia|jb)` in the *active* orbital
/// window — the same object [`crate::rimp2::spin_components_from_g`] and
/// [`crate::dlpno_mp2::dlpno_mp2_spin_components`] consume, so no new integral build
/// is needed at the call site. `eps` are the full MO energies; `first_occ` and
/// `nocc_total` locate the active window inside them (frozen core aware).
///
/// The `i < j` mirror factor of 2 is applied here, matching the dense kernel's
/// unique-pair weighting, so `sum_{i<=j} e_ij` is the MP2 correlation energy.
///
/// # Errors
///
/// [`FerricError::General`] when `g` is not `(nocc*nvir, nocc*nvir)`, when `eps` is
/// too short for the requested window, or when a denominator `e_i + e_j - e_a - e_b`
/// is non-negative — the latter means the reference is not a valid MP2 starting
/// point (zero or inverted gap) and silently dividing would fabricate a number.
pub fn estimate_pair_energies(
    g: ArrayView2<f64>,
    eps: &[f64],
    nocc: usize,
    nvir: usize,
    first_occ: usize,
    nocc_total: usize,
) -> Result<PairEnergies, FerricError> {
    let want = nocc * nvir;
    if g.nrows() != want || g.ncols() != want {
        return Err(FerricError::General(format!(
            "estimate_pair_energies: g is {:?}, expected ({want}, {want}) for \
             nocc={nocc}, nvir={nvir}",
            g.dim()
        )));
    }
    if eps.len() < nocc_total + nvir || first_occ + nocc > eps.len() {
        return Err(FerricError::General(format!(
            "estimate_pair_energies: eps has {} entries, too short for first_occ={first_occ}, \
             nocc={nocc}, nocc_total={nocc_total}, nvir={nvir}",
            eps.len()
        )));
    }

    let mut e = Array2::<f64>::zeros((nocc, nocc));
    for i in 0..nocc {
        let e_i = eps[first_occ + i];
        for j in i..nocc {
            let e_j = eps[first_occ + j];
            // Same unique-pair weighting as the dense kernel: (i,j) stands in for
            // its (j,i) mirror.
            let fac = if i == j { 1.0 } else { 2.0 };
            let mut acc = 0.0;
            for a in 0..nvir {
                let e_a = eps[nocc_total + a];
                for b in 0..nvir {
                    let d = e_i + e_j - e_a - eps[nocc_total + b];
                    if d >= 0.0 {
                        return Err(FerricError::General(format!(
                            "estimate_pair_energies: non-negative MP2 denominator {d} for \
                             (i={i}, j={j}, a={a}, b={b}); the reference has a zero or \
                             inverted gap and is not a valid MP2 starting point"
                        )));
                    }
                    let iajb = g[(i * nvir + a, j * nvir + b)];
                    let ibja = g[(i * nvir + b, j * nvir + a)];
                    acc += (2.0 * iajb - ibja) * iajb / d;
                }
            }
            let v = fac * acc;
            e[(i, j)] = v;
            e[(j, i)] = v;
        }
    }
    Ok(PairEnergies { nocc, e })
}

/// Build occupied pair domains by ORCA-style estimated-pair-energy screening.
///
/// A pair `(i, j)` with `i < j` is retained when `|e_ij| >= t_cut_pairs`. Diagonal
/// pairs `(i, i)` are **always** retained, exactly as in the distance screen: they
/// carry the bulk of the correlation energy, and dropping them is not a locality
/// approximation but an error.
///
/// The returned value is the same [`crate::pair_domains::PairDomains`] the distance screen produces, so
/// every existing consumer works unchanged. The `coupling_cutoff_bohr` screen is
/// still geometric — only the *pair list* criterion is replaced.
///
/// `centers` are the Boys centers, carried for diagnostics and used for the
/// pair-centroid coupling screen. Pass `f64::INFINITY` to couple all retained pairs.
///
/// # Exactness
///
/// `t_cut_pairs = 0.0` (with an infinite coupling cutoff) retains everything and the
/// result is [`crate::pair_domains::PairDomains::is_complete`], so downstream energies are unchanged.
///
/// # Errors
///
/// [`FerricError::General`] on a negative threshold or cutoff, on `centers` whose
/// second dimension is not 3, or when `centers` and `pair_energies` disagree on
/// `nocc`.
pub fn build_pair_domains_by_energy(
    centers: &Array2<f64>,
    pair_energies: &PairEnergies,
    t_cut_pairs: f64,
    coupling_cutoff_bohr: f64,
) -> Result<PairDomains, FerricError> {
    if centers.ncols() != 3 {
        return Err(FerricError::General(format!(
            "Boys centers must be (nocc, 3); got ({}, {})",
            centers.nrows(),
            centers.ncols()
        )));
    }
    if centers.nrows() != pair_energies.nocc {
        return Err(FerricError::General(format!(
            "build_pair_domains_by_energy: centers has {} rows but pair energies were \
             built for nocc={}",
            centers.nrows(),
            pair_energies.nocc
        )));
    }
    if !(t_cut_pairs >= 0.0) {
        return Err(FerricError::General(format!(
            "t_cut_pairs must be >= 0 (got {t_cut_pairs}); use 0.0 to disable screening"
        )));
    }
    if coupling_cutoff_bohr < 0.0 {
        return Err(FerricError::General(format!(
            "coupling cutoff must be >= 0 (got {coupling_cutoff_bohr}); use f64::INFINITY \
             to disable screening"
        )));
    }
    let nocc = centers.nrows();

    // --- Pair list: i <= j, kept on estimated energy. Diagonals always kept. ---
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for i in 0..nocc {
        for j in i..nocc {
            if i == j || pair_energies.e_ij(i, j).abs() >= t_cut_pairs {
                pairs.push((i, j));
            }
        }
    }

    // --- Coupling screen: unchanged, geometric on pair centroids. ---
    let centroid = |p: (usize, usize)| -> [f64; 3] {
        std::array::from_fn(|ax| 0.5 * (centers[(p.0, ax)] + centers[(p.1, ax)]))
    };
    let cents: Vec<[f64; 3]> = pairs.iter().map(|&p| centroid(p)).collect();

    let coup_cut_sq = coupling_cutoff_bohr * coupling_cutoff_bohr;
    let mut coupled: Vec<Vec<usize>> = Vec::with_capacity(pairs.len());
    for p in 0..pairs.len() {
        let mut row = Vec::new();
        for q in 0..pairs.len() {
            if p == q {
                row.push(q);
                continue;
            }
            let d: f64 = (0..3).map(|ax| (cents[p][ax] - cents[q][ax]).powi(2)).sum();
            if d <= coup_cut_sq {
                row.push(q);
            }
        }
        coupled.push(row);
    }

    Ok(PairDomains {
        nocc,
        pairs,
        coupled,
        centers: centers.to_owned(),
        // The distance screen was not applied; record that honestly rather than
        // stuffing the energy threshold into a field documented as Bohr.
        pair_cutoff: f64::INFINITY,
        coupling_cutoff: coupling_cutoff_bohr,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rimp2::spin_components_from_g;
    use ndarray::array;

    /// Deterministic pseudo-random filler — a structured matrix could hide an
    /// indexing error.
    fn fill(n: usize, seed: u64) -> Array2<f64> {
        let mut s = seed;
        let v: Vec<f64> = (0..n * n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 33) as f64 / (1u64 << 31) as f64) - 1.0
            })
            .collect();
        Array2::from_shape_vec((n, n), v).unwrap()
    }

    /// A small synthetic system: 3 occupied, 4 virtual, well-separated energies.
    fn synthetic() -> (Array2<f64>, Vec<f64>, usize, usize) {
        let (nocc, nvir) = (3usize, 4usize);
        let g = fill(nocc * nvir, 0xC0FFEE);
        let eps = vec![-1.0, -0.8, -0.6, 0.4, 0.7, 1.1, 1.6];
        (g, eps, nocc, nvir)
    }

    fn centers3() -> Array2<f64> {
        array![[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [20.0, 0.0, 0.0]]
    }

    /// THE EXACTNESS GUARANTEE, tested first: a zero threshold screens nothing.
    ///
    /// Every downstream consumer is allowed to assume that a complete `PairDomains`
    /// reproduces the dense result, so this is the load-bearing test for the whole
    /// module.
    #[test]
    fn zero_threshold_retains_every_pair() {
        let (g, eps, nocc, nvir) = synthetic();
        let pe = estimate_pair_energies(g.view(), &eps, nocc, nvir, 0, nocc).unwrap();
        let d = build_pair_domains_by_energy(&centers3(), &pe, 0.0, f64::INFINITY).unwrap();

        assert_eq!(d.pairs.len(), nocc * (nocc + 1) / 2, "must keep every i<=j pair");
        assert!(d.is_complete(), "t_cut_pairs = 0 must produce complete domains");
        assert_eq!(d.pair_retention(), 1.0);
        assert_eq!(d.coupling_retention(), 1.0);
    }

    /// The estimator must be the SAME energy the dense kernel sums.
    ///
    /// This is what makes the threshold interpretable: `t_cut_pairs` is denominated
    /// in the units of the quantity being approximated, not in some proxy.
    #[test]
    fn pair_energies_sum_to_mp2_correlation() {
        let (g, eps, nocc, nvir) = synthetic();
        let pe = estimate_pair_energies(g.view(), &eps, nocc, nvir, 0, nocc).unwrap();
        let dense = spin_components_from_g(&g, &eps, nocc, nvir, 0, nocc);

        let diff = (pe.total() - dense.e_total).abs();
        assert!(
            diff < 1e-12,
            "pair energies sum to {} but MP2 E_corr is {}; diff {diff:e}",
            pe.total(),
            dense.e_total
        );
    }

    /// Frozen-core windows must be handled: the estimator still reproduces the
    /// dense kernel when the active window does not start at orbital 0.
    #[test]
    fn pair_energies_respect_a_frozen_core_window() {
        // 4 total occupied, 1 frozen -> active nocc = 3; 4 virtuals.
        let (nocc_active, nvir) = (3usize, 4usize);
        let (first_occ, nocc_total) = (1usize, 4usize);
        let g = fill(nocc_active * nvir, 0xBADF00D);
        let eps = vec![-3.0, -1.0, -0.8, -0.6, 0.4, 0.7, 1.1, 1.6];

        let pe =
            estimate_pair_energies(g.view(), &eps, nocc_active, nvir, first_occ, nocc_total)
                .unwrap();
        let dense = spin_components_from_g(&g, &eps, nocc_active, nvir, first_occ, nocc_total);
        assert!((pe.total() - dense.e_total).abs() < 1e-12);
    }

    /// Raising the threshold must monotonically shrink the retained pair list.
    ///
    /// This is the "dial not cliff" property in its testable form.
    #[test]
    fn retention_falls_monotonically_with_threshold() {
        let (g, eps, nocc, nvir) = synthetic();
        let pe = estimate_pair_energies(g.view(), &eps, nocc, nvir, 0, nocc).unwrap();

        let mut last = usize::MAX;
        for t in [0.0, 1e-6, 1e-4, 1e-2, 1.0, 1e6] {
            let d = build_pair_domains_by_energy(&centers3(), &pe, t, f64::INFINITY).unwrap();
            assert!(d.pairs.len() <= last, "retention rose when the threshold tightened");
            last = d.pairs.len();
            // Hard floor: the diagonal is never screened.
            assert!(d.pairs.len() >= nocc, "diagonal pairs were screened out at t={t}");
        }
        assert_eq!(last, nocc, "an absurd threshold must leave exactly the diagonal");
    }

    /// Diagonal pairs survive any threshold, however absurd.
    #[test]
    fn diagonal_pairs_are_never_screened() {
        let (g, eps, nocc, nvir) = synthetic();
        let pe = estimate_pair_energies(g.view(), &eps, nocc, nvir, 0, nocc).unwrap();
        let d = build_pair_domains_by_energy(&centers3(), &pe, f64::MAX, f64::INFINITY).unwrap();
        assert_eq!(d.pairs.len(), nocc);
        for i in 0..nocc {
            assert!(d.pairs.contains(&(i, i)), "diagonal ({i},{i}) was screened out");
        }
        for row in &d.coupled {
            assert!(!row.is_empty(), "a pair must always couple to itself");
        }
    }

    /// The discarded-energy budget must agree with what screening actually removes.
    ///
    /// `discarded_energy(t)` is the module's a-priori error estimate; if it did not
    /// match the pairs the builder drops, the threshold would not be interpretable
    /// as an error budget.
    #[test]
    fn discarded_energy_matches_the_dropped_pairs() {
        let (g, eps, nocc, nvir) = synthetic();
        let pe = estimate_pair_energies(g.view(), &eps, nocc, nvir, 0, nocc).unwrap();

        for t in [0.0, 1e-4, 1e-2, 1.0] {
            let d = build_pair_domains_by_energy(&centers3(), &pe, t, f64::INFINITY).unwrap();
            let kept: f64 = d.pairs.iter().map(|&(i, j)| pe.e_ij(i, j)).sum();
            let budget = pe.discarded_energy(t);
            assert!(
                (kept + budget - pe.total()).abs() < 1e-12,
                "t={t}: kept {kept} + discarded {budget} != total {}",
                pe.total()
            );
        }
    }

    /// Screening must actually be able to drop something, or the criterion is inert.
    #[test]
    fn a_mid_threshold_drops_some_but_not_all_off_diagonals() {
        let (g, eps, nocc, nvir) = synthetic();
        let pe = estimate_pair_energies(g.view(), &eps, nocc, nvir, 0, nocc).unwrap();
        // Pick a threshold between the smallest and largest off-diagonal magnitude.
        let mut offs: Vec<f64> =
            (0..nocc).flat_map(|i| ((i + 1)..nocc).map(move |j| (i, j))).map(|(i, j)| pe.e_ij(i, j).abs()).collect();
        offs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(offs.len() >= 2, "test premise: need >= 2 off-diagonal pairs");
        let t = 0.5 * (offs[0] + offs[offs.len() - 1]);

        let d = build_pair_domains_by_energy(&centers3(), &pe, t, f64::INFINITY).unwrap();
        assert!(!d.is_complete(), "threshold {t} was inert");
        assert!(d.pairs.len() > nocc, "threshold {t} dropped every off-diagonal");
    }

    /// Bad inputs are caller bugs and must error, not panic or silently proceed.
    #[test]
    fn invalid_inputs_are_errors() {
        let (g, eps, nocc, nvir) = synthetic();
        let pe = estimate_pair_energies(g.view(), &eps, nocc, nvir, 0, nocc).unwrap();

        // Negative / NaN threshold.
        assert!(build_pair_domains_by_energy(&centers3(), &pe, -1.0, f64::INFINITY).is_err());
        assert!(build_pair_domains_by_energy(&centers3(), &pe, f64::NAN, f64::INFINITY).is_err());
        // Negative coupling cutoff.
        assert!(build_pair_domains_by_energy(&centers3(), &pe, 0.0, -1.0).is_err());
        // Wrong centers shape.
        let bad = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
        assert!(build_pair_domains_by_energy(&bad, &pe, 0.0, f64::INFINITY).is_err());
        // nocc mismatch between centers and pair energies.
        let two = array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        assert!(build_pair_domains_by_energy(&two, &pe, 0.0, f64::INFINITY).is_err());
        // Wrong g shape.
        let bad_g = fill(nocc * nvir + 1, 7);
        assert!(estimate_pair_energies(bad_g.view(), &eps, nocc, nvir, 0, nocc).is_err());
        // eps too short.
        assert!(estimate_pair_energies(g.view(), &eps[..3], nocc, nvir, 0, nocc).is_err());
    }

    /// A zero / inverted gap must be a hard error, never a fabricated energy.
    #[test]
    fn non_negative_denominator_is_an_error() {
        let (g, _, nocc, nvir) = synthetic();
        // HOMO and LUMO degenerate -> e_i + e_j - e_a - e_b = 0 for the top pair.
        let eps = vec![-1.0, -0.8, -0.6, -0.6, 0.7, 1.1, 1.6];
        assert!(estimate_pair_energies(g.view(), &eps, nocc, nvir, 0, nocc).is_err());
    }
}
