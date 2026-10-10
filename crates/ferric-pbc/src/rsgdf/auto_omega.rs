//! Per-cell choice of the RS-GDF Ewald split ω (`gdf_omega = "auto"`;
//! issue #227).
//!
//! # What ω moves
//!
//! `1/r = erfc(ωr)/r + erf(ωr)/r`: ω decides how much work goes to the
//! short-range (SR) real-space triplet walk and how much to the long-range
//! (LR) G-space sum. Any ω gives the same fit up to the screening precision
//! (`reference/pbc/bench/omega_sweep.py`: ≤ 4e-9 Ha/cell over ω = 0.25..2
//! on every study cell whose SCF has a single minimum; Si has several Gamma
//! RHF solutions and its energy differences are SCF states, not ω), so the choice is a cost decision.
//!
//! * SR: `N_sr3(ω)` shell triplets, each ~0.17 µs of 6-thread wall. It FALLS
//!   with ω (screening radii go as `1/ν`, `1/ν² = 1/p_min + 1/α_min + 1/ω²`)
//!   and saturates once `ω² ≫ p_min, α_min`: the diffuse primitives, not ω,
//!   then set the radius. The count is the build's own walk
//!   ([`super::sr_triplet_estimate`]: rotation and range split included, no
//!   integrals, exact up to [`AUTO_OMEGA_COUNT_BUDGET`] triplets and a
//!   deterministic sample of the (pair, image) cells beyond it), so no
//!   screening rule is re-implemented here.
//! * LR: `N_G(ω) = Ω (2ω √ln(1/p))³ / (12π²)` half-sphere G vectors (a
//!   closed form; the build's `half_gvectors` count agrees to ~1%), each
//!   costing `a · W + b · nao² naux`: `W` is the pair-FT work per G
//!   ([`crate::pair_ft::pair_ft_work`] on the lowest candidate's sphere: the
//!   plan's survivor × shell-size sum, no FT evaluated) and `nao² naux` the
//!   J3/J2 GEMM size. `N_G ∝ ω³`.
//!
//! The predicted SR3 + LR time is minimised over a fixed log-spaced
//! candidate set; ω moves to a candidate only if it predicts a saving of at
//! least [`AUTO_OMEGA_SWITCH_MARGIN`] over the status quo
//! [`DEFAULT_RSGDF_OMEGA`](super::DEFAULT_RSGDF_OMEGA), so a flat optimum
//! keeps today's build bit for bit. Everything else in a run (hcore, SCF,
//! the metric solve) is ω-independent except the default hcore split, whose
//! cap follows ω (a second-order term the model leaves out).
//!
//! # Calibration and scope
//!
//! The three cost constants ([`FITTED`]) were fitted by least squares on
//! four calibration cells only (diamond_prim, dry ice, LiF, Si; Gamma RHF,
//! cc-pVDZ / cc-pVDZ-RI, 6 threads, no range split) and then checked on four
//! held-out cells (see the PR / `reference/pbc/bench/omega_sweep.py`). They
//! are 6-thread-wall seconds; the choice depends on their RATIOS, which the
//! 1-thread timings of the issue (dry ice 98.5 s vs 55.4 s) keep to ~10%.
//! The model is for the Gamma-point energy build; the k-point builds (full
//! sphere, different walk) and the derivative builds (unrotated walk) are
//! not calibrated, and the bindings refuse `"auto"` there.

use super::split::sr_triplet_estimate;
use super::{RsGdfConfig, DEFAULT_RSGDF_OMEGA};
use crate::hcore::half_gvectors;
use crate::lattice::Cell;
use crate::pair_ft::pair_ft_work;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use std::f64::consts::PI;

/// The ω values the chooser considers (Bohr⁻¹): log-spaced by √2 and
/// containing [`DEFAULT_RSGDF_OMEGA`]. Measured optima of the study cells lie
/// in 0.35..1.4; the ends are one step beyond.
pub const AUTO_OMEGA_CANDIDATES: [f64; 7] = [0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0];

/// Triplets the chooser's SR walk visits per candidate before it scales
/// the visited cells up ([`sr_triplet_estimate`]; exact below this). The
/// exact walk costs ~20 ns per triplet against ~170 ns to compute one, so the
/// full count at the seven candidates would cost several times the build's
/// own SR time on a dense cell. At 1 M triplets the estimate was within 5.3%
/// of the exact count at every candidate of five study cells (diamond_prim
/// 0.5%, dry ice 1.4%, expanded dry ice 5.3%, Ar 3.3%, Si 0.8%) for
/// ~20-40 ms per candidate, ~0.2-0.3 s for the whole choice.
pub const AUTO_OMEGA_COUNT_BUDGET: u64 = 1_000_000;

/// ω leaves the default only for a predicted SR3 + LR saving of at least this
/// fraction of the default's predicted SR3 + LR time. Set a priori to the
/// model's demonstrated error on the calibration cells (10-15% in absolute
/// time, tighter in the difference of two candidates), not tuned on the
/// held-out cells.
pub const AUTO_OMEGA_SWITCH_MARGIN: f64 = 0.10;

/// The cost model's three constants (seconds, 6-thread wall).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CostModel {
    /// Per SR 3-centre shell triplet.
    pub s_per_sr3_triplet: f64,
    /// Per half-sphere G vector and unit of pair-FT work `W` (millions).
    pub s_per_g_per_mwork: f64,
    /// Per half-sphere G vector and million of `nao² naux`.
    pub s_per_g_per_mgemm: f64,
}

/// Constants fitted on the calibration cells only.
pub const FITTED: CostModel = CostModel {
    s_per_sr3_triplet: 169.2e-9,
    s_per_g_per_mwork: 3.709e-4,
    s_per_g_per_mgemm: 3.217e-5,
};

/// One evaluated candidate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Candidate {
    pub omega: f64,
    /// SR 3-centre shell triplets the build would compute.
    pub n_sr3: f64,
    /// Half-sphere G vectors (closed form).
    pub n_g: f64,
    /// Predicted SR3 + LR seconds.
    pub predicted_s: f64,
}

/// The chooser's result: the ω and the table it was read from.
#[derive(Debug, Clone, PartialEq)]
pub struct AutoOmega {
    /// The chosen RS-GDF ω (Bohr⁻¹), one of [`AUTO_OMEGA_CANDIDATES`].
    pub omega: f64,
    pub candidates: Vec<Candidate>,
    /// Pair-FT work per G vector `W` (millions) and `nao² naux` (millions).
    pub work_m: f64,
    pub gemm_m: f64,
}

/// `N_G(ω)`: the half-sphere G count `Ω gcut³ / (12π²)` with
/// `gcut = 2ω √ln(1/precision)` (the sphere the LR sum walks).
pub fn half_sphere_g_count(volume: f64, omega: f64, precision: f64) -> f64 {
    let gcut = 2.0 * omega * (1.0 / precision).ln().sqrt();
    volume * gcut.powi(3) / (12.0 * PI * PI)
}

/// Predicted SR3 + LR seconds.
pub fn predicted_seconds(model: &CostModel, n_sr3: f64, n_g: f64, work_m: f64, gemm_m: f64) -> f64 {
    model.s_per_sr3_triplet * n_sr3
        + n_g * (model.s_per_g_per_mwork * work_m + model.s_per_g_per_mgemm * gemm_m)
}

/// The ω to use given evaluated candidates: the cheapest one if it saves at
/// least `margin` of the [`DEFAULT_RSGDF_OMEGA`] candidate's predicted time,
/// else the default. Falls back to the default when the table is empty, has
/// no default entry or holds a non-finite prediction (never a silent NaN
/// pick).
pub fn choose_omega(candidates: &[Candidate], margin: f64) -> f64 {
    let Some(reference) = candidates
        .iter()
        .find(|c| c.omega == DEFAULT_RSGDF_OMEGA)
        .filter(|c| c.predicted_s.is_finite() && c.predicted_s > 0.0)
    else {
        return DEFAULT_RSGDF_OMEGA;
    };
    if candidates.iter().any(|c| !c.predicted_s.is_finite()) {
        return DEFAULT_RSGDF_OMEGA;
    }
    // Cheapest; an exact tie goes to the candidate closer to the default.
    let best = candidates
        .iter()
        .min_by(|a, b| {
            a.predicted_s.total_cmp(&b.predicted_s).then_with(|| {
                let da = (a.omega / DEFAULT_RSGDF_OMEGA).ln().abs();
                let db = (b.omega / DEFAULT_RSGDF_OMEGA).ln().abs();
                da.total_cmp(&db)
            })
        })
        .expect("non-empty: the reference was found");
    if best.predicted_s <= (1.0 - margin) * reference.predicted_s {
        best.omega
    } else {
        DEFAULT_RSGDF_OMEGA
    }
}

/// Choose the RS-GDF ω of a Gamma-point energy build of `(cell, obs, aux)`
/// under `base` (precision, range split, column rotation and budget are read
/// from it; its `omega` is ignored). Cost: one integral-free SR walk per
/// candidate plus one pair-FT screen.
pub fn auto_rsgdf_omega(
    cell: &Cell,
    obs: &PreparedBasis,
    aux: &PreparedBasis,
    base: &RsGdfConfig,
) -> Result<AutoOmega, FerricError> {
    let lowest = AUTO_OMEGA_CANDIDATES[0];
    let gcut = 2.0 * lowest * (1.0 / base.precision).ln().sqrt();
    let probe = half_gvectors(cell, gcut)?;
    // A cell whose lowest sphere holds no G vector (a tiny-volume limit)
    // cannot be priced; keep the default.
    let work_m = if probe.is_empty() {
        f64::NAN
    } else {
        pair_ft_work(cell, obs, &probe, base.precision)? as f64 / 1e6
    };
    let nao = obs.nbasis() as f64;
    let gemm_m = nao * nao * aux.nbasis() as f64 / 1e6;
    let mut candidates = Vec::with_capacity(AUTO_OMEGA_CANDIDATES.len());
    for &omega in &AUTO_OMEGA_CANDIDATES {
        let cfg = RsGdfConfig { omega, ..*base };
        let n_sr3 = sr_triplet_estimate(cell, obs, aux, &cfg, AUTO_OMEGA_COUNT_BUDGET)?;
        let n_g = half_sphere_g_count(cell.volume(), omega, base.precision);
        candidates.push(Candidate {
            omega,
            n_sr3,
            n_g,
            predicted_s: predicted_seconds(&FITTED, n_sr3, n_g, work_m, gemm_m),
        });
    }
    let omega = choose_omega(&candidates, AUTO_OMEGA_SWITCH_MARGIN);
    Ok(AutoOmega {
        omega,
        candidates,
        work_m,
        gemm_m,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cand(omega: f64, predicted_s: f64) -> Candidate {
        Candidate {
            omega,
            n_sr3: 0.0,
            n_g: 0.0,
            predicted_s,
        }
    }

    /// Synthetic SR ~ 1/ω^k, LR ~ ω³ table over the real candidate set.
    fn table(sr: f64, lr: f64) -> Vec<Candidate> {
        AUTO_OMEGA_CANDIDATES
            .iter()
            .map(|&w| cand(w, sr / w.powi(2) + lr * w.powi(3)))
            .collect()
    }

    #[test]
    fn candidates_contain_the_default_and_are_increasing() {
        assert!(AUTO_OMEGA_CANDIDATES.contains(&DEFAULT_RSGDF_OMEGA));
        assert!(AUTO_OMEGA_CANDIDATES.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn g_count_is_cubic_in_omega_and_linear_in_volume() {
        let n = |v, w| half_sphere_g_count(v, w, 1e-13);
        assert!((n(1000.0, 1.0) / n(1000.0, 0.5) - 8.0).abs() < 1e-12);
        assert!((n(2000.0, 0.7) / n(1000.0, 0.7) - 2.0).abs() < 1e-12);
        // dry ice (Ω = 1200.4 Bohr³) at ω = 1: the build counts 13372.
        let dry = n(1200.418105, 1.0);
        assert!((dry / 13372.0 - 1.0).abs() < 0.02, "{dry}");
    }

    #[test]
    fn long_range_dominated_cells_choose_a_smaller_omega_than_short_range_ones() {
        // LR-heavy: the optimum sits below the default.
        let lr_heavy = choose_omega(&table(1.0, 5.0), AUTO_OMEGA_SWITCH_MARGIN);
        // SR-heavy: the optimum sits above it.
        let sr_heavy = choose_omega(&table(500.0, 0.05), AUTO_OMEGA_SWITCH_MARGIN);
        assert!(lr_heavy < DEFAULT_RSGDF_OMEGA, "{lr_heavy}");
        assert!(sr_heavy > DEFAULT_RSGDF_OMEGA, "{sr_heavy}");
    }

    #[test]
    fn omega_is_monotone_in_the_cost_ratio() {
        // Raising the LR cost can only move the choice down, never up.
        let mut last = f64::INFINITY;
        for lr in [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0] {
            let w = choose_omega(&table(5.0, lr), AUTO_OMEGA_SWITCH_MARGIN);
            assert!(w <= last, "lr {lr}: {w} > {last}");
            last = w;
        }
    }

    #[test]
    fn the_choice_is_always_a_candidate() {
        for (sr, lr) in [(1e-9, 1.0), (1.0, 1e-9), (1.0, 1.0), (1e6, 1e-6)] {
            let w = choose_omega(&table(sr, lr), AUTO_OMEGA_SWITCH_MARGIN);
            assert!(AUTO_OMEGA_CANDIDATES.contains(&w), "{w}");
        }
    }

    #[test]
    fn a_flat_optimum_keeps_the_default() {
        // 5% saving at ω = 1.4 < the 10% margin: stay at 1.
        let t: Vec<_> = AUTO_OMEGA_CANDIDATES
            .iter()
            .map(|&w| cand(w, if w == 1.4 { 95.0 } else { 100.0 }))
            .collect();
        assert_eq!(
            choose_omega(&t, AUTO_OMEGA_SWITCH_MARGIN),
            DEFAULT_RSGDF_OMEGA
        );
        // 20% saving crosses it.
        let t: Vec<_> = AUTO_OMEGA_CANDIDATES
            .iter()
            .map(|&w| cand(w, if w == 1.4 { 80.0 } else { 100.0 }))
            .collect();
        assert_eq!(choose_omega(&t, AUTO_OMEGA_SWITCH_MARGIN), 1.4);
        // margin 0 takes any strict improvement; margin 1 never moves.
        assert_eq!(choose_omega(&t, 0.0), 1.4);
        assert_eq!(choose_omega(&t, 1.0), DEFAULT_RSGDF_OMEGA);
    }

    #[test]
    fn degenerate_tables_fall_back_to_the_default() {
        assert_eq!(choose_omega(&[], 0.1), DEFAULT_RSGDF_OMEGA);
        assert_eq!(
            choose_omega(&[cand(0.5, 1.0), cand(0.7, 2.0)], 0.1),
            DEFAULT_RSGDF_OMEGA,
            "no default entry"
        );
        let mut t = table(1.0, 5.0);
        t[2].predicted_s = f64::NAN;
        assert_eq!(choose_omega(&t, 0.1), DEFAULT_RSGDF_OMEGA, "NaN entry");
        let mut t = table(1.0, 5.0);
        t[4].predicted_s = 0.0;
        assert_eq!(choose_omega(&t, 0.1), DEFAULT_RSGDF_OMEGA, "zero reference");
    }

    /// The fitted constants against measured stage times (default build, 6
    /// threads, cc-pVDZ / cc-pVDZ-RI, `omega_sweep.py`; counters from the same
    /// runs): `(cell, ω, SR3 triplets, half-G, W [M], nao²·naux [M],
    /// measured SR 3-centre + LR seconds)`. Three calibration cells and one
    /// held-out. Residuals at the time of the fit: -1%/-6%, +12%, -19%, +13%; the
    /// 25% bar rejects a constant off by 1.3x and any swap of the two LR terms.
    const MEASURED: [(&str, f64, f64, f64, f64, f64, f64); 5] = [
        (
            "dryice",
            1.0,
            15_182_440.0,
            13_372.0,
            0.659_896,
            18.967,
            14.92,
        ),
        (
            "dryice",
            0.5,
            30_105_032.0,
            1_643.0,
            0.659_896,
            18.967,
            6.58,
        ),
        (
            "diamond_prim",
            1.0,
            44_660_940.0,
            843.0,
            0.491_780,
            0.088,
            6.86,
        ),
        ("si_prim", 1.0, 22_367_204.0, 2_980.0, 0.861, 0.197, 5.90),
        (
            "diamond_conv (held out)",
            1.0,
            164_659_920.0,
            3_381.0,
            1.967,
            5.620,
            27.39,
        ),
    ];

    #[test]
    fn fitted_constants_reproduce_the_measured_stage_times() {
        for (name, w, n3, ng, work, gemm, meas) in MEASURED {
            let p = predicted_seconds(&FITTED, n3, ng, work, gemm);
            let rel = p / meas - 1.0;
            assert!(
                rel.abs() < 0.25,
                "{name} at ω = {w}: model {p:.2} s vs {meas} s ({rel:+.2})"
            );
        }
    }

    #[test]
    fn prediction_is_the_sum_of_its_parts() {
        let m = CostModel {
            s_per_sr3_triplet: 2.0,
            s_per_g_per_mwork: 3.0,
            s_per_g_per_mgemm: 5.0,
        };
        // 2·10 + 7·(3·0.5 + 5·0.25) = 20 + 7·2.75
        assert!((predicted_seconds(&m, 10.0, 7.0, 0.5, 0.25) - 39.25).abs() < 1e-12);
    }
}
