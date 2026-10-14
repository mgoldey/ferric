//! Estimated noise floor of the SCF energy change `|dE|`, and the advisory
//! "your `energy_conv` is below it" warning (issue #410).
//!
//! # What the floor is
//!
//! Once the density has settled, `dE = E_n - E_(n-1)` no longer shrinks: it
//! jitters at a level set by the numerics, not by the physics. A tolerance
//! `energy_conv` below that level is met only by chance, so the SCF runs on
//! until the jitter happens to dip under it (12-27 extra iterations on the
//! 71-atom case of #410).
//!
//! # The model (measured; table and fit in the PR that added this module)
//!
//! `floor = rounding + screening`, where
//!
//! * `rounding = ROUND_COEFF * eps * |E| * sqrt(nbf)` is the floating-point
//!   floor of the energy sum. It is what remains with density-fitted J and
//!   in the open-shell solvers (RI-J alkanes C10/C20 and benzene: 3e-13 to
//!   1.3e-12 Ha, within 25 % of this formula).
//! * `screening = SCREEN_COEFF * integral_thresh * nbf^SCREEN_EXP` is added ONLY
//!   for the closed-shell DIRECT path (conventional J and K, default K builder).
//!   That path updates the Fock matrix incrementally from the density change
//!   (`rhf.rs`, full rebuild every 8th iteration); each update drops quartets
//!   whose bound falls below `integral_thresh` for the SMALL density change, and
//!   the dropped pieces accumulate until the next full rebuild. The result is
//!   100-1000x above the rounding floor, grows linearly with `integral_thresh`
//!   (alkane C10/STO-3G: 4.8e-7, 2.4e-8, 9.1e-11, 1.1e-12 Ha at 1e-8, 1e-10,
//!   1e-12, 1e-14), grows with system size, does NOT grow with `|E|` (HBr,
//!   E = -2573, is quieter than SiH4, E = -291, at equal `nbf`), and is
//!   bit-identical across 1, 2 and 4 threads.
//!
//! # Fit range
//!
//! The size dependence was fitted on 48 windowed-RMS points, nbf 24 to 338
//! (water to C48H98, plus the 239-function 71-atom case of #410). The free
//! pooled fit has exponent 2.0, but the points with nbf >= 100 alone give 2.25,
//! and compact 3-D systems are ~2x noisier than chains of equal nbf, so the
//! shipped model uses the steeper exponent with a coefficient set so that the
//! floor is NOT below the measured RMS for any measured system with nbf >= 100
//! (a false alarm at large nbf costs a line of text; a miss costs the user
//! tens of iterations). Small systems are over-predicted by up to ~5x.
//! MEASURED up to nbf ~340; beyond that it is an extrapolation.
//!
//! The estimate is good to a factor of a few (the windowed RMS of one run varies
//! by ~3x between windows; see [`crate::noise_floor::WARN_FACTOR`]), not a
//! bound. Functional (grid) runs, the LinK / COSX exchange builders, PCM /
//! COSMO, polarizable embedding and non-default screening are outside what was
//! measured; [`crate::noise_floor::floor_for_config`] returns `None` for them
//! rather than guess. The point-charge / field energy of an embedding is not in
//! the `|E|` estimate (it enters only the rounding term).
//!
//! The functions here are pure and nothing in the solvers reads them back, so
//! energies and iteration counts are unchanged.

use ferric_core::mol::Molecule;

/// Coefficient of the screening term `SCREEN_COEFF * integral_thresh * nbf^SCREEN_EXP`.
pub const SCREEN_COEFF: f64 = 0.0125;
/// Exponent of `nbf` in the screening term (see the module docs, "Fit range").
pub const SCREEN_EXP: f64 = 2.25;
/// Coefficient of the rounding term `ROUND_COEFF * eps * |E| * sqrt(nbf)`.
pub const ROUND_COEFF: f64 = 0.5;
/// `energy_conv` is reported as below the floor when `energy_conv < WARN_FACTOR * floor`.
pub const WARN_FACTOR: f64 = 0.3;

/// Which solver family runs the SCF. Only the closed-shell solver updates the
/// direct Fock matrix incrementally by default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverFamily {
    /// `solve_rhf` (RHF and the closed-shell references of the post-HF kinds).
    ClosedShell,
    /// `solve_uhf` / `solve_rohf`.
    OpenShell,
}

/// Inputs of the floor estimate, all known before the first SCF iteration
/// except `e_abs`, which can be a rough estimate (see [`estimate_total_energy_abs`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NoiseFloorInputs {
    /// `|E_total|` in Hartree (rounding term only).
    pub e_abs: f64,
    /// Number of basis functions.
    pub nbf: usize,
    /// Direct-SCF integral screening threshold (`RhfConfig::integral_thresh`).
    pub integral_thresh: f64,
    /// Whether the Fock matrix is built by the incremental direct J/K path
    /// (the only source of the screening term).
    pub incremental_direct: bool,
}

/// Estimated RMS of `|dE|` once the density has settled, in Hartree.
pub fn de_noise_floor(i: &NoiseFloorInputs) -> f64 {
    let nbf = i.nbf.max(1) as f64;
    let rounding = ROUND_COEFF * f64::EPSILON * i.e_abs.abs() * nbf.sqrt();
    let screening = if i.incremental_direct {
        SCREEN_COEFF * i.integral_thresh.max(0.0) * nbf.powf(SCREEN_EXP)
    } else {
        0.0
    };
    rounding + screening
}

/// True when `energy_conv` is low enough that the `dE` criterion is met only
/// by chance: `energy_conv < WARN_FACTOR * floor`. A non-positive or NaN
/// `energy_conv` is below any finite floor and also unreachable, so it counts.
/// A non-finite floor (NaN / infinite `|E|`) is no evidence of anything: `false`.
pub fn energy_conv_below_floor(energy_conv: f64, floor: f64) -> bool {
    floor.is_finite() && !(energy_conv >= WARN_FACTOR * floor)
}

/// Crude `|E_total|` from the nuclear charges alone: a Thomas-Fermi atom,
/// `0.6 * Zeff^(7/3)` Hartree per atom, with `Zeff = z - n_core_ecp` (0 for a
/// ghost). The prefactor is the TF value (0.7687) scaled by 0.78 so that the
/// all-electron errors are centred. MEASURED against the SCF energy (def2-SVP
/// unless noted): water +3 %, methane +4 %, benzene +4 %, HBr -6 %, SiH4 -2 %,
/// alkane C10/STO-3G +5 %. An ECP atom is NOT well described (HI/def2-SVP, I with
/// a 28-electron core: estimate 3.7x too high, vs 21x without the core
/// subtraction). Only the rounding term reads this, and that term is
/// `~1e-12 Ha`, three orders below any `energy_conv` anyone sets, so even the
/// ECP overestimate cannot move a verdict on the incremental-direct path, where
/// the screening term (independent of `|E|`) dominates.
pub fn estimate_total_energy_abs(mol: &Molecule) -> f64 {
    mol.atoms
        .iter()
        .map(|a| 0.6 * f64::from(a.effective_z().max(0)).powf(7.0 / 3.0))
        .sum()
}

/// Whether the floor model covers this configuration. Measured: plain HF with
/// the default Schwarz screening and the default (direct) K builder, with or
/// without density-fitted J/K and with or without a point-charge / field
/// embedding. NOT measured, so no claim is made: functionals (grid noise),
/// the LinK / COSX exchange builders, PCM / COSMO solvation, polarizable
/// embedding, and non-default `screening` (CSB, CSAM).
pub fn is_characterised(config: &crate::rhf::RhfConfig) -> bool {
    config.xc.is_none()
        && matches!(config.k_builder.as_deref(), None | Some("direct"))
        && config.pcm.is_none()
        && config.cosmo.is_none()
        && config.polarizable.is_none()
        && config.screening == crate::screening::ScreeningKind::default()
}

/// True when a density-fitting aux request means "fit" (`None` and the empty
/// opt-out sentinel do not).
pub fn aux_is_fitted(aux: Option<&str>) -> bool {
    aux.is_some_and(|s| !s.is_empty())
}

/// Whether the closed-shell solver will take the incremental direct J/K path
/// for this configuration, mirroring `solve_rhf`'s own condition (no density
/// fitting of either matrix, default K builder) and `incremental_enabled` (the
/// solver's `FERRIC_SCF_INCREMENTAL` switch, [`crate::rhf::closed_shell_incremental_enabled`]).
/// Always false for the open-shell solvers.
pub fn incremental_direct_expected(
    config: &crate::rhf::RhfConfig,
    family: SolverFamily,
    incremental_enabled: bool,
) -> bool {
    family == SolverFamily::ClosedShell
        && incremental_enabled
        && is_characterised(config)
        && !aux_is_fitted(config.df_j_aux.as_deref())
        && !aux_is_fitted(config.df_k_aux.as_deref())
}

/// The floor for a configuration, or `None` outside the measured scope
/// ([`is_characterised`]). `incremental_direct` is whether the incremental
/// direct path is (or will be) in use.
pub fn floor_for_config(
    config: &crate::rhf::RhfConfig,
    e_abs: f64,
    nbf: usize,
    incremental_direct: bool,
) -> Option<f64> {
    (is_characterised(config) && e_abs.is_finite()).then(|| {
        de_noise_floor(&NoiseFloorInputs {
            e_abs,
            nbf,
            integral_thresh: config.integral_thresh,
            incremental_direct,
        })
    })
}

/// The advisory text for a user-supplied `energy_conv`, or `None` when the
/// setting is above the estimated floor or the configuration is outside the
/// measured scope.
///
/// Called ONCE, before the first SCF iteration, by the Python and CLI front
/// ends: `nbf` and the molecule are known by then, and the one input that is
/// not (`|E|`) enters only the rounding term, where the nuclear-charge
/// estimate ([`estimate_total_energy_abs`]) is accurate enough. (Waiting for
/// the first iteration's energy would need a side channel out of the solver
/// for no gain in accuracy.)
pub fn energy_conv_warning(
    mol: &Molecule,
    nbf: usize,
    config: &crate::rhf::RhfConfig,
    family: SolverFamily,
) -> Option<String> {
    energy_conv_warning_with(
        mol,
        nbf,
        config,
        family,
        crate::rhf::closed_shell_incremental_enabled(),
    )
}

/// [`energy_conv_warning`] with the incremental-Fock switch given explicitly
/// instead of read from the environment (so tests do not depend on it).
pub fn energy_conv_warning_with(
    mol: &Molecule,
    nbf: usize,
    config: &crate::rhf::RhfConfig,
    family: SolverFamily,
    incremental_enabled: bool,
) -> Option<String> {
    let floor = floor_for_config(
        config,
        estimate_total_energy_abs(mol),
        nbf,
        incremental_direct_expected(config, family, incremental_enabled),
    )?;
    energy_conv_below_floor(config.energy_conv, floor)
        .then(|| floor_message(config.energy_conv, floor, nbf))
}

/// The clause shared by the pre-SCF warning and the exit report.
pub fn floor_sentence(energy_conv: f64, floor: f64) -> String {
    format!(
        "energy_conv {energy_conv:.1e} is below the estimated noise floor of dE (~{floor:.1e} Ha \
         for this system), so the dE criterion can be met only by chance"
    )
}

/// The full advisory.
pub fn floor_message(energy_conv: f64, floor: f64, nbf: usize) -> String {
    format!(
        "{} ({nbf} basis functions). The SCF is judged on the density change (density_conv); \
         dE is only a loose not-still-descending bound, so a tighter energy_conv adds \
         iterations without improving the result. Use an energy_conv above ~{:.0e}, or the \
         default.",
        floor_sentence(energy_conv, floor),
        floor
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rhf::RhfConfig;

    fn inputs(e_abs: f64, nbf: usize, thresh: f64, incr: bool) -> NoiseFloorInputs {
        NoiseFloorInputs {
            e_abs,
            nbf,
            integral_thresh: thresh,
            incremental_direct: incr,
        }
    }

    /// The reporter's 71-atom case (#410): E = -1936, 239 STO-3G functions,
    /// conventional closed-shell SCF at the default `integral_thresh`.
    fn reporter() -> NoiseFloorInputs {
        inputs(1936.0, 239, 1e-12, true)
    }

    fn water() -> Molecule {
        Molecule::parse_xyz("3\nH2O\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.26\n", 0, 1).unwrap()
    }

    #[test]
    fn floor_is_monotone_in_every_input() {
        let base = de_noise_floor(&inputs(300.0, 100, 1e-12, true));
        assert!(de_noise_floor(&inputs(600.0, 100, 1e-12, true)) > base);
        assert!(de_noise_floor(&inputs(300.0, 200, 1e-12, true)) > base);
        assert!(de_noise_floor(&inputs(300.0, 100, 1e-10, true)) > base);
        // The same system on a path without the incremental screening term is
        // quieter, and still positive (rounding).
        let quiet = de_noise_floor(&inputs(300.0, 100, 1e-12, false));
        assert!(quiet > 0.0 && quiet < base);
        // With no incremental path the threshold is irrelevant.
        assert_eq!(
            quiet,
            de_noise_floor(&inputs(300.0, 100, 1e-8, false)),
            "integral_thresh must not enter the rounding-only floor"
        );
    }

    /// The floor against the measured RMS of dE (windowed over the iterations
    /// after the density settled; the table in the PR description): within a
    /// factor 5 everywhere, and NOT below the measurement for any system with
    /// nbf >= 100 (the conservative side at large size).
    #[test]
    fn floor_matches_the_measured_rms_anchors() {
        // (|E|, nbf, measured RMS of dE) on the incremental direct path.
        for (e, nbf, rms) in [
            (76.0, 24, 4.5e-12),   // water / cc-pVDZ
            (231.0, 66, 8.3e-11),  // benzene / 6-31G
            (231.0, 114, 1.4e-10), // benzene / cc-pVDZ
            (387.0, 72, 5.0e-10),  // alkane C10 / STO-3G (the noisy window)
            (391.0, 134, 6.9e-10), // alkane C10 / 6-31G
            (773.0, 142, 5.9e-10), // alkane C20 / STO-3G
            (1875.0, 86, 6.3e-10), // CCl4 / def2-SVP (noisy window)
            (1235.0, 226, 1.1e-9), // alkane C32 / STO-3G
            (781.0, 264, 1.5e-9),  // alkane C20 / 6-31G
            (1853.0, 338, 2.2e-9), // alkane C48 / STO-3G
            (1936.0, 239, 1.9e-9), // the #410 case, 6458-charge field
            (1936.0, 239, 2.7e-9), // the #410 case, vacuum
        ] {
            let f = de_noise_floor(&inputs(e, nbf, 1e-12, true));
            let r = f / rms;
            assert!(
                (0.3..5.0).contains(&r),
                "nbf {nbf}: floor/measured = {r:.2}"
            );
            if nbf >= 100 {
                assert!(r >= 1.0, "nbf {nbf}: floor below measured ({r:.2})");
            }
        }
        // Rounding-only anchors: FERRIC_SCF_INCREMENTAL=0 and RI-J runs.
        for (e, nbf, rms) in [
            (76.0, 24, 4.4e-14),   // water / cc-pVDZ, full rebuild
            (387.0, 72, 3.7e-13),  // alkane C10 / STO-3G, full rebuild
            (231.0, 114, 3.3e-13), // benzene / cc-pVDZ, full rebuild
            (781.0, 264, 1.3e-12), // alkane C20 / 6-31G, RI-JK
        ] {
            let f = de_noise_floor(&inputs(e, nbf, 1e-12, false));
            let r = f / rms;
            assert!(
                (0.65..1.5).contains(&r),
                "nbf {nbf}: floor/measured = {r:.2}"
            );
        }
    }

    /// The warning condition is two-sided: it fires for the reporter's
    /// `energy_conv = 1e-10` and not for the default or the common loose values.
    #[test]
    fn fires_for_the_reporters_setting_and_not_for_ordinary_ones() {
        let f = de_noise_floor(&reporter());
        assert!(energy_conv_below_floor(1e-10, f), "floor {f:e}");
        assert!(energy_conv_below_floor(3e-10, f));
        for ec in [1e-3, 1e-6, 1e-8, 1e-9] {
            assert!(
                !energy_conv_below_floor(ec, f),
                "energy_conv {ec:e} must not warn (floor {f:e})"
            );
        }
        // Medium system (benzene / cc-pVDZ, nbf 114): 1e-9 and above are fine.
        let m = de_noise_floor(&inputs(231.0, 114, 1e-12, true));
        assert!(!energy_conv_below_floor(1e-9, m));
        assert!(energy_conv_below_floor(3e-11, m));
        // Water / STO-3G: even 1e-10 is far above the floor.
        let w = de_noise_floor(&inputs(75.0, 7, 1e-12, true));
        assert!(!energy_conv_below_floor(1e-10, w), "floor {w:e}");
    }

    /// A NaN / infinite energy (a diverged SCF) must not produce a floor or a claim.
    #[test]
    fn non_finite_energy_gives_no_floor_and_no_claim() {
        let c = RhfConfig::default();
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(floor_for_config(&c, bad, 100, true).is_none(), "{bad}");
        }
        assert!(floor_for_config(&c, 100.0, 100, true).is_some());
        for bad in [f64::NAN, f64::INFINITY] {
            assert!(!energy_conv_below_floor(1e-12, bad));
            assert!(!energy_conv_below_floor(0.0, bad));
        }
    }

    /// Beyond the measured range (nbf > ~340) the model is an extrapolation of
    /// the large-system exponent; at 1000 functions 1e-8 is flagged and 1e-6 is not.
    #[test]
    fn extrapolation_beyond_the_fit_range_errs_towards_warning() {
        let f = de_noise_floor(&inputs(5000.0, 1000, 1e-12, true));
        assert!(energy_conv_below_floor(1e-8, f), "floor {f:e}");
        assert!(!energy_conv_below_floor(1e-6, f), "floor {f:e}");
    }

    #[test]
    fn unreachable_energy_conv_counts_as_below_the_floor() {
        let f = de_noise_floor(&inputs(76.0, 24, 1e-12, true));
        assert!(energy_conv_below_floor(0.0, f));
        assert!(energy_conv_below_floor(-1.0, f));
        assert!(energy_conv_below_floor(f64::NAN, f));
    }

    /// Measured error table (see `estimate_total_energy_abs`): water real 75.96.
    #[test]
    fn the_energy_estimate_tracks_the_nuclear_charges() {
        let e = estimate_total_energy_abs(&water());
        assert!((0.9 * 75.96..1.1 * 75.96).contains(&e), "{e}");
        // Ghost centres carry no charge.
        let mut g = water();
        g.atoms[1].ghost = true;
        assert!(estimate_total_energy_abs(&g) < e);
    }

    #[test]
    fn ecp_core_electrons_are_removed_from_the_estimate() {
        let mut hi = Molecule::parse_xyz("2\nHI\nH 0 0 0\nI 0 0 3.0\n", 0, 1).unwrap();
        let all_electron = estimate_total_energy_abs(&hi);
        hi.atoms[1].n_core_ecp = 28;
        let with_ecp = estimate_total_energy_abs(&hi);
        // Real E(HI/def2-SVP) = -297.2; without the subtraction the estimate is ~21x.
        assert!(
            with_ecp < all_electron / 4.0,
            "{with_ecp} vs {all_electron}"
        );
        assert!((200.0..1500.0).contains(&with_ecp), "{with_ecp}");
    }

    #[test]
    fn characterised_scope_excludes_functionals_and_pluggable_k() {
        let mut c = RhfConfig::default();
        assert!(is_characterised(&c));
        c.k_builder = Some("direct".into());
        assert!(is_characterised(&c));
        c.k_builder = Some("link".into());
        assert!(!is_characterised(&c));
        assert!(floor_for_config(&c, 100.0, 50, true).is_none());
        c.k_builder = None;
        c.xc = Some("pbe".into());
        assert!(!is_characterised(&c));
        c.xc = None;
        assert!(is_characterised(&c));
        c.screening = crate::screening::ScreeningKind::Csam;
        assert!(!is_characterised(&c));
        c.screening = crate::screening::ScreeningKind::default();
        c.cosmo = Some(crate::cosmo::CosmoConfig::default());
        assert!(!is_characterised(&c));
    }

    #[test]
    fn incremental_direct_needs_closed_shell_and_no_density_fitting() {
        let mut c = RhfConfig::default();
        // the solver's FERRIC_SCF_INCREMENTAL=0 switch turns the incremental path off
        assert!(!incremental_direct_expected(
            &c,
            SolverFamily::ClosedShell,
            false
        ));
        assert!(incremental_direct_expected(
            &c,
            SolverFamily::ClosedShell,
            true
        ));
        assert!(!incremental_direct_expected(
            &c,
            SolverFamily::OpenShell,
            true
        ));
        c.df_j_aux = Some("def2-universal-jkfit".into());
        assert!(!incremental_direct_expected(
            &c,
            SolverFamily::ClosedShell,
            true
        ));
        c.df_j_aux = Some(String::new()); // "exact" sentinel: not fitted
        assert!(incremental_direct_expected(
            &c,
            SolverFamily::ClosedShell,
            true
        ));
        c.df_k_aux = Some("def2-universal-jkfit".into());
        assert!(!incremental_direct_expected(
            &c,
            SolverFamily::ClosedShell,
            true
        ));
    }

    /// 71 atoms (C31 N5 O4 F2 + 29 H): enough nuclear charge for a realistic
    /// `|E|` estimate; the geometry is irrelevant to the estimator.
    fn big_xyz() -> String {
        let mut s = String::from("71\nbig\n");
        let mut k = 0.0;
        for (sym, n) in [("C", 31), ("N", 5), ("O", 4), ("F", 2), ("H", 29)] {
            for _ in 0..n {
                s.push_str(&format!("{sym} {:.3} 0.0 0.0\n", 1.5 * k));
                k += 1.0;
            }
        }
        s
    }

    /// The pre-SCF entry point end to end on real molecules.
    #[test]
    fn warning_text_names_the_setting_and_the_floor() {
        let mut c = RhfConfig {
            energy_conv: 1e-16,
            ..RhfConfig::default()
        };
        let text = energy_conv_warning_with(&water(), 7, &c, SolverFamily::ClosedShell, true)
            .expect("fires");
        assert!(text.contains("energy_conv 1.0e-16"), "{text}");
        assert!(text.contains("noise floor"), "{text}");
        assert!(text.contains("7 basis functions"), "{text}");
        c.energy_conv = 1e-3;
        assert!(
            energy_conv_warning_with(&water(), 7, &c, SolverFamily::ClosedShell, true).is_none()
        );
        // Reporter class: 239 functions, `|E|` from the estimate.
        c.energy_conv = 1e-10;
        let big = Molecule::parse_xyz(&big_xyz(), 0, 1).unwrap();
        assert!(energy_conv_warning_with(&big, 239, &c, SolverFamily::ClosedShell, true).is_some());
        assert!(
            energy_conv_warning_with(&big, 239, &c, SolverFamily::OpenShell, true).is_none(),
            "open-shell solvers rebuild J/K every iteration: rounding-level floor"
        );
        c.df_j_aux = Some("def2-universal-jkfit".into());
        assert!(energy_conv_warning_with(&big, 239, &c, SolverFamily::ClosedShell, true).is_none());
    }
}
