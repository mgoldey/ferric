//! Analytic nuclear gradient of SCS-MP2(2terfc), the energy of
//! [`crate::scs::scs_mp2_2terfc`]:
//!
//! ```text
//!   E = E_HF[1/r] + c_os E_OS[τ1] + c_ss (E_SS[τ2] − E_SS[τ1]),
//!   τk = terfc(r; r0(k))/r,  r0(2) > r0(1)
//! ```
//!
//! The correlation part is a sum of two RI-MP2-shaped terms on one Coulomb
//! reference, each a spin-component scaling of RI-MP2 on its own operator:
//!
//! ```text
//!   term 1: τ1, (c_os, −c_ss)      term 2: τ2, (0, c_ss)
//! ```
//!
//! Each term builds its amplitudes, scaled unrelaxed density, `Imat` and
//! 3c/2c RI response on its own terfc kernel (the WD-1 terfc pieces,
//! `book/src/15-correlated-gradients/02-rimp2-gradient.md#rimp2-gradient-attenuated-erfc-ri-mp2`,
//! with the spin-component split of
//! `book/src/15-correlated-gradients/02-rimp2-gradient.md#rimp2-gradient-scs-mp2`). The reference,
//! its Fock
//! stationarity and the Z-vector stay Coulomb, and since every first
//! variation of the energy is linear in `(Imat, P)`, the two terms close ONE
//! Z-vector on the summed right-hand side
//! (`book/src/15-correlated-gradients/02-rimp2-gradient.md#rimp2-gradient`). On a UHF
//! reference the same two terms run through the unrestricted gradient, with
//! the same-spin (`αα`, `ββ`) and opposite-spin (`αβ`) products weighted
//! separately
//! (`book/src/15-correlated-gradients/07-u-rimp2-gradient.md#u-rimp2-gradient-terfc-scs-mp2-2terfc`).
//! A ROHF reference is refused, as for every
//! unrestricted RI-MP2 gradient
//! ([`crate::u_rimp2_gradient::ROHF_REFUSAL`]).

use crate::gradient::{rimp2_terms_gradient_timed, solve_rhf_reference, CorrelationTerm};
use crate::rimp2::{RiMp2Config, SpinScaling};
use crate::scs::{check_scs_2terfc_config, scs_mp2_2terfc, ScsMp2TerfcConfig};
use crate::u_rimp2_gradient::{solve_reference, u_terms_gradient_timed, UCorrelation};
use crate::zvector::ZvectorControls;
use ferric_core::basis::BasisSet;
use ferric_core::external_potential::ExternalPotential;
use ferric_core::mol::Molecule;
use ferric_core::timing::PhaseAccumulator;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::{Operator, OperatorKind};
use ferric_scf::rhf::RhfConfig;
use ferric_scf::scf_jk::ScfJk;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::warm_start::ScfWarmStart;
use ferric_scf::ScfResult;
use ndarray::Array2;

/// The two correlation terms of `config` (operator, spin scaling), checked
/// as [`scs_mp2_2terfc`] checks them, plus finite coefficients.
pub(crate) fn scs_2terfc_terms(
    config: &ScsMp2TerfcConfig,
) -> Result<[(Operator, SpinScaling); 2], FerricError> {
    check_scs_2terfc_config(config)?;
    if !(config.c_os.is_finite() && config.c_ss.is_finite()) {
        return Err(FerricError::General(format!(
            "SCS-MP2(2terfc) gradient: c_os and c_ss must be finite (got {}, {})",
            config.c_os, config.c_ss
        )));
    }
    Ok([
        (
            Operator::terfc(config.r0_bonded),
            SpinScaling {
                c_os: config.c_os,
                c_ss: -config.c_ss,
            },
        ),
        (
            Operator::terfc(config.r0_nonbonded),
            SpinScaling {
                c_os: 0.0,
                c_ss: config.c_ss,
            },
        ),
    ])
}

/// The RI-MP2 configuration [`scs_mp2_2terfc`] computes its energy with.
fn scs_2terfc_ri(config: &ScsMp2TerfcConfig) -> RiMp2Config {
    RiMp2Config {
        frozen_core: config.frozen_core,
        memory_budget_bytes: config.memory_budget_bytes,
        ..Default::default()
    }
}

fn require_coulomb_reference(op: Operator) -> Result<(), FerricError> {
    if op.is_composite() || op.kind != OperatorKind::Coulomb {
        return Err(FerricError::General(format!(
            "SCS-MP2(2terfc) gradient: the reference SCF must be Coulomb HF (its Schwarz bounds \
             are for {:?})",
            op.kind
        )));
    }
    Ok(())
}

fn closed_terms(terms: [(Operator, SpinScaling); 2]) -> [CorrelationTerm; 2] {
    terms.map(|(op, scaling)| CorrelationTerm { op, scaling })
}

fn open_terms(terms: [(Operator, SpinScaling); 2]) -> [UCorrelation; 2] {
    terms.map(|(op, scaling)| UCorrelation { op, scaling })
}

/// `dE/dR`, `(natoms, 3)`, of the SCS-MP2(2terfc) total
/// [`scs_mp2_2terfc`] reports for `config` on the converged Coulomb RHF
/// `rhf`, whose J/K route `bounds` describes (four-centre, or the RI-JK
/// route `rhf` carries). `ext` adds the external potential's
/// Hellmann-Feynman and classical terms, as in
/// [`crate::gradient::rimp2_gradient_analytical`].
///
/// # Errors
///
/// A non-Coulomb or X2C reference, the `config` refusals of
/// [`scs_mp2_2terfc`] (non-finite or non-positive cutoffs, `r0(2) ≤ r0(1)`),
/// non-finite coefficients, and the errors of the RI-MP2 gradient
/// (unconverged reference or Z-vector).
pub fn scs_mp2_2terfc_gradient_analytical(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    bounds: &SchwarzBounds,
    rhf: &ScfResult,
    config: &ScsMp2TerfcConfig,
    ext: Option<&ExternalPotential>,
) -> Result<Array2<f64>, FerricError> {
    ferric_scf::x2c::refuse_relativistic(rhf.relativistic, "SCS-MP2(2terfc) analytic gradient")?;
    require_coulomb_reference(bounds.op)?;
    let terms = closed_terms(scs_2terfc_terms(config)?);
    let mut acc = PhaseAccumulator::new().report_on_drop("SCS-MP2(2terfc) gradient");
    rimp2_terms_gradient_timed(
        mol,
        obs,
        dfbs,
        &terms,
        bounds,
        rhf,
        &scs_2terfc_ri(config),
        ext,
        ZvectorControls::DEFAULT,
        &mut acc,
    )
}

/// The total closed-shell SCS-MP2(2terfc) energy (Coulomb RHF solved here at
/// `density_conv = 1e-9`) and its analytic gradient at one geometry, the
/// reference on `scf_jk` (`"exact"` or `"rijk"`) and warm-started through
/// `warm`.
pub fn total_scs_mp2_2terfc_gradient_on(
    mol: &Molecule,
    obs_basis: &BasisSet,
    aux_basis: &BasisSet,
    config: &ScsMp2TerfcConfig,
    ext: Option<&ExternalPotential>,
    scf_jk: &ScfJk,
    warm: &mut ScfWarmStart,
) -> Result<(f64, Array2<f64>), FerricError> {
    let terms = closed_terms(scs_2terfc_terms(config)?);
    let ri = scs_2terfc_ri(config);
    let (rhf, obs, dfbs, bounds) = solve_rhf_reference(
        mol,
        obs_basis,
        aux_basis,
        Operator::coulomb(),
        &ri,
        ext,
        scf_jk,
        warm,
    )?;
    let mut acc = PhaseAccumulator::new().report_on_drop("SCS-MP2(2terfc) energy + gradient");
    let energy = acc.time("scs_mp2_2terfc energy", || {
        scs_mp2_2terfc(mol, &obs, &dfbs, &rhf, config).map(|r| r.total_energy)
    })?;
    let grad = rimp2_terms_gradient_timed(
        mol,
        &obs,
        &dfbs,
        &terms,
        &bounds,
        &rhf,
        &ri,
        ext,
        ZvectorControls::DEFAULT,
        &mut acc,
    )?;
    Ok((energy, grad))
}

/// `dE/dR`, `(natoms, 3)`, of the SCS-MP2(2terfc) total
/// [`scs_mp2_2terfc`] reports for `config` on the converged, aufbau Coulomb
/// UHF `uhf` (`E_OS = E_αβ`, `E_SS = E_αα + E_ββ`), whose J/K route `bounds`
/// describes. `ext` as in
/// [`crate::u_rimp2_gradient::u_rimp2_gradient_analytical`].
///
/// # Errors
///
/// The errors of [`crate::u_rimp2_gradient::u_rimp2_gradient_analytical`]
/// (a ROHF, restricted, unconverged, non-aufbau or X2C reference; no beta
/// electron; an unconverged Z-vector), a non-Coulomb reference, and the
/// `config` refusals of [`scs_mp2_2terfc_gradient_analytical`].
pub fn u_scs_mp2_2terfc_gradient_analytical(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    bounds: &SchwarzBounds,
    uhf: &ScfResult,
    config: &ScsMp2TerfcConfig,
    ext: Option<&ExternalPotential>,
) -> Result<Array2<f64>, FerricError> {
    require_coulomb_reference(bounds.op)?;
    let terms = open_terms(scs_2terfc_terms(config)?);
    let mut acc = PhaseAccumulator::new().report_on_drop("U-SCS-MP2(2terfc) gradient");
    u_terms_gradient_timed(
        mol,
        obs,
        dfbs,
        &terms,
        bounds,
        uhf,
        &scs_2terfc_ri(config),
        ext,
        ZvectorControls::DEFAULT,
        &mut acc,
    )
}

/// The total open-shell SCS-MP2(2terfc) energy (the Coulomb UHF of the
/// molecule's charge and multiplicity solved here at `density_conv = 1e-9`)
/// and its analytic gradient at one geometry, the reference on `scf_jk`
/// (`"exact"` or `"rijk"`) and warm-started through `warm`.
pub fn total_u_scs_mp2_2terfc_gradient_on(
    mol: &Molecule,
    obs_basis: &BasisSet,
    aux_basis: &BasisSet,
    config: &ScsMp2TerfcConfig,
    ext: Option<&ExternalPotential>,
    scf_jk: &ScfJk,
    warm: &mut ScfWarmStart,
) -> Result<(f64, Array2<f64>), FerricError> {
    let scf_base = RhfConfig {
        density_conv: 1e-9,
        external_potential: ext.cloned(),
        ..Default::default()
    };
    total_u_scs_mp2_2terfc_gradient_with(
        mol, obs_basis, aux_basis, config, ext, scf_jk, warm, scf_base,
    )
}

/// [`total_u_scs_mp2_2terfc_gradient_on`] with the UHF solved on `scf_base`
/// (before the geometry-task and `scf_jk` adjustments), for the
/// finite-difference tests.
#[allow(clippy::too_many_arguments)]
#[doc(hidden)]
pub fn total_u_scs_mp2_2terfc_gradient_with(
    mol: &Molecule,
    obs_basis: &BasisSet,
    aux_basis: &BasisSet,
    config: &ScsMp2TerfcConfig,
    ext: Option<&ExternalPotential>,
    scf_jk: &ScfJk,
    warm: &mut ScfWarmStart,
    scf_base: RhfConfig,
) -> Result<(f64, Array2<f64>), FerricError> {
    let terms = open_terms(scs_2terfc_terms(config)?);
    let ri = scs_2terfc_ri(config);
    let (uhf, obs, dfbs, bounds) = solve_reference(
        mol,
        obs_basis,
        aux_basis,
        Operator::coulomb(),
        &ri,
        scf_jk,
        warm,
        scf_base,
    )?;
    let mut acc = PhaseAccumulator::new().report_on_drop("U-SCS-MP2(2terfc) energy + gradient");
    let energy = acc.time("scs_mp2_2terfc energy", || {
        scs_mp2_2terfc(mol, &obs, &dfbs, &uhf, config).map(|r| r.total_energy)
    })?;
    let grad = u_terms_gradient_timed(
        mol,
        &obs,
        &dfbs,
        &terms,
        &bounds,
        &uhf,
        &ri,
        ext,
        ZvectorControls::DEFAULT,
        &mut acc,
    )?;
    Ok((energy, grad))
}

/// The open-shell SCS-MP2(2terfc) energy on the reference of
/// [`total_u_scs_mp2_2terfc_gradient_with`] (same SCF settings and route),
/// without the gradient: the finite-difference reference of the tests.
/// Returns the UHF too, so a test can check its convergence.
#[doc(hidden)]
pub fn total_u_scs_mp2_2terfc_energy_with(
    mol: &Molecule,
    obs_basis: &BasisSet,
    aux_basis: &BasisSet,
    config: &ScsMp2TerfcConfig,
    scf_jk: &ScfJk,
    scf_base: RhfConfig,
) -> Result<(f64, ScfResult), FerricError> {
    let (uhf, obs, dfbs, _) = solve_reference(
        mol,
        obs_basis,
        aux_basis,
        Operator::coulomb(),
        &scs_2terfc_ri(config),
        scf_jk,
        &mut ScfWarmStart::new(false),
        scf_base,
    )?;
    let e = scs_mp2_2terfc(mol, &obs, &dfbs, &uhf, config)?.total_energy;
    Ok((e, uhf))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gradient::rimp2_gradient_analytical_timed;
    use crate::u_rimp2_gradient::u_correlated_gradient_timed;
    use ferric_core::basis;
    use ferric_core::units::ANGSTROM_TO_BOHR;
    use ferric_scf::rhf::solve_rhf;
    use ferric_scf::scf_jk::ScfJkMode;
    use ferric_scf::Spin;
    use ferric_testkit::compare::{max_abs, max_abs_diff};
    use std::path::Path;

    const WATER: &str = "3\nwater, off C2v\nO 0.02 -0.01 0.117790\nH 0.0 0.785453 -0.451161\nH 0.03 -0.735453 -0.491161\n";
    const OH: &str = "2\nOH\nO 0 0 0\nH 0 0 0.97\n";

    /// Every anchor compares Z-vector solves, so each sits far below the bar.
    const TIGHT: ZvectorControls = ZvectorControls {
        max_iter: 200,
        tol: 1e-12,
    };

    /// The bar of the anchors. Each compares gradients assembled from up to
    /// four Z-vector solves with `max |r| < 1e-12`; one residual moves the
    /// gradient by `≲ ‖∂_R Xvo‖ ‖A⁻¹‖ · 1e-12 ≲ 20 · 1e-12`, four of them
    /// `≲ 8e-11`. Operation-order rounding of the summed `Imat`, `P` and RI
    /// response is `≈ 1e-14` relative to `max |g| ≲ 0.1`. So `1e-10`.
    const ANCHOR_BAR: f64 = 1e-10;

    static TABLES: std::sync::Once = std::sync::Once::new();

    /// The terfc table engine reads `FERRIC_TERF_TABLE_DIR`; point it at the
    /// committed tables when unset, and fail (never skip) when absent.
    fn ensure_tables() {
        TABLES.call_once(|| {
            if std::env::var("FERRIC_TERF_TABLE_DIR").is_ok() {
                return;
            }
            let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../terf-tables");
            assert!(
                repo.join("16_4_2.bin").is_file(),
                "terfc tables not found at {}",
                repo.display()
            );
            std::env::set_var("FERRIC_TERF_TABLE_DIR", &repo);
        });
    }

    fn scf_cfg(fit: bool) -> RhfConfig {
        let mut cfg = RhfConfig {
            density_conv: 1e-11,
            max_iter: 300,
            ..Default::default()
        }
        .for_geometry_task();
        if fit {
            ScfJk {
                mode: ScfJkMode::RiJk,
                aux: Some("def2-universal-jkfit".to_string()),
            }
            .apply(&mut cfg);
        }
        cfg
    }

    /// The published aTZ parameters (r0 0.75 / 1.05 Å, c_os 1.27, c_ss 4.05)
    /// with `frozen_core`.
    fn published(frozen_core: usize) -> ScsMp2TerfcConfig {
        ScsMp2TerfcConfig {
            frozen_core,
            ..Default::default()
        }
    }

    struct Closed {
        mol: Molecule,
        obs: PreparedBasis,
        dfbs: PreparedBasis,
        bounds: SchwarzBounds,
        rhf: ScfResult,
    }

    fn closed(xyz: &str, fit: bool) -> Closed {
        ensure_tables();
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("6-31g").unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        let coul = Operator::coulomb();
        let bounds = SchwarzBounds::compute(coul, &obs).unwrap();
        let ctx = ferric_core::parallel::ParallelContext::default();
        let rhf = solve_rhf(&ctx, &mol, &obs, coul, &bounds, &scf_cfg(fit)).unwrap();
        assert!(rhf.converged);
        assert_eq!(rhf.df_jk.is_some(), fit, "route not recorded");
        Closed {
            mol,
            obs,
            dfbs,
            bounds,
            rhf,
        }
    }

    impl Closed {
        fn terms(&self, terms: &[(Operator, SpinScaling)], fc: usize) -> Array2<f64> {
            let terms: Vec<CorrelationTerm> = terms
                .iter()
                .map(|&(op, scaling)| CorrelationTerm { op, scaling })
                .collect();
            rimp2_terms_gradient_timed(
                &self.mol,
                &self.obs,
                &self.dfbs,
                &terms,
                &self.bounds,
                &self.rhf,
                &RiMp2Config {
                    frozen_core: fc,
                    ..Default::default()
                },
                None,
                TIGHT,
                &mut PhaseAccumulator::new(),
            )
            .unwrap()
        }

        /// The established single-term path (its own Z-vector).
        fn single(&self, op: Operator, scaling: SpinScaling, fc: usize) -> Array2<f64> {
            rimp2_gradient_analytical_timed(
                &self.mol,
                &self.obs,
                &self.dfbs,
                op,
                &self.bounds,
                &self.rhf,
                &RiMp2Config {
                    frozen_core: fc,
                    ..Default::default()
                },
                scaling,
                None,
                TIGHT,
                &mut PhaseAccumulator::new(),
            )
            .unwrap()
        }
    }

    /// INDEPENDENT CONSTRUCTION. The two-term gradient with ONE Z-vector on
    /// the summed right-hand side equals the sum of the two single-term
    /// gradients (each its own Z-vector, the established RI-MP2 path) minus
    /// one copy of the reference gradient (a term with scaling `(0, 0)`,
    /// whose correlation response vanishes). Non-vacuity: the two term
    /// gradients each differ from the reference by more than 1e-4.
    fn one_zvector_is_the_sum(fit: bool, fc: usize) {
        let c = closed(WATER, fit);
        let [t1, t2] = scs_2terfc_terms(&published(fc)).unwrap();
        let g = c.terms(&[t1, t2], fc);
        let g1 = c.single(t1.0, t1.1, fc);
        let g2 = c.single(t2.0, t2.1, fc);
        let g0 = c.single(
            t1.0,
            SpinScaling {
                c_os: 0.0,
                c_ss: 0.0,
            },
            fc,
        );
        let d = max_abs_diff(&g, &(&(&g1 + &g2) - &g0));
        let (n1, n2) = (max_abs_diff(&g1, &g0), max_abs_diff(&g2, &g0));
        eprintln!(
            "2terfc fit={fit} fc={fc}: max|g - (g1 + g2 - g0)| = {d:.3e}; term sizes \
             {n1:.3e}, {n2:.3e}; max|g| = {:.3e}",
            max_abs(&g)
        );
        assert!(d < ANCHOR_BAR, "one Z-vector vs two: {d:.3e}");
        assert!(n1 > 1e-4 && n2 > 1e-4, "vacuous terms: {n1:.3e}, {n2:.3e}");
    }

    #[test]
    fn scs_2terfc_one_zvector_is_the_sum_of_its_terms() {
        one_zvector_is_the_sum(false, 0);
    }

    #[test]
    fn scs_2terfc_one_zvector_is_the_sum_of_its_terms_rijk_frozen_core() {
        one_zvector_is_the_sum(true, 1);
    }

    /// EXACTNESS ANCHOR (trivial limit). With both cutoffs equal the
    /// same-spin terms cancel, `E_SS[τ] − E_SS[τ] = 0`, and the energy is
    /// `c_os E_OS[τ]`: the gradient is the single-term SOS gradient
    /// `(c_os, 0)` on `τ`. Non-vacuity: at the published cutoffs the gradient
    /// differs from that SOS gradient by more than 1e-4.
    #[test]
    fn scs_2terfc_gradient_matches_sos_in_the_trivial_limit() {
        let c = closed(WATER, false);
        let cfg = published(0);
        let tau = Operator::terfc(cfg.r0_bonded);
        let coincide = [
            (
                tau,
                SpinScaling {
                    c_os: cfg.c_os,
                    c_ss: -cfg.c_ss,
                },
            ),
            (
                tau,
                SpinScaling {
                    c_os: 0.0,
                    c_ss: cfg.c_ss,
                },
            ),
        ];
        let g = c.terms(&coincide, 0);
        let sos = c.single(
            tau,
            SpinScaling {
                c_os: cfg.c_os,
                c_ss: 0.0,
            },
            0,
        );
        let d = max_abs_diff(&g, &sos);
        let terms = scs_2terfc_terms(&cfg).unwrap();
        let gap = max_abs_diff(&c.terms(&terms, 0), &sos);
        eprintln!("r0(1) = r0(2): max|g - g_SOS| = {d:.3e}; published gap {gap:.3e}");
        assert!(d < ANCHOR_BAR, "trivial limit: {d:.3e}");
        assert!(gap > 1e-4, "non-vacuity: {gap:.3e}");
    }

    /// The RHF written as the UHF with identical α and β orbitals, densities
    /// and Fock matrices: a UHF stationary point.
    fn lift_to_unrestricted(rhf: &ScfResult) -> ScfResult {
        let mut u = rhf.clone();
        u.spin = Spin::Unrestricted;
        u.density_beta = Some(rhf.density_alpha.clone());
        u.mos_beta = Some(rhf.mos_alpha.clone());
        u.eps_beta = Some(rhf.eps_alpha.clone());
        u.fock_beta = Some(rhf.fock_alpha.clone());
        u
    }

    fn u_terms(
        c: &Closed,
        uhf: &ScfResult,
        terms: &[(Operator, SpinScaling)],
        fc: usize,
    ) -> Array2<f64> {
        let terms: Vec<UCorrelation> = terms
            .iter()
            .map(|&(op, scaling)| UCorrelation { op, scaling })
            .collect();
        u_terms_gradient_timed(
            &c.mol,
            &c.obs,
            &c.dfbs,
            &terms,
            &c.bounds,
            uhf,
            &RiMp2Config {
                frozen_core: fc,
                ..Default::default()
            },
            None,
            TIGHT,
            &mut PhaseAccumulator::new(),
        )
        .unwrap()
    }

    /// EXACTNESS ANCHOR (closed-shell limit). A closed-shell RHF written as a
    /// UHF runs the open-shell terfc and SCS-MP2(2terfc) gradients; each must
    /// equal its closed-shell gradient on the same orbitals. The
    /// SCS-MP2(2terfc) terms carry `c_os ≠ c_ss`, so this also checks the
    /// same-spin / opposite-spin weighting of `x^σ` and `P^σ` against the
    /// closed-shell `T̃` split.
    fn closed_limit(fit: bool, fc: usize) {
        let c = closed(WATER, fit);
        let uhf = lift_to_unrestricted(&c.rhf);
        let tau = Operator::terfc(0.75 * ANGSTROM_TO_BOHR);
        let plain = [(tau, SpinScaling::MP2)];
        let d_terfc = max_abs_diff(&u_terms(&c, &uhf, &plain, fc), &c.terms(&plain, fc));
        let terms = scs_2terfc_terms(&published(fc)).unwrap();
        let g_r = c.terms(&terms, fc);
        let d_scs = max_abs_diff(&u_terms(&c, &uhf, &terms, fc), &g_r);
        eprintln!(
            "closed limit fit={fit} fc={fc}: terfc {d_terfc:.3e}, 2terfc {d_scs:.3e}; \
             max|g| = {:.3e}",
            max_abs(&g_r)
        );
        assert!(d_terfc < ANCHOR_BAR, "terfc closed limit: {d_terfc:.3e}");
        assert!(d_scs < ANCHOR_BAR, "2terfc closed limit: {d_scs:.3e}");
    }

    #[test]
    fn open_shell_terfc_gradients_match_closed_in_the_trivial_limit() {
        closed_limit(false, 0);
    }

    #[test]
    fn open_shell_terfc_gradients_match_closed_in_the_trivial_limit_rijk_frozen_core() {
        closed_limit(true, 1);
    }

    /// INDEPENDENT CONSTRUCTION on a genuine open shell (OH doublet, UHF):
    /// the two-term gradient with one coupled Z-vector equals the sum of the
    /// single-term gradients minus the reference gradient.
    #[test]
    fn open_shell_scs_2terfc_one_zvector_is_the_sum_of_its_terms() {
        ensure_tables();
        let mol = Molecule::parse_xyz(OH, 0, 2).unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("6-31g").unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        let coul = Operator::coulomb();
        let bounds = SchwarzBounds::compute(coul, &obs).unwrap();
        let ctx = ferric_core::parallel::ParallelContext::default();
        let uhf = ferric_scf::uhf::solve_uhf(&ctx, &mol, &obs, &bounds, &scf_cfg(false)).unwrap();
        assert!(uhf.converged && uhf.spin == Spin::Unrestricted);
        let ri = RiMp2Config::default();
        let one = |corr: UCorrelation| {
            u_correlated_gradient_timed(
                &mol,
                &obs,
                &dfbs,
                corr,
                &bounds,
                &uhf,
                &ri,
                None,
                TIGHT,
                &mut PhaseAccumulator::new(),
            )
            .unwrap()
        };
        let terms = open_terms(scs_2terfc_terms(&published(0)).unwrap());
        let g = u_terms_gradient_timed(
            &mol,
            &obs,
            &dfbs,
            &terms,
            &bounds,
            &uhf,
            &ri,
            None,
            TIGHT,
            &mut PhaseAccumulator::new(),
        )
        .unwrap();
        let g0 = one(UCorrelation::uniform(terms[0].op, 0.0));
        let sum = &(&one(terms[0]) + &one(terms[1])) - &g0;
        let d = max_abs_diff(&g, &sum);
        let size = max_abs_diff(&g, &g0);
        eprintln!("OH 2terfc: max|g - (g1 + g2 - g0)| = {d:.3e}; correlation part {size:.3e}");
        assert!(d < ANCHOR_BAR, "{d:.3e}");
        assert!(size > 1e-4, "non-vacuity: {size:.3e}");
    }

    /// The public entry points refuse a ROHF reference, a non-Coulomb
    /// reference and the configurations the energy refuses.
    #[test]
    fn scs_2terfc_gradient_refuses_what_it_does_not_differentiate() {
        ensure_tables();
        let mol = Molecule::parse_xyz(OH, 0, 2).unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("sto-3g").unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        let coul = Operator::coulomb();
        let bounds = SchwarzBounds::compute(coul, &obs).unwrap();
        let ctx = ferric_core::parallel::ParallelContext::default();
        let good = published(0);
        let rohf =
            ferric_scf::rohf::solve_rohf(&ctx, &mol, &obs, coul, &bounds, &RhfConfig::default())
                .unwrap();
        let err =
            u_scs_mp2_2terfc_gradient_analytical(&mol, &obs, &dfbs, &bounds, &rohf, &good, None)
                .unwrap_err();
        assert!(format!("{err}").contains("needs a UHF reference"), "{err}");
        let uhf =
            ferric_scf::uhf::solve_uhf(&ctx, &mol, &obs, &bounds, &RhfConfig::default()).unwrap();
        assert!(u_scs_mp2_2terfc_gradient_analytical(
            &mol, &obs, &dfbs, &bounds, &uhf, &good, None
        )
        .is_ok());
        let erf_bounds = SchwarzBounds::compute(Operator::erf(0.3), &obs).unwrap();
        assert!(u_scs_mp2_2terfc_gradient_analytical(
            &mol,
            &obs,
            &dfbs,
            &erf_bounds,
            &uhf,
            &good,
            None
        )
        .is_err());
        for bad in [
            ScsMp2TerfcConfig {
                r0_nonbonded: good.r0_bonded,
                ..good.clone()
            },
            ScsMp2TerfcConfig {
                r0_bonded: f64::NAN,
                ..good.clone()
            },
            ScsMp2TerfcConfig {
                c_ss: f64::INFINITY,
                ..good.clone()
            },
        ] {
            assert!(u_scs_mp2_2terfc_gradient_analytical(
                &mol, &obs, &dfbs, &bounds, &uhf, &bad, None
            )
            .is_err());
        }
        let rohf_terfc = crate::att_rimp2_gradient::u_terfc_rimp2_gradient_analytical(
            &mol,
            &obs,
            &dfbs,
            &bounds,
            &rohf,
            Operator::terfc(1.4),
            &RiMp2Config::default(),
            None,
        )
        .unwrap_err();
        assert!(
            format!("{rohf_terfc}").contains("needs a UHF reference"),
            "{rohf_terfc}"
        );
    }
}
