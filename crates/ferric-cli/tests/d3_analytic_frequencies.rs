//! KS + D3(BJ) frequencies on the analytic route
//! (`ferric_scf::frequencies::harmonic_frequencies_with_geometric_correction`):
//! the analytic moving-grid KS Hessian plus `ferric_d3::d3bj_hessian`, against
//! the finite-difference route it replaced (central differences of the
//! corrected analytic gradient).
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo test -p ferric-cli --release --test d3_analytic_frequencies
//! ```
//!
//! * `d3_analytic_route_matches_the_fd_route_within_its_truncation_bar`:
//!   B3LYP-D3(BJ)/def2-SVP water. The FD route runs at `δ` and `δ/2`; its
//!   truncation error at `δ` is `(4/3)|H(δ) − H(δ/2)|` to leading order, and
//!   the bar is twice that (Hessian entries and wavenumbers alike). The
//!   analytic route must be `AnalyticMovingGrid` with no gradient
//!   evaluations, and must equal the plain analytic KS Hessian plus
//!   `d3bj_hessian` to round-off (the D3 part is ~1e-5 Ha/Bohr², below the FD
//!   bar, so this exactness check is what sees a dropped D3 Hessian).
//! * `explicit_moving_grid_refusals_name_the_reference`: ROKS and a
//!   range-separated hybrid with D3 and `hessian = AnalyticMovingGrid` are
//!   refused by name, before the SCF; `Auto` runs finite differences.
//!
//! # Mutations (each measured to fail)
//!
//! * `with_correction_hessian` adds no D3 Hessian (`&res.cartesian_hessian +
//!   &c.hessian` -> `res.cartesian_hessian.clone()`): the exactness check;
//! * the analytic route skipped (`analytic_route` call replaced by `None`,
//!   D3 silently back on finite differences): the `hessian_source` check.

use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::operator::Operator;
use ferric_scf::frequencies::{
    harmonic_frequencies, harmonic_frequencies_with_geometric_correction, FrequencyConfig,
    FrequencyReference, FrequencyResult, GeometricCorrection, HessianMethod, HessianSource,
};
use ferric_scf::rhf::RhfConfig;
use ferric_testkit::compare::max_abs;
use ndarray::Array2;

/// Near-equilibrium water (Angstrom), tilted off the symmetry planes.
const H2O: &str = "3\nwater\nO 0.010 -0.020 0.117\nH 0.020 0.757 -0.469\nH -0.010 -0.757 -0.470\n";
const BASIS: &str = "def2-svp";
const XC: &str = "B3LYP";
const DELTA: f64 = 5e-3;

fn cfg(xc: &str) -> RhfConfig {
    RhfConfig {
        xc: Some(xc.into()),
        ri_j_precision: Some(ferric_scf::df_j::RiJPrecision::F64),
        max_iter: 300,
        // Tight: the FD route's noise is ~err/δ (SCF error 1e-11 -> 2e-9).
        energy_conv: 1e-13,
        density_conv: 1e-11,
        ..Default::default()
    }
}

fn d3(m: &Molecule) -> Result<GeometricCorrection, ferric_core::FerricError> {
    let p = ferric_d3::d3bj_params_for_functional(XC)?;
    let rows = ferric_d3::d3bj_gradient_for_molecule(m, &p)?;
    Ok(GeometricCorrection {
        energy: ferric_d3::d3bj_energy_for_molecule(m, &p)?,
        gradient: Array2::from_shape_fn((rows.len(), 3), |(k, a)| rows[k][a]),
        hessian: ferric_d3::d3bj_hessian_for_molecule(m, &p)?,
    })
}

fn run(mol: &Molecule, c: &RhfConfig, f: &FrequencyConfig) -> FrequencyResult {
    harmonic_frequencies_with_geometric_correction(
        &ParallelContext::default(),
        mol,
        BASIS,
        Operator::coulomb(),
        c,
        f,
        d3,
    )
    .expect("KS + D3 frequencies")
}

fn max_dnu(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
}

#[test]
fn d3_analytic_route_matches_the_fd_route_within_its_truncation_bar() {
    if ferric_integrals::engine::libint_max_deriv_order() < 2 {
        eprintln!("SKIP: libint2 has no second derivatives");
        return;
    }
    let mol = Molecule::parse_xyz(H2O, 0, 1).unwrap();
    let c = cfg(XC);
    let an = run(&mol, &c, &FrequencyConfig::default());
    assert_eq!(an.hessian_source, HessianSource::AnalyticMovingGrid);
    assert_eq!(an.n_gradient_evaluations, 0);

    // Exactness of the plumbing: analytic route = plain analytic KS + D3.
    let plain = harmonic_frequencies(
        &ParallelContext::default(),
        &mol,
        BASIS,
        Operator::coulomb(),
        &c,
        &FrequencyConfig::default(),
    )
    .unwrap();
    assert_eq!(plain.hessian_source, HessianSource::AnalyticMovingGrid);
    let h_d3 = d3(&mol).unwrap().hessian;
    let plumb = max_abs(&(&an.cartesian_hessian - &plain.cartesian_hessian - &h_d3));
    let e_d3 = an.energy - plain.energy - an.correction_energy;
    eprintln!(
        "analytic route - (KS + D3) = {plumb:.2e} (max |H_D3| {:.3e}); energy {e_d3:.1e}",
        max_abs(&h_d3)
    );
    assert!(plumb < 1e-12, "{plumb:.3e}");
    assert!(e_d3.abs() < 1e-12, "{e_d3:.3e}");
    assert!(an.correction_energy < 0.0);

    let fd = |delta: f64| {
        run(
            &mol,
            &c,
            &FrequencyConfig {
                hessian: HessianMethod::FiniteDifference,
                delta,
                ..Default::default()
            },
        )
    };
    let (fd1, fd2) = (fd(DELTA), fd(0.5 * DELTA));
    assert_eq!(fd1.hessian_source, HessianSource::FiniteDifference);
    let trunc_h = 4.0 / 3.0 * max_abs(&(&fd1.cartesian_hessian - &fd2.cartesian_hessian));
    let trunc_nu = 4.0 / 3.0 * max_dnu(&fd1.frequencies, &fd2.frequencies);
    let dh = max_abs(&(&an.cartesian_hessian - &fd1.cartesian_hessian));
    let dnu = max_dnu(&an.frequencies, &fd1.frequencies);
    eprintln!(
        "{XC}-D3(BJ)/{BASIS} water: |H_an - H_fd(δ)| {dh:.3e} vs bar {:.3e} (truncation \
         {trunc_h:.3e}); |ν_an - ν_fd(δ)| {dnu:.3e} cm^-1 vs bar {:.3e}; |H_an - H_fd(δ/2)| \
         {:.3e}; asymmetry {:.1e}; ν {:?}; E_D3 {:.6e}",
        2.0 * trunc_h,
        2.0 * trunc_nu,
        max_abs(&(&an.cartesian_hessian - &fd2.cartesian_hessian)),
        fd1.asymmetry,
        an.frequencies,
        an.correction_energy
    );
    assert!(dh < 2.0 * trunc_h, "Hessian {dh:.3e}");
    assert!(dnu < 2.0 * trunc_nu, "frequencies {dnu:.3e}");
    // The energies of both routes are the same corrected energy.
    assert!((an.energy - fd1.energy).abs() < 1e-9);
}

#[test]
fn explicit_moving_grid_refusals_name_the_reference() {
    let h2 = Molecule::parse_xyz("2\nh2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
    let explicit = FrequencyConfig {
        hessian: HessianMethod::AnalyticMovingGrid,
        ..Default::default()
    };
    let cases = [
        (
            "ROKS",
            cfg("LDA"),
            FrequencyReference::Rohf,
            "open-shell (Rohf, i.e. ROKS) reference",
        ),
        (
            "RSH",
            cfg("HSE06"),
            FrequencyReference::Rhf,
            "a range-separated hybrid ('HSE06')",
        ),
    ];
    for (label, c, reference, fragment) in cases {
        let e = harmonic_frequencies_with_geometric_correction(
            &ParallelContext::default(),
            &h2,
            "sto-3g",
            Operator::coulomb(),
            &c,
            &FrequencyConfig {
                reference,
                ..explicit.clone()
            },
            d3,
        )
        .expect_err(label)
        .to_string();
        eprintln!("{label}: {e}");
        assert!(e.contains(fragment), "{label}: {e}");
    }
    // `Auto` with an unsupported reference: finite differences, not an error.
    let auto = harmonic_frequencies_with_geometric_correction(
        &ParallelContext::default(),
        &h2,
        "sto-3g",
        Operator::coulomb(),
        &RhfConfig {
            energy_conv: 1e-11,
            density_conv: 1e-9,
            ..cfg("HSE06")
        },
        &FrequencyConfig::default(),
        d3,
    )
    .unwrap();
    assert_eq!(auto.hessian_source, HessianSource::FiniteDifference);
}
