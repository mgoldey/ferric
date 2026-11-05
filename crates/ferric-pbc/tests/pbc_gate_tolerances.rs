//! Pins for the numeric refusal gates of the open-shell force and stress
//! paths that no SCF-level test reaches. Pure-function inputs (S = 1, C = 1):
//! each pair of cases brackets the gate's tolerance, so loosening or removing
//! the gate flips an assertion. The tolerances are tabulated in
//! `crates/ferric-pbc/theory.md` ("Refusal gates and tolerances").

use ferric_pbc::grad::rohf_orbital_gradient;
use ndarray::Array2;

fn diag(v: &[f64]) -> Array2<f64> {
    let n = v.len();
    Array2::from_shape_fn((n, n), |(i, j)| if i == j { v[i] } else { 0.0 })
}

fn gradient(
    c: &Array2<f64>,
    da: &[f64],
    db: &[f64],
) -> Result<ferric_pbc::grad::RohfOrbitalGradient, ferric_core::FerricError> {
    let n = da.len();
    let z = Array2::<f64>::zeros((n, n));
    rohf_orbital_gradient(c, &Array2::eye(n), &diag(da), &diag(db), &z, &z)
}

/// MO occupations must be 0/1 to `RO_OCC_TOL = 1e-6` (grad.rs).
#[test]
fn rohf_occupation_tolerance_is_one_micro() {
    let c = Array2::<f64>::eye(3);
    gradient(&c, &[1.0, 1.0 - 5e-7, 0.0], &[1.0, 0.0, 0.0]).expect("5e-7 is inside the gate");
    let e = gradient(&c, &[1.0, 1.0 - 2e-6, 0.0], &[1.0, 0.0, 0.0])
        .expect_err("2e-6 is outside the gate")
        .to_string();
    assert!(e.contains("not 0/1"), "{e}");
}

/// A beta-occupied, alpha-empty MO is not a ROHF state (structural, no tolerance).
#[test]
fn rohf_beta_only_mo_is_refused() {
    let c = Array2::<f64>::eye(3);
    let e = gradient(&c, &[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0])
        .expect_err("beta-only MO")
        .to_string();
    assert!(e.contains("β-occupied but α-empty"), "{e}");
}

/// The class counts must carry `tr(D_σ S)` to 1e-6 (grad.rs): the MO set
/// spans two of three AOs, so a density with weight on the third is not that
/// set's. Each MO is still exactly 0/1, so only the count gate can fire.
#[test]
fn rohf_electron_count_tolerance_is_one_micro() {
    let c = Array2::from_shape_fn((3, 2), |(i, j)| if i == j { 1.0 } else { 0.0 });
    gradient(&c, &[1.0, 1.0, 1e-7], &[1.0, 0.0, 0.0]).expect("1e-7 of tr(D S) is inside the gate");
    let e = gradient(&c, &[1.0, 1.0, 3e-6], &[1.0, 0.0, 0.0])
        .expect_err("3e-6 of tr(D S) is outside the gate")
        .to_string();
    assert!(e.contains("do not carry"), "{e}");
}
