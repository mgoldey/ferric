//! `w_pdep::redress_eigenpotentials` inverts the PDEP export exactly.
//!
//! `PdepRpaResult::eigenpotentials` holds physical aux coefficients
//! c = F_pdepᵀ·u (u = the eigensolver's dressed eigenvectors, F_pdep = the
//! RPA intermediates' `v_inv_sqrt`). GW/BSE re-dress them into `mo_b`'s basis
//! (B̃ = F_gw·(Q|mn), F_gw = Cholesky L⁻¹) with d = F_gw⁻ᵀ·c. These tests pin:
//!
//! 1. Coulomb (both factors are the Cholesky L⁻¹ of the same metric):
//!    redress returns `dressed_eigenvectors` to machine precision, with unit
//!    column norms, AND equals what the pre-fix pipeline fed GW
//!    (c_old = F_pdep·u, d_old = F_gw⁻¹·c_old) — so GW/BSE consume the same
//!    dressed vectors as before, up to roundoff.
//! 2. erf (F_pdep is the SYMMETRIC eigh V^{-1/2}): redress with that same
//!    symmetric factor returns u (the symmetric/pseudo-inverse branch).

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_gw::{mo_b, w_pdep};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_integrals::threeindex::coulomb_metric_2c;
use ferric_mp2::rimp2::metric_inverse_sqrt;
use ferric_rpa::{run_pdep_rpa, PdepRpaConfig};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;
use ndarray_linalg::{Diag, SolveTriangular, UPLO};

fn max_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    assert_eq!(a.dim(), b.dim());
    a.iter()
        .zip(b.iter())
        .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()))
}

fn setup(
    obs_name: &str,
) -> (
    Molecule,
    PreparedBasis,
    PreparedBasis,
    ferric_scf::ScfResult,
) {
    let ctx = ParallelContext::default();
    let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled(obs_name).unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
    (mol, obs, dfbs, rhf)
}

fn cfg() -> PdepRpaConfig {
    PdepRpaConfig {
        frozen_core: 0,
        eigensolver_conv_thresh: 1e-10,
        need_eigenvalues_freq: false,
        ..Default::default()
    }
}

#[test]
fn coulomb_redress_recovers_dressed_eigenvectors_and_matches_prefix_pipeline() {
    let (mol, obs, dfbs, rhf) = setup("cc-pvdz");
    let op = Operator::coulomb();
    let pdep = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &cfg()).unwrap();
    let u = &pdep.dressed_eigenvectors;

    let mob = mo_b::build_full_b(&mol, &obs, &dfbs, op, &rhf, 0, None).unwrap();
    let (d, dev) = w_pdep::redress_with_check(&mob.v_inv_sqrt, &pdep.eigenpotentials).unwrap();
    let err_u = max_abs_diff(&d, u);

    // Pre-fix pipeline: c_old = F_pdep·u, d_old = F_gw⁻¹·c_old (lower solve).
    let f_pdep = metric_inverse_sqrt(&coulomb_metric_2c(op, &dfbs).unwrap(), op).unwrap();
    let c_old = f_pdep.dot(u);
    let d_old = mob
        .v_inv_sqrt
        .solve_triangular(UPLO::Lower, Diag::NonUnit, &c_old)
        .unwrap();
    let err_old = max_abs_diff(&d, &d_old);
    eprintln!(
        "Coulomb: M={} max|‖d‖²-1|={dev:.3e} max|d-u|={err_u:.3e} max|d-d_prefix|={err_old:.3e}",
        u.ncols()
    );
    assert!(
        dev <= 1e-10,
        "redressed column norms deviate from 1 by {dev:.3e}"
    );
    assert!(
        err_u <= 1e-10,
        "redress does not recover the dressed eigenvectors: {err_u:.3e}"
    );
    assert!(
        err_old <= 1e-10,
        "GW/BSE dressed vectors changed vs the pre-fix pipeline by {err_old:.3e}"
    );
}

#[test]
fn erf_symmetric_factor_redress_recovers_dressed_eigenvectors() {
    let (mol, obs, dfbs, rhf) = setup("sto-3g");
    let op = Operator::erf(0.2);
    let pdep = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &cfg()).unwrap();
    // The same (deterministic) symmetric factor the PDEP intermediates used.
    let f = metric_inverse_sqrt(&coulomb_metric_2c(op, &dfbs).unwrap(), op).unwrap();
    let scale = f.iter().fold(0.0_f64, |m, &x| m.max(x.abs()));
    let asym = max_abs_diff(&f, &f.t().to_owned());
    assert!(
        asym <= 1e-12 * scale,
        "erf factor expected symmetric, asym {asym:.3e} (scale {scale:.3e})"
    );
    let upper = (0..f.nrows())
        .flat_map(|i| ((i + 1)..f.ncols()).map(move |j| (i, j)))
        .fold(0.0_f64, |m, (i, j)| m.max(f[(i, j)].abs()));
    assert!(upper > 1e-6 * scale, "erf factor unexpectedly triangular");
    let (d, dev) = w_pdep::redress_with_check(&f, &pdep.eigenpotentials).unwrap();
    let err_u = max_abs_diff(&d, &pdep.dressed_eigenvectors);
    eprintln!(
        "erf(0.2): M={} max|‖d‖²-1|={dev:.3e} max|d-u|={err_u:.3e}",
        d.ncols()
    );
    assert!(
        dev <= 1e-8,
        "redressed column norms deviate from 1 by {dev:.3e}"
    );
    assert!(
        err_u <= 1e-8,
        "redress does not recover the dressed eigenvectors: {err_u:.3e}"
    );
}
