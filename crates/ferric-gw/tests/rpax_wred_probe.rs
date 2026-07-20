//! Investigation (task #68): probe the PDEP static dielectric eigenvalues
//! lambda_alpha(0) and the derived reduced screening weight w_red =
//! 1/lambda - 1 that bse.rs's `screened()` closures use to mix W into
//! (A+B)/(A-B). Suspicion: if any lambda_alpha(0) is negative or very close
//! to zero, w_red blows up or flips sign, which would explain a
//! screening-only (A-B) instability that the bare-exchange kernel doesn't
//! show (see rpax_bare_ab_check.rs, which proves the bare kernel is fine).
//!
//! Run: cargo test -p ferric-gw --release --test rpax_wred_probe -- --ignored --nocapture

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::config::{
    Chi0Backend, Chi0Sparsity, Eigensolver, PdepRpaConfig, QuadratureConfig, QuadratureScheme,
    SternheimerConfig,
};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

fn pdep_cfg() -> PdepRpaConfig {
    PdepRpaConfig {
        quadrature: QuadratureConfig {
            scheme: QuadratureScheme::GaussLegendre,
            n_points: 16,
            u0: 0.5,
        },
        eigensolver_conv_thresh: 1e-7,
        eigensolver_max_vecs: 0,
        trunc_thresh: 0.0,
        run_diagnostics: false,
        frozen_core: 0,
        chi0_backend: Chi0Backend::Dense,
        chi0_sparsity: Chi0Sparsity::Dense,
        eigensolver: Eigensolver::Davidson,
        sternheimer: SternheimerConfig::default(),
        memory_budget_bytes: None,
        need_inv_dielectric_freq: false,
        verbose: false,
    }
}

#[test]
#[ignore = "fast: PDEP static eigenvalue probe; --release --ignored --nocapture"]
fn wred_probe_water_pbe() {
    let xyz = "3\nH2O\nO 0.0 0.0 0.117790\nH 0.0 0.755453 -0.471161\nH 0.0 -0.755453 -0.471161\n";
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let ctx = ParallelContext::default();
    let ks = solve_rhf(
        &ctx,
        &mol,
        &obs,
        op,
        &bounds,
        &RhfConfig {
            xc: Some("PBE".to_string()),
            ..Default::default()
        },
    )
    .unwrap();

    let pdep = ferric_rpa::run_pdep_rpa(&mol, &obs, &dfbs, op, &ks, &pdep_cfg()).unwrap();
    let lam = &pdep.eigenvalues_static;
    eprintln!("n_eigenpotentials = {}", pdep.n_eigenpotentials);
    eprintln!("lambda_alpha(0) stats: min={:.6} max={:.6}", lam.iter().cloned().fold(f64::MAX, f64::min), lam.iter().cloned().fold(f64::MIN, f64::max));
    let n_neg = lam.iter().filter(|&&l| l < 0.0).count();
    let n_small = lam.iter().filter(|&&l| l.abs() < 1e-3).count();
    eprintln!("count(lambda<0) = {n_neg}   count(|lambda|<1e-3) = {n_small}");
    // print the 10 most extreme (closest to 0, and most negative if any)
    let mut sorted: Vec<f64> = lam.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    eprintln!("10 smallest lambda: {:?}", &sorted[..10.min(sorted.len())]);
    eprintln!("10 largest lambda: {:?}", &sorted[sorted.len().saturating_sub(10)..]);

    // w_red = 1/lambda - 1
    let w_red: Vec<f64> = lam.iter().map(|&l| 1.0 / l - 1.0).collect();
    let mut wsorted = w_red.clone();
    wsorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    eprintln!("w_red: min={:.6} max={:.6}", wsorted[0], wsorted[wsorted.len()-1]);
    eprintln!("10 smallest w_red: {:?}", &wsorted[..10.min(wsorted.len())]);
    eprintln!("10 largest w_red: {:?}", &wsorted[wsorted.len().saturating_sub(10)..]);
}
