//! `ScfResult::convergence` reports the signals the exit gate judged: its
//! `iteration` is the returned iteration count and its `de` is |E_n - E_{n-1}|
//! between the returned energy and the energy one iteration earlier.
//!
//! E_{n-1} comes from a second solve of the same system stopped at
//! `max_iter = n - 1`: the trajectory up to that point is the same, so its
//! best-effort energy is the iteration-(n-1) energy. The loose thresholds keep
//! |E_n - E_{n-1}| between 1e-9 and 1e-6 Ha, far above the 1e-12 Ha tolerance, so
//! a ΔE taken against the wrong energy (for instance the previous energy
//! recorded before the gate reads it, which makes ΔE zero) fails the check.
//! RHF and UHF each keep their own iteration loop, so each is checked; ROHF
//! returns no convergence report.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::result::ScfResult;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf_best_effort;
use std::path::PathBuf;

#[derive(Clone, Copy)]
enum Driver {
    Rhf,
    Uhf,
}

fn solve(driver: Driver, rel: &str, mult: usize, max_iter: usize) -> ScfResult {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata/molecules")
        .join(rel);
    let mol = Molecule::load_xyz_with_charge(p.to_str().unwrap(), 0, mult).unwrap();
    let bs = basis::bundled("sto-3g").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let cfg = RhfConfig::default()
        .with_energy_conv(1e-5)
        .with_density_conv(1e-3)
        .with_max_iter(max_iter);
    let ctx = ParallelContext::default();
    match driver {
        Driver::Rhf => solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg),
        Driver::Uhf => solve_uhf_best_effort(&ctx, &mol, &prep, &bounds, &cfg, None),
    }
    .unwrap()
}

fn check(driver: Driver, rel: &str, mult: usize) {
    let full = solve(driver, rel, mult, 100);
    assert!(full.converged, "{rel}: reference solve did not converge");
    let n = full.iterations;
    assert!(
        n >= 3,
        "{rel}: converged in {n} iterations, too few to probe"
    );
    let rep = full
        .convergence
        .as_ref()
        .expect("converged solve carries a report");
    assert_eq!(
        rep.iteration, n,
        "{rel}: report iteration vs returned count"
    );

    let short = solve(driver, rel, mult, n - 1);
    assert!(!short.converged, "{rel}: max_iter = n - 1 still converged");
    assert_eq!(short.iterations, n - 1, "{rel}: truncated iteration count");
    let de = (full.energy - short.energy).abs();
    eprintln!(
        "{rel}: n={n} E_n={:.15} E_n-1={:.15} |dE|={de:.3e} report.de={:.3e}",
        full.energy, short.energy, rep.de
    );
    assert!(
        de > 1e-10,
        "{rel}: |E_n - E_n-1| = {de:.3e} is too small to tell a wrong ΔE apart"
    );
    assert!(
        (rep.de - de).abs() <= 1e-12,
        "{rel}: report.de = {:.6e} but |E_n - E_n-1| = {de:.6e}",
        rep.de
    );
}

#[test]
fn rhf_report_de_is_the_last_energy_step() {
    check(Driver::Rhf, "water.xyz", 1);
}

#[test]
fn uhf_report_de_is_the_last_energy_step() {
    check(Driver::Uhf, "validation/nh2.xyz", 2);
}
