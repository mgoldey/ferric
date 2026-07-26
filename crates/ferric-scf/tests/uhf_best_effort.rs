//! `solve_uhf_best_effort` must preserve the converged-so-far state on failure.
//!
//! Historically `solve_uhf` discarded everything and returned a bare Err when it hit
//! max_iter. That made an open-shell convergence LADDER impossible: a ladder works by
//! carrying a failed rung's density forward as the next rung's guess, which is exactly
//! what `solve_rhf_ladder` does via RHF's `build_nonconverged`.
//!
//! These tests pin both halves of the split: the best-effort variant returns usable
//! state, and `solve_uhf` keeps its historical hard-error contract so existing callers
//! (which all unwrap or `?`) do not silently start accepting garbage.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::RhfConfig;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::{solve_uhf, solve_uhf_best_effort};
use ferric_scf::result::ScfExit;

/// An OH radical with a 1-iteration cap: guaranteed not to converge.
fn setup() -> (Molecule, PreparedBasis, SchwarzBounds) {
    let mol = Molecule::parse_xyz("2\n\nO 0.0 0.0 0.0\nH 0.0 0.0 0.97\n", 0, 2).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("sto-3g").unwrap()).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &obs).unwrap();
    (mol, obs, bounds)
}

#[test]
fn best_effort_returns_usable_state_when_it_fails() {
    let (mol, obs, bounds) = setup();
    let cfg = RhfConfig { max_iter: 1, density_conv: 1e-14, ..Default::default() };

    let r = solve_uhf_best_effort(&ParallelContext::default(), &mol, &obs, &bounds, &cfg, None)
        .expect("best-effort must return Ok even when the SCF does not converge");

    assert!(!r.converged, "test premise: this SCF must NOT have converged");
    assert_eq!(r.exit, ScfExit::MaxIter);

    // The point of the exercise: the density and MOs must be REAL, not zeros. A ladder
    // restart consumes exactly these.
    let n = obs.nbasis();
    assert_eq!(r.density_total.dim(), (n, n));
    assert_eq!(r.mos_alpha.dim(), (n, n));
    assert!(r.density_beta.is_some(), "UHF result must carry a beta density");
    assert!(r.mos_beta.is_some(), "UHF result must carry beta MOs");

    let trace: f64 = (0..n).map(|i| r.density_total[[i, i]]).sum();
    assert!(
        trace > 1.0,
        "density trace {trace:.6} is not a real density -- the best-effort state is empty"
    );
    assert!(r.density_total.iter().all(|v| v.is_finite()), "density has non-finite entries");
    assert!(r.mos_alpha.iter().all(|v| v.is_finite()), "MOs have non-finite entries");
    assert!(r.energy.is_finite(), "energy is not finite");
}

/// `solve_uhf` must still hard-error, so existing callers keep failing loudly.
#[test]
fn solve_uhf_keeps_its_error_contract() {
    let (mol, obs, bounds) = setup();
    let cfg = RhfConfig { max_iter: 1, density_conv: 1e-14, ..Default::default() };

    let r = solve_uhf(&ParallelContext::default(), &mol, &obs, &bounds, &cfg);
    assert!(
        r.is_err(),
        "solve_uhf must keep returning Err on non-convergence -- every existing caller \
         unwraps or ?s it, so returning Ok(converged:false) would silently feed them an \
         unconverged reference"
    );
}

/// When it DOES converge, both entry points agree and report success.
#[test]
fn converged_results_agree_between_the_two_entry_points() {
    let (mol, obs, bounds) = setup();
    let cfg = RhfConfig { max_iter: 200, density_conv: 1e-8, ..Default::default() };

    let a = solve_uhf(&ParallelContext::default(), &mol, &obs, &bounds, &cfg).unwrap();
    let b = solve_uhf_best_effort(&ParallelContext::default(), &mol, &obs, &bounds, &cfg, None)
        .unwrap();

    assert!(a.converged && b.converged, "OH/STO-3G UHF should converge");
    assert!(
        (a.energy - b.energy).abs() < 1e-10,
        "the two entry points disagree on a converged result: {:.12} vs {:.12}",
        a.energy,
        b.energy
    );
}
