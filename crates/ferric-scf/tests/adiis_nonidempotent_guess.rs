//! ADIIS/EDIIS must not trust a non-idempotent starting density (#411).
//!
//! The energy-based DIIS variants minimize a functional built from the
//! energies and Fock matrices of the stored densities. That is only meaningful
//! for genuine idempotent densities (`D S D = 2 D`, `E(D) >= E_gs`). A MINAO or
//! user-supplied guess is NOT idempotent and can carry an "energy" far BELOW
//! the ground state; the functional then puts all its weight on that entry
//! (`c = [1, ~1e-17]`), the extrapolated Fock is the guess's Fock again, the
//! next density repeats the previous one exactly, and the `dp_rms` gate
//! reports `converged = true` at iteration 3 on a wrong energy (smeltery's
//! 71-atom pose in a 6458-charge field: -1934.99 Ha against -1936.03 Ha).
//!
//! The cheap reproducer is the converged water density scaled by 1.25: a
//! 12.5-electron non-idempotent density whose energy lies ~30 Ha below the
//! ground state (E(lambda D) = lambda A + lambda^2 B with A < 0 < B).
//!
//! Exactness anchor: the ADIIS/EDIIS energy equals the plain-DIIS energy from
//! the SAME guess to the density-conv level.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::diis::DiisFlavor;
use ferric_scf::result::ScfResult;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

struct Water {
    mol: Molecule,
    prep: PreparedBasis,
    bounds: SchwarzBounds,
    d_conv: ndarray::Array2<f64>,
    e_conv: f64,
}

fn water() -> Water {
    let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
    let bs = basis::bundled("sto-3g").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();
    let cfg = RhfConfig {
        energy_conv: 1e-12,
        density_conv: 1e-10,
        max_iter: 100,
        ..Default::default()
    };
    let r = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
    assert!(r.converged);
    Water {
        mol,
        prep,
        bounds,
        d_conv: r.density_r().clone(),
        e_conv: r.energy,
    }
}

fn run(w: &Water, flavor: DiisFlavor, guess: Option<&ndarray::Array2<f64>>) -> ScfResult {
    let cfg = RhfConfig {
        energy_conv: 1e-10,
        density_conv: 1e-8,
        max_iter: 100,
        diis_flavor: flavor,
        diis_switch_thresh: 1e-1,
        init_guess_density: guess.cloned(),
        ..Default::default()
    };
    solve_rhf(
        &ParallelContext::default(),
        &w.mol,
        &w.prep,
        Operator::coulomb(),
        &w.bounds,
        &cfg,
    )
    .unwrap()
}

/// The reproducer really has the defect it claims: the guess energy is far
/// below the ground state (otherwise the test below would assert nothing).
#[test]
fn the_scaled_guess_energy_is_below_the_ground_state() {
    let w = water();
    let d = &w.d_conv * 1.25;
    // One SCF iteration reports the energy of the guess density.
    let cfg = RhfConfig {
        max_iter: 1,
        init_guess_density: Some(d),
        ..Default::default()
    };
    let r = solve_rhf(
        &ParallelContext::default(),
        &w.mol,
        &w.prep,
        Operator::coulomb(),
        &w.bounds,
        &cfg,
    )
    .unwrap();
    assert!(
        r.energy < w.e_conv - 1.0,
        "guess energy {} must lie well below E_gs {}",
        r.energy,
        w.e_conv
    );
}

#[test]
fn adiis_and_ediis_do_not_falsely_converge_from_a_non_idempotent_guess() {
    let w = water();
    let guess = &w.d_conv * 1.25;
    let plain = run(&w, DiisFlavor::Pulay, Some(&guess));
    assert!(plain.converged);
    assert!((plain.energy - w.e_conv).abs() < 1e-7);

    for flavor in [DiisFlavor::Adiis, DiisFlavor::Ediis] {
        let r = run(&w, flavor, Some(&guess));
        assert!(
            r.iterations > 3,
            "{flavor:?} stopped at iteration {} (the repeated-density false convergence)",
            r.iterations
        );
        assert!(r.converged, "{flavor:?} did not converge");
        assert!(
            (r.energy - plain.energy).abs() < 1e-7,
            "{flavor:?} E={} vs plain DIIS {}",
            r.energy,
            plain.energy
        );
    }
}

/// Default-guess anchor: from the ordinary MINAO guess ADIIS/EDIIS still agree
/// with plain DIIS (the vetting must not disturb cases that worked).
#[test]
fn adiis_and_ediis_match_plain_diis_from_the_default_guess() {
    let w = water();
    let plain = run(&w, DiisFlavor::Pulay, None);
    for flavor in [DiisFlavor::Adiis, DiisFlavor::Ediis] {
        let r = run(&w, flavor, None);
        assert!(r.converged);
        assert!((r.energy - plain.energy).abs() < 1e-7);
    }
}
