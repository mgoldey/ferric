//! The RHF virtual-block level shift lands on the virtuals of the occupation
//! MOM KEPT, not on those of the aufbau ordering MOM overrode.
//!
//! The shift adds `s * S C_v C_v^T S` built from the previous iteration's
//! orbitals `C`, with `C_v = C[:, nocc..]`. `c_prev` used to be cloned BEFORE
//! `mom_reorder`, so under MOM it carried the aufbau ordering: when MOM kept an
//! above-aufbau orbital occupied, the shift hit that OCCUPIED orbital and left
//! the true virtual alone.
//!
//! # Fixture
//!
//! CO / 6-31G at 1.13 Angstrom from the core-Hamiltonian guess
//! (`use_sad_guess = false`) with MOM engaged from iteration 1. The hcore
//! guess's first aufbau occupation is poor, MOM keeps it, and the unshifted run
//! converges to an occupied set 5.5 Ha above the ground state: MOM is holding
//! orbitals aufbau would not occupy, which is the situation the bug needs.
//!
//! # What separates the two builds (measured 2026-11-15, same fixture)
//!
//! | build | shift 0.5 | iterations |
//! |---|---|---|
//! | `c_prev` cloned after `mom_reorder` | -112.6672208312 Ha (the ground state) | 20 |
//! | `c_prev` cloned before it (the old code) | -111.3070043356 Ha (1.36 Ha above) | 36 |
//!
//! The shifted path is chaotic in detail (a MOM-held set moves with the shift),
//! so this pins the measured outcome rather than a derived identity; the bars
//! are the converged-energy agreement (1e-7 Ha, density_conv 1e-9) and an
//! iteration ceiling at 1.5x the measured 20. The artifact hypothesis (a
//! shift on the wrong orbitals reproduces the old numbers) is the second row.
use ferric_core::{basis, mol::Molecule, parallel::ParallelContext};
use ferric_integrals::{basis_bridge::PreparedBasis, operator::Operator};
use ferric_scf::{
    rhf::{solve_rhf, RhfConfig},
    screening::SchwarzBounds,
};

const CO: &str = "2\nco\nC 0 0 0\nO 0 0 1.13\n";

#[test]
fn level_shift_acts_on_the_virtuals_of_the_occupation_mom_kept() {
    let mol = Molecule::parse_xyz(CO, 0, 1).unwrap();
    let prep = PreparedBasis::new(&mol, &basis::bundled("6-31g").unwrap()).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    let ctx = ParallelContext::default();
    let op = Operator::coulomb();
    let base = RhfConfig {
        density_conv: 1e-9,
        energy_conv: 1e-11,
        max_iter: 150,
        use_sad_guess: false,
        ..Default::default()
    };
    let solve = |cfg: &RhfConfig| solve_rhf(&ctx, &mol, &prep, op, &bounds, cfg).unwrap();
    let with_mom = |shift: f64| {
        solve(&RhfConfig {
            mom_after_iter: 1,
            level_shift: shift,
            ..base.clone()
        })
    };

    let ground = solve(&base);
    assert!(ground.converged);

    // Precondition: unshifted MOM holds an occupied set far above the ground
    // state (aufbau would not occupy it).
    let held = with_mom(0.0);
    assert!(held.converged);
    println!(
        "ground {:.10}; MOM {:.10} ({} it)",
        ground.energy, held.energy, held.iterations
    );
    assert!(
        held.energy - ground.energy > 5.0,
        "MOM must hold an above-aufbau occupation here: {} vs ground {}",
        held.energy,
        ground.energy
    );

    let shifted = with_mom(0.5);
    println!(
        "MOM + shift 0.5: {:.10} ({} it)",
        shifted.energy, shifted.iterations
    );
    assert!(shifted.converged);
    assert!(
        (shifted.energy - ground.energy).abs() < 1e-7,
        "shift 0.5 under MOM must reach the ground state: {} vs {}",
        shifted.energy,
        ground.energy
    );
    assert!(
        shifted.iterations <= 30,
        "shift 0.5 under MOM took {} iterations (measured 20)",
        shifted.iterations
    );
}
