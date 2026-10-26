//! The orbital-gradient veto must not refuse a converged state whose AO
//! commutator carries a residual in lindep-dropped overlap modes.
//!
//! With near-null overlap modes dropped by the canonical orthogonalizer, the AO
//! commutator FDS − SDF keeps a component in the dropped subspace that no
//! rotation of the kept orbitals can remove, so its max-norm parks above zero
//! at the converged state. MEASURED here (water/aug-cc-pVTZ,
//! `FERRIC_LINDEP_THRESH=1e-3`, `density_conv` 1e-8): AO `err_max` parks at
//! 2.54e-4 (RHF) and 1.445e-4 (UHF cation), both above `sqrt(density_conv)` =
//! 1e-4. A veto read off the AO norm against `sqrt(density_conv)` (the first
//! version of the guard) refused these runs until `max_iter`; origin/main
//! converges them in 14 / 13 iterations. The veto reads the orthonormal-basis
//! gradient Xᵀ(FDS−SDF)X instead, with the bound `max(sqrt(dconv), 1e-3)`.
//!
//! ONE test in its own binary: the threshold is a process-wide env knob
//! (`FERRIC_LINDEP_THRESH`, no config field), set once before any solve, so
//! no concurrent test in this process can observe a different value.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;

const WATER: &str = "3\nwater\nO 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692\n";

fn setup(charge: i32, mult: usize) -> (Molecule, PreparedBasis, SchwarzBounds) {
    let mol = Molecule::parse_xyz(WATER, charge, mult).unwrap();
    let bs = basis::bundled("aug-cc-pvtz").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    (mol, prep, bounds)
}

#[test]
fn dropped_lindep_modes_do_not_trip_the_gradient_veto() {
    std::env::set_var("FERRIC_LINDEP_THRESH", "1e-3");
    let ctx = ParallelContext::default();
    let cfg = RhfConfig {
        density_conv: 1e-8,
        energy_conv: 1e-10,
        max_iter: 100,
        ..Default::default()
    };

    let (mol, prep, bounds) = setup(0, 1);
    let r = solve_rhf(&ctx, &mol, &prep, Operator::coulomb(), &bounds, &cfg).unwrap();
    let rep = r.convergence.unwrap();
    println!(
        "RHF water/aTZ lindep 1e-3: E {:.10} conv={} iters={} AO err_max={:.3e}",
        r.energy, r.converged, r.iterations, rep.err_max
    );
    // Not vacuous: the AO residual is above sqrt(density_conv).
    assert!(rep.err_max > cfg.density_conv.sqrt(), "no AO floor here");
    assert!(
        r.converged && rep.gradient_vetoes.is_none(),
        "{}",
        rep.describe()
    );
    assert_eq!(r.iterations, 14);

    let (mol, prep, bounds) = setup(1, 2);
    let r = solve_uhf(&ctx, &mol, &prep, &bounds, &cfg).expect("UHF must converge");
    let rep = r.convergence.unwrap();
    println!(
        "UHF water+/aTZ lindep 1e-3: E {:.10} iters={} AO err_max={:.3e}",
        r.energy, r.iterations, rep.err_max
    );
    assert!(rep.err_max > cfg.density_conv.sqrt(), "no AO floor here");
    assert!(rep.gradient_vetoes.is_none());
    assert_eq!(r.iterations, 13);
}
