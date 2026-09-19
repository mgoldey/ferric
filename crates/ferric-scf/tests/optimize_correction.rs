//! `optimize_geometry_with_correction`: byte-identity when unused, and a hard
//! refusal when a correction reports an energy it cannot differentiate.
//!
//! The hook exists so `ferric-cli` can add D3(BJ) dispersion to a geometry
//! optimization without `ferric-scf` depending on an empirical dispersion
//! model. Two properties make it safe to have added:
//!
//! 1. **Byte-identity when unused.** `optimize_geometry` now routes through
//!    this function, so if that changed any result -- even in the last bit --
//!    it would silently perturb every existing optimization. The
//!    `de != 0.0 || dg.is_some()` guard skips the addition rather than adding
//!    0.0, so no floating-point operation enters the pre-existing path.
//!
//! 2. **An energy without a gradient is refused.** Such a correction makes the
//!    optimizer walk one surface while reporting another, converging to a
//!    geometry that is a stationary point of NEITHER -- exactly what the D3
//!    `with_gradient` rejection used to prevent, so it must not reappear one
//!    level down.

use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::operator::Operator;
use ferric_scf::optimize::{optimize_geometry, optimize_geometry_with_correction, OptimizeConfig};
use ferric_scf::rhf::RhfConfig;
use ndarray::Array2;

/// H2 at 0.74 A.
///
/// The xyz format is ANGSTROM, which is worth stating in code: writing `1.4`
/// here intending Bohr starts the molecule at 2.65 Bohr, and the resulting
/// outward relaxation looks exactly like a runaway optimizer. That misreading
/// cost real time while this hook was being added.
fn h2() -> Molecule {
    Molecule::parse_xyz("2\nH2\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\n", 0, 1).expect("parse")
}

fn cfg() -> OptimizeConfig {
    OptimizeConfig {
        max_steps: 40,
        ..Default::default()
    }
}

#[test]
fn a_zero_correction_is_byte_identical_to_no_correction() {
    let ctx = ParallelContext::default();
    let mol = h2();
    let op = Operator::coulomb();
    let scf = RhfConfig::default();

    let plain = optimize_geometry(&ctx, &mol, "sto-3g", op, &scf, &cfg()).expect("plain");
    let hooked = optimize_geometry_with_correction(&ctx, &mol, "sto-3g", op, &scf, &cfg(), |_| {
        Ok((0.0, None))
    })
    .expect("hooked");

    // BIT-for-bit via to_bits(), not a tolerance: the docstring claims
    // identity, and a tolerance would let that quietly become "close".
    assert_eq!(
        plain.energy.to_bits(),
        hooked.energy.to_bits(),
        "a zero correction changed the energy: {} vs {}",
        plain.energy,
        hooked.energy
    );
    assert_eq!(plain.steps, hooked.steps, "step count changed");
    for (a, b) in plain.mol.atoms.iter().zip(&hooked.mol.atoms) {
        assert_eq!(a.x.to_bits(), b.x.to_bits());
        assert_eq!(a.y.to_bits(), b.y.to_bits());
        assert_eq!(a.zpos.to_bits(), b.zpos.to_bits());
    }
}

#[test]
fn an_energy_correction_without_a_gradient_is_refused() {
    let ctx = ParallelContext::default();
    let err = optimize_geometry_with_correction(
        &ctx,
        &h2(),
        "sto-3g",
        Operator::coulomb(),
        &RhfConfig::default(),
        &cfg(),
        |_| Ok((-0.001, None)),
    )
    .expect_err("an energy without a gradient must be refused");
    assert!(
        err.to_string().contains("gradient"),
        "the error must name the missing piece, got: {err}"
    );
}

#[test]
fn a_mis_shaped_correction_gradient_is_refused() {
    let ctx = ParallelContext::default();
    let err = optimize_geometry_with_correction(
        &ctx,
        &h2(),
        "sto-3g",
        Operator::coulomb(),
        &RhfConfig::default(),
        &cfg(),
        // 3 rows for a 2-atom molecule: broadcasting or truncating would apply
        // forces to the wrong nuclei.
        |_| Ok((-0.001, Some(Array2::<f64>::zeros((3, 3))))),
    )
    .expect_err("a mis-shaped gradient must be refused");
    assert!(err.to_string().contains("gradient"));
}

/// A CONSTANT correction shifts the energy but not the geometry.
///
/// The cheapest end-to-end check that the correction reaches BOTH halves: with
/// only the energy wired the geometry would still match, and with only the
/// gradient wired the energy would not shift. Requiring both pins the pair.
#[test]
fn a_constant_correction_shifts_the_energy_and_not_the_geometry() {
    let ctx = ParallelContext::default();
    let mol = h2();
    let op = Operator::coulomb();
    let scf = RhfConfig::default();
    const SHIFT: f64 = -0.25;

    let plain = optimize_geometry(&ctx, &mol, "sto-3g", op, &scf, &cfg()).expect("plain");
    let shifted = optimize_geometry_with_correction(&ctx, &mol, "sto-3g", op, &scf, &cfg(), |m| {
        Ok((SHIFT, Some(Array2::<f64>::zeros((m.atoms.len(), 3)))))
    })
    .expect("shifted");

    assert!(
        (shifted.energy - (plain.energy + SHIFT)).abs() < 1e-12,
        "energy should be offset by exactly {SHIFT}: got {} vs {}",
        shifted.energy,
        plain.energy + SHIFT
    );
    let bond = |r: &ferric_scf::optimize::OptimizeResult| {
        (r.mol.atoms[1].zpos - r.mol.atoms[0].zpos).abs()
    };
    assert!(
        (bond(&plain) - bond(&shifted)).abs() < 1e-9,
        "a zero-gradient correction must not move the geometry: {} vs {}",
        bond(&plain),
        bond(&shifted)
    );
}
