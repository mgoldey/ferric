//! An EMBEDDED saddle search that SUCCEEDS, and the symmetry condition it needs.
//!
//! `examples/qmmm_saddle.rs` originally showed only a REFUSAL -- H2 in an MM
//! field has no saddle, so `find_saddle` correctly declines. That is worth
//! showing, but a refusal alone does not demonstrate the workflow works: code
//! that rejected everything would print exactly the same thing. This is the
//! positive half, and it is a test rather than an example because the thing it
//! pins is easy to break silently.
//!
//! ## The system
//!
//! NH3 umbrella inversion. The planar D3h form is a genuine first-order saddle:
//! ONE imaginary mode at -1051.5 cm^-1. Smallest unambiguous closed-shell TS
//! that is not already the starting geometry.
//!
//! Two candidates were REJECTED during construction, recorded so they are not
//! retried: linear H3+ and linear H2O are both SECOND-order saddles (2
//! imaginary, -1068 and -2328.7 cm^-1, each doubly degenerate). The bend of a
//! linear molecule comes in a perpendicular pair, so "the linear form of a bent
//! molecule" is almost never a transition state.
//!
//! ## THE FINDING this test exists to protect
//!
//! An MM field that BREAKS THE SYMMETRY DEFINING THE SADDLE does not make the
//! search harder -- it removes the target.
//!
//! With an ANTISYMMETRIC charge pair (-0.2 above the plane, +0.2 below) there
//! is a constant force along z, and planar NH3 stops being a stationary point:
//! MEASURED max|g_z| = 3.8e-3 at the planar geometry, essentially independent
//! of the N-H distance, against ~1e-16 in the gas phase. `find_saddle` then
//! runs to max_steps and reports `converged = false` -- correctly, because
//! under that field there is nothing there to find. It reads as a solver bug.
//!
//! With a SYMMETRIC pair (both -0.2) the embedding still shifts the energy, by
//! 5.0e-4 Ha, but preserves the reflection symmetry, max|g_z| returns to ~1e-17
//! and the saddle survives.
//!
//! Both halves are asserted below. The negative half is the one that would
//! otherwise be rediscovered by debugging the optimizer.

use ferric_core::error::FerricError;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::frequencies::{harmonic_frequencies, FrequencyConfig, FrequencyReference};
use ferric_scf::gradient::rhf_gradient;
use ferric_scf::qmmm::{QmSelection, QmmmAtom, QmmmSystem};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::saddle::{find_saddle, SaddleConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::{Array1, Array2};

/// STO-3G planar optimum, MEASURED by scanning the residual gradient: g_max
/// falls 5.2e-3 -> 6.1e-4 between 1.010 and 1.006 A. A textbook 1.01 leaves a
/// BOND-STRETCH gradient the convergence test rightly refuses, which looks like
/// a failed search in a coordinate unrelated to the umbrella.
const R_NH: f64 = 1.006;
/// N lifted off the H3 plane: inside the saddle's basin, but not already there.
const H_LIFT: f64 = 0.15;

fn pyramidal_nh3() -> Result<Molecule, FerricError> {
    let a: f64 = 120.0_f64.to_radians();
    Molecule::parse_xyz(
        &format!(
            "4\nNH3\nN 0.0 0.0 {H_LIFT:.6}\nH {:.6} 0.0 0.0\nH {:.6} {:.6} 0.0\n\
             H {:.6} {:.6} 0.0\n",
            R_NH,
            R_NH * a.cos(),
            R_NH * a.sin(),
            R_NH * (2.0 * a).cos(),
            R_NH * (2.0 * a).sin()
        ),
        0,
        1,
    )
}

/// `q_below` is the knob: equal to `q_above` keeps the reflection symmetry,
/// opposite destroys it.
fn embedded_search(
    q_above: f64,
    q_below: f64,
) -> Result<ferric_scf::saddle::SaddleResult, FerricError> {
    let op = Operator::coulomb();
    let basis = "sto-3g";
    let a: f64 = 120.0_f64.to_radians();
    let atoms = vec![
        QmmmAtom::new("N", 7, 0.0, 0.0, H_LIFT, 0.0),
        QmmmAtom::new("H", 1, R_NH, 0.0, 0.0, 0.0),
        QmmmAtom::new("H", 1, R_NH * a.cos(), R_NH * a.sin(), 0.0, 0.0),
        QmmmAtom::new(
            "H",
            1,
            R_NH * (2.0 * a).cos(),
            R_NH * (2.0 * a).sin(),
            0.0,
            0.0,
        ),
        QmmmAtom::new("X", 0, 0.0, 0.0, 6.0, q_above),
        QmmmAtom::new("X", 0, 0.0, 0.0, -6.0, q_below),
    ];
    let system = QmmmSystem::new(&atoms, QmSelection::Indices(vec![0, 1, 2, 3]), 0, 1)?;
    let mut scf = RhfConfig {
        external_potential: system.to_external_potential(),
        ..Default::default()
    };
    scf.max_iter = 300;

    let s1 = scf.clone();
    let eg = move |m: &Molecule| -> Result<(f64, Array1<f64>), FerricError> {
        let bs = ferric_core::basis::bundled(basis)?;
        let prep = PreparedBasis::new(m, &bs)?;
        let b = SchwarzBounds::compute(op, &prep)?;
        let res = solve_rhf(&ParallelContext::default(), m, &prep, op, &b, &s1)?;
        let g = rhf_gradient(m, &prep, op, &b, &res, s1.external_potential.as_ref())?;
        Ok((res.energy, Array1::from_iter(g.iter().copied())))
    };
    let s2 = scf.clone();
    let hs = move |m: &Molecule| -> Result<Array2<f64>, FerricError> {
        let fc = FrequencyConfig {
            reference: FrequencyReference::Rhf,
            ..Default::default()
        };
        Ok(
            harmonic_frequencies(&ParallelContext::default(), m, basis, op, &s2, &fc)?
                .cartesian_hessian,
        )
    };

    find_saddle(
        &pyramidal_nh3()?,
        &SaddleConfig {
            max_steps: 60,
            trust_radius: 0.10,
            ..Default::default()
        },
        eg,
        hs,
    )
}

#[test]
fn a_symmetry_preserving_mm_field_still_admits_the_saddle() {
    let res = embedded_search(-0.2, -0.2).expect("the pyramidal start has a negative mode");

    assert!(
        res.converged,
        "embedded search did not converge in 60 steps"
    );
    assert_eq!(
        res.n_imaginary, 1,
        "a transition state has EXACTLY one imaginary mode, got {}",
        res.n_imaginary
    );
    assert!(res.is_transition_state());

    // The check a `converged` flag does NOT give you: is it the RIGHT saddle?
    // Umbrella inversion means planar, so every z must agree. A converged
    // search with one imaginary mode can still be a different saddle.
    let zs: Vec<f64> = res.mol.atoms.iter().map(|a| a.zpos).collect();
    let spread = zs.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        - zs.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(
        spread < 0.05,
        "z spread {spread:.4} Bohr -- the structure is not planar, so this is \
         not the umbrella TS even though the search converged"
    );
}

#[test]
fn a_symmetry_breaking_mm_field_removes_the_saddle_entirely() {
    // THE NEGATIVE HALF. Opposite signs put a constant force along z, so the
    // planar geometry is no longer stationary and there is no saddle to reach.
    // Asserting this keeps the test above honest: without it, a `find_saddle`
    // that ignored `external_potential` altogether would pass the positive
    // case, because the GAS-PHASE saddle is right there.
    let res = embedded_search(-0.2, 0.2).expect("the start still has a negative mode");
    assert!(
        !res.is_transition_state(),
        "an antisymmetric MM field applies a constant z force (MEASURED \
         max|g_z| = 3.8e-3 at the planar geometry vs ~1e-16 in gas phase), so \
         there is no stationary point to converge on. Reporting a transition \
         state here means the embedding is being ignored."
    );
}
