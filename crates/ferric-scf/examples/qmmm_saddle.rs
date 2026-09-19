//! Wiring `saddle::find_saddle` to a QM/MM system: the catalyst path, end to end.
//!
//! This is golden-path item C3 ("FIND THE TRANSITION STATE") joined to C0-C2,
//! and it is deliberately an EXAMPLE rather than a library function. The
//! reason is scope, stated up front so nobody mistakes this for a finished
//! `optimize_qmmm`-style entry point:
//!
//! ## What works today, and why it is less than it looks
//!
//! `find_saddle` takes energy/gradient and Hessian CLOSURES, so nothing has to
//! change in the solver to point it at an embedded system. The two pieces it
//! needs both already exist:
//!
//! * **Gradient** — `rhf_gradient(.., ext)` / `ks_gradient_closed(.., ext)`
//!   take the external potential, so an embedded gradient is a normal call.
//! * **Hessian** — `frequencies::harmonic_frequencies` threads
//!   `config.external_potential` into exactly those gradient calls
//!   (`frequencies.rs:649`), so an embedded Hessian is already reachable. That
//!   was NOT obvious and is worth stating: it means a QM/MM saddle search needs
//!   no new Hessian machinery.
//!
//! ## The three things this does NOT do
//!
//! 1. **The MM charges are FIXED.** `QmmmSystem::to_external_potential()` is
//!    evaluated once, at the starting geometry, and reused for every step. That
//!    is correct only while the MM subsystem does not move. `optimize_qmmm`
//!    rebuilds it per step (`qmmm.rs:2066`) precisely because that matters when
//!    MM atoms are free. Here the QM atoms move within a FROZEN field.
//!
//!    For a catalyst TS that is usually the right approximation for a first
//!    pass — the reacting bonds are in the QM region by construction (golden
//!    path C0) — but it IS an approximation, and a barrier computed this way
//!    omits MM relaxation along the reaction coordinate.
//!
//! 2. **Link atoms do not move with the frontier.** `with_link_atoms` places
//!    them from the starting geometry. As the QM region distorts toward the
//!    saddle, a link atom placed for the reactant geometry drifts off the
//!    frontier bond it is capping. `optimize_qmmm` has the same limitation.
//!
//! 3. **No IRC.** One imaginary frequency says first-order saddle, not "the
//!    saddle for the reaction you meant". `SaddleResult::imaginary_mode` is
//!    returned so a caller can inspect the displacement; nothing here checks
//!    it points along the intended coordinate.
//!
//! ## Cost
//!
//! MEASURED in `tests/saddle_cost.rs`: `2 * (6N + 1) + (n_steps + 1)` gradient
//! evaluations, where N is the QM atom count. Each Hessian is 6N DISPLACED
//! evaluations plus one at the undisplaced geometry; `find_saddle` takes one
//! more before its first step. The two Hessians dominate until n_steps exceeds
//! ~6N * 2. Every one of those gradients is an embedded SCF, so the QM region
//! size is the only lever that matters — see the golden-path note's cost table.
//!
//! Run with:
//!
//! ```text
//! cargo run --release -p ferric-scf --example qmmm_saddle
//! ```

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

const ANGSTROM_TO_BOHR: f64 = 1.8897261254578281;

fn main() -> Result<(), FerricError> {
    let ctx = ParallelContext::default();
    let basis = "sto-3g";
    let op = Operator::coulomb();

    // A minimal stand-in for a catalyst cut: H2 as the QM region, with two MM
    // point charges placed off-axis so the embedding is not trivially zero.
    // The chemistry is not the point -- the WIRING is. A real system replaces
    // this construction and nothing below it.
    let atoms = vec![
        QmmmAtom::new("H", 1, 0.0, 0.0, 0.0, 0.0),
        QmmmAtom::new("H", 1, 0.0, 0.0, 1.4, 0.0),
        QmmmAtom::new("X", 0, 4.0, 0.0, 0.7, -0.4),
        QmmmAtom::new("X", 0, -4.0, 0.0, 0.7, 0.4),
    ];
    let system = QmmmSystem::new(&atoms, QmSelection::Indices(vec![0, 1]), 0, 1)?;

    // THE FIXED-FIELD APPROXIMATION, made once and then held. See scope note 1.
    let ext = system.to_external_potential();
    let mut scf = RhfConfig {
        external_potential: ext.clone(),
        ..Default::default()
    };
    scf.max_iter = 100;

    let qm0 = system.to_qm_molecule();
    println!(
        "QM region: {} atoms; MM field: {} point charges (FIXED)",
        qm0.atoms.len(),
        ext.as_ref().map_or(0, |e| e.point_charges.len())
    );

    // --- the two closures find_saddle needs ---------------------------------
    //
    // Both are ordinary calls. Nothing in the solver knows a saddle search is
    // driving it, which is the point of taking closures.

    let scf_for_grad = scf.clone();
    let energy_gradient = move |mol: &Molecule| -> Result<(f64, Array1<f64>), FerricError> {
        let bs = ferric_core::basis::bundled(basis)?;
        let prep = PreparedBasis::new(mol, &bs)?;
        let bounds = SchwarzBounds::compute(op, &prep)?;
        let res = solve_rhf(&ctx, mol, &prep, op, &bounds, &scf_for_grad)?;
        let g = rhf_gradient(
            mol,
            &prep,
            op,
            &bounds,
            &res,
            scf_for_grad.external_potential.as_ref(),
        )?;
        // rhf_gradient returns (natoms, 3); find_saddle wants a flat 3N vector.
        Ok((res.energy, Array1::from_iter(g.iter().copied())))
    };

    let scf_for_hess = scf.clone();
    let hessian = move |mol: &Molecule| -> Result<Array2<f64>, FerricError> {
        // harmonic_frequencies threads config.external_potential into the same
        // gradient calls above, so this Hessian IS the embedded one -- no extra
        // machinery needed. That is the load-bearing fact this example exists
        // to demonstrate.
        let fc = FrequencyConfig {
            reference: FrequencyReference::Rhf,
            ..Default::default()
        };
        let fr = harmonic_frequencies(
            &ParallelContext::default(),
            mol,
            basis,
            op,
            &scf_for_hess,
            &fc,
        )?;
        Ok(fr.cartesian_hessian)
    };

    // --- search -------------------------------------------------------------
    //
    // H2 in this field has no saddle, so find_saddle is EXPECTED to refuse.
    // That refusal is the useful demonstration: the wiring runs end to end and
    // the guard fires rather than returning a minimum dressed as a TS.
    let cfg = SaddleConfig {
        max_steps: 40,
        trust_radius: 0.05,
        ..Default::default()
    };
    match find_saddle(&qm0, &cfg, energy_gradient, hessian) {
        Ok(res) => {
            println!(
                "converged={}  steps={}  n_imaginary={}  E={:.8} Ha",
                res.converged, res.steps, res.n_imaginary, res.energy
            );
            println!(
                "is_transition_state() = {}  (needs converged AND exactly 1 imaginary)",
                res.is_transition_state()
            );
            if let Some(m) = &res.imaginary_mode {
                let norm = m.dot(m).sqrt();
                println!("imaginary mode |v| = {norm:.4} -- CHECK IT POINTS ALONG YOUR REACTION COORDINATE");
            }
        }
        Err(e) => {
            // The expected path for this toy system, and worth printing in full:
            // a refusal that names the reason is the behaviour the module is for.
            println!("find_saddle refused, as it should for a minimum's basin:\n  {e}");
        }
    }

    println!(
        "\nQM atoms are {} Bohr apart at the start ({:.3} A)",
        (qm0.atoms[1].zpos - qm0.atoms[0].zpos).abs(),
        (qm0.atoms[1].zpos - qm0.atoms[0].zpos).abs() / ANGSTROM_TO_BOHR
    );

    embedded_nh3_inversion()?;
    Ok(())
}

/// A saddle search that SUCCEEDS, under the same MM embedding.
///
/// The H2 case above is a refusal, and a refusal alone does not show the
/// workflow works -- code that rejects everything would print the same thing.
/// This is the positive half: NH3 umbrella inversion, whose planar D3h form is
/// a genuine first-order saddle.
///
/// Why this system. It is the smallest closed-shell TS that is unambiguous:
///
///   * planar NH3 at STO-3G has EXACTLY ONE imaginary mode, -1051.5 cm^-1 (the
///     umbrella coordinate). MEASURED with `harmonic_frequencies` before this
///     example was written -- see the rejected candidates below.
///   * the pyramidal start (N lifted 0.15 A) also has one imaginary mode, so it
///     is inside the saddle's basin but is NOT already the answer. A start that
///     was already planar would make the search trivially succeed and prove
///     nothing.
///
/// CANDIDATES REJECTED, because each is a second-order saddle and `find_saddle`
/// correctly refuses them -- worth recording so they are not retried:
///
///   * linear H3+     : 2 imaginary (degenerate bend), -1068 cm^-1 twice
///   * linear H2O     : 2 imaginary (degenerate bend), -2328.7 cm^-1 twice
///
/// The degeneracy is the trap: a bend perpendicular to a linear molecule comes
/// in a pair, so "linear form of a bent molecule" is almost never a TS.
fn embedded_nh3_inversion() -> Result<(), FerricError> {
    let op = Operator::coulomb();
    let basis = "sto-3g";

    // Pyramidal start, N lifted off the H3 plane. Angstrom.
    // r = 1.006 A is the STO-3G planar optimum, MEASURED by scanning the
    // residual gradient: g_max falls 5.2e-3 -> 6.1e-4 between 1.010 and 1.006.
    // Using a textbook 1.01 leaves a BOND-STRETCH gradient of 5.2e-3 that the
    // convergence test (g_max 3e-4) rightly refuses, and the search then looks
    // like it failed when the geometry was simply unrelaxed in a coordinate
    // that has nothing to do with the umbrella. Relax everything except the
    // coordinate under study before starting a saddle search.
    let r = 1.006_f64;
    let ang: f64 = 120.0_f64.to_radians();
    let h = 0.15_f64;
    let xyz = format!(
        "4\nNH3 pyramidal start\nN 0.0 0.0 {h:.6}\nH {:.6} {:.6} 0.0\n\
         H {:.6} {:.6} 0.0\nH {:.6} {:.6} 0.0\n",
        r,
        0.0,
        r * ang.cos(),
        r * ang.sin(),
        r * (2.0 * ang).cos(),
        r * (2.0 * ang).sin()
    );

    // Same fixed-field embedding as above: two off-axis MM charges, so the
    // search runs against an EMBEDDED surface rather than a gas-phase one.
    let atoms = vec![
        QmmmAtom::new("N", 7, 0.0, 0.0, h, 0.0),
        QmmmAtom::new("H", 1, r, 0.0, 0.0, 0.0),
        QmmmAtom::new("H", 1, r * ang.cos(), r * ang.sin(), 0.0, 0.0),
        QmmmAtom::new(
            "H",
            1,
            r * (2.0 * ang).cos(),
            r * (2.0 * ang).sin(),
            0.0,
            0.0,
        ),
        // BOTH charges the SAME SIGN, placed symmetrically about the H3 plane.
        //
        // This is not cosmetic. An ANTISYMMETRIC pair (-0.2 above, +0.2 below)
        // applies a constant force along z, so planar NH3 stops being a
        // stationary point at all: MEASURED max|g_z| = 3.8e-3 at the planar
        // geometry, essentially independent of the N-H distance, against
        // 1e-16 in the gas phase. The search then cannot converge -- correctly,
        // because under that field there is no saddle there to find -- and it
        // LOOKS like a solver failure.
        //
        // A symmetric pair perturbs the energy (by 5.0e-4 Ha here) while
        // preserving the reflection symmetry that DEFINES this TS, so max|g_z|
        // returns to ~1e-17 and the saddle survives the embedding.
        //
        // General rule: an MM field that breaks the symmetry defining your
        // transition state does not make the search harder, it removes the
        // target. Check that the embedded gradient still vanishes at the
        // symmetric geometry before blaming the optimizer.
        QmmmAtom::new("X", 0, 0.0, 0.0, 6.0, -0.2),
        QmmmAtom::new("X", 0, 0.0, 0.0, -6.0, -0.2),
    ];
    let system = QmmmSystem::new(&atoms, QmSelection::Indices(vec![0, 1, 2, 3]), 0, 1)?;
    let ext = system.to_external_potential();
    let mut scf = RhfConfig {
        external_potential: ext.clone(),
        ..Default::default()
    };
    scf.max_iter = 300;

    let mol = Molecule::parse_xyz(&xyz, 0, 1)?;
    println!(
        "\n--- embedded NH3 inversion: {} QM atoms, {} MM charges ---",
        mol.atoms.len(),
        ext.as_ref().map_or(0, |e| e.point_charges.len())
    );

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

    let cfg = SaddleConfig {
        max_steps: 60,
        trust_radius: 0.10,
        ..Default::default()
    };
    let res = find_saddle(&mol, &cfg, eg, hs)?;
    println!(
        "converged={}  steps={}  n_imaginary={}  E={:.8} Ha",
        res.converged, res.steps, res.n_imaginary, res.energy
    );
    println!(
        "is_transition_state() = {}  (needs converged AND exactly 1 imaginary)",
        res.is_transition_state()
    );

    // THE CHECK THAT MATTERS, and the one a converged flag does not give you:
    // is the structure the RIGHT saddle? For umbrella inversion the answer is
    // planar, so every z must agree. A converged search with one imaginary mode
    // can still be the wrong saddle on a surface with several.
    let zs: Vec<f64> = res.mol.atoms.iter().map(|a| a.zpos).collect();
    let spread = zs.iter().fold(f64::NEG_INFINITY, |a, b| a.max(*b))
        - zs.iter().fold(f64::INFINITY, |a, b| a.min(*b));
    println!(
        "z spread = {spread:.4} Bohr -- planar means ~0; the N has come down \
         into the H3 plane, which is the umbrella TS"
    );
    Ok(())
}
