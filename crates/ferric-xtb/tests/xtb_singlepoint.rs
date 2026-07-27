//! Validation of the xtb binding.
//!
//! Only compiled/run with `--features xtb`, and libxtb must be on the runtime
//! loader path:
//!
//! ```text
//! LD_LIBRARY_PATH=$HOME/.local/lib/x86_64-linux-gnu \
//!   cargo test -p ferric-xtb --features xtb
//! ```
//!
//! The energy reference is the **xtb CLI binary's own output** on the same
//! geometry, which is an independent check on the binding (different entry
//! point into the library) rather than a self-check.
//!
//! # These tests MUST run serially
//!
//! libxtb has process-global mutable state and is not thread-safe: this suite
//! passes 8/8 serially and fails 3/8 under the default parallel harness (see
//! the threading section of `calculator.rs`). Every test therefore takes a
//! process-wide lock via [`xtb_lock`], so the suite is correct regardless of
//! `--test-threads`.

#![cfg(feature = "xtb")]

use ferric_core::mol::Molecule;
use ferric_xtb::{XtbCalculator, XtbConfig, XtbMethod};
use std::sync::{Mutex, MutexGuard, OnceLock};

/// Serialises every xtb call in this binary. libxtb is not thread-safe (see the
/// module doc), and cargo's harness runs tests on separate threads by default.
fn xtb_lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    // Poisoning is irrelevant here: the lock guards a foreign library, not data
    // whose invariants a panicking test could have broken.
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

/// Water at the geometry in `testdata/molecules/water.xyz` (Angstrom in the
/// file; `parse_xyz` converts to Bohr).
fn water() -> Molecule {
    let xyz = "3
water optimized HF/cc-pVDZ
O   0.000000   0.000000   0.117790
H   0.000000   0.755453  -0.471161
H   0.000000  -0.755453  -0.471161
";
    Molecule::parse_xyz(xyz, 0, 1).expect("parse water")
}

/// GFN2-xTB total energy for water must match the xtb CLI's own printed
/// TOTAL ENERGY on the identical geometry.
///
/// Reference (`xtb --gfn 2 --sp water.xyz`, xtb 6.7.1, this build):
///   TOTAL ENERGY  -5.070325128562 Eh   (gradient norm 0.139602571206 Eh/a0)
///
/// Tolerance 1e-8 Ha: the CLI prints 12 decimals, so any real discrepancy
/// (units, geometry transfer, charge/uhf) would be orders of magnitude larger.
#[test]
fn gfn2_water_energy_matches_xtb_cli() {
    let _guard = xtb_lock();
    const XTB_CLI_GFN2_WATER: f64 = -5.070325128562;

    let mol = water();
    let mut calc = XtbCalculator::new_gfn2(&mol).expect("build GFN2 calculator");
    let out = calc.singlepoint().expect("GFN2 single point");

    let dev = (out.energy - XTB_CLI_GFN2_WATER).abs();
    assert!(
        dev < 1e-8,
        "GFN2 water energy {:.12} Ha deviates from xtb CLI {:.12} Ha by {:.3e}",
        out.energy,
        XTB_CLI_GFN2_WATER,
        dev
    );

    // The CLI also prints the gradient norm, so this is a second independent
    // check -- and specifically one on the gradient's units (Hartree/Bohr).
    const XTB_CLI_GFN2_WATER_GNORM: f64 = 0.139602571206;
    let gnorm = out.gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
    let gdev = (gnorm - XTB_CLI_GFN2_WATER_GNORM).abs();
    assert!(
        gdev < 1e-8,
        "GFN2 water gradient norm {gnorm:.12} deviates from xtb CLI {:.12} Ha/Bohr by {gdev:.3e}",
        XTB_CLI_GFN2_WATER_GNORM
    );
}

/// GFN1-xTB on the same geometry, again against the CLI.
///
/// Reference (`xtb --gfn 1 --sp water.xyz`, xtb 6.7.1, this build):
///   TOTAL ENERGY  -5.768602223514 Eh
#[test]
fn gfn1_water_energy_matches_xtb_cli() {
    let _guard = xtb_lock();
    const XTB_CLI_GFN1_WATER: f64 = -5.768602223514;

    let mol = water();
    let cfg = XtbConfig {
        method: XtbMethod::Gfn1,
        ..Default::default()
    };
    let mut calc = XtbCalculator::new(&mol, cfg).expect("build GFN1 calculator");
    let out = calc.singlepoint().expect("GFN1 single point");

    let dev = (out.energy - XTB_CLI_GFN1_WATER).abs();
    assert!(
        dev < 1e-8,
        "GFN1 water energy {:.12} Ha deviates from xtb CLI {:.12} Ha by {:.3e}",
        out.energy,
        XTB_CLI_GFN1_WATER,
        dev
    );
}

/// GFN-FF (force field, not self-consistent) on the same geometry.
///
/// Reference (`xtb --gfnff --sp water.xyz`, xtb 6.7.1, this build):
///   TOTAL ENERGY  -0.327252506734 Eh
///
/// The energy is on a completely different scale from GFN1/GFN2 (a force field
/// has no electronic energy), which is itself a check that the method selector
/// actually reaches a different Hamiltonian rather than silently falling back.
#[test]
fn gfnff_water_energy_matches_xtb_cli() {
    let _guard = xtb_lock();
    const XTB_CLI_GFNFF_WATER: f64 = -0.327252506734;

    let mol = water();
    let cfg = XtbConfig {
        method: XtbMethod::GfnFf,
        ..Default::default()
    };
    let mut calc = XtbCalculator::new(&mol, cfg).expect("build GFN-FF calculator");
    let out = calc.singlepoint().expect("GFN-FF single point");

    let dev = (out.energy - XTB_CLI_GFNFF_WATER).abs();
    assert!(
        dev < 1e-8,
        "GFN-FF water energy {:.12} Ha deviates from xtb CLI {:.12} Ha by {:.3e}",
        out.energy,
        XTB_CLI_GFNFF_WATER,
        dev
    );
}

/// Charge and multiplicity must reach xtb: the water radical cation
/// (charge +1, doublet -> xtb `--chrg 1 --uhf 1`) against the CLI.
///
/// Reference (`xtb wcat.xyz --gfn 2 --sp --chrg 1 --uhf 1`, xtb 6.7.1):
///   TOTAL ENERGY  -4.396840287753 Eh
///
/// Without this, a binding that silently dropped `charge`/`uhf` would still
/// pass every neutral-singlet test above.
#[test]
fn charge_and_multiplicity_reach_xtb() {
    let _guard = xtb_lock();
    const XTB_CLI_WATER_CATION: f64 = -4.396840287753;

    let xyz = "3
water cation
O   0.000000   0.000000   0.117790
H   0.000000   0.755453  -0.471161
H   0.000000  -0.755453  -0.471161
";
    // charge +1, multiplicity 2 (doublet) => uhf = 1 unpaired electron.
    let mol = Molecule::parse_xyz(xyz, 1, 2).expect("parse water cation");
    let mut calc = XtbCalculator::new_gfn2(&mol).expect("build calculator");
    let e = calc.energy().expect("single point");

    let dev = (e - XTB_CLI_WATER_CATION).abs();
    assert!(
        dev < 1e-8,
        "water cation energy {e:.12} Ha deviates from xtb CLI {:.12} Ha by {dev:.3e}",
        XTB_CLI_WATER_CATION
    );

    // Sanity: the cation must be well above the neutral, or charge was ignored.
    assert!(
        e > -5.0,
        "cation energy {e:.6} looks like the neutral molecule -- charge not applied?"
    );
}

/// The gradient this binding returns must be *bit-for-bit what xtb itself
/// computes*, checked against the analytic gradient the xtb CLI writes to its
/// `gradient` file for the identical geometry (`xtb --gfn 2 --grad water.xyz`).
///
/// This is the transfer check that is actually meaningful here: it proves the
/// `[natoms][3]` layout, the row/column ordering and the sign convention all
/// survive the FFI boundary. See `gfn2_gradient_disagrees_with_finite_difference`
/// for why this is NOT paired with a passing FD assertion.
#[test]
fn gfn2_gradient_matches_xtb_cli_gradient_file() {
    let _guard = xtb_lock();
    // From the CLI's `gradient` file, xtb 6.7.1, this build:
    //   -1.3570272628572E-17   4.7232145325676E-18  -8.6408120409245E-02
    //    1.0797447332021E-17  -6.4379084252548E-02   4.3204060204623E-02
    //    2.7728252965514E-18   6.4379084252548E-02   4.3204060204623E-02
    const CLI_GRAD: [[f64; 3]; 3] = [
        [-1.3570272628572E-17, 4.7232145325676E-18, -8.6408120409245E-02],
        [1.0797447332021E-17, -6.4379084252548E-02, 4.3204060204623E-02],
        [2.7728252965514E-18, 6.4379084252548E-02, 4.3204060204623E-02],
    ];

    let mol = water();
    let mut calc = XtbCalculator::new_gfn2(&mol).expect("build calculator");
    let g = calc.singlepoint().expect("single point").gradient;

    for atom in 0..3 {
        for axis in 0..3 {
            let dev = (g[[atom, axis]] - CLI_GRAD[atom][axis]).abs();
            assert!(
                dev < 1e-10,
                "gradient[{atom}][{axis}] = {:.14} but the xtb CLI wrote {:.14} (dev {dev:.3e})",
                g[[atom, axis]],
                CLI_GRAD[atom][axis]
            );
        }
    }
}

/// KNOWN-BAD, DOCUMENTED: xtb 6.7.1's analytic gradient does not agree with a
/// finite difference of its own energy, and the discrepancy is **xtb's, not this
/// binding's**.
///
/// Evidence gathered when this binding was written (2026-07-27, xtb 6.7.1
/// commit a59bca3, gfortran 13.3, built from source both with and without
/// OpenMP -- identical results):
///
/// 1. The energy is correct: this binding reproduces the xtb CLI's total energy
///    to <1e-8 Ha for GFN1/GFN2/GFN-FF (the three tests above).
/// 2. The gradient transfer is correct: this binding reproduces the CLI's own
///    `gradient` file to <1e-10 (`gfn2_gradient_matches_xtb_cli_gradient_file`),
///    and the gradient norm to 4e-11.
/// 3. The CLI contradicts itself, with this binding uninvolved: finite
///    differences of the CLI's *own* printed energies give +3.638e-3 Ha/Bohr for
///    the water O-z component while the CLI's own analytic gradient says
///    -8.641e-2 Ha/Bohr. The FD value is stable over three decades of step size
///    (1e-2/1e-3/1e-4 Bohr), so it is not FD noise.
/// 4. Running `xtb --opt` on H2 (0.75 Ang) drives the bond to 2.92 Ang and ends
///    at a *higher* energy (-0.800 Ha) than it started (-0.982 Ha) -- an
///    optimizer following a correct gradient cannot do that.
/// 5. Decisively: **xtb's own unit tests fail the same way.** `meson test` on
///    this source tree reports `unit - xtb:gfn2` "Floating point value
///    missmatch, expected 0.6457E-2 but got 0.1348", plus the same class of
///    failure in `unit - xtb:gfn1` and `unit - xtb:hessian`. Those tests never
///    touch ferric.
///
/// So: **energies from this binding are validated and usable for conformer
/// screening; gradients must not be trusted until the upstream defect is
/// resolved** (suspect the bundled dftd4/multicharge subproject versions that
/// meson resolved to `HEAD` rather than a pinned tag).
///
/// This test asserts the *known* broken state so that it FAILS LOUDLY if a
/// future xtb build fixes the gradient -- at which point delete this test and
/// restore a real analytic-vs-FD check.
#[test]
fn gfn2_gradient_disagrees_with_finite_difference() {
    let _guard = xtb_lock();
    let mol = water();

    let analytic = {
        let mut calc = XtbCalculator::new_gfn2(&mol).expect("build calculator");
        calc.singlepoint().expect("single point").gradient
    };

    const H: f64 = 1e-4; // Bohr
    let energy_at = |atom: usize, axis: usize, delta: f64| -> f64 {
        let mut m = mol.clone();
        let a = &mut m.atoms[atom];
        match axis {
            0 => a.x += delta,
            1 => a.y += delta,
            _ => a.zpos += delta,
        }
        let mut calc = XtbCalculator::new_gfn2(&m).expect("build displaced calculator");
        calc.energy().expect("displaced energy")
    };

    // The O z-component: analytic -8.64e-2 vs FD +3.64e-3 in xtb 6.7.1.
    let fd = (energy_at(0, 2, H) - energy_at(0, 2, -H)) / (2.0 * H);
    let an = analytic[[0, 2]];

    assert!(
        (an - fd).abs() > 1e-3,
        "xtb's analytic gradient now AGREES with finite difference \
         (analytic {an:.10} vs FD {fd:.10}). The upstream gradient defect \
         documented on this test appears to be FIXED -- delete this test and \
         restore a real analytic-vs-FD validation, and re-enable gradients in \
         the crate docs."
    );

    eprintln!(
        "KNOWN xtb 6.7.1 gradient defect: analytic {an:.10} vs FD {fd:.10} Ha/Bohr \
         (see this test's doc comment)"
    );
}

/// Translating the whole molecule must leave the energy unchanged (xTB is
/// translationally invariant) and the gradient must sum to ~zero across atoms
/// (Newton's third law). Catches a coordinate-transfer bug that a single-point
/// energy comparison alone could miss.
///
/// Note this passes even under the gradient defect documented on
/// `gfn2_gradient_disagrees_with_finite_difference`: xtb's analytic gradient is
/// internally well-formed (correct symmetry, sums to zero) -- it just does not
/// match the derivative of its own energy.
#[test]
fn translational_invariance_and_gradient_sum() {
    let _guard = xtb_lock();
    let mol = water();
    let mut calc = XtbCalculator::new_gfn2(&mol).expect("build calculator");
    let out = calc.singlepoint().expect("single point");

    let mut shifted = mol.clone();
    for a in &mut shifted.atoms {
        a.x += 1.5;
        a.y -= 0.75;
        a.zpos += 2.25;
    }
    let mut calc2 = XtbCalculator::new_gfn2(&shifted).expect("build shifted calculator");
    let e_shifted = calc2.energy().expect("shifted energy");

    assert!(
        (out.energy - e_shifted).abs() < 1e-9,
        "energy not translation-invariant: {:.12} vs {:.12}",
        out.energy,
        e_shifted
    );

    for axis in 0..3 {
        let s: f64 = (0..mol.atoms.len()).map(|a| out.gradient[[a, axis]]).sum();
        assert!(
            s.abs() < 1e-8,
            "gradient does not sum to zero along axis {axis}: {s:.3e}"
        );
    }
}

/// Ghost atoms have no xTB counterpart and must be a typed error, not a silently
/// different system.
#[test]
fn ghost_atoms_are_rejected() {
    let _guard = xtb_lock();
    let xyz = "2
ghost dimer
O   0.000000   0.000000   0.000000
@O  0.000000   0.000000   2.000000
";
    let mol = Molecule::parse_xyz(xyz, 0, 1).expect("parse ghost");
    // `XtbCalculator` holds raw FFI pointers and deliberately does not derive
    // Debug, so match rather than `unwrap_err`.
    match XtbCalculator::new_gfn2(&mol) {
        Ok(_) => panic!("expected a ghost-atom error, but the calculator was built"),
        Err(e) => assert!(
            e.to_string().contains("ghost"),
            "expected a ghost-atom error, got: {e}"
        ),
    }
}

/// Unknown method strings must hard-error rather than silently defaulting
/// (repo config-honesty convention).
#[test]
fn unknown_method_string_errors() {
    assert_eq!(XtbMethod::parse_config_str("gfn2").unwrap(), XtbMethod::Gfn2);
    assert_eq!(XtbMethod::parse_config_str("GFN1-xTB").unwrap(), XtbMethod::Gfn1);
    assert_eq!(XtbMethod::parse_config_str("gfn-ff").unwrap(), XtbMethod::GfnFf);
    assert!(XtbMethod::parse_config_str("gfn3").is_err());
    assert!(XtbMethod::parse_config_str("").is_err());
}
