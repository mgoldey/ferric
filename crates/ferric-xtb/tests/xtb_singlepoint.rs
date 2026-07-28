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
///   TOTAL ENERGY  -5.070325128562 Eh   (gradient norm 0.008098996263 Eh/a0)
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
    const XTB_CLI_GFN2_WATER_GNORM: f64 = 0.008098996263;
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
/// survive the FFI boundary. It is paired with
/// `gfn2_gradient_matches_finite_difference`, which independently checks that
/// the value xtb computes is the true derivative of its own energy.
#[test]
fn gfn2_gradient_matches_xtb_cli_gradient_file() {
    let _guard = xtb_lock();
    // From the CLI's `gradient` file, xtb 6.7.1 built at -O2 (see the module
    // doc for why the optimisation level matters):
    //    9.2510791586002E-17  -2.3032361083061E-17   3.6377058406328E-03
    //   -8.6487037493821E-17  -4.7824880967663E-03  -1.8188529203163E-03
    //   -6.0237540921806E-18   4.7824880967663E-03  -1.8188529203164E-03
    const CLI_GRAD: [[f64; 3]; 3] = [
        [9.2510791586002E-17, -2.3032361083061E-17, 3.6377058406328E-03],
        [-8.6487037493821E-17, -4.7824880967663E-03, -1.8188529203163E-03],
        [-6.0237540921806E-18, 4.7824880967663E-03, -1.8188529203164E-03],
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

/// The analytic gradient must be the true derivative of xtb's own energy:
/// every component is checked against a central finite difference of the
/// energies this same binding returns.
///
/// This is the check that makes gradients *usable* (forces, geometry
/// optimisation) rather than merely faithfully transferred. It is the
/// complement of `gfn2_gradient_matches_xtb_cli_gradient_file`: that test
/// proves ferric reports what xtb computed, this one proves what xtb computed
/// is right.
///
/// # History: this used to assert the opposite
///
/// Until 2026-07-27 this test asserted a *known-broken* upstream state, because
/// xtb's analytic gradient disagreed with FD by ~20x (water O-z: analytic
/// -8.641e-2 vs FD +3.638e-3 Ha/Bohr), `xtb --opt` on H2 ran the bond from
/// 0.75 to 2.92 Ang while raising the energy, and xtb's own `meson test` failed
/// `unit - xtb:gfn1`, `gfn2` and `hessian` with "expected 0.6457E-2 but got
/// 0.1348".
///
/// **Root cause: a gfortran 13.3 miscompilation of xtb 6.7.1 at `-O3`**, not a
/// bug in xtb's source, its dftd4/multicharge subprojects (both were correctly
/// pinned to v3.5.0/v0.2.0), or this binding. `meson --buildtype=release`
/// selects `-O3`; rebuilding the identical source at `-O0`, `-O1` or `-O2`
/// gives a gradient that matches FD to ~7 significant figures and makes xtb's
/// own gfn1/gfn2/hessian unit tests pass. Only GFN1/GFN2 were affected -- GFN0
/// uses a different gradient path (`peeq_module.f90`) and was always correct.
/// Ruled out by experiment: OpenBLAS threading (identical result at
/// `OPENBLAS_NUM_THREADS=1`) and the BLAS backend itself (identical under
/// `LD_PRELOAD` of reference netlib BLAS/LAPACK).
///
/// So libxtb **must be built at `-O2` or lower** with gfortran 13.x. If this
/// test starts failing again, check the optimisation level of the installed
/// libxtb before suspecting ferric.
#[test]
fn gfn2_gradient_matches_finite_difference() {
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

    // 1e-6 Ha/Bohr comfortably covers central-difference truncation at
    // h=1e-4 Bohr while still being ~3 orders tighter than the -O3 defect.
    const TOL: f64 = 1e-6;

    for atom in 0..3 {
        for axis in 0..3 {
            let fd = (energy_at(atom, axis, H) - energy_at(atom, axis, -H)) / (2.0 * H);
            let an = analytic[[atom, axis]];
            let dev = (an - fd).abs();
            assert!(
                dev < TOL,
                "gradient[{atom}][{axis}]: analytic {an:.10} vs finite difference \
                 {fd:.10} Ha/Bohr (dev {dev:.3e} > {TOL:.0e}). If libxtb was built \
                 at -O3 with gfortran 13.x, that is the known miscompilation -- \
                 rebuild at -O2 (see this test's doc comment)."
            );
        }
    }
}

/// Translating the whole molecule must leave the energy unchanged (xTB is
/// translationally invariant) and the gradient must sum to ~zero across atoms
/// (Newton's third law). Catches a coordinate-transfer bug that a single-point
/// energy comparison alone could miss.
///
/// Note this is a weaker check than it looks: it passed even under the `-O3`
/// miscompilation described on `gfn2_gradient_matches_finite_difference`,
/// because that defect left the gradient internally well-formed (correct
/// symmetry, summing to zero) while still not being the derivative of the
/// energy. Translational invariance alone cannot catch that -- the FD test is
/// what does.
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
