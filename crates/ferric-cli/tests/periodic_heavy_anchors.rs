//! Issue #498, "fixtures beyond H/STO-3G": the CLI-vs-library anchors of the
//! wired periodic derivative routes, on a basis with p shells.
//!
//! Every #498 binding anchor before this file used bundled STO-3G H cells
//! (s functions only), so nothing in the binding-level suite touched an
//! l > 0 shell in a force or stress contraction. Here the cells of
//! `periodic_rsh_open_mgga_derivs.rs` (H2 closed shell, H3 doublet, cubic
//! a = 4 Bohr, dense-AFT J/K, exxdiv ewald, omega 0.8, XC grid 20x50 with a
//! 14 Bohr cutoff, 1x1x2 Gamma-centred mesh or the Gamma point) carry STO-3G
//! H plus one p shell (zeta 1.0, read through `[basis] path` as G94). The
//! extra shell keeps the dense-AFT G sphere of the H/STO-3G cells (it is set
//! by the tightest exponent), which is what keeps these per-commit.
//!
//! Routes here (CLI = library BIT FOR BIT, energy and force at the
//! optimizer's moved geometry): the closed-shell RSH force, k-point HSE06 and
//! Gamma CAM-B3LYP. Every other wired route (closed-shell RSH stress,
//! open-shell UKS SCAN / r2SCAN force and stress, k-point UKS HSE06 force,
//! Gamma UKS CAM-B3LYP force) takes 40-80 s per test on a p-shell cell and is
//! in `validation_periodic_heavy_anchors.rs`, with the d-shell cells and the
//! heavy-atom (LiH / LiH2, STO-3G) cells. The CLI has no Gamma stress; the
//! Gamma RSH stress is anchored in Python (test_pbc_heavy_anchors.py), which
//! also holds the physics check (a central FD of the binding's own energy)
//! for every route on the p-shell cells (nightly, non-blocking).
//!
//! What this file can and cannot see: CLI = library bit for bit catches any
//! glue defect on a p-shell cell (a dropped `[basis] path`, a config field
//! the CLI restates differently, a grid or exxdiv mismatch); the basis guard
//! (`nbasis = 8` printed by the CLI and `PreparedBasis::nbasis()` on the
//! library side) catches a fall-back to an s-only basis. It CANNOT see a
//! defect inside the library's l > 0 derivative contraction, because both
//! sides call the same library. That is the FD anchors' job: one runs here,
//! per commit and PR-gating (the library Gamma RKS CAM-B3LYP force on the
//! p-shell cell against a directional FD of the library energy); the rest are
//! in the Python file, which CI runs only in the nightly, non-blocking
//! coverage job and the validation tier.
//!
//! Measured (2026-10-26, test profile, one test at a time): bit-equal; k-point
//! HSE06 30 s, Gamma CAM-B3LYP 20 s; the FD anchor 9 s.

mod heavy_anchors_cli;

use heavy_anchors_cli::*;

#[test]
fn kpoint_rks_hse06_force_p_shell_equals_the_library_bit_for_bit() {
    force_anchor(
        "krks-hse-p",
        H2_SP,
        "rhf",
        "HSE06",
        "[1, 1, 2]",
        lib_krks_force,
    );
}

#[test]
fn gamma_rks_cam_b3lyp_force_p_shell_equals_the_library_bit_for_bit() {
    force_anchor("grks-cam-p", H2_SP, "rhf", CAM, "", lib_gamma_rks_rsh_force);
}

/// The PR-gating PHYSICS anchor of the p-shell derivative path (the Python
/// FD anchors run only in the nightly, non-blocking coverage job and in the
/// validation tier): the library Gamma RKS CAM-B3LYP force on H2 + p (the
/// route the CLI anchor above pins to the CLI) against a central FD of the
/// library's own energy along [`generic_direction`], h = 1e-4 Bohr.
///
/// Mutant (`if ax > 0` -> `if ax > 1` at the x lowering term of the k-point
/// pair-FT derivative, crates/ferric-pbc/src/pair_ft/residues.rs, which the
/// Gamma range-separated force runs through; s shells are blind to it):
/// KILLED, miss 4.6e-5, while the two CLI = library anchors above pass under
/// it (both sides call the mutated library). Measured clean: miss 5.2e-10
/// (Prescott kernel 5.2e-10); axis shares of g . d 6.6e-3 / 1.3e-3 / 6.8e-2;
/// 9 s.
#[test]
fn gamma_rks_cam_b3lyp_force_p_shell_matches_fd_of_the_library_energy() {
    let x0: Vec<f64> = H2_SP.atoms.iter().flatten().copied().collect();
    let d = generic_direction(H2_SP.atoms.len());
    let (_, g) = lib_gamma_rks_rsh_force(&H2_SP, CAM, &x0);
    let moved = |s: f64| -> Vec<f64> { x0.iter().zip(&d).map(|(x, v)| x + s * v).collect() };
    let ep = lib_gamma_rks_rsh_energy(&H2_SP, CAM, &moved(FD_H));
    let em = lib_gamma_rks_rsh_energy(&H2_SP, CAM, &moved(-FD_H));
    let fd = (ep - em) / (2.0 * FD_H);
    let gd: f64 = g.iter().zip(&d).map(|(a, b)| a * b).sum();
    // Each Cartesian branch contributes: a defect in any one axis moves g . d.
    for axis in 0..3 {
        let share: f64 = (0..H2_SP.atoms.len())
            .map(|i| g[3 * i + axis] * d[3 * i + axis])
            .sum();
        eprintln!("axis {axis}: share of g.d {share:.3e}");
        assert!(
            share.abs() > 1e-4,
            "axis {axis}: share {share:e} is trivial"
        );
    }
    let miss = (gd - fd).abs();
    eprintln!("Gamma RKS CAM-B3LYP p-shell: g.d {gd:.12e} FD {fd:.12e} miss {miss:.2e}");
    assert!(miss < FD_BAR, "analytic g.d {gd} vs FD {fd}: miss {miss:e}");
}

const FD_H: f64 = 1e-4;
/// Measured miss 5.2e-10 (the h^2 floor of the H/STO-3G cells); bar ~100x
/// that and ~900x below the p-lowering mutant's 4.6e-5.
const FD_BAR: f64 = 5e-8;
