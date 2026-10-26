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
//! for every route on the p-shell cells, per commit.
//!
//! What this file can and cannot see: CLI = library bit for bit catches any
//! glue defect on a p-shell cell (a dropped `[basis] path`, a config field
//! the CLI restates differently, a grid or exxdiv mismatch); the basis guard
//! (`nbasis = 8` printed by the CLI and `PreparedBasis::nbasis()` on the
//! library side) catches a fall-back to an s-only basis. It CANNOT see a
//! defect inside the library's l > 0 derivative contraction, because both
//! sides call the same library: that is the FD anchors' job (the Python
//! file's mutation ledger).
//!
//! Measured (2026-10-26, test profile, one test at a time): bit-equal; k-point
//! HSE06 30 s, Gamma CAM-B3LYP 20 s.

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
