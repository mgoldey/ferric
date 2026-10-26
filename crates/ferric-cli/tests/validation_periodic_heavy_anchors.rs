//! Issue #498, "fixtures beyond H/STO-3G" (validation tier): the slower
//! CLI-vs-library anchors of `periodic_heavy_anchors.rs` (see its module doc
//! for the cell, the settings and what a bit-equal anchor can and cannot see).
//!
//! * p shells (H + one p, zeta 1.0): the k-point RKS HSE06 stress, the
//!   k-point UKS SCAN force, SCAN / r2SCAN stress and HSE06 force, and the
//!   Gamma UKS CAM-B3LYP force.
//! * d shells (H + p + pure d, zeta 1.0): k-point RKS HSE06 force and stress,
//!   k-point UKS SCAN force.
//! * A heavy atom, bundled STO-3G (Li 1s, 2s, 2p on a Z = 3 nucleus): the
//!   k-point RKS HSE06 force and stress on LiH, the k-point UKS SCAN force on
//!   the LiH2 doublet. Li's 1s exponent (16.1) sets a dense-AFT G sphere ~10x
//!   the H cells', so each derivative evaluation costs minutes, not seconds.
//!
//! Measured 2026-10-26 (test profile, one test at a time; the times are in
//! the ignore reasons): every CLI run and library call bit-equal.

mod heavy_anchors_cli;

use heavy_anchors_cli::*;

// ──────────────────────────────────────────────────────────── p shells ──

#[test]
#[ignore = "validation: p-shell k-point RKS HSE06 stress, ~40 s"]
fn kpoint_rks_hse06_stress_p_shell_equals_the_library_bit_for_bit() {
    stress_anchor("krks-hse-p-stress", H2_SP, "rhf", "HSE06", lib_krks_stress);
}

#[test]
#[ignore = "validation: p-shell k-point UKS SCAN force, ~60 s"]
fn kpoint_uks_scan_force_p_shell_equals_the_library_bit_for_bit() {
    force_anchor(
        "kuks-scan-p",
        H3_SP,
        "uhf",
        "SCAN",
        "[1, 1, 2]",
        lib_kuks_force,
    );
}

#[test]
#[ignore = "validation: p-shell Gamma UKS CAM-B3LYP force, ~55 s"]
fn gamma_uks_cam_b3lyp_force_p_shell_equals_the_library_bit_for_bit() {
    force_anchor("guks-cam-p", H3_SP, "uhf", CAM, "", lib_gamma_uks_rsh_force);
}

#[test]
#[ignore = "validation: p-shell k-point UKS SCAN stress, ~62 s"]
fn kpoint_uks_scan_stress_p_shell_equals_the_library_bit_for_bit() {
    stress_anchor("kuks-scan-p-stress", H3_SP, "uhf", "SCAN", lib_kuks_stress);
}

#[test]
#[ignore = "validation: p-shell k-point UKS r2SCAN stress, ~63 s"]
fn kpoint_uks_r2scan_stress_p_shell_equals_the_library_bit_for_bit() {
    stress_anchor(
        "kuks-r2scan-p-stress",
        H3_SP,
        "uhf",
        "r2SCAN",
        lib_kuks_stress,
    );
}

#[test]
#[ignore = "validation: p-shell k-point UKS HSE06 force, ~80 s"]
fn kpoint_uks_hse06_force_p_shell_equals_the_library_bit_for_bit() {
    force_anchor(
        "kuks-hse-p",
        H3_SP,
        "uhf",
        "HSE06",
        "[1, 1, 2]",
        lib_kuks_force,
    );
}

// ──────────────────────────────────────────────────────────── d shells ──

#[test]
#[ignore = "validation: d-shell k-point RKS HSE06 force, ~95 s"]
fn kpoint_rks_hse06_force_d_shell_equals_the_library_bit_for_bit() {
    force_anchor(
        "krks-hse-d",
        H2_SPD,
        "rhf",
        "HSE06",
        "[1, 1, 2]",
        lib_krks_force,
    );
}

#[test]
#[ignore = "validation: d-shell k-point RKS HSE06 stress, ~140 s"]
fn kpoint_rks_hse06_stress_d_shell_equals_the_library_bit_for_bit() {
    stress_anchor("krks-hse-d-stress", H2_SPD, "rhf", "HSE06", lib_krks_stress);
}

#[test]
#[ignore = "validation: d-shell k-point UKS SCAN force, ~190 s"]
fn kpoint_uks_scan_force_d_shell_equals_the_library_bit_for_bit() {
    force_anchor(
        "kuks-scan-d",
        H3_SPD,
        "uhf",
        "SCAN",
        "[1, 1, 2]",
        lib_kuks_force,
    );
}

// ───────────────────────────────────────────────────── heavy atom (Li) ──

#[test]
#[ignore = "validation: LiH k-point RKS HSE06 force (dense AFT, Li 1s G sphere), ~9 min"]
fn kpoint_rks_hse06_force_lih_equals_the_library_bit_for_bit() {
    force_anchor(
        "krks-hse-lih",
        LIH,
        "rhf",
        "HSE06",
        "[1, 1, 2]",
        lib_krks_force,
    );
}

#[test]
#[ignore = "validation: LiH k-point RKS HSE06 stress (dense AFT, Li 1s G sphere), ~13 min"]
fn kpoint_rks_hse06_stress_lih_equals_the_library_bit_for_bit() {
    stress_anchor("krks-hse-lih-stress", LIH, "rhf", "HSE06", lib_krks_stress);
}

#[test]
#[ignore = "validation: LiH2 k-point UKS SCAN force (dense AFT, Li 1s G sphere), ~8 min"]
fn kpoint_uks_scan_force_lih2_equals_the_library_bit_for_bit() {
    force_anchor(
        "kuks-scan-lih2",
        LIH2,
        "uhf",
        "SCAN",
        "[1, 1, 2]",
        lib_kuks_force,
    );
}
