//! Weekly half of `periodic_rsh_open_mgga_derivs.rs` (issue #498): the four
//! open-shell UKS CLI == library force anchors that run two optimizer steps
//! and check a sibling functional (k-point UKS SCAN, r2SCAN and HSE06 on a
//! Gamma-centred 1x1x2 mesh, Gamma UKS CAM-B3LYP; all dense AFT). Each took
//! 190-245 s on CI (runs 40957025137 / 40953725119, mean), above the 120 s
//! per-commit threshold, so they moved here. The fixtures, the exactness
//! anchor, the sibling gaps and the mutation ledger are documented in
//! `periodic_rsh_open_mgga_derivs.rs`; the harness is
//! `rsh_open_mgga_cli/mod.rs`.
//!
//! Per-commit canaries left in `periodic_rsh_open_mgga_derivs.rs`: the same
//! four routes (k UKS HSE06, k UKS SCAN, Gamma UKS CAM-B3LYP bit for bit at
//! the input geometry, `*_force_at_the_input_geometry_equals_the_library`;
//! k UKS r2SCAN bit for bit there WITH the SCAN sibling check,
//! `kpoint_uks_r2scan_force_at_the_input_geometry_differs_from_scan`).
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo test -p ferric-cli \
//!   --test validation_periodic_rsh_open_mgga_derivs -- --ignored --nocapture
//! ```

mod rsh_open_mgga_cli;
use rsh_open_mgga_cli::*;

#[test]
#[ignore = "validation: periodic open-shell UKS force CLI == library (two steps + sibling); CI ~190 s"]
fn kpoint_uks_scan_force_equals_the_library_bit_for_bit() {
    let cli = optimize("kuks-scan-opt", "uhf", "SCAN", "[1, 1, 2]");
    force_anchor("kuks-scan-opt", cli, lib_kuks_force, "SCAN", "PBE");
}

#[test]
#[ignore = "validation: periodic open-shell UKS force CLI == library (two steps + sibling); CI ~193 s"]
fn kpoint_uks_r2scan_force_equals_the_library_bit_for_bit() {
    let cli = optimize("kuks-r2scan-opt", "uhf", "r2SCAN", "[1, 1, 2]");
    force_anchor("kuks-r2scan-opt", cli, lib_kuks_force, "r2SCAN", "SCAN");
}

#[test]
#[ignore = "validation: periodic open-shell UKS force CLI == library (two steps + sibling); CI ~244 s"]
fn kpoint_uks_hse06_force_equals_the_library_bit_for_bit() {
    let cli = optimize("kuks-hse-opt", "uhf", "HSE06", "[1, 1, 2]");
    force_anchor("kuks-hse-opt", cli, lib_kuks_force, "HSE06", "PBE0");
}

#[test]
#[ignore = "validation: periodic open-shell UKS force CLI == library (two steps + sibling); CI ~199 s"]
fn gamma_uks_cam_b3lyp_force_equals_the_library_bit_for_bit() {
    let cli = optimize("guks-cam-opt", "uhf", CAM, "");
    force_anchor("guks-cam-opt", cli, lib_gamma_uks_rsh_force, CAM, "HSE06");
}
