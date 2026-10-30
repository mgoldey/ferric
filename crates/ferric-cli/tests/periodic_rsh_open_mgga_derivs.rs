//! Issue #498: the library routes wired to the CLI by it.
//!
//! * ROKS with range-separated hybrids (energy; library `solve_kroks`,
//!   `gamma_rsh::gamma_roks_any`) at a k-mesh and at the Gamma point.
//! * Closed-shell range-separated hybrid FORCES on the dense-AFT J/K
//!   (`task = "optimize"`; library `kpoint_rks_gradient`,
//!   `gamma_rsh::gamma_rks_rsh_gradient`) at a k-mesh and at the Gamma point.
//! * Open-shell UKS SCAN / r2SCAN forces on the dense-AFT J/K
//!   (`kpoint_uks_gradient`). The matching STRESS anchors (closed-shell
//!   HSE06, UKS SCAN, closed-shell CAM-B3LYP on RS-GDF) are cases of
//!   `periodic_kpoint_stress.rs`.
//! * Open-shell UKS range-separated hybrid FORCES on the dense-AFT J/K
//!   (`kpoint_uks_gradient` at a k-mesh, `gamma_rsh::gamma_uks_rsh_gradient`
//!   through `gamma_uks_any_keep` at the Gamma point).
//! * Closed-shell range-separated hybrid FORCES on the k-point RS-GDF J/K
//!   (`kpoint_rks_gradient` on the `KRsGdf::build_rsh` source the SCF kept).
//!
//! EXACTNESS ANCHOR (CLI vs library, BIT FOR BIT): the CLI energy, and for an
//! optimize run the returned geometry and the gradient evaluated there, are
//! read at full precision from the run log's `run_end` record (serde_json
//! parses floats correctly rounded since #489). The library is then called
//! on the SAME cell (the logged Bohr positions written into the parsed
//! molecule), mesh, grid and configs (the CLI's own config builders restated
//! field by field), and every number must be identical to the last bit.
//! Fixtures: H2 (closed shell) and an H3 doublet in a 4 Bohr cubic cell,
//! STO-3G, dense AFT, exxdiv ewald, XC grid 20x50 with a 14 Bohr cutoff.
//!
//! Each anchor also checks its library value against a sibling that a
//! mis-routed CLI would have produced (the global hybrid PBE0 for HSE06, PBE
//! for SCAN): those differ by far more than one ulp, so a bit-equal pass
//! cannot come from the wrong route.
//!
//! Measured (2026-10-20, the UKS RSH and RS-GDF rows 2026-10-23): every
//! anchor bit-equal. Siblings at the same geometry: k ROKS HSE06 vs PBE0
//! 2.5e-4 Ha, Gamma ROKS CAM-B3LYP vs B3LYP 2.5e-2 Ha; force k HSE06 vs PBE0
//! 5.1e-4, Gamma CAM-B3LYP vs HSE06 2.0e-3, UKS SCAN vs PBE 4.9e-3, r2SCAN vs
//! SCAN 1.4e-3, k UKS HSE06 vs PBE0 4.2e-4, Gamma UKS CAM-B3LYP vs HSE06
//! 3.6e-3, k RKS CAM-B3LYP vs B3LYP on RS-GDF 5.7e-4 Ha/Bohr.
//!
//! Mutation ledger (applied to the CLI source, rebuilt, this file run, then
//! reverted), all KILLED:
//! * Gamma ROKS with `exxdiv = none` forced: CLI -1.11639 vs library -1.57434.
//! * k-point ROKS HSE06 grid `n_radial + 1`: CLI -1.62024 vs -1.62172.
//! * k-point force XC config with `exxdiv = none`; the Gamma RSH force given a
//!   config with `n_radial + 1`; the UKS force on a 110-point angular grid:
//!   each makes the CLI run fail (the library's exxdiv / Fock guards refuse
//!   the mismatched derivative), so the anchor test fails.
//! * The run log reporting the FIRST evaluated gradient instead of the one at
//!   the returned geometry: all four force anchors fail on the bits (e.g. k
//!   HSE06 z-force -1.74e-2 vs +5.2e-3).
//! * (2026-10-23) the Gamma UKS range-separated force given a config with
//!   `n_radial + 1`: the CLI run fails (the routed k-point Fock guard refuses
//!   it, fock_alpha off by 1.17e-3); the k-point UKS force run with the
//!   K-swap spin defect (`KUSpinMutation::PulayKSwap`): the HSE06 anchor fails
//!   on the bits (z-force of atom 0 0.142 vs 0.107).
//! * (periodic_kpoint_stress.rs) the k-point stress XC config with `exxdiv =
//!   none`: the HSE06 stress anchors fail (the CLI run errors); the SCAN /
//!   r2SCAN ones pass (no exact exchange: an equivalent mutant there).
//!
//! Refusals with exact wording: derivatives of an unvalidated
//! range-separated hybrid (wB97X; RKS and UKS, k and Gamma), ROKS meta-GGA
//! derivatives (its energy runs: `periodic_open_mgga_energy.rs`), UKS
//! meta-GGA derivatives on RS-GDF. The open-shell range-separated
//! derivatives and the Gamma ones on RS-GDF, refused here before, are wired
//! (#498) and anchored in `periodic_open_rsh_derivs.rs` and its validation
//! twin.
//!
//! TIERS. The four open-shell UKS force anchors with two optimizer steps and
//! a sibling check (k-point UKS HSE06 / SCAN / r2SCAN, Gamma UKS CAM-B3LYP;
//! 190-245 s each on CI) run weekly in
//! `validation_periodic_rsh_open_mgga_derivs.rs`. Per commit, each of those
//! routes keeps a CANARY here: the same CLI == library bit-for-bit force at
//! the INPUT geometry (`max_steps = 0`, one CLI SCF + gradient, no sibling).
//! The canaries still kill the K-swap spin defect and every config mismatch
//! of the ledger above; the "first gradient instead of the returned one"
//! mutant is killed per commit by the closed-shell two-step anchors below.
//! Shared harness: `rsh_open_mgga_cli/mod.rs`.

mod rsh_open_mgga_cli;
use rsh_open_mgga_cli::*;

// ───────────────────────────────────────────────────────── ROKS energies ──

#[test]
fn kpoint_roks_hse06_energy_equals_the_library_bit_for_bit() {
    let lib = lib_kroks_energy("HSE06");
    roks_energy_anchor(
        "kroks-hse",
        "HSE06",
        "[1, 1, 2]",
        lib,
        lib_kroks_energy("PBE0"),
    );
}

#[test]
fn gamma_roks_cam_b3lyp_energy_equals_the_library_bit_for_bit() {
    let lib = lib_gamma_roks_energy(CAM);
    roks_energy_anchor("groks-cam", CAM, "", lib, lib_gamma_roks_energy("B3LYP"));
}

// ──────────────────────────────────────────────────────────────── forces ──

#[test]
fn kpoint_rks_hse06_force_equals_the_library_bit_for_bit() {
    let cli = optimize("krks-hse-opt", "rhf", "HSE06", "[1, 1, 2]");
    force_anchor("krks-hse-opt", cli, lib_krks_force, "HSE06", "PBE0");
}

#[test]
fn gamma_rks_cam_b3lyp_force_equals_the_library_bit_for_bit() {
    let cli = optimize("grks-cam-opt", "rhf", CAM, "");
    force_anchor("grks-cam-opt", cli, lib_gamma_rks_rsh_force, CAM, "HSE06");
}

#[test]
fn kpoint_rks_cam_b3lyp_rsgdf_force_equals_the_library_bit_for_bit() {
    let cli = optimize_on("krks-cam-gdf-opt", "rhf", CAM, "[1, 1, 2]", RSGDF);
    force_anchor("krks-cam-gdf-opt", cli, lib_krks_force_rsgdf, CAM, "B3LYP");
}

/// CANARY of the weekly k-point UKS HSE06 force anchor: CLI == library bit
/// for bit at the input geometry (no optimizer step, no sibling). The
/// K-swap spin defect (`KUSpinMutation::PulayKSwap`) fails it on the bits.
#[test]
fn kpoint_uks_hse06_force_at_the_input_geometry_equals_the_library() {
    let tag = "kuks-hse-f0";
    let cli = optimize_steps(tag, "uhf", "HSE06", "[1, 1, 2]", "", 0);
    force_bits_anchor(tag, cli, &lib_kuks_force, "HSE06");
}

/// CANARY of the weekly k-point UKS SCAN / r2SCAN force anchors (the
/// open-shell meta-GGA tau terms): CLI == library bit for bit at the input
/// geometry.
#[test]
fn kpoint_uks_scan_force_at_the_input_geometry_equals_the_library() {
    let tag = "kuks-scan-f0";
    let cli = optimize_steps(tag, "uhf", "SCAN", "[1, 1, 2]", "", 0);
    force_bits_anchor(tag, cli, &lib_kuks_force, "SCAN");
}

/// CANARY of the weekly Gamma UKS CAM-B3LYP force anchor (the Gamma UKS
/// range-separated route, `gamma_uks_any_keep` + `gamma_uks_rsh_gradient`):
/// CLI == library bit for bit at the input geometry.
#[test]
fn gamma_uks_cam_b3lyp_force_at_the_input_geometry_equals_the_library() {
    let tag = "guks-cam-f0";
    let cli = optimize_steps(tag, "uhf", CAM, "", "", 0);
    force_bits_anchor(tag, cli, &lib_gamma_uks_rsh_force, CAM);
}

// ────────────────────────────────────────────────────────────── refusals ──

#[test]
fn refusals_have_exact_wording() {
    // The closed- and open-shell RSH derivatives of HSE06 / CAM-B3LYP run on
    // either J/K at a k-mesh and at Gamma (anchored above, in
    // periodic_kpoint_stress.rs and in periodic_open_rsh_derivs.rs, #498).
    // Range-separated hybrids outside the force/stress validation (wB97X)
    // are refused by name for derivatives, at a k-mesh and at Gamma (dense).
    let base = Input {
        tag: "",
        kind: "rhf",
        functional: "HYB_GGA_XC_WB97X",
        task: "optimize",
        kmesh: "",
        cell_extra: "",
        steps: 0,
    };
    let wb = "HYB_GGA_XC_WB97X: the range-separated hybrid force and stress are validated \
              for HSE06 / CAM-B3LYP (HYB_GGA_XC_CAM_B3LYP) only";
    for (tag, kmesh, prefix) in [
        (
            "wb-k-opt",
            "[1, 1, 2]",
            "method.task = \"optimize\" with [cell] kmesh: ",
        ),
        ("wb-g-opt", "", "method.task = \"optimize\" with [cell]: "),
    ] {
        let e = refused(&Input { tag, kmesh, ..base });
        assert!(e.contains(prefix) && e.contains(wb), "{tag}: {e}");
    }
    let e = refused(&Input {
        tag: "wb-k-stress",
        task: "energy",
        kmesh: "[1, 1, 2]",
        cell_extra: "stress = true",
        ..base
    });
    assert!(
        e.contains("[cell] stress = true: ") && e.contains(wb),
        "{e}"
    );
    // ... and for the open-shell UKS force (k-point and Gamma).
    for (tag, kmesh) in [("wb-uks-k-opt", "[1, 1, 2]"), ("wb-uks-g-opt", "")] {
        let e = refused(&Input {
            tag,
            kind: "uhf",
            kmesh,
            ..base
        });
        assert!(e.contains("optimize") && e.contains(wb), "{tag}: {e}");
    }
    // ROKS meta-GGA derivatives (the energy runs, #498: the derivative gate
    // names them) and UKS meta-GGA derivatives on RS-GDF.
    let e = refused(&Input {
        tag: "roks-scan",
        kind: "rohf",
        functional: "SCAN",
        task: "optimize",
        kmesh: "[1, 1, 2]",
        cell_extra: "",
        steps: 0,
    });
    assert!(
        e.contains("method.task = \"optimize\" with [cell] kmesh: ")
            && e.contains("SCAN: the k-point ROKS meta-GGA force and stress"),
        "{e}"
    );
    let e = refused(&Input {
        tag: "uks-scan-rsgdf",
        kind: "uhf",
        functional: "SCAN",
        task: "optimize",
        kmesh: "[1, 1, 2]",
        cell_extra: RSGDF,
        steps: 0,
    });
    assert!(e.contains("dense-AFT J/K only"), "{e}");
}
