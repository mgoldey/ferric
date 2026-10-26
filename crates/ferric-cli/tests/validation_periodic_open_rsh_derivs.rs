//! Validation tier of issue #498 (open-shell RSH derivatives and the Gamma
//! RSH derivatives on RS-GDF, wired to the CLI): the CLI == library anchors
//! that do not fit the per-commit budget (`periodic_open_rsh_derivs.rs` keeps
//! one anchor per gap plus the FD checks), and the sibling checks that make
//! a bit-equal pass meaningful. Shared harness: `open_rsh_cli/mod.rs`.
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo test -p ferric-cli \
//!   --test validation_periodic_open_rsh_derivs -- --ignored --nocapture
//! ```
//!
//! # Artifact hypothesis next to the physics hypothesis
//!
//! * Right wiring: every CLI number equals the library one at the same
//!   settings bit for bit, and the sibling a mis-routed run would have
//!   produced (another range-separated hybrid; the dense-AFT J/K for an
//!   RS-GDF run) differs by far more than one ulp.
//! * Wrong wiring that still runs (a functional, grid, exxdiv, J/K or spin
//!   state other than the SCF's): the library's routed Fock guards refuse it
//!   (the CLI run fails), or the numbers differ in the bits.
//!
//! # Measured (2026-10-26)
//!
//! Every anchor bit-equal (energy, force, stress), including the p-shell
//! cell (STO-3G + one p shell, zeta 1, on every H: 12 functions for H3; the
//! basis size is asserted on both sides) and a three-step Gamma ROKS HSE06
//! optimization (returned geometry moved 0.22 Bohr). Siblings at the same
//! geometry (max |d|, Ha/Bohr or Ha/Bohr^3): k ROKS CAM-B3LYP RS-GDF force vs
//! HSE06 RS-GDF 5.15e-3, vs CAM-B3LYP dense 1.28e-5; k UKS CAM-B3LYP RS-GDF vs
//! dense 1.30e-5; Gamma ROKS CAM-B3LYP vs HSE06 3.80e-3; Gamma ROKS HSE06
//! RS-GDF vs dense 3.81e-5; Gamma RKS HSE06 RS-GDF vs dense 5.97e-6; Gamma UKS
//! CAM-B3LYP RS-GDF vs HSE06 RS-GDF 3.57e-3; k ROKS HSE06 vs CAM-B3LYP
//! 5.14e-3; p-cell: k ROKS HSE06 vs CAM-B3LYP 5.12e-3, k UKS CAM-B3LYP RS-GDF
//! vs dense 3.53e-5. Stress: k ROKS CAM-B3LYP RS-GDF vs dense 6.88e-7, k ROKS
//! HSE06 vs k UKS HSE06 3.73e-5, k UKS HSE06 vs CAM-B3LYP 7.98e-4 (p-cell
//! 7.63e-4); Gamma UKS CAM-B3LYP RS-GDF vs dense 3.70e-6, Gamma RKS HSE06
//! RS-GDF vs dense 1.12e-7. The whole file runs in ~530 s on 3 threads.

mod open_rsh_cli;
use open_rsh_cli::*;

fn opt<'a>(tag: &'a str, spin: Spin, f: &'a str, kpoints: bool, rsgdf: bool) -> Run<'a> {
    Run {
        kpoints,
        rsgdf,
        ..Run::new(tag, spin, f, "optimize")
    }
}

/// CLI == library bit for bit, and the library force of each `sibling`
/// (another functional and / or the other J/K) at the same geometry
/// differs by more than `1e-6`.
fn force_case(r: &Run, siblings: &[(&str, bool)]) {
    let c = force_anchor(r);
    for &(f, g) in siblings {
        let (_, gs) = lib_force(&r.with(f, g), Some(&c.1));
        let d = max_diff(&c.2, &gs);
        eprintln!(
            "{}: sibling ({f}, rsgdf {g}) force differs by {d:.2e}",
            r.tag
        );
        assert!(d > 1e-6, "{}: sibling ({f}, {g}) too close: {d:e}", r.tag);
    }
}

/// CLI == library bit for bit (stress), and each sibling's library stress
/// differs by more than `1e-8`.
fn stress_case(r: &Run, siblings: &[(Spin, &str, bool)]) {
    let (_, s) = stress_anchor(r);
    for &(spin, f, g) in siblings {
        let (_, ss) = lib_stress(&Run {
            spin,
            ..r.with(f, g)
        });
        let d = max_diff(&s, &ss);
        eprintln!(
            "{}: sibling ({spin:?}, {f}, rsgdf {g}) stress differs by {d:.2e}",
            r.tag
        );
        assert!(d > 1e-8, "{}: sibling too close: {d:e}", r.tag);
    }
}

// ─────────────────────────────────────────────────────────────── forces ──

#[test]
#[ignore = "validation: periodic open-shell RSH derivatives (CLI)"]
fn kpoint_roks_cam_b3lyp_rsgdf_force() {
    let r = opt("v-kroks-cam-gdf", Spin::Roks, CAM, true, true);
    force_case(&r, &[("HSE06", true), (CAM, false)]);
}

#[test]
#[ignore = "validation: periodic open-shell RSH derivatives (CLI)"]
fn kpoint_uks_cam_b3lyp_rsgdf_force() {
    let r = opt("v-kuks-cam-gdf", Spin::Uks, CAM, true, true);
    force_case(&r, &[(CAM, false)]);
}

#[test]
#[ignore = "validation: periodic open-shell RSH derivatives (CLI)"]
fn gamma_roks_cam_b3lyp_dense_force_siblings() {
    // The per-commit anchor's siblings, at the fixture geometry (library).
    let r = opt("v-groks-cam", Spin::Roks, CAM, false, false);
    let (_, g) = lib_force(&r, None);
    let (_, gs) = lib_force(&r.with("HSE06", false), None);
    let d = max_diff(&g, &gs);
    eprintln!("v-groks-cam: HSE06 sibling force differs by {d:.2e}");
    assert!(d > 1e-6, "{d:e}");
}

#[test]
#[ignore = "validation: periodic open-shell RSH derivatives (CLI)"]
fn gamma_roks_hse06_rsgdf_force() {
    let r = opt("v-groks-hse-gdf", Spin::Roks, "HSE06", false, true);
    force_case(&r, &[("HSE06", false)]);
}

#[test]
#[ignore = "validation: periodic open-shell RSH derivatives (CLI)"]
fn gamma_rks_hse06_rsgdf_force() {
    let r = opt("v-grks-hse-gdf", Spin::Rks, "HSE06", false, true);
    force_case(&r, &[("HSE06", false)]);
}

#[test]
#[ignore = "validation: periodic open-shell RSH derivatives (CLI)"]
fn gamma_uks_cam_b3lyp_rsgdf_force() {
    let r = opt("v-guks-cam-gdf", Spin::Uks, CAM, false, true);
    force_case(&r, &[("HSE06", true)]);
}

/// The per-commit k ROKS HSE06 force anchor's sibling, at the fixture
/// geometry (library).
#[test]
#[ignore = "validation: periodic open-shell RSH derivatives (CLI)"]
fn kpoint_roks_hse06_force_sibling() {
    let r = opt("v-sib-kroks", Spin::Roks, "HSE06", true, false);
    let (_, g) = lib_force(&r, None);
    let (_, gs) = lib_force(&r.with(CAM, false), None);
    let d = max_diff(&g, &gs);
    eprintln!("v-sib-kroks: CAM-B3LYP sibling force differs by {d:.2e}");
    assert!(d > 1e-6, "{d:e}");
}

/// A multi-step Gamma ROKS HSE06 optimization (three BFGS steps, dense AFT):
/// the logged geometry is a MOVED one and the gradient logged there equals
/// the library at that geometry bit for bit (the run log reports the
/// gradient of the returned geometry, not the first one evaluated).
#[test]
#[ignore = "validation: periodic open-shell RSH derivatives (CLI)"]
fn gamma_roks_hse06_three_step_optimize() {
    let r = Run {
        steps: 3,
        ..opt("v-groks-hse-opt3", Spin::Roks, "HSE06", false, false)
    };
    let (_, x, _) = force_anchor(&r);
    let x0: Vec<f64> = H3.iter().flatten().copied().collect();
    let moved = max_diff(&x, &x0);
    eprintln!("v-groks-hse-opt3: the returned geometry moved by {moved:.2e} Bohr");
    assert!(moved > 1e-3, "the optimizer did not move: {moved:e}");
}

// ───────────────────────────────────────────────────────────── stresses ──

fn stress<'a>(tag: &'a str, spin: Spin, f: &'a str, rsgdf: bool) -> Run<'a> {
    Run {
        rsgdf,
        ..Run::new(tag, spin, f, "stress")
    }
}

#[test]
#[ignore = "validation: periodic open-shell RSH derivatives (CLI)"]
fn kpoint_roks_cam_b3lyp_rsgdf_stress() {
    let r = stress("v-kroks-cam-gdf-s", Spin::Roks, CAM, true);
    stress_case(&r, &[(Spin::Roks, CAM, false)]);
}

#[test]
#[ignore = "validation: periodic open-shell RSH derivatives (CLI)"]
fn kpoint_uks_cam_b3lyp_rsgdf_stress() {
    stress_case(&stress("v-kuks-cam-gdf-s", Spin::Uks, CAM, true), &[]);
}

#[test]
#[ignore = "validation: periodic open-shell RSH derivatives (CLI)"]
fn kpoint_roks_hse06_dense_stress() {
    let r = stress("v-kroks-hse-s", Spin::Roks, "HSE06", false);
    stress_case(&r, &[(Spin::Uks, "HSE06", false)]);
}

/// The per-commit UKS HSE06 stress anchor's sibling (library).
#[test]
#[ignore = "validation: periodic open-shell RSH derivatives (CLI)"]
fn kpoint_uks_hse06_stress_sibling() {
    let r = stress("v-sib-kuks-s", Spin::Uks, "HSE06", false);
    let (_, s) = lib_stress(&r);
    let (_, sc) = lib_stress(&r.with(CAM, false));
    let d = max_diff(&s, &sc);
    eprintln!("UKS HSE06 vs CAM-B3LYP stress differs by {d:.2e}");
    assert!(d > 1e-8, "sibling too close: {d:e}");
}

/// The library side of the Python Gamma range-separated STRESS pins (the
/// CLI wires no Gamma stress): UKS HSE06 (dense AFT), UKS CAM-B3LYP and RKS
/// HSE06 (RS-GDF) at the fixtures, printed at full precision; each RS-GDF one
/// must differ from its dense-AFT sibling. (The Python test also checks the
/// Gamma UKS stress against the binding's own 1x1x1 k-point stress, an
/// independent construction.)
#[test]
#[ignore = "validation: periodic open-shell RSH derivatives (CLI)"]
fn gamma_rsh_stress_library_pins() {
    let gs = |tag, spin, f, rsgdf| Run {
        kpoints: false,
        ..stress(tag, spin, f, rsgdf)
    };
    let (e, s) = lib_stress(&gs("g-uks-hse", Spin::Uks, "HSE06", false));
    eprintln!("Gamma UKS HSE06 dense: E {e:?} sigma {s:?}");
    assert!(s.iter().any(|v| v.abs() > 1e-6));
    for (tag, spin, f) in [
        ("Gamma UKS CAM-B3LYP RS-GDF", Spin::Uks, CAM),
        ("Gamma RKS HSE06 RS-GDF", Spin::Rks, "HSE06"),
    ] {
        let (e, s) = lib_stress(&gs(tag, spin, f, true));
        eprintln!("{tag}: E {e:?} sigma {s:?}");
        assert!(s.iter().any(|v| v.abs() > 1e-6));
        let (_, sd) = lib_stress(&gs(tag, spin, f, false));
        let d = max_diff(&s, &sd);
        eprintln!("{tag} vs dense stress {d:.2e}");
        assert!(d > 1e-8, "{tag}: dense sibling too close: {d:e}");
    }
}

// ───────────────────────────────────── p-shell cell (STO-3G + p, ζ = 1) ──

/// The new routes on a cell whose basis carries p shells (12 functions for
/// H3): CLI == library bit for bit, with the basis guard on both sides.
#[test]
#[ignore = "validation: periodic open-shell RSH derivatives (CLI)"]
fn p_cell_kpoint_roks_hse06_force() {
    let r = Run {
        sp: true,
        ..opt("v-p-kroks-hse", Spin::Roks, "HSE06", true, false)
    };
    force_case(&r, &[(CAM, false)]);
}

#[test]
#[ignore = "validation: periodic open-shell RSH derivatives (CLI)"]
fn p_cell_kpoint_uks_hse06_stress() {
    let r = Run {
        sp: true,
        ..stress("v-p-kuks-hse-s", Spin::Uks, "HSE06", false)
    };
    stress_case(&r, &[(Spin::Uks, CAM, false)]);
}

#[test]
#[ignore = "validation: periodic open-shell RSH derivatives (CLI)"]
fn p_cell_kpoint_uks_cam_b3lyp_rsgdf_force() {
    let r = Run {
        sp: true,
        ..opt("v-p-kuks-cam-gdf", Spin::Uks, CAM, true, true)
    };
    force_case(&r, &[(CAM, false)]);
}
