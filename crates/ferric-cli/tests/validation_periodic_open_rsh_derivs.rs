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
//! Every anchor bit-equal (energy, force, stress). Siblings at the same
//! geometry (max |d|, Ha/Bohr or Ha/Bohr^3): k ROKS CAM-B3LYP RS-GDF force vs
//! HSE06 RS-GDF 5.15e-3, vs CAM-B3LYP dense 1.28e-5; Gamma ROKS CAM-B3LYP
//! dense vs HSE06 3.80e-3; Gamma RKS HSE06 RS-GDF vs dense 5.97e-6; Gamma UKS
//! CAM-B3LYP RS-GDF vs HSE06 RS-GDF 3.57e-3; the sibling test: k ROKS
//! HSE06 vs CAM-B3LYP 5.14e-3, k UKS CAM-B3LYP RS-GDF vs dense 1.30e-5, Gamma
//! ROKS HSE06 RS-GDF vs dense 3.81e-5. Stress: k ROKS CAM-B3LYP RS-GDF vs
//! dense 6.88e-7, k ROKS HSE06 vs k UKS HSE06 3.73e-5, k UKS HSE06 vs
//! CAM-B3LYP 7.98e-4; Gamma UKS CAM-B3LYP RS-GDF vs dense 3.70e-6. The whole
//! file runs in ~290 s on 3 threads.

mod open_rsh_cli;
use open_rsh_cli::*;

fn force(tag: &str, spin: Spin, f: &str, kpoints: bool, rsgdf: bool) -> (f64, Vec<f64>, Vec<f64>) {
    cli_force(&Run {
        kpoints,
        rsgdf,
        ..Run::new(tag, spin, f, "optimize")
    })
}

fn lib_force(spin: Spin, f: &str, x: &[f64], kpoints: bool, rsgdf: bool) -> (f64, Vec<f64>) {
    if kpoints {
        lib_k_force(spin, f, x, rsgdf)
    } else {
        lib_gamma_force(spin, f, x, rsgdf)
    }
}

/// CLI == library bit for bit, and the library force of `sibling` (another
/// functional and / or the other J/K) at the same geometry differs.
fn force_case(
    tag: &str,
    spin: Spin,
    f: &str,
    (kpoints, rsgdf): (bool, bool),
    sibling: (&str, bool),
) {
    let c = force(tag, spin, f, kpoints, rsgdf);
    assert_force_bit_equal(tag, &c, lib_force(spin, f, &c.1, kpoints, rsgdf));
    let (_, gs) = lib_force(spin, sibling.0, &c.1, kpoints, sibling.1);
    let d = max_diff(&c.2, &gs);
    eprintln!("{tag}: sibling {sibling:?} force differs by {d:.2e}");
    assert!(d > 1e-6, "{tag}: sibling {sibling:?} too close: {d:e}");
}

#[test]
#[ignore = "validation: periodic open-shell RSH derivatives (CLI)"]
fn kpoint_roks_cam_b3lyp_rsgdf_force() {
    let t = "v-kroks-cam-gdf";
    force_case(t, Spin::Roks, CAM, (true, true), ("HSE06", true));
    let c = force(t, Spin::Roks, CAM, true, true);
    let (_, gd) = lib_k_force(Spin::Roks, CAM, &c.1, false);
    let d = max_diff(&c.2, &gd);
    eprintln!("{t}: dense CAM-B3LYP force differs by {d:.2e}");
    assert!(d > 1e-6, "{t}: the dense sibling is too close: {d:e}");
}

#[test]
#[ignore = "validation: periodic open-shell RSH derivatives (CLI)"]
fn gamma_roks_cam_b3lyp_dense_force() {
    force_case(
        "v-groks-cam",
        Spin::Roks,
        CAM,
        (false, false),
        ("HSE06", false),
    );
}

#[test]
#[ignore = "validation: periodic open-shell RSH derivatives (CLI)"]
fn gamma_rks_hse06_rsgdf_force() {
    force_case(
        "v-grks-hse-gdf",
        Spin::Rks,
        "HSE06",
        (false, true),
        ("HSE06", false),
    );
}

#[test]
#[ignore = "validation: periodic open-shell RSH derivatives (CLI)"]
fn gamma_uks_cam_b3lyp_rsgdf_force() {
    force_case(
        "v-guks-cam-gdf",
        Spin::Uks,
        CAM,
        (false, true),
        ("HSE06", true),
    );
}

/// The siblings of the k ROKS HSE06 (per commit) and the two RS-GDF force
/// anchors below (a bit-equal pass there cannot come from a mis-routed run).
#[test]
#[ignore = "validation: periodic open-shell RSH derivatives (CLI)"]
fn per_commit_force_anchors_differ_from_their_siblings() {
    for (tag, spin, f, kpoints, rsgdf, sibling) in [
        (
            "v-sib-kroks",
            Spin::Roks,
            "HSE06",
            true,
            false,
            (CAM, false),
        ),
        ("v-sib-kuks-gdf", Spin::Uks, CAM, true, true, (CAM, false)),
        (
            "v-sib-groks-gdf",
            Spin::Roks,
            "HSE06",
            false,
            true,
            ("HSE06", false),
        ),
    ] {
        let (_, x, g) = force(tag, spin, f, kpoints, rsgdf);
        let (_, gs) = lib_force(spin, sibling.0, &x, kpoints, sibling.1);
        let d = max_diff(&g, &gs);
        eprintln!("{tag}: sibling {sibling:?} force differs by {d:.2e}");
        assert!(d > 1e-6, "{tag}: sibling {sibling:?} too close: {d:e}");
    }
}

/// Moved from the per-commit file (runtime): the Gamma ROKS force on RS-GDF
/// (G3 + G5) and the k-point UKS force on RS-GDF (G1).
#[test]
#[ignore = "validation: periodic open-shell RSH derivatives (CLI)"]
fn gamma_roks_hse06_rsgdf_force_equals_the_library_bit_for_bit() {
    let r = Run {
        kpoints: false,
        rsgdf: true,
        ..Run::new("groks-hse-gdf-f", Spin::Roks, "HSE06", "optimize")
    };
    let c = cli_force(&r);
    assert_force_bit_equal(r.tag, &c, lib_gamma_force(Spin::Roks, "HSE06", &c.1, true));
}

#[test]
#[ignore = "validation: periodic open-shell RSH derivatives (CLI)"]
fn kpoint_uks_cam_b3lyp_rsgdf_force_equals_the_library_bit_for_bit() {
    let r = Run {
        rsgdf: true,
        ..Run::new("kuks-cam-gdf-f", Spin::Uks, CAM, "optimize")
    };
    let c = cli_force(&r);
    assert_force_bit_equal(r.tag, &c, lib_k_force(Spin::Uks, CAM, &c.1, true));
}

fn stress_case(tag: &str, spin: Spin, f: &str, rsgdf: bool) -> Vec<f64> {
    let c = cli_stress(&Run {
        rsgdf,
        ..Run::new(tag, spin, f, "stress")
    });
    assert_stress_bit_equal(tag, &c, lib_k_stress(spin, f, rsgdf));
    c.1
}

#[test]
#[ignore = "validation: periodic open-shell RSH derivatives (CLI)"]
fn kpoint_roks_cam_b3lyp_rsgdf_stress() {
    let s = stress_case("v-kroks-cam-gdf-s", Spin::Roks, CAM, true);
    let (_, sd) = lib_k_stress(Spin::Roks, CAM, false);
    let d = max_diff(&s, &sd);
    eprintln!("v-kroks-cam-gdf-s: dense CAM-B3LYP stress differs by {d:.2e}");
    assert!(d > 1e-8, "dense sibling too close: {d:e}");
}

#[test]
#[ignore = "validation: periodic open-shell RSH derivatives (CLI)"]
fn kpoint_uks_cam_b3lyp_rsgdf_stress() {
    stress_case("v-kuks-cam-gdf-s", Spin::Uks, CAM, true);
}

#[test]
#[ignore = "validation: periodic open-shell RSH derivatives (CLI)"]
fn kpoint_roks_hse06_dense_stress() {
    let s = stress_case("v-kroks-hse-s", Spin::Roks, "HSE06", false);
    let (_, su) = lib_k_stress(Spin::Uks, "HSE06", false);
    let d = max_diff(&s, &su);
    eprintln!("v-kroks-hse-s: UKS HSE06 stress differs by {d:.2e}");
    assert!(d > 1e-8, "UKS sibling too close: {d:e}");
}

/// The per-commit UKS HSE06 stress anchor's sibling.
#[test]
#[ignore = "validation: periodic open-shell RSH derivatives (CLI)"]
fn per_commit_stress_anchor_differs_from_its_sibling() {
    let (_, s) = lib_k_stress(Spin::Uks, "HSE06", false);
    let (_, sc) = lib_k_stress(Spin::Uks, CAM, false);
    let d = max_diff(&s, &sc);
    eprintln!("UKS HSE06 vs CAM-B3LYP stress differs by {d:.2e}");
    assert!(d > 1e-8, "sibling too close: {d:e}");
}

/// The library side of the Python Gamma UKS range-separated STRESS pins
/// (`run_uks_gamma(with_stress=True)`, issue #498; the CLI wires no Gamma
/// stress): dense-AFT HSE06 and RS-GDF CAM-B3LYP at the H3 fixture,
/// printed at full precision; the RS-GDF one must differ from its dense-AFT
/// sibling. (The Python test also checks the Gamma stress against the
/// binding's own 1x1x1 k-point stress, an independent construction.)
#[test]
#[ignore = "validation: periodic open-shell RSH derivatives (CLI)"]
fn gamma_uks_rsh_stress_library_pins() {
    let (e, s) = lib_gamma_uks_stress("HSE06", false);
    eprintln!("Gamma UKS HSE06 dense: E {e:?} sigma {s:?}");
    assert!(s.iter().any(|v| v.abs() > 1e-6));
    let (e2, s2) = lib_gamma_uks_stress(CAM, true);
    eprintln!("Gamma UKS CAM-B3LYP RS-GDF: E {e2:?} sigma {s2:?}");
    let (_, s2d) = lib_gamma_uks_stress(CAM, false);
    let d = max_diff(&s2, &s2d);
    eprintln!("Gamma UKS CAM-B3LYP RS-GDF vs dense stress {d:.2e}");
    assert!(d > 1e-8, "dense sibling too close: {d:e}");
}
