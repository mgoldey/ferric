//! Weekly half of `periodic_kpoint_ks_optimize.rs` (k-point closed-shell
//! Kohn-Sham `task = "optimize"` wired to `kpoint_rks_gradient`): the three
//! anchors that took more than 120 s on CI (runs 40957025137 / 40953725119,
//! mean 139-158 s): the PBE0 1x1x3 anchor with its mesh, functional, cutoff
//! and exxdiv mutants; the real 30-step PBE0 optimization of stretched H2;
//! and the k-point meta-GGA force (SCAN dense 1x1x3, r2SCAN RS-GDF, TPSS
//! dense) with the GGA-parent mutant. Per-commit canaries:
//! `printed_force_equals_the_library_force_for_a_global_hybrid_canary`,
//! `printed_force_equals_the_library_force_for_scan_canary` (both in
//! `periodic_kpoint_ks_optimize.rs`) and, for the optimizer loop,
//! `periodic_kpoint_optimize.rs::k_point_optimize_converges_and_the_final_force_is_the_library_force`.
//! Shared harness: `kpoint_ks_optimize_cli/mod.rs`.

mod kpoint_ks_optimize_cli;
use kpoint_ks_optimize_cli::*;

#[test]
#[ignore = "validation: k-point KS PBE0 force CLI == library + mutants; CI ~139 s"]
fn printed_force_equals_the_library_force_for_a_global_hybrid_with_mutants() {
    let o = run(&loose_input("pbe0", "PBE0", "[1, 1, 3]", DENSE));
    let out = ok(&o, "pbe0");
    assert!(out.contains("analytic k-point RKS gradient"), "{out}");
    let g = printed_gradient(&out, 2);
    let at = printed_atoms(&out, 2);
    let lib = Lib {
        functional: "PBE0",
        mesh: [1, 1, 3],
        exxdiv_scf: ExxDiv::Ewald,
        rsgdf: false,
        cutoff: true,
    };
    assert!(
        g.iter().fold(0.0_f64, |m, v| m.max(v.abs())) > 1e-3,
        "trivial"
    );
    let d = max_diff(&g, &library_gradient(&at, lib, ExxDiv::Ewald));
    eprintln!("PBE0 CLI vs library force: {d:.2e}");
    assert!(d < BAR, "{d:e}");
    // MUTANT: a wrong mesh moves the force by far more than the bar.
    let bad = library_gradient(
        &at,
        Lib {
            mesh: [1, 1, 1],
            ..lib
        },
        ExxDiv::Ewald,
    );
    assert!(max_diff(&g, &bad) > 1e-4, "mesh mutant not detected");
    // MUTANT: the pure parent functional.
    let bad = library_gradient(
        &at,
        Lib {
            functional: "PBE",
            ..lib
        },
        ExxDiv::Ewald,
    );
    assert!(max_diff(&g, &bad) > 1e-4, "functional mutant not detected");
    // MUTANT: the explicit neighbour cutoff dropped (the default is not 14
    // Bohr): ~3e-6, far above the bar that BAR sets.
    let bad = library_gradient(
        &at,
        Lib {
            cutoff: false,
            ..lib
        },
        ExxDiv::Ewald,
    );
    let miss = max_diff(&g, &bad);
    eprintln!("cutoff mutant miss: {miss:.2e}");
    assert!(miss > 10.0 * BAR, "cutoff mutant not detected: {miss:e}");
    // ARTIFACT HYPOTHESIS: for a closed-shell system the exxdiv shift is uniform
    // on the occupied space, so it moves the energy by a constant and leaves
    // the density and the force unchanged: an SCF run with exxdiv = none gives
    // the SAME force (an identity, asserted), and the force cannot tell the
    // two apart. What guards the wiring is the library's Fock-consistency
    // check: a force handed an exxdiv the SCF did not run is refused.
    let same = library_gradient(
        &at,
        Lib {
            exxdiv_scf: ExxDiv::None,
            ..lib
        },
        ExxDiv::None,
    );
    assert!(max_diff(&g, &same) < BAR, "exxdiv identity violated");
    let e =
        try_library_gradient(&at, lib, ExxDiv::None).expect_err("a wrong exxdiv must be refused");
    assert!(e.contains("exxdiv"), "{e}");
}

#[test]
#[ignore = "validation: k-point KS PBE0 optimization converges, final force == library; CI ~150 s"]
fn a_real_optimization_converges_and_the_final_force_is_the_library_force() {
    let b = BOHR_RADIUS_ANGSTROM;
    let start = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0 / b]];
    let o = run(&Input {
        tag: "opt",
        xyz: xyz(&start),
        mult: 1,
        basis: "sto-3g",
        functional: "PBE0",
        kmesh: "[1, 1, 3]",
        jk: DENSE,
        optimize: "max_steps = 30",
    });
    let out = ok(&o, "opt");
    assert_eq!(value(&out, "converged"), "true", "{out}");
    let atoms = printed_atoms(&out, 2);
    let bond = ((atoms[1][2] - atoms[0][2]).powi(2)
        + (atoms[1][1] - atoms[0][1]).powi(2)
        + (atoms[1][0] - atoms[0][0]).powi(2))
    .sqrt();
    eprintln!("optimized bond {bond:.5} Bohr");
    assert!(
        bond < 1.0 / b - 0.05,
        "stretched H2 should contract: {bond}"
    );
    let g = printed_gradient(&out, 2);
    assert!(g.iter().fold(0.0_f64, |m, v| m.max(v.abs())) < 4.5e-4);
    let lib = Lib {
        functional: "PBE0",
        mesh: [1, 1, 3],
        exxdiv_scf: ExxDiv::Ewald,
        rsgdf: false,
        cutoff: true,
    };
    let d = max_diff(&g, &library_gradient(&atoms, lib, ExxDiv::Ewald));
    assert!(d < BAR, "{d:e}");
}

/// Pure meta-GGAs (SCAN, r2SCAN, TPSS) pass the energy and force gates: the
/// printed final force equals `kpoint_rks_gradient` on the same cell, dense
/// and RS-GDF.
#[test]
#[ignore = "validation: k-point KS meta-GGA force CLI == library (SCAN, r2SCAN, TPSS); CI ~158 s"]
fn printed_force_equals_the_library_force_for_meta_gga() {
    for (tag, f, kmesh, jk, rsgdf) in [
        ("scan", "SCAN", "[1, 1, 3]", DENSE, false),
        ("r2scan", "r2SCAN", "[1, 1, 2]", RSGDF, true),
        ("tpss", "TPSS", "[1, 1, 2]", DENSE, false),
    ] {
        let o = run(&loose_input(tag, f, kmesh, jk));
        let out = ok(&o, tag);
        assert!(out.contains("analytic k-point RKS gradient"), "{out}");
        let g = printed_gradient(&out, 2);
        let at = printed_atoms(&out, 2);
        let lib = Lib {
            functional: f,
            mesh: match kmesh {
                "[1, 1, 3]" => [1, 1, 3],
                _ => [1, 1, 2],
            },
            exxdiv_scf: ExxDiv::Ewald,
            rsgdf,
            cutoff: true,
        };
        assert!(
            g.iter().fold(0.0_f64, |m, v| m.max(v.abs())) > 1e-3,
            "trivial"
        );
        let d = max_diff(&g, &library_gradient(&at, lib, ExxDiv::Ewald));
        eprintln!("{f} CLI vs library force: {d:.2e}");
        assert!(d < BAR, "{f}: {d:e}");
        // MUTANT: the GGA parent is a different functional.
        let bad = library_gradient(
            &at,
            Lib {
                functional: "PBE",
                ..lib
            },
            ExxDiv::Ewald,
        );
        assert!(
            max_diff(&g, &bad) > 1e-4,
            "{f}: functional mutant not detected"
        );
    }
}
