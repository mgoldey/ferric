//! k-point open-shell routes at `[cell] kmesh`: UKS (`uhf` / `ksdft` with a
//! functional), ROKS (`rohf` + `[dft] functional`) and ROHF, energies and
//! `task = "optimize"` forces, wired to `solve_kuks`/`solve_kroks`/
//! `solve_krohf` and `kpoint_{uks,roks,rohf}_gradient`.
//!
//! * EXACTNESS ANCHOR (CLI vs library): the printed total energy equals the
//!   library SCF energy of the same cell, mesh, grid and configs (bar 5e-10),
//!   and the printed final gradient of a loose optimize run (stopped after
//!   three steps) equals the library force at the printed 8-decimal final
//!   geometry (bar 5e-8). Both are anchored on an H3 doublet in a 4 Bohr cubic
//!   cell (1x1x2 Gamma-centred mesh, STO-3G, dense AFT J/K, exxdiv ewald).
//! * Mutants of the wiring itself are in the report of the commit that adds
//!   this file; the in-test mutants assert that the library force of a
//!   DIFFERENT method, mesh or functional misses the printed force by far more
//!   than the bar, i.e. the anchors could tell a mis-routed run apart.
//! * Open-shell SCAN and HSE06 ENERGIES (issue #357): the CLI energy, read
//!   at full precision from the JSON run log, equals the library `solve_kuks`
//!   (Gamma-centred and Monkhorst-Pack 1x1x2) and, with no kmesh, the
//!   library `gamma_uks_any` to 1e-12 (the run log holds exactly the library
//!   value; see [`SAME_ARITHMETIC`]); the Gamma HSE06 run also equals the CLI
//!   1x1x1 run to SCF precision.
//!   Mutation ledger (run 2026-10-18, then reverted): the CLI Gamma UKS with
//!   `exxdiv = none` forced into its config: KILLED (CLI -1.3319 vs library
//!   -1.5513); the CLI Gamma UKS with a direct Ewald start and density_conv
//!   1e-6: SURVIVED (moved E by 2.2e-16; this 3-orbital cell converges to the
//!   same state either way, an equivalent mutant here). In-test mutants:
//!   the sibling global functional misses by >= 2.3e-4 (HSE06 vs PBE0) /
//!   2.5e-2 (SCAN vs PBE), the other centring by >= 1.7e-2.
//! * Refusals with exact wording: ROKS meta-GGA, open-shell HSE06 forces, UKS
//!   meta-GGA on RS-GDF and at Gamma.
//!
//! TIERS. The Monkhorst-Pack energy anchor over all three drivers (123 s on
//! CI) runs weekly in `validation_periodic_kpoint_open_shell.rs`. Per commit:
//! the UKS Monkhorst-Pack energies are anchored bit-tight by
//! `cli_open_shell_scan_and_hse06_equal_library_solve_kuks` (with the
//! centring mutant), and the ROHF one by
//! `printed_energy_equals_the_library_energy_on_a_monkhorst_pack_mesh_rohf_canary`.
//! Shared harness: `kpoint_open_shell_cli/mod.rs`.

mod kpoint_open_shell_cli;
use kpoint_open_shell_cli::*;

#[test]
fn printed_force_equals_the_library_force_for_uks_with_mutants() {
    let (g, at) = force_anchor("uks", Method::Uks, "PBE", "UKS");
    let lib = Lib {
        method: Method::Uks,
        functional: "PBE",
        mesh: [1, 1, 2],
        centring: MeshCentring::Gamma,
    };
    // MUTANT: a different method (the ROKS force of the same functional) is
    // not the UKS force; the anchor must be able to tell them apart.
    let (_, roks) = library(
        &at,
        Lib {
            method: Method::Roks,
            ..lib
        },
        ExxDiv::Ewald,
    );
    assert!(max_diff(&g, &roks) > 1e-5, "method mutant not detected");
    // MUTANT: wrong mesh, wrong functional.
    let (_, bad) = library(
        &at,
        Lib {
            mesh: [1, 1, 1],
            ..lib
        },
        ExxDiv::Ewald,
    );
    assert!(max_diff(&g, &bad) > 1e-4, "mesh mutant not detected");
    let (_, bad) = library(
        &at,
        Lib {
            functional: "LDA",
            ..lib
        },
        ExxDiv::Ewald,
    );
    assert!(max_diff(&g, &bad) > 1e-4, "functional mutant not detected");
}

#[test]
fn printed_force_equals_the_library_force_for_roks() {
    force_anchor("roks", Method::Roks, "PBE", "ROKS");
}

#[test]
fn printed_force_equals_the_library_force_for_rohf() {
    force_anchor("rohf", Method::Rohf, "", "ROHF");
}

#[test]
fn printed_force_equals_the_library_force_for_a_uks_global_hybrid() {
    force_anchor("uks-pbe0", Method::Uks, "PBE0", "UKS");
}

#[test]
fn printed_energy_equals_the_library_energy_for_every_open_shell_route() {
    for (tag, method, f, key) in [
        ("e-uks", Method::Uks, "PBE", "E_xc"),
        ("e-roks", Method::Roks, "PBE", "E_xc"),
        ("e-rohf", Method::Rohf, "", "<S^2>"),
    ] {
        let out = ok(&run(&input(tag, method, f, "energy")), tag);
        assert_eq!(value(&out, "converged"), "true", "{out}");
        let e: f64 = value(&out, "energy").parse().unwrap();
        let (el, _) = library(
            &H3,
            Lib {
                method,
                functional: f,
                mesh: [1, 1, 2],
                centring: MeshCentring::Gamma,
            },
            ExxDiv::Ewald,
        );
        eprintln!(
            "{tag}: CLI energy {e:.10} vs library {el:.10}: {:.2e}",
            (e - el).abs()
        );
        assert!((e - el).abs() < E_BAR, "{tag}: {e} vs {el}");
        // The route's own printout (KS routes print E_xc, ROHF does not).
        assert!(out.contains(key), "{tag}: {out}");
        assert_eq!(out.contains("E_xc"), method != Method::Rohf, "{tag}: {out}");
        assert!(
            out.contains("<S^2>") && out.contains("gaps"),
            "{tag}: {out}"
        );
    }
}

/// CANARY of the weekly Monkhorst-Pack anchor (UKS, ROKS, ROHF): the ROHF
/// driver on the half-shifted 1x1x2 mesh, with the centring mutant.
#[test]
fn printed_energy_equals_the_library_energy_on_a_monkhorst_pack_mesh_rohf_canary() {
    mp_mesh_energy_anchor("mp-rohf-c", Method::Rohf, "");
}

/// `run_end.energy` from the JSON run log of a `task = "energy"` input, as
/// the exact f64 (the stdout line has 10 decimals).
fn logged_energy(tag: &str) -> f64 {
    let log = std::fs::read_to_string(scratch(tag).join("run.jsonl")).expect("run log");
    for line in log.lines() {
        let v: serde_json::Value = serde_json::from_str(line).unwrap();
        if v.get("record").and_then(|r| r.as_str()) == Some("run_end") {
            return v["energy"]
                .as_f64()
                .unwrap_or_else(|| panic!("no energy in {line}"));
        }
    }
    panic!("no run_end record in:\n{log}");
}

/// The H3 doublet cell of [`library`] (same xyz round trip as the CLI).
fn h3_cell() -> (Cell, PreparedBasis) {
    // One file per call: the tests of this binary run in parallel.
    static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let p = scratch("lib-e").join(format!("l{n}.xyz"));
    std::fs::write(&p, xyz(&H3)).unwrap();
    let mol = Molecule::load_xyz_with_charge(p.to_str().unwrap(), 0, 2).expect("xyz");
    let lat = [[A_BOHR, 0.0, 0.0], [0.0, A_BOHR, 0.0], [0.0, 0.0, A_BOHR]];
    let cell = Cell::new(mol, lat).expect("cell");
    let prep = PreparedBasis::new(cell.mol(), &basis::bundled("sto-3g").unwrap()).unwrap();
    (cell, prep)
}

/// The CLI's dense-AFT cap (`[cell] max_eri_gb` default 0.5).
fn cli_max_eri_bytes() -> usize {
    ferric_core::memory::gib_to_bytes(0.5)
}

/// Library k-point UKS energy on the dense-AFT J/K with the CLI's configs
/// restated field by field (`crate::periodic::kuks_config`), so the two runs
/// are the same arithmetic.
fn library_kuks_energy(functional: &str, mesh: [usize; 3], centring: MeshCentring) -> f64 {
    let (cell, prep) = h3_cell();
    let mesh = KPointMesh::new(&cell, mesh, centring).unwrap();
    let mut c = KUksConfig::new(&cell, ExxDiv::Ewald, functional);
    c.kuhf.scf = KScfConfig {
        max_iter: 200,
        energy_conv: 1e-12,
        grad_conv: 1e-9,
        ..Default::default()
    };
    c.kuhf.hcore =
        PeriodicHcoreConfig::with_omega(OMEGA).with_sr_column_rotation(SrColumnRotation::Off);
    c.kuhf.dense = KDenseAftConfig {
        max_bytes: cli_max_eri_bytes(),
        ..Default::default()
    };
    c.grid = grid();
    solve_kuks(&cell, &prep, None, &mesh, &c)
        .unwrap_or_else(|e| panic!("solve_kuks {functional}: {e}"))
        .scf
        .energy
}

/// Library Gamma UKS energy (dense AFT) through the bindings' shared entry
/// `gamma_uks_any`, with the CLI's Gamma configs restated.
fn library_gamma_uks_energy(functional: &str) -> f64 {
    let (cell, prep) = h3_cell();
    let hcfg = PeriodicHcoreConfig {
        sr_column_rotation: SrColumnRotation::Off,
        ..PeriodicHcoreConfig::with_omega(OMEGA)
    };
    let hc = periodic_hcore(&cell, &prep, &hcfg).unwrap();
    let eri = DenseAftEri::build(
        &cell,
        &prep,
        &hc.s,
        ExxDiv::Ewald,
        DEFAULT_DENSE_AFT_PRECISION,
        cli_max_eri_bytes(),
    )
    .unwrap();
    let mut c = GammaUksConfig::new(functional);
    (c.grid, c.exxdiv) = (grid(), ExxDiv::Ewald);
    c.scf = RhfConfig {
        use_sad_guess: false,
        density_conv: 1e-10,
        max_iter: 200,
        ..Default::default()
    };
    let sizing = GammaRshSizing {
        rsgdf_omega: DEFAULT_RSGDF_OMEGA,
        budget_bytes: None,
        max_eri_bytes: cli_max_eri_bytes(),
        range_split: false,
    };
    gamma_uks_any(
        &cell,
        &prep,
        &hc,
        GammaUhfIntegrals::DenseAft(&eri),
        None,
        sizing,
        &c,
    )
    .unwrap_or_else(|e| panic!("gamma_uks_any {functional}: {e}"))
    .scf
    .energy
}

fn energy_input<'a>(tag: &'a str, functional: &'a str, kmesh: &'a str) -> Input<'a> {
    Input {
        kmesh,
        ..input(tag, Method::Uks, functional, "energy")
    }
}

/// CLI vs library bar for the same arithmetic in two processes. The run log
/// holds exactly the library value (k-point HSE06: -1.6224626688847699 in the
/// file on every run); the test reads it back through serde_json, which is
/// built without `float_roundtrip` and parses that literal 1 ulp off
/// (-1.62246266888477, deterministically). `==` would fail on that parse
/// artifact, so the bar is 1e-12, far below every mutant (>= 2.3e-4).
const SAME_ARITHMETIC: f64 = 1e-12;

/// Open-shell SCAN and HSE06 (issue #357; library PRs #479 / #480): the CLI
/// UKS energy on a k-mesh (Gamma-centred 1x1x2 and Monkhorst-Pack 1x1x2) is
/// the library `solve_kuks` energy (read at full precision from the JSON run
/// log) to [`SAME_ARITHMETIC`]. MUTANTS the anchor must tell apart: the global
/// functional of the same family (PBE for SCAN, PBE0 for HSE06) and the other
/// mesh centring.
#[test]
fn cli_open_shell_scan_and_hse06_equal_library_solve_kuks() {
    for (f, sibling) in [("SCAN", "PBE"), ("HSE06", "PBE0")] {
        for (kmesh, centring, ctag) in [
            ("[1, 1, 2]", MeshCentring::Gamma, "g"),
            (
                "[1, 1, 2]\ncentring = \"mp\"",
                MeshCentring::MonkhorstPack,
                "mp",
            ),
        ] {
            let tag = format!("bits-{f}-{ctag}");
            let out = ok(&run(&energy_input(&tag, f, kmesh)), &tag);
            assert_eq!(value(&out, "converged"), "true", "{out}");
            let cli = logged_energy(&tag);
            let lib = library_kuks_energy(f, [1, 1, 2], centring);
            eprintln!("{tag}: CLI {cli:?} library {lib:?}");
            assert!(
                (cli - lib).abs() <= SAME_ARITHMETIC,
                "{tag}: {cli:?} vs {lib:?}"
            );
            let other = match centring {
                MeshCentring::Gamma => MeshCentring::MonkhorstPack,
                MeshCentring::MonkhorstPack => MeshCentring::Gamma,
            };
            let d_mesh = (cli - library_kuks_energy(f, [1, 1, 2], other)).abs();
            let d_f = (cli - library_kuks_energy(sibling, [1, 1, 2], centring)).abs();
            eprintln!("{tag}: centring mutant {d_mesh:.2e}, {sibling} mutant {d_f:.2e}");
            assert!(
                d_mesh > 1e-6 && d_f > 1e-4,
                "{tag}: a mutant was not detected"
            );
        }
    }
}

/// Gamma-point UKS HSE06 (no kmesh): the CLI energy is the library
/// `gamma_uks_any` (Gamma J/XC/SCF + the k = 0 attenuated K) to
/// [`SAME_ARITHMETIC`] (measured 0), and
/// agrees with the CLI's 1x1x1 k-point UKS (a different J, XC and SCF code
/// over the same cell) to SCF precision. MUTANT: PBE0 on the same cell.
#[test]
fn cli_gamma_uks_hse06_equals_library_and_the_one_point_mesh() {
    let out = ok(&run(&energy_input("gam-hse", "HSE06", "")), "gam-hse");
    assert_eq!(value(&out, "converged"), "true", "{out}");
    let cli = logged_energy("gam-hse");
    let lib = library_gamma_uks_energy("HSE06");
    eprintln!("gam-hse: CLI {cli:?} library {lib:?}");
    assert!((cli - lib).abs() <= SAME_ARITHMETIC, "{cli:?} vs {lib:?}");
    ok(
        &run(&energy_input("k111-hse", "HSE06", "[1, 1, 1]")),
        "k111-hse",
    );
    let k111 = logged_energy("k111-hse");
    eprintln!("Gamma - 1x1x1: {:.2e}", cli - k111);
    assert!((cli - k111).abs() < 1e-8, "{cli:.12} vs {k111:.12}");
    let pbe0 = library_gamma_uks_energy("PBE0");
    assert!((cli - pbe0).abs() > 1e-4, "PBE0 mutant not detected");
}

/// HSE06 on the RS-GDF J/K at a k-mesh: the CLI energy is the library
/// `solve_kuks` with the same aux, within the SR-column-rotation screening
/// difference of the CLI's fit (bar 1e-8, as the closed-shell RS-GDF anchor),
/// and differs from the dense-AFT energy by the fit error only.
#[test]
fn cli_open_shell_hse06_rsgdf_equals_library_solve_kuks() {
    let mut i = energy_input("rsgdf-hse", "HSE06", "[1, 1, 2]");
    i.jk = "jk = \"rsgdf\"\nauxbasis = \"cc-pvdz-ri\"";
    ok(&run(&i), "rsgdf-hse");
    let cli = logged_energy("rsgdf-hse");
    let (cell, prep) = h3_cell();
    let aux = PreparedBasis::new(cell.mol(), &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let mesh = KPointMesh::new(&cell, [1, 1, 2], MeshCentring::Gamma).unwrap();
    let mut c = KUksConfig::new(&cell, ExxDiv::Ewald, "HSE06");
    c.kuhf.jk = KJkKind::RsGdf;
    c.kuhf.scf.max_iter = 200;
    c.kuhf.hcore = PeriodicHcoreConfig::with_omega(OMEGA);
    c.grid = grid();
    let lib = solve_kuks(&cell, &prep, Some(&aux), &mesh, &c)
        .unwrap()
        .scf
        .energy;
    let dense = library_kuks_energy("HSE06", [1, 1, 2], MeshCentring::Gamma);
    eprintln!(
        "rsgdf-hse: CLI - library {:.2e}, CLI - dense {:.2e}",
        cli - lib,
        cli - dense
    );
    assert!((cli - lib).abs() < 1e-8, "{cli:.12} vs {lib:.12}");
    assert!((cli - dense).abs() < 1e-3, "{cli:.12} vs dense {dense:.12}");
}

#[test]
fn open_shell_refusals_have_exact_messages() {
    // k-point ROKS SCAN runs as an energy on the dense-AFT J/K (#498,
    // periodic_open_mgga_energy.rs); RS-GDF and its force refuse at plan time.
    let tag = "roks-SCAN-rsgdf";
    let mut i = input(tag, Method::Roks, "SCAN", "energy");
    i.jk = "jk = \"rsgdf\"\nauxbasis = \"cc-pvdz-ri\"";
    let e = refused(&run(&i), tag);
    assert!(
        e.contains("[cell] kmesh with functional \"SCAN\" and jk = \"rsgdf\"")
            && e.contains("the k-point ROKS SCAN / r2SCAN energy is validated on the dense-AFT"),
        "{tag}: {e}"
    );
    let tag = "roks-opt-SCAN";
    let e = refused(&run(&input(tag, Method::Roks, "SCAN", "optimize")), tag);
    assert!(
        e.contains("method.task = \"optimize\" with [cell] kmesh")
            && e.contains("SCAN: the k-point ROKS meta-GGA force and stress"),
        "{tag}: {e}"
    );
    // ROKS wB97X has an energy here but no validated force: optimize is
    // refused by the derivative allowlist, before any integral (the ROKS and
    // UKS HSE06 / CAM-B3LYP forces run: periodic_rsh_open_mgga_derivs.rs).
    let tag = "roks-opt-wB97X";
    let wb = "HYB_GGA_XC_WB97X";
    let e = refused(&run(&input(tag, Method::Roks, wb, "optimize")), tag);
    assert!(
        e.contains("method.task = \"optimize\" with [cell] kmesh")
            && e.contains(
                "HYB_GGA_XC_WB97X: the range-separated hybrid force and stress are validated \
                 for HSE06 / CAM-B3LYP (HYB_GGA_XC_CAM_B3LYP) only"
            ),
        "{tag}: {e}"
    );
    // The open-shell meta-GGA force and stress need the dense-AFT J/K (the
    // RS-GDF energy runs, #498).
    let mut i = input("uks-scan-rsgdf", Method::Uks, "SCAN", "optimize");
    i.jk = "jk = \"rsgdf\"\nauxbasis = \"cc-pvdz-ri\"";
    let e = refused(&run(&i), "uks-scan-rsgdf");
    assert!(
        e.contains("method.task = \"optimize\" with [cell] kmesh")
            && e.contains("run on the dense-AFT J/K only"),
        "{e}"
    );
    // Gamma-point UKS / ROKS meta-GGA: the energy runs (#498); the force is
    // refused by name, pointing at the 1x1x1 mesh that has it.
    for (tag, method) in [
        ("gam-uks-scan-opt", Method::Uks),
        ("gam-roks-scan-opt", Method::Roks),
    ] {
        let i = Input {
            kmesh: "",
            ..input(tag, method, "SCAN", "optimize")
        };
        let e = refused(&run(&i), tag);
        assert!(
            e.contains("method.task = \"optimize\" with [cell]: ")
                && e.contains("SCAN / r2SCAN force and stress are not implemented")
                && e.contains("1x1x1 mesh"),
            "{tag}: {e}"
        )
    }
}
