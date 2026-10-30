//! `task = "optimize"` with `[cell] kmesh` and closed-shell Kohn-Sham (LDA,
//! GGA, global hybrids) is wired to `ferric_pbc::kxcgrad::kpoint_rks_gradient`.
//!
//! * EXACTNESS ANCHOR (CLI vs library): with loose optimizer thresholds the
//!   run stops after a few steps; the printed final gradient is the force at
//!   the printed final geometry, which must equal `kpoint_rks_gradient` called
//!   directly on the same cell, mesh, grid and configs (evaluated at the
//!   printed 8-decimal geometry, bar 5e-8). Mutants: a wrong mesh, a wrong
//!   functional and a wrong exxdiv each move the library force by far more
//!   than the bar (asserted).
//! * A real optimization (stretched H2, PBE0, 1x1x3) converges to the
//!   optimizer's thresholds, contracts the bond, and the printed final
//!   gradient equals the library force at the printed final geometry.
//! * Refusals with exact wording: meta-GGA, range-separated hybrid on RS-GDF,
//!   open-shell meta-GGA on RS-GDF, range_split, `gdf_omega = "auto"`.
//! * Gamma point (no kmesh), meta-GGA (issue #492): the printed final force of
//!   a TPSS optimize equals the routed library force
//!   (`gamma_mgga::gamma_rks_mgga_gradient` on `gamma_rks`), same bar; a
//!   meta-GGA optimize on `jk = "rsgdf"` is refused before any SCF, and the
//!   pre-SCF dense cap counts the routed force's 1x1x1 kernels. A k-point
//!   TPSS optimize on `jk = "rsgdf"` is refused by name.
//!
//! The explicit `neighbour_cutoff` is shared by the CLI input and the library
//! grid: the XC energy has a hard cutoff and the default depends on the cell.
//!
//! TIERS. Three anchors above 120 s on CI run weekly in
//! `validation_periodic_kpoint_ks_optimize.rs`: the PBE0 1x1x3 anchor with
//! its mesh / functional / cutoff / exxdiv mutants, the real 30-step PBE0
//! optimization, and the three-functional k-point meta-GGA force. Per-commit
//! canaries here: `printed_force_equals_the_library_force_for_a_global_hybrid_canary`
//! (PBE0 dense 1x1x2: CLI == library and the exxdiv Fock guard),
//! `printed_force_equals_the_library_force_for_scan_canary` (SCAN dense
//! 1x1x2) and, for the optimizer loop, the k-point HF
//! `periodic_kpoint_optimize.rs::k_point_optimize_converges_and_the_final_force_is_the_library_force`.
//! Shared harness: `kpoint_ks_optimize_cli/mod.rs`.

mod kpoint_ks_optimize_cli;
use kpoint_ks_optimize_cli::*;

/// CANARY of the weekly PBE0 1x1x3 anchor with mutants: a dense PBE0 force
/// on the 1x1x2 mesh equals the library force at the printed geometry, and
/// the library's Fock-consistency guard refuses a force handed an exxdiv the
/// SCF did not run (the guard that protects the exxdiv wiring).
#[test]
fn printed_force_equals_the_library_force_for_a_global_hybrid_canary() {
    let out = ok(
        &run(&loose_input("pbe0-c", "PBE0", "[1, 1, 2]", DENSE)),
        "pbe0-c",
    );
    assert!(out.contains("analytic k-point RKS gradient"), "{out}");
    let g = printed_gradient(&out, 2);
    let at = printed_atoms(&out, 2);
    let lib = Lib {
        functional: "PBE0",
        mesh: [1, 1, 2],
        exxdiv_scf: ExxDiv::Ewald,
        rsgdf: false,
        cutoff: true,
    };
    assert!(
        g.iter().fold(0.0_f64, |m, v| m.max(v.abs())) > 1e-3,
        "trivial"
    );
    let d = max_diff(&g, &library_gradient(&at, lib, ExxDiv::Ewald));
    eprintln!("PBE0 1x1x2 CLI vs library force: {d:.2e}");
    assert!(d < BAR, "{d:e}");
    let e =
        try_library_gradient(&at, lib, ExxDiv::None).expect_err("a wrong exxdiv must be refused");
    assert!(e.contains("exxdiv"), "{e}");
}

/// CANARY of the weekly k-point meta-GGA force anchor (SCAN 1x1x3, r2SCAN
/// RS-GDF, TPSS): the SCAN dense force on the 1x1x2 mesh equals the library
/// force at the printed geometry.
#[test]
fn printed_force_equals_the_library_force_for_scan_canary() {
    let out = ok(
        &run(&loose_input("scan-c", "SCAN", "[1, 1, 2]", DENSE)),
        "scan-c",
    );
    assert!(out.contains("analytic k-point RKS gradient"), "{out}");
    let g = printed_gradient(&out, 2);
    let at = printed_atoms(&out, 2);
    let lib = Lib {
        functional: "SCAN",
        mesh: [1, 1, 2],
        exxdiv_scf: ExxDiv::Ewald,
        rsgdf: false,
        cutoff: true,
    };
    assert!(
        g.iter().fold(0.0_f64, |m, v| m.max(v.abs())) > 1e-3,
        "trivial"
    );
    let d = max_diff(&g, &library_gradient(&at, lib, ExxDiv::Ewald));
    eprintln!("SCAN 1x1x2 CLI vs library force: {d:.2e}");
    assert!(d < BAR, "{d:e}");
}

#[test]
fn printed_force_equals_the_library_force_for_a_gga() {
    let o = run(&loose_input("pbe", "PBE", "[1, 1, 3]", DENSE));
    let out = ok(&o, "pbe");
    let g = printed_gradient(&out, 2);
    let at = printed_atoms(&out, 2);
    let lib = Lib {
        functional: "PBE",
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
    eprintln!("PBE CLI vs library force: {d:.2e}");
    assert!(d < BAR, "{d:e}");
}

#[test]
fn printed_force_equals_the_library_force_with_rsgdf_jk() {
    let o = run(&loose_input("rsgdf", "PBE0", "[1, 1, 2]", RSGDF));
    let out = ok(&o, "rsgdf");
    assert!(out.contains("analytic k-point RKS gradient"), "{out}");
    let g = printed_gradient(&out, 2);
    let at = printed_atoms(&out, 2);
    let lib = Lib {
        functional: "PBE0",
        mesh: [1, 1, 2],
        exxdiv_scf: ExxDiv::Ewald,
        rsgdf: true,
        cutoff: true,
    };
    assert!(
        g.iter().fold(0.0_f64, |m, v| m.max(v.abs())) > 1e-3,
        "trivial"
    );
    let d = max_diff(&g, &library_gradient(&at, lib, ExxDiv::Ewald));
    eprintln!("RS-GDF PBE0 CLI vs library force: {d:.2e}");
    assert!(d < BAR, "{d:e}");
}

#[test]
fn unsupported_kohn_sham_routes_are_refused_with_exact_messages() {
    // Range-separated hybrids: the HSE06 / CAM-B3LYP force runs on either
    // J/K (periodic_rsh_open_mgga_derivs.rs); the force allowlist names the
    // unvalidated ones.
    let e = refused(
        &run(&loose_input(
            "wb97x",
            "HYB_GGA_XC_WB97X",
            "[1, 1, 2]",
            RSGDF,
        )),
        "wb97x",
    );
    assert!(
        e.contains(
            "HYB_GGA_XC_WB97X: the range-separated hybrid force and stress are validated for \
             HSE06 / CAM-B3LYP (HYB_GGA_XC_CAM_B3LYP) only"
        ),
        "{e}"
    );
    // Hybrid meta-GGA and the raw TPSS component names stay refused by the
    // energy gate (TPSS itself has a force: `printed_force_equals_the_library_force_for_meta_gga`).
    for f in ["TPSSh", "MGGA_X_TPSS", "wB97X-V"] {
        let e = refused(&run(&loose_input("mgga-ref", f, "[1, 1, 2]", DENSE)), f);
        assert!(e.contains("[cell] kmesh with functional"), "{f}: {e}");
    }
    // TPSS derivatives are validated on the dense J only: RS-GDF is refused
    // by name (SCAN / r2SCAN on RS-GDF stay accepted: the meta-GGA test).
    let e = refused(
        &run(&loose_input("tpss-gdf", "TPSS", "[1, 1, 2]", RSGDF)),
        "tpss-gdf",
    );
    assert!(
        e.contains("method.task = \"optimize\" with [cell] kmesh")
            && e.contains(
                "TPSS: the k-point TPSS force and stress are validated on the \
                 dense-AFT J only"
            ),
        "{e}"
    );
    // Open-shell Kohn-Sham with a meta-GGA (ksdft with multiplicity > 1 is
    // UKS): the force runs on the dense-AFT J/K (periodic_rsh_open_mgga_derivs.rs);
    // on RS-GDF the energy gate already names the J/K.
    let mut i = loose_input("uks-scan", "SCAN", "[1, 1, 2]", RSGDF);
    i.xyz = xyz(&[[0.0, 0.0, 0.0]]);
    i.mult = 2;
    let e = refused(&run(&i), "uks-scan");
    assert!(e.contains("dense-AFT J/K only"), "{e}");
    // The range split and the automatic RS-GDF omega are energy-only.
    let split = "jk = \"rsgdf\"\nauxbasis = \"cc-pvdz-ri\"\nrange_split = true";
    let e = refused(
        &run(&loose_input("split", "PBE", "[1, 1, 2]", split)),
        "split",
    );
    assert!(
        e.contains("[cell] range_split is not supported with method.task = \"optimize\""),
        "{e}"
    );
    let auto = "jk = \"rsgdf\"\nauxbasis = \"cc-pvdz-ri\"\ngdf_omega = \"auto\"";
    let e = refused(&run(&loose_input("auto", "PBE", "[1, 1, 2]", auto)), "auto");
    assert!(
        e.contains(
            "[cell] gdf_omega = \"auto\" is calibrated for the Gamma-point build; with kmesh \
             set a number"
        ),
        "{e}"
    );
}

#[test]
fn a_non_converging_scf_step_is_a_typed_error_not_a_force() {
    // [scf] max_iter = 2 cannot converge the k-point KS SCF: the optimize step
    // must stop with the library's convergence error, never differentiate an
    // unconverged density. (`[scf]` is appended after `[optimize]`.)
    let mut i = loose_input("noconv", "PBE0", "[1, 1, 2]", DENSE);
    i.optimize = "max_steps = 3\n[scf]\nmax_iter = 2";
    let e = refused(&run(&i), "noconv");
    assert!(
        e.contains("solve_krks: PBE0 SCF did not converge in 2 iterations"),
        "{e}"
    );
}

#[test]
fn an_ecp_cell_passes_the_k_point_kohn_sham_force_gate() {
    // I carries the def2 ECP in def2-svp. The k-point KS force with an ECP is
    // validated in the library (ferric-pbc/tests/validation_kks_ecp_derivs.rs);
    // here the optimize must get past the gate to the SCF, which is capped at
    // two iterations so the test stays cheap (RS-GDF: the dense AFT of iodine's
    // tight primitives in this box exceeds the G-vector cap).
    let mut i = loose_input("ecp", "PBE", "[1, 1, 2]", ECP_RSGDF);
    i.xyz = "2\nHI\nH 0.0 0.0 0.0\nI 0.0 0.0 1.61\n".to_string();
    i.basis = "def2-svp";
    i.optimize = "max_steps = 3\n[scf]\nmax_iter = 2";
    let e = refused(&run(&i), "ecp");
    assert!(!e.contains("not validated"), "{e}");
    assert!(
        e.contains("solve_krks: PBE SCF did not converge in 2 iterations"),
        "{e}"
    );
}

// ------------------------------------------------- Gamma point, meta-GGA

/// A Gamma-point (no kmesh) RKS optimize input, otherwise as [`run`].
fn run_gamma(tag: &str, functional: &str, jk: &str) -> Output {
    let dir = scratch(tag);
    let xyz_path = dir.join("m.xyz");
    std::fs::write(&xyz_path, xyz(&H2_OFFAXIS)).unwrap();
    let toml = dir.join("in.toml");
    std::fs::write(
        &toml,
        format!(
            r#"[molecule]
xyz = "{xyz}"
[basis]
name = "sto-3g"
[method]
kind = "ksdft"
task = "optimize"
[dft]
functional = "{functional}"
[cell]
unit = "bohr"
lattice = [[{a}, 0.0, 0.0], [0.0, {a}, 0.0], [0.0, 0.0, {a}]]
exxdiv = "ewald"
omega = {OMEGA}
n_radial = {N_RADIAL}
n_angular = {N_ANGULAR}
neighbour_cutoff = {CUTOFF_BOHR}
{jk}
[optimize]
{LOOSE}
"#,
            xyz = xyz_path.display(),
            a = A_BOHR,
        ),
    )
    .unwrap();
    Command::new(ferric_cli_bin())
        .arg(&toml)
        .current_dir(workspace_root())
        .env("OPENBLAS_NUM_THREADS", "1")
        .output()
        .expect("run ferric-cli")
}

/// The routed Gamma meta-GGA force (`gamma_rks` + `gamma_rks_mgga_gradient`)
/// at `atoms_bohr`, with the CLI's configs restated.
fn gamma_library_gradient(atoms_bohr: &[[f64; 3]], functional: &str) -> Vec<f64> {
    use ferric_pbc::dense_aft::{
        DenseAftEri, DEFAULT_DENSE_AFT_MAX_BYTES, DEFAULT_DENSE_AFT_PRECISION,
    };
    use ferric_pbc::gamma_mgga::{gamma_rks_mgga_gradient, GammaMggaDerivConfig};
    let dir = scratch("glib");
    let p = dir.join("l.xyz");
    std::fs::write(&p, xyz(atoms_bohr)).unwrap();
    let mol = Molecule::load_xyz_with_charge(p.to_str().unwrap(), 0, 1).expect("xyz");
    let bs = basis::bundled("sto-3g").unwrap();
    let lat = [[A_BOHR, 0.0, 0.0], [0.0, A_BOHR, 0.0], [0.0, 0.0, A_BOHR]];
    let cell = Cell::new(mol, lat).expect("cell");
    let prep = PreparedBasis::new(cell.mol(), &bs).unwrap();
    let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
    let hc = ferric_pbc::hcore::periodic_hcore(&cell, &prep, &hcfg).unwrap();
    let eri = DenseAftEri::build(
        &cell,
        &prep,
        &hc.s,
        ExxDiv::Ewald,
        DEFAULT_DENSE_AFT_PRECISION,
        DEFAULT_DENSE_AFT_MAX_BYTES,
    )
    .unwrap();
    let mut c = ferric_pbc::dft::GammaRksConfig::new(functional);
    c.grid = PeriodicGridConfig::with_size(N_RADIAL, N_ANGULAR);
    c.grid.neighbour_cutoff = Some(CUTOFF_BOHR);
    c.exxdiv = ExxDiv::Ewald;
    c.scf.use_sad_guess = false;
    let r = ferric_pbc::dft::gamma_rks(
        &cell,
        &prep,
        &hc,
        ferric_pbc::uhf::GammaUhfIntegrals::DenseAft(&eri),
        &c,
    )
    .expect("gamma_rks");
    assert!(r.scf.converged);
    let cfg = GammaMggaDerivConfig::for_binding(DEFAULT_DENSE_AFT_MAX_BYTES, None);
    gamma_rks_mgga_gradient(&cell, &prep, &hcfg, &r.scf, &c, &cfg)
        .expect("gamma_rks_mgga_gradient")
        .grad
        .iter()
        .copied()
        .collect()
}

/// Gamma-point TPSS optimize (issue #492): the printed final force equals the
/// routed library force at the printed geometry (bar as the k-point rows);
/// MUTANT: the SCAN force at the same geometry misses it. A meta-GGA optimize
/// on RS-GDF is refused before any SCF.
#[test]
fn gamma_printed_force_equals_the_routed_library_force_for_tpss() {
    let out = ok(&run_gamma("gamma-tpss", "TPSS", DENSE), "gamma-tpss");
    let g = printed_gradient(&out, 2);
    let at = printed_atoms(&out, 2);
    assert!(
        g.iter().fold(0.0_f64, |m, v| m.max(v.abs())) > 1e-3,
        "trivial"
    );
    let d = max_diff(&g, &gamma_library_gradient(&at, "TPSS"));
    eprintln!("Gamma TPSS CLI vs routed library force: {d:.2e}");
    assert!(d < BAR, "{d:e}");
    let bad = gamma_library_gradient(&at, "SCAN");
    assert!(max_diff(&g, &bad) > 1e-4, "functional mutant not detected");
    let e = refused(
        &run_gamma("gamma-tpss-gdf", "TPSS", RSGDF),
        "gamma-tpss-gdf",
    );
    assert!(
        e.contains("method.task = \"optimize\" with [cell]") && e.contains("dense-AFT J only"),
        "{e}"
    );
}

/// The Gamma pre-SCF dense budget counts the routed force's 1x1x1 kernels
/// (32 nao^4 bytes) for a meta-GGA optimize, as the Python preflight does:
/// nao = 2, so plain 128 B and meta-GGA optimize 640 B; a 322 B cap runs the
/// PBE optimize and refuses the TPSS one before any SCF.
#[test]
fn gamma_meta_gga_optimize_counts_the_one_point_kernels_in_the_dense_cap() {
    let cap = "jk = \"dense\"\nmax_eri_gb = 3e-7";
    ok(&run_gamma("gamma-cap-pbe", "PBE", cap), "gamma-cap-pbe");
    let e = refused(&run_gamma("gamma-cap-tpss", "TPSS", cap), "gamma-cap-tpss");
    assert!(e.contains("dense AFT ERI tensors need 640 bytes"), "{e}");
}
