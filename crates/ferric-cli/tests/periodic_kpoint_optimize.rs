//! `task = "optimize"` with `[cell] kmesh` (k-point RHF / UHF forces) is wired
//! to the library force.
//!
//! * EXACTNESS ANCHOR (CLI vs library): with loose optimizer thresholds the run
//!   stops after a few steps; the printed final gradient is the force at the
//!   printed final geometry, which must equal `kpoint_rhf_gradient` /
//!   `kpoint_uhf_gradient` called directly on the same cell, mesh and configs
//!   (evaluated at the printed 8-decimal geometry, bar 5e-7). A mesh, spin,
//!   unit or wrong-J/K slip in the wiring moves it by orders of magnitude
//!   (mesh mutant asserted).
//! * A real optimization (stretched H2, 1x1x3) converges to the optimizer's
//!   gradient thresholds, lowers the energy, and the printed final gradient
//!   equals the library force at the printed final geometry.
//! * Refusals: ROHF (k-point Kohn-Sham: periodic_kpoint_ks_optimize.rs).
//!
//! TIERS. The rotating-basis (cc-pVDZ) ROHF RS-GDF force anchor (~135 s on
//! CI) runs weekly in `validation_periodic_kpoint_optimize.rs`. Per-commit
//! canaries: `rotating_basis_rsgdf_k_point_force_equals_the_library_force`
//! (the same rotating-basis RS-GDF force wiring, RHF) and
//! `periodic_kpoint_open_shell.rs::printed_force_equals_the_library_force_for_rohf`
//! (the k-point ROHF force wiring, dense J/K). Shared harness:
//! `kpoint_optimize_cli/mod.rs`.

mod kpoint_optimize_cli;
use kpoint_optimize_cli::*;

#[test]
fn printed_k_point_force_equals_the_library_force_rhf() {
    let o = run("rhf-grad", &xyz(&H2_OFFAXIS), "rhf", 1, "[1, 1, 3]", LOOSE);
    let out = ok(&o, "rhf");
    assert!(out.contains("analytic k-point RHF gradient"), "{out}");
    let g = printed_gradient(&out, 2);
    let at = printed_atoms(&out, 2);
    let lib = library_gradient(&at, 1, [1, 1, 3], false, None);
    assert!(
        g.iter().fold(0.0_f64, |m, v| m.max(v.abs())) > 1e-3,
        "trivial"
    );
    let d = max_diff(&g, &lib);
    eprintln!("RHF CLI vs library force: {d:.2e}");
    assert!(d < BAR, "{d:e}\n{g:?}\n{lib:?}");
    // MUTANT: a wrong mesh (1x1x1) moves the force by far more than the bar.
    let bad = library_gradient(&at, 1, [1, 1, 3], false, Some([1, 1, 1]));
    assert!(max_diff(&g, &bad) > 1e-4, "mesh mutant not detected");
}

#[test]
fn printed_k_point_force_equals_the_library_force_uhf() {
    let o = run("uhf-grad", &xyz(&H3_OFFAXIS), "uhf", 2, "[1, 1, 3]", LOOSE);
    let out = ok(&o, "uhf");
    assert!(out.contains("analytic k-point UHF gradient"), "{out}");
    let g = printed_gradient(&out, 3);
    let at = printed_atoms(&out, 3);
    let lib = library_gradient(&at, 2, [1, 1, 3], true, None);
    assert!(
        g.iter().fold(0.0_f64, |m, v| m.max(v.abs())) > 1e-3,
        "trivial"
    );
    let d = max_diff(&g, &lib);
    eprintln!("UHF CLI vs library force: {d:.2e}");
    assert!(d < BAR, "{d:e}\n{g:?}\n{lib:?}");
    // MUTANT: a wrong mesh (1x1x1) moves the open-shell force beyond the bar.
    let bad = library_gradient(&at, 2, [1, 1, 3], true, Some([1, 1, 1]));
    assert!(max_diff(&g, &bad) > 1e-4, "mesh mutant not detected");
}

#[test]
fn k_point_optimize_converges_and_the_final_force_is_the_library_force() {
    let start = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0 / BOHR_RADIUS_ANGSTROM]];
    let o = run("opt", &xyz(&start), "rhf", 1, "[1, 1, 3]", "max_steps = 30");
    let out = ok(&o, "opt");
    assert_eq!(value(&out, "converged"), "true", "{out}");
    let b = BOHR_RADIUS_ANGSTROM;
    let fin = rows(&out, "final geometry", 2);
    let atoms: Vec<[f64; 3]> = fin.iter().map(|r| [r[0] / b, r[1] / b, r[2] / b]).collect();
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
    let lib = library_gradient(&atoms, 1, [1, 1, 3], false, None);
    // The printed geometry has 8 decimals (Angstrom): the force changes by
    // ~k * 1e-8 over the rounding, below the bar.
    assert!(max_diff(&g, &lib) < 5e-7, "{g:?} vs {lib:?}");
}

/// Library RS-GDF RHF force with the configs the CLI builds for
/// `jk = "rsgdf"`, `auxbasis = "cc-pvdz-ri"` (default column rotation ON and
/// default RS-GDF omega); also returns the rotated hcore column count.
fn library_gradient_rsgdf_ccpvdz(atoms_bohr: &[[f64; 3]]) -> (Vec<f64>, usize) {
    let dir = scratch("lib-gdf");
    let p = dir.join("l.xyz");
    std::fs::write(&p, xyz(atoms_bohr)).unwrap();
    let mol = Molecule::load_xyz_with_charge(p.to_str().unwrap(), 0, 1).expect("xyz");
    let bs = basis::bundled("cc-pvdz").unwrap();
    let aux = basis::bundled("cc-pvdz-ri").unwrap();
    let lat = [[A_BOHR, 0.0, 0.0], [0.0, A_BOHR, 0.0], [0.0, 0.0, A_BOHR]];
    let cell = Cell::new(mol, lat).expect("cell");
    let prep = PreparedBasis::new(cell.mol(), &bs).unwrap();
    let auxp = PreparedBasis::new(cell.mol(), &aux).unwrap();
    let mesh = KPointMesh::new(&cell, [1, 1, 2], MeshCentring::Gamma).unwrap();
    let mut c = KRhfConfig::for_cell(&cell, ExxDiv::Ewald);
    c.scf.energy_conv = 1e-12;
    c.scf.grad_conv = 1e-9;
    c.hcore =
        PeriodicHcoreConfig::with_omega(OMEGA).with_sr_column_rotation(SrColumnRotation::on());
    c.jk = KJkKind::RsGdf;
    c.rsgdf.gdf.sr_column_rotation = SrColumnRotation::on();
    let (r, kept) = solve_krhf_keep(&cell, &prep, Some(&auxp), &mesh, &c).expect("scf");
    let rotated = kept.hk.sr_rotated_columns;
    let g = kpoint_rhf_gradient(
        &cell,
        &prep,
        &mesh,
        &c.hcore,
        &kept.hk,
        kept.grad_jk(Some(&auxp), &c.rsgdf).unwrap(),
        &r,
        ExxDiv::Ewald,
        &KGradConfig::default(),
    )
    .expect("gradient");
    (g.grad.iter().copied().collect(), rotated)
}

#[test]
fn rotating_basis_rsgdf_k_point_force_equals_the_library_force() {
    let o = run_with(
        "rsgdf-rot",
        &xyz(&H2_OFFAXIS),
        "rhf",
        1,
        "[1, 1, 2]",
        LOOSE,
        "cc-pvdz",
        "jk = \"rsgdf\"\nauxbasis = \"cc-pvdz-ri\"",
    );
    let out = ok(&o, "rsgdf-rot");
    // The default column rotation is on (the optimizer prints no stage table,
    // so the rotated-column count is read from the library run below).
    assert!(out.contains("sr_column_rotation = on (default)"), "{out}");
    let g = printed_gradient(&out, 2);
    let at = printed_atoms(&out, 2);
    let (lib, lib_rotated) = library_gradient_rsgdf_ccpvdz(&at);
    assert!(lib_rotated > 0);
    assert!(
        g.iter().fold(0.0_f64, |m, v| m.max(v.abs())) > 1e-3,
        "trivial"
    );
    let d = max_diff(&g, &lib);
    eprintln!("RS-GDF rotated CLI vs library force: {d:.2e}");
    assert!(d < BAR, "{d:e}\n{g:?}\n{lib:?}");
}
