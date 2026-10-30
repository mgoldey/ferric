//! Weekly half of `periodic_kpoint_optimize.rs`: the k-point ROHF force on
//! the RS-GDF J/K with a rotating (cc-pVDZ, p-shell) orbital basis, CLI ==
//! library at the printed geometry. It took ~135 s on CI (runs 40957025137 /
//! 40953725119, mean), above the 120 s per-commit threshold. Per-commit
//! canaries: `periodic_kpoint_optimize.rs::rotating_basis_rsgdf_k_point_force_equals_the_library_force`
//! (rotating-basis RS-GDF force, RHF) and
//! `periodic_kpoint_open_shell.rs::printed_force_equals_the_library_force_for_rohf`
//! (k-point ROHF force, dense). Shared harness: `kpoint_optimize_cli/mod.rs`.

mod kpoint_optimize_cli;
use kpoint_optimize_cli::*;

/// Library RS-GDF ROHF force with the configs the CLI builds for
/// `jk = "rsgdf"`, `auxbasis = "cc-pvdz-ri"` on cc-pVDZ (column rotation ON);
/// also returns the rotated hcore column count.
fn library_gradient_rohf_rsgdf_ccpvdz(atoms_bohr: &[[f64; 3]]) -> (Vec<f64>, usize) {
    let dir = scratch("lib-rohf-gdf");
    let p = dir.join("l.xyz");
    std::fs::write(&p, xyz(atoms_bohr)).unwrap();
    let mol = Molecule::load_xyz_with_charge(p.to_str().unwrap(), 0, 2).expect("xyz");
    let bs = basis::bundled("cc-pvdz").unwrap();
    let aux = basis::bundled("cc-pvdz-ri").unwrap();
    let lat = [[A_BOHR, 0.0, 0.0], [0.0, A_BOHR, 0.0], [0.0, 0.0, A_BOHR]];
    let cell = Cell::new(mol, lat).expect("cell");
    let prep = PreparedBasis::new(cell.mol(), &bs).unwrap();
    let auxp = PreparedBasis::new(cell.mol(), &aux).unwrap();
    let mesh = KPointMesh::new(&cell, [1, 1, 2], MeshCentring::Gamma).unwrap();
    let mut c = KRohfConfig::for_cell(&cell, ExxDiv::Ewald);
    c.scf.max_iter = 200;
    c.scf.energy_conv = 1e-12;
    c.scf.grad_conv = 1e-9;
    c.hcore =
        PeriodicHcoreConfig::with_omega(OMEGA).with_sr_column_rotation(SrColumnRotation::on());
    c.jk = KJkKind::RsGdf;
    c.rsgdf.gdf.sr_column_rotation = SrColumnRotation::on();
    let (r, kept) = solve_krohf_keep(&cell, &prep, Some(&auxp), &mesh, &c).expect("scf");
    let rotated = kept.hk.sr_rotated_columns;
    let cfg = KUksGradConfig {
        base: KRksGradConfig {
            grad: KGradConfig::default(),
            exxdiv: Some(ExxDiv::Ewald),
            ..Default::default()
        },
        ..Default::default()
    };
    let g = kpoint_rohf_gradient(
        &cell,
        &prep,
        &mesh,
        &c.hcore,
        &kept.hk,
        kept.grad_jk(Some(&auxp), &c.rsgdf).unwrap(),
        &r.scf,
        ExxDiv::Ewald,
        &cfg,
    )
    .expect("gradient");
    (g.grad.iter().copied().collect(), rotated)
}

/// The ROHF RS-GDF force on a rotating (p-shell) basis, end to end: the CLI's
/// printed final gradient equals the library force at the printed geometry.
#[test]
#[ignore = "validation: k-point ROHF RS-GDF force on a rotating cc-pVDZ basis, CLI == library; CI ~135 s"]
fn rotating_basis_rohf_rsgdf_k_point_force_equals_the_library_force() {
    let o = run_with(
        "rohf-rsgdf-rot",
        &xyz(&H3_OFFAXIS),
        "rohf",
        2,
        "[1, 1, 2]",
        LOOSE,
        "cc-pvdz",
        "jk = \"rsgdf\"\nauxbasis = \"cc-pvdz-ri\"",
    );
    let out = ok(&o, "rohf-rsgdf-rot");
    assert!(out.contains("sr_column_rotation = on (default)"), "{out}");
    let g = printed_gradient(&out, 3);
    let at = printed_atoms(&out, 3);
    let (lib, lib_rotated) = library_gradient_rohf_rsgdf_ccpvdz(&at);
    assert!(lib_rotated > 0);
    assert!(
        g.iter().fold(0.0_f64, |m, v| m.max(v.abs())) > 1e-3,
        "trivial"
    );
    let d = max_diff(&g, &lib);
    eprintln!("ROHF RS-GDF rotated CLI vs library force: {d:.2e}");
    assert!(d < BAR, "{d:e}\n{g:?}\n{lib:?}");
}
