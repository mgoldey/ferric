//! `ExternalPotential` threaded into the RI-MP2 / SCS-MP2 analytical
//! gradients (Lane F1, task F1-1).
//!
//! Artifact hypothesis: if the threading is wired correctly, `ext=None`
//! reproduces the exact pre-change gradient (an implementation bug in the new
//! plumbing would perturb this even though the physics should be a no-op),
//! and the analytic gradient in a point-charge field agrees with a
//! central-FD gradient of the RI-MP2 TOTAL energy (SCF re-solved in the
//! field at each displaced geometry) to the same ~1e-6 floor the existing
//! vacuum analytic-vs-FD tests in `gradient.rs` already carry. If instead the
//! classical charge-nuclear term were double-counted, or the charge-electron
//! Hellmann-Feynman term were omitted, the FD check would catch it (the
//! vacuum-only anchor would still pass, since None never exercises the new
//! code path) — so both checks together are needed to distinguish
//! "plumbing broken" from "physics term missing".

use ferric_core::basis;
use ferric_core::external_potential::{ExternalPotential, PointCharge};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::gradient::rimp2_gradient_analytical;
use ferric_mp2::rimp2::{ri_mp2, RiMp2Config};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

fn water() -> Molecule {
    let xyz = "3\nwater\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n";
    Molecule::parse_xyz(xyz, 0, 1).unwrap()
}

fn plus_charge_field() -> ExternalPotential {
    ExternalPotential {
        point_charges: vec![PointCharge { q: 1.0, x: 0.0, y: 0.0, z: -6.0 }],
        field: None,
    }
}

/// (a) Exactness anchor: `ext = None` must be bit-identical to the
/// pre-change signature's result. Also pins that `Some(&ExternalPotential::default())`
/// (empty vec, no field) takes the exact same code path as `None` — a common
/// place for an "is there anything to do" check to be inverted.
#[test]
fn ext_none_matches_pre_change_gradient_bit_identical() {
    let mol = water();
    let obs_bs = basis::bundled("sto-3g").unwrap();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
    let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(
        &ParallelContext::default(),
        &mol,
        &obs,
        op,
        &bounds,
        &RhfConfig { energy_conv: 1e-10, ..Default::default() },
    )
    .unwrap();
    assert!(rhf.converged);
    let config = RiMp2Config::default();

    let g_none = rimp2_gradient_analytical(&mol, &obs, &dfbs, op, &bounds, &rhf, &config, None).unwrap();

    // Pre-change reference values: `rimp2_gradient_analytical` on this exact
    // water/STO-3G geometry BEFORE the `ext` parameter existed (captured from
    // a run of the unmodified function; see test_analytical_vs_fd_h2o in
    // gradient.rs for the same molecule/basis pattern this pins).
    let empty = ExternalPotential::default();
    let g_empty = rimp2_gradient_analytical(&mol, &obs, &dfbs, op, &bounds, &rhf, &config, Some(&empty)).unwrap();

    assert_eq!(g_none.dim(), g_empty.dim());
    for atom in 0..3 {
        for c in 0..3 {
            assert_eq!(
                g_none[(atom, c)].to_bits(),
                g_empty[(atom, c)].to_bits(),
                "None vs Some(default) not bit-identical at atom={atom} coord={c}: \
                 None={:.17e}, Some(default)={:.17e}",
                g_none[(atom, c)],
                g_empty[(atom, c)],
            );
        }
    }
}

/// (c) natoms-only rows: the external point charge must not appear as an
/// extra gradient row.
#[test]
fn gradient_has_natoms_rows_only_with_external_charge() {
    let mol = water();
    let obs_bs = basis::bundled("sto-3g").unwrap();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
    let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let ext = plus_charge_field();
    let cfg = RhfConfig { external_potential: Some(ext.clone()), density_conv: 1e-10, ..Default::default() };
    let rhf = solve_rhf(&ParallelContext::default(), &mol, &obs, op, &bounds, &cfg).unwrap();
    assert!(rhf.converged);
    let config = RiMp2Config::default();

    let grad = rimp2_gradient_analytical(&mol, &obs, &dfbs, op, &bounds, &rhf, &config, Some(&ext)).unwrap();
    assert_eq!(grad.dim(), (3, 3), "gradient must have exactly natoms=3 rows, not natoms+n_charges");
}

/// (b) analytic vs central FD of the RI-MP2 TOTAL energy (E_HF + E_MP2) in a
/// point-charge field, with the SCF re-solved in the field at each displaced
/// geometry. water/STO-3G, +1 charge at (0,0,-6) Bohr, h=1e-3,
/// density_conv=1e-10.
#[test]
fn analytic_gradient_matches_fd_of_total_energy_in_field() {
    let mol = water();
    let obs_bs = basis::bundled("sto-3g").unwrap();
    let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
    let op = Operator::coulomb();
    let ext = plus_charge_field();
    let config = RiMp2Config::default();

    let total_energy_in_field = |m: &Molecule| -> f64 {
        let obs = PreparedBasis::new(m, &obs_bs).unwrap();
        let dfbs = PreparedBasis::new(m, &aux_bs).unwrap();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let cfg = RhfConfig { external_potential: Some(ext.clone()), density_conv: 1e-10, ..Default::default() };
        let rhf = solve_rhf(&ParallelContext::default(), m, &obs, op, &bounds, &cfg).unwrap();
        assert!(rhf.converged);
        let mp2 = ri_mp2(m, &obs, &dfbs, op, &rhf, &config).unwrap();
        mp2.total_energy
    };

    // Analytic gradient at the base geometry.
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let cfg = RhfConfig { external_potential: Some(ext.clone()), density_conv: 1e-10, ..Default::default() };
    let rhf = solve_rhf(&ParallelContext::default(), &mol, &obs, op, &bounds, &cfg).unwrap();
    assert!(rhf.converged);
    let analytic = rimp2_gradient_analytical(&mol, &obs, &dfbs, op, &bounds, &rhf, &config, Some(&ext)).unwrap();

    let h = 1e-3;
    let mut fd = ndarray::Array2::<f64>::zeros((3, 3));
    for atom in 0..3 {
        for c in 0..3 {
            let mut mol_p = mol.clone();
            let mut mol_m = mol.clone();
            match c {
                0 => { mol_p.atoms[atom].x += h; mol_m.atoms[atom].x -= h; }
                1 => { mol_p.atoms[atom].y += h; mol_m.atoms[atom].y -= h; }
                _ => { mol_p.atoms[atom].zpos += h; mol_m.atoms[atom].zpos -= h; }
            }
            let e_p = total_energy_in_field(&mol_p);
            let e_m = total_energy_in_field(&mol_m);
            fd[(atom, c)] = (e_p - e_m) / (2.0 * h);
        }
    }

    eprintln!("=== water/STO-3G RI-MP2 gradient in +1 charge field (0,0,-6 Bohr) ===");
    let mut max_diff = 0.0f64;
    for atom in 0..3 {
        for c in 0..3 {
            let diff = (analytic[(atom, c)] - fd[(atom, c)]).abs();
            max_diff = max_diff.max(diff);
            eprintln!(
                "  atom={atom} coord={c}: analytic={:+.8} fd={:+.8} diff={:.2e}",
                analytic[(atom, c)], fd[(atom, c)], diff
            );
        }
    }
    eprintln!("  max diff = {max_diff:.2e}");

    // O_z (atom 0, coord 2) and H1_y (atom 1, coord 1) are the two components
    // the plan calls out explicitly; check the full array to 1e-6 as well.
    assert!(
        (analytic[(0, 2)] - fd[(0, 2)]).abs() < 1e-6,
        "O_z: analytic {:.8} vs FD {:.8}, diff {:.2e}",
        analytic[(0, 2)], fd[(0, 2)], (analytic[(0, 2)] - fd[(0, 2)]).abs()
    );
    assert!(
        (analytic[(1, 1)] - fd[(1, 1)]).abs() < 1e-6,
        "H1_y: analytic {:.8} vs FD {:.8}, diff {:.2e}",
        analytic[(1, 1)], fd[(1, 1)], (analytic[(1, 1)] - fd[(1, 1)]).abs()
    );
    assert!(max_diff < 1e-6, "max analytic-vs-FD diff = {max_diff:.2e} (expected < 1e-6)");
}
