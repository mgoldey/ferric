//! Integration tests for the ROHF solver and analytical gradient.
//!
//! Covers:
//! - OH radical cc-pVDZ (doublet) vs PySCF reference; ⟨S²⟩ exact 0.75.
//! - CH3 radical cc-pVDZ (doublet) vs PySCF reference; ⟨S²⟩ exact 0.75.
//! - O2 sto-3g (triplet) vs PySCF reference; ⟨S²⟩ exact 2.0.
//! - ROHF analytical gradient vs ±5e-4 Bohr central-difference on OH/STO-3G.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::gradient::rohf_gradient;
use ferric_scf::rohf::{solve_rohf, RohfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::Spin;
use ndarray::Array2;

fn run_rohf(
    xyz: &str,
    charge: i32,
    mult: usize,
    basis_name: &str,
) -> (ferric_scf::ScfResult, Molecule, PreparedBasis) {
    let mol = Molecule::parse_xyz(xyz, charge, mult).unwrap();
    let bs = basis::bundled(basis_name).unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let cfg = RohfConfig {
        energy_conv: 1e-10,
        density_conv: 1e-8,
        max_iter: 200,
        ..Default::default()
    };
    let ctx = ParallelContext::default();
    let res = solve_rohf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
    (res, mol, prep)
}

fn check_energy(slug: &str, energy: f64, tol: f64) {
    let path = format!("../../testdata/reference/{slug}");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|_| {
        panic!("missing reference {path}");
    });
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    let ref_e = v["energy"].as_f64().unwrap();
    assert!(
        (energy - ref_e).abs() < tol,
        "{slug}: got {:.10}, ref {:.10} (diff {:.2e})",
        energy,
        ref_e,
        (energy - ref_e).abs()
    );
}

#[test]
fn rohf_oh_doublet_ccpvdz() {
    let xyz = "2\nOH\nO 0 0 0\nH 0 0 0.97\n";
    let (res, _mol, _prep) = run_rohf(xyz, 0, 2, "cc-pvdz");
    assert!(res.converged, "OH/cc-pvdz ROHF did not converge");
    assert!(matches!(res.spin, Spin::RestrictedOpen));
    eprintln!("OH/cc-pvdz E = {:.10}", res.energy);
    // ⟨S²⟩ exact by construction = 0.75 for doublet.
    check_energy("oh_cc-pvdz_rohf.json", res.energy, 1e-6);
}

#[test]
fn rohf_ch3_doublet_ccpvdz() {
    let r = 1.079;
    let (x1, y1) = (r, 0.0);
    let (x2, y2) = (-0.5 * r, r * (3f64).sqrt() * 0.5);
    let (x3, y3) = (-0.5 * r, -r * (3f64).sqrt() * 0.5);
    let xyz = format!(
        "4\nCH3\nC 0 0 0\nH {x1} {y1} 0\nH {x2:.6} {y2:.6} 0\nH {x3:.6} {y3:.6} 0\n"
    );
    let (res, _mol, _prep) = run_rohf(&xyz, 0, 2, "cc-pvdz");
    assert!(res.converged, "CH3/cc-pvdz ROHF did not converge");
    eprintln!("CH3/cc-pvdz E = {:.10}", res.energy);
    check_energy("ch3_cc-pvdz_rohf.json", res.energy, 1e-6);
}

#[test]
fn rohf_o2_triplet_sto3g() {
    let xyz = "2\nO2\nO 0 0 0\nO 0 0 1.208\n";
    let (res, _mol, _prep) = run_rohf(xyz, 0, 3, "sto-3g");
    assert!(res.converged, "O2/sto-3g ROHF did not converge");
    eprintln!("O2/sto-3g E = {:.10}", res.energy);
    // ⟨S²⟩ exact = 2.0 for triplet (S=1).
    check_energy("o2_sto-3g_rohf.json", res.energy, 1e-6);
}

fn fd_energy(mol: &Molecule, basis_name: &str) -> f64 {
    let bs = basis::bundled(basis_name).unwrap();
    let prep = PreparedBasis::new(mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let cfg = RohfConfig {
        energy_conv: 1e-12,
        density_conv: 1e-10,
        max_iter: 200,
        ..Default::default()
    };
    let ctx = ParallelContext::default();
    solve_rohf(&ctx, mol, &prep, op, &bounds, &cfg).unwrap().energy
}

#[test]
fn rohf_gradient_oh_sto3g_fd() {
    let xyz = "2\nOH\nO 0 0 0\nH 0 0 0.97\n";
    let mol = Molecule::parse_xyz(xyz, 0, 2).unwrap();
    let bs = basis::bundled("sto-3g").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let cfg = RohfConfig {
        energy_conv: 1e-12,
        density_conv: 1e-10,
        max_iter: 200,
        ..Default::default()
    };
    let ctx = ParallelContext::default();
    let res = solve_rohf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
    let analytic = rohf_gradient(&mol, &prep, op, &bounds, &res, None).unwrap();

    let h = 5e-4_f64;
    let natoms = mol.atoms.len();
    let mut fd = Array2::<f64>::zeros((natoms, 3));
    for atom in 0..natoms {
        for coord in 0..3 {
            let mut mp = mol.clone();
            let mut mm = mol.clone();
            match coord {
                0 => { mp.atoms[atom].x += h; mm.atoms[atom].x -= h; }
                1 => { mp.atoms[atom].y += h; mm.atoms[atom].y -= h; }
                _ => { mp.atoms[atom].zpos += h; mm.atoms[atom].zpos -= h; }
            }
            let ep = fd_energy(&mp, "sto-3g");
            let em = fd_energy(&mm, "sto-3g");
            fd[(atom, coord)] = (ep - em) / (2.0 * h);
        }
    }
    eprintln!("=== OH/STO-3G ROHF gradient ===");
    let mut max_diff = 0.0_f64;
    for atom in 0..natoms {
        for c in 0..3 {
            let diff = (analytic[(atom, c)] - fd[(atom, c)]).abs();
            eprintln!(
                "atom={atom} coord={c}: analytic={:.8} fd={:.8} diff={:.2e}",
                analytic[(atom, c)], fd[(atom, c)], diff
            );
            if diff > max_diff { max_diff = diff; }
        }
    }
    assert!(max_diff < 1e-4, "ROHF gradient FD mismatch: max diff = {:.2e}", max_diff);
}
