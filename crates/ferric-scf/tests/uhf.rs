//! Integration tests for the UHF solver and analytical gradient.
//!
//! Covers:
//! - H atom doublet (trivial 1-electron case)
//! - OH radical cc-pVDZ (open shell)
//! - CH3 radical cc-pVDZ (open shell)
//! - UHF analytical gradient vs central-difference (H atom, OH/STO-3G)
//!
//! Reference energies are loaded from `testdata/reference/*_uhf.json` when
//! present. If absent, only convergence and ⟨S²⟩ sanity is checked and the
//! observed energy is printed for the user to inspect.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::gradient::uhf_gradient;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::{solve_uhf, UhfConfig};
use ndarray::Array2;

fn s_squared_diag(
    c_a: &Array2<f64>,
    c_b: &Array2<f64>,
    s: &Array2<f64>,
    nocc_a: usize,
    nocc_b: usize,
) -> f64 {
    let s_true = 0.5 * (nocc_a as f64 - nocc_b as f64);
    let s_ideal = s_true * (s_true + 1.0);
    if nocc_a == 0 || nocc_b == 0 {
        return s_ideal;
    }
    let ca = c_a.slice(ndarray::s![.., ..nocc_a]);
    let cb = c_b.slice(ndarray::s![.., ..nocc_b]);
    let overlap_ab = ca.t().dot(s).dot(&cb);
    let sum_sq: f64 = overlap_ab.iter().map(|v| v * v).sum();
    s_ideal + (nocc_b as f64) - sum_sq
}

fn run_uhf(
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
    let cfg = UhfConfig {
        energy_conv: 1e-10,
        density_conv: 1e-8,
        max_iter: 200,
        ..Default::default()
    };
    let ctx = ParallelContext::default();
    let res = solve_uhf(&ctx, &mol, &prep, &bounds, &cfg).unwrap();
    (res, mol, prep)
}

fn maybe_check_energy(slug: &str, energy: f64, tol: f64) {
    let path = format!("../../testdata/reference/{slug}");
    if let Ok(text) = std::fs::read_to_string(&path) {
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        if let Some(ref_e) = v["energy"].as_f64() {
            assert!(
                (energy - ref_e).abs() < tol,
                "{slug}: got {:.10}, ref {:.10}",
                energy,
                ref_e
            );
        }
    } else {
        eprintln!("note: {slug} reference missing; energy observed = {:.10}", energy);
    }
}

#[test]
fn uhf_h_doublet_sto3g() {
    let (res, mol, prep) = run_uhf("1\nH\nH 0 0 0\n", 0, 2, "sto-3g");
    assert!(res.converged);
    let s = ferric_integrals::oneelectron::overlap(&prep);
    let nelec = mol.nelec() as i64;
    let two_s = mol.multiplicity as i64 - 1;
    let nocc_a = ((nelec + two_s) / 2) as usize;
    let nocc_b = ((nelec - two_s) / 2) as usize;
    let s2 = s_squared_diag(&res.mos_alpha, res.mos_beta.as_ref().unwrap(), &s, nocc_a, nocc_b);
    assert!((s2 - 0.75).abs() < 1e-8, "<S^2> = {}", s2);
    eprintln!("H/sto-3g  E = {:.10}  <S^2> = {:.6}", res.energy, s2);
    maybe_check_energy("h_sto-3g_uhf.json", res.energy, 1e-6);
}

#[test]
fn uhf_oh_doublet_ccpvdz() {
    // OH at r = 0.97 Å (Bohr ≈ 1.83); doublet.
    let xyz = "2\nOH\nO 0 0 0\nH 0 0 0.97\n";
    let (res, mol, prep) = run_uhf(xyz, 0, 2, "cc-pvdz");
    assert!(res.converged, "OH/cc-pvdz UHF did not converge");
    let s = ferric_integrals::oneelectron::overlap(&prep);
    let nelec = mol.nelec() as i64;
    let two_s = mol.multiplicity as i64 - 1;
    let nocc_a = ((nelec + two_s) / 2) as usize;
    let nocc_b = ((nelec - two_s) / 2) as usize;
    let s2 = s_squared_diag(&res.mos_alpha, res.mos_beta.as_ref().unwrap(), &s, nocc_a, nocc_b);
    eprintln!("OH/cc-pvdz E = {:.10}  <S^2> = {:.6}", res.energy, s2);
    // doublet ideal 0.75; allow up to 0.05 contamination
    assert!(s2 < 0.85 && s2 > 0.70, "<S^2> = {} out of range", s2);
    maybe_check_energy("oh_cc-pvdz_uhf.json", res.energy, 1e-6);
}

#[test]
fn uhf_ch3_doublet_ccpvdz() {
    // Planar CH3, doublet. Geometry: C at origin, three H at 1.079 Å, 120°.
    let r = 1.079;
    let (x1, y1) = (r, 0.0);
    let (x2, y2) = (-0.5 * r, r * (3f64).sqrt() * 0.5);
    let (x3, y3) = (-0.5 * r, -r * (3f64).sqrt() * 0.5);
    let xyz = format!(
        "4\nCH3\nC 0 0 0\nH {x1} {y1} 0\nH {x2:.6} {y2:.6} 0\nH {x3:.6} {y3:.6} 0\n"
    );
    let (res, mol, prep) = run_uhf(&xyz, 0, 2, "cc-pvdz");
    assert!(res.converged, "CH3/cc-pvdz UHF did not converge");
    let s = ferric_integrals::oneelectron::overlap(&prep);
    let nelec = mol.nelec() as i64;
    let two_s = mol.multiplicity as i64 - 1;
    let nocc_a = ((nelec + two_s) / 2) as usize;
    let nocc_b = ((nelec - two_s) / 2) as usize;
    let s2 = s_squared_diag(&res.mos_alpha, res.mos_beta.as_ref().unwrap(), &s, nocc_a, nocc_b);
    eprintln!("CH3/cc-pvdz E = {:.10}  <S^2> = {:.6}", res.energy, s2);
    assert!(s2 < 0.85 && s2 > 0.70, "<S^2> = {} out of range", s2);
    maybe_check_energy("ch3_cc-pvdz_uhf.json", res.energy, 1e-6);
}

fn fd_energy(mol: &Molecule, basis_name: &str) -> f64 {
    let bs = basis::bundled(basis_name).unwrap();
    let prep = PreparedBasis::new(mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let cfg = UhfConfig {
        energy_conv: 1e-12,
        density_conv: 1e-10,
        max_iter: 200,
        ..Default::default()
    };
    let ctx = ParallelContext::default();
    solve_uhf(&ctx, mol, &prep, &bounds, &cfg).unwrap().energy
}

#[test]
fn uhf_gradient_h_atom_fd() {
    // H atom: gradient should be (0,0,0) — trivial sanity.
    let mol = Molecule::parse_xyz("1\nH\nH 0 0 0\n", 0, 2).unwrap();
    let bs = basis::bundled("sto-3g").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let cfg = UhfConfig { energy_conv: 1e-12, ..Default::default() };
    let ctx = ParallelContext::default();
    let res = solve_uhf(&ctx, &mol, &prep, &bounds, &cfg).unwrap();
    let g = uhf_gradient(&mol, &prep, op, &bounds, &res).unwrap();
    for v in g.iter() {
        assert!(v.abs() < 1e-8, "H atom UHF gradient not zero: {}", v);
    }
}

#[test]
fn uhf_gradient_oh_sto3g_fd() {
    let xyz = "2\nOH\nO 0 0 0\nH 0 0 0.97\n";
    let mol = Molecule::parse_xyz(xyz, 0, 2).unwrap();
    let bs = basis::bundled("sto-3g").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let cfg = UhfConfig {
        energy_conv: 1e-12,
        density_conv: 1e-10,
        max_iter: 200,
        ..Default::default()
    };
    let ctx = ParallelContext::default();
    let res = solve_uhf(&ctx, &mol, &prep, &bounds, &cfg).unwrap();
    let analytic = uhf_gradient(&mol, &prep, op, &bounds, &res).unwrap();

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
    eprintln!("=== OH/STO-3G UHF gradient ===");
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
    assert!(max_diff < 1e-4, "UHF gradient FD mismatch: max diff = {:.2e}", max_diff);
}
