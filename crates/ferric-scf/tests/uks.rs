//! UKS regression suite — open-shell DFT for radicals.
//!
//! PySCF references generated with `dft.UKS(mol, xc=...); grids.level = 5`,
//! see commit message for the exact harness.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::RhfConfig;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;

fn run(xc: &str, xyz: &str, mult: usize, basis_name: &str) -> f64 {
    let mol = Molecule::parse_xyz(xyz, 0, mult).unwrap();
    let bs = basis::bundled(basis_name).unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let cfg = RhfConfig {
        xc: Some(xc.into()),
        energy_conv: 1e-9,
        density_conv: 1e-7,
        max_iter: 200,
        ..Default::default()
    };
    let res = solve_uhf(&ParallelContext::default(), &mol, &prep, &bounds, &cfg).unwrap();
    eprintln!("UKS {xc} {xyz} (mult={mult}, {basis_name}): E = {:.8} ({} iter)",
              res.energy, res.iterations);
    res.energy
}

#[test]
fn uks_h_atom_sto3g_lda() {
    let e = run("LDA", "1\nH\nH 0 0 0\n", 2, "sto-3g");
    // PySCF UKS LDA H/sto-3g: -0.43567
    assert!((e - (-0.43567)).abs() < 5e-4, "got E = {e}");
}

#[test]
fn uks_h_atom_ccpvdz_lda() {
    let e = run("LDA", "1\nH\nH 0 0 0\n", 2, "cc-pvdz");
    // PySCF UKS LDA H/cc-pvdz: -0.47747
    assert!((e - (-0.47747)).abs() < 5e-4, "got E = {e}");
}

#[test]
fn uks_h_atom_ccpvdz_pbe() {
    let e = run("PBE", "1\nH\nH 0 0 0\n", 2, "cc-pvdz");
    // PySCF UKS PBE H/cc-pvdz: -0.49863
    assert!((e - (-0.49863)).abs() < 5e-4, "got E = {e}");
}

#[test]
fn uks_h_atom_ccpvdz_b3lyp() {
    let e = run("B3LYP", "1\nH\nH 0 0 0\n", 2, "cc-pvdz");
    // PySCF UKS B3LYP H/cc-pvdz: -0.50126
    assert!((e - (-0.50126)).abs() < 5e-4, "got E = {e}");
}

#[test]
fn uks_oh_doublet_ccpvdz_pbe() {
    // OH (²Π) bond length ≈ 0.97 Å
    let e = run("PBE", "2\nOH\nO 0 0 0\nH 0 0 0.97\n", 2, "cc-pvdz");
    // PySCF UKS PBE OH/cc-pvdz: -75.6449
    assert!((e - (-75.6449)).abs() < 1e-3, "got E = {e}");
}

#[test]
fn uks_h_atom_ccpvdz_wb97x_v() {
    // wB97X-V on a hydrogen atom: includes VV10 nonlocal correlation.
    let mol = Molecule::parse_xyz("1\nH\nH 0 0 0\n", 0, 2).unwrap();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let cfg = RhfConfig {
        xc: Some("wB97X-V".into()),
        energy_conv: 1e-9, density_conv: 1e-7,
        max_iter: 200, ..Default::default()
    };
    let res = solve_uhf(&ParallelContext::default(), &mol, &prep, &bounds, &cfg).unwrap();
    eprintln!("UKS wB97X-V H/cc-pVDZ: E = {:.8} ({} iter)", res.energy, res.iterations);
    // PySCF UKS wB97X-V H/cc-pvdz: -0.49962
    assert!((res.energy - (-0.49962)).abs() < 5e-4, "got E = {}", res.energy);
}

#[test]
fn uks_oh_doublet_ccpvdz_wb97x_v() {
    let mol = Molecule::parse_xyz("2\nOH\nO 0 0 0\nH 0 0 0.97\n", 0, 2).unwrap();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let cfg = RhfConfig {
        xc: Some("wB97X-V".into()),
        energy_conv: 1e-7, density_conv: 1e-5,
        max_iter: 400, ..Default::default()
    };
    let res = solve_uhf(&ParallelContext::default(), &mol, &prep, &bounds, &cfg).unwrap();
    eprintln!("UKS wB97X-V OH/cc-pVDZ: E = {:.8} ({} iter)", res.energy, res.iterations);
    // PySCF UKS wB97X-V OH/cc-pvdz: -75.70361
    assert!((res.energy - (-75.70361)).abs() < 2e-3, "got E = {}", res.energy);
}

#[test]
fn uks_oh_doublet_ccpvdz_b3lyp() {
    // OH (²Π) hybrid B3LYP. Tighten DIIS / iterations: hybrid Hartree-exchange
    // plus the spin-polarized GGA potential introduces a slower-converging
    // doublet near-degeneracy. PySCF takes ~25-40 iters with default settings.
    let xyz = "2\nOH\nO 0 0 0\nH 0 0 0.97\n";
    let mol = Molecule::parse_xyz(xyz, 0, 2).unwrap();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let cfg = RhfConfig {
        xc: Some("B3LYP".into()),
        energy_conv: 1e-7,
        density_conv: 1e-5,
        max_iter: 400,
        ..Default::default()
    };
    let res = solve_uhf(&ParallelContext::default(), &mol, &prep, &bounds, &cfg).unwrap();
    eprintln!("UKS B3LYP OH/cc-pvdz: E = {:.8} ({} iter)", res.energy, res.iterations);
    assert!((res.energy - (-75.7319)).abs() < 1e-3, "got E = {}", res.energy);
}
