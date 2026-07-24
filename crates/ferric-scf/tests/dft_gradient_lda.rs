//! Validate the closed-shell LDA nuclear gradient against finite differences.
//!
//! The analytical gradient implements the "no grid response" approximation:
//! AO-basis-derivative + nuclear repulsion + 1e + 2e (no K). FD comparison
//! uses h = 5e-4 Bohr; analytical error should be ~5e-4 Ha/Bohr or better
//! (limited by grid-response neglect at standard (75, 110)).

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::ks_gradient::ks_gradient_closed;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;
use rayon::prelude::*;

fn rhf_cfg() -> RhfConfig {
    RhfConfig {
        xc: Some("LDA".into()),
        df_j_aux: Some("def2-universal-jkfit".into()),
        energy_conv: 1e-10,
        density_conv: 1e-8,
        ..Default::default()
    }
}

/// One-sided displacement's energy (a single independent RHF solve). Each
/// (atom, coord, sign) triple is fully independent of the others -- no shared
/// mutable state -- so the caller fans these out over rayon. Each individual
/// `solve_rhf` call already serializes its own BLAS calls internally
/// (opt_in_blas_threads' rayon-worker self-guard), so nesting this under an
/// outer rayon iterator is the same safe pattern the production code already
/// uses (e.g. rimp2.rs's per-pair parallelism over per-i BLAS3 GEMMs).
fn displaced_energy(mol: &Molecule, bs: &ferric_core::basis::BasisSet, cfg: &RhfConfig, atom: usize, coord: usize, delta: f64) -> f64 {
    let mut mol_d = mol.clone();
    match coord {
        0 => mol_d.atoms[atom].x += delta,
        1 => mol_d.atoms[atom].y += delta,
        _ => mol_d.atoms[atom].zpos += delta,
    }
    let prep = PreparedBasis::new(&mol_d, bs).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    solve_rhf(&ParallelContext::default(), &mol_d, &prep, Operator::coulomb(), &bounds, cfg)
        .unwrap()
        .energy
}

fn fd_gradient(xyz: &str, basis_name: &str, delta: f64) -> Array2<f64> {
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let natoms = mol.atoms.len();
    let cfg = rhf_cfg();
    let bs = basis::bundled(basis_name).unwrap();

    // Flatten to (atom, coord) pairs and run both displacements for each in
    // parallel -- 2*natoms*3 independent RHF solves total, previously serial.
    let pairs: Vec<(usize, usize)> = (0..natoms).flat_map(|a| (0..3).map(move |c| (a, c))).collect();
    let results: Vec<((usize, usize), f64)> = pairs
        .par_iter()
        .map(|&(atom, coord)| {
            let e_p = displaced_energy(&mol, &bs, &cfg, atom, coord, delta);
            let e_m = displaced_energy(&mol, &bs, &cfg, atom, coord, -delta);
            ((atom, coord), (e_p - e_m) / (2.0 * delta))
        })
        .collect();

    let mut grad = Array2::<f64>::zeros((natoms, 3));
    for ((atom, coord), g) in results {
        grad[(atom, coord)] = g;
    }
    grad
}

fn run_fd_test(label: &str, xyz: &str, basis_name: &str, tol: f64) {
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let bs = basis::bundled(basis_name).unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let cfg = rhf_cfg();
    let res = solve_rhf(&ParallelContext::default(), &mol, &prep, op, &bounds, &cfg).unwrap();

    let g_ana = ks_gradient_closed(&mol, &prep, &bs, op, &bounds, "LDA", &res, None).unwrap();
    let g_fd = fd_gradient(xyz, basis_name, 5e-4);

    eprintln!("=== {label} LDA gradient (analytic vs FD) ===");
    let mut max_diff = 0.0_f64;
    for a in 0..mol.atoms.len() {
        for c in 0..3 {
            let diff = (g_ana[(a, c)] - g_fd[(a, c)]).abs();
            max_diff = max_diff.max(diff);
            eprintln!(
                "  atom={a} coord={c}: ana={:+.6e} fd={:+.6e} diff={:.2e}",
                g_ana[(a, c)], g_fd[(a, c)], diff
            );
        }
    }
    eprintln!("  max diff: {max_diff:.2e}, tol: {tol:.0e}");
    for a in 0..mol.atoms.len() {
        for c in 0..3 {
            let diff = (g_ana[(a, c)] - g_fd[(a, c)]).abs();
            assert!(
                diff < tol,
                "{label} atom={a} coord={c}: ana={:+.6e} fd={:+.6e} diff={:.2e}",
                g_ana[(a, c)], g_fd[(a, c)], diff
            );
        }
    }
}

#[test]
fn lda_gradient_h2_sto3g_vs_fd() {
    run_fd_test("H2/sto-3g",
                "2\nH2\nH 0 0 0\nH 0 0 0.74\n",
                "sto-3g", 5e-4);
}

#[test]
fn lda_gradient_h2o_sto3g_vs_fd() {
    run_fd_test("H2O/sto-3g",
                "3\nH2O\nO 0 0 0\nH 0 0.7572 0.5868\nH 0 -0.7572 0.5868\n",
                "sto-3g", 1e-3);
}

#[test]
fn lda_gradient_h2_ccpvdz_vs_fd() {
    run_fd_test("H2/cc-pVDZ",
                "2\nH2\nH 0 0 0\nH 0 0 0.74\n",
                "cc-pvdz", 1e-3);
}
