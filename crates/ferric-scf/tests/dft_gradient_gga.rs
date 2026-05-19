//! FD validation of the PBE (GGA) and B3LYP (hybrid GGA) closed-shell nuclear
//! gradients. AO Hessians are only implemented for s/p shells, so these tests
//! restrict to basis sets without d functions (STO-3G, 6-31G).

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::ks_gradient::ks_gradient_closed;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;

fn rhf_cfg(xc: &str, hybrid: bool) -> RhfConfig {
    RhfConfig {
        xc: Some(xc.into()),
        df_j_aux: Some("def2-universal-jkfit".into()),
        df_k_aux: if hybrid { Some("def2-universal-jkfit".into()) } else { None },
        energy_conv: 1e-10,
        density_conv: 1e-8,
        ..Default::default()
    }
}

fn fd_gradient(xyz: &str, basis_name: &str, xc: &str, hybrid: bool, delta: f64) -> Array2<f64> {
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let natoms = mol.atoms.len();
    let mut grad = Array2::<f64>::zeros((natoms, 3));
    let cfg = rhf_cfg(xc, hybrid);
    let bs = basis::bundled(basis_name).unwrap();
    for atom in 0..natoms {
        for coord in 0..3 {
            let mut mol_p = mol.clone();
            let mut mol_m = mol.clone();
            match coord {
                0 => { mol_p.atoms[atom].x += delta; mol_m.atoms[atom].x -= delta; }
                1 => { mol_p.atoms[atom].y += delta; mol_m.atoms[atom].y -= delta; }
                _ => { mol_p.atoms[atom].zpos += delta; mol_m.atoms[atom].zpos -= delta; }
            }
            let prep_p = PreparedBasis::new(&mol_p, &bs).unwrap();
            let bounds_p = SchwarzBounds::compute(Operator::coulomb(), &prep_p).unwrap();
            let res_p = solve_rhf(
                &ParallelContext::default(), &mol_p, &prep_p, Operator::coulomb(), &bounds_p, &cfg,
            ).unwrap();
            let prep_m = PreparedBasis::new(&mol_m, &bs).unwrap();
            let bounds_m = SchwarzBounds::compute(Operator::coulomb(), &prep_m).unwrap();
            let res_m = solve_rhf(
                &ParallelContext::default(), &mol_m, &prep_m, Operator::coulomb(), &bounds_m, &cfg,
            ).unwrap();
            grad[(atom, coord)] = (res_p.energy - res_m.energy) / (2.0 * delta);
        }
    }
    grad
}

fn run_fd_test(label: &str, xyz: &str, basis_name: &str, xc: &str, hybrid: bool, tol: f64) {
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let bs = basis::bundled(basis_name).unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let cfg = rhf_cfg(xc, hybrid);
    let res = solve_rhf(&ParallelContext::default(), &mol, &prep, op, &bounds, &cfg).unwrap();

    let g_ana = ks_gradient_closed(&mol, &prep, &bs, op, &bounds, xc, &res).unwrap();
    let g_fd = fd_gradient(xyz, basis_name, xc, hybrid, 5e-4);

    eprintln!("=== {label} {xc} gradient (analytic vs FD) ===");
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
            assert!(diff < tol, "{label} {xc}: atom={a} coord={c} diff={diff:.2e}");
        }
    }
}

#[test]
fn pbe_gradient_h2_sto3g_vs_fd() {
    run_fd_test("H2/sto-3g", "2\nH2\nH 0 0 0\nH 0 0 0.74\n", "sto-3g", "PBE", false, 1e-3);
}

#[test]
fn pbe_gradient_h2o_sto3g_vs_fd() {
    run_fd_test("H2O/sto-3g",
                "3\nH2O\nO 0 0 0\nH 0 0.7572 0.5868\nH 0 -0.7572 0.5868\n",
                "sto-3g", "PBE", false, 2e-3);
}

#[test]
fn b3lyp_gradient_h2_sto3g_vs_fd() {
    run_fd_test("H2/sto-3g", "2\nH2\nH 0 0 0\nH 0 0 0.74\n", "sto-3g", "B3LYP", true, 1e-3);
}
