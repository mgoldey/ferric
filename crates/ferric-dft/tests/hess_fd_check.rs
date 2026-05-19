//! Verify analytical AO Hessians match a finite-difference of the gradient.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_dft::ao_grid::{eval_basis_and_grad_on_points, eval_basis_grad_hess_on_points};

fn check_basis(label: &str, basis_name: &str) {
    let xyz = "3\nH2O\nO 0 0 0\nH 0 0.7572 0.5868\nH 0 -0.7572 0.5868\n";
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let bs = basis::bundled(basis_name).unwrap();

    // Probe a few off-atom points.
    let pts = vec![
        [0.3_f64, 0.2, 0.4],
        [-0.5, 0.6, 0.1],
        [1.1, -0.3, 0.5],
    ];
    let (_chi, _dchi, ddchi) = eval_basis_grad_hess_on_points(&mol, &bs, &pts).unwrap();
    let h = 1e-5_f64;

    let mut max_diff = 0.0_f64;
    for (g, p) in pts.iter().enumerate() {
        for b in 0..3 {
            let mut pp = *p; pp[b] += h;
            let mut pm = *p; pm[b] -= h;
            let (_chi_p, dchi_p) = eval_basis_and_grad_on_points(&mol, &bs, &[pp]).unwrap();
            let (_chi_m, dchi_m) = eval_basis_and_grad_on_points(&mol, &bs, &[pm]).unwrap();
            let nbf = dchi_p.dim().1;
            for a in 0..3 {
                for mu in 0..nbf {
                    let fd = (dchi_p[(a, mu, 0)] - dchi_m[(a, mu, 0)]) / (2.0 * h);
                    let ana = ddchi[(a, b, mu, g)];
                    let diff = (ana - fd).abs();
                    if diff > max_diff { max_diff = diff; }
                }
            }
        }
    }
    eprintln!("[{label}] max |Hess_ana - Hess_FD| = {max_diff:.2e}");
    assert!(max_diff < 1e-5, "{label}: Hessian FD mismatch {max_diff:.2e}");
}

#[test] fn hess_sto3g() { check_basis("sto-3g", "sto-3g"); }
#[test] fn hess_631g() { check_basis("6-31g",  "6-31g"); }
#[test] fn hess_ccpvdz() { check_basis("cc-pvdz", "cc-pvdz"); }
