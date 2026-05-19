//! `eval_shell` (value-only) and `eval_shell_and_grad` (value+grad) must
//! produce identical χ for every basis function. Regression check covering
//! f-shells (def2-TZVP, aug-cc-pVTZ).

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_dft::ao_grid::{eval_basis_and_grad_on_points, eval_basis_on_points,
                          eval_basis_grad_hess_on_points};

fn check(label: &str, basis_name: &str) {
    let mol = Molecule::parse_xyz(
        "3\nH2O\nO 0 0 0\nH 0 0.7572 0.5868\nH 0 -0.7572 0.5868\n", 0, 1).unwrap();
    let bs = basis::bundled(basis_name).unwrap();
    let pts = vec![[0.3_f64, 0.2, 0.4], [-0.5, 0.6, 0.1], [1.1, -0.3, 0.5]];

    let chi_v = eval_basis_on_points(&mol, &bs, &pts).unwrap();
    let (chi_g, _) = eval_basis_and_grad_on_points(&mol, &bs, &pts).unwrap();
    let (chi_h, _, _) = eval_basis_grad_hess_on_points(&mol, &bs, &pts).unwrap();

    let mut max_vg = 0.0_f64;
    let mut max_vh = 0.0_f64;
    for ((idx, v), (_, vg)) in chi_v.indexed_iter().zip(chi_g.indexed_iter()) {
        max_vg = max_vg.max((v - vg).abs());
        let vh = chi_h[idx];
        max_vh = max_vh.max((v - vh).abs());
    }
    eprintln!("[{label}] |value-vs-grad-value| = {max_vg:.2e}, |value-vs-hess-value| = {max_vh:.2e}");
    assert!(max_vg < 1e-14, "{label}: value/grad value differ by {max_vg:.2e}");
    assert!(max_vh < 1e-14, "{label}: value/hess value differ by {max_vh:.2e}");
}

#[test] fn def2_tzvp() { check("def2-tzvp", "def2-tzvp"); }
#[test] fn aug_cc_pvtz() { check("aug-cc-pvtz", "aug-cc-pvtz"); }
