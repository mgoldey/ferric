use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_dft::ao_grid::{eval_basis_and_grad_on_points, eval_basis_on_points};

#[test]
fn gradient_matches_central_difference_h2() {
    let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 1.4\n", 0, 1).unwrap();
    let bs = basis::bundled("cc-pvdz").unwrap();

    // A random test point well away from any atom (avoid the cusp).
    let p = [[0.2_f64, -0.3, 0.7]];
    let (_chi, dchi) = eval_basis_and_grad_on_points(&mol, &bs, &p).unwrap();

    let eps = 1e-5_f64;
    for axis in 0..3 {
        let mut p_plus = p;
        let mut p_minus = p;
        p_plus[0][axis] += eps;
        p_minus[0][axis] -= eps;
        let chi_plus = eval_basis_on_points(&mol, &bs, &p_plus).unwrap();
        let chi_minus = eval_basis_on_points(&mol, &bs, &p_minus).unwrap();
        for mu in 0..chi_plus.nrows() {
            let fd = (chi_plus[(mu, 0)] - chi_minus[(mu, 0)]) / (2.0 * eps);
            let analytical = dchi[(axis, mu, 0)];
            let err = (fd - analytical).abs();
            assert!(err < 1e-6,
                "axis={axis}, mu={mu}: analytical={analytical:.8e}, fd={fd:.8e}, err={err:.2e}");
        }
    }
}

#[test]
fn gradient_matches_central_difference_h2o_covers_d_shells() {
    let mol = Molecule::parse_xyz(
        "3\nH2O\nO 0 0 0\nH 0 0.7572 0.5868\nH 0 -0.7572 0.5868\n", 0, 1).unwrap();
    let bs = basis::bundled("cc-pvdz").unwrap();

    // Test point near oxygen so d-shells contribute non-trivially, but
    // not directly on the nucleus.
    let p = [[0.15_f64, 0.1, -0.2]];
    let (_chi, dchi) = eval_basis_and_grad_on_points(&mol, &bs, &p).unwrap();

    let eps = 1e-5_f64;
    for axis in 0..3 {
        let mut p_plus = p;
        let mut p_minus = p;
        p_plus[0][axis] += eps;
        p_minus[0][axis] -= eps;
        let chi_plus = eval_basis_on_points(&mol, &bs, &p_plus).unwrap();
        let chi_minus = eval_basis_on_points(&mol, &bs, &p_minus).unwrap();
        for mu in 0..chi_plus.nrows() {
            let fd = (chi_plus[(mu, 0)] - chi_minus[(mu, 0)]) / (2.0 * eps);
            let analytical = dchi[(axis, mu, 0)];
            let err = (fd - analytical).abs();
            assert!(err < 1e-6,
                "axis={axis}, mu={mu}: analytical={analytical:.8e}, fd={fd:.8e}, err={err:.2e}");
        }
    }
}

#[test]
fn gradient_matches_central_difference_h2o_covers_f_shells() {
    // def2-TZVP has f shells on O and aug-cc-pVTZ has f on every heavy atom
    // (pure d/f via JSON). Probe a point off the cusp so radial derivatives
    // are well-conditioned, near oxygen so f-functions are non-trivial.
    let mol = Molecule::parse_xyz(
        "3\nH2O\nO 0 0 0\nH 0 0.7572 0.5868\nH 0 -0.7572 0.5868\n", 0, 1).unwrap();
    for basis_name in ["def2-tzvp", "aug-cc-pvtz"] {
        let bs = basis::bundled(basis_name).unwrap();
        let p = [[0.15_f64, 0.1, -0.2]];
        let (_chi, dchi) = eval_basis_and_grad_on_points(&mol, &bs, &p).unwrap();
        let eps = 1e-5_f64;
        let mut max_err = 0.0_f64;
        for axis in 0..3 {
            let mut p_plus = p;
            let mut p_minus = p;
            p_plus[0][axis] += eps;
            p_minus[0][axis] -= eps;
            let chi_plus = eval_basis_on_points(&mol, &bs, &p_plus).unwrap();
            let chi_minus = eval_basis_on_points(&mol, &bs, &p_minus).unwrap();
            for mu in 0..chi_plus.nrows() {
                let fd = (chi_plus[(mu, 0)] - chi_minus[(mu, 0)]) / (2.0 * eps);
                let analytical = dchi[(axis, mu, 0)];
                let err = (fd - analytical).abs();
                if err > max_err { max_err = err; }
            }
        }
        eprintln!("[{basis_name}] max |grad_ana - grad_FD| = {max_err:.2e}");
        assert!(max_err < 1e-5, "{basis_name}: AO grad FD mismatch {max_err:.2e}");
    }
}
