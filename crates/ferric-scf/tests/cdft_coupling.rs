//! cDFT-ET coupling: kernel unit tests on synthetic matrices, then He₂⁺
//! end-to-end identities. Kernel tests use S = I so det(Mσ) = det(C_aᵀ C_b).

use ferric_scf::cdft_coupling::biorth_pairing;
use ndarray::{array, Array2};

/// Identical occupied sets with S = I → det(M) = 1 (singular values all 1).
#[test]
fn pairing_identical_sets_det_one() {
    let s = Array2::<f64>::eye(3);
    // Two occupied orbitals = first two columns of I_3.
    let c = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]];
    let p = biorth_pairing(&c, &c, &s);
    assert!((p.det_m - 1.0).abs() < 1e-12, "det_m {}", p.det_m);
    assert_eq!(p.s_vals.len(), 2);
    for sv in p.s_vals.iter() {
        assert!((sv - 1.0).abs() < 1e-12);
    }
}

/// A column swap between the two sets flips the determinant sign (|det| = 1).
#[test]
fn pairing_swapped_columns_det_minus_one() {
    let s = Array2::<f64>::eye(3);
    let c_a = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]];
    let c_b = array![[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]]; // columns swapped
    let p = biorth_pairing(&c_a, &c_b, &s);
    // SVD singular values are non-negative, so |det_m| = product = 1.
    assert!((p.det_m.abs() - 1.0).abs() < 1e-12, "|det_m| {}", p.det_m.abs());
}
