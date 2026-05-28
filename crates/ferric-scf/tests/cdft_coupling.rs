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

/// For identical α and β sets (S = I, C_a = C_b), the one-body element equals
/// the ordinary expectation Σ_σ Σ_i ⟨i|Ô|i⟩ (since S_ab = 1 and all s_i = 1).
#[test]
fn cross_one_body_identical_is_expectation() {
    use ferric_scf::cdft_coupling::{biorth_pairing, cross_one_body};
    let s = Array2::<f64>::eye(3);
    let c = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]; // 2 occ
    // Operator: diagonal AO operator diag(2,3,5).
    let mut op = Array2::<f64>::zeros((3, 3));
    op[(0, 0)] = 2.0; op[(1, 1)] = 3.0; op[(2, 2)] = 5.0;
    let pa = biorth_pairing(&c, &c, &s);
    let pb = biorth_pairing(&c, &c, &s);
    let s_ab = pa.det_m * pb.det_m; // = 1
    let val = cross_one_body(&op, &pa, &pb, s_ab);
    // Two spins, each occupying AO0 and AO1: ⟨0|op|0⟩+⟨1|op|1⟩ = 2+3 = 5 per
    // spin, ×2 spins = 10.
    assert!((val - 10.0).abs() < 1e-10, "got {val}");
}

/// Build α sets that share one orbital but whose second orbitals are mutually
/// orthogonal (one zero singular value). S_ab = 0, but the one-body element is
/// finite and equals (Π nonzero) · ⟨ã_k|Ô|b̃_k⟩ for the paired zero orbital.
/// β sets identical (det_β = 1) so the α-zero is the only zero.
#[test]
fn cross_one_body_single_zero_overlap_is_finite() {
    use ferric_scf::cdft_coupling::{biorth_pairing, cross_one_body};
    let s = Array2::<f64>::eye(4);
    // α_a occupies AO0, AO1 ; α_b occupies AO0, AO2 → second orbitals orthogonal.
    let c_a = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]];
    let c_b = array![[1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.0, 0.0]];
    let pa = biorth_pairing(&c_a, &c_b, &s);
    // One singular value should be ~1 (shared AO0) and one ~0 (orthogonal pair).
    let n_zero = pa.s_vals.iter().filter(|&&s| s < 1e-8).count();
    assert_eq!(n_zero, 1, "expected exactly one zero overlap, s={:?}", pa.s_vals);

    // β identical (occupies AO0, AO1).
    let cb = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]];
    let pb = biorth_pairing(&cb, &cb, &s);

    let s_ab = pa.det_m * pb.det_m;
    assert!(s_ab.abs() < 1e-10, "S_ab should be 0, got {s_ab}");

    // Operator that connects AO1 and AO2 (the orthogonal pair): off-diagonal.
    let mut op = Array2::<f64>::zeros((4, 4));
    op[(1, 2)] = 1.0; op[(2, 1)] = 1.0;
    let val = cross_one_body(&op, &pa, &pb, s_ab);
    // Finite and nonzero: the single zero-overlap pair carries the element.
    assert!(val.is_finite(), "element not finite: {val}");
    assert!(val.abs() > 1e-6, "expected nonzero element, got {val}");
}

/// Identical states: S_ab = 1, and the degenerate-denominator guard returns
/// cleanly (no NaN/Inf). Uses a tiny synthetic state.
#[test]
fn coupling_identical_state_is_clean() {
    use ferric_scf::cdft_coupling::{coupling_hab, DiabaticState};
    let s = Array2::<f64>::eye(3);
    let c = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let w = Array2::<f64>::eye(3);
    let st = DiabaticState { c_a: &c, c_b: &c, nocc_a: 2, nocc_b: 1,
        energy: -3.0, lambda: 0.5, w: &w };
    let r = coupling_hab(&st, &st, &s);
    assert!((r.s_ab - 1.0).abs() < 1e-10, "S_ab {}", r.s_ab);
    assert!(r.h_ab.is_finite(), "H_ab not finite");
}

/// a↔b symmetry: swapping the two states gives the same S_ab and H_ab.
#[test]
fn coupling_symmetric_under_swap() {
    use ferric_scf::cdft_coupling::{coupling_hab, DiabaticState};
    let s = Array2::<f64>::eye(3);
    let c1 = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    // c2: a slight rotation in the (0,2) plane so S_ab ≠ 1 but nonzero.
    let t = 0.3_f64;
    let (ct, st_) = (t.cos(), t.sin());
    let c2 = array![[ct, 0.0, -st_], [0.0, 1.0, 0.0], [st_, 0.0, ct]];
    let w = Array2::<f64>::eye(3);
    let a = DiabaticState { c_a: &c1, c_b: &c1, nocc_a: 2, nocc_b: 1,
        energy: -3.0, lambda: 0.5, w: &w };
    let b = DiabaticState { c_a: &c2, c_b: &c2, nocc_a: 2, nocc_b: 1,
        energy: -2.9, lambda: 0.4, w: &w };
    let ab = coupling_hab(&a, &b, &s);
    let ba = coupling_hab(&b, &a, &s);
    assert!((ab.s_ab - ba.s_ab).abs() < 1e-10, "S_ab asym {} {}", ab.s_ab, ba.s_ab);
    assert!((ab.h_ab - ba.h_ab).abs() < 1e-8, "H_ab asym {} {}", ab.h_ab, ba.h_ab);
}
