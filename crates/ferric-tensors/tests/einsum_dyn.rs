//! `einsum_dyn` (run-time spec) against a naive index loop, and against the
//! `einsum!` macro, which parses with the same `ferric_einsum_spec`.

use ferric_tensors::{einsum, einsum_dyn, TensorError};
use ndarray::{ArrayD, IxDyn};
use std::collections::BTreeMap;

/// Deterministic, non-symmetric fill so a transposed axis cannot pass.
fn filled(shape: &[usize], seed: f64) -> ArrayD<f64> {
    let mut k = 0.0;
    ArrayD::from_shape_fn(IxDyn(shape), |_| {
        k += 1.0;
        (seed * k).sin() + 0.1 * k
    })
}

/// Reference: loop over every assignment of every index letter.
fn naive(spec: &str, a: &ArrayD<f64>, b: &ArrayD<f64>, scale: f64) -> ArrayD<f64> {
    let (ins, out) = spec.split_once("->").unwrap();
    let (la, lb) = ins.split_once(',').unwrap();
    let mut ext = BTreeMap::new();
    for (term, arr) in [(la, a), (lb, b)] {
        for (c, &n) in term.chars().zip(arr.shape()) {
            ext.insert(c, n);
        }
    }
    let letters: Vec<char> = ext.keys().copied().collect();
    let out_shape: Vec<usize> = out.chars().map(|c| ext[&c]).collect();
    let mut res = ArrayD::<f64>::zeros(IxDyn(&out_shape));
    let total: usize = letters.iter().map(|c| ext[c]).product();
    for flat in 0..total {
        let mut rem = flat;
        let mut val = BTreeMap::new();
        for c in letters.iter().rev() {
            val.insert(*c, rem % ext[c]);
            rem /= ext[c];
        }
        let idx = |t: &str| -> Vec<usize> { t.chars().map(|c| val[&c]).collect() };
        let v = a[IxDyn(&idx(la))] * b[IxDyn(&idx(lb))];
        res[IxDyn(&idx(out))] += scale * v;
    }
    res
}

fn assert_close(x: &ArrayD<f64>, y: &ArrayD<f64>) {
    assert_eq!(x.shape(), y.shape());
    for (p, q) in x.iter().zip(y.iter()) {
        assert!((p - q).abs() <= 1e-12 * (1.0 + q.abs()), "{p} vs {q}");
    }
}

#[test]
fn matches_naive_loop_across_spec_shapes() {
    let cases: &[(&str, &[usize], &[usize])] = &[
        ("ij,jk->ik", &[3, 4], &[4, 5]),
        ("ijcd,abcd->ijab", &[2, 3, 4, 5], &[3, 2, 4, 5]),
        ("ij,ij->", &[3, 4], &[3, 4]),
        ("ij,kl->ijkl", &[2, 3], &[4, 2]),
        ("kij,kij->ij", &[5, 3, 4], &[5, 3, 4]),
        ("pij,qij->pq", &[3, 4, 5], &[2, 4, 5]),
        // Free and batch indices listed out of operand order.
        ("ijk,kl->jil", &[2, 3, 4], &[4, 5]),
        ("kij,kji->ji", &[4, 2, 3], &[4, 3, 2]),
        ("iak,kbj->iajb", &[2, 3, 4], &[4, 5, 2]),
    ];
    for &(spec, sa, sb) in cases {
        let a = filled(sa, 0.37);
        let b = filled(sb, 1.13);
        let got = einsum_dyn(spec, a.view(), b.view(), -0.5).unwrap();
        assert_close(&got, &naive(spec, &a, &b, -0.5));
    }
}

#[test]
fn macro_and_dyn_agree_bit_for_bit() {
    let a = filled(&[2, 3, 4], 0.37);
    let b = filled(&[4, 5], 1.13);
    let m = einsum!("ijk,kl->jil", &a, &b);
    let d = einsum_dyn("ijk,kl->jil", a.view(), b.view(), 1.0).unwrap();
    assert_eq!(m, d);
    // The macro's out-of-order output was laid out in operand order before
    // the shared parser; pin it against the naive loop too.
    assert_close(&m, &naive("ijk,kl->jil", &a, &b, 1.0));
}

#[test]
fn reports_mismatches_as_errors() {
    let a = filled(&[3, 4], 0.37);
    let b = filled(&[5, 6], 1.13);
    assert!(matches!(
        einsum_dyn("ij,jk->ik", a.view(), b.view(), 1.0),
        Err(TensorError::Operand(_))
    ));
    assert!(matches!(
        einsum_dyn("ijk,jk->i", a.view(), b.view(), 1.0),
        Err(TensorError::Operand(_))
    ));
    assert!(matches!(
        einsum_dyn("ii,ij->j", a.view(), b.view(), 1.0),
        Err(TensorError::Spec(_))
    ));
}

#[test]
fn zero_extent_operands_give_zeros_not_a_panic() {
    // Empty free, contracted and batch axes, and the scalar output.
    let cases: &[(&str, &[usize], &[usize])] = &[
        ("ij,jk->ik", &[0, 3], &[3, 4]),
        ("ij,jk->ik", &[2, 0], &[0, 4]),
        ("ij,jk->ik", &[2, 3], &[3, 0]),
        ("kij,kij->ij", &[0, 2, 3], &[0, 2, 3]),
        ("ij,ij->", &[0, 3], &[0, 3]),
        ("ij,kl->ijkl", &[2, 0], &[3, 1]),
    ];
    for &(spec, sa, sb) in cases {
        let a = filled(sa, 0.37);
        let b = filled(sb, 1.13);
        let got = einsum_dyn(spec, a.view(), b.view(), -0.5).unwrap();
        assert_close(&got, &naive(spec, &a, &b, -0.5));
    }
}
