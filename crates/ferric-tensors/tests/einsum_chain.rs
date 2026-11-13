//! `einsum_chain_dyn`: a fixed left-to-right chain of binary contractions.
//!
//! Anchors: a chain equals the same binary `einsum_dyn` calls made by hand,
//! bitwise (the chain adds no arithmetic of its own); two operands are
//! `einsum_dyn` exactly; refusals name the step.

use ferric_tensors::{einsum_chain_dyn, einsum_dyn, TensorError};
use ndarray::{ArrayD, IxDyn};

fn fill(shape: &[usize], seed: u64) -> ArrayD<f64> {
    let mut x = seed;
    ArrayD::from_shape_fn(IxDyn(shape), |_| {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((x >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
    })
}

#[test]
fn chain_is_the_hand_written_binary_calls_bitwise() {
    let (a, b, c) = (fill(&[6, 9], 1), fill(&[9, 5], 2), fill(&[5, 4], 3));
    let ab = einsum_dyn("ij,jk->ik", a.view(), b.view(), 1.0).unwrap();
    let want = einsum_dyn("ik,kl->il", ab.view(), c.view(), -0.5).unwrap();
    let got = einsum_chain_dyn("ij,jk,kl->il", &[a.view(), b.view(), c.view()], -0.5).unwrap();
    assert_eq!(got, want);
}

#[test]
fn a_later_operand_keeps_a_shared_index_as_batch_for_the_step() {
    // j is in all three terms: step one keeps it (batch), step two sums it.
    let (a, b, c) = (fill(&[3, 7], 4), fill(&[7, 2], 5), fill(&[7, 5], 6));
    let ab = einsum_dyn("ij,jk->jik", a.view(), b.view(), 1.0);
    // Batch indices lead the step output, so the intermediate is "jik".
    let ab = ab.unwrap();
    let want = einsum_dyn("jik,jl->ikl", ab.view(), c.view(), 1.0).unwrap();
    let got = einsum_chain_dyn("ij,jk,jl->ikl", &[a.view(), b.view(), c.view()], 1.0).unwrap();
    assert_eq!(got, want);
}

#[test]
fn two_operands_are_einsum_dyn_exactly() {
    let (a, b) = (fill(&[8, 33], 7), fill(&[33, 6], 8));
    let want = einsum_dyn("ij,jk->ik", a.view(), b.view(), 2.0).unwrap();
    let got = einsum_chain_dyn("ij,jk->ik", &[a.view(), b.view()], 2.0).unwrap();
    assert_eq!(got, want);
}

#[test]
fn refusals_name_the_step_or_the_count() {
    let (a, b, c) = (fill(&[2, 3], 1), fill(&[3, 4], 2), fill(&[4, 5], 3));
    let v = [a.view(), b.view(), c.view()];
    // i is carried by no later operand and not by the output: a single-operand sum.
    let e = einsum_chain_dyn("ij,jk,kl->l", &v, 1.0).unwrap_err();
    assert!(
        matches!(e, TensorError::Spec(_)) && e.to_string().contains("step 1"),
        "{e}"
    );
    // The last step lists the output out of group order.
    let e = einsum_chain_dyn("ij,jk,kl->li", &v, 1.0).unwrap_err();
    assert!(e.to_string().contains("step 2"), "{e}");
    let e = einsum_chain_dyn("ij,jk,kl->il", &v[..2], 1.0).unwrap_err();
    assert!(
        e.to_string().contains("3 input terms but 2 operands"),
        "{e}"
    );
    let e = einsum_chain_dyn("ij,jk,kl->il", &v[..1], 1.0).unwrap_err();
    assert!(
        e.to_string().contains("3 input terms but 1 operands"),
        "{e}"
    );
}
