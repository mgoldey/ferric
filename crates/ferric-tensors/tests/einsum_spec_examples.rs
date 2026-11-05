//! Every example on `site/src/reference/einsum.md` ("Einsum spec grammar"),
//! with hand-computed expected values. The page cites this file; the last
//! test fails if a spec listed here is missing from the page.

use ferric_tensors::{einsum, einsum_dyn, TensorError};
use ndarray::{ArrayD, IxDyn};

fn arr(shape: &[usize], data: &[f64]) -> ArrayD<f64> {
    ArrayD::from_shape_vec(IxDyn(shape), data.to_vec()).unwrap()
}

/// A = [[1,2,3],[4,5,6]]
fn a23() -> ArrayD<f64> {
    arr(&[2, 3], &[1., 2., 3., 4., 5., 6.])
}

/// B = [[1,0],[0,1],[1,1]]
fn b32() -> ArrayD<f64> {
    arr(&[3, 2], &[1., 0., 0., 1., 1., 1.])
}

fn dynamic(spec: &str, a: &ArrayD<f64>, b: &ArrayD<f64>, scale: f64) -> ArrayD<f64> {
    einsum_dyn(spec, a.view(), b.view(), scale).unwrap()
}

#[test]
fn matrix_product_and_spaces() {
    let want = arr(&[2, 2], &[4., 5., 10., 11.]);
    assert_eq!(dynamic("ij,jk->ik", &a23(), &b32(), 1.0), want);
    assert_eq!(dynamic("ij , jk -> ik", &a23(), &b32(), 1.0), want);
    // Index letters are labels only: any distinct characters work.
    assert_eq!(dynamic("ab,bc->ac", &a23(), &b32(), 1.0), want);
    // Macro and run-time entry points agree.
    let m: ArrayD<f64> = einsum!("ij,jk->ik", &a23(), &b32());
    assert_eq!(m, want);
}

#[test]
fn scale_factor() {
    let want = arr(&[2, 2], &[-2., -2.5, -5., -5.5]);
    assert_eq!(dynamic("ij,jk->ik", &a23(), &b32(), -0.5), want);
    let m: ArrayD<f64> = einsum!("ij,jk->ik", &a23(), &b32(), -0.5);
    assert_eq!(m, want);
}

#[test]
fn scalar_output() {
    let d = dynamic("ij,ij->", &a23(), &a23(), 1.0);
    assert_eq!(d.ndim(), 0);
    assert_eq!(d[IxDyn(&[])], 91.0);
    let s: f64 = einsum!("ij,ij->", &a23(), &a23());
    assert_eq!(s, 91.0);
    let s: f64 = einsum!("ij,ij->", &a23(), &a23(), 2.0);
    assert_eq!(s, 182.0);
}

#[test]
fn matrix_vector_and_outer_product() {
    let v = arr(&[3], &[1., 1., 1.]);
    assert_eq!(dynamic("ij,j->i", &a23(), &v, 1.0), arr(&[2], &[6., 15.]));
    let (u, w) = (arr(&[2], &[1., 2.]), arr(&[3], &[3., 4., 5.]));
    assert_eq!(
        dynamic("i,j->ij", &u, &w, 1.0),
        arr(&[2, 3], &[3., 4., 5., 6., 8., 10.])
    );
}

#[test]
fn batch_indices_are_elementwise() {
    // Hadamard product: i and j are in both inputs and the output.
    assert_eq!(
        dynamic("ij,ij->ij", &a23(), &a23(), 1.0),
        arr(&[2, 3], &[1., 4., 9., 16., 25., 36.])
    );
    // Per-element dot over k: k is contracted, i and j are batch.
    let t = arr(&[2, 1, 3], &[1., 2., 3., 4., 5., 6.]);
    assert_eq!(
        dynamic("kij,kij->ij", &t, &t, 1.0),
        arr(&[1, 3], &[17., 29., 45.])
    );
}

#[test]
fn output_order_within_a_group() {
    // Free indices of the left operand may be listed in any order: i,j -> j,i.
    let a = arr(&[1, 2, 2], &[1., 2., 3., 4.]);
    let b = arr(&[2, 1], &[1., 1.]);
    let want = arr(&[2, 1, 1], &[3., 7.]);
    assert_eq!(dynamic("ijk,kl->jil", &a, &b, 1.0), want);
    let m: ArrayD<f64> = einsum!("ijk,kl->jil", &a, &b);
    assert_eq!(m, want);
}

#[test]
fn malformed_specs_are_errors() {
    let (a, b) = (a23(), b32());
    let cases: &[(&str, &str)] = &[
        ("ij,jk", "einsum spec must contain '->'"),
        ("ij->ij", "einsum spec must have exactly two comma-separated inputs"),
        ("ij,jk,kl->il", "einsum is binary: exactly two inputs"),
        (
            "ii,jk->jk",
            "index 'i' is repeated in the left operand (traces / diagonals of one operand are not supported)",
        ),
        (
            "ij,jk->kk",
            "index 'k' is repeated in the output (traces / diagonals of one operand are not supported)",
        ),
        (
            "ij,jk->k",
            "index 'i' appears in an input but is neither contracted nor in the output (implicit single-operand sums are not supported)",
        ),
        (
            "ij,jk->ikz",
            "output index 'z' is not a free or batch index of either input",
        ),
        (
            "ij,jk->ki",
            "output indices must be listed as (batch indices, then left-operand free indices, then right-operand free indices) to match the runtime axis order",
        ),
    ];
    for &(spec, msg) in cases {
        match einsum_dyn(spec, a.view(), b.view(), 1.0) {
            Err(TensorError::Spec(got)) => assert_eq!(got, msg, "{spec}"),
            other => panic!("{spec}: expected Spec error, got {other:?}"),
        }
    }
}

#[test]
fn operand_mismatches_are_errors_at_run_time() {
    let (a, b) = (a23(), b32());
    // Rank disagrees with the spec term.
    let err = einsum_dyn("ijk,jk->i", a.view(), b.view(), 1.0).unwrap_err();
    assert_eq!(
        err.to_string(),
        "einsum operand mismatch: left operand has 2 axes, spec 'ijk,jk->i' gives it 3"
    );
    // Shared index with different extents (A is 2x3, A again contracts 3 with 2).
    let err = einsum_dyn("ij,jk->ik", a.view(), a.view(), 1.0).unwrap_err();
    assert_eq!(
        err.to_string(),
        "einsum operand mismatch: shared index at left axis 1 has extent 3 but right axis 0 has 2"
    );
}

#[test]
fn zero_extent_operand_gives_zeros() {
    let z = ArrayD::<f64>::zeros(IxDyn(&[0, 3]));
    let out = dynamic("ij,jk->ik", &z, &b32(), 1.0);
    assert_eq!(out.shape(), &[0, 2]);
    let out = dynamic("ij,jk->ik", &a23(), &ArrayD::zeros(IxDyn(&[3, 0])), 1.0);
    assert_eq!(out.shape(), &[2, 0]);
}

/// Every spec exercised above appears on the page, so an example cannot be
/// removed from the tests while the page still shows it.
#[test]
fn page_lists_every_tested_spec() {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../site/src/reference/einsum.md");
    let Ok(page) = std::fs::read_to_string(&path) else {
        eprintln!("{} not found (packaged crate); skipped", path.display());
        return;
    };
    for spec in [
        "ij,jk->ik",
        "ij , jk -> ik",
        "ab,bc->ac",
        "ij,ij->",
        "ij,j->i",
        "i,j->ij",
        "ij,ij->ij",
        "kij,kij->ij",
        "ijk,kl->jil",
        "ij,jk",
        "ij->ij",
        "ij,jk,kl->il",
        "ii,jk->jk",
        "ij,jk->kk",
        "ij,jk->k",
        "ij,jk->ikz",
        "ij,jk->ki",
    ] {
        assert!(page.contains(&format!("`{spec}`")), "page misses `{spec}`");
    }
    assert!(page.contains("einsum_spec_examples.rs"));
}
