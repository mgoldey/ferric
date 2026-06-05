//! Runtime engine for a binary tensor contraction.
//!
//! [`einsum_binary_batched`] is what the `einsum!` macro lowers to;
//! [`einsum_binary`] is the no-batch, unit-scale special case. Given two
//! operands and the positions of their batch / free / contracted axes, the
//! engine permutes each operand so those axis groups are contiguous, reshapes,
//! calls GEMM (`general_mat_mul` — one call, or one per batch slice when there
//! are batch axes), applies the scale factor, and reshapes the result to the
//! requested output shape. Any required permutation copy is logged at debug
//! level so hot transposes are discoverable.

use ndarray::linalg::general_mat_mul;
use ndarray::{ArrayD, ArrayViewD, IxDyn};
use thiserror::Error;

/// Errors from a contraction.
#[derive(Debug, Error)]
pub enum TensorError {
    /// A contracted axis had different lengths in the two operands.
    #[error("contracted-dimension mismatch: left {left} vs right {right}")]
    ContractedDimMismatch { left: usize, right: usize },
    /// The computed 2D product shape could not be reshaped to the output shape.
    #[error("output reshape failed: product has {got} elements, output shape needs {want}")]
    OutputReshape { got: usize, want: usize },
}

/// Contract two operands over their contracted axes into one GEMM.
///
/// - `*_free`: axis positions (into the operand) that survive into the output.
/// - `*_contr`: axis positions that are summed over. Must be the same logical
///   axes in both operands (the macro guarantees ordering matches).
/// - `out_shape`: the shape of the result (free_left dims then free_right dims).
///   Empty slice means a scalar (returned as a length-1 `ArrayD` of shape `[]`).
///
/// This is the plain (no batch axes, unit scale) entry point. See
/// [`einsum_binary_batched`] for diagonal/batch axes and a scale factor.
pub fn einsum_binary(
    left: ArrayViewD<f64>,
    left_free: &[usize],
    left_contr: &[usize],
    right: ArrayViewD<f64>,
    right_free: &[usize],
    right_contr: &[usize],
    out_shape: &[usize],
) -> Result<ArrayD<f64>, TensorError> {
    let l2 = to_2d(left, left_free, left_contr, "left");
    let r2 = to_2d(right, right_contr, right_free, "right");

    let (lf, lc) = (l2.shape()[0], l2.shape()[1]);
    let (rc, rf) = (r2.shape()[0], r2.shape()[1]);
    if lc != rc {
        return Err(TensorError::ContractedDimMismatch { left: lc, right: rc });
    }

    let mut out2 = ArrayD::<f64>::zeros(IxDyn(&[lf, rf]));
    {
        let l2m = l2.view().into_dimensionality::<ndarray::Ix2>().unwrap();
        let r2m = r2.view().into_dimensionality::<ndarray::Ix2>().unwrap();
        let mut o2m = out2.view_mut().into_dimensionality::<ndarray::Ix2>().unwrap();
        general_mat_mul(1.0, &l2m, &r2m, 0.0, &mut o2m);
    }

    let want: usize = out_shape.iter().product::<usize>().max(1);
    let got = lf * rf;
    if got != want {
        return Err(TensorError::OutputReshape { got, want });
    }
    let out = out2
        .into_shape_with_order(IxDyn(out_shape))
        .map_err(|_| TensorError::OutputReshape { got, want })?;
    Ok(out)
}

/// Contract two operands with optional *batch* (diagonal) axes and a scale.
///
/// Batch axes are indices that appear in BOTH inputs and the OUTPUT — they are
/// iterated element-wise rather than contracted or outer-product'd. For each
/// fixed batch index the remaining free/contracted axes form an independent
/// contraction (`ij,ij->ij` Hadamard, `kij,kij->ij` per-element dot over `k`).
/// The whole result is multiplied by `scale`.
///
/// - `*_batch`: batch axis positions in each operand, in the SAME logical order
///   (the macro guarantees the orders match). `out_batch` gives their positions
///   in the output; batch dims always lead the output (`batch..., left_free...,
///   right_free...`), which the macro enforces.
/// - When `*_batch` is empty this is exactly [`einsum_binary`] times `scale`.
#[allow(clippy::too_many_arguments)]
pub fn einsum_binary_batched(
    left: ArrayViewD<f64>,
    left_batch: &[usize],
    left_free: &[usize],
    left_contr: &[usize],
    right: ArrayViewD<f64>,
    right_batch: &[usize],
    right_free: &[usize],
    right_contr: &[usize],
    out_shape: &[usize],
    scale: f64,
) -> Result<ArrayD<f64>, TensorError> {
    // Fast path: no batch axes -> one GEMM, then scale.
    if left_batch.is_empty() {
        let mut out =
            einsum_binary(left, left_free, left_contr, right, right_free, right_contr, out_shape)?;
        if scale != 1.0 {
            out.mapv_inplace(|x| x * scale);
        }
        return Ok(out);
    }

    // Batched path. Reshape each operand to (batch, free, contr): a 3D view
    // where index 0 walks the (flattened) batch space. Then for each batch
    // slice do one GEMM of (free x contr)·(contr x rfree) -> (free x rfree).
    let l3 = to_3d(left, left_batch, left_free, left_contr, "left");
    let r3 = to_3d(right, right_batch, right_contr, right_free, "right");

    let nb = l3.shape()[0];
    let (lf, lc) = (l3.shape()[1], l3.shape()[2]);
    let (rc, rf) = (r3.shape()[1], r3.shape()[2]);
    if nb != r3.shape()[0] {
        // Batch extents disagree: treat as a contracted-dim style mismatch.
        return Err(TensorError::ContractedDimMismatch { left: nb, right: r3.shape()[0] });
    }
    if lc != rc {
        return Err(TensorError::ContractedDimMismatch { left: lc, right: rc });
    }

    let mut out3 = ArrayD::<f64>::zeros(IxDyn(&[nb, lf, rf]));
    {
        let l3m = l3.view().into_dimensionality::<ndarray::Ix3>().unwrap();
        let r3m = r3.view().into_dimensionality::<ndarray::Ix3>().unwrap();
        let mut o3m = out3.view_mut().into_dimensionality::<ndarray::Ix3>().unwrap();
        for b in 0..nb {
            let lb = l3m.index_axis(ndarray::Axis(0), b);
            let rb = r3m.index_axis(ndarray::Axis(0), b);
            let mut ob = o3m.index_axis_mut(ndarray::Axis(0), b);
            general_mat_mul(scale, &lb, &rb, 0.0, &mut ob);
        }
    }

    let want: usize = out_shape.iter().product::<usize>().max(1);
    let got = nb * lf * rf;
    if got != want {
        return Err(TensorError::OutputReshape { got, want });
    }
    let out = out3
        .into_shape_with_order(IxDyn(out_shape))
        .map_err(|_| TensorError::OutputReshape { got, want })?;
    Ok(out)
}

/// Permute `op` to (batch, second, third) axis order and reshape to the 3D
/// shape (prod(batch), prod(second), prod(third)), copying to contiguous when
/// the permutation is non-trivial.
fn to_3d(
    op: ArrayViewD<f64>,
    batch: &[usize],
    second: &[usize],
    third: &[usize],
    which: &str,
) -> ArrayD<f64> {
    let order: Vec<usize> =
        batch.iter().chain(second.iter()).chain(third.iter()).copied().collect();
    let is_identity = order.iter().enumerate().all(|(i, &p)| i == p);
    if !is_identity {
        log::debug!("einsum: {which} operand permuted to {:?} (transpose copy)", order);
    }
    let nb: usize = batch.iter().map(|&ax| op.shape()[ax]).product::<usize>().max(1);
    let d1: usize = second.iter().map(|&ax| op.shape()[ax]).product::<usize>().max(1);
    let d2: usize = third.iter().map(|&ax| op.shape()[ax]).product::<usize>().max(1);
    let permuted = op.permuted_axes(order);
    permuted
        .as_standard_layout()
        .into_owned()
        .into_shape_with_order(IxDyn(&[nb, d1, d2]))
        .expect("einsum: 3D reshape after permutation")
}

/// Permute `op` so the axes in `first` come before the axes in `second`, then
/// reshape to 2D (prod(first), prod(second)). Copies to contiguous when the
/// permutation is non-trivial; logs the copy at debug level.
fn to_2d(op: ArrayViewD<f64>, first: &[usize], second: &[usize], which: &str) -> ArrayD<f64> {
    let order: Vec<usize> = first.iter().chain(second.iter()).copied().collect();
    let is_identity = order.iter().enumerate().all(|(i, &p)| i == p);
    if !is_identity {
        log::debug!("einsum: {which} operand permuted to {:?} (transpose copy)", order);
    }
    let rows: usize = first.iter().map(|&ax| op.shape()[ax]).product::<usize>().max(1);
    let cols: usize = second.iter().map(|&ax| op.shape()[ax]).product::<usize>().max(1);
    let permuted = op.permuted_axes(order);
    // `as_standard_layout` guarantees C-contiguous memory (copies when needed)
    // so that `into_shape_with_order` (which requires standard layout) succeeds.
    permuted
        .as_standard_layout()
        .into_owned()
        .into_shape_with_order(IxDyn(&[rows, cols]))
        .expect("einsum: 2D reshape after permutation")
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array, IxDyn};

    fn naive_matmul(a: &Array<f64, IxDyn>, b: &Array<f64, IxDyn>) -> Array<f64, IxDyn> {
        let (ni, nk) = (a.shape()[0], a.shape()[1]);
        let nj = b.shape()[1];
        let mut c = Array::zeros(IxDyn(&[ni, nj]));
        for i in 0..ni {
            for j in 0..nj {
                let mut s = 0.0;
                for k in 0..nk {
                    s += a[[i, k]] * b[[k, j]];
                }
                c[[i, j]] = s;
            }
        }
        c
    }

    #[test]
    fn matmul_contracted_last_first_no_transpose() {
        let a = Array::from_shape_vec(IxDyn(&[2, 3]), (0..6).map(|x| x as f64).collect()).unwrap();
        let b = Array::from_shape_vec(IxDyn(&[3, 2]), (0..6).map(|x| x as f64).collect()).unwrap();
        let out = einsum_binary(a.view(), &[0], &[1], b.view(), &[1], &[0], &[2, 2]).unwrap();
        let want = naive_matmul(&a, &b);
        assert!((&out - &want).iter().all(|x| x.abs() < 1e-12));
    }

    #[test]
    fn contraction_needs_transpose() {
        let a = Array::from_shape_vec(IxDyn(&[3, 2]), (0..6).map(|x| x as f64).collect()).unwrap();
        let b = Array::from_shape_vec(IxDyn(&[2, 3]), (0..6).map(|x| x as f64).collect()).unwrap();
        let out = einsum_binary(a.view(), &[1], &[0], b.view(), &[0], &[1], &[2, 2]).unwrap();
        let mut want = Array::zeros(IxDyn(&[2, 2]));
        for i in 0..2 { for j in 0..2 { let mut s=0.0; for k in 0..3 { s += a[[k,i]]*b[[j,k]]; } want[[i,j]]=s; } }
        assert!((&out - &want).iter().all(|x| x.abs() < 1e-12));
    }

    #[test]
    fn scalar_output() {
        let a = Array::from_shape_vec(IxDyn(&[2, 2]), vec![1.0,2.0,3.0,4.0]).unwrap();
        let b = Array::from_shape_vec(IxDyn(&[2, 2]), vec![1.0,1.0,1.0,1.0]).unwrap();
        let out = einsum_binary(a.view(), &[], &[0,1], b.view(), &[], &[0,1], &[]).unwrap();
        assert_eq!(out.len(), 1);
        assert!((out.iter().next().unwrap() - 10.0).abs() < 1e-12);
    }

    #[test]
    fn three_index_eri_build() {
        let p=2; let n=2;
        let bt = Array::from_shape_vec(IxDyn(&[p,n,n]), (0..p*n*n).map(|x| x as f64).collect()).unwrap();
        let out = einsum_binary(bt.view(), &[1,2], &[0], bt.view(), &[1,2], &[0], &[n,n,n,n]).unwrap();
        let mut want = Array::zeros(IxDyn(&[n,n,n,n]));
        for i in 0..n {for a in 0..n {for j in 0..n {for b in 0..n {
            let mut s=0.0; for pp in 0..p { s+= bt[[pp,i,a]]*bt[[pp,j,b]]; }
            want[[i,a,j,b]]=s;
        }}}}
        assert!((&out - &want).iter().all(|x| x.abs() < 1e-12));
    }
}
