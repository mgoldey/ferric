//! Runtime engine for a single binary tensor contraction.
//!
//! `einsum_binary` is what the `einsum!` macro lowers to. Given two operands and
//! the positions of their free vs contracted axes, it permutes each operand so
//! the contracted axes are grouped, reshapes to 2D, calls one GEMM
//! (`general_mat_mul`), and reshapes the result to the requested output shape.
//! Any required permutation copy is logged at debug level so hot transposes are
//! discoverable.

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
