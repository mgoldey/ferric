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

use ferric_core::blas_threads::{opt_in_blas_threads, with_blas_threads};
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
    // `to_2d_or_transpose` returns the operand already viewable as a matrix
    // whenever the required axis order is either the identity OR the exact
    // swap of the two axis groups — in the latter case it hands back the
    // TRANSPOSED view and ndarray's GEMM absorbs it into dgemm's transa/transb
    // flag, so no permutation copy is materialized at all. 72 of the ~130
    // workspace specs have at least one operand in that class (e.g. the CCSD
    // VVVV term `ijcd,abcd->ijab`), which is 52% of all copies that survived
    // the identity check. Only genuinely interleaved orders still copy.
    let l2 = to_2d_or_transpose(left, left_free, left_contr, "left");
    let r2 = to_2d_or_transpose(right, right_contr, right_free, "right");

    let (lf, lc) = (l2.view().shape()[0], l2.view().shape()[1]);
    let (rc, rf) = (r2.view().shape()[0], r2.view().shape()[1]);
    if lc != rc {
        return Err(TensorError::ContractedDimMismatch { left: lc, right: rc });
    }

    let mut out2 = ArrayD::<f64>::zeros(IxDyn(&[lf, rf]));
    {
        let l2m = l2.view().into_dimensionality::<ndarray::Ix2>().unwrap();
        let r2m = r2.view().into_dimensionality::<ndarray::Ix2>().unwrap();
        let mut o2m = out2.view_mut().into_dimensionality::<ndarray::Ix2>().unwrap();
        // Opt-in multi-threaded BLAS for this GEMM (default resolves to 1 —
        // a no-op — unless FERRIC_BLAS_THREADS is set; see blas_threads.rs's
        // hazard-model doc). einsum_binary is called from both rayon and
        // non-rayon contexts across the workspace; opt_in_blas_threads's
        // runtime rayon-worker guard forces 1 automatically when this runs
        // inside a parallel region, so wrapping here is safe either way.
        //
        // The k-axis is blocked in OUR code (see `gemm_kblocked`): coarse
        // pairwise summation, measured 1.10-1.40x more accurate than one
        // full-k GEMM against a compensated reference. That is an
        // unconditional win and applies at ANY thread count.
        //
        // It does NOT make the result thread-independent — OpenBLAS also
        // parallelizes over m/n, so it still splits inside a block; CCSD across
        // FERRIC_BLAS_THREADS 1..12 still shows 4 bit patterns (5 ulp,
        // 6.0e-16 rel). Accuracy and bit-reproducibility are separate
        // properties here; blocking buys the first, and only a reproducible
        // BLAS would buy the second. The default therefore stays at 1 thread —
        // raising it is ~1.7x on CCSD (24.5 -> 14.1 s at the 6-thread knee,
        // regressing past ~6 on this 12-core box) and costs only bit-identity,
        // never accuracy. See `gemm_kblocked`'s doc for the full table.
        with_blas_threads(opt_in_blas_threads(), || {
            gemm_kblocked(&l2m, &r2m, &mut o2m);
        });
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
        // Opt-in multi-threaded BLAS for every batch-slice GEMM in this loop
        // (default resolves to 1 — a no-op — unless FERRIC_BLAS_THREADS is
        // set). This is a plain sequential `for`, not a par_iter; the
        // runtime rayon-worker guard inside opt_in_blas_threads still forces
        // 1 if a caller ever nests this inside a rayon region, so the wrap
        // is safe either way. One resolve+set/restore for the whole loop
        // rather than per-slice.
        with_blas_threads(opt_in_blas_threads(), || {
            for b in 0..nb {
                let lb = l3m.index_axis(ndarray::Axis(0), b);
                let rb = r3m.index_axis(ndarray::Axis(0), b);
                let mut ob = o3m.index_axis_mut(ndarray::Axis(0), b);
                general_mat_mul(scale, &lb, &rb, 0.0, &mut ob);
            }
        });
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

/// Contraction-axis block size for [`gemm_kblocked`]. Matches
/// `three_index_source.rs`'s `DRESS_K_BLOCK`, which the DF dressing work
/// (`eb895df`) measured as the accuracy knee.
const GEMM_K_BLOCK: usize = 128;

/// `out = left · right`, accumulating over the contraction axis in fixed
/// blocks of [`GEMM_K_BLOCK`] rather than as one full-k GEMM.
///
/// **This is chosen for ACCURACY first, and reproducibility falls out of it.**
///
/// Blocking the k-axis is coarse pairwise summation: each block's partial
/// product is formed independently and the blocks are then added, so rounding
/// error grows like log(k/blk) instead of k. Measured against a
/// Neumaier-compensated reference (`benchmarks/harness/examples/
/// gemm_accuracy_vs_threads.rs`), max |error| over the product:
///
/// ```text
///   m=200 k=4000 n=200     full-k GEMM 1.421e-14    k-blocked(128) 1.066e-14   1.33x better
///   m=400 k=2000 n=400     full-k GEMM 1.121e-14    k-blocked(128) 7.994e-15   1.40x better
///   m=100 k=8000 n=100     full-k GEMM 1.954e-14    k-blocked(128) 1.776e-14   1.10x better
/// ```
///
/// The same measurement corrected an earlier misconception worth recording:
/// **threading dgemm does NOT cost accuracy.** At 1 vs 12 OpenBLAS threads the
/// max error is *identical* (1.421e-14 both, and at k=8000 the threaded result
/// is marginally BETTER at 1.821e-14 vs 1.954e-14) — only the bit pattern
/// moves. So the reflex "don't thread, it breaks determinism" was defending
/// the *serial full-k* order, which is neither the fastest nor the most
/// accurate of the candidates. Accuracy and reproducibility are separate
/// properties here, and this function buys the first one unconditionally.
///
/// **It does NOT buy the second, and that was a wrong hypothesis worth
/// recording.** Fixing the block boundaries in our own code does not make the
/// product thread-independent, because OpenBLAS also parallelizes over m/n —
/// not only over k — so it still splits work *inside* a 128-wide block.
/// Measured directly (k-blocked at 1 vs 4 vs 12 threads, compared to each
/// other rather than to serial full-k): bit-identical in only 1 of 6 cases.
/// End-to-end, CCSD across `FERRIC_BLAS_THREADS` 1/2/4/6/8/12 still shows 4
/// distinct bit patterns with the same 5-ulp spread as before. Anyone wanting
/// bit-reproducible threaded GEMM needs a reproducible BLAS, not blocking.
///
/// Larger blocks are not uniformly better or worse (256/512 measured at or
/// slightly above full-k); 128 is the knee that `eb895df` independently landed
/// on for the DF dressing, so both paths now share one constant and one
/// rationale.
fn gemm_kblocked(
    left: &ndarray::ArrayView2<f64>,
    right: &ndarray::ArrayView2<f64>,
    out: &mut ndarray::ArrayViewMut2<f64>,
) {
    let k = left.ncols();
    if k <= GEMM_K_BLOCK {
        general_mat_mul(1.0, left, right, 0.0, out);
        return;
    }
    let mut k0 = 0;
    while k0 < k {
        let k1 = (k0 + GEMM_K_BLOCK).min(k);
        let lb = left.slice(ndarray::s![.., k0..k1]);
        let rb = right.slice(ndarray::s![k0..k1, ..]);
        // beta = 0 on the first block (initializes `out`), 1 thereafter, so
        // the blocks accumulate in ascending-k order — fixed, and independent
        // of how many threads OpenBLAS uses inside each block.
        let beta = if k0 == 0 { 0.0 } else { 1.0 };
        general_mat_mul(1.0, &lb, &rb, beta, out);
        k0 = k1;
    }
}

/// Minimum element count before the permutation copy is worth fanning out over
/// rayon. Below this the dispatch overhead dominates a copy that is only a few
/// microseconds serially.
///
/// Calibrated on the measured copy rate (~2.5 GB/s single-threaded, i.e.
/// ~3e8 f64/s): 1<<16 elements is ~0.2 ms of work, comfortably above rayon's
/// few-µs dispatch cost while still catching every production-scale
/// contraction. Small unit-test tensors stay on the serial path.
const PAR_PERMUTE_MIN_ELEMS: usize = 1 << 16;

/// Copy `permuted` into a fresh C-contiguous array, fanning the gather out over
/// rayon when the operand is large enough to pay for the dispatch.
///
/// This is the hot part of `einsum!`. 83% of the workspace's ~130 einsum specs
/// need at least one operand permuted before it can be viewed as a 2D matrix,
/// and the copy is a strided gather that ndarray performs single-threaded via
/// `as_standard_layout().into_owned()`. Measured on `ijcd,abcd->ijab` at
/// no=10: the copy is 47% of the contraction at nv=40 and **70% at nv=80** —
/// it is memory-bandwidth-bound and gets relatively WORSE with size, while the
/// GEMM beside it holds ~38 GFLOP/s.
///
/// Parallelism is over the outermost axis of the *output* (post-permutation)
/// array, so each worker writes a disjoint contiguous slab and reads a strided
/// slice of the source. No accumulation happens here — this is a pure data
/// movement — so unlike a reduction it is exactly bit-identical regardless of
/// thread count: every output element is written exactly once, by one worker,
/// with the same value. (Contrast the summation-order discipline that governs
/// the GEMM paths.)
///
/// BLAS is untouched: this is rayon-only, so it composes with the
/// `blas_threads` hazard model (rayon owns parallelism, OpenBLAS pinned to 1)
/// without needing a thread-count decision.
fn permute_to_owned(permuted: ndarray::ArrayViewD<f64>) -> ArrayD<f64> {
    use rayon::prelude::*;

    // Already contiguous in the requested order: ndarray hands back a view with
    // no copy at all, so there is nothing to parallelize.
    if permuted.is_standard_layout() {
        return permuted.to_owned();
    }
    let shape = permuted.shape().to_vec();
    let n: usize = shape.iter().product();
    if n < PAR_PERMUTE_MIN_ELEMS || shape.is_empty() || shape[0] < 2 {
        return permuted.as_standard_layout().into_owned();
    }

    // Fan out over the outermost axis by splitting the flat output buffer into
    // equal per-slab chunks. `ndarray`'s own `rayon` feature is not enabled in
    // this workspace, so rather than turn it on for every crate we drive rayon
    // over the raw output slice and re-view each chunk — the output IS standard
    // layout by construction, so slab `k` is exactly `chunk k` of the buffer.
    let n0 = shape[0];
    let slab: usize = shape[1..].iter().product::<usize>().max(1);
    let mut out = ArrayD::<f64>::zeros(IxDyn(&shape));
    let buf = out.as_slice_mut().expect("freshly allocated ArrayD is contiguous");
    buf.par_chunks_mut(slab).enumerate().for_each(|(k, chunk)| {
        let src = permuted.index_axis(ndarray::Axis(0), k);
        // Each (n-1)-dim source slab is still strided; let ndarray's own
        // optimized assign drive the inner gather into the contiguous chunk.
        let mut dst = ndarray::ArrayViewMutD::from_shape(IxDyn(&shape[1..]), chunk)
            .expect("chunk length matches slab shape by construction");
        dst.assign(&src);
    });
    debug_assert_eq!(n0 * slab, permuted.len());
    out
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
    permute_to_owned(permuted.view())
        .into_shape_with_order(IxDyn(&[nb, d1, d2]))
        .expect("einsum: 3D reshape after permutation")
}

/// A 2D operand for GEMM: either a borrowed view onto the caller's data (no
/// copy) or an owned permuted array. `view()` yields the matrix either way.
enum Operand2d<'a> {
    /// Zero-copy: the operand was already usable as a matrix, possibly only
    /// after a transpose that `general_mat_mul` absorbs into dgemm's
    /// `transa`/`transb` flag.
    Borrowed(ndarray::ArrayView2<'a, f64>),
    /// The axis order was genuinely interleaved, so a gather was unavoidable.
    Owned(ArrayD<f64>),
}

impl Operand2d<'_> {
    fn view(&self) -> ndarray::ArrayView2<'_, f64> {
        match self {
            Operand2d::Borrowed(v) => v.reborrow(),
            Operand2d::Owned(a) => {
                a.view().into_dimensionality::<ndarray::Ix2>().expect("2D by construction")
            }
        }
    }
}

/// Like [`to_2d`], but avoids the permutation copy whenever the required axis
/// order is reachable by a pure transpose.
///
/// Three cases, in order:
///  1. The operand is already in `(first..., second...)` order → view it as
///     `(prod(first), prod(second))`, no copy.
///  2. It is in `(second..., first...)` order → view it as
///     `(prod(second), prod(first))` and TRANSPOSE the view. ndarray's
///     `general_mat_mul` maps a transposed view onto dgemm's `transa`/`transb`
///     flag, so BLAS reads the original memory with a stride — still no copy.
///  3. Anything else (genuinely interleaved axes) → fall back to [`to_2d`].
///
/// Case 2 is what pays: 72 of the workspace's ~130 specs have at least one
/// operand there, including the O(N^6) CCSD VVVV term `ijcd,abcd->ijab`.
///
/// The reshape in cases 1 and 2 requires the underlying data to be contiguous
/// in the grouped order, which is exactly `is_standard_layout()` on the
/// correspondingly-permuted view; when that does not hold (a non-contiguous
/// input slice) we fall back to the copy rather than reshape a strided view.
fn to_2d_or_transpose<'a>(
    op: ArrayViewD<'a, f64>,
    first: &[usize],
    second: &[usize],
    which: &str,
) -> Operand2d<'a> {
    let rows: usize = first.iter().map(|&ax| op.shape()[ax]).product::<usize>().max(1);
    let cols: usize = second.iter().map(|&ax| op.shape()[ax]).product::<usize>().max(1);

    // `into_shape_with_order` consumes the view, so hand it a clone — an
    // ArrayView clone is just a pointer + shape + strides, not the data.
    let fwd: Vec<usize> = first.iter().chain(second.iter()).copied().collect();
    if fwd.iter().enumerate().all(|(i, &p)| i == p) && op.is_standard_layout() {
        if let Ok(v) = op.clone().into_shape_with_order(ndarray::Ix2(rows, cols)) {
            return Operand2d::Borrowed(v);
        }
    }

    // Transposed grouping: the operand is laid out as (second..., first...).
    let rev: Vec<usize> = second.iter().chain(first.iter()).copied().collect();
    if rev.iter().enumerate().all(|(i, &p)| i == p) && op.is_standard_layout() {
        if let Ok(v) = op.clone().into_shape_with_order(ndarray::Ix2(cols, rows)) {
            log::debug!("einsum: {which} operand consumed transposed (no copy)");
            return Operand2d::Borrowed(v.reversed_axes());
        }
    }

    Operand2d::Owned(to_2d(op, first, second, which))
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
    // The result must be C-contiguous so `into_shape_with_order` (which requires
    // standard layout) succeeds; `permute_to_owned` guarantees that.
    permute_to_owned(permuted.view())
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

    /// `gemm_kblocked` blocks the contraction axis, which is coarse pairwise
    /// summation and therefore MORE accurate than one full-k GEMM — the reason
    /// it was chosen. This pins that claim against a Neumaier-compensated
    /// reference so a future "simplification" back to a single
    /// `general_mat_mul` cannot silently give up the accuracy.
    ///
    /// Uses a large k (that is where accumulation error lives) and operands of
    /// mixed magnitude, which is what makes summation order matter at all —
    /// uniformly-scaled random data barely distinguishes the orderings.
    #[test]
    fn kblocked_gemm_is_at_least_as_accurate_as_full_k() {
        use ndarray::Array2;

        let (m, k, n) = (24usize, 3000usize, 24usize);
        let mut s: u64 = 0x9e37_79b9_7f4a_7c15;
        let mut rnd = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        // Mixed magnitudes: a few large terms among many small ones is the
        // classic case where naive left-to-right accumulation loses digits.
        let a = Array2::from_shape_fn((m, k), |(_, p)| {
            let v = rnd();
            if p % 97 == 0 { v * 1e6 } else { v }
        });
        let b = Array2::from_shape_fn((k, n), |(p, _)| {
            let v = rnd();
            if p % 89 == 0 { v * 1e6 } else { v }
        });

        // Neumaier-compensated reference.
        let bt = b.t().as_standard_layout().into_owned();
        let mut want = Array2::<f64>::zeros((m, n));
        for i in 0..m {
            for j in 0..n {
                let (mut sum, mut c) = (0.0f64, 0.0f64);
                for p in 0..k {
                    let prod = a[[i, p]] * bt[[j, p]];
                    let t = sum + prod;
                    if sum.abs() >= prod.abs() {
                        c += (sum - t) + prod;
                    } else {
                        c += (prod - t) + sum;
                    }
                    sum = t;
                }
                want[[i, j]] = sum + c;
            }
        }

        let err_of = |got: &Array2<f64>| -> f64 {
            got.iter().zip(want.iter()).map(|(g, w)| (g - w).abs()).fold(0.0f64, f64::max)
        };

        let mut full = Array2::<f64>::zeros((m, n));
        general_mat_mul(1.0, &a.view(), &b.view(), 0.0, &mut full.view_mut());

        let mut blocked = Array2::<f64>::zeros((m, n));
        gemm_kblocked(&a.view(), &b.view(), &mut blocked.view_mut());

        let (e_full, e_blk) = (err_of(&full), err_of(&blocked));
        // Measured margin on this data is 1.167x (blocked 3.66e-4 vs full-k
        // 4.27e-4). Require a real improvement, not merely "no worse": a
        // >=1.05x floor still fails immediately if the blocking is removed
        // (which lands at ratio 1.000), while leaving headroom for BLAS
        // version differences. A "<= full * 1.05" style bound would NOT catch
        // that regression -- verified by mutation.
        let ratio = e_full / e_blk.max(1e-300);
        assert!(
            ratio >= 1.05,
            "k-blocked GEMM must be measurably MORE accurate than full-k \
             (blocked {e_blk:.3e} vs full-k {e_full:.3e}, ratio {ratio:.3} \
             < 1.05); if this fails, the k-blocking that justifies \
             gemm_kblocked's existence has been lost",
        );
        // And it must still be a correct product, not merely a precise one.
        assert!(
            e_blk / want.iter().fold(0.0f64, |m, w| m.max(w.abs())) < 1e-13,
            "k-blocked GEMM relative error {e_blk:.3e} is too large to be a \
             correct product",
        );
    }

    /// `permute_to_owned` fans the transpose gather out over rayon above
    /// `PAR_PERMUTE_MIN_ELEMS`. It must produce EXACTLY what the serial
    /// `as_standard_layout().into_owned()` produces — this is pure data
    /// movement (every output element written once, by one worker), so unlike
    /// a reduction it is bit-identical, not merely close.
    ///
    /// Shapes deliberately straddle the threshold and include a non-divisible
    /// leading extent (so the last rayon chunk is short) and a leading extent
    /// of 1 (which takes the serial short-circuit).
    /// `to_2d_or_transpose` lets `general_mat_mul` absorb a group-swap into
    /// dgemm's transa/transb flag instead of materializing a permuted copy. A
    /// wrong transpose here does not crash — it silently computes a DIFFERENT
    /// contraction — so this checks the engine against a brute-force
    /// index-by-index reference over real workspace specs.
    ///
    /// The specs are chosen to cover all three dispatch cases: an operand
    /// already contiguous (`Pia,Pjb->iajb` left), an operand reachable by pure
    /// transpose (`ijcd,abcd->ijab` right — the O(N^6) CCSD VVVV term), and a
    /// genuinely interleaved one that must still copy (`akic,kjbc->aijb`).
    /// Extents are deliberately distinct primes so a transposed-but-wrong
    /// result cannot accidentally match by symmetry.
    #[test]
    fn transpose_dispatch_matches_naive_reference() {
        // (spec, dims) with every index a distinct extent.
        let dims: std::collections::HashMap<char, usize> =
            [
                ('P', 7),
                ('i', 3),
                ('j', 4),
                ('a', 5),
                ('b', 2),
                ('c', 6),
                ('d', 3),
                ('k', 2),
                ('l', 4),
            ]
            .into_iter()
            .collect();
        let specs = ["Pia,Pjb->iajb", "ijcd,abcd->ijab", "akic,kjbc->aijb", "ijcd,klcd->ijkl"];

        for spec in specs {
            let (lhs, out) = spec.split_once("->").unwrap();
            let (ls, rs) = lhs.split_once(',').unwrap();
            let (ls, rs, out) = (ls.as_bytes(), rs.as_bytes(), out.as_bytes());
            let ext = |s: &[u8]| -> Vec<usize> { s.iter().map(|c| dims[&(*c as char)]).collect() };
            let (lsh, rsh, osh) = (ext(ls), ext(rs), ext(out));

            let mk = |shape: &[usize], seed: u64| {
                let n: usize = shape.iter().product();
                let mut s = seed;
                let v: Vec<f64> = (0..n)
                    .map(|_| {
                        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                        (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
                    })
                    .collect();
                Array::from_shape_vec(IxDyn(shape), v).unwrap()
            };
            let l = mk(&lsh, 0xabcd);
            let r = mk(&rsh, 0x1234);

            // Axis roles, exactly as the macro computes them.
            let contr: Vec<u8> =
                ls.iter().copied().filter(|c| rs.contains(c) && !out.contains(c)).collect();
            let pos = |s: &[u8], want: &[u8]| -> Vec<usize> {
                want.iter().map(|c| s.iter().position(|x| x == c).unwrap()).collect()
            };
            let lfree: Vec<u8> = ls.iter().copied().filter(|c| out.contains(c)).collect();
            let rfree: Vec<u8> = rs.iter().copied().filter(|c| out.contains(c)).collect();

            let got = einsum_binary(
                l.view(),
                &pos(ls, &lfree),
                &pos(ls, &contr),
                r.view(),
                &pos(rs, &rfree),
                &pos(rs, &contr),
                &osh,
            )
            .unwrap();

            // Brute force: walk every output index and every contracted index.
            let mut want = ArrayD::<f64>::zeros(IxDyn(&osh));
            let letters: Vec<u8> = out.iter().chain(contr.iter()).copied().collect();
            let extents: Vec<usize> = letters.iter().map(|c| dims[&(*c as char)]).collect();
            let total: usize = extents.iter().product();
            for flat in 0..total {
                // Decode a mixed-radix index over (out..., contr...).
                let mut rem = flat;
                let mut idx = std::collections::HashMap::new();
                for (k, &e) in letters.iter().zip(extents.iter()).rev() {
                    idx.insert(*k, rem % e);
                    rem /= e;
                }
                let pick = |s: &[u8]| -> Vec<usize> { s.iter().map(|c| idx[c]).collect() };
                let contrib = l[IxDyn(&pick(ls))] * r[IxDyn(&pick(rs))];
                want[IxDyn(&pick(out))] += contrib;
            }

            let max_dev = got
                .iter()
                .zip(want.iter())
                .map(|(g, w)| (g - w).abs())
                .fold(0.0f64, f64::max);
            assert!(
                max_dev < 1e-12,
                "{spec}: transpose-dispatch result differs from the naive \
                 reference by {max_dev:.3e}",
            );
        }
    }

    #[test]
    fn parallel_permute_matches_serial_bitwise() {
        // (shape, permutation) pairs; several exceed PAR_PERMUTE_MIN_ELEMS.
        let cases: Vec<(Vec<usize>, Vec<usize>)> = vec![
            (vec![7, 5, 3], vec![2, 0, 1]),          // small -> serial path
            (vec![41, 13, 17, 3], vec![2, 3, 0, 1]), // large, non-divisible
            (vec![64, 32, 32], vec![1, 2, 0]),       // large, power-of-two
            (vec![1, 400, 400], vec![2, 1, 0]),      // leading extent 1
            (vec![400, 400], vec![1, 0]),            // plain 2D transpose
        ];
        for (shape, order) in cases {
            let n: usize = shape.iter().product();
            let a =
                Array::from_shape_vec(IxDyn(&shape), (0..n).map(|x| x as f64 * 0.5 - 3.0).collect())
                    .unwrap();
            let permuted = a.view().permuted_axes(order.clone());
            let want = permuted.as_standard_layout().into_owned();
            let got = permute_to_owned(permuted.view());
            assert_eq!(got.shape(), want.shape(), "shape mismatch for {shape:?} / {order:?}");
            assert!(
                got.iter().zip(want.iter()).all(|(g, w)| g.to_bits() == w.to_bits()),
                "parallel permute must be BIT-identical to serial for {shape:?} / {order:?}",
            );
        }
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
