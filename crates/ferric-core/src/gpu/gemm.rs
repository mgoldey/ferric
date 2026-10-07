//! Row-major f64 GEMM on the device. `out = left · right` is computed as the
//! column-major product `outᵀ = rightᵀ · leftᵀ`, so a row-major standard
//! operand is passed as-is with op N and ld = ncols, and a transposed-standard
//! view is passed with op T and ld = nrows. The k axis is blocked at the same
//! width as the CPU (`ferric_tensors::einsum::GEMM_K_BLOCK` = 128) in ascending
//! order, beta 0 then 1 — the same coarse-pairwise structure, NOT bit-identical
//! to the CPU (cuBLAS blocks inside each slab its own way). See the parity test
//! for the bound that IS guaranteed.
use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublas::{Gemm, GemmConfig};
use cudarc::driver::{CudaSlice, CudaView};
use ndarray::{ArrayView2, ArrayViewMut2};

use super::device::{Device, GpuError};
use super::pool::DevicePool;
use super::stats;

/// How cuBLAS should read one row-major operand.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Desc {
    pub op: cublasOperation_t,
    pub ld: i32,
    /// Element offset per unit of k: `ld` when k runs along the stored
    /// column-major matrix's columns (op N), `1` when along its rows (op T).
    pub k_step: usize,
}

/// `Some` for the two layouts the device path accepts, `None` otherwise.
/// `k_is_cols` says whether the contraction index is this operand's COLUMN
/// axis in row-major terms (true for `left`, false for `right`).
pub(crate) fn col_major_desc<T>(v: &ArrayView2<T>, k_is_cols: bool) -> Option<Desc> {
    let (r, c) = (v.nrows(), v.ncols());
    if v.is_standard_layout() {
        // Row-major (r×c) memory read as column-major with ld = c is the
        // transpose (c×r) — exactly the factor the swapped product wants, so
        // op N. In that stored (c×r) matrix the row-major COLUMN axis is the
        // ROW axis, and the row-major ROW axis is the COLUMN axis. cuBLAS
        // sub-views: rows k0.. of a column-major matrix start at offset k0
        // (same ld); columns k0.. start at k0·ld. Hence k_step = 1 when k is
        // the row-major column axis (left operand), k_step = c (= ld) otherwise (right).
        let k_step = if k_is_cols { 1 } else { c };
        return Some(Desc {
            op: cublasOperation_t::CUBLAS_OP_N,
            ld: c as i32,
            k_step,
        });
    }
    if v.t().is_standard_layout() {
        // Memory is already column-major (r×c) with ld = r; op T supplies the
        // transpose. Here the row-major column axis IS the stored column axis
        // (offset k0·r per k) and the row-major row axis is the stored row
        // axis (offset k0).
        let k_step = if k_is_cols { r } else { 1 };
        return Some(Desc {
            op: cublasOperation_t::CUBLAS_OP_T,
            ld: r as i32,
            k_step,
        });
    }
    None
}

// Worked check of the four cases (the parity test pins them):
//   left  standard   (m×k, strides (k,1)):  stored col-major (k×m) ld=k, op N, k along rows    → k_step 1
//   right standard   (k×n, strides (n,1)):  stored col-major (n×k) ld=n, op N, k along columns → k_step n
//   left  transposed (m×k, strides (1,m)):  stored col-major (m×k) ld=m, op T, k along columns → k_step m
//   right transposed (k×n, strides (1,k)):  stored col-major (k×n) ld=k, op T, k along rows    → k_step 1

/// Bytes the pool is charged for one offload: both operands and the result.
pub fn offload_bytes(m: usize, k: usize, n: usize) -> usize {
    8 * (m * k + k * n + m * n)
}

/// One operand already on the device, described the way cuBLAS reads it after
/// the row-major → column-major swap (`outᵀ = rightᵀ·leftᵀ`): `view` starts at
/// the operand's first element, `ld` is its leading dimension in the stored
/// column-major matrix, `op` says whether cuBLAS transposes it, and `k_step`
/// is the element offset per unit of k (see `col_major_desc`).
///
/// Soundness: the fields are crate-private and the only constructors
/// ([`dev_operand`], `DevOperand::from_desc`) derive the descriptor from a
/// `rows × cols` shape and check `view.len() == rows * cols`; the `*_dev`
/// GEMMs then check that shape against the `(m, k, n)` they are given. A
/// descriptor from `col_major_desc` for a contiguous `rows × cols` matrix never
/// reads outside those `rows * cols` elements, so no caller can make the cuBLAS
/// call read past a buffer (no `unsafe` in the public signatures).
pub struct DevOperand<'a, T> {
    pub(crate) view: CudaView<'a, T>,
    pub(crate) op: cublasOperation_t,
    pub(crate) ld: i32,
    pub(crate) k_step: usize,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
}

impl<'a, T> DevOperand<'a, T> {
    pub(crate) fn from_desc(
        view: CudaView<'a, T>,
        d: Desc,
        rows: usize,
        cols: usize,
    ) -> Result<Self, GpuError> {
        if view.len() != rows * cols {
            return Err(GpuError::Layout(format!(
                "device operand holds {} elements but its {rows}x{cols} shape needs {}",
                view.len(),
                rows * cols
            )));
        }
        Ok(Self {
            view,
            op: d.op,
            ld: d.ld,
            k_step: d.k_step,
            rows,
            cols,
        })
    }
}

/// Describe a resident operand whose host twin `host` has the stored layout
/// (standard or transposed-standard, hence contiguous, the two layouts
/// `gemm_f64` accepts). `view` must hold exactly `host.len()` elements in the
/// host's memory order; anything else is a `GpuError::Layout`.
pub fn dev_operand<'a, T>(
    view: CudaView<'a, T>,
    host: &ArrayView2<T>,
    k_is_cols: bool,
) -> Result<DevOperand<'a, T>, GpuError> {
    let d = col_major_desc(host, k_is_cols).ok_or_else(|| {
        GpuError::Layout("operand is neither standard nor transposed-standard".into())
    })?;
    DevOperand::from_desc(view, d, host.nrows(), host.ncols())
}

/// Device-free part of the `*_dev` argument checks: operand shapes against
/// `(m, k, n)` and each output buffer's length against `m·n`.
pub(crate) fn check_dev_geometry(
    m: usize,
    k: usize,
    n: usize,
    left: (usize, usize),
    right: (usize, usize),
    outputs: &[(&str, usize)],
) -> Result<(), GpuError> {
    if [m, k, n].iter().any(|&d| d > i32::MAX as usize) {
        return Err(GpuError::Layout(format!(
            "dimension exceeds cuBLAS i32 range: {m}x{k}x{n}"
        )));
    }
    if left != (m, k) {
        return Err(GpuError::Layout(format!(
            "left operand is {}x{} but the product is {m}x{k}x{n}",
            left.0, left.1
        )));
    }
    if right != (k, n) {
        return Err(GpuError::Layout(format!(
            "right operand is {}x{} but the product is {m}x{k}x{n}",
            right.0, right.1
        )));
    }
    for &(name, len) in outputs {
        if len < m * n {
            return Err(GpuError::Layout(format!(
                "output buffer {name} holds {len} elements, {m}x{n} needs {}",
                m * n
            )));
        }
    }
    Ok(())
}

/// The k-blocked f64 loop on resident operands; `c` holds m·n f64 in the
/// column-major (n × m) layout that is row-major (m × n). Numerics unchanged:
/// `gemm_f64` is upload + this + download. Shapes and buffer lengths are
/// checked first (typed `GpuError::Layout`, nothing launched).
pub fn gemm_f64_dev(
    dev: &Device,
    m: usize,
    k: usize,
    n: usize,
    left: &DevOperand<'_, f64>,
    right: &DevOperand<'_, f64>,
    c: &mut CudaSlice<f64>,
    k_block: usize,
) -> Result<(), GpuError> {
    check_dev_geometry(
        m,
        k,
        n,
        (left.rows, left.cols),
        (right.rows, right.cols),
        &[("c", c.len())],
    )?;
    let cuda = |e: std::fmt::Arguments| GpuError::Cuda(e.to_string());
    let blas = dev.blas.lock().unwrap_or_else(|e| e.into_inner());
    let kb = k_block.max(1);
    let mut k0 = 0usize;
    while k0 < k {
        let k1 = (k0 + kb).min(k);
        let cfg = GemmConfig {
            transa: right.op,
            transb: left.op,
            m: n as i32,
            n: m as i32,
            k: (k1 - k0) as i32,
            alpha: 1.0,
            lda: right.ld,
            ldb: left.ld,
            beta: if k0 == 0 { 0.0 } else { 1.0 },
            ldc: n as i32,
        };
        let a_view = right.view.slice(k0 * right.k_step..);
        let b_view = left.view.slice(k0 * left.k_step..);
        // SAFETY: shapes and leading dimensions come from the descriptors and
        // the buffers hold exactly those elements.
        unsafe { blas.gemm(cfg, &a_view, &b_view, c) }
            .map_err(|e| cuda(format_args!("cublasDgemm k-block {k0}..{k1}: {e:?}")))?;
        k0 = k1;
    }
    Ok(())
}

pub fn gemm_f64(
    dev: &Device,
    pool: &DevicePool,
    left: &ArrayView2<f64>,
    right: &ArrayView2<f64>,
    out: &mut ArrayViewMut2<f64>,
    k_block: usize,
) -> Result<(), GpuError> {
    let (m, k) = (left.nrows(), left.ncols());
    let n = right.ncols();
    if right.nrows() != k || out.nrows() != m || out.ncols() != n {
        return Err(GpuError::Layout(format!(
            "shape mismatch: ({m}x{k})·({}x{n}) -> ({}x{})",
            right.nrows(),
            out.nrows(),
            out.ncols()
        )));
    }
    if !out.is_standard_layout() {
        return Err(GpuError::Layout(
            "output must be row-major contiguous".into(),
        ));
    }
    if m == 0 || n == 0 {
        return Ok(());
    }
    if [m, k, n].iter().any(|&d| d > i32::MAX as usize) {
        return Err(GpuError::Layout(format!(
            "dimension exceeds cuBLAS i32 range: {m}x{k}x{n}"
        )));
    }
    if k == 0 {
        out.fill(0.0);
        return Ok(());
    }
    let da = col_major_desc(left, true).ok_or_else(|| {
        GpuError::Layout("left operand is neither standard nor transposed-standard".into())
    })?;
    let db = col_major_desc(right, false).ok_or_else(|| {
        GpuError::Layout("right operand is neither standard nor transposed-standard".into())
    })?;
    let la = left
        .as_slice_memory_order()
        .ok_or_else(|| GpuError::Layout("left not contiguous".into()))?;
    let rb = right
        .as_slice_memory_order()
        .ok_or_else(|| GpuError::Layout("right not contiguous".into()))?;

    let bytes = offload_bytes(m, k, n);
    let _lease = pool.reserve("gemm_f64 operands+result", bytes)?;

    let s = &dev.stream;
    let cuda = |e: std::fmt::Arguments| GpuError::Cuda(e.to_string());
    let d_a: CudaSlice<f64> = s
        .clone_htod(la)
        .map_err(|e| cuda(format_args!("H2D left: {e:?}")))?;
    let d_b: CudaSlice<f64> = s
        .clone_htod(rb)
        .map_err(|e| cuda(format_args!("H2D right: {e:?}")))?;
    let mut d_c: CudaSlice<f64> = s
        .alloc_zeros(m * n)
        .map_err(|e| cuda(format_args!("alloc out: {e:?}")))?;

    let left_op = DevOperand::from_desc(d_a.slice(..), da, m, k)?;
    let right_op = DevOperand::from_desc(d_b.slice(..), db, k, n)?;
    gemm_f64_dev(dev, m, k, n, &left_op, &right_op, &mut d_c, k_block)?;
    let host = out.as_slice_mut().expect("standard layout checked above");
    s.memcpy_dtoh(&d_c, host)
        .map_err(|e| cuda(format_args!("D2H out: {e:?}")))?;
    s.synchronize()
        .map_err(|e| cuda(format_args!("sync: {e:?}")))?;
    stats::note_offloaded(8 * (m * k + k * n), 8 * m * n);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    // Pure descriptor logic: no device needed, runs in every CI configuration.
    #[test]
    fn descriptors_for_the_four_operand_cases() {
        let a = Array2::<f64>::zeros((5, 7)); // m=5, k=7 standard
        let d = col_major_desc(&a.view(), true).unwrap();
        assert_eq!(
            (d.op, d.ld, d.k_step),
            (cublasOperation_t::CUBLAS_OP_N, 7, 1)
        );
        let b = Array2::<f64>::zeros((7, 3)); // k=7, n=3 standard
        let d = col_major_desc(&b.view(), false).unwrap();
        assert_eq!(
            (d.op, d.ld, d.k_step),
            (cublasOperation_t::CUBLAS_OP_N, 3, 3)
        );
        let at = Array2::<f64>::zeros((7, 5)); // stored k x m, viewed .t() = m x k
        let d = col_major_desc(&at.t(), true).unwrap();
        assert_eq!(
            (d.op, d.ld, d.k_step),
            (cublasOperation_t::CUBLAS_OP_T, 5, 5)
        );
        let bt = Array2::<f64>::zeros((3, 7)); // stored n x k, viewed .t() = k x n
        let d = col_major_desc(&bt.t(), false).unwrap();
        assert_eq!(
            (d.op, d.ld, d.k_step),
            (cublasOperation_t::CUBLAS_OP_T, 7, 1)
        );
    }

    #[test]
    fn interleaved_view_has_no_descriptor() {
        let big = Array2::<f64>::zeros((8, 8));
        assert!(col_major_desc(&big.slice(ndarray::s![..4, ..4]), true).is_none());
    }

    #[test]
    fn negative_stride_view_has_no_descriptor() {
        let a = Array2::<f64>::zeros((8, 8));
        assert!(col_major_desc(&a.slice(ndarray::s![..;-1, ..]), true).is_none());
        assert!(col_major_desc(&a.slice(ndarray::s![.., ..;-1]), false).is_none());
    }

    #[test]
    fn dev_geometry_accepts_matching_shapes_and_exact_or_larger_outputs() {
        assert!(check_dev_geometry(4, 6, 5, (4, 6), (6, 5), &[("c", 20)]).is_ok());
        assert!(check_dev_geometry(4, 6, 5, (4, 6), (6, 5), &[("c", 99)]).is_ok());
    }

    #[test]
    fn dev_geometry_refuses_each_short_or_mismatched_buffer_by_name() {
        let e =
            check_dev_geometry(4, 6, 5, (4, 6), (6, 5), &[("c32", 20), ("c64", 19)]).unwrap_err();
        assert!(
            matches!(e, GpuError::Layout(ref s) if s.contains("c64")),
            "{e:?}"
        );
        let e = check_dev_geometry(4, 6, 5, (4, 7), (6, 5), &[]).unwrap_err();
        assert!(
            matches!(e, GpuError::Layout(ref s) if s.contains("left")),
            "{e:?}"
        );
        let e = check_dev_geometry(4, 6, 5, (4, 6), (5, 5), &[]).unwrap_err();
        assert!(
            matches!(e, GpuError::Layout(ref s) if s.contains("right")),
            "{e:?}"
        );
        let big = i32::MAX as usize + 1;
        let e = check_dev_geometry(big, 1, 1, (big, 1), (1, 1), &[]).unwrap_err();
        assert!(
            matches!(e, GpuError::Layout(ref s) if s.contains("i32")),
            "{e:?}"
        );
    }

    #[test]
    fn offload_bytes_is_eight_times_the_three_operands() {
        assert_eq!(offload_bytes(2, 3, 5), 8 * (6 + 15 + 10));
    }
}
