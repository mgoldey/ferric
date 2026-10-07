//! Strided-batched f64 GEMM and SYRK on resident operands.
//!
//! The single-matrix `DevOperand` constructors prove a view reaches one matrix.
//! A batch's last matrix starts `(batch − 1)·stride` elements later, so the
//! constructors here derive the geometry of ONE matrix from a host twin using
//! the length left for the LAST matrix (`view.len() − (batch − 1)·stride`,
//! checked), and `check_batched_geometry` repeats the same accounting before
//! any launch, delegating the single-matrix soundness rules (roles, shapes,
//! descriptor, per-panel extent, i32 range) to `gemm::check_dev_geometry`.
//! Reads grow monotonically with the batch index, so covering the last matrix
//! covers all of them. Outputs must not overlap (`c_stride ≥ m·n` when
//! `batch > 1`), so the result cannot depend on the order cuBLAS runs batches.
//!
//! Numerics: the product is NOT k-blocked (one cuBLAS call). The Higham bound
//! `|ĉ − c| ≤ γ_k·(|A||B|)` holds for every summation order of a classical
//! GEMM, so blocking buys no accuracy argument here (`gemm_f64_dev` blocks at
//! 128 only to mirror the CPU's structure).
use cudarc::cublas::sys::{self, cublasFillMode_t, cublasOperation_t};
use cudarc::cublas::{Gemm, GemmConfig, StridedBatchedConfig};
use cudarc::driver::{CudaSlice, CudaView, DevicePtr, DevicePtrMut};
use ndarray::ArrayView2;

use super::device::{Device, GpuError};
use super::gemm::{check_dev_geometry, DevOperand, OperandGeom, Role};

/// One operand of a strided batch: the first matrix's descriptor plus the
/// element offset between consecutive matrices (0 broadcasts one matrix).
pub struct DevBatchedOperand<'a, T> {
    pub(crate) operand: DevOperand<'a, T>,
    pub(crate) batch: usize,
    pub(crate) stride: usize,
}

/// `(batch − 1)·stride`, the element offset of the last matrix; typed refusal on overflow.
fn batch_tail(batch: usize, stride: usize) -> Result<usize, GpuError> {
    batch
        .saturating_sub(1)
        .checked_mul(stride)
        .ok_or_else(|| GpuError::Layout(format!("batch {batch} x stride {stride} overflows usize")))
}

fn new_batched<'a, T, H>(
    view: CudaView<'a, T>,
    first: &ArrayView2<H>,
    role: Role,
    batch: usize,
    stride: usize,
) -> Result<DevBatchedOperand<'a, T>, GpuError> {
    let tail = batch_tail(batch, stride)?;
    let last_len = view.len().checked_sub(tail).ok_or_else(|| {
        GpuError::Layout(format!(
            "device view holds {} elements but batch {batch} at stride {stride} starts its last matrix at element {tail}",
            view.len()
        ))
    })?;
    let geom = OperandGeom::derive_padded(role, first, last_len)?;
    Ok(DevBatchedOperand {
        operand: DevOperand { view, geom },
        batch,
        stride,
    })
}

/// LEFT operand of a strided batch. `first` is the host twin of matrix 0
/// (standard or transposed-standard, padded allowed); `view` starts at its first
/// element and must reach the last matrix, otherwise `GpuError::Layout`.
pub fn dev_batched_left<'a, T, H>(
    view: CudaView<'a, T>,
    first: &ArrayView2<H>,
    batch: usize,
    stride: usize,
) -> Result<DevBatchedOperand<'a, T>, GpuError> {
    new_batched(view, first, Role::Left, batch, stride)
}

/// RIGHT operand of a strided batch; see [`dev_batched_left`].
pub fn dev_batched_right<'a, T, H>(
    view: CudaView<'a, T>,
    first: &ArrayView2<H>,
    batch: usize,
    stride: usize,
) -> Result<DevBatchedOperand<'a, T>, GpuError> {
    new_batched(view, first, Role::Right, batch, stride)
}

/// Shape of one strided-batched product: `batch` independent `m×k · k×n`.
#[derive(Debug, Clone, Copy)]
pub struct BatchedDims {
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub batch: usize,
}

/// The product writes its output panels (the single definition shared by the
/// geometry check and the launch path, so the two cannot drift).
fn writes_output(d: BatchedDims) -> bool {
    d.batch > 0 && d.m > 0 && d.n > 0
}

/// The product reads its operands (a `k = 0` product is the zero matrix).
fn reads_operands(d: BatchedDims) -> bool {
    writes_output(d) && d.k > 0
}

/// Device-free soundness check; `left`/`right` are `(geom, view_len, stride)`,
/// `out` is `(c_len, c_stride)`. Total on all inputs (zero sizes, overflow).
pub(crate) fn check_batched_geometry(
    d: BatchedDims,
    left: (&OperandGeom, usize, usize),
    right: (&OperandGeom, usize, usize),
    out: (usize, usize),
) -> Result<(), GpuError> {
    let lay = |s: String| GpuError::Layout(s);
    if d.batch > i32::MAX as usize {
        return Err(lay(format!(
            "batch count {} exceeds cuBLAS i32 range",
            d.batch
        )));
    }
    for (name, stride) in [("left", left.2), ("right", right.2), ("c", out.1)] {
        if stride > i64::MAX as usize {
            return Err(lay(format!(
                "{name} batch stride {stride} exceeds cuBLAS c_longlong range"
            )));
        }
    }
    let (lt, rt, ct) = (
        batch_tail(d.batch, left.2)?,
        batch_tail(d.batch, right.2)?,
        batch_tail(d.batch, out.1)?,
    );
    let mn =
        d.m.checked_mul(d.n)
            .ok_or_else(|| lay(format!("output size {}x{} overflows usize", d.m, d.n)))?;
    let (writes, reads) = (writes_output(d), reads_operands(d));
    if writes {
        if d.batch > 1 && out.1 < mn {
            return Err(lay(format!(
                "batched outputs overlap: c_stride {} < m*n {mn}",
                out.1
            )));
        }
        let need = ct
            .checked_add(mn)
            .ok_or_else(|| lay("output extent overflows usize".into()))?;
        if out.0 < need {
            return Err(lay(format!(
                "output buffer holds {} elements, the last of {} batches ends at {need}",
                out.0, d.batch
            )));
        }
    }
    // The elements left for the LAST matrix of each operand.
    let remaining = |name: &str, len: usize, tail: usize| {
        if reads {
            len.checked_sub(tail).ok_or_else(|| {
                lay(format!(
                    "{name} view holds {len} elements but its last matrix starts at {tail}"
                ))
            })
        } else {
            // Nothing is read (`reads_operands`, the launch condition): no
            // extent applies, so the view length is not a constraint. Roles,
            // shapes and descriptors are still validated.
            Ok(usize::MAX)
        }
    };
    let (llen, rlen) = (
        remaining("left", left.1, lt)?,
        remaining("right", right.1, rt)?,
    );
    let outs: &[(&str, usize)] = if writes { &[("c", out.0)] } else { &[] };
    check_dev_geometry(
        d.m,
        d.k,
        d.n,
        d.k.max(1),
        (left.0, llen),
        (right.0, rlen),
        outs,
    )
}

/// `out_b = left_b · right_b` for `b in 0..d.batch`, one cuBLAS call. `c` holds
/// the outputs, `out_b` at `b·c_stride`, each row-major `m×n` (cuBLAS `ldc = n`).
/// Geometry is checked first (typed `GpuError::Layout`, nothing launched).
pub fn gemm_f64_strided_batched_dev(
    dev: &Device,
    d: BatchedDims,
    left: &DevBatchedOperand<'_, f64>,
    right: &DevBatchedOperand<'_, f64>,
    c: &mut CudaSlice<f64>,
    c_stride: usize,
) -> Result<(), GpuError> {
    if left.batch != d.batch || right.batch != d.batch {
        return Err(GpuError::Layout(format!(
            "operands were built for batches {} / {} but the product has batch {}",
            left.batch, right.batch, d.batch
        )));
    }
    check_batched_geometry(
        d,
        (&left.operand.geom, left.operand.view.len(), left.stride),
        (&right.operand.geom, right.operand.view.len(), right.stride),
        (c.len(), c_stride),
    )?;
    if !writes_output(d) {
        return Ok(());
    }
    if !reads_operands(d) {
        // k = 0: zero exactly the output panels (the gaps between panels, when
        // c_stride > m*n, are not part of the contract). Extents are checked.
        let mn = d.m * d.n;
        for b in 0..d.batch {
            let at = b * c_stride;
            dev.stream
                .memset_zeros(&mut c.slice_mut(at..at + mn))
                .map_err(|e| GpuError::Cuda(format!("memset c panel {b} (k = 0): {e:?}")))?;
        }
        return Ok(());
    }
    let cfg = StridedBatchedConfig {
        gemm: GemmConfig {
            transa: right.operand.geom.op,
            transb: left.operand.geom.op,
            m: d.n as i32,
            n: d.m as i32,
            k: d.k as i32,
            alpha: 1.0,
            lda: right.operand.geom.ld,
            ldb: left.operand.geom.ld,
            beta: 0.0,
            ldc: d.n as i32,
        },
        batch_size: d.batch as i32,
        stride_a: right.stride as i64,
        stride_b: left.stride as i64,
        stride_c: c_stride as i64,
    };
    let blas = dev.blas.lock().unwrap_or_else(|e| e.into_inner());
    // SAFETY: `check_batched_geometry` proved, for every batch, that each
    // operand's highest read lies inside its view (descriptor derived from the
    // role and the host twin, last-matrix extent checked), that output panels
    // do not overlap, and that `c` holds the last panel; all dimensions and
    // strides are within cuBLAS's i32 / c_longlong ranges.
    unsafe { blas.gemm_strided_batched(cfg, &right.operand.view, &left.operand.view, c) }
        .map_err(|e| GpuError::Cuda(format!("cublasDgemmStridedBatched: {e:?}")))
}

/// Device-free soundness check for [`syrk_f64_dev`].
pub(crate) fn check_syrk_geometry(
    n: usize,
    k: usize,
    a_len: usize,
    c_len: usize,
) -> Result<(), GpuError> {
    let lay = |s: String| GpuError::Layout(s);
    if n > i32::MAX as usize || k > i32::MAX as usize {
        return Err(lay(format!(
            "dimension exceeds cuBLAS i32 range: n={n} k={k}"
        )));
    }
    let need_a = n
        .checked_mul(k)
        .ok_or_else(|| lay(format!("{n}x{k} overflows usize")))?;
    let need_c = n
        .checked_mul(n)
        .ok_or_else(|| lay(format!("{n}x{n} overflows usize")))?;
    if n > 0 && k > 0 && a_len < need_a {
        return Err(lay(format!(
            "syrk input holds {a_len} elements, {n}x{k} needs {need_a}"
        )));
    }
    if n > 0 && c_len < need_c {
        return Err(lay(format!(
            "syrk output holds {c_len} elements, {n}x{n} needs {need_c}"
        )));
    }
    Ok(())
}

/// Symmetric rank-k update. `a` is the column-major `n×k` matrix with `ld = n`
/// (equivalently the row-major `k×n` matrix `Y`). The Fortran-LOWER triangle of
/// `c` (= the row-major UPPER triangle, the CPU path's convention) becomes
/// `A·Aᵀ` when `accumulate` is false, `c + A·Aᵀ` when true; the other triangle
/// is never read or written. `k = 0` zeroes the owned triangle only (or leaves
/// `c` alone when accumulating); `n = 0` is a no-op.
pub fn syrk_f64_dev(
    dev: &Device,
    n: usize,
    k: usize,
    a: &CudaView<'_, f64>,
    c: &mut CudaSlice<f64>,
    accumulate: bool,
) -> Result<(), GpuError> {
    check_syrk_geometry(n, k, a.len(), c.len())?;
    if n == 0 {
        return Ok(());
    }
    if k == 0 {
        if accumulate {
            return Ok(());
        }
        // Zero only the triangle this routine owns (row-major r <= c, the
        // Fortran lower): row r's run starts at r*n + r and is n - r long.
        for r in 0..n {
            dev.stream
                .memset_zeros(&mut c.slice_mut(r * n + r..(r + 1) * n))
                .map_err(|e| GpuError::Cuda(format!("memset c row {r} (k = 0): {e:?}")))?;
        }
        return Ok(());
    }
    let (alpha, beta) = (1.0f64, if accumulate { 1.0f64 } else { 0.0f64 });
    // The raw cuBLAS call below does not go through a cudarc wrapper that binds
    // the context, so bind it to this thread first.
    dev.ctx
        .bind_to_thread()
        .map_err(|e| GpuError::Cuda(format!("bind context before cublasDsyrk: {e:?}")))?;
    let blas = dev.blas.lock().unwrap_or_else(|e| e.into_inner());
    let (ap, _ra) = a.device_ptr(&dev.stream);
    let (cp, _rc) = c.device_ptr_mut(&dev.stream);
    // SAFETY: `check_syrk_geometry` proved `a` holds n·k and `c` holds n·n
    // elements; lda = ldc = n with trans = N reads an n×k matrix and writes an
    // n×n triangle; n, k are within cuBLAS's i32 range.
    unsafe {
        sys::cublasDsyrk_v2(
            *blas.handle(),
            cublasFillMode_t::CUBLAS_FILL_MODE_LOWER,
            cublasOperation_t::CUBLAS_OP_N,
            n as i32,
            k as i32,
            &alpha,
            ap as *const f64,
            n as i32,
            &beta,
            cp as *mut f64,
            n as i32,
        )
    }
    .result()
    .map_err(|e| GpuError::Cuda(format!("cublasDsyrk: {e:?}")))
}
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn dims(m: usize, k: usize, n: usize, batch: usize) -> BatchedDims {
        BatchedDims { m, k, n, batch }
    }

    /// Geometry of the DF-K stage-1 operands for one matrix: right = B_P
    /// (n×n standard, stride n²), left = C_occᵀ (nocc×n, transposed-standard,
    /// stride 0), derived from stride-only host twins exactly as the real
    /// constructors do.
    fn dfk_geoms(n: usize, nocc: usize) -> (OperandGeom, OperandGeom) {
        let b = Array2::<f64>::zeros((n, n));
        let c = Array2::<f64>::zeros((n, nocc));
        let right = OperandGeom::derive_padded(Role::Right, &b.view(), n * n).unwrap();
        let left = OperandGeom::derive_padded(Role::Left, &c.t(), n * nocc).unwrap();
        (left, right)
    }

    const N: usize = 7;
    const NOCC: usize = 3;
    const BATCH: usize = 5;

    fn check(
        d: BatchedDims,
        lg: &OperandGeom,
        rg: &OperandGeom,
        (llen, lstride): (usize, usize),
        (rlen, rstride): (usize, usize),
        (clen, cstride): (usize, usize),
    ) -> Result<(), GpuError> {
        check_batched_geometry(d, (lg, llen, lstride), (rg, rlen, rstride), (clen, cstride))
    }

    #[test]
    fn exact_extents_over_every_batch_are_accepted() {
        let (lg, rg) = dfk_geoms(N, NOCC);
        let d = dims(NOCC, N, N, BATCH);
        // left broadcast (stride 0): one matrix; right: BATCH matrices; out: BATCH panels
        let ok = check(
            d,
            &lg,
            &rg,
            (N * NOCC, 0),
            (BATCH * N * N, N * N),
            (BATCH * NOCC * N, NOCC * N),
        );
        assert!(ok.is_ok(), "{ok:?}");
    }

    #[test]
    fn every_batch_is_covered_by_the_extent_check() {
        let (lg, rg) = dfk_geoms(N, NOCC);
        let d = dims(NOCC, N, N, BATCH);
        // One element short of the LAST right matrix: matrix 0 is fine, so a
        // single-matrix check cannot see it.
        let short_right = check(
            d,
            &lg,
            &rg,
            (N * NOCC, 0),
            (BATCH * N * N - 1, N * N),
            (BATCH * NOCC * N, NOCC * N),
        );
        assert!(
            matches!(short_right, Err(GpuError::Layout(_))),
            "{short_right:?}"
        );
        // One short of the last output panel.
        let short_out = check(
            d,
            &lg,
            &rg,
            (N * NOCC, 0),
            (BATCH * N * N, N * N),
            (BATCH * NOCC * N - 1, NOCC * N),
        );
        assert!(
            matches!(short_out, Err(GpuError::Layout(_))),
            "{short_out:?}"
        );
        // A broadcast (stride 0) operand needs only one matrix; one short of that is refused.
        let short_left = check(
            d,
            &lg,
            &rg,
            (N * NOCC - 1, 0),
            (BATCH * N * N, N * N),
            (BATCH * NOCC * N, NOCC * N),
        );
        assert!(
            matches!(short_left, Err(GpuError::Layout(_))),
            "{short_left:?}"
        );
    }

    #[test]
    fn the_extent_follows_the_stride_not_the_batch_times_the_matrix_size() {
        let (lg, rg) = dfk_geoms(N, NOCC);
        let d = dims(NOCC, N, N, BATCH);
        let stride = N * N / 2; // overlapping READ windows are allowed
        let need = (BATCH - 1) * stride + N * N;
        assert!(check(
            d,
            &lg,
            &rg,
            (N * NOCC, 0),
            (need, stride),
            (BATCH * NOCC * N, NOCC * N)
        )
        .is_ok());
        let e = check(
            d,
            &lg,
            &rg,
            (N * NOCC, 0),
            (need - 1, stride),
            (BATCH * NOCC * N, NOCC * N),
        );
        assert!(matches!(e, Err(GpuError::Layout(_))), "{e:?}");
    }

    #[test]
    fn outputs_that_overlap_are_refused() {
        let (lg, rg) = dfk_geoms(N, NOCC);
        let overlapping = NOCC * N - 1;
        let e = check(
            dims(NOCC, N, N, BATCH),
            &lg,
            &rg,
            (N * NOCC, 0),
            (BATCH * N * N, N * N),
            (BATCH * NOCC * N, overlapping),
        );
        assert!(matches!(e, Err(GpuError::Layout(_))), "{e:?}");
        // one matrix: the output stride is never used, any value is accepted
        assert!(check(
            dims(NOCC, N, N, 1),
            &lg,
            &rg,
            (N * NOCC, 0),
            (N * N, N * N),
            (NOCC * N, 0)
        )
        .is_ok());
    }

    #[test]
    fn batch_and_stride_range_are_refused_not_wrapped() {
        let (lg, rg) = dfk_geoms(N, NOCC);
        let big = i32::MAX as usize + 1;
        let e = check(
            dims(NOCC, N, N, big),
            &lg,
            &rg,
            (N * NOCC, 0),
            (usize::MAX, N * N),
            (usize::MAX, NOCC * N),
        );
        assert!(
            matches!(e, Err(GpuError::Layout(_))),
            "batch beyond i32: {e:?}"
        );
        // (batch - 1) * stride overflows usize (3 · (2⁶³ − 1) > 2⁶⁴), with the stride itself in range
        let e = check(
            dims(NOCC, N, N, 4),
            &lg,
            &rg,
            (N * NOCC, 0),
            (usize::MAX, i64::MAX as usize),
            (4 * NOCC * N, NOCC * N),
        );
        assert!(
            matches!(e, Err(GpuError::Layout(_))),
            "stride overflow: {e:?}"
        );
        // a stride beyond cuBLAS's c_longlong
        let e = check(
            dims(NOCC, N, N, 1),
            &lg,
            &rg,
            (N * NOCC, 0),
            (N * N, i64::MAX as usize + 1),
            (NOCC * N, 0),
        );
        assert!(
            matches!(e, Err(GpuError::Layout(_))),
            "stride beyond i64: {e:?}"
        );
    }

    #[test]
    fn zero_sizes_are_total() {
        let (lg, rg) = dfk_geoms(N, NOCC);
        // m = 0, n = 0 or batch = 0: nothing is read or written, empty buffers are fine
        // (the operand SHAPES must still match the dims, so each case builds its own twins).
        let h_m0 = Array2::<f64>::zeros((0, N));
        let lg_m0 = OperandGeom::derive_padded(Role::Left, &h_m0.view(), 0).unwrap();
        assert!(check(
            dims(0, N, N, BATCH),
            &lg_m0,
            &rg,
            (0, 0),
            (0, N * N),
            (0, 0)
        )
        .is_ok());
        let g_n0 = Array2::<f64>::zeros((N, 0));
        let rg_n0 = OperandGeom::derive_padded(Role::Right, &g_n0.view(), 0).unwrap();
        assert!(check(
            dims(NOCC, N, 0, BATCH),
            &lg,
            &rg_n0,
            (0, 0),
            (0, N * N),
            (0, 0)
        )
        .is_ok());
        assert!(check(dims(NOCC, N, N, 0), &lg, &rg, (0, 0), (0, N * N), (0, 0)).is_ok());
        // k = 0 with real m, n, batch: the product is the zero matrix, so the OUTPUT
        // extent is still checked (the GEMM zeroes it) while the operands are not read.
        let h = Array2::<f64>::zeros((NOCC, 0));
        let g = Array2::<f64>::zeros((0, N));
        let lg0 = OperandGeom::derive_padded(Role::Left, &h.view(), 0).unwrap();
        let rg0 = OperandGeom::derive_padded(Role::Right, &g.view(), 0).unwrap();
        let ok = check(
            dims(NOCC, 0, N, BATCH),
            &lg0,
            &rg0,
            (0, 0),
            (0, 0),
            (BATCH * NOCC * N, NOCC * N),
        );
        assert!(ok.is_ok(), "{ok:?}");
        let e = check(
            dims(NOCC, 0, N, BATCH),
            &lg0,
            &rg0,
            (0, 0),
            (0, 0),
            (BATCH * NOCC * N - 1, NOCC * N),
        );
        assert!(matches!(e, Err(GpuError::Layout(_))), "{e:?}");
    }

    #[test]
    fn a_batch_count_that_disagrees_with_the_dims_is_a_layout_error_at_the_gemm_not_here() {
        // The pure check is stateless about the operands' own `batch` field; the
        // GEMM compares it with `BatchedDims::batch` (device test below). This
        // test pins that the helper itself has no hidden dependency on it.
        let (lg, rg) = dfk_geoms(N, NOCC);
        assert!(check(
            dims(NOCC, N, N, 2),
            &lg,
            &rg,
            (N * NOCC, 0),
            (2 * N * N, N * N),
            (2 * NOCC * N, NOCC * N)
        )
        .is_ok());
    }

    #[test]
    fn syrk_geometry_checks_both_buffers_and_is_total() {
        assert!(check_syrk_geometry(5, 7, 35, 25).is_ok());
        assert!(matches!(
            check_syrk_geometry(5, 7, 34, 25),
            Err(GpuError::Layout(_))
        ));
        assert!(matches!(
            check_syrk_geometry(5, 7, 35, 24),
            Err(GpuError::Layout(_))
        ));
        // k = 0 still needs C (it is zeroed); n = 0 needs nothing.
        assert!(check_syrk_geometry(5, 0, 0, 25).is_ok());
        assert!(matches!(
            check_syrk_geometry(5, 0, 0, 24),
            Err(GpuError::Layout(_))
        ));
        assert!(check_syrk_geometry(0, 9, 0, 0).is_ok());
        let big = i32::MAX as usize + 1;
        assert!(matches!(
            check_syrk_geometry(big, 1, usize::MAX, usize::MAX),
            Err(GpuError::Layout(_))
        ));
        assert!(matches!(
            check_syrk_geometry(1, big, usize::MAX, usize::MAX),
            Err(GpuError::Layout(_))
        ));
    }

    #[test]
    fn a_gapped_output_stride_needs_exactly_the_last_panel() {
        let (lg, rg) = dfk_geoms(N, NOCC);
        let d = dims(NOCC, N, N, BATCH);
        let mn = NOCC * N;
        let cs = mn + 3;
        let need = (BATCH - 1) * cs + mn;
        let r = (BATCH * N * N, N * N);
        let ok = check(d, &lg, &rg, (N * NOCC, 0), r, (need, cs));
        assert!(ok.is_ok(), "{ok:?}");
        let e = check(d, &lg, &rg, (N * NOCC, 0), r, (need - 1, cs));
        assert!(matches!(e, Err(GpuError::Layout(_))), "{e:?}");
    }

    #[test]
    fn a_nonbroadcast_left_operand_is_extent_checked_over_every_batch() {
        let (lg, rg) = dfk_geoms(N, NOCC);
        let d = dims(NOCC, N, N, BATCH);
        let stride = N * NOCC;
        let need = (BATCH - 1) * stride + N * NOCC;
        let c = (BATCH * NOCC * N, NOCC * N);
        let r = (BATCH * N * N, N * N);
        assert!(check(d, &lg, &rg, (need, stride), r, c).is_ok());
        let e = check(d, &lg, &rg, (need - 1, stride), r, c);
        assert!(matches!(e, Err(GpuError::Layout(_))), "{e:?}");
    }

    #[test]
    fn an_operand_in_the_wrong_slot_is_refused() {
        let (lg, rg) = dfk_geoms(N, NOCC);
        let d = dims(NOCC, N, N, BATCH);
        let c = (BATCH * NOCC * N, NOCC * N);
        let e = check(d, &rg, &lg, (BATCH * N * N, N * N), (N * NOCC, 0), c);
        assert!(matches!(e, Err(GpuError::Layout(_))), "{e:?}");
    }

    #[test]
    fn zero_axis_combinations_with_nonzero_strides_are_accepted() {
        // Operands are empty when m, k or n is zero, whatever the strides and batch.
        for (m, k, n, batch) in [
            (0, 4, 5, 3),
            (3, 4, 0, 3),
            (3, 0, 5, 3),
            (0, 0, 0, 0),
            (3, 4, 5, 0),
            (0, 4, 0, 2),
        ] {
            let h_l = Array2::<f64>::zeros((m, k));
            let h_r = Array2::<f64>::zeros((k, n));
            let lg = OperandGeom::derive_padded(Role::Left, &h_l.view(), m * k).unwrap();
            let rg = OperandGeom::derive_padded(Role::Right, &h_r.view(), k * n).unwrap();
            // only k = 0 with a real output still writes: it needs the last panel
            let writes = m > 0 && n > 0 && batch > 0;
            let clen = if writes { 3 * m * n + 8 } else { 0 };
            let d = dims(m, k, n, batch);
            let ok = check(d, &lg, &rg, (m * k, 17), (k * n, 19), (clen, m * n + 2));
            assert!(ok.is_ok(), "m={m} k={k} n={n} batch={batch}: {ok:?}");
        }
    }
}
