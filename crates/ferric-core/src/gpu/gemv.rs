//! Row-major f64 GEMV on resident operands (the RI-J passes: `d = A·w` and
//! `j = Aᵀ·c` over a resident `(naux × pair)` tensor).
//!
//! `A` is row-major `rows × cols`, i.e. the column-major `cols × rows` matrix
//! with `lda = cols`. `y = A·x` is cuBLAS op T on that matrix (`x` has `cols`
//! elements, `y` has `rows`); `y = Aᵀ·x` is op N (`x` has `rows`, `y` has
//! `cols`). The caller supplies only the shape and the transpose flag; the
//! leading dimension is derived here. `accumulate` selects beta 1 (add to `y`)
//! over beta 0 (overwrite), so a caller may split `A` into row chunks and
//! accumulate in ascending order.
//!
//! The Higham bound `|ŷ − y| ≤ γ_k·(|A||x|)` (k the contraction length) holds
//! for every summation order of a classical GEMV, so no particular cuBLAS
//! blocking is assumed.
use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublas::{Gemv, GemvConfig};
use cudarc::driver::{CudaView, CudaViewMut};

use super::device::{Device, GpuError};

/// Device-free soundness check for [`gemv_f64_dev`]; `lens` is the element
/// counts of `(a, x, y)`.
pub(crate) fn check_gemv_geometry(
    rows: usize,
    cols: usize,
    transpose: bool,
    lens: (usize, usize, usize),
) -> Result<(), GpuError> {
    let (a_len, x_len, y_len) = lens;
    let lay = GpuError::Layout;
    if rows > i32::MAX as usize || cols > i32::MAX as usize {
        return Err(lay(format!(
            "dimension exceeds cuBLAS i32 range: {rows}x{cols}"
        )));
    }
    let need_a = rows
        .checked_mul(cols)
        .ok_or_else(|| lay(format!("{rows}x{cols} overflows usize")))?;
    let (need_x, need_y) = if transpose {
        (rows, cols)
    } else {
        (cols, rows)
    };
    if a_len != need_a {
        return Err(lay(format!(
            "gemv matrix holds {a_len} elements, {rows}x{cols} needs {need_a}"
        )));
    }
    if x_len != need_x {
        return Err(lay(format!(
            "gemv input holds {x_len} elements, needs {need_x}"
        )));
    }
    if y_len != need_y {
        return Err(lay(format!(
            "gemv output holds {y_len} elements, needs {need_y}"
        )));
    }
    Ok(())
}

/// Shape and mode of one [`gemv_f64_dev`] call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemvSpec {
    pub rows: usize,
    pub cols: usize,
    pub transpose: bool,
    pub accumulate: bool,
}

/// `y = A·x` (`transpose = false`) or `y = Aᵀ·x` (`true`) with `A` the
/// row-major `rows × cols` resident matrix `a`; `accumulate` adds to `y`
/// instead of overwriting it. A zero dimension reads nothing: the output is
/// zeroed unless accumulating. Lengths must match exactly (typed
/// `GpuError::Layout`, nothing launched).
pub fn gemv_f64_dev(
    dev: &Device,
    spec: GemvSpec,
    a: &CudaView<'_, f64>,
    x: &CudaView<'_, f64>,
    y: &mut CudaViewMut<'_, f64>,
) -> Result<(), GpuError> {
    let GemvSpec {
        rows,
        cols,
        transpose,
        accumulate,
    } = spec;
    check_gemv_geometry(rows, cols, transpose, (a.len(), x.len(), y.len()))?;
    if rows == 0 || cols == 0 {
        if accumulate || y.is_empty() {
            return Ok(());
        }
        return dev
            .stream
            .memset_zeros(y)
            .map_err(|e| GpuError::Cuda(format!("memset y (empty gemv): {e:?}")));
    }
    let cfg = GemvConfig {
        // row-major A is the column-major (cols × rows) matrix: y = A x is op T.
        trans: if transpose {
            cublasOperation_t::CUBLAS_OP_N
        } else {
            cublasOperation_t::CUBLAS_OP_T
        },
        m: cols as i32,
        n: rows as i32,
        alpha: 1.0,
        lda: cols as i32,
        incx: 1,
        beta: if accumulate { 1.0 } else { 0.0 },
        incy: 1,
    };
    let blas = dev.blas.lock().unwrap_or_else(|e| e.into_inner());
    // SAFETY: `check_gemv_geometry` proved `a` holds exactly rows·cols elements
    // with lda = cols = m, `x` holds the op-dependent input length and `y` the
    // output length; every dimension is within i32.
    unsafe { blas.gemv(cfg, a, x, y) }
        .map_err(|e| GpuError::Cuda(format!("cublasDgemv {rows}x{cols} t={transpose}: {e:?}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_accepts_exact_lengths_and_rejects_everything_else() {
        assert!(check_gemv_geometry(3, 5, false, (15, 5, 3)).is_ok());
        assert!(check_gemv_geometry(3, 5, true, (15, 3, 5)).is_ok());
        assert!(check_gemv_geometry(0, 5, false, (0, 5, 0)).is_ok());
        assert!(check_gemv_geometry(3, 0, true, (0, 3, 0)).is_ok());
        for bad in [
            check_gemv_geometry(3, 5, false, (14, 5, 3)),
            check_gemv_geometry(3, 5, false, (15, 3, 5)),
            check_gemv_geometry(3, 5, true, (15, 5, 3)),
            check_gemv_geometry(3, 5, false, (15, 5, 4)),
            check_gemv_geometry(usize::MAX, 2, false, (0, 0, 0)),
            check_gemv_geometry(i32::MAX as usize + 1, 1, false, (0, 0, 0)),
        ] {
            assert!(matches!(bad, Err(GpuError::Layout(_))), "{bad:?}");
        }
    }
}
