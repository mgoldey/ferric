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
//!
//! [`gemv_f32mat_f64_dev`] is the same product with the matrix resident as f32
//! (4 B/element) and an f64 vector, accumulator and result, by a hand-written
//! kernel (`kernels/gemv_f32_f64.cu`). Its bound is [`mixed_gemv_error_factor`].
use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublas::{Gemv, GemvConfig};
use cudarc::driver::{CudaView, CudaViewMut, LaunchConfig, PushKernelArg};

use super::mixed_host::{gamma, U32, U64};

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

/// ε of the f32-matrix GEMV: for `A` the f64 matrix whose entries were rounded to
/// nearest f32 (`Â = A∘(1+δ)`, `|δ| ≤ u32`, normal range) and any f64 `x`,
///   `|ŷ − A x| ≤ ε·(|A||x|)`,  `ε = u32 + γ_k(u64)·(1 + u32)`,
/// `k` the contraction length (`cols` for `y = A·x`, `rows` for `y = Aᵀ·x`).
/// Derivation: `Σ Â_l x_l = Σ A_l x_l + Σ A_l δ_l x_l` contributes `≤ u32 Σ|A||x|`;
/// the f64 FMA accumulation (any summation order, the kernel's strided partial
/// sums and fixed reduction included) adds `≤ γ_k(u64) Σ|Â||x|` and
/// `Σ|Â||x| ≤ (1+u32) Σ|A||x|`. The vector is never rounded, so there is no
/// vector term. Against the f64 matrix `Â` itself (the "twin") the factor is
/// `γ_k(u64)(1+u32)` alone.
pub fn mixed_gemv_error_factor(k: usize) -> f64 {
    U32 + gamma(k, U64) * (1.0 + U32)
}

/// `y = A·x` (`transpose = false`) or `y = Aᵀ·x` (`true`) with `A` the resident
/// row-major `rows × cols` f32 matrix, `x` and `y` f64 (`accumulate` adds to `y`).
/// Same geometry rules, zero-dimension behaviour and typed refusals as
/// [`gemv_f64_dev`]; the kernel (`GpuError::Kernel`) is probed before anything is
/// checked or launched, so a missing kernel moves nothing. One launch covers any
/// `rows × cols` (64-bit indices), so no row chunking is needed.
pub fn gemv_f32mat_f64_dev(
    dev: &Device,
    spec: GemvSpec,
    a: &CudaView<'_, f32>,
    x: &CudaView<'_, f64>,
    y: &mut CudaViewMut<'_, f64>,
) -> Result<(), GpuError> {
    let GemvSpec {
        rows,
        cols,
        transpose,
        accumulate,
    } = spec;
    let kernels = dev.gemv_f32_f64()?;
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
    let (func, launch) = if transpose {
        (
            &kernels.t,
            LaunchConfig {
                grid_dim: (cols.div_ceil(32) as u32, 1, 1),
                block_dim: (32, 16, 1),
                shared_mem_bytes: 0,
            },
        )
    } else {
        (
            &kernels.n,
            LaunchConfig {
                grid_dim: (rows.min(1 << 20) as u32, 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            },
        )
    };
    let (rows64, cols64, acc) = (rows as u64, cols as u64, i32::from(accumulate));
    // SAFETY: the kernel signature is (const float* a, const double* x, double* y,
    // u64 rows, u64 cols, int accumulate); `check_gemv_geometry` proved `a` holds
    // rows·cols elements, `x` the op-dependent input length and `y` the output
    // length, and the kernel reads/writes exactly those ranges.
    unsafe {
        dev.stream
            .launch_builder(func)
            .arg(a)
            .arg(x)
            .arg(y)
            .arg(&rows64)
            .arg(&cols64)
            .arg(&acc)
            .launch(launch)
    }
    .map(|_| ())
    .map_err(|e| GpuError::Cuda(format!("gemv_f32_f64 {rows}x{cols} t={transpose}: {e:?}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixed_factor_is_the_f32_rounding_plus_a_negligible_f64_term() {
        let e = mixed_gemv_error_factor(1_000_000);
        assert!(e > U32 && e < U32 * (1.0 + 5e-3), "{e:e}");
        assert_eq!(mixed_gemv_error_factor(0), U32);
    }

    #[test]
    fn committed_gemv_ptx_names_both_entries_and_the_target() {
        let ptx = include_str!("kernels/gemv_f32_f64.ptx");
        for needle in [
            ".entry gemv_f32_f64_n",
            ".entry gemv_f32_f64_t",
            ".target sm_61",
            ".address_size 64",
        ] {
            assert!(ptx.contains(needle), "{needle} missing from the PTX");
        }
    }

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
