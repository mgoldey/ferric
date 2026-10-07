//! k-panelled SGEMM with f64 accumulation (plan §3.2 variant (b)):
//! ```text
//! for each panel [k0, k1):  c32 = A[:,k0:k1]·B[k0:k1,:]   (cublasSgemm, beta 0)
//!                           c64 += (double) c32          (axpy_f32_to_f64 PTX kernel)
//! ```
//! Operands are rounded to f32 (host `as f32`, round-to-nearest-even) once;
//! the result is f64. Bound per element vs the exact product:
//! (2u32 + γ_b(u32) + γ_{⌈k/b⌉}(u64))·(|A||B|)_ij — see `mixed_host.rs`.
//! Run-to-run on one device the result is expected bit-identical (cuBLAS
//! §2.1.4 reproducibility + a deterministic flush); the parity test measures it.
//!
//! Operand range: rounding to f32 turns |x| > f32::MAX into inf (the host
//! wrapper refuses that with `GpuError::F32Range` so the caller can run f64)
//! and loses the relative 2u32 term for f32-subnormal operands (|x| < 1.2e-38,
//! absolute error <= 7e-46 per element instead); the model bound assumes
//! normal-range operands.
//!
//! Variant (c) (hi/lo split, three products) was measured and is NOT shipped:
//! at (256,8192,256) b = 128 its rms e is 5.7e-9 against 5.8e-9 for variant (b),
//! ratio ~1.0 against the pre-registered 2x, because the error is the f32
//! panel accumulation, not the operand rounding.
//!
//! Why a kernel: cublasGemmEx offers no f32-input / f64-accumulate combination
//! (64F compute requires 64F A/B/C), and downloading each f32 panel to
//! accumulate on the host would move ⌈k/b⌉·4 bytes per output element instead
//! of 8 (4x the traffic at k = 912, b = 128: ⌈912/128⌉ = 8 panels, 8·4/8).
use cudarc::cublas::{Gemm, GemmConfig};
use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use ndarray::{ArrayView2, ArrayViewMut2};

use super::device::{Device, GpuError};
use super::gemm::{check_dev_geometry, col_major_desc, DevOperand, Role};
use super::pool::DevicePool;
use super::precision::mixed_k_panel_override;
use super::stats;

/// Panel width. Provisional, set by the quiet-box sweep: 128 mirrors
/// `GEMM_K_BLOCK` as a placeholder until
/// `benchmarks/harness/examples/gpu_mixed_gemm_sweep.rs` runs on a quiet box
/// (the accuracy half alone cannot choose: rms e falls monotonically with
/// smaller b, so the throughput half decides). Accuracy measured at
/// (256,8192,256), positive operands, rms e: b=64 3.9e-9, 128 5.8e-9,
/// 256 8.9e-9, 512 1.4e-8, 1024 2.4e-8; plain sgemm 1.6e-7.
/// Rule for the value: the smallest b in {64,128,256,512,1024} whose resident
/// mixed throughput is >= 0.25 x the resident plain-SGEMM throughput at BOTH
/// (393,912,5895) and (256,8192,256). Smaller b is always more accurate
/// (`mixed_host.rs`); this picks the most accurate b that keeps the lane at
/// >= 7x DGEMM by the §2 flush model. The measured table goes here.
pub const MIXED_K_PANEL_DEFAULT: usize = 128;

/// The panel width in force: the env override if set, else the default.
pub fn effective_k_panel() -> usize {
    mixed_k_panel_override().unwrap_or(MIXED_K_PANEL_DEFAULT)
}

/// Bytes charged for one host-view call: f32 operands, f32 scratch, f64 result.
pub fn mixed_offload_bytes(m: usize, k: usize, n: usize) -> usize {
    4 * (m * k + k * n) + 4 * m * n + 8 * m * n
}

/// Flush grid: 256 threads, at most 4096 blocks, grid-stride loop in the kernel.
fn flush_launch(n_elems: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n_elems.div_ceil(256).clamp(1, 4096) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Device core on resident f32 operands. `c32` and `c64` hold m·n elements
/// each (column-major n × m = row-major m × n); `c64` is zeroed here, so the
/// result is `left·right` (beta 0). Returns the number of panels flushed.
/// Operand shapes and both buffer lengths are checked first (typed
/// `GpuError::Layout`, nothing launched or written).
pub fn gemm_f32_f64acc_dev(
    dev: &Device,
    m: usize,
    k: usize,
    n: usize,
    left: &DevOperand<'_, f32>,
    right: &DevOperand<'_, f32>,
    c32: &mut CudaSlice<f32>,
    c64: &mut CudaSlice<f64>,
    k_panel: usize,
) -> Result<usize, GpuError> {
    let cuda = |e: std::fmt::Arguments| GpuError::Cuda(e.to_string());
    check_dev_geometry(
        m,
        k,
        n,
        k_panel,
        (&left.geom, left.view.len()),
        (&right.geom, right.view.len()),
        &[("c32", c32.len()), ("c64", c64.len())],
    )?;
    let func = dev.axpy_f32_to_f64()?; // typed Kernel error BEFORE any work
    if m == 0 || n == 0 {
        return Ok(0); // empty product: nothing to compute, nothing to write
    }
    let s = &dev.stream;
    s.memset_zeros(c64)
        .map_err(|e| cuda(format_args!("memset c64: {e:?}")))?;
    if k == 0 {
        return Ok(0); // c64 is now the zero matrix
    }
    let n_elems = (m * n) as u64;
    let launch = flush_launch(m * n);
    let blas = dev.blas.lock().unwrap_or_else(|e| e.into_inner());
    let kb = k_panel.max(1);
    let (mut k0, mut panels) = (0usize, 0usize);
    while k0 < k {
        let k1 = (k0 + kb).min(k);
        let cfg = GemmConfig::<f32> {
            transa: right.geom.op,
            transb: left.geom.op,
            m: n as i32,
            n: m as i32,
            k: (k1 - k0) as i32,
            alpha: 1.0,
            lda: right.geom.ld,
            ldb: left.geom.ld,
            beta: 0.0, // every panel starts fresh in f32; the sum lives in c64
            ldc: n as i32,
        };
        let a_view = right.view.slice(k0 * right.geom.k_step..);
        let b_view = left.view.slice(k0 * left.geom.k_step..);
        // SAFETY: `check_dev_geometry` above proved, for every panel, that the
        // highest element each operand read touches lies inside its view and
        // that c32 and c64 hold AT LEAST m·n elements each (they may hold more;
        // only the first m·n are written).
        unsafe { blas.gemm(cfg, &a_view, &b_view, c32) }
            .map_err(|e| cuda(format_args!("cublasSgemm panel {k0}..{k1}: {e:?}")))?;
        // SAFETY: the kernel signature is (double*, const float*, u64) and the
        // two buffers hold >= n_elems = m·n elements each (checked above).
        unsafe {
            s.launch_builder(func)
                .arg(&mut *c64)
                .arg(&*c32)
                .arg(&n_elems)
                .launch(launch)
        }
        .map_err(|e| cuda(format_args!("axpy_f32_to_f64 panel {k0}..{k1}: {e:?}")))?;
        k0 = k1;
        panels += 1;
    }
    Ok(panels)
}

/// `x as f32` for a whole operand, refusing any finite f64 that overflows to
/// inf in f32 (`GpuError::F32Range`, naming the operand and the first index).
/// Existing inf/NaN inputs pass through unchanged (they are the caller's data).
fn round_to_f32(label: &str, x: &[f64]) -> Result<Vec<f32>, GpuError> {
    let mut out = Vec::with_capacity(x.len());
    for (i, &v) in x.iter().enumerate() {
        let r = v as f32;
        if v.is_finite() && !r.is_finite() {
            return Err(GpuError::F32Range(format!(
                "{label}[{i}] = {v:e} exceeds f32::MAX"
            )));
        }
        out.push(r);
    }
    Ok(out)
}

/// Host-view convenience (einsum path, tests): rounds and uploads the
/// operands, runs the device core, downloads the f64 result. Same layout
/// rules and refusals as `gemm_f64`.
///
/// Precondition: operands in the f32 normal range. A finite |x| > f32::MAX is
/// refused with `GpuError::F32Range` (before any lease or transfer) so the
/// caller can run f64; f32-subnormal operands (|x| < 1.2e-38) are accepted but
/// lose the relative 2u32 term of the error bound. The kernel (`GpuError::Kernel`)
/// and range checks run before the pool is touched.
pub fn gemm_f32_f64acc(
    dev: &Device,
    pool: &DevicePool,
    left: &ArrayView2<f64>,
    right: &ArrayView2<f64>,
    out: &mut ArrayViewMut2<f64>,
    k_panel: usize,
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
    if col_major_desc(left, true).is_none() {
        return Err(GpuError::Layout(
            "left operand is neither standard nor transposed-standard".into(),
        ));
    }
    if col_major_desc(right, false).is_none() {
        return Err(GpuError::Layout(
            "right operand is neither standard nor transposed-standard".into(),
        ));
    }
    // Kernel availability BEFORE any conversion, lease or transfer, so the
    // caller's f64 fallback wastes no work.
    dev.axpy_f32_to_f64()?;
    let la = round_to_f32(
        "left",
        left.as_slice_memory_order()
            .ok_or_else(|| GpuError::Layout("left not contiguous".into()))?,
    )?;
    let rb = round_to_f32(
        "right",
        right
            .as_slice_memory_order()
            .ok_or_else(|| GpuError::Layout("right not contiguous".into()))?,
    )?;
    let bytes = mixed_offload_bytes(m, k, n);
    let _lease = pool.reserve("gemm_f32_f64acc operands+scratch+result", bytes)?;
    let s = &dev.stream;
    let cuda = |e: std::fmt::Arguments| GpuError::Cuda(e.to_string());
    let d_a: CudaSlice<f32> = s
        .clone_htod(&la)
        .map_err(|e| cuda(format_args!("H2D left: {e:?}")))?;
    let d_b: CudaSlice<f32> = s
        .clone_htod(&rb)
        .map_err(|e| cuda(format_args!("H2D right: {e:?}")))?;
    let mut c32: CudaSlice<f32> = s
        .alloc_zeros(m * n)
        .map_err(|e| cuda(format_args!("alloc c32: {e:?}")))?;
    let mut c64: CudaSlice<f64> = s
        .alloc_zeros(m * n)
        .map_err(|e| cuda(format_args!("alloc c64: {e:?}")))?;
    let left_op = DevOperand::new(d_a.slice(..), left, Role::Left)?;
    let right_op = DevOperand::new(d_b.slice(..), right, Role::Right)?;
    let panels = gemm_f32_f64acc_dev(
        dev, m, k, n, &left_op, &right_op, &mut c32, &mut c64, k_panel,
    )?;
    let host = out.as_slice_mut().expect("standard layout checked above");
    s.memcpy_dtoh(&c64, host)
        .map_err(|e| cuda(format_args!("D2H out: {e:?}")))?;
    s.synchronize()
        .map_err(|e| cuda(format_args!("sync: {e:?}")))?;
    stats::note_mixed(panels, 4 * (m * k + k * n), 8 * m * n);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn committed_ptx_names_the_entry_and_target() {
        let ptx = include_str!("kernels/axpy_f32_to_f64.ptx");
        assert!(
            ptx.contains(".entry axpy_f32_to_f64"),
            "entry point renamed?"
        );
        assert!(
            ptx.contains(".target sm_61"),
            "regenerate with -arch=compute_61"
        );
        assert!(ptx.contains(".address_size 64"));
    }

    #[test]
    fn f32_range_overflow_is_a_typed_error_naming_the_element() {
        let e = round_to_f32("left", &[1.0, 2.0, 1e300, 4.0]).unwrap_err();
        assert!(
            matches!(e, GpuError::F32Range(ref s) if s.contains("left[2]")),
            "{e:?}"
        );
        let ok = round_to_f32("right", &[f64::from(f32::MAX), -1e-300, 0.0]).unwrap();
        assert_eq!(ok, vec![f32::MAX, 0.0, 0.0]);
        // already-non-finite input is the caller's data, not an overflow
        assert!(round_to_f32("x", &[f64::INFINITY, f64::NAN]).is_ok());
    }

    #[test]
    fn flush_grid_is_capped_and_never_zero() {
        assert_eq!(flush_launch(1).grid_dim.0, 1);
        assert_eq!(flush_launch(256 * 10).grid_dim.0, 10);
        assert_eq!(flush_launch(1 << 30).grid_dim.0, 4096);
    }

    #[test]
    fn mixed_bytes_count_all_four_buffers() {
        assert_eq!(mixed_offload_bytes(2, 3, 5), 4 * (6 + 15) + 4 * 10 + 8 * 10);
    }
}
