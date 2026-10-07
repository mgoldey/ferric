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
//! Why a kernel: cublasGemmEx offers no f32-input / f64-accumulate combination
//! (64F compute requires 64F A/B/C), and downloading each f32 panel to
//! accumulate on the host would move ⌈k/b⌉·4 bytes per output element instead
//! of 8 (3.5x the traffic at k = 912, b = 128).
use cudarc::cublas::{Gemm, GemmConfig};
use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use ndarray::{ArrayView2, ArrayViewMut2};

use super::device::{Device, GpuError};
use super::gemm::{col_major_desc, DevOperand};
use super::pool::DevicePool;
use super::stats;

/// TEMP: replaced by `gpu::precision::mixed_k_panel_override` once Task 4.0
/// lands (one-line import swap). `FERRIC_GPU_MIXED_K_PANEL`; 0 or unset or
/// unparsable means no override.
fn mixed_k_panel_override() -> Option<usize> {
    std::env::var("FERRIC_GPU_MIXED_K_PANEL")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&b| b > 0)
}

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
    let func = dev.axpy_f32_to_f64()?; // typed Kernel error BEFORE any work
    let s = &dev.stream;
    s.memset_zeros(c64)
        .map_err(|e| cuda(format_args!("memset c64: {e:?}")))?;
    if k == 0 || m == 0 || n == 0 {
        return Ok(0);
    }
    let n_elems = (m * n) as u64;
    let launch = flush_launch(m * n);
    let blas = dev.blas.lock().unwrap_or_else(|e| e.into_inner());
    let kb = k_panel.max(1);
    let (mut k0, mut panels) = (0usize, 0usize);
    while k0 < k {
        let k1 = (k0 + kb).min(k);
        let cfg = GemmConfig::<f32> {
            transa: right.op,
            transb: left.op,
            m: n as i32,
            n: m as i32,
            k: (k1 - k0) as i32,
            alpha: 1.0,
            lda: right.ld,
            ldb: left.ld,
            beta: 0.0, // every panel starts fresh in f32; the sum lives in c64
            ldc: n as i32,
        };
        let a_view = right.view.slice(k0 * right.k_step..);
        let b_view = left.view.slice(k0 * left.k_step..);
        // SAFETY: shapes/leading dimensions come from the descriptors; the
        // buffers hold exactly those elements (same argument as gemm_f64).
        unsafe { blas.gemm(cfg, &a_view, &b_view, c32) }
            .map_err(|e| cuda(format_args!("cublasSgemm panel {k0}..{k1}: {e:?}")))?;
        // SAFETY: the kernel signature is (double*, const float*, u64) and the
        // two buffers hold >= n_elems elements each.
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

/// Host-view convenience (einsum path, tests): rounds and uploads the
/// operands, runs the device core, downloads the f64 result. Same layout
/// rules and refusals as `gemm_f64`.
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
    let da = col_major_desc(left, true).ok_or_else(|| {
        GpuError::Layout("left operand is neither standard nor transposed-standard".into())
    })?;
    let db = col_major_desc(right, false).ok_or_else(|| {
        GpuError::Layout("right operand is neither standard nor transposed-standard".into())
    })?;
    let la: Vec<f32> = left
        .as_slice_memory_order()
        .ok_or_else(|| GpuError::Layout("left not contiguous".into()))?
        .iter()
        .map(|&x| x as f32)
        .collect();
    let rb: Vec<f32> = right
        .as_slice_memory_order()
        .ok_or_else(|| GpuError::Layout("right not contiguous".into()))?
        .iter()
        .map(|&x| x as f32)
        .collect();
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
    let left_op = DevOperand {
        view: d_a.slice(..),
        op: da.op,
        ld: da.ld,
        k_step: da.k_step,
    };
    let right_op = DevOperand {
        view: d_b.slice(..),
        op: db.op,
        ld: db.ld,
        k_step: db.k_step,
    };
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
