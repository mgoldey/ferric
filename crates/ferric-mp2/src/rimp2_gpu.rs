//! RI-MP2 energy with `B_ov` resident on the device. Replaces ONLY the wide
//! GEMM `G_i = B_iᵀ·B_tail` of `rimp2::spin_components_from_b_ov_kappa_cpu`;
//! the pair arithmetic is the shared `rimp2::pair_energy`, folded over j
//! ascending and over i ascending exactly as the CPU path, so the device and
//! CPU energies differ only by the GEMM's rounding.
//!
//! Layout. `b_ov` is row-major (naux × nov), nov = nocc·nvir; in column-major
//! terms it is the (nov × naux) matrix S with ld = nov, S(r, P) = B[P, r].
//! `outᵀ = G_iᵀ (ntail × nvir) = B_tailᵀ · B_i` in column-major is
//!   A = rows [i·nvir, nov) of S  -> op N, lda = nov, offset i·nvir
//!   B = rows [i·nvir, (i+1)·nvir) of S, transposed -> op T, ldb = nov, offset i·nvir
//! and k (= P) runs along S's columns for both, so each k-panel starts at
//! offset k0·nov (`k_step = nov`). Both operands are column sub-blocks of the
//! resident matrix, described to the GEMM through
//! `gemm::dev_left_padded` / `dev_right_padded` from the host twin's block
//! views (the descriptor is derived from their strides, never hand-written).
//!
//! Threading. The device GEMM runs on the calling thread (never inside a rayon
//! worker — the dispatcher declines there). The host pair loop for one i is
//! parallel over j with disjoint outputs, then folded ascending: bit-identical
//! to the serial fold for any worker count (`parallel_pairs = false` is the
//! serial twin the tests compare against).
//!
//! Dimension arithmetic is checked throughout: a shape whose products do not
//! fit `usize` is a typed `GpuError::Layout`; zero-size inputs return zeros
//! (or refuse to the CPU path) without touching the device.
use ferric_core::gpu::device::{device, Device, GpuError};
use ferric_core::gpu::gemm::{dev_left_padded, dev_right_padded, gemm_f64_dev};
use ferric_core::gpu::pool::DevicePool;
use ferric_core::gpu::resident::DeviceMatrix;
use ferric_core::gpu::stats::{
    note_cpu, note_cpu_detail, note_mixed_fallback, note_offloaded, CpuReason,
};
use ferric_core::gpu::{GpuMode, GpuStatus, MixedKernel, Precision};
use ferric_tensors::einsum::GEMM_K_BLOCK;
use ndarray::{s, Array2, ArrayView2};
use rayon::prelude::*;

use crate::rimp2::{pair_energy, SpinComponents};

/// Dispatcher: `None` means "run the CPU path" (reason counted unless the
/// mode is off). Never panics; never touches CUDA when the mode is off.
pub fn try_spin_components_on_device(
    b_ov: &Array2<f64>,
    eps: &[f64],
    nocc: usize,
    nvir: usize,
    first_occ: usize,
    nocc_total: usize,
    kappa: Option<f64>,
) -> Option<SpinComponents> {
    let settings = ferric_core::gpu::settings();
    if settings.mode == GpuMode::Off {
        return None;
    }
    let GpuStatus::Ready(info) = ferric_core::gpu::status() else {
        return None;
    };
    if rayon::current_thread_index().is_some() {
        note_cpu(CpuReason::InsideRayonWorker);
        return None;
    }
    let pool = ferric_core::gpu::pool()?;
    let Ok(dev) = device(info.ordinal) else {
        note_cpu(CpuReason::CudaError);
        return None;
    };
    let precision = if settings.mixed_allows(MixedKernel::RiMp2Energy) {
        Precision::Mixed
    } else {
        Precision::F64
    };
    let run = |p| {
        spin_components_on_device(
            &dev, &pool, b_ov, eps, nocc, nvir, first_occ, nocc_total, kappa, p, true,
        )
    };
    match run(precision) {
        Ok(sc) => Some(sc),
        Err(GpuError::Kernel(reason)) if precision == Precision::Mixed => {
            // Mixed requested but the path is unavailable: f64 on the device,
            // counted and traced, never silent.
            if ferric_core::gpu::config::gpu_trace() {
                eprintln!(
                    "[gpu] RI-MP2 mixed path unavailable ({reason}); running f64 on the device"
                );
            }
            note_mixed_fallback();
            match run(Precision::F64) {
                Ok(sc) => Some(sc),
                Err(e) => {
                    count_cpu_fallback(&e);
                    None
                }
            }
        }
        Err(e) => {
            count_cpu_fallback(&e);
            None
        }
    }
}

fn count_cpu_fallback(e: &GpuError) {
    let reason = match e {
        GpuError::PoolFull { .. } => CpuReason::PoolFull,
        GpuError::Layout(_) => CpuReason::Layout,
        _ => CpuReason::CudaError,
    };
    note_cpu_detail(reason, &format!("(RI-MP2 energy: {e})"));
}

/// `nocc·nvir`, or a typed refusal when it overflows `usize`.
fn checked_nov(nocc: usize, nvir: usize) -> Result<usize, GpuError> {
    nocc.checked_mul(nvir).ok_or_else(|| {
        GpuError::Layout(format!(
            "RI-MP2 dimensions nocc={nocc} nvir={nvir} overflow usize"
        ))
    })
}

/// The device energy. `b_ov` row-major (naux × nocc·nvir). Charges the pool
/// for the resident tensor and the per-i scratch; refuses (`PoolFull`) before
/// any transfer when either does not fit. `Precision::Mixed` is a typed
/// refusal until the mixed arm ships.
pub fn spin_components_on_device(
    dev: &Device,
    pool: &DevicePool,
    b_ov: &Array2<f64>,
    eps: &[f64],
    nocc: usize,
    nvir: usize,
    first_occ: usize,
    nocc_total: usize,
    kappa: Option<f64>,
    precision: Precision,
    parallel_pairs: bool,
) -> Result<SpinComponents, GpuError> {
    let (naux, nov) = b_ov.dim();
    let want_nov = checked_nov(nocc, nvir)?;
    if nov != want_nov {
        return Err(GpuError::Layout(format!(
            "b_ov is {naux}x{nov} but nocc*nvir = {want_nov}"
        )));
    }
    let eps_needed = first_occ
        .saturating_add(nocc)
        .max(nocc_total.saturating_add(nvir));
    if eps.len() < eps_needed {
        return Err(GpuError::Layout(format!(
            "eps holds {} orbital energies but the index ranges reach {eps_needed}",
            eps.len()
        )));
    }
    if nocc == 0 || nvir == 0 {
        return Ok(SpinComponents {
            e_os: 0.0,
            e_ss: 0.0,
            e_total: 0.0,
        });
    }
    if naux == 0 {
        // No contraction depth: the CPU path owns this degenerate shape.
        return Err(GpuError::Layout("empty auxiliary dimension".into()));
    }
    let mut stage = Stage::new(dev, pool, b_ov, nvir, precision)?;
    let mut host = vec![0.0f64; nvir * nov];
    let mut partials: Vec<(f64, f64)> = Vec::with_capacity(nocc);
    for i in 0..nocc {
        let ntail = (nocc - i) * nvir;
        let block = &mut host[..nvir * ntail];
        stage.g_block(dev, b_ov, i, nvir, block)?;
        let g_i = ArrayView2::from_shape((nvir, ntail), &*block).expect("shape");
        let pair = |j: usize| pair_energy(&g_i, i, j, nvir, eps, first_occ, nocc_total, kappa);
        let pairs: Vec<(f64, f64)> = if parallel_pairs {
            (i..nocc).into_par_iter().map(pair).collect()
        } else {
            (i..nocc).map(pair).collect()
        };
        let (mut e_os_i, mut e_ss_i) = (0.0, 0.0);
        for (jj, (e_os_ij, e_ss_ij)) in pairs.into_iter().enumerate() {
            let fac = if jj == 0 { 1.0 } else { 2.0 }; // jj == 0 is j == i
            e_os_i += fac * e_os_ij;
            e_ss_i += fac * e_ss_ij;
        }
        partials.push((e_os_i, e_ss_i));
    }
    let (mut e_os, mut e_ss) = (0.0, 0.0);
    for (a, b) in partials {
        e_os += a;
        e_ss += b;
    }
    Ok(SpinComponents {
        e_os,
        e_ss,
        e_total: e_os + e_ss,
    })
}

/// Resident `B_ov` + the `G_i` scratch for one stage. Both are charged to the
/// pool for the stage's lifetime.
struct Stage {
    b: DeviceMatrix<f64>,
    c64: DeviceMatrix<f64>,
}

impl Stage {
    fn new(
        dev: &Device,
        pool: &DevicePool,
        b_ov: &Array2<f64>,
        nvir: usize,
        precision: Precision,
    ) -> Result<Self, GpuError> {
        if precision == Precision::Mixed {
            // The mixed arm (f32-resident B_ov) ships separately; the
            // dispatcher turns this into the counted f64 device path.
            return Err(GpuError::Kernel("mixed RI-MP2 not shipped".into()));
        }
        let nov = b_ov.ncols();
        // Scratch first: a pool too small for the scratch refuses before the
        // (larger) B_ov upload is even attempted. Its size is nvir·nov, the
        // widest block (i = 0); `zeros` checks the product.
        let c64 = DeviceMatrix::<f64>::zeros(
            dev,
            pool,
            "RI-MP2 energy (device): G_i scratch f64",
            nvir,
            nov,
        )?;
        let b = DeviceMatrix::<f64>::upload(dev, pool, "RI-MP2 B_ov (device, f64)", &b_ov.view())?;
        Ok(Self { b, c64 })
    }

    /// Computes G_i into the device scratch and downloads it (row-major
    /// (nvir × ntail)) into `host`. `b_ov` is the host twin whose block views
    /// the operand descriptors are derived from. `&mut self` because the GEMM
    /// takes `&mut CudaSlice` for the scratch.
    fn g_block(
        &mut self,
        dev: &Device,
        b_ov: &Array2<f64>,
        i: usize,
        nvir: usize,
        host: &mut [f64],
    ) -> Result<(), GpuError> {
        let cuda =
            |what: &str, e: &dyn std::fmt::Debug| GpuError::Cuda(format!("{what} G_{i}: {e:?}"));
        let nov = b_ov.ncols();
        let off = i.checked_mul(nvir).filter(|&o| o < nov).ok_or_else(|| {
            GpuError::Layout(format!("block {i} of width {nvir} is outside nov = {nov}"))
        })?;
        let (m, k, n) = (nvir, b_ov.nrows(), nov - off);
        let b_i = b_ov.slice(s![.., off..off + nvir]);
        let b_tail = b_ov.slice(s![.., off..]);
        let left = dev_left_padded(self.b.buf().slice(off..), &b_i.t())?;
        let right = dev_right_padded(self.b.buf().slice(off..), &b_tail)?;
        gemm_f64_dev(
            dev,
            m,
            k,
            n,
            &left,
            &right,
            self.c64.buf_mut(),
            GEMM_K_BLOCK,
        )?;
        // m·n ≤ nvir·nov (checked when the scratch was sized).
        note_offloaded(0, 8 * m * n);
        if host.len() != m * n {
            return Err(GpuError::Layout(format!(
                "host block holds {} elements, G_{i} has {}",
                host.len(),
                m * n
            )));
        }
        dev.stream
            .memcpy_dtoh(&self.c64.buf().slice(..m * n), host)
            .map_err(|e| cuda("D2H", &e))?;
        dev.stream.synchronize().map_err(|e| cuda("sync", &e))?;
        Ok(())
    }
}

/// One `G_i` block from the device (row-major (nvir × (nocc−i)·nvir)), for the
/// per-element bound tests.
#[doc(hidden)]
pub fn g_block_on_device(
    dev: &Device,
    pool: &DevicePool,
    b_ov: &Array2<f64>,
    i: usize,
    nvir: usize,
    precision: Precision,
) -> Result<Array2<f64>, GpuError> {
    let nov = b_ov.ncols();
    let ntail = i
        .checked_mul(nvir)
        .and_then(|off| nov.checked_sub(off))
        .filter(|&t| t > 0)
        .ok_or_else(|| {
            GpuError::Layout(format!("block {i} of width {nvir} is outside nov = {nov}"))
        })?;
    if b_ov.nrows() == 0 {
        return Err(GpuError::Layout("empty auxiliary dimension".into()));
    }
    let mut stage = Stage::new(dev, pool, b_ov, nvir, precision)?;
    let len = nvir
        .checked_mul(ntail)
        .ok_or_else(|| GpuError::Layout("G_i block overflows usize".into()))?;
    let mut host = vec![0.0f64; len];
    stage.g_block(dev, b_ov, i, nvir, &mut host)?;
    Ok(Array2::from_shape_vec((nvir, ntail), host).expect("shape"))
}
