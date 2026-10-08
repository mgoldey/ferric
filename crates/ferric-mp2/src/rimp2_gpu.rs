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
use ferric_core::gpu::mixed::{effective_k_panel, gemm_f32_f64acc_dev};
use ferric_core::gpu::pool::DevicePool;
use ferric_core::gpu::resident::DeviceMatrix;
use ferric_core::gpu::stats::{
    note_cpu, note_cpu_detail, note_mixed, note_mixed_fallback, note_offloaded, CpuReason,
};
use ferric_core::gpu::{GpuMode, GpuStatus, MixedKernel, Precision};
use ferric_tensors::einsum::GEMM_K_BLOCK;
use ndarray::{s, Array2, ArrayView2};
use rayon::prelude::*;
#[cfg(feature = "test-seams")]
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use crate::rimp2::{pair_energy, SpinComponents};
use crate::u_rimp2::{opposite_spin_block_energy, same_spin_block_energy, SpinChannel};

/// Test hook (`test-seams` feature; mirrors `FORCE_KERNEL_FAILURE`): when set to a
/// block index, the device block `G_i` with that `i` fails with a typed error
/// before any work, so the mid-loop failure path (counters, fallback) can be
/// exercised.
#[cfg(feature = "test-seams")]
#[doc(hidden)]
pub static FAIL_AT_BLOCK: AtomicUsize = AtomicUsize::new(usize::MAX);
/// Test hook (`test-seams` feature): the injected failure is a
/// `GpuError::Kernel` in the MIXED stage only (a mixed-path failure the
/// dispatcher turns into the f64 device path) when `true`, else a
/// `GpuError::Cuda` in either stage (a CPU fallback).
#[cfg(feature = "test-seams")]
#[doc(hidden)]
pub static FAIL_AS_KERNEL: AtomicBool = AtomicBool::new(false);

/// The block index an injected failure targets (`usize::MAX` = none).
#[cfg(feature = "test-seams")]
fn fail_at_block() -> usize {
    FAIL_AT_BLOCK.load(Ordering::Relaxed)
}

#[cfg(not(feature = "test-seams"))]
fn fail_at_block() -> usize {
    usize::MAX
}

#[cfg(feature = "test-seams")]
fn fail_as_kernel() -> bool {
    FAIL_AS_KERNEL.load(Ordering::Relaxed)
}

#[cfg(not(feature = "test-seams"))]
fn fail_as_kernel() -> bool {
    false
}

/// Counter updates of finished blocks, applied only when the whole call
/// succeeds so that a run that ends as the f64 device path or the CPU path
/// never shows mixed or offloaded work it threw away.
#[derive(Default)]
struct Tally {
    f64_blocks: Vec<usize>,
    mixed_blocks: Vec<(usize, usize)>,
}

impl Tally {
    fn commit(self) {
        for d2h in self.f64_blocks {
            note_offloaded(0, d2h);
        }
        for (panels, d2h) in self.mixed_blocks {
            note_mixed(panels, 0, d2h);
        }
    }
}

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
    mixed_ok: bool,
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
    let precision = if mixed_ok && settings.mixed_allows(MixedKernel::RiMp2Energy) {
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
        Err(e @ (GpuError::Kernel(_) | GpuError::F32Range(_))) if precision == Precision::Mixed => {
            // Mixed requested but the path is unavailable (flush kernel will
            // not load) or the operand does not survive f32 (a finite value
            // beyond f32::MAX): f64 on the device, counted and traced, never
            // silent and never a wrong number.
            if ferric_core::gpu::config::gpu_trace() {
                eprintln!("[gpu] RI-MP2 mixed path unavailable ({e}); running f64 on the device");
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
/// any transfer when either does not fit. `Precision::Mixed` keeps `B_ov` on the
/// device as f32 (half the bytes) and forms `G_i` by the k-panelled SGEMM with
/// f64 accumulation; it refuses with `GpuError::Kernel` (flush kernel will not
/// load; checked before any reservation) or `GpuError::F32Range` (a finite
/// `B_ov` element beyond `f32::MAX`), never a silent f64 run.
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
    stage.tally.commit();
    Ok(SpinComponents {
        e_os,
        e_ss,
        e_total: e_os + e_ss,
    })
}

/// Refuses (`GpuError::F32Range`, to the f64 device path) a `B_ov` whose f32
/// PANEL SUMS could overflow: with every finite |B| ≤ M and panels of b terms,
/// a panel sum of products is ≤ b·M², which stays below `f32::MAX` iff
/// M ≤ sqrt(f32::MAX / b). Storage alone (`upload_rounded`) only refuses an
/// element beyond `f32::MAX`; this guards the arithmetic. NaN and ±inf are the
/// caller's data, not a range violation: they pass through to a NaN energy, as
/// on the CPU path.
fn check_f32_arithmetic_range(b_ov: &Array2<f64>) -> Result<(), GpuError> {
    let b = effective_k_panel().clamp(1, b_ov.nrows().max(1));
    let limit = (f64::from(f32::MAX) / b as f64).sqrt();
    let max = b_ov
        .iter()
        .filter(|x| x.is_finite())
        .fold(0.0f64, |m, x| m.max(x.abs()));
    if max > limit {
        return Err(GpuError::F32Range(format!(
            "max|B_ov| = {max:e} exceeds sqrt(f32::MAX/b) = {limit:e} (panel width {b}): an f32 panel sum could overflow"
        )));
    }
    Ok(())
}

/// `B_ov` resident on the device in the precision of the stage.
enum Resident {
    F64(DeviceMatrix<f64>),
    F32(DeviceMatrix<f32>),
}

/// Resident `B_ov` + the `G_i` scratch for one stage. All are charged to the
/// pool, each under its own label, for the stage's lifetime. The mixed stage
/// also holds the f32 panel scratch.
struct Stage {
    b: Resident,
    c64: DeviceMatrix<f64>,
    c32: Option<DeviceMatrix<f32>>,
    tally: Tally,
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
            // The flush kernel is checked before ANY reservation or transfer, so
            // the dispatcher's f64 fallback wastes nothing.
            dev.axpy_f32_to_f64()?;
            check_f32_arithmetic_range(b_ov)?;
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
        match precision {
            Precision::F64 => {
                let b = DeviceMatrix::<f64>::upload(
                    dev,
                    pool,
                    "RI-MP2 B_ov (device, f64)",
                    &b_ov.view(),
                )?;
                Ok(Self {
                    b: Resident::F64(b),
                    c64,
                    c32: None,
                    tally: Tally::default(),
                })
            }
            Precision::Mixed => {
                let c32 = DeviceMatrix::<f32>::zeros(
                    dev,
                    pool,
                    "RI-MP2 energy (device): G_i scratch f32 panel",
                    nvir,
                    nov,
                )?;
                let b = DeviceMatrix::<f32>::upload_rounded(
                    dev,
                    pool,
                    "RI-MP2 B_ov (device, f32)",
                    &b_ov.view(),
                )?;
                Ok(Self {
                    b: Resident::F32(b),
                    c64,
                    c32: Some(c32),
                    tally: Tally::default(),
                })
            }
        }
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
        let kernel_class = fail_as_kernel();
        // a Kernel-class injection models a MIXED-path failure: the f64 stage is
        // not affected, so the dispatcher's f64 device fallback can complete
        if fail_at_block() == i && (!kernel_class || matches!(self.b, Resident::F32(_))) {
            return Err(if kernel_class {
                GpuError::Kernel(format!("injected failure at block {i} (test)"))
            } else {
                GpuError::Cuda(format!("injected failure at block {i} (test)"))
            });
        }
        let nov = b_ov.ncols();
        // The whole block [off, off + nvir) must lie inside the matrix: a width
        // that is not a multiple of nvir would otherwise leave a partial last
        // block and the slice below would panic.
        let off = i
            .checked_mul(nvir)
            .filter(|&o| o < nov && nvir <= nov - o)
            .ok_or_else(|| {
                GpuError::Layout(format!("block {i} of width {nvir} is outside nov = {nov}"))
            })?;
        let (m, k, n) = (nvir, b_ov.nrows(), nov - off);
        let b_i = b_ov.slice(s![.., off..off + nvir]);
        let b_tail = b_ov.slice(s![.., off..]);
        let mut mixed_panels = 0usize; // 0 = an f64 block
        match (&self.b, &mut self.c32) {
            (Resident::F64(b), _) => {
                let left = dev_left_padded(b.buf().slice(off..), &b_i.t())?;
                let right = dev_right_padded(b.buf().slice(off..), &b_tail)?;
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
            }
            (Resident::F32(b), Some(c32)) => {
                let left = dev_left_padded(b.buf().slice(off..), &b_i.t())?;
                let right = dev_right_padded(b.buf().slice(off..), &b_tail)?;
                let panels = gemm_f32_f64acc_dev(
                    dev,
                    m,
                    k,
                    n,
                    &left,
                    &right,
                    c32.buf_mut(),
                    self.c64.buf_mut(),
                    effective_k_panel(),
                )?;
                mixed_panels = panels;
            }
            (Resident::F32(_), None) => {
                return Err(GpuError::Layout(
                    "mixed stage without its f32 scratch".into(),
                ))
            }
        }
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
        // Counted only now, and only applied by `Tally::commit` when the whole
        // call succeeds.
        if mixed_panels > 0 {
            self.tally.mixed_blocks.push((mixed_panels, 8 * m * n));
        } else {
            self.tally.f64_blocks.push(8 * m * n);
        }
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
        .filter(|&t| t > 0 && t >= nvir)
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
    stage.tally.commit();
    Ok(Array2::from_shape_vec((nvir, ntail), host).expect("shape"))
}

// ---------------------------------------------------------------------------
// Unrestricted RI-MP2 energy.
//
// The same wide GEMM `G_i = B_iᵀ·B_right[:, col0..col0+ncols]` as above, with
// `B` resident per spin; the pair arithmetic is the shared
// `u_rimp2::{same,opposite}_spin_block_energy`, so the device and CPU energies
// differ ONLY in how `g_i` was formed. The per-i energies are summed serially in
// ascending i, exactly as the CPU `partials.into_iter().sum()`.
//
// Precision. f64 unless the caller passes `mixed_ok` (only `u_ri_mp2` with the
// Coulomb operator and no kappa does) AND the settings allow `rimp2-energy`:
// then each spin's `B_ov` is resident as f32 and every block (αα, ββ, αβ) is
// formed by the same k-panelled SGEMM with f64 accumulation as the closed-shell
// kernel (`gemm_f32_f64acc_dev`); the pair arithmetic stays f64. A mixed-path
// refusal (flush kernel unavailable, `B_ov` beyond the f32 arithmetic range)
// runs the f64 device path, counted in `mixed_fallback_f64`.
//
// Zero-size shapes (nocc or nvir of either spin 0, naux 0) are a typed
// `Layout` refusal from the `u_*_on_device` functions, which the dispatchers
// turn into the CPU path (counted as a layout fallback): there is no work, and
// the CPU path owns the value of the empty sum.
//
// Counters: every block's download bytes (and, mixed, its panel count) are
// collected and applied only after the last block succeeded, so a mid-loop
// failure leaves the counters untouched and the fallback reruns from scratch.

/// One spin's `B_ov` resident on the device (f64, or f32 for the mixed
/// kernel), with its host twin (the block views the operand descriptors are
/// derived from).
pub struct ResidentBov<'a> {
    m: Resident,
    host: &'a Array2<f64>,
}

impl ResidentBov<'_> {
    /// Rows of `B` (the auxiliary dimension).
    pub fn naux(&self) -> usize {
        match &self.m {
            Resident::F64(m) => m.rows(),
            Resident::F32(m) => m.rows(),
        }
    }
    /// Columns of `B` (`nocc·nvir`).
    pub fn cols(&self) -> usize {
        match &self.m {
            Resident::F64(m) => m.cols(),
            Resident::F32(m) => m.cols(),
        }
    }
}

/// Uploads `b_ov` once as f64, charged to `pool` under `label`.
pub fn upload_b_ov<'a>(
    dev: &Device,
    pool: &DevicePool,
    label: &str,
    b_ov: &'a Array2<f64>,
) -> Result<ResidentBov<'a>, GpuError> {
    upload_b_ov_prec(dev, pool, label, b_ov, Precision::F64)
}

/// Uploads `b_ov` once in `precision` (`Mixed`: rounded to f32, half the
/// bytes), charged to `pool` under `label`.
pub fn upload_b_ov_prec<'a>(
    dev: &Device,
    pool: &DevicePool,
    label: &str,
    b_ov: &'a Array2<f64>,
    precision: Precision,
) -> Result<ResidentBov<'a>, GpuError> {
    let m = match precision {
        Precision::F64 => {
            Resident::F64(DeviceMatrix::<f64>::upload(dev, pool, label, &b_ov.view())?)
        }
        Precision::Mixed => Resident::F32(DeviceMatrix::<f32>::upload_rounded(
            dev,
            pool,
            label,
            &b_ov.view(),
        )?),
    };
    Ok(ResidentBov { m, host: b_ov })
}

/// The `G_i` scratch of one U-RI-MP2 energy: the f64 result, plus the f32
/// panel for the mixed kernel. Each is charged to the pool under its own label.
pub struct UScratch {
    c64: DeviceMatrix<f64>,
    c32: Option<DeviceMatrix<f32>>,
}

impl UScratch {
    /// `rows × cols` elements of each buffer the `precision` needs.
    pub fn new(
        dev: &Device,
        pool: &DevicePool,
        rows: usize,
        cols: usize,
        precision: Precision,
    ) -> Result<Self, GpuError> {
        let c64 = DeviceMatrix::<f64>::zeros(
            dev,
            pool,
            "U-RI-MP2 energy (device): G_i scratch",
            rows,
            cols,
        )?;
        let c32 = match precision {
            Precision::F64 => None,
            Precision::Mixed => Some(DeviceMatrix::<f32>::zeros(
                dev,
                pool,
                "U-RI-MP2 energy (device): G_i scratch f32 panel",
                rows,
                cols,
            )?),
        };
        Ok(Self { c64, c32 })
    }
}

/// Where one `G_i` block comes from: `B_left` rows block `i` against
/// `B_right[:, col0..col0+ncols]`.
#[derive(Clone, Copy)]
pub struct UBlock {
    pub i: usize,
    pub nvir_left: usize,
    pub col0: usize,
    pub ncols: usize,
}

/// What one finished block moved: the bytes downloaded and the mixed panels
/// flushed (0 for an f64 block).
#[derive(Clone, Copy, Debug)]
pub struct UBlockTally {
    pub d2h: usize,
    pub panels: usize,
}

/// Applies the tallies of a whole successful call to the counters.
fn commit_u_tally(tally: Vec<UBlockTally>) {
    for t in tally {
        if t.panels > 0 {
            note_mixed(t.panels, 0, t.d2h);
        } else {
            note_offloaded(0, t.d2h);
        }
    }
}

/// The test-seam injection for block `i`: a `Kernel`-class failure models a
/// MIXED-path failure (an f64 stage is not affected); otherwise a `Cuda`
/// failure in either precision.
fn u_injected_failure(i: usize, mixed: bool) -> Result<(), GpuError> {
    if fail_at_block() != i {
        return Ok(());
    }
    if !fail_as_kernel() {
        return Err(GpuError::Cuda(format!(
            "injected failure at block {i} (test)"
        )));
    }
    if mixed {
        return Err(GpuError::Kernel(format!(
            "injected failure at block {i} (test)"
        )));
    }
    Ok(())
}

/// The checked geometry of one block: `(left offset, right end, block length)`.
fn u_block_geometry(
    left: &ResidentBov<'_>,
    right: &ResidentBov<'_>,
    blk: UBlock,
    out_len: usize,
) -> Result<(usize, usize, usize), GpuError> {
    let UBlock {
        i,
        nvir_left,
        col0,
        ncols,
    } = blk;
    let layout = |what: String| GpuError::Layout(format!("U-RI-MP2 block {i}: {what}"));
    if left.naux() != right.naux() {
        return Err(layout(format!("naux {} vs {}", left.naux(), right.naux())));
    }
    let off = i
        .checked_mul(nvir_left)
        .filter(|&o| o <= left.cols() && nvir_left <= left.cols() - o)
        .ok_or_else(|| {
            layout(format!(
                "left block of width {nvir_left} outside {}",
                left.cols()
            ))
        })?;
    let end = col0
        .checked_add(ncols)
        .filter(|&e| e <= right.cols())
        .ok_or_else(|| {
            layout(format!(
                "right columns {col0}+{ncols} outside {}",
                right.cols()
            ))
        })?;
    let len = nvir_left
        .checked_mul(ncols)
        .ok_or_else(|| layout("block size overflows usize".into()))?;
    if out_len != len {
        return Err(layout(format!(
            "host block holds {out_len} elements, needs {len}"
        )));
    }
    Ok((off, end, len))
}

/// `out = B_left[:, i·nvir_left..(i+1)·nvir_left]ᵀ · B_right[:, col0..col0+ncols]`
/// (row-major `nvir_left × ncols`), formed in `scratch` and downloaded into
/// `out`: an f64 GEMM when both tensors are f64, the k-panelled SGEMM with f64
/// accumulation when both are f32 (mixing the two is a `Layout` refusal).
/// Returns what the block moved; the caller counts it (`commit_u_tally`) only
/// when its whole call succeeds.
pub fn u_g_block_on_device(
    dev: &Device,
    left: &ResidentBov<'_>,
    right: &ResidentBov<'_>,
    blk: UBlock,
    scratch: &mut UScratch,
    out: &mut [f64],
) -> Result<UBlockTally, GpuError> {
    let i = blk.i;
    u_injected_failure(i, matches!(left.m, Resident::F32(_)))?;
    let (off, end, len) = u_block_geometry(left, right, blk, out.len())?;
    let (m, k, n) = (blk.nvir_left, left.naux(), blk.ncols);
    let b_i = left.host.slice(s![.., off..off + m]);
    let b_r = right.host.slice(s![.., blk.col0..end]);
    let panels = match (&left.m, &right.m, &mut scratch.c32) {
        (Resident::F64(l), Resident::F64(r), _) => {
            let lop = dev_left_padded(l.buf().slice(off..), &b_i.t())?;
            let rop = dev_right_padded(r.buf().slice(blk.col0..), &b_r)?;
            gemm_f64_dev(
                dev,
                m,
                k,
                n,
                &lop,
                &rop,
                scratch.c64.buf_mut(),
                GEMM_K_BLOCK,
            )?;
            0
        }
        (Resident::F32(l), Resident::F32(r), Some(c32)) => {
            let lop = dev_left_padded(l.buf().slice(off..), &b_i.t())?;
            let rop = dev_right_padded(r.buf().slice(blk.col0..), &b_r)?;
            gemm_f32_f64acc_dev(
                dev,
                m,
                k,
                n,
                &lop,
                &rop,
                c32.buf_mut(),
                scratch.c64.buf_mut(),
                effective_k_panel(),
            )?
        }
        _ => {
            return Err(GpuError::Layout(format!(
                "U-RI-MP2 block {i}: operand precisions differ or the f32 scratch is missing"
            )))
        }
    };
    let cuda = |what: &str, e: &dyn std::fmt::Debug| {
        GpuError::Cuda(format!("{what} U-RI-MP2 G_{i}: {e:?}"))
    };
    dev.stream
        .memcpy_dtoh(&scratch.c64.buf().slice(..len), out)
        .map_err(|e| cuda("D2H", &e))?;
    dev.stream.synchronize().map_err(|e| cuda("sync", &e))?;
    Ok(UBlockTally {
        d2h: 8 * len,
        panels,
    })
}

/// Shared dispatcher prologue: the device, the pool and the precision to try
/// first, or `None` (CPU path; a reason is counted unless the mode is off or
/// there is no device at all).
fn u_dispatch_context(mixed_ok: bool) -> Option<(std::sync::Arc<Device>, DevicePool, Precision)> {
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
    let precision = if mixed_ok && settings.mixed_allows(MixedKernel::RiMp2Energy) {
        Precision::Mixed
    } else {
        Precision::F64
    };
    Some((dev, pool, precision))
}

/// `nocc·nvir` of a channel after checking its shapes; `Err` for an
/// inconsistent or empty one.
fn u_channel_nov(ch: &SpinChannel<'_>) -> Result<usize, GpuError> {
    let nov = ch.nocc.checked_mul(ch.nvir).ok_or_else(|| {
        GpuError::Layout(format!(
            "U-RI-MP2 dimensions nocc={} nvir={} overflow usize",
            ch.nocc, ch.nvir
        ))
    })?;
    if ch.b.ncols() != nov {
        return Err(GpuError::Layout(format!(
            "b_ov is {}x{} but nocc*nvir = {nov}",
            ch.b.nrows(),
            ch.b.ncols()
        )));
    }
    let eps_needed = ch
        .first_occ
        .saturating_add(ch.nocc)
        .max(ch.nocc_total.saturating_add(ch.nvir));
    if ch.eps.len() < eps_needed {
        return Err(GpuError::Layout(format!(
            "eps holds {} orbital energies but the index ranges reach {eps_needed}",
            ch.eps.len()
        )));
    }
    if nov == 0 || ch.b.nrows() == 0 {
        return Err(GpuError::Layout("empty U-RI-MP2 spin channel".into()));
    }
    Ok(nov)
}

/// Runs `run` at `precision`; a mixed-path refusal (`Kernel`, `F32Range`)
/// reruns it in f64 on the device (counted, traced), any other failure is the
/// CPU path (counted).
fn u_run_with_fallback(
    precision: Precision,
    run: impl Fn(Precision) -> Result<f64, GpuError>,
) -> Option<f64> {
    let r = match run(precision) {
        Err(e @ (GpuError::Kernel(_) | GpuError::F32Range(_))) if precision == Precision::Mixed => {
            if ferric_core::gpu::config::gpu_trace() {
                eprintln!("[gpu] U-RI-MP2 mixed path unavailable ({e}); running f64 on the device");
            }
            note_mixed_fallback();
            run(Precision::F64)
        }
        r => r,
    };
    match r {
        Ok(e) => Some(e),
        Err(e) => {
            count_cpu_fallback(&e);
            None
        }
    }
}

/// Same-spin (αα or ββ) energy on the device, or `None` for the CPU path.
/// `mixed_ok`: this caller may use the `rimp2-energy` mixed kernel when the
/// settings allow it (Coulomb U-RI-MP2 only).
pub fn try_u_same_spin_on_device(ch: SpinChannel<'_>, mixed_ok: bool) -> Option<f64> {
    let (dev, pool, precision) = u_dispatch_context(mixed_ok)?;
    u_run_with_fallback(precision, |p| {
        u_same_spin_on_device_prec(&dev, &pool, ch, p)
    })
}

/// Opposite-spin (αβ) energy on the device, or `None` for the CPU path;
/// `mixed_ok` as for [`try_u_same_spin_on_device`].
pub fn try_u_opposite_spin_on_device(
    ch_a: SpinChannel<'_>,
    ch_b: SpinChannel<'_>,
    mixed_ok: bool,
) -> Option<f64> {
    let (dev, pool, precision) = u_dispatch_context(mixed_ok)?;
    u_run_with_fallback(precision, |p| {
        u_opposite_spin_on_device_prec(&dev, &pool, ch_a, ch_b, p)
    })
}

/// The mixed-path refusals checked before ANY reservation or transfer, so the
/// dispatcher's f64 fallback wastes nothing: the flush kernel loads and every
/// tensor is inside the f32 arithmetic range.
fn u_mixed_preflight(dev: &Device, tensors: &[&Array2<f64>]) -> Result<(), GpuError> {
    dev.axpy_f32_to_f64()?;
    for b in tensors {
        check_f32_arithmetic_range(b)?;
    }
    Ok(())
}

/// The device same-spin energy against an explicit pool, f64.
pub fn u_same_spin_on_device(
    dev: &Device,
    pool: &DevicePool,
    ch: SpinChannel<'_>,
) -> Result<f64, GpuError> {
    u_same_spin_on_device_prec(dev, pool, ch, Precision::F64)
}

/// The device same-spin energy against an explicit pool in `precision` (the
/// dispatcher's body). One scratch and one resident `B` ("U-RI-MP2 B_ov
/// same-spin").
pub fn u_same_spin_on_device_prec(
    dev: &Device,
    pool: &DevicePool,
    ch: SpinChannel<'_>,
    precision: Precision,
) -> Result<f64, GpuError> {
    let nov = u_channel_nov(&ch)?;
    if precision == Precision::Mixed {
        u_mixed_preflight(dev, &[ch.b])?;
    }
    // scratch first (refuses before the larger upload); widest block is i = 0
    let mut scratch = UScratch::new(dev, pool, ch.nvir, nov, precision)?;
    let bres = upload_b_ov_prec(dev, pool, "U-RI-MP2 B_ov same-spin", ch.b, precision)?;
    let mut host = vec![0.0f64; ch.nvir * nov];
    let mut partials = Vec::with_capacity(ch.nocc);
    let mut tally = Vec::with_capacity(ch.nocc);
    for i in 0..ch.nocc {
        let ncols = (ch.nocc - i) * ch.nvir;
        let block = &mut host[..ch.nvir * ncols];
        let blk = UBlock {
            i,
            nvir_left: ch.nvir,
            col0: i * ch.nvir,
            ncols,
        };
        tally.push(u_g_block_on_device(
            dev,
            &bres,
            &bres,
            blk,
            &mut scratch,
            block,
        )?);
        let g_i = ArrayView2::from_shape((ch.nvir, ncols), &*block).expect("shape");
        partials.push(same_spin_block_energy(&g_i, i, ch));
    }
    commit_u_tally(tally);
    Ok(partials.into_iter().sum())
}

/// The device opposite-spin energy against an explicit pool, f64.
pub fn u_opposite_spin_on_device(
    dev: &Device,
    pool: &DevicePool,
    ch_a: SpinChannel<'_>,
    ch_b: SpinChannel<'_>,
) -> Result<f64, GpuError> {
    u_opposite_spin_on_device_prec(dev, pool, ch_a, ch_b, Precision::F64)
}

/// The device opposite-spin energy against an explicit pool in `precision`.
/// Uploads `B_α` and `B_β` once each under "U-RI-MP2 B_ov alpha" / "U-RI-MP2
/// B_ov beta" (f32 under `Precision::Mixed`).
pub fn u_opposite_spin_on_device_prec(
    dev: &Device,
    pool: &DevicePool,
    ch_a: SpinChannel<'_>,
    ch_b: SpinChannel<'_>,
    precision: Precision,
) -> Result<f64, GpuError> {
    u_channel_nov(&ch_a)?;
    let nov_b = u_channel_nov(&ch_b)?;
    if ch_a.b.nrows() != ch_b.b.nrows() {
        return Err(GpuError::Layout(format!(
            "naux differs between spins: {} vs {}",
            ch_a.b.nrows(),
            ch_b.b.nrows()
        )));
    }
    if precision == Precision::Mixed {
        u_mixed_preflight(dev, &[ch_a.b, ch_b.b])?;
    }
    let mut scratch = UScratch::new(dev, pool, ch_a.nvir, nov_b, precision)?;
    // Both tensors must fit TOGETHER before either is transferred: a pool that
    // fits B_alpha but not B_beta refuses here, naming the tensor that does not
    // fit, with nothing uploaded and nothing left reserved (the probe leases
    // drop at the end of this block, before the real charges are taken).
    {
        let elem = match precision {
            Precision::F64 => 8,
            Precision::Mixed => 4,
        };
        let bytes = |c: &SpinChannel<'_>| c.b.len().saturating_mul(elem);
        let _alpha = pool.reserve("U-RI-MP2 B_ov alpha", bytes(&ch_a))?;
        let _beta = pool.reserve("U-RI-MP2 B_ov beta", bytes(&ch_b))?;
    }
    let ba = upload_b_ov_prec(dev, pool, "U-RI-MP2 B_ov alpha", ch_a.b, precision)?;
    let bb = upload_b_ov_prec(dev, pool, "U-RI-MP2 B_ov beta", ch_b.b, precision)?;
    let mut host = vec![0.0f64; ch_a.nvir * nov_b];
    let mut partials = Vec::with_capacity(ch_a.nocc);
    let mut tally = Vec::with_capacity(ch_a.nocc);
    for i in 0..ch_a.nocc {
        let blk = UBlock {
            i,
            nvir_left: ch_a.nvir,
            col0: 0,
            ncols: nov_b,
        };
        tally.push(u_g_block_on_device(
            dev,
            &ba,
            &bb,
            blk,
            &mut scratch,
            &mut host,
        )?);
        let g_i = ArrayView2::from_shape((ch_a.nvir, nov_b), &host[..]).expect("shape");
        partials.push(opposite_spin_block_energy(&g_i, i, ch_a, ch_b));
    }
    commit_u_tally(tally);
    Ok(partials.into_iter().sum())
}
