#![cfg(all(feature = "gpu", feature = "test-seams"))]
//! Mixed-precision RI-MP2 energy (f32-resident B_ov, k-panelled SGEMM with f64
//! accumulation) against the CPU f64 energy of the same B_ov. Every system uses
//! an RI-JK SCF reference (same B_ov and orbital energies on both sides); panel
//! width b = `effective_k_panel()`, the shipped default.
//!
//! BUDGET ROW (kernel `rimp2-energy`; every number is printed by this test):
//!   kernel         G_i = B_iᵀ·B_tail, k = naux; resident B_ov f32
//!   depth          n = naux per element, in panels of b = effective_k_panel()
//!   kappa_sum      max_ab S_ab/|g_ab| is unbounded (g_ab crosses zero), so the
//!                  budget carries the energy-level kappa_E = Σ fac|g|S/|D| / |E_os|
//!                  and the p99 of the element kappa on block i = 0, per system
//!   bound          per element eps_device(naux, b)·S + eta with
//!                  eps_device = (1+u32)²(1+γ_b(u32))(1+γ_⌈naux/b⌉(u64)) − 1 + γ_naux(u64)
//!                  (u32 = 2⁻²⁴; the last term is the f64 reference's own
//!                  rounding), eta the f32 underflow term; energy bound per
//!                  common/rimp2_error_bound.rs
//!   mitigation     f64 panel accumulation (b fixed), f64 host pair arithmetic,
//!                  f64 everything else; a finite B_ov with max|B| beyond
//!                  sqrt(f32::MAX/b) (a panel sum could overflow) or beyond
//!                  f32::MAX is refused to the f64 device path
//!   precision      f32 storage, f32 panel products, f64 sums
//!
//! WHAT EACH GATE CATCHES. The energy bound is a worst case over rounding signs,
//! 3 to 6 decades above every measured energy error; it catches gross defects
//! (wrong operand offset, dropped panel or pair weight: 1e-1 relative) and
//! NOTHING subtle. The subtle defects are pinned as follows.
//!  * ROUNDING MODE (round-to-nearest-even vs truncation): the GUARANTEE is the
//!    bit-exact upload test (`the_f32_upload_is_round_to_nearest_even_bit_for_bit`),
//!    which compares the device f32 B_ov with `as f32` element for element. The
//!    statistical band below also moves under truncation (1.29 vs the band edge
//!    1.13), but that margin (13%) is of the size of the device/twin spread
//!    between systems (0.87 at water/cc-pVDZ, 1.007 at benzene/aug-cc-pVTZ), so
//!    it is supplementary and device and system dependent: a build whose in-panel
//!    order is more accurate could put truncation inside the band.
//!  * PANEL STRUCTURE: `the_shipped_kernel_is_pinned_by_a_two_sided_per_element_measurement`
//!    measures the per-element RMS of err/S over all G blocks of real
//!    benzene/aug-cc-pVTZ (naux 912), device against the host twin at the shipped
//!    b and against host emulations of the defects; the accepted device/twin band
//!    is the geometric midpoint between the clean side (ratio 1) and the nearest
//!    measured defect on each side. It catches plain SGEMM (one panel; also run
//!    with the real device via FERRIC_GPU_MIXED_K_PANEL=1000000000), an f32 B_ov
//!    with an f64 GEMM (below the band), and wider panels (below). It does NOT
//!    catch a panel width below the shipped one (b = 32 reads 0.70: a more
//!    accurate kernel, a performance regression only, not pinned).
//!  * F64 FLUSH ACROSS PANELS (defect: panels summed in f32, one f64 flush at the
//!    end): the band also moves for it on this data (ratio 1.30, outside the
//!    band) but that is again supplementary; the deterministic guarantee is
//!    `the_f64_flush_across_panels_keeps_what_an_f32_sum_loses` (panel sums
//!    +2^20, 2^-9, -2^20 through the real `g_block`: the f64 flush returns 2^-9
//!    exactly, the f32-across-panels emulation 0).
//!  The RMS band is not a statement about other devices or builds; the two
//!  bit-exact/exact tests above are.
//! Measured panel-width sensitivity (host twin at width b' over the shipped
//! b = 64 twin, benzene/aug-cc-pVTZ, all blocks): b' = 32: 0.704 (inside the
//! band, not caught), 96: 1.270, 128: 1.506, 256: 2.389 (all above the band edge
//! 1.135, caught). Other measured ratios to the twin: device 1.007, plain SGEMM
//! 2.793, f32 B + f64 GEMM 0.240, truncating upload 1.287, f32-across-panels
//! 1.300.
//!
//! Decomposition (per system): device − CPU = storage + accumulation, where
//! storage = CPU f64 energy of the f32-rounded B_ov − CPU (factor eps_storage)
//! and accumulation = device − that (factor eps_accumulation); each is asserted
//! against its own bound. The device sees the SAME rounded B_ov as the CPU
//! storage run, so "the mixed energy is the f64 energy of a perturbed B_ov, up
//! to the accumulation bound" is the backward-error reading of the split.
//!
//! Counters are honest: a run that ends as the f64 device path or the CPU path
//! (mid-loop failure, range refusal) leaves `gemm_mixed`, `mixed_panels` and
//! `gemm_offloaded` unmoved.
//!
//! MEASURED at the shipped default b = 64 (the printed `error map` lines; RI-JK
//! SCF references; signed OS and SS partly cancel, so all three are listed):
//!   system (frozen core)     nocc nvir naux  E_corr     dE_OS     dE_SS     total     kcal/mol  bound OS/SS      kappa_E p99 kappa(i=0)
//!   h2o/cc-pvdz                 5   19    84  -0.204009  -4.9e-9   -2.1e-9   -7.0e-9   -4.4e-6   1.6e-6 / 2.8e-6   1.30   4.8e11
//!   c4h10/cc-pvdz              17   89   364  -0.599582  -2.8e-10  -3.7e-10  -6.5e-10  -4.1e-7   7.2e-6 / 1.4e-5   1.95   4.6e11
//!   c8h18/cc-pvdz              33  169   700  -1.183770  +1.1e-10  -8.3e-11  +2.6e-11  +1.6e-8   1.7e-5 / 3.3e-5   2.40   2.0e11
//!   alkane_12/cc-pvdz          49  249  1036  -1.767936  -1.7e-10  -9.1e-11  -2.6e-10  -1.6e-7   2.9e-5 / 5.6e-5   2.76   1.2e11
//!   h2o/aug-cc-pvtz             5   87   198  -0.283553  -6.1e-10  +3.7e-10  -2.4e-10  -1.5e-7   2.5e-6 / 4.6e-6   1.50   1.2e11
//!   benzene/aug-cc-pvdz (6)    15  171   570  -0.810690  +5.1e-10  +4.7e-10  +9.7e-10  +6.1e-7   7.8e-6 / 1.4e-5   1.65   2.1e12
//!   benzene/aug-cc-pvtz (6)    15  393   912  -0.963520  -8.7e-10  -1.0e-10  -9.7e-10  -6.1e-7   1.0e-5 / 2.0e-5   1.84   3.0e12
//! Decomposition at benzene/aug-cc-pvtz: storage dE_OS +7.5e-11, dE_SS +1.0e-10
//! (bound 3.2e-7 / 5.9e-7); accumulation dE_OS -9.4e-10, dE_SS -2.0e-10
//! (bound 1.0e-5 / 1.9e-5).
//! Two-sided per-element measurement (benzene/aug-cc-pvtz, naux 912, b 64,
//! 18345720 elements; RMS of err/S): device 1.796e-8; host twin 1.783e-8
//! (device/twin 1.007); plain SGEMM (one panel) 4.980e-8 (2.793x the twin);
//! f32 B + f64 GEMM 4.277e-9 (0.240x); truncating upload 2.296e-8 (1.287x).
//! Band for device/twin: (sqrt(0.240), sqrt(1.287)) = (0.49, 1.13); plain
//! SGEMM / device = 2.77 against sqrt(naux/b) = 3.78. Worst element / its bound:
//! 7.1e-2.
//! Per-element RMS against naux (cc-pVDZ, four systems; RMS of err/S at naux 84,
//! 364, 700, 1036: 4.9e-8, 2.5e-8, 2.1e-8, 1.8e-8): slope of ln RMS vs ln naux
//! -0.400 +- 0.027 (df 2), one-sided 95% upper limit -0.323; random-sign model
//! -0.5. The test fails only if that limit is >= 0. It is a DESCRIPTION of the
//! shipped kernel, not a law and not a gate on defects: naux is confounded with
//! the molecule (four systems; without water the slope is -0.314), and the
//! plain-SGEMM and truncating-upload emulations on the same blocks also fall
//! with naux (slopes -0.240 and -0.469). The only absolute guard is the
//! per-element bound (every element <= eps_device*S + eta). Nothing is asserted
//! about the total energy error vs N: its signed OS/SS parts scatter by two
//! decades through cancellation.
//! Teeth against the f64 device gate (gpu_rimp2_resident.rs): the f32-storage
//! defect exceeds that gate's bound on every system above (by 2.3x at
//! benzene/aug-cc-pvtz up to ~1e5x on water/cc-pvdz).
//!
//! No timings: only correctness gates run here.
use ferric_core::gpu::device::GpuError;
use ferric_core::gpu::device::{device, FORCE_KERNEL_FAILURE};
use ferric_core::gpu::mixed::{effective_k_panel, MIXED_K_PANEL_DEFAULT};
use ferric_core::gpu::mixed_host::{gamma, gemm_f32_f64acc_host, round_trip_f32, U32, U64};
use ferric_core::gpu::pool::DevicePool;
use ferric_core::gpu::resident::DeviceMatrix;
use ferric_core::gpu::{
    install, pool, probe, stats, GpuMode, GpuSettingsExplicit, GpuStatus, MixedKernel,
    MixedKernelSet, Precision,
};
use ferric_mp2::rimp2::{spin_components_from_b_ov_kappa, spin_components_from_b_ov_kappa_cpu};
use ferric_mp2::rimp2_gpu::{
    g_block_on_device, spin_components_on_device, FAIL_AS_KERNEL, FAIL_AT_BLOCK,
};
use ndarray::{linalg::general_mat_mul, s, Array2, ArrayView2};
use std::sync::atomic::Ordering;

#[path = "common/rimp2_error_bound.rs"]
mod bound;
#[path = "common/rimp2_eps.rs"]
mod eps;
#[path = "common/rimp2_gpu_fixture.rs"]
mod fixture;
#[path = "common/rimp2_stats.rs"]
mod stat;
use fixture::Prepared;

static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// `true` when a device is present; installs the mixed settings once, before
/// anything (the SCF in the fixture included) reads them.
fn ready() -> bool {
    if !matches!(probe(0), GpuStatus::Ready(_)) {
        eprintln!("skipping: no CUDA device");
        assert!(
            std::env::var("FERRIC_GPU_TESTS_REQUIRED").ok().as_deref() != Some("1"),
            "FERRIC_GPU_TESTS_REQUIRED=1 but no device"
        );
        return false;
    }
    static INSTALL: std::sync::Once = std::sync::Once::new();
    INSTALL.call_once(|| {
        install(GpuSettingsExplicit {
            mode: Some(GpuMode::Auto),
            memory_gb: Some(2.0),
            precision: Some(Precision::Mixed),
            mixed_kernels: Some(MixedKernelSet::EMPTY.with(MixedKernel::RiMp2Energy)),
            ..Default::default()
        })
        .expect("install");
    });
    true
}

fn run_on_device(
    p: &Prepared,
    kappa: Option<f64>,
    precision: Precision,
) -> Result<ferric_mp2::rimp2::SpinComponents, GpuError> {
    let dev = device(0).unwrap();
    let pool = pool().unwrap();
    spin_components_on_device(
        &dev,
        &pool,
        &p.b_ov,
        &p.eps,
        p.nocc,
        p.nvir,
        p.first_occ,
        p.nocc_total,
        kappa,
        precision,
        true,
    )
}

fn on_device(
    p: &Prepared,
    precision: Precision,
) -> Result<ferric_mp2::rimp2::SpinComponents, GpuError> {
    run_on_device(p, None, precision)
}

fn cpu_of(p: &Prepared, b: &Array2<f64>, kappa: Option<f64>) -> ferric_mp2::rimp2::SpinComponents {
    spin_components_from_b_ov_kappa_cpu(b, &p.eps, p.nocc, p.nvir, p.first_occ, p.nocc_total, kappa)
}

fn host_mixed(l: &ArrayView2<f64>, r: &ArrayView2<f64>, b: usize) -> Array2<f64> {
    let mut out = Array2::zeros((l.nrows(), r.ncols()));
    gemm_f32_f64acc_host(l, r, &mut out.view_mut(), b);
    out
}

/// Mutant 4, host emulation: the panels are summed in f32 and flushed to f64
/// ONCE at the end (f32 accumulation across panels).
fn host_f32_across_panels(l: &ArrayView2<f64>, r: &ArrayView2<f64>, b: usize) -> Array2<f64> {
    let (a32, b32) = (l.mapv(|x| x as f32), r.mapv(|x| x as f32));
    let k = l.ncols();
    let mut acc = Array2::<f32>::zeros((l.nrows(), r.ncols()));
    let mut k0 = 0;
    while k0 < k {
        let k1 = (k0 + b).min(k);
        let mut c = Array2::<f32>::zeros(acc.dim());
        general_mat_mul(
            1.0f32,
            &a32.slice(s![.., k0..k1]),
            &b32.slice(s![k0..k1, ..]),
            0.0f32,
            &mut c,
        );
        acc += &c;
        k0 = k1;
    }
    acc.mapv(f64::from)
}

/// Blocks sampled for the per-element statistics: at most 8, evenly spread.
fn sample_blocks(nocc: usize) -> Vec<usize> {
    (0..nocc).step_by(nocc.div_ceil(8).max(1)).collect()
}

/// Per-element statistics of the DEVICE G blocks (mixed) over `blocks`.
fn device_elem(p: &Prepared, blocks: &[usize], eg: f64, eta: f64) -> stat::Elem {
    let dev = device(0).unwrap();
    let pool = pool().unwrap();
    let babs = p.b_ov.mapv(f64::abs);
    let mut el = stat::Elem::new(eg, eta);
    for &i in blocks {
        let g_dev = g_block_on_device(&dev, &pool, &p.b_ov, i, p.nvir, Precision::Mixed).unwrap();
        let (g, sab) = stat::block_g_s(&p.b_ov, &babs, i, p.nvir);
        el.add(&g_dev.view(), &g.view(), &sab.view());
    }
    el
}

#[test]
fn the_f32_upload_is_round_to_nearest_even_bit_for_bit() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let dev = device(0).unwrap();
    let pool = DevicePool::with_capacity_bytes(1 << 24);
    // ties (1 + 2^-24, 1 + 3·2^-24), values 0.9 ulp above a float, negatives,
    // and a pseudo-random spread of magnitudes
    let mut v = vec![
        1.0 + U32,
        1.0 + 3.0 * U32,
        -(1.0 + 3.0 * U32),
        1.0 + 0.9 * 2.0 * U32,
        3.0e-39,
        -7.7e-39,
    ];
    let mut st = 12345u64;
    for _ in 0..4096 {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let m = (st >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
        v.push(m * 10f64.powi(((st >> 3) % 12) as i32 - 6));
    }
    let n = v.len();
    let host = Array2::from_shape_vec((n, 1), v).unwrap();
    let up = DeviceMatrix::<f32>::upload_rounded(&dev, &pool, "RN test", &host.view()).unwrap();
    let mut back = vec![0.0f32; n];
    dev.stream.memcpy_dtoh(up.buf(), &mut back).unwrap();
    dev.stream.synchronize().unwrap();
    let trunc = stat::truncate_f32(&host);
    let mut differs_from_truncation = 0;
    for (k, b) in back.iter().enumerate() {
        assert_eq!(b.to_bits(), (host[(k, 0)] as f32).to_bits(), "element {k}");
        if f64::from(*b) != trunc[(k, 0)] {
            differs_from_truncation += 1;
        }
    }
    assert!(
        differs_from_truncation > n / 4,
        "the data cannot tell RN from truncation"
    );
}

#[test]
fn the_shipped_kernel_is_pinned_by_a_two_sided_per_element_measurement() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let (p, _) = fixture::prepare_scf("benzene", "aug-cc-pvtz", "aug-cc-pvtz-rifit", 6);
    // the SHIPPED constant, not `effective_k_panel()`: this test pins the shipped
    // kernel, so a FERRIC_GPU_MIXED_K_PANEL override (or a regression of the
    // device's panel width) must show up as a device/twin mismatch
    let (naux, b) = (p.b_ov.nrows(), MIXED_K_PANEL_DEFAULT);
    let (eg, eta) = (
        eps::eps_device(naux, b),
        eps::eta_abs(naux, stat::max_abs(&p.b_ov)),
    );
    let babs = p.b_ov.mapv(f64::abs);
    let rn = round_trip_f32(&p.b_ov.view());
    let tr = stat::truncate_f32(&p.b_ov);
    let dev = device(0).unwrap();
    let pool = pool().unwrap();
    let mk = || stat::Elem::new(eg, eta);
    let (mut e_dev, mut e_twin, mut e_one, mut e_sto, mut e_trun) = (mk(), mk(), mk(), mk(), mk());
    let mut e_xp = mk(); // mutant 4: f32 accumulation across panels
    let sweep_b = [32usize, 96, 128, 256];
    let mut e_b: Vec<stat::Elem> = sweep_b.iter().map(|_| mk()).collect();
    for i in 0..p.nocc {
        let lo = i * p.nvir;
        let (g, sab) = stat::block_g_s(&p.b_ov, &babs, i, p.nvir);
        let view = |m: &Array2<f64>| {
            (
                m.slice(s![.., lo..lo + p.nvir]).t().to_owned(),
                m.slice(s![.., lo..]).to_owned(),
            )
        };
        let g_dev = g_block_on_device(&dev, &pool, &p.b_ov, i, p.nvir, Precision::Mixed).unwrap();
        e_dev.add(&g_dev.view(), &g.view(), &sab.view());
        let (l, r) = view(&p.b_ov);
        e_twin.add(
            &host_mixed(&l.view(), &r.view(), b).view(),
            &g.view(),
            &sab.view(),
        );
        // mutant 1: plain SGEMM (no f64 flush): one panel of depth naux
        e_one.add(
            &host_mixed(&l.view(), &r.view(), naux).view(),
            &g.view(),
            &sab.view(),
        );
        // mutant 2: f32-resident B with an f64 GEMM (storage error only)
        let (lr, rr) = view(&rn);
        e_sto.add(&lr.dot(&rr).view(), &g.view(), &sab.view());
        // mutant 4: panels summed in f32, one f64 flush at the end
        e_xp.add(
            &host_f32_across_panels(&l.view(), &r.view(), b).view(),
            &g.view(),
            &sab.view(),
        );
        // the panel-width sweep: what a device running another b would show
        for (e, &bb) in e_b.iter_mut().zip(&sweep_b) {
            e.add(
                &host_mixed(&l.view(), &r.view(), bb).view(),
                &g.view(),
                &sab.view(),
            );
        }
        // mutant 3: truncating instead of round-to-nearest upload (same kernel)
        let (lt, rt) = view(&tr);
        e_trun.add(
            &host_mixed(&lt.view(), &rt.view(), b).view(),
            &g.view(),
            &sab.view(),
        );
    }
    let rms = |e: &stat::Elem| e.rms();
    let (r_dev, r_one, r_sto, r_trun) = (
        rms(&e_dev) / rms(&e_twin),
        rms(&e_one) / rms(&e_twin),
        rms(&e_sto) / rms(&e_twin),
        rms(&e_trun) / rms(&e_twin),
    );
    eprintln!(
        "two-sided benzene/aug-cc-pvtz fc6 (naux {naux}, b {b}, {} elements): per-element RMS(err/S): device {:.3e}, host twin {:.3e}, one panel (plain SGEMM) {:.3e}, f32 B + f64 GEMM {:.3e}, truncating upload {:.3e} | ratios to the twin: device {r_dev:.3}, plain SGEMM {r_one:.3}, f32 storage only {r_sto:.3}, truncation {r_trun:.3} | worst/bound device {:.3e}",
        e_dev.n, rms(&e_dev), rms(&e_twin), rms(&e_one), rms(&e_sto), rms(&e_trun), e_dev.worst
    );
    assert!(
        e_dev.worst <= 1.0,
        "a device element exceeds eps_device·S + eta ({:e})",
        e_dev.worst
    );
    // both sides are measured: the clean ratio (device vs twin) and each
    // mutant's; the gate is only meaningful if every mutant sits on its side
    assert!(
        r_one > 1.0 && r_trun > 1.0 && r_sto < 1.0,
        "a mutant is not separable from the clean kernel"
    );
    let lo = r_sto.sqrt(); // geometric midpoint between the clean side (1) and the f32-storage-only side
    let hi = r_one.min(r_trun).sqrt(); // ... and the nearest of the plain-SGEMM / truncation sides
    assert!(
        lo < r_dev && r_dev < hi,
        "device/twin RMS ratio {r_dev:.3} is outside the band ({lo:.3}, {hi:.3}) set by the measured mutants"
    );
    // Measured, not asserted against the band: the fourth mutant and the panel
    // widths (the host twin at width b' is what a device running b' would show).
    let r_xp = rms(&e_xp) / rms(&e_twin);
    let r_b: Vec<f64> = e_b.iter().map(|e| rms(e) / rms(&e_twin)).collect();
    eprintln!(
        "mutant 4 (f32 accumulation across panels, one f64 flush at the end): ratio to the twin {r_xp:.3}, inside the band ({lo:.3}, {hi:.3}): {}",
        lo < r_xp && r_xp < hi
    );
    for (bb, r) in sweep_b.iter().zip(&r_b) {
        eprintln!(
            "panel width {bb}: ratio to the b = {b} twin {r:.3}; caught by the band: {}",
            !(lo < *r && *r < hi)
        );
    }
    // monotone in b: a wider panel never reads as a more accurate kernel
    assert!(
        r_b.windows(2).all(|w| w[1] > w[0]),
        "RMS is not monotone in the panel width: {r_b:?}"
    );
}

/// The cancelling construction that the RMS band cannot see (mutant 4): three
/// panels whose f32 sums are +2^20, 2^-9 and -2^20 (all exact in f32). The f64
/// flush keeps 2^-9; summing the panels in f32 loses it (ulp32(2^20) = 2^-3) and
/// returns 0. G_0[0][1] of this B_ov is exactly that sum.
#[test]
fn the_f64_flush_across_panels_keeps_what_an_f32_sum_loses() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let b = effective_k_panel();
    let naux = 3 * b;
    let tiny = 2.0f64.powi(-9);
    // B_ov (naux x 2), nvir = 1: column 0 all ones, column 1 the pattern
    let mut x = vec![0.0f64; naux];
    x[0] = 1048576.0; // 2^20, panel 1
    x[b] = tiny; // panel 2
    x[2 * b] = -1048576.0; // panel 3
    let b_ov = Array2::from_shape_fn((naux, 2), |(r, c)| if c == 0 { 1.0 } else { x[r] });
    let dev = device(0).unwrap();
    let pool = pool().unwrap();
    let g = g_block_on_device(&dev, &pool, &b_ov, 0, 1, Precision::Mixed).unwrap();
    assert_eq!(g.dim(), (1, 2));
    assert_eq!(
        g[(0, 1)],
        tiny,
        "the f64 flush must keep the middle panel's 2^-9"
    );
    // the emulated mutant loses it, so this construction has teeth
    let l = b_ov.slice(s![.., 0..1]).t().to_owned();
    let r = b_ov.slice(s![.., 0..2]).to_owned();
    let mutant = host_f32_across_panels(&l.view(), &r.view(), b);
    assert_eq!(
        mutant[(0, 1)],
        0.0,
        "the f32-across-panels mutant returns 0 here"
    );
}

#[test]
fn per_element_bound_holds_for_every_g_block_on_small_systems() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    for sys in ["h2o", "benzene"] {
        let (p, _) = fixture::prepare_scf(sys, "cc-pvdz", "cc-pvdz-ri", 0);
        let (naux, b) = (p.b_ov.nrows(), effective_k_panel());
        let (eg, eta) = (
            eps::eps_device(naux, b),
            eps::eta_abs(naux, stat::max_abs(&p.b_ov)),
        );
        let blocks: Vec<usize> = (0..p.nocc).collect();
        let el = device_elem(&p, &blocks, eg, eta);
        eprintln!("{sys}/cc-pvdz: naux {naux}, b {b}: per-element RMS(err/S) {:.3e}, worst/bound {:.3e} over {} elements", el.rms(), el.worst, el.n);
        assert!(
            el.worst <= 1.0,
            "{sys}: per-element bound violated ({:e})",
            el.worst
        );
    }
}

/// One system through the whole energy gate; returns (naux, per-element RMS).
fn gate(label: &str, p: &Prepared, cpu: &ferric_mp2::rimp2::SpinComponents) {
    let (naux, b) = (p.b_ov.nrows(), effective_k_panel());
    let eta = eps::eta_abs(naux, stat::max_abs(&p.b_ov));
    let s0 = stats();
    let mixed = on_device(p, Precision::Mixed).unwrap();
    let s1 = stats();
    assert!(
        s1.gemm_mixed - s0.gemm_mixed == p.nocc as u64
            && s1.mixed_fallback_f64 == s0.mixed_fallback_f64,
        "{label}: the mixed path was not taken for every block"
    );
    let storage = cpu_of(p, &round_trip_f32(&p.b_ov.view()), None);
    let eb = |eg: f64| {
        bound::energy_bound(
            &p.b_ov,
            &p.eps,
            p.nocc,
            p.nvir,
            p.first_occ,
            p.nocc_total,
            bound::BoundSpec {
                eps_g: eg,
                eta,
                kappa: None,
            },
        )
    };
    let full = eb(eps::eps_device(naux, b));
    let acc = eb(eps::eps_accumulation(naux, b));
    let sto = eb(eps::eps_storage(naux));
    let (dm_os, dm_ss) = (mixed.e_os - cpu.e_os, mixed.e_ss - cpu.e_ss);
    let (ds_os, ds_ss) = (storage.e_os - cpu.e_os, storage.e_ss - cpu.e_ss);
    let (da_os, da_ss) = (mixed.e_os - storage.e_os, mixed.e_ss - storage.e_ss);
    let kcal = 627.5094740631;
    let tot = mixed.e_total - cpu.e_total;
    eprintln!(
        "error map mixed {label}: nocc {} nvir {} naux {naux} b {b} | E_corr {:.9} | dE_os {dm_os:+.3e} dE_ss {dm_ss:+.3e} total {tot:+.3e} Eh ({:+.3e} kcal/mol, rel {:.3e}) | storage dE_os {ds_os:+.3e} dE_ss {ds_ss:+.3e} | accumulation dE_os {da_os:+.3e} dE_ss {da_ss:+.3e} | bounds full {:.3e} {:.3e} storage {:.3e} {:.3e} acc {:.3e} {:.3e} | kappa_E {:.3e} kappa_p99(i=0) {:.3e}",
        p.nocc, p.nvir, cpu.e_total, tot * kcal, tot.abs() / cpu.e_total.abs(),
        full.os, full.ss, sto.os, sto.ss, acc.os, acc.ss, full.kappa_e_os, full.kappa_p99_block0
    );
    assert!(
        dm_os.abs() <= full.os && dm_ss.abs() <= full.ss,
        "{label}: mixed energy error exceeds the deterministic bound"
    );
    assert!(
        ds_os.abs() <= sto.os && ds_ss.abs() <= sto.ss,
        "{label}: storage part exceeds its bound"
    );
    assert!(
        da_os.abs() <= acc.os && da_ss.abs() <= acc.ss,
        "{label}: accumulation part exceeds its bound"
    );
    // positive control: the mixed path moved the energy off the f64 noise class
    let floor = 1e3 * U64 * cpu.e_total.abs();
    assert!(
        dm_os.abs() + dm_ss.abs() >= floor,
        "{label}: mixed moved the energy by less than the f64 noise floor; not engaged?"
    );
    // teeth against the f64 device gate: the f32-storage defect exceeds its bound
    let f64b = bound::energy_bound(
        &p.b_ov,
        &p.eps,
        p.nocc,
        p.nvir,
        p.first_occ,
        p.nocc_total,
        bound::BoundSpec {
            eps_g: 2.0 * gamma(naux, U64),
            eta: 0.0,
            kappa: None,
        },
    );
    assert!(
        ds_os.abs() > f64b.os && ds_ss.abs() > f64b.ss,
        "{label}: the f32-storage defect ({:.3e}, {:.3e}) is inside the f64 gate ({:.3e}, {:.3e}); that gate is blind here",
        ds_os.abs(), ds_ss.abs(), f64b.os, f64b.ss
    );
}

#[test]
fn mixed_energy_decomposes_and_the_per_element_error_falls_with_the_auxiliary_depth() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let (mut pts, mut pts_one, mut pts_trunc) = (Vec::new(), Vec::new(), Vec::new());
    for sys in ["h2o", "c4h10", "c8h18", "alkane_12"] {
        let (p, cpu) = fixture::prepare_scf(sys, "cc-pvdz", "cc-pvdz-ri", 0);
        gate(&format!("{sys}/cc-pvdz"), &p, &cpu);
        let (naux, b) = (p.b_ov.nrows(), effective_k_panel());
        let (eg, eta) = (
            eps::eps_device(naux, b),
            eps::eta_abs(naux, stat::max_abs(&p.b_ov)),
        );
        let el = device_elem(&p, &sample_blocks(p.nocc), eg, eta);
        eprintln!("per-element {sys}: naux {naux}, {} elements: device RMS(err/S) {:.3e}, worst/bound {:.3e}, eps_device {eg:.3e}", el.n, el.rms(), el.worst);
        assert!(el.worst <= 1.0);
        pts.push((naux as f64, el.rms()));
        // the same blocks through two host-emulated defects, to state what the
        // slope can and cannot tell: plain SGEMM (one panel) and a truncating upload
        let blocks = sample_blocks(p.nocc);
        let babs = p.b_ov.mapv(f64::abs);
        let tr = stat::truncate_f32(&p.b_ov);
        let (mut e1, mut et) = (stat::Elem::new(eg, eta), stat::Elem::new(eg, eta));
        for &i in &blocks {
            let lo = i * p.nvir;
            let (g, sab) = stat::block_g_s(&p.b_ov, &babs, i, p.nvir);
            let lr = |m: &Array2<f64>| {
                (
                    m.slice(s![.., lo..lo + p.nvir]).t().to_owned(),
                    m.slice(s![.., lo..]).to_owned(),
                )
            };
            let (l, r) = lr(&p.b_ov);
            e1.add(
                &host_mixed(&l.view(), &r.view(), naux).view(),
                &g.view(),
                &sab.view(),
            );
            let (lt, rt) = lr(&tr);
            et.add(
                &host_mixed(&lt.view(), &rt.view(), b).view(),
                &g.view(),
                &sab.view(),
            );
        }
        pts_one.push((naux as f64, e1.rms()));
        pts_trunc.push((naux as f64, et.rms()));
    }
    // The per-element RMS of err/S against the auxiliary depth. Model: the
    // rounding errors of the naux terms of one element add with random signs
    // against S = Σ|terms|, so RMS(err/S) falls as naux^(-1/2) at fixed b. The
    // assertion is the one-sided 95% upper confidence limit of the fitted slope
    // being NEGATIVE (df = n - 2 = 2): it fails only if the data admit a slope of
    // 0 or more. It DESCRIBES the shipped kernel; it is not a gate on defects
    // (see the module docs): the plain-SGEMM and truncating-upload emulations
    // measured here fall with naux too.
    let (slope, se) = stat::loglog_fit(&pts);
    let df = pts.len() - 2;
    let upper = slope + stat::t95(df) * se;
    eprintln!("per-element RMS vs naux (cc-pvdz, 4 systems; naux is confounded with the molecule): slope of ln RMS vs ln naux = {slope:.3} +- {se:.3} (df {df}), one-sided 95% upper limit {upper:.3}; random-sign model -0.5");
    let (s1, _) = stat::loglog_fit(&pts[1..]);
    let (so, _) = stat::loglog_fit(&pts_one);
    let (st, _) = stat::loglog_fit(&pts_trunc);
    eprintln!("without the smallest system (3 points, no standard error at df 1): slope {s1:.3}; plain-SGEMM emulation slope {so:.3}; truncating-upload emulation slope {st:.3}");
    assert!(
        upper < 0.0,
        "the per-element RMS does not demonstrably fall with naux: upper limit {upper:.3}"
    );
}

#[test]
fn the_basis_axis_is_gated_on_real_aug_cc_systems() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    for (sys, obs, aux, fc) in [
        ("h2o", "aug-cc-pvtz", "aug-cc-pvtz-rifit", 0usize),
        ("benzene", "aug-cc-pvdz", "aug-cc-pvdz-rifit", 6),
        ("benzene", "aug-cc-pvtz", "aug-cc-pvtz-rifit", 6),
    ] {
        let (p, cpu) = fixture::prepare_scf(sys, obs, aux, fc);
        eprintln!(
            "{sys}/{obs} fc{fc}: nocc {} nvir {} naux {}: B_ov f32 {:.1} MB, G_i scratch {:.1} MB",
            p.nocc,
            p.nvir,
            p.b_ov.nrows(),
            4.0 * p.b_ov.len() as f64 / 1e6,
            (12 * p.nvir * p.b_ov.ncols()) as f64 / 1e6
        );
        gate(&format!("{sys}/{obs}/fc{fc}"), &p, &cpu);
    }
}

#[test]
fn damped_mp2_runs_through_the_mixed_path_within_its_bound() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let (p, _) = fixture::prepare_scf("h2o", "cc-pvdz", "cc-pvdz-ri", 0);
    let kappa = Some(1.1);
    let (naux, b) = (p.b_ov.nrows(), effective_k_panel());
    let eta = eps::eta_abs(naux, stat::max_abs(&p.b_ov));
    let cpu = cpu_of(&p, &p.b_ov, kappa);
    let s0 = stats();
    let mixed = run_on_device(&p, kappa, Precision::Mixed).unwrap();
    assert_eq!(
        stats().gemm_mixed - s0.gemm_mixed,
        p.nocc as u64,
        "the kappa path took the mixed lane"
    );
    let bd = bound::energy_bound(
        &p.b_ov,
        &p.eps,
        p.nocc,
        p.nvir,
        p.first_occ,
        p.nocc_total,
        bound::BoundSpec {
            eps_g: eps::eps_device(naux, b),
            eta,
            kappa,
        },
    );
    let (d_os, d_ss) = (mixed.e_os - cpu.e_os, mixed.e_ss - cpu.e_ss);
    eprintln!("error map mixed damped (kappa 1.1) h2o/cc-pvdz: dE_os {d_os:+.3e} dE_ss {d_ss:+.3e} | bound {:.3e} {:.3e}", bd.os, bd.ss);
    assert!(d_os.abs() <= bd.os && d_ss.abs() <= bd.ss);
    // the damping must act (it is not the undamped energy) and the mixed run
    // must differ from the f64 damped one (not silently f64)
    assert!((cpu.e_os - cpu_of(&p, &p.b_ov, None).e_os).abs() > 1e-4);
    assert!(d_os.abs() + d_ss.abs() >= 1e3 * U64 * cpu.e_total.abs());
}

#[test]
fn f32_resident_b_ov_is_a_labelled_reservation_of_half_the_f64_bytes() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let (p, _) = fixture::prepare_scf("h2o", "cc-pvdz", "cc-pvdz-ri", 0);
    let nov = p.b_ov.ncols();
    let (b32, c32, c64) = (4 * p.b_ov.len(), 4 * p.nvir * nov, 8 * p.nvir * nov);
    let mixed_need = b32 + c32 + c64;
    let f64_need = 8 * p.b_ov.len() + c64;
    assert!(f64_need > mixed_need, "fixture does not separate the two");
    let dev = device(0).unwrap();
    let pool = DevicePool::with_capacity_bytes(mixed_need);
    let run = |prec| {
        spin_components_on_device(
            &dev,
            &pool,
            &p.b_ov,
            &p.eps,
            p.nocc,
            p.nvir,
            p.first_occ,
            p.nocc_total,
            None,
            prec,
            true,
        )
    };
    run(Precision::Mixed).unwrap();
    assert_eq!(
        pool.peak_bytes(),
        mixed_need,
        "mixed reserves B_ov f32 + f32 panel scratch + f64 scratch, nothing else"
    );
    assert_eq!(
        pool.available_bytes(),
        pool.capacity_bytes(),
        "every reservation released"
    );
    let e = run(Precision::F64).unwrap_err();
    let GpuError::PoolFull { label, .. } = &e else {
        panic!("{e:?}")
    };
    assert!(
        label.contains("B_ov"),
        "the refusal names the B_ov reservation: {label}"
    );
    // one byte short for mixed: refused at the f32 B_ov (the last reservation)
    let tight = DevicePool::with_capacity_bytes(mixed_need - 1);
    let e = spin_components_on_device(
        &dev,
        &tight,
        &p.b_ov,
        &p.eps,
        p.nocc,
        p.nvir,
        p.first_occ,
        p.nocc_total,
        None,
        Precision::Mixed,
        true,
    )
    .unwrap_err();
    let GpuError::PoolFull { label, detail } = &e else {
        panic!("{e:?}")
    };
    assert!(label.contains("B_ov") && label.contains("f32"), "{label}");
    assert!(
        detail.contains("scratch"),
        "the report lists the scratch reservations: {detail}"
    );
    assert_eq!(tight.available_bytes(), tight.capacity_bytes());
}

#[test]
fn dispatcher_uses_mixed_when_allowed_and_falls_back_to_f64_device_when_the_kernel_is_unavailable()
{
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let (p, _) = fixture::prepare_scf("h2o", "cc-pvdz", "cc-pvdz-ri", 0);
    let s0 = stats();
    let via = spin_components_from_b_ov_kappa(
        &p.b_ov,
        &p.eps,
        p.nocc,
        p.nvir,
        p.first_occ,
        p.nocc_total,
        None,
    );
    let direct = on_device(&p, Precision::Mixed).unwrap();
    assert_eq!(
        via.e_total.to_bits(),
        direct.e_total.to_bits(),
        "the dispatcher takes the mixed path when allowed"
    );
    let s1 = stats();
    assert_eq!(
        s1.gemm_mixed - s0.gemm_mixed,
        2 * p.nocc as u64,
        "one mixed GEMM per block, twice"
    );
    assert_eq!(s1.mixed_fallback_f64, s0.mixed_fallback_f64);
    // Review Focus 2, dispatcher half
    FORCE_KERNEL_FAILURE.store(true, Ordering::Relaxed);
    let fallback = spin_components_from_b_ov_kappa(
        &p.b_ov,
        &p.eps,
        p.nocc,
        p.nvir,
        p.first_occ,
        p.nocc_total,
        None,
    );
    FORCE_KERNEL_FAILURE.store(false, Ordering::Relaxed);
    let s2 = stats();
    let f64dev = on_device(&p, Precision::F64).unwrap();
    assert_eq!(
        s2.mixed_fallback_f64 - s1.mixed_fallback_f64,
        1,
        "the fallback is counted"
    );
    assert_eq!(
        s2.gemm_mixed, s1.gemm_mixed,
        "no mixed GEMM may be counted on the fallback"
    );
    assert_eq!(
        s2.gemm_offloaded - s1.gemm_offloaded,
        p.nocc as u64,
        "the fallback ran the f64 DEVICE GEMMs, not the CPU"
    );
    assert_eq!(
        (
            s2.gemm_cpu_pool_full,
            s2.gemm_cpu_layout,
            s2.gemm_cpu_cuda_error
        ),
        (
            s1.gemm_cpu_pool_full,
            s1.gemm_cpu_layout,
            s1.gemm_cpu_cuda_error
        ),
        "no CPU fallback fired"
    );
    assert_eq!(
        fallback.e_total.to_bits(),
        f64dev.e_total.to_bits(),
        "the fallback is the f64 device energy"
    );
    // a direct mixed call with the kernel unavailable is a typed refusal BEFORE
    // any reservation or upload
    FORCE_KERNEL_FAILURE.store(true, Ordering::Relaxed);
    let u0 = stats().resident_uploads;
    let r = on_device(&p, Precision::Mixed);
    FORCE_KERNEL_FAILURE.store(false, Ordering::Relaxed);
    assert!(matches!(r, Err(GpuError::Kernel(_))), "{r:?}");
    assert_eq!(stats().resident_uploads, u0, "nothing was uploaded");
}

#[test]
fn a_mid_loop_failure_leaves_the_counters_honest_on_both_fallback_paths() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let (p, _) = fixture::prepare_scf("h2o", "cc-pvdz", "cc-pvdz-ri", 0);
    let half = p.nocc / 2;
    let f64dev = on_device(&p, Precision::F64).unwrap();
    let cpu = cpu_of(&p, &p.b_ov, None);
    let dispatch = || {
        spin_components_from_b_ov_kappa(
            &p.b_ov,
            &p.eps,
            p.nocc,
            p.nvir,
            p.first_occ,
            p.nocc_total,
            None,
        )
    };
    // (a) the mixed path dies after `half` blocks with a Kernel-class error
    FAIL_AT_BLOCK.store(half, Ordering::Relaxed);
    FAIL_AS_KERNEL.store(true, Ordering::Relaxed);
    let s0 = stats();
    let e = on_device(&p, Precision::Mixed);
    assert!(matches!(e, Err(GpuError::Kernel(_))), "{e:?}");
    let s1 = stats();
    assert_eq!(
        (s1.gemm_mixed, s1.mixed_panels, s1.gemm_offloaded),
        (s0.gemm_mixed, s0.mixed_panels, s0.gemm_offloaded),
        "a failed direct call counts nothing"
    );
    // the dispatcher: mixed fails mid-loop, the f64 device path (unaffected by
    // a Kernel-class injection) completes: only the f64 blocks are counted
    let via = dispatch();
    let s2 = stats();
    assert_eq!(
        (
            s2.gemm_mixed - s1.gemm_mixed,
            s2.mixed_panels - s1.mixed_panels,
            s2.gemm_offloaded - s1.gemm_offloaded
        ),
        (0, 0, p.nocc as u64),
        "the run that ended as f64 device shows no mixed work and exactly nocc f64 blocks"
    );
    assert_eq!(
        s2.mixed_fallback_f64 - s1.mixed_fallback_f64,
        1,
        "the mixed->f64 step is counted once"
    );
    assert_eq!(
        s2.gemm_cpu_cuda_error, s1.gemm_cpu_cuda_error,
        "no CPU fallback"
    );
    assert_eq!(
        via.e_total.to_bits(),
        f64dev.e_total.to_bits(),
        "the energy is the f64 device energy"
    );
    // (b) a Cuda-class failure mid-loop: straight to the CPU, counters clean
    FAIL_AS_KERNEL.store(false, Ordering::Relaxed);
    let s3 = stats();
    let via = dispatch();
    let s4 = stats();
    assert_eq!(
        (
            s4.gemm_mixed - s3.gemm_mixed,
            s4.mixed_panels - s3.mixed_panels,
            s4.gemm_offloaded - s3.gemm_offloaded,
            s4.mixed_fallback_f64 - s3.mixed_fallback_f64
        ),
        (0, 0, 0, 0)
    );
    assert_eq!(s4.gemm_cpu_cuda_error - s3.gemm_cpu_cuda_error, 1);
    assert_eq!(via.e_total.to_bits(), cpu.e_total.to_bits());
    // (c) the hook off: the mixed run completes, counts every block, and the
    // f64 device energy is unchanged by all of the above
    FAIL_AT_BLOCK.store(usize::MAX, Ordering::Relaxed);
    let s5 = stats();
    on_device(&p, Precision::Mixed).unwrap();
    assert_eq!(stats().gemm_mixed - s5.gemm_mixed, p.nocc as u64);
    assert_eq!(
        on_device(&p, Precision::F64).unwrap().e_total.to_bits(),
        f64dev.e_total.to_bits()
    );
}

#[test]
fn an_operand_beyond_the_f32_range_falls_back_to_the_f64_device_path_with_a_counter() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let (mut p, _) = fixture::prepare_scf("h2o", "cc-pvdz", "cc-pvdz-ri", 0);
    let b = effective_k_panel().clamp(1, p.b_ov.nrows());
    let limit = (f64::from(f32::MAX) / b as f64).sqrt();
    // (1) beyond f32::MAX (storage overflow), (2) inside f32 but above
    // sqrt(f32::MAX/b) (a panel sum of products could overflow)
    for (what, v) in [
        ("beyond f32::MAX", 1e300),
        ("above sqrt(f32::MAX/b)", 2.0 * limit),
    ] {
        let keep = p.b_ov[(0, 0)];
        p.b_ov[(0, 0)] = v;
        let direct = on_device(&p, Precision::Mixed);
        assert!(
            matches!(direct, Err(GpuError::F32Range(_))),
            "{what}: {direct:?}"
        );
        let s0 = stats();
        let via = spin_components_from_b_ov_kappa(
            &p.b_ov,
            &p.eps,
            p.nocc,
            p.nvir,
            p.first_occ,
            p.nocc_total,
            None,
        );
        let s1 = stats();
        let f64dev = on_device(&p, Precision::F64).unwrap();
        assert_eq!(
            s1.mixed_fallback_f64 - s0.mixed_fallback_f64,
            1,
            "{what}: counted"
        );
        assert_eq!(s1.gemm_mixed, s0.gemm_mixed, "{what}: no mixed GEMM");
        assert_eq!(
            s1.gemm_offloaded - s0.gemm_offloaded,
            p.nocc as u64,
            "{what}: f64 device GEMMs ran"
        );
        assert_eq!(
            via.e_total.to_bits(),
            f64dev.e_total.to_bits(),
            "{what}: the fallback is the f64 device energy"
        );
        p.b_ov[(0, 0)] = keep;
    }
    // just inside the guard is accepted (the guard is not off by a decade)
    p.b_ov[(0, 0)] = 0.5 * limit;
    assert!(on_device(&p, Precision::Mixed).is_ok());
    // RULING: a NaN B_ov is the caller's data, not an f32-range violation: it
    // passes through to a non-finite energy, as the CPU path produces one
    p.b_ov[(0, 0)] = f64::NAN;
    let nan = on_device(&p, Precision::Mixed).unwrap();
    assert!(!nan.e_total.is_finite());
}

#[test]
fn degenerate_shapes_never_panic_in_the_mixed_arm() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let dev = device(0).unwrap();
    let pool = DevicePool::with_capacity_bytes(1 << 26);
    let m = Precision::Mixed;
    let run = |b: &Array2<f64>, eps: &[f64], nocc, nvir, first, total| {
        spin_components_on_device(&dev, &pool, b, eps, nocc, nvir, first, total, None, m, true)
    };
    let z = run(&Array2::zeros((4, 0)), &[0.0; 4], 0, 3, 0, 0).unwrap();
    assert_eq!((z.e_os, z.e_ss, z.e_total), (0.0, 0.0, 0.0));
    let z = run(&Array2::zeros((4, 0)), &[0.0; 4], 3, 0, 0, 3).unwrap();
    assert_eq!(z.e_total, 0.0);
    let e = run(
        &Array2::zeros((0, 6)),
        &[-1.0, -1.0, 1.0, 1.0, 1.0, 1.0],
        2,
        3,
        0,
        2,
    );
    assert!(matches!(e, Err(GpuError::Layout(_))), "{e:?}");
    let e = run(&Array2::zeros((4, 7)), &[0.0; 12], 2, 3, 0, 2);
    assert!(matches!(e, Err(GpuError::Layout(_))), "{e:?}");
    let e = run(&Array2::zeros((1, 1)), &[0.0; 4], usize::MAX, 2, 0, 1);
    assert!(matches!(e, Err(GpuError::Layout(_))), "{e:?}");
    let ragged = Array2::<f64>::zeros((4, 10));
    let e = g_block_on_device(&dev, &pool, &ragged, 2, 4, m);
    assert!(matches!(e, Err(GpuError::Layout(_))), "{e:?}");
    // naux = 1 (a single panel of depth 1) and nvir = 1, nocc = 1 run
    let tiny = Array2::from_elem((1, 1), 0.3);
    let r = run(&tiny, &[-1.0, 1.0], 1, 1, 0, 1).unwrap();
    let want = spin_components_from_b_ov_kappa_cpu(&tiny, &[-1.0, 1.0], 1, 1, 0, 1, None);
    assert!(
        (r.e_total - want.e_total).abs() <= 1e-6 * want.e_total.abs(),
        "{r:?} vs {want:?}"
    );
    assert_eq!(pool.available_bytes(), pool.capacity_bytes());
}

/// The mixed error map covers the Coulomb, erfc and terfc RI-MP2 energies. The
/// callers that are not those entries (`spin_components_from_b_ov`: SR-MP2 in
/// RS-MP2+RPA, OO-MP2) and `spin_components_from_b_ov_kappa_f64` run the device in f64 under
/// `precision = "mixed"`: no mixed GEMM is counted, the result is the f64 device
/// result bit for bit, and it differs from the mixed one.
#[test]
fn callers_other_than_coulomb_ri_mp2_run_the_device_in_f64_under_mixed() {
    use ferric_mp2::rimp2::{spin_components_from_b_ov, spin_components_from_b_ov_kappa_f64};
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let (p, _) = fixture::prepare_scf("h2o", "cc-pvdz", "cc-pvdz-ri", 0);
    let f64dev = on_device(&p, Precision::F64).unwrap();
    let mixed = on_device(&p, Precision::Mixed).unwrap();
    assert_ne!(
        f64dev.e_total.to_bits(),
        mixed.e_total.to_bits(),
        "fixture cannot tell the mixed path from the f64 one"
    );
    let s0 = stats();
    let plain =
        spin_components_from_b_ov(&p.b_ov, &p.eps, p.nocc, p.nvir, p.first_occ, p.nocc_total);
    let gated = spin_components_from_b_ov_kappa_f64(
        &p.b_ov,
        &p.eps,
        p.nocc,
        p.nvir,
        p.first_occ,
        p.nocc_total,
        None,
    );
    let s1 = stats();
    assert_eq!(s1.gemm_mixed, s0.gemm_mixed, "a mixed GEMM ran");
    assert_eq!(s1.gemm_offloaded - s0.gemm_offloaded, 2 * p.nocc as u64);
    assert_eq!(plain.e_total.to_bits(), f64dev.e_total.to_bits());
    assert_eq!(gated.e_total.to_bits(), f64dev.e_total.to_bits());
}
