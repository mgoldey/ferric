#![cfg(feature = "gpu")]
//! Mixed-precision RI-MP2 energy (f32-resident B_ov, k-panelled SGEMM with f64
//! accumulation) against the CPU f64 energy of the same B_ov.
//!
//! BUDGET ROW (kernel `rimp2-energy`; every number is printed by this test and
//! kept in the MEASURED block below):
//!   kernel         G_i = B_iᵀ·B_tail, k = naux; resident B_ov f32
//!   depth          n = naux per element, in panels of b = MIXED_K_PANEL_DEFAULT
//!                  (128, the committed default; PROVISIONAL until the panel
//!                  sweep, see gpu/mixed.rs)
//!   kappa_sum      max_ab S_ab/|g_ab| is unbounded (g_ab crosses zero), so the
//!                  budget carries the energy-level kappa_E = Σ fac|g|S/|D| / |E_os|
//!                  and the p99 of the element kappa on block i = 0, per system
//!   bound          eps_G = 2·u32 + γ_b(u32) + γ_⌈naux/b⌉(u64) per element
//!                  (u32 = 2⁻²⁴), energy bound per common/rimp2_error_bound.rs
//!   mitigation     f64 panel accumulation (b fixed), f64 host pair arithmetic,
//!                  f64 everything else; the f32 range check refuses (to the f64
//!                  device path) a B_ov element beyond f32::MAX
//!   precision      f32 storage, f32 panel products, f64 sums
//! Gate: the mixed energy error is bounded by the deterministic energy bound on
//! every system, and the error decomposes as storage + accumulation:
//!   (device − CPU)        = (CPU on f32-rounded B_ov − CPU)   [storage, eps 2·u32]
//!                         + (device − CPU on f32-rounded B_ov) [accumulation]
//! with the accumulation part bounded by the same formula at its own factor
//! eps_acc = γ_b(u32) + γ_⌈naux/b⌉(u64) + γ_naux(u64) (the exact f64
//! reference's own rounding is in the last term). The device sees the SAME
//! rounded B_ov as the CPU storage run because `upload_rounded` is `as f32`
//! exactly as `round_trip_f32`; the equivalence (the mixed device energy IS the
//! f64 energy of a perturbed B_ov, up to the accumulation bound) is the
//! backward-error reading of the decomposition and is asserted by it.
//! MEASURED (b = 128; printed as `error map` lines; cc-pVDZ rows from an
//! exact-integral RHF, aug-cc and alkane_12 rows from an RI-JK RHF with loose thresholds, the
//! same orbitals on both sides of each comparison):
//!   system (frozen core)        nocc nvir naux  E_corr       |dE| Eh   rel      bound OS/SS        kappa_E  p99 kappa(i=0)
//!   h2o/cc-pvdz                    5   19    84  -0.204033   3.5e-9   1.7e-8   2.1e-6 / 3.7e-6    1.30     6.3e11
//!   c4h10/cc-pvdz                 17   89   364  -0.599645   1.6e-9   2.7e-9   1.4e-5 / 2.7e-5    1.95     1.4e11
//!   c8h18/cc-pvdz                 33  169   700  -1.183893   4.4e-11  3.7e-11  3.4e-5 / 6.6e-5    2.40     1.3e8
//!   alkane_12/cc-pvdz (RI-JK SCF) 49  249  1036  -1.767936   1.2e-10  6.9e-11  5.9e-5 / 1.1e-4    2.76     1.2e11
//!   h2o/aug-cc-pvtz                5   87   198  -0.283553   2.5e-9   8.9e-9   5.1e-6 / 9.2e-6    1.50     1.2e11
//!   benzene/aug-cc-pvdz (6)       15  171   570  -0.810690   1.3e-9   1.6e-9   1.5e-5 / 2.8e-5    1.65     2.1e12
//!   benzene/aug-cc-pvtz (6)       15  393   912  -0.963520   1.4e-10  1.5e-10  2.1e-5 / 3.9e-5    1.84     3.0e12
//! Decomposition at benzene/aug-cc-pvtz: storage (CPU on f32-rounded B_ov)
//! +7.5e-11 / +1.0e-10 (OS/SS), accumulation (device - storage) -5.0e-10 /
//! +4.7e-10, bounds 3.2e-7 / 5.9e-7 and 2.1e-5 / 3.8e-5.
//! Error law (cc-pVDZ, N = nocc, four systems): slope(|dE|) -1.80 +- 0.74,
//! slope(bound) 1.49 +- 0.05, slope(|dE|/bound) -3.29 +- 0.72; the asserted law
//! is slope(|dE|) <= slope(bound) + 2 SE (the ratio does not grow with N); four
//! noisy points do not resolve a sqrt(N) law.
//! Teeth (asserted on every system above): the f32-storage defect exceeds the
//! f64 device gate's bound (the gpu_rimp2_resident.rs bound, gamma_naux(u64)
//! class), by 2.3x at benzene/aug-cc-pvtz up to 1e4x on water/cc-pvdz.
//! Closing the kernel: this budget row is why `SHIPPED` contains `RiMp2Energy`.
//!
//! No timings: the CPU is contested. Only the correctness gates run here.
use ferric_core::gpu::device::GpuError;
use ferric_core::gpu::device::{device, FORCE_KERNEL_FAILURE};
use ferric_core::gpu::mixed::effective_k_panel;
use ferric_core::gpu::mixed_host::{gamma, mixed_error_factor, round_trip_f32, U32, U64};
use ferric_core::gpu::pool::DevicePool;
use ferric_core::gpu::{
    install, pool, probe, stats, GpuMode, GpuSettingsExplicit, GpuStatus, MixedKernel,
    MixedKernelSet, Precision,
};
use ferric_mp2::rimp2::{spin_components_from_b_ov_kappa, spin_components_from_b_ov_kappa_cpu};
use ferric_mp2::rimp2_gpu::{g_block_on_device, spin_components_on_device};
use ndarray::{linalg::general_mat_mul, s, Array2};
use std::sync::atomic::Ordering;

#[path = "common/rimp2_error_bound.rs"]
mod bound;
#[path = "common/rimp2_gpu_fixture.rs"]
mod fixture;
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

fn on_device(
    p: &Prepared,
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
        None,
        precision,
        true,
    )
}

fn cpu_of(p: &Prepared, b: &Array2<f64>) -> ferric_mp2::rimp2::SpinComponents {
    spin_components_from_b_ov_kappa_cpu(b, &p.eps, p.nocc, p.nvir, p.first_occ, p.nocc_total, None)
}

#[test]
fn every_g_block_meets_the_per_element_bound() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let dev = device(0).unwrap();
    let pool = pool().unwrap();
    let b = effective_k_panel();
    for (sys, obs, aux) in [
        ("h2o", "cc-pvdz", "cc-pvdz-ri"),
        ("benzene", "cc-pvdz", "cc-pvdz-ri"),
    ] {
        let (p, _) = fixture::prepare_scf(sys, obs, aux, 0);
        let naux = p.b_ov.nrows();
        let eps_g = (mixed_error_factor(naux, b) + gamma(naux, U64)) * 1.01;
        let babs = p.b_ov.mapv(f64::abs);
        for i in [0usize, p.nocc / 2, p.nocc - 1] {
            let g_dev =
                g_block_on_device(&dev, &pool, &p.b_ov, i, p.nvir, Precision::Mixed).unwrap();
            let blk = p.b_ov.slice(s![.., i * p.nvir..(i + 1) * p.nvir]);
            let g_cpu = blk.t().dot(&p.b_ov.slice(s![.., i * p.nvir..]));
            let mut sab = Array2::zeros(g_cpu.dim());
            general_mat_mul(
                1.0,
                &babs.slice(s![.., i * p.nvir..(i + 1) * p.nvir]).t(),
                &babs.slice(s![.., i * p.nvir..]),
                0.0,
                &mut sab,
            );
            let mut worst = 0.0f64;
            let mut kappas = Vec::new();
            for ((d, c), sv) in g_dev.iter().zip(g_cpu.iter()).zip(sab.iter()) {
                if *sv > 0.0 {
                    worst = worst.max((d - c).abs() / (eps_g * sv));
                    if c.abs() > 0.0 {
                        kappas.push(sv / c.abs());
                    }
                }
            }
            kappas.sort_by(|a, b| a.total_cmp(b));
            let p99 = kappas[(kappas.len() as f64 * 0.99) as usize];
            eprintln!(
                "{sys} i={i}: max |dg|/(eps_G S) = {worst:.3e}; kappa_sum max {:.3e} p99 {p99:.3e} (naux {naux}, b {b})",
                kappas.last().unwrap()
            );
            assert!(
                worst <= 1.0,
                "{sys} i={i}: per-element bound violated ({worst:e})"
            );
        }
    }
}

/// One system through the whole gate; returns (nocc, |dE| total, bound total).
fn gate(
    label: &str,
    p: &Prepared,
    cpu: &ferric_mp2::rimp2::SpinComponents,
    teeth: bool,
) -> (f64, f64, f64) {
    let b = effective_k_panel();
    let naux = p.b_ov.nrows();
    let s0 = stats();
    let mixed = on_device(p, Precision::Mixed).unwrap();
    let s1 = stats();
    assert!(
        s1.gemm_mixed - s0.gemm_mixed == p.nocc as u64
            && s1.mixed_fallback_f64 == s0.mixed_fallback_f64,
        "{label}: the mixed path was not taken for every block"
    );
    let storage = cpu_of(p, &round_trip_f32(&p.b_ov.view()));
    let eps_full = mixed_error_factor(naux, b) + gamma(naux, U64);
    let eps_acc = eps_full - 2.0 * U32;
    let full = bound::energy_bound(
        &p.b_ov,
        &p.eps,
        p.nocc,
        p.nvir,
        p.first_occ,
        p.nocc_total,
        eps_full * 1.01,
    );
    let acc = bound::energy_bound(
        &p.b_ov,
        &p.eps,
        p.nocc,
        p.nvir,
        p.first_occ,
        p.nocc_total,
        eps_acc * 1.01,
    );
    let sto = bound::energy_bound(
        &p.b_ov,
        &p.eps,
        p.nocc,
        p.nvir,
        p.first_occ,
        p.nocc_total,
        2.0 * U32,
    );
    let (dm_os, dm_ss) = (mixed.e_os - cpu.e_os, mixed.e_ss - cpu.e_ss);
    let (ds_os, ds_ss) = (storage.e_os - cpu.e_os, storage.e_ss - cpu.e_ss);
    let (da_os, da_ss) = (mixed.e_os - storage.e_os, mixed.e_ss - storage.e_ss);
    let kcal = 627.5094740631;
    eprintln!(
        "error map mixed {label}: nocc {} nvir {} naux {naux} b {b} | E_corr {:.9} | dE_mixed {dm_os:+.3e} {dm_ss:+.3e} (total {:+.3e} Eh = {:+.3e} kcal/mol, rel {:.3e}) | dE_storage {ds_os:+.3e} {ds_ss:+.3e} | dE_acc {da_os:+.3e} {da_ss:+.3e} | bounds full {:.3e} {:.3e} storage {:.3e} {:.3e} acc {:.3e} {:.3e} | ratios {:.3e} {:.3e} | kappa_E {:.3e} kappa_p99(i=0) {:.3e}",
        p.nocc, p.nvir, cpu.e_total, mixed.e_total - cpu.e_total, (mixed.e_total - cpu.e_total) * kcal,
        (mixed.e_total - cpu.e_total).abs() / cpu.e_total.abs(),
        full.os, full.ss, sto.os, sto.ss, acc.os, acc.ss, dm_os.abs() / full.os, dm_ss.abs() / full.ss,
        full.kappa_e_os, full.kappa_p99_block0
    );
    assert!(
        dm_os.abs() <= full.os && dm_ss.abs() <= full.ss,
        "{label}: mixed energy error exceeds the deterministic bound"
    );
    // decomposition: each part against its OWN bound
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
    if teeth {
        // The f64 device gate (gpu_rimp2_resident.rs) must reject this energy:
        // its bound is the f64 two-sided one (gamma_naux(u64) class), so the
        // f32-storage defect sits above it. Compare the storage defect with the
        // f64 per-element bound at 2·gamma_naux(u64).
        let f64b = bound::energy_bound(
            &p.b_ov,
            &p.eps,
            p.nocc,
            p.nvir,
            p.first_occ,
            p.nocc_total,
            2.0 * gamma(naux, U64),
        );
        assert!(
            ds_os.abs() > f64b.os && ds_ss.abs() > f64b.ss,
            "{label}: the f32-storage defect ({:.3e}, {:.3e}) is inside the f64 gate ({:.3e}, {:.3e}); that gate is blind here",
            ds_os.abs(), ds_ss.abs(), f64b.os, f64b.ss
        );
    }
    (
        p.nocc as f64,
        (mixed.e_total - cpu.e_total).abs(),
        full.os + full.ss,
    )
}

#[test]
fn mixed_energy_meets_the_bound_decomposes_and_follows_the_n_law_at_cc_pvdz() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let mut rows = Vec::new();
    for sys in ["h2o", "c4h10", "c8h18", "alkane_12"] {
        let (p, cpu) = fixture::prepare_scf(sys, "cc-pvdz", "cc-pvdz-ri", 0);
        rows.push(gate(&format!("{sys}/cc-pvdz"), &p, &cpu, true));
    }
    let fit = |f: &dyn Fn(&(f64, f64, f64)) -> f64| {
        bound::loglog_fit(&rows.iter().map(|r| (r.0, f(r))).collect::<Vec<_>>())
    };
    let (s_err, se_err, _) = fit(&|r| r.1);
    let (s_bnd, se_bnd, _) = fit(&|r| r.2);
    let (s_rat, se_rat, _) = fit(&|r| r.1 / r.2);
    eprintln!(
        "error law mixed (N = nocc, cc-pvdz): slope(|dE|) {s_err:.3} +- {se_err:.3} | slope(bound) {s_bnd:.3} +- {se_bnd:.3} | slope(|dE|/bound) {s_rat:.3} +- {se_rat:.3}"
    );
    assert!(
        s_err <= s_bnd + 2.0 * se_err,
        "measured error slope {s_err:.3} exceeds the bound's slope {s_bnd:.3} + 2 SE {se_err:.3}"
    );
}

/// One real system of the basis axis through the whole gate.
fn basis_axis(sys: &str, obs: &str, aux: &str, frozen: usize) {
    let (p, cpu) = fixture::prepare_scf(sys, obs, aux, frozen);
    eprintln!(
        "{sys}/{obs} fc{frozen}: nocc {} nvir {} naux {}: B_ov f32 {:.1} MB, G_i scratch {:.1} MB",
        p.nocc,
        p.nvir,
        p.b_ov.nrows(),
        4.0 * p.b_ov.len() as f64 / 1e6,
        (12 * p.nvir * p.b_ov.ncols()) as f64 / 1e6
    );
    gate(&format!("{sys}/{obs}/fc{frozen}"), &p, &cpu, true);
}

#[test]
fn the_basis_axis_is_gated_on_real_aug_cc_systems() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    // water in aug-cc-pVTZ (all electrons) and benzene in aug-cc-pVDZ with a
    // frozen core of 6 (nocc 15): a real benzene system at the production
    // occupied count; the full production shape is the next test
    basis_axis("h2o", "aug-cc-pvtz", "aug-cc-pvtz-rifit", 0);
    basis_axis("benzene", "aug-cc-pvdz", "aug-cc-pvdz-rifit", 6);
}

#[test]
fn benzene_aug_cc_pvtz_is_gated_at_the_production_shape() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    basis_axis("benzene", "aug-cc-pvtz", "aug-cc-pvtz-rifit", 6);
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
    // a pool of exactly the mixed requirement: mixed runs and peaks at exactly
    // that; f64 is refused (PoolFull) at the same capacity
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
    // one byte short for mixed: refused, naming the f32 B_ov
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
    assert!(
        label.contains("f32") && detail.contains("scratch"),
        "{label} / {detail}"
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
    // and a direct mixed call with the kernel unavailable is a typed refusal
    // BEFORE any reservation or upload
    FORCE_KERNEL_FAILURE.store(true, Ordering::Relaxed);
    let u0 = stats().resident_uploads;
    let r = on_device(&p, Precision::Mixed);
    FORCE_KERNEL_FAILURE.store(false, Ordering::Relaxed);
    assert!(matches!(r, Err(GpuError::Kernel(_))), "{r:?}");
    assert_eq!(stats().resident_uploads, u0, "nothing was uploaded");
}

#[test]
fn an_operand_beyond_f32_range_falls_back_to_the_f64_device_path_with_a_counter() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let (mut p, _) = fixture::prepare_scf("h2o", "cc-pvdz", "cc-pvdz-ri", 0);
    p.b_ov[(0, 0)] = 1e300; // finite in f64, +inf as f32
    let direct = on_device(&p, Precision::Mixed);
    assert!(matches!(direct, Err(GpuError::F32Range(_))), "{direct:?}");
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
    assert_eq!(s1.mixed_fallback_f64 - s0.mixed_fallback_f64, 1, "counted");
    assert_eq!(s1.gemm_mixed, s0.gemm_mixed, "no mixed GEMM");
    assert_eq!(
        s1.gemm_offloaded - s0.gemm_offloaded,
        p.nocc as u64,
        "f64 device GEMMs ran"
    );
    assert_eq!(
        via.e_total.to_bits(),
        f64dev.e_total.to_bits(),
        "the fallback is the f64 device energy"
    );
    // a non-finite operand is the caller's data, not an f32-range overflow: it
    // goes through (and yields a non-finite energy, as the CPU path would)
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
