#![cfg(feature = "gpu")]
//! Unrestricted RI-MP2 energy blocks on the device (f64, two resident B_ov) vs
//! the CPU path, on E_αα, E_ββ, E_αβ separately.
//!
//! The agreement gate is a DERIVED bound (`common/u_rimp2_gpu_bound.rs`), not a
//! tuned tolerance: the device and the CPU form the same wide block
//! G_i = B_iᵀ·B (accumulation depth k = naux) and feed it to the same
//! `same/opposite_spin_block_energy`, so they differ only in how each GEMM
//! rounded; any summation order satisfies |ĝ − g| ≤ γ_naux Σ_P|B B| (Higham,
//! ASNA Eq. 3.5), the two blocks differ by ≤ 2γ_naux·S, and the terms and the
//! m-term accumulation propagate that as derived there. This is the
//! closed-shell Phase-1 rule transposed to the antisymmetrised K = g_ab − g_ba
//! (SS) and the plain g² (OS); it is NOT the closed-shell error map.
//!
//! TWO-SIDED (both sides printed on every run; measured numbers are recorded
//! in the Task F2 report):
//!   side 1, the device is inside the bound: |E_dev − E_cpu| / bound ≪ 1;
//!   side 2, a defect is outside it: (a) B rounded through f32 on the CPU, the
//!   nearest subtle defect, exceeds the bound on the real systems (asserted);
//!   (b) a TRANSPOSED-OPERAND mutant of the device block (the GEMM's operands
//!   exchanged, `u_g_block_on_device` computing B_rightᵀ·B_left_i and reading
//!   it as the nvir × ncols block) applied once by temporary mutation of
//!   rimp2_gpu.rs and reverted: this binary FAILS on it, numbers in the report.
//!
//! Also pins: the downloaded bytes of every block, one upload per tensor under
//! the labels "U-RI-MP2 B_ov alpha"/"beta", the amplitude path never touching
//! the device, a mid-loop failure leaving the counters untouched and the CPU
//! path rerunning from scratch, and zero-size shapes (nocc_b = 0) not
//! panicking.
use ferric_core::gpu::device::{device, GpuError};
use ferric_core::gpu::mixed_host::round_trip_f32;
use ferric_core::gpu::pool::DevicePool;
use ferric_core::gpu::{install, pool, probe, stats, GpuMode, GpuSettingsExplicit, GpuStatus};
use ferric_mp2::rimp2::RiMp2Config;
use ferric_mp2::rimp2_gpu::{
    try_u_opposite_spin_on_device, try_u_same_spin_on_device, u_opposite_spin_on_device,
    u_same_spin_on_device, FAIL_AT_BLOCK,
};
use ferric_mp2::u_rimp2::{opposite_spin_pair_kernel, same_spin_pair_kernel, u_ri_mp2};
use std::sync::atomic::Ordering;

#[path = "common/u_rimp2_gpu_bound.rs"]
mod bound;
#[path = "common/u_rimp2_gpu_fixture.rs"]
mod fixture;
#[path = "common/u_rimp2_gpu_real.rs"]
mod real;
#[path = "common/u_rimp2_gpu_synth.rs"]
mod synth;
use fixture::{cpu_opp, cpu_same, Chan};

static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn ready() -> bool {
    if !matches!(probe(0), GpuStatus::Ready(_)) {
        eprintln!("skipping: no CUDA device");
        assert!(
            std::env::var("FERRIC_GPU_TESTS_REQUIRED").ok().as_deref() != Some("1"),
            "FERRIC_GPU_TESTS_REQUIRED=1 but no device"
        );
        return false;
    }
    install(GpuSettingsExplicit {
        mode: Some(GpuMode::Auto),
        memory_gb: Some(2.0),
        ..Default::default()
    })
    .expect("install");
    true
}

fn f32_rounded(c: &Chan) -> Chan {
    Chan {
        b: round_trip_f32(&c.b.view()),
        eps: c.eps.clone(),
        nocc: c.nocc,
        nvir: c.nvir,
        first_occ: c.first_occ,
        nocc_total: c.nocc_total,
    }
}

/// The three block energies on the device against the process pool.
fn device_energies(a: &Chan, b: &Chan) -> (f64, f64, f64) {
    let dev = device(0).unwrap();
    let pool = pool().unwrap();
    (
        u_same_spin_on_device(&dev, &pool, a.ch()).unwrap(),
        u_same_spin_on_device(&dev, &pool, b.ch()).unwrap(),
        u_opposite_spin_on_device(&dev, &pool, a.ch(), b.ch()).unwrap(),
    )
}

#[test]
fn u_rimp2_energy_matches_cpu_within_the_summation_order_bound() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    for (name, xyz, mult) in [
        ("O2 triplet", "o2.xyz", 3usize),
        ("CH3 doublet", "validation/ch3.xyz", 2),
    ] {
        let case = real::real_case(xyz, 0, mult);
        let (a, b) = real::chans(&case.amps);
        assert_ne!(a.nocc, b.nocc, "{name}: alpha and beta must differ");
        let (cpu_aa, cpu_bb, cpu_ab) = (cpu_same(&a), cpu_same(&b), cpu_opp(&a, &b));
        let s0 = stats();
        let (d_aa, d_bb, d_ab) = device_energies(&a, &b);
        let s1 = stats();
        // one GEMM per occupied block of each of the three energies
        assert_eq!(
            s1.gemm_offloaded - s0.gemm_offloaded,
            (a.nocc + b.nocc + a.nocc) as u64,
            "{name}: device path not taken as specified"
        );
        assert_eq!(
            (
                s1.gemm_cpu_cuda_error,
                s1.gemm_cpu_layout,
                s1.gemm_cpu_pool_full
            ),
            (
                s0.gemm_cpu_cuda_error,
                s0.gemm_cpu_layout,
                s0.gemm_cpu_pool_full
            ),
            "{name}: a fallback fired"
        );
        let (b_aa, b_bb, b_ab) = (
            bound::same_spin_bound(&a),
            bound::same_spin_bound(&b),
            bound::opposite_spin_bound(&a, &b),
        );
        // side 2a: B stored through f32 on the CPU, the nearest subtle defect
        let (fa, fb) = (f32_rounded(&a), f32_rounded(&b));
        let f_aa = (cpu_same(&fa) - cpu_aa).abs();
        let f_bb = (cpu_same(&fb) - cpu_bb).abs();
        let f_ab = (cpu_opp(&fa, &fb) - cpu_ab).abs();
        let e = [
            ("aa", (d_aa - cpu_aa).abs(), b_aa, f_aa),
            ("bb", (d_bb - cpu_bb).abs(), b_bb, f_bb),
            ("ab", (d_ab - cpu_ab).abs(), b_ab, f_ab),
        ];
        for (blk, err, bd, f32err) in e {
            eprintln!(
                "{name} E_{blk}: |dev-cpu| {err:.3e}, bound {bd:.3e}, err/bound {:.3e} | f32-storage defect/bound {:.3e}",
                err / bd,
                f32err / bd
            );
            assert!(err <= bd, "{name} E_{blk}: {err:e} > bound {bd:e}");
            assert!(
                f32err > bd,
                "{name} E_{blk}: the f32-storage defect {f32err:e} is inside the bound {bd:e}; the gate is blind"
            );
        }
        // the dispatchers reach the same numbers (bit-identical to the direct
        // calls: same code, serial sum)
        let via_aa = try_u_same_spin_on_device(a.ch()).expect("dispatcher takes the device");
        let via_ab =
            try_u_opposite_spin_on_device(a.ch(), b.ch()).expect("dispatcher takes the device");
        assert_eq!(
            (via_aa.to_bits(), via_ab.to_bits()),
            (d_aa.to_bits(), d_ab.to_bits())
        );
        // end to end, u_ri_mp2 on the main thread (device) vs inside a 1-thread
        // pool (CPU), every component inside its bound
        let cfg = RiMp2Config::default();
        let dev = u_ri_mp2(
            &case.mol,
            &case.obs,
            &case.dfbs,
            ferric_integrals::operator::Operator::coulomb(),
            &case.scf,
            &cfg,
        )
        .unwrap()
        .components;
        let cpu = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| {
                u_ri_mp2(
                    &case.mol,
                    &case.obs,
                    &case.dfbs,
                    ferric_integrals::operator::Operator::coulomb(),
                    &case.scf,
                    &cfg,
                )
                .unwrap()
                .components
            });
        assert!((dev.e_aa - cpu.e_aa).abs() <= b_aa, "{name} e2e E_aa");
        assert!((dev.e_bb - cpu.e_bb).abs() <= b_bb, "{name} e2e E_bb");
        assert!((dev.e_ab - cpu.e_ab).abs() <= b_ab, "{name} e2e E_ab");
    }
}

/// The O2/CH3 channels have naux ~ 100, where the device and CPU GEMMs round
/// identically (error exactly 0); a deeper contraction (naux 600, several
/// k-blocks) makes the two summation orders actually differ, so side 1 of the
/// two-sided check is measured on a non-zero error too.
#[test]
fn deep_contraction_differs_in_rounding_and_stays_inside_the_bound() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let a = synth::synthetic(600, 5, 18, 0, 41);
    let b = synth::synthetic(600, 4, 21, 0, 42);
    let (d_aa, d_bb, d_ab) = device_energies(&a, &b);
    let errs = [
        (
            "aa",
            (d_aa - cpu_same(&a)).abs(),
            bound::same_spin_bound(&a),
        ),
        (
            "bb",
            (d_bb - cpu_same(&b)).abs(),
            bound::same_spin_bound(&b),
        ),
        (
            "ab",
            (d_ab - cpu_opp(&a, &b)).abs(),
            bound::opposite_spin_bound(&a, &b),
        ),
    ];
    for (blk, err, bd) in errs {
        eprintln!(
            "deep naux 600 E_{blk}: err {err:.3e} bound {bd:.3e} err/bound {:.3e}",
            err / bd
        );
        assert!(err <= bd, "E_{blk}: {err:e} > {bd:e}");
    }
    // the same rule's other side on this shape: f32-rounded B is far outside
    let f = f32_rounded(&a);
    let f_aa = (cpu_same(&f) - cpu_same(&a)).abs();
    assert!(f_aa > bound::same_spin_bound(&a));
}

#[test]
fn two_b_tensors_are_uploaded_once_and_each_block_moves_the_documented_bytes() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let dev = device(0).unwrap();
    let a = synth::synthetic(150, 6, 20, 1, 11);
    let b = synth::synthetic(150, 4, 22, 1, 12);
    let (bytes_a, bytes_b) = ((8 * a.b.len()) as u64, (8 * b.b.len()) as u64);

    // same-spin: one upload of B, nocc GEMMs, block i is nvir x (nocc-i)*nvir
    let big = DevicePool::with_capacity_bytes(1 << 26);
    let s0 = stats();
    u_same_spin_on_device(&dev, &big, a.ch()).unwrap();
    let s1 = stats();
    let d2h: u64 = (0..a.nocc)
        .map(|i| (8 * a.nvir * a.nvir * (a.nocc - i)) as u64)
        .sum();
    assert_eq!(
        (
            s1.gemm_offloaded - s0.gemm_offloaded,
            s1.resident_uploads - s0.resident_uploads,
            s1.bytes_h2d - s0.bytes_h2d,
            s1.bytes_d2h - s0.bytes_d2h
        ),
        (a.nocc as u64, 1, bytes_a, d2h),
        "same-spin"
    );

    // opposite-spin: two uploads (alpha, beta), nocc_a GEMMs, each block is
    // nvir_a x nocc_b*nvir_b (the full beta width: no triangle)
    let s0 = stats();
    u_opposite_spin_on_device(&dev, &big, a.ch(), b.ch()).unwrap();
    let s1 = stats();
    let d2h = (a.nocc * 8 * a.nvir * b.nocc * b.nvir) as u64;
    assert_eq!(
        (
            s1.gemm_offloaded - s0.gemm_offloaded,
            s1.resident_uploads - s0.resident_uploads,
            s1.bytes_h2d - s0.bytes_h2d,
            s1.bytes_d2h - s0.bytes_d2h
        ),
        (a.nocc as u64, 2, bytes_a + bytes_b, d2h),
        "opposite-spin"
    );
    assert_eq!(big.available_bytes(), big.capacity_bytes(), "all released");

    // the labels: a pool one byte short of each upload names that tensor
    let scratch = 8 * a.nvir * b.b.ncols();
    let short_beta =
        DevicePool::with_capacity_bytes(scratch + bytes_a as usize + bytes_b as usize - 1);
    let e = u_opposite_spin_on_device(&dev, &short_beta, a.ch(), b.ch()).unwrap_err();
    assert!(matches!(e, GpuError::PoolFull { .. }), "{e}");
    assert!(e.to_string().contains("U-RI-MP2 B_ov beta"), "{e}");
    let short_alpha = DevicePool::with_capacity_bytes(scratch + bytes_a as usize - 1);
    let e = u_opposite_spin_on_device(&dev, &short_alpha, a.ch(), b.ch()).unwrap_err();
    assert!(e.to_string().contains("U-RI-MP2 B_ov alpha"), "{e}");
    assert_eq!(short_alpha.available_bytes(), short_alpha.capacity_bytes());
}

#[test]
fn amplitude_path_never_touches_the_device() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let a = synth::synthetic(120, 5, 18, 0, 3);
    let b = synth::synthetic(120, 4, 19, 0, 4);
    // on the main thread, where the energy-only kernels WOULD take the device
    let probe_dev = stats();
    let (e_ss, t_ss) = same_spin_pair_kernel(a.ch(), true);
    let (e_os, t_os) = opposite_spin_pair_kernel(a.ch(), b.ch(), true);
    let after = stats();
    assert!(t_ss.is_some() && t_os.is_some());
    assert!(e_ss.is_finite() && e_os.is_finite());
    assert_eq!(
        (
            after.gemm_offloaded - probe_dev.gemm_offloaded,
            after.resident_uploads - probe_dev.resident_uploads,
            after.bytes_h2d - probe_dev.bytes_h2d,
            after.bytes_d2h - probe_dev.bytes_d2h
        ),
        (0, 0, 0, 0),
        "the amplitude path must stay on the CPU"
    );
    // control: the energy-only kernel on the same inputs DOES offload
    let (e_dev, none) = same_spin_pair_kernel(a.ch(), false);
    assert!(e_dev.is_finite());
    let ctl = stats();
    assert!(none.is_none());
    assert_eq!(ctl.gemm_offloaded - after.gemm_offloaded, a.nocc as u64);
}

#[test]
fn a_mid_loop_failure_leaves_the_counters_and_reruns_the_cpu_path() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let a = synth::synthetic(100, 6, 15, 0, 5);
    let b = synth::synthetic(100, 5, 16, 0, 6);
    let (cpu_ss, cpu_os) = (cpu_same(&a), cpu_opp(&a, &b));
    FAIL_AT_BLOCK.store(3, Ordering::Relaxed);
    let s0 = stats();
    let (ss, os) = (
        same_spin_pair_kernel(a.ch(), false).0,
        opposite_spin_pair_kernel(a.ch(), b.ch(), false).0,
    );
    FAIL_AT_BLOCK.store(usize::MAX, Ordering::Relaxed);
    let s1 = stats();
    // blocks 0..3 finished before the failure, but no GEMM or download is
    // counted; the uploads did move bytes (the resident upload counts them when
    // it happens): same-spin B_alpha, then B_alpha and B_beta of opposite-spin
    assert_eq!(
        (s1.gemm_offloaded, s1.bytes_d2h),
        (s0.gemm_offloaded, s0.bytes_d2h),
        "blocks 0..3 finished but nothing may be counted"
    );
    assert_eq!(
        s1.bytes_h2d - s0.bytes_h2d,
        (8 * (2 * a.b.len() + b.b.len())) as u64
    );
    assert_eq!(s1.gemm_cpu_cuda_error - s0.gemm_cpu_cuda_error, 2);
    assert_eq!(
        (ss.to_bits(), os.to_bits()),
        (cpu_ss.to_bits(), cpu_os.to_bits()),
        "the fallback is the CPU path from scratch, bit for bit"
    );
}

#[test]
fn zero_size_shapes_do_not_panic_and_take_the_cpu_path() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let dev = device(0).unwrap();
    let pool = DevicePool::with_capacity_bytes(1 << 26);
    let a = synth::synthetic(60, 3, 10, 0, 7);
    // a one-electron-beta shape: no occupied beta orbital
    let b0 = synth::synthetic(60, 0, 13, 0, 8);
    assert_eq!(b0.b.dim(), (60, 0));
    let e = u_opposite_spin_on_device(&dev, &pool, a.ch(), b0.ch()).unwrap_err();
    assert!(matches!(e, GpuError::Layout(_)), "{e}");
    let e = u_same_spin_on_device(&dev, &pool, b0.ch()).unwrap_err();
    assert!(matches!(e, GpuError::Layout(_)), "{e}");
    // through the dispatchers: counted as a layout fallback, CPU energy
    let s0 = stats();
    let via = opposite_spin_pair_kernel(a.ch(), b0.ch(), false).0;
    let via_ss = same_spin_pair_kernel(b0.ch(), false).0;
    let s1 = stats();
    assert_eq!(s1.gemm_cpu_layout - s0.gemm_cpu_layout, 2);
    assert_eq!(s1.gemm_offloaded, s0.gemm_offloaded);
    assert_eq!(via.to_bits(), cpu_opp(&a, &b0).to_bits());
    assert_eq!(via_ss.to_bits(), cpu_same(&b0).to_bits());
    assert_eq!(via, 0.0);
    // inconsistent shapes are typed refusals, never an ndarray panic
    let mut bad = synth::synthetic(60, 3, 10, 0, 9);
    bad.nvir = 11; // b is 60 x 30, nocc * nvir = 33
    let e = u_same_spin_on_device(&dev, &pool, bad.ch()).unwrap_err();
    assert!(matches!(e, GpuError::Layout(_)), "{e}");
    let mut short = synth::synthetic(60, 3, 10, 0, 9);
    short.eps.truncate(5);
    let e = u_same_spin_on_device(&dev, &pool, short.ch()).unwrap_err();
    assert!(matches!(e, GpuError::Layout(_)), "{e}");
    let other = synth::synthetic(61, 3, 10, 0, 9); // naux differs
    let e = u_opposite_spin_on_device(&dev, &pool, a.ch(), other.ch()).unwrap_err();
    assert!(matches!(e, GpuError::Layout(_)), "{e}");
    let huge = Chan {
        b: ndarray::Array2::zeros((1, 1)),
        eps: vec![0.0; 4],
        nocc: usize::MAX,
        nvir: 2,
        first_occ: 0,
        nocc_total: 1,
    };
    let e = u_same_spin_on_device(&dev, &pool, huge.ch()).unwrap_err();
    assert!(matches!(e, GpuError::Layout(_)), "{e}");
    assert_eq!(pool.available_bytes(), pool.capacity_bytes());
}
