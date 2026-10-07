#![cfg(feature = "gpu")]
//! RI-MP2 energy with B_ov resident on the device (f64) vs the CPU path, on
//! E_os and E_ss separately.
//!
//! The agreement gate is a DERIVED bound, not a tuned tolerance. The device and
//! the CPU form the same wide block G_i = B_iᵀ·B_tail (accumulation depth
//! k = naux) and feed it to the same `pair_energy`, so they differ only in how
//! each GEMM rounded. Any summation order satisfies (Higham, ASNA Eq. 3.5)
//! |ĝ − g| ≤ γ_k Σ_P |B_P,ia B_P,jb| with γ_k = k·u/(1 − k·u), u = 2⁻⁵³, so
//! the two blocks differ by δ_ab ≤ 2·γ_naux·S_ab, S = |B_i|ᵀ|B_tail|. Per term
//!   OS: |Δ(g²/D)|             ≤ damp·δ(2|g| + δ)/|D|
//!   SS: |Δ(g_ab² − g_ab·g_ba)|≤ damp·[δ(2|g| + δ) + δ|g_ba| + |g_ab|δ' + δδ']/|D|
//! (damp = 1 without κ, (1 − e^{κD})² ≤ 1 with it), summed with the pair
//! weights. The accumulation of nvir² terms per pair (then ≤ nocc pairs, ≤
//! nocc blocks) rounds differently on the two sides: γ_m(Σ|t| + Σ|t̂|) with
//! m = nvir² + 2·nocc + 8 (the +8 covers the few operations inside one term),
//! Σ|t̂| ≤ Σ|t| + (the bound above). `energy_bound` computes this sum; no
//! machine constant is asserted.
//!
//! Sensitivity (measured on the GTX 1080 test box; the test prints both sides
//! on every run). On the two real SCF systems, |device − CPU| / bound is at
//! most ~1e-4 (the bound is a worst case over summation orders and signs, so
//! it is loose by design), and the nearest subtle defect, B_ov rounded through
//! f32 on the CPU, exceeds the bound by 1e1 .. 1e4 and is asserted to, so the
//! gate cannot go blind. The synthetic aTZ-shape system (random-sign B_ov,
//! heavy cancellation in the contraction) is reported but not given the
//! teeth assertion: there the worst-case bound is ~2x the f32-storage error.
//! Gross defects, measured once by temporary mutation of rimp2_gpu.rs (each
//! reverted, file diffed against the original), on water/cc-pVDZ:
//!   right operand view offset `slice(off..)` -> `slice(..)`: relative error
//!     dE_os 1.9e-1, dE_ss 6.1e-1; this test FAILS (error 2.9e-2 against a
//!     bound of 2.0e-14);
//!   pair weight `fac` forced to 1.0 (the j > i symmetry weight dropped):
//!     relative error dE_os 2.9e-1, dE_ss 5.0e-1; this test FAILS.
//! Both sit 12 orders above the bound, so the gate cannot pass them by looseness.
//!
//! Also pins: serial vs parallel pair fold bit-identical for 1, 3 and 4
//! explicit rayon workers; run-to-run device bit-identity; zero CPU fallbacks
//! during the device run; the dispatcher takes the device on an ample pool.
use ferric_core::gpu::device::{device, GpuError};
use ferric_core::gpu::mixed_host::round_trip_f32;
use ferric_core::gpu::pool::DevicePool;
use ferric_core::gpu::{
    install, pool, probe, stats, GpuMode, GpuSettingsExplicit, GpuStatus, Precision,
};
use ferric_mp2::rimp2::{spin_components_from_b_ov_kappa, spin_components_from_b_ov_kappa_cpu};
use ferric_mp2::rimp2_gpu::{g_block_on_device, spin_components_on_device};
use ndarray::{s, Array2};

#[path = "common/rimp2_gpu_fixture.rs"]
mod fixture;
use fixture::Prepared;

static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Unit roundoff of f64 (round to nearest).
const U: f64 = 1.0 / (1u64 << 53) as f64;

fn gamma(k: usize) -> f64 {
    let ku = k as f64 * U;
    ku / (1.0 - ku)
}

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

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(f64::MIN_POSITIVE)
}

/// A random but well-posed problem of RI-MP2 shape without an SCF: B_ov of
/// either sign, occupied eps < 0 < virtual eps (so every denominator is
/// negative and OS terms do not cancel).
fn synthetic(naux: usize, nocc: usize, nvir: usize, first_occ: usize, seed: u64) -> Prepared {
    let mut st = seed;
    let mut next = move || {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (st >> 11) as f64 / (1u64 << 53) as f64
    };
    let b_ov = Array2::from_shape_fn((naux, nocc * nvir), |_| 0.1 * (next() - 0.5));
    let nocc_total = first_occ + nocc;
    let mut eps = Vec::with_capacity(nocc_total + nvir);
    for _ in 0..nocc_total {
        eps.push(-1.5 + 1.2 * next());
    }
    for _ in 0..nvir {
        eps.push(0.2 + 1.8 * next());
    }
    Prepared {
        b_ov,
        eps,
        nocc,
        nvir,
        first_occ,
        nocc_total,
    }
}

/// The derived bound on |E_device − E_cpu| for (OS, SS); see the module docs.
fn energy_bound(p: &Prepared, kappa: Option<f64>) -> (f64, f64) {
    let naux = p.b_ov.nrows();
    let gk = gamma(naux);
    let gm = gamma(p.nvir * p.nvir + 2 * p.nocc + 8);
    let abs_b = p.b_ov.mapv(f64::abs);
    let (mut b_os, mut b_ss, mut abs_os, mut abs_ss) = (0.0, 0.0, 0.0, 0.0);
    for i in 0..p.nocc {
        let (lo, hi) = (i * p.nvir, (i + 1) * p.nvir);
        let g = p
            .b_ov
            .slice(s![.., lo..hi])
            .t()
            .dot(&p.b_ov.slice(s![.., lo..]));
        let sc = abs_b
            .slice(s![.., lo..hi])
            .t()
            .dot(&abs_b.slice(s![.., lo..]));
        for j in i..p.nocc {
            let fac = if i == j { 1.0 } else { 2.0 };
            let jcol = (j - i) * p.nvir;
            let e_ij = p.eps[p.first_occ + i] + p.eps[p.first_occ + j];
            for a in 0..p.nvir {
                for b in 0..p.nvir {
                    let (g_ab, g_ba) = (g[(a, jcol + b)].abs(), g[(b, jcol + a)].abs());
                    let (d_ab, d_ba) = (2.0 * gk * sc[(a, jcol + b)], 2.0 * gk * sc[(b, jcol + a)]);
                    let denom = e_ij - p.eps[p.nocc_total + a] - p.eps[p.nocc_total + b];
                    let damp = match kappa {
                        None => 1.0,
                        Some(k) => {
                            let d1 = 1.0 - (k * denom).exp();
                            d1 * d1
                        }
                    };
                    let w = fac * damp / denom.abs();
                    let sq = d_ab * (2.0 * g_ab + d_ab);
                    b_os += w * sq;
                    b_ss += w * (sq + d_ab * g_ba + g_ab * d_ba + d_ab * d_ba);
                    abs_os += w * g_ab * g_ab;
                    abs_ss += w * g_ab * (g_ab + g_ba);
                }
            }
        }
    }
    (
        b_os + gm * (2.0 * abs_os + b_os),
        b_ss + gm * (2.0 * abs_ss + b_ss),
    )
}

fn on_device(
    p: &Prepared,
    kappa: Option<f64>,
    parallel: bool,
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
        Precision::F64,
        parallel,
    )
}

fn cpu_of(
    p: &Prepared,
    b_ov: &Array2<f64>,
    kappa: Option<f64>,
) -> ferric_mp2::rimp2::SpinComponents {
    spin_components_from_b_ov_kappa_cpu(
        b_ov,
        &p.eps,
        p.nocc,
        p.nvir,
        p.first_occ,
        p.nocc_total,
        kappa,
    )
}

/// (name, problem, is a real SCF system).
fn systems() -> Vec<(String, Prepared, bool)> {
    let mut v = Vec::new();
    for (sys, obs, aux) in [
        ("water", "cc-pvdz", "cc-pvdz-ri"),
        ("benzene", "cc-pvdz", "cc-pvdz-ri"),
    ] {
        v.push((
            format!("{sys}/{obs}"),
            fixture::prepare_scf(sys, obs, aux, 0).0,
            true,
        ));
    }
    // The benzene/aug-cc-pVTZ shape the performance table is written for
    // (naux 912, nocc 15 with a frozen core of 6, nvir 393), synthetic data.
    v.push((
        "synthetic aTZ shape".into(),
        synthetic(912, 15, 393, 6, 11),
        false,
    ));
    v
}

#[test]
fn device_f64_energy_is_within_the_derived_bound_and_the_gate_has_teeth() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    for (name, p, real) in systems() {
        let cpu = cpu_of(&p, &p.b_ov, None);
        let s0 = stats();
        let dev = on_device(&p, None, true).unwrap();
        let s1 = stats();
        // one GEMM per occupied block, one resident upload, and the downloads
        // are exactly the G_i blocks: block i is nvir x (nocc - i)*nvir f64,
        // which pins the j >= i tail shape (a full-width block would move more)
        let d2h: u64 = (0..p.nocc)
            .map(|i| (8 * p.nvir * p.nvir * (p.nocc - i)) as u64)
            .sum();
        assert_eq!(
            (
                s1.gemm_offloaded - s0.gemm_offloaded,
                s1.resident_uploads - s0.resident_uploads,
                s1.bytes_d2h - s0.bytes_d2h
            ),
            (p.nocc as u64, 1, d2h),
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
        let (b_os, b_ss) = energy_bound(&p, None);
        let (d_os, d_ss) = ((dev.e_os - cpu.e_os).abs(), (dev.e_ss - cpu.e_ss).abs());
        // the nearest subtle defect: B_ov stored through f32, on the CPU
        let f32side = cpu_of(&p, &round_trip_f32(&p.b_ov.view()), None);
        let (f_os, f_ss) = (
            (f32side.e_os - cpu.e_os).abs(),
            (f32side.e_ss - cpu.e_ss).abs(),
        );
        eprintln!(
            "{name}: nocc {} nvir {} naux {}: rel dE_os {:.3e} dE_ss {:.3e} | err/bound os {:.3e} ss {:.3e} | f32-storage defect/bound os {:.3e} ss {:.3e}",
            p.nocc,
            p.nvir,
            p.b_ov.nrows(),
            rel(dev.e_os, cpu.e_os),
            rel(dev.e_ss, cpu.e_ss),
            d_os / b_os,
            d_ss / b_ss,
            f_os / b_os,
            f_ss / b_ss
        );
        assert!(
            d_os <= b_os,
            "{name}: E_os differs by {d_os:e} > bound {b_os:e}"
        );
        assert!(
            d_ss <= b_ss,
            "{name}: E_ss differs by {d_ss:e} > bound {b_ss:e}"
        );
        assert!(
            !real || (f_os > b_os && f_ss > b_ss),
            "{name}: the f32-storage defect ({f_os:e}, {f_ss:e}) is inside the bound ({b_os:e}, {b_ss:e}); the gate is blind"
        );
    }
}

#[test]
fn pair_fold_is_bit_identical_for_every_worker_count_and_run_to_run() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let p = synthetic(150, 9, 31, 2, 5);
    let serial = on_device(&p, None, false).unwrap();
    let bits = |s: &ferric_mp2::rimp2::SpinComponents| {
        (s.e_os.to_bits(), s.e_ss.to_bits(), s.e_total.to_bits())
    };
    for workers in [1usize, 3, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap();
        let par = pool.install(|| on_device(&p, None, true).unwrap());
        assert_eq!(bits(&par), bits(&serial), "{workers} workers vs serial");
    }
    let again = on_device(&p, None, false).unwrap();
    assert_eq!(bits(&again), bits(&serial), "run to run");
}

#[test]
fn kappa_path_takes_the_device_within_its_derived_bound() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let p = synthetic(120, 7, 23, 1, 3);
    let kappa = Some(1.1);
    let cpu = cpu_of(&p, &p.b_ov, kappa);
    let dev = on_device(&p, kappa, true).unwrap();
    let (b_os, b_ss) = energy_bound(&p, kappa);
    assert!((dev.e_os - cpu.e_os).abs() <= b_os);
    assert!((dev.e_ss - cpu.e_ss).abs() <= b_ss);
    // damping must actually act: the kappa energy is not the undamped one
    assert!(rel(cpu.e_os, cpu_of(&p, &p.b_ov, None).e_os) > 1e-3);
}

#[test]
fn per_element_g_blocks_match_the_cpu_within_two_gamma_s() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let dev = device(0).unwrap();
    let pool = pool().unwrap();
    let p = synthetic(300, 5, 17, 0, 9); // naux > 2 k-panels
    let gk = gamma(p.b_ov.nrows());
    let abs_b = p.b_ov.mapv(f64::abs);
    for i in [0usize, 2, 4] {
        let (lo, hi) = (i * p.nvir, (i + 1) * p.nvir);
        let want = p
            .b_ov
            .slice(s![.., lo..hi])
            .t()
            .dot(&p.b_ov.slice(s![.., lo..]));
        let sc = abs_b
            .slice(s![.., lo..hi])
            .t()
            .dot(&abs_b.slice(s![.., lo..]));
        let got = g_block_on_device(&dev, &pool, &p.b_ov, i, p.nvir, Precision::F64).unwrap();
        assert_eq!(got.dim(), want.dim());
        for (idx, &w) in want.indexed_iter() {
            let err = (got[idx] - w).abs();
            assert!(err <= 2.0 * gk * sc[idx], "i={i} {idx:?}: {err:e}");
        }
    }
}

#[test]
fn ample_pool_runs_the_device_path() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let p = synthetic(200, 6, 20, 0, 21);
    let process_pool = pool().unwrap();
    assert!(process_pool.available_bytes() > 8 * p.b_ov.len() + 8 * p.nvir * p.b_ov.ncols());
    let cpu = cpu_of(&p, &p.b_ov, None);
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
    assert_eq!(
        s1.gemm_offloaded - s0.gemm_offloaded,
        p.nocc as u64,
        "the dispatcher must offload exactly one GEMM per occupied block"
    );
    assert_eq!(s1.resident_uploads - s0.resident_uploads, 1);
    assert_eq!(
        s1.bytes_h2d - s0.bytes_h2d,
        (8 * p.b_ov.len()) as u64,
        "B_ov is uploaded once and nothing else moves host to device"
    );
    assert_eq!(
        (
            s1.gemm_cpu_pool_full,
            s1.gemm_cpu_layout,
            s1.gemm_cpu_cuda_error,
            s1.mixed_fallback_f64
        ),
        (
            s0.gemm_cpu_pool_full,
            s0.gemm_cpu_layout,
            s0.gemm_cpu_cuda_error,
            s0.mixed_fallback_f64
        )
    );
    let (b_os, b_ss) = energy_bound(&p, None);
    assert!((via.e_os - cpu.e_os).abs() <= b_os && (via.e_ss - cpu.e_ss).abs() <= b_ss);
    assert_eq!(
        process_pool.available_bytes(),
        process_pool.capacity_bytes(),
        "every reservation released when the energy returns"
    );
}

#[test]
fn degenerate_shapes_never_panic_and_mixed_is_a_typed_refusal() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let dev = device(0).unwrap();
    let pool = DevicePool::with_capacity_bytes(1 << 26);
    let run = |b: &Array2<f64>, eps: &[f64], nocc, nvir, first, total, prec| {
        spin_components_on_device(
            &dev, &pool, b, eps, nocc, nvir, first, total, None, prec, true,
        )
    };
    let f = Precision::F64;
    // zero occupied or virtual: exactly zero, no device work
    let z = run(&Array2::zeros((4, 0)), &[0.0; 4], 0, 3, 0, 0, f).unwrap();
    assert_eq!((z.e_os, z.e_ss, z.e_total), (0.0, 0.0, 0.0));
    let z = run(&Array2::zeros((4, 0)), &[0.0; 4], 3, 0, 0, 3, f).unwrap();
    assert_eq!(z.e_total, 0.0);
    // no auxiliary functions: typed refusal (the CPU path owns it)
    let e = run(
        &Array2::zeros((0, 6)),
        &[-1.0, -1.0, 1.0, 1.0, 1.0, 1.0],
        2,
        3,
        0,
        2,
        f,
    );
    assert!(matches!(e, Err(GpuError::Layout(_))), "{e:?}");
    // b_ov width disagrees with nocc * nvir
    let e = run(&Array2::zeros((4, 7)), &[0.0; 12], 2, 3, 0, 2, f);
    assert!(matches!(e, Err(GpuError::Layout(_))), "{e:?}");
    // eps too short for the index ranges
    let e = run(&Array2::zeros((4, 6)), &[0.0; 3], 2, 3, 0, 2, f);
    assert!(matches!(e, Err(GpuError::Layout(_))), "{e:?}");
    // nocc * nvir overflows usize
    let e = run(&Array2::zeros((1, 1)), &[0.0; 4], usize::MAX, 2, 0, 1, f);
    assert!(matches!(e, Err(GpuError::Layout(_))), "{e:?}");
    // the mixed arm ships separately: a typed refusal, never a silent f64 run
    let p = synthetic(8, 2, 3, 0, 1);
    let s0 = stats();
    let e = run(&p.b_ov, &p.eps, p.nocc, p.nvir, 0, p.nocc, Precision::Mixed);
    assert!(matches!(e, Err(GpuError::Kernel(_))), "{e:?}");
    let e = g_block_on_device(&dev, &pool, &p.b_ov, 0, p.nvir, Precision::Mixed);
    assert!(matches!(e, Err(GpuError::Kernel(_))), "{e:?}");
    // an out-of-range block index is refused, not indexed
    let e = g_block_on_device(&dev, &pool, &p.b_ov, p.nocc, p.nvir, Precision::F64);
    assert!(matches!(e, Err(GpuError::Layout(_))), "{e:?}");
    // a width that is not a multiple of nvir leaves a partial last block:
    // typed refusal for every block that is not whole (never an ndarray panic),
    // before anything is reserved or uploaded
    let ragged = Array2::<f64>::zeros((4, 10)); // nvir = 4: blocks 0, 1 whole, block 2 partial
    for i in [2usize, 3, 9] {
        let e = g_block_on_device(&dev, &pool, &ragged, i, 4, Precision::F64);
        assert!(matches!(e, Err(GpuError::Layout(_))), "i={i}: {e:?}");
    }
    let ok = g_block_on_device(&dev, &pool, &ragged, 1, 4, Precision::F64).unwrap();
    assert_eq!(
        ok.dim(),
        (4, 6),
        "the last whole block's tail is the rest of the row"
    );
    let s1 = stats();
    assert_eq!(
        (
            s1.gemm_offloaded - s0.gemm_offloaded,
            s1.resident_uploads - s0.resident_uploads
        ),
        (1, 1),
        "only the one valid block did any device work"
    );
    assert_eq!(pool.available_bytes(), pool.capacity_bytes());
}
