#![cfg(feature = "gpu")]
//! Pool pressure, both refusal points, in one binary (`install` is once per
//! process): a process pool too small for the G_i scratch (and a fortiori for
//! B_ov) makes the dispatcher run the CPU path, count `PoolFull` exactly once,
//! upload nothing, and return the mode-off energy bit for bit; a pool that
//! fits the scratch but not B_ov is refused AFTER the scratch was reserved and
//! leaves no charge behind. The ample direction is gpu_rimp2_resident.rs.
use ferric_core::gpu::device::{device, GpuError};
use ferric_core::gpu::pool::DevicePool;
use ferric_core::gpu::{install, probe, stats, GpuMode, GpuSettingsExplicit, GpuStatus, Precision};
use ferric_mp2::rimp2::{spin_components_from_b_ov_kappa, spin_components_from_b_ov_kappa_cpu};
use ferric_mp2::rimp2_gpu::spin_components_on_device;

#[path = "common/rimp2_gpu_fixture.rs"]
mod fixture;

/// Counters are process-wide: device tests in this binary run one at a time.
static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// `true` when a device is present. Installs the 1 MB process pool, once and
/// before anything reads the settings (`install` is once per process, and the
/// SCF in `prepare_scf` reads them).
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
            memory_gb: Some(0.001),
            ..Default::default()
        })
        .expect("install");
    });
    true
}

#[test]
fn tight_pool_runs_the_cpu_path_bit_identically_and_counts_it() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    // The process pool is 1 MB (see `ready`): benzene/cc-pVDZ needs a 1.45 MB
    // G_i scratch and a 420 x 1953 x 8 B = 6.6 MB B_ov (sizes asserted below,
    // not assumed).
    let (p, reference) = fixture::prepare_scf("benzene", "cc-pvdz", "cc-pvdz-ri", 0);
    let scratch_bytes = 8 * p.nvir * p.b_ov.ncols();
    let b_bytes = 8 * p.b_ov.len();
    assert!(
        scratch_bytes > 1_000_000 && b_bytes > scratch_bytes,
        "fixture no longer exercises a tight pool: scratch {scratch_bytes} B, B_ov {b_bytes} B"
    );

    let s0 = stats();
    let via_dispatch = spin_components_from_b_ov_kappa(
        &p.b_ov,
        &p.eps,
        p.nocc,
        p.nvir,
        p.first_occ,
        p.nocc_total,
        None,
    );
    let s1 = stats();
    let cpu = spin_components_from_b_ov_kappa_cpu(
        &p.b_ov,
        &p.eps,
        p.nocc,
        p.nvir,
        p.first_occ,
        p.nocc_total,
        None,
    );
    assert_eq!(
        s1.gemm_cpu_pool_full - s0.gemm_cpu_pool_full,
        1,
        "PoolFull must be counted exactly once"
    );
    assert_eq!(
        s1.gemm_offloaded, s0.gemm_offloaded,
        "no GEMM may reach the device"
    );
    assert_eq!(
        (s1.resident_uploads, s1.bytes_h2d),
        (s0.resident_uploads, s0.bytes_h2d),
        "nothing may be uploaded when refused"
    );
    assert_eq!(
        (s1.gemm_cpu_layout, s1.gemm_cpu_cuda_error),
        (s0.gemm_cpu_layout, s0.gemm_cpu_cuda_error),
        "the only fallback reason is the pool"
    );
    for (name, a, b) in [
        ("dispatch vs mode-off CPU", &via_dispatch, &cpu),
        ("dispatch vs the reference run", &via_dispatch, &reference),
    ] {
        assert_eq!(
            (a.e_os.to_bits(), a.e_ss.to_bits(), a.e_total.to_bits()),
            (b.e_os.to_bits(), b.e_ss.to_bits(), b.e_total.to_bits()),
            "{name}: a tight pool must give the CPU energy bit for bit"
        );
    }
}

#[test]
fn a_pool_that_fits_the_scratch_but_not_b_ov_is_refused_and_released() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let dev = device(0).unwrap();
    let (p, _) = fixture::prepare_scf("water", "cc-pvdz", "cc-pvdz-ri", 0);
    let scratch_bytes = 8 * p.nvir * p.b_ov.ncols();
    let b_bytes = 8 * p.b_ov.len();
    assert!(b_bytes > 1, "water B_ov is not empty");
    // room for the scratch and all of B_ov but one byte
    let pool = DevicePool::with_capacity_bytes(scratch_bytes + b_bytes - 1);
    let s0 = stats();
    let e = spin_components_on_device(
        &dev,
        &pool,
        &p.b_ov,
        &p.eps,
        p.nocc,
        p.nvir,
        p.first_occ,
        p.nocc_total,
        None,
        Precision::F64,
        true,
    )
    .unwrap_err();
    assert!(matches!(e, GpuError::PoolFull { .. }), "{e}");
    assert!(
        e.to_string().contains("B_ov"),
        "the refusal names B_ov: {e}"
    );
    assert_eq!(
        pool.available_bytes(),
        pool.capacity_bytes(),
        "scratch lease released"
    );
    let s1 = stats();
    assert_eq!(
        (s1.resident_uploads, s1.bytes_h2d),
        (s0.resident_uploads, s0.bytes_h2d)
    );
}
