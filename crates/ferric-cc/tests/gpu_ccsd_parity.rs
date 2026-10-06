#![cfg(feature = "gpu")]
//! Closed-shell CCSD (spin-adapted, DF) with every einsum on the device vs the
//! CPU, compared on the CCSD correlation energy (the total adds the identical
//! RHF energy and quantises at 1.4e-14, hiding last-bit differences). The
//! tolerance is derived from two measured sides, all as |dE_corr|/|E_corr|.
//!
//! Agreement side, same code GPU vs CPU (cc-pVDZ, cc-pvdz-ri), measured on
//! GPU 0 and GPU 1 with identical results:
//!     water  |dE| = 8.3e-17 Ha (rel 3.9e-16)
//!     NH3    |dE| = 8.3e-17 Ha (rel 4.1e-16)
//!     ethane |dE| = 1.1e-16 Ha (rel 3.2e-16)   largest observed: 4.1e-16
//!
//! Defect side, each mutant of `gemm_f64` applied once, water, outcome recorded:
//!   (a) last k-block drops its final k element: rel 9.0e-4 (|dE| 1.9e-4 Ha)
//!   (b) beta = 0 on every k-block (later blocks overwrite): rel 9.9e-1 (|dE| 2.1e-1 Ha)
//!   (c) operands rounded through f32 before upload: rel 2.5e-9 water, 2.2e-9 NH3,
//!       4.6e-9 ethane (|dE| 5.2e-10, 4.4e-10, 1.6e-9 Ha)
//!   (d) transa/transb swapped: cuBLAS rejects most shapes (34 of ~1001 GEMMs
//!       offloaded, rel 7.6e-1); caught by the fallback-counter assertion
//!       (gemm_cpu_cuda_error = 9274), not by the energy.
//! Smallest defect: (c) on NH3, 2.2e-9. The f32 mutant is damped by the CCSD
//! iteration (amplitudes converge to the fixed point of the perturbed residual),
//! so it lands near the 1e-9 class, well below the gross mutants.
//!
//! committed: REL_TOL = sqrt(4.1e-16 * 2.2e-9) = 9.5e-13 -> 9e-13 (1 s.f.), at
//! the 1e-13 floor (the k-blocked CPU path's accuracy class) or above.
//! The CPU reference runs inside a 1-thread rayon pool, where the worker rule
//! declines the device. The device run must offload at least one GEMM and have
//! ZERO cuda-error / layout / pool-full / below-threshold / inside-worker
//! fallbacks, so a device path that is silently refused cannot pass.
//!
//! Run: `OPENBLAS_NUM_THREADS=1 FERRIC_GPU_TESTS_REQUIRED=1 cargo test -p ferric-cc
//! --features gpu --test gpu_ccsd_parity -- --nocapture` (wrap in the shared GPU lock).
use ferric_cc::ccsd_closed_shell::ccsd_closed_shell;
use ferric_cc::CcConfig;
use ferric_core::basis;
use ferric_core::gpu::{install, probe, stats, GpuMode, GpuSettingsExplicit, GpuStatus};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

/// One device test at a time (stats deltas are process-global).
static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Derived above: geometric mean of the measured agreement and the defect.
const REL_TOL: f64 = 9e-13;

const SYSTEMS: [&str; 3] = ["water", "nh3", "c2h6"];

#[test]
fn ccsd_energy_on_device_matches_cpu_within_derived_tolerance() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !matches!(probe(0), GpuStatus::Ready(_)) {
        eprintln!("skipping: no CUDA device");
        assert!(std::env::var("FERRIC_GPU_TESTS_REQUIRED").ok().as_deref() != Some("1"));
        return;
    }
    install(GpuSettingsExplicit {
        mode: Some(GpuMode::Auto),
        min_flops: Some(0),
        // Explicit pool: install() sizes the process-wide pool first-wins, so a later
        // install_pool_for_tests would be a no-op. Any second test in this binary
        // must install IDENTICAL settings (install() errs on a different second set).
        memory_gb: Some(2.0),
        ..Default::default()
    })
    .expect("install explicit gpu settings");
    assert_ne!(ferric_core::gpu::settings().mode, GpuMode::Off);
    let mut worst = 0.0f64;
    for sys in SYSTEMS {
        // CPU reference: inside a 1-thread rayon pool the worker rule declines the device.
        let before = stats().gemm_offloaded;
        let cpu = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| run_ccsd(sys, "cc-pvdz"));
        assert_eq!(
            stats().gemm_offloaded,
            before,
            "{sys}: the CPU reference reached the device"
        );
        let s0 = stats();
        let gpu = run_ccsd(sys, "cc-pvdz");
        let s1 = stats();
        let offloaded = s1.gemm_offloaded - s0.gemm_offloaded;
        let cuda_err = s1.gemm_cpu_cuda_error - s0.gemm_cpu_cuda_error;
        let layout = s1.gemm_cpu_layout - s0.gemm_cpu_layout;
        let pool_full = s1.gemm_cpu_pool_full - s0.gemm_cpu_pool_full;
        let below = s1.gemm_cpu_below_threshold - s0.gemm_cpu_below_threshold;
        let in_worker = s1.gemm_cpu_inside_worker - s0.gemm_cpu_inside_worker;
        let rel = (gpu - cpu).abs() / cpu.abs();
        eprintln!(
            "{sys}: E_cpu={cpu:.17e} E_gpu={gpu:.17e} |dE|={:.3e} rel={rel:.3e} offloaded={offloaded} \
             cuda_error={cuda_err} layout={layout} pool_full={pool_full} below_threshold={below} inside_worker={in_worker}",
            (gpu - cpu).abs()
        );
        assert!(
            offloaded > 0,
            "{sys}: no GEMM reached the device; the comparison would be vacuous"
        );
        // A silent CPU fallback would make the comparison vacuous for that call: every
        // refusal class must be zero (pool 2 GB fits the largest operand set here;
        // threshold 0 declines nothing; the device run is outside any rayon worker).
        assert_eq!(
            (cuda_err, layout, pool_full, below, in_worker),
            (0, 0, 0, 0, 0),
            "{sys}: device run fell back to the CPU for some GEMMs"
        );
        worst = worst.max(rel);
    }
    assert!(
        worst <= REL_TOL,
        "CCSD GPU vs CPU worst rel {worst:.3e} > {REL_TOL:e}"
    );
}

/// SCF/CCSD setup as `validation_cc.rs` (same RHF thresholds, same RI aux), so
/// the runs differ only in where GEMMs execute. Returns the CCSD correlation energy (the total adds the identical RHF energy and quantises at 1.4e-14, hiding last-bit differences).
fn run_ccsd(sys: &str, basis_name: &str) -> f64 {
    let root = env!("CARGO_MANIFEST_DIR");
    let validation = format!("{root}/../../testdata/molecules/validation/{sys}.xyz");
    let xyz = if std::path::Path::new(&validation).exists() {
        validation
    } else {
        format!("{root}/../../testdata/molecules/{sys}.xyz")
    };
    let mol = Molecule::load_xyz_with_charge(&xyz, 0, 1).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled(basis_name).unwrap()).unwrap();
    let dfbs =
        PreparedBasis::new(&mol, &basis::bundled(&format!("{basis_name}-ri")).unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(
        &ParallelContext::default(),
        &mol,
        &obs,
        op,
        &bounds,
        &RhfConfig {
            max_iter: 200,
            energy_conv: 1e-11,
            density_conv: 1e-9,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(rhf.converged, "{sys}: RHF did not converge");
    let cc = ccsd_closed_shell(
        &mol,
        &obs,
        &dfbs,
        op,
        &rhf,
        &CcConfig {
            max_iter: 200,
            energy_conv: 1e-11,
            ..Default::default()
        },
    )
    .unwrap();
    cc.correlation_energy
}
