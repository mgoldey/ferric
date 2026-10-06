#![cfg(feature = "gpu")]
//! Closed-shell CCSD (spin-adapted, DF) with every einsum on the device vs the
//! CPU, compared on the CCSD correlation energy (the total adds the identical
//! RHF energy and quantises at 1.4e-14, hiding last-bit differences). The
//! tolerance is derived from two measured sides:
//!   same code, GPU vs CPU (cc-pVDZ, cc-pvdz-ri):
//!     water  |dE| = 8.3e-17 Ha (rel 3.9e-16)
//!     NH3    |dE| = 8.3e-17 Ha (rel 4.1e-16)
//!     ethane |dE| = 1.1e-16 Ha (rel 3.2e-16)
//!   transa/transb swapped in `gemm_f64` (defect), water: |dE| = 1.63e-1 Ha (rel 7.6e-1)
//!   committed: rel <= REL_TOL = sqrt(4.1e-16 * 7.6e-1) = 1.8e-8 -> 2e-8 (1 s.f.),
//!   above the 1e-13 floor of the k-blocked CPU path's accuracy class.
//! The CPU reference runs inside a 1-thread rayon pool, where the worker rule
//! declines the device; the test asserts that run moved no GEMM to the device
//! and that the device run offloaded at least one.
//!
//! Run: `OPENBLAS_NUM_THREADS=1 FERRIC_GPU_TESTS_REQUIRED=1 cargo test -p ferric-cc
//! --features gpu --test gpu_ccsd_parity -- --nocapture` (wrap in the shared GPU lock).
use ferric_cc::ccsd_closed_shell::ccsd_closed_shell;
use ferric_cc::CcConfig;
use ferric_core::basis;
use ferric_core::gpu::{
    install, install_pool_for_tests, probe, stats, GpuMode, GpuSettingsExplicit, GpuStatus,
};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

/// One device test at a time (stats deltas are process-global).
static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Derived above: geometric mean of the measured agreement and the defect.
const REL_TOL: f64 = 2e-8;

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
        ..Default::default()
    })
    .expect("install explicit gpu settings");
    assert_ne!(ferric_core::gpu::settings().mode, GpuMode::Off);
    let _pool = install_pool_for_tests(2 << 30);
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
        let gpu = run_ccsd(sys, "cc-pvdz");
        let offloaded = stats().gemm_offloaded - before;
        assert!(
            offloaded > 0,
            "{sys}: no GEMM reached the device; the comparison would be vacuous"
        );
        let rel = (gpu - cpu).abs() / cpu.abs();
        eprintln!(
            "{sys}: E_cpu={cpu:.17e} E_gpu={gpu:.17e} |dE|={:.3e} rel={rel:.3e} offloaded={offloaded}",
            (gpu - cpu).abs()
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
