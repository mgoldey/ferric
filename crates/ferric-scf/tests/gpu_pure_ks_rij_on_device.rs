#![cfg(feature = "gpu")]
//! A pure-functional RKS (RI-J, no K) with a usable device must build J on the
//! device. The closed-shell SCF overlaps the RI-J build with the XC digest under
//! one `rayon::join`, and the device RI-J path declines on a rayon worker, so
//! with the overlap on J silently stayed on the CPU every iteration (found
//! 2026-11-10, T019). The SCF now skips the overlap when a device is usable.
//!
//! Called from the test's own (non-rayon) thread, as the CLI calls it.
use ferric_core::basis;
use ferric_core::gpu::{install, probe, stats, GpuMode, GpuSettingsExplicit, GpuStatus};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig, JXC_OVERLAP_BUILDS};
use ferric_scf::screening::SchwarzBounds;
use std::sync::atomic::Ordering;

const WATER: &str = "3\nH2O\nO 0 0 0\nH 0 0.7572 0.5868\nH 0 -0.7572 0.5868\n";

#[test]
fn pure_functional_rks_builds_ri_j_on_the_device() {
    if !matches!(probe(0), GpuStatus::Ready(_)) {
        eprintln!("skipping: no CUDA device");
        assert!(
            std::env::var("FERRIC_GPU_TESTS_REQUIRED").ok().as_deref() != Some("1"),
            "FERRIC_GPU_TESTS_REQUIRED=1 but no device"
        );
        return;
    }
    install(GpuSettingsExplicit {
        mode: Some(GpuMode::Auto),
        ..Default::default()
    })
    .expect("install");
    let mol = Molecule::parse_xyz(WATER, 0, 1).unwrap();
    let prep = PreparedBasis::new(&mol, &basis::bundled("def2-svp").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let cfg = RhfConfig {
        xc: Some("PBE".into()),
        df_j_aux: Some("def2-universal-jkfit".into()),
        density_conv: 1e-6,
        max_iter: 6,
        ..Default::default()
    };
    let (a, ov0) = (stats(), JXC_OVERLAP_BUILDS.load(Ordering::Relaxed));
    let _ = solve_rhf(&ParallelContext::default(), &mol, &prep, op, &bounds, &cfg);
    let (b, ov1) = (stats(), JXC_OVERLAP_BUILDS.load(Ordering::Relaxed));
    assert!(
        b.dfj_device_builds > a.dfj_device_builds,
        "RI-J never ran on the device ({} overlapped J/XC builds, RI-J declined {})",
        ov1 - ov0,
        b.dfj_declined - a.dfj_declined
    );
    assert_eq!(ov1, ov0, "the J/XC overlap ran with a usable device");
}
