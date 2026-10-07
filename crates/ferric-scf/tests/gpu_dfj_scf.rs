#![cfg(feature = "gpu")]
//! RI-J through the real SCF solve path (not `DfJ::new`): a single-rank RHF with
//! RI-J reaches the device DF-J, and the forced-host solve never does. `DfJ` keeps
//! a `ctx` only when it spans more than one rank, so a one-rank `Some(ctx)` from
//! `build_df_jk` is eligible. The two energies agree to the DF-J rounding floor
//! (GEMV rounding enters through the metric solve and is second order in the
//! converged energy).
use ferric_core::basis;
use ferric_core::gpu::{install, probe, stats, GpuMode, GpuSettingsExplicit, GpuStatus};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::df_j_gpu::FORCE_HOST;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use std::sync::atomic::Ordering;

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
    static INSTALL: std::sync::Once = std::sync::Once::new();
    INSTALL.call_once(|| {
        install(GpuSettingsExplicit {
            mode: Some(GpuMode::Auto),
            memory_gb: Some(1.0),
            ..Default::default()
        })
        .expect("install");
    });
    true
}

#[test]
fn rhf_ri_j_through_the_solve_path_reaches_the_device() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
    let prep = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let cfg = RhfConfig {
        density_conv: 1e-8,
        df_j_aux: Some("def2-universal-jkfit".into()),
        df_k_aux: Some(String::new()),
        ..Default::default()
    };
    let run = |host: bool| {
        FORCE_HOST.store(host, Ordering::SeqCst);
        let r = solve_rhf(&ParallelContext::default(), &mol, &prep, op, &bounds, &cfg);
        FORCE_HOST.store(false, Ordering::SeqCst);
        r.unwrap().energy
    };
    let s0 = stats();
    let e_host = run(true);
    let s1 = stats();
    assert_eq!(
        s1.dfj_device_builds, s0.dfj_device_builds,
        "the forced-host solve built J on the device"
    );
    let e_dev = run(false);
    let s2 = stats();
    assert!(
        s2.dfj_device_builds > s1.dfj_device_builds,
        "the device solve never built J on the device (declined: {})",
        s2.dfj_declined - s1.dfj_declined
    );
    eprintln!("RI-J RHF water: host {e_host:.12} device {e_dev:.12}");
    assert!((e_dev - e_host).abs() < 1e-9, "dE = {:e}", e_dev - e_host);
}
