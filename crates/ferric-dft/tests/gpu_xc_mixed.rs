#![cfg(all(feature = "gpu", feature = "test-seams"))]
//! The mixed-precision `xc-vxc` kernel on the device (`[gpu] precision =
//! "mixed"`, `xc-vxc` shipped). Its own binary: the GPU settings are installed
//! once per process, and this one installs mixed.
//!
//! * The precision ladder: a first build is f64; a build whose density moved is
//!   mixed (`gemm_mixed` and `MIXED_VXC_BUILDS` move, V differs from f64); an
//!   unchanged density is f64 again. A meta-GGA (r2SCAN) is never mixed.
//! * The device error is the same class as the host twin's (`mixed_host`, the
//!   algorithm whose elementwise bound `ferric-dft`'s unit test pins): within
//!   a factor of 10 either way, and well below 1e-5 of max|V|. E_xc comes
//!   from the f64 density pass, so it agrees with f64 to rounding.
//! * Two device runs are bit-identical.
//! * A whole SCF (water PBE/cc-pVDZ, `density_conv` 1e-10) on the device in
//!   mixed agrees with the f64 host SCF to 1e-10 Eh.
use ferric_core::basis;
use ferric_core::gpu::{
    install, probe, stats, GpuMode, GpuSettingsExplicit, GpuStatus, MixedKernel, MixedKernelSet,
    Precision,
};
use ferric_core::mol::Molecule;
use ferric_dft::grid::AtomicGridConfig;
use ferric_dft::ks::{KsXc, MIXED_VXC_BUILDS};
use ferric_dft::xc_batch::{FORCE_HOST, MIXED_VXC_HOST_TWIN};
use ferric_dft::xc_trait::XcContribution;
use ndarray::Array2;
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
        assert!(MixedKernelSet::SHIPPED.contains(MixedKernel::XcVxc));
        install(GpuSettingsExplicit {
            mode: Some(GpuMode::Auto),
            memory_gb: Some(1.0),
            precision: Some(Precision::Mixed),
            ..Default::default()
        })
        .expect("install");
    });
    true
}

fn water() -> Molecule {
    Molecule::parse_xyz(
        "3\nH2O\nO 0 0 0\nH 0 0.7572 0.5868\nH 0 -0.7572 0.5868\n",
        0,
        1,
    )
    .unwrap()
}

fn ks(xc: &str) -> KsXc {
    let main = AtomicGridConfig {
        n_radial: 50,
        n_angular: 110,
        ..Default::default()
    };
    let nlc = AtomicGridConfig {
        n_radial: 50,
        n_angular: 50,
        ..Default::default()
    };
    KsXc::new(
        &water(),
        &basis::bundled("cc-pvdz").unwrap(),
        xc,
        &main,
        &nlc,
    )
    .unwrap()
}

/// `2·C·Cᵀ` for a deterministic `(nbf, 5)` block with entries in `[-0.3, 0.3]`.
fn density(nbf: usize, seed: u64) -> Array2<f64> {
    let mut x = seed;
    let c = Array2::from_shape_fn((nbf, 5), |_| {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((x >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 0.6
    });
    2.0 * c.dot(&c.t())
}

fn max_abs(a: &Array2<f64>) -> f64 {
    a.iter().fold(0.0_f64, |m, v| m.max(v.abs()))
}

/// Builds at D0, D1, D1 on a fresh evaluator: (E, V) of each.
fn ladder_run(xc: &str, host: bool, twin: bool) -> Vec<(f64, Array2<f64>)> {
    FORCE_HOST.store(host, Ordering::SeqCst);
    MIXED_VXC_HOST_TWIN.store(twin, Ordering::SeqCst);
    let k = ks(xc);
    let nbf = k.xc_screening_stats().nbf;
    let (d0, d1) = (density(nbf, 3), density(nbf, 5));
    let out = [&d0, &d1, &d1]
        .iter()
        .map(|d| {
            let mut f = Array2::<f64>::zeros((nbf, nbf));
            (k.add_xc(d, &mut f), f)
        })
        .collect();
    FORCE_HOST.store(false, Ordering::SeqCst);
    MIXED_VXC_HOST_TWIN.store(false, Ordering::SeqCst);
    out
}

#[test]
fn the_ladder_runs_moving_builds_mixed_inside_the_twin_error_class() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    // Meta-GGA τ never runs below f64: r2SCAN makes no mixed build.
    let m0 = MIXED_VXC_BUILDS.load(Ordering::SeqCst);
    let mgga_dev = ladder_run("R2SCAN", false, false);
    let mgga_host = ladder_run("R2SCAN", true, false);
    assert_eq!(
        MIXED_VXC_BUILDS.load(Ordering::SeqCst),
        m0,
        "r2SCAN ran mixed"
    );
    let dv = max_abs(&(&mgga_dev[1].1 - &mgga_host[1].1));
    assert!(
        dv < 1e-10 * max_abs(&mgga_host[1].1).max(1.0),
        "r2SCAN max|dV| {dv:e}"
    );
    for xc in ["LDA", "PBE"] {
        let f64_host = ladder_run(xc, true, false);
        let twin = ladder_run(xc, true, true);
        let (s0, m0) = (stats(), MIXED_VXC_BUILDS.load(Ordering::SeqCst));
        let dev = ladder_run(xc, false, false);
        let (s1, m1) = (stats(), MIXED_VXC_BUILDS.load(Ordering::SeqCst));
        let again = ladder_run(xc, false, false);
        assert_eq!(m1 - m0, 1, "{xc}: exactly the moving build is mixed");
        assert!(
            s1.gemm_mixed > s0.gemm_mixed,
            "{xc}: no mixed GEMM ran on the device"
        );
        assert_eq!(
            s1.mixed_fallback_f64, s0.mixed_fallback_f64,
            "{xc}: mixed fell back"
        );
        let vmax = max_abs(&f64_host[1].1);
        for i in 0..3 {
            let de = (dev[i].0 - f64_host[i].0).abs();
            assert!(de < 1e-10, "{xc} build {i}: E_xc moved by {de:e}");
            for (a, b) in dev[i].1.iter().zip(again[i].1.iter()) {
                assert_eq!(a.to_bits(), b.to_bits(), "{xc} build {i}: not reproducible");
            }
        }
        // Builds 0 and 2 are f64 on the device: rounding-level agreement.
        for i in [0, 2] {
            let dv = max_abs(&(&dev[i].1 - &f64_host[i].1));
            assert!(
                dv < 1e-10 * vmax.max(1.0),
                "{xc} f64 build {i}: max|dV| {dv:e}"
            );
        }
        // Build 1 is mixed: a real, bounded change of the twin's size.
        let dv_dev = max_abs(&(&dev[1].1 - &f64_host[1].1));
        let dv_twin = max_abs(&(&twin[1].1 - &f64_host[1].1));
        eprintln!("{xc}: max|V| {vmax:.3e}, mixed max|dV| device {dv_dev:.3e}, twin {dv_twin:.3e}");
        assert!(
            dv_twin > 0.0 && dv_dev > 0.0,
            "{xc}: the mixed build equals f64"
        );
        assert!(dv_dev < 1e-5 * vmax, "{xc}: mixed max|dV| {dv_dev:e}");
        assert!(
            dv_dev < 10.0 * dv_twin && dv_twin < 10.0 * dv_dev,
            "{xc}: device {dv_dev:e} and twin {dv_twin:e} are not the same error class"
        );
    }
}

#[test]
fn a_mixed_device_scf_matches_the_f64_host_scf() {
    use ferric_core::parallel::ParallelContext;
    use ferric_integrals::basis_bridge::PreparedBasis;
    use ferric_integrals::operator::Operator;
    use ferric_scf::rhf::{solve_rhf, RhfConfig};
    use ferric_scf::screening::SchwarzBounds;
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let mol = water();
    let prep = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let cfg = RhfConfig {
        xc: Some("PBE".into()),
        density_conv: 1e-10,
        energy_conv: 1e-8,
        ..Default::default()
    };
    let run = |host: bool| {
        FORCE_HOST.store(host, Ordering::SeqCst);
        let r = solve_rhf(&ParallelContext::default(), &mol, &prep, op, &bounds, &cfg);
        FORCE_HOST.store(false, Ordering::SeqCst);
        r.unwrap()
    };
    let host = run(true);
    let (s0, m0) = (stats(), MIXED_VXC_BUILDS.load(Ordering::SeqCst));
    let dev = run(false);
    let (s1, m1) = (stats(), MIXED_VXC_BUILDS.load(Ordering::SeqCst));
    assert!(
        m1 > m0 && s1.gemm_mixed > s0.gemm_mixed,
        "no build ran mixed on the device"
    );
    assert!(dev.converged && dev.iterations <= host.iterations + 3);
    let de = (dev.energy - host.energy).abs();
    eprintln!(
        "water PBE: host {:.12} ({} it), device mixed {:.12} ({} it, {} mixed builds), |dE| {de:.1e}",
        host.energy,
        host.iterations,
        dev.energy,
        dev.iterations,
        m1 - m0
    );
    assert!(de < 1e-10, "|dE| = {de:e}");
}
