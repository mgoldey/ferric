#![cfg(all(feature = "gpu", feature = "test-seams"))]
//! The screened KS XC grid with its per-batch GEMMs on the device, against the
//! same grid forced onto the host (`xc_batch::FORCE_HOST`), through the public
//! `KsXc` / `KsXcUks` Fock-build entry points: dense and factored densities,
//! GGA and meta-GGA (τ), closed and open shell. The device products are cuBLAS
//! dgemm (not bit-identical to OpenBLAS), so E_xc and V_xc agree to rounding:
//! the bars below are ~10³ above the measured-order γ_k·|A||B| bound and far
//! below any physical effect. Two device builds are bit-identical. The host
//! staging of the waves is page-locked (no heap fallback, bytes held).
use ferric_core::basis;
use ferric_core::gpu::{install, probe, stats, GpuMode, GpuSettingsExplicit, GpuStatus};
use ferric_core::mol::Molecule;
use ferric_dft::grid::AtomicGridConfig;
use ferric_dft::ks::{KsXc, KsXcUks};
use ferric_dft::xc_batch::FORCE_HOST;
use ferric_dft::xc_trait::{UksXcContribution, XcContribution};
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
        install(GpuSettingsExplicit {
            mode: Some(GpuMode::On), // the XC grid uses the device only under `on`
            memory_gb: Some(1.0),
            ..Default::default()
        })
        .expect("install");
    });
    true
}

fn water(charge: i32, mult: usize) -> Molecule {
    Molecule::parse_xyz(
        "3\nH2O\nO 0 0 0\nH 0 0.7572 0.5868\nH 0 -0.7572 0.5868\n",
        charge,
        mult,
    )
    .unwrap()
}

fn grids() -> (AtomicGridConfig, AtomicGridConfig) {
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
    (main, nlc)
}

/// A deterministic `(nbf, nocc)` block with entries in `[-0.3, 0.3]`.
fn c_occ(nbf: usize, nocc: usize, seed: u64) -> Array2<f64> {
    let mut x = seed;
    Array2::from_shape_fn((nbf, nocc), |_| {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((x >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 0.6
    })
}

fn max_abs(a: &Array2<f64>) -> f64 {
    a.iter().fold(0.0_f64, |m, v| m.max(v.abs()))
}

fn assert_close(what: &str, (e_h, v_h): (f64, &Array2<f64>), (e_d, v_d): (f64, &Array2<f64>)) {
    let de = (e_d - e_h).abs();
    let dv = max_abs(&(v_d - v_h));
    eprintln!("{what}: E_host {e_h:.14} |dE| {de:.2e} max|dV| {dv:.2e}");
    assert!(de < 1e-10, "{what}: |dE| = {de:e}");
    assert!(
        dv < 1e-10 * max_abs(v_h).max(1.0),
        "{what}: max|dV| = {dv:e}"
    );
}

/// `run(host)` with `FORCE_HOST` set for its duration.
fn on<T>(host: bool, run: impl FnOnce() -> T) -> T {
    FORCE_HOST.store(host, Ordering::SeqCst);
    let out = run();
    FORCE_HOST.store(false, Ordering::SeqCst);
    out
}

#[test]
fn closed_shell_gga_on_the_device_matches_the_host() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let mol = water(0, 1);
    let bs = basis::bundled("cc-pvdz").unwrap();
    let (main, nlc) = grids();
    let ks = KsXc::new(&mol, &bs, "PBE", &main, &nlc).unwrap();
    let nbf = ks.xc_screening_stats().nbf;
    let c = c_occ(nbf, 5, 7);
    let d = 2.0 * c.dot(&c.t());
    let build = |occ: bool| {
        let mut f = Array2::<f64>::zeros((nbf, nbf));
        let e = if occ {
            ks.add_xc_occ(&d, &c, &mut f)
        } else {
            ks.add_xc(&d, &mut f)
        };
        (e, f)
    };
    let s0 = stats();
    let host = on(true, || build(false));
    let host_occ = on(true, || build(true));
    let s1 = stats();
    assert_eq!(
        s1.xc_device_passes, s0.xc_device_passes,
        "a forced-host build ran on the device"
    );
    let dev = on(false, || build(false));
    let s2 = stats();
    assert!(
        ks.xc_on_device(),
        "the grid was not uploaded (declined: {})",
        s2.xc_declined - s1.xc_declined
    );
    assert_eq!(
        s2.xc_device_passes - s1.xc_device_passes,
        2,
        "density + potential pass"
    );
    // The host stage really is page-locked: no buffer fell back to the heap,
    // and the grid holds page-locked bytes (operands + results, f64).
    assert_eq!(
        s2.xc_pinned_fallback - s1.xc_pinned_fallback,
        0,
        "a page-locked staging buffer fell back to heap memory"
    );
    let pinned = ks.xc_pinned_staging_bytes();
    eprintln!("page-locked XC staging: {pinned} bytes");
    assert!(pinned > 0, "the XC host stage is not page-locked");
    let dev_occ = on(false, || build(true));
    let dev_again = on(false, || build(false));
    assert_close("PBE dense", (host.0, &host.1), (dev.0, &dev.1));
    assert_close(
        "PBE factored",
        (host_occ.0, &host_occ.1),
        (dev_occ.0, &dev_occ.1),
    );
    assert_eq!(
        dev.0.to_bits(),
        dev_again.0.to_bits(),
        "device E_xc is not reproducible"
    );
    assert!(
        dev.1
            .iter()
            .zip(dev_again.1.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "device V_xc is not reproducible"
    );
    let sym = max_abs(&(&dev.1 - &dev.1.t()));
    assert_eq!(sym, 0.0, "device V_xc is not exactly symmetric");
}

#[test]
fn open_shell_meta_gga_on_the_device_matches_the_host() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let mol = water(1, 2);
    let bs = basis::bundled("cc-pvdz").unwrap();
    let (main, nlc) = grids();
    let ks = KsXcUks::new(&mol, &bs, "R2SCAN", &main, &nlc).unwrap();
    let nbf = ks.xc_screening_stats().nbf;
    let (ca, cb) = (c_occ(nbf, 5, 11), c_occ(nbf, 4, 13));
    let (da, db) = (ca.dot(&ca.t()), cb.dot(&cb.t()));
    let build = |occ: bool| {
        let mut fa = Array2::<f64>::zeros((nbf, nbf));
        let mut fb = Array2::<f64>::zeros((nbf, nbf));
        let e = if occ {
            ks.add_xc_uks_occ((&da, &db), (Some(&ca), Some(&cb)), &mut fa, &mut fb)
        } else {
            ks.add_xc_uks(&da, &db, &mut fa, &mut fb)
        };
        (e, fa, fb)
    };
    let host = on(true, || build(false));
    let host_occ = on(true, || build(true));
    let s1 = stats();
    let dev = on(false, || build(false));
    let dev_occ = on(false, || build(true));
    let s2 = stats();
    assert!(
        ks.xc_on_device(),
        "the grid was not uploaded (declined: {})",
        s2.xc_declined - s1.xc_declined
    );
    assert_eq!(
        s2.xc_device_passes - s1.xc_device_passes,
        4,
        "two builds of two passes"
    );
    assert_close("r2SCAN alpha", (host.0, &host.1), (dev.0, &dev.1));
    assert_close("r2SCAN beta", (host.0, &host.2), (dev.0, &dev.2));
    assert_close(
        "r2SCAN alpha factored",
        (host_occ.0, &host_occ.1),
        (dev_occ.0, &dev_occ.1),
    );
    assert_close(
        "r2SCAN beta factored",
        (host_occ.0, &host_occ.2),
        (dev_occ.0, &dev_occ.2),
    );
}
