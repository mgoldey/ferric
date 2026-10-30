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
use ferric_dft::xc_batch::{SeamGuard, FORCE_HOST, POISON_XC_STAGING, XC_GEMM_LOOP};
use ferric_dft::xc_trait::{UksXcContribution, XcContribution};
use ndarray::Array2;

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
    let _seam = SeamGuard::set(&FORCE_HOST, host);
    run()
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

/// `run()` with the per-batch GEMM loop (`XC_GEMM_LOOP`) set for its duration.
fn looped<T>(on: bool, run: impl FnOnce() -> T) -> T {
    let _seam = SeamGuard::set(&XC_GEMM_LOOP, on);
    run()
}

/// The strided-batched launches over the zero-padded layout against the
/// per-batch loop at each batch's true `nact` (same device, same staging):
/// both are f64 cuBLAS products of the same numbers plus exact zeros, so they
/// differ only by summation order — rounding of the measured order (≤ 1e-15
/// relative, asserted at 1e-13) — and the batched result is reproducible.
#[test]
fn batched_device_gemms_match_the_per_batch_loop() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let bs = basis::bundled("cc-pvdz").unwrap();
    let (main, nlc) = grids();
    let close = |what: &str, (e_l, v_l): (f64, &Array2<f64>), (e_b, v_b): (f64, &Array2<f64>)| {
        let (de, dv) = ((e_b - e_l).abs(), max_abs(&(v_b - v_l)));
        eprintln!("{what}: batched vs loop |dE| {de:.2e} max|dV| {dv:.2e}");
        assert!(de < 1e-13 * e_l.abs().max(1.0), "{what}: |dE| = {de:e}");
        assert!(
            dv < 1e-13 * max_abs(v_l).max(1.0),
            "{what}: max|dV| = {dv:e}"
        );
    };
    // Closed-shell GGA: dense and factored densities (pass-1 forms) and pass 2.
    let ks = KsXc::new(&water(0, 1), &bs, "PBE", &main, &nlc).unwrap();
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
    for occ in [false, true] {
        let lp = on(false, || looped(true, || build(occ)));
        let b = on(false, || build(occ));
        let b2 = on(false, || build(occ));
        assert!(ks.xc_on_device(), "the grid was not uploaded");
        close(
            if occ { "PBE factored" } else { "PBE dense" },
            (lp.0, &lp.1),
            (b.0, &b.1),
        );
        assert!(
            b.0.to_bits() == b2.0.to_bits()
                && b.1
                    .iter()
                    .zip(b2.1.iter())
                    .all(|(x, y)| x.to_bits() == y.to_bits()),
            "batched build is not reproducible"
        );
    }
    // Open-shell meta-GGA: τ (four padded components) and two V matrices.
    let ks = KsXcUks::new(&water(1, 2), &bs, "R2SCAN", &main, &nlc).unwrap();
    let nbf = ks.xc_screening_stats().nbf;
    let (ca, cb) = (c_occ(nbf, 5, 11), c_occ(nbf, 4, 13));
    let (da, db) = (ca.dot(&ca.t()), cb.dot(&cb.t()));
    let build = || {
        let mut fa = Array2::<f64>::zeros((nbf, nbf));
        let mut fb = Array2::<f64>::zeros((nbf, nbf));
        let e = ks.add_xc_uks(&da, &db, &mut fa, &mut fb);
        (e, fa, fb)
    };
    let lp = on(false, || looped(true, build));
    let b = on(false, build);
    assert!(ks.xc_on_device(), "the open-shell grid was not uploaded");
    close("r2SCAN alpha", (lp.0, &lp.1), (b.0, &b.1));
    close("r2SCAN beta", (lp.0, &lp.2), (b.0, &b.2));
}

/// `run()` with the reused operand staging NaN-filled before every wave
/// (`POISON_XC_STAGING`) for its duration.
fn poisoned<T>(run: impl FnOnce() -> T) -> T {
    let _seam = SeamGuard::set(&POISON_XC_STAGING, true);
    run()
}

fn bits_equal(a: &Array2<f64>, b: &Array2<f64>) -> bool {
    a.iter()
        .zip(b.iter())
        .all(|(x, y)| x.to_bits() == y.to_bits())
}

/// Every padded element of a wave's operands (rows past `nact`, columns past
/// `nact` or `nb`, the gaps between τ's components) must be written as an
/// exact zero by the wave itself: the staging is reused, so a missing
/// zero-fill would read an earlier wave's numbers (or, on a first wave, the
/// allocation's zeros, which hid the defect). With the staging NaN-filled
/// before every wave, a padded operand element is NaN unless the fill zeroes
/// it, and NaN times the AO copy's zero padding is NaN in a real element, so
/// a missing zero-fill makes E or V NaN. Poisoned builds must equal clean
/// ones bit for bit (dense and factored GGA, open-shell meta-GGA).
/// MUTATION CHECK: deleting the zero-fills of `fill_amat_padded`, of the dense
/// branch of `gather_padded`, or of its factored branch (one at a time) makes
/// this test fail; the other tests in this file pass under each of them.
#[test]
fn nan_poisoned_staging_gives_the_clean_result_bit_for_bit() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let bs = basis::bundled("cc-pvdz").unwrap();
    let (main, nlc) = grids();
    let ks = KsXc::new(&water(0, 1), &bs, "PBE", &main, &nlc).unwrap();
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
    for occ in [false, true] {
        let clean = on(false, || build(occ));
        let dirty = on(false, || poisoned(|| build(occ)));
        assert!(ks.xc_on_device(), "the grid was not uploaded");
        assert!(
            dirty.0.is_finite() && dirty.1.iter().all(|v| v.is_finite()),
            "PBE occ={occ}: a padded staging element was not zero-filled (E {})",
            dirty.0
        );
        assert!(
            dirty.0.to_bits() == clean.0.to_bits() && bits_equal(&dirty.1, &clean.1),
            "PBE occ={occ}: the poisoned build differs from the clean one"
        );
    }
    let ks = KsXcUks::new(&water(1, 2), &bs, "R2SCAN", &main, &nlc).unwrap();
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
    for occ in [false, true] {
        let clean = on(false, || build(occ));
        let dirty = on(false, || poisoned(|| build(occ)));
        assert!(ks.xc_on_device(), "the open-shell grid was not uploaded");
        assert!(
            dirty.0.to_bits() == clean.0.to_bits()
                && bits_equal(&dirty.1, &clean.1)
                && bits_equal(&dirty.2, &clean.2),
            "r2SCAN occ={occ}: the poisoned build differs from the clean one (E {})",
            dirty.0
        );
    }
}
