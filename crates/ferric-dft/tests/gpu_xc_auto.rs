#![cfg(feature = "gpu")]
//! The KS XC grid uses the device only under an explicit `[gpu] mode = "on"`
//! (or a preset that sets it): under `auto` it stays on the CPU, because the
//! device path was measured slower than the CPU on a GTX 1080 (see
//! `ferric_dft::xc_batch::xc_grid_routes_to_device`). Its own binary: it
//! installs `mode = "auto"` for the process.
//!
//! * Config: the resolved mode of every way to ask for a device routes the XC
//!   grid as documented (no device needed).
//! * Behaviour (device needed, else skipped): with `auto` installed and a
//!   ready device, a Fock build makes no device XC pass.
use ferric_core::basis;
use ferric_core::gpu::{
    install, probe, stats, GpuMode, GpuPreset, GpuSettings, GpuSettingsExplicit, GpuStatus,
};
use ferric_core::mol::Molecule;
use ferric_dft::grid::AtomicGridConfig;
use ferric_dft::ks::KsXc;
use ferric_dft::xc_batch::xc_grid_routes_to_device;
use ferric_dft::xc_trait::XcContribution;
use ndarray::Array2;

fn resolved_mode(env: &[(&str, &str)], preset: Option<GpuPreset>) -> GpuMode {
    let get = |k: &str| {
        env.iter()
            .find(|(n, _)| *n == k)
            .map(|(_, v)| v.to_string())
    };
    let explicit = GpuSettingsExplicit {
        preset,
        ..Default::default()
    };
    GpuSettings::resolve(explicit, get).unwrap().0.mode
}

#[test]
fn only_an_explicit_on_routes_the_xc_grid_to_the_device() {
    for (env, preset, want) in [
        (vec![], None, false),
        (vec![("FERRIC_GPU", "off")], None, false),
        (vec![("FERRIC_GPU", "auto")], None, false),
        (vec![], Some(GpuPreset::Auto), false),
        (vec![], Some(GpuPreset::AutoMixed), false),
        (vec![("FERRIC_GPU", "on")], None, true),
        (vec![], Some(GpuPreset::On), true),
        (vec![], Some(GpuPreset::Mixed), true),
    ] {
        let mode = resolved_mode(&env, preset);
        assert_eq!(
            xc_grid_routes_to_device(mode),
            want,
            "env {env:?}, preset {preset:?} -> mode {mode:?}"
        );
    }
}

#[test]
fn under_auto_a_fock_build_makes_no_device_xc_pass() {
    if !matches!(probe(0), GpuStatus::Ready(_)) {
        eprintln!("skipping the behaviour half: no CUDA device");
        return;
    }
    install(GpuSettingsExplicit {
        mode: Some(GpuMode::Auto),
        memory_gb: Some(1.0),
        ..Default::default()
    })
    .expect("install");
    let mol = Molecule::parse_xyz(
        "3\nH2O\nO 0 0 0\nH 0 0.7572 0.5868\nH 0 -0.7572 0.5868\n",
        0,
        1,
    )
    .unwrap();
    let grid = AtomicGridConfig {
        n_radial: 30,
        n_angular: 50,
        ..Default::default()
    };
    let ks = KsXc::new(
        &mol,
        &basis::bundled("cc-pvdz").unwrap(),
        "PBE",
        &grid,
        &AtomicGridConfig::default_nlc(),
    )
    .unwrap();
    let nbf = ks.xc_screening_stats().nbf;
    let d = Array2::from_shape_fn((nbf, nbf), |(i, j)| 0.01 * (((i + j) % 5) as f64));
    let mut f = Array2::<f64>::zeros((nbf, nbf));
    let s0 = stats();
    ks.add_xc(&d, &mut f);
    let s1 = stats();
    assert_eq!(
        s1.xc_device_passes, s0.xc_device_passes,
        "mode = auto ran the XC grid on the device"
    );
}
