#![cfg(all(feature = "gpu", feature = "test-seams"))]
//! With the feature built and `[gpu] mode` at its default (off), the device RI-J
//! must be invisible: every counter unchanged, J bit-identical to the forced-host
//! build (the code path of a build without the feature). The same holds with the
//! unshipped mixed kernel `dfj-pack` allowed through the settings-override seam:
//! mode off wins before any precision decision.
use ferric_core::basis;
use ferric_core::gpu::config::{GpuSettings, GpuSettingsExplicit};
use ferric_core::gpu::{settings, stats, GpuMode, MixedKernel, MixedKernelSet, Precision};
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::df_j::DfJ;
use ferric_scf::fock::JBuilder;
use ndarray::Array2;
use std::sync::atomic::Ordering;

static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn mode_off_leaves_every_counter_alone_and_j_is_the_host_j() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    if settings().mode != GpuMode::Off {
        eprintln!("skipping: FERRIC_GPU is set in this environment");
        return;
    }
    let mol = Molecule::parse_xyz("3\nH2O\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.26\n", 0, 1).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let aux = PreparedBasis::new(&mol, &basis::bundled("def2-universal-jkfit").unwrap()).unwrap();
    let n = obs.nbasis();
    let d = Array2::from_shape_fn((n, n), |(i, j)| 0.01 * ((i * 7 + j * 3) % 11) as f64);
    let mut dfj = DfJ::new(Operator::coulomb(), &obs, &aux, usize::MAX).unwrap();
    let before = stats();
    let (mut j1, mut j2) = (Array2::zeros((n, n)), Array2::zeros((n, n)));
    dfj.build(&d, &mut j1).unwrap();
    dfj.build(&d, &mut j2).unwrap();
    assert_eq!(stats(), before, "mode off must not move any GPU counter");
    assert!(!dfj.device_resident_for_test());
    let bits = |m: &Array2<f64>| m.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&j1), bits(&j2));
    ferric_scf::df_j_gpu::FORCE_HOST.store(true, Ordering::SeqCst);
    let mut j3 = Array2::zeros((n, n));
    dfj.build(&d, &mut j3).unwrap();
    ferric_scf::df_j_gpu::FORCE_HOST.store(false, Ordering::SeqCst);
    assert_eq!(bits(&j1), bits(&j3));
}

#[test]
fn mode_off_with_dfj_pack_allowed_is_still_the_host_j_and_moves_no_counter() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    if settings().mode != GpuMode::Off {
        eprintln!("skipping: FERRIC_GPU is set in this environment");
        return;
    }
    let mol = Molecule::parse_xyz("3\nH2O\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.26\n", 0, 1).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let aux = PreparedBasis::new(&mol, &basis::bundled("def2-universal-jkfit").unwrap()).unwrap();
    let n = obs.nbasis();
    let d = Array2::from_shape_fn((n, n), |(i, j)| 0.01 * ((i * 7 + j * 3) % 11) as f64);
    let bits = |m: &Array2<f64>| m.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    let mut dfj = DfJ::new(Operator::coulomb(), &obs, &aux, usize::MAX).unwrap();
    let mut reference = Array2::zeros((n, n));
    dfj.build(&d, &mut reference).unwrap();

    let shipped = MixedKernelSet::SHIPPED.with(MixedKernel::DfjPack);
    let allowed = GpuSettings::resolve_with_default(
        GpuSettingsExplicit {
            mode: Some(GpuMode::Auto),
            precision: Some(Precision::Mixed),
            mixed_kernels: Some(shipped),
            ..Default::default()
        },
        |_| None,
        Precision::F64,
        shipped,
    )
    .unwrap()
    .0;
    *ferric_scf::df_k_gpu::MIXED_SETTINGS_OVERRIDE
        .lock()
        .unwrap() = Some(allowed);
    let before = stats();
    let mut j = Array2::zeros((n, n));
    dfj.build(&d, &mut j).unwrap();
    *ferric_scf::df_k_gpu::MIXED_SETTINGS_OVERRIDE
        .lock()
        .unwrap() = None;
    assert_eq!(stats(), before, "mode off must not move any GPU counter");
    assert!(!dfj.device_resident_for_test());
    assert_eq!(bits(&j), bits(&reference));
}
