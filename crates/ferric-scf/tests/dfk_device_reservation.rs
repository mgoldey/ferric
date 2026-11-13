#![cfg(feature = "gpu")]
//! The RI-J upload leaves room in the device pool for the same SCF's DF-K
//! (`DfJ::reserve_device_for_k`, sized by `DfK::device_footprint_bytes`).
//! Under `[gpu] mode = auto` a DF-K below its size gate stays on the CPU, so
//! it reserves nothing, even when its dressed tensor is the in-core square
//! band the device path would read. No device is needed: the gate reads the
//! installed mode, not the device status.
use ferric_core::basis;
use ferric_core::gpu::{install, settings, GpuMode, GpuSettingsExplicit};
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::df_k::{set_dfk_square_for_test, DfK};

#[test]
fn a_cpu_routed_dfk_reserves_no_device_memory() {
    install(GpuSettingsExplicit {
        mode: Some(GpuMode::Auto),
        ..Default::default()
    })
    .expect("auto installs with or without a device");
    assert_eq!(settings().mode, GpuMode::Auto);

    let mol = Molecule::load_xyz(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../testdata/molecules/water.xyz"
    ))
    .unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("def2-svp").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("def2-universal-jkfit").unwrap()).unwrap();
    assert!(obs.nbasis() < ferric_core::gpu::gate::AUTO_MIN_NBF_DF_K);

    // The square in-core band: the layout the device path uploads.
    set_dfk_square_for_test(true);
    let k = DfK::new(Operator::coulomb(), &obs, &dfbs, usize::MAX);
    set_dfk_square_for_test(false);
    let k = k.unwrap();
    assert!(
        k.dressed_incore_flat_for_test().is_some(),
        "precondition: the dressed tensor is the in-core square band"
    );
    assert_eq!(
        k.device_footprint_bytes(),
        0,
        "a DF-K that auto keeps on the CPU reserved device memory"
    );
}
