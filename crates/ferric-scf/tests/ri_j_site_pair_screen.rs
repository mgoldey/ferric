#![cfg(feature = "gpu")]
//! Under `[gpu] mode = auto` the f64 RI-J tensor keeps every AO pair (the
//! device upload reads the full packed triangle) only on a site the device
//! gate takes J on. A pure functional's site never routes to the device, so
//! its f64 tensor (a geometry task, or an explicit `ri_j_precision = "f64"`)
//! keeps the Schwarz pair screen; a global hybrid's site at the same size
//! routes to the device and keeps every pair. No device is needed: the gate
//! reads the installed mode, not the device status.
//!
//! n-dodecane/def2-SVP is 298 basis functions, the f64 RI-J threshold
//! (`AUTO_MIN_NBF_RI_J_F64`), the smallest size `auto` takes J at.
use ferric_core::basis;
use ferric_core::gpu::{install, settings, GpuMode, GpuSettingsExplicit};
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::df_j::{ri_j_pair_screen_on_site_for_test, ri_j_site_for_functional};

/// `RhfConfig::default().integral_thresh`, the threshold the SCF screens with.
const PAIR_THRESH: f64 = 1e-12;

#[test]
fn auto_screens_a_pure_functionals_f64_tensor_and_not_a_hybrids() {
    install(GpuSettingsExplicit {
        mode: Some(GpuMode::Auto),
        ..Default::default()
    })
    .expect("auto installs with or without a device");
    assert_eq!(settings().mode, GpuMode::Auto);

    let mol = Molecule::load_xyz(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../testdata/molecules/alkane_12.xyz"
    ))
    .unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("def2-svp").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("def2-universal-jkfit").unwrap()).unwrap();
    let n = obs.nbasis();
    assert_eq!(n, ferric_core::gpu::gate::AUTO_MIN_NBF_RI_J_F64);

    let screen = |xc| {
        ri_j_pair_screen_on_site_for_test(
            Operator::coulomb(),
            &obs,
            &dfbs,
            PAIR_THRESH,
            ri_j_site_for_functional(n, Some(xc)),
        )
        .unwrap()
    };
    let pure = screen("PBE").expect("a pure functional's f64 tensor is pair-screened");
    assert!(pure.ncols() < pure.total_pairs(), "the screen drops pairs");
    assert!(
        screen("B3LYP").is_none(),
        "a hybrid's f64 tensor keeps every pair for the device"
    );
}
