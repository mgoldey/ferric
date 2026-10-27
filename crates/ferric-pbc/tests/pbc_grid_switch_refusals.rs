//! Per-commit refusal pins for the periodic XC grid neighbour switch
//! (issue #495); the SCF-heavy FD checks are `validation_pbc_grid_switch_paths.rs`.

const CAM: &str = "HYB_GGA_XC_CAM_B3LYP";

/// The Gamma ROKS range-separated STRESS has no library entry: the request
/// gate refuses it by name before any SCF (the UKS one and the ROKS force are
/// wired, #498). Pinned so the bindings cannot report it before it exists.
#[test]
fn gamma_roks_rsh_stress_stays_refused() {
    use ferric_pbc::kxcgrad::Deriv;
    ferric_pbc::dft::check_gamma_uks_request(CAM, &[Deriv::Stress]).unwrap();
    ferric_pbc::dft::check_gamma_roks_request(CAM, &[Deriv::Force]).unwrap();
    let e = ferric_pbc::dft::check_gamma_roks_request(CAM, &[Deriv::Stress])
        .expect_err("Gamma ROKS RSH stress must be refused");
    assert!(e.to_string().contains("stress has no library entry"), "{e}");
}
