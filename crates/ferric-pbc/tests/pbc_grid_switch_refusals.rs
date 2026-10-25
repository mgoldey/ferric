//! Per-commit refusal pins for the periodic XC grid neighbour switch
//! (issue #495); the SCF-heavy FD checks are `validation_pbc_grid_switch_paths.rs`.

const CAM: &str = "HYB_GGA_XC_CAM_B3LYP";

/// The open-shell range-separated STRESS does not exist (switch or not): the
/// request gates refuse it by name before any SCF. Pinned so a later switch
/// wiring cannot silently report a stress without its range-separated term.
#[test]
fn open_shell_rsh_stress_stays_refused() {
    use ferric_pbc::kxcgrad::Deriv;
    let e = ferric_pbc::dft::check_gamma_uks_request(CAM, &[Deriv::Stress], false)
        .expect_err("Gamma UKS RSH stress must be refused");
    assert!(e.to_string().contains("stress"), "{e}");
    let e = ferric_pbc::dft::check_gamma_roks_request(CAM, true)
        .expect_err("Gamma ROKS RSH derivatives must be refused in the bindings");
    assert!(e.to_string().contains("range-separated"), "{e}");
}
