use ferric_dft::libxc::{xc_def_from_name, XcFunctional};

#[test]
fn xc_def_lda_resolves() {
    let def = xc_def_from_name("LDA").unwrap();
    assert!(def.cam.is_none(), "LDA should have no CAM coefficients");
    assert!(def.vv10.is_none(), "LDA should have no VV10");
    assert_eq!(def.funcs.len(), 2, "LDA = LDA_X + LDA_C_VWN");
}

#[test]
fn xc_def_pbe_resolves() {
    let def = xc_def_from_name("PBE").unwrap();
    assert!(def.cam.is_none());
    assert!(def.vv10.is_none());
    assert_eq!(def.funcs.len(), 2);
}

#[test]
fn xc_def_wb97xv_has_cam_and_vv10() {
    let def = xc_def_from_name("wB97X-V").unwrap();
    let cam = def.cam.expect("wB97X-V should have CAM coefficients");
    assert!((cam.omega - 0.3).abs() < 1e-6, "ω = {}", cam.omega);
    // libxc convention: alpha=1.0, beta=-0.833 → c_SR = alpha = 1.0, c_LR = alpha+beta = 0.167
    assert!((cam.c_sr - 1.0).abs() < 1e-6,   "c_SR = {}", cam.c_sr);
    assert!((cam.c_lr - 0.167).abs() < 5e-3, "c_LR = {}", cam.c_lr);
    let vv10 = def.vv10.expect("wB97X-V should have VV10");
    assert!((vv10.c - 0.01).abs() < 1e-6, "VV10 C = {}", vv10.c);
    assert!((vv10.b - 6.0).abs() < 1e-6,  "VV10 b = {}", vv10.b);
}

#[test]
fn lda_x_matches_slater_exchange_formula() {
    // LDA_X (Slater exchange): ε_x(ρ) = -3/(4π) · (3π² ρ)^(1/3)
    let func = XcFunctional::new("LDA_X", 1).unwrap();
    let rho = [1.0_f64, 0.5, 0.1];
    let mut exc = [0.0_f64; 3];
    let mut vrho = [0.0_f64; 3];
    func.eval_lda_unpolarized(&rho, &mut exc, &mut vrho);
    for i in 0..3 {
        let r = rho[i];
        let expected = -3.0 / (4.0 * std::f64::consts::PI)
            * (3.0 * std::f64::consts::PI.powi(2) * r).powf(1.0 / 3.0);
        assert!((exc[i] - expected).abs() < 1e-10,
                "ρ={r}: libxc ε_x = {}, expected {expected}", exc[i]);
    }
}

#[test]
fn pbe_x_evaluates_without_panic() {
    // PBE_X (Perdew-Burke-Ernzerhof exchange). At ρ → 0 with finite σ, the
    // functional may produce sentinel values; we test at well-conditioned points.
    let func = XcFunctional::new("GGA_X_PBE", 1).unwrap();
    let rho   = [1.0_f64, 0.5, 0.1, 0.01];
    let sigma = [0.5_f64, 0.1, 0.01, 0.001];
    let n = rho.len();
    let mut exc    = vec![0.0_f64; n];
    let mut vrho   = vec![0.0_f64; n];
    let mut vsigma = vec![0.0_f64; n];
    func.eval_gga_unpolarized(&rho, &sigma, &mut exc, &mut vrho, &mut vsigma);

    // PBE exchange must reduce to LDA exchange in the σ → 0 limit (uniform density).
    // For ρ = 1, σ ≈ 0: ε_x(PBE) ≈ ε_x(LDA) = -3/(4π) · (3π² · 1)^(1/3) ≈ -0.7385876
    let lda_func = XcFunctional::new("LDA_X", 1).unwrap();
    let mut lda_exc  = vec![0.0_f64; n];
    let mut lda_vrho = vec![0.0_f64; n];
    lda_func.eval_lda_unpolarized(&rho, &mut lda_exc, &mut lda_vrho);

    // At small σ (well-conditioned), PBE ε_x should be within a few % of LDA.
    // This is a sanity check that the FFI signatures and array shapes are right.
    for i in 0..n {
        assert!(exc[i] < 0.0, "ε_x(PBE) should be negative at ρ={}", rho[i]);
        // Crude bound: PBE enhancement factor F(s) for small s is ~1.0..1.2
        let ratio = exc[i] / lda_exc[i];
        assert!(ratio > 0.95 && ratio < 1.5,
                "PBE/LDA ratio out of expected range at ρ={}: ratio = {}", rho[i], ratio);
    }
}
