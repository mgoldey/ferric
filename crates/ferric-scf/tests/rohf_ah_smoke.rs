//! Smoke tests for the augmented-Hessian Newton step (rohf_ah module).
//!
//! Three tests cover correctness:
//!   1. Stationary-point no-op: at converged OH/HF, AH-step is a no-op.
//!   2. Benign open-shell agreement: H₂ triplet/LDA AH must reach the
//!      same energy as the DIIS variant.
//!   3. Regression guard: pure DIIS (ah_trigger=0) still converges with
//!      the new code paths compiled in.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_integrals::oneelectron;
use ferric_scf::rhf::{build_jk, RhfConfig};
use ferric_scf::rohf::solve_rohf;
use ferric_scf::rohf_ah::{rohf_ah_step, RohfAhInputs};
use ferric_scf::rohf_newton::RohfNewtonInputs;
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;

#[test]
fn rohf_ah_step_at_stationary_point_is_noop() {
    let mol = Molecule::parse_xyz("2\nOH\nO 0 0 0\nH 0 0 1.10\n", 0, 2).unwrap();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();

    let cfg = RhfConfig {
        energy_conv: 1e-10,
        density_conv: 1e-8,
        max_iter: 200,
        ..Default::default()
    };
    let res = solve_rohf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
    assert!(res.converged);

    // Reconstruct per-spin Focks from converged densities.
    let n = prep.nbasis();
    let h = oneelectron::hcore(&prep);
    let d_a = res.density_alpha.clone();
    let d_b = res.density_beta.as_ref().unwrap().clone();
    let d_tot = &d_a + &d_b;
    let mut j = Array2::<f64>::zeros((n, n));
    let mut k_a = Array2::<f64>::zeros((n, n));
    let mut k_b = Array2::<f64>::zeros((n, n));
    let mut k_dum = Array2::<f64>::zeros((n, n));
    let mut j_dum = Array2::<f64>::zeros((n, n));
    build_jk(&ctx, &prep, &bounds, 1e-12, &d_tot, &mut j, &mut k_dum).unwrap();
    build_jk(&ctx, &prep, &bounds, 1e-12, &d_a, &mut j_dum, &mut k_a).unwrap();
    j_dum.fill(0.0);
    build_jk(&ctx, &prep, &bounds, 1e-12, &d_b, &mut j_dum, &mut k_b).unwrap();
    let f_a: Array2<f64> = &h + &j - &k_a;
    let f_b: Array2<f64> = &h + &j - &k_b;
    let c = &res.mos_alpha;
    let f_a_mo = c.t().dot(&f_a).dot(c);
    let f_b_mo = c.t().dot(&f_b).dot(c);
    let nelec = mol.nelec() as usize;
    let nocc_open: usize = mol.multiplicity - 1;
    let nocc_double: usize = (nelec - nocc_open) / 2;
    let inputs = RohfNewtonInputs {
        prep: &prep,
        bounds: &bounds,
        c,
        eps: &res.eps_alpha,
        f_a_mo: &f_a_mo,
        f_b_mo: &f_b_mo,
        nocc_double,
        nocc_open,
        k_mix_sr: 1.0,
        fxc: None,
        thresh: 1e-12,
        ooc_budget: ferric_core::memory::resolve_budget_bytes(None),
    };
    let ah_inputs = RohfAhInputs { base: &inputs };

    let (c_new, kmax) = rohf_ah_step(&ctx, &ah_inputs, /*max_step=*/0.2,
                                    /*davidson_conv=*/1e-7, /*davidson_max_vecs=*/50).unwrap();
    eprintln!("AH stationary-point kmax = {:.3e}", kmax);
    assert!(kmax < 1e-5, "AH at stationary point should be a no-op: kmax = {kmax:.3e}");

    let diff = (&c_new - c).iter().fold(0.0f64, |m, &v| m.max(v.abs()));
    assert!(diff < 1e-5);
}

#[test]
fn rohf_ah_h2_triplet_lda_matches_diis() {
    let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 2.0\n", 0, 3).unwrap();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();

    let cfg_diis = RhfConfig {
        xc: Some("LDA".into()),
        energy_conv: 1e-9,
        density_conv: 1e-7,
        max_iter: 200,
        ..Default::default()
    };
    let cfg_ah = RhfConfig {
        ah_trigger: 1e-2,
        ..cfg_diis.clone()
    };
    let r_diis = solve_rohf(&ctx, &mol, &prep, op, &bounds, &cfg_diis).unwrap();
    let r_ah = solve_rohf(&ctx, &mol, &prep, op, &bounds, &cfg_ah).unwrap();

    eprintln!("H₂ triplet/LDA  DIIS: E={:.10}, iters={}", r_diis.energy, r_diis.iterations);
    eprintln!("H₂ triplet/LDA  AH:   E={:.10}, iters={}", r_ah.energy, r_ah.iterations);
    assert!(r_diis.converged && r_ah.converged);
    assert!(
        (r_diis.energy - r_ah.energy).abs() < 1e-6,
        "AH must reach the same energy as DIIS on a benign case; ΔE = {:.3e}",
        (r_diis.energy - r_ah.energy).abs()
    );
}

/// Diagnostic — not a hard assertion. Reports how AH performs on the
/// known-hard doublet OH/LDA case where PCG plateaus. If this lands an
/// energy ≥1 mHa below the DIIS run *and* hits density_conv=1e-3, we can
/// un-ignore roks_grad_oh_ccpvdz_lda in a follow-up.
#[test]
fn rohf_ah_oh_lda_diagnostic() {
    let mol = Molecule::parse_xyz("2\nOH\nO 0 0 0\nH 0 0 0.97\n", 0, 2).unwrap();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();

    let cfg_diis = RhfConfig {
        xc: Some("LDA".into()),
        energy_conv: 1e-7,
        density_conv: 1e-3,
        max_iter: 200,
        level_shift: 0.2,
        ..Default::default()
    };
    let cfg_ah = RhfConfig {
        ah_trigger: 1e-2,
        ..cfg_diis.clone()
    };
    let r_diis = solve_rohf(&ctx, &mol, &prep, op, &bounds, &cfg_diis);
    let r_ah   = solve_rohf(&ctx, &mol, &prep, op, &bounds, &cfg_ah);

    let (e_diis, conv_diis, iters_diis) = match r_diis {
        Ok(r) => (r.energy, r.converged, r.iterations),
        Err(ferric_core::FerricError::ScfConvergence { last_energy, iterations }) =>
            (last_energy, false, iterations),
        Err(e) => panic!("Unexpected DIIS error: {e:?}"),
    };
    let (e_ah, conv_ah, iters_ah) = match r_ah {
        Ok(r) => (r.energy, r.converged, r.iterations),
        Err(ferric_core::FerricError::ScfConvergence { last_energy, iterations }) =>
            (last_energy, false, iterations),
        Err(e) => panic!("Unexpected AH error: {e:?}"),
    };
    eprintln!("OH/LDA  DIIS: E={:.10} conv={} iters={}", e_diis, conv_diis, iters_diis);
    eprintln!("OH/LDA  AH:   E={:.10} conv={} iters={}", e_ah, conv_ah, iters_ah);
    eprintln!("OH/LDA  ΔE (AH − DIIS) = {:.3e} Ha", e_ah - e_diis);
    // No assertion — informative only.
}

#[test]
fn rohf_ah_disabled_still_works() {
    // Pure DIIS path with ah_trigger=0 must still converge unchanged.
    let mol = Molecule::parse_xyz("2\nOH\nO 0 0 0\nH 0 0 1.10\n", 0, 2).unwrap();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();

    let cfg = RhfConfig {
        energy_conv: 1e-10,
        density_conv: 1e-8,
        max_iter: 200,
        ..Default::default()
    };
    let r = solve_rohf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
    assert!(r.converged);
    assert!((r.energy - (-75.3745595033)).abs() < 1e-5,
            "DIIS regression: E = {:.10}", r.energy);
}
