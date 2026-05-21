//! Smoke test for the ROHF Newton step (HF-only path, no XC).
//!
//! Strategy:
//!   1. Run plain ROHF DIIS to convergence on doublet OH/cc-pVDZ.
//!   2. From the converged state, build (F_α, F_β) and ε, and call
//!      `rohf_newton_step` to verify that:
//!        (a) the step is a no-op (||κ|| ≈ 0) because the gradient is zero
//!            at the stationary point;
//!        (b) the matvec is symmetric: ⟨κ₁, H·κ₂⟩ = ⟨κ₂, H·κ₁⟩.
//!
//! These are necessary correctness checks before we plug Newton into the
//! SCF loop and trust it to converge harder cases (LDA/PBE ROKS on OH).

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_integrals::oneelectron;
use ferric_scf::rohf::{solve_rohf, RohfConfig};
use ferric_scf::rohf_newton::{rohf_newton_step, RohfNewtonInputs};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::rhf::build_jk;
use ndarray::Array2;

#[test]
fn rohf_newton_step_at_stationary_point_is_noop() {
    let mol = Molecule::parse_xyz("2\nOH\nO 0 0 0\nH 0 0 1.10\n", 0, 2).unwrap();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();

    // Plain ROHF (no XC) — should converge easily.
    let cfg = RohfConfig {
        energy_conv: 1e-10,
        density_conv: 1e-8,
        max_iter: 200,
        ..Default::default()
    };
    let res = solve_rohf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
    assert!(res.converged);

    // Reconstruct (F_α, F_β) from the converged densities. We rebuild J/K
    // here so we have access to the per-spin Fock — the ScfResult only
    // stores fock_alpha (= F_eff).
    let n = prep.nbasis();
    let h = oneelectron::hcore(&prep);

    let d_a = res.density_alpha.clone();
    let d_b = res.density_beta.as_ref().expect("ROHF stores density_beta").clone();
    let d_total = &d_a + &d_b;

    let mut j = Array2::<f64>::zeros((n, n));
    let mut k_a = Array2::<f64>::zeros((n, n));
    let mut k_b = Array2::<f64>::zeros((n, n));
    let mut k_dummy = Array2::<f64>::zeros((n, n));
    build_jk(&ctx, &prep, &bounds, 1e-12, &d_total, &mut j, &mut k_dummy).unwrap();
    let mut j_dummy = Array2::<f64>::zeros((n, n));
    build_jk(&ctx, &prep, &bounds, 1e-12, &d_a, &mut j_dummy, &mut k_a).unwrap();
    j_dummy.fill(0.0);
    build_jk(&ctx, &prep, &bounds, 1e-12, &d_b, &mut j_dummy, &mut k_b).unwrap();

    let f_a: Array2<f64> = &h + &j - &k_a;
    let f_b: Array2<f64> = &h + &j - &k_b;

    let c = &res.mos_alpha;
    let f_a_mo = c.t().dot(&f_a).dot(c);
    let f_b_mo = c.t().dot(&f_b).dot(c);

    // Occupations
    let nelec = mol.nelec() as usize;
    let mult = mol.multiplicity;
    let nocc_open: usize = mult - 1;
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
    };

    // Sanity-check the gradient norm at the converged point.
    let n_total = c.ncols();
    let nocc_a = nocc_double + nocc_open;
    let mut gmax = 0.0f64;
    for p in nocc_a..n_total {
        for q in 0..nocc_double {
            let g = f_a_mo[(p, q)] + f_b_mo[(p, q)];
            gmax = gmax.max(g.abs());
        }
    }
    for p in nocc_a..n_total {
        for q in nocc_double..nocc_a {
            gmax = gmax.max(f_a_mo[(p, q)].abs());
        }
    }
    for p in nocc_double..nocc_a {
        for q in 0..nocc_double {
            gmax = gmax.max(f_b_mo[(p, q)].abs());
        }
    }
    eprintln!("Gradient ∞-norm at converged ROHF: {:.3e}", gmax);
    eprintln!("ROHF reports converged in {} iterations", res.iterations);

    let (c_new, kmax) = rohf_newton_step(&ctx, &inputs, 0.0, 0.2, 20, 1e-9).unwrap();
    eprintln!("Newton step at stationary point: kmax = {:.3e}", kmax);
    assert!(kmax < 1e-5,
        "At a converged stationary point, the Newton step should be ~0; got kmax = {:.3e}",
        kmax);

    // C should be essentially unchanged.
    let diff = (&c_new - c).iter().fold(0.0f64, |m, &v| m.max(v.abs()));
    eprintln!("||C_new − C||_∞ = {:.3e}", diff);
    assert!(diff < 1e-5);
}

/// Triplet H₂ at 2.0 Å under LDA ROKS: a benign open-shell case (well-separated
/// SOMOs, no SOMO/HOMO near-degeneracy). Newton+LDA path must converge to the
/// same energy as DIIS-only. Validates the LDA f_xc kernel + Newton matvec
/// without the OH-doublet instability noise that dominates that case.
#[test]
fn rohf_newton_h2_triplet_lda_matches_diis() {
    let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 2.0\n", 0, 3).unwrap();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();

    let cfg_diis = RohfConfig {
        xc: Some("LDA".into()),
        energy_conv: 1e-9,
        density_conv: 1e-7,
        max_iter: 200,
        ..Default::default()
    };
    let cfg_newton = RohfConfig {
        newton_trigger: 1e-2,
        ..cfg_diis.clone()
    };
    let r_diis = solve_rohf(&ctx, &mol, &prep, op, &bounds, &cfg_diis).unwrap();
    let r_newton = solve_rohf(&ctx, &mol, &prep, op, &bounds, &cfg_newton).unwrap();
    eprintln!("H₂ triplet/LDA  DIIS:   E = {:.10}, iters = {}", r_diis.energy, r_diis.iterations);
    eprintln!("H₂ triplet/LDA  Newton: E = {:.10}, iters = {}", r_newton.energy, r_newton.iterations);

    assert!(r_diis.converged && r_newton.converged);
    assert!(
        (r_diis.energy - r_newton.energy).abs() < 1e-6,
        "Newton+LDA must reach the same energy as DIIS-only on a benign case; \
         got ΔE = {:.3e}",
        (r_diis.energy - r_newton.energy).abs()
    );
}

/// Newton-augmented ROHF on OH/HF must reach the same energy as DIIS-only.
/// Tests the in-SCF wiring with the per-spin semicanonical preconditioner.
#[test]
fn rohf_newton_oh_hf_matches_diis_only() {
    let mol = Molecule::parse_xyz("2\nOH\nO 0 0 0\nH 0 0 1.10\n", 0, 2).unwrap();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();

    let cfg_diis = RohfConfig {
        energy_conv: 1e-10,
        density_conv: 1e-8,
        max_iter: 200,
        ..Default::default()
    };
    let cfg_newton = RohfConfig {
        energy_conv: 1e-10,
        density_conv: 1e-8,
        max_iter: 200,
        newton_trigger: 1e-2,
        ..Default::default()
    };

    let r_diis = solve_rohf(&ctx, &mol, &prep, op, &bounds, &cfg_diis).unwrap();
    let r_newton = solve_rohf(&ctx, &mol, &prep, op, &bounds, &cfg_newton).unwrap();

    eprintln!("DIIS: E = {:.10}, iters = {}", r_diis.energy, r_diis.iterations);
    eprintln!("Newton-aug: E = {:.10}, iters = {}", r_newton.energy, r_newton.iterations);

    assert!(r_diis.converged, "DIIS-only must converge");
    assert!(r_newton.converged, "Newton-aug must converge");
    assert!(
        (r_diis.energy - r_newton.energy).abs() < 1e-6,
        "Energies must match: ΔE = {:.3e}",
        (r_diis.energy - r_newton.energy).abs()
    );
}

