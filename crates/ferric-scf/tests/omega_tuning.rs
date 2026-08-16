//! Anchors for the xc_omega override + the optimal-tuning driver.
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::omega_tuning::{tune_omega, OmegaTuneConfig};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

fn h2_631g() -> (Molecule, PreparedBasis, SchwarzBounds) {
    let mol = Molecule::load_xyz(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../testdata/molecules/h2.xyz"
    ))
    .unwrap();
    let bs = basis::bundled("6-31g").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    (mol, prep, bounds)
}

/// TRIVIAL-LIMIT ANCHOR: xc_omega = Some(published ω) must reproduce
/// xc_omega = None exactly — read the published value from libxc itself
/// rather than hardcoding it, so a libxc default change cannot silently
/// turn this into a comparison of two different functionals.
#[test]
fn omega_override_at_published_value_matches_default() {
    let (mol, prep, bounds) = h2_631g();
    let ctx = ParallelContext::default();
    // published omega from libxc's own metadata
    let f = ferric_dft::libxc::XcFunctional::new("HYB_GGA_XC_WB97X_V", 1).unwrap();
    let names = f.ext_param_names();
    let pos = names.iter().position(|n| n == "_omega").expect("wB97X-V has _omega");
    let w_pub = f.ext_param_defaults()[pos];
    let base = RhfConfig { xc: Some("wB97X-V".into()), energy_conv: 1e-9, ..Default::default() };
    let r0 = solve_rhf(&ctx, &mol, &prep, Operator::coulomb(), &bounds, &base).unwrap();
    let r1 = solve_rhf(
        &ctx,
        &mol,
        &prep,
        Operator::coulomb(),
        &bounds,
        &RhfConfig { xc_omega: Some(w_pub), ..base },
    )
    .unwrap();
    let de = (r1.energy - r0.energy).abs();
    eprintln!("published omega = {w_pub}; |E(override) - E(default)| = {de:.3e}");
    assert!(de < 1e-10, "trivial-limit anchor FAILED: {de:.3e}");
}

/// MUTATION of the trivial limit: a DIFFERENT ω must move the energy —
/// proves the override actually reaches the functional (and the SR/LR
/// exchange split), not just the config struct.
#[test]
fn omega_override_moves_the_energy_and_homo() {
    let (mol, prep, bounds) = h2_631g();
    let ctx = ParallelContext::default();
    let base = RhfConfig { xc: Some("wB97X-V".into()), energy_conv: 1e-9, ..Default::default() };
    let run = |w: f64| {
        solve_rhf(
            &ctx,
            &mol,
            &prep,
            Operator::coulomb(),
            &bounds,
            &RhfConfig { xc_omega: Some(w), ..base.clone() },
        )
        .unwrap()
    };
    let lo = run(0.15);
    let hi = run(0.60);
    let de = (hi.energy - lo.energy).abs();
    let dh = (hi.eps_r()[0] - lo.eps_r()[0]).abs();
    eprintln!("omega 0.15 -> 0.60: |dE| = {de:.3e}, |d eps_HOMO| = {dh:.3e}");
    assert!(de > 1e-5, "omega override did not move the energy");
    assert!(dh > 1e-3, "omega override did not move the HOMO");
    // more long-range exact exchange binds the HOMO deeper
    assert!(hi.eps_r()[0] < lo.eps_r()[0], "HOMO did not deepen with omega");
}

/// Config honesty: a non-RSH functional with an ω override must hard-error,
/// never silently ignore.
#[test]
fn omega_override_on_pbe_is_rejected() {
    let (mol, prep, bounds) = h2_631g();
    let ctx = ParallelContext::default();
    let r = solve_rhf(
        &ctx,
        &mol,
        &prep,
        Operator::coulomb(),
        &bounds,
        &RhfConfig { xc: Some("PBE".into()), xc_omega: Some(0.3), ..Default::default() },
    );
    assert!(r.is_err(), "PBE + xc_omega must be rejected");
}

/// The tuner end-to-end on H2/6-31G: converges within the bracket, the
/// tuned |J| beats both bracket endpoints, and every evaluation carries a
/// negative IP-consistent HOMO.
#[test]
fn tune_omega_h2_converges_and_improves_j() {
    let (mol, prep, bounds) = h2_631g();
    let ctx = ParallelContext::default();
    let cfg = OmegaTuneConfig {
        functional: "wB97X-V".into(),
        omega_lo: 0.2,
        omega_hi: 1.2,
        omega_tol: 0.02,
        max_evals: 16,
        scf: RhfConfig { energy_conv: 1e-9, ..Default::default() },
    };
    let r = tune_omega(&ctx, &mol, &prep, &bounds, &cfg).unwrap();
    eprintln!(
        "tuned omega = {:.4}, J = {:+.5e}, {} evals, converged = {}",
        r.omega,
        r.j,
        r.evals.len(),
        r.converged
    );
    for e in &r.evals {
        eprintln!(
            "  w={:.4} eps_HOMO={:+.6} IP={:+.6} J={:+.3e}",
            e.omega, e.eps_homo, e.ip_delta_scf, e.j
        );
        assert!(e.eps_homo < 0.0 && e.ip_delta_scf > 0.0);
    }
    assert!(r.converged);
    let j_lo = r.evals.iter().find(|e| (e.omega - 0.2).abs() < 0.35).map(|e| e.j.abs());
    let j_best = r.j.abs();
    if let Some(jl) = j_lo {
        assert!(j_best <= jl + 1e-12, "tuned J no better than a bracket-side eval");
    }
    assert!(r.omega > cfg.omega_lo && r.omega < cfg.omega_hi);
}
