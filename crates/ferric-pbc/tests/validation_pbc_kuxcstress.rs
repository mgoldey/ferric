//! Weekly half of `pbc_kuxcstress.rs` (k-point open-shell UKS stress): the
//! tests whose per-commit CI time was >= 120 s (runs 40957025137 /
//! 40953725119, mean 142-195 s each), moved here so the per-commit shards
//! stay short. Every test is `#[ignore = "validation: ..."]`. Hypotheses and
//! the anchor table are in the doc of `pbc_kuxcstress.rs`; the helpers are
//! `kuxcstress_support/mod.rs`. Per-commit canaries that remain:
//! `refusals_and_the_per_spin_fock_guard_canary`,
//! `a_equal_one_with_zero_xc_is_the_uhf_stress_on_the_one_point_mesh`,
//! `xc_spin_swap_mutant_is_caught_on_the_one_point_mesh`, and the 1x1x1 vs
//! Gamma anchor `one_point_mesh_matches_the_gamma_uks_stress_per_part`.

mod common;
#[path = "kuxcstress_support/mod.rs"]
mod support;

use support::*;

// ------------------------------------------------------------------ (1)

/// (1) WIRING ANCHOR, per part. a = 1 with a zero XC IS the k-point UHF stress:
/// same `assemble(Unrestricted)`, but the KS entry rebuilds F_sigma itself
/// (guarded) and takes its exxdiv from the config. Artifact hypothesis: a leaked
/// XC term, a K scaled by anything but 1, a swapped spin Fock would show as
/// nonzero; physics predicts 0 to roundoff. BLIND to every a-scaling defect
/// (ExchUnscaled/ExchScaledTwice are identities at a = 1: asserted EXACTLY 0).
#[test]
#[ignore = "validation: k-UKS stress a = 1 zero-XC == UHF stress per part, H3 1x1x3 dense ewald; CI ~188 s"]
fn a_equal_one_with_zero_xc_is_the_uhf_stress_per_part() {
    a_equal_one_case(Jk::Dense, ExxDiv::Ewald);
}

/// Typed refusals (exact wording) and the per-spin Fock guard from every side.
#[test]
#[ignore = "validation: k-UKS stress refusals and per-spin Fock guard, full sweep; CI ~195 s"]
fn refusals_and_the_per_spin_fock_guard() {
    let fx = fixture(&H3, G111, "PBE0", Jk::Dense, ExxDiv::Ewald);
    let ok_cfg = scfg(Some(ExxDiv::Ewald), None, None, None);
    let run = |f: &str, s: &KUScfResult, cfg: &KUksStressConfig| stress_with(&fx, f, s, cfg);
    for (name, key) in [
        // A range-separated name is differentiated since #492 (on J/K that
        // carry its range separation; `pbc_kuks_rsh_stress.rs`): on these
        // bare kernels the shared input check refuses it.
        (
            "HSE06",
            "but the dense kernels carry the bare Coulomb exchange; build the J/K with KDenseAftConfig::rsh set",
        ),
        // SCAN is covered (`pbc_kuxc_mgga.rs`): a PBE0 state handed in as
        // SCAN is refused by the Fock guard.
        ("SCAN", "differs from the SCF's stored fock_alpha"),
        ("wB97X-V", "VV10"),
        ("B2PLYP", "double hybrid"),
    ] {
        let e = run(name, &fx.res.scf, &ok_cfg)
            .err()
            .unwrap_or_else(|| panic!("{name} accepted"))
            .to_string();
        eprintln!("{name}: {e}");
        assert!(e.contains(key), "{name}: {e}");
    }
    let e = run("PBE0", &fx.res.scf, &scfg(None, None, None, None))
        .expect_err("no exxdiv")
        .to_string();
    assert!(
        e.contains("kpoint_uks_stress: a hybrid functional (exact exchange 0.25) needs KRksStressConfig::exxdiv"),
        "{e}"
    );
    let ok = run("PBE0", &fx.res.scf, &ok_cfg).unwrap();
    assert!(ok.hf.fock_mismatch < 1e-12, "{:e}", ok.hf.fock_mismatch);
    assert!(ok.hf.commutator < UKS_COMMUTATOR_TOL / 100.0);
    let pbe = fixture(&H3, G111, "PBE", Jk::Dense, ExxDiv::Ewald);
    let guard_msg = "kpoint_uks_stress: the Fock matrix rebuilt";
    let cases: Vec<(&str, Result<KUksStress, ferric_core::FerricError>)> = vec![
        (
            "exxdiv None on an Ewald SCF",
            run(
                "PBE0",
                &fx.res.scf,
                &scfg(Some(ExxDiv::None), None, None, None),
            ),
        ),
        ("B3LYP on a PBE0 SCF", run("B3LYP", &fx.res.scf, &ok_cfg)),
        ("PBE stress on a PBE0 SCF", run("PBE", &fx.res.scf, &ok_cfg)),
        (
            "PBE0 stress on a PBE SCF",
            run("PBE0", &pbe.res.scf, &ok_cfg),
        ),
    ];
    for (n, r) in cases {
        let e = r
            .err()
            .unwrap_or_else(|| panic!("{n} accepted"))
            .to_string();
        eprintln!("{n}: {e}");
        assert!(e.contains(guard_msg), "{n}: {e}");
    }
    // Tampers of each spin's Fock are refused, naming the spin.
    let mut t = fx.res.scf.clone();
    t.fock_alpha[0][(0, 1)] += Complex64::new(1e-6, 0.0);
    let e = run("PBE0", &t, &ok_cfg)
        .expect_err("alpha tamper")
        .to_string();
    assert!(e.contains(guard_msg) && e.contains("fock_alpha"), "{e}");
    let mut t = fx.res.scf.clone();
    t.fock_beta[0][(0, 1)] += Complex64::new(1e-6, 0.0);
    let e = run("PBE0", &t, &ok_cfg)
        .expect_err("beta tamper")
        .to_string();
    assert!(e.contains(guard_msg) && e.contains("fock_beta"), "{e}");
    let mut t = fx.res.scf.clone();
    std::mem::swap(&mut t.fock_alpha, &mut t.fock_beta);
    let e = run("PBE0", &t, &ok_cfg).expect_err("swapped").to_string();
    assert!(e.contains(guard_msg) && e.contains("fock_alpha"), "{e}");
    // Unconverged SCF.
    let mut t = fx.res.scf.clone();
    t.converged = false;
    assert!(run("PBE0", &t, &ok_cfg).is_err());
    // A pure functional needs no exxdiv and has no exchange part.
    let a = stress_with(&pbe, "PBE", &pbe.res.scf, &scfg(None, None, None, None)).unwrap();
    assert_eq!(mmax(&a.hf.parts.exchange), 0.0);
    // The exchange fraction of the diagnostic entry is validated.
    let z = zero_xc(&H3, G111, Jk::Dense, ExxDiv::None, 0.0);
    for bad in [-0.1, 1.5, f64::NAN] {
        let e = null_stress(&z, bad, &scfg(None, None, None, None))
            .err()
            .unwrap_or_else(|| panic!("a = {bad} accepted"))
            .to_string();
        assert!(e.contains("outside [0, 1]"), "{e}");
    }
}

/// Default-set subset of the mutant sweep: the three defects that most need the
/// open-shell Gamma anchor (a spin swap in the XC strain, the other spin's
/// exchange in the Pulay Fock, the exchange fraction applied twice). Each must
/// miss the Gamma reference by far more than the 1e-5 mutant bar (misses
/// measured in the full sweep: XcSpinSwap 1.6e-1, PulayKSwap 4.6e-1,
/// ExchScaledTwice 3.0e-1).
#[test]
#[ignore = "validation: k-UKS stress spin / scaling mutants vs the Gamma UKS anchor; CI ~142 s"]
fn spin_and_scaling_mutants_are_caught_by_the_gamma_anchor() {
    let f = "PBE0";
    let gam = gamma_side(&cell_of(&H3), f);
    let fx = fixture(&H3, G111, f, Jk::Dense, ExxDiv::Ewald);
    for (label, xm, sm) in [
        ("XcSpinSwap", None, Some(KUStressMutation::XcSpinSwap)),
        ("PulayKSwap", None, Some(KUStressMutation::PulayKSwap)),
        (
            "ExchScaledTwice",
            Some(KXcStressMutation::ExchScaledTwice),
            None,
        ),
    ] {
        let d = mdiff(&stress_mut(&fx, xm, sm, None).de_deps, &gam.de_deps);
        eprintln!("1x1x1 vs Gamma under {label}: miss {d:.2e}");
        assert!(d > MUTANT_BAR, "{label} survives the Gamma anchor: {d:e}");
    }
}
