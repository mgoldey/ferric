//! Weekly half of `pbc_kuxcgrad.rs` (k-point open-shell UKS force): the tests
//! whose per-commit CI time was >= 120 s (runs 40957025137 / 40953725119,
//! mean 145 / 180 s), moved here so the per-commit shards stay short. Every
//! test is `#[ignore = "validation: ..."]`. Hypotheses are in the doc of
//! `pbc_kuxcgrad.rs`; the helpers are `kuxcgrad_support/mod.rs`. Per-commit
//! canaries that remain: `one_point_mesh_matches_the_gamma_uks_gradient_per_part_pbe`
//! and `refusals_and_the_per_spin_fock_guard_canary`.

mod common;
#[path = "kuxcgrad_support/mod.rs"]
mod support;

use support::*;

#[test]
#[ignore = "validation: k-UKS PBE0 force 1x1x1 vs Gamma UKS per part + spin / scaling mutants; CI ~145 s"]
fn one_point_mesh_matches_the_gamma_uks_gradient_per_part() {
    one_point_vs_gamma("PBE0", true);
}

#[test]
#[ignore = "validation: k-UKS force refusals and per-spin Fock guard, full sweep; CI ~180 s"]
fn refusals_and_the_per_spin_fock_guard() {
    let fx = fixture(&H3, [1, 1, 1], "PBE0", Jk::Dense, ExxDiv::Ewald);
    let ok_cfg = gcfg(Some(ExxDiv::Ewald), None, None, None);
    let run = |f: &str, s: &KUScfResult, cfg: &KUksGradConfig| force_with(&fx, f, s, cfg);
    // Typed refusals, entry-point specific.
    for (name, key) in [
        // HSE06 is covered on kernels built with its range separation
        // (`pbc_kuks_rsh_force.rs`); these PBE0 kernels are bare: refused.
        ("HSE06", "bare Coulomb exchange"),
        ("wB97X-V", "VV10"),
        // SCAN is covered (`pbc_kuxc_mgga.rs`): a PBE0 state handed in as
        // SCAN is refused by the Fock guard.
        ("SCAN", "differs from the SCF's stored fock_alpha"),
    ] {
        let e = run(name, &fx.res.scf, &ok_cfg)
            .err()
            .unwrap_or_else(|| panic!("{name} accepted"))
            .to_string();
        eprintln!("{name}: {e}");
        assert!(e.contains(key), "{name}: {e}");
    }
    let e = run("PBE0", &fx.res.scf, &gcfg(None, None, None, None))
        .expect_err("no exxdiv")
        .to_string();
    assert!(
        e.contains("kpoint_uks_gradient: a hybrid functional"),
        "{e}"
    );
    let ok = run("PBE0", &fx.res.scf, &ok_cfg).unwrap();
    assert!(ok.hf.fock_mismatch < 1e-12, "{:e}", ok.hf.fock_mismatch);
    // Guard from every side.
    let pbe = fixture(&H3, [1, 1, 1], "PBE", Jk::Dense, ExxDiv::Ewald);
    let cases: Vec<(&str, Result<KUksGradient, ferric_core::FerricError>)> = vec![
        (
            "exxdiv None on an Ewald SCF",
            run(
                "PBE0",
                &fx.res.scf,
                &gcfg(Some(ExxDiv::None), None, None, None),
            ),
        ),
        ("B3LYP on a PBE0 SCF", run("B3LYP", &fx.res.scf, &ok_cfg)),
        ("PBE force on a PBE0 SCF", run("PBE", &fx.res.scf, &ok_cfg)),
        (
            "PBE0 force on a PBE SCF",
            run("PBE0", &pbe.res.scf, &ok_cfg),
        ),
    ];
    for (n, r) in cases {
        let e = r
            .err()
            .unwrap_or_else(|| panic!("{n} accepted"))
            .to_string();
        eprintln!("{n}: {e}");
        assert!(
            e.contains("kpoint_uks_gradient: the Fock matrix rebuilt"),
            "{n}: {e}"
        );
    }
    // Tampers of each spin's Fock are refused, naming the spin.
    let mut t = fx.res.scf.clone();
    t.fock_alpha[0][(0, 1)] += Complex64::new(1e-6, 0.0);
    let e = run("PBE0", &t, &ok_cfg)
        .expect_err("alpha tamper")
        .to_string();
    assert!(e.contains("fock_alpha"), "{e}");
    let mut t = fx.res.scf.clone();
    t.fock_beta[0][(0, 1)] += Complex64::new(1e-6, 0.0);
    let e = run("PBE0", &t, &ok_cfg)
        .expect_err("beta tamper")
        .to_string();
    assert!(e.contains("fock_beta"), "{e}");
    // Swapped spin Focks.
    let mut t = fx.res.scf.clone();
    std::mem::swap(&mut t.fock_alpha, &mut t.fock_beta);
    let e = run("PBE0", &t, &ok_cfg)
        .expect_err("swapped spin Focks")
        .to_string();
    assert!(
        e.contains("kpoint_uks_gradient: the Fock matrix rebuilt") && e.contains("fock_alpha"),
        "{e}"
    );
    // A pure functional needs no exxdiv and has no exchange part.
    let a = force_with(&pbe, "PBE", &pbe.res.scf, &gcfg(None, None, None, None)).unwrap();
    assert_eq!(max_abs(&a.hf.parts.exchange), 0.0);
}
