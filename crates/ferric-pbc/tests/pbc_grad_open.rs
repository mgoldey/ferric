//! Gamma-point periodic UHF / RKS / UKS analytic forces
//! (`ferric_pbc::grad::{gamma_uhf_gradient, gamma_rks_gradient,
//! gamma_uks_gradient}`), the Rust port of `reference/pbc/pbc_grad_open.py`
//! (FINDINGS "Iteration 17").
//!
//! What is anchored against what:
//! * EXACTNESS ANCHOR: analytic vs central finite difference of ferric's OWN
//!   Gamma energy (`periodic_hcore` + `DenseAftEri` + `gamma_uhf` /
//!   `gamma_rks` / `gamma_uks`), the XC grid REBUILT at each displaced
//!   geometry (points ride on their home atom, weights recomputed) — the
//!   derivative of the energy as defined, so it needs the full grid
//!   response. Grid (40, 50), SSF, D = 8 (the prototype's anchor grid).
//!   Prototype floors at h = 1e-4: H3 UHF 3.8e-9, H3 UKS LDA/PBE/PBE0
//!   3.3e-9..3.5e-9; the Rust RHF anchor adds the primitive screens and
//!   reaches 2.4e-9 on H2. Bar 1e-7 (as `pbc_grad.rs`).
//!   H2 a = 4 KS uses h = 5e-5: at h ≥ 1e-4 displaced points cross the hard
//!   `D` neighbour mask and the ENERGY jumps by 1e-11..1e-9 Ha (prototype
//!   h-scan: (1,y) +4.1e-8 at 1e-4, −2.2e-11 at 5e-5; (0,z) −5.8e-10 at
//!   5e-5, +2.05e-6 at 2e-4).
//! * Mutants (the hidden `GammaGradConfig::mutation` knob), each must miss FD
//!   by > 1e-3 (prototype, H3: smallest miss 5.7e-3 = `XcNoGga` on PBE0; the
//!   response pieces 1.4e-1..1.8e-1; spin mutants 9.2e-3..1.3e-1). The bar
//!   sits four orders above the passing residual and ~6x below the smallest
//!   measured miss.
//! * Identities on ONE density (no second SCF, so they test the formula,
//!   not SCF convergence): closed-shell UHF(D/2, D/2) ≡ RHF(D) and
//!   UKS(D/2, D/2) ≡ RKS(D) ≤ 1e-12 (prototype 0 / 1.1e-16); the three spin
//!   mutants are EXACTLY blind there (algebra; the UKS/RKS pair at 1e-11,
//!   see `XC_IDENTITY_BAR`); F(ewald) ≡ F(none) on an
//!   open-shell idempotent density ≤ 1e-11 (prototype 1.5e-12 across two
//!   SCFs), and the RHF-form Madelung mutant breaks it by > 1e-3 (prototype
//!   1.3e-1 UHF, 3.3e-2 PBE0).
//! * Translation invariance ΣF ≤ 1e-10 with the full response (exact by
//!   construction: `Σ_A (ao + point) = 0` per point and `Σ_A dW = 0`); the
//!   `NoPointMotion` and `XcNoAo` mutants break it (prototype 1e-4..3e-4) —
//!   a check on the image fold of the weight derivative, and the only test
//!   here that the sum of forces can pass or fail.
//!
//! Not covered (as in the prototype): FD with FROZEN points/weights (the
//! drivers always rebuild the grid; the response is instead pinned by the
//! two response mutants), the Becke scheme's full anchor (the weight
//! derivative is FD-checked for both schemes in ferric-dft's
//! `partition_weight_derivative_matches_fd`), meta-GGA, RSH, ROHF/ROKS,
//! RS-GDF J/K forces, k-points, stress.
//!
//! The tests whose per-commit CI time was >= 120 s
//! (`h3_uks_fd_anchor_catches_each_mutant`) run weekly from
//! `validation_pbc_grad_open.rs`; the shared helpers are
//! `pbc_grad_open_support/mod.rs`.

mod common;
#[path = "pbc_grad_open_support/mod.rs"]
mod support;

use support::*;

#[test]
fn h3_uhf_force_matches_fd_of_own_energy_both_exxdiv() {
    let su = setup(h3_cell(), &pyscf_sto3g_h());
    let (none, ewald) = h3_uhf_ref(&su);
    let fdv = h3_uhf_fd();
    for (k, (scf, exx)) in [(&none, ExxDiv::None), (&ewald, ExxDiv::Ewald)]
        .into_iter()
        .enumerate()
    {
        let g = uhf_grad(&su, scf, exx, None);
        let err = max_fd_err(&g.grad, fdv, k);
        eprintln!(
            "H3 UHF doublet {exx:?}: max|analytic − FD| = {err:.2e}, |ΣF| {:.1e}, comm {:.1e}\n{:.10}",
            g.net_force, g.commutator, g.grad
        );
        assert!(err < FD_BAR, "{exx:?}: {err:e}");
        assert!(g.net_force < NET_FORCE_BAR, "ΣF {:e}", g.net_force);
        assert!(g.e_xc.is_none() && g.n_grid_points == 0);
    }
}

#[test]
fn h3_uhf_fd_anchor_catches_each_spin_mutant() {
    let su = setup(h3_cell(), &pyscf_sto3g_h());
    let (none, ewald) = h3_uhf_ref(&su);
    let fdv = h3_uhf_fd();
    // (mutant, exxdiv): MadelungTotal is visible only with ewald (prototype
    // 1.3e-1); WSpinSum 1.3e-1 / 6.6e-2; ExchTotal 3.7e-2.
    for (m, exx) in [
        (GradMutation::WSpinSum, ExxDiv::None),
        (GradMutation::WSpinSum, ExxDiv::Ewald),
        (GradMutation::ExchTotal, ExxDiv::None),
        (GradMutation::ExchTotal, ExxDiv::Ewald),
        (GradMutation::MadelungTotal, ExxDiv::Ewald),
    ] {
        let (scf, k) = match exx {
            ExxDiv::None => (&none, 0),
            ExxDiv::Ewald => (&ewald, 1),
        };
        let g = uhf_grad(&su, scf, exx, Some(m));
        let err = max_fd_err(&g.grad, fdv, k);
        eprintln!(
            "H3 UHF mutant {m:?} ({exx:?}): max|analytic − FD| = {err:.2e}, |ΣF| {:.1e} \
             (translation invariance cannot see it)",
            g.net_force
        );
        assert!(
            err > MUTANT_BAR,
            "{m:?} ({exx:?}) escaped the FD anchor: {err:e}"
        );
    }
}

/// F(ewald) ≡ F(none) on ONE open-shell idempotent density (the per-spin
/// Madelung term cancels `−v_M D_σ` inside `W_σ` exactly), and the RHF-form
/// Madelung mutant breaks it.
#[test]
fn h3_uhf_ewald_and_none_forces_agree_on_one_density() {
    let su = setup(h3_cell(), &pyscf_sto3g_h());
    let (_, ewald) = h3_uhf_ref(&su);
    let gn = uhf_grad(&su, &ewald, ExxDiv::None, None);
    let ge = uhf_grad(&su, &ewald, ExxDiv::Ewald, None);
    let d = max_diff(&gn.grad, &ge.grad);
    eprintln!(
        "H3 UHF: |F_ewald − F_none| = {d:.2e} (v_M = {})",
        ge.madelung
    );
    assert!(ge.madelung > 0.1);
    assert!(d < EWALD_NONE_BAR, "{d:e}");
    let gm = uhf_grad(
        &su,
        &ewald,
        ExxDiv::Ewald,
        Some(GradMutation::MadelungTotal),
    );
    let dm = max_diff(&gn.grad, &gm.grad);
    eprintln!("H3 UHF MadelungTotal mutant: |F_ewald − F_none| = {dm:.2e}");
    assert!(dm > MUTANT_BAR, "{dm:e}");
}

#[test]
fn h3_uks_force_matches_fd_of_own_energy_with_grid_response() {
    let su = setup(h3_cell(), &pyscf_sto3g_h());
    let fdv = h3_uks_fd();
    for (k, (cfg, scf)) in h3_uks_refs(&su).iter().enumerate() {
        let g = uks_grad(&su, scf, cfg, None);
        let err = max_fd_err(&g.grad, fdv, k);
        let resp = &g.parts.xc_point + &g.parts.xc_weight;
        let resp_max = resp.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        eprintln!(
            "H3 UKS {} {:?}: max|analytic − FD| = {err:.2e}, |ΣF| {:.1e}, comm {:.1e}, \
             npts {}, |grid response| {resp_max:.2e}\n{:.10}",
            cfg.functional, cfg.exxdiv, g.net_force, g.commutator, g.n_grid_points, g.grad
        );
        assert!(err < FD_BAR, "{} {:?}: {err:e}", cfg.functional, cfg.exxdiv);
        assert!(g.net_force < NET_FORCE_BAR, "ΣF {:e}", g.net_force);
        // The response is live (prototype ~2-3e-3 at (40, 50)): an anchor
        // that passed with it absent would not be testing it.
        assert!(resp_max > 1e-4, "grid response {resp_max:e}");
    }
}

/// Per-commit canary of `validation_pbc_grad_open.rs`'s
/// `h3_uks_fd_anchor_catches_each_mutant` without its FD sweep: on the H3
/// doublet UKS PBE (the GGA route, ewald), every XC / spin mutant moves the
/// analytic force by more than `MUTANT_BAR`, and the two AO-term bookkeeping
/// mutants break translation invariance. The unmutated force is pinned to
/// the FD of the own energy by
/// `h3_uks_force_matches_fd_of_own_energy_with_grid_response`, so a mutant
/// that moves it this far would miss that FD; the exchange / Madelung
/// mutants are caught on the same exchange code by
/// `h3_uhf_fd_anchor_catches_each_spin_mutant`.
#[test]
fn h3_uks_pbe_mutants_move_the_force() {
    let su = setup(h3_cell(), &pyscf_sto3g_h());
    let cfg = uks_cfg("PBE", ExxDiv::Ewald, None);
    let scf = uks(&su, &cfg).scf;
    let g0 = uks_grad(&su, &scf, &cfg, None);
    for m in [
        GradMutation::XcNoAo,
        GradMutation::NoPointMotion,
        GradMutation::NoWeightDeriv,
        GradMutation::WSpinSum,
        GradMutation::XcNoGga,
    ] {
        let g = uks_grad(&su, &scf, &cfg, Some(m));
        let d = max_diff(&g.grad, &g0.grad);
        eprintln!(
            "H3 UKS PBE mutant {m:?}: |F_mut − F| = {d:.2e}, |ΣF| {:.1e}",
            g.net_force
        );
        assert!(d > MUTANT_BAR, "{m:?} does not move the force: {d:e}");
        if matches!(m, GradMutation::XcNoAo | GradMutation::NoPointMotion) {
            assert!(g.net_force > 1e-6, "{m:?}: ΣF {:e}", g.net_force);
        }
    }
}

#[test]
fn h3_pbe0_ewald_and_none_forces_agree_on_one_density() {
    let su = setup(h3_cell(), &pyscf_sto3g_h());
    let r = uks(&su, &uks_cfg("PBE0", ExxDiv::Ewald, None));
    let gn = uks_grad(&su, &r.scf, &uks_cfg("PBE0", ExxDiv::None, None), None);
    let ge = uks_grad(&su, &r.scf, &uks_cfg("PBE0", ExxDiv::Ewald, None), None);
    let d = max_diff(&gn.grad, &ge.grad);
    eprintln!("H3 PBE0: |F_ewald − F_none| = {d:.2e}");
    assert!((ge.exact_exchange_fraction - 0.25).abs() < 1e-12);
    assert!(d < EWALD_NONE_BAR, "{d:e}");
    let gm = uks_grad(
        &su,
        &r.scf,
        &uks_cfg("PBE0", ExxDiv::Ewald, None),
        Some(GradMutation::MadelungTotal),
    );
    let dm = max_diff(&gn.grad, &gm.grad);
    eprintln!("H3 PBE0 MadelungTotal mutant: |F_ewald − F_none| = {dm:.2e}");
    assert!(dm > MUTANT_BAR, "{dm:e}");
}

// ---------------------------------------------------- closed-shell identities

/// UHF(D/2, D/2) ≡ RHF(D) on one density, and the three spin mutants are
/// exactly blind on it (their difference from the correct form is zero
/// algebraically for `D_α = D_β`, `F_α = F_β`).
#[test]
fn closed_shell_uhf_gradient_is_the_rhf_gradient() {
    let cell = cell_at(&GRAD_H2_ATOMS, cubic(4.0), 1);
    let su = setup(cell, &pyscf_sto3g_h());
    let eri = su.eri.clone().with_exxdiv(&su.cell, ExxDiv::Ewald).unwrap();
    let scf = gamma_rhf(&su.cell, &su.prep, &su.hc, &eri);
    let grhf = gamma_rhf_gradient_with(
        &su.cell,
        &su.prep,
        &hcore_cfg(),
        &su.hc,
        &su.eri,
        &scf,
        ExxDiv::Ewald,
        &GammaGradConfig::default(),
    )
    .unwrap();
    let u = as_unrestricted(&scf);
    let guhf = uhf_grad(&su, &u, ExxDiv::Ewald, None);
    let d = max_diff(&grhf.grad, &guhf.grad);
    eprintln!("H2: |F_UHF(D/2,D/2) − F_RHF(D)| = {d:.2e}");
    assert!(d < IDENTITY_BAR, "{d:e}");
    for m in [
        GradMutation::WSpinSum,
        GradMutation::MadelungTotal,
        GradMutation::ExchTotal,
    ] {
        let gm = uhf_grad(&su, &u, ExxDiv::Ewald, Some(m));
        let dm = max_diff(&gm.grad, &guhf.grad);
        assert!(
            dm < IDENTITY_BAR,
            "{m:?} should be blind on a closed shell: {dm:e}"
        );
    }
}

/// UKS(D/2, D/2) ≡ RKS(D) on one density (polarized vs unpolarized libxc
/// kernel and the per-spin bookkeeping), PBE and PBE0 with ewald.
#[test]
fn closed_shell_uks_gradient_is_the_rks_gradient() {
    let cell = cell_at(&GRAD_H2_ATOMS, cubic(4.0), 1);
    let su = setup(cell, &pyscf_sto3g_h());
    for xc in ["PBE", "PBE0"] {
        let rcfg = rks_cfg(xc, ExxDiv::Ewald);
        let scf = rks(&su, &rcfg);
        let grks = rks_grad(&su, &scf, &rcfg, None);
        let guks = uks_grad(
            &su,
            &as_unrestricted(&scf),
            &uks_cfg(xc, ExxDiv::Ewald, None),
            None,
        );
        let d = max_diff(&grks.grad, &guks.grad);
        let de = (grks.e_xc.unwrap() - guks.e_xc.unwrap()).abs();
        eprintln!("H2 {xc}: |F_UKS(D/2,D/2) − F_RKS(D)| = {d:.2e}, |ΔE_xc| {de:.2e}");
        assert!(d < XC_IDENTITY_BAR, "{xc}: {d:e}");
        assert!(grks.net_force < NET_FORCE_BAR, "ΣF {:e}", grks.net_force);
    }
}

/// Closed-shell RKS on H2 a = 4 at h = 5e-5 (module doc), LDA and PBE0.
#[test]
fn h2_rks_force_matches_fd_of_own_energy() {
    let su = setup(cell_at(&GRAD_H2_ATOMS, cubic(4.0), 1), &pyscf_sto3g_h());
    let fdv = h2_rks_fd();
    for (k, xc) in H2_RKS_XCS.iter().enumerate() {
        let cfg = rks_cfg(xc, ExxDiv::Ewald);
        let scf = rks(&su, &cfg);
        let g = rks_grad(&su, &scf, &cfg, None);
        let err = max_fd_err(&g.grad, fdv, k);
        eprintln!(
            "H2 RKS {xc}: max|analytic − FD| = {err:.2e}, |ΣF| {:.1e}, comm {:.1e}\n{:.10}",
            g.net_force, g.commutator, g.grad
        );
        assert!(err < FD_BAR, "{xc}: {err:e}");
        assert!(g.net_force < NET_FORCE_BAR, "ΣF {:e}", g.net_force);
        // Response mutants on a closed shell (prototype H2: 1.1e-1 each).
        for m in [GradMutation::NoPointMotion, GradMutation::NoWeightDeriv] {
            let gm = rks_grad(&su, &scf, &cfg, Some(m));
            let e = max_fd_err(&gm.grad, fdv, k);
            eprintln!("  mutant {m:?}: {e:.2e}");
            assert!(e > MUTANT_BAR, "{xc} {m:?}: {e:e}");
        }
    }
}

#[test]
fn gradients_reject_the_wrong_spin_kind() {
    let su = setup(cell_at(&GRAD_H2_ATOMS, cubic(4.0), 1), &pyscf_sto3g_h());
    let scf = gamma_rhf(&su.cell, &su.prep, &su.hc, &su.eri);
    let err = gamma_uhf_gradient_with(
        &su.cell,
        &su.prep,
        &hcore_cfg(),
        &su.hc,
        &su.eri,
        &scf,
        ExxDiv::None,
        &GammaGradConfig::default(),
    )
    .expect_err("a restricted result must be refused by the UHF gradient");
    assert!(err.to_string().contains("unrestricted"), "{err}");
    let err = gamma_rks_gradient_with(
        &su.cell,
        &su.prep,
        &hcore_cfg(),
        &su.hc,
        &su.eri,
        &as_unrestricted(&scf),
        &rks_cfg("PBE", ExxDiv::None),
        &GammaGradConfig::default(),
    )
    .expect_err("an unrestricted result must be refused by the RKS gradient");
    assert!(err.to_string().contains("restricted"), "{err}");
}

#[test]
#[ignore = "slow: triclinic 4H s+p triplet, 8 displaced dense-AFT builds + grids + 16 SCFs; run with --release -- --ignored"]
fn triclinic_sp_triplet_uhf_and_uks_force_match_fd() {
    let bs = sp_basis_h();
    let su = setup(cell_at(&TRI_MOVED, TRI_A, 3), &bs);
    let r_uhf = uhf(&su, &uhf_cfg(ExxDiv::None, None)).scf;
    let pbe_cfg = uks_cfg("PBE", ExxDiv::None, None);
    let r_pbe = uks(&su, &pbe_cfg).scf;
    let seeds = [mos_of(&r_uhf), mos_of(&r_pbe)];
    let energies = |c: &Cell| {
        let s = setup(c.clone(), &bs);
        vec![
            uhf(&s, &uhf_cfg(ExxDiv::None, seeds[0].clone())).scf.energy,
            uks(&s, &uks_cfg("PBE", ExxDiv::None, seeds[1].clone()))
                .scf
                .energy,
        ]
    };
    let f1 = fd(&TRI_MOVED, TRI_A, 3, &TRI_COMPS, FD_H, &energies);
    let f2 = fd(&TRI_MOVED, TRI_A, 3, &TRI_COMPS, 0.5 * FD_H, &energies);
    let fdv = richardson(&f1, &f2);
    let g_uhf = uhf_grad(&su, &r_uhf, ExxDiv::None, None);
    let g_pbe = uks_grad(&su, &r_pbe, &pbe_cfg, None);
    for (k, (name, g)) in [("UHF", &g_uhf), ("UKS PBE", &g_pbe)]
        .into_iter()
        .enumerate()
    {
        let err = max_fd_err(&g.grad, &fdv, k);
        let err_h = max_fd_err(&g.grad, &f1, k);
        eprintln!(
            "tri s+p triplet {name}: max|analytic − FD_Richardson| = {err:.2e} \
             (plain h = 1e-4: {err_h:.2e}), |ΣF| {:.1e}\n{:.10}",
            g.net_force, g.grad
        );
        // p shells: the libint 3-centre derivative floor measured on the RHF
        // triclinic anchor is 1.85e-7 (pbc_grad.rs); bar from that floor.
        assert!(err < 5e-7, "{name}: {err:e}");
        assert!(g.net_force < NET_FORCE_BAR, "ΣF {:e}", g.net_force);
    }
}
