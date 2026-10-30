//! Weekly validation half of `pbc_grad_ro.rs` (every test
//! `#[ignore = "validation: ..."]`): the tests whose per-commit CI time
//! was >= 120 s (mean of runs 40957025137 and 40953725119), moved so the
//! fast tier stays balanced over 8 shards. The module docs, hypotheses and
//! bars are in `pbc_grad_ro.rs`; the shared helpers are
//! `pbc_grad_ro_support/mod.rs`.
//!
//! Fast-tier canaries that keep these invariants per commit:
//! * `h3_roks_force_matches_fd_of_own_energy_with_grid_response` (the unmutated
//!   FD anchor)
//! * `h3_roks_pbe_mutants_move_the_force_and_uhf_form_w_is_an_identity` (PBE:
//!   every mutant moves the force; W_UHF identity)
//! * `h3_rohf_fd_anchor_catches_each_mutant`,
//!   `h3_rohf_uhf_form_w_is_an_identity_at_convergence`

mod common;
#[path = "pbc_grad_ro_support/mod.rs"]
mod support;

use support::*;

#[test]
#[ignore = "validation: PBC Gamma ROKS force, H3 FD anchor vs every W/XC mutant; CI ~218 s"]
fn h3_roks_fd_anchor_catches_each_mutant_and_uhf_form_w_is_an_identity() {
    let su = setup(h3_cell(), &pyscf_sto3g_h());
    let fdv = h3_roks_fd();
    for (k, (cfg, scf)) in h3_roks_refs(&su).iter().enumerate() {
        let hybrid = cfg.functional == "PBE0";
        let mut muts = vec![
            GradMutation::RoWFeffEps,
            GradMutation::RoWNoCo,
            GradMutation::WSpinSum,
            GradMutation::RoXcRksForm,
        ];
        if hybrid {
            muts.push(GradMutation::ExchTotal);
            if cfg.exxdiv == ExxDiv::Ewald {
                muts.push(GradMutation::MadelungTotal);
            }
        }
        for m in muts {
            let g = roks_grad(&su, scf, cfg, Some(m));
            let err = max_fd_err(&g.grad, fdv, k);
            eprintln!(
                "H3 ROKS {} {:?} mutant {m:?}: max|analytic − FD| = {err:.2e}, |ΣF| {:.1e}",
                cfg.functional, cfg.exxdiv, g.net_force
            );
            assert!(
                err > MUTANT_BAR,
                "{} {:?}: {m:?} escaped the FD anchor: {err:e}",
                cfg.functional,
                cfg.exxdiv
            );
        }
        let g = roks_grad(&su, scf, cfg, None);
        let gu = roks_grad(&su, scf, cfg, Some(GradMutation::RoWUhfForm));
        let df = max_diff(&g.grad, &gu.grad);
        eprintln!(
            "H3 ROKS {} {:?}: |F(W_UHF) − F(W_RO)| {df:.2e} (identity)",
            cfg.functional, cfg.exxdiv
        );
        assert!(df < W_IDENTITY_FORCE_BAR, "{df:e}");
    }
}
