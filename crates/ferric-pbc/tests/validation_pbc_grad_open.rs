//! Weekly validation half of `pbc_grad_open.rs` (every test
//! `#[ignore = "validation: ..."]`): the tests whose per-commit CI time
//! was >= 120 s (mean of runs 40957025137 and 40953725119), moved so the
//! fast tier stays balanced over 8 shards. The module docs, hypotheses and
//! bars are in `pbc_grad_open.rs`; the shared helpers are
//! `pbc_grad_open_support/mod.rs`.
//!
//! Fast-tier canaries that keep these invariants per commit:
//! * `h3_uks_force_matches_fd_of_own_energy_with_grid_response` (the unmutated
//!   FD anchor)
//! * `h3_uks_pbe_mutants_move_the_force` (every XC/spin mutant moves the PBE
//!   force off the FD-anchored one)
//! * `h3_uhf_fd_anchor_catches_each_spin_mutant`

mod common;
#[path = "pbc_grad_open_support/mod.rs"]
mod support;

use support::*;

#[test]
#[ignore = "validation: PBC Gamma UKS force, H3 FD anchor vs every XC/spin mutant; CI ~130 s"]
fn h3_uks_fd_anchor_catches_each_mutant() {
    let su = setup(h3_cell(), &pyscf_sto3g_h());
    let fdv = h3_uks_fd();
    let refs = h3_uks_refs(&su);
    for (k, (cfg, scf)) in refs.iter().enumerate() {
        let hybrid = cfg.functional == "PBE0";
        let gga = cfg.functional != "LDA";
        let mut muts = vec![
            GradMutation::XcNoAo,
            GradMutation::NoPointMotion,
            GradMutation::NoWeightDeriv,
            GradMutation::WSpinSum,
        ];
        if gga {
            muts.push(GradMutation::XcNoGga);
        }
        if hybrid {
            muts.push(GradMutation::ExchTotal);
            if cfg.exxdiv == ExxDiv::Ewald {
                muts.push(GradMutation::MadelungTotal);
            }
        }
        for m in muts {
            let g = uks_grad(&su, scf, cfg, Some(m));
            let err = max_fd_err(&g.grad, fdv, k);
            eprintln!(
                "H3 UKS {} {:?} mutant {m:?}: max|analytic − FD| = {err:.2e}, |ΣF| {:.1e}",
                cfg.functional, cfg.exxdiv, g.net_force
            );
            assert!(
                err > MUTANT_BAR,
                "{} {:?}: {m:?} escaped the FD anchor: {err:e}",
                cfg.functional,
                cfg.exxdiv
            );
            // Dropping either half of the AO-term bookkeeping breaks the
            // translation invariance the full response restores.
            if matches!(m, GradMutation::XcNoAo | GradMutation::NoPointMotion) {
                assert!(g.net_force > 1e-6, "{m:?}: ΣF {:e}", g.net_force);
            }
        }
    }
}
