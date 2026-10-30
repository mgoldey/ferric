//! Weekly validation half of `pbc_grad_rsgdf.rs` (every test
//! `#[ignore = "validation: ..."]`): the tests whose per-commit CI time
//! was >= 120 s (mean of runs 40957025137 and 40953725119), moved so the
//! fast tier stays balanced over 8 shards. The module docs, hypotheses and
//! bars are in `pbc_grad_rsgdf.rs`; the shared helpers are
//! `pbc_grad_rsgdf_support/mod.rs`.
//!
//! Fast-tier canaries that keep these invariants per commit:
//! * `h3_uhf_force_matches_fd_and_catches_each_fit_mutant` (open-shell RS-GDF
//!   FD anchor + the same fit mutants)
//! * `closed_shell_uks_rsgdf_gradient_is_the_rks_gradient` (the UKS RS-GDF
//!   force path)

mod common;
#[path = "pbc_grad_rsgdf_support/mod.rs"]
mod support;

use support::*;

#[test]
#[ignore = "validation: PBC Gamma UKS PBE0 RS-GDF force, H3 FD + fit mutants; CI ~120 s"]
fn h3_uks_pbe0_force_matches_fd_and_catches_each_fit_mutant() {
    let su = h3_setup();
    let r = h3_pbe0_ref(&su);
    let none = r.none_stage.clone().expect("none stage");
    let fdv = h3_uks_fd();
    for (k, (scf, exx)) in [(&none, ExxDiv::None), (&r.scf, ExxDiv::Ewald)]
        .into_iter()
        .enumerate()
    {
        let cfg = uks_cfg("PBE0", exx, None);
        let g = uks_grad(&su, scf, &cfg, None);
        let err = max_fd_err(&g.grad, fdv, k);
        report(&format!("H3 UKS PBE0 cc-pvdz-ri {exx:?}"), &g, err);
        assert!((g.exact_exchange_fraction - 0.25).abs() < 1e-12);
        assert!(err < FD_BAR, "{exx:?}: {err:e}");
        assert!(g.net_force < NET_FORCE_BAR, "ΣF {:e}", g.net_force);
        for m in FIT_MUTANTS {
            let gm = uks_grad(&su, scf, &cfg, Some(m));
            let e = max_fd_err(&gm.grad, fdv, k);
            eprintln!("  mutant {m:?} ({exx:?}): {e:.2e}");
            assert!(
                e > MUTANT_BAR,
                "{m:?} ({exx:?}) escaped the FD anchor: {e:e}"
            );
        }
    }
}
