//! Weekly half of `pbc_kxcgrad_hybrid.rs` (k-point closed-shell hybrid KS
//! force): `smoke_mutants_are_caught`, whose per-commit CI time was >= 120 s
//! (runs 40957025137 / 40953725119, mean 206 s), moved here so the
//! per-commit shards stay short. Every test is
//! `#[ignore = "validation: ..."]`. Hypotheses are in the doc of
//! `pbc_kxcgrad_hybrid.rs`; the helpers are `kxcgrad_hybrid_support/mod.rs`.
//! Per-commit canaries that remain:
//! `dense_exchange_scale_mutants_are_caught_on_the_one_point_mesh` and
//! `gamma_rsgdf_anchor_catches_the_fit_scale_mutants`.

mod common;
#[path = "kxcgrad_hybrid_support/mod.rs"]
mod support;

use support::*;

/// Production-wired defects of the exchange scale (`KGradConfig::mutation`,
/// `KRksGradConfig::mutation`), each against the FD-anchored force. Artifact
/// hypothesis: a mutant that is an IDENTITY on a route/exxdiv must show exactly
/// 0 there (asserted), otherwise it must miss by far more than the bar.
#[test]
#[ignore = "validation: k-RKS hybrid force exchange-scale mutant sweep (dense/RS-GDF ewald, dense none), H2 1x1x3; CI ~206 s"]
fn smoke_mutants_are_caught() {
    use KGradMutation as G;
    // Dense AFT + Ewald: every dense-reachable defect.
    let fx = fixture(&H2, [1, 1, 3], "PBE0", Jk::Dense, ExxDiv::Ewald);
    let base = force(&fx);
    for m in [
        G::ExchDenseUnscaled,
        G::ExchMadelungUnscaled,
        G::ExchJScaled,
        G::ExchScaledTwice,
        G::ExchSpinFactor,
    ] {
        let d = miss_g(&fx, m, &base);
        eprintln!("dense ewald {m:?}: miss {d:.2e}");
        assert!(d > MUTANT_BAR, "{m:?} survives: {d:e}");
    }
    // The fit-only defect is an IDENTITY on the dense route.
    assert_eq!(miss_g(&fx, G::ExchFitUnscaled, &base), 0.0);
    // A wrong remainder ALONE (X1/X3/X4 and the Pulay V_xc with PBE instead
    // of the PBE0 remainder; only the guard skipped), and the guard-skip alone
    // as the control (must be exactly 0, so the miss is the remainder's).
    let d = parts_diff(
        &force_mut(&fx, Some(KXcMutation::WrongRemainder), None).grad,
        &base.grad,
    );
    eprintln!("WrongRemainder (PBE for PBE0), remainder ALONE: miss {d:.2e}");
    assert!(d > MUTANT_BAR);
    let d0 = parts_diff(
        &force_mut(&fx, Some(KXcMutation::GuardSkippedOnly), None).grad,
        &base.grad,
    );
    assert_eq!(d0, 0.0, "the guard-skip control must not change the force");
    // RS-GDF + Ewald.
    let fg = fixture(&H2, [1, 1, 3], "PBE0", Jk::Gdf, ExxDiv::Ewald);
    let gb = force(&fg);
    for m in [
        G::ExchFitUnscaled,
        G::ExchMadelungUnscaled,
        G::ExchScaledTwice,
        G::ExchSpinFactor,
        G::ExchJScaled,
    ] {
        let d = miss_g(&fg, m, &gb);
        eprintln!("gdf ewald {m:?}: miss {d:.2e}");
        assert!(d > MUTANT_BAR, "gdf {m:?} survives: {d:e}");
    }
    // The dense-only defect is an identity on RS-GDF (no dense K walk exists).
    assert_eq!(miss_g(&fg, G::ExchDenseUnscaled, &gb), 0.0);
    // exxdiv None: the Madelung defect is an identity (v_M = 0): BLIND by
    // construction, asserted so it cannot be mistaken for coverage.
    let fn0 = fixture(&H2, [1, 1, 3], "PBE0", Jk::Dense, ExxDiv::None);
    let b0 = force(&fn0);
    assert_eq!(miss_g(&fn0, G::ExchMadelungUnscaled, &b0), 0.0);
}
