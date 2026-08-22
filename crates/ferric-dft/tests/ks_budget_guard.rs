//! Memory-budget behavior in `KsXc::new` / `KsXcUks::new`.
//!
//! The resident χ + ∇χ grid cache is 4·nbf·npts·8 bytes (×2 with VV10's NLC
//! grid). When `FERRIC_ERI3_BUDGET_GB` says the *main*-grid cache alone
//! cannot fit, construction no longer fails: it falls back to a batched
//! per-iteration evaluation (never materializing the full cache) — see
//! `ks.rs`'s `GridCache::Batched`. VV10 is the one exception: its NLC grid's
//! own O(npts²) pair sum needs its own cache fully resident regardless (not
//! batchable), so an over-budget VV10 functional still returns
//! `KsXcError::OverBudget`.
//!
//! Single test fn in its own integration-test binary: the env-var mutation
//! must not race other tests that construct `KsXc` in the same process.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_dft::grid::AtomicGridConfig;
use ferric_dft::ks::{KsXc, KsXcError, KsXcUks};

fn h2o() -> Molecule {
    Molecule::parse_xyz(
        "3\nH2O\nO 0 0 0\nH 0 0.7572 0.5868\nH 0 -0.7572 0.5868\n", 0, 1,
    ).unwrap()
}

#[test]
fn over_budget_batches_instead_of_failing_and_under_budget_uses_full_cache() {
    let mol = h2o();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let main = AtomicGridConfig::default();
    let nlc = AtomicGridConfig { n_radial: 50, n_angular: 50, ..Default::default() };

    // Tiny budget: even H2O/cc-pVDZ's cache (a few MB) cannot fit 1e-6 GB.
    // A non-VV10 functional must NOT fail — it falls back to batching.
    std::env::set_var("FERRIC_ERI3_BUDGET_GB", "1e-6");

    KsXc::new(&mol, &bs, "B3LYP", &main, &nlc)
        .expect("KsXc::new must NOT fail under a tiny budget (batched fallback instead)");

    // UKS path takes the same fallback.
    KsXcUks::new(&mol, &bs, "B3LYP", &main, &nlc)
        .expect("KsXcUks::new must NOT fail under a tiny budget (batched fallback instead)");

    // VV10 functionals: the NLC grid's cache (not batchable — see vv10.rs's
    // O(npts²) pair sum) must still fit in the budget on its own, or this is
    // a hard failure with the message flagging VV10.
    let err = KsXc::new(&mol, &bs, "wB97X-V", &main, &nlc)
        .expect_err("VV10 KsXc::new must still fail under a 1 kB budget (NLC grid isn't batchable)");
    match &err {
        KsXcError::OverBudget { needed_gb, budget_gb, nbf, npts, .. } => {
            assert!(*needed_gb > *budget_gb);
            assert!(*nbf > 0 && *npts > 0);
            let msg = format!("{err}");
            assert!(msg.contains("GB"), "message should carry the numbers: {msg}");
        }
        other => panic!("expected OverBudget, got: {other:?}"),
    }
    assert!(format!("{err}").contains("VV10"), "VV10 flagged in: {err}");

    // Generous budget: construction succeeds via the Full cache path (and
    // unset = unlimited works too).
    std::env::set_var("FERRIC_ERI3_BUDGET_GB", "64");
    KsXc::new(&mol, &bs, "B3LYP", &main, &nlc)
        .expect("KsXc::new must succeed under a 64 GB budget");

    std::env::remove_var("FERRIC_ERI3_BUDGET_GB");
    KsXcUks::new(&mol, &bs, "B3LYP", &main, &nlc)
        .expect("KsXcUks::new must succeed with no budget set");
}
