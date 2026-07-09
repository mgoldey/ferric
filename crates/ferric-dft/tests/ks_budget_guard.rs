//! Memory-budget fail-fast in `KsXc::new` / `KsXcUks::new`.
//!
//! The resident χ + ∇χ grid cache is 4·nbf·npts·8 bytes (×2 with VV10's NLC
//! grid). When `FERRIC_ERI3_BUDGET_GB` says that cannot fit, construction must
//! return `KsXcError::OverBudget` (with the numbers in the message) instead of
//! letting the allocation abort the process.
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
fn over_budget_fails_fast_and_under_budget_succeeds() {
    let mol = h2o();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let main = AtomicGridConfig::default();
    let nlc = AtomicGridConfig { n_radial: 50, n_angular: 50 };

    // Tiny budget: even H2O/cc-pVDZ's cache (a few MB) cannot fit 1e-6 GB.
    std::env::set_var("FERRIC_ERI3_BUDGET_GB", "1e-6");

    let err = KsXc::new(&mol, &bs, "B3LYP", &main, &nlc)
        .err()
        .expect("KsXc::new must fail under a 1 kB budget");
    match &err {
        KsXcError::OverBudget { needed_gb, budget_gb, nbf, npts, .. } => {
            assert!(*needed_gb > *budget_gb);
            assert!(*nbf > 0 && *npts > 0);
            let msg = format!("{err}");
            assert!(msg.contains("GB"), "message should carry the numbers: {msg}");
        }
        other => panic!("expected OverBudget, got: {other:?}"),
    }

    // UKS path takes the same guard.
    assert!(
        matches!(
            KsXcUks::new(&mol, &bs, "B3LYP", &main, &nlc),
            Err(KsXcError::OverBudget { .. })
        ),
        "KsXcUks::new must fail under a 1 kB budget"
    );

    // VV10 functionals double the estimate; still over budget here, and the
    // message should say the NLC grid participates.
    let err = KsXc::new(&mol, &bs, "wB97X-V", &main, &nlc)
        .err()
        .expect("VV10 KsXc::new must fail under a 1 kB budget");
    assert!(format!("{err}").contains("VV10"), "VV10 flagged in: {err}");

    // Generous budget: construction succeeds (and unset = unlimited works too).
    std::env::set_var("FERRIC_ERI3_BUDGET_GB", "64");
    KsXc::new(&mol, &bs, "B3LYP", &main, &nlc)
        .expect("KsXc::new must succeed under a 64 GB budget");

    std::env::remove_var("FERRIC_ERI3_BUDGET_GB");
    KsXcUks::new(&mol, &bs, "B3LYP", &main, &nlc)
        .expect("KsXcUks::new must succeed with no budget set");
}
