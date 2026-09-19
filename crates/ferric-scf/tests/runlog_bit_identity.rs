//! The one non-negotiable property of [`ferric_scf::runlog`]: logging is
//! OBSERVATION, never PARTICIPATION. Turning the JSON run log on must not move
//! a single bit of a single energy.
//!
//! Per this project's "EXACTNESS ANCHOR FIRST" convention these tests were
//! written and made to pass BEFORE any solver was wired to emit a record. The
//! anchor here is not a trivial limit of an approximation but its exact
//! analogue: the feature must be invisible to the number.
//!
//! # Why one process, log-off first
//!
//! The sink is a process-global `OnceLock`, so a test binary has exactly one
//! logging state for its whole life. The single test below therefore runs its
//! reference SCFs *before* installing a sink and the comparison SCFs *after* —
//! a real off-then-on comparison inside one process, with no cross-process
//! float variation to explain away. It also runs the reference twice, to pin
//! the baseline (two log-off runs agree bit-for-bit) so that a failure can only
//! mean the log moved the number.
//!
//! It is ONE test, not three, because cargo runs a binary's tests
//! concurrently: a sibling test could observe either logging state, and the
//! ordering the comparison depends on would be a coin flip. It runs in its own
//! binary (one integration test file = one binary) so installing a global sink
//! cannot leak into another test file's expectations.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::{runlog, solve_rohf, solve_uhf};

fn water() -> (Molecule, ferric_core::basis::BasisSet) {
    let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
    let bs = basis::bundled("sto-3g").unwrap();
    (mol, bs)
}

/// An open-shell reference for the UHF/ROHF channels: the OH radical
/// (doublet), built inline so the test does not depend on a testdata file
/// that may not exist.
fn oh_doublet() -> (Molecule, ferric_core::basis::BasisSet) {
    let mol = Molecule::parse_xyz("2\nOH\nO 0.0 0.0 0.0\nH 0.0 0.0 0.97\n", 0, 2).unwrap();
    let bs = basis::bundled("sto-3g").unwrap();
    (mol, bs)
}

/// Every energy this file compares, from one closed- and one open-shell SCF.
/// Bundled so a single `assert_eq!` on the whole vector catches a drift in any
/// channel, not just the first.
fn all_energies() -> Vec<(&'static str, u64)> {
    let ctx = ParallelContext::default();
    let mut out = Vec::new();

    let (mol, bs) = water();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let cfg = RhfConfig::default();

    let rhf = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
    out.push(("rhf", rhf.energy.to_bits()));

    // Closed-shell KS-DFT is `solve_rhf` with `xc` set — the same loop, so the
    // same emit site, but it exercises the XC branch that also touches the
    // iteration body.
    let mut ks_cfg = RhfConfig::default();
    ks_cfg.xc = Some("LDA".to_string());
    let ks = solve_rhf(&ctx, &mol, &prep, op, &bounds, &ks_cfg).unwrap();
    out.push(("rks-lda", ks.energy.to_bits()));

    let (omol, obs) = oh_doublet();
    let oprep = PreparedBasis::new(&omol, &obs).unwrap();
    let obounds = SchwarzBounds::compute(op, &oprep).unwrap();

    let uhf = solve_uhf(&ctx, &omol, &oprep, &obounds, &RhfConfig::default()).unwrap();
    out.push(("uhf", uhf.energy.to_bits()));

    let rohf = solve_rohf(&ctx, &omol, &oprep, op, &obounds, &RhfConfig::default()).unwrap();
    out.push(("rohf", rohf.energy.to_bits()));

    out
}

/// THE test: identical energies, bit for bit, with the run log installed.
///
/// The reference runs first (no sink installed yet), then a sink is installed
/// into a temp file and the same four SCFs run again. Both halves execute the
/// same code; the only difference is whether `runlog::log()` returns `Some`,
/// which is exactly the condition every emit site is gated on.
///
/// Mutation-tested: making any emit site mutate the value it logs (e.g.
/// rounding `energy` and feeding it back) fails this test.
#[test]
fn logging_does_not_move_any_scf_energy() {
    // --- reference: no sink installed ---
    assert!(
        runlog::log().is_none(),
        "a sink was installed before the reference run; this test must run first \
         in its binary (it is the only test here that installs one)"
    );
    let before = all_energies();

    // Baseline, inside this same test rather than a sibling one: the log-OFF
    // path is itself bit-reproducible. Without it a failure below could be
    // blamed on ordinary run-to-run float variation rather than on the log.
    // It lives here and not in its own `#[test]` because the sink is a
    // process-global `OnceLock` and cargo runs tests in one binary
    // concurrently — a sibling test could observe either state.
    let before_again = all_energies();
    assert_eq!(
        before, before_again,
        "the SCF is not bit-reproducible run to run, so the comparison below \
         could not attribute a difference to the log"
    );

    // --- comparison: sink installed, records really being written ---
    let dir = std::env::temp_dir().join(format!("ferric-runlog-bitid-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("run.jsonl");
    assert!(runlog::init(&path), "could not install the run log");
    assert!(runlog::log().is_some(), "sink did not install");

    let after = all_energies();

    assert_eq!(
        before, after,
        "an SCF energy changed when the JSON run log was turned on — logging is \
         observation, not participation"
    );

    // The log must actually have been written, or this test proves nothing:
    // a sink that silently dropped every record would also leave the energies
    // untouched. Require per-iteration records from every channel above.
    let text = std::fs::read_to_string(&path).expect("run log was not created");
    let iter_lines: Vec<&str> = text
        .lines()
        .filter(|l| l.contains("\"record\":\"scf_iter\""))
        .collect();
    assert!(
        iter_lines.len() > 10,
        "expected many scf_iter records from four SCF runs, got {} — the \
         bit-identity assertion above is vacuous if nothing was logged",
        iter_lines.len()
    );
    for method in ["rhf", "uhf", "rohf"] {
        assert!(
            text.contains(&format!("\"method\":\"{method}\"")),
            "no scf_iter records for {method}; that channel's emit site is missing, \
             so this test does not cover it"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}
