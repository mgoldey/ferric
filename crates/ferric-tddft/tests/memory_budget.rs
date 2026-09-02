//! Memory-budget gates for the TDDFT dense-response path.
//!
//! Until this landed, `ferric-tddft` had NO budget machinery at all: not one
//! `check_alloc`, not one `resolve_budget`, and no budget field on
//! [`TddftConfig`], so `[memory] budget_gb` and a caller-supplied ceiling were
//! both unreachable from the crate. Meanwhile the Casida branch holds NINE
//! co-resident `(nocc·nvir)²` matrices, and `nocc·nvir` grows as N², so those
//! are N⁴ terms — the steepest allocation in the crate, entirely ungated.
//!
//! The tests here pin the three ways a guard goes wrong:
//!
//!   1. `*_over_budget_*` — it must actually REFUSE, and the message must name
//!      the largest term. A bare "needs 41 GB, have 10 GB" is what made the
//!      historical incidents slow to diagnose.
//!   2. `ample_budget_still_runs_to_completion` — it must NOT refuse a job that
//!      fits. An over-estimating guard is also a bug: it turns "would have run"
//!      into "refused", and that failure mode is invisible unless tested for
//!      directly.
//!   3. `caller_budget_is_honoured_not_discarded` — the budget on the config
//!      must REACH the allocation. The whole class of defect this work targets
//!      is a plumbed-but-ignored budget (`resolve_budget_bytes(None)` deep in
//!      the stack), which no amount of testing the guard in isolation catches.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_tddft::{run_tddft, TddftConfig, TddftMethod};

/// Water / STO-3G: small enough that all four tests together are seconds of
/// RHF, large enough that the dense `dim²` matrices (dim = 5·2 = 10) exceed the
/// starved ceiling below. Every test runs the SAME system; only the budget
/// changes, so a difference in outcome is attributable to the guard and nothing
/// else.
fn fixture() -> (Molecule, PreparedBasis, PreparedBasis, ferric_scf::ScfResult) {
    let xyz = "3\nH2O\nO 0.0 0.0 0.117790\nH 0.0 0.755453 -0.471161\nH 0.0 -0.755453 -0.471161\n";
    let mol = Molecule::parse_xyz(xyz, 0, 1).expect("parse H2O");
    let obs = PreparedBasis::new(&mol, &basis::bundled("sto-3g").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let ctx = ParallelContext::default();
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
    (mol, obs, dfbs, rhf)
}

/// 1 GB — vastly more than water/STO-3G needs (its dense matrices are under a
/// kilobyte each), so anything that fails under this budget is the guard
/// over-charging, not the system being large.
const AMPLE: usize = 1_000_000_000;

/// 1 kB — smaller than any real allocation on the path, so the gate must fire
/// before the first big buffer regardless of system size.
const STARVED: usize = 1_000;

#[test]
fn ample_budget_still_runs_to_completion() {
    // An over-estimating guard is also a bug. This is the direction that does
    // not announce itself: a job that WOULD have fit gets refused, and the only
    // symptom is a confusing error. Assert the ample-budget path runs end to
    // end and returns physical numbers, for BOTH methods — Casida charges 9
    // resident dim² matrices, TDA only 3, and an off-by-one in either count
    // shows up here first.
    let (mol, obs, dfbs, rhf) = fixture();
    for method in [TddftMethod::Tda, TddftMethod::Casida] {
        let cfg = TddftConfig {
            n_roots: 1,
            method,
            memory_budget_bytes: Some(AMPLE),
        };
        let r = run_tddft(&mol, &obs, &dfbs, &rhf, &cfg, 1.0)
            .unwrap_or_else(|e| panic!("{method:?} refused an ample 1 GB budget: {e}"));
        assert!(!r.excitation_energies.is_empty(), "{method:?}: no roots returned");
        assert!(
            r.excitation_energies[0] > 0.0 && r.excitation_energies[0] < 10.0,
            "{method:?}: unphysical excitation energy {:?}",
            r.excitation_energies
        );
    }
}

#[test]
fn casida_over_budget_errors_and_names_the_largest_term() {
    let (mol, obs, dfbs, rhf) = fixture();
    let cfg = TddftConfig {
        n_roots: 1,
        method: TddftMethod::Casida,
        memory_budget_bytes: Some(STARVED),
    };
    let err = run_tddft(&mol, &obs, &dfbs, &rhf, &cfg, 1.0)
        .expect_err("a 1 kB budget must not be enough for the Casida dense matrices")
        .to_string();
    assert!(err.contains("TDDFT/Casida"), "must name the method: {err}");
    assert!(err.contains("memory plan"), "must carry the plan breakdown: {err}");
    // The breakdown sorts largest-first, so the culprit is the first row. With
    // the RI tensors declared, `B(P|ab)` (naux·nvir²) is the biggest term at
    // this shape; whichever it is, SOME named reservation must appear — a bare
    // "needs X, have Y" is exactly the diagnosis-hostile message this replaces.
    assert!(
        err.contains("B(P|ab) [b_vv]") || err.contains("A (ia,jb)"),
        "breakdown must name a reservation, not just a total: {err}"
    );
}

#[test]
fn tda_over_budget_errors_and_names_the_largest_term() {
    let (mol, obs, dfbs, rhf) = fixture();
    let cfg = TddftConfig {
        n_roots: 1,
        method: TddftMethod::Tda,
        memory_budget_bytes: Some(STARVED),
    };
    let err = run_tddft(&mol, &obs, &dfbs, &rhf, &cfg, 1.0)
        .expect_err("a 1 kB budget must not be enough for the TDA dense matrices")
        .to_string();
    assert!(err.contains("TDDFT/TDA"), "must name the method: {err}");
    assert!(err.contains("memory plan"), "must carry the plan breakdown: {err}");
    assert!(
        err.contains("B(P|ab) [b_vv]") || err.contains("A (ia,jb)"),
        "breakdown must name a reservation, not just a total: {err}"
    );
}

#[test]
fn caller_budget_is_honoured_not_discarded() {
    // THE pin for the plumbing, distinct from the guard tests above. Same
    // system, same method, same everything — the ONLY difference between the
    // two runs is `memory_budget_bytes`. If the field were accepted and then
    // dropped on the floor (the `resolve_budget_bytes(None)` defect this crate
    // was full of), both runs would resolve the same env/auto ceiling and both
    // would succeed, and this test would fail.
    let (mol, obs, dfbs, rhf) = fixture();
    let base = TddftConfig { n_roots: 1, method: TddftMethod::Casida, ..Default::default() };

    let ample = TddftConfig { memory_budget_bytes: Some(AMPLE), ..base.clone() };
    assert!(
        run_tddft(&mol, &obs, &dfbs, &rhf, &ample, 1.0).is_ok(),
        "the ample-budget run must succeed, or the comparison below proves nothing"
    );

    let starved = TddftConfig { memory_budget_bytes: Some(STARVED), ..base };
    assert!(
        run_tddft(&mol, &obs, &dfbs, &rhf, &starved, 1.0).is_err(),
        "a caller-supplied 1 kB ceiling was ignored — the budget is not reaching the \
         allocations, which is the entire defect this field exists to fix"
    );
}

/// A hybrid/DFT reference (`c_hf != 1.0`) must still produce a result — the
/// missing-XC-kernel WARNING is a diagnostic, not a failure path.
///
/// `run_tddft` warns on stderr when `c_hf != 1.0` because the `(ia|f_xc|jb)`
/// kernel response is unimplemented, so those excitation energies omit a
/// physical term. That warning exists because the omission was previously
/// SILENT: a caller running TDDFT on a DFT reference got approximate numbers
/// that looked converged and complete.
///
/// This test pins the two things that must remain true: the warning path does
/// not turn a working calculation into an error, and it does not corrupt the
/// result. It deliberately does NOT assert the excitation energy against a
/// reference value — with the kernel term missing, no such reference is
/// meaningful, which is precisely why the warning is there.
#[test]
fn hybrid_c_hf_still_returns_a_result_despite_the_missing_xc_kernel() {
    let (mol, obs, dfbs, rhf) = fixture();
    let cfg = TddftConfig { n_roots: 1, method: TddftMethod::Tda, memory_budget_bytes: Some(AMPLE) };

    // B3LYP-like exact-exchange fraction: exercises the c_hf != 1.0 branch.
    let r = run_tddft(&mol, &obs, &dfbs, &rhf, &cfg, 0.2)
        .expect("a hybrid c_hf must still run; the missing f_xc term is a warning, not an error");

    assert!(!r.excitation_energies.is_empty(), "no roots returned for a hybrid reference");
    assert!(
        r.excitation_energies[0].is_finite(),
        "hybrid c_hf produced a non-finite excitation energy: {:?}",
        r.excitation_energies
    );
}
