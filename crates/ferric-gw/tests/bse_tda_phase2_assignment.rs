//! Phase-2 BSE-TDA benchmark statistic, recomputed with a REAL one-to-one
//! (energy, oscillator-strength) assignment instead of the naive
//! nearest-computed-energy heuristic.
//!
//! This test is the gate on the published number. It reads two committed,
//! static inputs — the TBE reference table
//! (`testdata/reference/thiel_set_subset.json`) and the raw ferric stdout for
//! all ten molecules (`benchmarks/bse-tda-pilot/phase2-logs/*.log`) — and
//! recomputes the MAE/MSE from scratch. It runs NO quantum chemistry: the
//! aug-cc-pVDZ sweep behind those logs is expensive and was already done (see
//! `docs/bse-tda-phase2-results.md`), and the defect being fixed here is in
//! the ASSIGNMENT, not in the computed roots. Re-running the SCF/GW/BSE would
//! not change a single computed (Ω, f) pair.
//!
//! What it pins:
//!
//! 1. The naive baseline still reproduces (MAE 0.482 / MSE +0.450 eV), so the
//!    old and new numbers are known to come from the same underlying data.
//! 2. The naive baseline is demonstrably broken — non-injective on FOUR of
//!    the ten molecules (the published caveat named three; see
//!    `naive_matching_collapses_on_four_molecules_not_the_documented_three`)
//!    and character-mismatched on ethylene.
//! 3. The corrected statistic, and the direction of the correction (UP).
//! 4. The assignment is stable across the intensity weight over a wide window,
//!    and the two states where it is NOT are reported as ambiguous rather than
//!    quietly included.
//!
//! See `crates/ferric-gw/src/assign.rs` for the cost function and the
//! rationale for gating on the intensity channel only.

use ferric_gw::assign::{
    assign_states, assignment_stability_sweep, naive_is_injective, naive_nearest_energy,
    osc_dissimilarity, StateRecord,
};
use std::path::{Path, PathBuf};

/// Intensity weight used for the headline number. Mid-window of the stable
/// plateau established by `assignment_is_stable_across_the_intensity_weight`
/// below; the statistic is reported as a RANGE over that whole window, so this
/// value selects a representative assignment, not the answer.
const W_HEADLINE: f64 = 1.0;

/// Intensity gate. A pair whose `d_f` exceeds this is reported unassigned.
/// 0.30 sits between "same character, different magnitude" (the ok states here
/// top out at 0.22) and "different character" (the flagged ones start at
/// 0.31).
const DF_GATE: f64 = 0.30;

/// Weight window over which the assignment is checked for stability.
const W_WINDOW: [f64; 11] = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0];

fn repo_root() -> PathBuf {
    // CARGO_MANIFEST_DIR = <root>/crates/ferric-gw
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("repo root")
        .to_path_buf()
}

/// One benchmark molecule: its reference states and ferric's computed roots.
struct Case {
    name: String,
    labels: Vec<String>,
    reference: Vec<StateRecord>,
    computed: Vec<StateRecord>,
}

/// Parse `n   Omega(eV)   f_osc` rows out of a ferric BSE-TDA stdout log.
fn parse_log(path: &Path, max_roots: usize) -> Vec<StateRecord> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    let mut out = Vec::new();
    for line in text.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() != 3 {
            continue;
        }
        // Column 0 must be an integer index, 1 and 2 floats. Anything else in
        // the log (headers, warnings, "nocc = 8  nvir = 74") fails this.
        let (Ok(_idx), Ok(e), Ok(fo)) = (
            f[0].parse::<usize>(),
            f[1].parse::<f64>(),
            f[2].parse::<f64>(),
        ) else {
            continue;
        };
        if !f[1].contains('.') || !f[2].contains('.') {
            continue;
        }
        out.push(StateRecord::new(e, fo));
        if out.len() >= max_roots {
            break;
        }
    }
    assert!(
        out.len() >= 5,
        "parsed only {} roots from {path:?} — log format changed?",
        out.len()
    );
    out
}

/// Minimal extraction of the fields this test needs from the reference JSON.
/// Deliberately hand-rolled rather than serde-derived: the file carries a large
/// `_meta` provenance block this test has no business modeling.
fn load_cases(max_roots: usize) -> Vec<Case> {
    let root = repo_root();
    let raw = std::fs::read_to_string(root.join("testdata/reference/thiel_set_subset.json"))
        .expect("read thiel_set_subset.json");
    let v: serde_json::Value = serde_json::from_str(&raw).expect("parse reference json");
    let mols = v["molecules"].as_array().expect("molecules array");
    assert_eq!(mols.len(), 10, "benchmark is 10 molecules");

    let mut cases = Vec::new();
    for m in mols {
        let name = m["name"].as_str().expect("name").to_string();
        let mut labels = Vec::new();
        let mut reference = Vec::new();
        for s in m["states"].as_array().expect("states") {
            labels.push(s["label"].as_str().expect("label").to_string());
            let e = s["excitation_energy_eV"].as_f64().expect("energy");
            // A missing/null f means symmetry-forbidden. Forbidden IS zero
            // intensity — map it to 0.0 rather than inventing a value, and
            // note that `osc_dissimilarity`'s F_FLOOR keeps that from being
            // over-penalized against a numerically-tiny computed f.
            let f = s["oscillator_strength"].as_f64().unwrap_or(0.0);
            reference.push(StateRecord::new(e, f));
        }
        let log = root.join(format!(
            "benchmarks/bse-tda-pilot/phase2-logs/{name}_bse.log"
        ));
        let computed = parse_log(&log, max_roots);
        cases.push(Case {
            name,
            labels,
            reference,
            computed,
        });
    }
    let n_states: usize = cases.iter().map(|c| c.reference.len()).sum();
    assert_eq!(n_states, 17, "benchmark is 17 reference states");
    cases
}

fn mae_mse(diffs: &[f64]) -> (f64, f64) {
    let n = diffs.len() as f64;
    (
        diffs.iter().map(|d| d.abs()).sum::<f64>() / n,
        diffs.iter().sum::<f64>() / n,
    )
}

/// PRECONDITION / PROVENANCE GATE: the naive nearest-energy heuristic must
/// still reproduce the published 0.482 / +0.450 eV from these same logs.
///
/// Without this, a change in the corrected number could not be attributed to
/// the assignment — it could equally be a different log file, a different
/// reference table, or a parser bug. This test ties both numbers to one input.
#[test]
fn naive_baseline_still_reproduces_the_published_0_482_ev() {
    let cases = load_cases(15);
    let mut diffs = Vec::new();
    for c in &cases {
        let picks = naive_nearest_energy(&c.reference, &c.computed);
        for (r, p) in picks.iter().enumerate() {
            let i = p.expect("naive always picks something");
            diffs.push(c.computed[i].energy_ev - c.reference[r].energy_ev);
        }
    }
    assert_eq!(diffs.len(), 17);
    let (mae, mse) = mae_mse(&diffs);
    assert!(
        (mae - 0.482).abs() < 5e-4,
        "naive MAE {mae:.4} != published 0.482 — inputs have drifted"
    );
    assert!(
        (mse - 0.450).abs() < 5e-4,
        "naive MSE {mse:+.4} != published +0.450 — inputs have drifted"
    );
}

/// The DEFECT, pinned — and a correction to the published caveat.
///
/// `docs/bse-tda-phase2-results.md` (and the VALIDATION.md row quoting it)
/// says "all three N-heterocycles (pyrazine/pyridine/pyrimidine) have two
/// reference states collapsing onto one ferric root". That UNDERCOUNTS:
/// CYCLOPROPENE collapses too. Its two reference states (1¹B1 at 6.671 eV and
/// 1¹B2 at 6.729 eV) are both nearest to the same computed root n=1 at 6.997
/// eV — visible in that doc's OWN per-molecule table, which lists "n=1, 6.997"
/// on both cyclopropene rows, but not carried into the prose caveat.
///
/// So the naive heuristic is non-injective on FOUR of ten molecules, not
/// three. This test pins the true set; it was written asserting the
/// documented three and failed, which is how the discrepancy was found.
#[test]
fn naive_matching_collapses_on_four_molecules_not_the_documented_three() {
    let cases = load_cases(15);
    let mut collapsed: Vec<&str> = Vec::new();
    for c in &cases {
        if !naive_is_injective(&c.reference, &c.computed) {
            collapsed.push(&c.name);
        }
    }
    collapsed.sort_unstable();
    assert_eq!(
        collapsed,
        // NOT the three the prose caveat named — cyclopropene is the fourth.
        vec!["cyclopropene", "pyrazine", "pyridine", "pyrimidine"],
        "the collapse set changed"
    );
}

/// The OTHER defect, pinned: ethylene's bright valence π→π\* reference
/// (f=0.346) is assigned by the naive matcher to a DARK computed root, and by
/// the joint-cost assignment to the bright one.
#[test]
fn ethylene_bright_state_moves_from_a_dark_root_to_a_bright_one() {
    let cases = load_cases(15);
    let c = cases
        .iter()
        .find(|c| c.name == "ethylene")
        .expect("ethylene");
    let bright = c
        .labels
        .iter()
        .position(|l| l == "1^1B1u")
        .expect("1^1B1u state");
    assert!(
        (c.reference[bright].osc - 0.346).abs() < 1e-9,
        "reference f changed"
    );

    let naive = naive_nearest_energy(&c.reference, &c.computed);
    let naive_root = naive[bright].expect("naive picks");
    assert!(
        c.computed[naive_root].osc < 1e-6,
        "precondition: naive must pick a DARK root, got f={}",
        c.computed[naive_root].osc
    );

    let new = assign_states(&c.reference, &c.computed, W_HEADLINE, f64::INFINITY);
    let new_root = new[bright].index().expect("assigned");
    assert_ne!(new_root, naive_root, "the assignment must have changed");
    assert!(
        c.computed[new_root].osc > 0.3,
        "the bright reference must take a BRIGHT root, got f={}",
        c.computed[new_root].osc
    );
}

/// STABILITY: the assignment must not be an artifact of the one free parameter.
///
/// Over w ∈ [0.5, 3.0], 15 of the 17 states must hold a single, fixed root.
/// The two that do not are the honest ambiguity this benchmark carries, and
/// they are named here so that a future change to either the cost function or
/// the underlying data has to confront them explicitly.
#[test]
fn assignment_is_stable_across_the_intensity_weight() {
    let cases = load_cases(15);
    let mut ambiguous: Vec<String> = Vec::new();
    let mut stable = 0usize;
    for c in &cases {
        let sweep = assignment_stability_sweep(&c.reference, &c.computed, &W_WINDOW, f64::INFINITY);
        // Per-state: collect the distinct roots this reference state took.
        for r in 0..c.reference.len() {
            let mut roots: Vec<Option<usize>> =
                sweep.per_weight.iter().map(|(_, a)| a[r].index()).collect();
            roots.sort_unstable();
            roots.dedup();
            if roots.len() == 1 {
                stable += 1;
            } else {
                ambiguous.push(format!("{}/{}", c.name, c.labels[r]));
            }
        }
    }
    ambiguous.sort();
    assert_eq!(
        stable, 15,
        "expected 15/17 weight-independent assignments, got {stable} (ambiguous: {ambiguous:?})"
    );
    assert_eq!(
        ambiguous,
        vec!["cyclopropene/1^1B1", "pyrimidine/1^1B2"],
        "the ambiguous set changed"
    );
}

/// TOO-CLEAN STOP CONDITION (project CLAUDE.md): the results doc documents real
/// ambiguity in at least four of ten molecules. An assignment that flags
/// nothing would be evidence the cost function is inert, not evidence the data
/// is clean. This asserts the flagging is non-trivial in BOTH directions:
/// several states are flagged, and most are not.
#[test]
fn flagging_is_neither_vacuous_nor_universal() {
    let cases = load_cases(15);
    let (mut gated, mut kept) = (0usize, 0usize);
    for c in &cases {
        let a = assign_states(&c.reference, &c.computed, W_HEADLINE, DF_GATE);
        for x in &a {
            if x.index().is_some() {
                kept += 1
            } else {
                gated += 1
            }
        }
    }
    assert_eq!(gated + kept, 17);
    assert!(
        gated >= 3,
        "only {gated} states flagged — the documented ambiguity is not being detected"
    );
    assert!(
        kept >= 10,
        "only {kept} states kept — the gate is throwing away the benchmark"
    );
}

/// THE HEADLINE: the corrected statistic, and the fact that it went UP.
///
/// Both numbers are pinned: the all-pairs statistic (directly comparable to
/// the old 0.482 / +0.450 on the same N=17) and the confident subset (the
/// states surviving the intensity gate, with the count stated).
#[test]
fn corrected_statistic_is_worse_than_the_naive_one_and_both_are_pinned() {
    let cases = load_cases(15);
    let (mut all, mut confident) = (Vec::new(), Vec::new());
    for c in &cases {
        let a = assign_states(&c.reference, &c.computed, W_HEADLINE, f64::INFINITY);
        for (r, x) in a.iter().enumerate() {
            let i = x.index().expect("ungated assignment pairs everything");
            let d = c.computed[i].energy_ev - c.reference[r].energy_ev;
            all.push(d);
            if osc_dissimilarity(c.computed[i].osc, c.reference[r].osc) <= DF_GATE {
                confident.push(d);
            }
        }
    }
    assert_eq!(all.len(), 17);
    assert_eq!(
        confident.len(),
        13,
        "13 of 17 states survive the intensity gate"
    );

    let (mae_all, mse_all) = mae_mse(&all);
    let (mae_c, mse_c) = mae_mse(&confident);

    assert!(
        (mae_all - 0.641).abs() < 5e-4,
        "all-pairs MAE {mae_all:.4} != 0.641"
    );
    assert!(
        (mse_all - 0.598).abs() < 5e-4,
        "all-pairs MSE {mse_all:+.4} != +0.598"
    );
    assert!(
        (mae_c - 0.622).abs() < 5e-4,
        "confident MAE {mae_c:.4} != 0.622"
    );
    assert!(
        (mse_c - 0.565).abs() < 5e-4,
        "confident MSE {mse_c:+.4} != +0.565"
    );

    // The DIRECTION is the load-bearing claim: the results doc called 0.482 a
    // LOWER bound, and a correct injective assignment cannot beat a per-state
    // energy argmin. A corrected number BELOW 0.482 would mean the assignment
    // or the parsing is wrong, not that ferric improved.
    assert!(
        mae_all > 0.482,
        "corrected MAE {mae_all:.4} is BELOW the naive lower bound 0.482 — \
         impossible for an injective assignment; the matcher or parser is broken"
    );
    assert!(
        mse_all > 0.450,
        "corrected MSE {mse_all:+.4} is below the naive +0.450"
    );
}

/// The injectivity fix ALONE (w = 0, intensity channel off) already moves the
/// number. Separating the two contributions keeps the writeup honest about
/// which defect cost how much: non-injectivity is worth ~+0.10 eV of MAE
/// before any intensity information is used at all.
#[test]
fn injectivity_alone_accounts_for_a_named_share_of_the_correction() {
    let cases = load_cases(15);
    let mut diffs = Vec::new();
    for c in &cases {
        let a = assign_states(&c.reference, &c.computed, 0.0, f64::INFINITY);
        for (r, x) in a.iter().enumerate() {
            let i = x.index().expect("paired");
            diffs.push(c.computed[i].energy_ev - c.reference[r].energy_ev);
        }
    }
    let (mae, mse) = mae_mse(&diffs);
    assert!(
        (mae - 0.578).abs() < 5e-4,
        "w=0 (injective, f-blind) MAE {mae:.4} != 0.578"
    );
    assert!((mse - 0.556).abs() < 5e-4, "w=0 MSE {mse:+.4} != +0.556");
    assert!(
        mae > 0.482 && mae < 0.641,
        "the injectivity-only number must sit strictly between naive and full"
    );
}

/// ROBUSTNESS: the number of candidate computed roots offered to the matcher
/// is an implementation detail, not a parameter. The result must be identical
/// whether the matcher sees 5 roots or 50.
///
/// If this ever fails, some reference state is reaching far up the computed
/// spectrum, which is the broken-cost signature (chasing intensity across a
/// large energy gap) and must be investigated before any number is quoted.
#[test]
fn statistic_is_independent_of_the_candidate_pool_size() {
    let mut reference_diffs: Option<Vec<f64>> = None;
    for &n_cand in &[5usize, 10, 15, 20, 30, 50] {
        let cases = load_cases(n_cand);
        let mut diffs = Vec::new();
        for c in &cases {
            let a = assign_states(&c.reference, &c.computed, W_HEADLINE, f64::INFINITY);
            for (r, x) in a.iter().enumerate() {
                let i = x.index().expect("paired");
                diffs.push(c.computed[i].energy_ev - c.reference[r].energy_ev);
            }
        }
        match &reference_diffs {
            None => reference_diffs = Some(diffs),
            Some(r0) => assert_eq!(
                &diffs, r0,
                "assignment changed at candidate-pool size {n_cand} — a reference \
                 state is reaching deep into the computed spectrum"
            ),
        }
    }
}
