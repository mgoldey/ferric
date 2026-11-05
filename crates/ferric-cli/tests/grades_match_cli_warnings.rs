//! The warning the CLI prints for a `method.kind` must agree with its grade.
//!
//! Grades live in ONE table, [`ferric_core::certificate::METHOD_KIND_GRADES`]
//! (a ferric-core unit test keeps it equal to the Grade column of the
//! `method.kind` matrix on `site/src/reference/validation.md`). The CLI holds
//! only warning TEXT in [`ferric_cli::EPISTEMIC_WARNINGS`]; a kind graded Smoke,
//! Spike or Stub must have a warning, and no other kind may.
//!
//! MUTATIONS this catches:
//! * demoting a kind in the table (e.g. `ccsd` Proven -> Smoke) without adding
//!   a warning -> `cli_warnings_match_the_grade_table` fails (unwarned).
//! * promoting a kind (e.g. `gw` Smoke -> Proven) while the CLI still warns
//!   -> fails (Proven kind warns).
//! * a new `SUPPORTED_METHOD_KINDS` entry with no grade row -> fails.

use ferric_core::certificate::{Grade, METHOD_KIND_GRADES};
use std::collections::BTreeMap;

/// Whether a grade must print an epistemic warning.
fn must_warn(g: Grade) -> bool {
    matches!(g, Grade::Smoke | Grade::Spike | Grade::Stub)
}

/// Every disagreement between the grade table and the CLI's warning table.
fn disagreements(grades: &[(&str, Grade)], warned: &[&str], dispatched: &[&str]) -> Vec<String> {
    let table: BTreeMap<&str, Grade> = grades.iter().copied().collect();
    let mut bad = Vec::new();
    for k in dispatched {
        if !table.contains_key(k) {
            bad.push(format!(
                "`{k}` is dispatched by the CLI but has no grade row"
            ));
        }
    }
    for k in warned {
        if !table.contains_key(k) {
            bad.push(format!("`{k}` is warned by the CLI but has no grade row"));
        }
    }
    for (k, g) in &table {
        let w = warned.contains(k);
        if must_warn(*g) != w {
            bad.push(format!("`{k}`: graded {g}; CLI warns: {w}"));
        }
    }
    bad
}

#[test]
fn cli_warnings_match_the_grade_table() {
    // Negative control: the table must hold every grade class the check
    // distinguishes, or a vacuous table would pass.
    assert!(METHOD_KIND_GRADES.len() >= 20);
    assert!(METHOD_KIND_GRADES.iter().any(|(_, g)| must_warn(*g)));
    assert!(METHOD_KIND_GRADES
        .iter()
        .any(|(_, g)| *g >= Grade::ProvenNarrow));
    assert!(METHOD_KIND_GRADES
        .iter()
        .any(|(_, g)| *g == Grade::Ungraded));

    let warned: Vec<&str> = ferric_cli::EPISTEMIC_WARNINGS
        .iter()
        .map(|(k, _)| *k)
        .collect();
    let bad = disagreements(
        METHOD_KIND_GRADES,
        &warned,
        ferric_cli::SUPPORTED_METHOD_KINDS,
    );
    assert!(
        bad.is_empty(),
        "METHOD_KIND_GRADES and the CLI's EPISTEMIC_WARNINGS disagree:\n  {}",
        bad.join("\n  ")
    );
}

/// The checker must be able to fail, in each direction.
#[test]
fn the_checker_reports_each_kind_of_disagreement() {
    let grades = [
        ("a", Grade::Proven),
        ("b", Grade::ProvenNarrow),
        ("c", Grade::Smoke),
        ("d", Grade::Ungraded),
        ("e", Grade::Spike),
    ];
    assert!(disagreements(&grades, &["c", "e"], &["a", "b", "c", "d", "e"]).is_empty());
    // `e` Spike but unwarned; `d` warned although not graded; `a` Proven but
    // warned; `f` dispatched with no row.
    let bad = disagreements(&grades, &["a", "c", "d"], &["a", "b", "c", "d", "e", "f"]);
    for needle in ["`a`", "`d`", "`e`", "`f`"] {
        assert!(
            bad.iter().any(|m| m.contains(needle)),
            "{needle} not reported: {bad:?}"
        );
    }
    assert_eq!(bad.len(), 4, "{bad:?}");
}
