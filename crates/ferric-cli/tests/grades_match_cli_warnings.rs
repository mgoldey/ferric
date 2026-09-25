//! The grade the published page gives each `method.kind` and the warning the
//! CLI prints for it must agree.
//!
//! The page (`site/src/reference/validation.md`, the `## CLI `method.kind`
//! matrix`) states a grade per kind in its Grade column. The CLI encodes the
//! same grades as [`ferric_cli::PROVEN_METHOD_KINDS`] (silent) and
//! [`ferric_cli::EPISTEMIC_WARNINGS`] (prints a `[warning]`); a kind in
//! neither is "not graded" and is also silent. Both sides are read from their
//! source of truth -- the page's markdown and the crate's public consts --
//! never hand-listed here.
//!
//! MUTATIONS this catches:
//! * demoting a kind on the page (e.g. `ccsd` Proven -> Smoke) without adding
//!   a warning -> `page_grades_match_the_cli` fails (Smoke kind unwarned).
//! * promoting a kind on the page (e.g. `gw` Smoke -> Proven) while the CLI
//!   still warns -> fails (Proven kind warns / not in the Proven list).
//! * a new `SUPPORTED_METHOD_KINDS` entry with no matrix row -> fails.
//! * a kind in both the Proven list and the warning table -> fails.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

const PAGE: &str = "site/src/reference/validation.md";
const MATRIX_HEADING: &str = "## CLI `method.kind` matrix";

/// Workspace root, resolved at RUN time -- see `ccsd_dispatch.rs` for why
/// `env!("CARGO_MANIFEST_DIR")` is wrong under `cargo nextest archive`.
fn workspace_root() -> PathBuf {
    let looks_like_root = |p: &Path| {
        p.join("Cargo.toml").is_file() && p.join("site").is_dir() && p.join("crates").is_dir()
    };
    if let Ok(cwd) = std::env::current_dir() {
        let mut here: Option<&Path> = Some(cwd.as_path());
        while let Some(p) = here {
            if looks_like_root(p) {
                return p.to_path_buf();
            }
            here = p.parent();
        }
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("ferric-cli manifest dir should be workspace_root/crates/ferric-cli")
        .to_path_buf()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Class {
    /// "Proven", "Proven (narrow)", "Proven (narrow, ...)".
    Proven,
    /// "Smoke" or "Spike": the CLI must warn.
    Warned,
    /// "not graded": in neither CLI list.
    NotGraded,
}

/// `[text](target)` -> `text`; anything else unchanged.
fn strip_link(cell: &str) -> &str {
    let c = cell.trim();
    if let (Some(rest), Some(close)) = (c.strip_prefix('['), c.find("](")) {
        if c.ends_with(')') {
            return &rest[..close - 1];
        }
    }
    c
}

fn classify(grade: &str) -> Result<Class, String> {
    let g = strip_link(grade).trim_matches('*').trim();
    if g == "Proven" || g.starts_with("Proven (narrow") {
        Ok(Class::Proven)
    } else if g == "Smoke" || g == "Spike" {
        Ok(Class::Warned)
    } else if g == "not graded" {
        Ok(Class::NotGraded)
    } else {
        Err(format!("unrecognised grade {grade:?}"))
    }
}

/// Split a markdown table row into cells, honouring `\|` escapes.
fn cells(row: &str) -> Vec<String> {
    row.trim()
        .trim_matches('|')
        .replace("\\|", "\u{0}")
        .split('|')
        .map(|c| c.replace('\u{0}', "|").trim().to_string())
        .collect()
}

/// kind -> grade cell, from the table under [`MATRIX_HEADING`]. The Grade
/// column is located by its header, not by position.
fn matrix_grades(page: &str) -> Result<BTreeMap<String, String>, String> {
    let start = page
        .find(MATRIX_HEADING)
        .ok_or_else(|| format!("no `{MATRIX_HEADING}` heading"))?;
    let mut rows = page[start..]
        .lines()
        .skip_while(|l| !l.trim_start().starts_with('|'))
        .take_while(|l| l.trim_start().starts_with('|'));
    let header = cells(rows.next().ok_or("matrix has no header row")?);
    let grade_col = header
        .iter()
        .position(|h| h == "Grade")
        .ok_or_else(|| format!("matrix header has no `Grade` column: {header:?}"))?;
    let mut out = BTreeMap::new();
    for row in rows.skip(1) {
        let c = cells(row);
        let kind = c[0]
            .strip_prefix('`')
            .and_then(|k| k.strip_suffix('`'))
            .ok_or_else(|| format!("first cell is not a `kind`: {:?}", c[0]))?;
        let grade = c
            .get(grade_col)
            .ok_or_else(|| format!("row `{kind}` has no Grade cell"))?;
        if out.insert(kind.to_string(), grade.clone()).is_some() {
            return Err(format!("kind `{kind}` has two matrix rows"));
        }
    }
    Ok(out)
}

/// Every disagreement between the page's grades and the CLI's two lists.
fn disagreements(
    page_grades: &BTreeMap<String, String>,
    proven: &[&str],
    warned: &[&str],
    dispatched: &[&str],
) -> Vec<String> {
    let proven: BTreeSet<&str> = proven.iter().copied().collect();
    let warned: BTreeSet<&str> = warned.iter().copied().collect();
    let mut bad = Vec::new();
    for k in proven.intersection(&warned) {
        bad.push(format!(
            "`{k}` is in both PROVEN_METHOD_KINDS and EPISTEMIC_WARNINGS"
        ));
    }
    for k in dispatched {
        if !page_grades.contains_key(*k) {
            bad.push(format!(
                "`{k}` is dispatched by the CLI but has no matrix row"
            ));
        }
    }
    for k in proven.iter().chain(warned.iter()) {
        if !page_grades.contains_key(*k) {
            bad.push(format!("`{k}` is graded by the CLI but has no matrix row"));
        }
    }
    for (kind, grade) in page_grades {
        let k = kind.as_str();
        let (in_p, in_w) = (proven.contains(k), warned.contains(k));
        match classify(grade) {
            Err(e) => bad.push(format!("`{kind}`: {e}")),
            Ok(Class::Proven) if !in_p || in_w => bad.push(format!(
                "`{kind}`: page says {grade:?}; CLI Proven list: {in_p}, CLI warns: {in_w}"
            )),
            Ok(Class::Warned) if !in_w || in_p => bad.push(format!(
                "`{kind}`: page says {grade:?}; CLI warns: {in_w}, CLI Proven list: {in_p}"
            )),
            Ok(Class::NotGraded) if in_p || in_w => bad.push(format!(
                "`{kind}`: page says not graded; CLI Proven list: {in_p}, CLI warns: {in_w}"
            )),
            Ok(_) => {}
        }
    }
    bad
}

#[test]
fn page_grades_match_the_cli() {
    let path = workspace_root().join(PAGE);
    let page = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let grades = matrix_grades(&page).unwrap_or_else(|e| panic!("{PAGE}: {e}"));

    // Negative control: an empty or mis-located parse must not pass.
    assert!(
        grades.len() >= 20,
        "parsed only {} matrix rows from {PAGE}: {grades:?}",
        grades.len()
    );
    let classes: BTreeSet<Class> = grades.values().filter_map(|g| classify(g).ok()).collect();
    for c in [Class::Proven, Class::Warned, Class::NotGraded] {
        assert!(
            classes.contains(&c),
            "no {c:?} kind parsed from {PAGE}; the Grade column is not being read"
        );
    }

    let warned: Vec<&str> = ferric_cli::EPISTEMIC_WARNINGS
        .iter()
        .map(|(k, _)| *k)
        .collect();
    let bad = disagreements(
        &grades,
        ferric_cli::PROVEN_METHOD_KINDS,
        &warned,
        ferric_cli::SUPPORTED_METHOD_KINDS,
    );
    assert!(
        bad.is_empty(),
        "grades on {PAGE} and the CLI's PROVEN_METHOD_KINDS / EPISTEMIC_WARNINGS disagree:\n  {}",
        bad.join("\n  ")
    );
}

/// The checker must be able to fail, in each direction.
#[test]
fn the_checker_reports_each_kind_of_disagreement() {
    let page = format!(
        "{MATRIX_HEADING}\n\n| `method.kind` | Family | Grade | Caveat |\n|---|---|---|---|\n\
         | `a` | x | [Proven](#anchors) | — |\n\
         | `b` | x | Proven (narrow, closed shell) | — |\n\
         | `c` | x | Smoke | y |\n\
         | `d` | x | not graded | y |\n\
         | `e` | x | Spike | y |\n"
    );
    let grades = matrix_grades(&page).unwrap();
    assert_eq!(grades.len(), 5, "{grades:?}");
    // Consistent: no complaints.
    assert!(disagreements(
        &grades,
        &["a", "b"],
        &["c", "e"],
        &["a", "b", "c", "d", "e"]
    )
    .is_empty());
    // `b` missing from the Proven list; `e` Spike but unwarned; `d` warned
    // although not graded; `f` dispatched with no row.
    let bad = disagreements(
        &grades,
        &["a"],
        &["c", "d"],
        &["a", "b", "c", "d", "e", "f"],
    );
    for needle in ["`b`", "`d`", "`e`", "`f`"] {
        assert!(
            bad.iter().any(|m| m.contains(needle)),
            "{needle} not reported: {bad:?}"
        );
    }
    assert_eq!(bad.len(), 4, "{bad:?}");
}
