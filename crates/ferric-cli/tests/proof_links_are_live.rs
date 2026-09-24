//! Every validation row's proof link resolves, and every validation test is
//! linked (validation-campaign design §2.5, next to the §2.1 tier guard in
//! `ci_validation_tier_guard.rs`).
//!
//! A grade on `site/src/reference/validation.md` is a claim; its Proof column
//! links to the committed evidence as
//! `https://github.com/mgoldey/ferric/blob/main/<path>` (file-level, pinned to
//! `main`). Both sets are DERIVED — the links from the page's markdown, the
//! validation files from the filesystem — never hand-listed, and there is no
//! allowlist (allowlists rot).
//!
//! MUTATIONS this catches:
//! * renaming or deleting a linked file, e.g.
//!   `crates/ferric-scf/tests/validation_open_shell_scf.rs` →
//!   `validation_uhf.rs` without updating the page →
//!   `every_proof_link_resolves` fails (dead link), AND
//!   `every_validation_file_is_linked` fails (the new name is unlinked).
//! * deleting a row's proof link from the page →
//!   `every_validation_file_is_linked` fails.
//! * a `#L123` line anchor or a link pinned to a branch/SHA other than `main`
//!   → `every_proof_link_resolves` fails (both rot silently).
//!
//! NOTE: docs-only PRs (`site/**`) skip CI by design (ci.yml `paths-ignore`),
//! so a page edit that breaks a link is caught on the next code PR or the
//! nightly run, not on the docs PR itself.

use std::path::{Path, PathBuf};

const PAGE: &str = "site/src/reference/validation.md";
const REPO_URL: &str = "https://github.com/mgoldey/ferric/";

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

/// Every `blob/` or `tree/` URL on the page that points into this repo, as
/// written (up to the first markdown/HTML/table terminator).
fn repo_links(page: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = page;
    while let Some(at) = rest.find(REPO_URL) {
        let tail = &rest[at..];
        let end = tail
            .find(|c: char| {
                c.is_whitespace() || matches!(c, ')' | ']' | '>' | '<' | '"' | '\'' | '|' | '`')
            })
            .unwrap_or(tail.len());
        let url = &tail[..end];
        // Only file/directory links are proof links; issue, PR and Actions
        // links (`issues/..`, `pull/..`) are ordinary prose references.
        let after = &url[REPO_URL.len()..];
        if after.starts_with("blob/") || after.starts_with("tree/") {
            out.push(url.to_string());
        }
        rest = &tail[end..];
    }
    out
}

/// A link resolved to a repo-relative path, or the reason it is invalid.
fn link_path(url: &str) -> Result<String, String> {
    let after = &url[REPO_URL.len()..];
    let path = after
        .strip_prefix("blob/main/")
        .or_else(|| after.strip_prefix("tree/main/"))
        .ok_or_else(|| {
            format!("{url}: proof links must be `blob/main/<path>` (or `tree/main/<dir>`), pinned to main")
        })?;
    let (path, frag) = match path.split_once('#') {
        Some((p, f)) => (p, Some(f)),
        None => (path, None),
    };
    if let Some(f) = frag {
        if f.starts_with('L') && f[1..].starts_with(|c: char| c.is_ascii_digit()) {
            return Err(format!(
                "{url}: line anchors (#L..) rot on every edit; link the file"
            ));
        }
    }
    if path.is_empty() {
        return Err(format!("{url}: links the repo root, not a proof file"));
    }
    Ok(path.trim_end_matches('/').to_string())
}

fn page_text() -> String {
    let p = workspace_root().join(PAGE);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

/// Every data row of the page's `## Anchors` table, as (capability, proof
/// cell). The Proof cell is the LAST column.
fn anchor_rows(page: &str) -> Vec<(String, String)> {
    let Some(start) = page.find("## Anchors") else {
        return Vec::new();
    };
    page[start..]
        .lines()
        .skip_while(|l| !l.trim_start().starts_with('|'))
        .take_while(|l| l.trim_start().starts_with('|'))
        .skip(2) // header row and the |---| separator
        .map(|l| {
            let cells: Vec<&str> = l
                .trim()
                .trim_matches('|')
                .split('|')
                .map(str::trim)
                .collect();
            (
                cells.first().copied().unwrap_or("").to_string(),
                cells.last().copied().unwrap_or("").to_string(),
            )
        })
        .collect()
}

/// Rows whose Proof cell does not carry a live `blob/main` / `tree/main`
/// link. A `″` (ditto) cell inherits the previous row's proof; a ditto with
/// no valid row above it is itself a problem.
fn anchor_row_problems(page: &str, root: &Path) -> Vec<String> {
    let mut problems = Vec::new();
    let mut previous_ok = false;
    for (cap, proof) in anchor_rows(page) {
        if proof == "″" || proof == "\"" {
            if !previous_ok {
                problems.push(format!("`{cap}`: ditto proof with no proven row above it"));
            }
            continue;
        }
        let live = repo_links(&proof)
            .iter()
            .filter_map(|u| link_path(u).ok())
            .any(|p| root.join(p).exists());
        if !live {
            problems.push(format!("`{cap}`: no live proof link in `{proof}`"));
        }
        previous_ok = live;
    }
    problems
}

#[test]
fn every_anchor_row_has_a_proof_link() {
    let page = page_text();
    let rows = anchor_rows(&page);
    assert!(
        rows.len() >= 5,
        "found only {} Anchors rows on {PAGE}; the table parser is not seeing the table",
        rows.len()
    );
    let problems = anchor_row_problems(&page, &workspace_root());
    assert!(
        problems.is_empty(),
        "Anchors rows without a proof link on {PAGE}:\n  {}",
        problems.join("\n  ")
    );
}

/// The row checker must be able to fail: a row with only prose in its Proof
/// cell, and a ditto row under it, are both reported. Deleting a row's link
/// on the real page fails `every_anchor_row_has_a_proof_link` the same way.
#[test]
fn the_anchor_row_checker_rejects_a_row_without_proof() {
    let root = workspace_root();
    let good = format!("[`x`]({REPO_URL}blob/main/Cargo.toml)");
    let page = format!(
        "## Anchors\n\n| Capability | Proof |\n|---|---|\n| proven | {good} |\n| inherits | ″ |\n\
         | unproven | `some_test_name` |\n| orphan ditto | ″ |\n"
    );
    let problems = anchor_row_problems(&page, &root);
    assert_eq!(problems.len(), 2, "{problems:?}");
    assert!(problems[0].contains("`unproven`"), "{problems:?}");
    assert!(problems[1].contains("`orphan ditto`"), "{problems:?}");
}

#[test]
fn every_proof_link_resolves() {
    let root = workspace_root();
    let mut bad = Vec::new();
    for url in repo_links(&page_text()) {
        match link_path(&url) {
            Ok(path) if root.join(&path).exists() => {}
            Ok(path) => bad.push(format!("{url}: `{path}` does not exist in the tree")),
            Err(e) => bad.push(e),
        }
    }
    assert!(
        bad.is_empty(),
        "dead or malformed proof links on {PAGE}:\n  {}",
        bad.join("\n  ")
    );
}

#[test]
fn every_validation_file_is_linked() {
    let root = workspace_root();
    // Only links in an Anchors Proof cell count: a validation file linked
    // from prose (while its row's Proof cell keeps only the generator) is not
    // the proof of any row.
    let linked: std::collections::BTreeSet<String> = anchor_rows(&page_text())
        .iter()
        .flat_map(|(_, proof)| repo_links(proof))
        .filter_map(|u| link_path(&u).ok())
        .collect();
    let mut on_disk = Vec::new();
    for c in std::fs::read_dir(root.join("crates")).expect("read crates/") {
        let tests = c.unwrap().path().join("tests");
        let Ok(rd) = std::fs::read_dir(&tests) else {
            continue;
        };
        for f in rd {
            let f = f.unwrap().path();
            let stem = f
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned();
            if f.is_file()
                && f.extension().is_some_and(|e| e == "rs")
                && stem.starts_with("validation_")
            {
                on_disk.push(
                    f.strip_prefix(&root)
                        .unwrap()
                        .to_string_lossy()
                        .replace('\\', "/"),
                );
            }
        }
    }
    on_disk.sort();
    assert!(
        !on_disk.is_empty(),
        "no crates/*/tests/validation_*.rs found — the walk is broken or the tier is empty"
    );
    let unlinked: Vec<&String> = on_disk.iter().filter(|p| !linked.contains(*p)).collect();
    assert!(
        unlinked.is_empty(),
        "validation test files not linked from any Anchors Proof cell on {PAGE} (add a \
         {REPO_URL}blob/main/<path> link to the Proof cell of the row it proves):\n  {}",
        unlinked
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join("\n  ")
    );
}

/// The link parser itself must be able to fail (a parser that finds nothing
/// makes both tests above vacuous).
#[test]
fn the_link_parser_sees_and_rejects_what_it_should() {
    let md = format!(
        "| row | [t]({REPO_URL}blob/main/crates/x/tests/validation_a.rs) | \
         <{REPO_URL}blob/main/scripts/validation/gen_a.py#usage> \
         {REPO_URL}blob/main/y.rs#L12 {REPO_URL}blob/feat/z.rs \
         see [#12]({REPO_URL}issues/12) |"
    );
    let links = repo_links(&md);
    assert_eq!(
        links.len(),
        4,
        "the issues/ link must be skipped: {links:?}"
    );
    assert_eq!(
        link_path(&links[0]).unwrap(),
        "crates/x/tests/validation_a.rs"
    );
    assert_eq!(link_path(&links[1]).unwrap(), "scripts/validation/gen_a.py");
    assert!(link_path(&links[2]).is_err(), "#L anchors must be rejected");
    assert!(
        link_path(&links[3]).is_err(),
        "non-main refs must be rejected"
    );
}
