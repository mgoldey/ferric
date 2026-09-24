//! The validation tier cannot silently go unrun, or silently run per-commit.
//!
//! Validation tests (ferric vs committed external references; design in the
//! wiki at superpowers/specs/2026-09-24-validation-campaign-design.md §2.1)
//! live in `crates/<crate>/tests/validation_<topic>.rs`, every `#[test]`
//! marked `#[ignore = "validation: <VALIDATION.md row>"]`. The per-commit
//! nextest shards skip ignored tests; the weekly `validation` job in
//! `.github/workflows/ci.yml` runs them with
//! `--run-ignored only -E 'binary(/^validation_/)'`.
//!
//! That arrangement has four ways to fail QUIETLY, and this regular-tier test
//! pins each by DERIVING the file set from the filesystem (a hand-listed set
//! lets a new file go unrun — the derive-the-guard-set lesson) and checking it
//! against the job's own text:
//!
//! * a validation file the job's filter or build glob does not match — its
//!   tests are ignored per-commit AND skipped weekly, i.e. never run.
//!   MUTATION: change the job's filter to `binary(/^validation_scf/)`, or
//!   rename a file to `validate_foo.rs` while it still carries
//!   `validation:` ignores → `every_validation_file_is_selected_by_the_job`
//!   / `validation_ignores_live_only_in_validation_files` fail.
//! * a test in a validation file WITHOUT the `validation:` ignore — it then
//!   runs in every per-commit shard (minutes of SCF on each PR), which is how
//!   a tier boundary erodes. MUTATION: delete one
//!   `#[ignore = "validation: UHF / ROHF"]` line in
//!   `crates/ferric-scf/tests/validation_open_shell_scf.rs` →
//!   `every_validation_test_carries_a_validation_ignore` fails.
//! * the job's `if:` keyed on a cron string that no `schedule:` entry fires —
//!   the job then never runs on a schedule at all. MUTATION: edit the `if:` to
//!   `'17 5 * * 0'` → `the_job_is_weekly_and_manual_only` fails.
//! * another job passing `--run-ignored` — validation tests then run
//!   per-commit. MUTATION: add `--run-ignored all` to the test-shard run →
//!   `no_other_job_runs_ignored_tests` fails.

use std::path::{Path, PathBuf};

/// The naming CONVENTION (independent of what ci.yml says): the guard derives
/// the file set from this, then checks the job's filter against it.
const CONVENTION_PREFIX: &str = "validation_";
/// The ignore-reason prefix every validation `#[test]` must carry.
const IGNORE_ATTR_PREFIX: &str = "#[ignore = \"validation: ";

/// Workspace root, resolved at RUN time -- see `ccsd_dispatch.rs` for why
/// `env!("CARGO_MANIFEST_DIR")` is wrong under `cargo nextest archive`.
fn workspace_root() -> PathBuf {
    let looks_like_root = |p: &Path| {
        p.join("Cargo.toml").is_file() && p.join(".github").is_dir() && p.join("crates").is_dir()
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

fn ci_yml() -> String {
    let p = workspace_root().join(".github/workflows/ci.yml");
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

/// The text of the `validation:` job: from its `  validation:` key line to the
/// next job key at the same (2-space) indentation, or a top-level key.
fn validation_job(ci: &str) -> String {
    let lines: Vec<&str> = ci.lines().collect();
    let start = lines
        .iter()
        .position(|l| l.trim_end() == "  validation:")
        .expect("ci.yml has no `validation:` job");
    let is_job_key = |l: &str| {
        l.len() > 2
            && l.starts_with("  ")
            && !l[2..].starts_with(' ')
            && !l[2..].starts_with('#')
            && l.trim_end().ends_with(':')
    };
    let is_top_level = |l: &str| !l.is_empty() && !l.starts_with(' ') && !l.starts_with('#');
    let end = lines[start + 1..]
        .iter()
        .position(|l| is_job_key(l) || is_top_level(l))
        .map(|i| start + 1 + i)
        .unwrap_or(lines.len());
    lines[start..end].join("\n")
}

/// Every `crates/*/tests/*.rs` (top level only — only those are binaries),
/// as (crate dir name, file stem, path).
fn test_files() -> Vec<(String, String, PathBuf)> {
    let crates = workspace_root().join("crates");
    let mut out = Vec::new();
    for c in std::fs::read_dir(&crates).expect("read crates/") {
        let c = c.unwrap().path();
        let tests = c.join("tests");
        let Ok(rd) = std::fs::read_dir(&tests) else {
            continue;
        };
        for f in rd {
            let f = f.unwrap().path();
            if f.extension().is_some_and(|e| e == "rs") && f.is_file() {
                out.push((
                    c.file_name().unwrap().to_string_lossy().into_owned(),
                    f.file_stem().unwrap().to_string_lossy().into_owned(),
                    f,
                ));
            }
        }
    }
    out.sort();
    out
}

fn validation_files() -> Vec<(String, String, PathBuf)> {
    let v: Vec<_> = test_files()
        .into_iter()
        .filter(|(_, stem, _)| stem.starts_with(CONVENTION_PREFIX))
        .collect();
    assert!(
        !v.is_empty(),
        "no crates/*/tests/{CONVENTION_PREFIX}*.rs found — the tier is empty or the walk is broken"
    );
    v
}

/// The regex inside the job's `-E 'binary(/RE/)'`, reduced to the literal
/// prefix it matches. Only the `^literal` form is evaluated; anything richer
/// is refused so the guard never silently approves a filter it cannot read.
fn filter_prefix(job: &str) -> String {
    let at = job
        .find("-E 'binary(/")
        .expect("validation job has no `-E 'binary(/.../)'` filter");
    let rest = &job[at + "-E 'binary(/".len()..];
    let re = &rest[..rest.find("/)'").expect("unterminated binary(/.../) filter")];
    let lit = re
        .strip_prefix('^')
        .unwrap_or_else(|| panic!("filter regex {re:?} is not anchored with ^; extend this guard"));
    assert!(
        !lit.is_empty() && lit.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'),
        "filter regex {re:?} is not a plain ^literal prefix; extend this guard to evaluate it"
    );
    lit.to_string()
}

#[test]
fn every_validation_file_is_selected_by_the_job() {
    let job = validation_job(&ci_yml());
    let prefix = filter_prefix(&job);
    assert!(
        job.contains("--run-ignored only"),
        "the validation job must pass `--run-ignored only` (validation tests are all #[ignore])"
    );
    // The build step derives `-p <crate> --test <stem>` from a glob; it must
    // be the same prefix as the filter, or a file can be filtered-in but
    // never built (nextest then silently has nothing to run for it).
    let glob = format!("crates/*/tests/{prefix}*.rs");
    assert!(
        job.contains(&glob),
        "the validation job's build step must glob `{glob}` (same prefix as its filter)"
    );
    let mut missed = Vec::new();
    for (krate, stem, path) in validation_files() {
        if !stem.starts_with(&prefix) {
            missed.push(format!("{} (filter prefix {prefix:?})", path.display()));
        }
        // Cargo only auto-discovers tests/*.rs as a binary named after the
        // stem when autotests is on (or an explicit [[test]] names it).
        let manifest = workspace_root()
            .join("crates")
            .join(&krate)
            .join("Cargo.toml");
        let toml = std::fs::read_to_string(&manifest).unwrap();
        let autotests_off = toml
            .lines()
            .any(|l| l.split('#').next().unwrap().replace(' ', "") == "autotests=false");
        if autotests_off && !toml.contains(&format!("name = \"{stem}\"")) {
            missed.push(format!(
                "{} (autotests = false, no [[test]] entry)",
                path.display()
            ));
        }
    }
    assert!(
        missed.is_empty(),
        "validation files the weekly job would never run:\n  {}",
        missed.join("\n  ")
    );
}

#[test]
fn every_validation_test_carries_a_validation_ignore() {
    let mut bad = Vec::new();
    for (_, _, path) in validation_files() {
        let src = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = src.lines().collect();
        let mut n_tests = 0;
        for (i, line) in lines.iter().enumerate() {
            if line.trim() != "#[test]" {
                continue;
            }
            n_tests += 1;
            // The attribute run around this #[test]: attribute lines above it,
            // and every line below it up to the `fn`.
            let mut attrs: Vec<&str> = lines[..i]
                .iter()
                .rev()
                .take_while(|l| l.trim_start().starts_with("#["))
                .copied()
                .collect();
            attrs.extend(lines[i + 1..].iter().take_while(|l| !l.contains("fn ")));
            let ok = attrs.iter().any(|a| {
                a.trim()
                    .strip_prefix(IGNORE_ATTR_PREFIX)
                    .and_then(|r| r.strip_suffix("\"]"))
                    .is_some_and(|row| !row.trim().is_empty())
            });
            if !ok {
                bad.push(format!("{}:{}", path.display(), i + 1));
            }
        }
        if n_tests == 0 {
            bad.push(format!("{} (no #[test] at all)", path.display()));
        }
    }
    assert!(
        bad.is_empty(),
        "validation tests without `{IGNORE_ATTR_PREFIX}<row>\"]` would run in EVERY per-commit \
         shard:\n  {}",
        bad.join("\n  ")
    );
}

#[test]
fn validation_ignores_live_only_in_validation_files() {
    // A `validation:` ignore outside a validation_*.rs file is ignored
    // per-commit AND not selected weekly: never run anywhere.
    let mut stray = Vec::new();
    for (_, stem, path) in test_files() {
        if stem.starts_with(CONVENTION_PREFIX) {
            continue;
        }
        let src = std::fs::read_to_string(&path).unwrap();
        for (i, line) in src.lines().enumerate() {
            if line.trim_start().starts_with(IGNORE_ATTR_PREFIX) {
                stray.push(format!("{}:{}", path.display(), i + 1));
            }
        }
    }
    assert!(
        stray.is_empty(),
        "`validation:` ignores outside crates/*/tests/{CONVENTION_PREFIX}*.rs never run:\n  {}",
        stray.join("\n  ")
    );
}

#[test]
fn the_job_is_weekly_and_manual_only() {
    let ci = ci_yml();
    let job = validation_job(&ci);
    let if_line = job
        .lines()
        .find(|l| l.trim_start().starts_with("if:"))
        .expect("validation job has no `if:` gate — it would run on every push/PR");
    assert!(
        !if_line.contains("push") && !if_line.contains("pull_request"),
        "validation job must not run on push/pull_request: {if_line}"
    );
    assert!(
        if_line.contains("workflow_dispatch"),
        "validation job must allow manual dispatch"
    );
    let key = "github.event.schedule == '";
    let at = if_line
        .find(key)
        .unwrap_or_else(|| panic!("validation `if:` must key on github.event.schedule: {if_line}"));
    let cron = &if_line[at + key.len()..];
    let cron = &cron[..cron.find('\'').expect("unterminated cron string")];
    assert!(
        ci.lines().any(|l| l.trim() == format!("- cron: '{cron}'")),
        "validation `if:` keys on cron '{cron}', which no `schedule:` entry fires"
    );
    // Weekly: the day-of-week field is a single day, and day-of-month/month
    // are wildcards.
    let fields: Vec<&str> = cron.split_whitespace().collect();
    assert_eq!(fields.len(), 5, "cron '{cron}'");
    assert!(
        fields[2] == "*" && fields[3] == "*" && fields[4].parse::<u8>().is_ok_and(|d| d <= 7),
        "validation cron '{cron}' is not a once-a-week schedule"
    );
}

#[test]
fn no_other_job_runs_ignored_tests() {
    let ci = ci_yml();
    let job = validation_job(&ci);
    let outside = ci.replacen(&job, "", 1);
    // nextest's `--run-ignored` AND libtest's `-- --ignored` /
    // `-- --include-ignored`: any of them in another job would run the
    // validation tier per-commit (a `cargo test` that builds a validation_*
    // binary runs its ignored tests with either libtest flag). Matched as
    // whole tokens so a flag named in a comment or a longer word cannot trip
    // it; comment lines are skipped entirely.
    const IGNORE_FLAGS: [&str; 3] = ["--run-ignored", "--ignored", "--include-ignored"];
    let hits: Vec<&str> = outside
        .lines()
        .filter(|l| !l.trim_start().starts_with('#'))
        .filter(|l| {
            l.split(|c: char| c.is_whitespace() || c == '=' || c == '"' || c == '\'')
                .any(|tok| IGNORE_FLAGS.contains(&tok))
        })
        .collect();
    assert!(
        hits.is_empty(),
        "only the validation job may run ignored tests (--run-ignored, --ignored, \
         --include-ignored); elsewhere they run the validation tier per-commit:\n  {}",
        hits.join("\n  ")
    );
}

// ── Python half of the tier: `@pytest.mark.validation` ──────────────────────
//
// pyproject's `addopts = -m "not validation"` excludes these from every
// default pytest run, so the ONLY place they run is the validation job's
// `pytest -m validation <paths>` step. Same two quiet failure modes as the
// Rust half, pinned the same way (file set DERIVED from the tree):
//
// * a marked file outside every path that step passes to pytest — excluded
//   per-commit AND never collected weekly. MUTATION: change the step's path
//   `crates/ferric-python/tests/` to `crates/ferric-python/tests/qmmm/` →
//   `every_validation_pytest_is_collected_by_the_job` fails.
// * another job passing `-m validation` — the tier then runs per-commit.
//   MUTATION: add `-m validation` to the coverage job's pytest →
//   `no_other_job_selects_the_validation_marker` fails.
// * the extension failing to import — conftest.py SKIPS the suite, which
//   exits green. MUTATION: delete the `python -c "import ferric"` line →
//   `every_validation_pytest_is_collected_by_the_job` fails.

/// The marker the Python tier keys on, as written in a test module.
const PY_MARKER: &str = "pytest.mark.validation";

/// Every `*.py` under the repo (skipping build/VCS/venv/untracked-notes dirs)
/// that carries the validation marker, as repo-relative '/'-joined paths.
fn python_validation_files() -> Vec<String> {
    const SKIP: [&str; 5] = ["target", "wiki", "node_modules", "__pycache__", "site"];
    let root = workspace_root();
    let mut out = Vec::new();
    let mut stack = vec![root.clone()];
    while let Some(dir) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&dir) else {
            continue;
        };
        for e in rd.flatten() {
            let p = e.path();
            let name = e.file_name().to_string_lossy().into_owned();
            if p.is_dir() {
                if !name.starts_with('.') && !SKIP.contains(&name.as_str()) {
                    stack.push(p);
                }
            } else if name.ends_with(".py") {
                let Ok(src) = std::fs::read_to_string(&p) else {
                    continue;
                };
                if src.contains(PY_MARKER) {
                    let rel = p.strip_prefix(&root).unwrap();
                    out.push(
                        rel.components()
                            .map(|c| c.as_os_str().to_string_lossy().into_owned())
                            .collect::<Vec<_>>()
                            .join("/"),
                    );
                }
            }
        }
    }
    out.sort();
    out
}

/// The path arguments of every `pytest -m validation ...` command in `job`.
fn pytest_validation_paths(job: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in job.lines().filter(|l| !l.trim_start().starts_with('#')) {
        let toks: Vec<&str> = line.split_whitespace().collect();
        let Some(at) = toks
            .windows(3)
            .position(|w| w[0] == "pytest" && w[1] == "-m" && w[2] == "validation")
        else {
            continue;
        };
        out.extend(
            toks[at + 3..]
                .iter()
                .filter(|t| !t.starts_with('-'))
                .map(|t| t.trim_end_matches('/').to_string()),
        );
    }
    out
}

#[test]
fn every_validation_pytest_is_collected_by_the_job() {
    let job = validation_job(&ci_yml());
    let files = python_validation_files();
    if files.is_empty() {
        return; // no Python half yet: nothing to collect
    }
    let paths = pytest_validation_paths(&job);
    assert!(
        !paths.is_empty(),
        "files carry `{PY_MARKER}` but the validation job runs no `pytest -m validation <paths>`:\n  {}",
        files.join("\n  ")
    );
    let missed: Vec<&String> = files
        .iter()
        .filter(|f| {
            !paths
                .iter()
                .any(|p| f.starts_with(&format!("{p}/")) || *f == p)
        })
        .collect();
    assert!(
        missed.is_empty(),
        "validation pytest files outside every path the job passes to pytest ({paths:?}) are \
         never run:\n  {missed:?}"
    );
    assert!(
        job.contains("python -c \"import ferric\""),
        "the validation job must `python -c \"import ferric\"` before pytest: conftest.py skips \
         the whole suite on a failed import, which would exit green with nothing run"
    );
}

#[test]
fn no_other_job_selects_the_validation_marker() {
    let ci = ci_yml();
    let job = validation_job(&ci);
    let outside = ci.replacen(&job, "", 1);
    let hits: Vec<&str> = outside
        .lines()
        .filter(|l| !l.trim_start().starts_with('#'))
        .filter(|l| {
            let toks: Vec<&str> = l
                .split(|c: char| c.is_whitespace() || c == '"' || c == '\'')
                .filter(|t| !t.is_empty())
                .collect();
            toks.windows(2)
                .any(|w| w[0] == "-m" && w[1] == "validation")
        })
        .collect();
    assert!(
        hits.is_empty(),
        "only the validation job may select `-m validation`; elsewhere it runs the Python \
         validation tier per-commit:\n  {}",
        hits.join("\n  ")
    );
}
