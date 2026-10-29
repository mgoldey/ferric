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
//!
//! A test too long for any shard runs in its own `validation-<name>` job and
//! is excluded from the shards' filter; the last section below checks that
//! every such test runs exactly once.
//!
//! It also pins the malloc-perturb canary list to ONE source
//! (`scripts/ci-malloc-perturb-canaries.sh`), which build-tests packs into
//! the small canary archive and malloc-perturb runs (#525): a second copy of
//! the list would let the archive and the run drift, and a canary missing
//! from the archive is only caught on a runner. MUTATION: rename a listed
//! binary, or paste the list back inline into the malloc-perturb job, or
//! point malloc-perturb at a `nextest-shard-<k>` archive →
//! `malloc_perturb_canaries_have_one_source` fails.
//!
//! And it pins the PRIMARY archive download path (#525): test-shard and
//! malloc-perturb `exec` `scripts/ci-download-artifact.sh` (stall detection
//! + resume), with actions/download-artifact only as a fallback whose use is
//! reported. Without this, a typo in the invocation would send every run to
//! the fallback and stay green. MUTATION: misspell the artifact name or the
//! script path in either invocation, or drop the fallback report step →
//! `archive_downloads_go_through_the_stall_detecting_script` fails.
//!
//! And the per-shard archives (#525): build-tests uploads exactly
//! `nextest-shard-1..N` for an N-entry test-shard matrix and runs the
//! exactly-once guard, and test-shard runs its whole archive with no
//! `--partition` (`shard_archives_match_the_matrix`), and every archive is
//! recompressed with `zstd --long=27` before the guard lists it
//! (`shard_archives_are_long_window_and_readable`); the SLOW list lives
//! only in `scripts/ci-shard-plan.sh` and each entry runs nightly
//! (`slow_binaries_have_one_source_and_run_nightly`).

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

/// The text of the sharded `validation:` job.
fn validation_job(ci: &str) -> String {
    job_text(ci, "validation").expect("ci.yml has no `validation:` job")
}

/// Every single-test validation job (`  validation-<name>:`), as (key, text).
/// These run tests too long for a shard; the shards exclude them.
fn solo_validation_jobs(ci: &str) -> Vec<(String, String)> {
    ci.lines()
        .filter_map(|l| {
            let key = l.strip_prefix("  validation-")?.strip_suffix(':')?;
            (!key.contains(' ')).then(|| format!("validation-{key}"))
        })
        .map(|k| {
            let t = job_text(ci, &k).expect("job key just found");
            (k, t)
        })
        .collect()
}

/// The single-quoted argument of the (only) nextest `-E '...'` filter in `job`
/// (comment lines and `sed -E` are not nextest filters).
fn nextest_expr(job: &str) -> String {
    let hits: Vec<&str> = job
        .lines()
        .filter(|l| !l.trim_start().starts_with('#') && !l.contains("sed "))
        .filter_map(|l| l.split_once("-E '").map(|(_, r)| r))
        .collect();
    assert_eq!(
        hits.len(),
        1,
        "expected exactly one nextest `-E '...'` filter, found {}",
        hits.len()
    );
    let rest = hits[0];
    rest[..rest.find('\'').expect("unterminated -E '...'")].to_string()
}

/// `binary_id(=CRATE::BIN) and test(=NAME)` -> (CRATE, BIN, NAME).
fn exact_test(expr: &str) -> (String, String, String) {
    let parse = || -> Option<(String, String, String)> {
        let rest = expr.strip_prefix("binary_id(=")?;
        let (id, rest) = rest.split_once(") and test(=")?;
        let name = rest.strip_suffix(')')?;
        let (krate, bin) = id.split_once("::")?;
        Some((krate.to_string(), bin.to_string(), name.to_string()))
    };
    parse().unwrap_or_else(|| {
        panic!("{expr:?} is not `binary_id(=CRATE::BIN) and test(=NAME)`; extend this guard")
    })
}

/// The text of the job keyed `  <key>:`: from its key line to the next job key
/// at the same (2-space) indentation, or a top-level key.
fn job_text(ci: &str, key: &str) -> Option<String> {
    let lines: Vec<&str> = ci.lines().collect();
    let want = format!("  {key}:");
    let start = lines.iter().position(|l| l.trim_end() == want)?;
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
    Some(lines[start..end].join("\n"))
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

/// The sharded job's filter: `binary(/^PREFIX/)`, optionally followed by
/// ` and not (T1 or T2 ...)` naming the tests that run in their own
/// `validation-*` job. Returns (PREFIX, [T1, T2, ...]). Only these forms are
/// evaluated; anything richer is refused so the guard never silently approves
/// a filter it cannot read.
fn shard_filter(job: &str) -> (String, Vec<String>) {
    let expr = nextest_expr(job);
    let rest = expr
        .strip_prefix("binary(/")
        .unwrap_or_else(|| panic!("validation filter {expr:?} does not start with `binary(/`"));
    let close = rest.find("/)").expect("unterminated binary(/.../) filter");
    let re = &rest[..close];
    let lit = re
        .strip_prefix('^')
        .unwrap_or_else(|| panic!("filter regex {re:?} is not anchored with ^; extend this guard"));
    assert!(
        !lit.is_empty() && lit.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'),
        "filter regex {re:?} is not a plain ^literal prefix; extend this guard to evaluate it"
    );
    let tail = &rest[close + "/)".len()..];
    let excluded = if tail.is_empty() {
        Vec::new()
    } else {
        let inner = tail
            .strip_prefix(" and not (")
            .and_then(|t| t.strip_suffix(')'))
            .unwrap_or_else(|| {
                panic!("validation filter tail {tail:?} is not ` and not (...)`; extend this guard")
            });
        inner.split(" or ").map(str::to_string).collect()
    };
    (lit.to_string(), excluded)
}

/// The literal prefix of the sharded job's `binary(/^PREFIX/)` filter.
fn filter_prefix(job: &str) -> String {
    shard_filter(job).0
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
    let mut outside = ci.replacen(&validation_job(&ci), "", 1);
    for (_, solo) in solo_validation_jobs(&ci) {
        outside = outside.replacen(&solo, "", 1);
    }
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
        "only the validation jobs (`validation`, `validation-*`) may run ignored tests \
         (--run-ignored, --ignored, --include-ignored); elsewhere they run the validation \
         tier per-commit:\n  {}",
        hits.join("\n  ")
    );
}

/// The workflow-level `concurrency: group:` expression.
fn concurrency_group(ci: &str) -> String {
    let lines: Vec<&str> = ci.lines().collect();
    let start = lines
        .iter()
        .position(|l| l.trim_end() == "concurrency:")
        .expect("ci.yml has no top-level `concurrency:` block");
    lines[start + 1..]
        .iter()
        .take_while(|l| l.is_empty() || l.starts_with(' '))
        .find_map(|l| l.trim_start().strip_prefix("group:"))
        .expect("`concurrency:` block has no `group:`")
        .trim()
        .to_string()
}

/// The scheduled run must be alone in its concurrency group.
///
/// GitHub keeps at most ONE pending run per group and cancels the older
/// pending one when a newer run arrives, whatever `cancel-in-progress` says.
/// A key of `github.ref` alone is `refs/heads/main` for both a main push and
/// the nightly schedule run, so a schedule run queued behind a push was
/// cancelled (with no jobs) by the next push — four times in 2026-10-10..18,
/// including both Sunday runs, so the weekly validation job never ran.
///
/// MUTATION: restore the old key
/// `${{ github.workflow }}-${{ github.event.pull_request.number || github.ref }}`
/// → this test fails (no `github.run_id`, unguarded `github.ref`).
#[test]
fn scheduled_runs_never_share_a_concurrency_group_with_pushes() {
    let group = concurrency_group(&ci_yml());
    assert!(
        group.contains("github.event_name"),
        "concurrency group must include github.event_name so a schedule run never shares a \
         group with a push to the same ref: {group}"
    );
    assert!(
        group.contains("github.run_id"),
        "concurrency group must fall back to github.run_id so a schedule/dispatch run is alone \
         in its group (a newer arrival cancels an older PENDING run regardless of \
         cancel-in-progress): {group}"
    );
    let unguarded_ref = group
        .replace("github.event_name == 'push' && github.ref", "")
        .contains("github.ref");
    assert!(
        !unguarded_ref,
        "github.ref in the concurrency group must be reached only for `push` events; for a \
         schedule run it is refs/heads/main, the same key a main push uses: {group}"
    );
}

/// The validation job's matrix and its nextest `--partition count:K/N` must
/// agree: matrix `[1, 2]` against `/3` would silently never run a third of
/// the tier, and every shard would still be green.
///
/// MUTATION: change the matrix to `partition: [1, 2]` → this test fails.
#[test]
fn validation_shards_cover_every_partition() {
    let job = validation_job(&ci_yml());
    let key = "--partition \"count:${{ matrix.partition }}/";
    let at = job
        .find(key)
        .unwrap_or_else(|| panic!("validation job must partition with `{key}N\"`"));
    let rest = &job[at + key.len()..];
    let n: usize = rest[..rest.find('"').expect("unterminated --partition")]
        .parse()
        .expect("partition total is not an integer");
    let matrix_line = job
        .lines()
        .map(str::trim)
        .find_map(|l| l.strip_prefix("partition: "))
        .expect("validation job has no `partition:` matrix");
    let shards: Vec<usize> = matrix_line
        .trim_start_matches('[')
        .trim_end_matches(']')
        .split(',')
        .map(|t| t.trim().parse().expect("non-integer matrix partition"))
        .collect();
    assert_eq!(
        shards,
        (1..=n).collect::<Vec<_>>(),
        "validation matrix partitions {shards:?} must be exactly 1..={n} to cover count:K/{n}"
    );
    // The Python half is pinned to one shard by `matrix.partition == K`. A K
    // outside 1..=N (e.g. the shard count lowered under it) skips every one of
    // those steps on every shard, and the job is still green.
    // MUTATION: change one `matrix.partition == 4` to `== 5` → this fails.
    let pins: Vec<usize> = job
        .match_indices("matrix.partition == ")
        .map(|(at, m)| {
            let rest = &job[at + m.len()..];
            let end = rest
                .find(|c: char| !c.is_ascii_digit())
                .unwrap_or(rest.len());
            rest[..end]
                .parse()
                .expect("non-integer `matrix.partition ==` shard")
        })
        .collect();
    assert!(
        !pins.is_empty(),
        "validation job pins no step to a shard; the Python half must run on exactly one"
    );
    for k in &pins {
        assert!(
            (1..=n).contains(k),
            "a validation step is pinned to shard {k}, outside 1..={n}: it never runs"
        );
    }
    // The Python half's setup (uv / maturin) and its pytest must share a runner.
    // MUTATION: pin one of those steps to a different shard → this fails.
    assert!(
        pins.windows(2).all(|w| w[0] == w[1]),
        "the Python-half steps are pinned to different shards {pins:?}; they must share one runner"
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
    // Non-comment lines only (a commented-out `# ... import ferric` must not
    // satisfy this), and the import must come BEFORE the pytest run.
    let live: Vec<&str> = job
        .lines()
        .filter(|l| !l.trim_start().starts_with('#'))
        .collect();
    let import_at = live
        .iter()
        .position(|l| l.contains("python -c \"import ferric\""));
    let pytest_at = live.iter().position(|l| l.contains("-m validation"));
    assert!(
        matches!((import_at, pytest_at), (Some(i), Some(p)) if i < p),
        "the validation job must run `python -c \"import ferric\"` (not commented out) BEFORE \
         `pytest -m validation`: conftest.py skips the whole suite on a failed import, which \
         would exit green with nothing run (import at {import_at:?}, pytest at {pytest_at:?})"
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

// ── Single-test validation jobs (`validation-<name>`) ────────────────────────
//
// A test too long for any shard runs in its own job, and the sharded filter
// excludes it with `and not (...)`. Quiet failures pinned below:
//
// * excluded from the shards but run by no `validation-*` job: never run.
//   MUTATION: delete the `validation-cr-co6` job, or change the test name in
//   its `-E` only → `every_test_runs_exactly_once_across_shards_and_solo_jobs`
//   fails.
// * run by a `validation-*` job AND left in the shards: runs twice, and still
//   costs a shard its time budget. MUTATION: drop the `and not (...)` from the
//   sharded filter → the same test fails.
// * a solo job naming a test that does not exist, not building its binary,
//   or gated differently from the shards. MUTATION: rename the test in BOTH
//   filters, or edit the solo job's `if:` → `solo_validation_jobs_are_well_formed`
//   fails.

#[test]
fn every_test_runs_exactly_once_across_shards_and_solo_jobs() {
    let ci = ci_yml();
    let (_, mut excluded) = shard_filter(&validation_job(&ci));
    let mut solo: Vec<String> = solo_validation_jobs(&ci)
        .iter()
        .map(|(_, t)| nextest_expr(t))
        .collect();
    excluded.sort();
    solo.sort();
    let mut distinct = solo.clone();
    distinct.dedup();
    assert_eq!(
        distinct.len(),
        solo.len(),
        "two `validation-*` jobs run the same test: {solo:?}"
    );
    let never: Vec<_> = excluded.iter().filter(|e| !solo.contains(e)).collect();
    let twice: Vec<_> = solo.iter().filter(|e| !excluded.contains(e)).collect();
    assert!(
        never.is_empty(),
        "excluded from the validation shards but run by no `validation-*` job (never run): \
         {never:?}"
    );
    assert!(
        twice.is_empty(),
        "run by a `validation-*` job but NOT excluded from the shards (runs twice): {twice:?}"
    );
}

#[test]
fn solo_validation_jobs_are_well_formed() {
    let ci = ci_yml();
    let shard = validation_job(&ci);
    let if_line = |job: &str, key: &str| {
        job.lines()
            .find(|l| l.trim_start().starts_with("if:"))
            .unwrap_or_else(|| panic!("{key} has no `if:` gate"))
            .trim()
            .to_string()
    };
    let shard_if = if_line(&shard, "validation");
    let prefix = filter_prefix(&shard);
    for (key, job) in solo_validation_jobs(&ci) {
        assert_eq!(
            if_line(&job, &key),
            shard_if,
            "{key} must share the validation job's `if:`"
        );
        for flag in [
            "--profile validation",
            "--run-ignored only",
            "--no-tests=fail",
        ] {
            assert!(job.contains(flag), "{key} must pass `{flag}`");
        }
        let (krate, bin, name) = exact_test(&nextest_expr(&job));
        assert!(
            bin.starts_with(&prefix),
            "{key}: {bin} is not a {prefix}* binary"
        );
        assert!(
            job.contains(&format!("-p {krate} --test {bin}")),
            "{key} must build `-p {krate} --test {bin}`"
        );
        let path = validation_files()
            .into_iter()
            .find(|(k, stem, _)| *k == krate && *stem == bin)
            .map(|(_, _, p)| p)
            .unwrap_or_else(|| panic!("{key}: no crates/{krate}/tests/{bin}.rs"));
        let src = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = src.lines().collect();
        let decl = format!("fn {name}() {{");
        let at = lines
            .iter()
            .position(|l| l.trim() == decl)
            .unwrap_or_else(|| panic!("{key}: {} has no `{decl}`", path.display()));
        assert!(
            lines[..at]
                .iter()
                .rev()
                .take_while(|l| l.trim_start().starts_with("#["))
                .any(|l| l.trim() == "#[test]"),
            "{key}: `{name}` in {} is not a #[test]",
            path.display()
        );
    }
}

const CANARY_SCRIPT: &str = "scripts/ci-malloc-perturb-canaries.sh";

/// The text of job `name`, panicking if ci.yml has no such job.
fn job(ci: &str, name: &str) -> String {
    job_text(ci, name).unwrap_or_else(|| panic!("ci.yml has no `{name}:` job"))
}

/// The canary binary ids, read from the `CANARY_IDS=( ... )` array in
/// `scripts/ci-malloc-perturb-canaries.sh`.
fn canary_ids() -> Vec<String> {
    let p = workspace_root().join(CANARY_SCRIPT);
    let text = std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    let body = text
        .split_once("CANARY_IDS=(")
        .and_then(|(_, rest)| rest.split_once(')'))
        .map(|(body, _)| body)
        .unwrap_or_else(|| panic!("{CANARY_SCRIPT} has no `CANARY_IDS=( ... )` array"));
    body.split_whitespace().map(str::to_owned).collect()
}

#[test]
fn malloc_perturb_canaries_have_one_source() {
    let ids = canary_ids();
    assert!(!ids.is_empty(), "{CANARY_SCRIPT} lists no canary binaries");

    // Every id names a real test binary (crate dir == package name here).
    let files: Vec<(String, String)> = test_files().into_iter().map(|(c, f, _)| (c, f)).collect();
    for id in &ids {
        let (krate, bin) = id
            .split_once("::")
            .unwrap_or_else(|| panic!("canary `{id}` is not a `crate::binary` id"));
        assert!(
            files.iter().any(|(c, f)| c == krate && f == bin),
            "canary `{id}` has no crates/{krate}/tests/{bin}.rs: renamed or deleted, so the \
             canary archive would silently lack it"
        );
    }

    let ci = ci_yml();
    // No second copy of the list anywhere in ci.yml (comments may name them).
    let inline: Vec<&str> = ci
        .lines()
        .filter(|l| !l.trim_start().starts_with('#'))
        .filter(|l| {
            ids.iter()
                .any(|id| l.contains(id.split_once("::").unwrap().1))
        })
        .collect();
    assert!(
        inline.is_empty(),
        "ci.yml must read the canary list from {CANARY_SCRIPT}, not repeat it:\n  {}",
        inline.join("\n  ")
    );

    let build = job(&ci, "build-tests");
    for needle in [
        "scripts/ci-malloc-perturb-canaries.sh filter",
        "--archive-file /tmp/canary-archive.tar.zst",
        "name: canary-archive",
    ] {
        assert!(
            build.contains(needle),
            "build-tests must contain `{needle}`"
        );
    }
    let run = job(&ci, "malloc-perturb");
    for needle in [
        "scripts/ci-malloc-perturb-canaries.sh ids",
        "scripts/ci-malloc-perturb-canaries.sh filter",
        "--archive-file /tmp/canary-archive.tar.zst",
        "name: canary-archive",
    ] {
        assert!(
            run.contains(needle),
            "malloc-perturb must contain `{needle}`"
        );
    }
    assert!(
        !run.contains("nextest-shard"),
        "malloc-perturb must use the small canary archive, not a test-shard archive (#525)"
    );
}

/// Non-comment lines of `text`, joined -- so a needle cannot be satisfied by
/// a comment that merely names it.
fn code_only(text: &str) -> String {
    text.lines()
        .filter(|l| !l.trim_start().starts_with('#'))
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn archive_downloads_go_through_the_stall_detecting_script() {
    let script = workspace_root().join("scripts/ci-download-artifact.sh");
    assert!(script.is_file(), "{} is missing", script.display());

    assert!(
        workspace_root()
            .join("scripts/ci-archive-advice.sh")
            .is_file(),
        "scripts/ci-archive-advice.sh is missing"
    );
    let ci = ci_yml();
    assert!(
        code_only(&job(&ci, "build-tests"))
            .contains("archives_uploaded_at: ${{ steps.uploaded.outputs.at }}"),
        "build-tests must export archives_uploaded_at (#525)"
    );
    {
        let build = code_only(&job(&ci, "build-tests"));
        assert!(
            build.contains("archive_digests: ${{ steps.uploaded.outputs.digests }}")
                && step_text(&build, "Record when the archives were uploaded")
                    .contains("echo \"digests=$D\" >> \"$GITHUB_OUTPUT\""),
            "build-tests must export archive_digests from its record step (#525)"
        );
    }
    // The advice's expiry wording and age threshold come from RETENTION_DAYS
    // in the script; EVERY upload-artifact step in build-tests (the archives
    // the downstream jobs download) must keep exactly that long. The set is
    // derived from ci.yml and `retention-days` parsed per step, so a new
    // upload is checked too and key order does not matter.
    // MUTATIONS: add a 6th upload with retention-days: 1 -> fails; set one
    // shard back to 1 -> fails; reorder an upload's `with:` keys -> passes.
    let advice_path = workspace_root().join("scripts/ci-archive-advice.sh");
    let advice_src = std::fs::read_to_string(&advice_path)
        .unwrap_or_else(|e| panic!("{}: {e}", advice_path.display()));
    let days = advice_src
        .lines()
        .find_map(|l| l.trim().strip_prefix("RETENTION_DAYS="))
        .expect("scripts/ci-archive-advice.sh must set RETENTION_DAYS=");
    let build = code_only(&job(&ci, "build-tests"));
    let build_lines: Vec<&str> = build.lines().collect();
    let step_starts: Vec<usize> = build_lines
        .iter()
        .enumerate()
        .filter(|(_, l)| {
            let t = l.trim_start();
            t.starts_with("- name:") || t.starts_with("- uses:")
        })
        .map(|(i, _)| i)
        .collect();
    let mut uploads = Vec::new();
    for (k, &a) in step_starts.iter().enumerate() {
        let b = step_starts.get(k + 1).copied().unwrap_or(build_lines.len());
        let step = &build_lines[a..b];
        if !step
            .iter()
            .any(|l| l.contains("uses: actions/upload-artifact@"))
        {
            continue;
        }
        // Skip the step's own `- name:`; the `with:` block's `name:` is the
        // artifact.
        let field = |key: &str| -> Option<String> {
            step[1..]
                .iter()
                .find_map(|l| l.trim().strip_prefix(key).map(|v| v.trim().to_owned()))
        };
        let name = field("name:").unwrap_or_else(|| step[0].trim().to_owned());
        let retention = field("retention-days:");
        assert_eq!(
            retention.as_deref(),
            Some(days),
            "build-tests uploads `{name}` with retention-days {retention:?}; every \
             archive upload must keep scripts/ci-archive-advice.sh's RETENTION_DAYS={days} (#525)"
        );
        uploads.push(name);
    }
    assert!(
        uploads.len() >= 5 && uploads.iter().any(|n| n == "canary-archive"),
        "found too few build-tests archive uploads to check: {uploads:?}"
    );
    for (job_name, invocation, advice, download_step, check_step, run_step) in [
        (
            "test-shard",
            "run: exec scripts/ci-download-artifact.sh nextest-shard-${{ matrix.partition }} \
             nextest-shard-${{ matrix.partition }}.tar.zst /tmp",
            "timeout 60 scripts/ci-archive-advice.sh nextest-shard-${{ matrix.partition }} \"$ARCHIVE\" || true",
            "Download the archive",
            "Check the archive",
            "Run shard ${{ matrix.partition }} of 4",
        ),
        (
            "malloc-perturb",
            "run: exec scripts/ci-download-artifact.sh canary-archive canary-archive.tar.zst /tmp",
            "timeout 60 scripts/ci-archive-advice.sh canary-archive /tmp/canary-archive.tar.zst || true",
            "Download the canary archive",
            "Check the canary archive",
            "Run the canary set under MALLOC_PERTURB_ 1, 170, 2",
        ),
    ] {
        let text = code_only(&job(&ci, job_name));
        // A missing/corrupt archive says WHICH re-run can fix it: an EXPIRED
        // one (past retention-days) needs "Re-run ALL jobs". Advisory:
        // `|| true`, the step still exits 1. The check is its OWN step,
        // the only one besides the download holding the token, so the test
        // processes never inherit it (#525).
        // MUTATIONS: move the token onto the run step, or the advice call
        // into it -> fails.
        let check = step_text(&text, check_step);
        for needle in [
            advice,
            "exit 1",
            "GITHUB_TOKEN: ${{ github.token }}",
            "CI_ARCHIVE_UPLOADED_AT: ${{ needs.build-tests.outputs.archives_uploaded_at }}",
            // Bytes + sha256 as uploaded: tells a failed transfer (re-run
            // the job) from a corrupt upload (re-run all jobs).
            "CI_ARCHIVE_DIGESTS: ${{ needs.build-tests.outputs.archive_digests }}",
        ] {
            assert!(
                check.contains(needle),
                "{job_name}'s `{check_step}` step must contain `{needle}` (#525)"
            );
        }
        assert_eq!(
            text.matches("scripts/ci-archive-advice.sh").count(),
            1,
            "{job_name} must call the advice script only from `{check_step}`"
        );
        let run = step_text(&text, run_step);
        for leak in ["github.token", "GITHUB_TOKEN", "ci-archive-advice"] {
            assert!(
                !run.contains(leak),
                "{job_name}'s `{run_step}` step runs the tests and must not see `{leak}` (#525)"
            );
        }
        assert!(
            step_text(&text, download_step).contains("GITHUB_TOKEN: ${{ github.token }}"),
            "{job_name}'s `{download_step}` step needs GITHUB_TOKEN (#525)"
        );
        for needle in [
            invocation,
            // The script reads this run's artifacts through the REST API.
            "actions: read",
            // The fallback runs only when the script failed, and says so.
            "if: steps.download.outcome != 'success'",
            "name: Report the download-artifact fallback",
            "GITHUB_STEP_SUMMARY",
        ] {
            assert!(
                text.contains(needle),
                "{job_name} must contain `{needle}` (#525)"
            );
        }
    }
}

const SHARD_PLAN_SCRIPT: &str = "scripts/ci-shard-plan.sh";

/// The `partition: [1, 2, ...]` matrix of `job`, as numbers.
fn partition_matrix(job_text: &str) -> Vec<u32> {
    let line = code_only(job_text)
        .lines()
        .find_map(|l| l.trim().strip_prefix("partition: [").map(str::to_owned))
        .expect("job has no `partition: [...]` matrix");
    line.trim_end_matches(']')
        .split(',')
        .map(|v| {
            v.trim()
                .parse()
                .unwrap_or_else(|e| panic!("partition value {v:?}: {e}"))
        })
        .collect()
}

/// Per-shard archives (#525): build-tests packs one archive per test-shard
/// job and test-shard runs ALL of its own, so the artifact set, the matrix and
/// the assign count must agree. A shard without an uploaded archive fails
/// only on a runner; an uploaded archive no shard downloads is a set of
/// tests that silently never runs; a leftover `--partition` would run only
/// a quarter of each shard's quarter.
///
/// MUTATIONS (each fails this test): drop the `Upload shard archive 4` step;
/// add `5` to the matrix; set `SHARDS=3`; re-add
/// `--partition hash:${{ matrix.partition }}/4` to the shard run; delete the
/// `ci-shard-plan.sh verify` call.
#[test]
fn shard_archives_match_the_matrix() {
    assert!(
        workspace_root().join(SHARD_PLAN_SCRIPT).is_file(),
        "{SHARD_PLAN_SCRIPT} is missing"
    );
    let ci = ci_yml();
    let shard = code_only(&job(&ci, "test-shard"));
    let build = code_only(&job(&ci, "build-tests"));

    let matrix = partition_matrix(&shard);
    let n = matrix.len() as u32;
    assert_eq!(
        matrix,
        (1..=n).collect::<Vec<_>>(),
        "test-shard's partition matrix must be 1..=N"
    );

    // build-tests: the shard count it plans for and the artifacts it uploads.
    let shards_decl: Vec<&str> = build
        .lines()
        .filter_map(|l| l.trim().strip_prefix("SHARDS="))
        .collect();
    assert_eq!(
        shards_decl,
        [n.to_string().as_str()],
        "build-tests must set SHARDS={n} exactly once (the test-shard matrix size)"
    );
    assert!(
        build.contains(&format!(
            "{SHARD_PLAN_SCRIPT} assign \"$PLAN-full.json\" scripts/ci-shard-weights.tsv \"$SHARDS\""
        )),
        "build-tests must run `{SHARD_PLAN_SCRIPT} assign ... \"$SHARDS\"`"
    );
    assert!(
        build.contains(&format!("{SHARD_PLAN_SCRIPT} verify \"$PLAN-full.json\"")),
        "build-tests must run the exactly-once guard (`{SHARD_PLAN_SCRIPT} verify`)"
    );
    let mut uploaded: Vec<String> = build
        .lines()
        .filter_map(|l| l.trim().strip_prefix("name: nextest-shard-"))
        .map(str::to_owned)
        .collect();
    uploaded.sort();
    let want: Vec<String> = (1..=n).map(|k| k.to_string()).collect();
    assert_eq!(
        uploaded, want,
        "build-tests must upload exactly nextest-shard-1..{n}"
    );
    for k in 1..=n {
        let needle = format!("path: /tmp/nextest-shard-{k}.tar.zst");
        assert!(
            build.contains(&needle),
            "build-tests must contain `{needle}`"
        );
    }
    assert!(
        !build.contains("name: nextest-archive"),
        "build-tests still uploads the single nextest-archive (#525)"
    );

    // test-shard: its own archive, run whole.
    for needle in [
        "ARCHIVE=/tmp/nextest-shard-${{ matrix.partition }}.tar.zst",
        "--archive-file \"$ARCHIVE\"",
        "name: nextest-shard-${{ matrix.partition }}",
        "SLOW=\"$(scripts/ci-shard-plan.sh slow-filter)\"",
        "-E \"not ($SLOW)\"",
        // An empty archive must fail the shard, not pass it.
        "--no-tests=fail",
    ] {
        assert!(
            shard.contains(needle),
            "test-shard must contain `{needle}` (#525)"
        );
    }
    assert!(
        !shard.contains("--partition"),
        "test-shard must not pass --partition: each shard runs its whole archive (#525)"
    );

    // build-tests plans and verifies the shards with the SAME deferred set
    // test-shard runs with. A build-side filter that differs from the run's
    // would make the exactly-once guard certify a different test set from the
    // one the shards actually run.
    for needle in [
        "SLOW=\"$(scripts/ci-shard-plan.sh slow-filter)\"",
        // The full fast-tier list the plan and the guard start from ...
        "-E \"not ($SLOW)\" --message-format json > \"$PLAN-full.json\"",
        // ... and each archive's list, which the guard compares with it.
        "--workspace-remap . -E \"not ($SLOW)\" --message-format json > \"$PLAN-$k.json\"",
    ] {
        assert!(
            build.contains(needle),
            "build-tests must contain `{needle}`: the same SLOW filter as test-shard (#525)"
        );
    }
}

/// The text of the step `- name: <name>` inside `job_text`, up to the next
/// step.
fn step_text(job_text: &str, name: &str) -> String {
    let want = format!("- name: {name}");
    let lines: Vec<&str> = job_text.lines().collect();
    let start = lines
        .iter()
        .position(|l| l.trim() == want)
        .unwrap_or_else(|| panic!("no step `{want}`"));
    let end = lines[start + 1..]
        .iter()
        .position(|l| {
            l.trim_start().starts_with("- name:") || l.trim_start().starts_with("- uses:")
        })
        .map(|i| start + 1 + i)
        .unwrap_or(lines.len());
    lines[start..end].join("\n")
}

/// Long-window shard archives (#525). build-tests recompresses each shard
/// archive with `zstd --long=27`, and every reader must accept it.
///
/// 27 is a hard ceiling, not a tuning knob: it is the largest window a zstd
/// decoder accepts by default, and nextest (which has no window option) reads
/// the archive with a default decoder. Measured locally: `cargo nextest list
/// --archive-file` reads a --long=27 archive and fails a --long=28 one with
/// "Frame requires too much memory for decoding".
///
/// MUTATIONS (each fails this test): `--long=28` anywhere; drop `pipefail`
/// from the archive step (a decompressor dying half way then leaves a
/// SHORTER but valid frame that passes `zstd -t`); move the recompression
/// after the exactly-once guard (the guard would then certify the
/// originals, not the uploaded files); drop `--long=27` from test-shard's or
/// the download script's `zstd -t`; replace the `exit 1` after the
/// entry/byte-count `::error::` with `true` (the mismatch would then be
/// logged and uploaded anyway); add `set +o pipefail` (or any `set +`) to
/// the archive step; drop the decompressed byte-count comparison.
#[test]
fn shard_archives_are_long_window_and_readable() {
    let ci = ci_yml();
    let build = code_only(&job(&ci, "build-tests"));
    let archive_step = step_text(&build, "Archive the test binaries");
    let shard = code_only(&job(&ci, "test-shard"));
    let dl_path = workspace_root().join("scripts/ci-download-artifact.sh");
    let dl = code_only(
        &std::fs::read_to_string(&dl_path).unwrap_or_else(|e| panic!("{}: {e}", dl_path.display())),
    );

    assert!(
        archive_step.contains("set -euo pipefail"),
        "the archive step pipes zstd into zstd and must `set -euo pipefail`"
    );
    let recompress = "zstd -dcq \"$A\" | zstd -q --long=27 -T0 -3 -c > \"$A.long\"";
    for needle in [
        "A=\"/tmp/nextest-shard-$k.tar.zst\"",
        recompress,
        "zstd -tq --long=27 \"$A.long\"",
        // Same tar entry count before and after.
        "AFTER_N=$(tar -I 'zstd --long=27' -tf \"$A.long\" | wc -l)",
        "\"$AFTER_N\" -ne \"$BEFORE_N\"",
        // Same decompressed byte count: a cut on a member boundary keeps
        // the entry count but not this.
        "BEFORE_RAW=$(zstd -dcq \"$A\" | wc -c)",
        "AFTER_RAW=$(zstd -dcq --long=27 \"$A.long\" | wc -c)",
        "\"$AFTER_RAW\" -ne \"$BEFORE_RAW\"",
        "mv \"$A.long\" \"$A\"",
    ] {
        assert!(
            archive_step.contains(needle),
            "build-tests' archive step must contain `{needle}` (#525)"
        );
    }
    // Nothing in the step may switch the shell options back off: a
    // `set +o pipefail` (or `set +e`) anywhere in it reopens the
    // shorter-but-valid-frame hole the checks above exist for.
    let relaxed: Vec<&str> = archive_step
        .lines()
        .map(str::trim)
        .filter(|l| l.starts_with("set +") || l.contains("+o pipefail"))
        .collect();
    assert!(
        relaxed.is_empty(),
        "the archive step must not relax its shell options: {relaxed:?}"
    );
    // The mismatch check must STOP the job: its `::error::` line is followed
    // directly by `exit 1`, inside the recompress loop.
    let lines: Vec<&str> = archive_step.lines().map(str::trim).collect();
    let err = lines
        .iter()
        .position(|l| {
            l.contains("::error::nextest-shard-$k changed in the --long=27 recompression")
        })
        .expect("archive step lacks the recompression mismatch ::error::");
    assert_eq!(
        lines.get(err + 1).copied(),
        Some("exit 1"),
        "the recompression mismatch ::error:: must be followed by `exit 1`"
    );
    let at = |needle: &str| {
        archive_step
            .find(needle)
            .unwrap_or_else(|| panic!("archive step lacks `{needle}`"))
    };
    assert!(
        at("mv \"$A.long\" \"$A\"") < at("cargo nextest list --archive-file"),
        "the --long=27 recompression must run BEFORE the exactly-once guard lists the \
         archives, so the guard reads the files that are uploaded"
    );
    assert!(
        shard.contains("zstd -tq --long=27 \"$ARCHIVE\""),
        "test-shard's integrity check must pass --long=27"
    );
    assert!(
        dl.contains("zstd -tq --long=27 \"$OUT\""),
        "scripts/ci-download-artifact.sh's integrity check must pass --long=27"
    );
    for (what, text) in [("ci.yml", code_only(&ci)), ("ci-download-artifact.sh", dl)] {
        for (i, _) in text.match_indices("--long=") {
            let window = &text[i + "--long=".len()..]
                .chars()
                .take_while(char::is_ascii_digit)
                .collect::<String>();
            assert_eq!(
                window, "27",
                "{what} uses --long={window}: nextest cannot read a window above 27"
            );
        }
    }
}

/// The SLOW (deferred-to-nightly) list lives ONLY in scripts/ci-shard-plan.sh,
/// and every entry has a `--test` line in slow-tests, or it runs nowhere.
/// MUTATIONS: paste a `binary(dft_pbe)` list back into a job, or delete the
/// `--test dft_pbe` from slow-tests → this fails.
#[test]
fn slow_binaries_have_one_source_and_run_nightly() {
    let p = workspace_root().join(SHARD_PLAN_SCRIPT);
    let text = std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    let names: Vec<String> = text
        .split_once("SLOW_BINARIES=(")
        .and_then(|(_, rest)| rest.split_once(')'))
        .map(|(body, _)| body)
        .unwrap_or_else(|| panic!("{SHARD_PLAN_SCRIPT} has no `SLOW_BINARIES=( ... )` array"))
        .split_whitespace()
        .map(str::to_owned)
        .collect();
    assert!(
        names.len() >= 10,
        "{SHARD_PLAN_SCRIPT} lists only {} SLOW binaries",
        names.len()
    );

    let ci = ci_yml();
    let code = code_only(&ci);
    for name in &names {
        let inline = format!("binary({name})");
        assert!(
            !code.contains(&inline),
            "ci.yml repeats `{inline}`: read the SLOW list from {SHARD_PLAN_SCRIPT} instead"
        );
    }
    let slow = code_only(&job(&ci, "slow-tests"));
    for name in &names {
        let has_line = slow.lines().any(|l| {
            l.split_whitespace()
                .collect::<Vec<_>>()
                .windows(2)
                .any(|w| w[0] == "--test" && w[1] == name)
        });
        assert!(
            has_line,
            "SLOW binary `{name}` has no `--test {name}` in slow-tests: deferred from the \
             fast tier AND never run nightly"
        );
    }
}
