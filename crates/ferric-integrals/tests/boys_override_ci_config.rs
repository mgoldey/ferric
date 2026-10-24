//! CI must FAIL, not warn, when the libint2 `boys.h` override cannot be
//! applied (issue #234). build.rs only does that when
//! `FERRIC_REQUIRE_PATCHED_BOYS` is set, and nothing sets it except the CI
//! configuration itself: ci.yml's workflow-level `env:`, an inline `export`
//! in each wheels.yml build script (maturin-action does not forward the
//! host environment), and a Dockerfile build ARG. Deleting any one of those
//! lines reverts that build to a warning that nobody reads, so a new libint2
//! pin or a stale cache would ship without the fixes again. These tests pin
//! the configuration so such an edit fails here instead.
//!
//! Local and sdist builds deliberately do NOT require the override (a distro
//! or from-source libint2 of another version is a documented install path),
//! which is why this is keyed on ferric's own CI files rather than on
//! `CI`/`GITHUB_ACTIONS`: those are also set when a downstream project's CI
//! pip-installs ferric's sdist against its own libint2.
//!
//! The checks match exact text on purpose. A harmless reformat (a trailing
//! comment on the line, an unquoted `1` in YAML, a different quote style)
//! FAILS loudly by design: update the expected text here together with the
//! config rather than loosening the match until a real weakening slips by.
//!
//! The workspace root is found at RUN time (walking up from the cwd), so the
//! test works from a `cargo nextest archive` extraction. Where `.github` is
//! absent (e.g. a Docker build context, which `.dockerignore` strips of
//! `.github`), the tests skip with a printed reason.
//!
//! Mutation ledger (each edit applied to the real file or a scratch copy of
//! the root, the named test failed, the edit reverted):
//! - delete `FERRIC_REQUIRE_PATCHED_BOYS: "1"` from ci.yml's `env:`:
//!   `ci_yml_requires_the_override_for_every_job` fails.
//! - delete one of the three `export FERRIC_REQUIRE_PATCHED_BOYS=1` lines in
//!   wheels.yml: `every_wheels_yml_libint_build_requires_the_override` fails.
//! - insert `cargo test -p ferric-integrals --no-run` between a
//!   `LIBINT2_PREFIX` export and the requirement export:
//!   `every_wheels_yml_libint_build_requires_the_override` fails.
//! - add a 4th `export LIBINT2_PREFIX=` build site, or a YAML
//!   `LIBINT2_PREFIX:` env key, to wheels.yml:
//!   `every_wheels_yml_libint_build_requires_the_override` fails.
//! - `ARG FERRIC_REQUIRE_PATCHED_BOYS=0` in the Dockerfile:
//!   `no_ci_config_weakens_the_requirement` and
//!   `dockerfile_requires_the_override_before_building` fail.
//! - move the Dockerfile ARG above `FROM ... AS ferric-builder` (it is then
//!   visible only to FROM lines): `dockerfile_requires_the_override_before_building` fails.
//! - move the Dockerfile ARG after `RUN cargo build`:
//!   `dockerfile_requires_the_override_before_building` fails.
//! - add `FERRIC_ALLOW_UNPATCHED_BOYS: "1"` to a ci.yml job's `env:`:
//!   `no_ci_config_weakens_the_requirement` fails.

use std::path::{Path, PathBuf};

#[allow(dead_code)]
mod policy {
    include!("../build_support/boys_policy.rs");
}

use policy::{ALLOW_VAR, REQUIRE_VAR, STOCK_VAR};

/// Workspace root holding `.github`, resolved at run time (see
/// crates/ferric-cli/tests/ci_validation_tier_guard.rs). `None` when no such
/// root exists, e.g. inside a Docker build context.
fn workspace_root() -> Option<PathBuf> {
    let looks_like_root = |p: &Path| {
        p.join("Cargo.toml").is_file() && p.join(".github").is_dir() && p.join("crates").is_dir()
    };
    if let Ok(cwd) = std::env::current_dir() {
        let mut here: Option<&Path> = Some(cwd.as_path());
        while let Some(p) = here {
            if looks_like_root(p) {
                return Some(p.to_path_buf());
            }
            here = p.parent();
        }
    }
    let fallback = Path::new(env!("CARGO_MANIFEST_DIR")).parent()?.parent()?;
    looks_like_root(fallback).then(|| fallback.to_path_buf())
}

/// The text of `rel` under the workspace root, or `None` (with the reason
/// printed) when there is no root with `.github` to check.
fn repo_file(rel: &str) -> Option<String> {
    let Some(root) = workspace_root() else {
        eprintln!(
            "SKIP: no workspace root with .github above {:?} (e.g. a Docker build context); \
             the CI configuration cannot be checked here",
            std::env::current_dir().ok()
        );
        return None;
    };
    let path = root.join(rel);
    Some(
        std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display())),
    )
}

/// Lines with shell/YAML/Dockerfile `#` comments removed (whole-line only,
/// which is all these files use around the variables in question).
fn code_lines(text: &str) -> Vec<&str> {
    text.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .collect()
}

const CI_FILES: [&str; 3] = [
    ".github/workflows/ci.yml",
    ".github/workflows/wheels.yml",
    "Dockerfile",
];

/// The number of wheels.yml scripts that build against a libint2 prefix:
/// the wheel build, the MALLOC_PERTURB_ job and the MPI wheel build. Exact,
/// so a new build site must be added here (and is then checked).
const WHEELS_LIBINT_SITES: usize = 3;

#[test]
fn ci_yml_requires_the_override_for_every_job() {
    let Some(text) = repo_file(".github/workflows/ci.yml") else {
        return;
    };
    let lines: Vec<&str> = text.lines().collect();
    let start = lines
        .iter()
        .position(|l| *l == "env:")
        .expect("ci.yml has no workflow-level `env:` block");
    let end = lines[start..]
        .iter()
        .position(|l| *l == "jobs:")
        .map(|i| start + i)
        .expect("ci.yml has no `jobs:` after its `env:` block");
    let want = format!("{REQUIRE_VAR}: \"1\"");
    assert!(
        lines[start..end].iter().any(|l| l.trim() == want),
        "ci.yml's workflow-level env: must set `{want}` so every job's build \
         fails when the boys.h override cannot be applied (issue #234)"
    );
}

#[test]
fn every_wheels_yml_libint_build_requires_the_override() {
    let Some(text) = repo_file(".github/workflows/wheels.yml") else {
        return;
    };
    let lines = code_lines(&text);
    if let Some(key) = lines.iter().find(|l| l.starts_with("LIBINT2_PREFIX:")) {
        panic!(
            "wheels.yml sets `{key}` as a YAML env key; this test only checks \
             `export LIBINT2_PREFIX=` build scripts, so extend it to cover the new form"
        );
    }
    let want = format!("export {REQUIRE_VAR}=1");
    let mut sites = 0;
    for (i, line) in lines.iter().enumerate() {
        if !line.starts_with("export LIBINT2_PREFIX=") {
            continue;
        }
        sites += 1;
        // The requirement must be exported before anything after the prefix
        // export can build (a cargo or maturin line) and before the next
        // build site's prefix export.
        let mut required = false;
        for next in &lines[i + 1..] {
            if *next == want {
                required = true;
                break;
            }
            if next.contains("cargo") || next.contains("maturin") {
                panic!(
                    "wheels.yml: `{next}` follows `{line}` before `{want}`; that \
                     build would only warn when the boys.h override cannot be \
                     applied (issue #234)"
                );
            }
            if next.starts_with("export LIBINT2_PREFIX=") {
                break;
            }
        }
        assert!(
            required,
            "wheels.yml line `{line}` builds against a libint2 prefix without \
             `{want}` after it (issue #234)"
        );
    }
    assert_eq!(
        sites, WHEELS_LIBINT_SITES,
        "wheels.yml has {sites} `export LIBINT2_PREFIX=` build sites, expected \
         {WHEELS_LIBINT_SITES}; update WHEELS_LIBINT_SITES once each new site \
         exports {REQUIRE_VAR}=1"
    );
}

#[test]
fn dockerfile_requires_the_override_before_building() {
    let Some(text) = repo_file("Dockerfile") else {
        return;
    };
    let lines = code_lines(&text);
    let want = format!("ARG {REQUIRE_VAR}=1");
    let builder = lines
        .iter()
        .position(|l| l.starts_with("FROM ") && l.ends_with(" AS ferric-builder"))
        .expect("Dockerfile has no `FROM ... AS ferric-builder` stage");
    let arg = lines
        .iter()
        .position(|l| *l == want)
        .unwrap_or_else(|| panic!("Dockerfile must declare `{want}` (issue #234)"));
    let first_build = lines
        .iter()
        .position(|l| l.starts_with("RUN") && (l.contains("cargo") || l.contains("maturin build")))
        .expect("Dockerfile has no cargo/maturin build step");
    assert!(
        builder < arg,
        "`{want}` must come after `FROM ... AS ferric-builder`: an ARG above \
         the first FROM is visible only to FROM lines"
    );
    assert!(
        arg < first_build,
        "`{want}` must come before the first cargo/maturin RUN step"
    );
}

#[test]
fn no_ci_config_weakens_the_requirement() {
    for file in CI_FILES {
        let Some(text) = repo_file(file) else {
            return;
        };
        for line in code_lines(&text) {
            for var in [ALLOW_VAR, STOCK_VAR] {
                assert!(
                    !line.contains(var),
                    "{file}: `{line}` sets {var}, which builds without the \
                     boys.h override fixes (issue #234)"
                );
            }
            if line.contains(REQUIRE_VAR) {
                let truthy = [format!("{REQUIRE_VAR}=1"), format!("{REQUIRE_VAR}: \"1\"")];
                assert!(
                    truthy.iter().any(|t| line.contains(t.as_str())),
                    "{file}: `{line}` sets {REQUIRE_VAR} to something other \
                     than 1 (issue #234)"
                );
            }
        }
    }
}
