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
//! Mutation ledger (each edit applied to the real file, the named test
//! failed, the edit reverted):
//! - delete `FERRIC_REQUIRE_PATCHED_BOYS: "1"` from ci.yml's `env:`:
//!   `ci_yml_requires_the_override_for_every_job` fails.
//! - delete one of the three `export FERRIC_REQUIRE_PATCHED_BOYS=1` lines in
//!   wheels.yml: `every_wheels_yml_libint_build_requires_the_override` fails.
//! - `ARG FERRIC_REQUIRE_PATCHED_BOYS=0` in the Dockerfile:
//!   `no_ci_config_weakens_the_requirement` and
//!   `dockerfile_requires_the_override_before_building` fail.
//! - add `FERRIC_ALLOW_UNPATCHED_BOYS: "1"` to a ci.yml job's `env:`:
//!   `no_ci_config_weakens_the_requirement` fails.
//! - move the Dockerfile ARG after `RUN cargo build`:
//!   `dockerfile_requires_the_override_before_building` fails.

#[allow(dead_code)]
mod policy {
    include!("../build_support/boys_policy.rs");
}

use policy::{ALLOW_VAR, REQUIRE_VAR, STOCK_VAR};

fn repo_file(rel: &str) -> String {
    let path = format!("{}/../../{rel}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {path}: {e}"))
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

#[test]
fn ci_yml_requires_the_override_for_every_job() {
    let text = repo_file(".github/workflows/ci.yml");
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
    let text = repo_file(".github/workflows/wheels.yml");
    let lines = code_lines(&text);
    let want = format!("export {REQUIRE_VAR}=1");
    let mut sites = 0;
    for (i, line) in lines.iter().enumerate() {
        if !line.starts_with("export LIBINT2_PREFIX=") {
            continue;
        }
        sites += 1;
        // Same script block: the exports are adjacent in every build script.
        let window = &lines[i + 1..(i + 6).min(lines.len())];
        assert!(
            window.contains(&want.as_str()),
            "wheels.yml line `{line}` builds against a libint2 prefix without \
             `{want}` right after it; that build would only warn when the \
             boys.h override cannot be applied (issue #234)"
        );
    }
    assert!(
        sites >= 3,
        "expected the two wheel builds and the MALLOC_PERTURB_ job to export \
         LIBINT2_PREFIX; found {sites} (if the layout changed, update this test)"
    );
}

#[test]
fn dockerfile_requires_the_override_before_building() {
    let text = repo_file("Dockerfile");
    let lines = code_lines(&text);
    let want = format!("ARG {REQUIRE_VAR}=1");
    let arg = lines
        .iter()
        .position(|l| *l == want)
        .unwrap_or_else(|| panic!("Dockerfile must declare `{want}` (issue #234)"));
    let first_build = lines
        .iter()
        .position(|l| l.starts_with("RUN") && (l.contains("cargo") || l.contains("maturin build")))
        .expect("Dockerfile has no cargo/maturin build step");
    assert!(
        arg < first_build,
        "`{want}` must come before the first cargo/maturin RUN step"
    );
}

#[test]
fn no_ci_config_weakens_the_requirement() {
    for file in CI_FILES {
        let text = repo_file(file);
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
