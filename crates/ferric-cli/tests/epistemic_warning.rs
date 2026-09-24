//! Confirms the CLI prints a one-line epistemic-status warning on stderr for
//! Smoke/Stub-grade `method.kind` values (per docs/VALIDATION.md), and stays
//! silent for Proven/Proven (narrow) methods. See
//! `crates/ferric-cli/src/lib.rs`'s `EPISTEMIC_WARNINGS` table (was
//! main.rs -- moved to a [lib] target so ferric-python could depend on it).

use std::path::{Path, PathBuf};
use std::process::Command;

/// Path to the `ferric-cli` binary, resolved at RUN TIME.
///
/// `env!("CARGO_BIN_EXE_ferric-cli")` is baked in at COMPILE time. Under
/// `cargo nextest archive` the binary is extracted to a fresh temporary
/// directory in the RUNNING job, and that constant still points at the BUILD
/// job's `target/debug/` -- which does not exist there. MEASURED: the archive
/// does carry the executable ("419 binaries, including 3 non-test binaries"),
/// so the failure is the stale path, not a missing file.
///
/// Prefer a sibling of the currently-running test binary
/// (`<extract-dir>/target/debug/deps/<test>` -> `../ferric-cli`), which is
/// where nextest puts it, then fall back to the compile-time path for plain
/// `cargo test`.
fn ferric_cli_bin() -> PathBuf {
    if let Ok(exe) = std::env::current_exe() {
        // .../target/<profile>/deps/<test-binary>  ->  .../target/<profile>/
        if let Some(profile_dir) = exe.parent().and_then(Path::parent) {
            let p = profile_dir.join("ferric-cli");
            if p.is_file() {
                return p;
            }
        }
    }
    PathBuf::from(env!("CARGO_BIN_EXE_ferric-cli"))
}

/// Workspace root, resolved at RUN TIME.
///
/// `env!("CARGO_MANIFEST_DIR")` is baked in when the test binary is COMPILED.
/// Under `cargo nextest archive` the binary is built in one job and run in
/// another, whose checkout lives at a different path -- so the compile-time
/// directory does not exist and `Command::current_dir` fails with a bare
/// `NotFound` that reads as "the ferric-cli binary is missing" (MEASURED: 36
/// of 37 shard failures, all of them this).
///
/// So: walk up from the CURRENT directory to the nearest ancestor holding a
/// workspace `Cargo.toml` alongside `examples/` and `testdata/`, and fall back
/// to the compile-time path when that fails (the ordinary `cargo test` case,
/// where it is correct and the cwd may be anywhere).
fn workspace_root() -> PathBuf {
    let looks_like_root = |p: &std::path::Path| {
        p.join("Cargo.toml").is_file() && p.join("examples").is_dir() && p.join("testdata").is_dir()
    };
    if let Ok(cwd) = std::env::current_dir() {
        let mut here: Option<&std::path::Path> = Some(cwd.as_path());
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

fn run_example(toml_relpath: &str) -> std::process::Output {
    Command::new(ferric_cli_bin())
        .arg(toml_relpath)
        .current_dir(workspace_root())
        .env("OPENBLAS_NUM_THREADS", "1")
        .output()
        .expect("failed to run ferric-cli binary")
}

#[test]
fn smoke_grade_method_warns_on_stderr() {
    // tdhf-static-polarizability is Smoke-grade per docs/VALIDATION.md
    // ("TDHF / RPAx polarizability ... Smoke — negative verdict for C6").
    let output = run_example("examples/water-tdhf-static-alpha.toml");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("[warning] method.kind = \"tdhf-static-polarizability\" is Smoke-grade"),
        "expected epistemic warning on stderr, got:\n{stderr}"
    );
    // The warning must never land on stdout (a user may pipe/parse stdout).
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        !stdout.contains("[warning]"),
        "epistemic warning leaked onto stdout:\n{stdout}"
    );
    // The shipped example must also RUN. It did not: it left [gw] scissor at
    // 0.0, which the negative-alpha-diagonal guard refuses (alpha_xx =
    // -2.68), and this test only looked at the warning, so it stayed green on
    // an example that exits 1. Removing `scissor = 0.36` from the example
    // fails this assertion.
    assert!(
        output.status.success(),
        "examples/water-tdhf-static-alpha.toml must run as shipped; stderr:\n{stderr}"
    );
    // ...and print the number its header quotes (measured 5.203810 a.u.; the
    // DOSD reference is 9.64). A header that drifts from the run fails here.
    let iso: f64 = stdout
        .lines()
        .find_map(|l| l.trim().strip_prefix("alpha_iso (static) ="))
        .and_then(|rest| rest.split_whitespace().next())
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| panic!("no alpha_iso line in stdout:\n{stdout}"));
    assert!(
        (iso - 5.20).abs() < 0.01,
        "alpha_iso = {iso}, but the example header quotes 5.20 a.u."
    );
}

#[test]
fn proven_grade_method_does_not_warn() {
    // rhf is Proven per docs/VALIDATION.md (testdata/reference/*_rhf.json, <=1e-8 Ha).
    let output = run_example("examples/water-rhf.toml");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("[warning]"),
        "rhf is Proven-grade and should not print an epistemic warning, got stderr:\n{stderr}"
    );
    assert!(
        output.status.success(),
        "water-rhf.toml should run to completion; stderr:\n{stderr}"
    );
}
