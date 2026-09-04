//! Confirms the CLI prints a one-line epistemic-status warning on stderr for
//! Smoke/Stub-grade `method.kind` values (per docs/VALIDATION.md), and stays
//! silent for Proven/Proven (narrow) methods. See
//! `crates/ferric-cli/src/lib.rs`'s `EPISTEMIC_WARNINGS` table (was
//! main.rs -- moved to a [lib] target so ferric-python could depend on it).

use std::path::PathBuf;
use std::process::Command;

/// Workspace root, derived from this crate's manifest dir (`<root>/crates/ferric-cli`)
/// so example TOMLs' relative `testdata/...` paths resolve regardless of the
/// directory `cargo test` invokes the test binary from.
fn workspace_root() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent() // crates/
        .and_then(|p| p.parent()) // workspace root
        .expect("ferric-cli manifest dir should be workspace_root/crates/ferric-cli")
        .to_path_buf()
}

fn run_example(toml_relpath: &str) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_ferric-cli"))
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
