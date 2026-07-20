//! Confirms the CLI's live per-iteration SCF trace (`--verbose`/`-v` flag,
//! `[scf] verbose = true` TOML key) is opt-in and additive:
//!  (a) default (neither flag nor TOML key set) prints nothing extra to
//!      stdout beyond today's post-hoc summary — byte-identical to before;
//!  (b) `--verbose`/`-v` and `[scf] verbose = true` both turn on a
//!      parseable "SCF iter=... E=... dE=... dp_rms=... err_max=..." line
//!      per iteration on stdout.
//!
//! Mirrors `epistemic_warning.rs`'s `Command::new(env!("CARGO_BIN_EXE_..."))`
//! pattern for driving the real CLI binary end-to-end.

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

fn run_cli(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_ferric-cli"))
        .args(args)
        .current_dir(workspace_root())
        .env("OPENBLAS_NUM_THREADS", "1")
        .output()
        .expect("failed to run ferric-cli binary")
}

#[test]
fn default_run_has_no_per_iteration_trace_on_stdout() {
    let output = run_cli(&["examples/water-rhf.toml"]);
    assert!(
        output.status.success(),
        "water-rhf.toml should run to completion; stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        !stdout.contains("SCF iter="),
        "default (no --verbose, no [scf] verbose) run must not print a \
         per-iteration trace; got stdout:\n{stdout}"
    );
    // The existing post-hoc summary must still be present, unchanged.
    assert!(stdout.contains("iterations ="), "missing post-hoc summary:\n{stdout}");
    assert!(stdout.contains("converged  ="), "missing post-hoc summary:\n{stdout}");
    assert!(stdout.contains("energy     ="), "missing post-hoc summary:\n{stdout}");
}

#[test]
fn cli_verbose_flag_prints_per_iteration_trace_on_stdout() {
    for flag in ["--verbose", "-v"] {
        let output = run_cli(&[flag, "examples/water-rhf.toml"]);
        assert!(
            output.status.success(),
            "water-rhf.toml with {flag} should run to completion; stderr:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8_lossy(&output.stdout);
        let trace_lines: Vec<&str> =
            stdout.lines().filter(|l| l.starts_with("SCF iter=")).collect();
        assert!(
            trace_lines.len() >= 2,
            "{flag} should print at least 2 SCF iteration trace lines, got {}:\n{stdout}",
            trace_lines.len()
        );
        // Format sanity: iter 1's line has the expected fields, parseable.
        let first = trace_lines[0];
        assert!(first.contains("iter=   1"), "unexpected iter field: {first}");
        assert!(first.contains("E="), "missing energy field: {first}");
        assert!(first.contains("dE="), "missing dE field: {first}");
        assert!(first.contains("dp_rms="), "missing dp_rms field: {first}");
        assert!(first.contains("err_max="), "missing err_max field: {first}");
        // Flag order shouldn't matter.
        let output2 = run_cli(&["examples/water-rhf.toml", flag]);
        let stdout2 = String::from_utf8_lossy(&output2.stdout);
        assert!(
            stdout2.lines().any(|l| l.starts_with("SCF iter=")),
            "{flag} after the TOML path should also enable the trace, got:\n{stdout2}"
        );
    }
}

#[test]
fn toml_scf_verbose_key_prints_per_iteration_trace_without_cli_flag() {
    // Write a temp TOML that mirrors examples/water-rhf.toml plus `[scf]
    // verbose = true`, so a queued/batch job can opt in without changing the
    // invocation command.
    let root = workspace_root();
    let toml_body = r#"
[molecule]
xyz = "testdata/molecules/water.xyz"

[basis]
name = "sto-3g"

[method]
kind = "rhf"

[scf]
max_iter = 100
energy_conv = 1e-8
density_conv = 1e-7
diis_size = 8
integral_thresh = 1e-12
verbose = true
"#;
    let tmp_path = root.join("target").join("test-water-rhf-verbose.toml");
    std::fs::write(&tmp_path, toml_body).expect("write temp TOML");

    let output = Command::new(env!("CARGO_BIN_EXE_ferric-cli"))
        .arg(tmp_path.to_str().unwrap())
        .current_dir(&root)
        .env("OPENBLAS_NUM_THREADS", "1")
        .output()
        .expect("failed to run ferric-cli binary");

    let _ = std::fs::remove_file(&tmp_path);

    assert!(
        output.status.success(),
        "TOML verbose=true run should succeed; stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.lines().any(|l| l.starts_with("SCF iter=")),
        "[scf] verbose = true should print a per-iteration trace without \
         needing --verbose, got stdout:\n{stdout}"
    );
}
