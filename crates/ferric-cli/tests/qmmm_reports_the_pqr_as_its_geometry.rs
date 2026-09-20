//! With `[qmmm]`, the geometry comes from the PQR -- and the CLI must SAY so.
//!
//! Every `on {}` header and the run log's molecule `path` read
//! `cfg.molecule.xyz`. When `[qmmm]` is present that file is never read: the
//! PQR supplies both coordinates and MM charges, and an xyz supplies no
//! charges, so the PQR has to win. The headers were still naming the xyz.
//!
//! That is not cosmetic, and this test exists because the stale header
//! actually misled someone. `examples/water-qmmm.toml` names
//! `testdata/molecules/water.xyz`, an optimized HF/cc-pVDZ water, while the
//! PQR carries a DIFFERENT geometry. The run printed `on water.xyz`, so the
//! obvious way to get the vacuum reference for a G4 binding difference --
//! delete `[qmmm]` and re-run -- silently switched molecules and moved the
//! answer by 0.13 kcal/mol, into exactly the difference the procedure exists
//! to protect.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Resolved at RUN TIME: `env!("CARGO_BIN_EXE_...")` points at the BUILD job
/// under `cargo nextest archive`. Same reasoning as `epistemic_warning.rs`.
fn ferric_cli_bin() -> PathBuf {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(profile_dir) = exe.parent().and_then(Path::parent) {
            let p = profile_dir.join("ferric-cli");
            if p.is_file() {
                return p;
            }
        }
    }
    PathBuf::from(env!("CARGO_BIN_EXE_ferric-cli"))
}

fn workspace_root() -> PathBuf {
    let looks_like_root = |p: &Path| {
        p.join("Cargo.toml").is_file() && p.join("examples").is_dir() && p.join("testdata").is_dir()
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

#[test]
fn the_header_names_the_pqr_not_the_unused_xyz() {
    let root = workspace_root();
    let out = Command::new(ferric_cli_bin())
        .arg("examples/water-qmmm.toml")
        .current_dir(&root)
        .env("OPENBLAS_NUM_THREADS", "1")
        .output()
        .expect("failed to run ferric-cli");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "run failed:\n{stdout}");

    let header = stdout
        .lines()
        .find(|l| l.starts_with("RHF/"))
        .unwrap_or_else(|| panic!("no RHF header in:\n{stdout}"));

    assert!(
        header.contains("water_na.pqr"),
        "header must name the PQR the geometry came from, got: {header}"
    );
    // The negative half. Without it the assertion above passes on a header
    // that names BOTH, which would be just as confusing.
    assert!(
        !header.contains("water.xyz"),
        "header must NOT name the unread xyz, got: {header}"
    );

    // And the energy must be untouched by the reporting change.
    assert!(
        stdout.contains("-74.9653197421"),
        "embedded energy changed:\n{stdout}"
    );

    // Clean up the run log this writes next to the example.
    let _ = std::fs::remove_file(root.join("examples/water-qmmm.ferric.jsonl"));
}

#[test]
fn a_plain_run_still_names_its_xyz() {
    // Negative control: the rewrite must apply ONLY when [qmmm] is present.
    // Without this, setting `molecule.xyz` to a constant would pass the test
    // above.
    let root = workspace_root();
    let out = Command::new(ferric_cli_bin())
        .arg("examples/water-rhf.toml")
        .current_dir(&root)
        .env("OPENBLAS_NUM_THREADS", "1")
        .output()
        .expect("failed to run ferric-cli");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "run failed:\n{stdout}");
    let header = stdout
        .lines()
        .find(|l| l.starts_with("RHF/"))
        .unwrap_or_else(|| panic!("no RHF header in:\n{stdout}"));
    assert!(
        header.contains(".xyz"),
        "a non-QM/MM run must still name its xyz, got: {header}"
    );
    let _ = std::fs::remove_file(root.join("examples/water-rhf.ferric.jsonl"));
}
