//! `[mp2] r0_sweep` must be a pure amortization: sweeping N values of r0 in one
//! job has to give exactly what N separate single-r0 jobs give.
//!
//! This exists because an equivalent feature was present, uncommitted, during
//! the 2026-07-22/23 A24 production sweeps and was then lost — leaving
//! committed output data that no committed code could regenerate, and a
//! 5x-more-expensive path as the only way to redo it. A regression test is the
//! cheap way to keep that from happening twice.
//!
//! The saving is real and worth protecting: the SCF dominates a single-r0 job
//! at aug-cc-pVQZ, so an N-point scan reusing one SCF is roughly N times
//! cheaper than N runs.

use std::path::PathBuf;
use std::process::Command;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("ferric-cli manifest dir should be workspace_root/crates/ferric-cli")
        .to_path_buf()
}

/// terf needs the interpolation tables; skip cleanly when they are absent
/// rather than failing on an unrelated machine.
fn terf_dir() -> Option<String> {
    for c in [
        std::env::var("FERRIC_TERF_TABLE_DIR").unwrap_or_default(),
        "/home/matt/qc/terf-tables-data".into(),
    ] {
        if !c.is_empty() && PathBuf::from(&c).join("16_4_2.bin").exists() {
            return Some(c);
        }
    }
    None
}

fn run(body: &str, name: &str, tdir: &str) -> String {
    let root = workspace_root();
    let path = root.join("target").join(format!("r0_sweep_test_{name}.toml"));
    std::fs::write(&path, body).expect("write temp toml");
    let out = Command::new(env!("CARGO_BIN_EXE_ferric-cli"))
        .arg(&path)
        .current_dir(&root)
        .env("OPENBLAS_NUM_THREADS", "1")
        .env("FERRIC_TERF_TABLE_DIR", tdir)
        .output()
        .expect("failed to run ferric-cli");
    assert!(
        out.status.success(),
        "ferric-cli failed for {name}:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

fn corr_energies(stdout: &str) -> Vec<String> {
    stdout
        .lines()
        .filter_map(|l| l.trim().strip_prefix("E_corr Δ-form (B)"))
        .map(|s| s.trim_start_matches([' ', '=']).trim().to_string())
        .collect()
}

fn toml_for(r0_line: &str) -> String {
    format!(
        r#"
[molecule]
xyz = "testdata/molecules/water.xyz"
[basis]
name = "cc-pvdz"
[method]
kind = "rs-mp2-rpa"
[mp2]
auxbasis = "cc-pvdz-ri"
attenuator = "terf"
{r0_line}
formulation = "delta-lr"
frozen_core = 1
"#
    )
}

#[test]
fn r0_sweep_matches_separate_single_r0_runs() {
    let Some(tdir) = terf_dir() else {
        eprintln!("skipping: terf interpolation tables not found");
        return;
    };

    let swept = corr_energies(&run(
        &toml_for("r0_sweep = [0.75, 1.0]"),
        "sweep",
        &tdir,
    ));
    assert_eq!(swept.len(), 2, "expected one result block per r0, got {swept:?}");

    let a = corr_energies(&run(&toml_for("r0 = 0.75"), "single_a", &tdir));
    let b = corr_energies(&run(&toml_for("r0 = 1.0"), "single_b", &tdir));
    assert_eq!(a.len(), 1);
    assert_eq!(b.len(), 1);

    assert_eq!(
        swept,
        vec![a[0].clone(), b[0].clone()],
        "r0_sweep must reproduce separate single-r0 runs exactly — it shares one \
         SCF, which is an amortization, NOT an approximation"
    );

    // Guard against a vacuous pass: the two r0 must actually give different
    // energies, or "they all match" would prove nothing about r0 being applied.
    assert_ne!(
        a[0], b[0],
        "r0 = 0.75 and r0 = 1.0 produced identical energies — r0 may be ignored"
    );
}

#[test]
fn r0_sweep_rejects_erf_and_empty() {
    let Some(tdir) = terf_dir() else { return };
    let root = workspace_root();

    for (name, body, want) in [
        (
            "erf",
            toml_for("r0_sweep = [0.75, 1.0]").replace("\"terf\"", "\"erf\""),
            "requires attenuator",
        ),
        ("empty", toml_for("r0_sweep = []"), "empty"),
        ("negative", toml_for("r0_sweep = [0.5, -1.0]"), "> 0"),
    ] {
        let path = root.join("target").join(format!("r0_sweep_bad_{name}.toml"));
        std::fs::write(&path, &body).unwrap();
        let out = Command::new(env!("CARGO_BIN_EXE_ferric-cli"))
            .arg(&path)
            .current_dir(&root)
            .env("OPENBLAS_NUM_THREADS", "1")
            .env("FERRIC_TERF_TABLE_DIR", &tdir)
            .output()
            .unwrap();
        assert!(!out.status.success(), "[{name}] should have been rejected");
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(
            err.contains(want),
            "[{name}] expected an error mentioning {want:?}, got:\n{err}"
        );
    }
}
