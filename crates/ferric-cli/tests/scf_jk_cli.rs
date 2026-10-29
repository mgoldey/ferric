//! `[scf] jk` through the real `ferric-cli` binary.
//!
//! HOW THESE FAIL IF BROKEN:
//! * dropping the `apply_scf_jk` call in `run()` makes the shipped
//!   `examples/water-rimp2-rijk.toml` run the four-centre SCF: its reference
//!   energy then EQUALS the exact run (the `> 1e-6` assert fails) and misses
//!   the explicit `df_j_aux`/`df_k_aux` run by the RI-JK fitting error;
//! * logging the raw `[scf] jk` key instead of the effective route makes
//!   rs-mp2-rpa with no `jk` log `null` instead of "rijk";
//! * removing a load-time refusal (oo-rimp2 fitted, df_increments + jk,
//!   rs-mp2-rpa + k_builder) lets the run start instead of failing.
//!
//! Energies come from the JSON run log (`run_end.energy` is the reference SCF
//! energy for a correlated kind, at full f64 precision), not the 10-decimal
//! printout.

use std::path::{Path, PathBuf};
use std::process::Command;

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
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("ferric-cli manifest dir should be workspace_root/crates/ferric-cli")
        .to_path_buf()
}

/// Run the TOML `body` and return the process output and the parsed JSON
/// run-log records (empty when the run was refused before logging).
fn run(tag: &str, body: &str) -> (std::process::Output, Vec<serde_json::Value>) {
    let root = workspace_root();
    let dir = root.join("target");
    std::fs::create_dir_all(&dir).expect("create target/");
    let toml = dir.join(format!("scf_jk_cli_{tag}.toml"));
    let log = dir.join(format!("scf_jk_cli_{tag}.jsonl"));
    let _ = std::fs::remove_file(&log);
    std::fs::write(&toml, body).expect("write temp toml");
    let out = Command::new(ferric_cli_bin())
        .arg(&toml)
        .arg("--json")
        .arg(&log)
        .current_dir(&root)
        .env("OPENBLAS_NUM_THREADS", "1")
        .env("RAYON_NUM_THREADS", "2")
        .output()
        .expect("failed to run ferric-cli");
    let records = std::fs::read_to_string(&log)
        .unwrap_or_default()
        .lines()
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect();
    (out, records)
}

fn run_ok(tag: &str, body: &str) -> Vec<serde_json::Value> {
    let (out, records) = run(tag, body);
    assert!(
        out.status.success(),
        "{tag} must run.\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    records
}

fn record<'a>(records: &'a [serde_json::Value], kind: &str) -> &'a serde_json::Value {
    records
        .iter()
        .find(|r| r["record"] == kind)
        .unwrap_or_else(|| panic!("no {kind} record in {records:?}"))
}

fn scf_energy(records: &[serde_json::Value]) -> f64 {
    record(records, "run_end")["energy"]
        .as_f64()
        .expect("run_end.energy is a number")
}

fn shipped_example() -> String {
    std::fs::read_to_string(workspace_root().join("examples/water-rimp2-rijk.toml"))
        .expect("read examples/water-rimp2-rijk.toml")
}

/// The shipped example's RI-JK reference differs from the exact one by the
/// fitting error and equals the explicit `df_j_aux`/`df_k_aux` spelling (same
/// `RhfConfig`, no DF-guess pre-stage) to 1e-12.
#[test]
fn shipped_rijk_example_runs_the_fitted_reference() {
    let example = shipped_example();
    let jk_lines = "jk = \"rijk\"\njk_aux = \"def2-universal-jkfit\"\n";
    assert!(
        example.contains(jk_lines),
        "the example's [scf] lines moved; update this test:\n{example}"
    );
    let rijk = run_ok("example", &example);
    let exact = run_ok("exact", &example.replace(jk_lines, ""));
    let explicit = run_ok(
        "explicit",
        &example.replace(
            jk_lines,
            "df_j_aux = \"def2-universal-jkfit\"\ndf_k_aux = \"def2-universal-jkfit\"\n\
             df_guess = false\n",
        ),
    );
    let (e_rijk, e_exact, e_explicit) =
        (scf_energy(&rijk), scf_energy(&exact), scf_energy(&explicit));
    assert!(
        (e_rijk - e_exact).abs() > 1e-6,
        "jk = \"rijk\" ran the four-centre SCF: {e_rijk:.12} vs exact {e_exact:.12}"
    );
    assert!(
        (e_rijk - e_explicit).abs() <= 1e-12,
        "jk = \"rijk\" is not df_j_aux = df_k_aux = jkfit: {e_rijk:.14} vs {e_explicit:.14}"
    );
    let cfg = |r: &[serde_json::Value]| record(r, "run_start")["config"].clone();
    let c = cfg(&rijk);
    assert_eq!(c["scf_jk"], "rijk", "{c}");
    assert_eq!(c["jk_aux"], "def2-universal-jkfit", "{c}");
    assert_eq!(c["k_builder"], serde_json::Value::Null, "{c}");
    assert_eq!(cfg(&exact)["scf_jk"], "exact");
    assert_eq!(cfg(&explicit)["scf_jk"], "rijk");
}

/// rs-mp2-rpa with no `jk` runs RI-JK (its historical default), and the
/// run log says so instead of echoing the absent key as null.
#[test]
fn rs_mp2_rpa_logs_its_effective_rijk_default() {
    let body = "[molecule]\nxyz = \"testdata/molecules/water.xyz\"\n\n\
                [basis]\nname = \"sto-3g\"\n\n[method]\nkind = \"rs-mp2-rpa\"\n\n\
                [mp2]\nauxbasis = \"cc-pvdz-ri\"\n";
    let r = run_ok("rs_default", body);
    let c = &record(&r, "run_start")["config"];
    assert_eq!(c["scf_jk"], "rijk", "{c}");
    assert_eq!(c["jk_aux"], "def2-universal-jkfit", "{c}");
}

fn assert_refused(tag: &str, kind: &str, scf: &str, needle: &str) {
    let body = format!(
        "[molecule]\nxyz = \"testdata/molecules/water.xyz\"\n\n\
         [basis]\nname = \"sto-3g\"\n\n[method]\nkind = \"{kind}\"\n\n\
         [scf]\n{scf}\n[mp2]\nauxbasis = \"cc-pvdz-ri\"\n"
    );
    let (out, _) = run(tag, &body);
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        !out.status.success(),
        "{tag} must be refused; stderr: {err}"
    );
    assert!(err.contains(needle), "{tag}: want {needle:?} in: {err}");
}

#[test]
fn fitted_references_that_would_not_run_are_refused_at_load() {
    assert_refused(
        "oo_rijk",
        "oo-rimp2",
        "jk = \"rijk\"\n",
        "orbital-optimised MP2 rebuilds exact four-centre J/K",
    );
    assert_refused(
        "dfi_rijk",
        "rimp2",
        "jk = \"rijk\"\ndf_increments = true\n",
        "df_increments cannot be combined with a fitted reference",
    );
    assert_refused(
        "dfi_rijcosx",
        "rimp2",
        "jk = \"rijcosx\"\ndf_increments = true\n",
        "df_increments cannot be combined with a fitted reference",
    );
    assert_refused(
        "rs_link",
        "rs-mp2-rpa",
        "k_builder = \"link\"\n",
        "[scf] k_builder = \"link\" conflicts with [scf] jk = \"rijk\"",
    );
}
