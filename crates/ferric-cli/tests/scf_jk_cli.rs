//! `[scf] jk` through the real `ferric-cli` binary.
//!
//! HOW THESE FAIL IF BROKEN:
//! * an omitted `jk` that ran four-centre (or a certificate that did not
//!   name the fitted reference) fails `omitted_jk_runs_and_names_...`;
//! * dropping the `apply_scf_jk` call in `run()` makes the shipped
//!   `examples/water-rimp2-rijk.toml` run the four-centre SCF: its reference
//!   energy then EQUALS the exact run (the `> 1e-6` assert fails) and misses
//!   the explicit `df_j_aux`/`df_k_aux` run by the RI-JK fitting error;
//! * logging the raw `[scf] jk` key instead of the effective route makes
//!   rs-mp2-rpa with no `jk` log `null` instead of "rijk";
//! * removing a load-time refusal (oo-rimp2 + rijcosx, df_increments + jk,
//!   rs-mp2-rpa + k_builder) lets the run start instead of failing.
//!
//! Energies come from the JSON run log (`run_end.energy` is the reference SCF
//! energy for a correlated kind, at full f64 precision), not the 10-decimal
//! printout.

use std::path::{Path, PathBuf};
use std::process::Command;

#[path = "support/cli_bin.rs"]
mod cli_bin;
use cli_bin::ferric_cli_bin;

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
    let exact = run_ok("exact", &example.replace(jk_lines, "jk = \"exact\"\n"));
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

/// The approximation details of the correlated result's certificate.
fn certificate_details(records: &[serde_json::Value], name: &str) -> Vec<String> {
    let r = records
        .iter()
        .find(|r| r["record"] == "result")
        .unwrap_or_else(|| panic!("no result record in {records:?}"));
    r["components"]["certificate"]["approximations"]
        .as_array()
        .expect("approximations")
        .iter()
        .filter(|a| a["name"] == name)
        .map(|a| a["detail"].as_str().unwrap_or_default().to_string())
        .collect()
}

/// A closed-shell rimp2 with `jk` omitted runs RI-JK on the JK-fit set
/// matched to cc-pVDZ (cc-pvtz-jkfit): the energy equals the explicit
/// spelling to 1e-12 and differs from `jk = "exact"` by the fitting error;
/// the printout names the route and why, the run log records it, and the
/// certificate carries the reference's RI-J and RI-K with that aux. A
/// default that ran four-centre while logging RI-JK fails the energy
/// asserts; one that ran RI-JK while the certificate said exact fails the
/// certificate asserts.
#[test]
fn omitted_jk_runs_and_names_the_matched_rijk_reference() {
    let body = "[molecule]\nxyz = \"testdata/molecules/water.xyz\"\n\n\
                [basis]\nname = \"cc-pvdz\"\n\n[method]\nkind = \"rimp2\"\n\n\
                [mp2]\nauxbasis = \"cc-pvdz-ri\"\n";
    let (out, omitted) = run("omitted", body);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains(
            "Reference SCF J/K: rijk(cc-pvtz-jkfit) ([scf] jk omitted: the closed-shell \
             MP2-family default)"
        ),
        "{stdout}"
    );
    let with = |scf: &str| body.replace("[mp2]", &format!("[scf]\n{scf}\n[mp2]"));
    let explicit = run_ok(
        "omitted_explicit",
        &with("jk = \"rijk\"\njk_aux = \"cc-pvtz-jkfit\""),
    );
    let exact = run_ok("omitted_exact", &with("jk = \"exact\""));
    let (e_omit, e_expl, e_exact) = (
        scf_energy(&omitted),
        scf_energy(&explicit),
        scf_energy(&exact),
    );
    assert!(
        (e_omit - e_expl).abs() <= 1e-12,
        "omitted jk is not rijk(cc-pvtz-jkfit): {e_omit:.14} vs {e_expl:.14}"
    );
    assert!(
        (e_omit - e_exact).abs() > 1e-7,
        "omitted jk ran the four-centre SCF: {e_omit:.14} vs exact {e_exact:.14}"
    );
    let c = &record(&omitted, "run_start")["config"];
    assert_eq!(
        (&c["scf_jk"], &c["jk_aux"]),
        (&"rijk".into(), &"cc-pvtz-jkfit".into())
    );
    for name in ["scf ri-j", "scf ri-k"] {
        let d = certificate_details(&omitted, name);
        assert!(
            d.len() == 1 && d[0].starts_with("aux cc-pvtz-jkfit"),
            "{name}: {d:?}"
        );
        assert!(certificate_details(&exact, name).is_empty(), "exact {name}");
    }
}

/// An omitted `jk` with a key only the four-centre route reads keeps the
/// four-centre reference and says why.
#[test]
fn omitted_jk_with_a_four_centre_key_stays_exact() {
    let body = "[molecule]\nxyz = \"testdata/molecules/water.xyz\"\n\n\
                [basis]\nname = \"sto-3g\"\n\n[method]\nkind = \"rimp2\"\n\n\
                [scf]\nk_builder = \"link\"\n[mp2]\nauxbasis = \"cc-pvdz-ri\"\n";
    let (out, records) = run("omitted_link", body);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains(
            "Reference SCF J/K: exact, k_builder=link ([scf] jk omitted: [scf] k_builder \
             reads the four-centre route)"
        ),
        "{stdout}"
    );
    assert_eq!(record(&records, "run_start")["config"]["scf_jk"], "exact");
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
        "oo_rijcosx",
        "oo-rimp2",
        "jk = \"rijcosx\"\n",
        "COSX exchange has no OO route",
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
