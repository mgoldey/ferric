//! Post-SCF certificates through the CLI binary: printed after the result
//! and written to the JSON run log (`result.components.certificate` for the
//! methods that log a `result` record, a `certificate` record for GW, BSE,
//! RPAx and PDEP-RPA), carrying the method's table grade and settings.
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
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf()
}

fn run(tag: &str, body: &str) -> (String, Vec<serde_json::Value>) {
    let root = workspace_root();
    let dir = root.join("target").join("post_scf_certificate_cli_test");
    std::fs::create_dir_all(&dir).expect("create the temp dir");
    let toml = dir.join(format!("{tag}.toml"));
    let log = dir.join(format!("{tag}.jsonl"));
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
        .expect("run ferric-cli");
    assert!(
        out.status.success(),
        "{tag}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let records = std::fs::read_to_string(&log)
        .expect("read the run log")
        .lines()
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect();
    (String::from_utf8_lossy(&out.stdout).into_owned(), records)
}

const WATER: &str =
    "[molecule]\nxyz = \"testdata/molecules/water.xyz\"\n[basis]\nname = \"sto-3g\"\n";

fn result_certificate<'a>(records: &'a [serde_json::Value], kind: &str) -> &'a serde_json::Value {
    let r = records
        .iter()
        .find(|r| r["record"] == "result" && r["kind"] == kind)
        .unwrap_or_else(|| panic!("no {kind} result record in {records:?}"));
    &r["components"]["certificate"]
}

fn names(c: &serde_json::Value) -> Vec<String> {
    c["approximations"]
        .as_array()
        .expect("approximations")
        .iter()
        .map(|a| a["name"].as_str().unwrap().to_string())
        .collect()
}

#[test]
fn rimp2_result_carries_its_certificate() {
    let (stdout, records) = run(
        "rimp2",
        &format!(
            "{WATER}[method]\nkind = \"rimp2\"\n[mp2]\nauxbasis = \"cc-pvdz-ri\"\nfrozen_core = 1\n"
        ),
    );
    assert!(
        stdout.contains("certificate\n  capability  = rimp2\n"),
        "{stdout}"
    );
    let c = result_certificate(&records, "rimp2");
    assert_eq!(c["capability"], "rimp2");
    assert_eq!(c["grade"], "proven");
    assert_eq!(c["build"]["commit"], ferric_build_info::COMMIT);
    assert_eq!(names(c), ["ri", "frozen-core"]);
    assert_eq!(c["approximations"][0]["detail"], "aux cc-pvdz-ri");
    let back: ferric_core::certificate::Certificate =
        serde_json::from_value(c.clone()).expect("certificate JSON parses");
    assert_eq!(back.grade, ferric_core::certificate::Grade::Proven);
}

#[test]
fn local_drpa_at_finite_eps_is_ungraded() {
    let (_, records) = run(
        "drpa_local",
        &format!(
            "{WATER}[method]\nkind = \"drpa\"\n[mp2]\nauxbasis = \"cc-pvdz-ri\"\n\
             [local]\nscheme = \"amplitude-threshold\"\neps = 1e-4\n"
        ),
    );
    let c = result_certificate(&records, "drpa");
    assert_eq!(c["capability"], "drpa");
    assert_eq!(c["grade"], "ungraded");
    assert!(names(c).contains(&"local".to_string()), "{c}");
    assert!(c["approximations"]
        .as_array()
        .unwrap()
        .iter()
        .any(|a| a["detail"]
            .as_str()
            .unwrap()
            .starts_with("amplitude-threshold eps = 1e-4")));
}

#[test]
fn ccsd_t_result_carries_its_table_grade() {
    let (_, records) = run(
        "ccsd_t",
        &format!("{WATER}[method]\nkind = \"ccsd(t)\"\n[mp2]\nauxbasis = \"cc-pvdz-ri\"\n"),
    );
    let c = result_certificate(&records, "ccsd(t)");
    assert_eq!(c["capability"], "ccsd(t)");
    assert_eq!(c["grade"], "proven_narrow");
}

#[test]
fn gw_logs_a_certificate_record_with_the_matched_settings_caveat() {
    let (stdout, records) = run(
        "gw",
        &format!(
            "{WATER}[method]\nkind = \"gw\"\n[rpa]\nauxbasis = \"cc-pvdz-ri\"\n\
             n_quad = 16\n[gw]\nmethod = \"g0w0\"\nqp_mos = [3, 6]\n"
        ),
    );
    assert!(
        stdout.contains("certificate\n  capability  = gw\n"),
        "{stdout}"
    );
    let c = records
        .iter()
        .find(|r| r["record"] == "certificate")
        .unwrap_or_else(|| panic!("no certificate record in {records:?}"));
    assert_eq!(c["capability"], "gw");
    assert_eq!(c["grade"], "smoke");
    let warned = c["warnings"].as_array().unwrap().iter().any(|w| {
        w.as_str()
            .unwrap()
            .contains("this run uses n_quad = 16, trunc_thresh = 1e-4")
    });
    assert!(warned, "{c}");
}
