//! The SCF certificate through the CLI binary: printed after the energy and
//! written to the JSON run log (`run_end.extra.certificate` for closed-shell
//! kinds, a `certificate` record for the open-shell runs that return before
//! `run_end`), stamped with the build.
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
    let dir = root.join("target").join("scf_certificate_cli_test");
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

fn record<'a>(records: &'a [serde_json::Value], kind: &str) -> &'a serde_json::Value {
    records
        .iter()
        .find(|r| r["record"] == kind)
        .unwrap_or_else(|| panic!("no {kind} record in {records:?}"))
}

#[test]
fn rhf_prints_and_logs_a_proven_stamped_certificate() {
    let (stdout, records) = run(
        "rhf",
        "[molecule]\nxyz = \"testdata/molecules/water.xyz\"\n[basis]\nname = \"sto-3g\"\n\
         [method]\nkind = \"rhf\"\n",
    );
    assert!(
        stdout.contains("certificate\n  capability  = rhf\n"),
        "{stdout}"
    );
    assert!(
        stdout.contains("  grade       = Proven (site/src/reference/validation.md)"),
        "{stdout}"
    );
    let c = &record(&records, "run_end")["extra"]["certificate"];
    assert_eq!(c["capability"], "rhf");
    assert_eq!(c["grade"], "proven");
    assert_eq!(c["build"]["commit"], ferric_build_info::COMMIT);
    assert_eq!(c["warnings"], serde_json::json!([]));
    // Round-trips into the strict core type.
    let back: ferric_core::certificate::Certificate =
        serde_json::from_value(c.clone()).expect("certificate JSON parses");
    assert_eq!(back.grade, ferric_core::certificate::Grade::Proven);
}

#[test]
fn uhf_logs_a_certificate_record_with_the_stability_caveat() {
    let (stdout, records) = run(
        "uhf",
        "[molecule]\nxyz = \"testdata/molecules/h_atom.xyz\"\nmultiplicity = 2\n\
         [basis]\nname = \"sto-3g\"\n[method]\nkind = \"uhf\"\n",
    );
    assert!(
        stdout.contains("certificate\n  capability  = uhf\n"),
        "{stdout}"
    );
    let c = record(&records, "certificate");
    assert_eq!(c["capability"], "uhf");
    assert_eq!(c["grade"], "proven");
    let warned = c["warnings"].as_array().unwrap().iter().any(|w| {
        w.as_str()
            .unwrap()
            .starts_with("internal stability not checked")
    });
    assert!(warned, "{c}");
}
