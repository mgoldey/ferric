//! The certificates of `task = "optimize"` (a `certificate` run-log record) and
//! `task = "frequencies"` (`certificate` inside the `frequencies` record)
//! through the CLI binary, stamped with the build.
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
    let dir = root.join("target").join("task_certificate_cli_test");
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

fn approx<'a>(c: &'a serde_json::Value, name: &str) -> &'a str {
    c["approximations"]
        .as_array()
        .unwrap()
        .iter()
        .find(|a| a["name"] == name)
        .unwrap_or_else(|| panic!("no {name} approximation in {c}"))["detail"]
        .as_str()
        .unwrap()
}

const H2: &str = "[molecule]\nxyz = \"testdata/molecules/h2.xyz\"\n[basis]\nname = \"sto-3g\"\n";

#[test]
fn optimize_prints_and_logs_the_final_point_certificate() {
    let (stdout, records) = run(
        "opt_h2",
        &format!("{H2}[method]\nkind = \"rhf\"\ntask = \"optimize\"\n"),
    );
    assert!(
        stdout.contains("certificate\n  capability  = optimize:rhf\n"),
        "{stdout}"
    );
    let c = record(&records, "certificate");
    assert_eq!(c["capability"], "optimize:rhf");
    assert_eq!(c["grade"], "proven");
    assert_eq!(c["grade_source"], "site/src/reference/validation.md");
    assert_eq!(c["build"]["commit"], ferric_build_info::COMMIT);
    assert_eq!(approx(c, "gradient"), "analytic");
    assert!(approx(c, "optimizer").starts_with("BFGS"), "{c}");
    assert_eq!(c["warnings"], serde_json::json!([]));
    let mut body = c.clone();
    // The run-log envelope is not part of the certificate.
    body.as_object_mut()
        .unwrap()
        .retain(|k, _| !matches!(k.as_str(), "record" | "seq" | "t" | "ts"));
    let back: ferric_core::certificate::Certificate =
        serde_json::from_value(body).expect("certificate JSON parses");
    assert_eq!(back.grade, ferric_core::certificate::Grade::Proven);
}

#[test]
fn an_unconverged_optimization_is_not_graded() {
    let (_, records) = run(
        "opt_h2_one_step",
        &format!("{H2}[method]\nkind = \"rhf\"\ntask = \"optimize\"\n[optimize]\nmax_steps = 1\n"),
    );
    let c = record(&records, "certificate");
    assert_eq!(c["grade"], "ungraded");
    let why = c["warnings"].as_array().unwrap().iter().any(|w| {
        w.as_str()
            .unwrap()
            .contains("grade capped from Proven to not graded: the optimizer did not converge")
    });
    assert!(why, "{c}");
}

#[test]
fn rimp2_optimize_records_its_gradient_and_caps_at_the_scf() {
    let (_, records) = run(
        "opt_rimp2",
        &format!("{H2}[method]\nkind = \"rimp2\"\ntask = \"optimize\"\n"),
    );
    let c = record(&records, "certificate");
    assert_eq!(c["capability"], "optimize:rimp2");
    assert_eq!(c["grade"], "proven");
    assert_eq!(approx(c, "gradient"), "analytic (Z-vector)");
}

#[test]
fn frequencies_certificate_sits_beside_the_frequencies_fields() {
    let (stdout, records) = run(
        "freq_h2_analytic",
        &format!("{H2}[method]\nkind = \"rhf\"\ntask = \"frequencies\"\n"),
    );
    assert!(
        stdout.contains("certificate\n  capability  = frequencies:rhf\n"),
        "{stdout}"
    );
    let f = record(&records, "frequencies");
    assert!(f["frequencies_cm1"].is_array());
    let c = &f["certificate"];
    assert_eq!(c["capability"], "frequencies:rhf");
    assert_eq!(c["grade"], "proven");
    assert_eq!(c["build"]["commit"], ferric_build_info::COMMIT);
    assert_eq!(approx(c, "hessian"), f["hessian_source"].as_str().unwrap());
    assert_eq!(approx(c, "hessian"), "analytic");
}

#[test]
fn finite_difference_frequencies_record_delta_and_gradient_count() {
    let (_, records) = run(
        "freq_h2_fd",
        &format!(
            "{H2}[method]\nkind = \"rhf\"\ntask = \"frequencies\"\n[frequencies]\nhessian = \"fd\"\n"
        ),
    );
    let f = record(&records, "frequencies");
    let c = &f["certificate"];
    let h = approx(c, "hessian");
    assert!(h.starts_with("finite-difference"), "{h}");
    assert!(
        h.contains(&format!(
            "{} gradient evaluation(s)",
            f["n_gradient_evaluations"]
        )),
        "{h}"
    );
    assert_eq!(c["grade"], "proven");
}
