//! The SCF certificate through the CLI binary: printed after the energy and
//! written to the JSON run log (`run_end.extra.certificate` for closed-shell
//! kinds, a `certificate` record for the open-shell runs that return before
//! `run_end`), stamped with the build.
use std::path::PathBuf;
use std::process::Command;

#[path = "support/cli_bin.rs"]
mod cli_bin;
use cli_bin::ferric_cli_bin;

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
    // Water / STO-3G is inside the RHF energy anchor: its value, no warning.
    let e = &c["measured_error"];
    assert_eq!(e["value"], 1.8e-12, "{c}");
    assert_eq!(e["units"], "Ha");
    assert_eq!(
        e["evidence"],
        "crates/ferric-mp2/tests/validation_basis_coverage.rs"
    );
    // Round-trips into the strict core type.
    let back: ferric_core::certificate::Certificate =
        serde_json::from_value(c.clone()).expect("certificate JSON parses");
    assert_eq!(back.grade, ferric_core::certificate::Grade::Proven);
}

/// An `[external_potential]` is recorded as an approximation; the RHF
/// anchor keeps its value and the grade, and no warning is added.
#[test]
fn rhf_with_an_external_potential_records_the_approximation() {
    let (_, records) = run(
        "rhf_field",
        "[molecule]\nxyz = \"testdata/molecules/water.xyz\"\n[basis]\nname = \"sto-3g\"\n\
         [method]\nkind = \"rhf\"\n[external_potential]\nfield = [0.0, 0.0, 0.001]\n",
    );
    let c = &record(&records, "run_end")["extra"]["certificate"];
    assert_eq!(c["grade"], "proven", "{c}");
    assert_eq!(c["measured_error"]["value"], 1.8e-12, "{c}");
    assert_eq!(c["warnings"], serde_json::json!([]), "{c}");
    assert!(
        c["approximations"]
            .as_array()
            .unwrap()
            .iter()
            .any(|a| a["name"] == "external-potential"),
        "{c}"
    );
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

#[test]
fn rks_outside_its_anchor_warns_without_changing_the_grade() {
    // The RKS anchor covers H2S, HCl, SiH4 / def2-SVP, def2-TZVP with PBE,
    // B3LYP, HSE06: water / STO-3G is outside on element and basis.
    let (stdout, records) = run(
        "rks_outside",
        "[molecule]\nxyz = \"testdata/molecules/water.xyz\"\n[basis]\nname = \"sto-3g\"\n\
         [method]\nkind = \"ksdft\"\n[dft]\nfunctional = \"PBE\"\n",
    );
    let c = &record(&records, "run_end")["extra"]["certificate"];
    assert_eq!(c["grade"], "proven", "{c}");
    assert_eq!(c["measured_error"]["value"], 4.2e-12, "{c}");
    let w: Vec<&str> = c["warnings"]
        .as_array()
        .unwrap()
        .iter()
        .map(|w| w.as_str().unwrap())
        .collect();
    let ev = "crates/ferric-scf/tests/validation_ks_energies.rs";
    assert_eq!(
        w,
        [
            format!("element O (covered: H, Si, S, Cl) is outside the RKS energy anchor ({ev})"),
            format!("basis sto-3g is outside the RKS energy anchor ({ev})"),
        ],
        "{c}"
    );
    assert!(stdout.contains("  warning     = element O"), "{stdout}");
}

#[test]
fn rimp2_inside_its_anchor_carries_the_measured_error_and_no_warning() {
    let (_stdout, records) = run(
        "rimp2_inside",
        "[molecule]\nxyz = \"testdata/molecules/water.xyz\"\n[basis]\nname = \"cc-pvdz\"\n\
         [method]\nkind = \"rimp2\"\n[mp2]\nauxbasis = \"cc-pvdz-ri\"\n",
    );
    let r = records
        .iter()
        .find(|r| r["record"] == "result")
        .unwrap_or_else(|| panic!("no result record in {records:?}"));
    let c = &r["components"]["certificate"];
    assert_eq!(c["capability"], "rimp2", "{c}");
    assert_eq!(c["measured_error"]["value"], 2.4e-10, "{c}");
    assert_eq!(c["measured_error"]["quantity"], "correlation energy");
    assert_eq!(c["warnings"], serde_json::json!([]), "{c}");
}
