//! The order in which the molecular run reports input errors around the
//! prepared system (`prepare_system` in `src/lib.rs`): a frozen-core error
//! (read on the ECP molecule before any integral), then a basis error, then
//! a bad `[scf] screening` value. Each is `error: ...` on stderr with exit
//! status 1 and nothing after it. An input wrong in several ways reports the
//! first, as the run did before the system existed.

use std::path::PathBuf;
use std::process::{Command, Output};

#[path = "support/cli_bin.rs"]
mod cli_bin;
use cli_bin::ferric_cli_bin;

const HI: &str = "2\n\nH 0 0 0\nI 0 0 1.61\n";
const WATER: &str = "3\n\nO 0 0 0.117\nH 0 0.757 -0.467\nH 0 -0.757 -0.467\n";

/// Run the CLI on `body` with `{xyz}` replaced by a file holding `xyz`. The
/// files go to a per-process directory under the system temp dir.
fn run(tag: &str, xyz: &str, body: &str) -> Output {
    let dir: PathBuf =
        std::env::temp_dir().join(format!("ferric-prepare-error-order-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let xyz_path = dir.join(format!("{tag}.xyz"));
    let toml_path = dir.join(format!("{tag}.toml"));
    std::fs::write(&xyz_path, xyz).expect("write xyz");
    let body = body.replace("{xyz}", &xyz_path.display().to_string());
    std::fs::write(&toml_path, body).expect("write toml");
    Command::new(ferric_cli_bin())
        .arg(&toml_path)
        .current_dir(&dir)
        .env("OPENBLAS_NUM_THREADS", "1")
        .output()
        .expect("failed to run ferric-cli binary")
}

fn toml(basis: &str, kind: &str, extra: &str) -> String {
    format!(
        "[molecule]\nxyz = \"{{xyz}}\"\n\n[basis]\nname = \"{basis}\"\n\n\
         [method]\nkind = \"{kind}\"\n\n{extra}"
    )
}

/// The one `error:` line of a failed run, after checking status 1.
fn error_line(out: &Output) -> String {
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "stderr:\n{stderr}");
    let errors: Vec<&str> = stderr.lines().filter(|l| l.starts_with("error:")).collect();
    assert_eq!(errors.len(), 1, "stderr:\n{stderr}");
    assert!(out.stdout.is_empty(), "stdout should be empty");
    errors[0].to_string()
}

const BAD_SCREENING: &str = "[scf]\nscreening = \"bogus\"\n";
const MISSING_I: &str =
    "error: basis error: basis \"cc-pvdz\" has no shells for I (Z=53); it covers H-Ar, Ca-Kr";
const SCREENING_ERR: &str = "error: [scf] screening: General error: unknown screening kind \
     \"bogus\"; expected \"schwarz\", \"csb\" or \"csam\"";

#[test]
fn a_bad_screening_value_alone_is_named() {
    let out = run("screening", WATER, &toml("cc-pvdz", "rhf", BAD_SCREENING));
    assert_eq!(error_line(&out), SCREENING_ERR);
}

#[test]
fn a_basis_error_is_reported_before_a_bad_screening_value() {
    let alone = run("basis", HI, &toml("cc-pvdz", "rhf", ""));
    assert_eq!(error_line(&alone), MISSING_I);
    let both = run(
        "basis_screening",
        HI,
        &toml("cc-pvdz", "rhf", BAD_SCREENING),
    );
    assert_eq!(error_line(&both), MISSING_I);
}

#[test]
fn a_frozen_core_error_is_reported_before_a_bad_screening_value() {
    let mp2 = "[mp2]\nauxbasis = \"cc-pvdz-ri\"\nfrozen_core = 9\n";
    let both = run(
        "fc_screening",
        WATER,
        &toml("cc-pvdz", "rimp2", &format!("{mp2}\n{BAD_SCREENING}")),
    );
    let line = error_line(&both);
    assert!(
        line.starts_with("error: [mp2] frozen_core = 9 freezes all 5 occupied orbital(s)"),
        "{line}"
    );
}
