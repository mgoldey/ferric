//! `rs-mp2-rpa` with `attenuator = "terf"` and formulation B refuses PDEP
//! truncation through the real binary: a nonzero exit and a message naming
//! `trunc_thresh`.
//!
//! `[rpa] trunc_thresh` is the only dRPA option the CLI passes to
//! `rs-mp2-rpa`, so it is the only one of the four library refusals
//! (`ferric-rpa/tests/terf_refusals.rs`) the CLI can reach. The refusal fires
//! before any integral is built, so no terf table is read. The control runs
//! the same input with erf and must succeed.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

#[path = "support/cli_bin.rs"]
mod cli_bin;
use cli_bin::ferric_cli_bin;

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

fn run(name: &str, attenuator: &str) -> Output {
    let toml_text = format!(
        "[molecule]\nxyz = \"testdata/molecules/h2.xyz\"\n[basis]\nname = \"sto-3g\"\n\
         [method]\nkind = \"rs-mp2-rpa\"\n\
         [mp2]\nauxbasis = \"cc-pvdz-ri\"\nattenuator = \"{attenuator}\"\nformulation = \"delta-lr\"\n\
         [rpa]\ntrunc_thresh = 1e-4\n"
    );
    let dir = std::env::temp_dir().join(format!("ferric_terf_refusals_{name}"));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("input.toml");
    std::fs::write(&path, toml_text).unwrap();
    let out = Command::new(ferric_cli_bin())
        .arg(&path)
        .arg("--no-json")
        .current_dir(workspace_root())
        .env("OPENBLAS_NUM_THREADS", "1")
        .output()
        .expect("failed to run ferric-cli");
    let _ = std::fs::remove_dir_all(&dir);
    out
}

#[test]
fn terf_b_with_trunc_thresh_is_refused_by_the_cli() {
    let out = run("terf", "terf");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "stderr:\n{stderr}");
    assert!(
        stderr.contains("trunc_thresh") && stderr.contains("is not supported"),
        "stderr does not name the refused option:\n{stderr}"
    );
}

#[test]
fn erf_b_with_trunc_thresh_runs() {
    let out = run("erf", "erf");
    assert!(
        out.status.success(),
        "erf B with trunc_thresh failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}
