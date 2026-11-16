//! `[optimize] coordinates = "internal"` with a fixed-frame term
//! (`[external_potential]` charges or field, `[qmmm]` MM charges) is refused
//! through the real binary before any SCF: exit status 1 and an error that
//! names the cause and suggests `cartesian`. The library rule and its
//! invariant counter-cases are `crates/ferric-scf/tests/internal_coords_fixed_frame.rs`.
//!
//! The refusal is at optimizer dispatch, which no SCF precedes; the library
//! test pins "before any SCF" with an unknown basis name.

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

fn run(name: &str, toml_text: &str) -> (Option<i32>, String) {
    let dir = std::env::temp_dir().join(format!("ferric_internal_fixed_frame_{name}"));
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
    (
        out.status.code(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

const HEAD: &str = "[molecule]\nxyz = \"testdata/molecules/water.xyz\"\n[basis]\nname = \"sto-3g\"\n\
                    [method]\nkind = \"rhf\"\ntask = \"optimize\"\n[optimize]\ncoordinates = \"internal\"\n";

fn assert_refused(name: &str, extra: &str, needle: &str) {
    assert_refused_kind(name, "rhf", extra, needle);
}

fn assert_refused_kind(name: &str, kind: &str, extra: &str, needle: &str) {
    let head = HEAD.replace("kind = \"rhf\"", &format!("kind = \"{kind}\""));
    let (code, stderr) = run(name, &format!("{head}{extra}"));
    assert_eq!(code, Some(1), "{name}: {stderr}");
    assert!(stderr.contains("cannot optimize with"), "{name}: {stderr}");
    assert!(
        stderr.contains(needle),
        "{name}: cause {needle:?} missing: {stderr}"
    );
    assert!(stderr.contains("cartesian"), "{name}: {stderr}");
}

#[test]
fn point_charges_with_internal_coordinates_are_refused() {
    assert_refused(
        "charges",
        "[[external_potential.point_charges]]\nq = 0.4\nx = 4.0\ny = 0.0\nz = 0.0\n",
        "charges",
    );
}

#[test]
fn a_field_with_internal_coordinates_is_refused() {
    assert_refused(
        "field",
        "[external_potential]\nfield = [0.02, 0.0, 0.0]\n",
        "field",
    );
}

#[test]
fn qmmm_with_internal_coordinates_is_refused() {
    assert_refused(
        "qmmm",
        "[qmmm]\npqr = \"testdata/molecules/water_na.pqr\"\nqm_indices = [0, 1, 2]\n",
        "charges",
    );
}

/// The RI-MP2 optimizer is driven through the closure entry point, which has
/// no SCF config of its own: only the CLI dispatch guard covers it.
#[test]
fn rimp2_with_charges_and_internal_coordinates_is_refused() {
    assert_refused_kind(
        "rimp2",
        "rimp2",
        "[mp2]\nauxbasis = \"def2-svp-rifit\"\n\
         [[external_potential.point_charges]]\nq = 0.4\nx = 4.0\ny = 0.0\nz = 0.0\n",
        "charges",
    );
}
