//! `[cdft]` and `[qmmm]` polarizable embedding refuse what they cannot honour,
//! through the real binary and before any integral is computed.
//!
//! The numerical equivalence with Python `run_cdft` / `run_qmmm` is
//! `crates/ferric-python/tests/test_cli_cdft_thole.py`; the value rules are
//! unit-tested in `src/cdft.rs` and `src/config.rs`. These pin that the
//! refusals reach the user as a nonzero exit and a message naming the key.

use std::path::{Path, PathBuf};
use std::process::Command;

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

/// Run the CLI on `toml_text`; the status must be 1 and stderr must hold `needle`.
fn assert_refused(name: &str, toml_text: &str, needle: &str) {
    let dir = std::env::temp_dir().join(format!("ferric_cdft_thole_{name}"));
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
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "{name}: stderr:\n{stderr}");
    assert!(
        stderr.contains(needle),
        "{name}: stderr lacks {needle:?}:\n{stderr}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

const LIH: &str = "[molecule]\nxyz = \"testdata/molecules/lih.xyz\"\n[basis]\nname = \"sto-3g\"\n";
const CON: &str = "[cdft]\n[[cdft.constraint]]\natoms = [0]\ntarget = 2.2\n";

#[test]
fn cdft_on_a_kind_that_does_not_run_it_is_refused() {
    assert_refused(
        "kind",
        &format!("{LIH}[method]\nkind = \"rhf\"\n{CON}"),
        "method.kind = \"uhf\" only",
    );
}

#[test]
fn cdft_is_energy_only() {
    assert_refused(
        "task",
        &format!("{LIH}[method]\nkind = \"uhf\"\ntask = \"optimize\"\n{CON}"),
        "task = \"energy\" only",
    );
}

#[test]
fn cdft_refuses_the_scf_keys_it_does_not_read() {
    assert_refused(
        "scfkey",
        &format!("{LIH}[method]\nkind = \"uhf\"\n[scf]\nsoscf = true\n{CON}"),
        "[scf] soscf is not read by a [cdft] run",
    );
}

#[test]
fn cdft_without_a_constraint_is_refused() {
    assert_refused(
        "empty",
        &format!("{LIH}[method]\nkind = \"uhf\"\n[cdft]\nlambda_tol = 1e-5\n"),
        "at least one [[cdft.constraint]]",
    );
}

#[test]
fn cdft_atom_index_out_of_range_is_refused_before_the_scf() {
    assert_refused(
        "atom",
        &format!(
            "{LIH}[method]\nkind = \"uhf\"\n[cdft]\n[[cdft.constraint]]\natoms = [5]\ntarget = 1.0\n"
        ),
        "names atom 5",
    );
}

const WATER_THOLE: &str = "[molecule]\nxyz = \"unused.xyz\"\n[basis]\nname = \"sto-3g\"\n";
const PQR: &str = "pqr = \"testdata/molecules/water_thole_sites.pqr\"\nqm_indices = [0, 1, 2]\n";
const ALPHAS: &str = "polarizabilities_angstrom3 = [0.0, 0.0, 0.0, 0.07, 0.07, 0.16]\n";

#[test]
fn polarizable_embedding_refuses_a_geometry_task() {
    assert_refused(
        "poltask",
        &format!(
            "{WATER_THOLE}[method]\nkind = \"rhf\"\ntask = \"optimize\"\n[qmmm]\n{PQR}{ALPHAS}"
        ),
        "task = \"energy\" only",
    );
}

#[test]
fn polarizable_embedding_refuses_a_correlated_kind() {
    assert_refused(
        "polkind",
        &format!("{WATER_THOLE}[method]\nkind = \"rimp2\"\n[qmmm]\n{PQR}{ALPHAS}"),
        "\"rhf\", \"uhf\" and \"ksdft\"",
    );
}

#[test]
fn thole_a_without_polarizabilities_is_refused() {
    assert_refused(
        "tholea",
        &format!("{WATER_THOLE}[method]\nkind = \"rhf\"\n[qmmm]\n{PQR}thole_a = 2.0\n"),
        "thole_a needs polarizabilities_angstrom3",
    );
}

#[test]
fn a_wrong_length_polarizability_list_is_refused() {
    assert_refused(
        "pollen",
        &format!(
            "{WATER_THOLE}[method]\nkind = \"rhf\"\n[qmmm]\n{PQR}polarizabilities_angstrom3 = [0.0, 0.5]\n"
        ),
        "6 atoms",
    );
}

#[test]
fn polarizable_embedding_refuses_x2c() {
    assert_refused(
        "polx2c",
        &format!(
            "{WATER_THOLE}[method]\nkind = \"rhf\"\n[scf]\nrelativistic = \"x2c1e\"\n[qmmm]\n{PQR}{ALPHAS}"
        ),
        "with [scf] relativistic is not supported",
    );
}

#[test]
fn cdft_refuses_x2c() {
    assert_refused(
        "cdftx2c",
        &format!("{LIH}[method]\nkind = \"uhf\"\n[scf]\nrelativistic = \"x2c1e\"\n{CON}"),
        "relativistic",
    );
}
