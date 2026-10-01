//! The `[cell]` CLI path reproduces PySCF's periodic RHF energy end to end.
//!
//! `examples/h2-cell-rhf.toml` (Gamma) and `h2-cell-kpts-rhf.toml` (2x2x2)
//! are pinned against PySCF 2.13 FFTDF (exxdiv = 'ewald', precision 1e-12):
//! ferric agreed to 2e-10 and 1.4e-9 Ha/cell (measured 2026-09-25). The bar
//! 1e-8 sits above that FFTDF precision floor and far below the 0.6 Ha gap
//! between the two meshes, so a CLI that silently dropped the k-mesh (or the
//! Madelung shift, 0.35-0.71 Ha here) fails.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Resolved at RUN TIME: `env!("CARGO_BIN_EXE_...")` points at the BUILD job
/// under `cargo nextest archive`. Same reasoning as `epistemic_warning.rs`.
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

fn energy_of(example: &str) -> f64 {
    let out = Command::new(ferric_cli_bin())
        .arg(example)
        .current_dir(workspace_root())
        .env("OPENBLAS_NUM_THREADS", "1")
        .output()
        .expect("run ferric-cli");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "{example} failed:\n{stdout}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let line = stdout
        .lines()
        .find(|l| l.trim_start().starts_with("energy "))
        .unwrap_or_else(|| panic!("no energy line in:\n{stdout}"));
    line.split('=')
        .nth(1)
        .unwrap()
        .split_whitespace()
        .next()
        .unwrap()
        .parse()
        .unwrap()
}

#[test]
fn gamma_cell_rhf_matches_pyscf() {
    let e = energy_of("examples/h2-cell-rhf.toml");
    assert!((e - (-1.658_517_536_0)).abs() < 1e-8, "E = {e}");
}

#[test]
fn kpoint_cell_rhf_matches_pyscf() {
    let e = energy_of("examples/h2-cell-kpts-rhf.toml");
    assert!((e - (-1.055_419_673_0)).abs() < 1e-8, "E = {e}");
}
