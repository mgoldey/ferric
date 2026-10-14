//! The CLI prints `warning: [scf] energy_conv ... below the estimated noise floor`
//! on stderr for a WRITTEN `[scf] energy_conv` the dE criterion cannot reach,
//! and nothing for a default or an ordinary value (issue #410). The decision
//! logic is unit-tested in `ferric-scf/src/noise_floor.rs` and
//! `ferric-cli/src/lib.rs`; this pins the end-to-end stderr contract.

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
        .expect("workspace root")
        .to_path_buf()
}

/// Run alkane C10 / STO-3G (72 functions, exact J/K; dE floor ~2e-10, so the
/// advisory threshold is ~6e-11 at the default `integral_thresh`) for ONE
/// iteration with the given extra `[scf]` lines and `FERRIC_SCF_INCREMENTAL`
/// (`None`: unset, the default gate); return stderr.
fn stderr_for(tag: &str, scf_lines: &str, incremental: Option<&str>) -> String {
    let dir = std::env::temp_dir().join(format!("ferric_ecnf_{}_{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let toml = dir.join("run.toml");
    let xyz = workspace_root().join("testdata/molecules/alkane_10.xyz");
    std::fs::write(
        &toml,
        format!(
            "[molecule]\nxyz = \"{}\"\n[basis]\nname = \"sto-3g\"\n[method]\nkind = \"rhf\"\n\
             [scf]\nmax_iter = 1\n{scf_lines}\n",
            xyz.display()
        ),
    )
    .unwrap();
    let mut cmd = Command::new(ferric_cli_bin());
    cmd.arg(&toml)
        .current_dir(workspace_root())
        .env("OPENBLAS_NUM_THREADS", "1")
        .env_remove("FERRIC_SCF_INCREMENTAL");
    if let Some(v) = incremental {
        cmd.env("FERRIC_SCF_INCREMENTAL", v);
    }
    let out = cmd.output().expect("failed to run ferric-cli");
    assert!(
        !String::from_utf8_lossy(&out.stdout).contains("warning: [scf]"),
        "the advisory must go to stderr, never stdout"
    );
    String::from_utf8_lossy(&out.stderr).into_owned()
}

/// The incremental Fock path is forced on: by default `energy_conv = 1e-11` is
/// below `1e4 * integral_thresh` and turns it off (issue #415), which removes
/// the screening noise this warns about (see the next test).
#[test]
fn a_written_tight_energy_conv_warns_on_stderr() {
    let e = stderr_for("tight", "energy_conv = 1e-11", Some("1"));
    assert!(
        e.contains("warning: [scf] energy_conv 1.0e-11 is below the estimated noise floor"),
        "{e}"
    );
    assert!(e.contains("72 basis functions"), "{e}");
}

/// The same setting with the default gate: the incremental Fock is off, only the
/// rounding floor (~4e-13 Ha here) is left, so 1e-11 is reachable and not advised on.
#[test]
fn a_tight_energy_conv_that_turns_the_incremental_fock_off_is_silent() {
    let e = stderr_for("tight_default", "energy_conv = 1e-11", None);
    assert!(!e.contains("noise floor"), "{e}");
}

#[test]
fn defaults_and_ordinary_values_are_silent() {
    for (tag, lines) in [
        ("default", ""),
        // The 1e-3 default would be below the floor once integral_thresh is this
        // loose, but it is not the user's choice: still silent.
        ("default_loose_thresh", "integral_thresh = 1e-4"),
        ("ordinary", "energy_conv = 1e-8"),
    ] {
        let e = stderr_for(tag, lines, None);
        assert!(!e.contains("noise floor"), "{tag}:\n{e}");
    }
}
