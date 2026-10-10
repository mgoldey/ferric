//! `[cell] sr_column_rotation` at a k-point mesh is HONOURED by the CLI: the
//! plan's resolved value reaches both the k-point hcore and the k-point
//! RS-GDF fit (a key that only changed the header would leave every run on
//! the library default).
//!
//! H2 / cc-pVDZ (one rotated s column per H) on a 1x1x2 Gamma-centred mesh
//! with `jk = "rsgdf"`:
//!
//! * absent (default) and `true`: the stage-table counter `k rsgdf SR3
//!   rotated columns` is 2 and the SR 3-centre triplet count is the rotated
//!   walk's;
//! * `false`: the counter is 0 and the triplet count is the unrotated
//!   walk's, strictly larger (negative control: the key CHANGES the run);
//! * the three energies agree to the printed precision (the rotation is
//!   exact to the screening precision).

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

/// Runs the input and returns stdout.
fn run(tag: &str, key: &str) -> String {
    let dir = std::env::temp_dir().join(format!("ferric-krot-{}-{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let toml = dir.join("in.toml");
    let xyz = workspace_root().join("testdata/molecules/h2.xyz");
    std::fs::write(
        &toml,
        format!(
            r#"[molecule]
xyz = "{xyz}"
[basis]
name = "cc-pvdz"
[method]
kind = "rhf"
[output]
json = "{json}"
[cell]
unit = "bohr"
lattice = [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]
kmesh = [1, 1, 2]
exxdiv = "ewald"
jk = "rsgdf"
auxbasis = "cc-pvdz-ri"
{key}
"#,
            xyz = xyz.display(),
            json = dir.join("run.jsonl").display(),
        ),
    )
    .unwrap();
    let out = Command::new(ferric_cli_bin())
        .arg(&toml)
        .current_dir(workspace_root())
        .env("OPENBLAS_NUM_THREADS", "1")
        .output()
        .expect("run ferric-cli");
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    assert!(
        out.status.success(),
        "{tag}: ferric-cli failed\n{stdout}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    stdout
}

/// The integer after `name` on its counter line.
fn counter(stdout: &str, name: &str) -> usize {
    stdout
        .lines()
        .find(|l| l.trim_start().starts_with(name))
        .unwrap_or_else(|| panic!("no counter {name:?} in:\n{stdout}"))
        .split_whitespace()
        .last()
        .and_then(|t| t.parse().ok())
        .unwrap_or_else(|| panic!("counter {name:?} is not an integer"))
}

fn energy(stdout: &str) -> f64 {
    let line = stdout
        .lines()
        .find(|l| l.trim_start().starts_with("energy "))
        .unwrap_or_else(|| panic!("no energy line in:\n{stdout}"));
    line.split('=')
        .nth(1)
        .and_then(|r| r.split_whitespace().next())
        .and_then(|t| t.parse().ok())
        .unwrap_or_else(|| panic!("cannot parse {line:?}"))
}

#[test]
fn kpoint_sr_column_rotation_key_reaches_the_builds() {
    const ROT: &str = "k rsgdf SR3 rotated columns";
    const TRIP: &str = "k rsgdf SR3 triplets";
    let default = run("default", "");
    let on = run("on", "sr_column_rotation = true");
    let off = run("off", "sr_column_rotation = false");
    assert!(default.contains("sr_column_rotation = on (default)"));
    assert!(off.contains("sr_column_rotation = off"));
    assert_eq!(counter(&default, ROT), 2);
    assert_eq!(counter(&on, ROT), 2);
    assert_eq!(counter(&off, ROT), 0, "false must reach the k-point fit");
    assert_eq!(counter(&default, TRIP), counter(&on, TRIP));
    assert!(
        counter(&off, TRIP) > counter(&on, TRIP),
        "unrotated walk {} vs rotated {}",
        counter(&off, TRIP),
        counter(&on, TRIP)
    );
    let (ed, e_on, e_off) = (energy(&default), energy(&on), energy(&off));
    assert!((ed - e_on).abs() < 1e-9 && (e_on - e_off).abs() < 1e-8);
}
