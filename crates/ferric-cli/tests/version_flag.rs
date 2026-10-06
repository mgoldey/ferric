//! `ferric --version` reports the same build identity as `ferric.build_info()`:
//! one `key: value` line per field, every value fixed at compile time.

use std::path::{Path, PathBuf};
use std::process::Command;

/// The `ferric-cli` binary, resolved at RUN TIME.
///
/// `env!("CARGO_BIN_EXE_ferric-cli")` is baked in when the test binary is
/// COMPILED. CI builds tests in one job and runs them from a `cargo nextest
/// archive` in another, so that path points at the build job's `target/debug/`
/// -- which does not exist on the shard runner.
///
/// Prefer a sibling of the currently-running test binary
/// (`<extract-dir>/target/debug/deps/<test>` -> `../ferric-cli`), which is
/// where nextest puts it, then fall back to the compile-time path for plain
/// `cargo test`.
fn ferric_cli_bin() -> PathBuf {
    if let Ok(exe) = std::env::current_exe() {
        // .../target/<profile>/deps/<test-binary>  ->  .../target/<profile>/
        if let Some(profile_dir) = exe.parent().and_then(Path::parent) {
            let p = profile_dir.join("ferric-cli");
            if p.is_file() {
                return p;
            }
        }
    }
    PathBuf::from(env!("CARGO_BIN_EXE_ferric-cli"))
}

fn version_output(flag: &str) -> String {
    let out = Command::new(ferric_cli_bin())
        .arg(flag)
        .output()
        .expect("ferric runs");
    assert!(out.status.success(), "{flag} must exit 0");
    String::from_utf8(out.stdout).unwrap()
}

#[test]
fn version_prints_every_identity_field_and_matches_the_embedded_values() {
    let text = version_output("--version");
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 5, "{text}");
    assert_eq!(lines[0], format!("ferric {}", ferric_build_info::VERSION));
    assert_eq!(lines[1], format!("commit: {}", ferric_build_info::COMMIT));
    assert_eq!(
        lines[2],
        format!("dirty: {}", ferric_build_info::dirty_str())
    );
    assert_eq!(lines[3], format!("profile: {}", ferric_build_info::PROFILE));
    assert!(lines[4].starts_with("libint: "), "{text}");
    assert!(
        lines[4].len() > "libint: ".len(),
        "libint must never be empty"
    );
    assert!(
        !ferric_build_info::VERSION.contains('+'),
        "no PyPI-rejected local version"
    );
}

#[test]
fn short_flag_is_the_same_as_long() {
    assert_eq!(version_output("-V"), version_output("--version"));
}
