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

/// The `libint:` value rebuilt independently from the integrals crate's API:
/// `<version> (eri a,b,c; eri3 ...; eri2 ...; onebody ...; g12 n|none)`, an
/// ungenerated class printing `none`.
fn expected_libint_value() -> String {
    let am = ferric_integrals::libint_max_am();
    let join = |k: &str| {
        if am[k].is_empty() {
            return "none".to_string();
        }
        am[k]
            .iter()
            .map(|l| l.to_string())
            .collect::<Vec<_>>()
            .join(",")
    };
    let g12 = ferric_integrals::libint_g12_max_am()
        .map(|l| l.to_string())
        .unwrap_or_else(|| "none".into());
    format!(
        "{} (eri {}; eri3 {}; eri2 {}; onebody {}; g12 {g12})",
        ferric_integrals::libint_version(),
        join("eri"),
        join("eri3"),
        join("eri2"),
        join("onebody"),
    )
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
    assert_eq!(
        lines[4],
        format!("libint: {}", expected_libint_value()),
        "{text}"
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
