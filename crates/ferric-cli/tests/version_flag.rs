//! `ferric --version` reports the same build identity as `ferric.build_info()`:
//! one `key: value` line per field, every value fixed at compile time.

use std::process::Command;

#[path = "support/cli_bin.rs"]
mod cli_bin;
use cli_bin::ferric_cli_bin;

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

/// `--version` into a full device (`> /dev/full`, every write fails with
/// `ENOSPC`) is an error with status 1, not a panic (status 101).
#[cfg(target_os = "linux")]
#[test]
fn version_into_a_full_device_is_an_error_not_a_panic() {
    let full = std::fs::OpenOptions::new()
        .write(true)
        .open("/dev/full")
        .expect("open /dev/full");
    let out = Command::new(ferric_cli_bin())
        .arg("--version")
        .stdout(full)
        .output()
        .expect("ferric runs");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "{stderr}");
    assert!(
        stderr.starts_with("error: cannot write the --version text to stdout: "),
        "{stderr}"
    );
    assert!(!stderr.contains("panicked"), "{stderr}");
}

/// `--help` with stderr on a full device: status 1, and the error message,
/// which would go to the same stderr, is dropped instead of panicking.
#[cfg(target_os = "linux")]
#[test]
fn help_into_a_full_device_is_an_error_not_a_panic() {
    let full = std::fs::OpenOptions::new()
        .write(true)
        .open("/dev/full")
        .expect("open /dev/full");
    let out = Command::new(ferric_cli_bin())
        .arg("--help")
        .stderr(full)
        .output()
        .expect("ferric runs");
    assert_eq!(out.status.code(), Some(1));
    assert!(out.stdout.is_empty());
}

/// The reader of the pipe exits before ferric writes (`ferric --help 2>&1 |
/// true`, ferric delayed so the read end is closed first): the broken pipe
/// is success, status 0, with nothing on the original stderr. If the reader
/// were still there the write succeeds and the status is 0 as well, so
/// the delay only makes the test exercise the broken pipe, not pass.
#[cfg(unix)]
#[test]
fn a_broken_pipe_is_success() {
    for (flag, redirect) in [("--version", ""), ("--help", "2>&1"), ("-h", "2>&1")] {
        let script = format!(
            "{{ sleep 0.3; \"$0\" {flag} {redirect}; echo \"status=$?\" >&3; }} 3>&2 | true"
        );
        let out = Command::new("bash")
            .arg("-c")
            .arg(&script)
            .arg(ferric_cli_bin())
            .output()
            .expect("bash runs");
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(stderr, "status=0\n", "{flag}: {stderr}");
    }
}
