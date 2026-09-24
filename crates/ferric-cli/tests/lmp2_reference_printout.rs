//! `[mp2] lmp2_reference` is OPT-IN, and a run without it must say so
//! honestly: no "NaN" canonical energy, no NaN difference line, and a `null`
//! (not a NaN string, not a number) `e_corr_canonical_ri` in the JSON run log.
//!
//! The canonical reference is a full N^5 canonical RI-MP2 over the global
//! (naux, nocc·nvir) tensor -- exactly what `lmp2-direct` exists to avoid --
//! so it defaults off. Before that change the library computed it
//! unconditionally and the CLI printed it unconditionally; flipping only the
//! library default would have printed `E_corr(canonical RI) = NaN Ha` and a
//! `NaN` error line. These tests drive the real binary end to end.
//!
//! Water/6-31G with cc-pVDZ-RI: seconds per run.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Path to the `ferric-cli` binary, resolved at RUN TIME -- see
/// `dispersion_guards.rs` for why `env!("CARGO_BIN_EXE_ferric-cli")` alone is
/// wrong under `cargo nextest archive`.
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

/// Workspace root, resolved at RUN TIME -- see `ccsd_dispatch.rs`.
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

/// Run `kind` on water/6-31G, with `extra` appended to `[mp2]`, logging JSON
/// to a per-test file. Returns (stdout, the `result` record's components).
fn run(tag: &str, kind: &str, extra: &str) -> (String, serde_json::Value) {
    let root = workspace_root();
    let toml_path = root.join("target").join(format!("lmp2_ref_{tag}.toml"));
    let log_path = root.join("target").join(format!("lmp2_ref_{tag}.jsonl"));
    let _ = std::fs::remove_file(&log_path);
    let body = format!(
        "[molecule]\nxyz = \"testdata/molecules/water.xyz\"\n\n\
         [basis]\nname = \"6-31g\"\n\n\
         [method]\nkind = \"{kind}\"\n\n\
         [mp2]\nauxbasis = \"cc-pvdz-ri\"\nfrozen_core = 1\nlmp2_eps = 1e-3\n{extra}\n\
         [output]\njson = \"{}\"\n",
        log_path.display()
    );
    std::fs::write(&toml_path, body).expect("write temp toml");
    let out = Command::new(ferric_cli_bin())
        .arg(&toml_path)
        .current_dir(&root)
        .env("OPENBLAS_NUM_THREADS", "1")
        .output()
        .expect("failed to run ferric-cli binary");
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    assert!(
        out.status.success(),
        "{kind} run failed.\nstdout:\n{stdout}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let log = std::fs::read_to_string(&log_path).expect("read JSON run log");
    let result = log
        .lines()
        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
        .find(|v| v["record"] == "result")
        .unwrap_or_else(|| panic!("no result record in the run log:\n{log}"));
    assert_eq!(result["kind"], kind, "{result}");
    (stdout, result["components"].clone())
}

/// Default (`lmp2_reference` absent), both kinds: the printout says the
/// reference was not computed and contains no NaN anywhere; the run log's
/// `e_corr_canonical_ri` is `null` while `e_corr` is a real number.
///
/// Fails if reverted: with the old unconditional `println!` of
/// `r.e_corr_canonical_ri` (NaN once the library default is off) stdout
/// contains "NaN" twice and no "not computed"; with the old
/// `json!({"e_corr_canonical_ri": r.e_corr_canonical_ri})` the field is still
/// null, so the log half alone would NOT catch that revert -- the stdout
/// asserts are the load-bearing ones. If the CLI default
/// (`Mp2Cfg::lmp2_reference`) were reverted to on, the printout shows a
/// number instead of "not computed" and the log field is a number.
#[test]
fn default_run_prints_no_nan_and_logs_null_reference() {
    for kind in ["lmp2", "lmp2-direct"] {
        let (stdout, comps) = run(&format!("off_{kind}"), kind, "");
        assert!(
            !stdout.contains("NaN"),
            "{kind}: NaN leaked into the printout:\n{stdout}"
        );
        assert!(
            stdout.contains("E_corr(canonical RI)  = not computed"),
            "{kind}: printout must state the reference was not computed:\n{stdout}"
        );
        assert!(
            stdout.contains("lmp2_reference = true"),
            "{kind}: printout must name the opt-in key:\n{stdout}"
        );
        assert!(
            !stdout.contains("threshold error") && !stdout.contains("total error"),
            "{kind}: no error-vs-reference line without a reference:\n{stdout}"
        );
        assert!(
            comps["e_corr_canonical_ri"].is_null(),
            "{kind}: run log must carry null for the uncomputed reference: {comps}"
        );
        assert!(comps["e_corr"].is_f64(), "{kind}: {comps}");
    }
}

/// Opt-in: `lmp2_reference = true` restores the value, the signed error line
/// and a numeric run-log field.
///
/// Fails if reverted: without the key wired into `AmplitudeLmp2Config`
/// (`compute_reference: want_ref`) the library default (off) returns a NaN
/// reference that the printout then shows as "NaN" (and the log as the
/// string "NaN"/null, not a number); without the key parsed at all the TOML
/// is rejected by deny_unknown_fields and the run fails.
#[test]
fn opt_in_run_prints_the_reference_and_its_error() {
    let (stdout, comps) = run("on_lmp2", "lmp2", "lmp2_reference = true\n");
    assert!(!stdout.contains("NaN"), "{stdout}");
    assert!(!stdout.contains("not computed"), "{stdout}");
    assert!(stdout.contains("threshold error       = "), "{stdout}");
    let e_ref = comps["e_corr_canonical_ri"]
        .as_f64()
        .unwrap_or_else(|| panic!("reference missing from the run log: {comps}"));
    let e_corr = comps["e_corr"].as_f64().unwrap();
    assert!(e_ref < 0.0, "{comps}");
    // eps = 1e-3 truncation is one-sided (under-correlation) and small
    let de = e_corr - e_ref;
    assert!((0.0..5e-2).contains(&de), "de = {de:+.3e}");
}
