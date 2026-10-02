//! The CLI separates the correlation METHOD (`method.kind` = `rimp2`, `drpa`,
//! `linlccd`) from its LOCAL APPROXIMATION (`[local]`). These tests drive the
//! real binary end to end and pin what that separation promises:
//!
//! * no `[local]` is the EXACT method, and the printout and run log say so
//!   (`"<method> (exact)"`, `"local": null`);
//! * `[local] scheme = "amplitude-threshold"` states its threshold and kept
//!   fraction in the printout and logs a `local` object;
//! * exact `drpa` IS the ε = 0 Riccati solve, to <= 1e-12 Ha;
//! * the opt-in exact reference (`[local] reference`) is honest when off: no
//!   NaN in the printout, `null` in the log;
//! * the removed kinds/keys fail, and an exact `drpa` that cannot fit the
//!   memory budget is refused BEFORE the SCF with a pointer to `pdep-rpa`.
//!
//! Water/6-31G with cc-pVDZ-RI: seconds per run.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Path to the `ferric-cli` binary, resolved at RUN TIME -- see
/// `dispersion_guards.rs` for why `env!("CARGO_BIN_EXE_ferric-cli")` alone is
/// wrong under `cargo nextest archive`.
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

struct Run {
    ok: bool,
    stdout: String,
    stderr: String,
    results: Vec<serde_json::Value>,
    /// Whether the run log holds any SCF iteration record (`scf_iter` or
    /// `guess_scf_iter`): false means the SCF never started.
    scf_started: bool,
}

/// Run water/6-31G under `kind` with the given extra TOML sections, logging
/// JSON to a per-test file.
fn run(tag: &str, kind: &str, extra: &str) -> Run {
    let root = workspace_root();
    let toml_path = root.join("target").join(format!("local_cli_{tag}.toml"));
    let log_path = root.join("target").join(format!("local_cli_{tag}.jsonl"));
    let _ = std::fs::remove_file(&log_path);
    let body = format!(
        "[molecule]\nxyz = \"testdata/molecules/water.xyz\"\n\n\
         [basis]\nname = \"6-31g\"\n\n\
         [method]\nkind = \"{kind}\"\n\n\
         [scf]\nmax_iter = 200\ndensity_conv = 1e-10\ndf_guess = false\n\n\
         {extra}\n\
         [output]\njson = \"{}\"\n",
        log_path.display()
    );
    std::fs::write(&toml_path, body).expect("write temp toml");
    let out = Command::new(ferric_cli_bin())
        .arg(&toml_path)
        .current_dir(&root)
        .env("OPENBLAS_NUM_THREADS", "1")
        .env("RAYON_NUM_THREADS", "2")
        .output()
        .expect("failed to run ferric-cli binary");
    let records: Vec<serde_json::Value> = std::fs::read_to_string(&log_path)
        .map(|log| {
            log.lines()
                .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
                .collect()
        })
        .unwrap_or_default();
    let scf_started = records.iter().any(|v| {
        v["record"]
            .as_str()
            .is_some_and(|r| r.ends_with("scf_iter"))
    });
    let results = records
        .into_iter()
        .filter(|v| v["record"] == "result")
        .collect();
    Run {
        ok: out.status.success(),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
        results,
        scf_started,
    }
}

fn ok_run(tag: &str, kind: &str, extra: &str) -> Run {
    let r = run(tag, kind, extra);
    assert!(
        r.ok,
        "{kind} run failed.\nstdout:\n{}\nstderr:\n{}",
        r.stdout, r.stderr
    );
    assert!(!r.results.is_empty(), "no result record in the run log");
    for rec in &r.results {
        assert_eq!(rec["kind"], kind, "{rec}");
    }
    r
}

const MP2: &str = "[mp2]\nauxbasis = \"cc-pvdz-ri\"\nfrozen_core = 1\n";

fn local(body: &str) -> String {
    format!("{MP2}\n[local]\nscheme = \"amplitude-threshold\"\n{body}\n")
}

/// The exact `drpa` is the ε = 0 Riccati solve: the two runs agree to
/// <= 1e-12 Ha; the exact printout says "(exact)" and logs `local: null`,
/// the ε = 0 local one states its threshold and logs the object.
#[test]
fn exact_drpa_equals_local_eps_zero_and_says_exact() {
    let ex = ok_run("drpa_exact", "drpa", MP2);
    let lo = ok_run("drpa_eps0", "drpa", &local("eps = 0.0"));
    assert!(ex.stdout.contains("dRPA (exact)"), "{}", ex.stdout);
    assert!(!ex.stdout.contains("local:"), "{}", ex.stdout);
    assert!(ex.results[0]["components"]["local"].is_null());
    assert!(
        lo.stdout
            .contains("dRPA (local: amplitude threshold, eps = 0.0e0; kept 100.00% of amplitudes)"),
        "{}",
        lo.stdout
    );
    let l = &lo.results[0]["components"]["local"];
    assert_eq!(l["scheme"], "amplitude-threshold", "{l}");
    assert_eq!(l["eps"], 0.0, "{l}");
    assert_eq!(l["keep_fraction"], 1.0, "{l}");
    assert_eq!(l["integral_direct"], false, "{l}");
    let (a, b) = (
        ex.results[0]["total"].as_f64().unwrap(),
        lo.results[0]["total"].as_f64().unwrap(),
    );
    assert!((a - b).abs() <= 1e-12, "exact {a:.14} vs eps=0 {b:.14}");
}

/// A finite ε prints the threshold and the kept fraction, and moves the
/// energy (reachability: a run that ignored eps would print 100%).
#[test]
fn local_drpa_states_eps_and_keep_fraction() {
    let lo = ok_run("drpa_eps3", "drpa", &local("eps = 1e-3\nreference = true"));
    let line = lo
        .stdout
        .lines()
        .find(|l| l.starts_with("dRPA (local: amplitude threshold, eps = 1.0e-3; kept "))
        .unwrap_or_else(|| panic!("no model line:\n{}", lo.stdout));
    assert!(!line.contains("kept 100.00%"), "{line}");
    let c = &lo.results[0]["components"];
    let keep = c["local"]["keep_fraction"].as_f64().unwrap();
    assert!(0.0 < keep && keep < 1.0, "{c}");
    assert_eq!(c["local"]["eps"], 1e-3);
    // the opt-in reference: threshold error one-sided-ish, small
    let de = c["e_corr"].as_f64().unwrap() - c["e_corr_plasmon_canonical"].as_f64().unwrap();
    assert!(de.abs() < 5e-2 && de != 0.0, "de = {de:e}");
}

/// Exact `rimp2` / `linlccd` say so and log `local: null`; local runs log the
/// object; and with the reference OFF there is no NaN and the log carries
/// `null` (the printout names the opt-in key).
#[test]
fn rimp2_and_linlccd_state_their_model() {
    let ex = ok_run("rimp2_exact", "rimp2", MP2);
    assert!(ex.stdout.contains("RI-MP2 (exact)"), "{}", ex.stdout);
    assert!(ex.results[0]["components"]["local"].is_null());

    for (tag, direct) in [
        ("rimp2_local", ""),
        ("rimp2_direct", "integral_direct = true\n"),
    ] {
        let lo = ok_run(tag, "rimp2", &local(&format!("eps = 1e-3\n{direct}")));
        assert!(!lo.stdout.contains("NaN"), "{}", lo.stdout);
        assert!(
            lo.stdout.contains("E_corr(canonical RI)  = not computed")
                && lo.stdout.contains("[local] reference = true"),
            "{}",
            lo.stdout
        );
        assert!(
            lo.stdout.contains("MP2 (local: amplitude threshold"),
            "{}",
            lo.stdout
        );
        let c = &lo.results[0]["components"];
        assert!(c["e_corr_canonical_ri"].is_null(), "{c}");
        assert_eq!(c["local"]["eps"], 1e-3, "{c}");
        assert_eq!(c["local"]["integral_direct"], !direct.is_empty(), "{c}");
    }

    let lex = ok_run("linlccd_exact", "linlccd", MP2);
    assert!(lex.stdout.contains("LinLCCD(hh) (exact)"), "{}", lex.stdout);
    assert!(lex.results[0]["components"]["local"].is_null());
    let llo = ok_run(
        "linlccd_eps0",
        "linlccd",
        &local("eps = 0.0\nreference = true"),
    );
    let c = &llo.results[0]["components"];
    assert_eq!(c["local"]["eps"], 0.0, "{c}");
    // eps = 0 reproduces the exact LinLCCD of the same variant
    let (a, b) = (
        c["e_corr"].as_f64().unwrap(),
        lex.results[0]["components"]["e_corr"].as_f64().unwrap(),
    );
    assert!((a - b).abs() < 1e-8, "local eps=0 {a:.12} vs exact {b:.12}");
    assert!(
        (c["e_corr_exact"].as_f64().unwrap() - b).abs() < 1e-9,
        "{c}"
    );
}

/// Refused configurations fail through the binary, before any SCF output:
/// the removed kinds and keys, `[local]` without `eps`, and an exact `drpa`
/// whose Riccati solve cannot fit the budget (pointing at `pdep-rpa`).
#[test]
fn refusals_fail_before_the_scf() {
    let cases: &[(&str, &str, &str, &str)] = &[
        ("old_kind_lmp2", "lmp2", MP2, "unsupported method.kind"),
        (
            "old_kind_lmp2d",
            "lmp2-direct",
            MP2,
            "unsupported method.kind",
        ),
        (
            "old_kind_lla",
            "linlccd-amplitude",
            MP2,
            "unsupported method.kind",
        ),
        ("old_key", "rimp2", "[mp2]\nlmp2_eps = 1e-4\n", "lmp2_eps"),
        (
            "old_drpa_key",
            "drpa",
            "[mp2]\ndrpa_eps = 1e-4\n",
            "drpa_eps",
        ),
        (
            "no_eps",
            "drpa",
            "[local]\nscheme = \"amplitude-threshold\"\n",
            "requires eps",
        ),
        (
            "eps_none",
            "drpa",
            "[local]\neps = 1e-4\n",
            "scheme = \"none\"",
        ),
        (
            "direct_drpa",
            "drpa",
            "[local]\nscheme = \"amplitude-threshold\"\neps = 1e-4\nintegral_direct = true\n",
            "integral_direct",
        ),
        (
            "local_rhf",
            "rhf",
            "[local]\nscheme = \"amplitude-threshold\"\neps = 1e-4\n",
            "[local] applies to",
        ),
        (
            "exact_drpa_oom",
            "drpa",
            "[memory]\nbudget_gb = 1e-4\n",
            "pdep-rpa",
        ),
    ];
    for (tag, kind, extra, needle) in cases {
        let r = run(tag, kind, extra);
        assert!(!r.ok, "{tag}: must fail.\nstdout:\n{}", r.stdout);
        assert!(
            r.stderr.contains(needle),
            "{tag}: stderr lacks {needle:?}:\n{}",
            r.stderr
        );
        assert!(
            !r.scf_started && !r.stdout.contains("RHF energy") && r.results.is_empty(),
            "{tag}: refused only after the SCF started:\n{}",
            r.stdout
        );
    }
}
