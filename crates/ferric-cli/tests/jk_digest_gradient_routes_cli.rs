//! `[scf] jk_digest` on the two gradient routes that re-solve their
//! reference at each geometry from reduced settings: the RI-MP2 family
//! (`rimp2`, through `ScfJk::jk_digest`) and `pdep-rpa` (through
//! `RpaGradientRoute::jk_digest`), `task = "optimize"`. Both runs are
//! accepted (PR #717 refused the key here), and the key reaches every SCF
//! of the run: config `legacy` logs bitwise the SCF energies of an unset
//! config under `FERRIC_JK_DIGEST=legacy`, and config `pair` logs different
//! last bits that agree to roundoff.
//!
//! The four-centre reference of a small molecule normally runs on the
//! in-core ERI store, which no digest reaches, so every run gets a memory
//! budget (`FERRIC_MEM_BUDGET_GB`, about 107 kB) whose quarter the store of
//! water / 6-31G (33 kB) exceeds, and `eri = "auto"` builds J/K direct.
//! This couples the test to the auto rule's budget fraction
//! (`incore_jk::AUTO_BUDGET_DIVISOR`) and to that store size: should either
//! change so the store fits, the pair and legacy runs would become bitwise
//! one and the "config pair is bitwise legacy" assert would say so.
//! `[memory] budget_gb` gives the correlation part its own budget.
//!
//! `rimp2` names `jk = "exact"` so its gradient route is four-centre (an
//! omitted `jk` is RI-JK, on which the digest is inert); the last test
//! checks that `jk_digest` does not move an omitted `jk` off that RI-JK
//! default. Dropping the CLI's lowering on either route makes the
//! per-geometry SCFs of config `legacy` run `pair`, and the first assert of
//! `check` fails.

use std::path::{Path, PathBuf};
use std::process::Command;

#[path = "support/cli_bin.rs"]
mod cli_bin;
use cli_bin::ferric_cli_bin;

/// The workspace root at RUN time: the nearest ancestor of the current
/// directory that looks like it (a relocated nextest archive runs from an
/// extracted copy, where the compile-time `CARGO_MANIFEST_DIR` does not
/// exist), else the compile-time path (as `full_examples_anchor` does).
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
        .and_then(Path::parent)
        .expect("ferric-cli manifest dir should be workspace_root/crates/ferric-cli")
        .to_path_buf()
}

/// The input of `kind` with `[scf]` body `scf`.
fn input(kind: &str, scf: &str) -> String {
    let section = match kind {
        "rimp2" => "[mp2]\nauxbasis = \"cc-pvdz-ri\"\n",
        _ => "[rpa]\nauxbasis = \"cc-pvdz-ri\"\nn_quad = 4\n",
    };
    format!(
        "[molecule]\nxyz = \"testdata/molecules/water.xyz\"\n[basis]\nname = \"6-31g\"\n\
         [method]\nkind = \"{kind}\"\ntask = \"optimize\"\n[scf]\n{scf}{section}\
         [memory]\nbudget_gb = 2\n[optimize]\nmax_steps = 2\n"
    )
}

/// Run `body` with `FERRIC_JK_DIGEST = env` (unset when `None`) and return
/// the run log's records.
fn run(tag: &str, body: &str, env: Option<&str>) -> Vec<serde_json::Value> {
    let root = workspace_root();
    let dir = root.join("target");
    std::fs::create_dir_all(&dir).expect("create target/");
    let toml = dir.join(format!("jk_digest_route_{tag}.toml"));
    let log = dir.join(format!("jk_digest_route_{tag}.jsonl"));
    let _ = std::fs::remove_file(&log);
    std::fs::write(&toml, body).expect("write temp toml");
    let mut cmd = Command::new(ferric_cli_bin());
    cmd.arg(&toml)
        .arg("--json")
        .arg(&log)
        .current_dir(&root)
        .env("OPENBLAS_NUM_THREADS", "1")
        .env("RAYON_NUM_THREADS", "2")
        .env("FERRIC_MEM_BUDGET_GB", "0.0001")
        .env_remove("FERRIC_JK_DIGEST");
    if let Some(v) = env {
        cmd.env("FERRIC_JK_DIGEST", v);
    }
    let out = cmd.output().expect("failed to run ferric-cli");
    assert!(
        out.status.success(),
        "{tag} must run.\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    std::fs::read_to_string(&log)
        .expect("run log")
        .lines()
        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
        .collect()
}

/// The energy of every `scf_iter` record of `records`, in order.
fn scf_energies_of(tag: &str, records: &[serde_json::Value]) -> Vec<f64> {
    let energies: Vec<f64> = records
        .iter()
        .filter(|r| r["record"] == "scf_iter")
        .map(|r| r["energy"].as_f64().expect("scf_iter energy"))
        .collect();
    assert!(!energies.is_empty(), "{tag}: no scf_iter records");
    energies
}

/// [`run`], then [`scf_energies_of`].
fn scf_energies(tag: &str, body: &str, env: Option<&str>) -> Vec<f64> {
    scf_energies_of(tag, &run(tag, body, env))
}

fn check(kind: &str, exact: &str) {
    let legacy = scf_energies(
        &format!("{kind}_legacy"),
        &input(kind, &format!("{exact}jk_digest = \"legacy\"\n")),
        None,
    );
    let env_legacy = scf_energies(
        &format!("{kind}_env_legacy"),
        &input(kind, exact),
        Some("legacy"),
    );
    let pair = scf_energies(
        &format!("{kind}_pair"),
        &input(kind, &format!("{exact}jk_digest = \"pair\"\n")),
        None,
    );
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(
        bits(&legacy),
        bits(&env_legacy),
        "{kind}: config legacy is not the unset config under FERRIC_JK_DIGEST=legacy"
    );
    assert_eq!(
        pair.len(),
        legacy.len(),
        "{kind}: SCF iteration counts differ"
    );
    let differ = pair
        .iter()
        .zip(&legacy)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    eprintln!(
        "{kind}: {differ} of {} SCF energies differ under config pair",
        pair.len()
    );
    assert!(differ > 0, "{kind}: config pair is bitwise legacy");
    for (i, (a, b)) in pair.iter().zip(&legacy).enumerate() {
        assert!(a.is_finite(), "{kind}[{i}]: {a}");
        assert!(
            (a - b).abs() <= 1e-10 * b.abs(),
            "{kind}[{i}]: pair {a:e} vs legacy {b:e}"
        );
    }
}

#[test]
fn jk_digest_reaches_the_rimp2_optimize_scfs() {
    check("rimp2", "jk = \"exact\"\n");
}

#[test]
fn jk_digest_reaches_the_pdep_rpa_optimize_scfs() {
    check("pdep-rpa", "");
}

/// `jk_digest` does not change the J/K route: an `rimp2` optimize with `jk`
/// omitted logs the RI-JK default (`scf_jk = "rijk"` on the matched set)
/// with and without the key, and since every J/K build of that route is
/// fitted the digest is inert there: the SCF energies are bitwise one run.
#[test]
fn jk_digest_keeps_the_omitted_rijk_route() {
    let unset = run("rimp2_rijk_unset", &input("rimp2", ""), None);
    let set = run(
        "rimp2_rijk_pair",
        &input("rimp2", "jk_digest = \"pair\"\n"),
        None,
    );
    let route = |r: &[serde_json::Value]| {
        let start = r
            .iter()
            .find(|x| x["record"] == "run_start")
            .expect("run_start");
        (
            start["config"]["scf_jk"].clone(),
            start["config"]["jk_aux"].clone(),
        )
    };
    assert_eq!(
        route(&unset),
        ("rijk".into(), "def2-universal-jkfit".into())
    );
    assert_eq!(route(&set), route(&unset));
    let bits = |tag, r: &[serde_json::Value]| {
        scf_energies_of(tag, r)
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>()
    };
    assert_eq!(bits("set", &set), bits("unset", &unset));
}
