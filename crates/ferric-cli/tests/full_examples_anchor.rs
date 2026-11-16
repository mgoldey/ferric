//! Exactness anchor of `examples/full/`: each fully specified file and the
//! minimal input in its header describe the same run.
//!
//! Each file writes every key at the value an omitted key resolves to, so the
//! CLI must take the same path on both inputs. The test runs both through the
//! real `ferric-cli` binary, single-threaded (`RAYON_NUM_THREADS`,
//! `OPENBLAS_NUM_THREADS` and `FERRIC_BLAS_THREADS` all 1, so a run is bit
//! for bit reproducible), and requires the two JSON run logs to be identical
//! after removing the fields that are not results: wall-clock times, the
//! timestamp, memory high-water marks, and the `run_start` fields that echo a
//! key *as written* (an omitted `ri_j_precision` is logged as `null` with
//! source `default`, the written one as `"f32"` with source `explicit`; the
//! resolved precision is compared), and the labels that say why a value was
//! chosen (`precision_source`, "requested f32: default" against "explicit",
//! "[default]" against "[explicit grid keys]" in the certificate). The
//! free-atom guess SCFs interleave their records, so the logs are compared as
//! multisets of records with `seq` removed; every record still carries its
//! own iteration number. Every energy, iteration, route and certificate
//! entry must then agree exactly: no tolerance is applied.
//!
//! What it cannot see: a default that only matters on a branch these small
//! runs never take (a later convergence-ladder rung, a memory tier the budget
//! never reaches). The negative control at the end shows the comparison does
//! see a one-key change.
//!
//! `mp2-v` and `scs-mp2-2terfc` default to the terfc operator and need
//! `FERRIC_TERF_TABLE_DIR`; without it they are skipped with a message (same
//! policy as `cli_wired_keys.rs`). Set `FERRIC_FULL_EXAMPLES=rhf,ccsd` to run
//! a subset.

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

const MIN_BEGIN: &str = "# --- minimal input (begin) ---";
const MIN_END: &str = "# --- minimal input (end) ---";

/// The minimal input embedded in a full file's header.
fn minimal_of(full: &str) -> String {
    let start = full
        .find(MIN_BEGIN)
        .expect("header has the minimal-input marker")
        + MIN_BEGIN.len();
    let end = full.find(MIN_END).expect("header has the end marker");
    full[start..end]
        .lines()
        .filter_map(|l| l.strip_prefix("#   "))
        .map(|l| format!("{l}\n"))
        .collect()
}

const TERF_KINDS: &[&str] = &["mp2-v", "scs-mp2-2terfc"];

fn terf_tables() -> bool {
    std::env::var("FERRIC_TERF_TABLE_DIR")
        .map(|d| !d.is_empty() && Path::new(&d).join("16_4_2.bin").exists())
        .unwrap_or(false)
}

/// Run `toml` (written as `dir/<stem>.toml`) and return its run log, the
/// `[output] json = true` default path beside the input.
fn run(dir: &Path, stem: &str, toml: &str) -> Result<Vec<serde_json::Value>, String> {
    let root = workspace_root();
    let path = dir.join(format!("{stem}.toml"));
    let log = dir.join(format!("{stem}.ferric.jsonl"));
    std::fs::write(&path, toml).unwrap();
    let _ = std::fs::remove_file(&log);
    let out = Command::new(ferric_cli_bin())
        .arg(&path)
        .current_dir(&root)
        .env("OPENBLAS_NUM_THREADS", "1")
        .env("RAYON_NUM_THREADS", "1")
        .env("FERRIC_BLAS_THREADS", "1")
        .output()
        .expect("failed to run ferric-cli");
    if !out.status.success() {
        return Err(format!(
            "{} failed:\nstdout:\n{}\nstderr:\n{}",
            path.display(),
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    let text = std::fs::read_to_string(&log)
        .map_err(|e| format!("{}: no run log at {}: {e}", path.display(), log.display()))?;
    Ok(text
        .lines()
        .map(|l| serde_json::from_str(l).expect("run-log line parses"))
        .map(normalize)
        .collect())
}

/// Fields that are not results of the calculation.
const VOLATILE: &[&str] = &[
    "t",
    "wall_s",
    "cpu_s",
    "peak_rss_bytes",
    "timestamp",
    "timings",
    "path",
    // The auto budget is 0.8 x the RAM available at that moment.
    "memory_budget_bytes",
    "build_seconds",
    // Why a value was chosen ("default", "explicit", "geometry-task
    // default"); the value itself is compared.
    "precision_source",
    // The free-atom guess SCFs of different elements interleave their
    // records, so record order (and with it `seq`) is not reproducible; the
    // logs are compared as multisets of records instead.
    "seq",
];

/// Provenance labels inside certificate strings: the same value, chosen by
/// default or written explicitly.
fn strip_labels(s: &str) -> String {
    let mut out = s.replace(" [explicit grid keys]", " [default]");
    for prec in ["f32", "f64"] {
        for src in [
            "default",
            "explicit",
            "geometry-task default",
            "gpu default",
        ] {
            out = out.replace(&format!(" (requested {prec}: {src})"), "");
        }
    }
    out
}

/// `run_start.config` fields that echo a key as written (`null` when it was
/// omitted) rather than the value the run used. The RI precisions are
/// resolved in `ri_j_precision_resolved` and the `ri_j_tensor` record; the
/// CAS and OP-MP3 runs log their `[scf]` keys as written.
const AS_WRITTEN: &[&str] = &[
    "ri_j_precision",
    "ri_j_precision_source",
    "ri_k_precision",
    "max_iter",
    "energy_conv",
    "density_conv",
    // `functional = "LDA"` as written; the resolved functional is `xc`.
    "functional",
];

fn strip(v: &mut serde_json::Value) {
    match v {
        serde_json::Value::Object(m) => {
            for k in VOLATILE {
                m.remove(*k);
            }
            for x in m.values_mut() {
                strip(x);
            }
        }
        serde_json::Value::Array(a) => a.iter_mut().for_each(strip),
        serde_json::Value::String(st) => *st = strip_labels(st),
        _ => {}
    }
}

fn normalize(mut v: serde_json::Value) -> serde_json::Value {
    strip(&mut v);
    if v["record"] == "run_start" {
        if let Some(cfg) = v.get_mut("config").and_then(|c| c.as_object_mut()) {
            for k in AS_WRITTEN {
                cfg.remove(*k);
            }
        }
        // The binary's own path and build are the same for both runs.
        if let Some(m) = v.as_object_mut() {
            m.remove("git_sha");
        }
    }
    v
}

/// The records of one log that the other lacks (logs compared as multisets;
/// see `seq` in [`VOLATILE`]), at most three each way, as a message.
fn first_difference(a: &[serde_json::Value], b: &[serde_json::Value]) -> Option<String> {
    let key = |v: &serde_json::Value| v.to_string();
    let mut sa: Vec<String> = a.iter().map(key).collect();
    let mut sb: Vec<String> = b.iter().map(key).collect();
    sa.sort();
    sb.sort();
    let only = |x: &[String], y: &[String]| -> Vec<String> {
        let mut y = y.to_vec();
        x.iter()
            .filter(|r| match y.iter().position(|q| q == *r) {
                Some(i) => {
                    y.remove(i);
                    false
                }
                None => true,
            })
            .take(3)
            .cloned()
            .collect()
    };
    let (ma, mb) = (only(&sa, &sb), only(&sb, &sa));
    if ma.is_empty() && mb.is_empty() {
        return None;
    }
    Some(format!(
        "{} vs {} records; only in the minimal run: {ma:?}; only in the full run: {mb:?}",
        a.len(),
        b.len()
    ))
}

fn scratch() -> PathBuf {
    let dir = workspace_root().join("target/full_examples_anchor");
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn full_examples_match_their_minimal_inputs_in_the_trivial_limit() {
    let root = workspace_root();
    let only: Option<Vec<String>> = std::env::var("FERRIC_FULL_EXAMPLES")
        .ok()
        .map(|s| s.split(',').map(str::to_string).collect());
    let mut files: Vec<PathBuf> = std::fs::read_dir(root.join("examples/full"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "toml"))
        .collect();
    files.sort();
    assert!(!files.is_empty(), "no files in examples/full/");
    let dir = scratch();
    let mut failures = Vec::new();
    let mut compared = 0;
    for path in &files {
        let stem = path.file_stem().unwrap().to_string_lossy().into_owned();
        if only.as_ref().is_some_and(|o| !o.contains(&stem)) {
            continue;
        }
        let full = std::fs::read_to_string(path).unwrap();
        let kind = stem
            .split("-optimize")
            .next()
            .unwrap()
            .split("-frequencies")
            .next()
            .unwrap();
        if TERF_KINDS.contains(&kind) && !terf_tables() {
            eprintln!("skip {stem}: needs FERRIC_TERF_TABLE_DIR (terfc operator)");
            continue;
        }
        let tag = stem.replace(['(', ')'], "_");
        let a = run(&dir, &format!("{tag}.min"), &minimal_of(&full));
        let b = run(&dir, &format!("{tag}.full"), &full);
        match (a, b) {
            (Ok(a), Ok(b)) => {
                if let Some(d) = first_difference(&a, &b) {
                    failures.push(format!("{stem}: {d}"));
                }
                compared += 1;
            }
            (Err(e), _) | (_, Err(e)) => failures.push(format!("{stem}: {e}")),
        }
    }
    eprintln!("compared {compared} full examples against their minimal inputs");
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
}

/// Negative control: the comparison sees a one-key change. The rhf file with
/// `density_conv` moved from 1e-6 to 1e-7 must give a different run log.
#[test]
fn a_changed_default_changes_the_run_log() {
    let root = workspace_root();
    let full = std::fs::read_to_string(root.join("examples/full/rhf.toml")).unwrap();
    let changed = full.replacen("density_conv = 1e-6", "density_conv = 1e-7", 1);
    assert_ne!(
        full, changed,
        "rhf.toml no longer writes density_conv = 1e-6"
    );
    let dir = scratch();
    let a = run(&dir, "control.full", &full).unwrap();
    let b = run(&dir, "control.changed", &changed).unwrap();
    assert!(
        first_difference(&a, &b).is_some(),
        "the run logs of density_conv = 1e-6 and 1e-7 are identical after normalization; \
         the anchor comparison would not see a changed default"
    );
}
