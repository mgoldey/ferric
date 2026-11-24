//! Thread-count determinism of the REAL default path, one process per run.
//!
//! The in-process determinism tests (`ferric-mp2/tests/
//! mo_stream_dressing_threads.rs`, `laplace_energy_is_thread_independent.rs`,
//! ...) run their drivers inside `pool.install(..)`. Inside `install` the
//! caller is a rayon worker, and `opt_in_blas_threads()` returns 1 on a worker,
//! so those tests only ever see the `blas_threads = 1` case. The default is
//! different: off a worker the serial sections (SCF Fock diagonalization and
//! DIIS, the RI metric Cholesky `cholesky_inverse_sqrt`) run OpenBLAS at the
//! rayon POOL SIZE, and OpenBLAS splits those reductions by thread count, so
//! the last bits move with `RAYON_NUM_THREADS` (book-review-ch11-12 E2,
//! measured on water/cc-pVDZ).
//!
//! The rayon global pool and the OpenBLAS baseline are process-wide, so each
//! run here is its own `ferric-cli` process at `RAYON_NUM_THREADS` = 1 or 3.
//! `FERRIC_BLAS_THREADS` is REMOVED from the child env: `.cargo/config.toml`
//! sets it to 1 for every cargo-launched process, which would silently pin
//! the "default" runs to 1 and make case (b) a copy of case (a).
//!
//! Contracts, for RHF + RI-MP2 and RHF + Laplace SOS-MP2:
//!   (a) `[parallel] blas_threads = 1`: every reported energy is
//!       BIT-IDENTICAL at 1 and 3 rayon threads;
//!   (b) defaults: they agree within `default_drift_bound`, the documented
//!       last-bit drift (about 1e-14 relative, floored at 1e-13 Ha).
//! Reachability: each run's `[ferric] threads:` line is checked, so the
//! default runs at 3 threads really run serial BLAS at 3 (else (b) would be
//! (a) again) and the `blas_threads = 1` runs really run at 1.
//!
//! Mutation (recorded 2026-11-19): `with_blas_threads` ignoring its argument
//! and running the rayon pool size off a worker made (a) fail with an
//! assertion on the bit comparison (every energy of both kinds, e.g. RI-MP2
//! e_corr ...192c vs ...18b2). Unmutated, (a) is bit-identical and (b) drifts
//! 1.85e-13 Ha in E_SCF, so the default runs do take a different path.

use std::path::{Path, PathBuf};
use std::process::Command;

#[path = "support/cli_bin.rs"]
mod cli_bin;
use cli_bin::ferric_cli_bin;

/// The documented default-settings drift across thread counts: "about 1e-14
/// relative" (`[parallel] blas_threads` doc, config.rs), floored at the
/// "~1e-13 Ha" of CLAUDE.md "threading" for the small correlation energies.
/// MEASURED 2026-11-19 here (rayon 1 vs 3): E_SCF and E_total move 1.85e-13
/// Ha (13 ulps at 76 Ha, 2.4e-15 relative), e_corr 3-4e-15 Ha. A flat 1e-13
/// Ha bound fails on the 76 Ha totals, so the bound is relative.
fn default_drift_bound(e: f64) -> f64 {
    (1e-14 * e.abs()).max(1e-13)
}

/// The JSON-log energy fields compared, per kind. Every one present in the
/// log must match; a kind whose log carries none of them fails loudly.
/// Read from the `result` and `run_end` records only.
const ENERGY_KEYS: &[&str] = &[
    "energy",
    "total",
    "e_scf_reference",
    "e_corr",
    "e_os",
    "e_ss",
];

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

/// One run's identity: kind, rayon threads, explicit serial BLAS count.
#[derive(Clone, Copy, Debug)]
struct Run {
    kind: &'static str,
    rayon: usize,
    blas_one: bool,
}

fn toml_for(kind: &str, blas_one: bool, xyz: &Path) -> String {
    let parallel = if blas_one {
        "\n[parallel]\nblas_threads = 1\n"
    } else {
        ""
    };
    let mp2_extra = if kind == "laplace-sos-mp2" {
        "c_os = 1.3\nn_quad = 7\n"
    } else {
        ""
    };
    format!(
        "[molecule]\nxyz = \"{}\"\n\n[basis]\nname = \"cc-pvdz\"\n\n[method]\nkind = \"{kind}\"\n\n\
         [mp2]\nauxbasis = \"cc-pvdz-ri\"\n{mp2_extra}\n[memory]\nbudget_gb = 1.0\n{parallel}",
        xyz.display()
    )
}

/// Spawn one `ferric-cli` process for `run`; return the energy fields of its
/// JSON log (name -> f64 bits) and its `[ferric] threads:` line.
fn spawn(run: Run, dir: &Path) -> std::process::Child {
    let tag = format!(
        "{}-r{}-{}",
        run.kind,
        run.rayon,
        if run.blas_one { "b1" } else { "bdef" }
    );
    let toml = dir.join(format!("{tag}.toml"));
    let xyz = workspace_root().join("testdata/molecules/water.xyz");
    std::fs::write(&toml, toml_for(run.kind, run.blas_one, &xyz)).unwrap();
    Command::new(ferric_cli_bin())
        .arg("--json")
        .arg(dir.join(format!("{tag}.jsonl")))
        .arg(&toml)
        .current_dir(workspace_root())
        .env("RAYON_NUM_THREADS", run.rayon.to_string())
        .env("OPENBLAS_NUM_THREADS", "1")
        .env_remove("FERRIC_BLAS_THREADS")
        .env_remove("FERRIC_LANCZOS_BLAS_THREADS")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("failed to spawn ferric-cli")
}

struct Outcome {
    energies: Vec<(String, f64)>,
    serial_blas: usize,
}

fn collect(run: Run, child: std::process::Child, dir: &Path) -> Outcome {
    let out = child.wait_with_output().unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "{run:?} failed; stderr:\n{stderr}\nstdout:\n{}",
        String::from_utf8_lossy(&out.stdout)
    );
    let line = stderr
        .lines()
        .find(|l| l.starts_with("[ferric] threads:"))
        .unwrap_or_else(|| panic!("{run:?}: no threads line in stderr:\n{stderr}"));
    // "... in parallel regions, N in serial sections"
    let serial_blas: usize = line
        .rsplit(", ")
        .next()
        .and_then(|s| s.split_whitespace().next())
        .and_then(|n| n.parse().ok())
        .unwrap_or_else(|| panic!("{run:?}: unparsable threads line: {line}"));
    let tag = format!(
        "{}-r{}-{}",
        run.kind,
        run.rayon,
        if run.blas_one { "b1" } else { "bdef" }
    );
    let log = std::fs::read_to_string(dir.join(format!("{tag}.jsonl"))).unwrap();
    let mut energies = Vec::new();
    for rec in log.lines().filter(|l| !l.trim().is_empty()) {
        let v: serde_json::Value = serde_json::from_str(rec).unwrap();
        let record = v["record"].as_str().unwrap_or("");
        if record == "result" || record == "run_end" {
            collect_keys(&v, record, &mut energies);
        }
    }
    assert!(
        !energies.is_empty(),
        "{run:?}: no energy fields ({ENERGY_KEYS:?}) in the JSON log:\n{log}"
    );
    Outcome {
        energies,
        serial_blas,
    }
}

/// Every numeric field named in `ENERGY_KEYS`, anywhere in the record, keyed
/// by its JSON path so two runs compare field for field.
fn collect_keys(v: &serde_json::Value, path: &str, out: &mut Vec<(String, f64)>) {
    match v {
        serde_json::Value::Object(m) => {
            for (k, x) in m {
                let p = format!("{path}/{k}");
                match x.as_f64() {
                    Some(f) if ENERGY_KEYS.contains(&k.as_str()) => out.push((p, f)),
                    _ => collect_keys(x, &p, out),
                }
            }
        }
        serde_json::Value::Array(a) => {
            for (i, x) in a.iter().enumerate() {
                collect_keys(x, &format!("{path}[{i}]"), out);
            }
        }
        _ => {}
    }
}

#[test]
fn energies_across_rayon_thread_counts_in_separate_processes() {
    let dir = std::env::temp_dir().join(format!("ferric-thread-det-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    let mut runs = Vec::new();
    for kind in ["rimp2", "laplace-sos-mp2"] {
        for blas_one in [true, false] {
            for rayon in [1, 3] {
                runs.push(Run {
                    kind,
                    rayon,
                    blas_one,
                });
            }
        }
    }
    // Four processes at a time (1+3 rayon threads each pair): the runs are
    // independent and seconds long, and concurrency cannot change a process's
    // bits (each sets its own pool and BLAS count).
    let mut outcomes = Vec::new();
    for chunk in runs.chunks(4) {
        let children: Vec<_> = chunk.iter().map(|&r| (r, spawn(r, &dir))).collect();
        for (r, c) in children {
            outcomes.push((r, collect(r, c, &dir)));
        }
    }
    let _ = std::fs::remove_dir_all(&dir);

    let mut failures = Vec::new();
    for pair in outcomes.chunks(2) {
        let [(r1, o1), (r3, o3)] = pair else {
            unreachable!()
        };
        assert_eq!((r1.rayon, r3.rayon), (1, 3));
        // Reachability: the serial BLAS count is what each case claims.
        let expect3 = if r3.blas_one { 1 } else { 3 };
        assert_eq!(o1.serial_blas, 1, "{r1:?}: serial BLAS threads");
        assert_eq!(
            o3.serial_blas, expect3,
            "{r3:?}: serial BLAS threads (default must be the rayon pool size)"
        );
        let names1: Vec<_> = o1.energies.iter().map(|e| &e.0).collect();
        let names3: Vec<_> = o3.energies.iter().map(|e| &e.0).collect();
        assert_eq!(names1, names3, "{}: different energy fields", r1.kind);
        for ((name, e1), (_, e3)) in o1.energies.iter().zip(&o3.energies) {
            let ok = if r1.blas_one {
                e1.to_bits() == e3.to_bits()
            } else {
                (e1 - e3).abs() <= default_drift_bound(*e1)
            };
            eprintln!(
                "{} blas_threads={} {name}: {e1:?} vs {e3:?} (diff {:e})",
                r1.kind,
                if r1.blas_one { "1" } else { "default" },
                e1 - e3
            );
            if !ok {
                failures.push(format!(
                    "{} {} {name}: rayon 1 -> {e1:?} ({:#018x}), rayon 3 -> {e3:?} ({:#018x})",
                    r1.kind,
                    if r1.blas_one {
                        "blas_threads=1 must be bit-identical"
                    } else {
                        "default must agree within max(1e-13 Ha, 1e-14 |E|)"
                    },
                    e1.to_bits(),
                    e3.to_bits()
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
