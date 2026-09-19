//! Writer-level guarantees of [`ferric_scf::runlog`], each written so that a
//! specific plausible BREAKAGE fails it.
//!
//! Per this repo's "a test you have never seen fail is an assumption"
//! convention, every test below was run against a deliberately broken build of
//! `runlog.rs` before being kept. The mutation each one kills is named in its
//! doc comment; the ledger is in the module doc of `runlog_bit_identity.rs`'s
//! sibling report. Tests whose mutation could NOT be made to fail them are not
//! here — they would be decoration.
//!
//! # Why a subprocess
//!
//! The sink is a process-global `OnceLock`, so a single test process can hold
//! exactly one sink for its whole life, and the durability property under test
//! (a KILLED process leaves a usable log) cannot be observed from inside the
//! process being killed. These tests therefore drive a tiny helper binary via
//! `cargo run --example`, which is the only way to observe both.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Locate the compiled `runlog_probe` example next to this test binary.
///
/// `cargo test` puts integration-test binaries in `target/<profile>/deps/` and
/// examples in `target/<profile>/examples/`, so the example is found relative
/// to `current_exe` rather than by shelling out to cargo (which would deadlock
/// on the build lock this test already holds).
fn probe_bin() -> PathBuf {
    let exe = std::env::current_exe().expect("current_exe");
    // .../target/<profile>/deps/runlog_writer-<hash>
    let profile_dir = exe
        .parent()
        .and_then(Path::parent)
        .expect("target/<profile>");
    let p = profile_dir.join("examples").join("runlog_probe");
    assert!(
        p.exists(),
        "probe binary {} not built. Run `cargo test -p ferric-scf` (which builds \
         examples) rather than `--test runlog_writer` alone.",
        p.display()
    );

    // STALENESS GUARD. `cargo test --test runlog_writer` does NOT rebuild
    // examples, so this test can silently run a probe compiled from a
    // DIFFERENT revision of runlog.rs than the one under test. That is not
    // hypothetical: it happened here. A mutation-testing harness was killed
    // mid-run with a "never write" mutation applied; the probe it had built
    // stayed on disk, and the next `--test runlog_writer` reported three
    // failures that had nothing to do with the (by then restored) source.
    //
    // A test that can pass or fail because of a file it did not build is not
    // measuring the code. Refuse to run against a probe older than the module
    // it exercises, and say what to do about it.
    let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/runlog.rs");
    if let (Ok(bin_m), Ok(src_m)) = (p.metadata(), src.metadata()) {
        if let (Ok(bin_t), Ok(src_t)) = (bin_m.modified(), src_m.modified()) {
            assert!(
                bin_t >= src_t,
                "the probe binary {} is OLDER than {}, so these tests would \
                 exercise a stale build rather than the current runlog.rs. \
                 Rebuild it: `cargo build -p ferric-scf --example runlog_probe` \
                 (or run the whole `cargo test -p ferric-scf`, which builds \
                 examples).",
                p.display(),
                src.display()
            );
        }
    }
    p
}

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ferric-runlog-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d.join("run.jsonl")
}

/// Every line of `path` that parses as JSON, plus the count of lines that did
/// not (a truncated final line is expected after a kill; anything else is a
/// bug).
fn parse_lines(path: &Path) -> (Vec<serde_json::Value>, usize) {
    let text = std::fs::read_to_string(path).unwrap_or_default();
    let mut ok = Vec::new();
    let mut bad = 0;
    for line in text.lines() {
        match serde_json::from_str::<serde_json::Value>(line) {
            Ok(v) => ok.push(v),
            Err(_) => bad += 1,
        }
    }
    (ok, bad)
}

/// MUTATION KILLED: dropping a record (`emit` returning early for some kinds,
/// or the loop writing every other one).
///
/// Asserts the exact count AND the `seq` sequence, which is strictly
/// 0,1,2,...: a gap is the signature of a dropped record even if the count
/// were somehow right.
#[test]
fn every_record_reaches_the_file_with_no_gaps() {
    let path = scratch("all-records");
    let out = Command::new(probe_bin())
        .args(["emit", path.to_str().unwrap(), "50"])
        .output()
        .expect("probe");
    assert!(out.status.success(), "probe failed: {out:?}");

    let (recs, bad) = parse_lines(&path);
    assert_eq!(bad, 0, "a record did not parse as JSON");
    assert_eq!(recs.len(), 50, "expected 50 records, got {}", recs.len());
    for (i, r) in recs.iter().enumerate() {
        assert_eq!(
            r["seq"].as_u64(),
            Some(i as u64),
            "seq gap at record {i}: {r}"
        );
        assert_eq!(r["iter"].as_u64(), Some(i as u64), "iter mismatch at {i}");
    }
}

/// MUTATION KILLED: buffering to exit (wrapping the file in a `BufWriter`, or
/// removing the per-record `flush`).
///
/// THE central property of this module. The probe is SIGKILLed partway through
/// — no unwinding, no `Drop`, no atexit — and the records written before the
/// kill must still be on disk and parseable. A buffered writer leaves an empty
/// or truncated-mid-buffer file and fails this.
///
/// SIGKILL specifically, not SIGTERM/SIGINT: a signal a process can handle
/// proves nothing about what survives an OOM kill, which is the real-world
/// case this guards.
#[test]
fn a_killed_run_leaves_a_usable_partial_log() {
    let path = scratch("killed");
    // The probe emits slowly and forever; we kill it mid-stream.
    let mut child = Command::new(probe_bin())
        .args(["slow", path.to_str().unwrap()])
        .spawn()
        .expect("spawn probe");

    // Wait until at least MIN_RECORDS records are on disk, then kill without
    // warning. Counting PARSED records rather than bytes ties the wait
    // condition to the assertion below: a byte threshold would make the test
    // flaky the moment a field was added or removed.
    const MIN_RECORDS: usize = 8;
    let mut waited = 0u64;
    loop {
        if parse_lines(&path).0.len() >= MIN_RECORDS {
            break;
        }
        assert!(
            waited < 20_000,
            "probe never wrote {MIN_RECORDS} records; it is not flushing per record"
        );
        std::thread::sleep(std::time::Duration::from_millis(10));
        waited += 10;
    }
    child.kill().expect("kill");
    let _ = child.wait();

    let (recs, bad) = parse_lines(&path);
    assert!(
        recs.len() >= MIN_RECORDS,
        "a SIGKILLed run left only {} parseable records — records are being \
         buffered rather than flushed per record, which is the exact failure \
         this module exists to prevent",
        recs.len()
    );
    // A kill can land mid-`write`, so AT MOST one trailing line may be
    // truncated. More than one means whole records were lost from the middle.
    assert!(
        bad <= 1,
        "{bad} unparseable lines after a kill; at most the final partial line \
         may be truncated"
    );
    // Everything that did land must be complete, in order, and usable.
    for (i, r) in recs.iter().enumerate() {
        assert_eq!(r["seq"].as_u64(), Some(i as u64), "seq gap at {i}");
        assert!(r["energy"].is_number(), "record {i} lost a field: {r}");
    }
}

/// MUTATION KILLED: swallowing a write error silently (no `warned` flag, or
/// `emit` ignoring the `Err` from `write_all`).
///
/// The probe points its log at a path that cannot be opened (a directory), so
/// `init` must fail. Two things must then hold, and both are asserted: the
/// run still SUCCEEDS (exit 0 — a logging failure never fails a calculation),
/// and it SAYS SO on stderr (a silent failure is how a log goes missing
/// unnoticed, which is the situation that motivated this module).
#[test]
fn an_unopenable_log_warns_and_does_not_fail_the_run() {
    let dir = std::env::temp_dir().join(format!("ferric-runlog-dir-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    // A directory path can never be opened as a file.
    let out = Command::new(probe_bin())
        .args(["emit", dir.to_str().unwrap(), "5"])
        .output()
        .expect("probe");

    assert!(
        out.status.success(),
        "an unopenable log path failed the run; logging must never do that"
    );
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("could not open") && err.contains("continuing without"),
        "an unopenable log path produced no warning on stderr; a silently \
         missing log is the failure mode this module exists to prevent. \
         stderr was: {err}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// MUTATION KILLED: `disable()` not poisoning the sink (a later `init`
/// installing a log the user explicitly refused with `[output] json = false`).
#[test]
fn disable_wins_over_a_later_init() {
    let path = scratch("disabled");
    let out = Command::new(probe_bin())
        .args(["disabled", path.to_str().unwrap()])
        .output()
        .expect("probe");
    assert!(out.status.success(), "probe failed: {out:?}");
    assert!(
        !path.exists(),
        "a log was created after `disable()`; an explicit opt-out was ignored"
    );
}

/// MUTATION KILLED: emitting a record with no trailing newline, which would
/// run two records together into one unparseable line.
///
/// Checked separately from the count test because a missing newline still
/// yields the right BYTES; only the line structure breaks, and JSON Lines is
/// the entire contract with a reader.
#[test]
fn records_are_one_per_line() {
    let path = scratch("newlines");
    let out = Command::new(probe_bin())
        .args(["emit", path.to_str().unwrap(), "20"])
        .output()
        .expect("probe");
    assert!(out.status.success());
    let text = std::fs::read_to_string(&path).unwrap();
    assert_eq!(
        text.lines().count(),
        20,
        "records did not land one per line"
    );
    assert!(text.ends_with('\n'), "the final record has no newline");
    assert!(
        !text.contains("}{"),
        "two records ran together on one line — a newline is missing"
    );
}
