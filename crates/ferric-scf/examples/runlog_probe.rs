//! Test helper for `tests/runlog_writer.rs`: drives
//! [`ferric_scf::runlog`] in a SEPARATE process.
//!
//! A separate process is not a convenience here, it is the only way to test
//! two of the module's properties:
//!
//! * the sink is a process-global `OnceLock`, so one process can hold exactly
//!   one sink for its whole life — `init`, `disable` and "init failed" cannot
//!   all be exercised from a single test binary;
//! * "a SIGKILLed run leaves a usable partial log" cannot be observed from
//!   inside the process being killed.
//!
//! Modes (argv):
//!   `emit <path> <n>` — install a log at `<path>`, write `<n>` `scf_iter`
//!       records, exit 0. Exits 0 even when `<path>` cannot be opened, which
//!       is itself the assertion: logging never fails a run.
//!   `slow <path>`     — install a log and emit forever, ~5 ms apart, so the
//!       parent can SIGKILL it mid-stream.
//!   `disabled <path>` — call `disable()` and THEN `init(<path>)`; the file
//!       must not appear.

use ferric_scf::runlog;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("");
    let path = std::path::PathBuf::from(args.get(2).cloned().unwrap_or_default());

    match mode {
        "emit" => {
            let n: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(10);
            // Deliberately NOT asserted: an unopenable path returns false and
            // the loop below then emits into a no-op sink. The parent test
            // asserts this process still exits 0.
            let _ = runlog::init(&path);
            for i in 0..n {
                emit_one(i);
            }
        }
        "slow" => {
            assert!(runlog::init(&path), "probe: could not open {path:?}");
            // Deliberately unbounded: the PARENT kills this probe mid-stream
            // to test a truncated log. `loop` + an explicit counter says that,
            // where `for i in 0..` reads as an accident (and trips clippy's
            // unbounded-range lint).
            let mut i = 0usize;
            loop {
                emit_one(i);
                i += 1;
                // Slow enough that the parent reliably kills us mid-stream
                // rather than after we would have finished.
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
        }
        "disabled" => {
            runlog::disable();
            // Must NOT install: `disable` poisons the `OnceLock`.
            let _ = runlog::init(&path);
            emit_one(0);
        }
        other => {
            eprintln!("probe: unknown mode {other:?}");
            std::process::exit(2);
        }
    }
}

/// One `scf_iter` record with an `iter`/`energy` the parent can check.
///
/// `iter` mirrors the record index so the parent can assert ordering
/// independently of the `seq` envelope the module adds — two witnesses to a
/// dropped record rather than one.
fn emit_one(i: usize) {
    if let Some(rl) = runlog::log() {
        rl.scf_iter(
            "probe",
            None,
            i,
            -74.5 - i as f64 * 1e-6,
            1e-3,
            1e-4,
            1e-3,
            1e-5,
            Some(1e-6),
        );
    }
}
