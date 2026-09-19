//! Measure the JSON run log's per-record cost directly.
//!
//! The honest way to measure per-iteration I/O is to time the emit path
//! itself. Timing a whole `ferric` process instead would bury a
//! microsecond-scale per-record cost under process startup, basis parsing and
//! integral setup — the difference would sit inside the run-to-run noise and
//! the measurement would report "no overhead" whether or not there was any.
//! That is a measurement that cannot fail, which is not a measurement.
//!
//! Reports the per-record cost, which the caller compares against the cost of
//! a real SCF iteration. Run:
//!
//! ```text
//! cargo run --release -p ferric-scf --example runlog_overhead -- <path> <n>
//! ```

use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = std::path::PathBuf::from(
        args.get(1)
            .cloned()
            .unwrap_or_else(|| "/tmp/ferric-overhead.jsonl".to_string()),
    );
    let n: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(10_000);

    // --- baseline: the call with NO sink installed (what a library caller,
    // and any run with `--no-json`, pays). This is the cost of the
    // `runlog::log()` check at each emit site and nothing else.
    let t0 = Instant::now();
    for i in 0..n {
        emit(i);
    }
    let off = t0.elapsed();

    // --- with a sink: serialize + write + flush, per record.
    assert!(
        ferric_scf::runlog::init(&path),
        "could not open {}",
        path.display()
    );
    let t1 = Instant::now();
    for i in 0..n {
        emit(i);
    }
    let on = t1.elapsed();

    let off_ns = off.as_nanos() as f64 / n as f64;
    let on_ns = on.as_nanos() as f64 / n as f64;
    println!("records            : {n}");
    println!("log OFF  per record: {off_ns:9.1} ns");
    println!(
        "log ON   per record: {on_ns:9.1} ns  ({:.2} us)",
        on_ns / 1e3
    );
    println!("added    per record: {:9.1} ns", on_ns - off_ns);
    let bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
    println!(
        "bytes written      : {bytes} ({} B/record)",
        bytes / n as u64
    );
}

fn emit(i: usize) {
    if let Some(rl) = ferric_scf::runlog::log() {
        rl.scf_iter(
            "bench",
            Some(0),
            i,
            -74.963_146_800_039_1,
            1.2e-9,
            3.4e-10,
            5.6e-9,
            7.8e-11,
            Some(9.1e-12),
        );
    }
}
