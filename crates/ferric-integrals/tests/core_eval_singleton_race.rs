//! libint2's core-integral evaluators (`TennoGmEval`, behind `Operator::yukawa`
//! and `Operator::slater_geminal`; `FmEval_Chebyshev7`, behind Coulomb/erf/erfc)
//! are process singletons whose `instance(m_max)` REPLACES the static
//! `std::shared_ptr` when a caller asks for a larger `m_max`, and every other
//! caller copies that `shared_ptr` with no lock. A copy racing a replacement
//! reads a freed object (the old instance's `max_m()`, or its control block),
//! so concurrent callers can SIGSEGV. ThreadSanitizer reports it as a data race
//! between `instance()`'s `operator=` and the unlocked `instance_->max_m()`
//! read, then as a read of memory freed by that `operator=`.
//!
//! The shim serializes every `instance()` call on `libint_ctor_mutex`: engine
//! constructors always did; the `scf_tenno_gm` / `scf_boys_fm` test hooks did
//! not, which is how a parallel `tenno_table_edge` run (engine constructors and
//! `tenno_gm` calls at different `m_max` in flight at once) crashed once in the
//! wheels perturb canary.
//!
//! The test runs the race in fresh child processes (the singleton only grows
//! once per process): reader threads hammer `instance(0)` while the main thread
//! grows `m_max` 1..=max, for both evaluators. MEASURED 2026-11-11 (libint2
//! 2.7.2 with the override, MALLOC_PERTURB_=1, a 12-core box): without the lock
//! 100/100 standalone children crashed (SIGSEGV or a glibc abort in free), and
//! this test failed 40/40 children in each of two runs; with the lock, 0/500
//! children.

use ferric_integrals::engine::{boys_fm_chebyshev7, tenno_gm, TennoKind};
use std::io::Read;
use std::process::{Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

const CHILD_ENV: &str = "FERRIC_CORE_EVAL_RACE_CHILD";
const CHILDREN: usize = 40;
const READERS: usize = 3;

fn hammer(read: impl Fn() + Sync, grow: impl Fn(usize), max_m: usize) {
    let done = AtomicBool::new(false);
    std::thread::scope(|s| {
        for _ in 0..READERS {
            s.spawn(|| {
                while !done.load(Ordering::Relaxed) {
                    read();
                }
            });
        }
        for m in 1..=max_m {
            grow(m);
        }
        done.store(true, Ordering::Relaxed);
    });
}

/// Child body: only does work when spawned by the parent test.
#[test]
fn core_eval_race_child() {
    if std::env::var_os(CHILD_ENV).is_none() {
        return;
    }
    hammer(
        || {
            tenno_gm(TennoKind::Yukawa, 1.0, 3.0, 0, 1.0).expect("tenno m=0");
        },
        |m| {
            tenno_gm(TennoKind::Yukawa, 1.0, 3.0, m, 1.0).expect("tenno grow");
        },
        20, // TennoGmEval's table mmax
    );
    hammer(
        || {
            boys_fm_chebyshev7(3.0, 0).expect("boys m=0");
        },
        |m| {
            boys_fm_chebyshev7(3.0, m).expect("boys grow");
        },
        40,
    );
    println!("CORE_EVAL_RACE_CHILD_OK");
}

/// A child that has not finished in this long is hung (a healthy child takes
/// well under a second); it is killed and counted as a failure.
const CHILD_TIMEOUT: Duration = Duration::from_secs(60);

/// How a child ended: its exit status (`None` if it was killed on timeout) and
/// the last lines of its stderr.
fn run_child(exe: &std::path::Path) -> (Option<ExitStatus>, bool, String) {
    let mut child = Command::new(exe)
        .args(["core_eval_race_child", "--exact", "--nocapture"])
        .env(CHILD_ENV, "1")
        .env("RUST_TEST_THREADS", "1")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn child");
    // Drain both pipes on threads so a chatty child cannot block on a full pipe.
    let mut out_pipe = child.stdout.take().unwrap();
    let mut err_pipe = child.stderr.take().unwrap();
    let out_t = std::thread::spawn(move || {
        let mut s = String::new();
        out_pipe.read_to_string(&mut s).ok();
        s
    });
    let err_t = std::thread::spawn(move || {
        let mut s = String::new();
        err_pipe.read_to_string(&mut s).ok();
        s
    });
    let start = Instant::now();
    let status = loop {
        if let Some(st) = child.try_wait().expect("wait child") {
            break Some(st);
        }
        if start.elapsed() > CHILD_TIMEOUT {
            child.kill().ok();
            child.wait().ok();
            break None;
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    let stdout = out_t.join().unwrap_or_default();
    let stderr = err_t.join().unwrap_or_default();
    let ok = stdout.contains("CORE_EVAL_RACE_CHILD_OK");
    let lines: Vec<&str> = stderr.lines().collect();
    let tail = lines[lines.len().saturating_sub(8)..].join("\n");
    (status, ok, tail)
}

/// `exited 0`, `exit code N`, `killed by signal N (SIGSEGV)` or `TIMED OUT`.
fn describe(status: Option<ExitStatus>) -> String {
    use std::os::unix::process::ExitStatusExt;
    match status {
        None => format!("TIMED OUT after {CHILD_TIMEOUT:?} (killed)"),
        Some(st) => match (st.code(), st.signal()) {
            (Some(c), _) => format!("exit code {c}"),
            (None, Some(sig)) => {
                let name = match sig {
                    6 => " (SIGABRT)",
                    7 => " (SIGBUS)",
                    11 => " (SIGSEGV)",
                    _ => "",
                };
                format!("killed by signal {sig}{name}")
            }
            _ => format!("{st:?}"),
        },
    }
}

#[test]
fn concurrent_core_eval_instance_growth_does_not_crash() {
    let exe = std::env::current_exe().expect("current exe");
    let mut failures = Vec::new();
    for i in 0..CHILDREN {
        let (status, ok, tail) = run_child(&exe);
        if !(ok && status.is_some_and(|s| s.success())) {
            failures.push(format!(
                "child {i}: {}; stderr tail:\n{tail}",
                describe(status)
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {CHILDREN} children failed racing the libint2 core-evaluator singleton:\n{}",
        failures.len(),
        failures.join("\n---\n")
    );
}
