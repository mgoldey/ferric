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
//! grows `m_max` 1..=max, for both evaluators. Without the lock a child crashes
//! in roughly 20% of runs on 4 threads; with it, never.

use ferric_integrals::engine::{boys_fm_chebyshev7, tenno_gm, TennoKind};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

const CHILD_ENV: &str = "FERRIC_CORE_EVAL_RACE_CHILD";
const CHILDREN: usize = 40;
const READERS: usize = 3;

fn hammer(read: impl Fn() + Sync, grow: impl Fn(usize), max_m: usize) {
    let done = Arc::new(AtomicBool::new(false));
    std::thread::scope(|s| {
        for _ in 0..READERS {
            let done = Arc::clone(&done);
            let read = &read;
            s.spawn(move || {
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

#[test]
fn concurrent_core_eval_instance_growth_does_not_crash() {
    let exe = std::env::current_exe().expect("current exe");
    let mut failures = Vec::new();
    for i in 0..CHILDREN {
        let out = std::process::Command::new(&exe)
            .args(["core_eval_race_child", "--exact", "--nocapture"])
            .env(CHILD_ENV, "1")
            .env("RUST_TEST_THREADS", "1")
            .output()
            .expect("spawn child");
        let ok = out.status.success()
            && String::from_utf8_lossy(&out.stdout).contains("CORE_EVAL_RACE_CHILD_OK");
        if !ok {
            failures.push((i, out.status));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {CHILDREN} children crashed racing the libint2 core-evaluator singleton: {failures:?}",
        failures.len()
    );
}
