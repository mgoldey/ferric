//! OpenBLAS thread control: the process baseline, the serial-section thread
//! count (a user setting), scoped raises, and the thread stack size that makes
//! threaded LAPACK safe off the main thread.
//!
//! # The model
//!
//! * **Parallel regions** (rayon) run OpenBLAS at the process *baseline*, which
//!   [`init_threading`](crate::blas_threads::init_threading) pins to 1 unless the user set `OPENBLAS_NUM_THREADS`.
//!   rayon owns the cores there; a baseline above 1 multiplies rayon threads by
//!   BLAS threads. That oversubscription is a user configuration choice, not
//!   something ferric designs around.
//! * **Serial sections** (the per-iteration Fock diagonalization, the DIIS
//!   commutator, orthogonalizer and density GEMMs, and the other sites that
//!   wrap their work in `with_blas_threads(opt_in_blas_threads(), ..)`) run
//!   outside any rayon region, where the cores would otherwise sit idle. They
//!   use [`opt_in_blas_threads`](crate::blas_threads::opt_in_blas_threads) threads: by default the size of the global
//!   rayon pool (so `RAYON_NUM_THREADS` is the one thread count a user sets),
//!   overridable by `[parallel] blas_threads` (CLI), `ferric.set_blas_threads`
//!   (Python) or `FERRIC_BLAS_THREADS`. `1` gives results that are
//!   bit-identical across thread counts; any other value lets OpenBLAS split
//!   reductions by thread count, which moves results at the last bit.
//!
//! # Why threaded LAPACK used to crash (resolved 2026-11-08)
//!
//! OpenBLAS's threaded LU (`getrf_parallel`, reached by `.inv()`, `.det()`,
//! `.solve()`) needs more than 2 MiB of C stack from 2 BLAS threads up. Rust
//! gives every spawned thread — rayon workers, `std::thread::spawn`, and the
//! test harness's threads — a 2 MiB stack; only the main thread gets
//! `ulimit -s` (8 MiB). The earlier "LU is not reentrant / crashes even on a
//! plain 8 MB thread" diagnosis was a 2 MiB *test-harness* thread. With
//! [`THREAD_STACK_BYTES`](crate::blas_threads::THREAD_STACK_BYTES) stacks, LU / eigh / Cholesky / GEMM at any BLAS
//! thread count run correctly on any thread, including several threads at
//! once (crash matrix in `wiki/notes/blas-threads-2026-11-08/`).
//! [`init_threading`](crate::blas_threads::init_threading) applies that size to every thread spawned afterwards.
//!
//! Canonical home: this module lives in `ferric-core` (dependency-free, so any
//! crate in the workspace can reach it without a new dependency edge).
//! `ferric_integrals::blas_threads` re-exports everything from here.

use std::cell::Cell;
use std::os::raw::c_int;
use std::sync::atomic::{AtomicI32, AtomicUsize, Ordering};

// OpenBLAS runtime API. Present in the OpenBLAS we link (verified: `nm -D`
// shows openblas_set_num_threads / openblas_get_num_threads as T). The `#[link]`
// is implicit — the binary already links openblas. NOT weakly linked: a build
// against a reference-LAPACK/BLAS (lacking these symbols) fails at link time
// with an undefined-symbol error, not a runtime no-op.
extern "C" {
    fn openblas_get_num_threads() -> c_int;
    fn openblas_set_num_threads(n: c_int);
}

/// Stack size for every thread the process spawns (rayon workers,
/// `std::thread` workers), applied by [`init_threading`].
///
/// Measured 2026-11-08 (OpenBLAS 0.3.26 pthread, `MAX_THREADS=64`): each
/// `dgetrf_parallel` frame takes about 527 KiB of stack and the routine
/// recurses (4 frames deep at n = 113), so `.inv()` / `.det()` at 2 or more
/// BLAS threads overflows a 2 MiB or 3 MiB stack and runs clean from 4 MiB up
/// at every size (to n = 2000) and thread count tried. 64 MiB leaves room for
/// deeper recursion at large n and for an OpenBLAS built with a larger
/// `NUM_THREADS` (the frame grows with its square). Stack memory is reserved,
/// not committed, until touched, so idle workers cost address space only.
pub const THREAD_STACK_BYTES: usize = 64 << 20;

/// The OpenBLAS thread count parallel regions run at, as recorded by
/// [`init_threading`] (or, if that never ran, by the first outermost
/// [`with_blas_threads`] scope). `0` = not recorded yet.
static BASELINE: AtomicI32 = AtomicI32::new(0);

/// Serial-section BLAS threads set explicitly by the program (CLI
/// `[parallel] blas_threads`, Python `ferric.set_blas_threads`). `0` = unset.
static EXPLICIT_SERIAL: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    /// (nesting depth of [`with_blas_threads`] scopes on this thread, the count
    /// the innermost one set) — what a nested scope restores on exit.
    static SCOPE: Cell<(usize, c_int)> = const { Cell::new((0, 0)) };
}

/// Establish ferric's threading model at program startup.
///
/// 1. **Thread stacks.** If `RUST_MIN_STACK` is unset, set it to
///    [`THREAD_STACK_BYTES`], so every `std::thread` and rayon worker spawned
///    afterwards (including pools built with `ThreadPoolBuilder` and no
///    explicit `stack_size`) can run threaded LAPACK; and build the global
///    rayon pool with at least that stack (a no-op if the pool already exists).
/// 2. **BLAS baseline.** Pin OpenBLAS to 1 thread for parallel regions, unless
///    the user set `OPENBLAS_NUM_THREADS` (respected as given). Serial sections
///    raise above this baseline per [`opt_in_blas_threads`].
///
/// Historically the baseline was set via `.cargo/config.toml [env]`, which only
/// applies to `cargo`-invoked processes — the release binary run directly and
/// the Python extension inherited nothing. Call once, early, from every entry
/// point (CLI `main`, Python module init), before any thread is spawned: Rust
/// reads `RUST_MIN_STACK` once, at the first spawn. Idempotent.
pub fn init_threading() {
    if std::env::var_os("RUST_MIN_STACK").is_none() {
        std::env::set_var("RUST_MIN_STACK", THREAD_STACK_BYTES.to_string());
    }
    let stack = std::env::var("RUST_MIN_STACK")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(THREAD_STACK_BYTES)
        .max(THREAD_STACK_BYTES);
    // Err = the global pool already exists (an earlier rayon call, or a host
    // that configured it); its workers then carry whatever stack size the
    // first spawn read, which is why this function must run first.
    let _ = rayon::ThreadPoolBuilder::new()
        .stack_size(stack)
        .build_global();

    // Respect an explicit user override (any value, including >1 for a caller
    // who deliberately wants multi-threaded BLAS inside rayon regions).
    if let Ok(v) = std::env::var("OPENBLAS_NUM_THREADS") {
        if let Ok(n) = v.trim().parse::<usize>() {
            let n = n.max(1) as c_int;
            unsafe { openblas_set_num_threads(n) };
            BASELINE.store(n, Ordering::SeqCst);
            return;
        }
    }
    // No explicit override → 1 BLAS thread in parallel regions. Mirror it into
    // the env so child processes / late OpenBLAS initialization observe it.
    unsafe { openblas_set_num_threads(1) };
    BASELINE.store(1, Ordering::SeqCst);
    std::env::set_var("OPENBLAS_NUM_THREADS", "1");
}

/// Set the serial-section BLAS thread count for the rest of the process
/// (`Some(n)`, floored at 1), or return to the env/default resolution
/// (`None`). The CLI calls this from `[parallel] blas_threads`; Python exposes
/// it as `ferric.set_blas_threads`.
pub fn set_serial_blas_threads(n: Option<usize>) {
    EXPLICIT_SERIAL.store(n.map_or(0, |n| n.max(1)), Ordering::SeqCst);
}

/// The value last given to [`set_serial_blas_threads`], if any.
pub fn explicit_serial_blas_threads() -> Option<usize> {
    let n = EXPLICIT_SERIAL.load(Ordering::SeqCst);
    (n > 0).then_some(n)
}

/// BLAS thread count for serial sections — call sites *outside* any rayon
/// parallel region (see the module doc).
///
/// Precedence: [`set_serial_blas_threads`] (TOML / Python) >
/// `FERRIC_BLAS_THREADS` > the size of the global rayon pool. `0` from the env
/// means "the default". Called on a rayon worker it returns 1: a serial
/// section that runs inside a parallel region (the free-atom SCFs of the SAD
/// guess run in a one-thread pool; tests run whole SCFs inside
/// `pool.install`) is one of the pool's tasks and stays on one BLAS thread.
///
/// `FERRIC_LANCZOS_BLAS_THREADS` (Lanczos/Davidson only, kept for back-compat)
/// takes precedence over this in those solvers.
pub fn opt_in_blas_threads() -> usize {
    opt_in_blas_threads_with(|k| std::env::var(k).ok())
}

/// The serial BLAS-thread descriptor (`FERRIC_BLAS_THREADS`), routed through
/// [`ConfigVar`](crate::config::ConfigVar) for the uniform parse/audit.
///
/// A thread count is a performance knob, so a malformed value degrades to the
/// default (`0` = the rayon pool size) rather than aborting a run.
pub(crate) const BLAS_THREADS: crate::config::ConfigVar<usize> = crate::config::ConfigVar {
    env_name: "FERRIC_BLAS_THREADS",
    default: 0,
    parse: |s| Ok(s.trim().parse::<usize>().unwrap_or(0)),
    validate: crate::config::accept_any,
};

/// [`opt_in_blas_threads`] with an injected env lookup — the testable core.
///
/// Injection instead of `set_var` in tests: the env is process-global and the
/// test harness runs tests on parallel threads.
#[doc(hidden)]
pub fn opt_in_blas_threads_with(get: impl Fn(&str) -> Option<String>) -> usize {
    if rayon::current_thread_index().is_some() {
        return 1;
    }
    let explicit = EXPLICIT_SERIAL.load(Ordering::SeqCst);
    resolve_serial_blas_threads(
        (explicit > 0).then_some(explicit),
        get,
        rayon::current_num_threads(),
    )
}

/// The precedence rule behind [`opt_in_blas_threads`], free of process state:
/// `explicit` > env > `auto`; a `0` falls through; the result is at least 1.
#[doc(hidden)]
pub fn resolve_serial_blas_threads(
    explicit: Option<usize>,
    get: impl Fn(&str) -> Option<String>,
    auto: usize,
) -> usize {
    let v = BLAS_THREADS
        .resolve(explicit.filter(|&n| n > 0), get)
        .map(|r| r.value)
        .unwrap_or(0);
    if v == 0 {
        auto.max(1)
    } else {
        v
    }
}

/// The OpenBLAS thread count in effect right now (process-global).
pub fn current_blas_threads() -> usize {
    unsafe { openblas_get_num_threads() }.max(1) as usize
}

/// Run `f` with OpenBLAS at `n` threads (clamped to >= 1), then restore.
///
/// A nested scope restores the count its enclosing scope on the same thread
/// set; the outermost scope restores the process baseline (see
/// [`init_threading`]). Restoring the baseline, not "whatever the count was on
/// entry", matters when two threads open scopes at once (parallel tests, a
/// host calling ferric from several threads): OpenBLAS's count is
/// process-global, and an interleaved save/restore of entry values can leave a
/// raised count behind for every later parallel region.
///
/// `n > 1` must not be requested on a rayon worker (debug-asserted): that is a
/// parallel region, which runs at the baseline. [`opt_in_blas_threads`]
/// already returns 1 there.
pub fn with_blas_threads<R>(n: usize, f: impl FnOnce() -> R) -> R {
    let n = n.max(1) as c_int;
    debug_assert!(
        n == 1 || rayon::current_thread_index().is_none(),
        "with_blas_threads({n}) on a rayon worker: serial-section BLAS threads \
         are raised outside parallel regions only"
    );
    struct Restore(c_int, (usize, c_int));
    impl Drop for Restore {
        fn drop(&mut self) {
            SCOPE.with(|s| s.set(self.1));
            unsafe { openblas_set_num_threads(self.0) }
        }
    }
    let outer = SCOPE.with(|s| s.get());
    let restore_to = if outer.0 > 0 {
        outer.1
    } else {
        let now = unsafe { openblas_get_num_threads() }.max(1);
        match BASELINE.compare_exchange(0, now, Ordering::SeqCst, Ordering::SeqCst) {
            Ok(_) => now,
            Err(b) => b,
        }
    };
    let _restore = Restore(restore_to, outer);
    SCOPE.with(|s| s.set((outer.0 + 1, n)));
    unsafe { openblas_set_num_threads(n) };
    f()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // The BLAS thread count and OPENBLAS_NUM_THREADS are process-global, and
    // the default test harness runs tests on parallel threads — serialize
    // every test that touches them.
    static GLOBAL_THREAD_STATE: Mutex<()> = Mutex::new(());

    #[test]
    fn sets_and_restores() {
        let _g = GLOBAL_THREAD_STATE
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let outer = unsafe { openblas_get_num_threads() };
        let inside = with_blas_threads(1, || unsafe { openblas_get_num_threads() });
        assert_eq!(inside, 1, "guard should set BLAS threads to 1 inside");
        let after = unsafe { openblas_get_num_threads() };
        assert_eq!(after, outer, "guard should restore the prior count");
    }

    #[test]
    fn init_threading_pins_to_one_by_default() {
        let _g = GLOBAL_THREAD_STATE
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        // No explicit override → baseline pinned to 1.
        std::env::remove_var("OPENBLAS_NUM_THREADS");
        init_threading();
        assert_eq!(unsafe { openblas_get_num_threads() }, 1);
        assert_eq!(std::env::var("OPENBLAS_NUM_THREADS").as_deref(), Ok("1"));
        // Thread stacks: RUST_MIN_STACK is set (or kept) for every later spawn.
        assert!(std::env::var("RUST_MIN_STACK").is_ok());
    }

    #[test]
    fn init_threading_respects_explicit_override() {
        let _g = GLOBAL_THREAD_STATE
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        // Explicit OPENBLAS_NUM_THREADS is honored, not clobbered to 1.
        std::env::set_var("OPENBLAS_NUM_THREADS", "3");
        init_threading();
        assert_eq!(unsafe { openblas_get_num_threads() }, 3);
        // An outermost scope restores that baseline.
        with_blas_threads(2, || ());
        assert_eq!(unsafe { openblas_get_num_threads() }, 3);
        // Back to the default baseline for the other tests in this binary.
        std::env::remove_var("OPENBLAS_NUM_THREADS");
        init_threading();
        assert_eq!(unsafe { openblas_get_num_threads() }, 1);
    }

    // The resolver tests use the injected-lookup / pure variants so they never
    // mutate the real (process-global) env.

    fn env(v: &'static str) -> impl Fn(&str) -> Option<String> {
        move |k: &str| (k == "FERRIC_BLAS_THREADS").then(|| v.to_string())
    }

    #[test]
    fn serial_blas_threads_default_is_the_rayon_pool_size() {
        assert_eq!(resolve_serial_blas_threads(None, |_| None, 6), 6);
        assert_eq!(resolve_serial_blas_threads(None, |_| None, 0), 1);
        // The live resolver, off any rayon worker, with nothing set, follows
        // the global pool. (EXPLICIT_SERIAL is only written by
        // `explicit_setting_wins_over_env`, which holds the lock.)
        let _g = GLOBAL_THREAD_STATE
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        assert_eq!(
            opt_in_blas_threads_with(|_| None),
            rayon::current_num_threads()
        );
    }

    #[test]
    fn serial_blas_threads_respects_env() {
        assert_eq!(resolve_serial_blas_threads(None, env("3"), 6), 3);
        assert_eq!(resolve_serial_blas_threads(None, env("1"), 6), 1);
    }

    #[test]
    fn serial_blas_threads_zero_or_garbage_env_means_default() {
        assert_eq!(resolve_serial_blas_threads(None, env("0"), 6), 6);
        assert_eq!(resolve_serial_blas_threads(None, env("not-a-number"), 6), 6);
    }

    #[test]
    fn explicit_setting_wins_over_env() {
        assert_eq!(resolve_serial_blas_threads(Some(2), env("5"), 6), 2);
        assert_eq!(resolve_serial_blas_threads(Some(1), env("5"), 6), 1);
        let _g = GLOBAL_THREAD_STATE
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        set_serial_blas_threads(Some(1));
        let got = opt_in_blas_threads_with(env("5"));
        set_serial_blas_threads(None);
        assert_eq!(got, 1, "set_serial_blas_threads(Some(1)) must win over env");
        assert_eq!(opt_in_blas_threads_with(env("5")), 5);
    }

    #[test]
    fn opt_in_blas_threads_forces_one_inside_rayon_worker() {
        // Outside rayon: honors the (injected) env var.
        assert_eq!(opt_in_blas_threads_with(env("4")), 4);
        // Inside a rayon worker: 1, whatever is set — a parallel region runs
        // at the baseline (this is how the SAD free-atom SCFs, which run in a
        // one-thread pool, stay single-threaded).
        let inside = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| opt_in_blas_threads_with(env("4")));
        assert_eq!(
            inside, 1,
            "rayon-worker guard must force 1 regardless of env"
        );
    }

    #[test]
    fn nested_compose() {
        let _g = GLOBAL_THREAD_STATE
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        with_blas_threads(4, || {
            let n4 = unsafe { openblas_get_num_threads() };
            assert_eq!(n4, 4);
            with_blas_threads(1, || {
                assert_eq!(unsafe { openblas_get_num_threads() }, 1);
            });
            // restored to 4 after the inner scope
            assert_eq!(unsafe { openblas_get_num_threads() }, 4);
        });
    }

    /// Two threads whose scopes interleave A-enter, B-enter, A-exit, B-exit
    /// must leave the process at the baseline. Restoring "the count on entry"
    /// (the pre-2026-11-08 rule) leaves B's entry value — A's raised 3 —
    /// behind for every later parallel region.
    #[test]
    fn interleaved_scopes_on_two_threads_leave_the_baseline() {
        use std::sync::mpsc::channel;
        let _g = GLOBAL_THREAD_STATE
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let base = unsafe { openblas_get_num_threads() };
        let (a_in_tx, a_in_rx) = channel::<()>();
        let (b_in_tx, b_in_rx) = channel::<()>();
        let (a_out_tx, a_out_rx) = channel::<()>();
        let a = std::thread::spawn(move || {
            with_blas_threads(3, || {
                a_in_tx.send(()).unwrap();
                b_in_rx.recv().unwrap();
            });
            a_out_tx.send(()).unwrap();
        });
        let b = std::thread::spawn(move || {
            a_in_rx.recv().unwrap();
            with_blas_threads(2, || {
                b_in_tx.send(()).unwrap();
                a_out_rx.recv().unwrap();
            });
        });
        a.join().unwrap();
        b.join().unwrap();
        assert_eq!(
            unsafe { openblas_get_num_threads() },
            base,
            "interleaved scopes leaked a raised BLAS count"
        );
    }

    /// The debug assertion: raising BLAS threads on a rayon worker is a bug at
    /// the call site.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "on a rayon worker")]
    fn raising_on_a_rayon_worker_is_a_debug_assertion() {
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| with_blas_threads(2, || ()));
    }
}
