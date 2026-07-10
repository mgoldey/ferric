//! Scoped OpenBLAS thread control. Inside rayon-parallel regions we pin BLAS to
//! 1 thread so nested OpenBLAS threads don't oversubscribe (and don't overflow
//! the 2 MB rayon worker stack via dgetrf/eigh — see openblas-rayon-dgetrf-crash).
//! Outside rayon, large GEMMs can use N BLAS threads. This guard sets the count
//! for a scope and restores it after.

use std::os::raw::c_int;

// OpenBLAS runtime API. Present in the OpenBLAS we link (verified: `nm -D`
// shows openblas_set_num_threads / openblas_get_num_threads as T). The `#[link]`
// is implicit — the binary already links openblas. Wrapped weakly so a
// reference-LAPACK build (no symbol) degrades to a no-op instead of a link error.
extern "C" {
    fn openblas_get_num_threads() -> c_int;
    fn openblas_set_num_threads(n: c_int);
}

/// Establish ferric's safe default threading model at program startup.
///
/// ferric **owns its parallelism through rayon** (ERI batches, frequency
/// quadrature, …) and calls OpenBLAS for the inner GEMM/SYRK/EIGH. If OpenBLAS
/// also spins up its own pool (its default = `nproc` threads), the result is a
/// *product* of threads (rayon × OpenBLAS) that oversubscribes the machine —
/// 3–5× slowdown — and can overflow the 2 MB rayon worker stack on big
/// dgetrf/eigh (see `openblas-rayon-dgetrf-crash`). The standard discipline is
/// to pin OpenBLAS to **1 thread per process** and let rayon do the threading;
/// the few large GEMMs outside rayon opt back into N threads via
/// [`with_blas_threads`].
///
/// Historically this was set via `.cargo/config.toml [env]`, which only applies
/// to `cargo`-invoked processes — the **release binary run directly** and the
/// **Python extension imported into a host process** inherited nothing and hit
/// the footgun. This function fixes that by setting the baseline in-code.
///
/// Idempotent and override-friendly: if the user has explicitly set
/// `OPENBLAS_NUM_THREADS` in the environment, that value is respected (we do not
/// override an explicit choice). Call once, early, from every entry point
/// (CLI `main`, Python module init). Calling it from a library function is fine
/// too — it is cheap and idempotent.
pub fn init_threading() {
    // Respect an explicit user override (any value, including >1 for a caller
    // who deliberately wants multi-threaded BLAS and manages rayon themselves).
    if let Ok(v) = std::env::var("OPENBLAS_NUM_THREADS") {
        if let Ok(n) = v.trim().parse::<usize>() {
            unsafe { openblas_set_num_threads(n.max(1) as c_int) };
            return;
        }
    }
    // No explicit override → ferric's safe default: 1 BLAS thread, rayon owns
    // the parallelism. Mirror it into the env so any child processes / late
    // OpenBLAS initialization observe the same baseline.
    unsafe { openblas_set_num_threads(1) };
    std::env::set_var("OPENBLAS_NUM_THREADS", "1");
}

/// Umbrella opt-in BLAS thread count for call sites *outside* any rayon
/// parallel region.
///
/// Defaults to **1** (matches [`init_threading`]'s process-wide default; a
/// caller who does nothing gets byte-identical behavior to before this
/// function existed). Raising it is opt-in via the `FERRIC_BLAS_THREADS`
/// environment variable, parsed the same way as the longstanding
/// `FERRIC_LANCZOS_BLAS_THREADS` (see `lanczos::lanczos_blas_threads`, the
/// precedent this generalizes): non-numeric/absent → default, present →
/// `.max(1)`.
///
/// ## Hazard model — read before wrapping a new call site
///
/// A raise here is safe **only** at call sites with a call-path proof that no
/// enclosing rayon parallel region is active when the wrapped closure runs.
/// Two independent hazards fire when that proof doesn't hold:
///
///  1. **Oversubscription.** rayon workers × OpenBLAS threads multiplies —
///     3–5× slowdown (see [`init_threading`]'s doc comment).
///  2. **Stack overflow.** A multi-threaded OpenBLAS `eigh`/`dgetrf` running
///     on a 2 MB rayon worker stack can overflow it outright (this is what
///     78bc70b reverted — see `openblas-rayon-dgetrf-crash`). This is not a
///     slowdown, it's a crash.
///
/// A raise must **never** reach the SAD / free-atom solve paths. Those run
/// inside `run_serial` — a single-thread rayon pool used specifically to
/// avoid the 18× per-atom-SCF regression rayon causes there (see
/// `rayon-penalty-on-free-atom-scf`). A single-thread rayon pool is still a
/// rayon pool: any BLAS call inside it is, by definition, "inside a rayon
/// parallel region" for purposes of hazard (2) above, so it is exactly the
/// rayon-adjacent hazard this doc warns about — even though only one worker
/// is active. Sites reachable from SAD/free-atom code must gate the raise
/// off with an explicit config flag (never a bare env read that fires
/// unconditionally), or must not call this function at all.
///
/// ## Precedence
///
/// `FERRIC_LANCZOS_BLAS_THREADS` (Lanczos-specific, kept for back-compat) >
/// `FERRIC_BLAS_THREADS` (this umbrella) > `1` (safe default). Lanczos's own
/// resolver (`lanczos::lanczos_blas_threads`) implements this by falling back
/// to this function when its own var is unset.
///
/// ## Runtime rayon-worker guard
///
/// Belt-and-suspenders on top of the call-path proof: if this function is
/// itself invoked from a rayon worker thread (`rayon::current_thread_index()
/// .is_some()`), it returns `1` regardless of the env var. A correct
/// call-path proof means this branch never fires in practice, but a future
/// refactor that accidentally moves a wrapped site inside a `par_iter` (or
/// runs it under `run_serial`'s single-thread pool — see the hazard note
/// above) degrades to today's safe default instead of silently reintroducing
/// the 78bc70b stack-overflow class of bug.
pub fn opt_in_blas_threads() -> usize {
    opt_in_blas_threads_with(|k| std::env::var(k).ok())
}

/// [`opt_in_blas_threads`] with an injected env lookup — the testable core.
///
/// Why injection instead of `set_var` in tests: `FERRIC_BLAS_THREADS` (and the
/// OpenBLAS thread count it drives through [`with_blas_threads`]) is
/// process-global, and the default test harness runs tests on parallel
/// threads. A test that raises the global count while any other test in the
/// same binary does BLAS work concurrently hits OpenBLAS's documented
/// non-thread-safety in multi-threaded mode — observed as silently corrupted
/// GEMM output, not a crash. Injecting the lookup lets precedence/guard tests
/// run with zero global mutation. Real-env raise tests live in dedicated
/// integration-test binaries (their own process) where nothing else runs.
#[doc(hidden)]
pub fn opt_in_blas_threads_with(get: impl Fn(&str) -> Option<String>) -> usize {
    if rayon::current_thread_index().is_some() {
        return 1;
    }
    if let Some(v) = get("FERRIC_BLAS_THREADS") {
        if let Ok(n) = v.trim().parse::<usize>() {
            return n.max(1);
        }
    }
    1
}

/// Run `f` with OpenBLAS pinned to `n` threads, restoring the previous count
/// afterward (including on panic). `n` is clamped to >= 1.
pub fn with_blas_threads<R>(n: usize, f: impl FnOnce() -> R) -> R {
    let n = n.max(1) as c_int;
    // Save + set. If the symbols are absent the build fails to link; we accept
    // that for the OpenBLAS build path (the project's standard). A guard struct
    // restores on drop so early-return/panic still restores.
    struct Restore(c_int);
    impl Drop for Restore {
        fn drop(&mut self) {
            unsafe { openblas_set_num_threads(self.0) }
        }
    }
    let prev = unsafe { openblas_get_num_threads() };
    let _restore = Restore(prev.max(1));
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
        let _g = GLOBAL_THREAD_STATE.lock().unwrap_or_else(|e| e.into_inner());
        let outer = unsafe { openblas_get_num_threads() };
        let inside = with_blas_threads(1, || unsafe { openblas_get_num_threads() });
        assert_eq!(inside, 1, "guard should set BLAS threads to 1 inside");
        let after = unsafe { openblas_get_num_threads() };
        assert_eq!(after, outer, "guard should restore the prior count");
    }

    #[test]
    fn init_threading_pins_to_one_by_default() {
        let _g = GLOBAL_THREAD_STATE.lock().unwrap_or_else(|e| e.into_inner());
        // No explicit override → baseline pinned to 1.
        std::env::remove_var("OPENBLAS_NUM_THREADS");
        init_threading();
        assert_eq!(unsafe { openblas_get_num_threads() }, 1);
        assert_eq!(std::env::var("OPENBLAS_NUM_THREADS").as_deref(), Ok("1"));
    }

    #[test]
    fn init_threading_respects_explicit_override() {
        let _g = GLOBAL_THREAD_STATE.lock().unwrap_or_else(|e| e.into_inner());
        // Explicit OPENBLAS_NUM_THREADS is honored, not clobbered to 1.
        std::env::set_var("OPENBLAS_NUM_THREADS", "3");
        init_threading();
        assert_eq!(unsafe { openblas_get_num_threads() }, 3);
        std::env::remove_var("OPENBLAS_NUM_THREADS");
    }

    // The opt_in resolver tests use the injected-lookup variant so they never
    // mutate the real (process-global) env — see opt_in_blas_threads_with's
    // doc comment for why set_var-based tests are a data race here.

    #[test]
    fn opt_in_blas_threads_defaults_to_one() {
        assert_eq!(opt_in_blas_threads_with(|_| None), 1);
    }

    #[test]
    fn opt_in_blas_threads_respects_umbrella_env() {
        let get = |k: &str| (k == "FERRIC_BLAS_THREADS").then(|| "3".to_string());
        assert_eq!(opt_in_blas_threads_with(get), 3);
    }

    #[test]
    fn opt_in_blas_threads_clamps_to_at_least_one() {
        let get = |k: &str| (k == "FERRIC_BLAS_THREADS").then(|| "0".to_string());
        assert_eq!(opt_in_blas_threads_with(get), 1);
    }

    #[test]
    fn opt_in_blas_threads_ignores_garbage() {
        let get = |k: &str| (k == "FERRIC_BLAS_THREADS").then(|| "not-a-number".to_string());
        assert_eq!(opt_in_blas_threads_with(get), 1);
    }

    #[test]
    fn opt_in_blas_threads_forces_one_inside_rayon_worker() {
        let get = |k: &str| (k == "FERRIC_BLAS_THREADS").then(|| "4".to_string());
        // Outside rayon: honors the (injected) env var.
        assert_eq!(opt_in_blas_threads_with(get), 4);
        // Inside a rayon worker: the runtime guard overrides the env var to 1,
        // even though it is set — this is the belt-and-suspenders check that
        // a wrapped call site accidentally moved inside a par_iter (or run
        // under run_serial's single-thread pool) still degrades safely.
        let inside = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| opt_in_blas_threads_with(get));
        assert_eq!(inside, 1, "rayon-worker guard must force 1 regardless of env");
    }

    #[test]
    fn nested_compose() {
        let _g = GLOBAL_THREAD_STATE.lock().unwrap_or_else(|e| e.into_inner());
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
}
