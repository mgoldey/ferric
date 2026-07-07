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
