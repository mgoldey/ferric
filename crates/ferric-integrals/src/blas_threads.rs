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

    #[test]
    fn sets_and_restores() {
        let outer = unsafe { openblas_get_num_threads() };
        let inside = with_blas_threads(1, || unsafe { openblas_get_num_threads() });
        assert_eq!(inside, 1, "guard should set BLAS threads to 1 inside");
        let after = unsafe { openblas_get_num_threads() };
        assert_eq!(after, outer, "guard should restore the prior count");
    }

    #[test]
    fn nested_compose() {
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
