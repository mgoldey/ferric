//! Serialising tests that touch the process-global memory pool.
//!
//! `ferric_core::memory::pool` keeps one process-wide pool. Two tests in one
//! binary that install, charge or clear it concurrently see each other's
//! state (a killed mutation can look like a survivor, a sibling can be
//! refused a charge it never asked for). Hold a [`CleanSlot`] for the whole
//! test: it takes a binary-wide lock and clears the global pool on entry
//! **and** on exit. Both halves are needed.

use std::sync::{Mutex, MutexGuard, OnceLock};

use ferric_core::memory::pool::clear_global;

/// The binary-wide lock behind [`CleanSlot`]. A poisoned lock (an earlier
/// test panicked while holding it) is recovered, so one failure does not
/// cascade into every sibling.
///
/// The `static` lives in this crate, so it is one lock per test binary, the
/// same scope the per-file copies had.
pub fn pool_lock() -> MutexGuard<'static, ()> {
    static L: OnceLock<Mutex<()>> = OnceLock::new();
    match L.get_or_init(|| Mutex::new(())).lock() {
        Ok(g) => g,
        Err(e) => e.into_inner(),
    }
}

/// Holds [`pool_lock`] and clears the global pool on acquire and on drop.
///
/// It touches only the pool: it does not read, set or restore any environment
/// variable. A test that sets one (`FERRIC_*`) must restore it itself.
#[must_use = "the lock is released when the slot is dropped"]
pub struct CleanSlot(#[allow(dead_code)] MutexGuard<'static, ()>);

impl CleanSlot {
    pub fn acquire() -> Self {
        let g = pool_lock();
        clear_global();
        Self(g)
    }
}

impl Drop for CleanSlot {
    fn drop(&mut self) {
        clear_global();
    }
}
