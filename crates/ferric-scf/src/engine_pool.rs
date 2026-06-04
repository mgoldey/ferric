//! Per-thread 2e integral-engine pool for the direct (exact-ERI) Fock builders.
//!
//! ## Why this exists
//!
//! libint2 engine construction is **expensive** (it allocates scratch sized by
//! the basis's max angular momentum and max primitive count, and builds
//! recurrence tables) and is **serialized behind a global mutex** in the shim
//! (libint2 `Engine` ctors are not thread-safe). The direct builders parallelize
//! with `into_par_iter().fold(|| Engine::new_2e(...), ...)`. Rayon calls a
//! `fold` init closure **once per work-chunk, not once per thread** — for a small
//! molecule the shell-pair list splits into dozens of chunks, so the engine was
//! constructed dozens of times per Fock build, every one of them queueing on the
//! global ctor mutex. With N threads all stuck on that mutex, wall time
//! *exploded* (PH3/aug-cc-pVDZ: 9.6 s at RAYON=1 vs >120 s at RAYON=8 — the
//! threads spent all their time contending, not computing). The effect is worst
//! for high-angular-momentum heavy-element bases (Si, P, S, Cl) where each
//! construction is slowest.
//!
//! ## The fix
//!
//! Build **exactly one engine per rayon worker thread** up front (so the mutex is
//! hit at most `num_threads` times total), store them in a pool indexed by
//! `rayon::current_thread_index()`, and have each parallel task borrow its
//! thread's engine via a `Mutex`. Construction count drops from O(chunks) to
//! O(threads), killing the contention while keeping full parallelism.

use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::operator::Operator;
use ferric_core::FerricError;
use std::sync::Mutex;

/// A pool of 2e engines, one slot per rayon worker thread (plus one spare for
/// the calling thread / non-rayon contexts at index `len-1`).
pub struct EnginePool {
    engines: Vec<Mutex<Engine>>,
}

impl EnginePool {
    /// Construct `num_threads + 1` engines (serialized, but bounded). The `+1`
    /// covers `current_thread_index() == None` (work run on a non-pool thread).
    pub fn new(op: Operator, prep: &PreparedBasis, precision: f64) -> Result<Self, FerricError> {
        let n = rayon::current_num_threads().max(1) + 1;
        let mut engines = Vec::with_capacity(n);
        for _ in 0..n {
            engines.push(Mutex::new(Engine::new_2e(op, prep, precision)?));
        }
        Ok(EnginePool { engines })
    }

    /// Run `f` with this thread's engine. Indexed by `current_thread_index()`;
    /// falls back to the spare slot for non-rayon threads. The per-slot `Mutex`
    /// is uncontended in practice (one thread maps to one slot), so the lock is
    /// effectively free — it exists only to satisfy `&mut Engine` borrowing.
    #[inline]
    pub fn with<R>(&self, f: impl FnOnce(&mut Engine) -> R) -> R {
        let idx = rayon::current_thread_index().unwrap_or(self.engines.len() - 1);
        // Guard against an index beyond the pool (shouldn't happen, but be safe).
        let slot = idx.min(self.engines.len() - 1);
        let mut eng = self.engines[slot].lock().unwrap();
        f(&mut eng)
    }
}
