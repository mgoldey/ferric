//! [`crate::memory::pool::MemoryPool`]: one process-global pool of bytes that allocations are
//! **debited** against, rather than a ceiling every call site re-reads.
//!
//! # Why this is not just `MemoryPlan`
//!
//! [`crate::memory::plan::MemoryPlan`] is the right *description* of a method's
//! allocations — named reservations, lifetimes, a breakdown report — and this
//! module deliberately does not reinvent any of that. But a plan is inert,
//! per-method, `Clone`, and mutated through `&mut self`. Three things it
//! structurally cannot do:
//!
//! 1. **Compose across subsystems.** Two plans built in two different crates
//!    against the same ceiling both pass `check()`. `sub_plan` composes only
//!    when one caller can hand the child to the other, which is exactly the
//!    threading that has not happened at ~50 sites.
//! 2. **Be shared across rayon workers.** `&mut self` reservations cannot come
//!    from worker threads.
//! 3. **Release.** A plan's reservation lives as long as the plan value. A
//!    transient buffer inside an SCF iteration has to give its bytes back, or
//!    an 80-iteration loop exhausts the pool on iteration two.
//!
//! So the pool is the *ledger* (atomic, shared, RAII) and the plan stays the
//! *description*. A `MemoryPlan` gains an optional handle to a pool
//! ([`crate::memory::plan::MemoryPlan::with_pool`]); when it has one, `check()` debits
//! the pool for the plan's projected peak and hands back a
//! [`crate::memory::pool::Reservation`] guard. When it has none, behaviour is exactly what it is
//! today — that is the trivial limit, and it is pinned by
//! `tests/mwe_pool_no_budget_is_a_noop.rs`.
//!
//! # The measured defect this fixes
//!
//! A 27-atom def2-SVP B3LYP RI-JK single point printed
//! `memory budget: 4.72 GiB [source: auto (0.8 × available RAM)]` and then
//! died at `MAXRSS 6.04 GiB` to a global (`CONSTRAINT_NONE`) OOM kill. Not one
//! gate had failed: the DF 3-index tensor asked "do I fit in 4.72 GiB?" (yes),
//! the grid AO cache asked "do I fit in 4.72 GiB?" (yes), and the process held
//! the sum. With a pool, whichever of those two asks second sees only what the
//! first left.
//!
//! # Cost
//!
//! One `fetch_update` on a single `AtomicUsize` per reservation, and one
//! `fetch_sub` on drop. Reservations happen per stage (and per batch at the
//! finest), not per loop iteration inside a GEMM, so this is not on a hot path
//! — but it is cheap enough to be. The per-plane name map is behind a `Mutex`
//! that is touched only on reserve/release, never on the fast path, and never
//! while the atomic is being updated.
//!
//! # Global vs threaded
//!
//! The pool is threaded explicitly wherever a config already carries a budget,
//! and is *additionally* installed in a process-global slot
//! ([`crate::memory::pool::install_global`]) by the CLI entry point. The global is what lets leaf
//! gates like `ferric_integrals::ao_grid::check_ao_grid_budget` — which take
//! no budget argument and are called from five crates — debit the same ledger
//! without threading a parameter through every intermediate signature. That is
//! a deliberate, bounded use of ambient state: the ambient thing is now a
//! *debited* ledger rather than a re-readable ceiling, so reading it twice
//! cannot hand out the same bytes twice.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};

use crate::FerricError;

/// A process-global pool of budgeted bytes that reservations debit and
/// release.
///
/// Cheap to clone (`Arc` inside). Every clone refers to the same ledger.
#[derive(Debug, Clone)]
pub struct MemoryPool {
    inner: Arc<PoolInner>,
}

#[derive(Debug)]
struct PoolInner {
    /// The ceiling. Never changes for the life of the pool.
    capacity: usize,
    /// Bytes currently outstanding across all live reservations.
    outstanding: AtomicUsize,
    /// High-water mark of `outstanding`, for the end-of-run report.
    peak: AtomicUsize,
    /// label -> currently outstanding bytes under that label. Only touched on
    /// reserve/release, never on a compute path.
    by_label: Mutex<HashMap<String, usize>>,
}

/// An RAII handle to bytes debited from a [`crate::memory::pool::MemoryPool`].
///
/// Dropping it returns the bytes to the pool. This is the property a plain
/// ceiling check cannot have: a transient buffer inside an SCF iteration
/// releases when it goes out of scope, so an 80-iteration loop does not
/// exhaust the pool.
///
/// A reservation made against a pool-less (unbudgeted) context is *inert*: it
/// holds no pool, debits nothing, and releases nothing. That is the trivial
/// limit — see [`Reservation::is_inert`].
#[derive(Debug)]
pub struct Reservation {
    pool: Option<MemoryPool>,
    label: String,
    bytes: usize,
}

impl MemoryPool {
    /// A pool with an explicit capacity in bytes.
    pub fn with_capacity_bytes(capacity: usize) -> Self {
        Self {
            inner: Arc::new(PoolInner {
                capacity,
                outstanding: AtomicUsize::new(0),
                peak: AtomicUsize::new(0),
                by_label: Mutex::new(HashMap::new()),
            }),
        }
    }

    /// A pool whose capacity comes from the standard resolution chain
    /// (explicit config → `FERRIC_MEM_BUDGET_GB` → legacy env → cgroup/RAM
    /// auto-detect → 2 GiB fallback). See [`super::resolve_budget`].
    ///
    /// Call this **once**, at a process entry point, and
    /// [`crate::memory::pool::install_global`] it.
    pub fn resolve(explicit: Option<usize>) -> Self {
        Self::with_capacity_bytes(super::resolve_budget_bytes(explicit))
    }

    /// The ceiling, in bytes.
    pub fn capacity_bytes(&self) -> usize {
        self.inner.capacity
    }

    /// Bytes currently held by live reservations.
    pub fn outstanding_bytes(&self) -> usize {
        self.inner.outstanding.load(Ordering::Acquire)
    }

    /// High-water mark of [`outstanding_bytes`](Self::outstanding_bytes) over
    /// the life of the pool.
    pub fn peak_bytes(&self) -> usize {
        self.inner.peak.load(Ordering::Acquire)
    }

    /// Capacity minus what is outstanding, floored at zero.
    ///
    /// This is the number a gate should compare against — NOT
    /// [`capacity_bytes`](Self::capacity_bytes), which is the whole point of
    /// the pool.
    pub fn available_bytes(&self) -> usize {
        self.inner
            .capacity
            .saturating_sub(self.inner.outstanding.load(Ordering::Acquire))
    }

    /// Debit `bytes` under `label`, returning an RAII guard that credits them
    /// back on drop.
    ///
    /// Fails with a typed error naming the plane, the shortfall, and the
    /// current occupancy breakdown when the pool cannot cover the request.
    /// Never panics, never truncates silently.
    ///
    /// Thread-safe: concurrent callers contend on one `fetch_update`, so two
    /// workers each asking for 3 GiB of a 4.72 GiB pool cannot both succeed.
    pub fn reserve(&self, label: &str, bytes: usize) -> Result<Reservation, FerricError> {
        let cap = self.inner.capacity;
        // REFUSE when the request would exceed the ceiling. `Some(next)`
        // unconditionally (the shape this had while it was being written)
        // makes `fetch_update` always succeed, which makes the `Err` arm below
        // dead code and the pool purely advisory -- the exact defect this type
        // exists to fix. The CAS loop is what makes the check atomic against
        // concurrent reservations from rayon workers: two threads that each
        // fit individually cannot both commit past the ceiling.
        let outcome =
            self.inner
                .outstanding
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |cur| {
                    let next = cur.saturating_add(bytes);
                    (next <= cap).then_some(next)
                });
        match outcome {
            Ok(prev) => {
                let now = prev.saturating_add(bytes);
                self.inner.peak.fetch_max(now, Ordering::AcqRel);
                if let Ok(mut m) = self.inner.by_label.lock() {
                    *m.entry(label.to_string()).or_insert(0) += bytes;
                }
                Ok(Reservation {
                    pool: Some(self.clone()),
                    label: label.to_string(),
                    bytes,
                })
            }
            Err(cur) => Err(FerricError::General(format!(
                "memory pool exhausted: \"{label}\" needs {:.3} GB but only {:.3} GB of the \
                 {:.3} GB pool is free (short by {:.3} GB) — raise [memory] budget_gb / \
                 FERRIC_MEM_BUDGET_GB, or shrink the system\n{}",
                bytes as f64 / 1e9,
                cap.saturating_sub(cur) as f64 / 1e9,
                cap as f64 / 1e9,
                bytes.saturating_sub(cap.saturating_sub(cur)) as f64 / 1e9,
                self.occupancy_report(),
            ))),
        }
    }

    /// [`reserve`](Self::reserve) that returns `None` instead of an error when
    /// the request does not fit.
    ///
    /// For gates that have a legitimate fallback (batch the grid, spill the
    /// tensor to disk) rather than a hard failure.
    pub fn try_reserve(&self, label: &str, bytes: usize) -> Option<Reservation> {
        self.reserve(label, bytes).ok()
    }

    /// The largest `k` for which `k × per_unit_bytes` fits in what is
    /// currently available. At least 1 — a zero width cannot make progress,
    /// and a caller that genuinely does not fit should be failing
    /// [`reserve`](Self::reserve), not looping forever on empty batches.
    pub fn fit_units(&self, per_unit_bytes: usize) -> usize {
        if per_unit_bytes == 0 {
            return 1;
        }
        (self.available_bytes() / per_unit_bytes).max(1)
    }

    /// Who is holding what, largest first.
    pub fn occupancy_report(&self) -> String {
        let mut s = format!(
            "  memory pool: {:.3} GB outstanding of {:.3} GB (peak {:.3} GB)\n",
            self.outstanding_bytes() as f64 / 1e9,
            self.inner.capacity as f64 / 1e9,
            self.peak_bytes() as f64 / 1e9,
        );
        let Ok(m) = self.inner.by_label.lock() else {
            s.push_str("    <occupancy map poisoned>\n");
            return s;
        };
        let mut rows: Vec<(&String, &usize)> = m.iter().filter(|(_, &b)| b > 0).collect();
        rows.sort_by_key(|(l, b)| (std::cmp::Reverse(**b), (*l).clone()));
        for (label, bytes) in rows {
            s.push_str(&format!(
                "    {:>10.3} GB  {}\n",
                *bytes as f64 / 1e9,
                label
            ));
        }
        s
    }

    fn release(&self, label: &str, bytes: usize) {
        // saturating_sub, not wrapping: a double-release (which the type
        // system makes hard but not impossible across a mem::forget) must not
        // wrap the ledger to a near-infinite occupancy that refuses everything.
        let _ = self
            .inner
            .outstanding
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |cur| {
                Some(cur.saturating_sub(bytes))
            });
        if let Ok(mut m) = self.inner.by_label.lock() {
            if let Some(e) = m.get_mut(label) {
                *e = e.saturating_sub(bytes);
                if *e == 0 {
                    m.remove(label);
                }
            }
        }
    }
}

impl Reservation {
    /// A reservation that holds nothing, against no pool. Dropping it does
    /// nothing. This is what every gate produces when no pool is installed,
    /// and it is what makes the unbudgeted path bit-identical to today.
    pub fn inert(label: impl Into<String>) -> Self {
        Self {
            pool: None,
            label: label.into(),
            bytes: 0,
        }
    }

    /// True when this reservation is not backed by a pool.
    pub fn is_inert(&self) -> bool {
        self.pool.is_none()
    }

    /// Bytes debited (0 when inert).
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// The plane name this was reserved under.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Give the bytes back now rather than at end of scope. Idempotent.
    pub fn release_now(&mut self) {
        if let Some(pool) = self.pool.take() {
            pool.release(&self.label, self.bytes);
            self.bytes = 0;
        }
    }
}

impl Drop for Reservation {
    /// Credit the reservation back to THE POOL IT CAME FROM.
    ///
    /// `self.pool` is a clone holding its own `Arc<PoolInner>`, so this
    /// releases into the originating pool even if the process-global slot has
    /// since been cleared or replaced. That is the semantics
    /// `an_installed_pool_does_not_leak_capacity_across_clears` asserts: a
    /// guard must never credit an unrelated pool.
    ///
    /// An `inert` reservation (no pool installed) has `pool: None` and is a
    /// no-op here, which is what keeps the no-budget path bit-identical.
    ///
    /// This body was EMPTY while the type was being written, so nothing was
    /// ever credited back: `outstanding` only ever grew, and a long SCF loop
    /// would exhaust the pool on transient buffers that had already been
    /// freed. Measured by its own test -- dropping a 3 GB guard left
    /// `outstanding_bytes()` at 3_000_000_000 instead of 0.
    fn drop(&mut self) {
        let Some(pool) = self.pool.take() else {
            return;
        };
        let bytes = self.bytes;
        // saturating_sub, not wrapping: a double-release would otherwise wrap
        // to a huge outstanding value and lock the pool out permanently.
        let _ = pool
            .inner
            .outstanding
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |cur| {
                Some(cur.saturating_sub(bytes))
            });
        // Scoped so the MutexGuard is released before `pool` goes out of
        // scope at the end of this body.
        if let Ok(mut m) = pool.inner.by_label.lock() {
            if let Some(slot) = m.get_mut(&self.label) {
                *slot = slot.saturating_sub(bytes);
                if *slot == 0 {
                    m.remove(&self.label);
                }
            }
            drop(m);
        }
        // `peak` is deliberately NOT decremented: it is a high-water mark for
        // the end-of-run report, not a live gauge.
        drop(pool);
    }
}

/// The process-global pool slot.
///
/// `RwLock<Option<..>>` rather than `OnceLock<..>` so tests can install and
/// clear a pool; production installs once at the entry point and never
/// touches it again. The `OnceLock` wrapper makes first access lock-free-ish
/// and avoids a `lazy_static` dependency.
fn global_slot() -> &'static RwLock<Option<MemoryPool>> {
    static SLOT: OnceLock<RwLock<Option<MemoryPool>>> = OnceLock::new();
    SLOT.get_or_init(|| RwLock::new(None))
}

/// Install `pool` as the process-global pool, returning whatever was there.
///
/// Call this ONCE, at the entry point (CLI arm, Python binding, driver).
pub fn install_global(pool: MemoryPool) -> Option<MemoryPool> {
    match global_slot().write() {
        Ok(mut g) => g.replace(pool),
        Err(_) => None,
    }
}

/// Remove the process-global pool, returning it. After this, every gate is
/// back to its unbudgeted (inert) behaviour.
pub fn clear_global() -> Option<MemoryPool> {
    match global_slot().write() {
        Ok(mut g) => g.take(),
        Err(_) => None,
    }
}

/// The process-global pool, if one has been installed.
pub fn global() -> Option<MemoryPool> {
    global_slot().read().ok().and_then(|g| g.clone())
}

/// Debit the process-global pool if one is installed; otherwise return an
/// inert reservation.
///
/// This is the shape every migrated gate uses. With no pool installed it is a
/// pure no-op that cannot fail, which is the trivial-limit anchor.
pub fn reserve_global(label: &str, bytes: usize) -> Result<Reservation, FerricError> {
    match global() {
        Some(pool) => pool.reserve(label, bytes),
        None => Ok(Reservation::inert(label)),
    }
}

/// [`reserve_global`] with a fallback: `None` when a pool is installed and the
/// request does not fit; `Some(inert)` when no pool is installed.
///
/// For gates whose over-budget answer is "batch it", not "fail".
pub fn try_reserve_global(label: &str, bytes: usize) -> Option<Reservation> {
    match global() {
        Some(pool) => pool.try_reserve(label, bytes),
        None => Some(Reservation::inert(label)),
    }
}

/// What the global pool has left, or `None` when no pool is installed.
///
/// A gate that needs to *size* something (a batch width) rather than admit it
/// uses this; `None` means "unbudgeted, keep doing what you did before".
pub fn global_available_bytes() -> Option<usize> {
    global().map(|p| p.available_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: usize = 1_000_000_000;

    #[test]
    fn two_reservations_summing_past_the_pool_cannot_both_succeed() {
        // THE property the pre-pool code lacked: a 4.72 GiB pool, two
        // subsystems asking 3 GiB each. Both passed before; the second must
        // fail now.
        let pool = MemoryPool::with_capacity_bytes(4_720 * GB / 1000);
        let _a = pool.reserve("DF 3-index", 3 * GB).expect("first fits");
        let err = pool.reserve("KS grid AO cache", 3 * GB).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("KS grid AO cache"), "{msg}");
        assert!(msg.contains("short by"), "{msg}");
        // The breakdown names the incumbent, so the report says WHICH plane is
        // dominant.
        assert!(msg.contains("DF 3-index"), "{msg}");
    }

    #[test]
    fn a_reservation_releases_on_drop() {
        let pool = MemoryPool::with_capacity_bytes(4 * GB);
        {
            let _r = pool.reserve("transient", 3 * GB).unwrap();
            assert_eq!(pool.outstanding_bytes(), 3 * GB);
            assert!(pool.reserve("other", 3 * GB).is_err());
        }
        assert_eq!(pool.outstanding_bytes(), 0);
        // Same request now succeeds — a long SCF loop is not spuriously
        // exhausted.
        assert!(pool.reserve("other", 3 * GB).is_ok());
    }

    #[test]
    fn peak_records_the_high_water_mark_after_release() {
        let pool = MemoryPool::with_capacity_bytes(10 * GB);
        {
            let _a = pool.reserve("a", 3 * GB).unwrap();
            let _b = pool.reserve("b", 4 * GB).unwrap();
            assert_eq!(pool.outstanding_bytes(), 7 * GB);
        }
        assert_eq!(pool.outstanding_bytes(), 0);
        assert_eq!(pool.peak_bytes(), 7 * GB, "peak must survive the release");
    }

    #[test]
    fn concurrent_reservations_do_not_oversubscribe() {
        use std::sync::atomic::AtomicUsize;
        // 8 threads each asking for a fifth of the pool: at most 5 can win,
        // and the ledger must never exceed capacity.
        let pool = MemoryPool::with_capacity_bytes(10 * GB);
        let granted = Arc::new(AtomicUsize::new(0));
        let held: Mutex<Vec<Reservation>> = Mutex::new(Vec::new());
        rayon::scope(|s| {
            for i in 0..8 {
                let pool = pool.clone();
                let granted = granted.clone();
                let held = &held;
                s.spawn(move |_| {
                    if let Ok(r) = pool.reserve(&format!("w{i}"), 2 * GB) {
                        granted.fetch_add(1, Ordering::SeqCst);
                        held.lock().unwrap().push(r);
                    }
                });
            }
        });
        let n = granted.load(Ordering::SeqCst);
        assert_eq!(n, 5, "exactly capacity/request may be granted, got {n}");
        assert!(
            pool.outstanding_bytes() <= pool.capacity_bytes(),
            "ledger oversubscribed: {} > {}",
            pool.outstanding_bytes(),
            pool.capacity_bytes()
        );
        drop(held);
        assert_eq!(pool.outstanding_bytes(), 0);
    }

    #[test]
    fn an_inert_reservation_debits_nothing_and_cannot_fail() {
        let r = Reservation::inert("nothing");
        assert!(r.is_inert());
        assert_eq!(r.bytes(), 0);
        drop(r);
    }

    #[test]
    fn available_is_capacity_minus_outstanding_not_capacity() {
        let pool = MemoryPool::with_capacity_bytes(10 * GB);
        assert_eq!(pool.available_bytes(), 10 * GB);
        let _a = pool.reserve("a", 6 * GB).unwrap();
        assert_eq!(
            pool.available_bytes(),
            4 * GB,
            "available must be DEBITED, not re-read as the ceiling"
        );
    }

    #[test]
    fn fit_units_shrinks_as_the_pool_fills_and_never_returns_zero() {
        let pool = MemoryPool::with_capacity_bytes(10 * GB);
        assert_eq!(pool.fit_units(GB), 10);
        let _a = pool.reserve("a", 7 * GB).unwrap();
        assert_eq!(pool.fit_units(GB), 3);
        let _b = pool.reserve("b", 3 * GB).unwrap();
        assert_eq!(pool.available_bytes(), 0);
        assert_eq!(pool.fit_units(GB), 1, "a zero width cannot make progress");
    }

    #[test]
    fn release_now_is_idempotent() {
        let pool = MemoryPool::with_capacity_bytes(10 * GB);
        let mut r = pool.reserve("a", 5 * GB).unwrap();
        r.release_now();
        assert_eq!(pool.outstanding_bytes(), 0);
        r.release_now();
        assert_eq!(
            pool.outstanding_bytes(),
            0,
            "double release must not wrap the ledger"
        );
    }

    #[test]
    fn an_over_capacity_request_fails_rather_than_saturating() {
        let pool = MemoryPool::with_capacity_bytes(GB);
        let err = pool.reserve("absurd", usize::MAX).unwrap_err().to_string();
        assert!(err.contains("absurd"), "{err}");
        assert_eq!(
            pool.outstanding_bytes(),
            0,
            "a failed reserve debits nothing"
        );
    }

    #[test]
    fn try_reserve_returns_none_instead_of_erroring() {
        let pool = MemoryPool::with_capacity_bytes(GB);
        // Bind the guard: a temporary would drop (and credit) immediately,
        // making the second request fit and the test vacuous.
        let _held = pool.try_reserve("fits", GB / 2).expect("half fits");
        assert!(pool.try_reserve("does not", GB).is_none());
    }

    #[test]
    fn occupancy_report_sorts_the_dominant_plane_first() {
        let pool = MemoryPool::with_capacity_bytes(10 * GB);
        let _small = pool.reserve("small_plane", GB).unwrap();
        let _big = pool.reserve("dominant_plane", 5 * GB).unwrap();
        let rep = pool.occupancy_report();
        let big = rep.find("dominant_plane").unwrap();
        let small = rep.find("small_plane").unwrap();
        assert!(big < small, "dominant plane must sort first:\n{rep}");
    }

    #[test]
    fn same_label_twice_accumulates_and_unwinds() {
        let pool = MemoryPool::with_capacity_bytes(10 * GB);
        let a = pool.reserve("batch", GB).unwrap();
        let b = pool.reserve("batch", 2 * GB).unwrap();
        assert!(pool.occupancy_report().contains("3.000 GB  batch"));
        drop(b);
        assert_eq!(pool.outstanding_bytes(), GB);
        drop(a);
        assert_eq!(pool.outstanding_bytes(), 0);
        assert!(!pool.occupancy_report().contains("batch"));
    }
}
