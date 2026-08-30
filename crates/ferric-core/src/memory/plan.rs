//! [`MemoryPlan`]: a memory budget as a *value* you spend, not a place you read.
//!
//! # Why this exists
//!
//! [`resolve_budget_bytes`](super::resolve_budget_bytes) answers "what is the
//! ceiling?". It does that well. But it hands the *whole* ceiling to every
//! caller that asks, and on the tree at `0e876a75` there were 121 such callers
//! across 48 files, against 32 [`check_alloc`](super::check_alloc) gates. Three
//! consequences, all observed in this repo:
//!
//! 1. **The gates do not compose.** Two stages that each independently pass
//!    `check_alloc` still OOM when run back to back, because neither subtracts
//!    what the other left resident. Exactly one site in the tree accounts for
//!    prior residency (`ferric_rpa::energy::quad_panel_width`, which does
//!    `budget.saturating_sub(y_bytes)`); everywhere else the
//!    [`Share`](super::Share) enum stands in with an author-time guess
//!    (`budget/4`, `budget/2`). A constant fraction cannot track what is
//!    actually resident at run time.
//!
//! 2. **Estimators drift from allocators.** Every method carries a hand-written
//!    second implementation of its own allocation shapes —
//!    `ferric_rpa::budget::estimate_peak_bytes` says so in its own docs
//!    ("Mirrors `properties::accumulate_atom_centred_dipoles`"). Kept in sync by
//!    hand, they drift by construction, and the drift is silent until it is an
//!    OOM: the LNO-coupled path predicted 0.055 GB and peaked at 7.3 GB, and the
//!    2026-07-13 RPA incident was an estimator missing a grid term that "would
//!    have passed a 16-17 GB job as fitting a 10 GB budget".
//!
//! 3. **The budget is ambient mutable state.** Because it is read from env vars
//!    at arbitrary call depth, tests that want to observe it must `set_var` and
//!    hold a serialization mutex (see `ferric_scf::rhf` and `ferric_dft::ks`).
//!
//! # The shape of the fix
//!
//! A plan is inert data describing the allocations a method *will* make. You
//! declare shapes up front, ask the plan whether they fit, and then allocate
//! **through** the plan:
//!
//! ```
//! use ferric_core::memory::plan::{MemoryPlan, Lifetime};
//!
//! let mut plan = MemoryPlan::with_budget_bytes(64 * 1024 * 1024, "RI-MP2");
//! plan.reserve("B(P|ia)", 100 * 20 * 200, Lifetime::Resident);
//! plan.reserve_per_worker("t2 scratch", 20 * 200, 4);
//! plan.check()?;                                   // fail fast, with a breakdown
//! let b = plan.alloc2("B(P|ia)", (100, 20 * 200))?; // shape-checked against the reservation
//! # assert_eq!(b.shape(), &[100, 4000]);
//! # Ok::<(), ferric_core::FerricError>(())
//! ```
//!
//! [`MemoryPlan::alloc2`] is the part that kills defect (2): an array can only
//! be allocated under a label that was declared, and its element count is
//! checked against the declaration. A forgotten term becomes a loud failure at
//! the allocation site instead of a silent underestimate found later by the OOM
//! killer. The estimate and the allocation are the same expression, so they
//! cannot disagree.
//!
//! [`MemoryPlan::remaining`] and [`MemoryPlan::sub_plan`] address defect (1):
//! what is left is a real subtraction over what has been declared, not a
//! fraction chosen when the code was written.
//!
//! # Cost
//!
//! Reservations happen once per stage, not once per loop iteration, so the
//! bookkeeping is a handful of integer adds on a path that is about to touch
//! hundreds of megabytes. [`MemoryPlan::alloc2`] allocates exactly what
//! `Array2::zeros` would; the plan adds a `HashMap` lookup.

use std::collections::HashMap;

use ndarray::{Array1, Array2, Array3};

use crate::FerricError;

/// Bytes per `f64` — the element type of every large ferric tensor.
pub const F64_BYTES: usize = 8;

/// How long an allocation stays live, which determines how it contributes to
/// the peak.
///
/// The distinction matters because summing everything as if it were resident
/// over-estimates badly on streaming paths (and an over-estimating guard is
/// also a bug — it refuses jobs that would have fit), while summing everything
/// as if it were transient under-estimates and OOMs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lifetime {
    /// Live for the whole method. Sums into the peak unconditionally.
    ///
    /// The 3-index tensor, MO coefficients, converged amplitudes.
    Resident,
    /// Live only within one stage, then dropped before the next stage
    /// allocates. Only the *largest* transient counts toward the peak, since
    /// they do not coexist.
    ///
    /// Per-stage scratch: a transform buffer, a panel, a chunk.
    Transient,
    /// Held concurrently by every rayon worker. Contributes
    /// `workers × bytes` — this is the multiplier the 2026-07-13 RPA incident
    /// missed, and the shape behind the LNO-coupled breach.
    PerWorker,
}

/// One declared allocation: a label, a shape, and how it counts toward peak.
#[derive(Debug, Clone)]
pub struct Reservation {
    /// Human-readable identity, e.g. `"B(P|ia)"`. Also the key used by
    /// [`MemoryPlan::alloc2`] and friends.
    pub label: String,
    /// Element count (not bytes).
    pub elems: usize,
    /// Bytes per element. [`F64_BYTES`] for the usual dense tensors.
    pub elem_bytes: usize,
    /// How this contributes to the peak.
    pub lifetime: Lifetime,
    /// Concurrent holders, for [`Lifetime::PerWorker`]. Always 1 otherwise.
    pub workers: usize,
}

impl Reservation {
    /// Total bytes this reservation contributes, including the worker
    /// multiplier. Saturating: a nonsense shape reports `usize::MAX` rather
    /// than wrapping to a small number that would pass a check.
    pub fn bytes(&self) -> usize {
        self.elems.saturating_mul(self.elem_bytes).saturating_mul(self.workers)
    }
}

/// A memory budget together with everything declared against it.
///
/// Construct one per method entry point (or derive a child with
/// [`sub_plan`](MemoryPlan::sub_plan)), declare the large allocations, call
/// [`check`](MemoryPlan::check) once, then allocate through it.
#[derive(Debug, Clone)]
pub struct MemoryPlan {
    budget_bytes: usize,
    label: String,
    reservations: Vec<Reservation>,
    index: HashMap<String, usize>,
}

impl MemoryPlan {
    /// A plan against an explicit byte ceiling.
    ///
    /// `label` names the method for diagnostics, e.g. `"RI-MP2"`.
    pub fn with_budget_bytes(budget_bytes: usize, label: impl Into<String>) -> Self {
        Self {
            budget_bytes,
            label: label.into(),
            reservations: Vec::new(),
            index: HashMap::new(),
        }
    }

    /// A plan whose ceiling comes from the standard resolution chain
    /// (explicit config → `FERRIC_MEM_BUDGET_GB` → legacy env → cgroup/RAM
    /// auto-detect → default). See [`resolve_budget`](super::resolve_budget).
    ///
    /// Call this **once**, at an entry point (CLI arm, Python binding, driver),
    /// and thread the resulting plan down. Resolving deep in the call stack is
    /// the ambient-state defect this type exists to retire.
    pub fn resolve(explicit: Option<usize>, label: impl Into<String>) -> Self {
        Self::with_budget_bytes(super::resolve_budget_bytes(explicit), label)
    }

    /// The ceiling, in bytes.
    pub fn budget_bytes(&self) -> usize {
        self.budget_bytes
    }

    /// The method label.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Everything declared so far.
    pub fn reservations(&self) -> &[Reservation] {
        &self.reservations
    }

    /// Declare an allocation of `elems` `f64`s.
    ///
    /// Re-declaring an existing label **replaces** it rather than
    /// double-counting, so a routine that refines an estimate as shapes become
    /// known (e.g. after truncation determines the retained rank) stays
    /// correct.
    pub fn reserve(&mut self, label: &str, elems: usize, lifetime: Lifetime) -> &mut Self {
        self.reserve_sized(label, elems, F64_BYTES, lifetime, 1)
    }

    /// Declare a per-worker allocation of `elems` `f64`s held concurrently by
    /// `workers` rayon threads. Contributes `workers × elems × 8` bytes.
    ///
    /// Pass `rayon::current_num_threads().max(1)` unless the fan-out is
    /// explicitly bounded to something smaller.
    pub fn reserve_per_worker(&mut self, label: &str, elems: usize, workers: usize) -> &mut Self {
        self.reserve_sized(label, elems, F64_BYTES, Lifetime::PerWorker, workers.max(1))
    }

    /// Declare an allocation with a non-`f64` element size (e.g. `usize`
    /// index lists, `Complex64` frequency data).
    pub fn reserve_sized(
        &mut self,
        label: &str,
        elems: usize,
        elem_bytes: usize,
        lifetime: Lifetime,
        workers: usize,
    ) -> &mut Self {
        let r = Reservation {
            label: label.to_string(),
            elems,
            elem_bytes,
            lifetime,
            workers: workers.max(1),
        };
        match self.index.get(label) {
            Some(&i) => self.reservations[i] = r,
            None => {
                self.index.insert(label.to_string(), self.reservations.len());
                self.reservations.push(r);
            }
        }
        self
    }

    /// Projected peak bytes: every [`Resident`](Lifetime::Resident) and
    /// [`PerWorker`](Lifetime::PerWorker) reservation, plus the single largest
    /// [`Transient`](Lifetime::Transient) one (transients do not coexist).
    pub fn peak_bytes(&self) -> usize {
        let mut fixed: usize = 0;
        let mut largest_transient: usize = 0;
        for r in &self.reservations {
            match r.lifetime {
                Lifetime::Resident | Lifetime::PerWorker => {
                    fixed = fixed.saturating_add(r.bytes())
                }
                Lifetime::Transient => largest_transient = largest_transient.max(r.bytes()),
            }
        }
        fixed.saturating_add(largest_transient)
    }

    /// Budget minus projected peak, floored at zero: what a further allocation
    /// may still claim.
    ///
    /// This replaces the [`Share`](super::Share) fractions. Instead of guessing
    /// that scratch may have a quarter of the budget, declare what is resident
    /// and ask what is left.
    pub fn remaining(&self) -> usize {
        self.budget_bytes.saturating_sub(self.peak_bytes())
    }

    /// The largest `k` for which `k × per_unit_elems × 8 × workers` still fits
    /// in [`remaining`](MemoryPlan::remaining).
    ///
    /// This is the one calculation every panel/chunk/band-width helper in the
    /// tree open-codes (`quad_panel_width`, `resolve_band_bytes`,
    /// `triple_chunk_len`, `dipole_band_width`, `MO_STREAM_CHUNK` sizing).
    /// Returns at least 1: a width of zero cannot make progress, and a caller
    /// that genuinely does not fit should be failing
    /// [`check`](MemoryPlan::check), not silently looping forever on
    /// zero-width panels.
    pub fn fit_width(&self, per_unit_elems: usize, workers: usize) -> usize {
        let per_unit = per_unit_elems.saturating_mul(F64_BYTES).saturating_mul(workers.max(1));
        if per_unit == 0 {
            return 1;
        }
        (self.remaining() / per_unit).max(1)
    }

    /// Fail fast if the declared peak exceeds the budget, with a
    /// per-reservation breakdown.
    ///
    /// The breakdown is the point: a bare "needs 41 GB, have 10 GB" tells you
    /// nothing about *which* term blew up, which is why the historical
    /// incidents took so long to diagnose.
    pub fn check(&self) -> Result<(), FerricError> {
        let peak = self.peak_bytes();
        if peak <= self.budget_bytes {
            return Ok(());
        }
        Err(FerricError::General(format!(
            "{} requires {:.2} GB; budget is {:.2} GB — raise [memory] budget_gb / \
             FERRIC_MEM_BUDGET_GB or shrink the system\n{}",
            self.label,
            peak as f64 / 1e9,
            self.budget_bytes as f64 / 1e9,
            self.report(),
        )))
    }

    /// A human-readable breakdown, largest contributor first.
    pub fn report(&self) -> String {
        let mut rows: Vec<&Reservation> = self.reservations.iter().collect();
        rows.sort_by_key(|r| std::cmp::Reverse(r.bytes()));
        let mut s = format!(
            "  memory plan [{}]: peak {:.3} GB of {:.3} GB budget\n",
            self.label,
            self.peak_bytes() as f64 / 1e9,
            self.budget_bytes as f64 / 1e9,
        );
        for r in rows {
            let kind = match r.lifetime {
                Lifetime::Resident => "resident".to_string(),
                Lifetime::Transient => "transient".to_string(),
                Lifetime::PerWorker => format!("per-worker x{}", r.workers),
            };
            s.push_str(&format!(
                "    {:>10.3} GB  {:<12}  {}\n",
                r.bytes() as f64 / 1e9,
                kind,
                r.label,
            ));
        }
        s
    }

    /// The `source -> value` audit line, for logging the budget once at the
    /// entry point.
    pub fn audit_line(&self) -> String {
        format!(
            "memory budget: {:.2} GiB  [plan: {}]",
            self.budget_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            self.label,
        )
    }

    /// A child plan limited to what this plan has left, for a sub-computation
    /// whose own allocations should not be able to exceed the parent's
    /// remaining headroom.
    ///
    /// This is how nested methods compose (SCF inside a gradient, RI-MP2 inside
    /// a double hybrid) without each one assuming it owns the whole box.
    pub fn sub_plan(&self, label: impl Into<String>) -> Self {
        Self::with_budget_bytes(self.remaining(), label)
    }

    /// Look up a declared reservation and verify `elems` matches it.
    ///
    /// This is the drift guard: allocating a shape that was never declared, or
    /// a different shape than was declared, is an error rather than a silent
    /// discrepancy between the estimate and reality.
    fn checked(&self, label: &str, elems: usize) -> Result<(), FerricError> {
        let Some(&i) = self.index.get(label) else {
            return Err(FerricError::General(format!(
                "memory plan [{}]: allocation \"{label}\" was never reserved — declare it with \
                 reserve() before allocating, so the pre-flight estimate accounts for it",
                self.label,
            )));
        };
        let r = &self.reservations[i];
        if r.elems != elems {
            return Err(FerricError::General(format!(
                "memory plan [{}]: allocation \"{label}\" reserved {} elements but allocated {} \
                 — the estimate has drifted from the allocation; fix the reserve() call",
                self.label, r.elems, elems,
            )));
        }
        Ok(())
    }

    /// Allocate a declared 1-D array, shape-checked against its reservation.
    pub fn alloc1(&self, label: &str, n: usize) -> Result<Array1<f64>, FerricError> {
        self.checked(label, n)?;
        Ok(Array1::zeros(n))
    }

    /// Allocate a declared 2-D array, shape-checked against its reservation.
    pub fn alloc2(&self, label: &str, dim: (usize, usize)) -> Result<Array2<f64>, FerricError> {
        self.checked(label, dim.0.saturating_mul(dim.1))?;
        Ok(Array2::zeros(dim))
    }

    /// Allocate a declared 3-D array, shape-checked against its reservation.
    pub fn alloc3(
        &self,
        label: &str,
        dim: (usize, usize, usize),
    ) -> Result<Array3<f64>, FerricError> {
        self.checked(label, dim.0.saturating_mul(dim.1).saturating_mul(dim.2))?;
        Ok(Array3::zeros(dim))
    }

    /// Allocate a declared `Vec<f64>`, length-checked against its reservation.
    pub fn alloc_vec(&self, label: &str, n: usize) -> Result<Vec<f64>, FerricError> {
        self.checked(label, n)?;
        Ok(vec![0.0; n])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: usize = 1_000_000_000;

    #[test]
    fn peak_sums_resident_and_takes_largest_transient() {
        let mut p = MemoryPlan::with_budget_bytes(10 * GB, "t");
        p.reserve("resident_a", 1_000_000, Lifetime::Resident); // 8 MB
        p.reserve("resident_b", 2_000_000, Lifetime::Resident); // 16 MB
        p.reserve("scratch_small", 1_000_000, Lifetime::Transient); // 8 MB
        p.reserve("scratch_big", 5_000_000, Lifetime::Transient); // 40 MB
        // 8 + 16 resident, plus only the LARGEST transient (40), not both.
        assert_eq!(p.peak_bytes(), (1_000_000 + 2_000_000 + 5_000_000) * F64_BYTES);
    }

    #[test]
    fn per_worker_multiplies_by_worker_count() {
        // The 2026-07-13 RPA incident shape: scratch that is budget-blind
        // until you multiply by the fan-out.
        let mut one = MemoryPlan::with_budget_bytes(10 * GB, "t");
        one.reserve_per_worker("scratch", 1_000_000, 1);
        let mut twelve = MemoryPlan::with_budget_bytes(10 * GB, "t");
        twelve.reserve_per_worker("scratch", 1_000_000, 12);
        assert_eq!(twelve.peak_bytes(), 12 * one.peak_bytes());
    }

    #[test]
    fn check_fails_over_budget_and_names_the_largest_term() {
        let mut p = MemoryPlan::with_budget_bytes(GB, "RI-MP2");
        p.reserve("small", 1_000, Lifetime::Resident);
        p.reserve("the_huge_one", 500_000_000, Lifetime::Resident); // 4 GB
        let err = p.check().unwrap_err().to_string();
        assert!(err.contains("RI-MP2"), "{err}");
        assert!(err.contains("the_huge_one"), "breakdown must name the term: {err}");
        // Largest first, so the culprit is the first row of the breakdown.
        let huge = err.find("the_huge_one").unwrap();
        let small = err.find("small").unwrap();
        assert!(huge < small, "largest contributor must sort first: {err}");
    }

    #[test]
    fn check_passes_when_it_fits() {
        let mut p = MemoryPlan::with_budget_bytes(GB, "t");
        p.reserve("a", 1_000_000, Lifetime::Resident); // 8 MB
        assert!(p.check().is_ok());
    }

    #[test]
    fn remaining_subtracts_what_is_declared() {
        let mut p = MemoryPlan::with_budget_bytes(GB, "t");
        assert_eq!(p.remaining(), GB);
        p.reserve("a", 100_000_000 / F64_BYTES, Lifetime::Resident); // 100 MB
        assert_eq!(p.remaining(), GB - 100_000_000);
    }

    #[test]
    fn remaining_floors_at_zero_rather_than_wrapping() {
        let mut p = MemoryPlan::with_budget_bytes(GB, "t");
        p.reserve("a", 10 * GB, Lifetime::Resident);
        assert_eq!(p.remaining(), 0, "must not wrap around to a huge budget");
    }

    #[test]
    fn fit_width_solves_the_panel_width() {
        let mut p = MemoryPlan::with_budget_bytes(GB, "t");
        p.reserve("resident", 100_000_000 / F64_BYTES, Lifetime::Resident); // 100 MB
        // 900 MB left; each unit is 1000 f64 = 8000 bytes, 1 worker.
        assert_eq!(p.fit_width(1000, 1), 900_000_000 / 8000);
        // Doubling the workers halves the width.
        assert_eq!(p.fit_width(1000, 2), 900_000_000 / 16000);
    }

    #[test]
    fn fit_width_never_returns_zero() {
        let mut p = MemoryPlan::with_budget_bytes(1000, "t");
        p.reserve("hog", 10_000, Lifetime::Resident);
        assert_eq!(p.remaining(), 0);
        assert_eq!(p.fit_width(1_000_000, 8), 1, "a zero width cannot make progress");
    }

    #[test]
    fn alloc_rejects_an_undeclared_label() {
        let p = MemoryPlan::with_budget_bytes(GB, "t");
        let err = p.alloc2("never_reserved", (10, 10)).unwrap_err().to_string();
        assert!(err.contains("never reserved"), "{err}");
    }

    #[test]
    fn alloc_rejects_a_shape_that_drifted_from_the_reservation() {
        // The LNO-coupled / RPA-incident defect class, made impossible: the
        // code allocates more than it declared, and says so at the site.
        let mut p = MemoryPlan::with_budget_bytes(GB, "t");
        p.reserve("b_tensor", 100, Lifetime::Resident);
        let err = p.alloc2("b_tensor", (100, 100)).unwrap_err().to_string();
        assert!(err.contains("drifted"), "{err}");
        assert!(err.contains("reserved 100 elements but allocated 10000"), "{err}");
    }

    #[test]
    fn alloc_succeeds_and_zeroes_when_the_shape_matches() {
        let mut p = MemoryPlan::with_budget_bytes(GB, "t");
        p.reserve("m", 12, Lifetime::Resident);
        let a = p.alloc2("m", (3, 4)).unwrap();
        assert_eq!(a.shape(), &[3, 4]);
        assert!(a.iter().all(|&x| x == 0.0));

        p.reserve("v", 5, Lifetime::Resident);
        assert_eq!(p.alloc1("v", 5).unwrap().len(), 5);
        assert_eq!(p.alloc_vec("v", 5).unwrap().len(), 5);

        p.reserve("t3", 24, Lifetime::Resident);
        assert_eq!(p.alloc3("t3", (2, 3, 4)).unwrap().shape(), &[2, 3, 4]);
    }

    #[test]
    fn re_reserving_replaces_rather_than_double_counting() {
        // A routine that refines an estimate once truncation is known must not
        // be punished for it.
        let mut p = MemoryPlan::with_budget_bytes(GB, "t");
        p.reserve("x", 1_000_000, Lifetime::Resident);
        p.reserve("x", 2_000_000, Lifetime::Resident);
        assert_eq!(p.peak_bytes(), 2_000_000 * F64_BYTES);
        assert_eq!(p.reservations().len(), 1);
    }

    #[test]
    fn sub_plan_inherits_only_the_remaining_headroom() {
        let mut p = MemoryPlan::with_budget_bytes(GB, "outer");
        p.reserve("resident", 100_000_000 / F64_BYTES, Lifetime::Resident); // 100 MB
        let child = p.sub_plan("inner");
        assert_eq!(child.budget_bytes(), GB - 100_000_000);
        assert_eq!(child.label(), "inner");
    }

    #[test]
    fn sub_plan_check_fails_for_a_child_that_would_breach_the_parent() {
        // Two stages that each "fit" in isolation must not both pass.
        let mut p = MemoryPlan::with_budget_bytes(GB, "outer");
        p.reserve("resident", 900_000_000 / F64_BYTES, Lifetime::Resident); // 900 MB
        let mut child = p.sub_plan("inner");
        child.reserve("scratch", 500_000_000 / F64_BYTES, Lifetime::Resident); // 500 MB
        assert!(
            child.check().is_err(),
            "500 MB must not fit in the 100 MB the parent left"
        );
    }

    #[test]
    fn bytes_saturate_instead_of_wrapping_on_absurd_shapes() {
        let mut p = MemoryPlan::with_budget_bytes(GB, "t");
        p.reserve("absurd", usize::MAX / 2, Lifetime::Resident);
        assert_eq!(p.peak_bytes(), usize::MAX, "must saturate, not wrap to something small");
        assert!(p.check().is_err());
    }

    #[test]
    fn reserve_sized_honours_a_non_f64_element_size() {
        let mut p = MemoryPlan::with_budget_bytes(GB, "t");
        p.reserve_sized("idx", 1_000, 4, Lifetime::Resident, 1);
        assert_eq!(p.peak_bytes(), 4_000);
    }

    #[test]
    fn zero_workers_is_treated_as_one() {
        let mut p = MemoryPlan::with_budget_bytes(GB, "t");
        p.reserve_per_worker("s", 1_000, 0);
        assert_eq!(p.peak_bytes(), 1_000 * F64_BYTES);
    }
}
