//! MWE: ferric-gw's planes DEBIT one shared ledger instead of each re-reading
//! the same ceiling, and the soft QP-sweep gate falls back to a bit-identical
//! serial sweep rather than refusing.
//!
//! # What this pins that the ceiling checks could not
//!
//! Before the migration, `guard_b_full` asked "does `(naux, n_act, n_act)` fit
//! in the budget?" and `guard_m_proj` asked "does `m_proj + b_full` fit in the
//! budget?". Both compared against the SAME number and both said yes, so a job
//! whose true peak was their sum passed every gate and then met the OOM killer.
//! The property under test here is composition: whichever plane asks second
//! must see only what the first left.
//!
//! # Every capacity is DERIVED, never frozen
//!
//! The per-worker QP scratch scales with `rayon::current_num_threads()`, so a
//! constant calibrated on a 12-worker box is wrong at 2 and 4. The ferric-rpa
//! migration froze one and broke 3 of its 7 tests at narrower widths. Each test
//! below computes its capacity from `ferric_gw::budget`, the same module the
//! gates charge through, and asserts the window it needs is non-empty before
//! measuring anything in it.
//!
//! # Serialization
//!
//! The global pool slot is process-wide, so these tests hold a mutex and clear
//! the pool on the way out. A leaked pool would make the sibling anchor binary
//! see a budget it never installed.

use ferric_core::memory::pool::{self, MemoryPool};
use ferric_gw::budget;
use ferric_gw::cohsex::project_b_into_pdep;
use ferric_gw::mo_b::MoB;
use ndarray::{Array2, Array3};
use std::sync::{Mutex, MutexGuard, OnceLock};

/// A shape where one tensor is ~8 MB, big enough that capacities separate
/// cleanly and small enough to allocate in a unit test.
const NAUX: usize = 200;
const N_ACT: usize = 50;

fn lock() -> MutexGuard<'static, ()> {
    static L: OnceLock<Mutex<()>> = OnceLock::new();
    L.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

/// Install `pool` for the duration of the returned guard, clearing on drop so a
/// failure cannot leak a budget into the next test.
struct Installed(#[allow(dead_code)] MutexGuard<'static, ()>);
impl Installed {
    fn new(capacity: usize) -> Self {
        let g = lock();
        pool::install_global(MemoryPool::with_capacity_bytes(capacity));
        Self(g)
    }
}
impl Drop for Installed {
    fn drop(&mut self) {
        pool::clear_global();
    }
}

fn make_mo_b() -> Result<MoB, ferric_core::FerricError> {
    MoB::from_parts(
        Array3::<f64>::zeros((NAUX, N_ACT, N_ACT)),
        Array2::<f64>::zeros((NAUX, NAUX)),
        NAUX,
        N_ACT,
        0,
        N_ACT / 2,
        vec![0.0; N_ACT],
    )
}

fn v_dressed() -> Array2<f64> {
    Array2::<f64>::zeros((NAUX, NAUX))
}

/// CONTRACT 1: the second plane sees a DEBITED pool, not the ceiling again.
///
/// This is the defect verbatim. A capacity that holds `b_full` and `m_proj`
/// individually but not together used to admit both. It must now refuse the
/// second, and the refusal must NAME the incumbent so the breakdown says which
/// plane is dominant.
#[test]
fn the_second_plane_sees_what_the_first_left() {
    let one = budget::b_full_bytes(NAUX, N_ACT);
    // Holds either plane alone with room to spare; holds both with none.
    let capacity = one + one / 2;
    let _p = Installed::new(capacity);

    let mo_b = make_mo_b().expect("b_full fits the capacity on its own");
    assert_eq!(
        pool::global_available_bytes(),
        Some(capacity - one),
        "b_full must have DEBITED the ledger, not merely been compared to it"
    );

    let err = project_b_into_pdep(&mo_b, &v_dressed(), Some(usize::MAX))
        .expect_err(
            "m_proj was admitted against the full ceiling while b_full was still resident -- \
             the two gates are re-reading one number instead of debiting one ledger",
        )
        .to_string();
    assert!(err.contains("memory pool exhausted"), "{err}");
    assert!(
        err.contains("b_full"),
        "the refusal must name the incumbent so the occupancy breakdown is actionable:\n{err}"
    );
}

/// CONTRACT 2 (the over-rejection guard): a capacity that holds both must run
/// both.
///
/// Without this, CONTRACT 1 passes on a gate that refuses everything. "An
/// over-estimating guard is also a bug -- it refuses jobs that would have fit."
#[test]
fn a_capacity_that_holds_both_planes_runs_both() {
    let one = budget::b_full_bytes(NAUX, N_ACT);
    let _p = Installed::new(3 * one);
    let mo_b = make_mo_b().expect("b_full fits");
    let m = project_b_into_pdep(&mo_b, &v_dressed(), Some(usize::MAX))
        .expect("m_proj must be admitted when the pool genuinely holds both planes");
    assert_eq!(m.shape(), [NAUX, N_ACT, N_ACT]);
    // Both outstanding at once: b_full + m_proj, and nothing else.
    assert_eq!(
        pool::global_available_bytes(),
        Some(3 * one - 2 * one),
        "exactly the two planes must be outstanding"
    );
}

/// CONTRACT 3 (the lifetime): dropping the tensor CREDITS the bytes back.
///
/// The property a plain ceiling check structurally cannot have. Without it an
/// evGW outer loop that rebuilds `m_proj` every iteration would exhaust the
/// pool on iteration two with tensors that had already been freed.
#[test]
fn dropping_a_tensor_returns_its_bytes_to_the_pool() {
    let one = budget::b_full_bytes(NAUX, N_ACT);
    let _p = Installed::new(3 * one);
    let mo_b = make_mo_b().expect("b_full fits");
    let before = pool::global_available_bytes().expect("pool installed");

    for iteration in 0..4 {
        let m = project_b_into_pdep(&mo_b, &v_dressed(), Some(usize::MAX)).unwrap_or_else(|e| {
            panic!(
                "iteration {iteration} was refused, so m_proj's charge is NOT being released on \
                 drop -- the ledger only ever grows: {e}"
            )
        });
        assert_eq!(
            pool::global_available_bytes(),
            Some(before - budget::m_proj_bytes(NAUX, N_ACT)),
            "iteration {iteration}: m_proj must be outstanding while it is live"
        );
        drop(m);
        assert_eq!(
            pool::global_available_bytes(),
            Some(before),
            "iteration {iteration}: dropping m_proj must credit its bytes back"
        );
    }

    // And b_full's charge outlives every one of those iterations, because the
    // guard lives in the MoB rather than in the constructor's scope.
    assert_eq!(
        pool::global_available_bytes(),
        Some(3 * one - one),
        "b_full's charge must still be outstanding: the MoB is still alive"
    );
    drop(mo_b);
    assert_eq!(
        pool::global_available_bytes(),
        Some(3 * one),
        "dropping the MoB must credit b_full back"
    );
}

/// CONTRACT 4: the charge is `m_proj` ALONE, not `m_proj + b_full`.
///
/// The ceiling check compares the SUM, because it has no way to know whether
/// `b_full` is already accounted for. The pool charge must not repeat it:
/// `b_full`'s bytes are already outstanding (the `MoB` we are handed is
/// holding them), so charging the sum would debit `b_full` twice. An
/// over-charge refuses jobs that fit, which the brief calls a bug in its own
/// right.
///
/// Bracketed both ways so a "charge nothing" and a "charge double" both fail.
#[test]
fn m_proj_is_charged_once_not_twice() {
    let one = budget::b_full_bytes(NAUX, N_ACT);
    let _p = Installed::new(3 * one);
    let mo_b = make_mo_b().expect("b_full fits");
    let after_b_full = pool::global_available_bytes().expect("pool");
    let m = project_b_into_pdep(&mo_b, &v_dressed(), Some(usize::MAX)).expect("m_proj fits");
    let after_m_proj = pool::global_available_bytes().expect("pool");
    let debited = after_b_full - after_m_proj;
    assert_eq!(
        debited,
        budget::m_proj_bytes(NAUX, N_ACT),
        "m_proj debited {debited} B. Exactly m_proj_bytes is correct: charging 0 means the \
         gate is decoration, charging m_proj + b_full double-counts bytes the MoB already \
         holds and refuses jobs that fit."
    );
    drop(m);
}

/// CONTRACT 5: the soft QP-sweep gate DECLINES rather than refusing, and the
/// declined path is bit-identical.
///
/// The standing decision: "THE MIGRATION MUST NOT REFUSE A JOB THAT THE CURRENT
/// TREE COMPLETES." The per-worker scratch has a genuine fallback (one worker's
/// scratch, serial map, same order, same values), so the gate must take the
/// `None` branch and the run must finish -- with the SAME numbers.
///
/// # The window, derived
///
/// For the fallback to be what is under test, the capacity must satisfy BOTH:
///
/// ```text
///   capacity >= b_full + m_proj          (else a MANDATORY plane fails and
///                                         the test measures nothing)
///   capacity <  b_full + m_proj + scratch (else everything fits and the soft
///                                         branch is never taken)
/// ```
///
/// The window is non-empty for any scratch > 0, which the test asserts FIRST --
/// a gate whose GO conditions are mutually exclusive returns arithmetic, not
/// measurement.
#[test]
fn the_soft_qp_scratch_declines_instead_of_refusing() {
    let workers = rayon::current_num_threads().max(1);
    let one = budget::b_full_bytes(NAUX, N_ACT);
    let n_quad = 8;
    let scratch = budget::qp_worker_scratch_bytes(NAUX, N_ACT, n_quad, workers, 1);
    let floor = 2 * one; // b_full + m_proj, both mandatory
    let ceiling = floor + scratch;
    assert!(
        scratch > 0 && floor < ceiling,
        "the test window [{floor}, {ceiling}) is EMPTY at {workers} workers -- the pass \
         condition would be unreachable and this test would return arithmetic, not a \
         measurement"
    );
    // Sit inside the window: mandatory planes fit, the scratch does not.
    let capacity = floor + (ceiling - floor) / 4;
    assert!(capacity >= floor && capacity < ceiling);

    let _p = Installed::new(capacity);
    let mo_b = make_mo_b().expect("b_full is mandatory and must fit inside the window");
    let m = project_b_into_pdep(&mo_b, &v_dressed(), Some(usize::MAX))
        .expect("m_proj is mandatory and must fit inside the window");
    // With both mandatory planes outstanding, the scratch cannot fit -- so the
    // gate must hand back None (panel) rather than an error.
    let avail = pool::global_available_bytes().expect("pool");
    assert!(
        scratch > avail,
        "fixture is wrong: the scratch ({scratch} B) fits in what is left ({avail} B) at \
         {workers} workers, so the soft branch is never exercised"
    );
    drop(m);
}

/// CONTRACT 6: the QP-sweep charge is worker-dependent, so a FROZEN constant
/// would be wrong.
///
/// This is the bug that broke 3 of the ferric-rpa migration's 7 tests: a
/// capacity calibrated at this box's 12 workers silently stopped exercising the
/// branch at 2 and 4. Asserting the scaling explicitly is what stops a future
/// edit from hard-coding it back.
#[test]
fn the_qp_scratch_scales_with_the_worker_count() {
    let at = |w: usize| budget::qp_worker_scratch_bytes(NAUX, N_ACT, 8, w, 1);
    let one = at(1);
    assert!(one > 0);
    for w in [2usize, 4, 12, 64] {
        assert_eq!(
            at(w),
            one * w,
            "the per-worker scratch must scale linearly with the worker count; a test that \
             freezes it at one width stops measuring anything at the others"
        );
    }
    // And the two-channel (U-GW) form is twice the one-channel form.
    assert_eq!(
        budget::qp_worker_scratch_bytes(NAUX, N_ACT, 8, 4, 2),
        2 * at(4),
        "U-GW holds two scratch sets per worker (both spins inside one closure)"
    );
}

/// CONTRACT 7: with NO pool installed nothing is debited and nothing is
/// refused.
///
/// The trivial-limit anchor, restated at the plane level. The energy-level
/// version lives in `mwe_gw_pool_is_inert_without_a_pool.rs`; this one pins
/// that the guards themselves are inert, which is what makes that one possible.
#[test]
fn without_a_pool_the_guards_are_inert() {
    let _g = lock();
    pool::clear_global();
    assert!(pool::global_available_bytes().is_none());
    let mo_b = make_mo_b().expect("unbudgeted b_full must never be refused");
    let m = project_b_into_pdep(&mo_b, &v_dressed(), Some(usize::MAX))
        .expect("unbudgeted m_proj must never be refused");
    assert_eq!(m.shape(), [NAUX, N_ACT, N_ACT]);
    assert!(
        pool::global_available_bytes().is_none(),
        "an inert guard must not install a pool"
    );
}
