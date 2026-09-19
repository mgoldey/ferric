//! The gradient planes are DEBITED against the shared pool, and they COMPOSE.
//!
//! The anchor next door (`mwe_gradient_pool_is_inert_without_a_pool.rs`) pins
//! the trivial limit: no pool, nothing changes. That test passes just as
//! happily against a tree where no gradient gate exists at all, so on its own
//! it proves nothing about the migration. This file is the other half: it
//! shows the charges are REAL.
//!
//! Three properties, each of which fails on the pre-migration tree:
//!
//! 1. Running a gradient against a pool leaves a non-zero `peak_bytes()`.
//!    Before the migration the gradient path took no reservation at all, so
//!    the peak stayed at whatever the SCF left and the gradient was invisible
//!    to the ledger.
//! 2. A pool too small for the gradient's NON-BATCHABLE working set (the grid
//!    and its `weight1` response array) REFUSES, with an error naming the
//!    plane — rather than proceeding to an OOM kill.
//! 3. A charge held by one subsystem NARROWS what the gradient may take. This
//!    is the composition property: two planes that each fit the whole ceiling
//!    must not both be admitted when their sum does not fit. It is the entire
//!    reason the pool exists, and the reason a re-readable ceiling
//!    (`MemoryPlan::resolve`) is not good enough.
//!
//!    Since the XC gradient began batching the DFT grid, "narrows" is literal:
//!    a crowded pool makes it take NARROWER BATCHES and still finish, rather
//!    than refuse. The assertion follows the property, not the old proxy — see
//!    `an_incumbent_charge_narrows_what_the_gradient_may_take` for why the
//!    refusal it used to require would now pin the worse behaviour.
//!
//! ## Reachability of the pass condition (per CLAUDE.md)
//!
//! Each assertion is checked in both directions in the same process: the
//! small-pool cases assert `is_err()` AND the ample-pool control asserts
//! `is_ok()` on the SAME input, so a gate that refused everything (or
//! admitted everything) fails one of the two. A test whose GO conditions
//! cannot both be reached returns arithmetic, not measurement.

use std::sync::{Mutex, MutexGuard, OnceLock};

use ferric_core::memory::pool::{clear_global, global, install_global, MemoryPool};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::gradient::rhf_gradient;
use ferric_scf::ks_gradient::ks_gradient_closed;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::ScfResult;
use ndarray::Array2;

fn global_lock() -> MutexGuard<'static, ()> {
    static L: OnceLock<Mutex<()>> = OnceLock::new();
    match L.get_or_init(|| Mutex::new(())).lock() {
        Ok(g) => g,
        Err(e) => e.into_inner(),
    }
}

struct CleanSlot(#[allow(dead_code)] MutexGuard<'static, ()>);

impl CleanSlot {
    fn acquire() -> Self {
        let g = global_lock();
        clear_global();
        Self(g)
    }
}

impl Drop for CleanSlot {
    fn drop(&mut self) {
        clear_global();
    }
}

fn water() -> Molecule {
    Molecule::load_xyz("../../testdata/molecules/water.xyz").expect("water.xyz")
}

/// Converge the SCF ONCE, with no pool installed, and hand back everything the
/// gradient needs.
///
/// Deliberately outside the pool window: this file is measuring what the
/// GRADIENT charges, and an SCF run inside the window would fill the ledger
/// with SCF planes and make the gradient's own charge impossible to attribute.
struct Fixture {
    mol: Molecule,
    bs: ferric_core::basis::BasisSet,
    obs: PreparedBasis,
    bounds: SchwarzBounds,
    scf: ScfResult,
}

fn fixture(xc: Option<&str>) -> Fixture {
    let mol = water();
    let bs = ferric_core::basis::bundled("6-31g").expect("6-31g");
    let obs = PreparedBasis::new(&mol, &bs).expect("prepared basis");
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).expect("schwarz");
    let ctx = ParallelContext::default();
    let cfg = RhfConfig {
        xc: xc.map(str::to_string),
        ..Default::default()
    };
    let scf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &cfg).expect("SCF must converge");
    Fixture {
        mol,
        bs,
        obs,
        bounds,
        scf,
    }
}

impl Fixture {
    fn hf_gradient(&self) -> Result<Array2<f64>, FerricError> {
        rhf_gradient(
            &self.mol,
            &self.obs,
            Operator::coulomb(),
            &self.bounds,
            &self.scf,
            None,
        )
    }

    fn ks_gradient(&self) -> Result<Array2<f64>, FerricError> {
        ks_gradient_closed(
            &self.mol,
            &self.obs,
            &self.bs,
            Operator::coulomb(),
            &self.bounds,
            "PBE",
            &self.scf,
            None,
        )
    }
}

#[test]
fn the_hf_gradient_debits_the_pool() {
    let _s = CleanSlot::acquire();
    let fx = fixture(None);

    install_global(MemoryPool::with_capacity_bytes(8_000_000_000));
    let peak_before = global().expect("pool").peak_bytes();
    fx.hf_gradient().expect("ample pool must admit");
    let peak_after = global().expect("pool").peak_bytes();
    let outstanding = global().expect("pool").outstanding_bytes();
    clear_global();

    assert!(
        peak_after > peak_before,
        "the HF gradient took NO reservation: peak {peak_before} -> {peak_after}. \
         On the pre-migration tree this path was invisible to the ledger, which \
         is the defect being fixed -- if this assertion is passing vacuously the \
         charge is not wired up."
    );
    assert_eq!(
        outstanding, 0,
        "every gradient reservation must be released when the gradient returns \
         (outstanding {outstanding} B) -- a leaked charge exhausts the pool over \
         an optimization's many steps"
    );
}

/// Each of the two HF-gradient list planes is charged SEPARATELY, and each is
/// named in its own refusal.
///
/// # Why `the_hf_gradient_debits_the_pool` is not enough
///
/// That test asserts only "the peak moved", which any ONE of the two charges
/// satisfies. MEASURED: deleting the quartet-list charge alone left it green,
/// and deleting the pair-list charge alone left it green; only deleting BOTH
/// turned it red. A test that cannot tell two planes apart cannot tell you
/// which one regressed, so it is pinned per plane here.
///
/// The probe is a pool sized to admit one list but not the other. Both lists
/// are built before their charge is taken (deliberately — see the comments at
/// the charge sites; sizing the build on the ledger would let the pool pick
/// the serial-vs-parallel branch and move a number), so the refusal names
/// whichever plane asked for more than was left.
#[test]
fn each_hf_gradient_list_plane_is_charged_under_its_own_name() {
    let _s = CleanSlot::acquire();
    let fx = fixture(None);

    // Reachability control: with room, this admits.
    install_global(MemoryPool::with_capacity_bytes(8_000_000_000));
    let ample = fx.hf_gradient();
    clear_global();
    assert!(
        ample.is_ok(),
        "reachability control failed: an ample pool must admit. Got {ample:?}"
    );

    // Size the two planes from the basis, so the thresholds track nsh rather
    // than pinning one machine's byte count.
    let nsh = fx.obs.nshells();
    let pair_bytes = (nsh * (nsh + 1) / 2) * std::mem::size_of::<(usize, usize)>();

    // --- Plane 1: the 1e shell-pair list.
    //
    // A pool far too small for ANY list. `oneelectron_gradient` runs first, so
    // the pair list is the first plane to ask and its name must be in the
    // message -- a refusal that does not say which buffer was too big is not
    // actionable.
    install_global(MemoryPool::with_capacity_bytes(8));
    let tight = fx.hf_gradient();
    clear_global();
    let msg = tight
        .expect_err("an 8-byte pool cannot hold either derivative list")
        .to_string();
    assert!(
        msg.contains("1e shell-pair list"),
        "the first refusal must name the 1e shell-pair list; got: {msg}"
    );

    // --- Plane 2: the screened quartet list.
    //
    // THE POINT OF THIS HALF: a pool that comfortably holds the pair list but
    // cannot hold the quartet list. Without it, deleting the quartet charge is
    // INVISIBLE -- the pair list always refuses first and masks it. MEASURED:
    // with only the probe above, removing the quartet-list charge left every
    // test in this file green.
    //
    // The quartet list is O(nsh^4)-ish and the pair list O(nsh^2), so a pool
    // of a few times the pair list separates them at any basis worth testing.
    install_global(MemoryPool::with_capacity_bytes(pair_bytes * 4));
    let mid = fx.hf_gradient();
    clear_global();
    let msg = mid.expect_err(
        "a pool holding only a few pair lists cannot hold the O(nsh^4) \
         screened quartet list",
    );
    let msg = msg.to_string();
    assert!(
        msg.contains("screened quartet list"),
        "the SECOND refusal must name the screened quartet list -- if it names \
         the pair list instead the threshold is mis-sized, and if it is not a \
         refusal at all the quartet-list charge is missing; got: {msg}"
    );
}

#[test]
fn the_ks_gradient_debits_the_pool_and_releases_it() {
    let _s = CleanSlot::acquire();
    let fx = fixture(Some("PBE"));

    install_global(MemoryPool::with_capacity_bytes(8_000_000_000));
    fx.ks_gradient().expect("ample pool must admit");
    let peak = global().expect("pool").peak_bytes();
    let outstanding = global().expect("pool").outstanding_bytes();
    clear_global();

    assert!(
        peak > 0,
        "the KS-DFT gradient took NO reservation -- the ddchi/mdchi planes are \
         the largest allocation in the gradient stack and must be on the ledger"
    );
    assert_eq!(
        outstanding, 0,
        "the KS gradient leaked {outstanding} B: the AO-plane charge must be \
         credited back when chi/dchi/ddchi drop"
    );
}

#[test]
fn a_pool_too_small_refuses_the_ks_gradient_by_name() {
    let _s = CleanSlot::acquire();
    let fx = fixture(Some("PBE"));

    // Control FIRST, so the GO condition is demonstrably reachable in this
    // same process: an ample pool admits this exact call.
    install_global(MemoryPool::with_capacity_bytes(8_000_000_000));
    let ample = fx.ks_gradient();
    clear_global();
    assert!(
        ample.is_ok(),
        "reachability control failed: the ample-pool case must ADMIT, else the \
         refusal below proves nothing (a gate that refuses everything would \
         also 'pass'). Got {ample:?}"
    );

    // A pool that cannot hold the grid AO planes must refuse, not OOM.
    install_global(MemoryPool::with_capacity_bytes(1_000_000));
    let tight = fx.ks_gradient();
    clear_global();

    let err = tight.expect_err("a 1 MB pool cannot hold the KS gradient's AO planes");
    let msg = err.to_string();
    assert!(
        msg.contains("XC gradient") || msg.contains("memory"),
        "the refusal must NAME the plane that did not fit, so the user can act \
         on it rather than guess; got: {msg}"
    );
}

/// THE COMPOSITION PROPERTY, as the batched XC gradient now expresses it.
///
/// # What changed, and why the assertion moved
///
/// This test used to require that a crowded pool make the KS gradient
/// **refuse**. That was the right property when the XC gradient evaluated the
/// whole DFT grid at once: with no way to shrink its working set, "does not
/// fit" and "must not run" were the same statement, and a refusal was the only
/// observable that distinguished composing against the ledger from double-
/// spending a ceiling.
///
/// The XC gradient now batches the grid (`ferric_dft::gradient`'s
/// `resolve_grad_batch_size`), so it has the same always-available fallback
/// `ks.rs`'s SCF grid gate has had all along: when the full working set does
/// not fit, take NARROWER BATCHES rather than error. `ks.rs` makes exactly this
/// argument at `check_grid_budget_reserved` — "a gate whose over-budget answer
/// is 'batch it' must keep answering 'batch it' against a pool, NOT start
/// erroring". Keeping the refusal here would have pinned the gradient to the
/// worse of the two behaviours: refusing a drug-sized optimization it can
/// actually complete.
///
/// So the property is unchanged — an incumbent charge must NARROW what the
/// gradient takes — and what is asserted is the narrowing itself rather than
/// its old proxy. A gradient that ignored the ledger would keep the SAME wide
/// batch and charge the SAME peak no matter who else is holding the pool;
/// that is the failure this catches, and it is a strictly sharper observable
/// than `is_err()` (which one over-wide batch could also produce, for the
/// wrong reason).
///
/// What still REFUSES is the non-batchable part: the grid and its `weight1`
/// response array are built before any AO plane exists and are indexed by
/// absolute grid index, so batching cannot shrink them —
/// `a_pool_too_small_refuses_the_ks_gradient_by_name` covers that, and it is
/// what keeps a truly hopeless pool from proceeding to an OOM kill.
#[test]
fn an_incumbent_charge_narrows_what_the_gradient_may_take() {
    let _s = CleanSlot::acquire();
    let fx = fixture(Some("PBE"));

    // Measure what the gradient takes with the pool to itself.
    install_global(MemoryPool::with_capacity_bytes(8_000_000_000));
    fx.ks_gradient().expect("ample pool must admit");
    let needed = global().expect("pool").peak_bytes();
    clear_global();
    assert!(
        needed > 0,
        "nothing was charged; cannot size the tight pool"
    );

    // Reachability control: a pool sized for the gradient alone admits it, and
    // the peak it records is the uncrowded baseline.
    install_global(MemoryPool::with_capacity_bytes(needed * 2));
    let alone = fx.ks_gradient();
    let alone_peak = global().expect("pool").peak_bytes();
    clear_global();
    assert!(
        alone.is_ok(),
        "reachability control failed: the gradient must fit a pool sized for \
         it, else the narrowing below is not about composition. Got {alone:?}"
    );

    // The same pool, but another subsystem is already holding most of it.
    // Against a re-readable ceiling (`MemoryPlan::resolve`) the gradient would
    // see the full capacity and take its full-width batch, because a ceiling
    // does not know what is outstanding. Against the pool it must see only
    // what is left and narrow accordingly.
    let pool = MemoryPool::with_capacity_bytes(needed * 2);
    install_global(pool.clone());
    let incumbent_bytes = needed * 2 - needed / 2;
    let _incumbent = pool
        .reserve("pretend DF 3-index tensor", incumbent_bytes)
        .expect("incumbent fits");
    let incumbent_only_peak = pool.peak_bytes();
    let crowded = fx.ks_gradient();
    let crowded_peak = pool.peak_bytes();
    drop(_incumbent);
    clear_global();

    // It must still produce an answer — that is the whole point of batching.
    let crowded = crowded.expect(
        "the batched XC gradient must COMPLETE under a crowded pool by taking \
         narrower batches, not refuse. A refusal here means the batch sizing \
         is not reaching the over-budget case it exists to serve.",
    );
    assert!(
        crowded.iter().all(|v| v.is_finite()),
        "the crowded run must return a real gradient, not NaNs"
    );

    // THE COMPOSITION PROPERTY: what the gradient itself added to the ledger
    // must be SMALLER when someone else is holding the pool. Subtracting the
    // incumbent's own charge isolates the gradient's contribution, so this
    // cannot pass merely because the incumbent inflated the total.
    let crowded_added = crowded_peak.saturating_sub(incumbent_only_peak);
    assert!(
        crowded_added < alone_peak,
        "COMPOSITION FAILED: with {incumbent_bytes} B already spoken for, the \
         gradient still charged {crowded_added} B of its own -- as much as the \
         {alone_peak} B it takes with the pool to itself. It is sizing against \
         the CAPACITY rather than against what is left, which is the exact \
         double-spend the pool exists to prevent."
    );
}
