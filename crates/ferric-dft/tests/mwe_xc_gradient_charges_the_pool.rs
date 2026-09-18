//! The XC nuclear-gradient AO planes are DEBITED against the shared pool.
//!
//! # Why this file exists, and not just the ferric-scf composition test
//!
//! `ferric-scf`'s `mwe_gradient_planes_compose_in_one_pool.rs` drives
//! `ks_gradient_closed`, which calls the 2-electron gradient (whose screened
//! quartet list is now charged) BEFORE the XC gradient. That charge alone
//! satisfies every "peak > 0" / "small pool refuses" assertion in that file —
//! MEASURED: reverting all six `commit()` calls in
//! `ferric_dft::gradient` back to `check()` left all four of its tests
//! GREEN. It was a surviving mutation, and this file is the fix.
//!
//! The lesson generalizes: a composite entry point cannot attribute a charge
//! to the plane you think you are testing. To test a plane you must call the
//! function that allocates it.
//!
//! So this file calls `xc_gradient_closed_gga_from_density` DIRECTLY. Nothing
//! else on that path touches the pool, so any reservation observed here is the
//! AO grid planes — `chi` + `dchi` + `ddchi` (13 planes of `nbf·npts`), plus
//! the `m`/`mdchi` planes allocated on top of them. `ddchi`, the AO SECOND
//! derivatives, is the single largest allocation in ferric's entire gradient
//! stack.
//!
//! ## Reachability of the pass condition (per CLAUDE.md)
//!
//! The refusal test asserts the ample-pool control ADMITS the same call in the
//! same process before asserting the tight pool REFUSES it. A gate that
//! refused unconditionally would fail the control; one that admitted
//! unconditionally would fail the refusal. Both directions are exercised, so
//! the pass condition is a measurement rather than arithmetic.

use std::sync::{Mutex, MutexGuard, OnceLock};

use ferric_core::basis;
use ferric_core::memory::pool::{clear_global, global, install_global, MemoryPool};
use ferric_core::mol::Molecule;
use ferric_dft::grid::AtomicGridConfig;
use ferric_integrals::basis_bridge::PreparedBasis;
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

fn h2o() -> Molecule {
    Molecule::parse_xyz(
        "3\nH2O\nO 0 0 0\nH 0 0.7572 0.5868\nH 0 -0.7572 0.5868\n",
        0,
        1,
    )
    .unwrap()
}

/// Call the closed-shell GGA XC gradient directly on a crude but valid
/// density.
///
/// The NUMBERS are irrelevant here — this file measures allocation
/// bookkeeping, not physics (the XC gradient's correctness is covered by
/// `dft_gradient_mgga.rs` and the PySCF reference tests). What matters is that
/// the call allocates the full χ + ∇χ + ∇∇χ working set, which it does for any
/// density.
fn run_gga_xc_gradient() -> Result<Array2<f64>, ferric_dft::gradient::KsGradError> {
    let mol = h2o();
    let bs = basis::bundled("6-31g").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let nbf = prep.nbasis();
    // A normalized-ish diagonal density: enough electrons on the grid for
    // libxc to produce finite potentials, with no SCF dependency.
    let mut d = Array2::<f64>::zeros((nbf, nbf));
    let nelec = mol.nelec() as f64;
    for i in 0..nbf {
        d[(i, i)] = nelec / nbf as f64;
    }
    let grid_cfg = AtomicGridConfig::default();
    ferric_dft::gradient::xc_gradient_closed_gga_from_density(
        &mol,
        &bs,
        &d,
        "PBE",
        &grid_cfg,
        prep.shell_to_atom(),
        prep.shell_offsets(),
        prep.shell_dims(),
    )
}

#[test]
fn the_xc_gradient_ao_planes_are_debited_and_released() {
    let _s = CleanSlot::acquire();

    install_global(MemoryPool::with_capacity_bytes(8_000_000_000));
    run_gga_xc_gradient().expect("an ample pool must admit the XC gradient");
    let peak = global().expect("pool").peak_bytes();
    let outstanding = global().expect("pool").outstanding_bytes();
    clear_global();

    assert!(
        peak > 0,
        "the XC gradient took NO reservation. `ddchi` (AO second derivatives, \
         9 planes of nbf*npts) is the largest single allocation in the gradient \
         stack and it must be on the ledger -- if this is 0 the plan is calling \
         check() (a ceiling read that forgets) rather than commit() (a debit)."
    );
    assert_eq!(
        outstanding, 0,
        "the XC gradient leaked {outstanding} B. The charge must be credited \
         back when chi/dchi/ddchi drop, or a geometry optimization exhausts the \
         pool after a handful of steps."
    );
}

/// The AO planes must stay debited for as long as the TENSORS are resident —
/// not merely be counted and handed straight back.
///
/// # Why a peak/`>0` assertion cannot see this
///
/// `eval_basis_grad_hess_on_points` already calls
/// `ferric_integrals::ao_grid::check_ao_grid_budget`, which debits the pool
/// for the 13 AO planes — and then drops the guard before `chi`/`dchi`/`ddchi`
/// are even allocated (it is the `.map(|_| ())` non-`_reserved` variant). So
/// on the gradient path the pool is charged for a moment and credited back
/// while the tensors live on. Any assertion phrased as "peak > 0" or "a 1 MB
/// pool refuses" is satisfied by that momentary charge alone.
///
/// MEASURED: with all six `commit()` calls in `ferric_dft::gradient` reverted
/// to `check()`, the peak/refusal tests in this very file stayed GREEN. They
/// were measuring the inner guard, not the migration.
///
/// This test separates them by SIZE: the module's plan declares the 13 AO
/// planes PLUS the `m`/`mdchi` planes and the AO-partial transients allocated
/// on top of them, so what it HOLDS is strictly larger than the 13 the inner
/// guard would have counted and released.
///
/// # Why the observable is now the PEAK, not a refusal
///
/// It used to be a refusal: a pool sized between 13 and the module's ~36
/// planes admitted the mutant (inner guard only) and refused the real code.
/// Two things changed when the XC gradient began batching the grid.
///
/// 1. An over-budget pool no longer refuses — it narrows the batch. So "a pool
///    of N planes refuses" stopped being an available observable at all, for
///    the same reason it stopped being one in
///    `an_incumbent_charge_narrows_what_the_xc_gradient_may_take`.
/// 2. The gradient now calls `eval_basis_grad_hess_on_points_unchecked`, so
///    the inner 13-plane guard this test was contrasting against is no longer
///    on this path. It was a DOUBLE charge: the plan already declares those 13
///    planes, and the inner guard debited them again on top, which is why the
///    old held working set measured ~36 planes rather than the ~21 the plan
///    actually declares. Removing it is why the benzene/cc-pVDZ ample-pool
///    demand fell from 3.03 GB to 1.86 GB with no batching involved.
///
/// So the discrimination is made directly: run against an ample pool (where
/// the width is the whole grid, i.e. the pre-batching working set) and require
/// the recorded PEAK to exceed 13 planes. A tree that charged only the AO
/// tensors — or only a transient guard — cannot reach that, and a tree that
/// charged nothing reaches 0.
#[test]
fn the_held_charge_covers_more_than_the_transient_ao_guard() {
    let _s = CleanSlot::acquire();

    let mol = h2o();
    let bs = basis::bundled("6-31g").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let nbf = prep.nbasis();
    let npts = ferric_dft::grid::build_atomic_grid(&mol, &AtomicGridConfig::default()).len();
    let plane = nbf * npts * 8;
    let ao_planes_only = 13 * plane;

    // MEASURED on H2O/6-31G at the default grid (nbf=13, npts=24750,
    // plane = 2.574 MB):
    //   AO tensors alone            13 planes = 33.46 MB
    //   what the plan declares      ~21 planes = 54.25 MB
    //       = 13 AO + 4 (m, mdchi) + 3 (AO-partial transients)
    //         + grid & weight1 (2.77 MB, ~1.08 planes)
    // Expressed in PLANES, not as a frozen byte count, so it tracks nbf and
    // npts rather than pinning one machine's measurement.
    install_global(MemoryPool::with_capacity_bytes(8_000_000_000));
    let ample = run_gga_xc_gradient();
    let peak = global().expect("pool").peak_bytes();
    clear_global();

    assert!(
        ample.is_ok(),
        "reachability control failed: an ample pool must admit, else the peak \
         below is not a measurement of the gradient. Got {ample:?}"
    );
    assert!(
        peak > ao_planes_only,
        "the XC gradient charged {peak} B, which is no more than the {ao_planes_only} B \
         of AO tensors alone ({} planes of {}). The m/mdchi planes and the \
         AO-partial transients are allocated ON TOP of chi/dchi/ddchi and must \
         be on the ledger while they are resident -- if they are not, a \
         geometry optimization is unaccounted for ~7 planes at the moment it \
         is closest to the ceiling.",
        peak / plane.max(1),
        13,
    );
    // ...and not absurdly more, which would mean something is charged twice.
    // This is the assertion that would have caught the inner AO guard's
    // double charge (it put the peak at ~36 planes against a ~21-plane plan).
    assert!(
        peak < 30 * plane,
        "the XC gradient charged {peak} B = {} planes against a plan that \
         declares ~21. Something is being charged TWICE -- that is what the \
         now-removed inner `check_ao_grid_budget` call did on this path.",
        peak / plane.max(1),
    );
}

/// What STILL refuses, now that over-budget generally means "batch it".
///
/// Batching shrinks the AO planes arbitrarily far — down to one grid point at
/// a time — so no pool is too small for them alone. What it cannot shrink is
/// the grid and its `weight1` response array: both are built before any AO
/// plane exists, both are indexed by ABSOLUTE grid index by the weight-response
/// scatter, and both outlive every batch. A pool that cannot hold those has
/// nothing left to fall back to, and must say so rather than proceed to an OOM
/// kill. That is the case this pins, and it is the gradient's analogue of
/// `ks.rs`'s non-batchable VV10 NLC grid.
#[test]
fn a_pool_too_small_refuses_the_xc_gradient_rather_than_ooming() {
    let _s = CleanSlot::acquire();

    // Reachability control: the same call must ADMIT when there is room.
    install_global(MemoryPool::with_capacity_bytes(8_000_000_000));
    let ample = run_gga_xc_gradient();
    clear_global();
    assert!(
        ample.is_ok(),
        "reachability control failed: an ample pool must admit, else the \
         refusal below proves nothing. Got {ample:?}"
    );

    install_global(MemoryPool::with_capacity_bytes(1_000_000));
    let tight = run_gga_xc_gradient();
    clear_global();

    let err = tight.expect_err(
        "a 1 MB pool cannot hold the XC gradient's AO planes and must refuse, \
         not proceed to an OOM kill",
    );
    let msg = err.to_string();
    assert!(
        msg.contains("GGA XC gradient"),
        "the refusal must NAME the plane so the user can act on it; got: {msg}"
    );
}

/// THE COMPOSITION PROPERTY, as the batched XC gradient now expresses it.
///
/// # Why this stopped asserting a refusal
///
/// This test used to require that a crowded pool make the XC gradient
/// **refuse**. That was right when the gradient evaluated the whole DFT grid
/// at once: with no way to shrink its working set, "does not fit" and "must
/// not run" were the same statement.
///
/// The gradient now batches the grid (`gradient::resolve_grad_batch_size`), so
/// over-budget has a real answer other than an error — take narrower batches —
/// exactly as `ks.rs`'s SCF grid gate has always done
/// (`check_grid_budget_reserved`: "a gate whose over-budget answer is 'batch
/// it' must keep answering 'batch it' against a pool, NOT start erroring").
/// Keeping the refusal would pin the worse behaviour: refusing a drug-sized
/// optimization the code can now complete.
///
/// The PROPERTY is unchanged — an incumbent must narrow what the gradient
/// takes — and it is now asserted directly rather than through its old proxy.
/// A gradient that ignored the ledger would take the same wide batch and
/// charge the same bytes regardless of who else holds the pool; that is what
/// this catches.
#[test]
fn an_incumbent_charge_narrows_what_the_xc_gradient_may_take() {
    let _s = CleanSlot::acquire();

    install_global(MemoryPool::with_capacity_bytes(8_000_000_000));
    run_gga_xc_gradient().expect("ample pool must admit");
    let needed = global().expect("pool").peak_bytes();
    clear_global();
    assert!(needed > 0, "nothing charged; cannot size the tight pool");

    // Control: a pool sized for the gradient alone admits it, and the peak it
    // records is the uncrowded baseline.
    install_global(MemoryPool::with_capacity_bytes(needed * 2));
    let alone = run_gga_xc_gradient();
    let alone_peak = global().expect("pool").peak_bytes();
    clear_global();
    assert!(
        alone.is_ok(),
        "reachability control failed: the gradient must fit a pool sized for it"
    );

    // THE COMPOSITION PROPERTY. Against `MemoryPlan::resolve` (a re-readable
    // ceiling) the gradient sees the full capacity no matter who else is
    // holding it, and takes its full-width batch. Against the pool it sees
    // only the remainder and must narrow.
    let pool = MemoryPool::with_capacity_bytes(needed * 2);
    install_global(pool.clone());
    let incumbent_bytes = needed * 2 - needed / 2;
    let incumbent = pool
        .reserve("pretend SCF grid cache", incumbent_bytes)
        .expect("incumbent fits");
    let incumbent_only_peak = pool.peak_bytes();
    let crowded = run_gga_xc_gradient();
    let crowded_peak = pool.peak_bytes();
    drop(incumbent);
    clear_global();

    crowded.expect(
        "the batched XC gradient must COMPLETE under a crowded pool by taking \
         narrower batches, not refuse -- that is the whole point of batching, \
         and a refusal here means the sizing never reaches the over-budget \
         case it exists to serve.",
    );

    // Subtracting the incumbent's own charge isolates what the GRADIENT added,
    // so this cannot pass merely because the incumbent inflated the total.
    let crowded_added = crowded_peak.saturating_sub(incumbent_only_peak);
    assert!(
        crowded_added < alone_peak,
        "COMPOSITION FAILED: with {incumbent_bytes} B already spoken for, the \
         XC gradient still charged {crowded_added} B of its own -- as much as \
         the {alone_peak} B it takes with the pool to itself. A geometry \
         optimization holds an SCF grid cache and a DF tensor while it computes \
         forces; sizing against CAPACITY rather than against the remainder is \
         exactly the OOM the pool exists to prevent."
    );
}
