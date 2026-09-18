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
/// This test separates them by SIZE. The module's plan declares the 13 AO
/// planes PLUS the `m`/`mdchi` planes and transients allocated on top of them
/// (~20 planes), so its peak is strictly larger than the inner guard's 13. A
/// pool sized between the two therefore:
///
///   * ADMITS when the only charge is the inner, immediately-released one
///     (the mutant), and
///   * REFUSES when the module's full working set is held (the real code).
///
/// which is exactly the discrimination the other assertions lack.
#[test]
fn the_held_charge_covers_more_than_the_transient_ao_guard() {
    let _s = CleanSlot::acquire();

    let mol = h2o();
    let bs = basis::bundled("6-31g").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let nbf = prep.nbasis();
    let npts = ferric_dft::grid::build_atomic_grid(&mol, &AtomicGridConfig::default()).len();
    let plane = nbf * npts * 8;
    let inner_ao_guard = 13 * plane;

    // MEASURED on H2O/6-31G at the default grid (nbf=13, npts=24750):
    //   inner AO guard alone (the mutant):  13 planes = 33.46 MB
    //   module's held working set (real):   36 planes = 93.06 MB
    // The band that separates them is (13, 36) planes. 24 sits in the middle
    // of it with room on both sides for a different basis/grid to shift the
    // ratio, and is expressed in PLANES rather than as a frozen byte count so
    // it tracks nbf and npts instead of pinning one machine's measurement.
    //
    // The module's peak exceeds 13 planes because the plan declares, on top of
    // the AO tensors, the m/mdchi planes (4), the AO-partial transients (3),
    // and the already-resident grid + weight1 response arrays -- all of which
    // the inner guard is blind to.
    let between = 24 * plane;
    assert!(
        between > inner_ao_guard,
        "threshold must sit ABOVE the inner guard or it cannot discriminate"
    );

    install_global(MemoryPool::with_capacity_bytes(between));
    let got = run_gga_xc_gradient();
    clear_global();

    assert!(
        got.is_err(),
        "a pool of {between} B (24 planes) admitted the XC gradient, but the \
         declared working set is ~36 planes. That means the only thing charged \
         was the 13-plane AO guard inside eval_basis_grad_hess_on_points -- \
         which is RELEASED before chi/dchi/ddchi are allocated (it is the \
         non-`_reserved` variant). The module's own m/mdchi planes would then \
         be completely unaccounted while resident."
    );

    // Reachability control: the same call must ADMIT once the pool can hold
    // the whole declared working set, so the refusal above is about SIZE and
    // not about the gradient being unconditionally broken.
    install_global(MemoryPool::with_capacity_bytes(8_000_000_000));
    let ample = run_gga_xc_gradient();
    clear_global();
    assert!(
        ample.is_ok(),
        "reachability control failed: an ample pool must admit. Got {ample:?}"
    );
}

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

#[test]
fn an_incumbent_charge_narrows_what_the_xc_gradient_may_take() {
    let _s = CleanSlot::acquire();

    install_global(MemoryPool::with_capacity_bytes(8_000_000_000));
    run_gga_xc_gradient().expect("ample pool must admit");
    let needed = global().expect("pool").peak_bytes();
    clear_global();
    assert!(needed > 0, "nothing charged; cannot size the tight pool");

    // Control: a pool sized for the gradient alone admits it.
    install_global(MemoryPool::with_capacity_bytes(needed * 2));
    let alone = run_gga_xc_gradient();
    clear_global();
    assert!(
        alone.is_ok(),
        "reachability control failed: the gradient must fit a pool sized for it"
    );

    // THE COMPOSITION PROPERTY. Against `MemoryPlan::resolve` (a re-readable
    // ceiling) the gradient sees the full capacity no matter who else is
    // holding it, and sails through. Against the pool it sees only the
    // remainder.
    let pool = MemoryPool::with_capacity_bytes(needed * 2);
    install_global(pool.clone());
    let incumbent = pool
        .reserve("pretend SCF grid cache", needed * 2 - needed / 2)
        .expect("incumbent fits");
    let crowded = run_gga_xc_gradient();
    drop(incumbent);
    clear_global();

    assert!(
        crowded.is_err(),
        "COMPOSITION FAILED: the XC gradient was admitted against a pool whose \
         free space was already spoken for. A geometry optimization holds an \
         SCF grid cache and a DF tensor while it computes forces; that is \
         exactly this shape, and it is the OOM the pool exists to prevent."
    );
}
