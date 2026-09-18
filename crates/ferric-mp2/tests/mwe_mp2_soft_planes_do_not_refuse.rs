//! The SOFT-charged MP2 planes must never refuse a job the pre-migration tree
//! completes.
//!
//! # The standing decision this enforces
//!
//! "THE MIGRATION MUST NOT REFUSE A JOB THAT THE CURRENT TREE COMPLETES. If
//! you find a site where hard-charging would refuse such a job, make it soft."
//! An over-estimating guard is as much a bug as a missing one -- it refuses
//! jobs that would have fit.
//!
//! # Which planes are soft, and why (MEASURED)
//!
//! Three, all documented at their reservation sites:
//!
//! 1. **`b_ov` in the closed-shell energy lane** (`ri_mp2_spin_components`).
//!    This is the one the contract below pins, because it is the one with a
//!    measured refusal. `stream_dressed_mo_band_budgeted` READS the AO
//!    3-index source while FILLING `b_ov`, so the two genuinely co-reside --
//!    and `ThreeIndexSource` holds its own hard pool charge for the AO tensor.
//!    Hard-charging `b_ov` on top therefore refuses across a real budget band
//!    the pre-migration tree completes. MEASURED, benzene/cc-pVDZ: the window
//!    is [ao, ao + b_ov) = [43.67, 50.23] MB.
//!
//! 2. The per-worker `g_i` energy transient (`nocc·nvir²·8 × n_workers`). Its
//!    SIZE reads `rayon::current_num_threads()`, which is not a configuration,
//!    so a hard gate would make ADMISSION depend on `RAYON_NUM_THREADS`.
//!    NOTE it is NOT what the contract below detects: `drop(src)` releases the
//!    AO tensor before `g_i` is charged, so by then the ledger holds only
//!    `b_ov` and `g_i` fits easily. Mutation M3 (soft made hard) SURVIVED
//!    against a `g_i`-targeted contract at 2, 4 and 12 workers for exactly
//!    that reason. Its softness is a defensive choice, not a measured one, and
//!    this file says so rather than implying a measurement it does not have.
//!
//! 3. The Laplace-MP2 AO per-worker quadrature panel set
//!    (`per_task_budget_bytes × n_workers`). `warn_if_task_workers_exceed_nominal`
//!    already documents that this aggregate CAN exceed the budget and
//!    deliberately only WARNS -- because the alternative is dividing the panel
//!    width by the live worker count, which would make the energy depend on
//!    `RAYON_NUM_THREADS`. That trade is settled in this crate and a hard
//!    charge would silently reverse it.
//!
//! # Artifact hypothesis (stated before measuring)
//!
//! If the soft branch is REAL: a pool sized inside the `[ao, ao + b_ov)`
//! window admits the run and returns the SAME energy it returns unbudgeted.
//! If the gate were secretly hard, the same pool returns a `memory pool
//! exhausted` error naming `b_ov`. Those predictions differ, so this test can
//! distinguish them.
//!
//! A soft gate whose None branch does not actually stream would be a LIE. Here
//! the None branch is genuinely a no-op on behaviour -- nothing downstream
//! reads any of these guards, and the loops they cover run at whatever
//! concurrency rayon supplies with or without a reservation. That is pinned by
//! the energy equality below: if the soft branch changed anything, the
//! energies would differ.
//!
//! # Every figure here is DERIVED, never frozen at this box's width
//!
//! `g_i` is per-worker, so any constant calibrated against it on this 12-wide
//! box goes red on a 2- or 4-wide CI runner. Nothing below is hardcoded: the
//! capacity comes from `refusal_window_capacity()`, which is a function of the
//! allocation shapes alone, and the one place a worker count appears
//! (`g_i_bytes`) reads `rayon::current_num_threads()` live. Verified green at
//! RAYON_NUM_THREADS = 2, 4 and 12.
//!
//! The fixture is benzene/cc-pVDZ rather than water/cc-pVDZ because the
//! refusal window has to be wide enough to target: `b_ov` is 6.56 MB at
//! benzene against 64 kB at water, and CONTRACT 1 asserts a 1 MB floor on it.

use std::sync::{Mutex, MutexGuard, OnceLock};

use ferric_core::memory::pool::{clear_global, install_global, MemoryPool};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::rimp2::{ri_mp2, RiMp2Config};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

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

struct Fixture {
    mol: Molecule,
    obs: PreparedBasis,
    dfbs: PreparedBasis,
    rhf: ferric_scf::ScfResult,
    nao: usize,
    naux: usize,
    nocc: usize,
    nvir: usize,
}

fn fixture() -> Fixture {
    let mol = Molecule::load_xyz("../../testdata/molecules/benzene.xyz").expect("benzene.xyz");
    let bs = ferric_core::basis::bundled("cc-pvdz").expect("cc-pvdz");
    let dfbs_set = ferric_core::basis::bundled("cc-pvdz-ri").expect("cc-pvdz-ri");
    let obs = PreparedBasis::new(&mol, &bs).expect("prepared obs");
    let dfbs = PreparedBasis::new(&mol, &dfbs_set).expect("prepared dfbs");
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).expect("schwarz");
    let ctx = ParallelContext::default();
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).expect("rhf");
    let nao = obs.nbasis();
    let naux = dfbs.nbasis();
    let nocc = (mol.nelec() as usize) / 2;
    let nvir = nao - nocc;
    Fixture {
        mol,
        obs,
        dfbs,
        rhf,
        nao,
        naux,
        nocc,
        nvir,
    }
}

impl Fixture {
    fn ao_bytes(&self) -> usize {
        self.naux * self.nao * self.nao * 8
    }
    fn b_flat_bytes(&self) -> usize {
        self.naux * self.nocc * self.nvir * 8
    }
    /// The per-worker `g_i` transient at the ambient worker count. This is the
    /// SOFT plane.
    fn g_i_bytes(&self) -> usize {
        self.nocc * self.nvir * self.nvir * 8 * rayon::current_num_threads().max(1)
    }
    /// A capacity strictly inside the measured `[ao, ao + b_ov)` refusal
    /// window: the AO tensor fits in core, and b_ov then does not fit on top.
    /// Derived from the allocation shapes, so it is width-independent.
    fn refusal_window_capacity(&self) -> usize {
        self.ao_bytes() + self.b_flat_bytes() / 2
    }
    fn run(&self) -> Result<f64, ferric_core::FerricError> {
        ri_mp2(
            &self.mol,
            &self.obs,
            &self.dfbs,
            Operator::coulomb(),
            &self.rhf,
            &RiMp2Config::default(),
        )
        .map(|r| r.mp2_corr)
    }
}

/// CONTRACT 1: the budget CONTRACT 2 uses is REACHABLE -- it sits strictly
/// inside the measured refusal window, so a hard charge really would refuse
/// it, and the pre-existing ceiling check really would admit it.
///
/// # Why a window, and not a ratio
///
/// An earlier version asserted "the soft plane is at least 10% of the hard
/// planes". That bar was wrong twice over. It was about `g_i`, which is
/// per-worker, so the ratio falls with the rayon pool width (MEASURED at
/// benzene/cc-pVDZ: 5.8% at 2 workers, 11.6% at 4, 34.7% at 12) -- a test
/// green on this 12-wide box and red on a 2-wide CI runner. And `g_i` is not
/// the plane the contract can detect anyway: `drop(src)` frees the AO tensor
/// before `g_i` is charged, so mutation M3 SURVIVED against it at every width.
///
/// What the contract actually detects is `b_ov`, whose refusal window is fixed
/// by the allocation shapes alone and so is width-independent:
/// `[ao, ao + b_ov)`. The last assertion is the one that took a survived
/// mutation to find -- the capacity must also clear `check_mo_side_alloc`'s
/// pre-existing CEILING check, or the run dies there and the contract
/// measures a gate that has nothing to do with the pool.
#[test]
fn the_contract_budget_sits_inside_the_measured_refusal_window() {
    let f = fixture();
    let capacity = f.refusal_window_capacity();
    assert!(
        capacity > f.ao_bytes(),
        "CONTRACT 2's capacity ({capacity}) must EXCEED the AO tensor ({}) -- below it the \
         tensor spills, nothing overlaps, and a hard charge would not refuse either, so \
         the contract would prove nothing. (nao={}, naux={}, nocc={}, nvir={})",
        f.ao_bytes(),
        f.nao,
        f.naux,
        f.nocc,
        f.nvir
    );
    assert!(
        capacity < f.ao_bytes() + f.b_flat_bytes(),
        "and it must fall SHORT of ao + b_ov ({}), or both fit and a hard charge is \
         admitted too",
        f.ao_bytes() + f.b_flat_bytes()
    );
    assert!(
        f.b_flat_bytes() > 1_000_000,
        "the refusal window is only {} bytes wide, too thin to target reliably",
        f.b_flat_bytes()
    );
    let ceiling_needs = f.b_flat_bytes() + f.g_i_bytes();
    assert!(
        capacity > ceiling_needs,
        "capacity {capacity} is below what check_mo_side_alloc demands ({ceiling_needs} = \
         b_ov + g_i at {} workers), so the run fails that pre-existing CEILING check before \
         reaching any pool charge and CONTRACT 2 measures the wrong gate entirely",
        rayon::current_num_threads().max(1)
    );
}

/// CONTRACT 2: a pool sized inside the measured refusal window must still
/// COMPLETE the run, with an unchanged energy.
///
/// This is the contract mutation M3 (turn the soft gate hard) must fail. It is
/// the direct encoding of the standing decision: the pre-migration tree runs
/// this job, so the migrated tree must too.
#[test]
fn a_budget_inside_the_refusal_window_still_completes() {
    let _s = CleanSlot::acquire();
    let f = fixture();

    // Reference: what the unbudgeted tree returns.
    let unbudgeted = f.run().expect("unbudgeted run must succeed");

    // Strictly inside the measured [ao, ao + b_ov) refusal window: the AO
    // tensor is big enough to be kept IN CORE (so ThreeIndexSource takes its
    // whole hard charge) and the pool then has too little left for b_ov.
    // CONTRACT 1 pins that this really is inside the window AND above the
    // pre-existing ceiling check.
    let capacity = f.refusal_window_capacity();

    let pool = MemoryPool::with_capacity_bytes(capacity);
    install_global(pool.clone());
    let outcome = f.run();
    let peak = pool.peak_bytes();
    clear_global();

    let got = match outcome {
        Ok(v) => v,
        Err(e) => panic!(
            "a {capacity}-byte pool REFUSED a job the unbudgeted tree completes \
             (E_corr = {unbudgeted:.12}). `b_ov` must be charged SOFTLY: it is filled \
             WHILE the AO 3-index source is still being read, so it genuinely overlaps a \
             tensor ThreeIndexSource has already hard-charged, and a hard charge here \
             refuses every budget in [{}, {}) that the tree completes.\n{e}",
            f.ao_bytes(),
            f.ao_bytes() + f.b_flat_bytes()
        ),
    };
    assert_eq!(
        got.to_bits(),
        unbudgeted.to_bits(),
        "the soft branch must be behaviourally inert: falling back must not change the \
         energy by one bit ({got:.17e} vs {unbudgeted:.17e}). If it does, the None branch \
         is not merely 'not reserved' -- it is taking a different code path."
    );
    // And the run must still have charged the HARD planes. A soft fallback
    // that quietly turned every charge off would also pass the two assertions
    // above.
    assert!(
        peak >= f.ao_bytes(),
        "the run completed but charged only {peak} bytes -- under the AO tensor's {}. \
         Soft-charging one plane must not disable the hard ones.",
        f.ao_bytes()
    );
}
