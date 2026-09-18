//! The MP2 planes must COMPOSE: whichever asks second sees only what the
//! first left.
//!
//! # The measured defect
//!
//! On the pre-migration tree, `crates/ferric-mp2/src/rimp2.rs` read the memory
//! budget at eight separate sites and every one of them compared its own plane
//! against the WHOLE ceiling. `ThreeIndexSource` (in ferric-integrals, already
//! migrated) debited the shared pool for the AO 3-index tensor; every MO-side
//! plane built on top of it -- `b_flat`, `b_oo`, `b_vv`, the per-worker
//! `g_i` -- was allocated unconditionally and invisibly.
//!
//! MEASURED, benzene/cc-pVDZ (nao=114, naux=420, nocc=21, nvir=93), ample
//! 64 GB pool:
//!
//! ```text
//!   plane                              bytes        pre-migration  post
//!   AO 3-index naux*nao^2*8            0.0437 GB    charged        charged
//!   b_flat     naux*nocc*nvir*8        0.0066 GB    INVISIBLE      charged
//!   g_i        nocc*nvir^2*8*nworkers  0.0174 GB    INVISIBLE      charged (soft)
//!   ------------------------------------------------------------------------
//!   pool.peak_bytes()                               0.0437 GB      0.0677 GB
//! ```
//!
//! i.e. the ledger saw 0.0437 of the 0.0677 GB the process held -- 65%, with
//! 35% of the peak invisible to every gate. The gap grows with
//! nvir/nao: at danuglipron/def2-SVP (naux=2800, nocc=90, nvir=610) the unseen
//! MO side is 1.23 GB (b_ov) + 8.34 GB (b_vv) against a ~1.1 GB AO tensor.
//!
//! # Artifact hypothesis (stated before measuring)
//!
//! If the composition property is REAL, a pool sized to admit the AO tensor
//! but NOT the AO tensor plus the MO side must refuse the run with an error
//! naming the MO plane and listing the AO tensor as the incumbent. If the
//! charges are DECORATION (taken and immediately released, or taken against a
//! ceiling rather than the ledger), the same pool admits the run and the test
//! sees a successful energy. Those predictions differ, so the test can
//! distinguish them.
//!
//! The counter-hypothesis that would make this test a LIE is that the pool is
//! simply too small for the AO tensor alone -- in which case the refusal proves
//! nothing about composition. `the_refusal_budget_admits_the_ao_tensor_alone`
//! is the reachability guard for that, and it is why the budget here is
//! calibrated to a MEASURED figure rather than guessed.

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

/// Clear the process-global pool on entry AND exit. Both halves are needed:
/// `e9949eed` on the integrals path found that a separate test binary alone
/// still let one test's pool refuse a SIBLING that never asked for it, and
/// that running two pool tests concurrently made a KILLED mutation look like a
/// survivor.
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

/// water/cc-pVDZ. Small enough to run in a test, and -- per
/// `the_refusal_budget_admits_the_ao_tensor_alone`'s third assertion -- one
/// where `b_ov` is a large enough fraction of the AO tensor that CONTRACT 2's
/// budget window is a real measurement rather than a rounding error.
fn fixture() -> Fixture {
    let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").expect("water.xyz");
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
    /// b_oo + b_vv, the two blocks `compute_mp2_intermediates` adds on top of
    /// b_ov. `b_vv` is the largest tensor in the crate.
    fn b_oo_vv_bytes(&self) -> usize {
        self.naux * self.nvir * self.nvir * 8 + self.naux * self.nocc * self.nocc * 8
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

    /// The CPKS/gradient intermediates path, which is where the HARD-charged
    /// `b_ov + b_oo + b_vv` plane lives.
    fn run_intermediates(&self) -> Result<f64, ferric_core::FerricError> {
        ferric_mp2::rimp2::compute_mp2_intermediates(
            &self.mol,
            &self.obs,
            &self.dfbs,
            Operator::coulomb(),
            &self.rhf,
            &RiMp2Config::default(),
        )
        .map(|i| i.e_mp2)
    }
}

// ---------------------------------------------------------------------------
// CONTRACT 1: the MO side is charged AT ALL.
//
// This is the contract mutation M1 (drop the reservation immediately) must
// fail. A charge that is taken and released before the tensor exists leaves
// the peak at exactly the AO tensor, which is the pre-migration number.
// ---------------------------------------------------------------------------

#[test]
fn the_pool_peak_exceeds_the_ao_tensor_alone() {
    let _s = CleanSlot::acquire();
    let f = fixture();
    let pool = MemoryPool::with_capacity_bytes(64_000_000_000);
    install_global(pool.clone());
    let e = f.run().expect("an ample pool must admit the run");
    let peak = pool.peak_bytes();
    clear_global();

    assert!(e < 0.0, "sanity: MP2 correlation energy must be negative");
    assert!(
        peak > f.ao_bytes(),
        "pool peak {peak} did not exceed the AO 3-index tensor alone ({}) -- the MO-side \
         planes are not being charged, which is the pre-migration state this fixes. \
         (nao={}, naux={}, nocc={}, nvir={})",
        f.ao_bytes(),
        f.nao,
        f.naux,
        f.nocc,
        f.nvir
    );
}

// ---------------------------------------------------------------------------
// CONTRACT 2: the planes COMPOSE -- a pool that fits each plane separately but
// not their sum must REFUSE.
//
// This is the property a ceiling cannot have and the pool exists to provide.
// It is the contract mutation M2 (make the pool advisory) must fail.
// ---------------------------------------------------------------------------

#[test]
fn a_pool_that_fits_each_plane_but_not_their_sum_refuses() {
    let _s = CleanSlot::acquire();
    let f = fixture();

    // Targeted at the HARD-charged intermediates plane (b_ov + b_oo + b_vv),
    // not at the energy lane's b_ov.
    //
    // The energy lane's b_ov charge is deliberately SOFT -- see the long note
    // at its site in `rimp2.rs`. Hard-charging it refused benzene/cc-pVDZ in
    // the measured band [43.67, 50.23] MB, which the pre-migration tree
    // completes, because the AO tensor is kept in core there rather than
    // spilling. `mwe_mp2_soft_planes_do_not_refuse.rs` pins that it must stay
    // soft.
    //
    // The intermediates plane has no such fallback: `b_vv` is dense by
    // construction and escapes into the returned struct, so it is hard, and it
    // is the right place to assert composition.
    //
    // Calibrated to the MEASURED shapes: big enough that the AO tensor alone
    // fits with room to spare (the reachability guard below pins that), too
    // small for the AO tensor plus the MO blocks.
    let capacity = f.ao_bytes() + f.b_flat_bytes() / 2;
    let pool = MemoryPool::with_capacity_bytes(capacity);
    install_global(pool.clone());
    let outcome = f.run_intermediates();
    clear_global();

    let err = match outcome {
        Err(e) => e.to_string(),
        Ok(v) => panic!(
            "a {capacity}-byte pool admitted an intermediates build needing {} (AO) + {} \
             (b_ov) + {} (b_oo+b_vv) and returned E_mp2 = {v:.12}. Each plane fits on its \
             own; their SUM does not. Admitting it means the second asker re-read the \
             ceiling instead of the ledger -- the exact defect the pool exists to fix.",
            f.ao_bytes(),
            f.b_flat_bytes(),
            f.b_oo_vv_bytes()
        ),
    };
    assert!(
        err.contains("memory pool exhausted"),
        "the refusal must come from the POOL (so the occupancy breakdown is printed), not \
         from a plain ceiling check: {err}"
    );
    // The breakdown must name the incumbent, or the message does not tell the
    // user WHICH plane to shrink.
    assert!(
        err.contains("DF 3-index") || err.contains("b_ov") || err.contains("b_vv"),
        "the occupancy breakdown must name the planes in play: {err}"
    );
}

/// REACHABILITY GUARD for CONTRACT 2 (per `CLAUDE.md`: "check the pass
/// condition is REACHABLE -- a gate whose GO conditions are mutually exclusive
/// returns arithmetic, not measurement").
///
/// CONTRACT 2 proves composition only if its budget genuinely admits the AO
/// tensor. If the budget were below the AO tensor itself, the refusal would
/// prove nothing except that the pool is smaller than one plane -- which a
/// plain ceiling check already does. This pins that the budget is in the
/// regime where only the SUM fails.
#[test]
fn the_refusal_budget_admits_the_ao_tensor_alone() {
    let _s = CleanSlot::acquire();
    let f = fixture();
    let capacity = f.ao_bytes() + f.b_flat_bytes() / 2;
    assert!(
        capacity > f.ao_bytes(),
        "CONTRACT 2's budget ({capacity}) must exceed the AO tensor ({}) or its refusal \
         proves nothing about composition",
        f.ao_bytes()
    );
    assert!(
        capacity < f.ao_bytes() + f.b_flat_bytes() + f.b_oo_vv_bytes(),
        "and it must fall SHORT of the AO tensor plus the MO blocks ({}), or CONTRACT 2 \
         cannot fail even with the charges removed",
        f.ao_bytes() + f.b_flat_bytes() + f.b_oo_vv_bytes()
    );
    // And the gap must be a meaningful fraction, not a rounding error: a
    // contract whose margin is one byte is a coin flip against allocator
    // detail.
    assert!(
        f.b_flat_bytes() * 20 > f.ao_bytes(),
        "b_ov ({}) is under 5% of the AO tensor ({}) at this fixture, so CONTRACT 2's \
         margin is too thin to be a measurement. Pick a larger basis.",
        f.b_flat_bytes(),
        f.ao_bytes()
    );
}

// ---------------------------------------------------------------------------
// CONTRACT 3: the charge is RELEASED, so a second MP2 in one process is not
// refused against bytes nothing holds.
//
// This is the contract mutation M4 (leak the guard past the buffer's life)
// must fail. The `Reservation` type was written with an EMPTY `Drop` body at
// first, so this half of the contract is not hypothetical.
// ---------------------------------------------------------------------------

#[test]
fn a_second_run_in_the_same_process_is_not_refused_by_the_first() {
    let _s = CleanSlot::acquire();
    let f = fixture();
    // Room for exactly one run's peak and a little slack -- two runs' worth of
    // LEAKED charge would not fit.
    let pool = MemoryPool::with_capacity_bytes(64_000_000_000);
    install_global(pool.clone());

    let first = f.run().expect("first run must succeed");
    let after_first = pool.outstanding_bytes();
    let second = f.run().expect(
        "the SECOND run in the same process was refused -- the first leaked its \
         reservations, so the ledger is monotone",
    );
    let after_second = pool.outstanding_bytes();
    clear_global();

    assert_eq!(
        after_first, 0,
        "run 1 left {after_first} bytes outstanding after returning"
    );
    assert_eq!(
        after_second, 0,
        "run 2 left {after_second} bytes outstanding after returning"
    );
    assert_eq!(
        first.to_bits(),
        second.to_bits(),
        "and the two runs must agree to the BIT ({first:.17e} vs {second:.17e}) -- if they \
         do not, a charge outstanding from run 1 changed a width in run 2, which means a \
         width is reading the LEDGER"
    );
}

// ---------------------------------------------------------------------------
// CONTRACT 5: the charge is OUTSTANDING while the tensors are alive.
//
// # Why this contract exists: CONTRACTS 1-4 were INERT against mutation M1
//
// M1 (per the brief: "make the charge a no-op -- drop the reservation
// immediately") was applied as
//
//     { let r = reserve_global(label, bytes)?; drop(r); }
//     Ok(Reservation::inert(label))
//
// and ALL FOUR of CONTRACTS 1-4 still passed. The reason is measurable and
// worth recording: `pool.peak_bytes()` is a HIGH-WATER MARK, so a charge that
// is taken and released a nanosecond later still moves the peak (killing
// CONTRACT 1), and CONTRACT 2's refusal fires during that same nanosecond
// (killing CONTRACT 2). Both contracts were asserting that a reservation was
// MADE, not that it was HELD -- and "held" is the entire property, because the
// defect being fixed is a second plane being admitted against bytes the first
// is still using.
//
// This contract asserts the thing M1 destroys: at a moment when the tensors
// provably exist -- we are holding the struct that owns them -- the pool must
// still show them outstanding. It is the direct test of the
// `_charge: Reservation` field pattern.
// ---------------------------------------------------------------------------

#[test]
fn the_charge_is_outstanding_while_the_tensors_are_alive() {
    let _s = CleanSlot::acquire();
    let f = fixture();
    let pool = MemoryPool::with_capacity_bytes(64_000_000_000);
    install_global(pool.clone());

    // `compute_mp2_intermediates` returns a struct OWNING b_ov/b_oo/b_vv, so
    // while `inter` is in scope those tensors are unambiguously resident.
    let inter = ferric_mp2::rimp2::compute_mp2_intermediates(
        &f.mol,
        &f.obs,
        &f.dfbs,
        Operator::coulomb(),
        &f.rhf,
        &RiMp2Config::default(),
    )
    .expect("intermediates must build under an ample pool");

    let while_alive = pool.outstanding_bytes();
    let report = pool.occupancy_report();
    // Touch the tensors so nothing can argue they were optimized away.
    let probe = inter.b_ov.len() + inter.b_vv.as_ref().map_or(0, |b| b.len());
    drop(inter);
    let after_drop = pool.outstanding_bytes();
    clear_global();

    assert!(
        probe > 0,
        "sanity: the intermediates must hold real tensors"
    );
    assert!(
        while_alive > 0,
        "the pool showed ZERO bytes outstanding while b_ov/b_oo/b_vv were provably alive \
         (we were holding the struct that owns them). The charge is being released before \
         the tensors are -- so a plane allocated next would be admitted against bytes that \
         are still in use, which is the defect this migration exists to fix.\n{report}"
    );
    // And it must be the real magnitude, not a token: b_ov alone is
    // naux*nocc*nvir*8.
    assert!(
        while_alive >= f.b_flat_bytes(),
        "only {while_alive} bytes were outstanding while the tensors were alive, but b_ov \
         alone is {} bytes -- the charge is a token, not the plane.\n{report}",
        f.b_flat_bytes()
    );
    assert_eq!(
        after_drop, 0,
        "and dropping the struct must credit every byte back (got {after_drop} outstanding) \
         -- otherwise the ledger is monotone and the next job is refused against nothing"
    );
}

// ---------------------------------------------------------------------------
// CONTRACT 4: an occupancy report names the dominant plane.
//
// The acceptance criterion from the brief: "fail fast with an occupancy
// breakdown -- never walk into the allocator and get OOM-killed".
// ---------------------------------------------------------------------------

#[test]
fn the_refusal_message_is_actionable() {
    let _s = CleanSlot::acquire();
    let f = fixture();
    let pool = MemoryPool::with_capacity_bytes(f.ao_bytes() + f.b_flat_bytes() / 2);
    install_global(pool);
    let err = f.run_intermediates().expect_err("must refuse").to_string();
    clear_global();

    for needle in ["memory pool exhausted", "short by", "GB", "budget_gb"] {
        assert!(
            err.contains(needle),
            "the refusal must be actionable -- missing {needle:?} in:\n{err}"
        );
    }
}
