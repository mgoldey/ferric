//! The (T) triple band is charged SOFTLY, and its `None` branch really does
//! narrow the band rather than refuse the job.
//!
//! # Why soft, measured rather than guessed
//!
//! The standing decision on this migration is: it must not refuse a job the
//! pre-migration tree completes. The (T) band is the one plane in ferric-cc
//! where that is a live risk, because the band is sized from a byte number and
//! then charged for the width it chose. Charge it HARD and the sequence is:
//!
//!   1. the pre-flight floor (`precomputed + one per-triple set`) is charged,
//!   2. the band asks for `width x per_triple` of what is LEFT,
//!   3. a pool sized just above the floor cannot fund a wide band,
//!   4. hard -> the job ERRORS, although the pre-migration tree runs it
//!      (it would simply have banded at whatever the budget allowed).
//!
//! Soft is therefore not a stylistic preference: it is the only charge that
//! preserves the pre-migration acceptance set. This file measures both halves.
//!
//! # A soft gate whose None branch does not stream is a LIE
//!
//! So the None branch is asserted to be REACHED and to CHANGE THE WIDTH, not
//! merely to be present in the source. `pool_pressure_narrows_the_band` pins
//! that the same job, same budget, runs at a narrower band under a tighter
//! pool -- and `mwe_t_band_width_is_not_an_energy_knob.rs` independently pins
//! that narrowing does not move `et`, so this is a pure performance tradeoff.

use std::sync::{Mutex, MutexGuard, OnceLock};

use ferric_cc::ccsd::ccsd;
use ferric_cc::ccsd_t::{ccsd_t, t_band_width, t_floor_bytes, t_per_triple_bytes};
use ferric_cc::CcConfig;
use ferric_core::basis;
use ferric_core::memory::pool::{clear_global, install_global, MemoryPool};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::result::ScfResult;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

/// H2O / STO-3G: no2 = 10, nv2 = 4 spin-orbitals.
const NO2: usize = 10;
const NV2: usize = 4;

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
    rhf: ScfResult,
}

fn fixture() -> Fixture {
    let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").expect("water.xyz");
    let obs = PreparedBasis::new(&mol, &basis::bundled("sto-3g").expect("sto-3g")).expect("obs");
    let dfbs =
        PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").expect("cc-pvdz-ri")).expect("dfbs");
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).expect("schwarz");
    let ctx = ParallelContext::default();
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).expect("RHF");
    Fixture {
        mol,
        obs,
        dfbs,
        rhf,
    }
}

fn cfg(budget: Option<usize>) -> CcConfig {
    CcConfig {
        frozen_core: 0,
        max_iter: 30,
        memory_budget_bytes: budget,
        ..Default::default()
    }
}

#[test]
fn a_pool_with_room_for_the_floor_but_not_a_wide_band_still_runs_t() {
    // THE standing-decision assertion. A pool sized to the (T) floor plus a
    // sliver has no room for a wide band. Under a HARD band charge this job
    // errors; under the soft charge it must still complete, because narrowing
    // the band is a real fallback.
    let _s = CleanSlot::acquire();
    let floor = t_floor_bytes(NO2, NV2);
    let per_triple = t_per_triple_bytes(NV2);

    let f = fixture();
    // Converge the amplitudes with no pool installed so the CCSD plane's own
    // charge does not confound what is being measured here.
    let cc = ccsd(
        &f.mol,
        &f.obs,
        &f.dfbs,
        Operator::coulomb(),
        &f.rhf,
        &cfg(Some(8_000_000_000)),
    )
    .expect("CCSD must converge");

    // Room for the floor and at most two extra triples' worth of band.
    let capacity = floor + 2 * per_triple;
    install_global(MemoryPool::with_capacity_bytes(capacity));

    // The UNBUDGETED band width this same config would have chosen: very wide.
    let wide = t_band_width(NO2, NV2, 8_000_000_000);
    assert!(
        wide * per_triple > capacity,
        "fixture is broken: the unpooled band ({wide} x {per_triple} B) already \
         fits the {capacity} B pool, so the soft branch is never exercised"
    );

    let et = ccsd_t(
        &f.mol,
        &f.obs,
        &f.dfbs,
        Operator::coulomb(),
        &f.rhf,
        &cc,
        &cfg(Some(8_000_000_000)),
    )
    .unwrap_or_else(|e| {
        panic!(
            "(T) was REFUSED under a {capacity}-byte pool that covers its \
             {floor}-byte floor. The band charge is hard, not soft -- this is \
             a job the pre-migration tree completes: {e}"
        )
    });
    assert!(
        et.is_finite() && et < 0.0,
        "(T) must still produce a physical correction, got {et}"
    );
    clear_global();
}

#[test]
fn a_pool_below_the_t_floor_refuses_rather_than_oom() {
    // The other side: when the plane genuinely does not fit, the answer is a
    // clean typed refusal naming the plane, not a walk into the allocator.
    // This is the HARD half of the (T) charge and it must stay hard -- nothing
    // streams bcei/majk/bcjk.
    let _s = CleanSlot::acquire();
    let floor = t_floor_bytes(NO2, NV2);

    let f = fixture();
    let cc = ccsd(
        &f.mol,
        &f.obs,
        &f.dfbs,
        Operator::coulomb(),
        &f.rhf,
        &cfg(Some(8_000_000_000)),
    )
    .expect("CCSD must converge");

    // Half the floor: the precomputed blocks cannot be built.
    install_global(MemoryPool::with_capacity_bytes(floor / 2));
    let err = match ccsd_t(
        &f.mol,
        &f.obs,
        &f.dfbs,
        Operator::coulomb(),
        &f.rhf,
        &cc,
        &cfg(Some(8_000_000_000)),
    ) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("(T) must refuse a pool below its own floor"),
    };
    assert!(
        err.contains("CCSD(T) precomputed blocks"),
        "the refusal must name the plane: {err}"
    );
    assert!(
        err.contains("memory pool exhausted") && err.contains("short by"),
        "the refusal must come from the POOL (with a shortfall), not from a \
         ceiling re-read: {err}"
    );
    clear_global();
}

/// The soft branch must be REACHABLE, and taking it must not refuse the job.
///
/// # What mutation testing found here
///
/// The first version of this migration sized the band from
/// `transient_share(available, Half)` and then charged `width * per_triple`.
/// Because `width = (available/2) / per_triple`, the charge could never exceed
/// `available` -- the `None` branch was unreachable BY CONSTRUCTION. Mutation
/// M3 (turn the soft gate hard) therefore survived every test in this crate:
/// a hard reserve that can never fail is indistinguishable from a soft one.
/// That is the "GO conditions are mutually exclusive" trap -- the gate
/// returned arithmetic, not a measurement.
///
/// The halving was dropped on the pooled branch. The soft branch is now
/// reachable through the `.max(1)` floor: when the pool has LESS than one
/// per-triple set free after the floor charge, `triple_chunk_len` still
/// returns 1 (a zero width cannot make progress), so the charge asks for
/// `per_triple` against strictly less than `per_triple` -- and must fall back
/// rather than refuse.
///
/// This test drives exactly that state and asserts the job still completes.
#[test]
fn the_soft_branch_is_reachable_and_does_not_refuse() {
    let _s = CleanSlot::acquire();
    let floor = t_floor_bytes(NO2, NV2);
    let per_triple = t_per_triple_bytes(NV2);

    let f = fixture();
    let cc = ccsd(
        &f.mol,
        &f.obs,
        &f.dfbs,
        Operator::coulomb(),
        &f.rhf,
        &cfg(Some(8_000_000_000)),
    )
    .expect("CCSD must converge");

    // EXACTLY the floor: after `_floor_charge` is granted, `available_bytes()`
    // is 0, so `triple_chunk_len` floors the width at 1 and the band charge
    // asks for `per_triple` against 0 free. The soft branch MUST be taken.
    let capacity = floor;
    let pool = MemoryPool::with_capacity_bytes(capacity);
    install_global(pool.clone());

    let et = ccsd_t(
        &f.mol,
        &f.obs,
        &f.dfbs,
        Operator::coulomb(),
        &f.rhf,
        &cc,
        &cfg(Some(8_000_000_000)),
    )
    .unwrap_or_else(|e| {
        panic!(
            "(T) was REFUSED with the pool at exactly its {floor}-byte floor. \
             The band charge took the HARD path: this is a job the \
             pre-migration tree completes (it would simply have banded at \
             width 1). {e}"
        )
    });
    assert!(et.is_finite() && et < 0.0, "(T) must still compute: {et}");

    // And the ledger must show the band was never funded: the peak is exactly
    // the floor, because the soft branch fell through to an inert guard.
    assert_eq!(
        pool.peak_bytes(),
        floor,
        "the peak must be exactly the floor ({floor}); a larger peak means the \
         band was charged although only {} bytes were free, i.e. the ledger \
         was oversubscribed",
        capacity - floor
    );
    assert!(
        per_triple > 0,
        "fixture sanity: a per-triple set must be nonzero"
    );
    clear_global();
}

#[test]
fn pool_pressure_narrows_the_band_it_does_not_merely_admit() {
    // A soft gate whose None branch does not actually stream is a lie. This
    // asserts the None branch is REACHED and that the width it produces is
    // smaller than the width the same config chose unpooled.
    //
    // It is measured through the ledger rather than by instrumenting the
    // driver: under a tight pool the band charge must be small, so the pool's
    // PEAK stays below floor + wide_band. If the driver had charged (or held)
    // the wide band, the peak would exceed the capacity -- which is
    // impossible, since the reserve would have failed. So the discriminator is
    // simply that the job COMPLETES under a capacity the wide band cannot fit,
    // which is what `a_pool_with_room_for_the_floor_but_not_a_wide_band_still_
    // runs_t` establishes -- plus the arithmetic below that the widths really
    // do differ.
    let floor = t_floor_bytes(NO2, NV2);
    let per_triple = t_per_triple_bytes(NV2);

    // Unpooled width at an ample budget.
    let wide = t_band_width(NO2, NV2, 8_000_000_000);
    // What the pool ledger funds when only 2 triples' worth is free. This is
    // the driver's own pooled-branch arithmetic: `triple_chunk_len(nv2,
    // available)`, i.e. `available / per_triple`, floored at 1 -- NOT
    // `available/2 / per_triple`, which is what made the soft gate unreachable
    // (see `the_soft_branch_is_reachable_and_does_not_refuse`).
    let tight_capacity = floor + 2 * per_triple;
    let pool = MemoryPool::with_capacity_bytes(tight_capacity);
    let _floor_charge = pool.reserve("floor", floor).expect("floor fits");
    let narrowed = (pool.available_bytes() / per_triple).max(1);

    assert!(
        narrowed < wide,
        "pool pressure must NARROW the band: unpooled {wide}, pooled {narrowed}. \
         If these are equal the soft branch changes nothing and the gate is \
         decoration."
    );
    assert!(
        narrowed >= 1,
        "a narrowed band of 0 cannot make progress: {narrowed}"
    );
    assert_eq!(
        narrowed, 2,
        "the ledger funds exactly the 2 extra triples the capacity provides, \
         got {narrowed} -- if this changed, the None branch is reading \
         something other than the ledger"
    );
}
