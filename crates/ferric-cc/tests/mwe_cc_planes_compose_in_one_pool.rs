//! The property the pre-pool ferric-cc tree did not have: two CC planes that
//! each fit the ceiling individually must NOT both be admitted when their sum
//! exceeds it.
//!
//! # The defect, in ferric-cc's own terms
//!
//! Before this migration every ferric-cc guard asked the same question of the
//! same full number:
//!
//! ```text
//!   ccsd.rs:189          plan.check()       "does 2*(2nv)^4 + DIIS fit in B?"
//!   ccsd_t.rs:412        check_alloc(..)    "does bcei+majk+bcjk fit in B?"
//! ```
//!
//! A CCSD+(T) job runs those back to back, and the (T) step runs while the
//! CCSD driver's amplitudes and DIIS ring are still resident. Both gates
//! passed against `B`; the process held the sum. This is the same shape as the
//! measured KS-DFT incident (budget 4.72 GiB printed, MAXRSS 6.04 GiB, OOM
//! kill, not one gate failed) -- ferric-cc simply has the largest tensors in
//! the repo, so it has the most exposure to it.
//!
//! # What is asserted here
//!
//! 1. A pool sized between "one plane" and "two planes" admits the first and
//!    REFUSES the second. Under a plain ceiling both would pass.
//! 2. The refusal names the plane and shows the occupancy breakdown, so the
//!    report says WHICH plane is dominant rather than a bare total.
//! 3. The charge is RELEASED when the driver returns, so a loop over many CC
//!    jobs does not exhaust the pool on the second one. (A migration that
//!    charged without releasing would look correct on a single run and fail
//!    on any sequence -- the `Drop` body of `Reservation` was empty while it
//!    was being written, for exactly this reason.)
//! 4. Charging is not decoration: the pool's `peak_bytes()` after a run is
//!    nonzero and at least the plan's projected peak.

use std::sync::{Mutex, MutexGuard, OnceLock};

use ferric_cc::ccsd::{ccsd, ccsd_memory_plan, CcsdShape};
use ferric_cc::{ccsd_t::ccsd_t, CcConfig};
use ferric_core::basis;
use ferric_core::memory::pool::{clear_global, global, install_global, MemoryPool};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::result::ScfResult;
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

fn cfg() -> CcConfig {
    CcConfig {
        frozen_core: 0,
        max_iter: 30,
        ..Default::default()
    }
}

/// The projected peak the CCSD driver charges on this fixture, taken from the
/// driver's OWN plan rather than reconstructed by hand -- a test that
/// recomputes a threshold is a second implementation and therefore a drift
/// surface.
fn ccsd_peak_bytes() -> usize {
    ccsd_memory_plan(
        CcsdShape {
            no: 5,
            nv: 2,
            nbas: 7,
            naux: 70,
            diis_subspace: CcConfig::default().diis_subspace,
        },
        Some(usize::MAX / 4),
    )
    .peak_bytes()
}

#[test]
fn two_cc_planes_summing_past_the_pool_cannot_both_be_admitted() {
    let _s = CleanSlot::acquire();
    let peak = ccsd_peak_bytes();
    assert!(peak > 0, "fixture is broken: CCSD plan has a zero peak");

    // A pool that fits ONE such plane with room to spare but not two. Under
    // the pre-pool ceiling both would pass, because both asked the same
    // question of the same full number.
    let pool = MemoryPool::with_capacity_bytes(peak + peak / 2);
    install_global(pool.clone());

    // A resident incumbent standing in for whatever else the process is
    // holding when the CC driver starts (in the real incident: the DF 3-index
    // tensor and the KS grid AO cache).
    let _incumbent = pool
        .reserve("incumbent plane", peak)
        .expect("the first plane must fit");

    let f = fixture();
    let err = match ccsd(&f.mol, &f.obs, &f.dfbs, Operator::coulomb(), &f.rhf, &cfg()) {
        Err(e) => e.to_string(),
        Ok(_) => panic!(
            "CCSD was ADMITTED with only {:.0} bytes free against a {peak}-byte \
             peak -- the gate is re-reading a ceiling, not debiting a ledger",
            pool.available_bytes()
        ),
    };

    // The refusal must be actionable: it names the plane and shows who is
    // holding what. A bare "needs X GB" is what made the historical incidents
    // slow to diagnose.
    assert!(
        err.contains("CCSD"),
        "the refusal must name the refused plane: {err}"
    );
    assert!(
        err.contains("incumbent plane") || err.contains("memory plan"),
        "the refusal must carry an occupancy or plan breakdown: {err}"
    );

    clear_global();
}

#[test]
fn the_same_job_is_admitted_once_the_incumbent_releases() {
    // The other half of the property: refusing must be about OCCUPANCY, not
    // about the job. If this failed, the gate would be refusing a job that
    // fits -- an over-estimating guard, which is also a bug.
    let _s = CleanSlot::acquire();
    let peak = ccsd_peak_bytes();
    let pool = MemoryPool::with_capacity_bytes(peak + peak / 2);
    install_global(pool.clone());
    let f = fixture();

    {
        let _incumbent = pool.reserve("incumbent plane", peak).expect("first fits");
        assert!(
            ccsd(&f.mol, &f.obs, &f.dfbs, Operator::coulomb(), &f.rhf, &cfg()).is_err(),
            "must be refused while the incumbent holds the pool"
        );
    } // incumbent released here

    assert_eq!(
        pool.outstanding_bytes(),
        0,
        "the incumbent must have released on drop"
    );
    let _ = ccsd(&f.mol, &f.obs, &f.dfbs, Operator::coulomb(), &f.rhf, &cfg())
        .expect("the very same job must be ADMITTED once the pool is free again");
    clear_global();
}

#[test]
fn a_cc_driver_releases_its_charge_when_it_returns() {
    // LIFETIME. A charge that outlives its tensors turns a long job into a
    // slow leak: the second CC call in a loop would be refused although the
    // first one's memory is long gone. `Reservation::drop` was an empty body
    // while it was being written, so this is a measured failure mode, not a
    // hypothetical one.
    let _s = CleanSlot::acquire();
    let peak = ccsd_peak_bytes();
    // Room for exactly one plane at a time.
    let pool = MemoryPool::with_capacity_bytes(peak + peak / 2);
    install_global(pool.clone());
    let f = fixture();

    for round in 0..3 {
        let _ = ccsd(&f.mol, &f.obs, &f.dfbs, Operator::coulomb(), &f.rhf, &cfg()).unwrap_or_else(
            |e| panic!("round {round} was refused, so a previous round leaked its charge: {e}"),
        );
        assert_eq!(
            pool.outstanding_bytes(),
            0,
            "round {round} left {} bytes outstanding after returning",
            pool.outstanding_bytes()
        );
    }
    clear_global();
}

#[test]
fn the_charge_is_not_decoration_the_peak_is_recorded() {
    // A gate that debits zero would pass every test above. `peak_bytes()` is
    // the high-water mark of the ledger, so it is nonzero only if something
    // was really charged.
    let _s = CleanSlot::acquire();
    install_global(MemoryPool::with_capacity_bytes(64_000_000_000));
    let f = fixture();
    let cc = ccsd(&f.mol, &f.obs, &f.dfbs, Operator::coulomb(), &f.rhf, &cfg())
        .expect("ample pool must admit");
    // (T) too, so both migrated shapes are exercised.
    ccsd_t(
        &f.mol,
        &f.obs,
        &f.dfbs,
        Operator::coulomb(),
        &f.rhf,
        &cc,
        &cfg(),
    )
    .expect("ample pool must admit (T)");

    let pool = global().expect("pool installed");
    let peak = pool.peak_bytes();
    let expected = ccsd_peak_bytes();
    assert!(
        peak >= expected,
        "the pool recorded a peak of {peak} bytes, below the CCSD plan's own \
         projected peak of {expected} -- the charge is decoration"
    );
    assert_eq!(
        pool.outstanding_bytes(),
        0,
        "everything must have been released by the time both drivers returned"
    );
    clear_global();
}
