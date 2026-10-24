//! CAS-CI's two memory gates must compose through the shared pool.
//!
//! # The defect
//!
//! Both gates (`build_active_space_integrals`' transform chain and
//! `run_cas_ci`'s Davidson subspace) were `check_alloc` against the whole
//! resolved ceiling. A pool installed by the caller was invisible to them:
//! bytes another subsystem already held did not narrow their room, and their
//! own bytes were never debited, so a later plane could not see them either.
//! Both now go through `MemoryPlan::from_global_pool(..).commit()`.
//!
//! # Artifact hypothesis (stated before measuring)
//!
//! If the gates COMPOSE, an outside incumbent that leaves just under the
//! Davidson peak free must refuse the run with the Davidson label, while the
//! transform chain (smaller, earlier) still fits; and an ample pool's
//! high-water must reach the Davidson peak. If they are ceiling checks, the
//! incumbent is invisible (the run is admitted) and the pool never sees the
//! Davidson bytes (high-water stays at the transform chain or zero). The
//! predictions differ, so the tests can tell them apart.

use std::sync::{Mutex, MutexGuard, OnceLock};

use ferric_ci::integrals::active_space_integral_elems;
use ferric_ci::{run_cas_ci, CasCiConfig, CasCiResult};
use ferric_core::memory::pool::{clear_global, install_global, reserve_global, MemoryPool};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::ScfResult;

const AMPLE: usize = 64_000_000_000;

fn global_lock() -> MutexGuard<'static, ()> {
    static L: OnceLock<Mutex<()>> = OnceLock::new();
    match L.get_or_init(|| Mutex::new(())).lock() {
        Ok(g) => g,
        Err(e) => e.into_inner(),
    }
}

/// Clear the process-global pool on entry AND exit, so one test's pool can
/// never refuse a sibling (see ferric-mp2's `mwe_mp2_planes_compose_in_one_pool.rs`).
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
    prep: PreparedBasis,
    rhf: ScfResult,
}

/// H2O/STO-3G, CAS(8e, 6o) above a frozen O 1s: N_det = C(6,4)^2 = 225.
fn fixture() -> Fixture {
    let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").expect("water.xyz");
    let bs = ferric_core::basis::bundled("sto-3g").expect("sto-3g");
    let prep = PreparedBasis::new(&mol, &bs).expect("prepared basis");
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).expect("schwarz");
    let rhf = solve_rhf(
        &ParallelContext::default(),
        &mol,
        &prep,
        op,
        &bounds,
        &RhfConfig::default(),
    )
    .expect("rhf");
    Fixture { mol, prep, rhf }
}

impl Fixture {
    fn config(&self, memory_budget_bytes: Option<usize>) -> CasCiConfig {
        CasCiConfig {
            n_active: 6,
            n_elec_active: (4, 4),
            active_start: 1,
            memory_budget_bytes,
            ..Default::default()
        }
    }
    fn run(&self, memory_budget_bytes: Option<usize>) -> Result<CasCiResult, String> {
        run_cas_ci(
            &self.mol,
            &self.prep,
            &self.rhf,
            self.config(memory_budget_bytes),
        )
        .map_err(|e| e.to_string())
    }
    /// The Davidson plan's peak: two bases of max_subspace vectors + 5 work
    /// vectors, all N_det doubles.
    fn davidson_peak(&self) -> usize {
        let ms = self.config(None).max_subspace.max(1);
        225 * (2 * ms + 5) * 8
    }
    /// The transform chain's peak (ao_eri + mo_eri + t1 + t2 + t3).
    fn transform_peak(&self) -> usize {
        active_space_integral_elems(self.prep.nbasis(), 1 + 6) * 8
    }
    /// Run under a 64 GB pool with an incumbent that leaves `room` bytes free.
    fn run_with_room(&self, room: usize) -> Result<CasCiResult, String> {
        let pool = MemoryPool::with_capacity_bytes(AMPLE);
        install_global(pool);
        let incumbent = reserve_global("incumbent", AMPLE - room).expect("incumbent");
        let out = self.run(None);
        drop(incumbent);
        clear_global();
        out
    }
}

fn expect_refusal(out: Result<CasCiResult, String>, what: &str) -> String {
    match out {
        Ok(r) => panic!("{what}: admitted ({} dets)", r.n_determinants),
        Err(e) => e,
    }
}

/// Inertness anchor: an ample pool changes no bit of the energy.
#[test]
fn an_ample_pool_does_not_move_the_casci_energy_one_bit() {
    let _s = CleanSlot::acquire();
    let f = fixture();
    let no_pool = f.run(None).expect("no pool must run");
    assert_eq!(no_pool.n_determinants, 225);
    let pool = MemoryPool::with_capacity_bytes(AMPLE);
    install_global(pool);
    let with_pool = f.run(None).expect("an ample pool must admit");
    clear_global();
    assert_eq!(no_pool.e_total.to_bits(), with_pool.e_total.to_bits());
    assert_eq!(no_pool.iterations, with_pool.iterations);
}

/// Composes: an incumbent leaving just under the Davidson peak free refuses
/// at the Davidson gate (the transform chain, which fits, ran first).
#[test]
fn an_incumbent_narrows_the_davidson_gate() {
    let _s = CleanSlot::acquire();
    let f = fixture();
    let room = f.davidson_peak() - 1;
    assert!(
        room > f.transform_peak(),
        "the room must admit the transform chain so the Davidson gate is reached"
    );
    let err = expect_refusal(f.run_with_room(room), "Davidson room - 1 B");
    assert!(err.contains("CAS-CI Davidson subspace"), "{err}");
}

/// Same for the earlier gate: just under the transform peak refuses there.
#[test]
fn an_incumbent_narrows_the_transform_chain_gate() {
    let _s = CleanSlot::acquire();
    let f = fixture();
    let err = expect_refusal(
        f.run_with_room(f.transform_peak() - 1),
        "transform room - 1 B",
    );
    assert!(err.contains("CAS-CI dense AO ERIs"), "{err}");
}

/// Reachability guard for both refusals: exactly the larger peak free admits
/// (the transform charge is released before Davidson commits).
#[test]
fn room_for_the_larger_peak_admits() {
    let _s = CleanSlot::acquire();
    let f = fixture();
    let room = f.davidson_peak().max(f.transform_peak());
    let r = f
        .run_with_room(room)
        .expect("room for the larger of the two non-overlapping peaks must admit");
    assert!(r.converged);
}

/// An explicit budget still binds under an ample pool (`with_pool` takes the min).
#[test]
fn an_explicit_budget_still_binds_under_an_ample_pool() {
    let _s = CleanSlot::acquire();
    let f = fixture();
    let pool = MemoryPool::with_capacity_bytes(AMPLE);
    install_global(pool);
    let out = f.run(Some(f.davidson_peak() - 1));
    clear_global();
    let err = expect_refusal(out, "explicit budget Davidson - 1 B");
    assert!(err.contains("CAS-CI Davidson subspace"), "{err}");
}

/// The pool is DEBITED (high-water reaches each plane's peak) and every
/// reservation is released when the run returns.
#[test]
fn both_planes_are_debited_and_released() {
    let _s = CleanSlot::acquire();
    let f = fixture();
    let pool = MemoryPool::with_capacity_bytes(AMPLE);
    install_global(pool.clone());
    let r = f.run(None).expect("ample pool");
    assert!(r.converged);
    let (outstanding, peak) = (pool.outstanding_bytes(), pool.peak_bytes());
    clear_global();
    assert_eq!(outstanding, 0, "a reservation outlived run_cas_ci");
    assert_eq!(
        peak,
        f.davidson_peak().max(f.transform_peak()),
        "pool high-water must be the larger committed plane (Davidson {}, transform {})",
        f.davidson_peak(),
        f.transform_peak()
    );
}

/// The transform chain alone debits its peak. Run in isolation because in a
/// full CAS-CI the larger Davidson plane sets the high-water and would hide a
/// transform gate that only checks.
#[test]
fn the_transform_chain_is_debited_on_its_own() {
    let _s = CleanSlot::acquire();
    let f = fixture();
    let pool = MemoryPool::with_capacity_bytes(AMPLE);
    install_global(pool.clone());
    let ints =
        ferric_ci::integrals::build_active_space_integrals(&f.mol, &f.prep, &f.rhf, 1, 6, None);
    let (outstanding, peak) = (pool.outstanding_bytes(), pool.peak_bytes());
    clear_global();
    assert!(ints.is_ok(), "ample pool must admit the transform chain");
    assert_eq!(
        outstanding, 0,
        "the transform reservation outlived the call"
    );
    assert_eq!(peak, f.transform_peak(), "transform chain not debited");
}
