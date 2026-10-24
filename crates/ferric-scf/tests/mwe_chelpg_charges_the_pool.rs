//! The CHELPG grid gate must charge the shared pool, not only the ceiling.
//!
//! Separate test binary on purpose: an installed process-global pool would
//! otherwise be visible to (and could refuse) unrelated in-module SCF tests
//! running concurrently in the lib harness.
//!
//! The charge is MEASURED (pool high-water of one ample run) rather than
//! recomputed here; the in-module test `chelpg_peak_counts_kept_values_and_output`
//! pins the 64 B/pt figure itself.

use std::sync::{Mutex, MutexGuard, OnceLock};

use ferric_core::basis;
use ferric_core::memory::pool::{clear_global, install_global, reserve_global, MemoryPool};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::properties::chelpg_charges;
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

/// Clear the process-global pool on entry AND exit.
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

/// Water/STO-3G RHF, converged BEFORE any pool is installed.
fn water() -> (Molecule, PreparedBasis, ScfResult) {
    let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").expect("water.xyz");
    let prep = PreparedBasis::new(&mol, &basis::bundled("sto-3g").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let rhf = solve_rhf(
        &ParallelContext::default(),
        &mol,
        &prep,
        op,
        &bounds,
        &RhfConfig::default(),
    )
    .unwrap();
    (mol, prep, rhf)
}

/// CHELPG under a 64 GB pool whose incumbent leaves `room` bytes free.
fn run_with_room(
    mol: &Molecule,
    prep: &PreparedBasis,
    rhf: &ScfResult,
    room: usize,
) -> Result<Vec<f64>, String> {
    install_global(MemoryPool::with_capacity_bytes(AMPLE));
    let incumbent = reserve_global("incumbent", AMPLE - room).expect("incumbent");
    let out = chelpg_charges(mol, prep, rhf.density_r()).map_err(|e| e.to_string());
    drop(incumbent);
    clear_global();
    out
}

/// Pin every CHELPG grid knob to its documented default so the candidate
/// count below does not move with the caller's environment, and pin the
/// ambient memory budget (`chelpg_charges` takes no explicit one) to an
/// ample 64 GiB so the CHELPG ceiling gate never refuses: the unified var
/// outranks the legacy ones and host auto-detection in `resolve_budget`.
/// Safe here: this binary holds a single test, and the pin runs before the
/// SCF spawns any worker thread that could read the environment concurrently.
fn pin_chelpg_settings() {
    for (k, v) in [
        ("FERRIC_CHELPG_SPACING", "0.3"),
        ("FERRIC_CHELPG_MARGIN", "2.8"),
        ("FERRIC_CHELPG_VDW_SCALE", "1.0"),
        ("FERRIC_CHELPG_OUTER_CUTOFF", "2.8"),
        (ferric_core::memory::ENV_UNIFIED, "64"),
    ] {
        std::env::set_var(k, v);
    }
}

#[test]
fn chelpg_is_charged_at_its_peak_refused_one_byte_short_and_released() {
    let _s = CleanSlot::acquire();
    pin_chelpg_settings();
    let (mol, prep, rhf) = water();

    // Measure the charge, and check the pool is inert to the result.
    let ambient = chelpg_charges(&mol, &prep, rhf.density_r()).unwrap();
    let pool = MemoryPool::with_capacity_bytes(AMPLE);
    install_global(pool.clone());
    let pooled = chelpg_charges(&mol, &prep, rhf.density_r()).unwrap();
    let (outstanding, peak) = (pool.outstanding_bytes(), pool.peak_bytes());
    clear_global();
    let bits = |q: &[f64]| q.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(
        bits(&ambient),
        bits(&pooled),
        "an ample pool moved a charge"
    );
    assert_eq!(outstanding, 0, "the CHELPG reservation outlived the call");
    // At the pinned spacing 0.3 / margin 2.8 Bohr, water's CHELPG grid is
    // 16,275 candidate points (in-module guard test); the charge is 64 B for
    // each.
    assert_eq!(peak, 64 * 16_275, "pool high-water is not the CHELPG peak");

    // One byte short of the peak refuses at the CHELPG gate.
    let err = match run_with_room(&mol, &prep, &rhf, peak - 1) {
        Ok(q) => panic!(
            "{} B free admitted a {peak} B CHELPG charge: {q:?}",
            peak - 1
        ),
        Err(e) => e,
    };
    assert!(err.contains("chelpg candidate grid"), "{err}");

    // Reachability: room for exactly the peak admits.
    let q = run_with_room(&mol, &prep, &rhf, peak).expect("room for the full peak");
    assert_eq!(bits(&q), bits(&ambient));
}
