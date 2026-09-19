//! The RI-MP2 gradient's own MO-side planes are DEBITED against the pool.
//!
//! # What this covers, and what it deliberately does not
//!
//! `integral_response_gradient_3c2c` builds four tensors of its own on top of
//! the `Mp2Intermediates` handed to it:
//!
//!   `x_ov`, `y_ov`, `c_fit`  — each `(naux, nov)`
//!   `gamma_2c`               — `(naux, naux)`
//!
//! all four co-resident by the time the 2-centre block at the bottom runs.
//! Those are this function's to charge. The intermediates themselves
//! (`t2`, `b_ov`, `v_inv_sqrt`) are charged by their BUILDER through
//! `Mp2Intermediates::_charge`, so charging them again here would double-debit
//! one allocation — the mistake `Mp2Intermediates::uncharged` is named after.
//!
//! ## Artifact hypothesis (stated before measuring)
//!
//! If the charge is REAL, a pool sized below the four tensors refuses with a
//! message naming them, while an ample pool admits and returns to a zero
//! ledger. If instead the assertion were being satisfied by some OTHER plane
//! charged upstream (the failure mode that made three assertions in the
//! ferric-dft sibling file survive their mutation), then deleting this
//! function's charge would leave the test green. That is checked directly:
//! the tight-pool threshold below is derived from THESE tensors' size, and
//! the refusal is required to name THIS label.

use std::sync::{Mutex, MutexGuard, OnceLock};

use ferric_core::basis;
use ferric_core::memory::pool::{clear_global, global, install_global, MemoryPool};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::gradient::rimp2_gradient_analytical;
use ferric_mp2::rimp2::RiMp2Config;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
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

fn h2() -> Molecule {
    Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap()
}

/// Everything the gradient needs, with the SCF converged OUTSIDE the pool
/// window so the ledger measures only the gradient.
struct Fixture {
    mol: Molecule,
    obs: PreparedBasis,
    dfbs: PreparedBasis,
    bounds: SchwarzBounds,
    rhf: ferric_scf::ScfResult,
    cfg: RiMp2Config,
}

fn h2o() -> Molecule {
    Molecule::parse_xyz("3\nH2O\nO 0 0 0\nH 0 0.757 0.587\nH 0 -0.757 0.587\n", 0, 1).unwrap()
}

fn fixture() -> Fixture {
    fixture_for(h2(), "sto-3g")
}

/// [`fixture`] with an explicit molecule and orbital basis, for the per-worker
/// probe, which needs a shape where the gradient's own planes -- not an
/// upstream tensor -- set the pool floor.
fn fixture_for(mol: Molecule, obs_name: &str) -> Fixture {
    let obs_bs = basis::bundled(obs_name).unwrap();
    let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let ctx = ParallelContext::default();
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
    Fixture {
        mol,
        obs,
        dfbs,
        bounds,
        rhf,
        cfg: RiMp2Config::default(),
    }
}

impl Fixture {
    fn gradient(&self) -> Result<Array2<f64>, FerricError> {
        rimp2_gradient_analytical(
            &self.mol,
            &self.obs,
            &self.dfbs,
            Operator::coulomb(),
            &self.bounds,
            &self.rhf,
            &self.cfg,
            None,
        )
    }
}

#[test]
fn the_mp2_gradient_debits_and_releases_the_pool() {
    let _s = CleanSlot::acquire();
    let fx = fixture();

    install_global(MemoryPool::with_capacity_bytes(8_000_000_000));
    fx.gradient().expect("an ample pool must admit");
    let peak = global().expect("pool").peak_bytes();
    let outstanding = global().expect("pool").outstanding_bytes();
    clear_global();

    assert!(peak > 0, "the MP2 gradient took NO reservation at all");
    assert_eq!(
        outstanding, 0,
        "the MP2 gradient leaked {outstanding} B -- every reservation must be \
         credited back when its tensor drops, or an optimization exhausts the \
         pool over its steps"
    );
}

#[test]
fn the_mo_side_gradient_planes_refuse_under_their_own_name() {
    let _s = CleanSlot::acquire();
    let fx = fixture();

    // Reachability control FIRST: this exact call must ADMIT when there is
    // room, else a refusal below would prove nothing.
    install_global(MemoryPool::with_capacity_bytes(8_000_000_000));
    let ample = fx.gradient();
    clear_global();
    assert!(
        ample.is_ok(),
        "reachability control failed: an ample pool must admit. Got {ample:?}"
    );

    // The pool must be sized to admit everything UPSTREAM of this function and
    // refuse only its own tensors.
    //
    // A merely "tiny" pool does NOT test this: the DF 3-index tensor is
    // charged long before the gradient's MO-side planes, so it refuses first
    // and masks them. MEASURED: at 64 bytes the error came back naming
    // "DF 3-index (P|mn) in-core" and this assertion failed for the wrong
    // reason. That is the same masking that let three assertions in the
    // ferric-dft sibling survive their mutation.
    //
    // So: measure the peak first, then install a pool a hair below it. The
    // upstream planes (all smaller, and released before the peak) still fit;
    // the MO-side planes that DEFINE the peak do not.
    install_global(MemoryPool::with_capacity_bytes(8_000_000_000));
    fx.gradient().expect("ample pool must admit");
    let peak = global().expect("pool").peak_bytes();
    clear_global();
    assert!(peak > 0, "nothing charged; cannot size the probe");

    install_global(MemoryPool::with_capacity_bytes(peak * 3 / 4));
    let tight = fx.gradient();
    clear_global();

    let msg = tight
        .expect_err("a pool below the measured peak must refuse")
        .to_string();
    assert!(
        msg.contains("RI-MP2 gradient"),
        "the refusal must NAME an RI-MP2 GRADIENT plane. If it names an \
         upstream plane instead (the DF 3-index tensor, say) the probe is \
         mis-sized and this test is measuring someone else's charge; got: {msg}"
    );
}

/// The per-worker transient charge must be SOFT.
///
/// Its size reads `rayon::current_num_threads()`, which is not configuration.
/// A hard gate on it would make ADMISSION depend on `RAYON_NUM_THREADS`, so
/// the same job would be accepted on 4 workers and refused on 12 — the
/// nondeterminism `charge_mo_side_soft` documents, and a refusal the
/// pre-migration tree never makes.
///
/// The property: a pool sized to hold the hard MO-side planes but NOT the
/// per-worker transients on top of them must still COMPLETE. If the transient
/// charge were hard, this would refuse.
#[test]
fn the_per_worker_transient_charge_does_not_refuse_a_job_that_fits() {
    let _s = CleanSlot::acquire();
    // H2O/STO-3G, and the choice is load-bearing -- see the slack argument at
    // the `hard_floor` derivation below. What this shape buys is that the
    // gradient's OWN hard planes (76608 B) exceed the upstream DF 3-index
    // tensor (32928 B), so the pool floor is set INSIDE the function under
    // test and there is no upstream slack for a hard per-worker charge to
    // hide in. H2/cc-pVDZ, which this probe used before, has it the other way
    // round (df3 22400 B > hard 12320 B) and leaves 10080 B of slack -- enough
    // that the soft->hard mutation SURVIVED at 2 rayon workers. MEASURED, both
    // shapes, at 1/2/4/12 workers.
    let fx = fixture_for(h2o(), "sto-3g");

    // Size the probe from the HARD planes only, computed independently here.
    //
    // Sizing it from the measured peak does NOT work and is worth recording:
    // under a mutation that makes the per-worker charge hard, the peak GROWS
    // to include it, so a pool "at the peak" fits by construction and the
    // mutation survives. MEASURED — that exact probe left MUTANT7 (soft ->
    // hard) green. The probe has to be blind to the transient it is testing.
    //
    // The hard planes are x_ov + y_ov + c_fit (each naux x nov) + gamma_2c
    // (naux x naux); the per-worker transients sit on top of them. A pool
    // holding exactly the hard planes must therefore still COMPLETE if the
    // transient charge is soft, and REFUSE if it is hard -- provided nothing
    // else leaves room for the transient, which the slack guard below pins.
    let naux = fx.dfbs.nbasis();
    let nocc = (fx.mol.nelec() / 2) as usize;
    let nvir = fx.obs.nbasis() - nocc;
    let nov = nocc * nvir;
    let hard = 3 * naux * nov * 8 + naux * naux * 8;

    // The per-worker term this is meant to exclude: workers x max(tt_i, g3c).
    let workers = rayon::current_num_threads().max(1);
    let tt_i = nvir * nov * 8;
    let max_np = fx.dfbs.shell_dims().iter().copied().max().unwrap_or(0);
    let nbas = fx.obs.nbasis();
    let g3c = max_np * nbas * nbas * 8;
    let per_worker = workers * tt_i.max(g3c);

    // The pool must also clear everything UPSTREAM of this function (the DF
    // 3-index tensor and the z-vector pipeline), or the probe refuses for a
    // reason that has nothing to do with the per-worker charge -- MEASURED:
    // a pool sized at `hard` alone came back naming "DF 3-index (P|mn)
    // in-core", because `b_ov` is still outstanding when this function's own
    // planes go live and `hard` alone does not describe that co-residency.
    //
    // The capacity is DERIVED, not read off an ample run's `peak_bytes()`. A
    // soft charge that FITS is still debited and still lifts the high-water
    // mark, so the peak tracks the worker count -- MEASURED on the H2/cc-pVDZ
    // shape this probe used before: ample-run peak = 24416 / 30336 / 62336 B
    // at 2 / 4 / 12 rayon workers over a floor that never moved from 24416 B.
    // Subtracting a FRACTION of `per_worker` from that thread-varying number
    // is what made this test thread-dependent: `peak - per_worker/2` came to
    // 38336 B at 12 workers (cleared the floor, green) but 22336 B at 4 and
    // 20416 B at 2 (below it, so the run refused on the DF 3-index plane and
    // the probe blamed the soft charge). The floor below reads only basis
    // dimensions, so it cannot move with RAYON_NUM_THREADS.
    //
    //     floor = 3*naux*nov*8 + naux*naux*8  (this function's hard planes)
    //           + naux*nov*8                  (b_ov, still outstanding)
    let b_ov = naux * nov * 8;
    let df3 = naux * nbas * nbas * 8;
    let hard_floor = hard + b_ov;

    // DISCRIMINATION GUARD, and the one that picked this molecule/basis.
    //
    // The upstream DF 3-index tensor is RELEASED before this function's planes
    // go live. So if that tensor were the tallest hard plane, the pool would
    // have to be sized for IT, and the difference would be slack sitting free
    // at the moment the per-worker transient is charged:
    //
    //     slack = max(0, df3 - hard)
    //
    // A per-worker term smaller than that slack fits even when hard-charged,
    // and the probe silently stops discriminating. MEASURED on H2/cc-pVDZ:
    // df3 = 22400 B against hard = 12320 B leaves 10080 B of slack, and the
    // soft->hard mutation SURVIVED at 2 rayon workers (per_worker = 8000 B,
    // which fits) while being caught at 4 and 12. H2O/STO-3G inverts that --
    // hard = 76608 B against df3 = 32928 B, so the floor is set INSIDE the
    // function under test and the slack is zero at every worker count.
    let slack = df3.saturating_sub(hard);
    assert!(
        per_worker > slack,
        "probe is not discriminating at {workers} rayon workers: the \
         per-worker transient is {per_worker} B but {slack} B is free at the \
         moment it is charged (the upstream DF 3-index tensor, {df3} B, is \
         taller than this function's own hard planes, {hard} B, and is \
         released before them). A HARD per-worker charge would fit in that \
         slack, so this test could not tell soft from hard. Pick a shape where \
         the gradient's own planes set the floor."
    );

    // Run at EXACTLY the derived floor. Every hard plane fits and there is not
    // one spare byte for the per-worker transient, so a HARD per-worker charge
    // must refuse here and a SOFT one must shrug and run at lower concurrency.
    //
    // MEASURED on H2O/STO-3G: the derived floor is 83328 B (= 76608 B of
    // MO-side planes + 6720 B of b_ov) and the gradient admits at exactly that
    // capacity at 1, 2, 4 and 12 rayon workers.
    install_global(MemoryPool::with_capacity_bytes(hard_floor));
    let got = fx.gradient();
    let floor_peak = global().expect("pool").peak_bytes();
    clear_global();

    assert!(
        got.is_ok(),
        "a pool sized for the HARD planes refused the gradient: {got:?}. A \
         soft, thread-count-derived charge must degrade to running at whatever \
         concurrency is available, never to a refusal -- otherwise admission \
         depends on RAYON_NUM_THREADS, and the same job is accepted on 4 \
         workers and refused on 12."
    );

    // The floor must be TIGHT, or the assertion above is satisfied by slack
    // rather than by the charge being soft. Requiring the run to high-water
    // the pool EXACTLY is what keeps the probe discriminating -- and it fires
    // if a hard plane is ever added or removed, instead of letting this test
    // quietly measure someone else's charge (the masking the header warns of).
    assert_eq!(
        floor_peak, hard_floor,
        "the derived hard floor ({hard_floor} B = MO-side planes {hard} B + \
         b_ov {b_ov} B) is not tight: the run high-watered {floor_peak} B. \
         With slack in the pool a HARD per-worker charge would fit here too \
         and the assertion above would pass for the wrong reason."
    );
}
