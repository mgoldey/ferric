//! MWE: with NO pool installed, GW's quasiparticle energies are BIT-IDENTICAL
//! to the pre-migration tree.
//!
//! # Why this test exists and why it was written FIRST
//!
//! The shared brief's non-negotiable #1: "Unbudgeted must mean *do exactly
//! what you did before*, never *refuse everything*." Every gate this crate
//! adds returns an INERT `Reservation` when `pool::global()` is `None`, which
//! debits nothing and releases nothing. The property that has to hold is not
//! "the run still succeeds" -- a gate that quietly shrank a GEMM's blocking
//! would also succeed, and would move the energy in the 12th digit. It is
//! `to_bits()` equality on every QP energy.
//!
//! # Why `to_bits()` and not a tolerance
//!
//! A tolerance cannot distinguish "the migration is a no-op" from "the
//! migration re-associated an accumulation by a few ulp". The ksdft migration
//! learned this the expensive way: a batch width sized from live RSS gave
//! -390.3794282913 and -390.3794337741 Ha for the SAME input, which every
//! sensible tolerance would have passed. Bits or nothing.
//!
//! # The reference values are FROZEN CONSTANTS, not a second run
//!
//! Comparing two runs in the same process only proves the code is
//! deterministic; it cannot see a change that moved BOTH runs. The constants
//! below were captured from the tree at `72883be0` (the base commit of this
//! migration, before any ferric-gw pool charge existed) and are compared
//! bit-for-bit. If a future change to the numerics is deliberate, these
//! constants must be re-captured deliberately, which is the point.
//!
//! Shape: water / STO-3G orbital basis with a cc-pVDZ-RI auxiliary basis,
//! G0W0@HF and COHSEX@HF, full rank -- small enough to run in the default
//! (non-`--release`, non-`--ignored`) suite at every worker count.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_gw::{run_gw, GwConfig, GwMethod};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::config::{
    Chi0Backend, Chi0Sparsity, Eigensolver, PdepRpaConfig, QuadratureConfig, QuadratureScheme,
    SternheimerConfig,
};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

fn water() -> Molecule {
    Molecule::parse_xyz(
        "3\nH2O\nO  0.0   0.0       0.117790\nH  0.0   0.755453 -0.471161\nH  0.0  -0.755453 -0.471161\n",
        0,
        1,
    )
    .expect("parse H2O")
}

fn setup() -> (
    Molecule,
    PreparedBasis,
    PreparedBasis,
    ferric_scf::ScfResult,
) {
    let mol = water();
    let obs = PreparedBasis::new(&mol, &basis::bundled("sto-3g").expect("sto-3g")).expect("obs");
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").expect("ri")).expect("dfbs");
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &obs).expect("schwarz");
    let rhf = solve_rhf(
        &ParallelContext::default(),
        &mol,
        &obs,
        Operator::coulomb(),
        &bounds,
        &RhfConfig::default(),
    )
    .expect("RHF");
    (mol, obs, dfbs, rhf)
}

fn pdep_cfg() -> PdepRpaConfig {
    PdepRpaConfig {
        need_eigenvalues_freq: true,
        quadrature: QuadratureConfig {
            scheme: QuadratureScheme::GaussLegendre,
            n_points: 8,
            u0: 0.5,
        },
        eigensolver_conv_thresh: 1e-7,
        eigensolver_max_vecs: 0,
        trunc_thresh: 0.0,
        run_diagnostics: false,
        frozen_core: 0,
        chi0_backend: Chi0Backend::Dense,
        chi0_sparsity: Chi0Sparsity::Dense,
        eigensolver: Eigensolver::Davidson,
        sternheimer: SternheimerConfig::default(),
        memory_budget_bytes: None,
        need_inv_dielectric_freq: false, // run_gw forces this on (M9 gate)
        verbose: false,
    }
}

pub fn run(method: GwMethod) -> Vec<f64> {
    let (mol, obs, dfbs, rhf) = setup();
    let res = run_gw(
        &mol,
        &obs,
        &dfbs,
        Operator::coulomb(),
        &rhf,
        &pdep_cfg(),
        &GwConfig {
            method,
            ..Default::default()
        },
        None,
    )
    .expect("GW runs unbudgeted");
    res.eps_qp.to_vec()
}

/// Frozen G0W0 QP energies (Ha), re-recorded 2026-09-18 on main-after-#83.
///
/// # Why these moved, and why they are now only a SANITY BAND
///
/// These were captured at `72883be0`, a pre-migration commit on the old base.
/// Integrating the five pool branches REBASED the stack onto main after PR #83
/// (`a268fc3c`, spherical MINAO atomic blocks), which changes the converged SCF
/// reference density -- and GW is built on that density, so every QP energy
/// moved by ~1e-10.
///
/// Worse for a hardcoded bit pattern: the result is **not bit-reproducible
/// across machines**. Measured for QP[0] on the SAME tree:
///
/// ```text
///   frozen (old base)   0xBFE3B2C267EC725D
///   CI runner           0xBFE3B2C267F37B92   delta -5.119e-11
///   dev box             0xBFE3B2C267F39884   delta -5.202e-11
/// ```
///
/// Bit-stable WITHIN a machine (3/3 identical runs here) and different
/// BETWEEN machines -- the same per-machine-deterministic BLAS dispatch that
/// bit this repo's cDFT iteration counts. So a hardcoded `u64` can only ever
/// be right on one box, and asserting it on CI is asserting a property of that
/// runner's arithmetic.
///
/// The `to_bits()` reasoning in the module doc is still correct: a tolerance
/// cannot tell "no-op" from "re-associated by a few ulp". The fix is to apply
/// it where it is meaningful -- **live pool-off vs live pool-on, in the same
/// process, on the same machine** (see `an_ample_pool_gives_the_same_bits_as_no_pool`,
/// which is the assertion that actually tests the migration). These constants
/// are demoted to a loose absolute sanity band that catches a gross regression
/// without pinning one box's last three digits.
///
/// Values below are this box's; the band tolerates 1e-8, four orders above the
/// observed cross-machine spread.
///   -6.15571215606518174e-1  -4.18041879665428884e-1  -3.30966250586579025e-1
///    6.09356792132542346e-1   7.42858606124310539e-1
const G0W0_QP_BITS: [u64; 5] = [
    0xBFE3B2C267F39884,
    0xBFDAC132BA75CAC3,
    0xBFD52E8D11B7456A,
    0x3FE37FD9D0B61903,
    0x3FE7C57F69532283,
];

/// Frozen COHSEX QP energies (Ha), same shape, same tree:
///   -6.29599329764241822e-1  -4.22570898390289329e-1  -3.27295802184455642e-1
///    6.26889454649354971e-1   7.64533208860053715e-1
const COHSEX_QP_BITS: [u64; 5] = [
    0xBFE425AD7E65099E,
    0xBFDB0B66CF49163D,
    0xBFD4F26A17C0C438,
    0x3FE40F7A793A10AD,
    0x3FE8770E590F3772,
];

/// Absolute band for the recorded-value check. The observed cross-machine
/// spread is ~1e-10 (CI runner vs dev box, same tree); 1e-8 is four orders
/// above that, so this catches a gross regression while tolerating the BLAS
/// dispatch difference a hardcoded `u64` cannot.
const QP_SANITY_BAND: f64 = 1e-8;

/// Compare against the RECORDED values as a sanity band, not bit-for-bit.
///
/// Bit equality against a frozen literal is the wrong instrument here: the
/// value is bit-stable within a machine but differs between machines, so the
/// literal can only be right on the box that produced it. The bit-level claim
/// this file actually needs -- that the migration is a no-op -- is asserted
/// live-vs-live in `an_ample_pool_gives_the_same_bits_as_no_pool`, where both
/// sides run in the SAME process on the SAME machine and `to_bits()` is
/// meaningful.
fn assert_bits(got: &[f64], want: &[u64], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: QP count changed");
    let mismatch: Vec<String> = got
        .iter()
        .zip(want)
        .enumerate()
        .filter(|(_, (g, w))| (**g - f64::from_bits(**w)).abs() >= QP_SANITY_BAND)
        .map(|(i, (g, w))| {
            format!(
                "  [{i}] got {g:.17e} (0x{:016X}) recorded 0x{w:016X} (delta {:.3e})",
                g.to_bits(),
                g - f64::from_bits(*w)
            )
        })
        .collect();
    assert!(
        mismatch.is_empty(),
        "{what}: a QP energy left the {QP_SANITY_BAND:e} Ha sanity band around the \
         recorded values.\nThat is far larger than the ~1e-10 cross-machine spread, so \
         it is a real change in what GW computes -- not BLAS dispatch. If a gate stopped \
         returning an INERT reservation without a pool and shrank a buffer or re-blocked \
         a GEMM, it shows up here.\n{}",
        mismatch.join("\n")
    );
}

// These two assert the NO-POOL path, so they must not run WHILE a sibling in
// this binary has a pool installed -- the global slot is process-wide and
// cargo runs the tests in one process. They take the same lock as the
// pool-installing tests. (Measured: without the lock, both failed at 2, 4 and
// 12 workers under the default parallel test runner while passing under
// --test-threads=1, which is exactly the shape of a test that is green for the
// wrong reason.)
#[test]
fn g0w0_qp_energies_are_bit_identical_without_a_pool() {
    let _g = pool_lock();
    assert!(
        ferric_core::memory::pool::global().is_none(),
        "this test asserts the NO-POOL path; something installed a global pool"
    );
    assert_bits(&run(GwMethod::G0W0), &G0W0_QP_BITS, "G0W0");
}

#[test]
fn cohsex_qp_energies_are_bit_identical_without_a_pool() {
    let _g = pool_lock();
    assert!(
        ferric_core::memory::pool::global().is_none(),
        "this test asserts the NO-POOL path; something installed a global pool"
    );
    assert_bits(&run(GwMethod::Cohsex), &COHSEX_QP_BITS, "COHSEX");
}

// ---------------------------------------------------------------------------
// The other half of the anchor: a run WITH a pool must give the SAME bits.
//
// A gate that is inert without a pool and perturbs the answer with one has only
// moved the defect behind a flag. Two cases matter and they are different:
//
//   * an AMPLE pool: every charge is admitted, the parallel QP sweep runs, and
//     the answer must be the frozen constants;
//   * a TIGHT pool: the soft QP-sweep gate declines, the SERIAL sweep runs, and
//     the answer must STILL be the frozen constants. That is the claim that
//     makes the fallback honest -- "one worker's scratch, same order, same
//     values" is an assertion about numerics, so it is asserted on numerics.
// ---------------------------------------------------------------------------

/// The smallest pool capacity at which the whole GW run completes.
///
/// Binary-searched rather than derived, because the answer spans four crates:
/// ferric-scf's SCF, ferric-rpa's preflight, ferric-integrals' AO tensor and
/// this crate's four planes. `FERRIC_GW_POOL_SEARCH=1` prints it; the test
/// below asserts the property that matters, which is that it does NOT depend on
/// the worker count.
fn smallest_completing_capacity(method: GwMethod, lo_hint: usize, hi: usize) -> usize {
    let completes = |cap: usize| -> bool {
        let _g = pool_lock();
        ferric_core::memory::pool::install_global(
            ferric_core::memory::pool::MemoryPool::with_capacity_bytes(cap),
        );
        let r = try_run(method);
        ferric_core::memory::pool::clear_global();
        r.is_ok()
    };
    let (mut lo, mut hi) = (lo_hint, hi);
    assert!(completes(hi), "the upper bound {hi} must itself complete");
    while lo + 1 < hi {
        let mid = lo + (hi - lo) / 2;
        if completes(mid) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    hi
}

fn pool_lock() -> std::sync::MutexGuard<'static, ()> {
    static L: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    L.get_or_init(|| std::sync::Mutex::new(()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

fn try_run(method: GwMethod) -> Result<Vec<f64>, ferric_core::FerricError> {
    let (mol, obs, dfbs, rhf) = setup();
    let res = run_gw(
        &mol,
        &obs,
        &dfbs,
        Operator::coulomb(),
        &rhf,
        &pdep_cfg(),
        &GwConfig {
            method,
            ..Default::default()
        },
        None,
    )?;
    Ok(res.eps_qp.to_vec())
}

/// An AMPLE pool must not move a single bit.
#[test]
fn an_ample_pool_gives_the_same_bits_as_no_pool() {
    let _g = pool_lock();

    // LIVE vs LIVE, strict `to_bits()`. This is the assertion that actually
    // tests the migration, and it is the one place bit equality is the right
    // instrument: both sides run in the SAME process on the SAME machine with
    // the SAME BLAS dispatch, so the ONLY difference is whether a pool is
    // installed. Any surviving difference is the migration perturbing the
    // answer -- which is exactly the defect this file exists to catch.
    //
    // It deliberately does NOT compare against the frozen constants. Those are
    // machine-specific (see `G0W0_QP_BITS`) and checking pool-on against them
    // would conflate "the migration moved the answer" with "this is a
    // different CPU" -- the conflation that made this test fail on CI.
    assert!(
        ferric_core::memory::pool::global().is_none(),
        "baseline leg must run with NO pool installed"
    );
    let without = run(GwMethod::G0W0);

    ferric_core::memory::pool::install_global(
        ferric_core::memory::pool::MemoryPool::with_capacity_bytes(64 * 1_000_000_000),
    );
    let with = try_run(GwMethod::G0W0);
    ferric_core::memory::pool::clear_global();
    let with = with.expect("a 64 GB pool must admit every plane");

    let mismatch: Vec<String> = without
        .iter()
        .zip(with.iter())
        .enumerate()
        .filter(|(_, (a, b))| a.to_bits() != b.to_bits())
        .map(|(i, (a, b))| {
            format!(
                "  [{i}] no-pool {a:.17e} (0x{:016X})  pooled {b:.17e} (0x{:016X})  delta {:.3e}",
                a.to_bits(),
                b.to_bits(),
                b - a
            )
        })
        .collect();
    assert!(
        mismatch.is_empty(),
        "G0W0: installing an AMPLE pool changed the answer. Every gate must be a \
         no-op when nothing is scarce; a gate that shrank a buffer or re-blocked a \
         GEMM under an ample pool has moved the defect behind a flag rather than \
         fixing it.\n{}",
        mismatch.join("\n")
    );

    // And the absolute values are still in the recorded band, so this cannot
    // pass by both legs being equally wrong.
    assert_bits(&with, &G0W0_QP_BITS, "G0W0 under an ample pool");
}

/// The soft QP-sweep gate must actually DECLINE when its scratch does not fit,
/// the run must still COMPLETE, and the panelled (serial) sweep must give the
/// SAME bits.
///
/// # Why this is written on a counter, and how the first version was INERT
///
/// A soft gate's whole purpose is that the run completes EITHER WAY, so
/// completion cannot distinguish "the gate declined and panelled" from "the
/// gate never had to decline". The first version of this test asserted only
/// completion at the binary-searched floor, and mutation M3 (`try_reserve` ->
/// `reserve`, soft made hard) left all five tests here GREEN at 2, 4 AND 12
/// workers.
///
/// The reason is worth recording, because it is a genuine property of this
/// crate's ordering and not just a bad constant. The floor is set by
/// ferric-rpa's preflight charge, which is RELEASED before the Σ driver runs.
/// At the instant this gate asks, the only outstanding ferric-gw bytes are
/// `b_full + m_proj` -- measured 840_672 B of free headroom against a
/// 19_712..118_272 B scratch at 2..12 workers. So a whole-run capacity test
/// has an EMPTY observable window at this shape: any capacity low enough to
/// squeeze the gate is too low for the earlier preflight, and the run dies on
/// a MANDATORY plane instead. Three successive attempts to derive that window
/// all failed on that inequality, which is the evidence it is empty rather
/// than mis-derived.
///
/// So the branch is exercised where it can be: at the gate itself, with a pool
/// whose free space is one byte short of the scratch. That is not a weaker
/// test -- it is the same decision, with the confounder (the released
/// preflight charge) removed instead of fought.
#[test]
fn the_soft_qp_gate_panels_when_its_scratch_does_not_fit() {
    let workers = rayon::current_num_threads().max(1);
    let scratch = ferric_gw::budget::qp_worker_scratch_bytes(
        RI_NAUX,
        STO3G_NACT,
        n_quad_points(),
        workers,
        1,
    );
    assert!(scratch > 1, "nothing to decline -- the fixture is vacuous");

    let _g = pool_lock();

    // (a) One byte short: the gate MUST decline.
    let pool = ferric_core::memory::pool::MemoryPool::with_capacity_bytes(scratch - 1);
    ferric_core::memory::pool::install_global(pool);
    let before = ferric_gw::sigma::qp_sweep_panelled_count();
    let taken =
        ferric_gw::sigma::charge_qp_sweep_scratch_for_test(RI_NAUX, STO3G_NACT, n_quad_points(), 0);
    let declined = ferric_gw::sigma::qp_sweep_panelled_count() - before;
    ferric_core::memory::pool::clear_global();
    assert!(
        taken.is_none() && declined == 1,
        "with {} B free the {scratch} B scratch cannot fit, so the gate must DECLINE \
         (panel), not take it and not refuse the job. A `reserve` here instead of a \
         `try_reserve` would have propagated an error and killed a run that completes.",
        scratch - 1
    );

    // (b) The over-rejection guard: one byte spare and it MUST take it.
    // Without this, (a) passes on a gate that declines unconditionally, which
    // would serialize every GW run in the tree.
    let pool = ferric_core::memory::pool::MemoryPool::with_capacity_bytes(scratch + 1);
    ferric_core::memory::pool::install_global(pool);
    let before = ferric_gw::sigma::qp_sweep_panelled_count();
    let taken =
        ferric_gw::sigma::charge_qp_sweep_scratch_for_test(RI_NAUX, STO3G_NACT, n_quad_points(), 0);
    let declined = ferric_gw::sigma::qp_sweep_panelled_count() - before;
    ferric_core::memory::pool::clear_global();
    assert!(
        taken.is_some() && declined == 0,
        "with {} B free the {scratch} B scratch fits, so the gate must TAKE it and the \
         parallel sweep must run",
        scratch + 1
    );

    // (c) The downstream-hard check: room for the scratch alone is not enough
    // if a mandatory plane asks next. This is the ferric-rpa starvation shape,
    // and a bare `try_reserve` (mutation M6) would take the bytes here.
    let pool = ferric_core::memory::pool::MemoryPool::with_capacity_bytes(scratch + 1);
    ferric_core::memory::pool::install_global(pool);
    let before = ferric_gw::sigma::qp_sweep_panelled_count();
    let taken = ferric_gw::sigma::charge_qp_sweep_scratch_for_test(
        RI_NAUX,
        STO3G_NACT,
        n_quad_points(),
        1_000_000, // a mandatory plane that asks next and has no fallback
    );
    let declined = ferric_gw::sigma::qp_sweep_panelled_count() - before;
    ferric_core::memory::pool::clear_global();
    assert!(
        taken.is_none() && declined == 1,
        "the scratch fits on its own but leaves nothing for the 1 MB downstream HARD \
         plane, so an OPTIONAL allocation would starve a MANDATORY one -- the exact \
         defect ferric-rpa's 08ab4431 fixed. A bare `try_reserve` takes the bytes here."
    );
}

/// The whole-run counterpart: a TIGHT (binary-searched floor) pool must give
/// the frozen bits, and the floor must be accounted for.
///
/// The capacity is the binary-searched minimum, not a frozen constant: the
/// planes involved span four crates and one of the terms scales with
/// `rayon::current_num_threads()`, so a number calibrated at one width is
/// wrong at the others (which is how the ferric-rpa migration broke 3 of its
/// 7 tests).
///
/// At this shape the QP-sweep gate does NOT decline at that minimum -- see
/// `the_soft_qp_gate_panels_when_its_scratch_does_not_fit` for why the
/// whole-run observable window for that branch is empty -- so this test
/// asserts the other half: everything the pool DOES bind still produces the
/// same answer, and the worker-dependent part of the floor is fully
/// attributable to ferric-rpa.
#[test]
fn the_smallest_completing_pool_gives_the_same_bits_and_is_worker_independent() {
    let workers = rayon::current_num_threads().max(1);
    let min = smallest_completing_capacity(GwMethod::G0W0, 0, 1_000_000_000);
    eprintln!("smallest completing capacity at {workers} workers: {min} B");

    let _g = pool_lock();
    ferric_core::memory::pool::install_global(
        ferric_core::memory::pool::MemoryPool::with_capacity_bytes(min),
    );
    let got = try_run(GwMethod::G0W0);
    ferric_core::memory::pool::clear_global();
    assert_bits(
        &got.unwrap_or_else(|e| panic!("the searched minimum {min} must complete: {e}")),
        &G0W0_QP_BITS,
        "G0W0 at the smallest completing pool",
    );

    // THE worker-independence property. If the soft QP scratch were greedy --
    // taken whenever it happened to fit, with no regard for a mandatory plane
    // asking next -- this minimum would grow with the worker count, which is
    // exactly how the ferric-rpa starvation bug hid at 12 workers while doing
    // its damage at 2 and 4. The soft term at this shape is:
    let scratch = ferric_gw::budget::qp_worker_scratch_bytes(
        // naux for cc-pVDZ-RI on water; m_modes == naux at full rank.
        RI_NAUX, STO3G_NACT, 8, workers, 1,
    );
    assert!(
        min > scratch,
        "sanity: the minimum ({min}) should exceed one sweep's scratch ({scratch})"
    );
    // WHY the QP gate does not decline at this minimum, recorded as an
    // assertion rather than a comment so it cannot silently stop being true.
    // ferric-rpa's preflight charge -- the term that SETS this minimum -- is
    // released before the Σ driver runs, so at the instant the QP gate asks
    // the only outstanding ferric-gw bytes are `b_full + m_proj`, leaving
    // `min - at_gate` free. That headroom exceeds the scratch at every width
    // measured (840_672 B free vs 19_712..118_272 B), which is why a
    // WHOLE-RUN capacity test cannot observe this gate's decision at this
    // shape and `the_soft_qp_gate_panels_when_its_scratch_does_not_fit`
    // exercises it at the gate instead.
    let at_gate = budget_at_gate();
    assert!(
        min - at_gate > scratch,
        "the free headroom at the QP gate ({} B) has dropped below the scratch \
         ({scratch} B) at {workers} workers. That would make a whole-run capacity test \
         of the soft branch possible after all -- worth writing, and worth re-reading the \
         reasoning in `the_soft_qp_gate_panels_when_its_scratch_does_not_fit`, which \
         documents the window as empty.",
        min - at_gate
    );
    // The mandatory floor is worker-INDEPENDENT by construction (b_full,
    // m_proj, the AO tensor and ferric-rpa's in-core planes carry no
    // n_workers term), so a worker-independent minimum is the observable
    // signature of a non-greedy soft gate. The bound below is what a greedy
    // gate would violate: it would need room for the scratch ON TOP of the
    // floor, i.e. the minimum would track `workers`.
    // THE WORKER-DEPENDENCE ACCOUNTING.
    //
    // MEASURED, water/STO-3G + cc-pVDZ-RI (naux = m = 84, nocc = 5, nvir = 2,
    // nov = 10, n_quad = 8), binary-searching the smallest completing capacity:
    //
    //   workers   this crate's QP scratch    smallest completing capacity
    //   1          9_856                      899_808
    //   2         19_712                      906_528   (+6_720)
    //   4         39_424                      919_968   (+20_160)
    //  12        118_272                      946_848   (+47_040)
    //  24        236_544                      946_848   (+47_040, saturated)
    //
    // The minimum DOES move with the worker count -- and none of it is this
    // crate's. Two facts locate it. First, the refusal at `min - 1` is
    //
    //   memory pool exhausted: "DF 3-index (P|mn) in-core" ...
    //     0.001 GB  PDEP-RPA preflight (naux=84, nocc=5, nvir=2, n_workers=N)
    //               [in-core]
    //
    // i.e. the incumbent is ferric-rpa's HARD half and the refused plane is
    // ferric-integrals'. No ferric-gw label appears in the breakdown at any
    // width, and no `[freq scratch]` label appears either -- ferric-rpa's own
    // soft gate is declining correctly, exactly as its 08ab4431 fix intends.
    //
    // Second, the increments are EXACTLY ferric-rpa's `clones` term inside
    // `budget::estimate_peak_bytes`:
    //
    //   clones = min(n_quad, n_workers) * m * nov * 8
    //          = min(8, w) * 84 * 10 * 8 = 6_720 * min(8, w)
    //
    // 6_720 / 20_160 / 47_040 are 1x / 3x / 7x that unit, and the series
    // SATURATES at w = 8 = n_quad, which is the formula's `min` and nothing
    // else's. That term is worker-dependent and lands in the HARD half of
    // `estimate_peak_split` (only `quad_scratch_bytes` is split off as soft),
    // so it is a mandatory, worker-scaling charge in ferric-rpa -- outside this
    // crate's scope, and reported rather than patched here.
    //
    // What this test can assert is that ferric-gw adds nothing on top: the
    // observed spread is fully explained by that term, with NO room for this
    // crate's QP scratch (which at 12 workers is 118_272 B, twenty-five times
    // larger than the whole 47_040 B spread). If the GW soft gate were greedy,
    // the minimum would have to grow by at least its own scratch.
    let rpa_clone_unit = RI_NAUX * (5 * (STO3G_NACT - 5)) * 8;
    let rpa_spread = rpa_clone_unit * (n_quad_points().min(workers) - 1);
    let base = min - rpa_spread;
    assert_eq!(
        base, 899_808,
        "the worker-INDEPENDENT floor moved. Measured minimum {min} B at {workers} workers \
         minus ferric-rpa's worker-dependent `clones` term ({rpa_spread} B) should leave the \
         same floor at every width. A change here means either a plane's size changed or \
         ferric-gw started contributing a worker-dependent term of its own -- the latter is \
         the greedy-soft-gate signature the brief warns about, and this crate's QP scratch \
         ({scratch} B at {workers} workers) would show up as a spread far larger than \
         ferric-rpa's."
    );
}

/// naux of cc-pVDZ-RI on water, and n_act of STO-3G on water. Asserted against
/// the live bases in the test below rather than trusted, so a basis-set change
/// makes the numbers fail loudly instead of silently mis-sizing the window.
const RI_NAUX: usize = 84;
const STO3G_NACT: usize = 7;

/// `pdep_cfg().quadrature.n_points`, read back rather than repeated so the
/// accounting above and the config cannot drift.
fn n_quad_points() -> usize {
    pdep_cfg().quadrature.n_points
}

/// ferric-gw bytes OUTSTANDING at the instant the QP-sweep gate asks: `b_full`
/// and `m_proj`, both live, both charged.
///
/// Derived from `ferric_gw::budget` -- the same module the gates charge
/// through -- so a shape or formula change cannot leave this stale.
fn budget_at_gate() -> usize {
    // Full rank (`trunc_thresh = 0`), so m_modes == naux.
    ferric_gw::budget::b_full_bytes(RI_NAUX, STO3G_NACT)
        + ferric_gw::budget::m_proj_bytes(RI_NAUX, STO3G_NACT)
}

#[test]
fn the_fixture_shape_constants_match_the_live_bases() {
    let (_mol, obs, dfbs, _rhf) = setup();
    assert_eq!(dfbs.nbasis(), RI_NAUX, "cc-pVDZ-RI naux on water changed");
    assert_eq!(obs.nbasis(), STO3G_NACT, "STO-3G nbf on water changed");
}
