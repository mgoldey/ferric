//! End-to-end exactness + cost gates for the open-shell combined direct J+K
//! Fock build and its incremental (ΔD) variant.
//!
//! The open-shell direct path used to rebuild the Fock matrices from THREE
//! separate quartet traversals every iteration — `DirectJ(D_α+D_β)`,
//! `DirectK(D_α)`, `DirectK(D_β)` — each screened on a single loose global
//! `max|D|` scalar rather than the tight six-pairwise shell table the
//! closed-shell path used, and with no incremental (differential) Fock build at
//! all. `solve_uhf`/`solve_rohf` now use one `DirectJK::build_uhf` pass with the
//! tight screen; a ΔD-incremental update also exists but is OPT-IN and off by
//! default (it measured ~1.0x — see `direct_jk::open_shell_incremental_enabled`).
//!
//! Both changes are gated by env kill-switches (`FERRIC_SCF_COMBINED_JK`,
//! `FERRIC_SCF_UHF_INCREMENTAL`), so every test here is a true in-binary A/B of
//! the new path against the historical one — no rebuild, no cross-commit
//! comparison.
//!
//! ## What each test is for
//!
//! * The `*_energy_matches_*` tests are the EXACTNESS ANCHORS: the physics must
//!   not move. They are the reason the cost tests below are allowed to exist.
//! * `combined_build_cuts_quartet_count_versus_three_pass` is the MEASUREMENT
//!   gate for the change that actually pays (3x fewer quartets), so a future
//!   refactor that silently reverts to the three-pass build fails loudly rather
//!   than just getting slower.
//! * `incremental_fock_does_not_regress_work_or_iterations` RECORDS the
//!   negative result for the incremental path rather than asserting a win.
//!
//! Every test takes `ENV_LOCK` for its whole body — the switches are process-
//! global env vars and cargo runs tests in threads.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::result::ScfResult;
use ferric_scf::rhf::RhfConfig;
use ferric_scf::screening::SchwarzBounds;
use std::sync::Mutex;

static ENV_LOCK: Mutex<()> = Mutex::new(());

/// CH3 radical (doublet), planar D3h, coordinates in Angstrom.
const CH3_XYZ: &str = "4\nCH3 doublet\nC 0.0000 0.0000 0.0000\nH 1.0790 0.0000 0.0000\n\
                       H -0.5395 0.9345 0.0000\nH -0.5395 -0.9345 0.0000\n";

fn setup(xyz: &str, charge: i32, mult: usize, bas: &str) -> (Molecule, PreparedBasis, SchwarzBounds) {
    let mol = Molecule::parse_xyz(xyz, charge, mult).unwrap();
    let bs = basis::bundled(bas).unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    (mol, prep, bounds)
}

fn setup_file(path: &str, charge: i32, mult: usize, bas: &str) -> (Molecule, PreparedBasis, SchwarzBounds) {
    let mol = Molecule::load_xyz_with_charge(path, charge, mult).unwrap();
    let bs = basis::bundled(bas).unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    (mol, prep, bounds)
}

fn tight_config() -> RhfConfig {
    RhfConfig { max_iter: 200, ..Default::default() }
}

/// Set the two kill-switches, run `f`, restore. Caller holds ENV_LOCK.
fn with_switches<T>(combined: bool, incremental: bool, f: impl FnOnce() -> T) -> T {
    // SAFETY (test-only): ENV_LOCK serializes every mutation in this file.
    unsafe {
        std::env::set_var("FERRIC_SCF_COMBINED_JK", if combined { "1" } else { "0" });
        // Open-shell incremental is opt-IN (default off) — see
        // `direct_jk::open_shell_incremental_enabled` for the measurements.
        std::env::set_var("FERRIC_SCF_UHF_INCREMENTAL", if incremental { "1" } else { "0" });
    }
    let out = f();
    unsafe {
        std::env::remove_var("FERRIC_SCF_COMBINED_JK");
        std::env::remove_var("FERRIC_SCF_UHF_INCREMENTAL");
    }
    out
}

fn run_uhf(mol: &Molecule, prep: &PreparedBasis, bounds: &SchwarzBounds) -> ScfResult {
    let ctx = ParallelContext::default();
    ferric_scf::uhf::solve_uhf(&ctx, mol, prep, bounds, &tight_config()).unwrap()
}

fn run_rohf(mol: &Molecule, prep: &PreparedBasis, bounds: &SchwarzBounds) -> ScfResult {
    let ctx = ParallelContext::default();
    ferric_scf::rohf::solve_rohf(&ctx, mol, prep, Operator::coulomb(), bounds, &tight_config())
        .unwrap()
}

/// The three configurations under test, for one open-shell system:
/// (three-pass baseline, combined single-pass, combined + incremental).
struct Arms {
    baseline: ScfResult,
    combined: ScfResult,
    incremental: ScfResult,
}

fn run_arms(
    mol: &Molecule,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    rohf: bool,
) -> Arms {
    let run = |m: &Molecule, p: &PreparedBasis, b: &SchwarzBounds| {
        if rohf { run_rohf(m, p, b) } else { run_uhf(m, p, b) }
    };
    Arms {
        baseline: with_switches(false, false, || run(mol, prep, bounds)),
        combined: with_switches(true, false, || run(mol, prep, bounds)),
        incremental: with_switches(true, true, || run(mol, prep, bounds)),
    }
}

/// Report + assert the exactness bar for one system. `EXACTNESS`: the combined
/// and incremental energies must reproduce the three-pass baseline to 1e-10 Ha.
///
/// 1e-10 is well below any chemical or basis-set scale and below the SCF
/// convergence threshold itself; it is a "the physics did not move" bar, not a
/// bit-identity claim (the combined path legitimately screens a DIFFERENT
/// quartet set — a tighter one — so exact bit agreement is not expected and
/// would in fact indicate the screen had not changed at all).
fn assert_arms(label: &str, a: &Arms) {
    assert!(a.baseline.converged, "{label}: three-pass baseline did not converge");
    assert!(a.combined.converged, "{label}: combined build did not converge");
    assert!(a.incremental.converged, "{label}: incremental build did not converge");

    let d_comb = (a.combined.energy - a.baseline.energy).abs();
    let d_incr = (a.incremental.energy - a.baseline.energy).abs();
    println!(
        "[{label}] E_baseline={:.12}  E_combined={:.12} (d={d_comb:.3e})  \
         E_incremental={:.12} (d={d_incr:.3e})",
        a.baseline.energy, a.combined.energy, a.incremental.energy
    );
    println!(
        "[{label}] quartets: baseline={} combined={} incremental={}  |  iters: {} {} {}",
        a.baseline.computed_quartets,
        a.combined.computed_quartets,
        a.incremental.computed_quartets,
        a.baseline.iterations,
        a.combined.iterations,
        a.incremental.iterations
    );
    assert!(
        d_comb < 1e-10,
        "{label}: combined single-pass J+K moved the energy by {d_comb:.3e} Ha \
         (baseline {:.12}, combined {:.12})",
        a.baseline.energy,
        a.combined.energy
    );
    assert!(
        d_incr < 1e-10,
        "{label}: incremental Fock moved the energy by {d_incr:.3e} Ha \
         (baseline {:.12}, incremental {:.12})",
        a.baseline.energy,
        a.incremental.energy
    );
}

/// EXACTNESS: CH3 doublet, UHF/cc-pVDZ.
#[test]
fn ch3_doublet_uhf_energy_matches_three_pass_baseline() {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (mol, prep, bounds) = setup(CH3_XYZ, 0, 2, "cc-pvdz");
    let arms = run_arms(&mol, &prep, &bounds, false);
    assert_arms("CH3/UHF/cc-pVDZ", &arms);
}

/// EXACTNESS: O2 triplet, UHF/cc-pVDZ. A genuinely spin-polarized case where
/// the two channels differ a lot, so a crossed/duplicated spin density in the
/// combined build would be obvious in the energy.
#[test]
fn o2_triplet_uhf_energy_matches_three_pass_baseline() {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (mol, prep, bounds) = setup_file("../../testdata/molecules/o2.xyz", 0, 3, "cc-pvdz");
    let arms = run_arms(&mol, &prep, &bounds, false);
    assert_arms("O2/UHF/cc-pVDZ", &arms);
}

/// EXACTNESS: ROHF. The Roothaan-coupled solver takes the same three-matrix
/// Fock build, so both changes apply — this pins that the ROHF-specific
/// coupling downstream is unaffected.
#[test]
fn oh_doublet_rohf_energy_matches_three_pass_baseline() {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (mol, prep, bounds) = setup_file("../../testdata/molecules/oh.xyz", 0, 2, "cc-pvdz");
    let arms = run_arms(&mol, &prep, &bounds, true);
    assert_arms("OH/ROHF/cc-pVDZ", &arms);
}

/// MEASUREMENT: the combined single-pass build must do substantially LESS
/// quartet work than the three-pass one it replaces.
///
/// Two independent effects push the same way: one traversal instead of three,
/// and a tighter (six-pairwise, spin-sum) screen instead of the global `max|D|`
/// scalar. The bar is deliberately loose (a 2× reduction) — the point is to
/// catch a silent revert to three passes, not to freeze a specific number that
/// legitimately moves with screening changes. The actual measured factor is
/// printed.
#[test]
fn combined_build_cuts_quartet_count_versus_three_pass() {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (mol, prep, bounds) = setup(CH3_XYZ, 0, 2, "cc-pvdz");
    let base = with_switches(false, false, || run_uhf(&mol, &prep, &bounds));
    let comb = with_switches(true, false, || run_uhf(&mol, &prep, &bounds));
    assert!(base.converged && comb.converged);
    // Compare per-iteration work: the two arms may converge in different
    // iteration counts, and total quartets would then conflate the two effects.
    let per_iter_base = base.computed_quartets as f64 / base.iterations as f64;
    let per_iter_comb = comb.computed_quartets as f64 / comb.iterations as f64;
    let factor = per_iter_base / per_iter_comb;
    println!(
        "[CH3/UHF] quartets/iter: three-pass {per_iter_base:.0}, combined {per_iter_comb:.0} \
         -> {factor:.2}x reduction"
    );
    assert!(
        factor > 2.0,
        "combined build should cut per-iteration quartet work by >2x, got {factor:.2}x \
         (three-pass {per_iter_base:.0}/iter, combined {per_iter_comb:.0}/iter)"
    );
}

/// MEASUREMENT (recorded, NOT a speedup claim): the incremental Fock build is
/// CORRECT but does essentially no work reduction at these sizes.
///
/// This test exists to pin the measurement that led to the open-shell
/// incremental path being left OPT-IN (default off), so a future reader does
/// not re-enable it expecting a win, and so that if the underlying screen
/// behaviour ever changes this assumption is re-examined rather than inherited.
///
/// Measured here (CH3/cc-pVDZ, default thresholds): full-rebuild and
/// incremental produce the SAME total quartet count. Larger runs recorded in
/// `direct_jk::open_shell_incremental_enabled`'s doc give 1.010x (alkane_4,
/// cc-pVDZ) and 1.046x (alkane_8, 6-31G) — real but ~1%.
///
/// The assertion is deliberately the honest one: incremental must not do MORE
/// work and must not need more iterations. A genuine future improvement would
/// make this test's printed ratio rise, and nothing here would block it.
#[test]
fn incremental_fock_does_not_regress_work_or_iterations() {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (mol, prep, bounds) = setup(CH3_XYZ, 0, 2, "cc-pvdz");
    let full = with_switches(true, false, || run_uhf(&mol, &prep, &bounds));
    let incr = with_switches(true, true, || run_uhf(&mol, &prep, &bounds));
    assert!(full.converged && incr.converged);
    let factor = full.computed_quartets as f64 / incr.computed_quartets as f64;
    println!(
        "[CH3/UHF] total quartets: full-rebuild {} ({} iters), incremental {} ({} iters) \
         -> {factor:.3}x (NOTE: ~1.0 is the expected, measured result here)",
        full.computed_quartets, full.iterations, incr.computed_quartets, incr.iterations
    );
    assert!(
        factor > 0.98,
        "incremental Fock did MORE quartet work than full rebuild ({factor:.3}x): \
         {} vs {}",
        incr.computed_quartets,
        full.computed_quartets
    );
    assert!(
        incr.iterations <= full.iterations,
        "incremental Fock must not slow convergence: {} iters vs {} full-rebuild",
        incr.iterations,
        full.iterations
    );
}

/// Thread-count bit-identity end to end: the open-shell SCF energy must not
/// depend on `RAYON_NUM_THREADS` on the new combined + incremental path.
///
/// The builder-level guard lives in `direct_jk.rs`; this is the whole-solver
/// version, which additionally covers the delta formation and the buffer
/// accumulate-vs-zero bookkeeping in the SCF loop.
#[test]
fn open_shell_energy_bit_identical_across_thread_counts() {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (mol, prep, bounds) = setup(CH3_XYZ, 0, 2, "cc-pvdz");
    let run = |threads: usize| -> f64 {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        pool.install(|| with_switches(true, true, || run_uhf(&mol, &prep, &bounds).energy))
    };
    let e1 = run(1);
    let e4 = run(4);
    println!("[CH3/UHF] E(1 thread)={e1:.17}  E(4 threads)={e4:.17}");
    assert_eq!(
        e1.to_bits(),
        e4.to_bits(),
        "open-shell SCF energy must be BIT-identical across thread counts: \
         1-thread {e1:.17} vs 4-thread {e4:.17}"
    );
}
