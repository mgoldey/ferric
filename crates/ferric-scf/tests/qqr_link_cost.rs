//! PHASE 1 COST MEASUREMENT: does wiring `QqrBounds` into LinK pay for itself?
//!
//! This file is a MEASUREMENT harness, not a correctness gate. Every test in it
//! is `#[ignore]`d and prints a table; nothing here asserts a timing (wall-clock
//! assertions on a shared box are noise generators, not tests).
//!
//! # The question
//!
//! `QqrBounds` refines Schwarz with a distance-decay envelope, screening ~0.7%
//! (alkane_6) to ~5% (alkane_16) more shell quartets than plain Schwarz at
//! production thresholds. But it is not free:
//!
//! * **construction**: a dense `O(nshells²)` pair-center + pair-extent table,
//!   built ON TOP of the Schwarz table it wraps (it owns a `SchwarzBounds`).
//! * **per-call**: `QqrBounds::estimate` does a Schwarz product PLUS two table
//!   loads, a 3-vector difference, a `sqrt`, and a divide — where Schwarz does
//!   one multiply.
//!
//! The per-call overhead is charged on EVERY surviving quartet AND on every
//! quartet it screens, plus twice more inside `SignificantPairs::build` and
//! `DensityPairs::build`. So a few percent fewer quartets has to beat a
//! constant-factor tax on the whole screening loop. That is the gate.
//!
//! # Protocol
//!
//! For each system: build the basis, converge ONE RHF density (shared by both
//! arms, so the comparison varies exactly one thing — the bound type), then for
//! each bound kind measure
//!
//!  (a) table construction wall time, and for QQR the Schwarz build it contains
//!      is timed separately so the MARGINAL QQR cost is visible;
//!  (b) `LinkK::new` (= `SignificantPairs::build`, which calls `estimate` over
//!      all `nsh²` pairs) plus `update_density` (`DensityPairs::build`, same
//!      order) — the per-iteration setup;
//!  (c) `build()` wall time and its returned computed-quartet count, repeated
//!      so a single noisy sample cannot decide the verdict.
//!
//! Times are reported as the MINIMUM over repeats, not the mean: on a shared
//! box the minimum is the estimate least contaminated by other load (the
//! contamination is one-sided — interference only ever makes a run slower).
//!
//! # Running
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 FERRIC_MEM_BUDGET_GB=2 \
//!   scripts/ferric-limited --max=4G --high=3600M -- \
//!   cargo test -p ferric-scf --test qqr_link_cost --release \
//!   -- --ignored --nocapture qqr_vs_schwarz_cost_alkane_8
//! ```
//!
//! Check `/proc/pressure/memory` before and after; DISCARD any run whose
//! `full avg10` is nonzero (see the repo's PSI discipline note).

use ferric_core::basis::{self, BasisSet};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::fock::KBuilder;
use ferric_scf::link_k::LinkK;
use ferric_scf::qqr::QqrBounds;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;
use std::time::{Duration, Instant};

/// Repeats for the per-iteration K build. The minimum over these is reported.
const K_BUILD_REPEATS: usize = 3;

/// Production screening threshold the cost question is asked at. This is
/// `RhfConfig::default().integral_thresh`'s regime; the QQR benefit table in
/// the task was measured at 1e-8/1e-10/1e-12 and the benefit SHRINKS as the
/// threshold tightens, so 1e-10 is the middle of the measured range.
const THRESHOLDS: [f64; 2] = [1e-8, 1e-10];

fn load_mol(stem: &str) -> Molecule {
    Molecule::load_xyz(&format!("../../testdata/molecules/{stem}.xyz"))
        .unwrap_or_else(|e| panic!("cannot load {stem}.xyz: {e}"))
}

/// Longest interatomic distance in Bohr — the "diameter" column of the benefit
/// table, recomputed here so the printed row is self-describing rather than
/// relying on a number copied from a prompt.
fn diameter_bohr(mol: &Molecule) -> f64 {
    let mut d: f64 = 0.0;
    for (i, a) in mol.atoms.iter().enumerate() {
        for b in mol.atoms.iter().skip(i + 1) {
            let dx = a.x - b.x;
            let dy = a.y - b.y;
            let dz = a.zpos - b.zpos;
            d = d.max((dx * dx + dy * dy + dz * dz).sqrt());
        }
    }
    d
}

/// One converged RHF density, shared by both bound arms.
fn converged_density(mol: &Molecule, prep: &PreparedBasis) -> Array2<f64> {
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, prep).expect("Schwarz bounds");
    let config = RhfConfig::default();
    let res = solve_rhf(&ParallelContext::default(), mol, prep, op, &bounds, &config)
        .expect("RHF solve");
    assert!(
        res.converged,
        "RHF did not converge — refusing to time a K build against an unconverged density"
    );
    res.density_total
}

struct Timing {
    /// Bound table construction (Schwarz alone, or Schwarz + QQR pair tables).
    table: Duration,
    /// `LinkK::new` + `update_density`: the pair-list builds, both of which
    /// sweep `estimate` over the full ordered-pair space.
    pairlists: Duration,
    /// Minimum `build()` wall time over `K_BUILD_REPEATS`.
    k_build: Duration,
    /// Computed shell quartets (the count `build()` returns).
    quartets: usize,
}

fn time_schwarz(
    prep: &PreparedBasis,
    d: &Array2<f64>,
    thresh: f64,
) -> (Timing, Array2<f64>) {
    let op = Operator::coulomb();
    let ctx = ParallelContext::default();
    let n = prep.nbasis();

    let t0 = Instant::now();
    let sb = SchwarzBounds::compute(op, prep).expect("Schwarz bounds");
    let table = t0.elapsed();

    let t1 = Instant::now();
    let mut link = LinkK::new(&ctx, prep, &sb, op, thresh, usize::MAX);
    link.update_density(d);
    let pairlists = t1.elapsed();

    let mut k = Array2::zeros((n, n));
    let mut best = Duration::MAX;
    let mut quartets = 0usize;
    for _ in 0..K_BUILD_REPEATS {
        k.fill(0.0);
        let t2 = Instant::now();
        quartets = link.build(d, &mut k).expect("LinK build");
        best = best.min(t2.elapsed());
    }
    (
        Timing { table, pairlists, k_build: best, quartets },
        k,
    )
}

fn time_qqr(
    mol: &Molecule,
    bs: &BasisSet,
    prep: &PreparedBasis,
    d: &Array2<f64>,
    thresh: f64,
) -> (Timing, Duration, Array2<f64>) {
    let op = Operator::coulomb();
    let ctx = ParallelContext::default();
    let n = prep.nbasis();

    let t0 = Instant::now();
    let sb = SchwarzBounds::compute(op, prep).expect("Schwarz bounds");
    let schwarz_part = t0.elapsed();
    let qb = QqrBounds::try_new(sb, mol, bs, prep, op).expect("QQR bounds");
    let table = t0.elapsed();

    let t1 = Instant::now();
    let mut link = LinkK::new(&ctx, prep, &qb, op, thresh, usize::MAX);
    link.update_density(d);
    let pairlists = t1.elapsed();

    let mut k = Array2::zeros((n, n));
    let mut best = Duration::MAX;
    let mut quartets = 0usize;
    for _ in 0..K_BUILD_REPEATS {
        k.fill(0.0);
        let t2 = Instant::now();
        quartets = link.build(d, &mut k).expect("LinK build");
        best = best.min(t2.elapsed());
    }
    (
        Timing { table, pairlists, k_build: best, quartets },
        schwarz_part,
        k,
    )
}

fn max_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f64, f64::max)
}

/// Print the full cost comparison for one system.
fn report(stem: &str, basis_name: &str) {
    let mol = load_mol(stem);
    let bs = basis::bundled(basis_name).expect("basis");
    let prep = PreparedBasis::new(&mol, &bs).expect("prepared basis");

    println!(
        "\n=== {stem} / {basis_name}: {} atoms, {} shells, {} basis fns, diameter {:.1} Bohr ===",
        mol.atoms.len(),
        prep.nshells(),
        prep.nbasis(),
        diameter_bohr(&mol),
    );

    let t_scf = Instant::now();
    let d = converged_density(&mol, &prep);
    println!("reference density: converged RHF in {:.2?}", t_scf.elapsed());

    for thresh in THRESHOLDS {
        let (ts, k_s) = time_schwarz(&prep, &d, thresh);
        let (tq, qqr_schwarz_part, k_q) = time_qqr(&mol, &bs, &prep, &d, thresh);

        let marginal_table = tq.table.saturating_sub(qqr_schwarz_part);
        let quartet_saving = 100.0
            * (ts.quartets as f64 - tq.quartets as f64)
            / (ts.quartets as f64).max(1.0);
        let k_speedup = ts.k_build.as_secs_f64() / tq.k_build.as_secs_f64();

        println!("\n--- thresh = {thresh:.0e} ---");
        println!(
            "  {:<8} table {:>10.3?}  pairlists {:>10.3?}  K build {:>10.3?}  quartets {:>12}",
            "Schwarz", ts.table, ts.pairlists, ts.k_build, ts.quartets
        );
        println!(
            "  {:<8} table {:>10.3?}  pairlists {:>10.3?}  K build {:>10.3?}  quartets {:>12}",
            "QQR", tq.table, tq.pairlists, tq.k_build, tq.quartets
        );
        println!(
            "  QQR marginal table cost (beyond the Schwarz it wraps): {:.3?}",
            marginal_table
        );
        println!(
            "  quartets screened by QQR beyond Schwarz: {:.3}%   K-build speedup: {:.4}x",
            quartet_saving, k_speedup
        );
        println!(
            "  per-iteration delta (QQR - Schwarz, pairlists + K build): {:+.3?}",
            (tq.pairlists.as_secs_f64() + tq.k_build.as_secs_f64())
                - (ts.pairlists.as_secs_f64() + ts.k_build.as_secs_f64())
        );
        println!("  max|K_QQR - K_Schwarz| = {:.3e}", max_abs_diff(&k_s, &k_q));
    }
}

#[test]
#[ignore = "cost measurement, minutes-scale; run explicitly (see module docs)"]
fn qqr_vs_schwarz_cost_alkane_8() {
    report("alkane_8", "cc-pvdz");
}

/// SCOPE CONTROL: a 3-D system, and the same system in a DIFFUSE basis.
///
/// The alkane cases are 1-D and gapped — the friendliest possible case for
/// LinK's density-pair screen, and therefore the case where QQR has the LEAST
/// left to prune once LinK has run. Concluding anything general from them alone
/// would be an assumption about molecular topology, not a measurement.
///
/// Benzene is compact and 3-D; aug-cc-pVDZ adds diffuse functions that widen
/// BOTH LinK pair lists (`SignificantPairs` via larger Schwarz factors,
/// `DensityPairs` via slower density decay). If QQR's marginal benefit is
/// still ~0.01% here, the dilution is a property of the LinK composition and
/// the alkane numbers generalize. If it jumps, the alkane result is
/// topology-specific and must not be cited beyond chains.
///
/// The artifact hypothesis, stated before running (repo rule): if the dilution
/// is real I expect benzene/aug-cc-pVDZ to stay well under ~0.5%; if the alkane
/// result was an artifact of 1-D geometry I expect it to rise toward the
/// full-population few-percent figure.
#[test]
#[ignore = "cost measurement, minutes-scale; run explicitly (see module docs)"]
fn qqr_vs_schwarz_cost_benzene_3d() {
    report("benzene", "cc-pvdz");
    report("benzene", "aug-cc-pvdz");
}

#[test]
#[ignore = "cost measurement, minutes-scale; run explicitly (see module docs)"]
fn qqr_vs_schwarz_cost_alkane_16() {
    report("alkane_16", "cc-pvdz");
}

#[test]
#[ignore = "cost measurement, many minutes; run explicitly (see module docs)"]
fn qqr_vs_schwarz_cost_alkane_20() {
    report("alkane_20", "cc-pvdz");
}
