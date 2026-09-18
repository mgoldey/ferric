//! `#[ignore]`d peak-memory and capacity measurements for the gradient planes.
//!
//! These are DELIBERATELY not part of the default suite. They run at
//! production shapes, take minutes, and the numbers they print are only
//! meaningful on a quiet box — on a loaded one they measure contention. Run
//! them serially:
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo test -p ferric-scf \
//!     --test gradient_pool_peak_measurement -- --ignored --nocapture --test-threads=1
//! ```
//!
//! What each one answers:
//!
//! * `report_gradient_plane_peaks_by_system_size` — WHICH plane dominates the
//!   gradient path, and how that shifts with the basis. This is the number
//!   that decides whether the charges are apportioned sensibly, and the one
//!   the migration could not obtain without a quiet box.
//! * `report_ks_gradient_peak_under_a_real_budget` — what the XC-gradient
//!   BATCHING bought, as a ratio. Note the distinction from the row above:
//!   that one installs an ample pool on purpose, so its KS column measures
//!   DEMAND and is unchanged by batching. Reading it as "the peak after
//!   batching" is a misattribution; this test is the one to read for that.
//! * `report_ks_gradient_peak_against_the_scf_that_precedes_it` — the
//!   composition question that actually matters in production: during a
//!   geometry optimization the SCF's planes are still resident when the
//!   gradient runs. This prints both, so the sum can be compared against a
//!   real `budget_gb`.
//! * `find_minimum_pool_that_completes_the_ks_gradient` — a capacity
//!   binary-search. It answers "what is the smallest `budget_gb` that still
//!   runs this job", which is what a refusal-regression would move.
//!
//! None of these assert a frozen number. They PRINT. A test that pinned a
//! measured byte count would fail on a different box for reasons that have
//! nothing to do with correctness, and the standing convention here is to
//! separate measurement from interpretation.

use ferric_core::memory::pool::{clear_global, global, install_global, MemoryPool};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::gradient::rhf_gradient;
use ferric_scf::ks_gradient::ks_gradient_closed;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

/// A large pool: big enough that nothing is refused, so `peak_bytes()` is a
/// pure measurement of demand rather than of the ceiling.
const AMPLE: usize = 200_000_000_000;

fn load(name: &str) -> Molecule {
    Molecule::load_xyz(&format!("../../testdata/molecules/{name}.xyz"))
        .unwrap_or_else(|e| panic!("{name}.xyz: {e}"))
}

#[test]
#[ignore = "production-scale measurement; run serially on a quiet box"]
fn report_gradient_plane_peaks_by_system_size() {
    println!(
        "\n{:<14} {:<12} {:>6} {:>14} {:>14} {:>9} {:>9}",
        "molecule", "basis", "nbf", "HF grad peak", "KS grad peak", "npts", "batch"
    );
    println!("{}", "-".repeat(86));

    // The four shapes the batching work is measured against, plus two larger
    // ones. `FERRIC_PEAK_ROWS=4` runs only the first four — the alkanes each
    // need their own SCF and dominate the runtime, and the scaling question
    // this answers is already settled by water→benzene.
    let all: &[(&str, &str)] = &[
        ("water", "6-31g"),
        ("water", "cc-pvdz"),
        ("benzene", "6-31g"),
        ("benzene", "cc-pvdz"),
        ("alkane_10", "6-31g"),
        ("alkane_16", "6-31g"),
    ];
    let n: usize = std::env::var("FERRIC_PEAK_ROWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(all.len())
        .min(all.len());

    for &(mol_name, basis) in &all[..n] {
        let mol = load(mol_name);
        let bs = ferric_core::basis::bundled(basis).expect("basis");
        let obs = PreparedBasis::new(&mol, &bs).expect("prepared");
        let nbf = obs.nbasis();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).expect("schwarz");
        let ctx = ParallelContext::default();

        // HF gradient.
        let scf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default())
            .expect("RHF must converge");
        install_global(MemoryPool::with_capacity_bytes(AMPLE));
        rhf_gradient(&mol, &obs, op, &bounds, &scf, None).expect("HF gradient");
        let hf_peak = global().expect("pool").peak_bytes();
        clear_global();

        // KS-DFT (PBE) gradient — the ddchi path.
        let kscfg = RhfConfig {
            xc: Some("PBE".into()),
            ..Default::default()
        };
        let ksscf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &kscfg).expect("KS-DFT must converge");
        install_global(MemoryPool::with_capacity_bytes(AMPLE));
        ks_gradient_closed(&mol, &obs, &bs, op, &bounds, "PBE", &ksscf, None).expect("KS gradient");
        let ks_peak = global().expect("pool").peak_bytes();
        // The width production would take against THIS pool. Printed beside
        // the peak because the two are the same statement: the peak is one
        // batch's planes plus the non-batchable grid, so a width equal to
        // `npts` means "no batching happened here" and the peak is the old
        // whole-grid number.
        let npts = ferric_dft::grid::build_atomic_grid(
            &mol,
            &ferric_dft::grid::AtomicGridConfig::default(),
        )
        .len();
        let width =
            ferric_dft::gradient::grad_batch_pts_for_test(nbf, npts, mol.atoms.len(), false);
        clear_global();

        println!(
            "{mol_name:<14} {basis:<12} {nbf:>6} {:>11.3} GB {:>11.3} GB {npts:>9} {width:>9}",
            hf_peak as f64 / 1e9,
            ks_peak as f64 / 1e9
        );
    }
    println!(
        "\nThe KS column is the one that matters: it carries ddchi (AO SECOND \n\
         derivatives, 9 of 13 AO planes) and should dominate the HF column by \n\
         a wide and GROWING margin. If it does not, the XC charge is \n\
         mis-apportioned.\n\
         \n\
         The `batch` column is the grid batch width the XC gradient would take \n\
         against THIS pool. These rows install an AMPLE pool deliberately (so \n\
         the peak measures demand, not the ceiling), which means batch == npts \n\
         and the KS peak is the FULL-GRID working set — the number to compare \n\
         against a real `budget_gb`. Under a production budget the width \n\
         shrinks and the peak with it; `ks_gradient_batching.rs` measures that \n\
         ratio directly.\n"
    );
}

/// THE NUMBER THE BATCHING WORK MOVED: the KS-gradient peak under a REAL
/// budget, beside the same job's unbatched (ample-pool) demand.
///
/// `report_gradient_plane_peaks_by_system_size` deliberately installs an ample
/// pool so its column measures demand; that column is therefore unchanged by
/// batching, and reading it as "the new peak" would be a misattribution. This
/// one installs the kind of budget a production run actually has, which is
/// what makes the gradient batch at all.
///
/// The budget is a FIXED constant, not a fraction of the measured demand: the
/// question a user asks is "does a 0.25 GB budget run my molecule", and a
/// budget that scaled with the system would answer a different question at
/// every row while looking like a trend.
///
/// # Choosing the constant
///
/// MEASURED first, then chosen — at 2 GB every row here reported a ratio of
/// exactly 1.0x, because the largest of them (benzene/cc-pVDZ) demands 1.86 GB
/// and therefore never batches. That is CORRECT behaviour (batch only when the
/// working set does not fit) but it is a vacuous measurement: a column of 1.0x
/// says nothing about batching, and a reader could mistake it for "batching
/// does not help". 0.25 GB is below every row's demand except the smallest, so
/// the ratio column reports real narrowing, and the small rows are left as the
/// control that shows an unbatched job is still unbatched.
#[test]
#[ignore = "production-scale measurement; run serially on a quiet box"]
fn report_ks_gradient_peak_under_a_real_budget() {
    // 0.25 GB — below benzene's 1.10/1.86 GB demand and water/cc-pVDZ's
    // 0.103 GB is just under it, so the table spans both sides of the
    // batch/no-batch decision. See the doc comment for why 2 GB was rejected.
    const BUDGET: usize = 250_000_000;

    println!(
        "\n{:<12} {:<10} {:>5} {:>8} {:>7} {:>13} {:>13} {:>7}",
        "molecule", "basis", "nbf", "npts", "batch", "unbatched peak", "batched peak", "ratio"
    );
    println!("{}", "-".repeat(84));

    let all: &[(&str, &str)] = &[
        ("water", "6-31g"),
        ("water", "cc-pvdz"),
        ("benzene", "6-31g"),
        ("benzene", "cc-pvdz"),
    ];
    let n: usize = std::env::var("FERRIC_PEAK_ROWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(all.len())
        .min(all.len());

    for &(mol_name, basis) in &all[..n] {
        let mol = load(mol_name);
        let bs = ferric_core::basis::bundled(basis).expect("basis");
        let obs = PreparedBasis::new(&mol, &bs).expect("prepared");
        let nbf = obs.nbasis();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).expect("schwarz");
        let ctx = ParallelContext::default();
        let cfg = RhfConfig {
            xc: Some("PBE".into()),
            ..Default::default()
        };
        let scf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &cfg).expect("KS-DFT must converge");
        let npts = ferric_dft::grid::build_atomic_grid(
            &mol,
            &ferric_dft::grid::AtomicGridConfig::default(),
        )
        .len();

        // Unbatched demand: an ample pool, so the width is the whole grid.
        install_global(MemoryPool::with_capacity_bytes(AMPLE));
        ks_gradient_closed(&mol, &obs, &bs, op, &bounds, "PBE", &scf, None).expect("KS gradient");
        let unbatched = global().expect("pool").peak_bytes();
        clear_global();

        // The same job under a budget a real run would have.
        install_global(MemoryPool::with_capacity_bytes(BUDGET));
        let width =
            ferric_dft::gradient::grad_batch_pts_for_test(nbf, npts, mol.atoms.len(), false);
        let r = ks_gradient_closed(&mol, &obs, &bs, op, &bounds, "PBE", &scf, None);
        let batched = global().expect("pool").peak_bytes();
        clear_global();
        assert!(
            r.is_ok(),
            "{mol_name}/{basis} must COMPLETE under {BUDGET} B, not refuse: {r:?}"
        );

        println!(
            "{mol_name:<12} {basis:<10} {nbf:>5} {npts:>8} {width:>7} {:>10.3} GB {:>10.3} GB {:>6.1}x",
            unbatched as f64 / 1e9,
            batched as f64 / 1e9,
            unbatched as f64 / batched.max(1) as f64,
        );
    }
    println!(
        "\nThe `batched peak` column is bounded by the budget by construction \n\
         — that is what the batch sizing does. What the `ratio` says is how \n\
         much of the old demand was the grid planes (batchable) rather than \n\
         the grid + weight1 arrays (not batchable), and therefore how far this \n\
         can be pushed on a bigger system.\n"
    );
}

#[test]
#[ignore = "production-scale measurement; run serially on a quiet box"]
fn report_ks_gradient_peak_against_the_scf_that_precedes_it() {
    // The production question: during a geometry optimization the SCF's own
    // planes are resident when the gradient runs, so what a `budget_gb` has to
    // cover is the SUM, not either one.
    let mol = load("benzene");
    let bs = ferric_core::basis::bundled("cc-pvdz").expect("basis");
    let obs = PreparedBasis::new(&mol, &bs).expect("prepared");
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).expect("schwarz");
    let ctx = ParallelContext::default();
    let cfg = RhfConfig {
        xc: Some("PBE".into()),
        df_j_aux: Some("def2-universal-jkfit".into()),
        ..Default::default()
    };

    // SCF inside the pool window: its peak is what an optimizer is holding.
    install_global(MemoryPool::with_capacity_bytes(AMPLE));
    let scf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &cfg).expect("KS-DFT must converge");
    let scf_peak = global().expect("pool").peak_bytes();

    // Gradient in the SAME pool, with the SCF result still alive.
    ks_gradient_closed(&mol, &obs, &bs, op, &bounds, "PBE", &scf, None).expect("KS gradient");
    let total_peak = global().expect("pool").peak_bytes();
    let report = global().expect("pool").occupancy_report();
    clear_global();

    println!("\nbenzene / cc-pVDZ / PBE / RI-J");
    println!("  SCF peak alone      : {:.3} GB", scf_peak as f64 / 1e9);
    println!("  SCF + gradient peak : {:.3} GB", total_peak as f64 / 1e9);
    println!(
        "  gradient adds       : {:.3} GB",
        total_peak.saturating_sub(scf_peak) as f64 / 1e9
    );
    println!("\nfinal occupancy:\n{report}");
}

#[test]
#[ignore = "capacity binary-search; run serially on a quiet box"]
fn find_minimum_pool_that_completes_the_ks_gradient() {
    // What is the smallest budget that still runs this job? That number is
    // what a refusal regression moves, and it is the honest way to check the
    // standing rule that the migration must not refuse a job the pre-migration
    // tree completes: run the same search with no pool installed (which cannot
    // refuse) and confirm the pooled floor is not far above it.
    let mol = load("water");
    let bs = ferric_core::basis::bundled("cc-pvdz").expect("basis");
    let obs = PreparedBasis::new(&mol, &bs).expect("prepared");
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).expect("schwarz");
    let ctx = ParallelContext::default();
    let cfg = RhfConfig {
        xc: Some("PBE".into()),
        ..Default::default()
    };
    let scf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &cfg).expect("KS-DFT must converge");

    let mut lo = 0usize;
    let mut hi = AMPLE;
    // Invariant: `lo` refuses, `hi` completes.
    while hi - lo > hi / 200 {
        let mid = lo + (hi - lo) / 2;
        install_global(MemoryPool::with_capacity_bytes(mid));
        let ok = ks_gradient_closed(&mol, &obs, &bs, op, &bounds, "PBE", &scf, None).is_ok();
        clear_global();
        if ok {
            hi = mid;
        } else {
            lo = mid;
        }
    }

    println!(
        "\nwater / cc-pVDZ / PBE: smallest pool that completes the KS gradient \
         = {:.4} GB\n  (largest that refuses = {:.4} GB)\n",
        hi as f64 / 1e9,
        lo as f64 / 1e9
    );
}
