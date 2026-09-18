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
        "\n{:<14} {:<12} {:>6} {:>14} {:>14}",
        "molecule", "basis", "nbf", "HF grad peak", "KS grad peak"
    );
    println!("{}", "-".repeat(66));

    for (mol_name, basis) in [
        ("water", "6-31g"),
        ("water", "cc-pvdz"),
        ("benzene", "6-31g"),
        ("benzene", "cc-pvdz"),
        ("alkane_10", "6-31g"),
        ("alkane_16", "6-31g"),
    ] {
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
        clear_global();

        println!(
            "{mol_name:<14} {basis:<12} {nbf:>6} {:>11.3} GB {:>11.3} GB",
            hf_peak as f64 / 1e9,
            ks_peak as f64 / 1e9
        );
    }
    println!(
        "\nThe KS column is the one that matters: it carries ddchi (AO SECOND \n\
         derivatives, 9 of 13 AO planes) and should dominate the HF column by \n\
         a wide and GROWING margin. If it does not, the XC charge is \n\
         mis-apportioned.\n"
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
