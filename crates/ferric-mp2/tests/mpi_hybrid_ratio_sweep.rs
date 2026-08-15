//! Hybrid rank x thread ratio sweep for MPI RI-MP2 (measurement harness).
//!
//! The hypothesis under test: when ONE parallel axis is too short to fill the
//! box, a SECOND axis multiplies where the first does not stretch. Ranks and
//! threads are those two axes here — `run_mpi_ri_mp2` distributes the aux band
//! and the occupied index `i` across ranks, and rayon threads the inner
//! GEMM/ERI work within each rank.
//!
//! This file measures ONLY wall time; correctness at every rank count is
//! already pinned by `mpi_rimp2_banding.rs` (cross-rank agreement to 1e-12).
//! The rank-aware pool width itself comes from
//! `ParallelContext::rayon_threads`, so a run under `mpirun -np N` gets
//! `floor(physical_cores / N)` threads per rank automatically.
//!
//! `#[ignore]` — this is a benchmark, not a correctness gate. Run explicitly:
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo test --release -p ferric-mp2 --features mpi \
//!     --test mpi_hybrid_ratio_sweep --no-run
//! mpirun -np 2 --bind-to core --map-by slot:PE=3 <bin> \
//!     hybrid_shape --ignored --nocapture --test-threads=1
//! ```
//!
//! ## Reading the numbers
//!
//! The box this was written for has 6 PHYSICAL cores with 2-way SMT (12
//! logical) on a SINGLE NUMA node. Two consequences:
//!
//!  * There is no NUMA locality to exploit, so no affinity logic beyond
//!    core binding.
//!  * SMT is a CONFOUNDER. Filling 12 logical cores on 6 physical ones can
//!    look like a speedup that is really just hyperthreading. Every
//!    configuration in the honest grid (1x6, 2x3, 3x2, 6x1) uses exactly 6
//!    physical cores; any 12-logical configuration must be reported
//!    SEPARATELY and never mixed into the same comparison.

#![cfg(feature = "mpi")]

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::mpi_rimp2::run_mpi_ri_mp2;
use ferric_mp2::rimp2::RiMp2Config;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

fn peak_rss_mib() -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|kb| kb.parse::<f64>().ok())
        })
        .map(|kb| kb / 1024.0)
        .unwrap_or(0.0)
}

/// Same DF-J/DF-K + hcore-guess RHF reference `mpi_rimp2_banding.rs` uses, and
/// for the same reason: it avoids the two pre-existing `ferric-scf` MPI bugs
/// documented at that file's `df_rhf_config` (the direct-path `build_jk`
/// Allreduce N-folding, and SAD's free-atom solves reusing the outer MPI
/// world). Neither is in scope here; using the same known-good reference keeps
/// this a measurement of RI-MP2, not of those bugs.
fn df_rhf_config() -> RhfConfig {
    RhfConfig {
        df_j_aux: Some("def2-universal-jkfit".into()),
        df_k_aux: Some("def2-universal-jkfit".into()),
        energy_conv: 1e-10,
        density_conv: 1e-8,
        use_sad_guess: false,
        ..Default::default()
    }
}

/// Time `run_mpi_ri_mp2` on one molecule/basis and print a machine-parseable
/// line. RHF is converged FIRST and excluded from the timer: the SCF has its
/// own (different) parallel structure, and including it would dilute exactly
/// the RI-MP2 scaling this sweep is trying to resolve.
fn time_shape(label: &str, xyz_rel: &str, obs_name: &str, aux_name: &str, reps: usize) {
    let ctx = ParallelContext::default();
    let mol = Molecule::load_xyz(&format!(
        "{}/../../testdata/molecules/{}",
        env!("CARGO_MANIFEST_DIR"),
        xyz_rel
    ))
    .unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled(obs_name).unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled(aux_name).unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &df_rhf_config()).unwrap();
    // NOT asserted. The hcore-guess + DF-JK reference `df_rhf_config` uses (see
    // its doc: it is chosen to dodge two pre-existing ferric-scf MPI bugs, not
    // for its convergence behavior) does not always reach the 1e-10/1e-8
    // thresholds on the larger systems here — a PRE-EXISTING SCF issue,
    // reproduced with this file's changes reverted, and entirely upstream of
    // the RI-MP2 code being timed. It does not invalidate a WALL-TIME
    // measurement: RI-MP2's cost depends on the tensor SHAPES (naux, nocc,
    // nvir), not on how converged the orbitals are, and every configuration in
    // the sweep is fed the identical `rhf` for a given shape. It WOULD
    // invalidate an energy claim, which is why the convergence flag is printed
    // on every line rather than silently dropped, and why correctness lives in
    // `mpi_rimp2_banding.rs` instead of here.
    let cfg = RiMp2Config::default();

    let nbas = obs.nbasis();
    let naux = dfbs.nbasis();
    let nocc = mol.nelec() as usize / 2;
    let nvir = nbas - nocc;

    // One untimed warm-up pass, then `reps` timed passes reported as the MIN.
    //
    // Min, not mean: on a shared box the noise is one-sided — an interfering
    // process can only ever make a run SLOWER, never faster — so the minimum
    // is the best estimate of the cost of the work itself, while a mean drags
    // in whatever else the machine was doing. (This is the standard choice for
    // wall-clock benchmarks and is why a single-pass timing on a busy box is
    // untrustworthy.) The warm-up pass absorbs first-touch page faults and any
    // lazy allocator growth so they are not charged to rep 1.
    //
    // `reps` is per-shape because these shapes differ by ~3 orders of
    // magnitude in cost: a 0.17 s shape needs repetition to rise above timer
    // and scheduler noise, a 40 s shape does not and repeating it would just
    // occupy the box.
    let _warm = run_mpi_ri_mp2(&ctx, &mol, &obs, &dfbs, op, &rhf, &cfg).unwrap();
    let mut secs = f64::INFINITY;
    let mut res = _warm;
    for _ in 0..reps.max(1) {
        // Every rank must enter the timed region together, or rank 0's clock
        // would also be measuring how late the slowest rank arrived from the
        // PREVIOUS rep rather than the cost of this one.
        if let Some(world) = ctx.world() {
            use mpi::traits::CommunicatorCollectives;
            world.barrier();
        }
        let t0 = std::time::Instant::now();
        res = run_mpi_ri_mp2(&ctx, &mol, &obs, &dfbs, op, &rhf, &cfg).unwrap();
        secs = secs.min(t0.elapsed().as_secs_f64());
    }

    // Only the root prints the headline number; per-rank lines would interleave.
    if ctx.is_root() {
        println!(
            "RATIO_SWEEP shape={label} ranks={} threads_per_rank={} nbas={nbas} naux={naux} \
             nocc={nocc} nvir={nvir} secs={secs:.3} mp2_corr={:.12} peak_rss_mib={:.1} \
             scf_converged={}",
            ctx.size,
            ctx.rayon_threads(),
            res.mp2_corr,
            peak_rss_mib(),
            rhf.converged
        );
    }
}

/// **Thread-hostile shape** (the hypothesis case). Water in a quadruple-zeta
/// basis: `nocc = 5` while `nbas`/`naux` are large. That makes the occupied
/// index a VERY short axis — the round-robin over `i` has only 5 items — while
/// the aux axis is long. This is the "too few rows to band widely" situation
/// that motivates a second axis. Note the two axes are hit differently: ranks
/// split BOTH the aux band (long, splits well) and `i` (short, splits badly at
/// >5 ranks), whereas threads within a rank see only whatever rayon regions
/// the inner code exposes. If hybrid ever wins, it wins here.
#[test]
#[ignore]
fn hybrid_shape_water_qz() {
    time_shape("water_def2qzvp", "water.xyz", "def2-qzvp", "def2-qzvp-rifit", 10);
}

/// **Control shape** that already threads well: a bigger molecule where `nocc`
/// is long (41) and the tensors are large enough to keep every rayon worker
/// busy on a single rank. Pure threading should be at or near the best
/// configuration here, and hybrid should show no advantage — that contrast is
/// what makes the thread-hostile result above interpretable rather than just
/// "MPI has overhead". Memory: full B_ov is ~67 MiB, so even 6 ranks each
/// holding a band plus the replicated metric stays well inside the box.
#[test]
#[ignore]
fn hybrid_shape_alkane10() {
    time_shape("alkane_10_ccpvdz", "alkane_10.xyz", "cc-pvdz", "cc-pvdz-ri", 2);
}
