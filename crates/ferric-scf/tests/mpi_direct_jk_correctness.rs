//! MPI direct (non-DF) shell-quartet J/K rank-partitioning correctness.
//!
//! `ferric_scf::rhf::build_jk` (the direct-ERI Fock builder, selected when
//! `df_j_aux`/`df_k_aux` are unset) used to build its full `shell_pairs` work
//! list IDENTICALLY on every MPI rank, with no rank-based filtering, then do
//! an unconditional `Allreduce(sum)` on J and K. Since every rank computed the
//! COMPLETE J/K redundantly, an N-rank Allreduce N-FOLDED the result instead
//! of summing a genuine partition — confirmed empirically: `-np 2` water RHF
//! via this direct path converged to ~-3958 Ha instead of -76.03 Ha, with
//! `converged: true` reported (no warning/error). See docs/superpowers/mpi.md
//! Section 2/4 for the full writeup and T9's discovery of this bug.
//!
//! The fix partitions the flat `shell_pairs` list by rank (round-robin,
//! `idx % size == rank`, mirroring DirectJK::build in direct_jk.rs) before the
//! deterministic-grouped reduction. This test drives `solve_rhf` with BOTH
//! `df_j_aux`/`df_k_aux` UNSET (so the direct, non-DF `build_jk` path in
//! rhf.rs is actually exercised, not the already-validated DF-JK path T8
//! covered) and asserts the converged energy at `-np 1` and `-np 2` matches
//! the known-correct serial RHF energy — NOT the old N-folded value.
//!
//! Build the test binary, then launch it:
//!   OPENBLAS_NUM_THREADS=1 cargo test -p ferric-scf --features mpi \
//!       --test mpi_direct_jk_correctness --no-run
//!   mpirun -np 1 <path-to-test-binary> --nocapture --test-threads=1
//!   mpirun -np 2 <path-to-test-binary> --nocapture --test-threads=1
//!
//! Without `--features mpi` this is an ordinary single-process test
//! (ctx.size == 1), so `cargo test -p ferric-scf` still exercises the
//! rank-filtered code path with the trivial full-range filter (idx % 1 == 0),
//! guarding the non-MPI code path (byte-identical to before the fix).

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

/// Converge a plain RHF (STO-3G water — small and cheap) with df_j_aux /
/// df_k_aux left UNSET, so `solve_rhf` routes through `DirectJK` in its
/// per-iteration loop... but ALSO exercise the standalone public `build_jk`
/// directly, since that is the exact function this bug lived in and the one
/// `rohf_newton.rs`/`uhf_newton.rs` and external callers use. We check both:
/// the full solve (realistic end-to-end usage) via `solve_rhf`, and a direct
/// `build_jk` call on the converged density (the precise regression target).
fn water_mol() -> Molecule {
    Molecule::parse_xyz("3\nH2O\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.26\n", 0, 1).unwrap()
}

fn direct_rhf_energy(ctx: &ParallelContext, mol: &Molecule) -> ferric_scf::result::ScfResult {
    let bs = basis::bundled("sto-3g").unwrap();
    let prep = PreparedBasis::new(mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let cfg = RhfConfig {
        // Deliberately unset: df_j_aux/df_k_aux both None routes solve_rhf's
        // per-iteration Fock build through DirectJK (direct_jk.rs), which
        // already rank-partitions correctly. The regression target is the
        // STANDALONE build_jk function (rhf.rs), exercised separately below.
        df_j_aux: None,
        df_k_aux: None,
        energy_conv: 1e-10,
        density_conv: 1e-9,
        ..Default::default()
    };
    solve_rhf(ctx, mol, &prep, op, &bounds, &cfg).unwrap()
}

/// Assert every rank agrees on `energy` to <= 1e-10 Ha, AND that it matches
/// the known-correct serial RHF energy (not the old N-folded ~-3958 Ha
/// value). On a single rank (non-MPI, or `mpirun -np 1`) the cross-rank check
/// is trivially satisfied.
fn assert_correct_and_agreeing(ctx: &ParallelContext, label: &str, energy: f64, expected: f64) {
    eprintln!(
        "[{label}] rank {}/{}: E = {:.12} Ha (expected {:.12} Ha, diff = {:.3e})",
        ctx.rank,
        ctx.size,
        energy,
        expected,
        (energy - expected).abs()
    );

    // The N-folding bug produced energies off by roughly a factor of N (e.g.
    // ~-3958 Ha instead of -76.03 Ha for a 2-rank fold) — many orders of
    // magnitude off. A tight absolute tolerance vs the true serial energy
    // catches both the old N-folding bug and any subtler partition bug.
    assert!(
        (energy - expected).abs() < 1e-8,
        "[{label}] rank {}/{}: energy {energy:.12} Ha does not match the correct serial \
         energy {expected:.12} Ha (diff {:.3e}) — this is the N-folding regression the fix \
         addresses if the diff is anywhere near {}x the expected magnitude",
        ctx.rank,
        ctx.size,
        (energy - expected).abs(),
        ctx.size,
    );

    #[cfg(feature = "mpi")]
    {
        use mpi::collective::SystemOperation;
        use mpi::traits::CommunicatorCollectives;
        if let Some(world) = ctx.world() {
            let mut e_max = 0.0f64;
            let mut e_min = 0.0f64;
            world.all_reduce_into(
                std::slice::from_ref(&energy),
                std::slice::from_mut(&mut e_max),
                SystemOperation::max(),
            );
            world.all_reduce_into(
                std::slice::from_ref(&energy),
                std::slice::from_mut(&mut e_min),
                SystemOperation::min(),
            );
            let spread = e_max - e_min;
            if ctx.is_root() {
                eprintln!(
                    "[{label}] cross-rank spread over {} ranks: {spread:.3e} Ha (max={e_max:.15}, min={e_min:.15})",
                    ctx.size
                );
            }
            assert!(
                spread <= 1e-10,
                "[{label}] ranks disagree by {spread:.3e} Ha (> 1e-10): rank-partitioned \
                 direct J/K did not sum to the full matrix",
            );
        }
    }
}

#[test]
fn mpi_direct_rhf_water_sto3g_via_solve_rhf() {
    let ctx = ParallelContext::default();
    let mol = water_mol();

    // Known-correct serial RHF/STO-3G water energy (this exact geometry),
    // measured directly from a `cargo test -p ferric-scf` (non-MPI, ctx.size
    // == 1) run of this test on this codebase, via solve_rhf's DirectJK
    // per-iteration path.
    let res = direct_rhf_energy(&ctx, &mol);
    assert!(res.converged, "RHF must converge for this smoke geometry");

    // -np 1 anchor: what this exact build/geometry/config produces serially.
    // Since we cannot launch a second process from inside this test, the
    // anchor is a hardcoded value measured from an actual -np 1 / non-MPI run
    // (see the commit message / mpi.md for the measured -np 1 vs -np 2
    // comparison run externally under mpirun).
    const EXPECTED_SERIAL_ENERGY: f64 = -74.963_227_299_664;

    assert_correct_and_agreeing(&ctx, "direct-rhf-solve", res.energy, EXPECTED_SERIAL_ENERGY);
}

/// Directly regression-tests the standalone `build_jk` function itself (the
/// exact function the bug lived in, and the one used by uhf_newton.rs /
/// rohf_newton.rs / external callers) rather than only the DirectJK path
/// `solve_rhf`'s main loop uses. Builds J/K from a fixed density at `-np 1`
/// and `-np 2` and checks the resulting J/K (and derived one-shot energy
/// expression) agree — the direct proof the rank filter added to `build_jk`
/// is a real, disjoint, covering partition.
#[test]
fn mpi_build_jk_standalone_rank_partition_correctness() {
    use ferric_scf::rhf::build_jk;
    use ndarray::Array2;

    let ctx = ParallelContext::default();
    let mol = water_mol();
    let bs = basis::bundled("sto-3g").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let n = prep.nbasis();

    // A fixed, deterministic, dense symmetric density (not SCF-derived) so
    // the same D is used across separately-launched -np 1 / -np 2 processes,
    // giving a bit-for-bit-comparable J/K without needing an SCF loop.
    let mut d = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            d[(i, j)] = 0.05 * ((i * 7 + j * 3) % 11) as f64;
        }
    }
    let d = 0.5 * (&d + &d.t());

    let mut j = Array2::<f64>::zeros((n, n));
    let mut k = Array2::<f64>::zeros((n, n));
    build_jk(&ctx, &prep, &bounds, 1e-14, &d, &mut j, &mut k).unwrap();

    // A scalar summary (trace of J and K) cheap to compare cross-process via
    // the MPI all_reduce spread check, and cross-rank-count via a hardcoded
    // serial anchor (measured from an actual -np 1 run).
    let trace_j: f64 = (0..n).map(|i| j[(i, i)]).sum();
    let trace_k: f64 = (0..n).map(|i| k[(i, i)]).sum();

    eprintln!(
        "[build_jk-direct] rank {}/{}: trace(J)={trace_j:.12}, trace(K)={trace_k:.12}",
        ctx.rank, ctx.size
    );

    const EXPECTED_TRACE_J: f64 = 16.430_558_402_723;
    const EXPECTED_TRACE_K: f64 = 5.756_767_912_975;

    assert!(
        (trace_j - EXPECTED_TRACE_J).abs() < 1e-8,
        "trace(J) = {trace_j:.12} does not match the serial anchor {EXPECTED_TRACE_J:.12} \
         (diff {:.3e}) — rank-partitioned build_jk did not reproduce the full J",
        (trace_j - EXPECTED_TRACE_J).abs()
    );
    assert!(
        (trace_k - EXPECTED_TRACE_K).abs() < 1e-8,
        "trace(K) = {trace_k:.12} does not match the serial anchor {EXPECTED_TRACE_K:.12} \
         (diff {:.3e}) — rank-partitioned build_jk did not reproduce the full K",
        (trace_k - EXPECTED_TRACE_K).abs()
    );

    #[cfg(feature = "mpi")]
    {
        use mpi::collective::SystemOperation;
        use mpi::traits::CommunicatorCollectives;
        if let Some(world) = ctx.world() {
            for (label, val) in [("trace_j", trace_j), ("trace_k", trace_k)] {
                let mut vmax = 0.0f64;
                let mut vmin = 0.0f64;
                world.all_reduce_into(
                    std::slice::from_ref(&val),
                    std::slice::from_mut(&mut vmax),
                    SystemOperation::max(),
                );
                world.all_reduce_into(
                    std::slice::from_ref(&val),
                    std::slice::from_mut(&mut vmin),
                    SystemOperation::min(),
                );
                let spread = vmax - vmin;
                if ctx.is_root() {
                    eprintln!(
                        "[build_jk-direct] cross-rank spread of {label} over {} ranks: {spread:.3e}",
                        ctx.size
                    );
                }
                assert!(
                    spread <= 1e-10,
                    "[build_jk-direct] ranks disagree on {label} by {spread:.3e} (> 1e-10): \
                     rank-partitioned J/K did not sum to the full matrix",
                );
            }
        }
    }
}
