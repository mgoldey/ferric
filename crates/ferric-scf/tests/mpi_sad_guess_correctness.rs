//! MPI correctness of the SAD (superposition-of-atomic-densities) initial
//! guess when it routes through the DIRECT (non-DF) `build_jk` Fock builder.
//!
//! Background: `ferric_scf::guess::free_atom_density` (guess.rs) builds each
//! element's free-atom density via a sub-`solve_rhf`/`solve_uhf` call using
//! `RhfConfig { use_sad_guess: false, ..Default::default() }` — i.e. with
//! `df_j_aux`/`df_k_aux` left `None`, which routes the free-atom sub-solve
//! through the DIRECT `build_jk` path (rhf.rs), not DF-JK. That sub-solve
//! uses its own fresh `ParallelContext::default()`.
//!
//! `build_jk` used to N-fold under multi-rank MPI (every rank built the FULL
//! J/K then did an unconditional `Allreduce(sum)`) — see
//! `mpi_direct_jk_correctness.rs` and docs/superpowers/mpi.md Section 2/4 for
//! the original bug and its fix (rank-partition the flat `shell_pairs` list
//! by `idx % ctx.size == ctx.rank` before reduction). `docs/superpowers/mpi.md`
//! reasoned that since `free_atom_density`'s sub-solve calls the SAME
//! `build_jk`, and `ParallelContext::default()` correctly derives the real
//! MPI rank/size when the `mpi` feature is active, the SAD-guess path should
//! be fixed as a side effect — but that had never been empirically verified
//! end-to-end (only reasoned through).
//!
//! This test verifies it directly: `solve_rhf` on water (O + H, 2 distinct
//! elements — so the free-atom guess actually exercises MORE than one
//! element/basis-block) with `use_sad_guess: true` (default) AND
//! `df_j_aux`/`df_k_aux` left `None` — so BOTH the free-atom guess
//! construction (inside `sad_guess`/`free_atom_density`) AND the outer
//! molecular SCF's own per-iteration Fock build go through the direct,
//! non-DF path. Run at `-np 1` and `-np 2` and assert the converged energy
//! matches the known-correct serial value and is cross-rank-identical,
//! mirroring `mpi_direct_jk_correctness.rs`'s assertion style.
//!
//! Build the test binary, then launch it:
//!   OPENBLAS_NUM_THREADS=1 cargo test -p ferric-scf --features mpi \
//!       --test mpi_sad_guess_correctness --no-run
//!   mpirun -np 1 <path-to-test-binary> --nocapture --test-threads=1
//!   mpirun -np 2 <path-to-test-binary> --nocapture --test-threads=1
//!
//! Without `--features mpi` this is an ordinary single-process test
//! (ctx.size == 1), so `cargo test -p ferric-scf` still exercises the
//! rank-filtered code path with the trivial full-range filter (idx % 1 == 0).

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

/// Water: 2 distinct elements (O, H), so the SAD free-atom guess must
/// actually build TWO different free-atom densities (not just one element
/// repeated), exercising `free_atom_density` for both Z=8 and Z=1.
fn water_mol() -> Molecule {
    Molecule::parse_xyz("3\nH2O\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.26\n", 0, 1).unwrap()
}

/// Converge RHF/STO-3G water via the SAD guess with df_j_aux/df_k_aux BOTH
/// left unset, so:
///  1. The free-atom sub-solves inside `sad_guess`/`free_atom_density` route
///     through the direct `build_jk` path (their own fresh
///     `RhfConfig::default()` also leaves df_j_aux/df_k_aux unset).
///  2. The outer molecular SCF's own per-iteration Fock build ALSO routes
///     through the direct path (`DirectJK`, already validated by T9/the
///     original build_jk fix — included here so the whole pipeline, guess
///     AND solve, is exercised under MPI in one shot).
fn sad_direct_rhf_energy(ctx: &ParallelContext, mol: &Molecule) -> ferric_scf::result::ScfResult {
    let bs = basis::bundled("sto-3g").unwrap();
    let prep = PreparedBasis::new(mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let cfg = RhfConfig {
        // Deliberately unset: forces BOTH the SAD free-atom guess sub-solves
        // AND the outer molecular SCF through the direct (non-DF) build_jk
        // Fock-builder path.
        df_j_aux: None,
        df_k_aux: None,
        use_sad_guess: true, // default, but explicit: this is the case under test
        energy_conv: 1e-10,
        density_conv: 1e-9,
        ..Default::default()
    };
    solve_rhf(ctx, mol, &prep, op, &bounds, &cfg).unwrap()
}

/// Assert every rank agrees on `energy` to <= 1e-10 Ha, AND that it matches
/// the known-correct serial RHF energy (not an N-folded value, which would be
/// off by roughly a factor of N as in the original build_jk bug).
fn assert_correct_and_agreeing(ctx: &ParallelContext, label: &str, energy: f64, expected: f64) {
    eprintln!(
        "[{label}] rank {}/{}: E = {:.12} Ha (expected {:.12} Ha, diff = {:.3e})",
        ctx.rank,
        ctx.size,
        energy,
        expected,
        (energy - expected).abs()
    );

    assert!(
        (energy - expected).abs() < 1e-8,
        "[{label}] rank {}/{}: energy {energy:.12} Ha does not match the correct serial \
         energy {expected:.12} Ha (diff {:.3e}) — if the diff is anywhere near {}x the \
         expected magnitude, the SAD free-atom guess sub-solve is NOT covered by the \
         build_jk rank-partition fix",
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
                "[{label}] ranks disagree by {spread:.3e} Ha (> 1e-10): the SAD-guess free-atom \
                 sub-solve's direct J/K did not sum to the full matrix on every rank",
            );
        }
    }
}

#[test]
fn mpi_sad_guess_direct_rhf_water_sto3g() {
    let ctx = ParallelContext::default();
    let mol = water_mol();

    let res = sad_direct_rhf_energy(&ctx, &mol);
    assert!(res.converged, "RHF with SAD guess must converge for this smoke geometry");

    // Known-correct serial RHF/STO-3G water energy for this geometry, with
    // use_sad_guess: true and df_j_aux/df_k_aux unset (SAD guess construction
    // AND molecular SCF both via the direct build_jk path). Measured from a
    // `cargo test -p ferric-scf` (non-MPI, ctx.size == 1) run of this exact
    // test/config on this codebase; identical to the plain-hcore-guess direct
    // RHF energy in mpi_direct_jk_correctness.rs (SAD and hcore guesses must
    // converge to the same SCF stationary point for this small, well-behaved
    // system — only the path taken to get there differs).
    const EXPECTED_SERIAL_ENERGY: f64 = -74.963_227_299_664;

    assert_correct_and_agreeing(&ctx, "sad-guess-direct-rhf", res.energy, EXPECTED_SERIAL_ENERGY);
}
