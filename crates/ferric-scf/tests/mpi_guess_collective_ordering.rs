//! Regression guard: the MINAO free-atom guess must issue its MPI collectives
//! in a rank-agreed order.
//!
//! ## The bug this pins
//!
//! `guess::minao_projection_guess` builds one density block per unique element.
//! For a light main-group element that block comes from `free_atom_density`,
//! which runs a full `solve_rhf` against a live MPI `ParallelContext` with
//! DF-J/DF-K aux bases configured — so it drives `DfJ`/`DfK`'s
//! `all_reduce_into` on MPI_COMM_WORLD. Building the elements under a rayon
//! `par_iter` therefore put two elements' collectives in flight concurrently,
//! in an order no two ranks agreed on. MPI matches collectives positionally
//! per communicator, so rank A's H reduce (naux 18) was matched against rank
//! B's O reduce (naux 77): `MPI_ERR_TRUNCATE`, and the job aborted before any
//! rank printed.
//!
//! Measured on `origin/main` @ c11ff82d (debug, water+benzene DF-B3LYP):
//! 9/10 runs aborted at np=2, 8/10 at np=3, 8/10 at np=4. A ONE-element
//! system (H2) took the `unique_zs.len() <= 1` branch, never reached the
//! par_iter, and was 0/10 — that contrast is what isolated the site.
//!
//! ## What this test asserts
//!
//! A multi-element molecule (water: O and H) is run through the guess on every
//! rank. Two things are checked:
//!
//!   1. The guess COMPLETES — under the bug this aborts the whole job rather
//!      than failing an assertion, so merely reaching the end is signal.
//!   2. Every rank produced a BIT-IDENTICAL guess density. This is the part
//!      that makes the test meaningful rather than vacuous: a reduce that
//!      silently matched the wrong pair of buffers would corrupt the density
//!      even on the runs that happen not to trip MPI's own size check.
//!
//! Without `--features mpi` this is an ordinary single-process test (size==1),
//! so `cargo test -p ferric-scf` still exercises the serial path.
//!
//! Build and launch:
//!   OPENBLAS_NUM_THREADS=1 cargo test -p ferric-scf --features mpi \
//!       --test mpi_guess_collective_ordering --no-run
//!   mpirun -np 4 <binary> --nocapture --test-threads=1

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_scf::guess::minao_projection_guess;

/// Water: two unique elements (O, H), so the guess's per-element build is the
/// multi-element path. Repeated, because the underlying defect was a race —
/// a single pass could pass by luck even with the bug present.
const WATER_XYZ: &str = "3\n\nO 0.0 0.0 0.0\nH 0.0 0.757 0.587\nH 0.0 -0.757 0.587\n";

/// Number of repeats per run. The pre-fix failure rate was ~85% for a single
/// benzene-scale pass; at 8 repeats of a 2-element guess the chance of a
/// bugged build slipping through untouched is negligible.
const REPEATS: usize = 8;

#[test]
fn minao_guess_is_collective_order_safe_across_ranks() {
    let ctx = ParallelContext::default();
    let mol = Molecule::parse_xyz(WATER_XYZ, 0, 1).unwrap();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();

    let mut last: Option<ndarray::Array2<f64>> = None;
    for i in 0..REPEATS {
        // Under the pre-fix code this call ABORTS the job (MPI_ERR_TRUNCATE)
        // rather than returning an Err, so reaching the next line is itself
        // the primary signal.
        let d = minao_projection_guess(&mol, &prep, prep.basis_set())
            .unwrap_or_else(|e| panic!("rank {}: MINAO guess failed on pass {i}: {e:?}", ctx.rank));

        // The guess is a deterministic function of (mol, basis), so repeated
        // builds on one rank must agree bit-for-bit. A build order that
        // corrupted a reduce would show up here.
        if let Some(prev) = &last {
            assert_eq!(
                prev, &d,
                "rank {}: MINAO guess differs between pass {} and the previous pass — \
                 the per-element build is not deterministic",
                ctx.rank, i
            );
        }
        last = Some(d);
    }

    let d = last.expect("REPEATS > 0");
    eprintln!(
        "[minao-order] rank {}/{}: guess built {REPEATS}x, trace {:.12}",
        ctx.rank,
        ctx.size,
        d.diag().sum()
    );

    // Cross-rank bit-identity. Every rank runs the identical deterministic
    // build, so the guess density must match EXACTLY across ranks. Reduced as
    // a max-abs difference against rank 0's copy, broadcast first.
    #[cfg(feature = "mpi")]
    {
        use mpi::collective::SystemOperation;
        use mpi::traits::{Communicator, CommunicatorCollectives, Root};
        if let Some(world) = ctx.world() {
            let mut reference = d.as_slice().unwrap().to_vec();
            world.process_at_rank(0).broadcast_into(&mut reference[..]);

            let local_max_diff = d
                .as_slice()
                .unwrap()
                .iter()
                .zip(reference.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);

            let mut global_max_diff = 0.0f64;
            world.all_reduce_into(
                std::slice::from_ref(&local_max_diff),
                std::slice::from_mut(&mut global_max_diff),
                SystemOperation::max(),
            );

            if ctx.is_root() {
                eprintln!(
                    "[minao-order] cross-rank max |ΔD| over {} ranks: {global_max_diff:.3e}",
                    ctx.size
                );
            }
            assert_eq!(
                global_max_diff, 0.0,
                "ranks disagree on the MINAO guess density by {global_max_diff:.3e} \
                 (expected exactly 0.0): the per-element guess build is not \
                 rank-consistent",
            );
        }
    }
}
