//! The shared-primitive kernel (`[scf] gc_kernel`) under real MPI.
//!
//! Self-checking at any rank count; meant to be launched under `mpirun` (the
//! CI `mpi` job runs it at 1, 2 and 3 ranks):
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo test -p ferric-scf --features mpi \
//!     --test mpi_gc_kernel --no-run
//! mpirun -np 2 <path-to-test-binary> mpi_gc_kernel --nocapture --test-threads=1
//! ```
//!
//! * `mpi_gc_kernel_matches_serial`: ethylene cc-pVDZ, rotated walk, kernel
//!   on. Every rank also builds the one-rank J/K on its own (a context with
//!   no MPI world) and requires the distributed J/K (striped by whole
//!   buckets, reduced across ranks) within 1e-12 of it, and the per-rank
//!   kernel counts, summed across ranks, equal to the one-rank counts: the
//!   same quartets served and exactly the same group evaluations (no bucket
//!   is split across ranks).
//! * `mpi_gc_kernel_disagreement_errors_everywhere`: with two or more ranks,
//!   rank 0 records `auto` and the others `off`. Every rank must return an
//!   error (the setting is agreed before the build's first reduction), and
//!   the ranks then meet in one more collective, which a hung rank would
//!   never reach. At one rank it checks that the build succeeds.
//!
//! Without `--features mpi`, or at one rank, both run as ordinary
//! single-process tests. These were not run under `mpirun` where they were
//! written (no MPI on that host); the assertions are the check.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::direct_jk::gc_kernel::{kernel_probe, KernelCounts};
use ferric_scf::direct_jk::{DirectJK, GcKernel, GcRotate};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;

fn setup() -> (PreparedBasis, Array2<f64>) {
    // Inline (testdata/molecules/c2h4.xyz): the CI mpirun job runs the
    // test binary from the workspace root, not from this crate's directory,
    // so a crate-relative path does not resolve there.
    const C2H4: &str = "6\nethylene D2h (rCC=1.339, rCH=1.086, HCH=117.4)\n\
        C 0.000000 0.000000 0.669500\n\
        C 0.000000 0.000000 -0.669500\n\
        H 0.000000 0.922832 1.237695\n\
        H 0.000000 -0.922832 1.237695\n\
        H 0.000000 0.922832 -1.237695\n\
        H 0.000000 -0.922832 -1.237695\n";
    let mol = Molecule::parse_xyz(C2H4, 0, 1).unwrap();
    let prep = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let n = prep.nbasis();
    let d = Array2::from_shape_fn((n, n), |(i, j)| {
        let v = ((i * 7 + j * 3) % 11) as f64;
        0.02 * v * (-0.01 * (i as f64 - j as f64).abs()).exp() - 0.03
    });
    let d = 0.5 * (&d + &d.t());
    (prep, d)
}

fn bounds(prep: &PreparedBasis, k: GcKernel) -> SchwarzBounds {
    let b = SchwarzBounds::compute(Operator::coulomb(), prep).unwrap();
    b.gc_kernel.set(Some(k));
    b
}

type Built = (Array2<f64>, Array2<f64>, Vec<KernelCounts>);

fn build(
    ctx: &ParallelContext,
    prep: &PreparedBasis,
    b: &SchwarzBounds,
    d: &Array2<f64>,
) -> Result<Built, ferric_core::FerricError> {
    let n = prep.nbasis();
    let mut djk = DirectJK::new(ctx, prep, b, 1e-12, usize::MAX).with_gc_rotation(GcRotate::Auto);
    let (mut j, mut k) = (Array2::zeros((n, n)), Array2::zeros((n, n)));
    kernel_probe::start();
    let r = djk.build(d, &mut j, &mut k);
    let c = kernel_probe::take();
    r.map(|_| (j, k, c))
}

/// Sum of `v` over the ranks of `ctx` (`v` itself without a world).
fn sum_ranks(ctx: &ParallelContext, v: u64) -> u64 {
    #[cfg(feature = "mpi")]
    if let Some(world) = ctx.world() {
        use mpi::collective::SystemOperation;
        use mpi::traits::CommunicatorCollectives;
        let mut out = 0u64;
        world.all_reduce_into(
            std::slice::from_ref(&v),
            std::slice::from_mut(&mut out),
            SystemOperation::sum(),
        );
        return out;
    }
    let _ = ctx;
    v
}

fn max_abs(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    a.iter()
        .zip(b)
        .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
}

#[test]
fn mpi_gc_kernel_matches_serial() {
    let ctx = ParallelContext::default();
    let (prep, d) = setup();
    let on = bounds(&prep, GcKernel::Auto);
    let (j, k, c) = build(&ctx, &prep, &on, &d).unwrap();
    let serial_bounds = bounds(&prep, GcKernel::Auto);
    let (j1, k1, c1) = build(&ParallelContext::for_rank(0, 1), &prep, &serial_bounds, &d).unwrap();
    assert_eq!((c.len(), c1.len()), (1, 1));
    let (ej, ek) = (max_abs(&j, &j1), max_abs(&k, &k1));
    let quartets = sum_ranks(&ctx, c[0].quartets as u64);
    let evals = sum_ranks(&ctx, c[0].evals as u64);
    eprintln!(
        "rank {}/{}: max|dJ| {ej:.2e} max|dK| {ek:.2e}; kernel quartets {quartets} (serial \
         {}), group evaluations {evals} (serial {})",
        ctx.rank, ctx.size, c1[0].quartets, c1[0].evals
    );
    assert!(c1[0].quartets > 0, "the kernel never ran");
    assert!(ej <= 1e-12 && ek <= 1e-12, "{ej:e} {ek:e}");
    assert_eq!(
        quartets, c1[0].quartets as u64,
        "kernel quartets across ranks"
    );
    assert_eq!(evals, c1[0].evals as u64, "a bucket was split across ranks");
}

#[test]
fn mpi_gc_kernel_disagreement_errors_everywhere() {
    let ctx = ParallelContext::default();
    let (prep, d) = setup();
    let mine = if ctx.rank == 0 {
        GcKernel::Auto
    } else {
        GcKernel::Off
    };
    let r = build(&ctx, &prep, &bounds(&prep, mine), &d);
    if ctx.size == 1 {
        assert!(r.is_ok(), "one rank cannot disagree: {:?}", r.err());
        return;
    }
    let msg = match &r {
        Ok(_) => panic!("rank {}: the build ran with ranks disagreeing", ctx.rank),
        Err(e) => e.to_string(),
    };
    eprintln!("rank {}/{}: refused: {msg}", ctx.rank, ctx.size);
    assert!(msg.contains("gc kernel"), "{msg}");
    // Every rank got here (none hangs in a reduction): one more collective.
    assert_eq!(sum_ranks(&ctx, 1), ctx.size as u64);
}
