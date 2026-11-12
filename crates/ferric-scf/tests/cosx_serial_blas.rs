//! The sparse COSX builder's two per-block GEMMs (`F = D[Λ,A] X[A,:]` and
//! `Ktilde[A,B] += X[A,:] G[B,:]^T`) run between the parallel A-build regions
//! at the serial-section BLAS count: by default the rayon pool size, `1` when
//! set to 1, and 1 inside a rayon pool. At `blas_threads = 1`, K is
//! bit-identical across pool sizes.
//!
//! One test in its own binary: it changes process-global state (the rayon
//! global pool, the serial setting, the OpenBLAS count).

use ferric_core::basis::bundled;
use ferric_core::blas_threads::{current_blas_threads, set_serial_blas_threads};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::blas_probe::COSX_BLOCK_GEMM;
use ferric_scf::cosx_k::{CosxConfig, CosxK};
use ferric_scf::fock::KBuilder;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;
use std::sync::atomic::Ordering;

const POOL: usize = 4;

const WATER: &str = "3
water
O  0.0000  0.0000  0.1173
H  0.0000  0.7572 -0.4692
H  0.0000 -0.7572 -0.4692
";

/// FAILS if the per-block GEMMs lose their `with_blas_threads(opt_in..)`
/// scope (the probe then reads the baseline 1 at the default setting), or if
/// a thread-count dependence enters the K build at `blas_threads = 1`.
///
/// The threaded-vs-single bar is 1e-12 on max|ΔK|: a threaded dgemm changes
/// only the summation order of ~1e3-term sums of O(1) products, so the
/// difference is a few ulps of |K| ~ O(1) (~1e-15), and 1e-12 leaves three
/// orders of headroom while still catching a wrong product.
#[test]
fn cosx_block_gemms_follow_the_serial_blas_setting() {
    std::env::remove_var("FERRIC_BLAS_THREADS");
    rayon::ThreadPoolBuilder::new()
        .num_threads(POOL)
        .build_global()
        .expect("this binary's first rayon use");
    let baseline = current_blas_threads();

    let mol = Molecule::parse_xyz(WATER, 0, 1).unwrap();
    let prep = PreparedBasis::new(&mol, &bundled("cc-pvdz").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();
    set_serial_blas_threads(Some(1));
    let rhf = solve_rhf(&ctx, &mol, &prep, op, &bounds, &RhfConfig::default()).unwrap();
    let d = rhf.density_total.clone();
    let nbf = prep.nbasis();

    let build = || {
        let mut kb = CosxK::new(&ctx, &mol, &prep, CosxConfig::default(), usize::MAX).unwrap();
        let mut k = Array2::<f64>::zeros((nbf, nbf));
        kb.build(&d, &mut k).unwrap();
        k
    };

    set_serial_blas_threads(None);
    let k_threaded = build();
    assert_eq!(
        COSX_BLOCK_GEMM.load(Ordering::Relaxed),
        POOL,
        "default: the COSX block GEMMs must run at the rayon pool size"
    );
    assert_eq!(current_blas_threads(), baseline, "baseline not restored");

    set_serial_blas_threads(Some(1));
    let k_single = build();
    assert_eq!(
        COSX_BLOCK_GEMM.load(Ordering::Relaxed),
        1,
        "blas_threads = 1 must be honoured"
    );
    let dk = (&k_threaded - &k_single)
        .iter()
        .fold(0.0f64, |m, v| m.max(v.abs()));
    assert!(dk <= 1e-12, "threaded vs single BLAS: max|dK| = {dk:e}");

    // blas_threads = 1: the same bits on any pool size.
    let on_pool = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(build)
    };
    let k1 = on_pool(1);
    assert_eq!(
        COSX_BLOCK_GEMM.load(Ordering::Relaxed),
        1,
        "inside a pool the block GEMMs run at 1"
    );
    assert_eq!(
        k1, k_single,
        "pool of 1 vs the global pool at blas_threads = 1"
    );
    for threads in [2, 6] {
        assert_eq!(on_pool(threads), k1, "K differs on a pool of {threads}");
    }
    set_serial_blas_threads(None);
}
