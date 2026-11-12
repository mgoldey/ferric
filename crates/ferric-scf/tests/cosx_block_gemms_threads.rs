//! The sparse COSX builder's two per-block GEMMs (`F = D[Λ,A] X[A,:]` and
//! `Ktilde[A,B] += X[A,:] G[B,:]^T`) run on the rayon pool through
//! `ferric_core::linalg::par_gemm`, at one BLAS thread per block, and K is
//! bit-identical across pool sizes at the default serial-BLAS setting.
//!
//! One test in its own binary: it builds the global rayon pool, changes the
//! serial setting and reads the process-global `par_gemm` probe.

use ferric_core::basis::bundled;
use ferric_core::blas_threads::set_serial_blas_threads;
use ferric_core::linalg::par_gemm_probe;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
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

/// On the global pool of 4 at the default serial setting (serial sections at
/// 4 BLAS threads), the block GEMMs go through `par_gemm` at one BLAS thread,
/// and K equals, bit for bit, K built on pools of 1, 2 and 6.
///
/// FAILS if a block GEMM goes back to a plain `.dot` on the main thread (no
/// `par_gemm` blocks recorded), or to a threaded BLAS call under the serial
/// count (K then depends on the pool size at the last bit).
#[test]
fn cosx_block_gemms_use_the_pool_and_are_thread_count_independent() {
    std::env::remove_var("FERRIC_BLAS_THREADS");
    rayon::ThreadPoolBuilder::new()
        .num_threads(POOL)
        .build_global()
        .expect("this binary's first rayon use");

    let mol = Molecule::parse_xyz(WATER, 0, 1).unwrap();
    let prep = PreparedBasis::new(&mol, &bundled("cc-pvdz").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();
    set_serial_blas_threads(Some(1));
    let rhf = solve_rhf(&ctx, &mol, &prep, op, &bounds, &RhfConfig::default()).unwrap();
    set_serial_blas_threads(None);
    let d = rhf.density_total.clone();
    let nbf = prep.nbasis();

    let build = || {
        let mut kb = CosxK::new(&ctx, &mol, &prep, CosxConfig::default(), usize::MAX).unwrap();
        let mut k = Array2::<f64>::zeros((nbf, nbf));
        kb.build(&d, &mut k).unwrap();
        k
    };

    par_gemm_probe::reset();
    let k_global = build();
    let blocks = par_gemm_probe::BLOCKS.load(Ordering::Relaxed);
    assert!(
        blocks > 0,
        "the COSX block GEMMs did not go through par_gemm"
    );
    let seen = par_gemm_probe::MAX_BLAS_THREADS.load(Ordering::Relaxed);
    assert_eq!(seen, 1, "COSX par_gemm blocks ran at {seen} BLAS threads");

    for threads in [1, 2, 6] {
        let k = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(build);
        assert_eq!(
            k, k_global,
            "K on a pool of {threads} differs from the pool of {POOL}"
        );
    }
}
