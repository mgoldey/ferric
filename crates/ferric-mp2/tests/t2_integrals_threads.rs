//! `compute_t2_and_integrals` builds `(ia|jb) = bᵀ·b` on the rayon pool
//! through `ferric_core::linalg::par_gemm`, with the bits of the serial
//! product on any pool size.
//!
//! Own binary: it reads the process-global `par_gemm` probe.

use ferric_core::linalg::{par_gemm_probe, PAR_GEMM_COL_BLOCK};
use ferric_mp2::oo_rimp2::compute_t2_and_integrals;
use ndarray::Array2;
use std::sync::atomic::Ordering;

/// nocc 3, nvir 90, naux 40: `nov = 270` spans three 128-column blocks, the
/// last one partial.
///
/// Asserts, on pools of 1, 2 and 6 threads:
/// - the product ran as `ceil(nov / 128)` `par_gemm` blocks;
/// - `eri_ov` has the bits of the serial `b.t().dot(&b)` in C order, which is
///   what the function computed before it used the pool;
/// - `t2[ia,jb]` is `eri_ov[ia,jb] / (e_i + e_j - e_a - e_b)` to the bit.
///
/// FAILS if the product leaves `par_gemm`, if the blocked product stops
/// reproducing the serial bits, or if the output is not C-ordered.
#[test]
fn t2_integrals_use_the_pool_and_keep_the_serial_bits() {
    let (nocc, nvir, naux) = (3usize, 90usize, 40usize);
    let nov = nocc * nvir;
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let b = Array2::from_shape_fn((naux, nov), |_| {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
    });
    let eps: Vec<f64> = (0..nocc)
        .map(|i| -1.0 + 0.2 * i as f64)
        .chain((0..nvir).map(|a| 0.3 + 0.05 * a as f64))
        .collect();

    let serial = b.t().dot(&b);
    let serial = serial.as_standard_layout();

    for threads in [1usize, 2, 6] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        par_gemm_probe::reset();
        let (t2, eri_ov) =
            pool.install(|| compute_t2_and_integrals(&b, &eps, nocc, nvir, nocc, 0, naux));
        assert_eq!(
            par_gemm_probe::BLOCKS.load(Ordering::Relaxed),
            nov.div_ceil(PAR_GEMM_COL_BLOCK),
            "{threads} threads: the (ia|jb) product did not run as par_gemm column blocks"
        );
        for ia in 0..nov {
            for jb in 0..nov {
                let want = serial[(ia, jb)];
                let got = eri_ov[ia * nov + jb];
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "{threads} threads: eri_ov[{ia},{jb}] {got:e} vs serial {want:e}"
                );
                let denom =
                    eps[ia / nvir] + eps[jb / nvir] - eps[nocc + ia % nvir] - eps[nocc + jb % nvir];
                assert_eq!(
                    t2[ia * nov + jb].to_bits(),
                    (want / denom).to_bits(),
                    "{threads} threads: t2[{ia},{jb}]"
                );
            }
        }
    }
}
