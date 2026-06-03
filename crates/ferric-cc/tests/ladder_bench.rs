//! Micro-benchmark: CCD ladder contractions, scalar Σ-loop vs einsum!/BLAS3.
//!
//! Isolates the O(N^5) ladder kernels (the CCD per-iteration hot path) at a
//! realistic size, comparing the pre-einsum scalar implementation (vendored
//! here from git history) against the live einsum! helpers in ferric_cc.
//! Both are checked to produce identical results, then timed.
//!
//! Run with:
//!   OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
//!     cargo test -p ferric-cc --release --test ladder_bench -- --ignored --nocapture

use std::time::Instant;
use ndarray::{Array3, Array4};
use ferric_cc::helpers::{contract_pp_ladder, contract_hh_ladder};
use ferric_tensors::{Axis, Tensor};

// ---- scalar reference implementations (vendored from pre-einsum commit) ----

/// Compute the Particle-Particle ladder term: L_abij = sum_P B^P_ab * (sum_cd B^P_cd * t_cdij)
/// RI complexity: O(N^5)
pub fn contract_pp_ladder_scalar(
    b_ab: &Array3<f64>,
    t2: &Array4<f64>,
) -> Array4<f64> {
    let (naux, nvir, _) = b_ab.dim();
    let (nocc, _, _, _) = t2.dim();
    let mut res = Array4::zeros((nocc, nvir, nocc, nvir));

    // Intermediate: X^P_ij = sum_cd B^P_cd * t_cdij
    // Shape: (naux, nocc, nocc)
    let mut x = Array3::zeros((naux, nocc, nocc));
    for p in 0..naux {
        let b_p = b_ab.slice(ndarray::s![p, .., ..]);
        for i in 0..nocc {
            for j in 0..nocc {
                let t_ij = t2.slice(ndarray::s![i, .., j, ..]);
                // Tr(B^P * T_ij)
                let mut sum = 0.0;
                for c in 0..nvir {
                    for d in 0..nvir {
                        sum += b_p[(c, d)] * t_ij[(c, d)];
                    }
                }
                x[(p, i, j)] = sum;
            }
        }
    }

    // Final contraction: R_iajb = sum_P B^P_ab * X^P_ij
    for i in 0..nocc {
        for j in 0..nocc {
            for p in 0..naux {
                let x_pij = x[(p, i, j)];
                let b_p = b_ab.slice(ndarray::s![p, .., ..]);
                for a in 0..nvir {
                    for b in 0..nvir {
                        res[(i, a, j, b)] += x_pij * b_p[(a, b)];
                    }
                }
            }
        }
    }
    res
}

/// Compute the Hole-Hole ladder term: H_abij = sum_kl (kl|ij) * t_abkl
/// RI complexity: O(N^5)
pub fn contract_hh_ladder_scalar(
    b_ij: &Array3<f64>,
    t2: &Array4<f64>,
) -> Array4<f64> {
    let (naux, nocc, _) = b_ij.dim();
    let (_, nvir, _, _) = t2.dim();
    let mut res = Array4::zeros((nocc, nvir, nocc, nvir));

    // Intermediate: Y^P_ab = sum_kl B^P_kl * t_abkl
    // Shape: (naux, nvir, nvir)
    let mut y = Array3::zeros((naux, nvir, nvir));
    for p in 0..naux {
        let b_p = b_ij.slice(ndarray::s![p, .., ..]);
        for a in 0..nvir {
            for b in 0..nvir {
                let mut sum = 0.0;
                for k in 0..nocc {
                    for l in 0..nocc {
                        sum += b_p[(k, l)] * t2[(k, a, l, b)];
                    }
                }
                y[(p, a, b)] = sum;
            }
        }
    }

    // Final contraction: R_iajb = sum_P B^P_ij * Y^P_ab
    for i in 0..nocc {
        for j in 0..nocc {
            for p in 0..naux {
                let b_pij = b_ij[(p, i, j)];
                let y_p = y.slice(ndarray::s![p, .., ..]);
                for a in 0..nvir {
                    for b in 0..nvir {
                        res[(i, a, j, b)] += b_pij * y_p[(a, b)];
                    }
                }
            }
        }
    }
    res
}

// ---- benchmark ----
fn fill(seed: &mut u64, n: usize) -> Vec<f64> {
    // cheap deterministic pseudo-random fill (no Math.random in tests)
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        v.push(((*seed >> 33) as f64 / u32::MAX as f64) - 0.5);
    }
    v
}

fn bench(naux: usize, nocc: usize, nvir: usize) {
    let mut seed = 0x1234_5678u64;
    let b_ab = Array3::from_shape_vec((naux, nvir, nvir), fill(&mut seed, naux*nvir*nvir)).unwrap();
    let b_ij = Array3::from_shape_vec((naux, nocc, nocc), fill(&mut seed, naux*nocc*nocc)).unwrap();
    let t2 = Array4::from_shape_vec((nocc, nvir, nocc, nvir), fill(&mut seed, nocc*nvir*nocc*nvir)).unwrap();

    // Wrap the loop-invariant B blocks ONCE (as the hoisted CCD loop does); t2
    // wrapped once here too (in CCD it is re-wrapped per iter as it mutates).
    let b_ab_t = Tensor::new(b_ab.clone().into_dyn(), [Axis::Aux, Axis::V, Axis::V]);
    let b_ij_t = Tensor::new(b_ij.clone().into_dyn(), [Axis::Aux, Axis::O, Axis::O]);
    let t2_t = Tensor::new(t2.clone().into_dyn(), [Axis::O, Axis::V, Axis::O, Axis::V]);

    // correctness: einsum result == scalar result
    let pp_e = contract_pp_ladder(&b_ab_t, &t2_t);
    let pp_s = contract_pp_ladder_scalar(&b_ab, &t2);
    let max_pp = pp_e.iter().zip(pp_s.iter()).map(|(a,b)| (a-b).abs()).fold(0.0f64, f64::max);
    let hh_e = contract_hh_ladder(&b_ij_t, &t2_t);
    let hh_s = contract_hh_ladder_scalar(&b_ij, &t2);
    let max_hh = hh_e.iter().zip(hh_s.iter()).map(|(a,b)| (a-b).abs()).fold(0.0f64, f64::max);
    assert!(max_pp < 1e-9 && max_hh < 1e-9, "einsum != scalar: pp {max_pp:.2e} hh {max_hh:.2e}");

    let reps = 5;
    let t = Instant::now();
    for _ in 0..reps { let _ = contract_pp_ladder_scalar(&b_ab, &t2); let _ = contract_hh_ladder_scalar(&b_ij, &t2); }
    let t_scalar = t.elapsed().as_secs_f64()/reps as f64;

    let t = Instant::now();
    for _ in 0..reps { let _ = contract_pp_ladder(&b_ab_t, &t2_t); let _ = contract_hh_ladder(&b_ij_t, &t2_t); }
    let t_einsum = t.elapsed().as_secs_f64()/reps as f64;

    println!("naux={naux:4} nocc={nocc:3} nvir={nvir:3}  scalar={:9.2}ms  einsum={:9.2}ms  speedup={:.1}x  (max|Δ|={:.1e})",
        t_scalar*1e3, t_einsum*1e3, t_scalar/t_einsum, max_pp.max(max_hh));
}

#[test]
#[ignore = "benchmark: run with --release --ignored --nocapture"]
fn ladder_scalar_vs_einsum() {
    println!();
    println!("=== CCD ladder contractions: scalar Σ-loop vs einsum!/BLAS3 ===");
    bench(100, 5, 20);    // ~water/cc-pVDZ scale
    bench(300, 10, 60);   // ~10-atom / cc-pVDZ scale
    bench(600, 15, 100);  // ~20-atom scale
}
