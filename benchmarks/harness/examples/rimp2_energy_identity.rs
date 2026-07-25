//! Verify the restructured MP2 energy accumulation (parallel over i, j>=i tail
//! GEMM) reproduces the ORIGINAL implementation (serial full-width g_all
//! precompute + parallel per-pair loop) to full f64 precision.
//!
//! The original is reimplemented verbatim here as the reference; the new one is
//! called through the library. Both are run on the same synthetic b_ov, so this
//! isolates the accumulation change from any integral-transform difference.
use ndarray::Array2;
use rayon::prelude::*;

use ferric_mp2::rimp2::spin_components_from_b_ov;

extern crate openblas_src as _;

/// The pre-change implementation, verbatim.
fn reference(
    b_ov: &Array2<f64>,
    eps: &[f64],
    nocc: usize,
    nvir: usize,
    first_occ: usize,
    nocc_total: usize,
) -> (f64, f64) {
    let g_all: Vec<Array2<f64>> = (0..nocc)
        .map(|i| {
            let b_i = b_ov.slice(ndarray::s![.., i * nvir..(i + 1) * nvir]);
            b_i.t().dot(b_ov)
        })
        .collect();
    let pairs: Vec<(usize, usize)> =
        (0..nocc).flat_map(|i| (i..nocc).map(move |j| (i, j))).collect();
    let partials: Vec<(f64, f64)> = pairs
        .par_iter()
        .map(|&(i, j)| {
            let fac = if i == j { 1.0 } else { 2.0 };
            let g_i = &g_all[i];
            let e_ij = eps[first_occ + i] + eps[first_occ + j];
            let (mut e_os_ij, mut e_ss_ij) = (0.0, 0.0);
            for a in 0..nvir {
                for b in 0..nvir {
                    let g_ab = g_i[(a, j * nvir + b)];
                    let g_ba = g_i[(b, j * nvir + a)];
                    let denom = e_ij - eps[nocc_total + a] - eps[nocc_total + b];
                    e_os_ij += g_ab * g_ab / denom;
                    e_ss_ij += g_ab * (g_ab - g_ba) / denom;
                }
            }
            (fac * e_os_ij, fac * e_ss_ij)
        })
        .collect();
    let (mut e_os, mut e_ss) = (0.0, 0.0);
    for (a, b) in partials {
        e_os += a;
        e_ss += b;
    }
    (e_os, e_ss)
}

fn main() {
    // Several shapes, including nvir=1 (the documented F-order dot trap) and a
    // frozen-core offset, so the comparison is not a single lucky geometry.
    let cases: [(usize, usize, usize, usize); 5] = [
        // (naux, nocc, nvir, first_occ)
        (60, 5, 12, 0),
        (120, 8, 40, 2),
        (200, 12, 60, 6),
        (40, 3, 1, 0),
        (912, 15, 393, 6),
    ];

    let mut worst = 0.0f64;
    for (naux, nocc, nvir, first_occ) in cases {
        let width = nocc * nvir;
        let nocc_total = first_occ + nocc;
        let mut b_ov = Array2::<f64>::zeros((naux, width));
        let mut s: u64 = 0x243f_6a88_85a3_08d3;
        for v in b_ov.iter_mut() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *v = ((s >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 0.05;
        }
        let eps: Vec<f64> = (0..nocc_total + nvir)
            .map(|i| {
                if i < nocc_total {
                    -0.9 - 0.07 * i as f64
                } else {
                    0.15 + 0.011 * i as f64
                }
            })
            .collect();

        let (ref_os, ref_ss) = reference(&b_ov, &eps, nocc, nvir, first_occ, nocc_total);
        let got = spin_components_from_b_ov(&b_ov, &eps, nocc, nvir, first_occ, nocc_total);

        let d_os = (ref_os - got.e_os).abs();
        let d_ss = (ref_ss - got.e_ss).abs();
        let d_tot = (ref_os + ref_ss - got.e_total).abs();
        let rel = if ref_os.abs() > 0.0 { d_tot / (ref_os + ref_ss).abs() } else { d_tot };
        worst = worst.max(rel);
        println!(
            "naux={naux:4} nocc={nocc:3} nvir={nvir:4} fc={first_occ}:  \
             ref={:.17e}  new={:.17e}  |d_os|={d_os:.2e} |d_ss|={d_ss:.2e} rel={rel:.2e}{}",
            ref_os + ref_ss,
            got.e_total,
            if d_tot == 0.0 { "  [BIT-IDENTICAL]" } else { "" }
        );
    }
    println!("\nworst relative deviation: {worst:.3e}");
}
