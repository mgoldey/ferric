//! Split the RI-MP2 energy accumulation into its two halves — the serial
//! `g_all` wide-GEMM precompute vs the rayon-parallel per-pair denominator
//! loop — to see which one owns the stage's wall time.
//!
//! Reproduces `spin_components_from_b_ov` inline (it has no public seam between
//! the halves) on a synthetic b_ov of production shape, so no SCF is needed.
use std::time::Instant;

use ndarray::Array2;

// Pull in the BLAS backend: `.dot()` on f64 arrays resolves to dgemm, which
// needs openblas linked even though no ferric crate is referenced here.
extern crate openblas_src as _;

fn main() {
    // benzene / aug-cc-pVTZ shape, frozen_core = 6
    let naux: usize = std::env::var("NAUX").ok().and_then(|s| s.parse().ok()).unwrap_or(912);
    let nocc: usize = std::env::var("NOCC").ok().and_then(|s| s.parse().ok()).unwrap_or(15);
    let nvir: usize = std::env::var("NVIR").ok().and_then(|s| s.parse().ok()).unwrap_or(393);
    let width = nocc * nvir;

    // Deterministic pseudo-random b_ov; magnitudes are irrelevant to timing.
    let mut b_ov = Array2::<f64>::zeros((naux, width));
    let mut s: u64 = 0x243f_6a88_85a3_08d3;
    for v in b_ov.iter_mut() {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *v = ((s >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 0.01;
    }
    let eps: Vec<f64> = (0..nocc + nvir)
        .map(|i| if i < nocc { -0.5 - 0.1 * i as f64 } else { 0.2 + 0.01 * i as f64 })
        .collect();
    let (first_occ, nocc_total) = (0usize, nocc);

    println!("naux={naux} nocc={nocc} nvir={nvir} width={width}");
    println!(
        "g_all memory = {:.2} GB",
        (nocc * nvir * width * 8) as f64 / 1e9
    );

    use rayon::prelude::*;

    // --- half 1: the SERIAL wide-GEMM precompute ---
    let t0 = Instant::now();
    let g_all: Vec<Array2<f64>> = (0..nocc)
        .map(|i| {
            let b_i = b_ov.slice(ndarray::s![.., i * nvir..(i + 1) * nvir]);
            b_i.t().dot(&b_ov)
        })
        .collect();
    let t_gall = t0.elapsed().as_secs_f64();

    // --- half 2: the PARALLEL per-pair loop ---
    let pairs: Vec<(usize, usize)> = (0..nocc).flat_map(|i| (i..nocc).map(move |j| (i, j))).collect();
    let t0 = Instant::now();
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
    let t_pairs = t0.elapsed().as_secs_f64();
    let e: f64 = partials.iter().map(|p| p.0 + p.1).sum();

    let total = t_gall + t_pairs;
    println!("\n(checksum {e:.10})");
    println!("{:<40} {:>9} {:>7}", "half", "sec", "%");
    println!("{:<40} {:>9.3} {:>6.1}%", "g_all wide-GEMM precompute (SERIAL)", t_gall, 100.0 * t_gall / total);
    println!("{:<40} {:>9.3} {:>6.1}%", "per-pair denominator loop (rayon)", t_pairs, 100.0 * t_pairs / total);
    println!("{:<40} {:>9.3}", "TOTAL", total);

    // FLOP accounting
    let gf_gall = 2.0 * nocc as f64 * naux as f64 * nvir as f64 * width as f64 / 1e9;
    let gf_pairs = 4.0 * (nocc * (nocc + 1) / 2) as f64 * (nvir * nvir) as f64 / 1e9;
    println!("\ng_all  {:.1} GFLOP -> {:.1} GFLOP/s", gf_gall, gf_gall / t_gall);
    println!("pairs  {:.1} GFLOP -> {:.1} GFLOP/s", gf_pairs, gf_pairs / t_pairs);
}
