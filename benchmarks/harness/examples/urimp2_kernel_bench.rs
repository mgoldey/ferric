//! Time the open-shell same-spin MP2 pair kernel at production shape.
//!
//! `u_rimp2::same_spin_pair_kernel`'s energy-only branch now uses the i<=j pair
//! symmetry and a j>=i tail GEMM (both halve the work). This measures the
//! energy-only path in isolation; there is no CLI route to open-shell RI-MP2,
//! so a config-driven benchmark cannot reach it.
//!
//! The kernel is `pub(crate)`, so this reimplements BOTH formulations locally
//! and times them head to head — that also makes it an independent check that
//! they agree.
use std::time::Instant;

use ndarray::Array2;
use rayon::prelude::*;

extern crate openblas_src as _;

/// Full-range, full-width: the pre-2026-07-25 formulation.
fn old(b: &Array2<f64>, eps: &[f64], nocc: usize, nvir: usize, fo: usize, nt: usize) -> f64 {
    let partials: Vec<f64> = (0..nocc)
        .into_par_iter()
        .map(|i| {
            let b_i = b.slice(ndarray::s![.., i * nvir..(i + 1) * nvir]);
            let g_i = b_i.t().dot(b);
            let eps_i = eps[fo + i];
            let mut energy_i = 0.0;
            for j in 0..nocc {
                let eps_j = eps[fo + j];
                for a in 0..nvir {
                    let eps_a = eps[nt + a];
                    for b_idx in 0..nvir {
                        let k = g_i[(a, j * nvir + b_idx)] - g_i[(b_idx, j * nvir + a)];
                        energy_i += k * k / (eps_i + eps_j - eps_a - eps[nt + b_idx]);
                    }
                }
            }
            0.25 * energy_i
        })
        .collect();
    partials.into_iter().sum()
}

/// i<=j symmetry + j>=i tail GEMM: the current formulation.
fn new(b: &Array2<f64>, eps: &[f64], nocc: usize, nvir: usize, fo: usize, nt: usize) -> f64 {
    let partials: Vec<f64> = (0..nocc)
        .into_par_iter()
        .map(|i| {
            let b_i = b.slice(ndarray::s![.., i * nvir..(i + 1) * nvir]);
            let b_tail = b.slice(ndarray::s![.., i * nvir..]);
            let g_i = b_i.t().dot(&b_tail);
            let eps_i = eps[fo + i];
            let mut energy_i = 0.0;
            for j in i..nocc {
                let fac = if i == j { 1.0 } else { 2.0 };
                let jcol = (j - i) * nvir;
                let eps_j = eps[fo + j];
                let mut e_ij = 0.0;
                for a in 0..nvir {
                    let eps_a = eps[nt + a];
                    for b_idx in 0..nvir {
                        let k = g_i[(a, jcol + b_idx)] - g_i[(b_idx, jcol + a)];
                        e_ij += k * k / (eps_i + eps_j - eps_a - eps[nt + b_idx]);
                    }
                }
                energy_i += fac * e_ij;
            }
            0.25 * energy_i
        })
        .collect();
    partials.into_iter().sum()
}

fn main() {
    // Open-shell alpha channel at roughly benzene-cation/aTZ scale.
    let naux: usize = std::env::var("NAUX").ok().and_then(|s| s.parse().ok()).unwrap_or(912);
    let nocc: usize = std::env::var("NOCC").ok().and_then(|s| s.parse().ok()).unwrap_or(21);
    let nvir: usize = std::env::var("NVIR").ok().and_then(|s| s.parse().ok()).unwrap_or(393);
    let width = nocc * nvir;
    let (fo, nt) = (0usize, nocc);

    let mut b = Array2::<f64>::zeros((naux, width));
    let mut s: u64 = 0x243f_6a88_85a3_08d3;
    for v in b.iter_mut() {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *v = ((s >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 0.01;
    }
    let eps: Vec<f64> = (0..nocc + nvir)
        .map(|i| if i < nocc { -0.9 - 0.05 * i as f64 } else { 0.15 + 0.01 * i as f64 })
        .collect();

    println!("naux={naux} nocc={nocc} nvir={nvir}");
    // Interleave A/B repeats: sequential runs drift with box state.
    let (mut t_old, mut t_new) = (f64::MAX, f64::MAX);
    let (mut e_old, mut e_new) = (0.0, 0.0);
    for _ in 0..3 {
        let t = Instant::now();
        e_old = old(&b, &eps, nocc, nvir, fo, nt);
        t_old = t_old.min(t.elapsed().as_secs_f64());
        let t = Instant::now();
        e_new = new(&b, &eps, nocc, nvir, fo, nt);
        t_new = t_new.min(t.elapsed().as_secs_f64());
    }
    let rel = (e_old - e_new).abs() / e_old.abs().max(1e-30);
    println!("old (full range, full width): {t_old:.3} s   E={e_old:.15e}");
    println!("new (i<=j, tail GEMM):        {t_new:.3} s   E={e_new:.15e}");
    println!("speedup {:.2}x   agreement rel={rel:.3e}", t_old / t_new);
}
