//! The error of a mixed-precision RI-MP2 energy with NO device (runs in CI).
//!
//! Two host measurements, both against the f64 CPU energy of the same B_ov:
//!
//! 1. f32 STORAGE alone: B_ov rounded through f32 (`round_trip_f32`), energy
//!    recomputed with the f64 kernel. This isolates "B_ov stored as f32" from
//!    "f32 panel accumulation"; on the device the latter is measured as the
//!    difference between the device and this (gpu_rimp2_mixed.rs). Per-element
//!    factor eps_G = 2·u32 (u32 = 2⁻²⁴, round to nearest), bound from
//!    `common/rimp2_error_bound.rs`.
//! 2. The host twin of the mixed GEMM (`gemm_f32_f64acc_host`: f32 operands,
//!    k-panels of width b in f32, panels summed in f64) forming every G_i:
//!    eps_G = 2·u32 + γ_b(u32) + γ_⌈naux/b⌉(u64) plus the f64 reference's own
//!    γ_naux(u64), for several panel widths b.
//!
//! Positive control: f32 rounding must move the energy far above the f64 noise
//! class (two valid f64 evaluation orders differ by ~1e-16 relative), so the
//! test asserts |ΔE| ≥ 1e3·u64·|E_corr| (1.1e-13 relative, three decades above
//! that class and several below the f32-storage error): a path that silently
//! skipped the rounding would sit inside the noise class and fail it.
//!
//! Error law (cc-pVDZ, all electrons correlated; N = nocc): the measured error
//! is fit on a log-log line against N and compared with the bound's own fit.
//! Model: |ΔE_storage| = |Σ_k (∂E/∂B_k) δB_k|, δB_k independent with
//! |δB_k| ≤ u32|B_k|. The deterministic bound grows like |E|·κ_E (slope of its
//! own fit); the measured error cannot grow faster than the bound it satisfies
//! at every N, so the asserted law is slope(|ΔE|) ≤ slope(bound) + 2·SE, SE the
//! standard error of the fitted error slope (a fit on three points has one
//! degree of freedom; no tighter statement is supportable). Measured (water,
//! C4, C8, and C12 with the device test): the error does NOT grow with N at
//! the bound's rate (slope 0.10 ± 0.82 over four exact-integral systems against a
//! bound slope of 1.30), i.e. |ΔE|/|bound| falls with N; the random-sign cancellation of
//! the f32 rounding terms is not resolved into a √N law by four noisy points,
//! so no lower limit is asserted.
//!
//! MEASURED (printed `error map` lines, cc-pVDZ, storage only, eps_G = 2·u32):
//!   system     nocc nvir naux  E_corr        dE_os      dE_ss      bound_os  bound_ss  kappa_E
//!   h2o          5   19    84  -0.204033457  3.14e-10   2.82e-10   4.74e-8   8.46e-8   1.30
//!   c4h10       17   89   364  -0.599645228  8.33e-11   2.36e-11   2.17e-7   4.13e-7   1.95
//!   c8h18       33  169   700  -1.183893422  3.50e-10   4.69e-11   5.22e-7   9.98e-7   2.40
//! (the C12 chain is measured by the device test; this CI test stops at C8 for time.)
//! Host twin (b = 8 / 128 / one panel): |ΔE_os| 1.4e-9 / 3.5e-9 / 3.5e-9 on
//! water (bound 2.4e-7 / 2.1e-6 / 2.1e-6).
use ferric_core::gpu::mixed_host::{
    gamma, gemm_f32_f64acc_host, mixed_error_factor, round_trip_f32, U32, U64,
};
use ferric_mp2::rimp2::spin_components_from_b_ov_kappa_cpu;
use ndarray::{s, Array2, ArrayView2};

#[path = "common/rimp2_error_bound.rs"]
mod bound;
#[path = "common/rimp2_gpu_fixture.rs"]
mod fixture;
use fixture::Prepared;

const SYSTEMS: [&str; 3] = ["h2o", "c4h10", "c8h18"];

/// (E_os, E_ss) with every G_i produced by `gemm(B_iᵀ, B_tail)`; the pair
/// arithmetic is the CPU path's (same formula, same fold order up to the
/// summation roundoff the bound accounts for).
fn energy_with(
    p: &Prepared,
    gemm: impl Fn(&ArrayView2<f64>, &ArrayView2<f64>) -> Array2<f64>,
) -> (f64, f64) {
    let (mut e_os, mut e_ss) = (0.0, 0.0);
    for i in 0..p.nocc {
        let lo = i * p.nvir;
        let g = gemm(
            &p.b_ov.slice(s![.., lo..lo + p.nvir]).t(),
            &p.b_ov.slice(s![.., lo..]),
        );
        for j in i..p.nocc {
            let fac = if i == j { 1.0 } else { 2.0 };
            let jcol = (j - i) * p.nvir;
            let e_ij = p.eps[p.first_occ + i] + p.eps[p.first_occ + j];
            for a in 0..p.nvir {
                for b in 0..p.nvir {
                    let (g_ab, g_ba) = (g[(a, jcol + b)], g[(b, jcol + a)]);
                    let d = e_ij - p.eps[p.nocc_total + a] - p.eps[p.nocc_total + b];
                    e_os += fac * g_ab * g_ab / d;
                    e_ss += fac * g_ab * (g_ab - g_ba) / d;
                }
            }
        }
    }
    (e_os, e_ss)
}

fn exact_gemm(l: &ArrayView2<f64>, r: &ArrayView2<f64>) -> Array2<f64> {
    l.dot(r)
}

fn mixed_gemm(b: usize) -> impl Fn(&ArrayView2<f64>, &ArrayView2<f64>) -> Array2<f64> {
    move |l, r| {
        let mut out = Array2::zeros((l.nrows(), r.ncols()));
        gemm_f32_f64acc_host(l, r, &mut out.view_mut(), b);
        out
    }
}

struct Row {
    n: f64,
    d_total: f64,
    bound_total: f64,
}

#[test]
fn f32_storage_error_is_within_its_bound_above_the_noise_floor_and_follows_the_law() {
    let mut rows = Vec::new();
    for sys in SYSTEMS {
        let (p, cpu) = fixture::prepare_scf(sys, "cc-pvdz", "cc-pvdz-ri", 0);
        let naux = p.b_ov.nrows();
        let rounded = round_trip_f32(&p.b_ov.view());
        let f32e = spin_components_from_b_ov_kappa_cpu(
            &rounded,
            &p.eps,
            p.nocc,
            p.nvir,
            p.first_occ,
            p.nocc_total,
            None,
        );
        let (d_os, d_ss) = ((f32e.e_os - cpu.e_os).abs(), (f32e.e_ss - cpu.e_ss).abs());
        let bd = bound::energy_bound(
            &p.b_ov,
            &p.eps,
            p.nocc,
            p.nvir,
            p.first_occ,
            p.nocc_total,
            2.0 * U32,
        );
        eprintln!(
            "error map storage {sys}/cc-pvdz: nocc {} nvir {} naux {naux} | E_corr {:.9} | dE_os {d_os:.3e} dE_ss {d_ss:.3e} | bound {:.3e} {:.3e} | ratio {:.3e} {:.3e} | kappa_E {:.3e} kappa_p99(i=0) {:.3e}",
            p.nocc, p.nvir, cpu.e_total, bd.os, bd.ss, d_os / bd.os, d_ss / bd.ss, bd.kappa_e_os, bd.kappa_p99_block0
        );
        assert!(
            d_os <= bd.os && d_ss <= bd.ss,
            "{sys}: storage error exceeds the deterministic bound"
        );
        let floor = 1e3 * U64 * cpu.e_total.abs();
        assert!(
            d_os + d_ss >= floor,
            "{sys}: f32 rounding moved the energy by only {:.3e} (floor {floor:.3e}); is the rounding applied?",
            d_os + d_ss
        );
        rows.push(Row {
            n: p.nocc as f64,
            d_total: (f32e.e_total - cpu.e_total).abs(),
            bound_total: bd.os + bd.ss,
        });
    }
    let (s_err, se_err, _) =
        bound::loglog_fit(&rows.iter().map(|r| (r.n, r.d_total)).collect::<Vec<_>>());
    let (s_bnd, se_bnd, _) = bound::loglog_fit(
        &rows
            .iter()
            .map(|r| (r.n, r.bound_total))
            .collect::<Vec<_>>(),
    );
    let (s_ratio, se_ratio, _) = bound::loglog_fit(
        &rows
            .iter()
            .map(|r| (r.n, r.d_total / r.bound_total))
            .collect::<Vec<_>>(),
    );
    eprintln!(
        "error law storage (N = nocc): slope(|dE|) {s_err:.3} +- {se_err:.3} | slope(bound) {s_bnd:.3} +- {se_bnd:.3} | slope(|dE|/bound) {s_ratio:.3} +- {se_ratio:.3}"
    );
    assert!(
        s_err <= s_bnd + 2.0 * se_err,
        "measured error slope {s_err:.3} exceeds the bound's slope {s_bnd:.3} + 2 SE {se_err:.3}"
    );
}

#[test]
fn the_host_twin_of_the_mixed_gemm_is_within_the_full_bound_for_every_panel_width() {
    for sys in ["h2o", "c4h10"] {
        let (p, cpu) = fixture::prepare_scf(sys, "cc-pvdz", "cc-pvdz-ri", 0);
        let naux = p.b_ov.nrows();
        // the formula used here reproduces the CPU energy up to the summation
        // roundoff of the exact GEMM (derived bound at eps_G = 2·gamma_naux(u64))
        let (x_os, x_ss) = energy_with(&p, exact_gemm);
        let eg0 = 2.0 * gamma(naux, U64);
        let b0 = bound::energy_bound(
            &p.b_ov,
            &p.eps,
            p.nocc,
            p.nvir,
            p.first_occ,
            p.nocc_total,
            eg0,
        );
        assert!(
            (x_os - cpu.e_os).abs() <= b0.os && (x_ss - cpu.e_ss).abs() <= b0.ss,
            "{sys}: reference energy loop"
        );
        for b in [8usize, 128, 1 << 30] {
            let (m_os, m_ss) = energy_with(&p, mixed_gemm(b));
            let eg = (mixed_error_factor(naux, b) + gamma(naux, U64)) * 1.01; // 1.01: first-order vs gamma_{b+2}
            let bd = bound::energy_bound(
                &p.b_ov,
                &p.eps,
                p.nocc,
                p.nvir,
                p.first_occ,
                p.nocc_total,
                eg,
            );
            let (d_os, d_ss) = ((m_os - cpu.e_os).abs(), (m_ss - cpu.e_ss).abs());
            eprintln!(
                "error map host twin {sys}/cc-pvdz b={b}: dE_os {d_os:.3e} dE_ss {d_ss:.3e} | bound {:.3e} {:.3e} | ratio {:.3e} {:.3e}",
                bd.os, bd.ss, d_os / bd.os, d_ss / bd.ss
            );
            assert!(
                d_os <= bd.os && d_ss <= bd.ss,
                "{sys} b={b}: host-twin energy error exceeds the bound"
            );
        }
    }
}
