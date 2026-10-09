//! The error of a mixed-precision RI-MP2 energy with NO device (runs in CI).
//!
//! Host measurements against the f64 CPU energy and f64 G_i blocks of the SAME
//! B_ov (RI-JK SCF reference, see `common/rimp2_gpu_fixture.rs`):
//!
//! 1. f32 STORAGE alone: B_ov rounded to nearest-even through f32
//!    (`round_trip_f32`), G_i and the energy recomputed in f64. Per-element
//!    factor `eps_storage` = 2·u32 + u32² + 2·γ_naux(u64) (u32 = 2⁻²⁴: both
//!    sides are f64 products, see `common/rimp2_eps.rs`). Asserted: every
//!    element and the energy inside their bounds, and a positive control: the
//!    per-element RMS error exceeds 2·γ_naux(u64), the largest difference two
//!    f64 evaluations of the same product can show, so a path that silently
//!    skipped the rounding fails it. The same blocks are also formed from a
//!    TRUNCATED (toward zero) B_ov; truncation must be measurably worse than
//!    round-to-nearest, and the ratio is printed.
//! 2. The host twin of the mixed GEMM (`gemm_f32_f64acc_host`: f32 operands,
//!    k-panels of width b in f32, panels summed in f64) forming every G_i, for
//!    b = 8, 64 (the shipped default) and one panel: every element inside
//!    `eps_device(naux, b)` and the energy inside its bound.
//!
//! Rounding mode: the truncation comparison above is a measurement here (2.5x to
//! 2.8x worse than round-to-nearest); the guarantee against a truncating upload
//! is the bit-exact device test in gpu_rimp2_mixed.rs, not these statistics.
//!
//! The file name predates the wording: the sweep below describes how the
//! storage RMS falls with naux (four systems, naux confounded with the
//! molecule); it is not a law and not a gate on defects.
//!
//! The deterministic bounds are worst cases over rounding signs, 3 to 6 decades
//! above the measured errors: they catch gross defects only (see
//! `common/rimp2_error_bound.rs`). The subtle defects are pinned in
//! gpu_rimp2_mixed.rs (bit-exact upload, deterministic flush construction,
//! supplementary per-element comparison).
//!
//! MEASURED (cc-pVDZ, RI-JK SCF; printed by this test; the binary runs in about
//! 3 s of test time after compilation, water and butane only):
//!   storage-only (RN) per-element RMS(err/S): h2o 1.68e-8, c4h10 7.65e-9; the
//!     truncated B_ov is 2.75x and 2.53x worse; worst element / eps_storage-bound
//!     0.90 and 0.64; storage energy errors dE_OS -1.9e-9 / -2.1e-10, dE_SS
//!     -9.3e-10 / -4.0e-11 (bounds 4.7e-8 / 2.2e-7 and 8.5e-8 / 4.1e-7).
//!   host twin per-element RMS: h2o b=8 2.7e-8, b=64 5.6e-8, one panel 6.9e-8;
//!     c4h10 b=8 1.4e-8, b=64 2.5e-8, one panel 4.0e-8.
//!   the ignored sweep (h2o, c4h10, c8h18, alkane_12; naux 84, 364, 700, 1036):
//!     storage RMS 1.68e-8, 7.65e-9, 6.16e-9, 5.74e-9; slope of ln RMS vs
//!     ln naux -0.442 +- 0.050 (df 2), one-sided 95% upper limit -0.297; the
//!     random-sign model is -0.5. It FAILS if the upper limit is >= 0.
use ferric_core::gpu::mixed_host::{gamma, gemm_f32_f64acc_host, round_trip_f32, U32, U64};
use ferric_mp2::rimp2::spin_components_from_b_ov_kappa_cpu;
use ndarray::{s, Array2, ArrayView2};

#[path = "common/rimp2_error_bound.rs"]
mod bound;
#[path = "common/rimp2_eps.rs"]
mod eps;
#[path = "common/rimp2_gpu_fixture.rs"]
mod fixture;
#[path = "common/rimp2_stats.rs"]
mod stats;
use fixture::Prepared;

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

fn mixed_gemm(b: usize) -> impl Fn(&ArrayView2<f64>, &ArrayView2<f64>) -> Array2<f64> {
    move |l, r| {
        let mut out = Array2::zeros((l.nrows(), r.ncols()));
        gemm_f32_f64acc_host(l, r, &mut out.view_mut(), b);
        out
    }
}

/// Per-element statistics of the storage-only error over all G blocks, for the
/// round-to-nearest and the truncated B_ov: (rms RN, rms trunc, worst RN).
fn storage_elements(p: &Prepared, eps_g: f64, eta: f64) -> (f64, f64, f64) {
    let babs = p.b_ov.mapv(f64::abs);
    let rn = round_trip_f32(&p.b_ov.view());
    let tr = stats::truncate_f32(&p.b_ov);
    let (mut e_rn, mut e_tr) = (stats::Elem::new(eps_g, eta), stats::Elem::new(eps_g, eta));
    for i in 0..p.nocc {
        let lo = i * p.nvir;
        let (g, sab) = stats::block_g_s(&p.b_ov, &babs, i, p.nvir);
        let blk = |m: &Array2<f64>| {
            m.slice(s![.., lo..lo + p.nvir])
                .t()
                .dot(&m.slice(s![.., lo..]))
        };
        e_rn.add(&blk(&rn).view(), &g.view(), &sab.view());
        e_tr.add(&blk(&tr).view(), &g.view(), &sab.view());
    }
    (e_rn.rms(), e_tr.rms(), e_rn.worst)
}

#[test]
fn round_to_nearest_even_is_what_as_f32_does_and_truncation_differs() {
    // 1 + 2^-24 is a tie between 1 and 1 + 2^-23: nearest-even gives 1; 1 + 3·2^-24
    // is a tie between 1 + 2^-23 and 1 + 2^-22: nearest-even gives 1 + 2^-22.
    let (tie_down, tie_up) = (1.0 + U32, 1.0 + 3.0 * U32);
    let v = Array2::from_shape_vec(
        (1, 4),
        vec![tie_down, tie_up, -tie_up, 1.0 + 0.9 * 2.0 * U32],
    )
    .unwrap();
    let rn = round_trip_f32(&v.view());
    assert_eq!(rn[(0, 0)], 1.0);
    assert_eq!(rn[(0, 1)], 1.0 + 4.0 * U32);
    assert_eq!(rn[(0, 2)], -(1.0 + 4.0 * U32));
    assert_eq!(rn[(0, 3)], 1.0 + 2.0 * U32, "0.9 ulp rounds UP to nearest");
    let tr = stats::truncate_f32(&v);
    assert_eq!(tr[(0, 0)], 1.0);
    assert_eq!(tr[(0, 1)], 1.0 + 2.0 * U32, "truncation goes toward zero");
    assert_eq!(tr[(0, 2)], -(1.0 + 2.0 * U32));
    assert_eq!(tr[(0, 3)], 1.0, "truncation drops 0.9 ulp");
}

#[test]
fn f32_storage_error_is_within_its_bounds_above_the_noise_floor_and_truncation_is_worse() {
    for sys in ["h2o", "c4h10"] {
        let (p, cpu) = fixture::prepare_scf(sys, "cc-pvdz", "cc-pvdz-ri", 0);
        let naux = p.b_ov.nrows();
        let (eg, eta) = (
            eps::eps_storage(naux),
            eps::eta_abs(naux, stats::max_abs(&p.b_ov)),
        );
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
        let (d_os, d_ss) = (f32e.e_os - cpu.e_os, f32e.e_ss - cpu.e_ss);
        let bd = bound::energy_bound(
            &p.b_ov,
            &p.eps,
            p.nocc,
            p.nvir,
            p.first_occ,
            p.nocc_total,
            bound::BoundSpec {
                eps_g: eg,
                eta,
                kappa: None,
            },
        );
        let (rms_rn, rms_tr, worst) = storage_elements(&p, eg, eta);
        eprintln!(
            "error map storage {sys}/cc-pvdz: nocc {} nvir {} naux {naux} | E_corr {:.9} | dE_os {d_os:+.3e} dE_ss {d_ss:+.3e} (total {:+.3e}) | bound {:.3e} {:.3e} | kappa_E {:.3e} kappa_p99(i=0) {:.3e} | per-element RMS(err/S): RN {rms_rn:.3e} truncation {rms_tr:.3e} (x{:.2}), worst/bound {worst:.3e}, eps_storage {eg:.3e}",
            p.nocc, p.nvir, cpu.e_total, d_os + d_ss, bd.os, bd.ss, bd.kappa_e_os, bd.kappa_p99_block0, rms_tr / rms_rn
        );
        assert!(
            d_os.abs() <= bd.os && d_ss.abs() <= bd.ss,
            "{sys}: storage energy error exceeds the deterministic bound"
        );
        assert!(
            worst <= 1.0,
            "{sys}: a storage element exceeds eps_storage·S + eta ({worst:e})"
        );
        assert!(
            rms_rn > 2.0 * gamma(naux, U64),
            "{sys}: per-element RMS {rms_rn:.3e} is inside the f64 noise class 2·γ_naux(u64) = {:.3e}; is the rounding applied?",
            2.0 * gamma(naux, U64)
        );
        assert!(
            rms_tr > rms_rn,
            "{sys}: truncation is not worse than round-to-nearest ({rms_tr:e} vs {rms_rn:e})"
        );
    }
}

#[test]
fn the_host_twin_of_the_mixed_gemm_is_within_its_bounds_for_every_panel_width() {
    for sys in ["h2o", "c4h10"] {
        let (p, cpu) = fixture::prepare_scf(sys, "cc-pvdz", "cc-pvdz-ri", 0);
        let naux = p.b_ov.nrows();
        let babs = p.b_ov.mapv(f64::abs);
        let eta = eps::eta_abs(naux, stats::max_abs(&p.b_ov));
        for b in [8usize, 64, 1 << 30] {
            let eg = eps::eps_device(naux, b);
            let (m_os, m_ss) = energy_with(&p, mixed_gemm(b));
            let bd = bound::energy_bound(
                &p.b_ov,
                &p.eps,
                p.nocc,
                p.nvir,
                p.first_occ,
                p.nocc_total,
                bound::BoundSpec {
                    eps_g: eg,
                    eta,
                    kappa: None,
                },
            );
            let (d_os, d_ss) = (m_os - cpu.e_os, m_ss - cpu.e_ss);
            let mut el = stats::Elem::new(eg, eta);
            for i in 0..p.nocc {
                let (g, sab) = stats::block_g_s(&p.b_ov, &babs, i, p.nvir);
                let lo = i * p.nvir;
                let got = mixed_gemm(b)(
                    &p.b_ov.slice(s![.., lo..lo + p.nvir]).t(),
                    &p.b_ov.slice(s![.., lo..]),
                );
                el.add(&got.view(), &g.view(), &sab.view());
            }
            // the accumulation part alone: the twin against the f64 product of
            // the f32-ROUNDED operands, at its own factor
            let rn = round_trip_f32(&p.b_ov.view());
            let mut acc = stats::Elem::new(eps::eps_accumulation(naux, b), eta);
            for i in 0..p.nocc {
                let lo = i * p.nvir;
                let (_, sab) = stats::block_g_s(&p.b_ov, &babs, i, p.nvir);
                let got = mixed_gemm(b)(
                    &p.b_ov.slice(s![.., lo..lo + p.nvir]).t(),
                    &p.b_ov.slice(s![.., lo..]),
                );
                let want = rn
                    .slice(s![.., lo..lo + p.nvir])
                    .t()
                    .dot(&rn.slice(s![.., lo..]));
                acc.add(&got.view(), &want.view(), &sab.view());
            }
            assert!(
                acc.worst <= 1.0,
                "{sys} b={b}: an accumulation-only element exceeds eps_accumulation·S + eta ({:e})",
                acc.worst
            );
            eprintln!(
                "error map host twin {sys}/cc-pvdz b={b}: dE_os {d_os:+.3e} dE_ss {d_ss:+.3e} | bound {:.3e} {:.3e} | per-element RMS(err/S) {:.3e}, worst/bound {:.3e}",
                bd.os, bd.ss, el.rms(), el.worst
            );
            assert!(
                d_os.abs() <= bd.os && d_ss.abs() <= bd.ss,
                "{sys} b={b}: host-twin energy error exceeds the bound"
            );
            assert!(
                el.worst <= 1.0,
                "{sys} b={b}: a host-twin element exceeds eps_device·S + eta"
            );
        }
    }
}

#[test]
#[ignore = "N sweep through C12 (RI-JK SCF); about two minutes, run explicitly"]
fn storage_per_element_rms_falls_with_the_auxiliary_depth() {
    let mut pts = Vec::new();
    for sys in ["h2o", "c4h10", "c8h18", "alkane_12"] {
        let (p, _) = fixture::prepare_scf(sys, "cc-pvdz", "cc-pvdz-ri", 0);
        let naux = p.b_ov.nrows();
        let (eg, eta) = (
            eps::eps_storage(naux),
            eps::eta_abs(naux, stats::max_abs(&p.b_ov)),
        );
        let (rms, _, worst) = storage_elements(&p, eg, eta);
        eprintln!(
            "per-element storage {sys}: naux {naux} RMS(err/S) {rms:.3e} worst/bound {worst:.3e}"
        );
        assert!(worst <= 1.0);
        pts.push((naux as f64, rms));
    }
    let (slope, se) = stats::loglog_fit(&pts);
    let upper = slope + stats::t95(pts.len() - 2) * se;
    eprintln!(
        "per-element storage RMS vs naux: slope of ln RMS vs ln naux = {slope:.3} +- {se:.3} (df {}), one-sided 95% upper limit {upper:.3}; random-sign model: -0.5",
        pts.len() - 2
    );
    assert!(
        upper < 0.0,
        "the storage RMS does not demonstrably fall with naux: upper limit {upper:.3}"
    );
}
