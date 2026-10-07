//! The deterministic energy-error bound shared by the mixed RI-MP2 tests.
//!
//! One element of the wide block G_i = B_iᵀ·B_tail (accumulation depth
//! k = naux) computed with a per-element relative-to-S error factor `eps_g`
//! satisfies |ĝ_ab − g_ab| ≤ eps_g · S_ab, S_ab = Σ_P |B_P,ia||B_P,jb|
//! (Higham, ASNA §3.1/3.5, standard model fl(x op y) = (x op y)(1+δ),
//! |δ| ≤ u). `eps_g` is 2·u32 for f32 storage alone, and
//! 2·u32 + γ_b(u32) + γ_⌈naux/b⌉(u64) (+ the f64 reference's own γ_naux(u64))
//! for the device kernel; the caller chooses it. With
//! d_ab = eps_g·S_ab and D = ε_i + ε_j − ε_a − ε_b (pair weight fac = 1 for
//! i = j, 2 otherwise), per term
//!   OS  g_ab²/D:           |Δ| ≤ fac·d_ab(2|g_ab| + d_ab)/|D|
//!   SS  (g_ab² − g_ab g_ba)/D:
//!       |Δ| ≤ fac·[d_ab(2|g_ab| + d_ab) + d_ab|g_ba| + |g_ab|d_ba + d_ab d_ba]/|D|
//! (d_ba = eps_g·S_ba is the perturbation of the transposed element). The two
//! evaluations also fold the same terms in f64 in the same order, so their
//! summation roundoff differs by at most γ_m(Σ|t| + Σ|t̂|), m = nvir² + 2·nocc
//! + 8 (the +8 covers the few operations inside one term), Σ|t̂| ≤ Σ|t| + the
//! bound above: that term is added, so the result bounds |E(ĝ) − E(g)| in full.
//! No machine constant is asserted; every number is derived from the data.
use ferric_core::gpu::mixed_host::{gamma, U64};
use ndarray::{linalg::general_mat_mul, s, Array2, ArrayView2};

pub struct EnergyBound {
    /// Bound on |ΔE_os|, |ΔE_ss| (host summation term included).
    pub os: f64,
    pub ss: f64,
    /// κ_E = Σ fac|g|S/|D| / |E_os|: the energy-level condition number, the
    /// factor by which a per-element relative perturbation e moves E_os
    /// (|ΔE_os|/|E_os| ≲ 2·e·κ_E to first order). E_os has one sign, so this
    /// has no cancellation in its denominator.
    pub kappa_e_os: f64,
    /// 99th percentile of the element-wise κ_sum = S_ab/|g_ab| over the i = 0
    /// block (elements with g = 0 excluded).
    pub kappa_p99_block0: f64,
}

struct Acc {
    os: f64,
    ss: f64,
    abs_os: f64,
    abs_ss: f64,
    s_os: f64,
}

/// One (i, j) pair's contribution; `g`/`sab` are the i-block, `jcol` the
/// column offset of j inside the tail.
fn add_pair(
    acc: &mut Acc,
    g: &ArrayView2<f64>,
    sab: &ArrayView2<f64>,
    jcol: usize,
    nvir: usize,
    e_ij: f64,
    eps: &[f64],
    nocc_total: usize,
    eps_g: f64,
    fac: f64,
) {
    for a in 0..nvir {
        for b in 0..nvir {
            let (g_ab, g_ba) = (g[(a, jcol + b)].abs(), g[(b, jcol + a)].abs());
            let (d_ab, d_ba) = (eps_g * sab[(a, jcol + b)], eps_g * sab[(b, jcol + a)]);
            let denom = (e_ij - eps[nocc_total + a] - eps[nocc_total + b]).abs();
            let w = fac / denom;
            let sq = d_ab * (2.0 * g_ab + d_ab);
            acc.os += w * sq;
            acc.ss += w * (sq + d_ab * g_ba + g_ab * d_ba + d_ab * d_ba);
            acc.abs_os += w * g_ab * g_ab;
            acc.abs_ss += w * g_ab * (g_ab + g_ba);
            acc.s_os += w * g_ab * sab[(a, jcol + b)];
        }
    }
}

fn p99(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let k = ((v.len() as f64 * 0.99) as usize).min(v.len() - 1);
    *v.select_nth_unstable_by(k, |a, b| a.total_cmp(b)).1
}

/// The bound for (OS, SS) at per-element factor `eps_g`; `b_ov` is the f64
/// (naux × nocc·nvir) tensor.
pub fn energy_bound(
    b_ov: &Array2<f64>,
    eps: &[f64],
    nocc: usize,
    nvir: usize,
    first_occ: usize,
    nocc_total: usize,
    eps_g: f64,
) -> EnergyBound {
    let babs = b_ov.mapv(f64::abs);
    let mut acc = Acc {
        os: 0.0,
        ss: 0.0,
        abs_os: 0.0,
        abs_ss: 0.0,
        s_os: 0.0,
    };
    let mut kappas = Vec::new();
    for i in 0..nocc {
        let lo = i * nvir;
        let g = b_ov
            .slice(s![.., lo..lo + nvir])
            .t()
            .dot(&b_ov.slice(s![.., lo..]));
        let mut sab = Array2::zeros(g.dim());
        general_mat_mul(
            1.0,
            &babs.slice(s![.., lo..lo + nvir]).t(),
            &babs.slice(s![.., lo..]),
            0.0,
            &mut sab,
        );
        if i == 0 {
            kappas = g
                .iter()
                .zip(sab.iter())
                .filter(|(c, sv)| c.abs() > 0.0 && **sv > 0.0)
                .map(|(c, sv)| sv / c.abs())
                .collect();
        }
        for j in i..nocc {
            let fac = if i == j { 1.0 } else { 2.0 };
            let e_ij = eps[first_occ + i] + eps[first_occ + j];
            add_pair(
                &mut acc,
                &g.view(),
                &sab.view(),
                (j - i) * nvir,
                nvir,
                e_ij,
                eps,
                nocc_total,
                eps_g,
                fac,
            );
        }
    }
    let gm = gamma(nvir * nvir + 2 * nocc + 8, U64);
    EnergyBound {
        os: acc.os + gm * (2.0 * acc.abs_os + acc.os),
        ss: acc.ss + gm * (2.0 * acc.abs_ss + acc.ss),
        kappa_e_os: acc.s_os / acc.abs_os,
        kappa_p99_block0: p99(kappas),
    }
}

/// Least-squares line through (ln x, ln y): (slope, standard error of the
/// slope, intercept). With n = 2 the standard error is reported as 0.
pub fn loglog_fit(points: &[(f64, f64)]) -> (f64, f64, f64) {
    let n = points.len() as f64;
    let (lx, ly): (Vec<f64>, Vec<f64>) = points.iter().map(|&(x, y)| (x.ln(), y.ln())).unzip();
    let (mx, my) = (lx.iter().sum::<f64>() / n, ly.iter().sum::<f64>() / n);
    let sxx: f64 = lx.iter().map(|x| (x - mx) * (x - mx)).sum();
    let sxy: f64 = lx.iter().zip(&ly).map(|(x, y)| (x - mx) * (y - my)).sum();
    let slope = sxy / sxx;
    let icpt = my - slope * mx;
    let rss: f64 = lx
        .iter()
        .zip(&ly)
        .map(|(x, y)| (y - icpt - slope * x).powi(2))
        .sum();
    let se = if points.len() > 2 {
        (rss / (n - 2.0) / sxx).sqrt()
    } else {
        0.0
    };
    (slope, se, icpt)
}
