//! The deterministic energy-error bound shared by the RI-MP2 device tests.
//!
//! One element of the wide block G_i = B_iᵀ·B_tail (accumulation depth
//! k = naux) computed with a per-element factor `eps_g` and absolute term `eta`
//! satisfies |ĝ_ab − g_ab| ≤ eps_g · S_ab + eta, S_ab = Σ_P |B_P,ia||B_P,jb|
//! (Higham, ASNA §3.1/3.5, standard model fl(x op y) = (x op y)(1+δ),
//! |δ| ≤ u; `eta` is the f32 underflow term, see `rimp2_eps.rs`, zero for f64).
//! With d_ab = eps_g·S_ab + eta and D = ε_i + ε_j − ε_a − ε_b (pair weight
//! fac = 1 for i = j, 2 otherwise, damping damp = (1 − e^{κD})² ≤ 1 when κ is
//! set, else 1), per term
//!   OS  damp·g_ab²/D:  |Δ| ≤ fac·damp·d_ab(2|g_ab| + d_ab)/|D|
//!   SS  damp·(g_ab² − g_ab g_ba)/D:
//!       |Δ| ≤ fac·damp·[d_ab(2|g_ab| + d_ab) + d_ab|g_ba| + |g_ab|d_ba + d_ab d_ba]/|D|
//! (d_ba is the perturbation of the transposed element). The two evaluations
//! also fold the same terms in f64 in the same order, so their summation
//! roundoff differs by at most γ_m(Σ|t| + Σ|t̂|), m = nvir² + 2·nocc + 8 (the
//! +8 covers the few operations inside one term), Σ|t̂| ≤ Σ|t| + the bound
//! above: that term is added, so the result bounds |E(ĝ) − E(g)| in full.
//!
//! WHAT THIS BOUND CAN AND CANNOT CATCH. It is a worst case over rounding
//! signs, of order (b + 2)·u32·S for the shipped kernel, 3 to 6 decades above
//! every measured energy error. It therefore catches gross defects (a wrong
//! operand offset, a dropped panel or pair weight: errors of 1e-1 relative) and
//! nothing subtle: a plain SGEMM without the f64 flush, an f32 B_ov with an f64
//! GEMM, a truncating instead of round-to-nearest upload, or f32 accumulation
//! across panels all pass it. `gpu_rimp2_mixed.rs` pins them with a bit-exact
//! upload test, a deterministic flush construction and a (supplementary)
//! measured per-element comparison; see its module docs.
use ferric_core::gpu::mixed_host::{gamma, U64};
use ndarray::{linalg::general_mat_mul, s, Array2, ArrayView2};

pub struct EnergyBound {
    /// Bound on |ΔE_os|, |ΔE_ss| (host summation term included).
    pub os: f64,
    pub ss: f64,
    /// κ_E = Σ fac·damp·|g|S/|D| / |E_os|: the energy-level condition number,
    /// the factor by which a per-element relative perturbation e moves E_os
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
struct PairCtx<'a> {
    g: ArrayView2<'a, f64>,
    sab: ArrayView2<'a, f64>,
    jcol: usize,
    nvir: usize,
    e_ij: f64,
    eps: &'a [f64],
    nocc_total: usize,
    eps_g: f64,
    eta: f64,
    kappa: Option<f64>,
    fac: f64,
}

fn add_pair(acc: &mut Acc, c: &PairCtx) {
    for a in 0..c.nvir {
        for b in 0..c.nvir {
            let (g_ab, g_ba) = (c.g[(a, c.jcol + b)].abs(), c.g[(b, c.jcol + a)].abs());
            let (d_ab, d_ba) = (
                c.eps_g * c.sab[(a, c.jcol + b)] + c.eta,
                c.eps_g * c.sab[(b, c.jcol + a)] + c.eta,
            );
            let denom = c.e_ij - c.eps[c.nocc_total + a] - c.eps[c.nocc_total + b];
            let damp = c.kappa.map_or(1.0, |k| {
                let d1 = 1.0 - (k * denom).exp();
                d1 * d1
            });
            let w = c.fac * damp / denom.abs();
            let sq = d_ab * (2.0 * g_ab + d_ab);
            acc.os += w * sq;
            acc.ss += w * (sq + d_ab * g_ba + g_ab * d_ba + d_ab * d_ba);
            acc.abs_os += w * g_ab * g_ab;
            acc.abs_ss += w * g_ab * (g_ab + g_ba);
            acc.s_os += w * g_ab * c.sab[(a, c.jcol + b)];
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

/// The error model of one evaluation: per-element factor `eps_g`, absolute
/// underflow term `eta`, optional κ-damping of the energy.
pub struct BoundSpec {
    pub eps_g: f64,
    pub eta: f64,
    pub kappa: Option<f64>,
}

/// The bound for (OS, SS) under `spec`; `b_ov` is the f64 (naux × nocc·nvir)
/// tensor.
pub fn energy_bound(
    b_ov: &Array2<f64>,
    eps: &[f64],
    nocc: usize,
    nvir: usize,
    first_occ: usize,
    nocc_total: usize,
    spec: BoundSpec,
) -> EnergyBound {
    let BoundSpec { eps_g, eta, kappa } = spec;
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
            add_pair(
                &mut acc,
                &PairCtx {
                    g: g.view(),
                    sab: sab.view(),
                    jcol: (j - i) * nvir,
                    nvir,
                    e_ij: eps[first_occ + i] + eps[first_occ + j],
                    eps,
                    nocc_total,
                    eps_g,
                    eta,
                    kappa,
                    fac: if i == j { 1.0 } else { 2.0 },
                },
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
