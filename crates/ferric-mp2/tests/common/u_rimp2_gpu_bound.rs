//! The derived device-vs-CPU energy bound for the U-RI-MP2 blocks.
//!
//! One element of the wide block G_i = B_iᵀ·B_other (accumulation depth
//! k = naux) differs between any two f64 summation orders by at most
//! δ_ab = 2·γ_naux·S_ab, S_ab = Σ_P |B_P,ia||B_P,jb| (Higham, ASNA Eq. 3.5,
//! γ_n = n·u/(1 − n·u), u = 2⁻⁵³; each order is bounded by γ_naux·S, so two
//! orders differ by at most twice that). With D = ε_i + ε_j − ε_a − ε_b the
//! terms perturb by
//!   OS  g²/D:                 |Δ| ≤ δ(2|g| + δ)/|D|
//!   SS  K²/D, K = g_ab − g_ba: |Δ| ≤ dK(2|K| + dK)/|D|,  dK = δ_ab + δ_ba
//! (SS weights: pair factor fac ∈ {1, 2} and the prefactor ¼). The two
//! evaluations also fold the same terms in f64 in the same order, so their
//! summation roundoff differs by at most γ_m(Σ|t| + Σ|t̂|) with Σ|t̂| ≤ Σ|t| +
//! the perturbation sum; m = nvir² + 2·nocc + 8 (SS: nvir² terms per pair, ≤
//! nocc pairs, ≤ nocc blocks, +8 for the operations inside one term) and
//! m = nvir_a·nocc_b·nvir_b + nocc_a + 8 (OS). No machine constant is asserted.
//!
//! The same propagation serves any per-element model δ_ab = ε_G·S_ab + η
//! ([`ElemModel`]): the f64 device gate above is ε_G = 2·γ_naux(u64), η = 0;
//! the mixed kernel (`gpu_u_rimp2_mixed.rs`) is ε_G = `eps_device(naux, b)`
//! (`rimp2_eps.rs`, the closed-shell kernel's factor: f32 storage, f32 panel
//! sums of width b, f64 sum of the panels, plus the f64 reference's own
//! γ_naux) and η its f32 underflow term.
use crate::fixture::Chan;
use ferric_core::gpu::mixed_host::{gamma, U64};
use ndarray::{s, Array2};

struct Block {
    g: Array2<f64>,
    d: Array2<f64>,
}

/// Per-element perturbation model δ_ab = `eps_g`·S_ab + `eta`.
#[derive(Clone, Copy)]
pub struct ElemModel {
    pub eps_g: f64,
    pub eta: f64,
}

impl ElemModel {
    /// Two f64 summation orders of depth `naux`: 2·γ_naux(u64), no absolute term.
    pub fn f64_orders(naux: usize) -> Self {
        Self {
            eps_g: 2.0 * gamma(naux, U64),
            eta: 0.0,
        }
    }
}

fn block(
    left: &Array2<f64>,
    i: usize,
    nvl: usize,
    right: &Array2<f64>,
    col0: usize,
    m: ElemModel,
) -> Block {
    let bi = left.slice(s![.., i * nvl..(i + 1) * nvl]);
    let br = right.slice(s![.., col0..]);
    let g = bi.t().dot(&br);
    let sab = bi.mapv(f64::abs).t().dot(&br.mapv(f64::abs));
    Block {
        g,
        d: sab.mapv(|x| m.eps_g * x + m.eta),
    }
}

fn finish(pert: f64, abs_t: f64, m: usize) -> f64 {
    pert + gamma(m, U64) * (2.0 * abs_t + pert)
}

/// Bound on |E_device − E_cpu| of the same-spin block (f64 device gate).
#[allow(dead_code)] // not every binary that includes this module uses both models
pub fn same_spin_bound(c: &Chan) -> f64 {
    same_spin_bound_with(c, ElemModel::f64_orders(c.b.nrows()))
}

/// Bound on |E_device − E_cpu| of the opposite-spin block (f64 device gate).
#[allow(dead_code)]
pub fn opposite_spin_bound(a: &Chan, b: &Chan) -> f64 {
    opposite_spin_bound_with(a, b, ElemModel::f64_orders(a.b.nrows()))
}

/// Same-spin bound under the per-element model `m`.
pub fn same_spin_bound_with(c: &Chan, m: ElemModel) -> f64 {
    let (mut pert, mut abs_t) = (0.0, 0.0);
    for i in 0..c.nocc {
        let blk = block(&c.b, i, c.nvir, &c.b, i * c.nvir, m);
        let ei = c.eps[c.first_occ + i];
        for j in i..c.nocc {
            let fac = if i == j { 1.0 } else { 2.0 };
            let jc = (j - i) * c.nvir;
            let ej = c.eps[c.first_occ + j];
            for a in 0..c.nvir {
                for b in 0..c.nvir {
                    let den = (ei + ej - c.eps[c.nocc_total + a] - c.eps[c.nocc_total + b]).abs();
                    let k = blk.g[(a, jc + b)] - blk.g[(b, jc + a)];
                    let dk = blk.d[(a, jc + b)] + blk.d[(b, jc + a)];
                    pert += 0.25 * fac * dk * (2.0 * k.abs() + dk) / den;
                    abs_t += 0.25 * fac * k * k / den;
                }
            }
        }
    }
    finish(pert, abs_t, c.nvir * c.nvir + 2 * c.nocc + 8)
}

/// Opposite-spin bound under the per-element model `m`.
pub fn opposite_spin_bound_with(a: &Chan, b: &Chan, m: ElemModel) -> f64 {
    let (mut pert, mut abs_t) = (0.0, 0.0);
    for i in 0..a.nocc {
        let blk = block(&a.b, i, a.nvir, &b.b, 0, m);
        let ei = a.eps[a.first_occ + i];
        for x in 0..a.nvir {
            for jj in 0..b.nocc {
                for y in 0..b.nvir {
                    let den = (ei + b.eps[b.first_occ + jj]
                        - a.eps[a.nocc_total + x]
                        - b.eps[b.nocc_total + y])
                        .abs();
                    let col = jj * b.nvir + y;
                    let (g, d) = (blk.g[(x, col)], blk.d[(x, col)]);
                    pert += d * (2.0 * g.abs() + d) / den;
                    abs_t += g * g / den;
                }
            }
        }
    }
    finish(pert, abs_t, a.nvir * b.nocc * b.nvir + a.nocc + 8)
}
