//! Per-element error factors of the RI-MP2 G_i block (Higham, ASNA §3.1/3.5,
//! standard model fl(x op y) = (x op y)(1+δ), |δ| ≤ u; u32 = 2⁻²⁴, u64 = 2⁻⁵³).
//! Every factor multiplies S_ab = Σ_P |B_P,ia||B_P,jb|; both sides of every
//! comparison here are FLOATING-POINT evaluations, so each f64 reference
//! contributes its own γ_naux(u64).
use ferric_core::gpu::mixed_host::{gamma, U32, U64};

/// Storage only: the CPU f64 product of the f32-rounded B_ov against the exact
/// f64 product. Each operand carries a relative error ≤ u32, so a term carries
/// (1+u32)² − 1 = 2·u32 + u32², and both products are f64 sums of depth naux:
/// 2·u32 + u32² + 2·γ_naux(u64).
pub fn eps_storage(naux: usize) -> f64 {
    2.0 * U32 + U32 * U32 + 2.0 * gamma(naux, U64)
}

/// The shipped device kernel against the exact f64 product. A term carries
/// (1+u32)² from storage; a panel of b products summed in f32 adds
/// (1 + γ_b(u32)); the ⌈naux/b⌉ panels are summed in f64, (1 + γ_⌈naux/b⌉(u64));
/// the f64 reference adds γ_naux(u64). The product form is exact, so no slack
/// factor is applied. `b` is clamped to [1, naux].
pub fn eps_device(naux: usize, b: usize) -> f64 {
    let naux = naux.max(1);
    let b = b.clamp(1, naux);
    let panels = naux.div_ceil(b);
    (1.0 + U32) * (1.0 + U32) * (1.0 + gamma(b, U32)) * (1.0 + gamma(panels, U64)) - 1.0
        + gamma(naux, U64)
}

/// The ACCUMULATION part alone: the device (or host twin) against the f64
/// product of the f32-ROUNDED B_ov (not the exact one): the f32 panel sums
/// (1 + γ_b(u32)), the f64 panel sum (1 + γ_⌈naux/b⌉(u64)) and the rounded
/// reference's own γ_naux(u64); S of the rounded operands is at most (1+u32)²
/// times S of the exact ones.
pub fn eps_accumulation(naux: usize, b: usize) -> f64 {
    let naux = naux.max(1);
    let b = b.clamp(1, naux);
    let panels = naux.div_ceil(b);
    ((1.0 + gamma(b, U32)) * (1.0 + gamma(panels, U64)) - 1.0 + gamma(naux, U64))
        * (1.0 + U32)
        * (1.0 + U32)
}

/// The f32 underflow term: a product or an operand below the f32 normal range
/// (1.2e-38) is rounded with an ABSOLUTE error ≤ 2⁻¹⁵⁰ instead of a relative
/// one, and sums of subnormals are exact. Per product term at most
/// 2⁻¹⁵⁰·(1 + 2·max|B|) (the product, plus the two operands each scaled by the
/// other, ≤ max|B|); naux terms: conservatively naux·2⁻¹⁴⁹·(2 + 2·max|B|). It
/// only matters where S_ab itself is below ~1e-30; elsewhere it is invisible.
pub fn eta_abs(naux: usize, max_abs_b: f64) -> f64 {
    naux as f64 * f64::from(f32::from_bits(1)) * (2.0 + 2.0 * max_abs_b)
}
