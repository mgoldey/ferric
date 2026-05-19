//! Abstraction over the XC contribution to the Kohn-Sham Fock matrix.
//!
//! Defined here (in `ferric-dft`, not `ferric-scf`) so a concrete `KsXc`
//! implementation can live alongside the grid/AO/libxc machinery without
//! creating a circular crate dependency.

use ndarray::Array2;

/// How exact exchange is mixed into the Fock matrix.
///
/// The SCF loop builds K according to:
///
/// * `omega == 0`, `sr == lr`: plain hybrid. Build one K with the Coulomb
///   operator and use `sr` as the mixing coefficient.
/// * `omega > 0`: range-separated hybrid. Build `K_SR(ω)` (via `Operator::erfc(ω)`)
///   and `K_LR(ω)` (via `Operator::erf(ω)`), then combine
///   `K_total = sr · K_SR(ω) + lr · K_LR(ω)`.
/// * `sr == 0 && lr == 0`: pure functional (LDA, PBE) with no exact exchange.
///
/// For pure HF (no DFT), `KMix { sr: 1.0, lr: 1.0, omega: 0.0 }` reduces to the
/// existing `F = h + 2J − K` path with a single full-Coulomb K.
#[derive(Debug, Clone, Copy)]
pub struct KMix {
    pub sr: f64,
    pub lr: f64,
    pub omega: f64,
}

impl Default for KMix {
    fn default() -> Self {
        // Plain HF: full exact exchange, no range-separation.
        Self { sr: 1.0, lr: 1.0, omega: 0.0 }
    }
}

/// Pure-DFT (semilocal + nonlocal correlation) contribution to the Fock matrix.
///
/// The caller (SCF) handles J and K itself; this trait covers only the V_xc
/// (and any VV10 V_nl) addition to F. The trait's `k_mix()` method tells the
/// caller how to build K.
pub trait XcContribution: Send + Sync {
    /// Adds V_xc (semilocal) + V_nl (VV10, if any) to `f` in place.
    /// Returns the corresponding energy contribution E_xc + E_nl in Ha.
    fn add_xc(&self, d: &Array2<f64>, f: &mut Array2<f64>) -> f64;

    /// How to build the exact-exchange contribution for this functional.
    fn k_mix(&self) -> KMix;
}
