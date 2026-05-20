/// GW-family method selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GwMethod {
    /// One-shot G0W0 on the input mean-field reference (HF in spike P0).
    G0W0,
    /// Static screened-exchange + Coulomb-hole (W(iω=0) only).
    Cohsex,
    /// Eigenvalue self-consistency on G; W frozen at iteration 0.
    EvGw0,
    /// Eigenvalue self-consistency on both G and W.
    EvGw,
    /// Density-matrix self-consistency for COHSEX.
    ScCohsex,
}
