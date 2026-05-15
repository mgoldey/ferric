pub mod config;
pub mod davidson;
pub mod diagnostics;
pub mod energy;
pub mod quadrature;
pub mod sternheimer;

use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::RhfResult;
use ndarray::Array2;

pub use config::PdepRpaConfig;

/// Results from a PDEP-RPA calculation.
#[derive(Debug)]
pub struct PdepRpaResult {
    /// RPA correlation energy in Hartree.
    pub e_rpa: f64,
    /// Number of eigenpotentials retained after truncation.
    pub n_eigenpotentials: usize,
    /// Static dielectric eigenvalues λ_α(0), length M.
    pub eigenvalues_static: Vec<f64>,
    /// Imaginary-frequency quadrature points ω_k.
    pub quad_freqs: Vec<f64>,
    /// Quadrature weights w_k.
    pub quad_weights: Vec<f64>,
    /// λ_α(iω_k) tensor, shape (N_quad, M).
    pub eigenvalues_freq: Array2<f64>,
    /// RI-dRPA sanity-check energy (None unless run_diagnostics=true).
    pub e_rpa_dft_diag: Option<f64>,
}

/// Top-level PDEP-RPA energy calculation.
pub fn run_pdep_rpa(
    _mol: &Molecule,
    _obs: &PreparedBasis,
    _dfbs: &PreparedBasis,
    _op: Operator,
    _rhf: &RhfResult,
    _config: &PdepRpaConfig,
) -> Result<PdepRpaResult, FerricError> {
    todo!("implemented in later tasks")
}
