/// Unified error type for the ferric workspace.
#[derive(Debug, thiserror::Error)]
pub enum FerricError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("XYZ parse error: {0}")]
    XyzParse(String),
    #[error("basis error: {0}")]
    Basis(String),
    #[error("libint error: {0}")]
    Libint(String),
    #[error("SCF did not converge after {iterations} iterations (last energy: {last_energy:.10})")]
    ScfConvergence { iterations: usize, last_energy: f64 },
    #[error("LAPACK error: {0}")]
    Lapack(String),
}
