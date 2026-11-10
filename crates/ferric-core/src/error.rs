/// Unified error type for the ferric workspace.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum FerricError {
    /// An I/O operation failed; converted from [`std::io::Error`] by `?`.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    /// Malformed XYZ input: empty text, bad atom count, too few lines or
    /// fields, unknown element, or an unparsable coordinate.
    #[error("XYZ parse error: {0}")]
    XyzParse(String),
    /// A basis set or ECP could not be loaded, parsed or looked up, or does not
    /// cover an element of the molecule.
    #[error("basis error: {0}")]
    Basis(String),
    /// An integral-engine (libint2 shim) call failed or was given inputs it
    /// cannot handle.
    #[error("libint error: {0}")]
    Libint(String),
    /// An SCF loop reached its iteration limit without meeting its
    /// convergence criteria.
    #[error("SCF did not converge after {iterations} iterations (last energy: {last_energy:.10})")]
    ScfConvergence {
        /// SCF iterations performed before giving up.
        iterations: usize,
        /// Total energy of the last iteration (Hartree).
        last_energy: f64,
    },
    /// A dense linear-algebra call (eigensolver, factorization, solve) failed
    /// or was given a malformed matrix.
    #[error("LAPACK error: {0}")]
    Lapack(String),
    /// An iterative procedure did not converge; the message names the solver
    /// (used where the structured [`FerricError::ScfConvergence`] is not).
    #[error("Convergence error: {0}")]
    Convergence(String),
    /// Any other failure, including invalid configuration and errors converted
    /// from other crates' error types.
    #[error("General error: {0}")]
    General(String),
    /// A memory budget or the shared pool cannot cover a request. Typed so an
    /// optional stage (a guess) can give way to it without matching on message
    /// text, and without swallowing any other failure. The message carries the
    /// occupancy breakdown, so `Display` adds no prefix.
    #[error("{0}")]
    MemoryExhausted(String),
}
