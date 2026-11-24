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
    /// A valid request for a combination of options the method does not
    /// implement, refused before any work is done. Typed so a caller (and a
    /// test) can tell "this combination has no meaning here" from a failed
    /// computation without matching on message text. `feature` names what was
    /// asked for, `reason` says why it has no meaning and what to use instead.
    #[error("{feature} is not supported: {reason}")]
    Unsupported {
        /// The refused option or combination, as the user spelled it.
        feature: String,
        /// Why the combination is refused, and the supported alternative.
        reason: String,
    },
}

/// What kind of failure a [`FerricError`] is, for a front end that has to
/// pick a class (the Python bindings' exception hierarchy). The match in
/// [`FerricError::category`] is exhaustive, so a new variant cannot be added
/// without deciding its category.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCategory {
    /// An iterative procedure did not converge.
    Convergence,
    /// A dense linear-algebra call failed.
    Numerical,
    /// A memory budget or pool cannot cover a request.
    Memory,
    /// A valid request for something the method does not implement.
    Unsupported,
    /// A basis set or ECP could not be loaded or does not cover the molecule.
    Basis,
    /// Anything else (untyped text, I/O, parsing, the integral engine).
    Other,
}

impl FerricError {
    /// The [`ErrorCategory`] of this error. Exhaustive on purpose.
    pub fn category(&self) -> ErrorCategory {
        match self {
            FerricError::ScfConvergence { .. } | FerricError::Convergence(_) => {
                ErrorCategory::Convergence
            }
            FerricError::Lapack(_) => ErrorCategory::Numerical,
            FerricError::MemoryExhausted(_) => ErrorCategory::Memory,
            FerricError::Unsupported { .. } => ErrorCategory::Unsupported,
            FerricError::Basis(_) => ErrorCategory::Basis,
            FerricError::Io(_)
            | FerricError::XyzParse(_)
            | FerricError::Libint(_)
            | FerricError::General(_) => ErrorCategory::Other,
        }
    }

    /// One value of every variant, for tests that must cover them all. The
    /// array length is not tied to the enum, but [`FerricError::category`]'s
    /// exhaustive match forces a decision for a new variant, and the unit
    /// test below checks every category here is reached.
    pub fn samples(msg: &str) -> Vec<(&'static str, FerricError)> {
        vec![
            (
                "io",
                FerricError::Io(std::io::Error::other(msg.to_string())),
            ),
            ("xyz_parse", FerricError::XyzParse(msg.into())),
            ("basis", FerricError::Basis(msg.into())),
            ("libint", FerricError::Libint(msg.into())),
            (
                "scf_convergence",
                FerricError::ScfConvergence {
                    iterations: 7,
                    last_energy: -1.5,
                },
            ),
            ("lapack", FerricError::Lapack(msg.into())),
            ("convergence", FerricError::Convergence(msg.into())),
            ("general", FerricError::General(msg.into())),
            ("memory_exhausted", FerricError::MemoryExhausted(msg.into())),
            ("unsupported", FerricError::unsupported("a feature", msg)),
        ]
    }

    /// Build a [`FerricError::Unsupported`].
    pub fn unsupported(feature: impl Into<String>, reason: impl Into<String>) -> Self {
        FerricError::Unsupported {
            feature: feature.into(),
            reason: reason.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Adding a variant fails `category`'s exhaustive match; this checks the
    /// sample list (which the Python mapping test iterates) keeps up and
    /// reaches every category.
    #[test]
    fn samples_cover_every_variant_and_category() {
        let samples = FerricError::samples("x");
        let mut seen = std::collections::HashSet::new();
        for (name, e) in &samples {
            seen.insert(format!("{:?}", e.category()));
            let variant = format!("{e:?}");
            assert!(!name.is_empty() && !variant.is_empty());
        }
        // One sample per variant: update with the enum (its `category` match
        // will not compile until the new variant has a category).
        assert_eq!(samples.len(), 10);
        for c in [
            "Convergence",
            "Numerical",
            "Memory",
            "Unsupported",
            "Basis",
            "Other",
        ] {
            assert!(seen.contains(c), "no sample reaches {c}");
        }
    }
}
