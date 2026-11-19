//! CLI error seam: the error type future `?`-based run functions will return,
//! and the transitional exit helpers that keep today's stderr/exit-code
//! behaviour byte-identical.

use std::fmt;

/// Error type for CLI run functions. Displays as its message only, so the
/// `error: {e}` line printed on exit is unchanged.
#[derive(Debug)]
pub enum CliError {
    /// An error from the numerical crates, kept as a typed source.
    Ferric(ferric_core::FerricError),
    /// A plain message (config validation, argument errors, ...).
    Msg(String),
}

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CliError::Ferric(e) => fmt::Display::fmt(e, f),
            CliError::Msg(s) => f.write_str(s),
        }
    }
}

impl std::error::Error for CliError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            CliError::Ferric(e) => Some(e),
            CliError::Msg(_) => None,
        }
    }
}

impl From<ferric_core::FerricError> for CliError {
    fn from(e: ferric_core::FerricError) -> Self {
        CliError::Ferric(e)
    }
}

impl From<String> for CliError {
    fn from(s: String) -> Self {
        CliError::Msg(s)
    }
}

impl From<&str> for CliError {
    fn from(s: &str) -> Self {
        CliError::Msg(s.to_string())
    }
}

/// Print `error: {msg}` to stderr and exit 1, the CLI's convention.
pub(crate) fn die(msg: impl fmt::Display) -> ! {
    eprintln!("error: {msg}");
    std::process::exit(1)
}

/// Unwrap a `Result`, or print `error: {e}` to stderr and exit 1.
///
/// Transitional: a step toward run functions returning `Result<_, CliError>`
/// (propagated with `?`), with the process exit happening only in `main`/`run`.
pub(crate) trait OrExit<T> {
    fn or_exit(self) -> T;
}

impl<T, E: fmt::Display> OrExit<T> for Result<T, E> {
    fn or_exit(self) -> T {
        self.unwrap_or_else(|e| die(e))
    }
}
