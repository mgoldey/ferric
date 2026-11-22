//! CLI error type: what every run function returns, and how the top level
//! ([`crate::run`]) reports it.
//!
//! The CLI's run functions return `Result<_, CliError>` and propagate with
//! `?`; only [`crate::run`] prints the error and calls `std::process::exit`.
//! Each variant fixes both the bytes written to stderr and the exit status,
//! so a returned error is reported exactly as the old in-place
//! `eprintln!` + `process::exit` was.

use std::fmt;

/// Error type for CLI run functions.
///
/// [`CliError::report`] writes it to stderr and [`CliError::exit_code`] is
/// the process status:
///
/// | variant  | stderr                | status |
/// |----------|-----------------------|--------|
/// | `Ferric` | `error: {e}\n`        | 1      |
/// | `Msg`    | `error: {msg}\n`      | 1      |
/// | `Report` | `{text}\n` (verbatim) | 1      |
/// | `Usage`  | `{text}\n` (verbatim) | 2      |
///
/// `Display` is the message only (no `error: ` prefix), the text a caller
/// that does not exit would show.
#[derive(Debug)]
pub enum CliError {
    /// An error from the numerical crates, kept as a typed source.
    Ferric(ferric_core::FerricError),
    /// A plain message (config validation, argument errors, ...), printed
    /// after `error: `.
    Msg(String),
    /// A message printed as written, with no `error: ` prefix added: the
    /// sites whose text starts differently (`config error: ...`,
    /// `error computing frequencies: ...`) and multi-line messages (an
    /// `error: ...` line followed by a hint). Exit status 1.
    Report(String),
    /// A command-line usage error, printed as written: the `--help` text, the
    /// synopsis line, or an `error: ...` line about a flag. Exit status 2.
    Usage(String),
}

impl CliError {
    /// [`CliError::Msg`] from anything printable: the message `error: {e}`
    /// shows.
    pub fn msg(e: impl fmt::Display) -> Self {
        CliError::Msg(e.to_string())
    }

    /// The process exit status this error ends the run with: 2 for a usage
    /// error, 1 otherwise.
    #[must_use]
    pub fn exit_code(&self) -> i32 {
        match self {
            CliError::Usage(_) => 2,
            CliError::Ferric(_) | CliError::Msg(_) | CliError::Report(_) => 1,
        }
    }

    /// The exact text written to stderr for this error, trailing newline
    /// included.
    #[must_use]
    pub fn stderr_text(&self) -> String {
        match self {
            CliError::Ferric(e) => format!("error: {e}\n"),
            CliError::Msg(s) => format!("error: {s}\n"),
            CliError::Report(s) | CliError::Usage(s) => format!("{s}\n"),
        }
    }

    /// Write [`CliError::stderr_text`] to stderr.
    pub fn report(&self) {
        eprint!("{}", self.stderr_text());
    }
}

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CliError::Ferric(e) => fmt::Display::fmt(e, f),
            CliError::Msg(s) | CliError::Report(s) | CliError::Usage(s) => f.write_str(s),
        }
    }
}

impl std::error::Error for CliError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            CliError::Ferric(e) => Some(e),
            CliError::Msg(_) | CliError::Report(_) | CliError::Usage(_) => None,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stderr_text_and_status_per_variant() {
        let cases = [
            (CliError::msg("bad key"), "error: bad key\n", 1),
            (
                CliError::Ferric(ferric_core::FerricError::Basis("no basis".into())),
                &*format!(
                    "error: {}\n",
                    ferric_core::FerricError::Basis("no basis".into())
                ),
                1,
            ),
            (
                CliError::Report("config error: x".into()),
                "config error: x\n",
                1,
            ),
            (
                CliError::Usage("usage: ferric".into()),
                "usage: ferric\n",
                2,
            ),
        ];
        for (e, text, code) in cases {
            assert_eq!(e.stderr_text(), text);
            assert_eq!(e.exit_code(), code);
        }
    }
}
