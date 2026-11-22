//! CLI error type: what every run function returns, and how the top level
//! ([`crate::run`]) prints it.
//!
//! The CLI's run functions return `Result<_, CliError>` and propagate with
//! `?`; only [`crate::run`] prints the error and calls `std::process::exit`.
//! Each variant fixes both the bytes written to stderr and the exit status,
//! so a returned error is reported exactly as the old in-place
//! `eprintln!` + `process::exit` was.

use std::fmt;

/// Error type for CLI run functions.
///
/// [`CliError::eprint`] writes it to stderr and [`CliError::exit_code`] is
/// the process status:
///
/// | variant    | stderr                | status |
/// |------------|-----------------------|--------|
/// | `Ferric`   | `error: {e}\n`        | 1      |
/// | `Msg`      | `error: {msg}\n`      | 1      |
/// | `Verbatim` | `{text}\n` (as is)    | 1      |
/// | `Usage`    | `{text}\n` (as is)    | 2      |
///
/// Build the string variants with [`CliError::msg`], [`CliError::verbatim`]
/// and [`CliError::usage`], which check (in debug builds) that a `Msg` does
/// not carry its own `error:` prefix and that usage text (always a literal)
/// has no trailing newline. `Msg`/`Verbatim` often wrap a library message,
/// and some end in a newline (a TOML parse error's caret block): that one
/// is printed as is, a blank line included, as it always was. There is deliberately no `From<String>`: each site picks the
/// variant, so a message is never prefixed twice or not at all by accident.
///
/// `Display` writes the stderr text without its trailing newline and without
/// the `error: ` that [`CliError::stderr_text`] adds to `Ferric`/`Msg`.
/// `Verbatim`/`Usage` text is written as is, so it can itself start with
/// `error: `; [`CliError::message`] strips that too.
#[derive(Debug)]
#[non_exhaustive]
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
    Verbatim(String),
    /// A command-line usage error, printed as written: the synopsis line,
    /// the flag list (no input file, or a second positional argument), or an
    /// `error: ...` line about a flag. Exit status 2. (`--help` itself is not
    /// an error: `try_run` prints the flag list and returns `Ok`.)
    Usage(String),
}

impl CliError {
    /// [`CliError::Msg`] from anything printable: `error: {e}` on stderr.
    pub fn msg(e: impl fmt::Display) -> Self {
        let s = e.to_string();
        debug_assert!(
            !s.starts_with("error:"),
            "Msg text carries its own prefix: {s:?}"
        );
        CliError::Msg(s)
    }

    /// [`CliError::Verbatim`]: `text` on stderr as written (status 1).
    pub fn verbatim(text: impl Into<String>) -> Self {
        let s = text.into();
        CliError::Verbatim(s)
    }

    /// [`CliError::Usage`]: `text` on stderr as written (status 2).
    pub fn usage(text: impl Into<String>) -> Self {
        let s = text.into();
        debug_assert!(!s.ends_with('\n'), "trailing newline: {s:?}");
        CliError::Usage(s)
    }

    /// The process exit status this error ends the run with: 2 for a usage
    /// error, 1 otherwise.
    #[must_use]
    pub fn exit_code(&self) -> i32 {
        match self {
            CliError::Usage(_) => 2,
            CliError::Ferric(_) | CliError::Msg(_) | CliError::Verbatim(_) => 1,
        }
    }

    /// The exact text written to stderr for this error, trailing newline
    /// included.
    #[must_use]
    pub fn stderr_text(&self) -> String {
        match self {
            CliError::Ferric(e) => format!("error: {e}\n"),
            CliError::Msg(s) => format!("error: {s}\n"),
            CliError::Verbatim(s) | CliError::Usage(s) => format!("{s}\n"),
        }
    }

    /// The message without any leading `error: `: [`Display`](fmt::Display)
    /// with that prefix removed from `Verbatim`/`Usage` text that starts with
    /// it. For a caller that shows the error in its own frame.
    #[must_use]
    pub fn message(&self) -> String {
        let s = self.to_string();
        match s.strip_prefix("error: ") {
            Some(rest) => rest.to_string(),
            None => s,
        }
    }

    /// Write [`CliError::stderr_text`] to stderr.
    pub fn eprint(&self) {
        eprint!("{}", self.stderr_text());
    }
}

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CliError::Ferric(e) => fmt::Display::fmt(e, f),
            CliError::Msg(s) | CliError::Verbatim(s) | CliError::Usage(s) => f.write_str(s),
        }
    }
}

impl std::error::Error for CliError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            CliError::Ferric(e) => Some(e),
            CliError::Msg(_) | CliError::Verbatim(_) | CliError::Usage(_) => None,
        }
    }
}

impl From<ferric_core::FerricError> for CliError {
    fn from(e: ferric_core::FerricError) -> Self {
        CliError::Ferric(e)
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
                CliError::verbatim("config error: x"),
                "config error: x\n",
                1,
            ),
            (CliError::usage("usage: ferric"), "usage: ferric\n", 2),
        ];
        for (e, text, code) in cases {
            assert_eq!(e.stderr_text(), text);
            assert_eq!(e.exit_code(), code);
        }
    }

    #[test]
    fn message_has_no_error_prefix() {
        assert_eq!(CliError::msg("bad key").message(), "bad key");
        assert_eq!(CliError::msg("bad key").to_string(), "bad key");
        let flag = CliError::usage("error: --json requires a path");
        assert_eq!(flag.to_string(), "error: --json requires a path");
        assert_eq!(flag.message(), "--json requires a path");
        assert_eq!(
            CliError::verbatim("config error: x").message(),
            "config error: x"
        );
    }
}
