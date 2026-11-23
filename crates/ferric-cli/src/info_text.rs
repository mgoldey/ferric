//! Writing the `--version` (stdout) and `--help` (stderr) text without
//! panicking.
//!
//! `print!` / `eprintln!` panic when the stream cannot be written (a full
//! disk behind `> /dev/full`, an I/O error on a redirected file). These two
//! informational outputs go through [`write_info`] instead, which writes the
//! same bytes and turns a failure into a return value:
//!
//! - **Broken pipe** (`EPIPE`, the reader went away, e.g. `ferric --version |
//!   true`): success, status 0. Rust ignores `SIGPIPE`, so the write returns
//!   the error instead of the process dying of the signal as a C tool would;
//!   the reader asked for no more output, so this is not an error of the
//!   run, and nothing more is written.
//! - **Any other write error**: [`CliError::Msg`] (`error: cannot write the
//!   --version text to stdout: ...`, status 1). The run asked for the text
//!   and it was not delivered. When the failing stream is stderr itself
//!   (`--help 2>/dev/full`), that message cannot be shown either;
//!   [`CliError::eprint`] drops it rather than panic, and the status
//!   still says the run failed.
//!
//! A closed descriptor (`ferric --version >&-`) is not an error at all: the
//! Rust runtime reopens closed standard descriptors on `/dev/null` at start.

use std::io::{self, Write};

use crate::error::CliError;

/// Which output [`write_info`] is writing, for its error message.
#[derive(Clone, Copy, Debug)]
pub(crate) enum InfoText {
    /// `--version` / `-V`, on stdout.
    Version,
    /// `--help` / `-h`, on stderr.
    Help,
}

impl InfoText {
    fn describe(self) -> &'static str {
        match self {
            InfoText::Version => "the --version text to stdout",
            InfoText::Help => "the --help text to stderr",
        }
    }
}

/// Write `text` to `out` exactly as given and flush it. See the module
/// documentation for how errors map to the result.
pub(crate) fn write_info(mut out: impl Write, which: InfoText, text: &str) -> Result<(), CliError> {
    match write!(out, "{text}").and_then(|()| out.flush()) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == io::ErrorKind::BrokenPipe => Ok(()),
        Err(e) => Err(CliError::msg(format!(
            "cannot write {}: {e}",
            which.describe()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Accepts `ok` bytes, then fails every write (and flush) with `kind`.
    struct Failing {
        ok: usize,
        kind: io::ErrorKind,
        got: Vec<u8>,
    }

    impl Write for Failing {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            if self.ok == 0 {
                return Err(io::Error::from(self.kind));
            }
            let n = buf.len().min(self.ok);
            self.ok -= n;
            self.got.extend_from_slice(&buf[..n]);
            Ok(n)
        }
        fn flush(&mut self) -> io::Result<()> {
            if self.ok == 0 {
                Err(io::Error::from(self.kind))
            } else {
                Ok(())
            }
        }
    }

    #[test]
    fn writes_the_text_unchanged() {
        let mut buf = Vec::new();
        let text = crate::version_text();
        write_info(&mut buf, InfoText::Version, &text).unwrap();
        assert_eq!(buf, text.as_bytes());
    }

    #[test]
    fn a_broken_pipe_is_success() {
        for ok in [0, 3] {
            let mut w = Failing {
                ok,
                kind: io::ErrorKind::BrokenPipe,
                got: Vec::new(),
            };
            assert!(write_info(&mut w, InfoText::Help, "usage: ferric\n").is_ok());
            assert_eq!(w.got, &b"usage: ferric\n"[..ok]);
        }
    }

    #[test]
    fn another_write_error_is_a_status_1_error() {
        let mut w = Failing {
            ok: 0,
            kind: io::ErrorKind::StorageFull,
            got: Vec::new(),
        };
        let e = write_info(&mut w, InfoText::Version, "ferric 0\n").unwrap_err();
        assert!(matches!(e, CliError::Msg(_)), "{e:?}");
        assert_eq!(e.exit_code(), 1);
        let text = e.stderr_text();
        assert!(
            text.starts_with("error: cannot write the --version text to stdout: "),
            "{text}"
        );
        let e = write_info(&mut w, InfoText::Help, "usage\n").unwrap_err();
        assert!(
            e.stderr_text()
                .starts_with("error: cannot write the --help text to stderr: "),
            "{e:?}"
        );
    }
}
