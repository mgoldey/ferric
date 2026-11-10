//! Operating-system calls of the 3-index spill file, behind one portable face.
//!
//! The spill path was written against Linux: `posix_fadvise` page-cache hints,
//! an `O_DIRECT` re-open through `/proc/self/fd`, positional `pread`s and
//! `statvfs`. Each is reached here through a function whose Linux body is the
//! original call, unchanged, so Linux behaviour is identical. Elsewhere:
//!
//! | call | Linux | macOS (other unix) | Windows |
//! |------|-------|--------------------|---------|
//! | [`fadvise`] | `posix_fadvise` | no-op (Apple has no `posix_fadvise`) | no-op |
//! | [`open_direct_handle`] | `O_DIRECT` via `/proc/self/fd` | `None` (buffered reads) | `None` |
//! | [`read_at`] / [`read_exact_at`] | `pread` (`FileExt`) | `pread` | `seek_read` |
//! | [`free_bytes_at`] | `statvfs` | `statvfs` | `None` ("could not measure") |
//!
//! The page-cache hints and direct IO are optimisations (see
//! `drop_page_cache` and `open_direct` in the parent module), so their absence
//! changes speed and page-cache footprint, never results. A `None` from
//! [`free_bytes_at`] is the documented "could not measure" outcome: the spill
//! preflight warns and proceeds unchecked.

use std::fs::File;

/// Page-cache advice for [`fadvise`].
#[derive(Clone, Copy, Debug)]
pub(super) enum Advice {
    /// The file will be read front to back (`POSIX_FADV_SEQUENTIAL`).
    Sequential,
    /// The range will not be read again; drop its clean pages (`POSIX_FADV_DONTNEED`).
    DontNeed,
}

/// Advisory page-cache hint on `[off, off + len)` (`len == 0`: to the end of
/// the file). Errors are ignored: the hint cannot change data. A no-op off
/// Linux.
pub(super) fn fadvise(file: &File, off: u64, len: u64, advice: Advice) {
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::io::AsRawFd;
        let advice = match advice {
            Advice::Sequential => libc::POSIX_FADV_SEQUENTIAL,
            Advice::DontNeed => libc::POSIX_FADV_DONTNEED,
        };
        // SAFETY: the descriptor comes from a live `File` borrowed for the
        // whole call. posix_fadvise is advisory and cannot corrupt data.
        unsafe {
            libc::posix_fadvise(
                file.as_raw_fd(),
                off as libc::off_t,
                len as libc::off_t,
                advice,
            );
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (file, off, len, advice);
    }
}

/// A second, read-only `O_DIRECT` handle on the same open file, or `None`
/// when the filesystem refuses direct IO. Linux only: it re-opens through
/// `/proc/self/fd` (the spill file is already unlinked) and `O_DIRECT` is a
/// Linux flag. Elsewhere always `None`, which selects the buffered reads.
pub(super) fn open_direct_handle(file: &File) -> Option<File> {
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::fs::OpenOptionsExt;
        use std::os::unix::io::AsRawFd;
        std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECT)
            .open(format!("/proc/self/fd/{}", file.as_raw_fd()))
            .ok()
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = file;
        None
    }
}

/// One positional read at byte `off`; returns the bytes read (short at the end
/// of the file). On Windows `seek_read` also moves the handle's cursor; the
/// spill code never relies on that cursor after a positional read.
pub(super) fn read_at(file: &File, buf: &mut [u8], off: u64) -> std::io::Result<usize> {
    #[cfg(unix)]
    {
        std::os::unix::fs::FileExt::read_at(file, buf, off)
    }
    #[cfg(windows)]
    {
        std::os::windows::fs::FileExt::seek_read(file, buf, off)
    }
}

/// Fill `buf` from byte `off`, or fail with `UnexpectedEof`.
pub(super) fn read_exact_at(file: &File, buf: &mut [u8], off: u64) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        std::os::unix::fs::FileExt::read_exact_at(file, buf, off)
    }
    #[cfg(windows)]
    {
        // Mirrors std's unix `read_exact_at` (Windows has no such method).
        let (mut buf, mut off) = (buf, off);
        while !buf.is_empty() {
            match read_at(file, buf, off) {
                Ok(0) => {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        "failed to fill whole buffer",
                    ))
                }
                Ok(n) => {
                    buf = &mut buf[n..];
                    off += n as u64;
                }
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }
}

/// Free bytes available to an unprivileged process on the filesystem holding
/// `path`, or `None` if the path cannot be statted.
///
/// `f_bavail` (not `f_bfree`): the reserved-blocks pool that `f_bfree` includes
/// is unavailable to a normal ferric process, so `f_bfree` would over-promise by
/// the usual 5% root reservation.
///
/// `None` on failure is deliberate. A missing/unstattable spill directory means
/// the probe could not measure, not that the disk is full: fabricating `0` there
/// would refuse every spill on any system whose `statvfs` we cannot read, which
/// is a worse failure than the one being fixed. The caller degrades to the old
/// (unchecked) behaviour in that case — but still warns. Windows has no probe
/// yet (`GetDiskFreeSpaceExW` would be one) and always takes that branch.
#[cfg(unix)]
pub(super) fn free_bytes_at(path: &std::path::Path) -> Option<u64> {
    use std::os::unix::ffi::OsStrExt;
    let c_path = std::ffi::CString::new(path.as_os_str().as_bytes()).ok()?;
    // SAFETY: `c_path` is a valid NUL-terminated C string that outlives the
    // call, and `buf` is a correctly-sized, properly-aligned `statvfs` we hand
    // over exclusively for the duration of the call. `statvfs` only writes into
    // `buf` and returns 0/-1.
    let mut buf = std::mem::MaybeUninit::<libc::statvfs>::uninit();
    let rc = unsafe { libc::statvfs(c_path.as_ptr(), buf.as_mut_ptr()) };
    if rc != 0 {
        return None;
    }
    // SAFETY: statvfs returned 0, so `buf` is fully initialized.
    let st = unsafe { buf.assume_init() };
    // f_frsize is the fragment size the f_b* counts are expressed in.
    //
    // The widening goes through `try_from` because libc's statvfs field types
    // are PLATFORM-DEPENDENT (`u64` on linux-gnu, `u32` on some 32-bit targets
    // and on macOS's `f_bavail`, `c_ulong` in general). `as` would be a
    // same-type cast here and a lossy one elsewhere; `u64::from` is not
    // implemented for every candidate type. This form compiles unchanged on all
    // of them, and the `ok()?` arm is genuinely unreachable on any target where
    // the field is unsigned and ≤64 bits — which is all of them — so it
    // degrades to "could not measure", never to a fabricated figure.
    //
    // clippy's `useless_conversion` fires because on THIS target (linux-gnu)
    // the fields already are `u64`, making the conversion an identity. Allowed
    // rather than removed: dropping it would silently truncate on any target
    // where the field is narrower, and re-adding it under a `cfg` for a
    // one-line widening is worse than the lint. On macOS `f_bavail` is `u32`,
    // where clippy instead flags the conversion as infallible
    // (`unnecessary_fallible_conversions`); same reasoning.
    #[allow(clippy::useless_conversion, clippy::unnecessary_fallible_conversions)]
    let frsize = u64::try_from(st.f_frsize).ok()?;
    #[allow(clippy::useless_conversion, clippy::unnecessary_fallible_conversions)]
    let bavail = u64::try_from(st.f_bavail).ok()?;
    frsize.checked_mul(bavail)
}

/// See the unix variant: no free-space probe on this platform yet, so the
/// spill preflight takes its documented "could not measure" branch.
#[cfg(not(unix))]
pub(super) fn free_bytes_at(path: &std::path::Path) -> Option<u64> {
    let _ = path;
    None
}
