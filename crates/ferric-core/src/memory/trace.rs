//! Lightweight, opt-in memory measurement helpers for ferric.
//!
//! # Why this exists
//!
//! Ferric's DFT memory optimization work needs a cheap, process-local way to
//! observe resident memory while tuning grid batching, AO masking, and
//! pruning. This module provides that observation without coupling to the
//! `MemoryPlan` budget machinery: it is purely informational, never affects
//! numerical results, and degrades silently on non-Linux or sandboxed systems.
//!
//! No `libc` (or other external) dependency is required: the Linux path uses a
//! fixed 4 KiB page size constant, and the non-Linux path is a no-op returning
//! `None`.

/// Read the current process's resident set size in bytes.
///
/// Linux-only: reads `/proc/self/statm` and multiplies the resident page count
/// by a fixed 4 KiB page size. No `libc` (or any other external crate) is
/// required; the page size is a compile-time constant, which keeps this module
/// dependency-free and lets it compile unchanged on non-Linux targets. Returns
/// `None` on any other platform or on read/parse failure.
/// Read the current process's resident set size in bytes.
///
/// Delegates to [`crate::memory::read_own_rss_bytes`] so there is a single
/// canonical RSS reader (from `/proc/self/status`, VmRSS). Returns `None` on
/// non-Linux or on read/parse failure.
pub fn current_rss_bytes() -> Option<usize> {
    crate::memory::read_own_rss_bytes()
}

/// Whether lightweight memory tracing is enabled via the `FERRIC_MEM_TRACE`
/// environment variable.
///
/// "Enabled" means the variable is present *and* non-empty, so an explicit
/// `FERRIC_MEM_TRACE=` does not turn tracing on.
pub fn trace_enabled() -> bool {
    std::env::var("FERRIC_MEM_TRACE")
        .map(|v| !v.is_empty())
        .unwrap_or(false)
}
