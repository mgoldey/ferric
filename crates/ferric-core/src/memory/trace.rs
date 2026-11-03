//! Lightweight, opt-in memory measurement helpers for ferric.
//!
//! # Why this exists
//!
//! Ferric's DFT/MP2 memory optimization work needs a cheap, process-local way
//! to observe resident memory while tuning. This module provides that
//! observation without coupling to the `MemoryPlan` budget machinery: it is
//! purely informational, never affects numerical results, and degrades
//! silently on non-Linux or sandboxed systems.

/// Read the current process's resident set size in bytes.
///
/// Delegates to [`crate::memory::read_own_rss_bytes`] (`/proc/self/status`,
/// VmRSS) so there is a single canonical RSS reader.
pub fn current_rss_bytes() -> Option<usize> {
    crate::memory::read_own_rss_bytes()
}

/// Whether lightweight memory tracing is enabled via the `FERRIC_MEM_TRACE`
/// environment variable.
///
/// "Enabled" means the variable is present *and* non-empty.
pub fn trace_enabled() -> bool {
    std::env::var("FERRIC_MEM_TRACE")
        .map(|v| !v.is_empty())
        .unwrap_or(false)
}
