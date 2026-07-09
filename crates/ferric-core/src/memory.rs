//! Unified memory-budget resolution for ferric.
//!
//! ferric's only budget mechanism is the `ThreeIndexSource` disk-spill: a
//! resident-bytes ceiling on the raw/dressed 3-index `(P|μν)` tensor and the
//! MO transforms / RPA Lanczos panels built from it. Historically that ceiling
//! was driven by TWO independent, mutually-unaware knobs — `FERRIC_OOC_BUDGET_GB`
//! (SCF DF-JK, default 2 GiB) and `FERRIC_ERI3_BUDGET_GB` (RI-MP2 / RPA, default
//! *unlimited*) — and the TOML `[memory]` section reached only the reference-SCF
//! stage. This module centralizes the resolution so every entry point resolves a
//! budget the same way and the answer is auditable.
//!
//! # Resolution precedence
//!
//! [`resolve_budget_bytes`] takes an optional caller-supplied `explicit` value
//! (from TOML `[memory]` / a Python kwarg / a config field) and returns the byte
//! ceiling using this precedence, highest first:
//!
//! 1. `explicit` — the caller passed a concrete budget (config field / kwarg).
//! 2. `FERRIC_MEM_BUDGET_GB` — the new unified env var (GiB).
//! 3. Legacy env vars `FERRIC_OOC_BUDGET_GB` / `FERRIC_ERI3_BUDGET_GB` (GiB),
//!    kept working for back-compat. If both are set, the smaller wins (the more
//!    conservative ceiling).
//! 4. Auto: `0.8 × detect_available_bytes()` — 80% of detected available RAM
//!    (cgroup limit ∧ `/proc/meminfo MemAvailable`), leaving headroom.
//! 5. Final fallback: [`DEFAULT_BUDGET_BYTES`] (2 GiB) when detection fails.
//!
//! The resolved value and the source it came from are returned together by
//! [`resolve_budget`] (see [`BudgetResolution`]) so callers can log the audit
//! line; [`resolve_budget_bytes`] is the thin wrapper that returns just the bytes.
//!
//! # M2 consumers
//!
//! This is the stable API the fail-fast-guard task (M2) builds on. Keep the
//! surface small: [`resolve_budget_bytes`], [`resolve_budget`],
//! [`detect_available_bytes`], and [`gib_to_bytes`].

/// Final fallback budget when nothing else resolves: 2 GiB.
pub const DEFAULT_BUDGET_BYTES: usize = 2 * 1024 * 1024 * 1024;

/// The new unified budget env var (value in GiB, e.g. `FERRIC_MEM_BUDGET_GB=8`).
pub const ENV_UNIFIED: &str = "FERRIC_MEM_BUDGET_GB";

/// Legacy SCF DF-JK budget env var (value in GiB). Kept for back-compat.
pub const ENV_LEGACY_OOC: &str = "FERRIC_OOC_BUDGET_GB";

/// Legacy RI-MP2 / RPA eri3 budget env var (value in GiB). Kept for back-compat.
pub const ENV_LEGACY_ERI3: &str = "FERRIC_ERI3_BUDGET_GB";

/// Fraction of detected available memory used as the auto budget (leaves
/// headroom for the OS page cache, BLAS scratch, and the non-tensor working set).
pub const AUTO_FRACTION: f64 = 0.8;

/// Convert a GiB figure to bytes, saturating (never panics on absurd input).
/// NaN or non-positive input maps to 0 (treated as "unset" by callers); `+inf`
/// saturates to `usize::MAX` (an explicitly unlimited budget).
pub fn gib_to_bytes(gib: f64) -> usize {
    if gib.is_nan() || gib <= 0.0 {
        return 0;
    }
    let bytes = gib * 1024.0 * 1024.0 * 1024.0;
    if bytes >= usize::MAX as f64 {
        usize::MAX
    } else {
        bytes as usize
    }
}

/// Where a resolved budget came from — for the audit log line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetSource {
    /// Caller-supplied explicit budget (TOML `[memory]` / config field / kwarg).
    Explicit,
    /// `FERRIC_MEM_BUDGET_GB`.
    UnifiedEnv,
    /// `FERRIC_OOC_BUDGET_GB` (legacy SCF).
    LegacyOocEnv,
    /// `FERRIC_ERI3_BUDGET_GB` (legacy RI-MP2/RPA).
    LegacyEri3Env,
    /// `0.8 ×` detected available RAM.
    AutoDetected,
    /// 2 GiB final fallback (detection failed, nothing set).
    Fallback,
}

impl BudgetSource {
    /// Human-readable label for logging.
    pub fn label(self) -> &'static str {
        match self {
            BudgetSource::Explicit => "explicit (config/TOML/kwarg)",
            BudgetSource::UnifiedEnv => "FERRIC_MEM_BUDGET_GB",
            BudgetSource::LegacyOocEnv => "FERRIC_OOC_BUDGET_GB (legacy)",
            BudgetSource::LegacyEri3Env => "FERRIC_ERI3_BUDGET_GB (legacy)",
            BudgetSource::AutoDetected => "auto (0.8 × available RAM)",
            BudgetSource::Fallback => "fallback (2 GiB)",
        }
    }
}

/// A resolved budget plus the source it came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BudgetResolution {
    pub bytes: usize,
    pub source: BudgetSource,
}

impl BudgetResolution {
    /// The `source -> value` audit line for debug/trace logging.
    pub fn audit_line(&self) -> String {
        format!(
            "memory budget: {:.2} GiB  [source: {}]",
            self.bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            self.source.label()
        )
    }
}

/// Read a GiB-valued env var; returns `Some(bytes)` only for a finite, positive
/// value. Unset / unparsable / non-positive → `None`.
fn env_gib_bytes(var: &str) -> Option<usize> {
    let raw = std::env::var(var).ok()?;
    let gib = raw.trim().parse::<f64>().ok()?;
    let bytes = gib_to_bytes(gib);
    if bytes == 0 {
        None
    } else {
        Some(bytes)
    }
}

/// Resolve the memory budget in bytes together with its source. See the module
/// docs for the precedence chain.
pub fn resolve_budget(explicit: Option<usize>) -> BudgetResolution {
    // 1. Explicit config / kwarg (0 is treated as "unset" — a real budget is >0).
    if let Some(b) = explicit {
        if b > 0 {
            return BudgetResolution { bytes: b, source: BudgetSource::Explicit };
        }
    }
    // 2. Unified env var.
    if let Some(b) = env_gib_bytes(ENV_UNIFIED) {
        return BudgetResolution { bytes: b, source: BudgetSource::UnifiedEnv };
    }
    // 3. Legacy env vars — if both set, the more conservative (smaller) wins.
    let legacy_ooc = env_gib_bytes(ENV_LEGACY_OOC);
    let legacy_eri3 = env_gib_bytes(ENV_LEGACY_ERI3);
    match (legacy_ooc, legacy_eri3) {
        (Some(o), Some(e)) => {
            let (bytes, source) = if o <= e {
                (o, BudgetSource::LegacyOocEnv)
            } else {
                (e, BudgetSource::LegacyEri3Env)
            };
            return BudgetResolution { bytes, source };
        }
        (Some(o), None) => {
            return BudgetResolution { bytes: o, source: BudgetSource::LegacyOocEnv }
        }
        (None, Some(e)) => {
            return BudgetResolution { bytes: e, source: BudgetSource::LegacyEri3Env }
        }
        (None, None) => {}
    }
    // 4. Auto: 0.8 × detected available RAM.
    if let Some(avail) = detect_available_bytes() {
        let budget = (avail as f64 * AUTO_FRACTION) as usize;
        if budget > 0 {
            return BudgetResolution { bytes: budget, source: BudgetSource::AutoDetected };
        }
    }
    // 5. Final fallback.
    BudgetResolution { bytes: DEFAULT_BUDGET_BYTES, source: BudgetSource::Fallback }
}

/// Resolve the memory budget in bytes. Thin wrapper over [`resolve_budget`] that
/// discards the source. This is the primary M2 entry point.
pub fn resolve_budget_bytes(explicit: Option<usize>) -> usize {
    resolve_budget(explicit).bytes
}

/// Detect available memory in bytes: the minimum of any active cgroup memory
/// limit and `/proc/meminfo`'s `MemAvailable`. Returns `None` if nothing can be
/// read (non-Linux, sandboxed, or missing files) so the caller falls back.
///
/// - cgroup v2: `/sys/fs/cgroup/memory.max` (`"max"` means no limit).
/// - cgroup v1: `/sys/fs/cgroup/memory/memory.limit_in_bytes` (a sentinel near
///   `u64::MAX`/`i64::MAX` means no limit).
/// - `/proc/meminfo` `MemAvailable:` (kB).
pub fn detect_available_bytes() -> Option<usize> {
    let cgroup = detect_cgroup_limit_bytes();
    let meminfo = std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|s| parse_meminfo_available(&s));
    match (cgroup, meminfo) {
        (Some(c), Some(m)) => Some(c.min(m)),
        (Some(c), None) => Some(c),
        (None, Some(m)) => Some(m),
        (None, None) => None,
    }
}

/// Read the active cgroup memory limit (v2 preferred, then v1). `None` if no
/// file readable or the limit is "unlimited".
fn detect_cgroup_limit_bytes() -> Option<usize> {
    if let Ok(s) = std::fs::read_to_string("/sys/fs/cgroup/memory.max") {
        if let Some(b) = parse_cgroup_v2_max(&s) {
            return Some(b);
        }
    }
    if let Ok(s) = std::fs::read_to_string("/sys/fs/cgroup/memory/memory.limit_in_bytes") {
        if let Some(b) = parse_cgroup_v1_limit(&s) {
            return Some(b);
        }
    }
    None
}

/// Parse cgroup v2 `memory.max`. `"max"` → no limit (`None`); otherwise the byte
/// value. A limit of 0 is treated as no useful limit (`None`).
pub fn parse_cgroup_v2_max(contents: &str) -> Option<usize> {
    let t = contents.trim();
    if t.is_empty() || t == "max" {
        return None;
    }
    let v = t.parse::<u64>().ok()?;
    if v == 0 {
        return None;
    }
    Some(clamp_u64_to_usize(v))
}

/// Parse cgroup v1 `memory.limit_in_bytes`. The classic "no limit" sentinel is a
/// value near `i64::MAX`/`u64::MAX` (a page-aligned huge number); treat anything
/// at or above a conservative threshold as unlimited (`None`).
pub fn parse_cgroup_v1_limit(contents: &str) -> Option<usize> {
    let v = contents.trim().parse::<u64>().ok()?;
    // The unlimited sentinel is typically 0x7FFFFFFFFFFFF000 (i64::MAX rounded
    // down to a page) or u64::MAX-page. Anything ≥ 2^60 (1 EiB) is not a real
    // container limit — treat as unlimited.
    const UNLIMITED_THRESHOLD: u64 = 1 << 60;
    if v == 0 || v >= UNLIMITED_THRESHOLD {
        return None;
    }
    Some(clamp_u64_to_usize(v))
}

/// Parse `/proc/meminfo`, returning `MemAvailable` in bytes. The field is
/// reported in kB (kibibytes, despite the "kB" label — Linux convention).
pub fn parse_meminfo_available(contents: &str) -> Option<usize> {
    for line in contents.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            // "MemAvailable:    16321234 kB"
            let kb = rest.split_whitespace().next()?.parse::<u64>().ok()?;
            return Some(clamp_u64_to_usize(kb.saturating_mul(1024)));
        }
    }
    None
}

fn clamp_u64_to_usize(v: u64) -> usize {
    if v > usize::MAX as u64 {
        usize::MAX
    } else {
        v as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // FERRIC_* budget env vars are process-global; the default harness runs
    // tests in parallel. Serialize every test that reads/writes them.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn clear_budget_env() {
        std::env::remove_var(ENV_UNIFIED);
        std::env::remove_var(ENV_LEGACY_OOC);
        std::env::remove_var(ENV_LEGACY_ERI3);
    }

    // ---- parser tests (fixture strings, never the live machine) ----

    #[test]
    fn cgroup_v2_max_parses() {
        assert_eq!(parse_cgroup_v2_max("8589934592\n"), Some(8589934592));
        assert_eq!(parse_cgroup_v2_max("  8589934592  "), Some(8589934592));
        assert_eq!(parse_cgroup_v2_max("max\n"), None);
        assert_eq!(parse_cgroup_v2_max(""), None);
        assert_eq!(parse_cgroup_v2_max("0"), None);
        assert_eq!(parse_cgroup_v2_max("garbage"), None);
    }

    #[test]
    fn cgroup_v1_limit_parses_and_detects_unlimited() {
        assert_eq!(parse_cgroup_v1_limit("4294967296\n"), Some(4294967296));
        // Classic i64::MAX-page-aligned unlimited sentinel.
        assert_eq!(parse_cgroup_v1_limit("9223372036854771712"), None);
        // u64::MAX-ish sentinel.
        assert_eq!(parse_cgroup_v1_limit("18446744073709551615"), None);
        assert_eq!(parse_cgroup_v1_limit("0"), None);
        assert_eq!(parse_cgroup_v1_limit("nan"), None);
    }

    #[test]
    fn meminfo_available_parses() {
        let fixture = "MemTotal:       32000000 kB\n\
                       MemFree:         1000000 kB\n\
                       MemAvailable:   16321234 kB\n\
                       Buffers:          200000 kB\n";
        assert_eq!(parse_meminfo_available(fixture), Some(16321234 * 1024));
        // Missing field → None.
        assert_eq!(parse_meminfo_available("MemTotal: 32000000 kB\n"), None);
        // Malformed value → None.
        assert_eq!(parse_meminfo_available("MemAvailable: notanumber kB\n"), None);
    }

    #[test]
    fn gib_conversion_edge_cases() {
        assert_eq!(gib_to_bytes(2.0), 2 * 1024 * 1024 * 1024);
        assert_eq!(gib_to_bytes(0.0), 0);
        assert_eq!(gib_to_bytes(-1.0), 0);
        assert_eq!(gib_to_bytes(f64::NAN), 0);
        assert_eq!(gib_to_bytes(f64::INFINITY), usize::MAX);
    }

    // ---- precedence tests (serialized; env is process-global) ----

    #[test]
    fn explicit_wins_over_everything() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_budget_env();
        std::env::set_var(ENV_UNIFIED, "16");
        std::env::set_var(ENV_LEGACY_OOC, "4");
        let r = resolve_budget(Some(gib_to_bytes(1.0)));
        assert_eq!(r.source, BudgetSource::Explicit);
        assert_eq!(r.bytes, gib_to_bytes(1.0));
        clear_budget_env();
    }

    #[test]
    fn explicit_zero_is_ignored() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_budget_env();
        std::env::set_var(ENV_UNIFIED, "16");
        // explicit=Some(0) is "unset" — must fall through to the unified env.
        let r = resolve_budget(Some(0));
        assert_eq!(r.source, BudgetSource::UnifiedEnv);
        assert_eq!(r.bytes, gib_to_bytes(16.0));
        clear_budget_env();
    }

    #[test]
    fn unified_env_wins_over_legacy() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_budget_env();
        std::env::set_var(ENV_UNIFIED, "16");
        std::env::set_var(ENV_LEGACY_OOC, "4");
        std::env::set_var(ENV_LEGACY_ERI3, "32");
        let r = resolve_budget(None);
        assert_eq!(r.source, BudgetSource::UnifiedEnv);
        assert_eq!(r.bytes, gib_to_bytes(16.0));
        clear_budget_env();
    }

    #[test]
    fn legacy_conservative_wins_when_both_set() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_budget_env();
        std::env::set_var(ENV_LEGACY_OOC, "8");
        std::env::set_var(ENV_LEGACY_ERI3, "3");
        let r = resolve_budget(None);
        // Smaller (3 GiB, eri3) wins.
        assert_eq!(r.source, BudgetSource::LegacyEri3Env);
        assert_eq!(r.bytes, gib_to_bytes(3.0));
        clear_budget_env();

        // Flip: OOC smaller.
        std::env::set_var(ENV_LEGACY_OOC, "2");
        std::env::set_var(ENV_LEGACY_ERI3, "9");
        let r = resolve_budget(None);
        assert_eq!(r.source, BudgetSource::LegacyOocEnv);
        assert_eq!(r.bytes, gib_to_bytes(2.0));
        clear_budget_env();
    }

    #[test]
    fn single_legacy_var_used() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_budget_env();
        std::env::set_var(ENV_LEGACY_ERI3, "5");
        let r = resolve_budget(None);
        assert_eq!(r.source, BudgetSource::LegacyEri3Env);
        assert_eq!(r.bytes, gib_to_bytes(5.0));
        clear_budget_env();
    }

    #[test]
    fn auto_or_fallback_when_nothing_set() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_budget_env();
        let r = resolve_budget(None);
        // On a live Linux box this is AutoDetected; in a sandbox with no
        // /proc/meminfo it is Fallback. Either is valid — just never an env
        // source, and always a positive budget.
        assert!(matches!(
            r.source,
            BudgetSource::AutoDetected | BudgetSource::Fallback
        ));
        assert!(r.bytes > 0);
        clear_budget_env();
    }
}
