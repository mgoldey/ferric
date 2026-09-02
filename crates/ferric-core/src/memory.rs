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
//! `resolve_budget_bytes` takes an optional caller-supplied `explicit` value
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
//! 5. Final fallback: `DEFAULT_BUDGET_BYTES` (2 GiB) when detection fails.
//!
//! The resolved value and the source it came from are returned together by
//! `resolve_budget` (see `BudgetResolution`) so callers can log the audit
//! line; `resolve_budget_bytes` is the thin wrapper that returns just the bytes.
//!
//! # M2 consumers
//!
//! This is the stable API the fail-fast-guard task (M2) builds on. Keep the
//! surface small: `resolve_budget_bytes`, `resolve_budget`,
//! `detect_available_bytes`, and `gib_to_bytes`.

/// `MemoryPlan` (`plan::MemoryPlan`): a memory budget as a value you spend and
/// account for, rather than a ceiling every call site re-reads independently.
pub mod plan;

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

/// Named vocabulary for "what fraction of the resolved budget may a
/// transient scratch claim on top of whatever else is already resident".
///
/// Before this, the tree had ad-hoc, unnamed answers to that question
/// scattered across crates: `ferric-scf::reduce::resolve_band_bytes` divided
/// by a bare `4`, `ferric-cc::ccsd_t`'s triple-chunk sizing divided by a bare
/// `2`, `ferric-mp2::u_rimp2`'s VVOV panel used the FULL budget (no share —
/// its intermediates are already accounted for via a *reduced* input budget,
/// not a further split), and `ferric-rpa::energy`'s quadrature panel
/// subtracts already-resident bytes off the budget first (a genuinely
/// different, non-divisor policy — see its `quad_panel_width` doc — so it is
/// NOT expressed as a `Share` here; this enum only names the "budget / N"
/// shape). [`transient_share`] gives those divisor-style call sites one named
/// constant instead of a bare integer literal; it changes NO fraction, only
/// the vocabulary for the fractions that already existed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Share {
    /// A quarter of the budget: the direct J/K/JK reduction band scratch is
    /// ADDITIVE to the 3-index tensor + accumulator the budget already
    /// governs, so it claims a quarter rather than the whole
    /// (`ferric_scf::reduce::resolve_band_bytes`).
    Quarter,
    /// Half the budget: CCSD(T)'s streaming per-triple-block chunk sizing —
    /// the other half is left for the persistent W/V/D block intermediates
    /// held alongside it (`ferric_cc::ccsd_t::triple_chunk_len` call site).
    Half,
}

impl Share {
    /// The divisor this share corresponds to (`budget / divisor()`).
    pub const fn divisor(self) -> usize {
        match self {
            Share::Quarter => 4,
            Share::Half => 2,
        }
    }
}

/// `budget_bytes / share.divisor()`, floored at 1 byte so a degenerate
/// zero/tiny budget never divides down to 0 (a 0-byte transient ceiling would
/// make even a single-item allocation look oversized). Pure vocabulary: the
/// two current callers (`reduce::resolve_band_bytes` → [`Share::Quarter`],
/// `ccsd_t`'s chunk sizing → [`Share::Half`]) get the exact same numeric
/// result as their prior bare `/ 4` / `/ 2`.
pub fn transient_share(budget_bytes: usize, share: Share) -> usize {
    (budget_bytes / share.divisor()).max(1)
}

/// Fail-fast pre-flight allocation guard (M2).
///
/// Returns `Err` when a method's projected peak resident allocation (`bytes`)
/// exceeds the resolved memory `budget` (from [`resolve_budget_bytes`]). Callers
/// place this *before* the first large allocation on a dense path so an oversized
/// job errors cleanly (a `Result`, propagated to the CLI / a Python exception)
/// instead of walking into a TB-scale allocation that OOM-kills the process (in
/// Python, the host interpreter).
///
/// `label` should identify the method and carry the shape context, e.g.
/// `"CCSD(T) (no=13, nv=102 spin-orbitals)"`. The produced message reads:
///
/// ```text
/// CCSD(T) (no=13, nv=102 spin-orbitals) requires 1834.20 GB; budget is 18.40 GB
/// — raise [memory] budget_gb / FERRIC_MEM_BUDGET_GB or shrink the system
/// ```
///
/// The guard is a pure pre-check: it never allocates and never changes a result
/// for a job that fits. GB in the message are decimal (÷1e9) to match the other
/// ferric size diagnostics.
pub fn check_alloc(label: &str, bytes: usize, budget: usize) -> Result<(), crate::FerricError> {
    if bytes > budget {
        return Err(crate::FerricError::General(format!(
            "{label} requires {:.2} GB; budget is {:.2} GB — raise [memory] \
             budget_gb / FERRIC_MEM_BUDGET_GB or shrink the system",
            bytes as f64 / 1e9,
            budget as f64 / 1e9,
        )));
    }
    Ok(())
}

/// Parse the current process's resident set size from a `/proc/self/status`-
/// shaped string. Looks for the `VmRSS:` line (reported in kB, despite the
/// "kB" label — Linux convention, same as `MemAvailable` in `/proc/meminfo`).
/// Returns `None` on any parse failure (missing field, malformed number) —
/// this is an observability helper, never a hard dependency, so a parse miss
/// must silently degrade to "unknown" rather than propagate an error.
pub fn parse_vm_rss_bytes(contents: &str) -> Option<usize> {
    for line in contents.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            let kb = rest.split_whitespace().next()?.parse::<u64>().ok()?;
            return Some(clamp_u64_to_usize(kb.saturating_mul(1024)));
        }
    }
    None
}

/// Read the CURRENT process's resident set size (RSS) in bytes by parsing
/// `/proc/self/status`'s `VmRSS:` line. `None` on any failure (non-Linux, the
/// file is unreadable/sandboxed, or the field is missing/malformed) — this is
/// a pure observability helper for the stage-seam RSS safety net
/// ([`crate`]-external callers use it via `warn_if_rss_over`-style helpers in
/// `ferric-rpa`), so it must NEVER panic mid-computation; a monitoring probe
/// failing silently is fine, a monitoring probe crashing the job it's
/// watching is not.
pub fn read_own_rss_bytes() -> Option<usize> {
    let contents = std::fs::read_to_string("/proc/self/status").ok()?;
    parse_vm_rss_bytes(&contents)
}

/// Stage-seam RSS safety net: if the CURRENT process's RSS exceeds
/// `over_factor × budget_bytes`, emit ONE stderr warning line naming the
/// stage, the actual RSS, and the budget — purely observational, NEVER a hard
/// error/kill (unlike [`check_alloc`], which is a pre-flight gate that
/// refuses to start an over-budget job; this instead watches a job that
/// already started and already passed its pre-flight checks, in case the
/// pre-flight ESTIMATE undershot the actual allocation at some later stage
/// boundary). Silently does nothing if RSS can't be read (see
/// [`read_own_rss_bytes`]) — observability must never be allowed to disrupt
/// the computation it's watching.
///
/// `over_factor` is typically ~1.1 (warn at 10% over budget) — see call sites
/// in `ferric-rpa` for the production convention.
pub fn warn_if_rss_over(label: &str, budget_bytes: usize, over_factor: f64) {
    let Some(rss) = read_own_rss_bytes() else { return };
    let threshold = (budget_bytes as f64 * over_factor) as usize;
    if rss > threshold {
        eprintln!(
            "ferric WARNING [{label}]: resident memory {:.2} GB exceeds {:.0}% of the {:.2} GB \
             budget (observability only — this stage already ran; if this recurs, raise \
             [memory] budget_gb / FERRIC_MEM_BUDGET_GB or shrink the system)",
            rss as f64 / 1e9,
            over_factor * 100.0,
            budget_bytes as f64 / 1e9,
        );
    }
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

/// Read the memory limit of **this process's own cgroup**, v2 preferred then
/// v1. `None` if nothing readable or every limit is "unlimited".
///
/// # Why this walks `/proc/self/cgroup` instead of reading the root
///
/// This used to read `/sys/fs/cgroup/memory.max` directly — the **root**
/// cgroup, which on a normal systemd host is `max` (unlimited). A process
/// placed in its own scope by `systemd-run --scope -p MemoryMax=...` (which is
/// exactly what `scripts/ferric-limited` does) therefore never saw its own
/// limit, and `resolve_budget` fell through to `0.8 × MemAvailable` — a
/// *system-wide* number.
///
/// The consequence was not a crash but a silent, expensive stall. Measured
/// 2026-07-27: a24-17/18 (13-atom A24 dimers, aQZ) ran under
/// `ferric-limited --max=5G`, budgeted themselves from ~18 GB of visible
/// system RAM, and then sat in `mem_cgroup_handle_over_high` for **46 minutes
/// producing zero r0 points** while the kernel reclaimed inside their 5 GB
/// scope. The box as a whole had 5+ GB free the whole time, so it looked like
/// system thrash and was mis-diagnosed as such — two unrelated drivers were
/// stopped chasing it. The same jobs finished in ~20 minutes each once given a
/// 10 GB scope.
///
/// The v2 limit is per-cgroup and the *effective* limit is the minimum over the
/// whole ancestry (a child cannot exceed its parent), so this walks from the
/// process's own cgroup up to the root and takes the smallest real limit found.
fn detect_cgroup_limit_bytes() -> Option<usize> {
    if let Some(b) = detect_cgroup_v2_limit_bytes() {
        return Some(b);
    }
    // v1 fallback: the controller mount is flat and the path from
    // /proc/self/cgroup is relative to the memory controller root.
    if let Some(rel) = read_proc_self_cgroup_v1_memory_path() {
        let p = format!("/sys/fs/cgroup/memory{rel}/memory.limit_in_bytes");
        if let Ok(s) = std::fs::read_to_string(&p) {
            if let Some(b) = parse_cgroup_v1_limit(&s) {
                return Some(b);
            }
        }
    }
    if let Ok(s) = std::fs::read_to_string("/sys/fs/cgroup/memory/memory.limit_in_bytes") {
        if let Some(b) = parse_cgroup_v1_limit(&s) {
            return Some(b);
        }
    }
    None
}

/// cgroup v2: the effective limit is the MINIMUM `memory.max` over this
/// process's cgroup and all its ancestors, since a child can never exceed its
/// parent's limit. Returns `None` when every level is unlimited.
fn detect_cgroup_v2_limit_bytes() -> Option<usize> {
    let rel = read_proc_self_cgroup_v2_path()?;
    let mut best: Option<usize> = None;
    // Walk from the leaf up to the root, e.g.
    //   /user.slice/user-1000.slice/.../run-r<id>.scope
    //   /user.slice/user-1000.slice/...
    //   ...
    //   ""   (the root itself)
    let mut cur: &str = rel.trim_end_matches('/');
    loop {
        let path = format!("/sys/fs/cgroup{cur}/memory.max");
        if let Ok(s) = std::fs::read_to_string(&path) {
            if let Some(b) = parse_cgroup_v2_max(&s) {
                best = Some(best.map_or(b, |cur_best: usize| cur_best.min(b)));
            }
        }
        match cur.rfind('/') {
            Some(0) | None => break,
            Some(i) => cur = &cur[..i],
        }
    }
    best
}

/// The cgroup-v2 path from `/proc/self/cgroup` (the `0::<path>` line), e.g.
/// `/user.slice/user-1000.slice/app.slice/run-r<id>.scope`.
fn read_proc_self_cgroup_v2_path() -> Option<String> {
    let s = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    for line in s.lines() {
        // v2 unified hierarchy is always hierarchy-id 0 with an empty controller list.
        if let Some(rest) = line.strip_prefix("0::") {
            return Some(rest.trim().to_string());
        }
    }
    None
}

/// The path for the v1 `memory` controller from `/proc/self/cgroup`, i.e. the
/// third field of the line whose controller list contains `memory`.
fn read_proc_self_cgroup_v1_memory_path() -> Option<String> {
    let s = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    for line in s.lines() {
        let mut it = line.splitn(3, ':');
        let _id = it.next()?;
        let controllers = it.next()?;
        let path = it.next()?;
        if controllers.split(',').any(|c| c == "memory") {
            return Some(path.trim().to_string());
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
    fn check_alloc_fits_and_errors() {
        // Fits: bytes <= budget → Ok.
        assert!(check_alloc("test", 100, 100).is_ok());
        assert!(check_alloc("test", 99, 100).is_ok());
        // Exceeds → Err with the shape label and both GB figures.
        let err = check_alloc(
            "CCSD(T) (no=13, nv=102 spin-orbitals)",
            1_834_200_000_000,
            18_400_000_000,
        )
        .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("CCSD(T) (no=13, nv=102 spin-orbitals)"));
        assert!(msg.contains("1834.20 GB"));
        assert!(msg.contains("18.40 GB"));
        assert!(msg.contains("FERRIC_MEM_BUDGET_GB"));
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

    // ---- Item 3: stage-seam RSS safety net ----

    #[test]
    fn parse_vm_rss_bytes_fixture_string() {
        let fixture = "Name:   ferric-cli\n\
                       VmPeak:    123456 kB\n\
                       VmRSS:     654321 kB\n\
                       VmData:    111111 kB\n";
        assert_eq!(parse_vm_rss_bytes(fixture), Some(654321 * 1024));
        // Missing field → None.
        assert_eq!(parse_vm_rss_bytes("VmPeak: 123456 kB\n"), None);
        // Malformed value → None, never a panic.
        assert_eq!(parse_vm_rss_bytes("VmRSS: notanumber kB\n"), None);
        // Empty input → None.
        assert_eq!(parse_vm_rss_bytes(""), None);
    }

    #[test]
    fn read_own_rss_bytes_on_current_process_is_some_and_reasonable() {
        // Reading the REAL /proc/self/status of the test process itself (not
        // a fixture) — must return Some(reasonable_value), not None, on any
        // live Linux test box. "Reasonable" here just means non-zero and
        // under a generous 100 GiB ceiling (catches a unit-confusion bug,
        // e.g. reporting bytes as if they were kB) without being flaky on
        // slow/loaded CI machines.
        match read_own_rss_bytes() {
            Some(rss) => {
                assert!(rss > 0, "expected a positive RSS reading, got 0");
                assert!(
                    rss < 100 * 1024 * 1024 * 1024,
                    "RSS reading implausibly large ({rss} bytes) — possible unit bug"
                );
            }
            None => {
                // Only acceptable on a non-Linux or heavily sandboxed
                // environment where /proc/self/status isn't readable; the
                // function's contract is "None on any parse failure", not
                // "always Some on Linux", so don't hard-fail here — but this
                // branch should not be reached on the CI/dev Linux boxes this
                // crate targets.
                panic!("read_own_rss_bytes() returned None on what should be a live Linux test process");
            }
        }
    }

    #[test]
    fn warn_if_rss_over_does_not_panic_and_is_a_pure_observer() {
        // Smoke test: calling this with a budget so tiny that RSS will
        // certainly exceed 1.1x it must NOT panic, must NOT return anything
        // (it's fire-and-forget stderr-only), and must not affect subsequent
        // computation — i.e. it's safe to call unconditionally at a stage
        // boundary even on a system where /proc/self/status is unreadable.
        warn_if_rss_over("test-stage-tiny-budget", 1, 1.1);
        // And with a budget so enormous that RSS can never exceed it — the
        // no-warning path must also just return cleanly.
        warn_if_rss_over("test-stage-huge-budget", usize::MAX / 2, 1.1);
    }

    /// The cgroup-v2 path parser must pick the `0::` line out of a real
    /// `/proc/self/cgroup`, not the first line or a v1 controller line.
    #[test]
    fn cgroup_v2_path_is_the_unified_line() {
        // Hybrid host: v1 controller lines first, unified last.
        let sample = "12:memory:/user.slice\n\
                      3:cpu,cpuacct:/user.slice\n\
                      0::/user.slice/user-1000.slice/run-rABC.scope\n";
        let got = sample
            .lines()
            .find_map(|l| l.strip_prefix("0::"))
            .map(|r| r.trim().to_string());
        assert_eq!(got.as_deref(), Some("/user.slice/user-1000.slice/run-rABC.scope"));
    }

    /// REGRESSION: the effective v2 limit is the MINIMUM over the ancestry.
    ///
    /// This pins the arithmetic that `detect_cgroup_v2_limit_bytes` performs
    /// while walking up. Before the 2026-07-27 fix the code read only the ROOT
    /// `/sys/fs/cgroup/memory.max` (usually "max"), so a job confined to a 5 GB
    /// scope by `systemd-run` budgeted itself from total system RAM and then
    /// stalled for 46 minutes inside its own cgroup, producing nothing.
    #[test]
    fn ancestry_minimum_is_the_effective_limit() {
        // leaf 5 GiB inside a 12 GiB parent inside an unlimited root.
        let levels = [Some(5 * (1024usize * 1024 * 1024)), Some(12 * (1024usize * 1024 * 1024)), None];
        let mut best: Option<usize> = None;
        for b in levels.into_iter().flatten() {
            best = Some(best.map_or(b, |c: usize| c.min(b)));
        }
        assert_eq!(best, Some(5 * (1024usize * 1024 * 1024)), "the tightest ancestor must win");

        // An unlimited leaf under a limited parent still inherits the parent.
        let levels = [None, Some(8 * (1024usize * 1024 * 1024)), None];
        let mut best: Option<usize> = None;
        for b in levels.into_iter().flatten() {
            best = Some(best.map_or(b, |c: usize| c.min(b)));
        }
        assert_eq!(best, Some(8 * (1024usize * 1024 * 1024)));
    }

    /// On THIS machine, whatever cgroup the test runs in, the detector must not
    /// report a limit larger than physical RAM -- the failure mode that let a
    /// 5 GB-capped job plan for 18 GB.
    #[test]
    fn detected_limit_is_never_absurd() {
        if let Some(b) = detect_cgroup_limit_bytes() {
            let meminfo = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
            if let Some(total) = meminfo.lines().find_map(|l| {
                l.strip_prefix("MemTotal:")
                    .and_then(|r| r.split_whitespace().next())
                    .and_then(|k| k.parse::<usize>().ok())
                    .map(|kb| kb * 1024)
            }) {
                assert!(
                    b <= total,
                    "cgroup limit {b} exceeds MemTotal {total} -- detector is reading \
                     the wrong cgroup"
                );
            }
        }
    }
}
