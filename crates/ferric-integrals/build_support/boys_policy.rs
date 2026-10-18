// Decision logic for the libint2 `boys.h` override (shim/libint2_overrides/).
//
// Pure: no I/O, no environment reads. `build.rs` `include!`s this file and
// feeds it the bytes of `<libint_root>/include/libint2/boys.h` and the three
// environment switches; `tests/boys_override_policy.rs` includes the same
// file to test every branch without running a build.

/// FNV-1a 64 of each STOCK libint2 `include/libint2/boys.h` for which
/// `shim/libint2_overrides/<version>/libint2/boys.h` holds a patched copy.
/// - 2.7.2: mpqc4 tarball `libint-2.7.2-mpqc4.tgz` and the wheels' export
///   `libint-2.7.2-ferric-small-1.tgz`, whose `boys.h` is byte-identical
///   (sha256 880592b9026352c871c96877a43788edd22edb0b1057ab5a73657c40b3a61505).
/// - 2.13.1: conda-forge `libint-2.13.1-h83f5b4b_0.conda` as installed by
///   scripts/install-libint.sh
///   (sha256 890ab0261a2be2517d3d4cedfd7d07e2a1c6be7b6f057dc7ea2ea418e17067c1).
pub const STOCK_BOYS_H_FNV1A64: [(&str, u64); 2] = [
    ("2.7.2", 0x1d51_ac52_f207_6ad8),
    ("2.13.1", 0x0f6f_0c23_194e_18ad),
];

/// Environment variable that turns "no override applied" into a build error.
/// CI, the wheel builds and the Dockerfile set it: each installs a
/// sha256-pinned stock libint2, so an unknown `boys.h` there means the pin or
/// a cache moved and the fixes were silently lost.
pub const REQUIRE_VAR: &str = "FERRIC_REQUIRE_PATCHED_BOYS";
/// Explicit opt-out of `REQUIRE_VAR`: build without the fixes, with a warning.
pub const ALLOW_VAR: &str = "FERRIC_ALLOW_UNPATCHED_BOYS";
/// Forces the stock header (A/B timing runs).
pub const STOCK_VAR: &str = "FERRIC_LIBINT2_STOCK_BOYS";

/// What the override fixes; named in every message that says it is missing.
pub const FIXES: &str = "libint2 GmEval copy fix, Boys T_crit table-edge fix (#433) and Yukawa/Slater table-edge and overflow fixes (#436, #439, #443)";

pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// The three switches, already parsed.
#[derive(Clone, Copy, Debug, Default)]
pub struct BoysEnv {
    /// `FERRIC_LIBINT2_STOCK_BOYS` is set.
    pub force_stock: bool,
    /// `FERRIC_REQUIRE_PATCHED_BOYS` is truthy.
    pub require: bool,
    /// `FERRIC_ALLOW_UNPATCHED_BOYS` is truthy.
    pub allow_unpatched: bool,
}

/// `1`, `true`, `yes`, `on` (any case) are truthy; unset, empty, `0` and
/// anything else are not, so `FERRIC_REQUIRE_PATCHED_BOYS=0` disables it.
pub fn env_flag(value: Option<&str>) -> bool {
    matches!(
        value.map(|v| v.trim().to_ascii_lowercase()).as_deref(),
        Some("1" | "true" | "yes" | "on")
    )
}

#[derive(Debug, PartialEq, Eq)]
pub enum BoysOutcome {
    /// Put `shim/libint2_overrides/<version>` first on the include path.
    Override(&'static str),
    /// Build against the prefix's own `boys.h`; print this as a cargo warning.
    Unpatched(String),
    /// Fail the build with this message.
    Refuse(String),
}

fn known_hashes() -> String {
    STOCK_BOYS_H_FNV1A64
        .iter()
        .map(|(v, h)| format!("{v} = {h:#018x}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Decide the override for the `boys.h` at `path`, whose bytes are
/// `contents` (`Err` carries the read error).
pub fn boys_outcome(path: &str, contents: Result<&[u8], String>, env: BoysEnv) -> BoysOutcome {
    let reason = if env.force_stock {
        format!("{STOCK_VAR} is set")
    } else {
        match contents {
            Err(e) => format!("cannot read {path} ({e})"),
            Ok(bytes) => {
                let hash = fnv1a64(bytes);
                if let Some((version, _)) =
                    STOCK_BOYS_H_FNV1A64.iter().find(|(_, h)| *h == hash)
                {
                    return BoysOutcome::Override(version);
                }
                format!(
                    "{path} (FNV-1a 64 {hash:#018x}) is not a stock libint2 boys.h ferric has an override for (known stock hashes: {})",
                    known_hashes()
                )
            }
        }
    };
    if env.require && !env.allow_unpatched {
        return BoysOutcome::Refuse(format!(
            "{reason}, so the {FIXES} would NOT be applied, and {REQUIRE_VAR} is set. \
             Point LIBINT2_PREFIX at a stock libint2 2.7.2 or 2.13.1 (scripts/install-libint.sh), \
             or set {ALLOW_VAR}=1 to build without the fixes \
             (see crates/ferric-integrals/shim/libint2_overrides/README.md)."
        ));
    }
    BoysOutcome::Unpatched(format!(
        "{reason}; {FIXES} NOT applied (erf/erfc integrals scale poorly across threads; a Boys argument of exactly 117.0 MAY read uninitialised memory if this libint2 still has the stock table-edge defect, issue #433; and Yukawa/Slater-geminal integrals with T == 1024 or U == 1000 exactly MAY read out of range, and with U above ~700 (2.7.2) or U + 2 sqrt(U T) above ~709 with T above 1024 (2.7.2 and 2.13.1) return NaN, if this libint2 still has the stock TennoGmEval defects, issues #436 and #439)"
    ))
}
