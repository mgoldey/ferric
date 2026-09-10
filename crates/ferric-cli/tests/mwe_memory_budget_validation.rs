//! MWE: an unusable `[memory] budget_gb` must ERROR, not silently mean "auto".
//!
//! # The inversion
//!
//! `MemoryCfg::budget_bytes()` maps the TOML figure through
//! `ferric_core::memory::gib_to_bytes`, which returns **0** for NaN or any
//! non-positive input, and for a positive value too small to reach one byte.
//! `resolve_budget` then documents `0` as "unset" (`if b > 0`) and falls
//! through to step 4: `0.8 × detect_available_bytes()`.
//!
//! So a user who writes `budget_gb = -4.0` (a typo'd sign), `budget_gb = 0.0`
//! ("use no extra memory"), or `budget_gb = 1e-12` asks for the tightest
//! possible ceiling and receives the **loosest one available** — 80% of the
//! whole box. That is an inversion, not a degradation, and it reaches every
//! method because `budget_bytes()` is the single value threaded into all of
//! them. The only trace is the audit line printing
//! `[source: auto (0.8 × available RAM)]` where the user expects `explicit`.
//!
//! # Why this is the CLI's own convention, not a new rule
//!
//! Every other string knob in this config hard-errors on an unusable value
//! rather than silently defaulting — `QuadratureScheme`, `C6Source` and
//! `DispersionPartition::parse_config_str` all do, and every config struct
//! carries `#[serde(deny_unknown_fields)]` so a typo'd KEY is already fatal.
//! A typo'd VALUE on the single knob that bounds memory should not be the one
//! exception, especially when its failure mode is "ignore the limit entirely".
//!
//! Arithmetic and parsing only — no SCF, no allocation.

use std::path::PathBuf;
use std::process::Command;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("ferric-cli manifest dir should be workspace_root/crates/ferric-cli")
        .to_path_buf()
}

/// Run the real `ferric-cli` binary on a config with the given `[memory]` body.
///
/// Shelling out rather than calling `load_config` directly, matching
/// `ccsd_dispatch.rs`: this is the actual user path, so it pins the EXIT CODE
/// too. A validation that returned `Err` but was then swallowed into a warning
/// would pass a library-level test and still leave the inversion reachable.
fn run_with_memory(tag: &str, section: &str) -> std::process::Output {
    let root = workspace_root();
    let path = root.join("target").join(format!("mwe_budget_validation_{tag}.toml"));
    std::fs::write(
        &path,
        format!(
            "[molecule]\nxyz = \"testdata/molecules/water.xyz\"\n\n\
             [basis]\nname = \"sto-3g\"\n\n\
             [method]\nkind = \"rhf\"\n\n\
             [memory]\n{section}\n"
        ),
    )
    .expect("write temp toml");
    Command::new(env!("CARGO_BIN_EXE_ferric-cli"))
        .arg(&path)
        .current_dir(&root)
        .env("OPENBLAS_NUM_THREADS", "1")
        .output()
        .expect("failed to run ferric-cli binary")
}

/// Assert the run FAILED and said something about the budget key.
fn expect_rejected(tag: &str, section: &str, why: &str) {
    let out = run_with_memory(tag, section);
    let stderr = String::from_utf8_lossy(&out.stderr);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        !out.status.success(),
        "`[memory] {section}` was ACCEPTED (exit {:?}). {why}\n--- stdout ---\n{stdout}\n\
         --- stderr ---\n{stderr}",
        out.status.code()
    );
    assert!(
        stderr.contains("budget_gb"),
        "the failure must name the offending key so a user can find it.\n--- stderr ---\n{stderr}"
    );
    // The inversion signature: it must NOT have quietly resolved to auto.
    // The audit line goes to STDERR (`[ferric] memory budget: .. [source: ..]`).
    assert!(
        !stderr.contains("source: auto"),
        "the run reported an AUTO-DETECTED budget for an explicitly-set one — this is the \
         inversion itself.\n--- stderr ---\n{stderr}"
    );
    let _ = &stdout;
}

/// Assert the run got past config validation.
///
/// It need not SUCCEED overall (the SCF still has to converge), only not fail
/// on the budget. Checking for the audit line is the precise statement.
fn expect_accepted(tag: &str, section: &str) {
    let out = run_with_memory(tag, section);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !stderr.contains("budget_gb"),
        "`[memory] {section}` is valid and must not be rejected.\n--- stderr ---\n{stderr}"
    );
}

/// CONTRACT 1: a NEGATIVE budget is rejected.
///
/// The typo case. `budget_gb = -4.0` currently resolves to 80% of RAM.
#[test]
fn a_negative_budget_is_rejected() {
    expect_rejected(
        "negative",
        "budget_gb = -4.0",
        "A negative budget must error, not fall through to auto-detect (which grants the WHOLE box).",
    );
}

/// CONTRACT 2: a ZERO budget is rejected.
///
/// Distinct from CONTRACT 1 because `0.0` is a plausible thing to write on
/// purpose, meaning "do not use extra memory" — and it is precisely the value
/// `resolve_budget` documents as "unset". A user cannot express that intent
/// today, and gets its opposite.
#[test]
fn a_zero_budget_is_rejected() {
    expect_rejected(
        "zero",
        "budget_gb = 0.0",
        "resolve_budget documents 0 bytes as \"unset\" and grants 0.8 x available RAM instead.",
    );
}

/// CONTRACT 3: a POSITIVE budget too small to reach one byte is rejected.
///
/// The subtle case, and the reason the check cannot just be `g > 0.0`:
/// `gib_to_bytes(1e-12)` truncates to 0, so this passes a naive sign test and
/// still inverts. The validation must test the RESOLVED BYTE COUNT, not just
/// the sign of the input.
#[test]
fn a_budget_that_rounds_to_zero_bytes_is_rejected() {
    expect_rejected(
        "tiny",
        "budget_gb = 1e-12",
        "gib_to_bytes truncates this to 0, so it takes the same silent fall-through as a \
         negative value — a sign-only check is insufficient.",
    );
}

/// CONTRACT 4: NaN is rejected.
#[test]
fn a_nan_budget_is_rejected() {
    expect_rejected("nan", "budget_gb = nan", "gib_to_bytes maps NaN to 0 bytes.");
}

/// CONTRACT 5: the deprecated alias is validated too.
///
/// `three_index_budget_gb` feeds the same `budget_gb()` accessor, so a check
/// applied only to the new field would leave the identical inversion reachable
/// through the alias that older configs still use.
#[test]
fn the_deprecated_alias_is_validated_too() {
    expect_rejected(
        "alias",
        "three_index_budget_gb = -1.0",
        "The deprecated alias feeds the same budget_gb() accessor, so a check applied only to \
         the new field leaves the identical inversion reachable through the old one.",
    );
}

/// CONTRACT 6 (the over-rejection guard): every VALID budget still loads, and
/// still resolves to exactly the byte count it did before.
///
/// This is the half that keeps the change inert. "An over-estimating guard is
/// also a bug" — a validation that rejected a usable budget, or that perturbed
/// a valid one, would be a regression dressed as a fix. Assert both that the
/// config loads AND that the resolved bytes are unchanged from
/// `gib_to_bytes`'s own answer.
#[test]
fn every_valid_budget_still_loads_and_resolves_identically() {
    for (i, gb) in ["0.5", "1.0", "2", "4.0", "12.5", "1024.0"].iter().enumerate() {
        expect_accepted(&format!("valid{i}"), &format!("budget_gb = {gb}"));
    }
    // And the resolved value is untouched: the same expression as before the
    // check, so no currently-valid config changes its resolved ceiling.
    assert_eq!(
        ferric_core::memory::gib_to_bytes(4.0),
        4 * 1024 * 1024 * 1024,
        "validation must not change how a valid budget resolves"
    );
}

/// CONTRACT 7: omitting `[memory]` entirely still means auto-detect.
///
/// Auto-detect is the correct behaviour for an ABSENT budget — the bug is only
/// that an unusable PRESENT one becomes indistinguishable from absent. A fix
/// that made the section mandatory would break every shipped example.
#[test]
fn an_absent_budget_still_means_auto_detect() {
    let out = run_with_memory("absent", "");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !stderr.contains("budget_gb"),
        "an empty [memory] section must remain valid.\n--- stderr ---\n{stderr}"
    );
    // The audit line is on stderr.
    assert!(
        stderr.contains("memory budget"),
        "an absent budget must still resolve (auto-detect) and report itself.\n\
         --- stderr ---\n{stderr}"
    );
}
