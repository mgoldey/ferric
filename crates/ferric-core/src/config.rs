//! Unified configuration ingestion for ferric — env vars and TOML, one pattern.
//!
//! ferric reads ~27 runtime settings from env vars, historically via scattered
//! ad-hoc `std::env::var(...)` calls with inconsistent parsing, per-site default
//! drift, and (worst) two incompatible bool idioms (`== Some("1")` vs `.is_ok()`,
//! so `FOO=0` disables one flag and enables another). This module is the single
//! pattern that replaces those, generalizing the [`crate::memory`] budget
//! resolver from one setting to a reusable per-setting descriptor.
//!
//! A setting is declared once as a [`ConfigVar<T>`] (env name + default + parse +
//! validate). [`ConfigVar::resolve`] applies the uniform precedence
//! **config/TOML > env > default** and returns a [`Resolved<T>`] that can emit an
//! [`Resolved::audit_line`] — the same auditability [`crate::memory`] gives the
//! budget. A malformed *explicitly-set* override (env or TOML) is a loud `Err`,
//! never a silent fallback to the default: a typo in a result-affecting knob must
//! not quietly change the answer.
//!
//! Debug-trace toggles route through the shared [`parse_toggle`] so `FERRIC_X=0`
//! means off everywhere. See `docs/config-style.md` for the dev-facing rules.
//!
//! This is the foundational module (migration Group 1); call-site substitutions
//! (Groups 2–8) are separate changes. The budget resolver in [`crate::memory`] is
//! intentionally NOT folded in — it keeps its specialized 5-tier chain (legacy-var
//! reconciliation + RAM auto-detect) and simply lives beside this module.

/// Which config channel supplied a resolved value — the audit-line source,
/// mirroring [`crate::memory::BudgetSource`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigSource {
    /// A caller-supplied value (TOML field / config field / Python kwarg).
    Explicit,
    /// The setting's env var.
    Env,
    /// The descriptor's built-in default (nothing was set).
    Default,
}

impl ConfigSource {
    /// Human-readable source for the audit line.
    pub fn label(self) -> &'static str {
        match self {
            ConfigSource::Explicit => "explicit (config/TOML/kwarg)",
            ConfigSource::Env => "env",
            ConfigSource::Default => "default",
        }
    }
}

/// A resolved setting plus where it came from and the env var it *would* read.
#[derive(Debug, Clone, Copy)]
pub struct Resolved<T> {
    pub value: T,
    pub source: ConfigSource,
    pub env_name: &'static str,
}

impl<T: std::fmt::Display> Resolved<T> {
    /// `name: value  [source: ...]` — one audit line, same shape as
    /// [`crate::memory::BudgetResolution::audit_line`].
    pub fn audit_line(&self) -> String {
        format!(
            "{}: {}  [source: {}]",
            self.env_name,
            self.value,
            self.source.label()
        )
    }
}

/// Declarative descriptor for one setting. `parse` is the ONE place the var's
/// string form becomes `T`; `validate` rejects out-of-range values LOUDLY
/// (returns `Err` — no silent fallback to the default on a malformed override).
pub struct ConfigVar<T: 'static> {
    pub env_name: &'static str,
    pub default: T,
    pub parse: fn(&str) -> Result<T, String>,
    pub validate: fn(&T) -> Result<(), String>,
}

impl<T: Clone + std::fmt::Display> ConfigVar<T> {
    /// Resolve with precedence **config/TOML > env > default**. `explicit` is the
    /// TOML/kwarg value (already-typed, `None` from a library path with no TOML);
    /// `get` is the injectable env lookup — [`env_lookup`] in production, a
    /// closure in tests (process-global `env::set_var` races the parallel test
    /// harness; inject instead).
    ///
    /// A present `explicit` or a present env string is *validated*, and a
    /// present-but-malformed value is an `Err` — absent falls through quietly.
    pub fn resolve(
        &self,
        explicit: Option<T>,
        get: impl Fn(&str) -> Option<String>,
    ) -> Result<Resolved<T>, String> {
        if let Some(v) = explicit {
            (self.validate)(&v)?;
            return Ok(Resolved {
                value: v,
                source: ConfigSource::Explicit,
                env_name: self.env_name,
            });
        }
        if let Some(raw) = get(self.env_name) {
            // Loud: a malformed *explicitly-set* override is an error, not a
            // silent default (fixes the `CPKS_WP=1O`-typo class of bug).
            let v = (self.parse)(raw.trim())?;
            (self.validate)(&v)?;
            return Ok(Resolved {
                value: v,
                source: ConfigSource::Env,
                env_name: self.env_name,
            });
        }
        Ok(Resolved {
            value: self.default.clone(),
            source: ConfigSource::Default,
            env_name: self.env_name,
        })
    }

    /// Convenience: [`resolve`](Self::resolve) against the real process env, no
    /// explicit override. The common library-path shape.
    pub fn get(&self) -> Result<Resolved<T>, String> {
        self.resolve(None, env_lookup)
    }
}

impl ConfigVar<bool> {
    /// A debug/trace toggle's resolved value against the process env, no explicit
    /// override. A malformed value logs a one-line warning and returns the
    /// default (`false` for a trace flag) — a DEBUG toggle must never abort a
    /// real run over a typo like `FERRIC_X=tru`. (This is the deliberate
    /// exception to the "malformed override Errs loudly" rule, which applies to
    /// *result-affecting* knobs, not diagnostic prints.)
    pub fn toggle(&self) -> bool {
        self.get().map(|r| r.value).unwrap_or_else(|e| {
            eprintln!("[config] {}: {e}; treating as off", self.env_name);
            self.default
        })
    }
}

/// The production env lookup: `|k| std::env::var(k).ok()`. Pass to
/// [`ConfigVar::resolve`] (or use [`ConfigVar::get`]). Kept as a named fn so call
/// sites read consistently and tests can substitute a closure.
pub fn env_lookup(key: &str) -> Option<String> {
    std::env::var(key).ok()
}

/// Shared bool parser for `*_TRACE`/toggle vars: canonicalizes
/// `1/true/on/yes` (case-insensitive) → `true`, `0/false/off/no/""` → `false`,
/// anything else → `Err`. Kills the "`SCF_TRACE` needs exactly `1` but
/// `ROHF_TRACE` fires on any value" split, so `FERRIC_X=0` means off everywhere.
pub fn parse_toggle(s: &str) -> Result<bool, String> {
    match s.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "on" | "yes" => Ok(true),
        "0" | "false" | "off" | "no" | "" => Ok(false),
        other => Err(format!(
            "invalid boolean toggle {other:?} (expected one of \
             1/true/on/yes or 0/false/off/no)"
        )),
    }
}

/// The trivial validator: accept everything. For settings whose `parse` already
/// establishes validity (e.g. bools via [`parse_toggle`], or any value is legal).
pub fn accept_any<T>(_v: &T) -> Result<(), String> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Build an injected env lookup from pairs (no process-global set_var).
    fn lookup(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let map: HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        move |k: &str| map.get(k).cloned()
    }

    fn lindep() -> ConfigVar<f64> {
        ConfigVar {
            env_name: "FERRIC_LINDEP_THRESH",
            default: 1e-6,
            parse: |s| s.parse::<f64>().map_err(|e| e.to_string()),
            validate: |v| {
                (v.is_finite() && *v > 0.0)
                    .then_some(())
                    .ok_or_else(|| "must be finite > 0".to_string())
            },
        }
    }

    #[test]
    fn precedence_explicit_over_env_over_default() {
        let v = lindep();
        // (1) explicit (TOML/config) WINS over env.
        let r = v
            .resolve(Some(1e-3), lookup(&[("FERRIC_LINDEP_THRESH", "1e-9")]))
            .unwrap();
        assert_eq!(r.value, 1e-3);
        assert_eq!(r.source, ConfigSource::Explicit);
        // (2) env fills in when explicit is None.
        let r = v
            .resolve(None, lookup(&[("FERRIC_LINDEP_THRESH", "1e-9")]))
            .unwrap();
        assert_eq!(r.value, 1e-9);
        assert_eq!(r.source, ConfigSource::Env);
        // (3) default when neither is set.
        let r = v.resolve(None, lookup(&[])).unwrap();
        assert_eq!(r.value, 1e-6);
        assert_eq!(r.source, ConfigSource::Default);
    }

    #[test]
    fn malformed_env_override_errs_loudly_not_silent_default() {
        let v = lindep();
        // A present-but-unparseable env value is an Err (the typo-swallowing bug).
        let err = v
            .resolve(None, lookup(&[("FERRIC_LINDEP_THRESH", "1e-")]))
            .unwrap_err();
        assert!(!err.is_empty());
        // A parseable-but-invalid value also Errs (validate runs on env too).
        assert!(v
            .resolve(None, lookup(&[("FERRIC_LINDEP_THRESH", "-1.0")]))
            .is_err());
        assert!(v
            .resolve(None, lookup(&[("FERRIC_LINDEP_THRESH", "0")]))
            .is_err());
    }

    #[test]
    fn explicit_invalid_value_is_validated_too() {
        let v = lindep();
        // An explicit (TOML/config) value is validated, not blindly trusted.
        assert!(v.resolve(Some(-3.0), lookup(&[])).is_err());
    }

    #[test]
    fn parse_toggle_canonicalizes_both_idioms() {
        for on in ["1", "true", "TRUE", "on", "On", "yes", " yes "] {
            assert_eq!(parse_toggle(on), Ok(true), "{on:?} should be on");
        }
        for off in ["0", "false", "OFF", "no", ""] {
            assert_eq!(parse_toggle(off), Ok(false), "{off:?} should be off");
        }
        // The key fix: "0" is OFF (was ON under the `.is_ok()` idiom).
        assert_eq!(parse_toggle("0"), Ok(false));
        // Garbage is a loud Err, not a silent true/false.
        assert!(parse_toggle("maybe").is_err());
        assert!(parse_toggle("2").is_err());
    }

    #[test]
    fn toggle_configvar_roundtrips() {
        let trace: ConfigVar<bool> = ConfigVar {
            env_name: "FERRIC_SCF_TRACE",
            default: false,
            parse: parse_toggle,
            validate: accept_any,
        };
        assert!(
            !trace.resolve(None, lookup(&[])).unwrap().value,
            "default off"
        );
        assert!(
            trace
                .resolve(None, lookup(&[("FERRIC_SCF_TRACE", "1")]))
                .unwrap()
                .value
        );
        assert!(
            !trace
                .resolve(None, lookup(&[("FERRIC_SCF_TRACE", "0")]))
                .unwrap()
                .value,
            "=0 is OFF (the cross-flag consistency fix)"
        );
        assert!(trace
            .resolve(None, lookup(&[("FERRIC_SCF_TRACE", "garbage")]))
            .is_err());
    }

    #[test]
    fn audit_line_shape_matches_budget() {
        let v = lindep();
        let r = v
            .resolve(None, lookup(&[("FERRIC_LINDEP_THRESH", "1e-9")]))
            .unwrap();
        assert_eq!(r.audit_line(), "FERRIC_LINDEP_THRESH: 0.000000001  [source: env]");
        let r = v.resolve(Some(1e-3), lookup(&[])).unwrap();
        assert_eq!(
            r.audit_line(),
            "FERRIC_LINDEP_THRESH: 0.001  [source: explicit (config/TOML/kwarg)]"
        );
    }
}
