//! How a correlated method's reference SCF builds J and K.
//!
//! One named switch, `scf_jk`, shared by every MP2-family entry point (Python
//! and the CLI), so that "RI-JK reference" means the same
//! [`RhfConfig`](crate::rhf::RhfConfig) fields everywhere:
//!
//! | `scf_jk`    | J                      | K                      |
//! |-------------|------------------------|------------------------|
//! | `"exact"`   | four-centre            | four-centre (`k_builder` direct/link/cosx) |
//! | `"rijk"`    | RI-J on `jk_aux`       | RI-K on `jk_aux`       |
//! | `"rijcosx"` | RI-J on `jk_aux`       | COSX (seminumerical)   |
//!
//! `jk_aux` defaults to [`DEFAULT_SCF_JK_AUX`](crate::scf_jk::DEFAULT_SCF_JK_AUX).
//! It is the SCF's JK-fitting basis and is deliberately a different knob from
//! the correlation method's RI auxiliary basis: an MP2-fit basis used for K
//! costs mHa-scale error.
//!
//! The parser is strict (config-honesty convention): an unknown mode, a
//! `jk_aux` that would be ignored (with `"exact"`) or that names no basis
//! (an opt-out spelling), and a `k_builder` that the mode would silently
//! override are all errors, never a quiet pick.

use crate::rhf::RhfConfig;

/// The JK-fitting basis `"rijk"` / `"rijcosx"` use when none is named: the
/// same default the SCF layer auto-applies for a functional.
pub const DEFAULT_SCF_JK_AUX: &str = crate::fock_assembly::DEFAULT_JK_AUX;

/// The accepted `scf_jk` spellings, in the order error messages list them.
pub const SCF_JK_MODES: &[&str] = &["exact", "rijk", "rijcosx"];

/// How the reference SCF builds J and K.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScfJkMode {
    /// Four-centre J and K (exchange from `k_builder`).
    Exact,
    /// RI-J + RI-K on one JK-fitting basis.
    RiJk,
    /// RI-J + COSX exchange (the RIJCOSX composite).
    RiJCosx,
}

impl ScfJkMode {
    /// Strict parse of an `scf_jk` value: exactly `"exact"`, `"rijk"` or
    /// `"rijcosx"`. Anything else is an error naming the accepted values.
    pub fn parse_config_str(s: &str) -> Result<Self, String> {
        match s {
            "exact" => Ok(Self::Exact),
            "rijk" => Ok(Self::RiJk),
            "rijcosx" => Ok(Self::RiJCosx),
            other => Err(format!(
                "unknown value \"{other}\"; expected one of {SCF_JK_MODES:?}"
            )),
        }
    }

    /// The canonical spelling, as echoed on results.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::RiJk => "rijk",
            Self::RiJCosx => "rijcosx",
        }
    }
}

/// The names a surface gives the three knobs, so errors quote what the
/// user actually typed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScfJkKeys {
    pub mode: &'static str,
    pub aux: &'static str,
    pub k_builder: &'static str,
}

impl ScfJkKeys {
    /// The Python MP2 entry points' kwargs.
    pub const PYTHON: Self = Self {
        mode: "scf_jk",
        aux: "jk_auxbasis",
        k_builder: "k_builder",
    };
    /// The CLI `[scf]` keys.
    pub const CLI: Self = Self {
        mode: "[scf] jk",
        aux: "[scf] jk_aux",
        k_builder: "[scf] k_builder",
    };
}

/// A resolved `scf_jk` choice: the mode plus the JK-fitting basis it uses
/// (`None` exactly when the mode is [`ScfJkMode::Exact`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScfJk {
    pub mode: ScfJkMode,
    pub aux: Option<String>,
}

impl ScfJk {
    /// Resolve the user's `scf_jk` / `jk_aux` / `k_builder` into one choice.
    ///
    /// `mode = None` takes `default` (an entry point's historical behaviour,
    /// so the unset case stays bit-identical). Errors:
    /// * an unknown `mode`;
    /// * `jk_aux` with `"exact"` (it would be read by nothing);
    /// * `jk_aux` spelled as an opt-out (`"exact"`, `"none"`, `""`, ...),
    ///   which would silently turn the fit off;
    /// * a `jk_aux` that is not a bundled basis;
    /// * any `k_builder` with `"rijk"` (RI-K replaces it);
    /// * a `k_builder` other than `"cosx"` with `"rijcosx"`.
    pub fn resolve(
        mode: Option<&str>,
        jk_aux: Option<&str>,
        k_builder: Option<&str>,
        default: ScfJkMode,
        keys: ScfJkKeys,
    ) -> Result<Self, String> {
        let mode = match mode {
            None => default,
            Some(m) => ScfJkMode::parse_config_str(m).map_err(|e| format!("{}: {e}", keys.mode))?,
        };
        check_k_builder(mode, k_builder, keys)?;
        let aux = match (mode, jk_aux) {
            (ScfJkMode::Exact, None) => None,
            (ScfJkMode::Exact, Some(a)) => {
                return Err(format!(
                    "{aux} = \"{a}\" is only read with {m} = \"rijk\" or \"rijcosx\"; \
                     {m} = \"exact\" fits nothing. Remove {aux} or choose a fitted mode",
                    aux = keys.aux,
                    m = keys.mode
                ))
            }
            (_, None) => Some(DEFAULT_SCF_JK_AUX.to_string()),
            (_, Some(a)) => Some(checked_aux(mode, a, keys)?),
        };
        Ok(Self { mode, aux })
    }

    /// Write this choice into the reference SCF's config. `"exact"` clears
    /// both fit names (pure HF never auto-defaults them, so this is the
    /// historical four-centre path) and leaves `k_builder` alone.
    pub fn apply(&self, cfg: &mut RhfConfig) {
        match self.mode {
            ScfJkMode::Exact => {
                cfg.df_j_aux = None;
                cfg.df_k_aux = None;
            }
            ScfJkMode::RiJk => {
                cfg.df_j_aux = self.aux.clone();
                cfg.df_k_aux = self.aux.clone();
                cfg.k_builder = None;
            }
            ScfJkMode::RiJCosx => {
                cfg.df_j_aux = self.aux.clone();
                cfg.df_k_aux = None;
                cfg.k_builder = Some("cosx".to_string());
            }
        }
    }
}

fn check_k_builder(
    mode: ScfJkMode,
    k_builder: Option<&str>,
    keys: ScfJkKeys,
) -> Result<(), String> {
    let (kbk, m) = (keys.k_builder, keys.mode);
    match (mode, k_builder) {
        (ScfJkMode::RiJk, Some(kb)) => Err(format!(
            "{kbk} = \"{kb}\" conflicts with {m} = \"rijk\": RI-K builds the exchange, so \
             {kbk} would be ignored. Remove {kbk}, or use {m} = \"rijcosx\" for COSX"
        )),
        (ScfJkMode::RiJCosx, Some(kb)) if kb != "cosx" => Err(format!(
            "{kbk} = \"{kb}\" conflicts with {m} = \"rijcosx\", whose exchange is COSX. \
             Remove {kbk}"
        )),
        _ => Ok(()),
    }
}

fn checked_aux(mode: ScfJkMode, aux: &str, keys: ScfJkKeys) -> Result<String, String> {
    let name = crate::rhf::normalize_df_aux(aux);
    if name.is_empty() {
        return Err(format!(
            "{k} = \"{aux}\" turns density fitting off, which contradicts {m} = \"{}\"; \
             use {m} = \"exact\" for four-centre J/K",
            mode.as_str(),
            k = keys.aux,
            m = keys.mode
        ));
    }
    ferric_core::basis::bundled(&name).map_err(|e| format!("{} = \"{name}\": {e}", keys.aux))?;
    Ok(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resolved(mode: Option<&str>, aux: Option<&str>, kb: Option<&str>) -> ScfJk {
        ScfJk::resolve(mode, aux, kb, ScfJkMode::Exact, ScfJkKeys::PYTHON).unwrap()
    }

    fn err(mode: Option<&str>, aux: Option<&str>, kb: Option<&str>) -> String {
        ScfJk::resolve(mode, aux, kb, ScfJkMode::Exact, ScfJkKeys::PYTHON).unwrap_err()
    }

    fn applied(jk: &ScfJk, k_builder: Option<&str>) -> RhfConfig {
        let mut cfg = RhfConfig {
            k_builder: k_builder.map(str::to_string),
            ..Default::default()
        };
        jk.apply(&mut cfg);
        cfg
    }

    #[test]
    fn unset_takes_the_entry_point_default_and_exact_is_the_untouched_config() {
        let exact = resolved(None, None, Some("link"));
        assert_eq!(exact.mode, ScfJkMode::Exact);
        assert_eq!(exact.aux, None);
        let cfg = applied(&exact, Some("link"));
        let plain = RhfConfig {
            k_builder: Some("link".into()),
            ..Default::default()
        };
        assert_eq!(cfg.df_j_aux, plain.df_j_aux);
        assert_eq!(cfg.df_k_aux, plain.df_k_aux);
        assert_eq!(cfg.k_builder, plain.k_builder);
        let rs = ScfJk::resolve(None, None, None, ScfJkMode::RiJk, ScfJkKeys::PYTHON).unwrap();
        assert_eq!(rs.mode, ScfJkMode::RiJk);
    }

    #[test]
    fn rijk_sets_both_fits_to_the_same_jk_basis_and_clears_k_builder() {
        let jk = resolved(Some("rijk"), None, None);
        let cfg = applied(&jk, None);
        assert_eq!(cfg.df_j_aux.as_deref(), Some(DEFAULT_SCF_JK_AUX));
        assert_eq!(cfg.df_k_aux.as_deref(), Some(DEFAULT_SCF_JK_AUX));
        assert_eq!(cfg.k_builder, None);
        let named = resolved(Some("rijk"), Some(" cc-pvtz-jkfit "), None);
        assert_eq!(named.aux.as_deref(), Some("cc-pvtz-jkfit"));
        let cfg = applied(&named, None);
        assert_eq!(cfg.df_j_aux.as_deref(), Some("cc-pvtz-jkfit"));
        assert_eq!(cfg.df_k_aux.as_deref(), Some("cc-pvtz-jkfit"));
    }

    #[test]
    fn rijcosx_fits_j_only_and_selects_cosx_exchange() {
        for kb in [None, Some("cosx")] {
            let jk = resolved(Some("rijcosx"), None, kb);
            let cfg = applied(&jk, kb);
            assert_eq!(cfg.df_j_aux.as_deref(), Some(DEFAULT_SCF_JK_AUX));
            assert_eq!(cfg.df_k_aux, None);
            assert_eq!(cfg.k_builder.as_deref(), Some("cosx"));
        }
    }

    #[test]
    fn exact_on_a_config_that_had_fits_clears_them() {
        let mut cfg = RhfConfig {
            df_j_aux: Some(DEFAULT_SCF_JK_AUX.into()),
            df_k_aux: Some(DEFAULT_SCF_JK_AUX.into()),
            ..Default::default()
        };
        resolved(Some("exact"), None, None).apply(&mut cfg);
        assert_eq!((cfg.df_j_aux, cfg.df_k_aux), (None, None));
    }

    #[test]
    fn strict_parse_and_conflicts_are_errors() {
        for bad in ["RIJK", "ri-jk", "rij", "", "cosx", "direct"] {
            assert!(
                err(Some(bad), None, None).contains("scf_jk: unknown value"),
                "{bad}"
            );
        }
        assert!(err(Some("exact"), Some("def2-universal-jkfit"), None).contains("only read"));
        assert!(err(None, Some("def2-universal-jkfit"), None).contains("only read"));
        for kb in ["direct", "link", "cosx"] {
            assert!(
                err(Some("rijk"), None, Some(kb)).contains("conflicts"),
                "{kb}"
            );
        }
        for kb in ["direct", "link"] {
            assert!(
                err(Some("rijcosx"), None, Some(kb)).contains("conflicts"),
                "{kb}"
            );
        }
        for off in ["exact", "none", "", "off"] {
            assert!(err(Some("rijk"), Some(off), None).contains("turns density fitting off"));
        }
        assert!(err(Some("rijk"), Some("no-such-jkfit"), None).contains("no-such-jkfit"));
    }

    #[test]
    fn exact_keeps_every_k_builder() {
        for kb in ["direct", "link", "cosx"] {
            let jk = resolved(Some("exact"), None, Some(kb));
            assert_eq!(applied(&jk, Some(kb)).k_builder.as_deref(), Some(kb));
        }
    }
}
