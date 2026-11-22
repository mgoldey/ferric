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

use crate::options::KBuilderKind;
use crate::rhf::RhfConfig;

/// The JK-fitting basis `"rijk"` / `"rijcosx"` use when none is named: the
/// same default the SCF layer auto-applies for a functional.
pub const DEFAULT_SCF_JK_AUX: &str = crate::fock_assembly::DEFAULT_JK_AUX;

/// The JK-fitting basis Weigend fitted for the orbital basis `orbital`
/// (case-insensitive bundled name): `cc-pvtz-jkfit` for cc-pVDZ and cc-pVTZ
/// (no cc-pVDZ-JKFIT set exists; the TZ set is the DZ partner),
/// `cc-pvqz-jkfit` for cc-pVQZ, `aug-cc-pv{d,t,q}z-jkfit` for the augmented
/// sets, and [`DEFAULT_SCF_JK_AUX`] for every other orbital basis.
pub fn matched_jk_aux(orbital: &str) -> &'static str {
    match orbital.trim().to_ascii_lowercase().as_str() {
        "cc-pvdz" | "cc-pvtz" => "cc-pvtz-jkfit",
        "cc-pvqz" => "cc-pvqz-jkfit",
        "aug-cc-pvdz" => "aug-cc-pvdz-jkfit",
        "aug-cc-pvtz" => "aug-cc-pvtz-jkfit",
        "aug-cc-pvqz" => "aug-cc-pvqz-jkfit",
        _ => DEFAULT_SCF_JK_AUX,
    }
}

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
    /// Key that selects the [`ScfJkMode`].
    pub mode: &'static str,
    /// Key that names the JK-fitting basis.
    pub aux: &'static str,
    /// Key that selects the exchange builder for an exact reference.
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

/// Why orbital-optimised MP2 refuses an RIJCOSX reference. Shared by the
/// Python `run_oo_rimp2` / `run_optimize(method="oo-rimp2")` and the CLI
/// `oo-rimp2` kind.
pub const OO_BUILDS_EXACT_OR_RIJK: &str = "orbital-optimised MP2 rebuilds J/K at every \
     orbital step on the reference's own route, which must be four-centre or RI-JK (RI-J and \
     RI-K on one JK-fitting basis); COSX exchange has no OO route, so an RIJCOSX reference \
     would change only the starting orbitals while the result claimed it";

/// A resolved `scf_jk` choice: the mode plus the JK-fitting basis it uses
/// (`None` exactly when the mode is [`ScfJkMode::Exact`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScfJk {
    /// How J and K are built.
    pub mode: ScfJkMode,
    /// JK-fitting basis name; `None` exactly when `mode` is
    /// [`ScfJkMode::Exact`].
    pub aux: Option<String>,
}

impl ScfJk {
    /// Resolve the user's `scf_jk` / `jk_aux` into one choice, checking the
    /// `k_builder` spelling next to them (`k_builder` as the user typed it,
    /// so a conflict quotes it; the frontend parses it into the config's
    /// [`KBuilderKind`], which an `"exact"` reference keeps).
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

    /// Does this choice density-fit the reference SCF at all?
    pub fn is_fitted(&self) -> bool {
        self.mode != ScfJkMode::Exact
    }

    /// Refuse RIJCOSX on a path that honours only four-centre or RI-JK
    /// J/K; `why` says what the path does instead. `"exact"` and `"rijk"`
    /// pass.
    pub fn require_exact_or_rijk(&self, keys: ScfJkKeys, why: &str) -> Result<(), String> {
        if self.mode != ScfJkMode::RiJCosx {
            return Ok(());
        }
        Err(format!(
            "{m} = \"{}\" is refused here: {why}. Use {m} = \"exact\" or \"rijk\"",
            self.mode.as_str(),
            m = keys.mode
        ))
    }

    /// A warning when the JK-fitting basis is not a JK-fit set (its name
    /// lacks "jkfit"): an MP2-fit (`-ri`/`-rifit`) basis used for K costs
    /// mHa-scale error. `None` when there is nothing to warn about.
    pub fn aux_warning(&self, keys: ScfJkKeys) -> Option<String> {
        let aux = self.aux.as_deref()?;
        (!aux.to_ascii_lowercase().contains("jkfit")).then(|| {
            format!(
                "{} = \"{aux}\" is not a JK-fitting set (its name lacks \"jkfit\"); an \
                 MP2/RI-fit basis used for exchange costs mHa-scale error. Prefer \
                 def2-universal-jkfit or a cc-pV*Z-jkfit set",
                keys.aux
            )
        })
    }

    /// The SCF J/K echo of a result printout: `"exact"`,
    /// `"exact, k_builder=link"` (a non-default exchange builder, the
    /// config's `k_builder`, on the four-centre path), or
    /// `"<mode>(<jk aux>)"`.
    pub fn label(&self, k_builder: Option<KBuilderKind>) -> String {
        match (&self.aux, k_builder) {
            (Some(a), _) => format!("{}({a})", self.mode.as_str()),
            (None, Some(kb)) if kb != KBuilderKind::Direct => {
                format!("{}, k_builder={kb}", self.mode.as_str())
            }
            (None, _) => self.mode.as_str().to_string(),
        }
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
                cfg.k_builder = Some(KBuilderKind::Cosx);
            }
        }
    }
}

/// The J/K route a closed-shell HF [`RhfConfig`] actually takes, for run
/// logs: what the SCF builds, not what the input spelled.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EffectiveJk {
    /// `"exact"`, `"rijk"`, `"rijcosx"`, or `"ri-j"` (RI-J with four-centre
    /// K) / `"mixed"` (any other combination of the raw fields).
    pub mode: &'static str,
    /// The J-fitting basis (`None` for four-centre J).
    pub j_aux: Option<String>,
    /// The K-fitting basis (`None` unless exchange is RI-K).
    pub k_aux: Option<String>,
    /// The pluggable exchange builder the SCF actually uses: `None` when the
    /// four-centre direct builder or RI-K builds K (the SCF ignores, with a
    /// warning, a `k_builder` that density fitting overrides).
    pub k_builder: Option<KBuilderKind>,
}

/// Classify `cfg`'s J/K fields the way the HF solvers act on them
/// (`fock_assembly::resolve_k_builder` for `need_k`, `omega = 0`): a fit
/// name of `""` is the "do not fit" sentinel; a DF-K, or a DF-J with a
/// non-COSX builder, overrides `k_builder`.
pub fn effective_jk(cfg: &RhfConfig) -> EffectiveJk {
    let fit = |a: &Option<String>| a.clone().filter(|s| !s.is_empty());
    let (j_aux, k_aux) = (fit(&cfg.df_j_aux), fit(&cfg.df_k_aux));
    let kb = cfg.k_builder.filter(|k| *k != KBuilderKind::Direct);
    let overridden = k_aux.is_some() || (j_aux.is_some() && kb != Some(KBuilderKind::Cosx));
    let k_builder = if overridden { None } else { kb };
    let mode = match (&j_aux, &k_aux, k_builder) {
        (None, None, _) => "exact",
        (Some(j), Some(k), _) if j == k => "rijk",
        (Some(_), None, Some(KBuilderKind::Cosx)) => "rijcosx",
        (Some(_), None, _) => "ri-j",
        _ => "mixed",
    };
    EffectiveJk {
        mode,
        j_aux,
        k_aux,
        k_builder,
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
        (ScfJkMode::RiJCosx, Some(kb)) if KBuilderKind::parse(kb) != Ok(KBuilderKind::Cosx) => {
            Err(format!(
                "{kbk} = \"{kb}\" conflicts with {m} = \"rijcosx\", whose exchange is COSX. \
             Remove {kbk}"
            ))
        }
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

    fn kind(kb: Option<&str>) -> Option<KBuilderKind> {
        KBuilderKind::parse_opt(kb).unwrap()
    }

    fn applied(jk: &ScfJk, k_builder: Option<&str>) -> RhfConfig {
        let mut cfg = RhfConfig {
            k_builder: kind(k_builder),
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
            k_builder: Some(KBuilderKind::Link),
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
            assert_eq!(cfg.k_builder, Some(KBuilderKind::Cosx));
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
        // The spelling is checked as typed: an unknown one conflicts like any
        // other (the frontend's parse into `KBuilderKind` reports it only
        // where no conflict does, as the solver did before).
        for kb in ["direct", "link", "cosx", "bogus", "COSX"] {
            assert!(
                err(Some("rijk"), None, Some(kb))
                    .contains(&format!("k_builder = \"{kb}\" conflicts")),
                "{kb}"
            );
        }
        for kb in ["direct", "link", "bogus", "COSX"] {
            assert!(
                err(Some("rijcosx"), None, Some(kb))
                    .contains(&format!("k_builder = \"{kb}\" conflicts")),
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
            assert_eq!(applied(&jk, Some(kb)).k_builder, kind(Some(kb)));
        }
    }

    #[test]
    fn rijcosx_is_refused_where_only_exact_or_rijk_is_honoured() {
        assert!(resolved(Some("exact"), None, Some("link"))
            .require_exact_or_rijk(ScfJkKeys::PYTHON, OO_BUILDS_EXACT_OR_RIJK)
            .is_ok());
        assert!(resolved(Some("rijk"), None, None)
            .require_exact_or_rijk(ScfJkKeys::PYTHON, OO_BUILDS_EXACT_OR_RIJK)
            .is_ok());
        let e = resolved(Some("rijcosx"), None, None)
            .require_exact_or_rijk(ScfJkKeys::CLI, OO_BUILDS_EXACT_OR_RIJK)
            .unwrap_err();
        assert!(e.contains("[scf] jk = \"rijcosx\" is refused"), "{e}");
        assert!(e.contains("COSX exchange has no OO route"), "{e}");
    }

    #[test]
    fn label_names_a_non_default_exchange_builder_only_on_the_exact_path() {
        let label = |m, kb| resolved(m, None, kb).label(kind(kb));
        assert_eq!(label(None, None), "exact");
        assert_eq!(label(None, Some("direct")), "exact");
        assert_eq!(label(Some("exact"), Some("link")), "exact, k_builder=link");
        assert_eq!(label(Some("exact"), Some("cosx")), "exact, k_builder=cosx");
        assert_eq!(
            label(Some("rijk"), None),
            format!("rijk({DEFAULT_SCF_JK_AUX})")
        );
        assert_eq!(
            label(Some("rijcosx"), Some("cosx")),
            format!("rijcosx({DEFAULT_SCF_JK_AUX})")
        );
    }

    #[test]
    fn a_non_jkfit_aux_is_warned_about() {
        assert_eq!(
            resolved(Some("rijk"), None, None).aux_warning(ScfJkKeys::PYTHON),
            None
        );
        assert_eq!(
            resolved(None, None, None).aux_warning(ScfJkKeys::PYTHON),
            None
        );
        let w = resolved(Some("rijk"), Some("cc-pvdz-ri"), None)
            .aux_warning(ScfJkKeys::CLI)
            .unwrap();
        assert!(
            w.contains("[scf] jk_aux = \"cc-pvdz-ri\" is not a JK-fitting set"),
            "{w}"
        );
    }

    #[test]
    fn effective_jk_reports_what_the_scf_builds() {
        let route = |j: Option<&str>, k: Option<&str>, kb: Option<&str>| {
            effective_jk(&RhfConfig {
                df_j_aux: j.map(str::to_string),
                df_k_aux: k.map(str::to_string),
                k_builder: kind(kb),
                ..Default::default()
            })
        };
        let jk = Some(DEFAULT_SCF_JK_AUX);
        let e = route(None, None, Some("link"));
        assert_eq!((e.mode, e.k_builder), ("exact", Some(KBuilderKind::Link)));
        assert_eq!(route(Some(""), Some(""), None).mode, "exact");
        assert_eq!(route(None, None, Some("direct")).k_builder, None);
        let e = route(jk, jk, Some("link"));
        assert_eq!((e.mode, e.k_builder), ("rijk", None), "DF-K overrides link");
        let e = route(jk, None, Some("cosx"));
        assert_eq!((e.mode, e.k_builder), ("rijcosx", Some(KBuilderKind::Cosx)));
        let e = route(jk, None, Some("link"));
        assert_eq!((e.mode, e.k_builder), ("ri-j", None), "DF-J overrides link");
        assert_eq!(route(jk, Some("cc-pvtz-jkfit"), None).mode, "mixed");
        assert_eq!(route(None, jk, None).mode, "mixed");
        for m in ["rijk", "rijcosx"] {
            let mut cfg = RhfConfig::default();
            resolved(Some(m), None, None).apply(&mut cfg);
            assert_eq!(effective_jk(&cfg).mode, m);
        }
    }
}
