//! Frontend-neutral SCF options and their lowering to [`RhfConfig`](crate::rhf::RhfConfig).
//!
//! Frontends (the Python `run_rhf` / `run_uhf` / `run_rohf` and the Kohn-Sham
//! `run_dft` / `run_uks` / `run_roks`, and the CLI `[scf]` section) parse
//! their own
//! surface syntax (strings, PyO3 objects, TOML) into an [`ScfOptions`](crate::options::ScfOptions), then
//! call [`ScfOptions::into_rhf_config`](crate::options::ScfOptions::into_rhf_config). The lowering is pure: no I/O, no
//! environment reads, no process exit, no printing. It owns the defaults (the
//! named consts below, which [`RhfConfig::default`](crate::rhf::RhfConfig::default) also reads) and the knob
//! semantics (e.g. "an explicit `level_shift` turns off the meta-GGA automatic
//! shift"), so two frontends cannot drift apart on them.
//!
//! Frontend-specific checks with their own error type and message (Python
//! `ValueError`s, say) stay in the frontend and run BEFORE the lowering; the
//! lowering's own checks are a backstop that returns [`FerricError`](ferric_core::FerricError).
//!
//! `UhfConfig` and `RohfConfig` are aliases of [`RhfConfig`](crate::rhf::RhfConfig), so the one
//! lowering serves all three Hartree-Fock flavors. Per-flavor policy (which
//! knobs a given entry point exposes) is expressed by which fields that
//! entry point sets: an unset (`None`) field always keeps the
//! [`RhfConfig::default`](crate::rhf::RhfConfig::default) value.
//!
//! Cross-option defaults that more than one frontend applies before filling
//! an [`ScfOptions`](crate::options::ScfOptions) live here as pure functions
//! (e.g. [`df_k_aux_or_default`](crate::options::df_k_aux_or_default), the
//! RIJCOSX "no RI-K default under COSX" rule).
//!
//! # Per-frontend defaults
//!
//! What each frontend puts in the [`ScfOptions`](crate::options::ScfOptions)
//! fields that differ between them when the caller says nothing ("unset" =
//! the field is `None`, so the [`RhfConfig::default`](crate::rhf::RhfConfig::default)
//! value applies). `JK` is [`DEFAULT_JK_AUX`](crate::options::DEFAULT_JK_AUX);
//! "no RI-K under COSX" is [`df_k_aux_or_default`](crate::options::df_k_aux_or_default).
//! A caller's explicit value always wins (aux spellings through
//! [`AuxBasis::parse`](crate::options::AuxBasis::parse)).
//!
//! | Frontend | `df_j_aux` | `df_k_aux` | `guess` | `cosx` | `ri_j_task` | `xc` |
//! |---|---|---|---|---|---|---|
//! | `run_rhf` / `run_uhf` / `run_rohf` | unset (exact J) | unset (exact K) | `guess=` kwarg, else SAD | the `cosx_*` kwargs | `Energy` | unset (HF) |
//! | `run_dft` | `JK` | `JK`; no RI-K under COSX | `guess=` kwarg, else SAD | the `cosx_*` kwargs | `Energy` | the functional, default `"LDA"` |
//! | `run_uks` / `run_roks` | `JK` | `JK`; no RI-K under COSX | `guess=` kwarg, else SAD | the `cosx_*` kwargs | `Energy` | the functional (required) |
//! | CLI `[scf]` | the kind's default: `JK` for Kohn-Sham, RPA/response and TDA/TDDFT kinds, else unset | as `df_j_aux`; no RI-K under COSX | `[scf] guess`, else SAD | `[scf] cosx_*` and the `[gpu]` fp64 router | from `method.task`: geometry for `optimize` / `frequencies` / `md`, else energy | the kind's functional (`[dft]`, `[rpa] xc`, `[tddft] xc`), else unset |
//!
//! `ri_j_task` is not an option but the argument of
//! [`ScfOptions::into_rhf_config`](crate::options::ScfOptions::into_rhf_config):
//! the task class the frontend runs.
//!
//! The CLI column is `Config::scf_xc_and_aux_defaults` and `ri_j_task_for`
//! in `ferric-cli`; the Python rows are
//! [`ScfEntryPoint::options`](crate::options::entry::ScfEntryPoint::options),
//! which all six bindings call, and which also lists the knobs each entry
//! point refuses because its solver cannot honour them
//! ([`ScfEntryPoint::refused`](crate::options::entry::ScfEntryPoint::refused)).

use ferric_core::FerricError;
use ndarray::Array2;

use crate::rhf::RhfConfig;

pub mod entry;

/// Default SCF iteration cap ([`RhfConfig::max_iter`]).
pub const DEFAULT_SCF_MAX_ITER: usize = 200;
/// Default loose `|ΔE|` gate in Hartree ([`RhfConfig::energy_conv`]).
pub const DEFAULT_SCF_ENERGY_CONV: f64 = 1e-3;
/// Default RMS density-change gate ([`RhfConfig::density_conv`]).
pub const DEFAULT_SCF_DENSITY_CONV: f64 = 1e-6;
/// Default DIIS subspace size ([`RhfConfig::diis_size`]).
pub const DEFAULT_DIIS_SIZE: usize = 8;
/// Default integral screening threshold ([`RhfConfig::integral_thresh`]).
pub const DEFAULT_INTEGRAL_THRESH: f64 = 1e-12;
/// The `newton_trigger` (DIIS `err_max`) a `soscf = true` request arms.
pub const SOSCF_NEWTON_TRIGGER: f64 = 1e-3;
/// Default JK-fit auxiliary basis wherever RI-J/RI-K is auto-selected.
pub const DEFAULT_JK_AUX: &str = "def2-universal-jkfit";
/// The exchange (K) builder an exact-exchange SCF uses
/// ([`RhfConfig::k_builder`]; `None` there is the default, [`Direct`]).
///
/// The frontends receive it as a string (Python `k_builder=`, CLI
/// `[scf] k_builder`) and parse it once with [`KBuilderKind::parse`]; past
/// that boundary it is never a string. Named `...Kind` because
/// [`crate::fock::KBuilder`] is the trait the builders implement.
///
/// [`Direct`]: KBuilderKind::Direct
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KBuilderKind {
    /// The direct four-centre builder (`"direct"`, the default). Spelled
    /// out, it is still a request: the solvers warn when density fitting
    /// overrides it.
    Direct,
    /// LinK screened exchange (`"link"`, [`crate::link_k`]).
    Link,
    /// Seminumerical chain-of-spheres exchange (`"cosx"`,
    /// [`crate::cosx_k`]), the K of the RIJCOSX composite.
    Cosx,
}

/// [`KBuilderKind::parse`]'s refusal: the spelling given. Its `Display` is
/// the user-facing message, and it lowers to [`FerricError::General`] (the
/// error the solvers raised for it before the frontends parsed it).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownKBuilder(pub String);

impl std::fmt::Display for UnknownKBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "unknown k_builder '{}': valid options are 'direct', 'link', 'cosx'",
            self.0
        )
    }
}

impl std::error::Error for UnknownKBuilder {}

impl From<UnknownKBuilder> for FerricError {
    fn from(e: UnknownKBuilder) -> Self {
        FerricError::General(e.to_string())
    }
}

impl KBuilderKind {
    /// Every kind, in the order the error message lists them.
    pub const ALL: [KBuilderKind; 3] = [Self::Direct, Self::Link, Self::Cosx];

    /// Strict, case-sensitive parse of a user spelling: exactly `"direct"`,
    /// `"link"` or `"cosx"`.
    pub fn parse(s: &str) -> Result<Self, UnknownKBuilder> {
        match s {
            "direct" => Ok(Self::Direct),
            "link" => Ok(Self::Link),
            "cosx" => Ok(Self::Cosx),
            other => Err(UnknownKBuilder(other.to_string())),
        }
    }

    /// Whether `s` spells [`KBuilderKind::Cosx`] (through the one
    /// whitelist). For the checks that read a spelling as typed, to quote it
    /// in their messages, before the frontend's own parse reports an unknown
    /// one: an unknown spelling is not COSX.
    pub fn spells_cosx(s: &str) -> bool {
        Self::parse(s) == Ok(Self::Cosx)
    }

    /// [`KBuilderKind::parse`] of an optional keyword (`None` = not given).
    pub fn parse_opt(s: Option<&str>) -> Result<Option<Self>, UnknownKBuilder> {
        s.map(Self::parse).transpose()
    }

    /// The canonical spelling (what [`KBuilderKind::parse`] accepts), used
    /// in messages, labels and the checkpoint identity.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Direct => "direct",
            Self::Link => "link",
            Self::Cosx => "cosx",
        }
    }
}

impl std::str::FromStr for KBuilderKind {
    type Err = UnknownKBuilder;

    /// [`KBuilderKind::parse`].
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

impl std::fmt::Display for KBuilderKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The RIJCOSX rule for the RI-K auxiliary basis: an `explicit` choice
/// always wins; otherwise the frontend's `default` applies, except under
/// `k_builder = "cosx"`, where COSX replaces RI-K and no RI-K default is
/// applied (`None`; a NAMED `df_k_aux` next to COSX, with HF exchange or a
/// global hybrid, stays the conflict the SCF refuses, while `df_k_aux = ""`
/// is fine and pure or range-separated functionals ignore COSX). RI-J is
/// unaffected.
///
/// Generic over the aux representation; the Python KS entry points and the
/// CLI `[scf]` lowering both pass [`AuxBasis`].
pub fn df_k_aux_or_default<T>(
    k_builder: Option<KBuilderKind>,
    explicit: Option<T>,
    default: Option<T>,
) -> Option<T> {
    explicit.or(if k_builder == Some(KBuilderKind::Cosx) {
        None
    } else {
        default
    })
}

/// The Kohn-Sham JK-fit defaults: RI-J on `jk` unless `df_j_aux` names a
/// choice, and RI-K on `jk` under the RIJCOSX rule
/// ([`df_k_aux_or_default`]: no RI-K default under COSX). The one home of
/// the Kohn-Sham JK-fit default, for the Python KS entry points
/// ([`ScfEntryPoint::options`](crate::options::entry::ScfEntryPoint::options),
/// over [`AuxBasis`]) and `ksdft_ladder` (over a lowered config's names).
/// Returns `(df_j_aux, df_k_aux)`: RI-J always has a choice, RI-K may not.
///
/// The CLI's per-kind defaults are deliberately separate
/// (`Config::scf_xc_and_aux_defaults`) and share only the RIJCOSX rule.
pub fn ks_jk_fit_defaults<T: Clone>(
    k_builder: Option<KBuilderKind>,
    df_j_aux: Option<T>,
    df_k_aux: Option<T>,
    jk: T,
) -> (T, Option<T>) {
    let df_k_aux = df_k_aux_or_default(k_builder, df_k_aux, Some(jk.clone()));
    (df_j_aux.unwrap_or(jk), df_k_aux)
}

/// A density-fitting auxiliary basis choice for [`RhfConfig::df_j_aux`] /
/// [`RhfConfig::df_k_aux`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuxBasis {
    /// Explicitly no density fitting (exact four-centre J or K). Lowers to
    /// the `Some("")` sentinel `RhfConfig` uses for an explicit opt-out.
    Off,
    /// A named auxiliary basis (trimmed; validated later by `basis::bundled`).
    /// Build it with [`AuxBasis::named`] or [`AuxBasis::parse`], which
    /// normalise the spelling; the variant is public for pattern matching.
    Named(String),
}

impl AuxBasis {
    /// Parse a user spelling: any of
    /// [`DF_AUX_OFF_SPELLINGS`](crate::rhf::DF_AUX_OFF_SPELLINGS)
    /// (case-insensitive, trimmed) is [`AuxBasis::Off`], anything else a
    /// basis name (see [`normalize_df_aux`](crate::rhf::normalize_df_aux)).
    pub fn parse(value: &str) -> Self {
        let name = crate::rhf::normalize_df_aux(value);
        if name.is_empty() {
            Self::Off
        } else {
            Self::Named(name)
        }
    }

    /// A basis named by a frontend (a default, not user input), through the
    /// same trim / opt-out normalisation as [`AuxBasis::parse`], so a name
    /// cannot reach [`AuxBasis::Named`] untrimmed or as an opt-out spelling.
    pub fn named(name: &str) -> Self {
        Self::parse(name)
    }

    /// [`DEFAULT_JK_AUX`], the auxiliary basis the Kohn-Sham entry points
    /// fit with when the caller names none.
    pub fn default_jk() -> Self {
        Self::named(DEFAULT_JK_AUX)
    }

    /// The `RhfConfig` string form (`""` for [`AuxBasis::Off`]).
    fn into_config_string(self) -> String {
        match self {
            Self::Off => String::new(),
            Self::Named(name) => name,
        }
    }
}

/// Orbital-Hessian stability handling after convergence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Stability {
    /// No stability check (the `RhfConfig` default).
    Off,
    /// Run the stability analysis and report the verdict, but do not act on
    /// it: sets [`RhfConfig::check_stability`] only.
    Check,
    /// Check stability and, on a saddle, descend and re-converge: sets both
    /// [`RhfConfig::check_stability`] and [`RhfConfig::scf_stability_descent`]
    /// (the descent needs a verdict to act on).
    Descend,
}

impl Stability {
    /// Whether the stability analysis runs ([`Stability::Check`] or
    /// [`Stability::Descend`]; the descent acts on its verdict).
    pub fn checks(self) -> bool {
        matches!(self, Stability::Check | Stability::Descend)
    }
}

/// Initial-guess family ([`RhfConfig::use_sad_guess`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Guess {
    /// Superposed atomic densities (MINAO projection; the default).
    Sad,
    /// The bare core-Hamiltonian guess.
    Core,
}

/// A validated [`RhfConfig::trah_trigger`]: a positive finite DIIS-error
/// threshold. [`TrahTrigger::new`] is the single place that rule lives.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrahTrigger(f64);

/// [`TrahTrigger::new`]'s refusal. Its `Display` is the user-facing message.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InvalidTrahTrigger(pub f64);

impl std::fmt::Display for InvalidTrahTrigger {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "trah_trigger must be a positive number, got {}", self.0)
    }
}

impl std::error::Error for InvalidTrahTrigger {}

impl From<InvalidTrahTrigger> for FerricError {
    fn from(e: InvalidTrahTrigger) -> Self {
        FerricError::General(e.to_string())
    }
}

impl TrahTrigger {
    /// Accept `value` when it is finite and > 0.
    pub fn new(value: f64) -> Result<Self, InvalidTrahTrigger> {
        if value.is_finite() && value > 0.0 {
            Ok(Self(value))
        } else {
            Err(InvalidTrahTrigger(value))
        }
    }

    /// The threshold.
    pub fn get(self) -> f64 {
        self.0
    }
}

/// User-tunable SCF knobs, already parsed into typed values. `None` keeps the
/// [`RhfConfig::default`] value of the field it lowers to; a `Some` is lowered
/// by its field's mapping (most copy the value; `soscf`, for example, sets
/// `newton_trigger`). An empty `Some` is not treated as `None`:
/// `irrep_nelec: Some(vec![])` lowers to `Some([])`, not the default `None`.
/// See [`ScfOptions::into_rhf_config`] for the mapping.
#[derive(Debug, Clone, Default)]
pub struct ScfOptions {
    /// [`RhfConfig::max_iter`].
    pub max_iter: Option<usize>,
    /// [`RhfConfig::energy_conv`].
    pub energy_conv: Option<f64>,
    /// [`RhfConfig::density_conv`].
    pub density_conv: Option<f64>,
    /// [`RhfConfig::diis_size`].
    pub diis_size: Option<usize>,
    /// [`RhfConfig::diis_flavor`].
    pub diis_flavor: Option<crate::diis::DiisFlavor>,
    /// [`RhfConfig::integral_thresh`].
    pub integral_thresh: Option<f64>,
    /// [`RhfConfig::k_builder`].
    pub k_builder: Option<KBuilderKind>,
    /// [`RhfConfig::cosx`].
    pub cosx: Option<crate::cosx_k::CosxConfig>,
    /// [`RhfConfig::df_j_aux`] (see [`AuxBasis::parse`]).
    pub df_j_aux: Option<AuxBasis>,
    /// [`RhfConfig::df_k_aux`] (see [`AuxBasis::parse`]).
    pub df_k_aux: Option<AuxBasis>,
    /// [`RhfConfig::ri_j_precision`].
    pub ri_j_precision: Option<crate::df_j::RiJPrecision>,
    /// [`RhfConfig::ri_k_precision`].
    pub ri_k_precision: Option<crate::df_k::RiKPrecision>,
    /// [`RhfConfig::level_shift`]. Setting it (to any value, 0 included) also
    /// clears [`RhfConfig::meta_gga_auto_level_shift`], so an explicit 0
    /// really runs unshifted.
    pub level_shift: Option<f64>,
    /// [`RhfConfig::mom_after_iter`].
    pub mom_after_iter: Option<usize>,
    /// [`RhfConfig::mom_reference`].
    pub mom_reference: Option<crate::mom::MomReference>,
    /// [`RhfConfig::smearing_sigma`].
    pub smearing_sigma: Option<f64>,
    /// Second-order SCF: `Some(true)` sets [`RhfConfig::newton_trigger`] to
    /// [`SOSCF_NEWTON_TRIGGER`], `Some(false)` to 0 (off).
    pub soscf: Option<bool>,
    /// [`RhfConfig::use_sad_guess`].
    pub guess: Option<Guess>,
    /// [`RhfConfig::guess_basis`].
    pub guess_basis: Option<crate::basis2::GuessBasis>,
    /// [`RhfConfig::init_guess_density`].
    pub init_guess_density: Option<Array2<f64>>,
    /// [`RhfConfig::external_potential`].
    pub external_potential: Option<ferric_core::external_potential::ExternalPotential>,
    /// [`RhfConfig::pcm`].
    pub pcm: Option<ferric_pcm::PcmConfig>,
    /// [`RhfConfig::three_index_budget_bytes`] (`None` = auto-detect).
    pub three_index_budget_bytes: Option<usize>,
    /// [`RhfConfig::check_stability`] and
    /// [`RhfConfig::scf_stability_descent`] (see [`Stability`]).
    pub stability: Option<Stability>,
    /// [`RhfConfig::symmetry`].
    pub symmetry: Option<bool>,
    /// [`RhfConfig::irrep_nelec`].
    pub irrep_nelec: Option<Vec<(String, usize)>>,
    /// [`RhfConfig::eri`].
    pub eri: Option<crate::incore_jk::EriMode>,
    /// [`RhfConfig::relativistic`].
    pub relativistic: Option<crate::x2c::Relativistic>,
    /// AURORA accelerator on/off ([`RhfConfig::aurora`]`.enabled`).
    pub aurora: Option<bool>,
    /// [`RhfConfig::trah_trigger`].
    pub trah_trigger: Option<TrahTrigger>,
    /// [`RhfConfig::xc`]: the exchange-correlation functional name (passed
    /// through verbatim; resolved by the KS solver). `None` keeps the
    /// default, Hartree-Fock.
    pub xc: Option<String>,
    /// [`RhfConfig::dft_grid`]: an explicit main KS grid. `None` keeps the
    /// default (`None`, the production grid on every KS path).
    pub dft_grid: Option<ferric_dft::grid::AtomicGridConfig>,
    /// [`RhfConfig::diis_switch_thresh`].
    pub diis_switch_thresh: Option<f64>,
    /// [`RhfConfig::screening`].
    pub screening: Option<crate::screening::ScreeningKind>,
    /// [`RhfConfig::verbose`].
    pub verbose: Option<bool>,
}

impl ScfOptions {
    /// Lower these options onto [`RhfConfig::default`] (also the
    /// `UhfConfig` / `RohfConfig` aliases). Pure: no I/O, no environment.
    ///
    /// Every field is validated by its type, so this lowering cannot fail
    /// today; the `Result` is the slot for cross-field rules a frontend
    /// lowering (the CLI `[scf]` section) brings. Rules that depend on the
    /// solver (e.g. `MomReference::Initial`, checked by
    /// `mom::check_reference` in the solver that implements it) are not
    /// repeated here.
    ///
    /// Fields of `RhfConfig` that `ScfOptions` does not expose keep their
    /// `RhfConfig::default()` value; adding a field to `RhfConfig` does not
    /// break this function (the test `every_rhf_config_field_is_classified`
    /// is where a new field must be classified).
    ///
    /// `ri_j_task` is [`RhfConfig::ri_j_task`], the task class an unset
    /// [`RhfConfig::ri_j_precision`] resolves against: the frontend derives
    /// it from the task it runs (it is not a user knob, so not an option).
    pub fn into_rhf_config(
        self,
        ri_j_task: crate::df_j::RiJTask,
    ) -> Result<RhfConfig, FerricError> {
        // Exhaustive destructure: a new ScfOptions field fails to compile
        // here until the lowering decides what it means.
        let ScfOptions {
            max_iter,
            energy_conv,
            density_conv,
            diis_size,
            diis_flavor,
            integral_thresh,
            k_builder,
            cosx,
            df_j_aux,
            df_k_aux,
            ri_j_precision,
            ri_k_precision,
            level_shift,
            mom_after_iter,
            mom_reference,
            smearing_sigma,
            soscf,
            guess,
            guess_basis,
            init_guess_density,
            external_potential,
            pcm,
            three_index_budget_bytes,
            stability,
            symmetry,
            irrep_nelec,
            eri,
            relativistic,
            aurora,
            trah_trigger,
            xc,
            dft_grid,
            diis_switch_thresh,
            screening,
            verbose,
        } = self;

        let mut c = RhfConfig::default();
        if let Some(v) = max_iter {
            c.max_iter = v;
        }
        if let Some(v) = energy_conv {
            c.energy_conv = v;
        }
        if let Some(v) = density_conv {
            c.density_conv = v;
        }
        if let Some(v) = diis_size {
            c.diis_size = v;
        }
        if let Some(v) = diis_flavor {
            c.diis_flavor = v;
        }
        if let Some(v) = integral_thresh {
            c.integral_thresh = v;
        }
        if k_builder.is_some() {
            c.k_builder = k_builder;
        }
        if let Some(v) = cosx {
            c.cosx = v;
        }
        if let Some(v) = df_j_aux {
            c.df_j_aux = Some(v.into_config_string());
        }
        if let Some(v) = df_k_aux {
            c.df_k_aux = Some(v.into_config_string());
        }
        if ri_j_precision.is_some() {
            c.ri_j_precision = ri_j_precision;
        }
        if ri_k_precision.is_some() {
            c.ri_k_precision = ri_k_precision;
        }
        if let Some(v) = level_shift {
            c.level_shift = v;
            c.meta_gga_auto_level_shift = false;
        }
        if let Some(v) = mom_after_iter {
            c.mom_after_iter = v;
        }
        if let Some(v) = mom_reference {
            c.mom_reference = v;
        }
        if smearing_sigma.is_some() {
            c.smearing_sigma = smearing_sigma;
        }
        if let Some(on) = soscf {
            c.newton_trigger = if on { SOSCF_NEWTON_TRIGGER } else { 0.0 };
        }
        if let Some(g) = guess {
            c.use_sad_guess = match g {
                Guess::Sad => true,
                Guess::Core => false,
            };
        }
        if guess_basis.is_some() {
            c.guess_basis = guess_basis;
        }
        if init_guess_density.is_some() {
            c.init_guess_density = init_guess_density;
        }
        if external_potential.is_some() {
            c.external_potential = external_potential;
        }
        if pcm.is_some() {
            c.pcm = pcm;
        }
        if let Some(v) = three_index_budget_bytes {
            c.three_index_budget_bytes = v;
        }
        if let Some(st) = stability {
            c.check_stability = st.checks();
            c.scf_stability_descent = st == Stability::Descend;
        }
        if let Some(v) = symmetry {
            c.symmetry = v;
        }
        if irrep_nelec.is_some() {
            c.irrep_nelec = irrep_nelec;
        }
        if let Some(v) = eri {
            c.eri = v;
        }
        if relativistic.is_some() {
            c.relativistic = relativistic;
        }
        if let Some(v) = aurora {
            c.aurora.enabled = v;
        }
        if let Some(t) = trah_trigger {
            c.trah_trigger = Some(t.get());
        }
        if xc.is_some() {
            c.xc = xc;
        }
        if dft_grid.is_some() {
            c.dft_grid = dft_grid;
        }
        if let Some(v) = diis_switch_thresh {
            c.diis_switch_thresh = v;
        }
        c.ri_j_task = ri_j_task;
        if let Some(v) = screening {
            c.screening = v;
        }
        if let Some(v) = verbose {
            c.verbose = v;
        }
        Ok(c)
    }
}

#[cfg(test)]
mod tests {
    //! The lowering must reproduce, field for field, the `RhfConfig` the
    //! Python `run_rhf` / `run_uhf` / `run_rohf` built with their own struct
    //! literals before the lowering existed. Each `old_*` fn below is that
    //! literal, copied verbatim with the Python-side parsers replaced by their
    //! (already parsed) outputs: it is the oracle. `RhfConfig` has no
    //! `PartialEq` (several nested types lack it), so configs are compared
    //! through their `Debug` text, which prints every field and every f64 in
    //! shortest round-trip form.
    use super::entry::{ScfEntryPoint, ScfKwargs};
    use super::*;
    use crate::cosx_k::CosxConfig;
    use crate::df_j::RiJPrecision;
    use crate::df_k::RiKPrecision;
    use crate::diis::DiisFlavor;
    use crate::incore_jk::EriMode;
    use crate::mom::MomReference;
    use crate::x2c::Relativistic;
    use ferric_core::external_potential::{ExternalPotential, PointCharge};
    use ferric_dft::grid::AtomicGridConfig;

    /// Every Python kwarg the three entry points take, already parsed, in the
    /// shape the old literals consumed them.
    #[derive(Clone, Default)]
    struct Kw {
        max_iter: Option<usize>,
        energy_conv: Option<f64>,
        density_conv: Option<f64>,
        diis_size: Option<usize>,
        integral_thresh: Option<f64>,
        k_builder: Option<&'static str>,
        cosx: Option<CosxConfig>,
        df_j_aux: Option<&'static str>,
        df_k_aux: Option<&'static str>,
        ri_j_precision: Option<RiJPrecision>,
        ri_k_precision: Option<RiKPrecision>,
        level_shift: Option<f64>,
        mom_after_iter: Option<usize>,
        mom_reference: Option<MomReference>,
        diis: Option<DiisFlavor>,
        smearing_sigma: Option<f64>,
        soscf: Option<bool>,
        guess_sad: Option<bool>,
        init_guess_density: Option<Array2<f64>>,
        external_potential: Option<ExternalPotential>,
        pcm: Option<ferric_pcm::PcmConfig>,
        budget_bytes: Option<usize>,
        stability_descent: Option<bool>,
        symmetry: bool,
        irrep_nelec: Option<Vec<(String, usize)>>,
        eri: Option<EriMode>,
        relativistic: Option<Relativistic>,
        aurora: Option<bool>,
        trah_trigger: Option<f64>,
        guess_basis: Option<crate::basis2::GuessBasis>,
        /// KS entry points only: `functional=`.
        functional: Option<&'static str>,
        /// KS entry points only: the resolved `grid_*` kwargs.
        dft_grid: Option<AtomicGridConfig>,
    }

    fn all_knobs() -> Kw {
        let cosx = CosxConfig {
            overlap_fit: !CosxConfig::default().overlap_fit,
            ..CosxConfig::default()
        };
        Kw {
            guess_basis: Some(crate::basis2::GuessBasis::bundled("sto-3g", "fock").unwrap()),
            max_iter: Some(77),
            energy_conv: Some(1e-7),
            density_conv: Some(3e-9),
            diis_size: Some(5),
            integral_thresh: Some(1e-11),
            k_builder: Some("cosx"),
            cosx: Some(cosx),
            df_j_aux: Some("def2-universal-jkfit"),
            df_k_aux: Some("Exact"),
            ri_j_precision: Some(RiJPrecision::F64),
            ri_k_precision: Some(RiKPrecision::F32),
            level_shift: Some(0.0),
            mom_after_iter: Some(4),
            mom_reference: Some(MomReference::Previous),
            diis: Some(DiisFlavor::Adiis),
            smearing_sigma: Some(0.01),
            soscf: Some(true),
            guess_sad: Some(false),
            init_guess_density: Some(Array2::from_shape_fn((3, 3), |(i, j)| {
                (i * 3 + j) as f64 * 0.125
            })),
            external_potential: Some(ExternalPotential {
                point_charges: vec![PointCharge {
                    q: -0.5,
                    x: 1.0,
                    y: 2.0,
                    z: 3.0,
                }],
                ..Default::default()
            }),
            pcm: Some(ferric_pcm::PcmConfig::default()),
            budget_bytes: Some(1 << 30),
            stability_descent: Some(true),
            symmetry: true,
            irrep_nelec: Some(vec![("A1".into(), 6), ("B2".into(), 2)]),
            eri: Some(EriMode::Direct),
            relativistic: Some(Relativistic::X2c1e),
            aurora: Some(true),
            trah_trigger: Some(1e-4),
            functional: Some("B3LYP"),
            dft_grid: Some(AtomicGridConfig {
                n_radial: 99,
                n_angular: 302,
                ..AtomicGridConfig::production()
            }),
        }
    }

    /// The `k_builder` spelling an oracle passed through verbatim, as the
    /// frontend now parses it (every spelling the oracles use is valid; the
    /// unknown ones are `k_builder_spellings_parse_once_with_the_solver_message`).
    fn kb(spelling: Option<&str>) -> Option<KBuilderKind> {
        KBuilderKind::parse_opt(spelling).unwrap()
    }

    // ── oracles: the pre-refactor literals ──

    fn old_run_rhf(k: Kw) -> RhfConfig {
        let descent = k.stability_descent.unwrap_or(false);
        let mut config = RhfConfig {
            max_iter: k.max_iter.unwrap_or_else(|| RhfConfig::default().max_iter),
            energy_conv: k.energy_conv.unwrap_or(1e-3),
            density_conv: k.density_conv.unwrap_or(1e-6),
            diis_size: k.diis_size.unwrap_or(8),
            integral_thresh: k.integral_thresh.unwrap_or(1e-12),
            k_builder: kb(k.k_builder),
            cosx: k.cosx.unwrap(),
            df_j_aux: k.df_j_aux.map(crate::rhf::normalize_df_aux),
            df_k_aux: k.df_k_aux.map(crate::rhf::normalize_df_aux),
            ri_j_precision: k.ri_j_precision,
            ri_k_precision: k.ri_k_precision,
            level_shift: k.level_shift.unwrap_or(0.0),
            meta_gga_auto_level_shift: k.level_shift.is_none(),
            mom_after_iter: k.mom_after_iter.unwrap_or(0),
            diis_flavor: k.diis.unwrap_or(DiisFlavor::Pulay),
            smearing_sigma: k.smearing_sigma,
            newton_trigger: if k.soscf.unwrap_or(false) { 1e-3 } else { 0.0 },
            use_sad_guess: k.guess_sad.unwrap_or(true),
            guess_basis: k.guess_basis,
            init_guess_density: k.init_guess_density,
            external_potential: k.external_potential,
            pcm: k.pcm,
            three_index_budget_bytes: k.budget_bytes.unwrap_or(0),
            check_stability: descent,
            scf_stability_descent: descent,
            symmetry: k.symmetry,
            irrep_nelec: k.irrep_nelec,
            eri: k.eri.unwrap_or(EriMode::Auto),
            relativistic: k.relativistic,
            ..Default::default()
        };
        // apply_accelerators
        config.aurora.enabled = k.aurora.unwrap_or(false);
        config.trah_trigger = k.trah_trigger;
        config
    }

    fn old_run_uhf(k: Kw) -> RhfConfig {
        let descent = k.stability_descent.unwrap_or(false);
        RhfConfig {
            max_iter: k.max_iter.unwrap_or_else(|| RhfConfig::default().max_iter),
            energy_conv: k.energy_conv.unwrap_or(1e-3),
            density_conv: k.density_conv.unwrap_or(1e-6),
            diis_size: k.diis_size.unwrap_or(8),
            integral_thresh: k.integral_thresh.unwrap_or(1e-12),
            k_builder: kb(k.k_builder),
            cosx: k.cosx.unwrap(),
            df_j_aux: k.df_j_aux.map(crate::rhf::normalize_df_aux),
            df_k_aux: k.df_k_aux.map(crate::rhf::normalize_df_aux),
            ri_j_precision: k.ri_j_precision,
            ri_k_precision: k.ri_k_precision,
            level_shift: k.level_shift.unwrap_or(0.0),
            meta_gga_auto_level_shift: k.level_shift.is_none(),
            mom_after_iter: k.mom_after_iter.unwrap_or(0),
            mom_reference: k.mom_reference.unwrap_or_default(),
            external_potential: k.external_potential,
            three_index_budget_bytes: k.budget_bytes.unwrap_or(0),
            use_sad_guess: k.guess_sad.unwrap_or(true),
            guess_basis: k.guess_basis,
            eri: k.eri.unwrap_or(EriMode::Auto),
            relativistic: k.relativistic,
            trah_trigger: k.trah_trigger,
            check_stability: descent,
            scf_stability_descent: descent,
            ..Default::default()
        }
    }

    fn old_run_rohf(k: Kw) -> RhfConfig {
        RhfConfig {
            max_iter: k.max_iter.unwrap_or_else(|| RhfConfig::default().max_iter),
            energy_conv: k.energy_conv.unwrap_or(1e-3),
            density_conv: k.density_conv.unwrap_or(1e-6),
            diis_size: k.diis_size.unwrap_or(8),
            integral_thresh: k.integral_thresh.unwrap_or(1e-12),
            k_builder: kb(k.k_builder),
            df_j_aux: k.df_j_aux.map(crate::rhf::normalize_df_aux),
            df_k_aux: k.df_k_aux.map(crate::rhf::normalize_df_aux),
            ri_j_precision: k.ri_j_precision,
            ri_k_precision: k.ri_k_precision,
            level_shift: k.level_shift.unwrap_or(0.0),
            meta_gga_auto_level_shift: k.level_shift.is_none(),
            mom_after_iter: k.mom_after_iter.unwrap_or(0),
            external_potential: k.external_potential,
            three_index_budget_bytes: k.budget_bytes.unwrap_or(0),
            guess_basis: k.guess_basis,
            eri: k.eri.unwrap_or(EriMode::Auto),
            relativistic: k.relativistic,
            ..Default::default()
        }
    }

    // ── the new path: the production policy, ScfEntryPoint::options ──
    //
    // `kwargs` is the one test-side step: it turns a `Kw` (the parsed
    // values of every kwarg) into the `ScfKwargs` the binding of `entry`
    // hands to `ScfEntryPoint::options` for a call that passes only the
    // kwargs that binding took before the lowering existed (the oracles'
    // surface): a kwarg it did not take then, including one added since
    // (roadmap 6b), stays at the value the binding passes when it is
    // omitted. Everything after that is the production code path. The 6b
    // kwargs are checked by `kwargs_added_in_6b_lower_to_their_fields`.

    fn guess(sad: Option<bool>) -> Guess {
        if sad.unwrap_or(true) {
            Guess::Sad
        } else {
            Guess::Core
        }
    }

    fn trah(t: Option<f64>) -> Option<TrahTrigger> {
        t.map(|v| TrahTrigger::new(v).unwrap())
    }

    /// The `functional=` an open-shell KS call passes (required there).
    const OPEN_KS_DEFAULT_FUNCTIONAL: &str = "PBE";

    fn kwargs(entry: ScfEntryPoint, k: &Kw) -> ScfKwargs<'static> {
        use ScfEntryPoint as E;
        let is = |es: &[E]| es.contains(&entry);
        let hf = !entry.is_kohn_sham();
        ScfKwargs {
            max_iter: k.max_iter,
            energy_conv: k.energy_conv,
            density_conv: k.density_conv,
            diis_size: if hf { k.diis_size } else { None },
            diis_flavor: is(&[E::Rhf]).then(|| k.diis.unwrap_or(DiisFlavor::Pulay)),
            integral_thresh: if hf { k.integral_thresh } else { None },
            k_builder: kb(k.k_builder),
            // The bindings always resolve the `cosx_*` kwargs: the default
            // COSX settings when none is passed.
            cosx: Some(if is(&[E::Rhf, E::Uhf, E::Dft]) {
                k.cosx.clone().unwrap_or_default()
            } else {
                CosxConfig::default()
            }),
            df_j_aux: k.df_j_aux,
            df_k_aux: k.df_k_aux,
            ri_j_precision: k.ri_j_precision,
            ri_k_precision: k.ri_k_precision,
            level_shift: k.level_shift,
            mom_after_iter: k.mom_after_iter,
            mom_reference: if is(&[E::Uhf]) { k.mom_reference } else { None },
            smearing_sigma: if is(&[E::Rhf]) {
                k.smearing_sigma
            } else {
                None
            },
            soscf: if is(&[E::Rhf]) { k.soscf } else { None },
            guess: Some(match entry {
                E::Rhf | E::Uhf | E::Uks => guess(k.guess_sad),
                E::Rohf | E::Dft | E::Roks => Guess::Sad,
            }),
            guess_basis: if is(&[E::Uks, E::Roks]) {
                None
            } else {
                k.guess_basis.clone()
            },
            init_guess_density: if is(&[E::Rhf, E::Dft]) {
                k.init_guess_density.clone()
            } else {
                None
            },
            external_potential: k.external_potential.clone(),
            pcm: if is(&[E::Rhf]) { k.pcm.clone() } else { None },
            three_index_budget_bytes: k.budget_bytes,
            stability_descent: if is(&[E::Rhf, E::Uhf]) {
                k.stability_descent
            } else {
                None
            },
            symmetry: is(&[E::Rhf, E::Dft]) && k.symmetry,
            irrep_nelec: if is(&[E::Rhf, E::Dft]) {
                k.irrep_nelec.clone()
            } else {
                None
            },
            eri: Some(if hf {
                k.eri.unwrap_or(EriMode::Auto)
            } else {
                EriMode::Auto
            }),
            relativistic: k.relativistic,
            aurora: if is(&[E::Rhf]) { k.aurora } else { None },
            trah_trigger: if is(&[E::Rhf, E::Uhf]) {
                trah(k.trah_trigger)
            } else {
                None
            },
            functional: match entry {
                E::Dft => k.functional,
                E::Uks | E::Roks => Some(k.functional.unwrap_or(OPEN_KS_DEFAULT_FUNCTIONAL)),
                _ => None,
            },
            dft_grid: if hf { None } else { k.dft_grid.clone() },
            // Post-SCF; no `RhfConfig` field to compare.
            dispersion: false,
        }
    }

    /// `entry` called with `k` (see [`kwargs`]), through the production
    /// policy and lowering.
    fn new_config(entry: ScfEntryPoint, k: &Kw) -> RhfConfig {
        entry
            .options(kwargs(entry, k))
            .unwrap()
            .into_rhf_config(crate::df_j::RiJTask::default())
            .unwrap()
    }

    fn new_run_rhf(k: Kw) -> RhfConfig {
        new_config(ScfEntryPoint::Rhf, &k)
    }

    fn new_run_uhf(k: Kw) -> RhfConfig {
        new_config(ScfEntryPoint::Uhf, &k)
    }

    fn new_run_rohf(k: Kw) -> RhfConfig {
        new_config(ScfEntryPoint::Rohf, &k)
    }

    /// Field-by-field equality via `Debug` text, except `guess_basis`: its
    /// `BasisSet` keeps shells in a `HashMap`, whose `Debug` order differs
    /// between two equal values, so it is compared with `PartialEq` instead.
    fn same_msg(a: &RhfConfig, b: &RhfConfig, msg: &str) {
        assert_eq!(a.guess_basis, b.guess_basis, "{msg}: guess_basis");
        let (mut a, mut b) = (a.clone(), b.clone());
        a.guess_basis = None;
        b.guess_basis = None;
        assert_eq!(format!("{a:#?}"), format!("{b:#?}"), "{msg}");
    }

    fn same(a: &RhfConfig, b: &RhfConfig) {
        same_msg(a, b, "lowering differs from the old literal");
    }

    /// The Python side always resolves `cosx=` for run_rhf/run_uhf (to the
    /// default CosxConfig when no cosx kwarg is given).
    fn defaults_with_cosx() -> Kw {
        Kw {
            cosx: Some(CosxConfig::default()),
            ..Kw::default()
        }
    }

    #[test]
    fn run_rhf_defaults_match_old_literal() {
        same(
            &old_run_rhf(defaults_with_cosx()),
            &new_run_rhf(defaults_with_cosx()),
        );
    }

    #[test]
    fn run_rhf_all_knobs_match_old_literal() {
        same(&old_run_rhf(all_knobs()), &new_run_rhf(all_knobs()));
    }

    #[test]
    fn run_uhf_defaults_match_old_literal() {
        same(
            &old_run_uhf(defaults_with_cosx()),
            &new_run_uhf(defaults_with_cosx()),
        );
    }

    #[test]
    fn run_uhf_all_knobs_match_old_literal() {
        same(&old_run_uhf(all_knobs()), &new_run_uhf(all_knobs()));
    }

    #[test]
    fn run_rohf_defaults_match_old_literal() {
        same(&old_run_rohf(Kw::default()), &new_run_rohf(Kw::default()));
    }

    #[test]
    fn run_rohf_all_knobs_match_old_literal() {
        same(&old_run_rohf(all_knobs()), &new_run_rohf(all_knobs()));
    }

    /// splitmix64: a seeded, dependency-free generator for the randomized
    /// equivalence test (no proptest/rand dev-dependency in the workspace).
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn coin(&mut self) -> bool {
            self.next() & 1 == 1
        }
        fn pick<T: Clone>(&mut self, xs: &[T]) -> T {
            xs[(self.next() % xs.len() as u64) as usize].clone()
        }
        fn maybe<T>(&mut self, f: impl FnOnce(&mut Self) -> T) -> Option<T> {
            if self.coin() {
                Some(f(self))
            } else {
                None
            }
        }
        /// A positive finite f64 spanning many decades.
        fn pos(&mut self) -> f64 {
            let mantissa = 1.0 + (self.next() % 1000) as f64 / 1000.0;
            mantissa * 10f64.powi((self.next() % 16) as i32 - 13)
        }
    }

    /// One random kwarg set: every knob independently unset or set to a
    /// value drawn from its domain (aux spellings include the opt-outs, in
    /// mixed case and with whitespace). `cosx` is always set, as the Python
    /// run_rhf/run_uhf always resolve it.
    fn random_kw(r: &mut Rng) -> Kw {
        const AUX: &[&str] = &[
            "def2-universal-jkfit",
            "  def2-svp-jkfit ",
            "Exact",
            "none",
            " OFF ",
            "",
            "conventional",
        ];
        let cosx = CosxConfig {
            overlap_fit: r.coin(),
            ..CosxConfig::default()
        };
        Kw {
            max_iter: r.maybe(|r| (r.next() % 500) as usize),
            energy_conv: r.maybe(Rng::pos),
            density_conv: r.maybe(Rng::pos),
            diis_size: r.maybe(|r| (r.next() % 20) as usize),
            integral_thresh: r.maybe(Rng::pos),
            k_builder: r.maybe(|r| r.pick(&["direct", "link", "cosx"])),
            cosx: Some(cosx),
            df_j_aux: r.maybe(|r| r.pick(AUX)),
            df_k_aux: r.maybe(|r| r.pick(AUX)),
            ri_j_precision: r.maybe(|r| r.pick(&[RiJPrecision::F32, RiJPrecision::F64])),
            ri_k_precision: r.maybe(|r| r.pick(&[RiKPrecision::F32, RiKPrecision::F64])),
            level_shift: r.maybe(|r| if r.coin() { 0.0 } else { r.pos() }),
            mom_after_iter: r.maybe(|r| (r.next() % 10) as usize),
            // `Initial` is refused by Python next to mom_after_iter/descent
            // before any config is built; `Previous` is always reachable.
            mom_reference: r.maybe(|_| MomReference::Previous),
            diis: r.maybe(|r| r.pick(&[DiisFlavor::Pulay, DiisFlavor::Adiis, DiisFlavor::Ediis])),
            smearing_sigma: r.maybe(Rng::pos),
            soscf: r.maybe(Rng::coin),
            guess_sad: r.maybe(Rng::coin),
            init_guess_density: r.maybe(|r| {
                let x = r.pos();
                Array2::from_shape_fn((2, 2), |(i, j)| x * (i + 2 * j) as f64)
            }),
            external_potential: r.maybe(|r| ExternalPotential {
                point_charges: vec![PointCharge {
                    q: -r.pos(),
                    x: r.pos(),
                    y: 0.0,
                    z: 1.0,
                }],
                field: if r.coin() {
                    Some([r.pos(), 0.0, 0.0])
                } else {
                    None
                },
                ..Default::default()
            }),
            pcm: r.maybe(|_| ferric_pcm::PcmConfig::default()),
            budget_bytes: r.maybe(|r| (r.next() % (1 << 34)) as usize + 1),
            stability_descent: r.maybe(Rng::coin),
            symmetry: r.coin(),
            irrep_nelec: r.maybe(|r| vec![("A1".into(), (r.next() % 9) as usize)]),
            eri: r.maybe(|r| r.pick(&[EriMode::Auto, EriMode::Direct, EriMode::Incore])),
            relativistic: r.maybe(|_| Relativistic::X2c1e),
            aurora: r.maybe(Rng::coin),
            trah_trigger: r.maybe(Rng::pos),
            guess_basis: r.maybe(|r| {
                let proj = r.pick(&["fock", "overlap"]);
                crate::basis2::GuessBasis::bundled("sto-3g", proj).unwrap()
            }),
            functional: r.maybe(|r| r.pick(&["PBE", "B3LYP", "wB97X-V", "TPSS", "LDA"])),
            dft_grid: r.maybe(|r| AtomicGridConfig {
                n_radial: 50 + (r.next() % 50) as usize,
                n_angular: r.pick(&[110, 194, 302]),
                prune: if r.coin() {
                    None
                } else {
                    AtomicGridConfig::production().prune
                },
                ..AtomicGridConfig::production()
            }),
        }
    }

    /// 256 seeded random kwarg sets per entry point: the lowering equals the
    /// pre-refactor literal (the `old_*` oracles) for each.
    #[test]
    fn random_kwargs_match_old_literals() {
        let mut r = Rng(0x5eed_f00d_cafe_0001);
        for case in 0..256 {
            let k = random_kw(&mut r);
            let msg = |who: &str| format!("{who}, case {case}");
            let (a, b) = (old_run_rhf(k.clone()), new_run_rhf(k.clone()));
            same_msg(&a, &b, &msg("run_rhf"));
            let (a, b) = (old_run_uhf(k.clone()), new_run_uhf(k.clone()));
            same_msg(&a, &b, &msg("run_uhf"));
            let (a, b) = (old_run_rohf(k.clone()), new_run_rohf(k));
            same_msg(&a, &b, &msg("run_rohf"));
        }
    }

    // ── Kohn-Sham entry points (run_dft / run_uks / run_roks) ──
    //
    // Oracles: the pre-lowering construction in `ferric-python/src/dft.rs`
    // (`run_dft` and the shared `run_open_ks` body), copied verbatim with the
    // two `ferric-python` helpers it called (`rhf_config_budgeted`,
    // `resolve_df_aux`) inlined below as they were.

    fn old_rhf_config_budgeted(k_builder: Option<&str>, budget_bytes: Option<usize>) -> RhfConfig {
        RhfConfig {
            k_builder: kb(k_builder),
            three_index_budget_bytes: budget_bytes.unwrap_or(0),
            ..Default::default()
        }
    }

    fn old_resolve_df_aux(requested: Option<&str>, default_aux: &str) -> Option<String> {
        match requested {
            None => Some(default_aux.to_string()),
            Some(v) => Some(crate::rhf::normalize_df_aux(v)),
        }
    }

    /// `run_dft` before the lowering. `k.cosx` is what `resolve_cosx_kwargs`
    /// returned (always `Some` there); `k.symmetry` the `symmetry=` kwarg.
    fn old_run_dft(k: Kw) -> RhfConfig {
        let k_builder = k.k_builder;
        let (max_iter, energy_conv, density_conv) = (k.max_iter, k.energy_conv, k.density_conv);
        let (level_shift, mom_after_iter) = (k.level_shift, k.mom_after_iter);
        let (df_j_aux, df_k_aux) = (k.df_j_aux, k.df_k_aux);
        let mut cfg = old_rhf_config_budgeted(k_builder, k.budget_bytes);
        cfg.max_iter = max_iter.unwrap_or(cfg.max_iter);
        cfg.energy_conv = energy_conv.unwrap_or(cfg.energy_conv);
        cfg.density_conv = density_conv.unwrap_or(cfg.density_conv);
        cfg.level_shift = level_shift.unwrap_or(cfg.level_shift);
        cfg.meta_gga_auto_level_shift = level_shift.is_none();
        cfg.mom_after_iter = mom_after_iter.unwrap_or(cfg.mom_after_iter);
        cfg.external_potential = k.external_potential;
        let xc_name = k.functional.unwrap_or("LDA").to_string();
        cfg.xc = Some(xc_name.clone());
        cfg.guess_basis = k.guess_basis;
        cfg.relativistic = k.relativistic;
        cfg.init_guess_density = k.init_guess_density;
        cfg.df_j_aux = old_resolve_df_aux(df_j_aux, "def2-universal-jkfit");
        cfg.ri_j_precision = k.ri_j_precision;
        cfg.ri_k_precision = k.ri_k_precision;
        cfg.df_k_aux = if k_builder == Some("cosx") && df_k_aux.is_none() {
            None
        } else {
            old_resolve_df_aux(df_k_aux, "def2-universal-jkfit")
        };
        cfg.cosx = k.cosx.unwrap();
        cfg.dft_grid = k.dft_grid;
        if k.symmetry || k.irrep_nelec.is_some() {
            cfg.symmetry = true;
            cfg.irrep_nelec = k.irrep_nelec;
        }
        cfg
    }

    /// `run_open_ks` (the body of `run_uks` and `run_roks`) before the
    /// lowering. `use_sad_guess` is `parse_guess(guess)`: `run_roks` passes
    /// `guess = None`, so `true`.
    fn old_run_open_ks(k: Kw, functional: &str, use_sad_guess: bool) -> RhfConfig {
        let (max_iter, energy_conv, density_conv) = (k.max_iter, k.energy_conv, k.density_conv);
        let (level_shift, mom_after_iter) = (k.level_shift, k.mom_after_iter);
        let mut config = old_rhf_config_budgeted(k.k_builder, k.budget_bytes);
        config.max_iter = max_iter.unwrap_or(config.max_iter);
        config.energy_conv = energy_conv.unwrap_or(config.energy_conv);
        config.density_conv = density_conv.unwrap_or(config.density_conv);
        config.level_shift = level_shift.unwrap_or(config.level_shift);
        config.meta_gga_auto_level_shift = level_shift.is_none();
        config.mom_after_iter = mom_after_iter.unwrap_or(config.mom_after_iter);
        config.use_sad_guess = use_sad_guess;
        config.external_potential = k.external_potential;
        config.xc = Some(functional.to_string());
        config.relativistic = k.relativistic;
        config.df_j_aux = old_resolve_df_aux(k.df_j_aux, "def2-universal-jkfit");
        config.df_k_aux = old_resolve_df_aux(k.df_k_aux, "def2-universal-jkfit");
        config.ri_j_precision = k.ri_j_precision;
        config.ri_k_precision = k.ri_k_precision;
        config.dft_grid = k.dft_grid;
        config
    }

    fn new_run_dft(k: Kw) -> RhfConfig {
        new_config(ScfEntryPoint::Dft, &k)
    }

    /// `run_uks` with `functional` and `guess` given by `use_sad_guess`.
    fn new_run_open_ks(k: Kw, functional: &'static str, use_sad_guess: bool) -> RhfConfig {
        let k = Kw {
            functional: Some(functional),
            guess_sad: Some(use_sad_guess),
            ..k
        };
        new_config(ScfEntryPoint::Uks, &k)
    }

    /// One KS call checked against its oracle: `run_dft` as given, `run_uks`
    /// with and without `guess="hcore"` and `run_roks` (guess always MINAO),
    /// for `functional` (default `"PBE"`; required there).
    fn check_ks(k: &Kw, msg: &str) {
        if k.irrep_nelec.is_some() && !k.symmetry {
            // The binding refuses this before the policy runs
            // ("run_dft: irrep_nelec requires symmetry=True"); the old
            // construction promoted it, the policy refuses it too.
            let entry = ScfEntryPoint::Dft;
            assert_eq!(
                entry.options(kwargs(entry, k)).unwrap_err(),
                entry::EntryOptionError::IrrepNelecWithoutSymmetry(entry),
                "{msg}"
            );
        } else {
            same_msg(
                &old_run_dft(k.clone()),
                &new_run_dft(k.clone()),
                &format!("run_dft, {msg}"),
            );
        }
        let f = k.functional.unwrap_or(OPEN_KS_DEFAULT_FUNCTIONAL);
        let open = [
            (ScfEntryPoint::Uks, Some(true)),
            (ScfEntryPoint::Uks, Some(false)),
            (ScfEntryPoint::Uks, None),
            (ScfEntryPoint::Roks, k.guess_sad),
        ];
        for (entry, guess_sad) in open {
            let k = Kw {
                functional: Some(f),
                guess_sad,
                ..k.clone()
            };
            let sad = entry == ScfEntryPoint::Roks || guess_sad.unwrap_or(true);
            let old = old_run_open_ks(k.clone(), f, sad);
            let new = new_config(entry, &k);
            let who = format!("{} (guess_sad={guess_sad:?}), {msg}", entry.name());
            if open_ks_takes_rijcosx_rule(&k) {
                // The one intended change (see
                // `open_ks_cosx_drops_the_ri_k_default`): everything but
                // df_k_aux is the old construction.
                assert_eq!(new.df_k_aux, None, "{who}");
                let old = RhfConfig {
                    df_k_aux: None,
                    ..old
                };
                same_msg(&old, &new, &format!("cosx: {who}"));
            } else {
                same_msg(&old, &new, &who);
            }
        }
    }

    /// Where `run_uks` / `run_roks` now differ from their old construction:
    /// `k_builder = "cosx"` with no explicit `df_k_aux`. The old body applied
    /// the RI-K default regardless, so COSX always met a df_k_aux conflict
    /// the caller never set; it now takes the RIJCOSX rule, as `run_dft`.
    fn open_ks_takes_rijcosx_rule(k: &Kw) -> bool {
        k.k_builder == Some("cosx") && k.df_k_aux.is_none()
    }

    #[test]
    fn open_ks_cosx_drops_the_ri_k_default() {
        let k = Kw {
            k_builder: Some("cosx"),
            ..Kw::default()
        };
        for sad in [true, false] {
            let c = new_run_open_ks(k.clone(), "PBE0", sad);
            assert_eq!(c.k_builder, Some(KBuilderKind::Cosx));
            assert_eq!(c.df_k_aux, None);
            assert_eq!(c.df_j_aux.as_deref(), Some(DEFAULT_JK_AUX));
            // The old construction kept the RI-K default (the bug).
            let old = old_run_open_ks(k.clone(), "PBE0", sad);
            assert_eq!(old.df_k_aux.as_deref(), Some(DEFAULT_JK_AUX));
        }
        // An explicit df_k_aux still wins (and stays the SCF's conflict).
        let k = Kw {
            df_k_aux: Some("def2-universal-jkfit"),
            ..k
        };
        assert_eq!(
            new_run_open_ks(k, "PBE0", true).df_k_aux.as_deref(),
            Some(DEFAULT_JK_AUX)
        );
        // Without COSX the RI-K default is unchanged.
        for kb in [None, Some("direct"), Some("link")] {
            let k = Kw {
                k_builder: kb,
                ..Kw::default()
            };
            let c = new_run_open_ks(k, "PBE0", true);
            assert_eq!(c.df_k_aux.as_deref(), Some(DEFAULT_JK_AUX), "{kb:?}");
        }
    }

    /// `KBuilderKind::parse` is the one home of the `k_builder` whitelist:
    /// exactly the three spellings, case-sensitive, untrimmed, and every
    /// other spelling refused with the message the solvers raised before the
    /// frontends parsed it (`FerricError::General`, so Python's text is
    /// unchanged: "General error: unknown k_builder ...").
    #[test]
    fn k_builder_spellings_parse_once_with_the_solver_message() {
        for kind in KBuilderKind::ALL {
            assert_eq!(KBuilderKind::parse(kind.as_str()), Ok(kind));
            assert_eq!(kind.as_str().parse::<KBuilderKind>(), Ok(kind));
            assert_eq!(
                KBuilderKind::spells_cosx(kind.as_str()),
                kind == KBuilderKind::Cosx
            );
            assert_eq!(kind.to_string(), kind.as_str());
        }
        assert_eq!(KBuilderKind::parse_opt(None), Ok(None));
        for bad in [
            "Link", "COSX", "Direct", " cosx", "cosx ", "", "bogus", "cosxx", "exact",
        ] {
            let e = KBuilderKind::parse(bad).unwrap_err();
            assert_eq!(e, UnknownKBuilder(bad.to_string()));
            let text =
                format!("unknown k_builder '{bad}': valid options are 'direct', 'link', 'cosx'");
            assert_eq!(e.to_string(), text);
            assert_eq!(
                FerricError::from(e).to_string(),
                format!("General error: {text}")
            );
            assert_eq!(KBuilderKind::parse_opt(Some(bad)).unwrap_err().0, bad);
            assert!(!KBuilderKind::spells_cosx(bad), "{bad:?}");
        }
    }

    #[test]
    fn ks_defaults_match_old_construction() {
        check_ks(&defaults_with_cosx(), "defaults");
        // run_dft with k_builder="cosx" and nothing else: RIJCOSX, no RI-K.
        let k = Kw {
            k_builder: Some("cosx"),
            ..defaults_with_cosx()
        };
        check_ks(&k, "cosx defaults");
        let c = new_run_dft(k);
        assert_eq!(c.df_k_aux, None);
        assert_eq!(c.df_j_aux.as_deref(), Some(DEFAULT_JK_AUX));
    }

    #[test]
    fn ks_all_knobs_match_old_construction() {
        check_ks(&all_knobs(), "all knobs");
        // all_knobs() sets symmetry; also the unsymmetric all-knobs call.
        let k = Kw {
            symmetry: false,
            irrep_nelec: None,
            ..all_knobs()
        };
        check_ks(&k, "all knobs, no symmetry");
    }

    /// 256 seeded random kwarg sets: run_dft / run_uks / run_roks equal
    /// their pre-lowering construction for each.
    #[test]
    fn ks_random_kwargs_match_old_construction() {
        let mut r = Rng(0x5eed_f00d_cafe_0006);
        for case in 0..256 {
            let k = random_kw(&mut r);
            check_ks(&k, &format!("case {case}"));
        }
    }

    /// The kwargs roadmap 6b added to each entry point (the drift a sibling
    /// with the same solver already took), each set to a non-default value:
    /// every one reaches its `RhfConfig` field.
    #[test]
    fn kwargs_added_in_6b_lower_to_their_fields() {
        use ScfEntryPoint as E;
        let added: [(E, &[&str]); 5] = [
            (E::Uhf, &["soscf", "smeared_charges", "solvent"]),
            (
                E::Rohf,
                &["guess", "soscf", "smeared_charges", "solvent", "cosx"],
            ),
            (
                E::Dft,
                &[
                    "diis_size",
                    "integral_thresh",
                    "eri",
                    "guess",
                    "smeared_charges",
                    "solvent",
                    "stability_descent",
                    "aurora",
                    "trah_trigger",
                ],
            ),
            (
                E::Uks,
                &[
                    "diis_size",
                    "integral_thresh",
                    "eri",
                    "soscf",
                    "smeared_charges",
                    "solvent",
                    "stability_descent",
                    "cosx",
                    "trah_trigger",
                    "guess_basis",
                    "mom_reference",
                ],
            ),
            (
                E::Roks,
                &[
                    "diis_size",
                    "integral_thresh",
                    "eri",
                    "guess",
                    "soscf",
                    "smeared_charges",
                    "solvent",
                    "cosx",
                    "guess_basis",
                ],
            ),
        ];
        let flipped_cosx = CosxConfig {
            overlap_fit: !CosxConfig::default().overlap_fit,
            ..CosxConfig::default()
        };
        let smeared = ExternalPotential {
            smeared_charges: vec![ferric_core::external_potential::SmearedCharge {
                q: -0.5,
                x: 0.0,
                y: 0.0,
                z: 4.0,
                width: 1.0,
            }],
            ..Default::default()
        };
        let gb = crate::basis2::GuessBasis::bundled("sto-3g", "fock").unwrap();
        for (entry, names) in added {
            let has = |n: &str| names.contains(&n);
            // Every listed name must be set below and checked below, so a
            // misspelt or unhandled name fails instead of passing vacuously.
            let mut set: Vec<&str> = Vec::new();
            let checked = std::cell::RefCell::new(Vec::<&str>::new());
            let mut kw = kwargs(entry, &Kw::default());
            if has("diis_size") {
                set.push("diis_size");
                kw.diis_size = Some(5);
            }
            if has("integral_thresh") {
                set.push("integral_thresh");
                kw.integral_thresh = Some(1e-10);
            }
            if has("eri") {
                set.push("eri");
                kw.eri = Some(EriMode::Direct);
            }
            if has("guess") {
                set.push("guess");
                kw.guess = Some(Guess::Core);
            }
            if has("soscf") {
                set.push("soscf");
                kw.soscf = Some(true);
            }
            if has("smeared_charges") {
                set.push("smeared_charges");
                kw.external_potential = Some(smeared.clone());
            }
            if has("solvent") {
                set.push("solvent");
                kw.pcm = Some(ferric_pcm::PcmConfig::default());
            }
            if has("stability_descent") {
                set.push("stability_descent");
                kw.stability_descent = Some(true);
            }
            if has("cosx") {
                set.push("cosx");
                kw.cosx = Some(flipped_cosx.clone());
            }
            if has("aurora") {
                set.push("aurora");
                kw.aurora = Some(true);
            }
            if has("trah_trigger") {
                set.push("trah_trigger");
                kw.trah_trigger = trah(Some(0.1));
            }
            if has("guess_basis") {
                set.push("guess_basis");
                kw.guess_basis = Some(gb.clone());
            }
            if has("mom_reference") {
                set.push("mom_reference");
                kw.mom_reference = Some(MomReference::Initial);
            }
            let c = entry
                .options(kw)
                .unwrap()
                .into_rhf_config(crate::df_j::RiJTask::default())
                .unwrap();
            let d = RhfConfig::default();
            let who = entry.name();
            let check = |n: &'static str, ok: bool| {
                if has(n) {
                    checked.borrow_mut().push(n);
                    assert!(ok, "{who}: {n}");
                }
            };
            check("diis_size", c.diis_size == 5 && d.diis_size != 5);
            check("integral_thresh", c.integral_thresh == 1e-10);
            check("eri", c.eri == EriMode::Direct && d.eri != EriMode::Direct);
            check("guess", !c.use_sad_guess && d.use_sad_guess);
            check("soscf", c.newton_trigger == SOSCF_NEWTON_TRIGGER);
            check(
                "smeared_charges",
                c.external_potential.as_ref() == Some(&smeared),
            );
            check("solvent", c.pcm.is_some());
            check(
                "stability_descent",
                c.check_stability && c.scf_stability_descent,
            );
            check(
                "cosx",
                format!("{:?}", c.cosx) == format!("{flipped_cosx:?}"),
            );
            check("aurora", c.aurora.enabled);
            check("trah_trigger", c.trah_trigger == Some(0.1));
            check("guess_basis", c.guess_basis.as_ref() == Some(&gb));
            check("mom_reference", c.mom_reference == MomReference::Initial);
            let mut want: Vec<&str> = names.to_vec();
            want.sort_unstable();
            set.sort_unstable();
            let mut checked = checked.into_inner();
            checked.sort_unstable();
            assert_eq!(set, want, "{who}: names with no setter");
            assert_eq!(checked, want, "{who}: names with no check");
            assert!(c.xc.is_some() == entry.is_kohn_sham(), "{who}");
        }
    }

    #[test]
    fn rijcosx_rule_drops_only_the_ri_k_default() {
        let named = |s: &str| Some(AuxBasis::named(s));
        let d = named(DEFAULT_JK_AUX);
        assert_eq!(
            df_k_aux_or_default(Some(KBuilderKind::Cosx), None, d.clone()),
            None
        );
        assert_eq!(
            df_k_aux_or_default(Some(KBuilderKind::Cosx), named("x"), d.clone()),
            named("x")
        );
        assert_eq!(
            df_k_aux_or_default(Some(KBuilderKind::Cosx), Some(AuxBasis::Off), d.clone()),
            Some(AuxBasis::Off)
        );
        for kb in [None, Some(KBuilderKind::Direct), Some(KBuilderKind::Link)] {
            assert_eq!(df_k_aux_or_default(kb, None, d.clone()), d, "{kb:?}");
            assert_eq!(df_k_aux_or_default::<String>(kb, None, None), None);
        }
    }

    #[test]
    fn aux_basis_parse_maps_opt_outs_to_off() {
        for s in ["", "  ", "none", "OFF", " Exact ", "conventional"] {
            assert_eq!(AuxBasis::parse(s), AuxBasis::Off, "{s:?}");
        }
        assert_eq!(
            AuxBasis::parse(" def2-universal-jkfit "),
            AuxBasis::Named("def2-universal-jkfit".into())
        );
        // `named` normalises exactly as `parse` does.
        for s in [" def2-svp-jkfit ", "EXACT", "", DEFAULT_JK_AUX] {
            assert_eq!(AuxBasis::named(s), AuxBasis::parse(s), "{s:?}");
        }
        assert_eq!(
            AuxBasis::default_jk(),
            AuxBasis::Named(DEFAULT_JK_AUX.into())
        );
    }

    #[test]
    fn stability_checks_unless_off() {
        assert!(!Stability::Off.checks());
        assert!(Stability::Check.checks());
        assert!(Stability::Descend.checks());
    }

    #[test]
    fn explicit_level_shift_disables_meta_gga_auto_shift() {
        let c = ScfOptions {
            level_shift: Some(0.0),
            ..Default::default()
        }
        .into_rhf_config(crate::df_j::RiJTask::default())
        .unwrap();
        assert!(!c.meta_gga_auto_level_shift);
        assert!(RhfConfig::default().meta_gga_auto_level_shift);
    }

    #[test]
    fn named_defaults_are_the_rhf_config_defaults() {
        let d = RhfConfig::default();
        assert_eq!(d.max_iter, DEFAULT_SCF_MAX_ITER);
        assert_eq!(d.energy_conv, DEFAULT_SCF_ENERGY_CONV);
        assert_eq!(d.density_conv, DEFAULT_SCF_DENSITY_CONV);
        assert_eq!(d.diis_size, DEFAULT_DIIS_SIZE);
        assert_eq!(d.integral_thresh, DEFAULT_INTEGRAL_THRESH);
        same(
            &ScfOptions::default()
                .into_rhf_config(crate::df_j::RiJTask::default())
                .unwrap(),
            &d,
        );
    }

    #[test]
    fn trah_trigger_rejects_non_positive_and_non_finite() {
        for v in [0.0, -1e-3, f64::NAN, f64::INFINITY] {
            let e = TrahTrigger::new(v).unwrap_err();
            assert_eq!(
                e.to_string(),
                format!("trah_trigger must be a positive number, got {v}")
            );
        }
        assert_eq!(TrahTrigger::new(1e-4).unwrap().get(), 1e-4);
    }

    #[test]
    fn stability_lowers_to_check_and_descent_flags() {
        let lower = |st| {
            let c = ScfOptions {
                stability: Some(st),
                ..Default::default()
            }
            .into_rhf_config(crate::df_j::RiJTask::default())
            .unwrap();
            (c.check_stability, c.scf_stability_descent)
        };
        assert_eq!(lower(Stability::Off), (false, false));
        assert_eq!(lower(Stability::Check), (true, false));
        assert_eq!(lower(Stability::Descend), (true, true));
        let d = RhfConfig::default();
        assert_eq!((d.check_stability, d.scf_stability_descent), (false, false));
    }

    /// The knobs only the CLI `[scf]` section sets today each lower to their
    /// one `RhfConfig` field.
    #[test]
    fn cli_knobs_lower_to_their_config_fields() {
        use crate::df_j::RiJTask;
        use crate::screening::ScreeningKind;
        let c = ScfOptions {
            diis_switch_thresh: Some(3e-2),
            screening: Some(ScreeningKind::Csb),
            verbose: Some(true),
            ..Default::default()
        }
        .into_rhf_config(RiJTask::Geometry)
        .unwrap();
        let expected = RhfConfig {
            diis_switch_thresh: 3e-2,
            ri_j_task: RiJTask::Geometry,
            screening: ScreeningKind::Csb,
            verbose: true,
            ..RhfConfig::default()
        };
        same(&c, &expected);
    }

    /// Exhaustive destructure of `RhfConfig`: adding a field fails to compile
    /// this test (the lowering itself still compiles, since it starts from
    /// `RhfConfig::default()`), forcing a decision on whether `ScfOptions`
    /// should expose it.
    /// Fields marked `lowered` are set by [`ScfOptions::into_rhf_config`];
    /// the rest keep `RhfConfig::default()`.
    #[test]
    fn every_rhf_config_field_is_classified() {
        let RhfConfig {
            max_iter: _,                  // lowered
            energy_conv: _,               // lowered
            density_conv: _,              // lowered
            diis_size: _,                 // lowered
            diis_flavor: _,               // lowered
            diis_switch_thresh: _,        // lowered
            smearing_sigma: _,            // lowered
            integral_thresh: _,           // lowered
            k_builder: _,                 // lowered
            cosx: _,                      // lowered
            df_j_aux: _,                  // lowered
            df_k_aux: _,                  // lowered
            ri_j_precision: _,            // lowered
            ri_j_task: _,                 // lowered (CLI task class; geometry drivers also set it)
            ri_k_precision: _,            // lowered
            xc: _,                        // lowered
            xc_omega: _,                  // default
            dft_grid: _,                  // lowered
            nlc_grid: _,                  // default
            level_shift: _,               // lowered
            meta_gga_auto_level_shift: _, // lowered (from level_shift)
            newton_trigger: _,            // lowered (from soscf)
            aurora: _,                    // lowered (.enabled)
            ah_trigger: _,                // default
            trah_trigger: _,              // lowered
            trah: _,                      // default
            mom_after_iter: _,            // lowered
            mom_reference: _,             // lowered
            rohf_occupation_guard: _,     // default
            constraints: _,               // default (cDFT)
            cdft_lambda_tol: _,           // default (cDFT)
            cdft_max_outer: _,            // default (cDFT)
            cdft_lambda_init: _,          // default (cDFT)
            cdft_stability_descent: _,    // default (cDFT)
            scf_stability_descent: _,     // lowered (stability_descent)
            fractional_occ: _,            // default
            three_index_budget_bytes: _,  // lowered
            init_guess_density: _,        // lowered
            guess_basis: _,               // lowered
            use_sad_guess: _,             // lowered
            stall_window: _,              // default (ladder)
            divergence_tol: _,            // default (ladder)
            external_potential: _,        // lowered
            cosmo: _,                     // default
            pcm: _,                       // lowered
            polarizable: _,               // default (QM/MM)
            verbose: _,                   // lowered
            check_stability: _,           // lowered (stability_descent)
            symmetry: _,                  // lowered
            irrep_nelec: _,               // lowered
            screening: _,                 // lowered
            raw_handoff: _,               // default
            eri: _,                       // lowered
            gc_rotation: _,               // default (not an entry-point knob)
            eri_route: _,                 // default (solver output)
            relativistic: _,              // lowered
        } = RhfConfig::default();
    }

    /// The struct-update literals of the Python bindings that roadmap item
    /// 11(a) moved onto `ScfOptions` (the `old_*` closures are those
    /// literals verbatim with their inputs parsed; the `new` sides mirror the
    /// bindings): `rhf_config_budgeted`, `run_saddle` / `run_irc` /
    /// `run_frequencies` (xc + MM field), `run_optimize` (MM field + PCM) and
    /// `run_cdft`.
    #[test]
    fn python_struct_update_sites_lower_through_options() {
        let same = |old: &RhfConfig, new: RhfConfig| {
            assert_eq!(format!("{old:?}"), format!("{new:?}"));
        };
        let lower = |o: ScfOptions| o.into_rhf_config(crate::df_j::RiJTask::Energy).unwrap();
        let ext = all_knobs().external_potential;

        // rhf_config_budgeted
        for (k_builder, budget) in [(None, None), (Some(KBuilderKind::Link), Some(1usize << 30))] {
            let old = RhfConfig {
                k_builder,
                three_index_budget_bytes: budget.unwrap_or(0),
                ..Default::default()
            };
            same(
                &old,
                lower(ScfOptions {
                    k_builder,
                    three_index_budget_bytes: budget,
                    ..Default::default()
                }),
            );
        }

        for (xc, ext, pcm) in [
            (None, None, None),
            (
                Some("B3LYP"),
                ext.clone(),
                Some(ferric_pcm::PcmConfig::default()),
            ),
        ] {
            // run_saddle / run_irc / run_frequencies
            let old = RhfConfig {
                xc: xc.map(|s| s.to_string()),
                external_potential: ext.clone(),
                ..Default::default()
            };
            same(
                &old,
                lower(ScfOptions {
                    xc: xc.map(|s| s.to_string()),
                    external_potential: ext.clone(),
                    ..Default::default()
                }),
            );
            // run_optimize
            let old = RhfConfig {
                external_potential: ext.clone(),
                pcm: pcm.clone(),
                ..Default::default()
            };
            same(
                &old,
                lower(ScfOptions {
                    external_potential: ext.clone(),
                    pcm: pcm.clone(),
                    ..Default::default()
                }),
            );
        }

        // run_cdft: every kwarg unset, then every kwarg set (an off spelling
        // for one auxiliary basis, a meta-GGA-relevant explicit level shift).
        struct Cdft {
            xc: Option<String>,
            dft_grid: Option<AtomicGridConfig>,
            max_iter: Option<usize>,
            energy_conv: Option<f64>,
            density_conv: Option<f64>,
            diis_size: Option<usize>,
            integral_thresh: Option<f64>,
            k_builder: Option<KBuilderKind>,
            df_j_aux: Option<&'static str>,
            df_k_aux: Option<&'static str>,
            ri_j_precision: Option<RiJPrecision>,
            ri_k_precision: Option<RiKPrecision>,
            level_shift: Option<f64>,
            mom_after_iter: Option<usize>,
            external_potential: Option<ExternalPotential>,
            budget_bytes: Option<usize>,
            use_sad_guess: bool,
            constraints: Vec<ferric_dft::cdft::Constraint>,
            lambda_tol: f64,
            max_outer: usize,
            stability_descent: Option<bool>,
        }
        let unset = Cdft {
            xc: None,
            dft_grid: None,
            max_iter: None,
            energy_conv: None,
            density_conv: None,
            diis_size: None,
            integral_thresh: None,
            k_builder: None,
            df_j_aux: None,
            df_k_aux: None,
            ri_j_precision: None,
            ri_k_precision: None,
            level_shift: None,
            mom_after_iter: None,
            external_potential: None,
            budget_bytes: None,
            use_sad_guess: true,
            constraints: Vec::new(),
            lambda_tol: 1e-5,
            max_outer: 30,
            stability_descent: None,
        };
        let set = Cdft {
            xc: Some("TPSS".to_string()),
            dft_grid: Some(AtomicGridConfig {
                n_radial: 75,
                n_angular: 302,
                ..Default::default()
            }),
            max_iter: Some(77),
            energy_conv: Some(1e-7),
            density_conv: Some(3e-9),
            diis_size: Some(5),
            integral_thresh: Some(1e-11),
            k_builder: Some(KBuilderKind::Cosx),
            df_j_aux: Some(" def2-universal-jkfit "),
            df_k_aux: Some("OFF"),
            ri_j_precision: Some(RiJPrecision::F64),
            ri_k_precision: Some(RiKPrecision::F32),
            level_shift: Some(0.1),
            mom_after_iter: Some(4),
            external_potential: ext,
            budget_bytes: Some(1 << 30),
            use_sad_guess: false,
            constraints: vec![ferric_dft::cdft::Constraint {
                fragment: vec![0],
                spin: ferric_dft::cdft::SpinChannel::Total,
                target: 2.0,
            }],
            lambda_tol: 1e-7,
            max_outer: 12,
            stability_descent: Some(false),
        };
        for c in [unset, set] {
            let defaults = RhfConfig::default();
            let old = RhfConfig {
                constraints: c.constraints.clone(),
                cdft_lambda_tol: c.lambda_tol,
                cdft_max_outer: c.max_outer,
                cdft_stability_descent: c
                    .stability_descent
                    .unwrap_or(defaults.cdft_stability_descent),
                xc: c.xc.clone(),
                dft_grid: c.dft_grid.clone(),
                max_iter: c.max_iter.unwrap_or(defaults.max_iter),
                energy_conv: c.energy_conv.unwrap_or(defaults.energy_conv),
                density_conv: c.density_conv.unwrap_or(defaults.density_conv),
                diis_size: c.diis_size.unwrap_or(defaults.diis_size),
                integral_thresh: c.integral_thresh.unwrap_or(defaults.integral_thresh),
                k_builder: c.k_builder,
                df_j_aux: c.df_j_aux.map(crate::rhf::normalize_df_aux),
                df_k_aux: c.df_k_aux.map(crate::rhf::normalize_df_aux),
                ri_j_precision: c.ri_j_precision,
                ri_k_precision: c.ri_k_precision,
                level_shift: c.level_shift.unwrap_or(defaults.level_shift),
                mom_after_iter: c.mom_after_iter.unwrap_or(defaults.mom_after_iter),
                external_potential: c.external_potential.clone(),
                three_index_budget_bytes: c.budget_bytes.unwrap_or(0),
                use_sad_guess: c.use_sad_guess,
                ..RhfConfig::default()
            };
            let mut new = lower(ScfOptions {
                xc: c.xc,
                dft_grid: c.dft_grid,
                max_iter: c.max_iter,
                energy_conv: c.energy_conv,
                density_conv: c.density_conv,
                diis_size: c.diis_size,
                integral_thresh: c.integral_thresh,
                k_builder: c.k_builder,
                df_j_aux: c.df_j_aux.map(AuxBasis::parse),
                df_k_aux: c.df_k_aux.map(AuxBasis::parse),
                ri_j_precision: c.ri_j_precision,
                ri_k_precision: c.ri_k_precision,
                mom_after_iter: c.mom_after_iter,
                external_potential: c.external_potential,
                three_index_budget_bytes: c.budget_bytes,
                guess: Some(if c.use_sad_guess {
                    Guess::Sad
                } else {
                    Guess::Core
                }),
                ..Default::default()
            });
            new.level_shift = c.level_shift.unwrap_or(defaults.level_shift);
            new.constraints = c.constraints;
            new.cdft_lambda_tol = c.lambda_tol;
            new.cdft_max_outer = c.max_outer;
            new.cdft_stability_descent = c
                .stability_descent
                .unwrap_or(defaults.cdft_stability_descent);
            same(&old, new);
        }
    }
}
