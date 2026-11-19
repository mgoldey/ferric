//! Frontend-neutral SCF options and their lowering to [`RhfConfig`](crate::rhf::RhfConfig).
//!
//! Frontends (the Python `run_rhf` / `run_uhf` / `run_rohf` today; the CLI
//! `[scf]` section and the KS entry points are follow-ups) parse their own
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

use ferric_core::FerricError;
use ndarray::Array2;

use crate::rhf::RhfConfig;

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

/// A density-fitting auxiliary basis choice for [`RhfConfig::df_j_aux`] /
/// [`RhfConfig::df_k_aux`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuxBasis {
    /// Explicitly no density fitting (exact four-centre J or K). Lowers to
    /// the `Some("")` sentinel `RhfConfig` uses for an explicit opt-out.
    Off,
    /// A named auxiliary basis (trimmed; validated later by `basis::bundled`).
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
    /// [`RhfConfig::k_builder`] (passed through verbatim).
    pub k_builder: Option<String>,
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
    pub fn into_rhf_config(self) -> Result<RhfConfig, FerricError> {
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
            let (check, descend) = match st {
                Stability::Off => (false, false),
                Stability::Check => (true, false),
                Stability::Descend => (true, true),
            };
            c.check_stability = check;
            c.scf_stability_descent = descend;
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
    use super::*;
    use crate::cosx_k::CosxConfig;
    use crate::df_j::RiJPrecision;
    use crate::df_k::RiKPrecision;
    use crate::diis::DiisFlavor;
    use crate::incore_jk::EriMode;
    use crate::mom::MomReference;
    use crate::x2c::Relativistic;
    use ferric_core::external_potential::{ExternalPotential, PointCharge};

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
        }
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
            k_builder: k.k_builder.map(|s| s.to_string()),
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
            k_builder: k.k_builder.map(|s| s.to_string()),
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
            k_builder: k.k_builder.map(|s| s.to_string()),
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

    // ── the new path: what each Python entry point now puts in ScfOptions ──

    fn aux(v: Option<&str>) -> Option<AuxBasis> {
        v.map(AuxBasis::parse)
    }

    fn guess(sad: Option<bool>) -> Option<Guess> {
        Some(if sad.unwrap_or(true) {
            Guess::Sad
        } else {
            Guess::Core
        })
    }

    fn stability(descent: Option<bool>) -> Option<Stability> {
        descent.map(|d| {
            if d {
                Stability::Descend
            } else {
                Stability::Off
            }
        })
    }

    fn trah(t: Option<f64>) -> Option<TrahTrigger> {
        t.map(|v| TrahTrigger::new(v).unwrap())
    }

    fn new_run_rhf(k: Kw) -> RhfConfig {
        ScfOptions {
            max_iter: k.max_iter,
            energy_conv: k.energy_conv,
            density_conv: k.density_conv,
            diis_size: k.diis_size,
            diis_flavor: Some(k.diis.unwrap_or(DiisFlavor::Pulay)),
            integral_thresh: k.integral_thresh,
            k_builder: k.k_builder.map(str::to_string),
            cosx: k.cosx,
            df_j_aux: aux(k.df_j_aux),
            df_k_aux: aux(k.df_k_aux),
            ri_j_precision: k.ri_j_precision,
            ri_k_precision: k.ri_k_precision,
            level_shift: k.level_shift,
            mom_after_iter: k.mom_after_iter,
            mom_reference: None,
            smearing_sigma: k.smearing_sigma,
            soscf: k.soscf,
            guess: guess(k.guess_sad),
            guess_basis: k.guess_basis,
            init_guess_density: k.init_guess_density,
            external_potential: k.external_potential,
            pcm: k.pcm,
            three_index_budget_bytes: k.budget_bytes,
            stability: stability(k.stability_descent),
            symmetry: Some(k.symmetry),
            irrep_nelec: k.irrep_nelec,
            eri: Some(k.eri.unwrap_or(EriMode::Auto)),
            relativistic: k.relativistic,
            aurora: k.aurora,
            trah_trigger: trah(k.trah_trigger),
        }
        .into_rhf_config()
        .unwrap()
    }

    fn new_run_uhf(k: Kw) -> RhfConfig {
        ScfOptions {
            max_iter: k.max_iter,
            energy_conv: k.energy_conv,
            density_conv: k.density_conv,
            diis_size: k.diis_size,
            diis_flavor: None,
            integral_thresh: k.integral_thresh,
            k_builder: k.k_builder.map(str::to_string),
            cosx: k.cosx,
            df_j_aux: aux(k.df_j_aux),
            df_k_aux: aux(k.df_k_aux),
            ri_j_precision: k.ri_j_precision,
            ri_k_precision: k.ri_k_precision,
            level_shift: k.level_shift,
            mom_after_iter: k.mom_after_iter,
            mom_reference: Some(k.mom_reference.unwrap_or_default()),
            smearing_sigma: None,
            soscf: None,
            guess: guess(k.guess_sad),
            guess_basis: k.guess_basis,
            init_guess_density: None,
            external_potential: k.external_potential,
            pcm: None,
            three_index_budget_bytes: k.budget_bytes,
            stability: stability(k.stability_descent),
            symmetry: None,
            irrep_nelec: None,
            eri: Some(k.eri.unwrap_or(EriMode::Auto)),
            relativistic: k.relativistic,
            aurora: None,
            trah_trigger: trah(k.trah_trigger),
        }
        .into_rhf_config()
        .unwrap()
    }

    fn new_run_rohf(k: Kw) -> RhfConfig {
        ScfOptions {
            max_iter: k.max_iter,
            energy_conv: k.energy_conv,
            density_conv: k.density_conv,
            diis_size: k.diis_size,
            diis_flavor: None,
            integral_thresh: k.integral_thresh,
            k_builder: k.k_builder.map(str::to_string),
            cosx: None,
            df_j_aux: aux(k.df_j_aux),
            df_k_aux: aux(k.df_k_aux),
            ri_j_precision: k.ri_j_precision,
            ri_k_precision: k.ri_k_precision,
            level_shift: k.level_shift,
            mom_after_iter: k.mom_after_iter,
            mom_reference: None,
            smearing_sigma: None,
            soscf: None,
            guess: None,
            guess_basis: k.guess_basis,
            init_guess_density: None,
            external_potential: k.external_potential,
            pcm: None,
            three_index_budget_bytes: k.budget_bytes,
            stability: None,
            symmetry: None,
            irrep_nelec: None,
            eri: Some(k.eri.unwrap_or(EriMode::Auto)),
            relativistic: k.relativistic,
            aurora: None,
            trah_trigger: None,
        }
        .into_rhf_config()
        .unwrap()
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

    #[test]
    fn aux_basis_parse_maps_opt_outs_to_off() {
        for s in ["", "  ", "none", "OFF", " Exact ", "conventional"] {
            assert_eq!(AuxBasis::parse(s), AuxBasis::Off, "{s:?}");
        }
        assert_eq!(
            AuxBasis::parse(" def2-universal-jkfit "),
            AuxBasis::Named("def2-universal-jkfit".into())
        );
    }

    #[test]
    fn explicit_level_shift_disables_meta_gga_auto_shift() {
        let c = ScfOptions {
            level_shift: Some(0.0),
            ..Default::default()
        }
        .into_rhf_config()
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
        same(&ScfOptions::default().into_rhf_config().unwrap(), &d);
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
            .into_rhf_config()
            .unwrap();
            (c.check_stability, c.scf_stability_descent)
        };
        assert_eq!(lower(Stability::Off), (false, false));
        assert_eq!(lower(Stability::Check), (true, false));
        assert_eq!(lower(Stability::Descend), (true, true));
        let d = RhfConfig::default();
        assert_eq!((d.check_stability, d.scf_stability_descent), (false, false));
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
            diis_switch_thresh: _,        // default
            smearing_sigma: _,            // lowered
            integral_thresh: _,           // lowered
            k_builder: _,                 // lowered
            cosx: _,                      // lowered
            df_j_aux: _,                  // lowered
            df_k_aux: _,                  // lowered
            ri_j_precision: _,            // lowered
            ri_j_task: _,                 // default (geometry drivers set it)
            ri_k_precision: _,            // lowered
            xc: _,                        // default (KS: follow-up)
            xc_omega: _,                  // default (KS: follow-up)
            dft_grid: _,                  // default (KS: follow-up)
            nlc_grid: _,                  // default (KS: follow-up)
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
            verbose: _,                   // default
            check_stability: _,           // lowered (stability_descent)
            symmetry: _,                  // lowered
            irrep_nelec: _,               // lowered
            screening: _,                 // default
            raw_handoff: _,               // default
            eri: _,                       // lowered
            eri_route: _,                 // default (solver output)
            relativistic: _,              // lowered
        } = RhfConfig::default();
    }
}
