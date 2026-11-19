//! The Python SCF entry points' keyword policy: one function from their
//! already-parsed keyword arguments to an [`ScfOptions`].
//!
//! `run_rhf`, `run_uhf`, `run_rohf` (Hartree-Fock) and `run_dft`,
//! `run_uks`, `run_roks` (Kohn-Sham) each parse their own surface syntax
//! (strings, PyO3 objects) into typed values, in the order that fixes which
//! error a call with several bad keywords raises, collect them in an
//! [`ScfKwargs`], and call [`ScfEntryPoint::options`]. That function is the
//! one place the per-entry-point policy lives: the Kohn-Sham JK-fit
//! defaults and the RIJCOSX rule, `run_dft`'s symmetry and `"LDA"` defaults,
//! and which knobs an entry point refuses because its solver cannot honour
//! them ([`ScfEntryPoint::refused`], each with its reason). The rows of the
//! per-frontend default table in [`crate::options`] are this function.
//!
//! A refused knob never reaches the solver: the Python signatures do not
//! offer it, and [`ScfEntryPoint::options`] returns an [`EntryOptionError`]
//! if one is set anyway, so a knob is never accepted and silently ignored.

use std::fmt;

use ndarray::Array2;

use super::{df_k_aux_or_default, AuxBasis, Guess, ScfOptions, Stability, TrahTrigger};

/// `run_dft`'s functional when the caller names none.
pub const DEFAULT_DFT_FUNCTIONAL: &str = "LDA";

/// One of the six Python molecular SCF entry points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScfEntryPoint {
    /// `run_rhf`: closed-shell Hartree-Fock (`solve_rhf`).
    Rhf,
    /// `run_uhf`: unrestricted Hartree-Fock (`solve_uhf`).
    Uhf,
    /// `run_rohf`: restricted open-shell Hartree-Fock (`solve_rohf`).
    Rohf,
    /// `run_dft`: closed-shell Kohn-Sham (`solve_rhf` with `xc`, through
    /// `ksdft_ladder`).
    Dft,
    /// `run_uks`: unrestricted Kohn-Sham (`solve_uhf` with `xc`).
    Uks,
    /// `run_roks`: restricted open-shell Kohn-Sham (`solve_rohf` with `xc`).
    Roks,
}

/// A keyword an entry point may refuse (see [`ScfEntryPoint::refused`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ScfKnob {
    /// `diis=` ([`ScfKwargs::diis_flavor`]).
    Diis,
    /// `smearing_sigma=`.
    SmearingSigma,
    /// `soscf=`.
    Soscf,
    /// `aurora=`.
    Aurora,
    /// `trah_trigger=`.
    TrahTrigger,
    /// `stability_descent=`.
    StabilityDescent,
    /// `mom_reference=`.
    MomReference,
    /// `symmetry=` / `irrep_nelec=`.
    Symmetry,
    /// `functional=`.
    Functional,
    /// The `grid_*` keywords ([`ScfKwargs::dft_grid`]).
    DftGrid,
}

impl ScfKnob {
    /// The Python keyword(s) this knob is spelled as.
    pub fn kwarg(self) -> &'static str {
        match self {
            ScfKnob::Diis => "diis",
            ScfKnob::SmearingSigma => "smearing_sigma",
            ScfKnob::Soscf => "soscf",
            ScfKnob::Aurora => "aurora",
            ScfKnob::TrahTrigger => "trah_trigger",
            ScfKnob::StabilityDescent => "stability_descent",
            ScfKnob::MomReference => "mom_reference",
            ScfKnob::Symmetry => "symmetry / irrep_nelec",
            ScfKnob::Functional => "functional",
            ScfKnob::DftGrid => "grid_*",
        }
    }

    /// Whether `kw` sets this knob (`symmetry=False` with no `irrep_nelec`
    /// is unset).
    fn is_set(self, kw: &ScfKwargs<'_>) -> bool {
        match self {
            ScfKnob::Diis => kw.diis_flavor.is_some(),
            ScfKnob::SmearingSigma => kw.smearing_sigma.is_some(),
            ScfKnob::Soscf => kw.soscf.is_some(),
            ScfKnob::Aurora => kw.aurora.is_some(),
            ScfKnob::TrahTrigger => kw.trah_trigger.is_some(),
            ScfKnob::StabilityDescent => kw.stability_descent.is_some(),
            ScfKnob::MomReference => kw.mom_reference.is_some(),
            ScfKnob::Symmetry => kw.symmetry || kw.irrep_nelec.is_some(),
            ScfKnob::Functional => kw.functional.is_some(),
            ScfKnob::DftGrid => kw.dft_grid.is_some(),
        }
    }
}

const DIIS_RHF_ONLY: &str =
    "the open-shell solvers extrapolate with Pulay DIIS only (RhfConfig::diis_flavor is read by \
     solve_rhf alone)";
const SMEARING_RHF_ONLY: &str =
    "Fermi smearing (RhfConfig::smearing_sigma) is implemented in solve_rhf only";
const AURORA_RHF_ONLY: &str =
    "the AURORA accelerator (RhfConfig::aurora) is implemented in solve_rhf only";
const SYMMETRY_OPEN: &str =
    "the point-group-blocked open-shell solves (ferric_scf::symmetric_open) return their own \
     frame-aware results and take a per-spin irrep_nelec, which have no Python binding yet";
const TRAH_NOT_ROHF: &str =
    "solve_rohf has no TRAH step (RhfConfig::trah_trigger is read by solve_rhf and solve_uhf)";
const STABILITY_NOT_ROHF: &str =
    "there is no restricted-open-shell orbital-Hessian stability analysis \
     (stability::StabilitySkip::Rohf), so no descent";
const MOM_REFERENCE_UHF_ONLY: &str =
    "mom_reference = \"initial\" is implemented for UHF/UKS only (mom::check_reference), and \
     \"previous\" is already the default";
const LADDER_OWNS: &str =
    "run_dft solves through ksdft_ladder, whose rungs set diis_flavor, newton_trigger and \
     smearing_sigma themselves";
const HF_NO_FUNCTIONAL: &str = "Hartree-Fock has no exchange-correlation functional";
const HF_NO_GRID: &str = "Hartree-Fock has no Kohn-Sham integration grid";

impl ScfEntryPoint {
    /// The Python name (`"run_rhf"`, ...).
    pub fn name(self) -> &'static str {
        match self {
            ScfEntryPoint::Rhf => "run_rhf",
            ScfEntryPoint::Uhf => "run_uhf",
            ScfEntryPoint::Rohf => "run_rohf",
            ScfEntryPoint::Dft => "run_dft",
            ScfEntryPoint::Uks => "run_uks",
            ScfEntryPoint::Roks => "run_roks",
        }
    }

    /// Kohn-Sham (`run_dft` / `run_uks` / `run_roks`).
    pub fn is_kohn_sham(self) -> bool {
        matches!(
            self,
            ScfEntryPoint::Dft | ScfEntryPoint::Uks | ScfEntryPoint::Roks
        )
    }

    /// The knobs this entry point refuses, each with the reason its solver
    /// cannot honour it. A sibling entry point takes each of them; every
    /// other [`ScfKwargs`] field is honoured by every entry point that
    /// offers it.
    pub fn refused(self) -> &'static [(ScfKnob, &'static str)] {
        use ScfKnob as K;
        match self {
            ScfEntryPoint::Rhf => &[
                (K::MomReference, MOM_REFERENCE_UHF_ONLY),
                (K::Functional, HF_NO_FUNCTIONAL),
                (K::DftGrid, HF_NO_GRID),
            ],
            ScfEntryPoint::Uhf => &[
                (K::Diis, DIIS_RHF_ONLY),
                (K::SmearingSigma, SMEARING_RHF_ONLY),
                (K::Aurora, AURORA_RHF_ONLY),
                (K::Symmetry, SYMMETRY_OPEN),
                (K::Functional, HF_NO_FUNCTIONAL),
                (K::DftGrid, HF_NO_GRID),
            ],
            ScfEntryPoint::Rohf => &[
                (K::Diis, DIIS_RHF_ONLY),
                (K::SmearingSigma, SMEARING_RHF_ONLY),
                (K::Aurora, AURORA_RHF_ONLY),
                (K::Symmetry, SYMMETRY_OPEN),
                (K::TrahTrigger, TRAH_NOT_ROHF),
                (K::StabilityDescent, STABILITY_NOT_ROHF),
                (K::MomReference, MOM_REFERENCE_UHF_ONLY),
                (K::Functional, HF_NO_FUNCTIONAL),
                (K::DftGrid, HF_NO_GRID),
            ],
            ScfEntryPoint::Dft => &[
                (K::Diis, LADDER_OWNS),
                (K::SmearingSigma, LADDER_OWNS),
                (K::Soscf, LADDER_OWNS),
                (K::MomReference, MOM_REFERENCE_UHF_ONLY),
            ],
            ScfEntryPoint::Uks => &[
                (K::Diis, DIIS_RHF_ONLY),
                (K::SmearingSigma, SMEARING_RHF_ONLY),
                (K::Aurora, AURORA_RHF_ONLY),
                (K::Symmetry, SYMMETRY_OPEN),
            ],
            ScfEntryPoint::Roks => &[
                (K::Diis, DIIS_RHF_ONLY),
                (K::SmearingSigma, SMEARING_RHF_ONLY),
                (K::Aurora, AURORA_RHF_ONLY),
                (K::Symmetry, SYMMETRY_OPEN),
                (K::TrahTrigger, TRAH_NOT_ROHF),
                (K::StabilityDescent, STABILITY_NOT_ROHF),
                (K::MomReference, MOM_REFERENCE_UHF_ONLY),
            ],
        }
    }

    /// The [`ScfOptions`] for a call of this entry point with `kw`. Pure.
    ///
    /// An unset (`None`) keyword keeps the `RhfConfig` default, except
    /// where this entry point's policy names its own default:
    ///
    /// * Kohn-Sham: `df_j_aux` defaults to
    ///   [`DEFAULT_JK_AUX`](crate::options::DEFAULT_JK_AUX), and so does
    ///   `df_k_aux` except under COSX
    ///   ([`df_k_aux_or_default`](crate::options::df_k_aux_or_default));
    ///   `xc` is the functional, `run_dft`'s defaulting to
    ///   [`DEFAULT_DFT_FUNCTIONAL`].
    /// * `run_rhf`: `symmetry` lowers as given; `run_dft`: an
    ///   `irrep_nelec` turns symmetry on (the binding refuses it without
    ///   `symmetry=True` before calling this).
    ///
    /// Errors: a knob of [`refused`](Self::refused) set, or `run_uks` /
    /// `run_roks` without a functional.
    pub fn options(self, kw: ScfKwargs<'_>) -> Result<ScfOptions, EntryOptionError> {
        if let Some((knob, reason)) = self.refused().iter().find(|(k, _)| k.is_set(&kw)) {
            return Err(EntryOptionError::Refused {
                entry: self,
                knob: *knob,
                reason,
            });
        }
        let ScfKwargs {
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
            stability_descent,
            symmetry,
            irrep_nelec,
            eri,
            relativistic,
            aurora,
            trah_trigger,
            functional,
            dft_grid,
        } = kw;
        let ks = self.is_kohn_sham();
        let xc = match self {
            ScfEntryPoint::Rhf | ScfEntryPoint::Uhf | ScfEntryPoint::Rohf => None,
            ScfEntryPoint::Dft => Some(functional.unwrap_or(DEFAULT_DFT_FUNCTIONAL).to_string()),
            ScfEntryPoint::Uks | ScfEntryPoint::Roks => Some(
                functional
                    .ok_or(EntryOptionError::MissingFunctional(self))?
                    .to_string(),
            ),
        };
        let symmetry = match self {
            ScfEntryPoint::Rhf => Some(symmetry),
            ScfEntryPoint::Dft => (symmetry || irrep_nelec.is_some()).then_some(true),
            // Refused above when set.
            _ => None,
        };
        // Kohn-Sham: RI-J / RI-K on by default (PySCF's density_fit
        // convention), with the RIJCOSX rule for RI-K. Hartree-Fock: exact
        // J and K unless named.
        let (df_j_aux, df_k_aux) = if ks {
            (
                Some(
                    df_j_aux
                        .map(AuxBasis::parse)
                        .unwrap_or_else(AuxBasis::default_jk),
                ),
                df_k_aux_or_default(
                    k_builder,
                    df_k_aux.map(AuxBasis::parse),
                    Some(AuxBasis::default_jk()),
                ),
            )
        } else {
            (df_j_aux.map(AuxBasis::parse), df_k_aux.map(AuxBasis::parse))
        };
        Ok(ScfOptions {
            max_iter,
            energy_conv,
            density_conv,
            diis_size,
            diis_flavor,
            integral_thresh,
            k_builder: k_builder.map(str::to_string),
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
            // The descent needs a stability verdict to act on, so the
            // keyword turns on both halves.
            stability: stability_descent.map(|d| {
                if d {
                    Stability::Descend
                } else {
                    Stability::Off
                }
            }),
            symmetry,
            irrep_nelec,
            eri,
            relativistic,
            aurora,
            trah_trigger,
            xc,
            dft_grid,
            // Not Python keywords.
            diis_switch_thresh: None,
            ri_j_task: None,
            screening: None,
            verbose: None,
        })
    }
}

/// The SCF keywords of the six entry points, already parsed into typed
/// values (`None` = not passed). A field an entry point does not offer stays
/// `None` (`false` for `symmetry`).
#[derive(Debug, Clone, Default)]
pub struct ScfKwargs<'a> {
    /// `max_iter=`.
    pub max_iter: Option<usize>,
    /// `energy_conv=`.
    pub energy_conv: Option<f64>,
    /// `density_conv=`.
    pub density_conv: Option<f64>,
    /// `diis_size=`.
    pub diis_size: Option<usize>,
    /// `diis=`.
    pub diis_flavor: Option<crate::diis::DiisFlavor>,
    /// `integral_thresh=`.
    pub integral_thresh: Option<f64>,
    /// `k_builder=` (verbatim).
    pub k_builder: Option<&'a str>,
    /// The resolved `cosx_*` keywords (the default COSX settings when the
    /// entry point offers them and none is passed).
    pub cosx: Option<crate::cosx_k::CosxConfig>,
    /// `df_j_aux=` (any spelling; see [`AuxBasis::parse`]).
    pub df_j_aux: Option<&'a str>,
    /// `df_k_aux=` (any spelling; see [`AuxBasis::parse`]).
    pub df_k_aux: Option<&'a str>,
    /// `ri_j_precision=`.
    pub ri_j_precision: Option<crate::df_j::RiJPrecision>,
    /// `ri_k_precision=`.
    pub ri_k_precision: Option<crate::df_k::RiKPrecision>,
    /// `level_shift=`.
    pub level_shift: Option<f64>,
    /// `mom_after_iter=`.
    pub mom_after_iter: Option<usize>,
    /// `mom_reference=`.
    pub mom_reference: Option<crate::mom::MomReference>,
    /// `smearing_sigma=`.
    pub smearing_sigma: Option<f64>,
    /// `soscf=`.
    pub soscf: Option<bool>,
    /// `guess=`.
    pub guess: Option<Guess>,
    /// `guess_basis=` / `guess_projection=`.
    pub guess_basis: Option<crate::basis2::GuessBasis>,
    /// `dm0=` of the closed-shell entry points (the open-shell ones pass
    /// their per-spin `dm0` to the solver directly).
    pub init_guess_density: Option<Array2<f64>>,
    /// `point_charges=` / `smeared_charges=` / `external_field=`.
    pub external_potential: Option<ferric_core::external_potential::ExternalPotential>,
    /// `solvent=` / `pcm_lebedev_order=`.
    pub pcm: Option<ferric_pcm::PcmConfig>,
    /// `memory_budget_gb=` in bytes (`None` = auto-detect).
    pub three_index_budget_bytes: Option<usize>,
    /// `stability_descent=`.
    pub stability_descent: Option<bool>,
    /// `symmetry=`.
    pub symmetry: bool,
    /// `irrep_nelec=`.
    pub irrep_nelec: Option<Vec<(String, usize)>>,
    /// `eri=`.
    pub eri: Option<crate::incore_jk::EriMode>,
    /// `relativistic=`.
    pub relativistic: Option<crate::x2c::Relativistic>,
    /// `aurora=`.
    pub aurora: Option<bool>,
    /// `trah_trigger=`.
    pub trah_trigger: Option<TrahTrigger>,
    /// `functional=` (Kohn-Sham).
    pub functional: Option<&'a str>,
    /// The resolved `grid_*` keywords (Kohn-Sham; `None` = production grid).
    pub dft_grid: Option<ferric_dft::grid::AtomicGridConfig>,
}

/// [`ScfEntryPoint::options`]'s refusal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EntryOptionError {
    /// A knob of [`ScfEntryPoint::refused`] was set.
    Refused {
        /// The entry point.
        entry: ScfEntryPoint,
        /// The knob.
        knob: ScfKnob,
        /// Why its solver cannot honour it.
        reason: &'static str,
    },
    /// `run_uks` / `run_roks` without a functional.
    MissingFunctional(ScfEntryPoint),
}

impl fmt::Display for EntryOptionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EntryOptionError::Refused {
                entry,
                knob,
                reason,
            } => write!(
                f,
                "{}: {} is not supported: {reason}",
                entry.name(),
                knob.kwarg()
            ),
            EntryOptionError::MissingFunctional(entry) => {
                write!(f, "{}: a functional is required", entry.name())
            }
        }
    }
}

impl std::error::Error for EntryOptionError {}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [ScfEntryPoint; 6] = [
        ScfEntryPoint::Rhf,
        ScfEntryPoint::Uhf,
        ScfEntryPoint::Rohf,
        ScfEntryPoint::Dft,
        ScfEntryPoint::Uks,
        ScfEntryPoint::Roks,
    ];

    fn ks_kw() -> ScfKwargs<'static> {
        ScfKwargs {
            functional: Some("PBE"),
            ..ScfKwargs::default()
        }
    }

    fn base(entry: ScfEntryPoint) -> ScfKwargs<'static> {
        if entry.is_kohn_sham() {
            ks_kw()
        } else {
            ScfKwargs::default()
        }
    }

    /// Every refused knob is refused with its reason, naming the entry
    /// point; the same knob is accepted by at least one sibling (otherwise
    /// it is not drift, and would not be listed).
    #[test]
    fn refused_knobs_are_refused_and_offered_by_a_sibling() {
        let set = |knob: ScfKnob, kw: &mut ScfKwargs<'static>| match knob {
            ScfKnob::Diis => kw.diis_flavor = Some(crate::diis::DiisFlavor::Adiis),
            ScfKnob::SmearingSigma => kw.smearing_sigma = Some(0.01),
            ScfKnob::Soscf => kw.soscf = Some(true),
            ScfKnob::Aurora => kw.aurora = Some(true),
            ScfKnob::TrahTrigger => kw.trah_trigger = Some(TrahTrigger::new(0.1).unwrap()),
            ScfKnob::StabilityDescent => kw.stability_descent = Some(true),
            ScfKnob::MomReference => kw.mom_reference = Some(crate::mom::MomReference::Previous),
            ScfKnob::Symmetry => kw.symmetry = true,
            ScfKnob::Functional => kw.functional = Some("PBE"),
            ScfKnob::DftGrid => {
                kw.dft_grid = Some(ferric_dft::grid::AtomicGridConfig::production())
            }
        };
        for entry in ALL {
            for &(knob, reason) in entry.refused() {
                let mut kw = base(entry);
                set(knob, &mut kw);
                let err = entry.options(kw).unwrap_err();
                assert_eq!(
                    err,
                    EntryOptionError::Refused {
                        entry,
                        knob,
                        reason
                    }
                );
                assert!(err.to_string().starts_with(entry.name()), "{err}");
                let offered = ALL.iter().any(|&sib| {
                    let mut kw = base(sib);
                    set(knob, &mut kw);
                    sib.options(kw).is_ok()
                });
                assert!(offered, "{knob:?} is refused everywhere");
            }
        }
    }

    #[test]
    fn open_shell_ks_needs_a_functional() {
        for entry in [ScfEntryPoint::Uks, ScfEntryPoint::Roks] {
            assert_eq!(
                entry.options(ScfKwargs::default()).unwrap_err(),
                EntryOptionError::MissingFunctional(entry)
            );
        }
        let c = ScfEntryPoint::Dft
            .options(ScfKwargs::default())
            .unwrap()
            .into_rhf_config()
            .unwrap();
        assert_eq!(c.xc.as_deref(), Some(DEFAULT_DFT_FUNCTIONAL));
    }
}
