//! Opt-in LOCAL build of the VV-HV localized virtual space.
//!
//! [`crate::lmp2_amplitude::build_vvhv`] (the DENSE build, the default) is
//! O(N³): a global Cholesky solve `S⁻¹ S_x`, the occupied projector
//! `C_o C_oᵀ S` as a dense product, a global eigh of the projected-minimal
//! Gram, an all-pairs Boys sweep, dense candidate Grams, a dense greedy
//! pivoted Cholesky, a global Löwdin eigh and a full `F·C_h`. This module
//! keeps that construction STEP FOR STEP and replaces each dense global
//! operation by a threshold-sparse one that is exact as its thresholds go to
//! zero (design: `wiki/notes/local-vvhv-design-2026-11-06.md`, Candidate A;
//! measured locality: `wiki/notes/local-vvhv-step0-2026-11-06.md`).
//!
//! WHY the dense build stays the default: step 0 measured the reach of the
//! span-critical intermediates at the τ the 1e-8 construction check needs
//! (τ ≈ 1e-9) at 62-95 Bohr on n-alkanes (6-31G), so a truncated product drops
//! nothing until the chain is ≈ 2 r(τ) ≈ 140-190 Bohr long (C60-C80), and the
//! wall-clock crossover against the (constant-factor-fixed) dense build lies
//! well past C100. The local build is therefore an opt-in
//! (`VirtualSpace::Local`, CLI `[local] virtual_space = "local"`).
//!
//! Every run reports which build it used ([`VirtualSpaceReport`]).

use ferric_core::basis::BasisSet;
use ferric_core::error::FerricError;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_scf::result::ScfResult;

use crate::lmp2_amplitude::{StageTimings, VvHv};
use crate::lmp2_certificate::CertValue;

/// Which VV-HV virtual-space build a local MP2 run uses.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum VirtualSpace {
    /// [`crate::lmp2_amplitude::build_vvhv`]: the dense O(N³) build (default).
    #[default]
    Dense,
    /// The threshold-sparse build of this module, with its thresholds.
    Local(LocalVvhvConfig),
}

impl VirtualSpace {
    /// The accepted spellings of [`VirtualSpace::parse_config_str`].
    pub const SPELLINGS: [&'static str; 2] = ["dense", "local"];

    /// Strict parse of the selector: `"dense"` or `"local"` (lower case),
    /// with `cfg` the thresholds of the local build. Anything else is an
    /// error, never a silent default.
    pub fn parse_config_str(s: &str, cfg: LocalVvhvConfig) -> Result<Self, String> {
        match s {
            "dense" => Ok(Self::Dense),
            "local" => Ok(Self::Local(cfg)),
            other => Err(format!(
                "virtual_space = {other:?}: expected one of {}",
                Self::SPELLINGS
                    .iter()
                    .map(|v| format!("\"{v}\""))
                    .collect::<Vec<_>>()
                    .join(", ")
            )),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Dense => "dense",
            Self::Local(_) => "local",
        }
    }
}

/// Thresholds of the local VV-HV build. Every one has a trivial limit in which
/// it is a no-op ([`LocalVvhvConfig::trivial`], the exactness-anchor
/// configuration).
///
/// The split follows the design's span-critical / basis-choice
/// classification: an error in a SPAN-CRITICAL step (the occupied projector,
/// the two orthonormalizations, the final guard) moves the ε = 0 energy at
/// first order, so it is truncated at the tight `tau`; an error in a
/// BASIS-CHOICE step (the projected-minimal solve, the valence spectral
/// projector, the hard-virtual pivot Gram and Cholesky rows, the Boys pair
/// screen) only rotates the basis inside the exact virtual space, so it is
/// truncated at the loose `tau_loose`.
#[derive(Debug, Clone, PartialEq)]
pub struct LocalVvhvConfig {
    /// Span-critical block drop threshold (Frobenius norm of an atom-group
    /// block). 0 drops nothing.
    pub tau: f64,
    /// Basis-choice block drop threshold. 0 drops nothing.
    pub tau_loose: f64,
    /// Neighbourhood radius (Bohr) of the domain-local `S⁻¹ S_x` solves:
    /// each atom group's projected minimal functions are solved in the AOs of
    /// the groups within this distance. `None` = one global solve.
    pub solve_radius_bohr: Option<f64>,
    /// Boys pair screen: pair (i, j) is not rotated in a sweep when
    /// `|q_ij| < boys_tau · |R_i − R_j|` (q_ij the dipole coupling, R the
    /// centroids); every skipped rotation then has `|θ| ≤ boys_tau`. 0 rotates
    /// every pair.
    pub boys_tau: f64,
}

impl Default for LocalVvhvConfig {
    /// The production thresholds (see `site/src/methods/mp2.md` for the
    /// measured justification).
    fn default() -> Self {
        Self {
            tau: 1e-9,
            tau_loose: 1e-6,
            solve_radius_bohr: Some(30.0),
            boys_tau: 1e-8,
        }
    }
}

impl LocalVvhvConfig {
    /// Every threshold at its no-op limit: the local build must then
    /// reproduce the dense build (the exactness anchor).
    pub fn trivial() -> Self {
        Self {
            tau: 0.0,
            tau_loose: 0.0,
            solve_radius_bohr: None,
            boys_tau: 0.0,
        }
    }

    /// Reject non-finite / negative thresholds and a non-positive radius.
    pub fn validate(&self) -> Result<(), FerricError> {
        let bad = |k: &str, x: f64| {
            FerricError::General(format!(
                "local VV-HV: {k} must be finite and >= 0 (got {x})"
            ))
        };
        for (k, x) in [
            ("tau", self.tau),
            ("tau_loose", self.tau_loose),
            ("boys_tau", self.boys_tau),
        ] {
            if !(x.is_finite() && x >= 0.0) {
                return Err(bad(k, x));
            }
        }
        if let Some(r) = self.solve_radius_bohr {
            if !(r.is_finite() && r > 0.0) {
                return Err(FerricError::General(format!(
                    "local VV-HV: solve_radius_bohr must be finite and > 0 (got {r})"
                )));
            }
        }
        Ok(())
    }
}

/// Sparsity counters and stage timers of one local VV-HV build (the
/// honest-measurement surface: the locality claim is these counters, so they
/// are reported, not inferred).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LocalVvhvStats {
    /// Atom groups (a heavy atom with its nearest hydrogens).
    pub n_groups: usize,
    /// Block-sparse products formed.
    pub n_products: usize,
    /// Atom-group blocks kept / dropped by the thresholds, over every
    /// truncated product and conversion.
    pub blocks_kept: u64,
    pub blocks_dropped: u64,
    /// Domain-local solve neighbourhood sizes (AOs), mean / max.
    pub solve_domain_aos_mean: f64,
    pub solve_domain_aos_max: usize,
    /// Trace-correcting purification iterations of the valence projector.
    pub purification_iterations: usize,
    /// Newton-Schulz iterations: valence Löwdin, hard-virtual Löwdin, guard.
    pub ns_iterations_valence: usize,
    pub ns_iterations_hard: usize,
    pub ns_iterations_guard: usize,
    /// Pivoted-Cholesky pivots taken (valence + hard) and heap pops.
    pub heap_pops: u64,
    /// Boys sweeps, and pair rotations evaluated / skipped by the screen.
    pub boys_sweeps: usize,
    pub boys_pairs_evaluated: u64,
    pub boys_pairs_screened: u64,
    /// Mean / max block-row fill (kept blocks per block row, as a fraction of
    /// the groups) of the final virtual coefficients.
    pub cv_row_fill_mean: f64,
    pub cv_row_fill_max: f64,
    /// Wall seconds of the build (excluding the construction check).
    pub t_build_s: f64,
}

/// Which virtual-space build a run used, and its counters.
#[derive(Debug, Clone, PartialEq)]
pub enum VirtualSpaceReport {
    /// The caller supplied the space (`*_with_virtuals` entry points).
    CallerSupplied,
    /// The dense default build.
    Dense,
    /// The local build, with the thresholds it ran at and its counters.
    Local {
        config: LocalVvhvConfig,
        stats: LocalVvhvStats,
    },
}

impl VirtualSpaceReport {
    pub fn build(&self) -> &'static str {
        match self {
            Self::CallerSupplied => "caller-supplied",
            Self::Dense => "dense",
            Self::Local { .. } => "local",
        }
    }

    /// `(t_spaces + t_loc) / t_corr` of a run, with `t_corr = t_spaces +
    /// t_assembly + t_solve` (the correlation stage; `t_assembly` includes
    /// `t_loc`, the reference is excluded).
    pub fn spaces_share(t: &StageTimings) -> (f64, f64) {
        let t_corr = t.t_spaces_s + t.t_assembly_s + t.t_solve_s;
        let share = if t_corr > 0.0 {
            (t.t_spaces_s + t.t_loc_s) / t_corr
        } else {
            f64::NAN
        };
        (share, t_corr)
    }

    /// The one line every amplitude-threshold LMP2 run prints: which build
    /// it used, why, and what it cost.
    pub fn log_line(&self, t: &StageTimings) -> String {
        let (share, t_corr) = Self::spaces_share(t);
        let cost = format!(
            "t_spaces {:.3} s + t_loc {:.3} s = {:.1}% of t_corr {:.3} s",
            t.t_spaces_s,
            t.t_loc_s,
            100.0 * share,
            t_corr
        );
        match self {
            Self::CallerSupplied => {
                format!("virtual space: caller-supplied VV-HV (not built here); {cost}")
            }
            Self::Dense => format!(
                "virtual space: dense VV-HV build (the default; O(N^3)). The local build \
                 ([local] virtual_space = \"local\") is opt-in because it pays off only well \
                 past ~C100. This run: {cost}"
            ),
            Self::Local { config, stats } => format!(
                "virtual space: LOCAL VV-HV build (opt-in): tau {:.0e}, tau_loose {:.0e}, \
                 solve_radius {}, boys_tau {:.0e}; blocks kept/dropped {}/{}, products {}, \
                 Newton-Schulz iters valence/hard/guard {}/{}/{}, purification iters {}, \
                 solve domain AOs mean/max {:.0}/{}, C_v row fill mean/max {:.2}/{:.2}, Boys \
                 sweeps {} (pairs evaluated/screened {}/{}); {cost}",
                config.tau,
                config.tau_loose,
                config
                    .solve_radius_bohr
                    .map_or("global".to_string(), |r| format!("{r} Bohr")),
                config.boys_tau,
                stats.blocks_kept,
                stats.blocks_dropped,
                stats.n_products,
                stats.ns_iterations_valence,
                stats.ns_iterations_hard,
                stats.ns_iterations_guard,
                stats.purification_iterations,
                stats.solve_domain_aos_mean,
                stats.solve_domain_aos_max,
                stats.cv_row_fill_mean,
                stats.cv_row_fill_max,
                stats.boys_sweeps,
                stats.boys_pairs_evaluated,
                stats.boys_pairs_screened,
            ),
        }
    }
}

impl VirtualSpaceReport {
    /// The report as result-dict fields (CLI run log `local` object, Python
    /// `local` dict): `virtual_space` (the build), `virtual_space_log` (the
    /// printed line) and `virtual_space_local` (thresholds + counters of the
    /// local build; null otherwise).
    pub fn fields(&self, t: &StageTimings) -> Vec<(&'static str, CertValue)> {
        use CertValue as V;
        let local = match self {
            Self::Local { config, stats } => V::Map(vec![
                ("tau", V::F64(config.tau)),
                ("tau_loose", V::F64(config.tau_loose)),
                ("solve_radius_bohr", config.solve_radius_bohr.into()),
                ("boys_tau", V::F64(config.boys_tau)),
                ("n_groups", V::Usize(stats.n_groups)),
                ("n_products", V::Usize(stats.n_products)),
                ("blocks_kept", V::Usize(stats.blocks_kept as usize)),
                ("blocks_dropped", V::Usize(stats.blocks_dropped as usize)),
                ("solve_domain_aos_mean", V::F64(stats.solve_domain_aos_mean)),
                ("solve_domain_aos_max", V::Usize(stats.solve_domain_aos_max)),
                (
                    "purification_iterations",
                    V::Usize(stats.purification_iterations),
                ),
                (
                    "ns_iterations_valence",
                    V::Usize(stats.ns_iterations_valence),
                ),
                ("ns_iterations_hard", V::Usize(stats.ns_iterations_hard)),
                ("ns_iterations_guard", V::Usize(stats.ns_iterations_guard)),
                ("heap_pops", V::Usize(stats.heap_pops as usize)),
                ("boys_sweeps", V::Usize(stats.boys_sweeps)),
                (
                    "boys_pairs_evaluated",
                    V::Usize(stats.boys_pairs_evaluated as usize),
                ),
                (
                    "boys_pairs_screened",
                    V::Usize(stats.boys_pairs_screened as usize),
                ),
                ("cv_row_fill_mean", V::F64(stats.cv_row_fill_mean)),
                ("cv_row_fill_max", V::F64(stats.cv_row_fill_max)),
                ("t_build_s", V::F64(stats.t_build_s)),
            ]),
            _ => V::Null,
        };
        vec![
            ("virtual_space", V::Str(self.build().to_string())),
            ("virtual_space_log", V::Str(self.log_line(t))),
            ("virtual_space_local", local),
        ]
    }
}

/// One local VV-HV build: the space, the AO index of every selected
/// hard-virtual candidate in pivot order (the dense build's trace), and the
/// counters.
#[derive(Debug, Clone)]
pub struct LocalVvhv {
    pub vv: VvHv,
    pub hard_aos: Vec<usize>,
    pub stats: LocalVvhvStats,
}

/// Build the VV-HV space with the local construction.
///
/// STATUS (step a): delegates to the dense build; the sparse stages replace
/// it one at a time, each against the trivial-limit anchor.
pub fn build_vvhv_local(
    mol: &Molecule,
    obs: &PreparedBasis,
    obs_bs: &BasisSet,
    rhf: &ScfResult,
    cfg: &LocalVvhvConfig,
) -> Result<LocalVvhv, FerricError> {
    cfg.validate()?;
    let t0 = std::time::Instant::now();
    let (vv, hard_aos) = crate::lmp2_amplitude::build_vvhv_traced_dense(mol, obs, obs_bs, rhf)?;
    let stats = LocalVvhvStats {
        t_build_s: t0.elapsed().as_secs_f64(),
        ..Default::default()
    };
    Ok(LocalVvhv {
        vv,
        hard_aos,
        stats,
    })
}

#[cfg(test)]
#[path = "vvhv_local_anchor.rs"]
pub(crate) mod anchor;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn virtual_space_parse_is_strict() {
        assert_eq!(
            VirtualSpace::parse_config_str("dense", LocalVvhvConfig::default()),
            Ok(VirtualSpace::Dense)
        );
        assert_eq!(
            VirtualSpace::parse_config_str("local", LocalVvhvConfig::trivial()),
            Ok(VirtualSpace::Local(LocalVvhvConfig::trivial()))
        );
        for bad in ["Dense", "LOCAL", "sparse", ""] {
            assert!(VirtualSpace::parse_config_str(bad, LocalVvhvConfig::default()).is_err());
        }
        assert_eq!(VirtualSpace::default(), VirtualSpace::Dense);
    }

    #[test]
    fn local_config_validation_rejects_bad_values() {
        assert!(LocalVvhvConfig::default().validate().is_ok());
        assert!(LocalVvhvConfig::trivial().validate().is_ok());
        for c in [
            LocalVvhvConfig {
                tau: -1.0,
                ..LocalVvhvConfig::default()
            },
            LocalVvhvConfig {
                tau_loose: f64::NAN,
                ..LocalVvhvConfig::default()
            },
            LocalVvhvConfig {
                boys_tau: f64::INFINITY,
                ..LocalVvhvConfig::default()
            },
            LocalVvhvConfig {
                solve_radius_bohr: Some(0.0),
                ..LocalVvhvConfig::default()
            },
        ] {
            assert!(c.validate().is_err(), "{c:?}");
        }
    }

    #[test]
    fn log_line_names_the_build_and_the_share() {
        let t = StageTimings {
            t_spaces_s: 1.0,
            t_loc_s: 0.5,
            t_assembly_s: 2.0,
            t_solve_s: 1.0,
            t_reference_s: 9.0,
        };
        let l = VirtualSpaceReport::Dense.log_line(&t);
        assert!(l.contains("dense VV-HV build"), "{l}");
        assert!(l.contains("37.5% of t_corr 4.000 s"), "{l}");
        let l = VirtualSpaceReport::Local {
            config: LocalVvhvConfig::default(),
            stats: LocalVvhvStats::default(),
        }
        .log_line(&t);
        assert!(
            l.contains("LOCAL VV-HV build") && l.contains("tau 1e-9"),
            "{l}"
        );
    }
}
