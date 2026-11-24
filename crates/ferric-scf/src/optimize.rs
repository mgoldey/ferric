//! Geometry optimization using analytical gradients.
//!
//! Implements the BFGS (Broyden-Fletcher-Goldfarb-Shanno) algorithm for
//! minimizing the molecular energy with respect to nuclear coordinates, in
//! either Cartesian coordinates (the default) or redundant internal
//! coordinates ([`CoordSystem`](crate::optimize::CoordSystem)).

use crate::result::ScfResult;
use crate::rhf::RhfConfig;
use crate::screening::SchwarzBounds;
use crate::warm_start::{solve_rhf_warm, solve_rohf_warm, solve_uhf_warm, ScfWarmStart};
use ferric_core::internal_coords::{
    generalized_inverse, gradient_to_internal, step_to_cartesian, InternalCoords, Primitive,
};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_core::timing::Stage;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ndarray::{Array1, Array2};
use ndarray_linalg::{Eigh, Solve, UPLO};

/// Which coordinate system the BFGS search runs in.
///
/// # Why Cartesian is the default
///
/// Cartesian is the pre-existing, long-validated path and is unchanged by the
/// addition of internals (pinned by
/// `tests/internal_coord_optimize.rs::selecting_internals_does_not_perturb_the_cartesian_path`,
/// which compares the two live in one process, and by
/// `cartesian_path_is_reproducible_and_matches_the_recorded_path`).
/// Internals are opt-in because they add machinery that can fail in ways
/// Cartesians cannot — bond perception can mis-assign connectivity on an
/// unusual geometry, and the back-transformation can fail to converge — and
/// because the measured iteration-count advantage, while real on floppy
/// systems, is not uniform across every system (see the ledger printed by
/// `iteration_count_ledger`). Making the *safer* path the default and the
/// *faster-on-hard-cases* path explicit is the honest ordering: a user who
/// opts in has decided their system is one where torsions dominate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CoordSystem {
    /// Plain Cartesian BFGS with a fixed trust-radius clip. The default.
    #[default]
    Cartesian,
    /// Redundant internal coordinates (bonds, angles, proper dihedrals) with a
    /// Schlegel empirical initial Hessian and an iterative step
    /// back-transformation. Falls back to Cartesian, with a warning, if the
    /// coordinates cannot be constructed for the input geometry.
    RedundantInternal,
}

impl CoordSystem {
    /// Parse a user-facing coordinate-system name: `"cartesian"` (also
    /// `"cart"`) or `"internal"` (also `"internals"`, `"redundant-internal"`,
    /// `"redundant_internal"`), case-insensitive. Strict: any other value is
    /// an error naming the accepted values, never a silent Cartesian.
    ///
    /// # Errors
    ///
    /// Returns a message naming the accepted values if `s` is not one of them.
    pub fn parse_config_str(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "cartesian" | "cart" => Ok(Self::Cartesian),
            "internal" | "internals" | "redundant-internal" | "redundant_internal" => {
                Ok(Self::RedundantInternal)
            }
            other => Err(format!(
                "unknown coordinates = \"{other}\"; expected \"cartesian\" or \"internal\""
            )),
        }
    }
}

impl CoordSystem {
    /// The config spelling: `"cartesian"` or `"internal"`.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cartesian => "cartesian",
            Self::RedundantInternal => "internal",
        }
    }
}

/// What in `cfg` makes the energy depend on where the molecule sits in the
/// lab frame, or `None` when the energy is invariant to a rigid rotation and
/// translation of the optimized atoms.
///
/// Fixed classical charges (point or Gaussian-smeared, including every
/// QM/MM `[qmmm]` MM charge), a uniform field and polarizable embedding sites
/// all stay put while the atoms move, so they exert a net force and torque.
/// COSMO and PCM cavities are built from the atoms and move with them, so
/// they keep the invariance and are not listed. A `fock_hook` is opaque and
/// is not inspected.
#[must_use]
pub fn fixed_frame_cause(cfg: &RhfConfig) -> Option<&'static str> {
    if let Some(ext) = cfg.external_potential.as_ref() {
        if !ext.point_charges.is_empty() || !ext.smeared_charges.is_empty() {
            return Some("fixed point or smeared charges (point_charges / QM/MM MM atoms)");
        }
        if ext.field.is_some() {
            return Some("a uniform external field");
        }
    }
    if cfg.polarizable.is_some() {
        return Some("fixed polarizable embedding sites");
    }
    None
}

/// Refuse `coordinates = "internal"` when the energy is not invariant to a
/// rigid rotation or translation of the optimized atoms ([`fixed_frame_cause`]).
///
/// Internal coordinates (bonds, angles, dihedrals) are blind to those six
/// motions, so a step in them can never remove a net force or torque, while
/// the convergence test reads the Cartesian gradient, which includes both.
/// The search then stalls (water in a 0.02 a.u. field: no convergence in 100
/// steps, 14 mHa high; Cartesian converges in 22). Call this before any SCF.
///
/// # Errors
///
/// A message naming the cause and suggesting `"cartesian"`.
pub fn refuse_internal_coordinates_in_fixed_frame(
    coord_system: CoordSystem,
    cfg: &RhfConfig,
) -> Result<(), FerricError> {
    if coord_system != CoordSystem::RedundantInternal {
        return Ok(());
    }
    match fixed_frame_cause(cfg) {
        None => Ok(()),
        Some(cause) => Err(FerricError::General(format!(
            "coordinates = \"internal\" cannot optimize with {cause}: the energy then depends \
             on the molecule's position and orientation, and internal coordinates cannot \
             translate or rotate it, so the search cannot follow the net force or torque and \
             does not converge. Use coordinates = \"cartesian\" (the default)."
        ))),
    }
}

/// Initial Hessian of the BFGS search.
///
/// Explicit in [`OptimizeConfig::initial_hessian`]; `None` there takes
/// [`DEFAULT_INITIAL_HESSIAN`] on the molecular entry points and
/// [`InitialHessian::Simple`] on [`optimize_coordinates`], which has no atoms
/// to build a model from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InitialHessian {
    /// The identity in Cartesians; Schlegel's empirical diagonal in internals
    /// ([`InternalCoords::initial_hessian_diagonal`]).
    Simple,
    /// The Lindh (1995) model Hessian ([`ferric_core::model_hessian`]). In
    /// Cartesians the full matrix, with the rigid-body block set to the
    /// identity and the other eigenvalues floored at [`MODEL_HESSIAN_FLOOR`];
    /// in internals the Lindh force constant of each primitive on the
    /// diagonal, with the same floor.
    Lindh,
}

impl InitialHessian {
    /// Parse `"simple"` or `"lindh"`, case-insensitive. Strict: any other
    /// value is an error naming the accepted values.
    ///
    /// # Errors
    ///
    /// Returns a message naming the accepted values if `s` is not one of them.
    pub fn parse_config_str(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "simple" => Ok(Self::Simple),
            "lindh" => Ok(Self::Lindh),
            other => Err(format!(
                "unknown initial_hessian = \"{other}\"; expected \"simple\" or \"lindh\""
            )),
        }
    }
}

impl InitialHessian {
    /// The config spelling: `"simple"` or `"lindh"`.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Simple => "simple",
            Self::Lindh => "lindh",
        }
    }
}

/// How the trust radius moves between steps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TrustUpdate {
    /// The radius stays at [`OptimizeConfig::trust_radius`] and every step is
    /// accepted.
    Fixed,
    /// The radius follows the ratio of the actual to the predicted energy
    /// change ([`ratio_trust_update`]), and a step that raises the energy is
    /// rejected. The default.
    #[default]
    Ratio,
}

impl TrustUpdate {
    /// Parse `"fixed"` or `"ratio"`, case-insensitive. Strict.
    ///
    /// # Errors
    ///
    /// Returns a message naming the accepted values if `s` is not one of them.
    pub fn parse_config_str(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "fixed" => Ok(Self::Fixed),
            "ratio" => Ok(Self::Ratio),
            other => Err(format!(
                "unknown trust_update = \"{other}\"; expected \"fixed\" or \"ratio\""
            )),
        }
    }
}

impl TrustUpdate {
    /// The config spelling: `"fixed"` or `"ratio"`.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fixed => "fixed",
            Self::Ratio => "ratio",
        }
    }
}

/// The search an optimization runs: coordinates, initial Hessian and trust
/// rule, each resolved (no `None`). [`OptimizeConfig::requested_search`] is
/// what a config asks for; [`OptimizeResult::search`] is what ran, which
/// differs when internal coordinates could not be built and the search fell
/// back to Cartesian.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SearchSettings {
    /// Coordinates of the BFGS search.
    pub coord_system: CoordSystem,
    /// Initial Hessian.
    pub initial_hessian: InitialHessian,
    /// Trust-radius rule.
    pub trust_update: TrustUpdate,
}

impl std::fmt::Display for SearchSettings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} coordinates, {} initial Hessian, {} trust radius",
            self.coord_system.as_str(),
            self.initial_hessian.as_str(),
            self.trust_update.as_str()
        )
    }
}

/// The initial Hessian an unset [`OptimizeConfig::initial_hessian`] takes on
/// the molecular entry points. Lindh: on eight molecules from water to
/// sulfamethoxazole at HF/STO-3G it cut the Cartesian step count from 538 to
/// 211 and the internal one from 252 to 139 (with [`TrustUpdate::Ratio`]);
/// `book/src/14-derivatives/08-geometry-search.md#geometry-search-lindh-model-hessian` has the
/// table.
pub const DEFAULT_INITIAL_HESSIAN: InitialHessian = InitialHessian::Lindh;

/// Floor on the model Hessian's eigenvalues (Cartesian) or diagonal
/// (internals), Hartree/Bohr² or Hartree/rad². Directions the model gives no
/// curvature (a linear molecule's bends, whose primitives are dropped) would
/// otherwise get an unbounded inverse. 1e-3 is below a single torsion's
/// Cartesian curvature (`k_tau |b|^2`, about 3e-3 for a bonded torsion), so
/// it does not stiffen a real mode.
pub const MODEL_HESSIAN_FLOOR: f64 = 1e-3;

/// Ratio below which the trust radius shrinks ([`ratio_trust_update`]).
pub const TRUST_SHRINK_RATIO: f64 = 0.25;
/// Ratio above which a step on the boundary grows the trust radius.
pub const TRUST_GROW_RATIO: f64 = 0.75;
/// Energy changes below this (Hartree) are treated as noise by the ratio rule.
pub const TRUST_ENERGY_NOISE: f64 = 1e-8;
/// Trust-radius bounds, as multiples of [`OptimizeConfig::trust_radius`]
/// (Cartesian) or of the starting scale 1 (internals).
pub const TRUST_SCALE_MIN: f64 = 0.01;
/// Upper bound for the Cartesian trust radius, as a multiple of
/// [`OptimizeConfig::trust_radius`].
pub const CARTESIAN_TRUST_SCALE_MAX: f64 = 10.0;
/// Upper bound for the internal-coordinate trust scale (it multiplies both the
/// distance and the angular radius).
pub const INTERNAL_TRUST_SCALE_MAX: f64 = 4.0;

/// One decision of the ratio trust-radius rule (Nocedal and Wright,
/// "Numerical Optimization", 2nd ed., Algorithm 4.1, with `eta = 0`).
///
/// `actual` and `predicted` are the energy change and the quadratic model's
/// prediction for the step (`predicted < 0` for a descent step);
/// `step_norm` is the step length in the units of `radius`. Returns whether
/// the step is accepted and the next radius:
///
/// * the step is rejected, or `predicted >= 0` (the model does not predict a
///   descent), or `r = actual / predicted < TRUST_SHRINK_RATIO`: radius
///   `TRUST_SHRINK_RATIO * step_norm`, so a rejected step never grows it;
/// * `r > TRUST_GROW_RATIO` and the step reached the boundary: radius doubled;
/// * otherwise unchanged; always clamped to `[r_min, r_max]`.
///
/// The step is accepted when the energy fell (`r > 0`), or rose by less
/// than [`TRUST_ENERGY_NOISE`]. When both changes are below the noise the
/// ratio carries no information and the radius is kept.
#[must_use]
pub fn ratio_trust_update(
    actual: f64,
    predicted: f64,
    step_norm: f64,
    radius: f64,
    r_min: f64,
    r_max: f64,
) -> (bool, f64) {
    let accept = actual < TRUST_ENERGY_NOISE;
    if predicted.abs() < TRUST_ENERGY_NOISE && actual.abs() < TRUST_ENERGY_NOISE {
        return (accept, radius);
    }
    let ratio = actual / predicted;
    let next = if !accept || predicted >= 0.0 || ratio < TRUST_SHRINK_RATIO {
        TRUST_SHRINK_RATIO * step_norm
    } else if ratio > TRUST_GROW_RATIO && step_norm >= 0.99 * radius {
        2.0 * radius
    } else {
        radius
    };
    (accept, next.clamp(r_min, r_max))
}

/// Configuration for geometry optimization.
#[derive(Debug, Clone)]
pub struct OptimizeConfig {
    /// Maximum number of optimization steps.
    pub max_steps: usize,
    /// Convergence threshold for the maximum gradient component (Hartree/Bohr).
    pub g_max_thresh: f64,
    /// Convergence threshold for the RMS gradient (Hartree/Bohr).
    pub g_rms_thresh: f64,
    /// Convergence threshold for the energy change (Hartree).
    pub e_conv: f64,
    /// Trust radius: a BFGS step longer than this (2-norm over the coordinate
    /// vector) is scaled down onto it. There is no line search; the scaled
    /// step is always accepted.
    pub trust_radius: f64,
    /// Coordinate system for the search. Defaults to
    /// [`CoordSystem::Cartesian`], which is bit-identical to the behaviour
    /// before internals existed.
    pub coord_system: CoordSystem,
    /// Initial Hessian; `None` takes the entry point's default (see
    /// [`InitialHessian`]).
    pub initial_hessian: Option<InitialHessian>,
    /// Trust-radius rule.
    pub trust_update: TrustUpdate,
    /// Start each geometry's SCF from the previous geometry's converged MOs
    /// (re-orthonormalized in the new overlap) instead of the default guess.
    /// See [`crate::warm_start`] for the construction, the cold-guess
    /// fallbacks and the retry on a failed warm start. Default `true`; the
    /// SCF energies it reaches agree with a cold start to within the SCF's own
    /// convergence tolerance, not bitwise.
    pub scf_warm_start: bool,
}

impl Default for OptimizeConfig {
    fn default() -> Self {
        Self {
            max_steps: 100,
            g_max_thresh: 4.5e-4,
            g_rms_thresh: 3.0e-4,
            e_conv: 1.0e-6,
            trust_radius: 0.1,
            coord_system: CoordSystem::Cartesian,
            initial_hessian: None,
            trust_update: TrustUpdate::default(),
            scf_warm_start: true,
        }
    }
}

impl OptimizeConfig {
    /// The search the molecular entry points run for this config: the
    /// coordinates as set, an unset initial Hessian as
    /// [`DEFAULT_INITIAL_HESSIAN`].
    #[must_use]
    pub fn requested_search(&self) -> SearchSettings {
        SearchSettings {
            coord_system: self.coord_system,
            initial_hessian: self.initial_hessian.unwrap_or(DEFAULT_INITIAL_HESSIAN),
            trust_update: self.trust_update,
        }
    }
}

/// Maximum passes of the iterative internal→Cartesian back-transformation
/// before the step is declared non-convergent and halved.
const BACKTRANSFORM_MAX_ITER: usize = 25;

/// Convergence tolerance for the back-transformation, as a maximum remaining
/// **Cartesian** correction in Bohr (see
/// [`ferric_core::internal_coords::step_to_cartesian`] for why the residual is
/// measured in Cartesians rather than in the internal coordinates themselves).
///
/// 1e-9 Bohr is five orders below the geometry change implied by the
/// optimizer's own gradient threshold, so the back-transformation is never what
/// limits the converged geometry, while staying comfortably above the level at
/// which the iteration would be chasing floating-point noise.
const BACKTRANSFORM_TOL: f64 = 1e-9;

/// Trust radius for the internal-coordinate step, in the primitives' own units.
///
/// Separate from [`OptimizeConfig::trust_radius`] (Bohr, Cartesian) because
/// radians and Bohr are not comparable: 0.3 rad is 17 degrees, a sensible
/// maximum torsional move, while 0.3 Bohr would be a reckless bond step. The
/// bond and angle components are clipped against the caller's Cartesian
/// `trust_radius`; angular components against this.
const INTERNAL_ANGULAR_TRUST: f64 = 0.3;

/// Result of a geometry optimization.
#[derive(Debug, Clone)]
#[must_use = "optimization result contains the relaxed geometry and energy"]
pub struct OptimizeResult {
    /// The molecule at the final geometry (coordinates in Bohr).
    pub mol: Molecule,
    /// Energy at `mol`, in Hartree.
    pub energy: f64,
    /// Optimizer steps taken (coordinate updates), at most
    /// [`OptimizeConfig::max_steps`].
    pub steps: usize,
    /// Whether, after at least one step, the energy change and the max and
    /// RMS Cartesian gradient components all fell below
    /// [`OptimizeConfig::e_conv`], [`OptimizeConfig::g_max_thresh`] and
    /// [`OptimizeConfig::g_rms_thresh`].
    pub converged: bool,
    /// Energy at every point the optimizer EVALUATED, in order.
    ///
    /// `energy`, `steps` and `converged` cannot distinguish "ran out of steps
    /// near a minimum" from "walked uphill and oscillated" -- a run that climbs
    /// -39.82 -> -39.53 Ha over 200 steps reports the same three fields as a
    /// slow approach. The trace is what makes that visible, and it is what
    /// `tools.viz.energy_plots.optimization_trace` plots.
    ///
    /// This is every EVALUATION, including the steps the ratio trust rule
    /// rejected. Under [`TrustUpdate::Fixed`] every step is accepted, so an
    /// uphill step shows up here as a rise in the trace.
    pub energy_trace: Vec<f64>,
    /// Evaluated steps the ratio trust rule rejected (always 0 under
    /// [`TrustUpdate::Fixed`]). Counted in `steps`, absent from `mol`.
    pub rejected_steps: usize,
    /// The search that ran (Cartesian if internal coordinates were asked for
    /// but could not be built).
    pub search: SearchSettings,
}

impl std::fmt::Display for OptimizeResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Optimized energy: {:.10} Ha ({} steps, converged: {})",
            self.energy, self.steps, self.converged
        )
    }
}

/// Optimize the molecular geometry using RHF (or closed-shell KS-DFT, via
/// `rhf_config.xc`) analytical gradients.
pub fn optimize_geometry(
    ctx: &ParallelContext,
    mol: &Molecule,
    basis_name: &str,
    op: Operator,
    rhf_config: &RhfConfig,
    opt_config: &OptimizeConfig,
) -> Result<OptimizeResult, FerricError> {
    optimize_geometry_with_correction(ctx, mol, basis_name, op, rhf_config, opt_config, |_| {
        Ok((0.0, None))
    })
}

/// [`optimize_geometry`] plus a caller-supplied ADDITIVE energy correction.
///
/// `correction(mol)` returns `(energy, Option<gradient>)` in Hartree and
/// Hartree/Bohr, both added to the SCF values at every step. The motivating
/// case is empirical dispersion: D3(BJ) is additive in the energy and
/// therefore in the gradient, but it is an EMPIRICAL model and `ferric-scf`
/// has no business depending on one. Passing it as a closure keeps this crate
/// dispersion-agnostic -- `ferric-cli` supplies D3, and a caller with a
/// different correction supplies that instead.
///
/// **An energy without a gradient is a hard error, not a zero.** A correction
/// that can report an energy but not its derivative would make the optimizer
/// walk one surface while reporting another, converging to a geometry that is
/// a stationary point of neither.
///
/// With the default `|_| Ok((0.0, None))` this is byte-identical to the
/// uncorrected optimizer: the `de != 0.0 || dg.is_some()` guard skips the
/// addition entirely, so no floating-point operation is introduced on the
/// path that existed before. Pinned by
/// `a_zero_correction_is_byte_identical_to_no_correction`.
pub fn optimize_geometry_with_correction(
    ctx: &ParallelContext,
    mol: &Molecule,
    basis_name: &str,
    op: Operator,
    rhf_config: &RhfConfig,
    opt_config: &OptimizeConfig,
    mut correction: impl FnMut(&Molecule) -> Result<(f64, Option<Array2<f64>>), FerricError>,
) -> Result<OptimizeResult, FerricError> {
    optimize_geometry_with_scf_correction(
        ctx,
        mol,
        basis_name,
        op,
        rhf_config,
        opt_config,
        |m, _scf| correction(m),
    )
}

/// [`optimize_geometry_with_correction`] whose correction also sees the
/// converged closed-shell SCF result at the geometry being evaluated.
///
/// For corrections that depend on the electron density, not only on the
/// nuclear positions -- MBD@rsSCS takes its per-atom polarizabilities from
/// Hirshfeld volumes of the SCF density. `correction(mol, scf)` is called once
/// per evaluated geometry, after the SCF and its gradient, with the
/// [`ScfResult`] that produced the SCF energy; its return value is combined
/// exactly as in [`optimize_geometry_with_correction`] (same guard, same
/// refusals), which delegates here.
pub fn optimize_geometry_with_scf_correction(
    ctx: &ParallelContext,
    mol: &Molecule,
    basis_name: &str,
    op: Operator,
    rhf_config: &RhfConfig,
    opt_config: &OptimizeConfig,
    mut correction: impl FnMut(&Molecule, &ScfResult) -> Result<(f64, Option<Array2<f64>>), FerricError>,
) -> Result<OptimizeResult, FerricError> {
    // Geometry task: an unset `ri_j_precision` resolves to f64 (jagged f32
    // surface; `RhfConfig::ri_j_task`), an explicit one is kept.
    refuse_internal_coordinates_in_fixed_frame(opt_config.coord_system, rhf_config)?;
    let rhf_config = &rhf_config.for_geometry_task();
    let mut warm = ScfWarmStart::new(opt_config.scf_warm_start);
    let out = run_bfgs(mol, opt_config, |m| {
        let t = Stage::start("opt-step:eval-total");
        let (e, g, scf) =
            compute_energy_gradient_and_result(ctx, m, basis_name, op, rhf_config, &mut warm)?;
        let (de, dg) = correction(m, &scf)?;
        let out = add_correction(e, g, de, dg);
        t.end();
        out
    });
    print_warm_start_summary(&warm);
    out
}

/// Print the [`ScfWarmStart::summary`] line after an optimization (nothing
/// when warm start is off).
fn print_warm_start_summary(warm: &ScfWarmStart) {
    if let Some(line) = warm.summary() {
        println!("{line}");
    }
}

/// Combine an SCF (energy, gradient) with an additive correction under the
/// rules of [`optimize_geometry_with_correction`]: `(0.0, None)` adds nothing
/// (no floating-point operation), an energy without a gradient and a
/// mis-shaped gradient are errors.
pub(crate) fn add_correction(
    e: f64,
    mut g: Array2<f64>,
    de: f64,
    dg: Option<Array2<f64>>,
) -> Result<(f64, Array2<f64>), FerricError> {
    if de != 0.0 || dg.is_some() {
        let dg = dg.ok_or_else(|| {
            FerricError::General(
                "geometry optimization: the energy correction returned a value but no \
                 gradient. Optimizing would follow the UNCORRECTED surface while \
                 reporting corrected energies, converging to a geometry that is a \
                 stationary point of neither."
                    .to_string(),
            )
        })?;
        if dg.shape() != g.shape() {
            return Err(FerricError::General(format!(
                "geometry optimization: correction gradient is {:?} but the SCF \
                 gradient is {:?}",
                dg.shape(),
                g.shape()
            )));
        }
        g = g + dg;
    }
    Ok((e + de, g))
}

/// Optimize the molecular geometry using UHF analytical gradients.
///
/// `mol.charge`/`mol.multiplicity` fix the spin state for every step (the
/// occupation is not re-derived from a Aufbau guess mid-optimization, mirroring
/// how `solve_uhf` itself works — the caller is responsible for choosing a
/// multiplicity that stays the correct ground state along the whole path).
pub fn optimize_geometry_uhf(
    ctx: &ParallelContext,
    mol: &Molecule,
    basis_name: &str,
    op: Operator,
    uhf_config: &RhfConfig,
    opt_config: &OptimizeConfig,
) -> Result<OptimizeResult, FerricError> {
    optimize_geometry_uhf_with_scf_correction(
        ctx,
        mol,
        basis_name,
        op,
        uhf_config,
        opt_config,
        |_, _| Ok((0.0, None)),
    )
}

/// [`optimize_geometry_uhf`] plus an additive correction that sees the
/// converged UHF/UKS result at each geometry: the open-shell sibling of
/// [`optimize_geometry_with_scf_correction`], with the same combination rules
/// (an energy without a gradient is an error; `(0.0, None)` is byte-identical
/// to no correction). Dispersion is the motivating case — D3(BJ) uses only
/// the geometry, MBD@rsSCS the UKS spin densities.
pub fn optimize_geometry_uhf_with_scf_correction(
    ctx: &ParallelContext,
    mol: &Molecule,
    basis_name: &str,
    op: Operator,
    uhf_config: &RhfConfig,
    opt_config: &OptimizeConfig,
    mut correction: impl FnMut(&Molecule, &ScfResult) -> Result<(f64, Option<Array2<f64>>), FerricError>,
) -> Result<OptimizeResult, FerricError> {
    // Geometry task: an unset `ri_j_precision` resolves to f64 (jagged f32
    // surface; `RhfConfig::ri_j_task`), an explicit one is kept.
    refuse_internal_coordinates_in_fixed_frame(opt_config.coord_system, uhf_config)?;
    let uhf_config = &uhf_config.for_geometry_task();
    let mut warm = ScfWarmStart::new(opt_config.scf_warm_start);
    let out = run_bfgs(mol, opt_config, |m| {
        let (e, g, scf) =
            compute_energy_and_gradient_uhf(ctx, m, basis_name, op, uhf_config, &mut warm)?;
        let (de, dg) = correction(m, &scf)?;
        add_correction(e, g, de, dg)
    });
    print_warm_start_summary(&warm);
    out
}

/// Optimize the molecular geometry using ROHF analytical gradients.
pub fn optimize_geometry_rohf(
    ctx: &ParallelContext,
    mol: &Molecule,
    basis_name: &str,
    op: Operator,
    rohf_config: &RhfConfig,
    opt_config: &OptimizeConfig,
) -> Result<OptimizeResult, FerricError> {
    optimize_geometry_rohf_with_scf_correction(
        ctx,
        mol,
        basis_name,
        op,
        rohf_config,
        opt_config,
        |_, _| Ok((0.0, None)),
    )
}

/// [`optimize_geometry_rohf`] plus an additive correction that sees the
/// converged ROHF/ROKS result at each geometry: the restricted-open-shell
/// sibling of [`optimize_geometry_uhf_with_scf_correction`], with the same
/// combination rules (an energy without a gradient is an error; `(0.0, None)`
/// is byte-identical to no correction). Dispersion is the motivating case —
/// D3(BJ) uses only the geometry, MBD@rsSCS the ROKS spin densities.
pub fn optimize_geometry_rohf_with_scf_correction(
    ctx: &ParallelContext,
    mol: &Molecule,
    basis_name: &str,
    op: Operator,
    rohf_config: &RhfConfig,
    opt_config: &OptimizeConfig,
    mut correction: impl FnMut(&Molecule, &ScfResult) -> Result<(f64, Option<Array2<f64>>), FerricError>,
) -> Result<OptimizeResult, FerricError> {
    // Geometry task: an unset `ri_j_precision` resolves to f64 (jagged f32
    // surface; `RhfConfig::ri_j_task`), an explicit one is kept.
    refuse_internal_coordinates_in_fixed_frame(opt_config.coord_system, rohf_config)?;
    let rohf_config = &rohf_config.for_geometry_task();
    let mut warm = ScfWarmStart::new(opt_config.scf_warm_start);
    let out = run_bfgs(mol, opt_config, |m| {
        let (e, g, scf) =
            compute_energy_and_gradient_rohf(ctx, m, basis_name, op, rohf_config, &mut warm)?;
        let (de, dg) = correction(m, &scf)?;
        add_correction(e, g, de, dg)
    });
    print_warm_start_summary(&warm);
    out
}

/// Shared BFGS driver: minimizes `energy_and_gradient(mol)` over nuclear
/// coordinates starting from `mol`. Identical algorithm for every reference
/// (RHF/RKS/UHF/ROHF) — only how energy+gradient are computed at a geometry
/// differs, which is captured entirely in the closure. A thin wrapper over
/// [`optimize_coordinates`] that flattens/unflattens the `Molecule` into a
/// coordinate vector at the boundary.
fn run_bfgs(
    mol: &Molecule,
    opt_config: &OptimizeConfig,
    energy_and_gradient: impl FnMut(&Molecule) -> Result<(f64, Array2<f64>), FerricError>,
) -> Result<OptimizeResult, FerricError> {
    if opt_config.coord_system == CoordSystem::RedundantInternal {
        // Build the coordinate system ONCE, from the starting geometry, and
        // reuse it for every step. Regenerating it mid-run would silently
        // invalidate the BFGS history, which is indexed against this exact
        // primitive list.
        match InternalCoords::generate(mol) {
            Ok(coords) => {
                if coords.connectivity.n_fragments > 1 {
                    println!(
                        "Note: {} disconnected fragments perceived; added {} interfragment \
                         link(s) so the internal coordinates span their relative motion.",
                        coords.connectivity.n_fragments,
                        coords.connectivity.interfragment_bonds.len()
                    );
                }
                return run_bfgs_internal(mol, &coords, opt_config, energy_and_gradient);
            }
            Err(e) => {
                // Falling back is better than failing: the Cartesian path
                // always works, and the user asked for an optimization, not
                // for a particular coordinate system. But say so loudly — a
                // silent downgrade would make the iteration counts
                // uninterpretable.
                println!(
                    "Warning: could not build redundant internal coordinates ({e}); \
                     falling back to Cartesian BFGS."
                );
            }
        }
    }
    run_bfgs_cartesian(mol, opt_config, energy_and_gradient)
}

/// The Cartesian search of [`run_bfgs`]: the seed from
/// [`OptimizeConfig::initial_hessian`], then [`bfgs_cartesian`].
fn run_bfgs_cartesian(
    mol: &Molecule,
    opt_config: &OptimizeConfig,
    mut energy_and_gradient: impl FnMut(&Molecule) -> Result<(f64, Array2<f64>), FerricError>,
) -> Result<OptimizeResult, FerricError> {
    let x0 = flatten_molecule_coords(mol);
    let mol_template = mol.clone();

    // Record every evaluation as it happens. The closure is the only place
    // that sees them all, and threading a callback through
    // `optimize_coordinates` would change a signature two callers share for
    // no gain over this.
    let mut energy_trace: Vec<f64> = Vec::new();
    let initial = opt_config
        .initial_hessian
        .unwrap_or(DEFAULT_INITIAL_HESSIAN);
    let seed = |x: &[f64]| -> Result<Array2<f64>, FerricError> {
        match initial {
            InitialHessian::Simple => Ok(Array2::eye(x.len())),
            InitialHessian::Lindh => {
                let mut m = mol_template.clone();
                set_molecule_coords(&mut m, x);
                lindh_inverse_hessian(&m)
            }
        }
    };
    let out = bfgs_cartesian(&x0, opt_config, &seed, |x| {
        let mut m = mol_template.clone();
        set_molecule_coords(&mut m, x);
        let (e, grad_arr) = energy_and_gradient(&m)?;
        energy_trace.push(e);
        Ok((e, flatten_gradient(&grad_arr).to_vec()))
    })?;

    let mut current_mol = mol_template;
    set_molecule_coords(&mut current_mol, &out.x);

    Ok(OptimizeResult {
        mol: current_mol,
        energy: out.energy,
        steps: out.steps,
        converged: out.converged,
        energy_trace,
        rejected_steps: out.rejected,
        search: SearchSettings {
            coord_system: CoordSystem::Cartesian,
            ..opt_config.requested_search()
        },
    })
}

/// Rigid-body motions of `mol` (translations, and rotations about the
/// centroid) as orthonormal columns of a `(3N, m)` matrix; `m` is 6, or 5 for
/// a linear molecule, or 3 for an atom.
fn rigid_body_basis(mol: &Molecule) -> Array2<f64> {
    let n = mol.atoms.len();
    let mut c = [0.0; 3];
    for a in &mol.atoms {
        c[0] += a.x / n as f64;
        c[1] += a.y / n as f64;
        c[2] += a.zpos / n as f64;
    }
    let mut cols: Vec<Array1<f64>> = Vec::new();
    let mut push = |mut v: Array1<f64>| {
        for u in &cols {
            let d = u.dot(&v);
            v = &v - &(u * d);
        }
        let nv = v.dot(&v).sqrt();
        if nv > 1e-6 {
            cols.push(v / nv);
        }
    };
    for k in 0..3 {
        let mut v = Array1::zeros(3 * n);
        for a in 0..n {
            v[3 * a + k] = 1.0;
        }
        push(v);
    }
    for axis in 0..3 {
        let (p, q) = ((axis + 1) % 3, (axis + 2) % 3);
        let mut v = Array1::zeros(3 * n);
        for (i, a) in mol.atoms.iter().enumerate() {
            let r = [a.x - c[0], a.y - c[1], a.zpos - c[2]];
            v[3 * i + q] += r[p];
            v[3 * i + p] -= r[q];
        }
        push(v);
    }
    let mut t = Array2::zeros((3 * n, cols.len()));
    for (j, v) in cols.iter().enumerate() {
        t.column_mut(j).assign(v);
    }
    t
}

/// Inverse of the Lindh model Hessian for the Cartesian search.
///
/// The model has the rigid-body motions `T` in its null space. On the
/// complement `Q` (orthonormal, `QᵀT = 0`) the projected model `QᵀHQ` is
/// inverted with its eigenvalues floored at [`MODEL_HESSIAN_FLOOR`]; on `T`
/// the inverse is the identity, the same as the simple start, so a gradient
/// with a rigid-body component (an external field) moves the molecule as it
/// did before:
///
/// `H⁻¹ = Q V diag(1/max(w, floor)) Vᵀ Qᵀ + T Tᵀ`.
fn lindh_inverse_hessian(mol: &Molecule) -> Result<Array2<f64>, FerricError> {
    let h = ferric_core::model_hessian::lindh_cartesian_hessian(mol)?;
    model_inverse_with_rigid_identity(&h, mol)
}

/// `H⁻¹ = Q V diag(1/max(w, floor)) Vᵀ Qᵀ + T Tᵀ` for a model `h` over the
/// atoms of `mol` (see [`lindh_inverse_hessian`]).
fn model_inverse_with_rigid_identity(
    h: &Array2<f64>,
    mol: &Molecule,
) -> Result<Array2<f64>, FerricError> {
    let n = h.nrows();
    let t = rigid_body_basis(mol);
    let p = Array2::<f64>::eye(n) - t.dot(&t.t());
    let (pw, pv) = p
        .eigh(UPLO::Upper)
        .map_err(|e| FerricError::General(format!("rigid-body projector eigh: {e}")))?;
    let keep: Vec<usize> = (0..n).filter(|&i| pw[i] > 0.5).collect();
    if keep.is_empty() {
        // Only rigid motions (one free atom): the simple start.
        return Ok(t.dot(&t.t()));
    }
    let mut q = Array2::zeros((n, keep.len()));
    for (j, &i) in keep.iter().enumerate() {
        q.column_mut(j).assign(&pv.column(i));
    }
    let hq = q.t().dot(h).dot(&q);
    let (w, v) = hq
        .eigh(UPLO::Upper)
        .map_err(|e| FerricError::General(format!("Lindh model Hessian eigh: {e}")))?;
    let qv = q.dot(&v);
    let mut scaled = qv.clone();
    for (j, &wj) in w.iter().enumerate() {
        let inv = 1.0 / wj.max(MODEL_HESSIAN_FLOOR);
        scaled.column_mut(j).mapv_inplace(|x| x * inv);
    }
    Ok(scaled.dot(&qv.t()) + t.dot(&t.t()))
}

/// The atoms of `mol` that share a Lindh term with an atom of `free`
/// (`free` included), sorted.
///
/// A term is a chain of at most four atoms whose `rho` product is at least
/// [`ferric_core::model_hessian::LINDH_RHO_CUTOFF`]. Every factor is at most
/// `rho_max`, so each link of such a chain has `rho >= cutoff / rho_max²`,
/// and every atom of a term that contains a free atom is within three such
/// links of it. The set is that three-link closure.
fn lindh_environment(mol: &Molecule, free: &[usize]) -> Vec<usize> {
    use ferric_core::model_hessian::{lindh_rho, LINDH_RHO_CUTOFF};
    let n = mol.atoms.len();
    let mut rho_max = 1.0f64;
    for i in 0..n {
        for j in i + 1..n {
            rho_max = rho_max.max(lindh_rho(mol, i, j));
        }
    }
    let pair_cut = LINDH_RHO_CUTOFF / (rho_max * rho_max);
    let mut inside = vec![false; n];
    let mut frontier: Vec<usize> = free.to_vec();
    for &i in free {
        inside[i] = true;
    }
    for _hop in 0..3 {
        let mut next = Vec::new();
        for &i in &frontier {
            for j in 0..n {
                if !inside[j] && lindh_rho(mol, i, j) >= pair_cut {
                    inside[j] = true;
                    next.push(j);
                }
            }
        }
        frontier = next;
    }
    (0..n).filter(|&i| inside[i]).collect()
}

/// Inverse Lindh model Hessian over the atoms `free` of `mol` (sorted,
/// distinct), with every other atom held fixed: the seed of a search that
/// moves only those atoms, as `optimize_qmmm` does.
///
/// With the others fixed the search's Hessian is the free-free block `H_ff`
/// of the model of the whole structure, so the terms that tie a free atom to
/// a fixed one (a boundary bond to an MM host) are kept. `H_ff` is built from
/// the model of [`lindh_environment`] only (the same block, without the
/// `(3N)²` matrix of a large MM region), and inverted as in
/// [`lindh_inverse_hessian`], with the rigid-body motions of the free atoms
/// on the identity. Fixed atoms make those motions physical, but the model
/// gives them no curvature unless a fixed atom is bonded, so the identity
/// (the simple start) is the safe choice there. With every atom free this is
/// [`lindh_inverse_hessian`] of `mol`, the molecular seed.
///
/// # Errors
///
/// Propagates the model's degenerate-primitive error.
pub(crate) fn lindh_inverse_hessian_free(
    mol: &Molecule,
    free: &[usize],
) -> Result<Array2<f64>, FerricError> {
    let n = mol.atoms.len();
    if free.len() == n && free.iter().enumerate().all(|(k, &i)| k == i) {
        return lindh_inverse_hessian(mol);
    }
    let sub = Molecule {
        atoms: free.iter().map(|&i| mol.atoms[i].clone()).collect(),
        ..mol.clone()
    };
    model_inverse_with_rigid_identity(&lindh_free_block(mol, free)?, &sub)
}

/// The free-free block `H_ff` of the Lindh model of `mol`, built from the
/// model of [`lindh_environment`] (the same block to rounding).
fn lindh_free_block(mol: &Molecule, free: &[usize]) -> Result<Array2<f64>, FerricError> {
    let env = lindh_environment(mol, free);
    let sub = Molecule {
        atoms: env.iter().map(|&i| mol.atoms[i].clone()).collect(),
        ..mol.clone()
    };
    let h_env = ferric_core::model_hessian::lindh_cartesian_hessian(&sub)?;
    let pos: Vec<usize> = free
        .iter()
        .map(|f| {
            env.binary_search(f)
                .expect("free atoms are in their environment")
        })
        .collect();
    let nf = free.len();
    let mut h_ff = Array2::<f64>::zeros((3 * nf, 3 * nf));
    for (a, &pa) in pos.iter().enumerate() {
        for (b, &pb) in pos.iter().enumerate() {
            for c in 0..3 {
                for d in 0..3 {
                    h_ff[(3 * a + c, 3 * b + d)] = h_env[(3 * pa + c, 3 * pb + d)];
                }
            }
        }
    }
    Ok(h_ff)
}

/// BFGS in redundant internal coordinates.
///
/// # The loop
///
/// 1. Compute energy and the **Cartesian** gradient at the current geometry.
/// 2. Test convergence on the **Cartesian** gradient — deliberately the same
///    test the Cartesian path uses, so "converged" means the same thing in
///    both coordinate systems and the two are comparable. A coordinate system
///    is a parameterization of the search, not of the convergence criterion.
/// 3. Transform the gradient to internals: `g_q = G^- B g_x`.
/// 4. Take a BFGS step in internals ([`internal_trial_step`]): each block
///    clipped onto its own radius under `fixed`, the whole step by one factor
///    under `ratio`.
/// 5. Back-transform iteratively to Cartesians; halve the step and retry if the
///    back-transformation fails to converge.
/// 6. Update the inverse Hessian with the internal-coordinate `s` and `y`.
///
/// # Why the Hessian is initialized from Schlegel rather than the identity
///
/// This is the whole point. In internals the diagonal force constants differ by
/// two orders of magnitude between a stretch and a torsion, and the empirical
/// guess gets each within about a factor of two
/// (`initial_hessian_is_within_an_order_of_magnitude_of_reality`). An identity
/// Hessian in *Cartesians* cannot express that separation at all, because every
/// Cartesian coordinate mixes stiff and soft modes.
fn run_bfgs_internal(
    mol: &Molecule,
    coords: &InternalCoords,
    opt_config: &OptimizeConfig,
    mut energy_and_gradient: impl FnMut(&Molecule) -> Result<(f64, Array2<f64>), FerricError>,
) -> Result<OptimizeResult, FerricError> {
    let nq = coords.len();
    let mut current = mol.clone();

    let (mut energy, grad_arr) = energy_and_gradient(&current)?;
    // Same trace the Cartesian path records -- see `OptimizeResult::energy_trace`.
    let mut energy_trace: Vec<f64> = vec![energy];
    let mut grad_cart = flatten_gradient(&grad_arr);

    // Inverse Hessian in internals, seeded from the empirical diagonal. BFGS
    // updates the INVERSE, so seed with 1/k.
    let initial = opt_config
        .initial_hessian
        .unwrap_or(DEFAULT_INITIAL_HESSIAN);
    let mut h_inv = internal_inverse_seed(coords, &current, initial)?;
    // Positive diagonal metric of the fallback step under the ratio rule.
    let h_seed = h_inv.clone();

    let mut prev_energy = energy;
    let mut converged = false;
    let mut step_idx = 0;
    let mut rejected = 0;
    let mut fresh_point = true;
    // Multiplies both the distance and the angular trust radius.
    let mut trust_scale = 1.0;

    println!(
        "Redundant internal coordinates: {} primitives ({} bonds, {} angles, {} dihedrals) \
         over {} atoms",
        nq,
        coords
            .primitives
            .iter()
            .filter(|p| matches!(p, Primitive::Bond(..)))
            .count(),
        coords
            .primitives
            .iter()
            .filter(|p| matches!(p, Primitive::Angle(..)))
            .count(),
        coords
            .primitives
            .iter()
            .filter(|p| matches!(p, Primitive::Dihedral(..)))
            .count(),
        coords.natoms,
    );
    println!("Step | Energy (Ha) | Delta E | Max Grad | RMS Grad");
    println!("-----+-------------+---------+----------+---------");

    while step_idx < opt_config.max_steps {
        if fresh_point {
            let n_cart = grad_cart.len();
            let g_max = grad_cart.iter().map(|g| g.abs()).fold(0.0f64, f64::max);
            let g_rms = (grad_cart.iter().map(|g| g * g).sum::<f64>() / n_cart as f64).sqrt();
            let e_diff = (energy - prev_energy).abs();
            print_step_row(step_idx, energy, prev_energy, g_max, g_rms);

            // Same convergence test as the Cartesian path, on the same
            // (Cartesian) gradient — so "converged" is one definition, not two.
            if step_idx > 0
                && e_diff < opt_config.e_conv
                && g_max < opt_config.g_max_thresh
                && g_rms < opt_config.g_rms_thresh
            {
                converged = true;
                break;
            }
        }

        let b = coords.b_matrix(&current)?;
        let ginv = generalized_inverse(&b)?;
        let g_int = gradient_to_internal(
            &b,
            &ginv,
            grad_cart.as_slice().expect("gradient is contiguous"),
        )?;

        // BFGS step in internals.
        let dist_trust = trust_scale * opt_config.trust_radius;
        let ang_trust = trust_scale * INTERNAL_ANGULAR_TRUST;
        let (dq, fell_back) = internal_trial_step(
            coords,
            &g_int,
            &h_inv,
            &h_seed,
            dist_trust,
            ang_trust,
            opt_config.trust_update,
        );
        if fell_back {
            println!(
                "     | step {}: -H g is not a descent direction; \
                 stepping along -H0 g (the seed metric)",
                step_idx + 1
            );
        }

        // The requested step is deliberately dropped inside the helper: from
        // here on only the ACHIEVED displacement (measured from the two
        // geometries by `bfgs_secant_step`) is allowed to reach the Hessian
        // update. Keeping the requested vector in scope is what made the old
        // "s = requested step" defect possible.
        let new_coords = back_transform_with_retries(coords, &current, &dq)?;

        let q_before = coords.values(&current)?;
        let mut trial = current.clone();
        set_molecule_coords(&mut trial, &new_coords);
        let q_after = coords.values(&trial)?;

        let (e_new, grad_arr_new) = energy_and_gradient(&trial)?;
        energy_trace.push(e_new);
        let grad_cart_new = flatten_gradient(&grad_arr_new);

        let b_new = coords.b_matrix(&trial)?;
        let ginv_new = generalized_inverse(&b_new)?;
        let g_int_new = gradient_to_internal(
            &b_new,
            &ginv_new,
            grad_cart_new.as_slice().expect("gradient is contiguous"),
        )?;

        // The BFGS `s` must be the displacement ACTUALLY achieved, not the one
        // requested: the back-transformation is iterative and the two differ
        // whenever it stops short. Using the requested step would feed the
        // Hessian update a lie about the curvature.
        let s = bfgs_secant_step(coords, &q_before, &q_after);
        debug_assert_eq!(s.len(), nq);

        let accept = match opt_config.trust_update {
            TrustUpdate::Fixed => true,
            TrustUpdate::Ratio => {
                // Quadratic model g.s + s.Bs/2 on the ACHIEVED internal step,
                // B = H^-1 (H is positive definite: the guarded update keeps it so).
                let bs = h_inv.solve(&s).map_err(|e| {
                    FerricError::General(format!("internal-coordinate model solve: {e}"))
                })?;
                let predicted = g_int.dot(&s) + 0.5 * s.dot(&bs);
                let step_norm =
                    trust_scale * internal_step_length(coords, &s, dist_trust, ang_trust);
                let (acc, next) = ratio_trust_update(
                    e_new - energy,
                    predicted,
                    step_norm,
                    trust_scale,
                    TRUST_SCALE_MIN,
                    INTERNAL_TRUST_SCALE_MAX,
                );
                trust_scale = next;
                acc
            }
        };

        let y = &g_int_new - &g_int;
        let update = guarded_inverse_bfgs_update(&mut h_inv, &s, &y);
        if accept {
            current = trial;
            prev_energy = energy;
            energy = e_new;
            grad_cart = grad_cart_new;
            fresh_point = true;
        } else {
            rejected += 1;
            fresh_point = false;
            print_rejected_row(step_idx, e_new, energy, trust_scale);
        }
        match update {
            BfgsUpdate::Applied => {}
            BfgsUpdate::SkippedNonPositiveCurvature => report_skipped_update(step_idx, &s, &y),
            BfgsUpdate::Degenerate => {
                // Reset to the initial guess, not to the identity: the
                // identity is meaningless in internals, where the coordinates
                // have different units.
                h_inv = internal_inverse_seed(coords, &current, initial)?;
            }
        }

        step_idx += 1;
    }

    Ok(OptimizeResult {
        mol: current,
        energy,
        steps: step_idx,
        converged,
        energy_trace,
        rejected_steps: rejected,
        search: SearchSettings {
            coord_system: CoordSystem::RedundantInternal,
            ..opt_config.requested_search()
        },
    })
}

/// Back-transform an internal step to Cartesians, halving it (up to four
/// times) when the nonlinear solve does not converge. A failed
/// back-transformation means the step left the regime where B is a good
/// linearization; the fix is a shorter step, not a different coordinate
/// system. Out of retries, the best iterate of the last attempt is taken
/// rather than stalling (it is a descent direction and the next gradient
/// corrects it), with a printed warning. Returns the new Cartesian
/// coordinates only.
fn back_transform_with_retries(
    coords: &InternalCoords,
    current: &Molecule,
    dq: &Array1<f64>,
) -> Result<Vec<f64>, FerricError> {
    let mut scale = 1.0;
    for attempt in 0..5 {
        let trial = dq * scale;
        let bt = step_to_cartesian(
            coords,
            current,
            &trial,
            BACKTRANSFORM_MAX_ITER,
            BACKTRANSFORM_TOL,
        )?;
        if bt.converged {
            return Ok(bt.coords);
        }
        if attempt == 4 {
            println!(
                "Warning: internal→Cartesian back-transformation did not converge \
                 (residual {:.2e} after {} passes, step scaled to {:.3}); \
                 accepting the best iterate.",
                bt.residual, bt.iterations, scale
            );
            return Ok(bt.coords);
        }
        scale *= 0.5;
    }
    unreachable!("the retry loop returns on its last pass")
}

/// Diagonal inverse Hessian in internals: `1/k` per primitive, `k` from
/// Schlegel ([`InitialHessian::Simple`]) or Lindh floored at
/// [`MODEL_HESSIAN_FLOOR`].
fn internal_inverse_seed(
    coords: &InternalCoords,
    mol: &Molecule,
    initial: InitialHessian,
) -> Result<Array2<f64>, FerricError> {
    let nq = coords.len();
    let h_diag = match initial {
        InitialHessian::Simple => coords.initial_hessian_diagonal(mol)?,
        InitialHessian::Lindh => Array1::from_iter(coords.primitives.iter().map(|&p| {
            ferric_core::model_hessian::lindh_force_constant(mol, p).max(MODEL_HESSIAN_FLOOR)
        })),
    };
    let mut h_inv = Array2::<f64>::zeros((nq, nq));
    for i in 0..nq {
        h_inv[(i, i)] = 1.0 / h_diag[i];
    }
    Ok(h_inv)
}

/// Length of an internal step relative to the trust radii: the larger of
/// `|s_dist| / dist_trust` and `|s_ang| / ang_trust`, 1 on the boundary.
fn internal_step_length(
    coords: &InternalCoords,
    s: &Array1<f64>,
    dist_trust: f64,
    ang_trust: f64,
) -> f64 {
    let (mut d, mut a) = (0.0, 0.0);
    for (idx, prim) in coords.primitives.iter().enumerate() {
        if prim.is_angular() {
            a += s[idx] * s[idx];
        } else {
            d += s[idx] * s[idx];
        }
    }
    (d.sqrt() / dist_trust).max(a.sqrt() / ang_trust)
}

/// The BFGS secant vector `s`: the internal displacement **actually achieved**
/// between two geometries, from their measured coordinate values.
///
/// # Why this takes the two geometries and not the requested step
///
/// The internal→Cartesian back-transformation is iterative, and in a redundant
/// coordinate set the requested `Δq` is generally not even reachable (ethane:
/// 28 primitives, rank 18 — see
/// [`ferric_core::internal_coords::step_to_cartesian`]). Measured on ethane
/// with an optimizer-sized step, the requested and achieved vectors differ by
/// 2.1e-2, which is larger than the 1e-2..2e-2 step components themselves.
/// Feeding BFGS the requested vector would therefore tell it the geometry moved
/// somewhere it did not, and the resulting curvature estimate would be wrong by
/// order unity — not a rounding detail.
///
/// Taking the two geometries as arguments makes it impossible to pass the
/// requested step by mistake: this function never sees it.
fn bfgs_secant_step(
    coords: &InternalCoords,
    q_before: &Array1<f64>,
    q_after: &Array1<f64>,
) -> Array1<f64> {
    let mut s = Array1::zeros(coords.len());
    for (idx, p) in coords.primitives.iter().enumerate() {
        let d = q_after[idx] - q_before[idx];
        // A torsion crossing the +/-pi branch cut must read as the short way
        // round, not as a ~2*pi excursion that would wreck the Hessian update.
        s[idx] = if matches!(p, Primitive::Dihedral(..)) {
            ferric_core::internal_coords::wrap_to_pi(d)
        } else {
            d
        };
    }
    s
}

/// Relative curvature threshold of the BFGS update: the pair `(s, y)` is used
/// only when `y·s > CURVATURE_EPS ‖y‖‖s‖`.
///
/// The secant condition `H'y = s` forces `yᵀH'y = y·s`, so no update that
/// satisfies it can be positive definite when `y·s ≤ 0`. A relative
/// threshold, not an absolute one, because `s` and `y` carry the units of the
/// coordinates (Bohr, radians) and of the gradient. 1e-8 rejects only pairs
/// that are orthogonal to within rounding or point the wrong way; every
/// positive-curvature pair the optimize regression set produces is far above
/// it, and for those the update is unchanged.
const CURVATURE_EPS: f64 = 1e-8;

/// What [`guarded_inverse_bfgs_update`] did with one `(s, y)` pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BfgsUpdate {
    /// `y·s > CURVATURE_EPS ‖y‖‖s‖`: the inverse BFGS update was applied.
    Applied,
    /// `y·s` is negative, or positive but below the relative threshold. `H`
    /// is left unchanged, so it stays positive definite and the next step
    /// `-Hg` is still a descent direction.
    SkippedNonPositiveCurvature,
    /// `|y·s| ≤ 1e-12`, typically a zero-length step at convergence. `H` is
    /// left unchanged here and the caller resets it to its initial guess.
    Degenerate,
}

/// Inverse-Hessian BFGS update for a minimizer, with a curvature guard.
///
/// With `ρ = 1/(y·s)`, applies
/// `H ← H + ρ²(y·s + yᵀHy) ssᵀ − ρ(Hy sᵀ + s yᵀH)` when
/// `y·s > CURVATURE_EPS ‖y‖‖s‖`, and otherwise leaves `H` unchanged (the
/// "skip" safeguard of a quasi-Newton method run without a line search). The
/// update keeps `H` positive definite exactly when `y·s > 0`; when `y·s < 0`
/// it would make `yᵀH'y = y·s < 0`, and a later step could point uphill. The
/// search has no line search and accepts every step, so nothing else would
/// catch it.
///
/// Used by the Cartesian ([`optimize_coordinates`]) and redundant-internal
/// (`run_bfgs_internal`) minimizers only. The saddle search carries an
/// indefinite Hessian with its own Bofill update (`saddle.rs::bofill_update`),
/// and the IRC walk takes steepest-descent steps with no Hessian, so neither
/// passes through here.
fn guarded_inverse_bfgs_update(
    h: &mut Array2<f64>,
    s: &Array1<f64>,
    y: &Array1<f64>,
) -> BfgsUpdate {
    let ys = y.dot(s);
    if ys.abs() <= 1e-12 {
        return BfgsUpdate::Degenerate;
    }
    if ys <= CURVATURE_EPS * y.dot(y).sqrt() * s.dot(s).sqrt() {
        return BfgsUpdate::SkippedNonPositiveCurvature;
    }
    let n = s.len();
    let hy = h.dot(y);
    let yhy = y.dot(&hy);
    let rho = 1.0 / ys;
    let term1 = (ys + yhy) * rho * rho;
    for i in 0..n {
        for j in 0..n {
            h[(i, j)] += term1 * s[i] * s[j] - rho * (hy[i] * s[j] + s[i] * hy[j]);
        }
    }
    BfgsUpdate::Applied
}

/// Print a line in the step table when an update is skipped.
fn report_skipped_update(step: usize, s: &Array1<f64>, y: &Array1<f64>) {
    println!(
        "     | step {}: y.s = {:.2e} <= 0 (negative curvature along the step); \
         BFGS update skipped, Hessian kept",
        step + 1,
        y.dot(s)
    );
}

/// Clip a proposed internal step, separately per coordinate type.
///
/// Bonds are clipped against `cartesian_trust` (Bohr); angles and dihedrals
/// against `angular_trust` (radians; [`INTERNAL_ANGULAR_TRUST`] times the
/// trust scale).
/// Mixing the two into a single vector norm would be a units error: adding
/// Bohr² to rad² produces a number with no meaning, and whichever group
/// happened to be numerically larger would dominate the clip.
fn clip_internal_step(
    coords: &InternalCoords,
    p: &Array1<f64>,
    cartesian_trust: f64,
    angular_trust: f64,
) -> Array1<f64> {
    let mut dist_norm = 0.0;
    let mut ang_norm = 0.0;
    for (idx, prim) in coords.primitives.iter().enumerate() {
        if prim.is_angular() {
            ang_norm += p[idx] * p[idx];
        } else {
            dist_norm += p[idx] * p[idx];
        }
    }
    let dist_norm = dist_norm.sqrt();
    let ang_norm = ang_norm.sqrt();

    let dist_scale = if dist_norm > cartesian_trust {
        cartesian_trust / dist_norm
    } else {
        1.0
    };
    let ang_scale = if ang_norm > angular_trust {
        angular_trust / ang_norm
    } else {
        1.0
    };

    let mut out = p.clone();
    for (idx, prim) in coords.primitives.iter().enumerate() {
        out[idx] *= if prim.is_angular() {
            ang_scale
        } else {
            dist_scale
        };
    }
    out
}

/// The requested internal step for one trust rule, and whether it fell back
/// to the seed metric.
///
/// `Fixed` is the search before the ratio rule: `-H g` with each block
/// clipped onto its own radius ([`clip_internal_step`]). `Ratio` takes
/// [`internal_model_direction`] and clips it by one factor
/// ([`clip_internal_step_uniform`]), so the step is a descent direction at
/// every radius.
fn internal_trial_step(
    coords: &InternalCoords,
    g: &Array1<f64>,
    h_inv: &Array2<f64>,
    h_seed: &Array2<f64>,
    dist_trust: f64,
    ang_trust: f64,
    rule: TrustUpdate,
) -> (Array1<f64>, bool) {
    match rule {
        TrustUpdate::Fixed => (
            clip_internal_step(coords, &-h_inv.dot(g), dist_trust, ang_trust),
            false,
        ),
        TrustUpdate::Ratio => {
            let (p, fell_back) = internal_model_direction(g, h_inv, h_seed);
            (
                clip_internal_step_uniform(coords, &p, dist_trust, ang_trust),
                fell_back,
            )
        }
    }
}

/// Clip a proposed internal step by ONE factor (`trust_update = "ratio"`).
///
/// `alpha = min(1, cartesian_trust / |p_dist|, angular_trust / |p_ang|)`
/// and the step is `alpha p`: each block stays within its own radius (the
/// units are never mixed), and the step keeps the model direction, so
/// `g.dq = alpha g.p < 0` whenever `p` descends. [`clip_internal_step`]
/// scales the two blocks by different factors. That step need not descend,
/// and the ratio of the two factors does not depend on the radius, so a
/// shrinking radius cannot recover a descent step
/// (`book/src/14-derivatives/08-geometry-search.md#geometry-search`).
fn clip_internal_step_uniform(
    coords: &InternalCoords,
    p: &Array1<f64>,
    cartesian_trust: f64,
    angular_trust: f64,
) -> Array1<f64> {
    let (mut dist_norm, mut ang_norm) = (0.0, 0.0);
    for (idx, prim) in coords.primitives.iter().enumerate() {
        if prim.is_angular() {
            ang_norm += p[idx] * p[idx];
        } else {
            dist_norm += p[idx] * p[idx];
        }
    }
    let (dist_norm, ang_norm) = (f64::sqrt(dist_norm), f64::sqrt(ang_norm));
    let mut alpha = 1.0f64;
    if dist_norm > cartesian_trust {
        alpha = alpha.min(cartesian_trust / dist_norm);
    }
    if ang_norm > angular_trust {
        alpha = alpha.min(angular_trust / ang_norm);
    }
    p * alpha
}

/// The internal model direction under the ratio rule: `p = -H g`, or
/// `-H0 g` with `H0` the positive diagonal seed when `-H g` is not a descent
/// direction (`g.p >= 0`, which a positive definite `H` rules out but
/// rounding can produce). Returns the direction and whether it fell back.
fn internal_model_direction(
    g: &Array1<f64>,
    h_inv: &Array2<f64>,
    h_seed: &Array2<f64>,
) -> (Array1<f64>, bool) {
    let p = -h_inv.dot(g);
    if g.dot(&p) < 0.0 {
        (p, false)
    } else {
        (-h_seed.dot(g), true)
    }
}

/// Coordinate-vector core of the BFGS driver: minimizes `f(x)` over a flat
/// `Vec<f64>` of length `x0.len()` (any multiple-of-3 layout the caller's
/// closure agrees with itself on — this function never interprets the
/// coordinates as atoms). Returns `(x_final, energy, steps, converged)`.
///
/// The step is `-Hg` scaled onto the trust radius; `H` starts at the
/// identity and is carried by `guarded_inverse_bfgs_update`, which skips a
/// pair with non-positive curvature so `H` stays positive definite. Under
/// [`TrustUpdate::Fixed`] every step is accepted; under
/// [`TrustUpdate::Ratio`] the radius follows [`ratio_trust_update`]. Pinned
/// bit-for-bit by `test_optimize_h2_sto3g_step_energy_anchor`; the guard by
/// `negative_curvature_step_does_not_send_the_minimizer_uphill`.
///
/// # Errors
///
/// Refuses `initial_hessian = Some(Lindh)`: this entry point has no atoms to
/// build the model from. Propagates errors from `f`.
pub fn optimize_coordinates(
    x0: &[f64],
    opt_config: &OptimizeConfig,
    f: impl FnMut(&[f64]) -> Result<(f64, Vec<f64>), FerricError>,
) -> Result<(Vec<f64>, f64, usize, bool), FerricError> {
    if opt_config.initial_hessian == Some(InitialHessian::Lindh) {
        return Err(FerricError::General(
            "initial_hessian = \"lindh\" needs a molecule; this optimizer (periodic \
             cells) has none, so only \"simple\" is available"
                .to_string(),
        ));
    }
    optimize_coordinates_seeded(x0, opt_config, &|x: &[f64]| Ok(Array2::eye(x.len())), f)
}

/// [`optimize_coordinates`] from the inverse Hessian `seed(x)` (also the
/// reset on a degenerate update pair), for a caller that has atoms to build a
/// model from (`optimize_qmmm`).
pub(crate) fn optimize_coordinates_seeded(
    x0: &[f64],
    opt_config: &OptimizeConfig,
    seed: &dyn Fn(&[f64]) -> Result<Array2<f64>, FerricError>,
    f: impl FnMut(&[f64]) -> Result<(f64, Vec<f64>), FerricError>,
) -> Result<(Vec<f64>, f64, usize, bool), FerricError> {
    let out = bfgs_cartesian(x0, opt_config, seed, f)?;
    if out.rejected > 0 {
        println!("{} step(s) rejected by the trust-radius rule", out.rejected);
    }
    Ok((out.x, out.energy, out.steps, out.converged))
}

/// What [`bfgs_cartesian`] returns.
struct CartesianOutcome {
    x: Vec<f64>,
    energy: f64,
    steps: usize,
    converged: bool,
    rejected: usize,
}

/// Print one row of the step table.
fn print_step_row(step_idx: usize, energy: f64, prev_energy: f64, g_max: f64, g_rms: f64) {
    println!(
        "{:4} | {:11.8} | {:7.1e} | {:8.2e} | {:8.2e}",
        step_idx,
        energy,
        if step_idx == 0 {
            0.0
        } else {
            energy - prev_energy
        },
        g_max,
        g_rms
    );
}

/// Print the line of a step the ratio rule rejected.
fn print_rejected_row(step_idx: usize, e_trial: f64, energy: f64, radius: f64) {
    println!(
        "{:4} | {:11.8} | {:+7.1e} | rejected: energy rose; trust radius -> {:.3e}",
        step_idx + 1,
        e_trial,
        e_trial - energy,
        radius
    );
}

/// The Cartesian BFGS loop, from the inverse Hessian `seed(x0)`. `seed` is
/// also the reset on a degenerate update pair.
fn bfgs_cartesian(
    x0: &[f64],
    opt_config: &OptimizeConfig,
    seed: &dyn Fn(&[f64]) -> Result<Array2<f64>, FerricError>,
    mut f: impl FnMut(&[f64]) -> Result<(f64, Vec<f64>), FerricError>,
) -> Result<CartesianOutcome, FerricError> {
    let n_coord = x0.len();
    let mut x = Array1::from_vec(x0.to_vec());

    // Initial energy and gradient
    let (mut energy, grad0) = f(x.as_slice().expect("x is contiguous"))?;
    let mut grad = Array1::from_vec(grad0);

    // Approximate inverse Hessian
    let mut h = seed(x0)?;

    let mut prev_energy = energy;
    let mut converged = false;
    let mut step_idx = 0;
    let mut rejected = 0;
    let mut fresh_point = true;
    let mut radius = opt_config.trust_radius;
    let r_min = TRUST_SCALE_MIN * opt_config.trust_radius;
    let r_max = CARTESIAN_TRUST_SCALE_MAX * opt_config.trust_radius;

    println!("Step | Energy (Ha) | Delta E | Max Grad | RMS Grad");
    println!("-----+-------------+---------+----------+---------");

    while step_idx < opt_config.max_steps {
        if fresh_point {
            let g_max = grad.iter().map(|g| g.abs()).fold(0.0f64, f64::max);
            let g_rms = (grad.iter().map(|g| g * g).sum::<f64>() / n_coord as f64).sqrt();
            let e_diff = (energy - prev_energy).abs();
            print_step_row(step_idx, energy, prev_energy, g_max, g_rms);
            if step_idx > 0
                && e_diff < opt_config.e_conv
                && g_max < opt_config.g_max_thresh
                && g_rms < opt_config.g_rms_thresh
            {
                converged = true;
                break;
            }
        }

        // BFGS step: p = -H * g, scaled onto the trust radius.
        let p = -h.dot(&grad);
        let p_norm = p.iter().map(|v| v * v).sum::<f64>().sqrt();
        let g_dot_p = grad.dot(&p);
        let (step, alpha) = if p_norm > radius {
            (&p * (radius / p_norm), radius / p_norm)
        } else {
            (p, 1.0)
        };

        let x_trial = &x + &step;
        let (e_new, grad_new) = f(x_trial.as_slice().expect("x is contiguous"))?;
        let grad_new = Array1::from_vec(grad_new);

        let accept = match opt_config.trust_update {
            TrustUpdate::Fixed => true,
            TrustUpdate::Ratio => {
                // s = alpha p and B = H^-1 give B s = -alpha g, so the
                // quadratic model's change g.s + s.Bs/2 is alpha g.p (1 - alpha/2).
                let predicted = alpha * g_dot_p * (1.0 - 0.5 * alpha);
                let step_norm = alpha * p_norm;
                let (acc, next) =
                    ratio_trust_update(e_new - energy, predicted, step_norm, radius, r_min, r_max);
                radius = next;
                acc
            }
        };

        // BFGS update for H, with the pair of this step whether or not it is
        // kept: the curvature it measured is real either way.
        let s = step; // x_trial - x
        let y = &grad_new - &grad;
        let update = guarded_inverse_bfgs_update(&mut h, &s, &y);

        if accept {
            x = x_trial;
            prev_energy = energy;
            energy = e_new;
            grad = grad_new;
            fresh_point = true;
        } else {
            rejected += 1;
            fresh_point = false;
            print_rejected_row(step_idx, e_new, energy, radius);
        }
        match update {
            BfgsUpdate::Applied => {}
            BfgsUpdate::SkippedNonPositiveCurvature => report_skipped_update(step_idx, &s, &y),
            BfgsUpdate::Degenerate => h = seed(x.as_slice().expect("x is contiguous"))?,
        }

        step_idx += 1;
    }

    Ok(CartesianOutcome {
        x: x.to_vec(),
        energy,
        steps: step_idx,
        converged,
        rejected,
    })
}

/// `solve_rhf` returns an unconverged result as `Ok` (the UHF and ROHF solvers
/// raise instead), so a geometry driver must refuse it itself: an energy and a
/// gradient from a density that never settled are not points on the surface,
/// and the search would walk on noise while reporting ordinary-looking steps.
pub(crate) fn require_converged(res: &ScfResult) -> Result<(), FerricError> {
    if res.converged {
        Ok(())
    } else {
        Err(FerricError::ScfConvergence {
            iterations: res.iterations,
            last_energy: res.energy,
        })
    }
}

/// Optimize a geometry on an arbitrary surface supplied as a closure.
///
/// `energy_and_gradient(mol)` returns `(E, dE/dR)` in Hartree and Hartree/Bohr
/// (`[natoms, 3]`). This is the same BFGS driver (trust-radius step scaling,
/// no line search), convergence tests and coordinate systems the SCF optimizers use; only the surface
/// differs. It is how a method that lives outside `ferric-scf` (GFN-xTB in
/// `ferric-xtb`) reaches the existing optimizer without a second one.
/// An error from the closure aborts the optimization and is returned.
pub fn optimize_geometry_with_energy_gradient(
    mol: &Molecule,
    opt_config: &OptimizeConfig,
    energy_and_gradient: impl FnMut(&Molecule) -> Result<(f64, Array2<f64>), FerricError>,
) -> Result<OptimizeResult, FerricError> {
    run_bfgs(mol, opt_config, energy_and_gradient)
}

/// Closed-shell SCF energy, its nuclear gradient, and the converged SCF
/// result that produced both (handed to density-dependent corrections).
pub(crate) fn compute_energy_gradient_and_result(
    ctx: &ParallelContext,
    mol: &Molecule,
    basis_name: &str,
    op: Operator,
    rhf_config: &RhfConfig,
    warm: &mut ScfWarmStart,
) -> Result<(f64, Array2<f64>, ScfResult), FerricError> {
    let t = Stage::start("opt-step:basis+prep");
    let bs = ferric_core::basis::bundled(basis_name)?;
    let prep = PreparedBasis::new(mol, &bs)?;
    t.end();
    // Honour `[scf] screening` here too: geometry optimization rebuilds the
    // bound at every step, and a step that silently dropped back to plain
    // Schwarz would make the optimizer's energies inconsistent with a
    // single-point run at the same geometry. `ScreeningKind::Schwarz` (the
    // default) delegates verbatim to `compute`, so this is byte-identical
    // unless CSB was explicitly selected.
    //
    // SCOPE: this governs the SCF only. `gradient.rs` carries its OWN inline
    // `bounds.q[(s1,s2)] * bounds.q[(s3,s4)]` screen that does not consult
    // `csb_m`, so the GRADIENT stays on plain Schwarz regardless. That is
    // sound (plain Schwarz is still a rigorous bound, just looser) and is
    // recorded as a known gap rather than silently assumed to be covered.
    let t = Stage::start("opt-step:schwarz");
    let bounds = SchwarzBounds::compute_for_screening(op, &prep, rhf_config.screening)?;
    t.end();
    // Fail before the SCF if the exchange will come from a COSX setup whose
    // gradient is refused (pruned grid, or overlap fit with a KS functional) — not after it.
    crate::gradient::preflight_cosx_restricted(rhf_config)?;
    // A fresh hand-off per evaluation: the SCF keeps its raw RI-JK tensor for
    // the gradient below, which takes and drops it (`crate::raw_handoff`).
    let rhf_config =
        &crate::raw_handoff::with_fresh_handoff(&crate::gradient::gradient_task_config(rhf_config));
    let t = Stage::start("opt-step:scf");
    let res = solve_rhf_warm(ctx, mol, &prep, op, &bounds, rhf_config, warm)?;
    t.end();
    require_converged(&res)?;
    // Differentiates the exchange the SCF actually built: exactly
    // `ks_gradient_closed` / `rhf_gradient` unless `k_builder = "cosx"` is in
    // effect, in which case the exchange term is the COSX derivative.
    let t = Stage::start("opt-step:gradient");
    let grad =
        crate::gradient::restricted_scf_gradient(mol, &prep, &bs, op, &bounds, rhf_config, &res)?;
    t.end();
    Ok((res.energy, grad, res))
}

pub(crate) fn compute_energy_and_gradient_uhf(
    ctx: &ParallelContext,
    mol: &Molecule,
    basis_name: &str,
    op: Operator,
    uhf_config: &RhfConfig,
    warm: &mut ScfWarmStart,
) -> Result<(f64, Array2<f64>, ScfResult), FerricError> {
    // `uhf_gradient` is HF-only (no XC term), so an `xc` run must route to
    // `ks_gradient_uks` instead. That function IS implemented (LDA/GGA/hybrid/
    // RSH/meta-GGA + VV10) and is FD- and PySCF-validated by
    // tests/uks_gradient.rs, tests/uks_gradient_rsh.rs and
    // tests/dft_gradient_mgga.rs.
    let bs = ferric_core::basis::bundled(basis_name)?;
    let prep = PreparedBasis::new(mol, &bs)?;
    // Honour `[scf] screening` here too: geometry optimization rebuilds the
    // bound at every step, and a step that silently dropped back to plain
    // Schwarz would make the optimizer's energies inconsistent with a
    // single-point run at the same geometry. `ScreeningKind::Schwarz` (the
    // default) delegates verbatim to `compute`, so this is byte-identical
    // unless CSB was explicitly selected.
    //
    // SCOPE: this governs the SCF only. `gradient.rs` carries its OWN inline
    // `bounds.q[(s1,s2)] * bounds.q[(s3,s4)]` screen that does not consult
    // `csb_m`, so the GRADIENT stays on plain Schwarz regardless. That is
    // sound (plain Schwarz is still a rigorous bound, just looser) and is
    // recorded as a known gap rather than silently assumed to be covered.
    let bounds = SchwarzBounds::compute_for_screening(op, &prep, uhf_config.screening)?;
    crate::gradient::preflight_cosx_unrestricted(uhf_config)?;
    // A fresh hand-off per evaluation: the SCF keeps its raw RI-JK tensor for
    // the gradient below, which takes and drops it (`crate::raw_handoff`).
    let uhf_config =
        &crate::raw_handoff::with_fresh_handoff(&crate::gradient::gradient_task_config(uhf_config));
    let res = solve_uhf_warm(ctx, mol, &prep, &bounds, uhf_config, warm)?;
    // Exactly `ks_gradient_uks` / `uhf_gradient` unless the SCF's exchange
    // came from COSX (then the COSX derivative; a fitted setup is refused).
    let grad =
        crate::gradient::unrestricted_scf_gradient(mol, &prep, &bs, op, &bounds, uhf_config, &res)?;
    Ok((res.energy, grad, res))
}

pub(crate) fn compute_energy_and_gradient_rohf(
    ctx: &ParallelContext,
    mol: &Molecule,
    basis_name: &str,
    op: Operator,
    rohf_config: &RhfConfig,
    warm: &mut ScfWarmStart,
) -> Result<(f64, Array2<f64>, ScfResult), FerricError> {
    // `restricted_open_scf_gradient` routes HF to `rohf_gradient` and an `xc`
    // run to `ks_gradient_roks` (FD-validated by tests/roks_gradient.rs,
    // LDA/PBE/B3LYP/wB97X-V), or to their COSX forms when the SCF built its
    // exchange with COSX. Same shape as the UHF path above.
    let bs = ferric_core::basis::bundled(basis_name)?;
    let prep = PreparedBasis::new(mol, &bs)?;
    // Honour `[scf] screening` here too: geometry optimization rebuilds the
    // bound at every step, and a step that silently dropped back to plain
    // Schwarz would make the optimizer's energies inconsistent with a
    // single-point run at the same geometry. `ScreeningKind::Schwarz` (the
    // default) delegates verbatim to `compute`, so this is byte-identical
    // unless CSB was explicitly selected.
    //
    // SCOPE: this governs the SCF only. `gradient.rs` carries its OWN inline
    // `bounds.q[(s1,s2)] * bounds.q[(s3,s4)]` screen that does not consult
    // `csb_m`, so the GRADIENT stays on plain Schwarz regardless. That is
    // sound (plain Schwarz is still a rigorous bound, just looser) and is
    // recorded as a known gap rather than silently assumed to be covered.
    let bounds = SchwarzBounds::compute_for_screening(op, &prep, rohf_config.screening)?;
    // A COSX setup whose ROHF/ROKS gradient is not implemented (the overlap
    // fit) is refused BEFORE the SCF; the gradient differentiates the
    // exchange the SCF actually built.
    crate::gradient::preflight_cosx_restricted_open(rohf_config)?;
    // A fresh hand-off per evaluation: the SCF keeps its raw RI-JK tensor for
    // the gradient below, which takes and drops it (`crate::raw_handoff`).
    let rohf_config = &crate::raw_handoff::with_fresh_handoff(
        &crate::gradient::gradient_task_config(rohf_config),
    );
    let res = solve_rohf_warm(ctx, mol, &prep, op, &bounds, rohf_config, warm)?;
    let grad = crate::gradient::restricted_open_scf_gradient(
        mol,
        &prep,
        &bs,
        op,
        &bounds,
        rohf_config,
        &res,
    )?;
    Ok((res.energy, grad, res))
}

fn flatten_gradient(grad: &Array2<f64>) -> Array1<f64> {
    let mut flat = Array1::zeros(grad.len());
    let mut idx = 0;
    for i in 0..grad.nrows() {
        for j in 0..3 {
            flat[idx] = grad[(i, j)];
            idx += 1;
        }
    }
    flat
}

/// Flatten a [`Molecule`]'s Cartesian coordinates into `(x0,y0,z0,x1,y1,z1,...)`.
fn flatten_molecule_coords(mol: &Molecule) -> Vec<f64> {
    let mut out = Vec::with_capacity(mol.atoms.len() * 3);
    for atom in &mol.atoms {
        out.push(atom.x);
        out.push(atom.y);
        out.push(atom.zpos);
    }
    out
}

/// Overwrite a [`Molecule`]'s Cartesian coordinates from a flat
/// `(x0,y0,z0,x1,y1,z1,...)` vector (the inverse of [`flatten_molecule_coords`]).
fn set_molecule_coords(mol: &mut Molecule, x: &[f64]) {
    let mut idx = 0;
    for atom in mol.atoms.iter_mut() {
        atom.x = x[idx];
        atom.y = x[idx + 1];
        atom.zpos = x[idx + 2];
        idx += 3;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::mol::Molecule;

    /// `solve_rhf` returns an unconverged SCF as `Ok`; the closed-shell
    /// optimizer must refuse it, not step on its density. Trivial limit: the
    /// same input with the default `max_iter` converges (test above).
    #[test]
    fn test_optimize_raises_on_unconverged_scf() {
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 1.0\n", 0, 1).unwrap();
        let rhf_config = RhfConfig {
            max_iter: 1,
            energy_conv: 1e-12,
            density_conv: 1e-12,
            ..Default::default()
        };
        let ctx = ParallelContext::default();
        let err = optimize_geometry(
            &ctx,
            &mol,
            "sto-3g",
            Operator::coulomb(),
            &rhf_config,
            &OptimizeConfig::default(),
        )
        .expect_err("a one-iteration SCF cannot converge");
        assert!(
            matches!(err, FerricError::ScfConvergence { iterations: 1, .. }),
            "{err:?}"
        );
    }

    #[test]
    fn test_optimize_h2_sto3g() {
        // Start from a stretched bond: 1.0 Angstrom = 1.89 Bohr
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 1.0\n", 0, 1).unwrap();
        let op = Operator::coulomb();
        let rhf_config = RhfConfig {
            energy_conv: 1e-10,
            ..Default::default()
        };
        let opt_config = OptimizeConfig {
            trust_radius: 0.1,
            ..Default::default()
        };

        let ctx = ParallelContext::default();
        let result = optimize_geometry(&ctx, &mol, "sto-3g", op, &rhf_config, &opt_config).unwrap();

        assert!(result.converged);
        let dist = (result.mol.atoms[0].zpos - result.mol.atoms[1].zpos).abs();
        eprintln!("H2/STO-3G optimized distance: {:.6} Bohr", dist);
        // STO-3G H2 bond length is ~1.346 Bohr
        assert!(
            (dist - 1.346).abs() < 1e-2,
            "dist = {dist}, expected ~1.346"
        );
    }

    /// F2-1 EXACTNESS ANCHOR. Pins the exact per-step energy trajectory
    /// `run_bfgs` produces for H2/STO-3G BEFORE it is rewritten on top of
    /// `optimize_coordinates` (the coordinate-vector core). The bit patterns
    /// below were captured from the pre-refactor implementation; after the
    /// refactor this test must still pass byte-for-byte (`to_bits()`
    /// equality, not a tolerance) — any drift means the refactor changed the
    /// algorithm, not just its shape.
    #[test]
    fn test_optimize_h2_sto3g_step_energy_anchor() {
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 1.0\n", 0, 1).unwrap();
        let op = Operator::coulomb();
        let rhf_config = RhfConfig {
            energy_conv: 1e-10,
            ..Default::default()
        };
        // The search the anchor was captured with (identity start, fixed
        // trust radius), pinned so the anchor tests the loop, not the defaults.
        let opt_config = OptimizeConfig {
            trust_radius: 0.1,
            initial_hessian: Some(InitialHessian::Simple),
            trust_update: TrustUpdate::Fixed,
            ..Default::default()
        };
        let ctx = ParallelContext::default();

        let result = optimize_geometry(&ctx, &mol, "sto-3g", op, &rhf_config, &opt_config).unwrap();
        assert!(result.converged);
        eprintln!(
            "[F2-1 anchor] H2/STO-3G final energy bits = {:#018x} ({:.15}), steps = {}",
            result.energy.to_bits(),
            result.energy,
            result.steps,
        );
        // Captured 2026-08-27 from the pre-refactor `run_bfgs` (this exact
        // test, run once before touching optimize.rs):
        //   [F2-1 anchor] H2/STO-3G final energy bits = 0xbff1e14dd9dd63b7
        //   (-1.117505885156999), steps = 7
        //
        // Bit identity holds on the machine the anchor was captured on, but
        // NOT across BLAS kernel sets: CI run 33216140352 (AMD EPYC 9V74,
        // OpenBLAS 0.3.20 with OPENBLAS_CORETYPE=Haswell) reproduced this
        // energy to 5 ulp (-1.117505885156998 vs ...999), which is GEMM
        // reduction-order noise, not an optimizer change. The refactor
        // invariant this test guards -- same minimum, same step count -- is
        // therefore asserted at 1e-12 Ha (six orders below the optimizer's
        // e_conv and ~1e4x above ulp noise), and exactly on `steps`.
        const ANCHOR_ENERGY: f64 = -1.117505885156999;
        const ANCHOR_STEPS: usize = 7;
        let de = (result.energy - ANCHOR_ENERGY).abs();
        assert!(
            de < 1e-12,
            "post-refactor energy {:.15} != pre-refactor anchor {:.15} (|Δ| {de:.3e})",
            result.energy,
            ANCHOR_ENERGY
        );
        assert_eq!(result.steps, ANCHOR_STEPS);
    }

    #[test]
    fn test_optimize_h2o_sto3g() {
        // Second molecule (widens past H2-only): H2O/STO-3G, started from a
        // distorted geometry (O-H 0.9/0.92 A-ish, non-equilibrium angle).
        // Reference: PySCF RHF/STO-3G geometric-optimizer result --
        //   O-H bond lengths (Bohr): 1.869732, 1.869731
        //   H-O-H angle (deg): 100.0258
        //   E_final = -74.9659011921 Ha
        let mol = Molecule::parse_xyz("3\nH2O\nO 0 0 0\nH 0 0.9 0\nH 0 -0.3 0.85\n", 0, 1).unwrap();
        let op = Operator::coulomb();
        let rhf_config = RhfConfig {
            energy_conv: 1e-10,
            ..Default::default()
        };
        let opt_config = OptimizeConfig {
            trust_radius: 0.1,
            ..Default::default()
        };

        let ctx = ParallelContext::default();
        let result = optimize_geometry(&ctx, &mol, "sto-3g", op, &rhf_config, &opt_config).unwrap();

        assert!(result.converged);
        let o = &result.mol.atoms[0];
        let h1 = &result.mol.atoms[1];
        let h2 = &result.mol.atoms[2];
        let r1 = ((o.x - h1.x).powi(2) + (o.y - h1.y).powi(2) + (o.zpos - h1.zpos).powi(2)).sqrt();
        let r2 = ((o.x - h2.x).powi(2) + (o.y - h2.y).powi(2) + (o.zpos - h2.zpos).powi(2)).sqrt();
        eprintln!("H2O/STO-3G optimized O-H distances: {r1:.6}, {r2:.6} Bohr (ref 1.869732)");
        assert!(
            (r1 - 1.869732).abs() < 1e-2,
            "r1 = {r1}, expected ~1.869732"
        );
        assert!(
            (r2 - 1.869732).abs() < 1e-2,
            "r2 = {r2}, expected ~1.869732"
        );
    }

    #[test]
    fn test_optimize_h2plus_uhf_sto3g() {
        // H2+ (one electron, doublet), started stretched at 1.5 Angstrom
        // (2.835 Bohr) -- well beyond the equilibrium bond -- and optimized
        // with UHF/STO-3G analytical gradients. UHF on a single-electron
        // system reduces exactly to RHF on that electron, so this is a
        // useful sanity check that the UHF optimize path (a) actually
        // iterates (not just prints one gradient) and (b) lands at a
        // reasonable minimum.
        let mol = Molecule::parse_xyz("2\nH2+\nH 0 0 0\nH 0 0 1.5\n", 1, 2).unwrap();
        let op = Operator::coulomb();
        let uhf_config = RhfConfig {
            energy_conv: 1e-10,
            ..Default::default()
        };
        let opt_config = OptimizeConfig {
            trust_radius: 0.1,
            ..Default::default()
        };

        let ctx = ParallelContext::default();
        let e0 = compute_energy_and_gradient_uhf(
            &ctx,
            &mol,
            "sto-3g",
            op,
            &uhf_config,
            &mut ScfWarmStart::new(false),
        )
        .unwrap()
        .0;

        let result =
            optimize_geometry_uhf(&ctx, &mol, "sto-3g", op, &uhf_config, &opt_config).unwrap();

        assert!(
            result.converged,
            "UHF H2+ optimization did not converge in {} steps",
            result.steps
        );
        assert!(
            result.steps > 0,
            "optimizer should take at least one step from a stretched start"
        );
        assert!(
            result.energy < e0,
            "optimized energy {} should be lower than initial energy {}",
            result.energy,
            e0
        );

        let dist = (result.mol.atoms[0].zpos - result.mol.atoms[1].zpos).abs();
        eprintln!(
            "H2+/UHF/STO-3G optimized distance: {:.6} Bohr, energy: {:.10} Ha",
            dist, result.energy
        );
        // Independent PySCF UHF/STO-3G geomeTRIC-optimizer reference (2026-07-21):
        //   dist = 2.004215 Bohr, E = -0.5826966474 Ha
        // ferric matches to ~1e-9 Ha / <1e-6 Bohr -- tightened from the old
        // loose +-0.3 Bohr sanity band now that a real reference exists.
        const PYSCF_DIST: f64 = 2.004215;
        const PYSCF_E: f64 = -0.5826966474;
        assert!(
            (dist - PYSCF_DIST).abs() < 1e-3,
            "dist = {dist}, expected {PYSCF_DIST} (PySCF)"
        );
        assert!(
            (result.energy - PYSCF_E).abs() < 1e-5,
            "energy = {}, expected {PYSCF_E} (PySCF)",
            result.energy
        );

        // Final gradient norm must be below the configured convergence
        // thresholds -- re-derive it directly rather than trusting the
        // driver's internal bookkeeping.
        let (_, grad_arr, _) = compute_energy_and_gradient_uhf(
            &ctx,
            &result.mol,
            "sto-3g",
            op,
            &uhf_config,
            &mut ScfWarmStart::new(false),
        )
        .unwrap();
        let grad = flatten_gradient(&grad_arr);
        let g_max = grad.iter().map(|g| g.abs()).fold(0.0f64, f64::max);
        assert!(
            g_max < opt_config.g_max_thresh,
            "final |g|_max = {g_max:.3e} not converged"
        );
    }

    #[test]
    fn test_optimize_oh_radical_rohf_sto3g() {
        // OH radical (doublet), started stretched at 1.3 Angstrom (vs the
        // experimental 0.9697 Angstrom in testdata/molecules/oh.xyz) and
        // optimized with ROHF/STO-3G analytical gradients.
        let mol = Molecule::parse_xyz("2\nOH\nO 0 0 0\nH 0 0 1.3\n", 0, 2).unwrap();
        let op = Operator::coulomb();
        let rohf_config = RhfConfig {
            energy_conv: 1e-10,
            ..Default::default()
        };
        let opt_config = OptimizeConfig {
            trust_radius: 0.1,
            ..Default::default()
        };

        let ctx = ParallelContext::default();
        let e0 = compute_energy_and_gradient_rohf(
            &ctx,
            &mol,
            "sto-3g",
            op,
            &rohf_config,
            &mut ScfWarmStart::new(false),
        )
        .unwrap()
        .0;

        let result =
            optimize_geometry_rohf(&ctx, &mol, "sto-3g", op, &rohf_config, &opt_config).unwrap();

        assert!(
            result.converged,
            "ROHF OH optimization did not converge in {} steps",
            result.steps
        );
        assert!(
            result.steps > 0,
            "optimizer should take at least one step from a stretched start"
        );
        assert!(
            result.energy < e0,
            "optimized energy {} should be lower than initial energy {}",
            result.energy,
            e0
        );

        let dist_bohr = (result.mol.atoms[0].zpos - result.mol.atoms[1].zpos).abs();
        let dist_ang = dist_bohr * ferric_core::units::BOHR_TO_ANGSTROM;
        eprintln!(
            "OH/ROHF/STO-3G optimized distance: {:.6} Bohr ({:.4} Ang), energy: {:.10} Ha",
            dist_bohr, dist_ang, result.energy
        );
        // STO-3G ROHF is a minimal basis, so the equilibrium bond will not
        // match the experimental 0.9697 Ang exactly -- minimal-basis HF
        // typically overbinds by several tenths of an Angstrom for OH.
        // Loose cross-check band: this catches optimizing to the wrong
        // stationary point (e.g. dissociation or a basis artifact far off).
        assert!(
            (dist_ang - 0.9697).abs() < 0.2,
            "dist = {dist_ang} Ang, expected within 0.2 Ang of experimental 0.9697"
        );
        // Independent PySCF ROHF/STO-3G geomeTRIC-optimizer reference
        // (2026-07-21): dist = 1.913998 Bohr, E = -74.3636983636 Ha. This
        // confirms the ~1 Ang overbinding above is genuine minimal-basis
        // ROHF physics (PySCF lands at the SAME stationary point ferric
        // does), not a ferric bug -- a real, tight computational
        // cross-check alongside the deliberately loose experimental band.
        const PYSCF_DIST_BOHR: f64 = 1.913998;
        const PYSCF_E: f64 = -74.3636983636;
        assert!(
            (dist_bohr - PYSCF_DIST_BOHR).abs() < 1e-3,
            "dist = {dist_bohr} Bohr, expected {PYSCF_DIST_BOHR} (PySCF)"
        );
        assert!(
            (result.energy - PYSCF_E).abs() < 1e-5,
            "energy = {}, expected {PYSCF_E} (PySCF)",
            result.energy
        );

        let (_, grad_arr, _) = compute_energy_and_gradient_rohf(
            &ctx,
            &result.mol,
            "sto-3g",
            op,
            &rohf_config,
            &mut ScfWarmStart::new(false),
        )
        .unwrap();
        let grad = flatten_gradient(&grad_arr);
        let g_max = grad.iter().map(|g| g.abs()).fold(0.0f64, f64::max);
        assert!(
            g_max < opt_config.g_max_thresh,
            "final |g|_max = {g_max:.3e} not converged"
        );
    }

    /// `f = (x0^2 - 1)^2 + x1^2 + x2^2`: a double well in `x0` with a local
    /// maximum at `x0 = 0` and negative curvature for `|x0| < 1/sqrt(3)`.
    fn double_well(x: &[f64]) -> Result<(f64, Vec<f64>), FerricError> {
        let a = x[0] * x[0] - 1.0;
        Ok((
            a * a + x[1] * x[1] + x[2] * x[2],
            vec![4.0 * x[0] * a, 2.0 * x[1], 2.0 * x[2]],
        ))
    }

    /// Started at `x0 = 0.1`, inside the negative-curvature region, the first
    /// step (0.1 -> 0.2, the trust radius) gives `y.s = -3.7e-2`. An
    /// unguarded update turns that into `H' = s/y < 0` along `x0`, the next
    /// step points uphill, and the search climbs back to the MAXIMUM at
    /// `x0 = 0`, where the gradient vanishes and it reports converged at
    /// `E = 1`. The guard skips that update, keeps `H = I`, and the search
    /// descends monotonically to the minimum at `x0 = 1`.
    #[test]
    fn negative_curvature_step_does_not_send_the_minimizer_uphill() {
        let mut trace = Vec::new();
        // Fixed trust radius: every evaluation is kept, so an uphill entry in
        // the trace is a step the search took.
        let cfg = OptimizeConfig {
            trust_update: TrustUpdate::Fixed,
            ..Default::default()
        };
        let (x, e, _steps, converged) = optimize_coordinates(&[0.1, 0.0, 0.0], &cfg, |x| {
            let r = double_well(x)?;
            trace.push(r.0);
            Ok(r)
        })
        .expect("model optimization runs");
        for (k, w) in trace.windows(2).enumerate() {
            assert!(
                w[1] <= w[0],
                "evaluation {} went uphill: {:.6} -> {:.6} (trace {trace:?})",
                k + 1,
                w[0],
                w[1]
            );
        }
        assert!(converged, "must converge (trace {trace:?})");
        assert!(
            (x[0] - 1.0).abs() < 1e-3 && e < 1e-6,
            "must reach the minimum x0 = 1, E = 0; got x0 = {}, E = {e}",
            x[0]
        );
    }

    /// The guard itself: a positive-definite `H` with a `y.s < 0` pair is left
    /// unchanged (the update it would replace makes `y^T H' y = y.s < 0`), and
    /// a `y.s > 0` pair gets exactly the textbook inverse update, bit for bit.
    #[test]
    fn guarded_update_skips_negative_curvature_and_is_exact_otherwise() {
        let h0 = Array2::from_shape_vec((3, 3), vec![1.0, 0.2, 0.0, 0.2, 0.8, 0.1, 0.0, 0.1, 0.5])
            .expect("3x3");
        let s = Array1::from_vec(vec![0.1, -0.05, 0.02]);

        let y_neg = Array1::from_vec(vec![-0.3, 0.02, 0.01]);
        assert!(y_neg.dot(&s) < 0.0);
        let mut h = h0.clone();
        assert_eq!(
            guarded_inverse_bfgs_update(&mut h, &s, &y_neg),
            BfgsUpdate::SkippedNonPositiveCurvature
        );
        assert_eq!(h, h0, "a skipped update must leave H untouched");

        let y_pos = Array1::from_vec(vec![0.12, -0.03, 0.015]);
        let ys = y_pos.dot(&s);
        assert!(ys > 0.0);
        let mut h = h0.clone();
        assert_eq!(
            guarded_inverse_bfgs_update(&mut h, &s, &y_pos),
            BfgsUpdate::Applied
        );
        let hy = h0.dot(&y_pos);
        let rho = 1.0 / ys;
        let term1 = (ys + y_pos.dot(&hy)) * rho * rho;
        let mut expect = h0.clone();
        for i in 0..3 {
            for j in 0..3 {
                expect[(i, j)] += term1 * s[i] * s[j] - rho * (hy[i] * s[j] + s[i] * hy[j]);
            }
        }
        assert_eq!(
            h, expect,
            "above the threshold the update is the plain BFGS one"
        );
        // Secant condition H' y = s.
        let hy_new = h.dot(&y_pos);
        for k in 0..3 {
            assert!((hy_new[k] - s[k]).abs() < 1e-14);
        }

        let mut h = h0.clone();
        assert_eq!(
            guarded_inverse_bfgs_update(&mut h, &Array1::zeros(3), &y_pos),
            BfgsUpdate::Degenerate
        );
    }
    // -- internal-coordinate step machinery -------------------------------
    //
    // These two tests exist because the integration suite
    // (tests/internal_coord_optimize.rs) did NOT kill the corresponding
    // mutations: it asserts that the optimizer converges and roughly how fast,
    // which a degraded-but-still-descending step satisfies. Convergence is too
    // coarse an observable to pin step CONSTRUCTION, so the construction is
    // pinned directly here.

    fn ethane_for_steps() -> Molecule {
        Molecule::parse_xyz(
            "8\nethane\nC  0.000  0.000  0.766\nC  0.000  0.000 -0.766\n\
             H  1.019  0.000  1.163\nH -0.509  0.883  1.163\nH -0.509 -0.883  1.163\n\
             H  1.019  0.000 -1.163\nH -0.509  0.883 -1.163\nH -0.509 -0.883 -1.163\n",
            0,
            1,
        )
        .unwrap()
    }

    /// The step clip must treat Bohr and radians as SEPARATE budgets.
    ///
    /// Mixing them is a units error: a single vector norm over a mixed-unit
    /// vector adds Bohr^2 to rad^2, and whichever group is numerically larger
    /// then dictates the clip for both. This test constructs a step that is
    /// large in the angular block and small in the distance block, and checks
    /// that the angular block is cut to its own trust radius while the
    /// distance block — already within budget — is left completely alone.
    ///
    /// Mutation M15 (apply the Bohr scale to angular coordinates too) survived
    /// the integration suite and is killed here.
    #[test]
    fn internal_step_clip_keeps_distance_and_angular_budgets_separate() {
        let mol = ethane_for_steps();
        let coords = InternalCoords::generate(&mol).unwrap();
        let n_ang = coords.primitives.iter().filter(|p| p.is_angular()).count();
        let n_dist = coords.len() - n_ang;
        assert!(
            n_ang > 0 && n_dist > 0,
            "this test needs both coordinate kinds: {n_dist} distance, {n_ang} angular"
        );

        // Distance block: deliberately tiny, total norm far inside the trust
        // radius. Angular block: deliberately huge.
        let cartesian_trust = 0.1;
        let small = 1e-4;
        let big = 1.0;
        let mut p = Array1::zeros(coords.len());
        for (idx, prim) in coords.primitives.iter().enumerate() {
            p[idx] = if prim.is_angular() { big } else { small };
        }
        let dist_norm_in = (n_dist as f64).sqrt() * small;
        assert!(
            dist_norm_in < cartesian_trust,
            "precondition: the distance block must already be inside the trust radius"
        );

        let out = clip_internal_step(&coords, &p, cartesian_trust, INTERNAL_ANGULAR_TRUST);

        let mut dist_norm_out = 0.0;
        let mut ang_norm_out = 0.0;
        for (idx, prim) in coords.primitives.iter().enumerate() {
            if prim.is_angular() {
                ang_norm_out += out[idx] * out[idx];
            } else {
                dist_norm_out += out[idx] * out[idx];
            }
        }
        let dist_norm_out = dist_norm_out.sqrt();
        let ang_norm_out = ang_norm_out.sqrt();
        eprintln!(
            "clip: distance {dist_norm_in:.4e} -> {dist_norm_out:.4e} (trust {cartesian_trust}), \
             angular {:.4e} -> {ang_norm_out:.4e} (trust {INTERNAL_ANGULAR_TRUST})",
            (n_ang as f64).sqrt() * big
        );

        // The angular block IS clipped, to its own budget.
        assert!(
            (ang_norm_out - INTERNAL_ANGULAR_TRUST).abs() < 1e-12,
            "angular block should be clipped to {INTERNAL_ANGULAR_TRUST}, got {ang_norm_out:.6e}"
        );
        // The distance block is untouched — bit-identical, because no scale
        // was applied at all. This is what fails if the two budgets are mixed:
        // the huge angular block would drag the distance block down with it.
        for (idx, prim) in coords.primitives.iter().enumerate() {
            if !prim.is_angular() {
                assert_eq!(
                    out[idx].to_bits(),
                    p[idx].to_bits(),
                    "distance component {idx} was scaled ({} -> {}) although its own \
                     block was within budget — the two budgets are being mixed",
                    p[idx],
                    out[idx]
                );
            }
        }
        assert!((dist_norm_out - dist_norm_in).abs() < 1e-18);
    }

    /// The BFGS secant vector `s` must be the displacement ACTUALLY achieved,
    /// not the one requested.
    ///
    /// The back-transformation is iterative and, in a redundant coordinate set,
    /// the requested `Δq` is generally not even reachable (see
    /// `ferric_core::internal_coords::step_to_cartesian`). Feeding the
    /// requested step into the Hessian update would tell BFGS the geometry
    /// moved somewhere it did not, corrupting the curvature estimate.
    ///
    /// Mutation M16 (use `dq_taken` instead of `q_after - q_before`) survived
    /// the integration suite and is killed here: this test measures the gap
    /// between requested and achieved directly and asserts it is real, so a
    /// driver that conflated them would be using a demonstrably wrong vector.
    #[test]
    fn requested_and_achieved_internal_steps_differ_in_a_redundant_set() {
        let mol = ethane_for_steps();
        let coords = InternalCoords::generate(&mol).unwrap();
        let b = coords.b_matrix(&mol).unwrap();
        let ginv = generalized_inverse(&b).unwrap();
        assert!(
            coords.len() > ginv.rank,
            "precondition: the set must be redundant ({} primitives, rank {})",
            coords.len(),
            ginv.rank
        );

        // A representative optimizer-sized step.
        let mut dq = Array1::zeros(coords.len());
        for (idx, prim) in coords.primitives.iter().enumerate() {
            dq[idx] = if prim.is_angular() { 0.02 } else { 0.01 };
        }

        let q_before = coords.values(&mol).unwrap();
        let bt = step_to_cartesian(
            &coords,
            &mol,
            &dq,
            BACKTRANSFORM_MAX_ITER,
            BACKTRANSFORM_TOL,
        )
        .unwrap();
        assert!(
            bt.converged,
            "the back-transformation should converge for this step"
        );

        let mut moved = mol.clone();
        set_molecule_coords(&mut moved, &bt.coords);
        let q_after = coords.values(&moved).unwrap();

        // THE ASSERTION: the secant vector the driver builds is the achieved
        // displacement. `bfgs_secant_step` is the function the driver calls,
        // and it cannot see `dq` at all.
        let s = bfgs_secant_step(&coords, &q_before, &q_after);

        let mut max_gap = 0.0f64;
        let mut max_err = 0.0f64;
        for idx in 0..coords.len() {
            let achieved = s[idx];
            max_gap = max_gap.max((achieved - dq[idx]).abs());
            // Independently recomputed achieved displacement (no wrapping is
            // needed at this step size, so a plain difference is the honest
            // cross-check of what `bfgs_secant_step` returned).
            max_err = max_err.max((achieved - (q_after[idx] - q_before[idx])).abs());
        }
        eprintln!(
            "secant step: max |s - requested| = {max_gap:.3e}, \
             max |s - (q_after - q_before)| = {max_err:.3e} \
             ({} primitives, rank {})",
            coords.len(),
            ginv.rank
        );

        // `s` IS the achieved displacement, exactly.
        assert!(
            max_err < 1e-14,
            "bfgs_secant_step returned something other than q_after - q_before \
             (max deviation {max_err:.3e})"
        );
        // And the achieved displacement is NOT the requested one — so a driver
        // that used the requested step would be using a demonstrably different
        // vector. Without this second half the test would be vacuous.
        assert!(
            max_gap > 1e-3,
            "requested and achieved steps agree to {max_gap:.3e}; if they were \
             really interchangeable this test would be vacuous — check that the \
             coordinate set is still redundant"
        );
    }

    /// `bfgs_secant_step` must take a torsion across the +/-pi branch cut the
    /// SHORT way. Without the wrap, a 2-degree move that happens to straddle
    /// the cut reads as a 358-degree one and destroys the Hessian update.
    #[test]
    fn bfgs_secant_step_wraps_torsions_across_the_branch_cut() {
        // H2O2 has exactly one dihedral, which makes the assertion unambiguous.
        let mol = Molecule::parse_xyz(
            "4\nH2O2\nO  0.000  0.734 -0.055\nO  0.000 -0.734 -0.055\n\
             H  0.839  0.885  0.435\nH -0.839 -0.885  0.435\n",
            0,
            1,
        )
        .unwrap();
        let coords = InternalCoords::generate(&mol).unwrap();
        let dih_idx = coords
            .primitives
            .iter()
            .position(|p| matches!(p, Primitive::Dihedral(..)))
            .expect("H2O2 has a dihedral");

        let pi = std::f64::consts::PI;
        let mut q_before = Array1::zeros(coords.len());
        let mut q_after = Array1::zeros(coords.len());
        // Straddle the cut: +179 deg -> -179 deg is a +2 deg move, not -358.
        q_before[dih_idx] = pi - 0.0175;
        q_after[dih_idx] = -pi + 0.0175;

        let s = bfgs_secant_step(&coords, &q_before, &q_after);
        eprintln!(
            "torsion {:.4} rad -> {:.4} rad gives s = {:.4} rad ({:.2} deg)",
            q_before[dih_idx],
            q_after[dih_idx],
            s[dih_idx],
            s[dih_idx].to_degrees()
        );
        assert!(
            (s[dih_idx] - 0.035).abs() < 1e-9,
            "expected the short way round (+0.035 rad), got {:.6} rad",
            s[dih_idx]
        );

        // A NON-torsion must NOT be wrapped: bonds have no periodicity, and
        // wrapping one would silently corrupt a large bond displacement.
        let bond_idx = coords
            .primitives
            .iter()
            .position(|p| matches!(p, Primitive::Bond(..)))
            .expect("H2O2 has bonds");
        let mut qb = Array1::zeros(coords.len());
        let mut qa = Array1::zeros(coords.len());
        qa[bond_idx] = 2.0 * pi + 0.5; // larger than 2*pi on purpose
        let s2 = bfgs_secant_step(&coords, &qb, &qa);
        assert!(
            (s2[bond_idx] - (2.0 * pi + 0.5)).abs() < 1e-12,
            "a bond displacement must never be wrapped: got {:.6}",
            s2[bond_idx]
        );
        let _ = &mut qb;
    }

    /// One optimization with the warm start on or off, keeping the
    /// per-evaluation warm-start records (the public drivers print them).
    fn optimize_recorded(
        mol: &Molecule,
        basis: &str,
        cfg: &RhfConfig,
        spin: crate::result::Spin,
        warm_on: bool,
    ) -> (OptimizeResult, Vec<crate::warm_start::WarmStartRecord>) {
        let ctx = ParallelContext::default();
        let op = Operator::coulomb();
        let opt = OptimizeConfig {
            scf_warm_start: warm_on,
            ..Default::default()
        };
        let mut warm = ScfWarmStart::new(warm_on);
        let r = run_bfgs(mol, &opt, |m| {
            let (e, g, _) = match spin {
                crate::result::Spin::Restricted => {
                    compute_energy_gradient_and_result(&ctx, m, basis, op, cfg, &mut warm)?
                }
                crate::result::Spin::Unrestricted => {
                    compute_energy_and_gradient_uhf(&ctx, m, basis, op, cfg, &mut warm)?
                }
                crate::result::Spin::RestrictedOpen => {
                    compute_energy_and_gradient_rohf(&ctx, m, basis, op, cfg, &mut warm)?
                }
            };
            Ok((e, g))
        })
        .unwrap();
        (r, warm.records().to_vec())
    }

    /// Warm vs cold optimization: same minimum within the optimizer's own
    /// tolerances, warm start actually used after the first evaluation, and
    /// fewer SCF iterations in total.
    ///
    /// Geometry bound: the optimizer stops once `g_max < g_max_thresh`
    /// (4.5e-4 Ha/Bohr); with the softest force constant of these molecules
    /// above ~0.2 Ha/Bohr² (bends), a converged geometry is determined only
    /// to `g_max_thresh / 0.2 ≈ 2.3e-3` Bohr per coordinate, so two converged
    /// runs may differ by at most twice that.
    fn warm_and_cold_optimizations_agree(
        xyz: &str,
        mult: usize,
        basis: &str,
        cfg: RhfConfig,
        spin: crate::result::Spin,
    ) {
        let mol = Molecule::parse_xyz(xyz, 0, mult).unwrap();
        let (rw, recw) = optimize_recorded(&mol, basis, &cfg, spin, true);
        let (rc, recc) = optimize_recorded(&mol, basis, &cfg, spin, false);
        let it = |r: &[crate::warm_start::WarmStartRecord]| -> Vec<usize> {
            r.iter()
                .map(|x| x.iterations + x.wasted_iterations)
                .collect()
        };
        let dx = rw
            .mol
            .atoms
            .iter()
            .zip(&rc.mol.atoms)
            .flat_map(|(a, b)| [a.x - b.x, a.y - b.y, a.zpos - b.zpos])
            .fold(0.0f64, |m, d| m.max(d.abs()));
        let de = (rw.energy - rc.energy).abs();
        eprintln!(
            "{basis} {spin:?}: steps warm {} cold {}; |dE| {de:.2e}; max|dx| {dx:.2e} Bohr; \
             SCF iterations warm {:?} (sum {}) cold {:?} (sum {})",
            rw.steps,
            rc.steps,
            it(&recw),
            it(&recw).iter().sum::<usize>(),
            it(&recc),
            it(&recc).iter().sum::<usize>(),
        );
        assert!(rw.converged && rc.converged);
        let opt = OptimizeConfig::default();
        assert!(de < opt.e_conv, "|dE| {de:e} >= e_conv");
        assert!(dx < 2.0 * opt.g_max_thresh / 0.2, "max|dx| {dx:e}");
        assert!(recw.iter().skip(1).all(|r| r.cold.is_none()), "{recw:?}");
        assert!(recc
            .iter()
            .all(|r| r.cold == Some(crate::warm_start::ColdReason::Disabled)));
        assert!(
            it(&recw).iter().sum::<usize>() < it(&recc).iter().sum::<usize>(),
            "warm start did not save SCF iterations"
        );
    }

    const DISTORTED_WATER: &str = "3\nH2O\nO 0 0 0\nH 0 0.9 0\nH 0 -0.3 0.85\n";

    #[test]
    fn warm_start_rhf_optimization_matches_cold() {
        warm_and_cold_optimizations_agree(
            DISTORTED_WATER,
            1,
            "6-31g",
            RhfConfig::default(),
            crate::result::Spin::Restricted,
        );
    }

    #[test]
    fn warm_start_uhf_radical_optimization_matches_cold() {
        warm_and_cold_optimizations_agree(
            "3\nNH2\nN 0 0 0\nH 0 0.95 0.2\nH 0 -0.6 0.85\n",
            2,
            "6-31g",
            RhfConfig::default(),
            crate::result::Spin::Unrestricted,
        );
    }

    #[test]
    fn warm_start_pbe_optimization_matches_cold() {
        warm_and_cold_optimizations_agree(
            DISTORTED_WATER,
            1,
            "6-31g",
            RhfConfig {
                xc: Some("PBE".to_string()),
                ..Default::default()
            },
            crate::result::Spin::Restricted,
        );
    }

    /// The ratio rule's decision table (the same cases as claim T2 of
    /// `scripts/geometry_search_steps_sympy.py`).
    #[test]
    fn ratio_trust_update_follows_the_decision_table() {
        let cases = [
            ((-0.1e-3, -1e-3, 0.1, 0.1), (true, 0.025)),
            ((1e-3, -1e-3, 0.1, 0.1), (false, 0.025)),
            ((-0.9e-3, -1e-3, 0.1, 0.1), (true, 0.2)),
            ((-0.9e-3, -1e-3, 0.05, 0.1), (true, 0.1)),
            ((-0.5e-3, -1e-3, 0.1, 0.1), (true, 0.1)),
            ((-1.5e-3, -1e-3, 0.1, 0.1), (true, 0.2)),
            ((5e-9, -5e-9, 0.1, 0.1), (true, 0.1)),
            ((-0.9e-3, -1e-3, 1.0, 1.0), (true, 1.0)),
            ((1e-3, -1e-3, 1e-3, 1e-3), (false, 1e-3)),
            // predicted >= 0: a rise predicted as a rise (r = 2 on the
            // boundary) and a fall predicted as a rise both shrink.
            ((2e-3, 1e-3, 0.1, 0.1), (false, 0.025)),
            ((-1e-3, 1e-3, 0.1, 0.1), (true, 0.025)),
        ];
        for ((a, p, s, r), want) in cases {
            let got = ratio_trust_update(a, p, s, r, 1e-3, 1.0);
            assert_eq!(got.0, want.0, "accept for {:?}", (a, p, s, r));
            assert!(
                (got.1 - want.1).abs() < 1e-15,
                "radius {got:?} for {:?}",
                (a, p, s, r)
            );
        }
    }

    /// A rejected step, or one whose predicted change is `>= 0`, never grows
    /// the radius, over a grid that reaches every branch of the rule.
    /// Mutation: dropping `!accept || predicted >= 0.0 ||` lets the grid's
    /// rises predicted as rises double the radius.
    #[test]
    fn ratio_trust_update_never_grows_on_rejection_or_a_non_descent_model() {
        let values = [-2e-3, -1e-3, -1e-5, -5e-9, 5e-9, 2e-8, 1e-5, 1e-3, 2e-3];
        let (mut n_rej, mut n_pos) = (0, 0);
        for actual in values {
            for predicted in values {
                for radius in [0.01, 0.1, 1.0] {
                    for frac in [0.25, 0.99, 1.0] {
                        let (acc, next) =
                            ratio_trust_update(actual, predicted, frac * radius, radius, 0.01, 4.0);
                        if !acc || predicted >= 0.0 {
                            n_rej += usize::from(!acc);
                            n_pos += usize::from(predicted >= 0.0);
                            assert!(
                                next <= radius,
                                "radius grew {radius} -> {next} (accept {acc}, actual \
                                 {actual:e}, predicted {predicted:e}, step {})",
                                frac * radius
                            );
                        }
                    }
                }
            }
        }
        assert!(n_rej > 50 && n_pos > 50, "grid too small: {n_rej} {n_pos}");
    }

    /// Internal step under the ratio rule: one clip factor, so the step is
    /// `alpha p` and descends. The constructed case is one where the per-block
    /// clip (the `fixed` search, and the ratio search before this fix) goes
    /// uphill at every radius: with `g = (1, 0.1)` on one bond and one angle
    /// and the positive definite block `H = [[1, -20], [-20, 1000]]`,
    /// `p = (1, -80)` descends (`g.p = -7`), but clipping the bond to 0.1 and
    /// the angle to 0.3 gives `g.dq = 0.1 - 0.03 > 0`. Mutation: the ratio arm
    /// of `internal_trial_step` calling `clip_internal_step`.
    #[test]
    fn ratio_internal_step_descends_where_the_per_block_clip_does_not() {
        let mol = ethane_for_steps();
        let coords = InternalCoords::generate(&mol).unwrap();
        let nq = coords.len();
        let i_d = coords
            .primitives
            .iter()
            .position(|p| !p.is_angular())
            .unwrap();
        let i_a = coords
            .primitives
            .iter()
            .position(|p| p.is_angular())
            .unwrap();
        let mut g = Array1::zeros(nq);
        g[i_d] = 1.0;
        g[i_a] = 0.1;
        let mut h = Array2::eye(nq) * 1e-6;
        h[(i_d, i_d)] = 1.0;
        h[(i_a, i_a)] = 1000.0;
        h[(i_d, i_a)] = -20.0;
        h[(i_a, i_d)] = -20.0;
        let seed = Array2::eye(nq);
        let p = -h.dot(&g);
        assert!(g.dot(&p) < 0.0, "precondition: -H g descends");
        for scale in [1.0, 1e-2, 1e-4] {
            let (dist, ang) = (0.1 * scale, INTERNAL_ANGULAR_TRUST * scale);
            let (fixed, _) =
                internal_trial_step(&coords, &g, &h, &seed, dist, ang, TrustUpdate::Fixed);
            assert!(
                g.dot(&fixed) > 0.0,
                "precondition: the per-block clip goes uphill at scale {scale}"
            );
            let (dq, fell_back) =
                internal_trial_step(&coords, &g, &h, &seed, dist, ang, TrustUpdate::Ratio);
            assert!(!fell_back);
            assert!(
                g.dot(&dq) < 0.0,
                "ratio step uphill at scale {scale}: g.dq = {:e}",
                g.dot(&dq)
            );
            // dq = alpha p with alpha = min(dist / |p_d|, ang / |p_a|) = ang / 80.
            let alpha = ang / 80.0;
            for k in 0..nq {
                assert!((dq[k] - alpha * p[k]).abs() <= 1e-15 * p[k].abs().max(1.0));
            }
        }
        // A model that does not descend falls back to the seed metric.
        let (dq, fell_back) = internal_trial_step(
            &coords,
            &g,
            &(-&h),
            &seed,
            0.1,
            INTERNAL_ANGULAR_TRUST,
            TrustUpdate::Ratio,
        );
        assert!(fell_back && g.dot(&dq) < 0.0);
    }

    /// The Cartesian Lindh seed: symmetric positive definite; the identity on
    /// the rigid-body motions; and the exact inverse of the model on the
    /// directions whose eigenvalue is above the floor (`H0 H_L v = v`).
    #[test]
    fn lindh_inverse_seed_inverts_the_model_off_the_rigid_body_space() {
        let mol = Molecule::parse_xyz(
            "9\nethanol\n\
             C -0.0478 -0.4252 0.0000\nC 1.2600 0.3599 0.0000\nO -1.1661 0.4545 0.0000\n\
             H -0.0951 -1.0675 0.8857\nH -0.0951 -1.0675 -0.8857\nH 2.1102 -0.3268 0.0000\n\
             H 1.3184 1.0000 0.8810\nH 1.3184 1.0000 -0.8810\nH -1.9867 -0.0507 0.0000\n",
            0,
            1,
        )
        .unwrap();
        let h0 = lindh_inverse_hessian(&mol).unwrap();
        let hl = ferric_core::model_hessian::lindh_cartesian_hessian(&mol).unwrap();
        let asym = (&h0 - &h0.t()).iter().fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(asym < 1e-10, "asymmetry {asym:e}");
        let (w0, _) = h0.eigh(UPLO::Upper).unwrap();
        assert!(w0[0] > 0.0, "H0 not positive definite: {}", w0[0]);
        let t = rigid_body_basis(&mol);
        assert_eq!(t.ncols(), 6);
        let ht = h0.dot(&t) - &t;
        assert!(ht.iter().fold(0.0f64, |m, v| m.max(v.abs())) < 1e-10);
        let (w, v) = hl.eigh(UPLO::Upper).unwrap();
        let mut checked = 0;
        for (j, &wj) in w.iter().enumerate() {
            if wj > 10.0 * MODEL_HESSIAN_FLOOR {
                let col = v.column(j).to_owned();
                let back = h0.dot(&hl.dot(&col)) - &col;
                assert!(back.dot(&back).sqrt() < 1e-9, "mode {j} (w = {wj})");
                checked += 1;
            }
        }
        assert!(checked >= 15, "only {checked} modes above the floor");
    }

    /// Ethanol, a water 6 Bohr from its hydroxyl H (inside the Lindh
    /// environment of the hydroxyl) and a water 80 Bohr away (outside it).
    fn ethanol_and_two_waters() -> Molecule {
        Molecule::parse_xyz(
            "15\nethanol + 2 H2O\n\
             C -0.0478 -0.4252 0.0000\nC 1.2600 0.3599 0.0000\nO -1.1661 0.4545 0.0000\n\
             H -0.0951 -1.0675 0.8857\nH -0.0951 -1.0675 -0.8857\nH 2.1102 -0.3268 0.0000\n\
             H 1.3184 1.0000 0.8810\nH 1.3184 1.0000 -0.8810\nH -1.9867 -0.0507 0.0000\n\
             O -4.9 -1.2 0.3\nH -5.5 -1.9 0.1\nH -5.4 -0.5 0.8\n\
             O 40.0 0.0 0.0\nH 40.6 0.7 0.0\nH 40.6 -0.7 0.0\n",
            0,
            1,
        )
        .unwrap()
    }

    /// The free-free block built from the Lindh environment is the block of
    /// the whole structure's model. Same terms, different summation order:
    /// each entry is a sum of at most a few hundred terms of size <= max|H|,
    /// so the two agree to ~1e2 eps max|H|; 1e-13 max|H| is the bar. The
    /// environment is checked to be a strict subset of the structure and a
    /// strict superset of the free atoms, so both the pruning and the fixed
    /// neighbours are exercised.
    #[test]
    fn lindh_free_block_matches_the_block_of_the_whole_structure() {
        let mol = ethanol_and_two_waters();
        let n = mol.atoms.len();
        // The hydroxyl O and H, and one methylene H: the rest is fixed.
        let free = [2usize, 3, 8];
        let env = lindh_environment(&mol, &free);
        assert!(env.len() < n, "environment is the whole structure");
        assert!(env.len() > free.len(), "environment adds no fixed atom");
        assert!(!env.contains(&12), "the far water is in the environment");
        assert!(env.contains(&9), "the near water is not in the environment");
        let full = ferric_core::model_hessian::lindh_cartesian_hessian(&mol).unwrap();
        let block = lindh_free_block(&mol, &free).unwrap();
        let scale = full.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let mut worst = 0.0f64;
        for (a, &fa) in free.iter().enumerate() {
            for (b, &fb) in free.iter().enumerate() {
                for c in 0..3 {
                    for d in 0..3 {
                        let diff = block[(3 * a + c, 3 * b + d)] - full[(3 * fa + c, 3 * fb + d)];
                        worst = worst.max(diff.abs());
                    }
                }
            }
        }
        assert!(
            worst < 1e-13 * scale,
            "block differs by {worst:e} (scale {scale:e})"
        );
        // The fixed neighbours put curvature on the free atoms: the block is
        // not the model of the free atoms alone.
        let alone = ferric_core::model_hessian::lindh_cartesian_hessian(&Molecule {
            atoms: free.iter().map(|&i| mol.atoms[i].clone()).collect(),
            ..mol.clone()
        })
        .unwrap();
        let gap = (&block - &alone).iter().fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(
            gap > 1e-2,
            "fixed neighbours change the block by only {gap:e}"
        );
    }

    /// With every atom free the QM/MM seed is the molecular seed, bit for bit.
    #[test]
    fn lindh_free_seed_with_every_atom_free_is_the_molecular_seed() {
        let mol = ethanol_and_two_waters();
        let all: Vec<usize> = (0..mol.atoms.len()).collect();
        let a = lindh_inverse_hessian_free(&mol, &all).unwrap();
        let b = lindh_inverse_hessian(&mol).unwrap();
        assert!(a
            .iter()
            .zip(b.iter())
            .all(|(x, y)| x.to_bits() == y.to_bits()));
    }

    /// `OptimizeResult::search` is the search that ran: internals asked for on
    /// an atom (no bonds, so no internal coordinates) fall back to Cartesian,
    /// and the result says so; a bonded molecule keeps internals. Unset
    /// fields resolve to the defaults.
    #[test]
    fn the_result_records_the_search_that_ran() {
        let ctx = ParallelContext::default();
        let rhf_config = RhfConfig::default();
        let cfg = OptimizeConfig {
            coord_system: CoordSystem::RedundantInternal,
            max_steps: 2,
            ..Default::default()
        };
        let he = Molecule::parse_xyz("1\nHe\nHe 0 0 0\n", 0, 1).unwrap();
        let r =
            optimize_geometry(&ctx, &he, "sto-3g", Operator::coulomb(), &rhf_config, &cfg).unwrap();
        assert_eq!(
            r.search,
            SearchSettings {
                coord_system: CoordSystem::Cartesian,
                initial_hessian: DEFAULT_INITIAL_HESSIAN,
                trust_update: TrustUpdate::Ratio,
            }
        );
        let h2 = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.8\n", 0, 1).unwrap();
        let r =
            optimize_geometry(&ctx, &h2, "sto-3g", Operator::coulomb(), &rhf_config, &cfg).unwrap();
        assert_eq!(r.search, cfg.requested_search());
        assert_eq!(r.search.coord_system, CoordSystem::RedundantInternal);
        assert_eq!(
            r.search.to_string(),
            "internal coordinates, lindh initial Hessian, ratio trust radius"
        );
    }
}
