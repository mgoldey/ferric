//! Optional COSX SCF grid SCHEDULE (ORCA style: a coarse exchange grid for the
//! early iterations, the production grid [`CosxConfig::grid`] once the density
//! has mostly settled, then the unchanged final-grid pass).
//!
//! OFF by default (`CosxConfig::schedule == None`): no [`ScheduleRun`] is
//! created, every helper below is a pass-through, and the SCF is the
//! unscheduled one bit for bit (`tests/cosx_grid_schedule.rs` anchors it).
//!
//! # What the schedule does, per iteration
//!
//! 1. Iterations `1..switch_iter` build K on [`CosxGridSchedule::coarse_grid`].
//! 2. The iteration after the first one whose incoming density change
//!    `max|ΔD|` (`ScfMonitor::dp_max`, the quantity the convergence gate
//!    already tracks) falls below [`CosxGridSchedule::switch_dp_max`] rebuilds
//!    the builder on the production grid and builds K there from then on.
//! 3. The convergence gate is CLOSED while on the coarse grid AND on the first
//!    production iteration, whose density (and the ΔD/ΔE it is judged by) was
//!    still produced by a coarse-grid Fock. The earliest accepted iteration is
//!    therefore the second production one: its density came from
//!    diagonalizing a production Fock and its ΔE compares two production-grid
//!    energies. The converged energy is defined by the production grid alone.
//!
//! # DIIS at the switch: the history is RESET
//!
//! The coarse-grid entries are commutators of a DIFFERENT Fock operator. Near
//! the switch they are small (that is what triggered it), so Pulay DIIS — which
//! favours the smallest error vectors — would keep extrapolating towards the
//! COARSE fixed point and only forget it as entries age out of the subspace.
//! Restarting the history costs a couple of plain steps from a density that is
//! already good to `switch_dp_max`, and makes every extrapolation after the
//! switch a combination of production-grid Focks only.
//!
//! The choice is made on that ground, not on speed. Measured once
//! (2026-10-08, water/cc-pVDZ RHF, default grids, density_conv 1e-8): reset
//! converges in 15 iterations (8 coarse + 7 production), keeping the history
//! in 14 (8 + 6), both to the unscheduled energy within 1.2e-13 Ha. One
//! molecule; not a general cost verdict.

use crate::cosx_k::{validate_grid, CosxConfig};
use crate::diis::HistoryReset;
use crate::fock::KBuilder;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_core::FerricError;
use ferric_dft::grid::AtomicGridConfig;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;

/// `(radial, peak angular, prune)` of the default coarse grid: the pruned
/// `sgx` row peaking at 110 (regions 14/26/50/110/50 — ORCA's AngularGrid 2
/// row, PySCF SGX level 1) on 25 radial shells. One SGX row below the
/// production default (35,194) on BOTH axes. The peak-50 row (ORCA's
/// AngularGrid 1) was not chosen: the angular order dominates the COSX error
/// (`site/src/methods/scf.md`), and the overlap fit inverts the numerical
/// overlap `S_num`, which a 50-point valence shell makes far worse conditioned
/// — a too-coarse first phase would trade grid cost for a failed fit or extra
/// production iterations, neither of which has been measured.
pub const COSX_DEFAULT_COARSE_GRID: (usize, usize, Option<ferric_dft::prune::PruneScheme>) =
    (25, 110, Some(ferric_dft::prune::PruneScheme::Sgx));

/// Default switch threshold on the incoming `max|ΔD|` (the convergence gate's
/// `dp_max`). 1e-3 is three orders above the default `density_conv` (1e-6,
/// gate at `dp_max < 1e-5`): the density is qualitatively settled, while
/// enough iterations remain for the production grid to define the fixed point.
pub const COSX_DEFAULT_SWITCH_DP_MAX: f64 = 1e-3;

/// The coarse-grid schedule carried in [`CosxConfig::schedule`].
#[derive(Debug, Clone)]
pub struct CosxGridSchedule {
    /// The exchange grid of the early iterations.
    pub coarse_grid: AtomicGridConfig,
    /// Switch to the production grid once the incoming `max|ΔD|` is below
    /// this. Must be finite and > 0.
    pub switch_dp_max: f64,
}

impl Default for CosxGridSchedule {
    fn default() -> Self {
        let (n_radial, n_angular, prune) = COSX_DEFAULT_COARSE_GRID;
        Self {
            coarse_grid: AtomicGridConfig {
                n_radial,
                n_angular,
                prune,
            },
            switch_dp_max: COSX_DEFAULT_SWITCH_DP_MAX,
        }
    }
}

impl CosxGridSchedule {
    /// Strict validation: a tabulated coarse grid and a finite positive
    /// switch threshold.
    pub fn validate(&self) -> Result<(), FerricError> {
        validate_grid(&self.coarse_grid)?;
        if !(self.switch_dp_max > 0.0 && self.switch_dp_max.is_finite()) {
            return Err(FerricError::General(format!(
                "cosx grid schedule: switch_dp_max = {} must be finite and > 0",
                self.switch_dp_max
            )));
        }
        Ok(())
    }
}

/// Which grid one iteration's K was built on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CosxGridPhase {
    /// [`CosxGridSchedule::coarse_grid`].
    Coarse,
    /// [`CosxConfig::grid`].
    Production,
}

/// What a scheduled SCF actually did, on `ScfResult::cosx_schedule`.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct CosxScheduleRecord {
    /// Grid of the K build of iteration `i + 1`, one entry per iteration.
    pub phases: Vec<CosxGridPhase>,
    /// Exchange-grid point count of the builder that made that K.
    pub npts: Vec<usize>,
    /// First iteration (1-based) built on the production grid; `None` if the
    /// run never left the coarse grid.
    pub switch_iter: Option<usize>,
}

/// Live schedule state of one SCF run. Exists only when a schedule is
/// configured AND the run's resolved K builder is COSX.
#[derive(Debug)]
pub(crate) struct ScheduleRun {
    switch_dp_max: f64,
    phase: CosxGridPhase,
    pending_switch: bool,
    record: CosxScheduleRecord,
}

/// Start a schedule for a run whose resolved pluggable K is `k_kind`. `None`
/// (no schedule, or K is not COSX) makes every helper here a pass-through.
pub(crate) fn start(
    k_kind: Option<&str>,
    cfg: &CosxConfig,
) -> Result<Option<ScheduleRun>, FerricError> {
    let Some(s) = cfg.schedule.as_ref().filter(|_| k_kind == Some("cosx")) else {
        return Ok(None);
    };
    s.validate()?;
    Ok(Some(ScheduleRun {
        switch_dp_max: s.switch_dp_max,
        phase: CosxGridPhase::Coarse,
        pending_switch: false,
        record: CosxScheduleRecord::default(),
    }))
}

/// Hard error for a solver without schedule support (`solver`, e.g.
/// "ROHF/ROKS") when a schedule is configured and K is COSX.
pub(crate) fn refuse(
    k_kind: Option<&str>,
    cfg: &CosxConfig,
    solver: &str,
) -> Result<(), FerricError> {
    if k_kind == Some("cosx") && cfg.schedule.is_some() {
        return Err(FerricError::General(format!(
            "COSX grid schedule is implemented for RHF/RKS and UHF/UKS, not {solver}; \
             unset it for {solver} runs"
        )));
    }
    Ok(())
}

/// Everything `fock_assembly::build_pluggable_k` needs, so the solver loops
/// can hand the schedule one value to rebuild the exchange builder from.
pub(crate) struct KBuilderArgs<'a, B> {
    pub kind: Option<&'a str>,
    pub ctx: &'a ParallelContext,
    pub mol: &'a Molecule,
    pub prep: &'a PreparedBasis,
    pub link_bound: &'a B,
    pub op: Operator,
    /// The run's COSX config (production grid).
    pub cosx: &'a CosxConfig,
    pub integral_thresh: f64,
    pub ooc_budget: usize,
}

impl<'a, B: crate::screening::Bound + Sync> KBuilderArgs<'a, B> {
    fn build(&self, cosx: &CosxConfig) -> Result<Option<Box<dyn KBuilder + 'a>>, FerricError> {
        crate::fock_assembly::build_pluggable_k(
            self.kind,
            self.ctx,
            self.mol,
            self.prep,
            self.link_bound,
            self.op,
            cosx,
            self.integral_thresh,
            self.ooc_budget,
        )
    }
}

/// The run's schedule state together with its FIRST pluggable K builder:
/// on the coarse grid when a schedule runs, else exactly
/// `build_pluggable_k(.., args.cosx, ..)` as before.
pub(crate) type StartedK<'a> = (Option<ScheduleRun>, Option<Box<dyn KBuilder + 'a>>);

/// Start the schedule (if any) and build the first pluggable K builder.
pub(crate) fn start_with_builder<'a, B: crate::screening::Bound + Sync>(
    args: &KBuilderArgs<'a, B>,
) -> Result<StartedK<'a>, FerricError> {
    let run = start(args.kind, args.cosx)?;
    let builder = args.build(&initial_config(&run, args.cosx))?;
    Ok((run, builder))
}

/// [`before_k_build_with`] for the solver loops: the switch rebuilds from
/// `args` on the production grid and resets `diis`.
pub(crate) fn before_k_build<'a, B: crate::screening::Bound + Sync, H: HistoryReset>(
    run: &mut Option<ScheduleRun>,
    iter: usize,
    k_builder: &mut Option<Box<dyn KBuilder + 'a>>,
    diis: &mut H,
    args: &KBuilderArgs<'a, B>,
) -> Result<(), FerricError> {
    before_k_build_with(
        run,
        iter,
        k_builder,
        || args.build(args.cosx),
        || diis.reset_history(),
    )
}

/// The config the run's FIRST COSX builder is built from: `cfg` itself, or a
/// copy on the coarse grid when a schedule is running.
fn initial_config(run: &Option<ScheduleRun>, cfg: &CosxConfig) -> CosxConfig {
    match (run, cfg.schedule.as_ref()) {
        (Some(_), Some(s)) => CosxConfig {
            grid: s.coarse_grid.clone(),
            ..cfg.clone()
        },
        _ => cfg.clone(),
    }
}

/// Call once per iteration BEFORE the K build. Performs a pending switch
/// (rebuilds `k_builder` with `rebuild`, resets DIIS with `reset_diis` — see
/// the module doc for why) and records this iteration's grid.
fn before_k_build_with<'a, R, D>(
    run: &mut Option<ScheduleRun>,
    iter: usize,
    k_builder: &mut Option<Box<dyn KBuilder + 'a>>,
    rebuild: R,
    reset_diis: D,
) -> Result<(), FerricError>
where
    R: FnOnce() -> Result<Option<Box<dyn KBuilder + 'a>>, FerricError>,
    D: FnOnce(),
{
    let Some(run) = run.as_mut() else {
        return Ok(());
    };
    if run.pending_switch {
        *k_builder = rebuild()?;
        reset_diis();
        run.pending_switch = false;
        run.phase = CosxGridPhase::Production;
        run.record.switch_iter = Some(iter);
    }
    run.record.phases.push(run.phase);
    run.record.npts.push(
        k_builder
            .as_ref()
            .and_then(|k| k.exchange_grid_npts())
            .unwrap_or(0),
    );
    Ok(())
}

/// Filter the convergence decision of iteration `iter` (call right after
/// `scf_converged`). On the coarse grid it arms the switch once
/// `dp_max < switch_dp_max` (or once the gate would have fired) and refuses
/// convergence; on the first production iteration it refuses convergence too.
pub(crate) fn gate(
    run: &mut Option<ScheduleRun>,
    iter: usize,
    dp_max: f64,
    conv: Option<crate::result::ScfExit>,
) -> Option<crate::result::ScfExit> {
    let Some(run) = run.as_mut() else {
        return conv;
    };
    match run.phase {
        CosxGridPhase::Coarse => {
            run.pending_switch = dp_max < run.switch_dp_max || conv.is_some();
            None
        }
        CosxGridPhase::Production => conv.filter(|_| run.record.switch_iter != Some(iter)),
    }
}

/// The record for `ScfResult::cosx_schedule` (`None` without a schedule).
pub(crate) fn record(run: &Option<ScheduleRun>) -> Option<CosxScheduleRecord> {
    run.as_ref().map(|r| r.record.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::result::ScfExit;

    fn run() -> Option<ScheduleRun> {
        let cfg = CosxConfig {
            schedule: Some(CosxGridSchedule::default()),
            ..CosxConfig::default()
        };
        start(Some("cosx"), &cfg).unwrap()
    }

    #[test]
    fn schedule_is_inert_without_cosx_or_without_a_schedule() {
        let cfg = CosxConfig {
            schedule: Some(CosxGridSchedule::default()),
            ..CosxConfig::default()
        };
        assert!(start(Some("link"), &cfg).unwrap().is_none());
        assert!(start(None, &cfg).unwrap().is_none());
        assert!(start(Some("cosx"), &CosxConfig::default())
            .unwrap()
            .is_none());
        let mut none = None;
        let c = Some(ScfExit::Converged);
        assert_eq!(gate(&mut none, 2, 0.0, c), c);
    }

    #[test]
    fn gate_refuses_coarse_and_first_production_iterations() {
        let mut r = run();
        let c = Some(ScfExit::Converged);
        // Coarse, ΔD above the switch and the gate not firing: no switch.
        assert_eq!(gate(&mut r, 3, 1e-2, None), None);
        let mut kb: Option<Box<dyn KBuilder>> = None;
        before_k_build_with(&mut r, 4, &mut kb, || Ok(None), || {}).unwrap();
        assert_eq!(r.as_ref().unwrap().phase, CosxGridPhase::Coarse);
        // Coarse, ΔD below: refused, switch armed for the next iteration.
        assert_eq!(gate(&mut r, 4, 1e-4, c), None);
        let mut reset = false;
        before_k_build_with(&mut r, 5, &mut kb, || Ok(None), || reset = true).unwrap();
        assert!(reset);
        // First production iteration: refused; the next one is accepted.
        assert_eq!(gate(&mut r, 5, 0.0, c), None);
        assert_eq!(gate(&mut r, 6, 0.0, c), c);
        let rec = record(&r).unwrap();
        assert_eq!(rec.switch_iter, Some(5));
        assert_eq!(
            rec.phases,
            vec![CosxGridPhase::Coarse, CosxGridPhase::Production]
        );
    }

    #[test]
    fn schedule_validation_is_strict() {
        let mut s = CosxGridSchedule::default();
        s.validate().unwrap();
        s.switch_dp_max = 0.0;
        assert!(s.validate().is_err());
        s.switch_dp_max = f64::NAN;
        assert!(s.validate().is_err());
        let s = CosxGridSchedule {
            coarse_grid: AtomicGridConfig {
                n_radial: 25,
                n_angular: 26,
                prune: Some(ferric_dft::prune::PruneScheme::Sgx),
            },
            ..Default::default()
        };
        assert!(s.validate().is_err());
    }
}
