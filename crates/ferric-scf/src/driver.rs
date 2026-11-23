//! Spin-generic SCF driver components shared by all six SCF variants.
//!
//! `solve_rhf` (RHF + RKS), `solve_uhf` (UHF + UKS) and `solve_rohf`
//! (ROHF + ROKS) are one iteration scheme instantiated over three spin
//! policies. This module owns everything that is NOT spin policy, passed in
//! as one prepared environment plus small per-iteration helpers:
//!
//! - [`ScfEnv`] / [`prepare`] — the geometry-only pre-loop environment
//!   (overlap, core Hamiltonian, nuclear + external-potential energy, solvent
//!   contexts, resolved memory budget, RSH exchange fitters), built once from
//!   the shared `RhfConfig` and handed to the solver.
//! - [`ScfMonitor`] — convergence bookkeeping (previous energy, ΔP signals,
//!   divergence streak, stall history). One implementation of the ΔP-primary
//!   gate (see `rhf::scf_converged`) and the stall/divergence early exits for
//!   every variant — UHF/ROHF previously ignored `stall_window` /
//!   `divergence_tol` silently; through this monitor they honor them (defaults
//!   `None` are byte-identical to the old behavior).
//! - [`solvent_terms`] — the per-iteration COSMO + IEF-PCM reaction-field
//!   fold, applied uniformly to however many spin Focks the policy carries.
//! - [`diagonalize_rect`] — the rectangular canonical-orthogonalizer Fock
//!   diagonalization shared by RHF and UHF (ROHF still uses its historical
//!   symmetric S^{-1/2} — a real policy difference, see below).
//! - [`density_change`] — the ΔP (rms, max) pair all variants gate on.
//!
//! ## What remains spin policy (deliberately per-solver)
//!
//! The occupation/orbital-update step: closed-shell aufbau (RHF), independent
//! α/β diagonalization + fractional occupation + per-spin MOM (UHF), and the
//! Roothaan effective-Fock coupling producing ONE MO set from two Focks
//! (ROHF); plus each variant's second-order (Newton/AH) machinery and ROHF's
//! symmetric orthogonalizer. Those differences are behavioral — folding them
//! into a trait is possible but must be validated against the slow
//! pathological systems (TM dimers, plateau radicals), not just the fast
//! suite; extract further seams here as they prove out.
//!
//! Exchange assembly (the RSH fold and the fitter pair) lives in
//! [`crate::fock_assembly`]; reduction banding in [`crate::reduce`].

use crate::df_k::DfK;
use crate::fock::{JBuilder, KBuilder};
use crate::rhf::{ConvergenceSignals, RhfConfig};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_core::timing::PhaseAccumulator;
use ferric_core::FerricError;
use ferric_dft::xc_trait::KMix;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::blas_threads::{opt_in_blas_threads, with_blas_threads};
use ferric_integrals::oneelectron;
use ndarray::Array2;
use ndarray_linalg::Eigh;
use std::cell::RefCell;

/// Geometry-only SCF environment, built once before the iteration loop.
/// Everything here depends only on (molecule, basis, config) — never on the
/// density — so it is identical for all spin policies.
pub(crate) struct ScfEnv<'a> {
    /// Overlap matrix S.
    pub s: Array2<f64>,
    /// Core Hamiltonian (T + V [+ V_ECP] [+ external point-charge/field terms]).
    pub h: Array2<f64>,
    /// Nuclear repulsion + external-potential nuclear terms.
    pub vnn: f64,
    /// Fully-resolved unified memory budget (TOML > env > auto) — governs the
    /// DF 3-index footprint and the direct builders' reduction band scratch.
    pub ooc_budget: usize,
    /// COSMO cavity (geometry-only); `None` when solvation is off.
    pub cosmo_cavity: Option<crate::cosmo::CosmoCavity>,
    /// IEF-PCM geometry setup; `None` when PCM is off.
    pub pcm_ctx: Option<ferric_pcm::PcmContext>,
    /// Polarizable-embedding p-shell site basis (geometry-only; built once
    /// from `config.polarizable.sites` at `config.polarizable.dipole_zeta`,
    /// mirroring `cosmo_cavity`/`pcm_ctx`). `None` when polarizable
    /// embedding is off OR configured with an empty site list.
    pub polarizable_site_basis: Option<ferric_integrals::site_basis::SiteBasis>,
    /// RSH SR/LR exchange fitters (ω > 0 only) — see `fock_assembly`.
    ///
    /// Behind a `RefCell` so the ONE pair that assembles the converged Fock can
    /// also be handed to the orbital-Hessian matvec (`crate::rsh_response`)
    /// without the solver giving up ownership for the rest of the loop. A
    /// second pair built with the same arguments would pay the DF build twice
    /// and reopen the Fock/Hessian divergence #314 closed; `DfK::build` needs
    /// `&mut self`, so shared read-only borrows alone will not do.
    pub dfk_sr: Option<RefCell<DfK<'a>>>,
    pub dfk_lr: Option<RefCell<DfK<'a>>>,
}

/// The budget RI-J/K setup decides against, allowing for what a deferred XC
/// evaluator (which allocates only after that setup) means to hold.
///
/// - An auto-detected budget reads live `MemAvailable`, which those bytes are not
///   yet out of. Without this it would rise by 0.8 of them and the job would
///   commit that memory twice: with no pool, nothing else would notice.
/// - The NLC cache is already allocated, and raw must leave room for it: see
///   [`budget_beside_nlc`].
///
/// Nothing here reads the thread count at an explicit budget.
pub(crate) fn raw_budget(config_bytes: usize, plan: ferric_dft::xc_trait::XcMemoryPlan) -> usize {
    use ferric_core::memory::{
        auto_cgroup_ceiling_bytes, pool, resolve_budget_after, BudgetSource,
    };
    let explicit = (config_bytes != 0).then_some(config_bytes);
    let res = resolve_budget_after(explicit, plan.resident_if_alone);
    budget_beside_nlc(
        res.bytes,
        res.source == BudgetSource::AutoDetected,
        plan.nlc_bytes,
        lowest_ceiling(
            pool::global().map(|p| p.capacity_bytes()),
            ceiling_beside_ao(auto_cgroup_ceiling_bytes(), plan.resident_if_alone),
        ),
    )
}

/// A cgroup ceiling less the AO blocks XC plans to keep. Those are charged later
/// against a budget that has not seen the tensor, so they must be left room here.
fn ceiling_beside_ao(ceiling: Option<usize>, resident_if_alone: usize) -> Option<usize> {
    ceiling.map(|c| c.saturating_sub(resident_if_alone))
}

/// The tighter of two optional ceilings.
fn lowest_ceiling(a: Option<usize>, b: Option<usize>) -> Option<usize> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    }
}

/// `budget` less the NLC cache, counted once.
///
/// A fixed budget has not seen the cache, so it comes off, with or without a pool:
/// the tensor and the cache must fit together. A live auto budget has already fallen
/// with the allocated cache as far as `MemAvailable` goes, so subtracting it again
/// would send a tensor that fits to a slower tier. It only has to stay within what
/// `ceiling` leaves beside the cache. The ceiling is whatever does not move with
/// usage: the pool, sized before the cache existed (raw picks `InCore` from this
/// figure alone and refuses if the pool cannot pay, so otherwise a budget just above
/// the tensor is refused where a smaller one, which gets a soft tier, runs), and the
/// cgroup limit, which the cache counts toward.
fn budget_beside_nlc(budget: usize, live: bool, nlc: usize, ceiling: Option<usize>) -> usize {
    if nlc == 0 {
        return budget;
    }
    if !live {
        return budget.saturating_sub(nlc);
    }
    match ceiling {
        Some(ceiling) => budget.min(ceiling.saturating_sub(nlc)),
        None => budget,
    }
}

/// Say that the MINAO guess failed and the SCF starts from hcore, on stderr and in
/// the run log. Numerics are untouched, but hcore can take more iterations and can
/// reach a different energy at the bit level, so it must not pass silently. The
/// free-atom solves behind MINAO draw on the shared pool, so tight memory is one
/// way to get here.
fn warn_guess_fallback(solver: &str, err: &FerricError) {
    eprintln!(
        "[ferric] WARNING: {solver}: the MINAO guess failed, so the SCF starts from the hcore \
         guess. It may take more iterations. Cause: {err}"
    );
    if let Some(rl) = crate::runlog::log() {
        rl.note(
            "guess_fallback",
            serde_json::json!({ "solver": solver, "guess": "hcore", "reason": err.to_string() }),
        );
    }
}

/// One element's MINAO block that could not be built.
pub(crate) struct LostGuessBlock {
    pub z: i32,
    pub route: &'static str,
    pub reason: String,
}

/// What losing `block` does to the guess. With every element lost the density is
/// zero, so the first Fock matrix is hcore; otherwise the rest of MINAO remains.
fn guess_block_lost_text(block: &LostGuessBlock, all_lost: bool) -> String {
    let LostGuessBlock { z, route, reason } = block;
    let sym = ferric_core::elements::z_to_symbol(*z).unwrap_or("?");
    let outcome = if all_lost {
        "no element has a block, so the guess density is zero and the first Fock matrix is hcore"
    } else {
        "the SCF starts from a partial MINAO guess, not from hcore"
    };
    format!(
        "[ferric] WARNING: MINAO: {route} failed for {sym} (Z={z}), so its block of the guess \
         density is left zero and {outcome}. It may take more iterations. Cause: {reason}"
    )
}

/// Report the blocks MINAO lost out of `n_elements`, on stderr and in the run log.
/// Numerics are untouched. The free-atom solves draw on the shared pool, so tight
/// memory is one way to get here.
pub(crate) fn report_lost_guess_blocks(mut lost: Vec<LostGuessBlock>, n_elements: usize) {
    lost.sort_by_key(|b| b.z);
    let all_lost = lost.len() >= n_elements;
    for block in &lost {
        eprintln!("{}", guess_block_lost_text(block, all_lost));
        if let Some(rl) = crate::runlog::log() {
            rl.note(
                "guess_block_lost",
                serde_json::json!({
                    "z": block.z,
                    "route": block.route,
                    "reason": block.reason,
                    "all_elements_lost": all_lost,
                }),
            );
        }
    }
}

/// What a deferred XC evaluator means to hold; nothing without one.
pub(crate) fn xc_plan<X>(xc: Option<&X>) -> ferric_dft::xc_trait::XcMemoryPlan
where
    X: ferric_dft::xc_trait::XcStorage + ?Sized,
{
    xc.map_or_else(Default::default, |x| x.memory_plan())
}

/// XC takes its memory, after RI-J/K setup. Records what it decided.
pub(crate) fn commit_xc<X>(xc: Option<&mut X>) -> Result<(), FerricError>
where
    X: ferric_dft::xc_trait::XcStorage + ?Sized,
{
    if let Some(x) = xc {
        let done = x
            .commit_storage()
            .map_err(|e| FerricError::General(format!("XC grid storage: {e}")))?;
        note_xc_commit(&done);
    }
    Ok(())
}

/// The MINAO guess density, or `None` after saying it failed.
pub(crate) fn minao_or_report(
    solver: &str,
    mol: &Molecule,
    prep: &PreparedBasis,
) -> Option<Array2<f64>> {
    crate::guess::minao_projection_guess(mol, prep, prep.basis_set())
        .map_err(|e| warn_guess_fallback(solver, &e))
        .ok()
}

/// The UKS/ROKS XC evaluator for `config`, or `None` for plain UHF/ROHF. It takes
/// no memory but its NLC cache until `commit_xc`.
///
/// `[memory] budget_gb` is threaded through because this is the largest single
/// allocation in a DFT job. It used to call the unbudgeted `new_with_omega`, which
/// resolves from env/auto-detect and so silently discarded
/// `config.three_index_budget_bytes`: `budget_gb = 4` on a 64 GB box still sized the
/// grid cache against ~51 GB. 0 means unset, matching
/// `rhf::resolve_three_index_budget`.
pub(crate) fn uks_xc_contribution(
    mol: &Molecule,
    prep: &PreparedBasis,
    config: &RhfConfig,
) -> Result<Option<Box<dyn ferric_dft::xc_trait::UksXcContribution>>, FerricError> {
    let Some(name) = config.xc.as_deref() else {
        return Ok(None);
    };
    let main = config
        .dft_grid
        .clone()
        .unwrap_or_else(ferric_dft::grid::AtomicGridConfig::production);
    let nlc = config
        .nlc_grid
        .clone()
        .unwrap_or_else(ferric_dft::grid::AtomicGridConfig::default_nlc);
    let ks = ferric_dft::ks::KsXcUks::new_deferred(
        mol,
        prep.basis_set(),
        name,
        &main,
        &nlc,
        config.xc_omega,
        (config.three_index_budget_bytes != 0).then_some(config.three_index_budget_bytes),
    )
    .map_err(|e| e.into_ferric(&format!("KsXcUks init for {name}")))?;
    Ok(Some(Box::new(ks)))
}

/// Record what XC storage decided, so that an uncharged amount reaches the run
/// log and not only stderr.
fn note_xc_commit(done: &ferric_dft::xc_trait::XcCommit) {
    if let Some(rl) = crate::runlog::log() {
        rl.note(
            "xc_storage",
            serde_json::json!({
                "resident": done.resident,
                "uncharged_bytes": done.uncharged_bytes,
            }),
        );
    }
}

/// Build the shared [`ScfEnv`]. `k_mix` comes from the solver's XC contribution
/// (closed- and open-shell XC are different trait objects, so XC itself stays
/// with the solver). The hcore is [`crate::x2c::reference_hcore`]: X2C-1e
/// when `config.relativistic` is set, otherwise `hcore_ecp_with_external`,
/// which always folds the ECP projector in: for an all-electron basis `ecp_potential` returns
/// `None` at zero cost, so the result is byte-identical to the plain hcore —
/// all-electron RHF/UHF/ROHF results are unchanged. (Historically UHF/ROHF
/// used `hcore_with_external`, silently omitting V_ECP for ECP-carrying
/// bases such as aug-cc-pvdz-pp.)
pub(crate) fn prepare<'a>(
    ctx: &'a ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    config: &RhfConfig,
    k_mix: &ferric_dft::xc_trait::KMix,
    pre: Option<(Array2<f64>, Array2<f64>, f64)>,
    xc_plan: ferric_dft::xc_trait::XcMemoryPlan,
) -> Result<ScfEnv<'a>, FerricError> {
    // `pre` = caller-supplied (S, h, V_nn) — the periodic injection path
    // (`rhf::solve_rhf_injected`). `None` runs exactly the molecular build
    // below, so every molecular caller is byte-identical.
    let (s, h, vnn) = match pre {
        Some(_) if config.relativistic.is_some() => {
            return Err(FerricError::General(format!(
                "relativistic = \"{}\" is not available for a caller-supplied (periodic) \
                 core Hamiltonian: X2C-1e is built for a molecule's nuclear potential only",
                config.relativistic.map_or("", |r| r.as_str())
            )));
        }
        Some(injected) => injected,
        None => {
            let s = oneelectron::overlap(prep);
            let h = crate::x2c::reference_hcore(
                prep,
                mol,
                prep.basis_set(),
                config.external_potential.as_ref(),
                config.relativistic,
            )?;
            let vnn = mol.nuclear_repulsion()
                + config.external_potential.as_ref().map_or(0.0, |ext| {
                    ext.charge_nuclear_energy(mol) + ext.field_nuclear_energy(mol)
                });
            (s, h, vnn)
        }
    };
    let cosmo_cavity = match config.cosmo.as_ref() {
        Some(cfg) => Some(crate::cosmo::CosmoCavity::build(mol, cfg)?),
        None => None,
    };
    let pcm_ctx = config
        .pcm
        .as_ref()
        .map(|pcfg| ferric_pcm::PcmContext::new(mol, pcfg))
        .transpose()?;
    let polarizable_site_basis = match config.polarizable.as_ref() {
        Some(pol) if !pol.sites.is_empty() => {
            let sites: Vec<[f64; 4]> = pol
                .sites
                .iter()
                .map(|s| [s.x, s.y, s.z, pol.dipole_zeta])
                .collect();
            Some(ferric_integrals::site_basis::SiteBasis::new(&sites, 1)?)
        }
        _ => None,
    };
    let ooc_budget = raw_budget(config.three_index_budget_bytes, xc_plan);
    let (dfk_sr, dfk_lr) = if k_mix.omega > 0.0 {
        // RSH exchange is density-fitted ONLY: there is no conventional
        // four-centre erf/erfc exchange path. The explicit opt-out
        // `df_k_aux = Some("")` (see `solve_rhf`) cannot be honoured here, and
        // passing it on would fail inside `basis::bundled("")` with an
        // unrelated-looking basis error. Refuse it by name instead.
        if config.df_k_aux.as_deref() == Some("") {
            return Err(FerricError::General(format!(
                "density-fitted exchange cannot be turned off for a \
                 range-separated functional (omega = {}): ferric builds RSH \
                 exchange only via RI-K. Leave df_k_aux unset or name an aux \
                 basis; df_j_aux may still be turned off.",
                k_mix.omega
            )));
        }
        let (sr, lr) = crate::raw_handoff::with_metric_capture(config, || {
            crate::fock_assembly::build_rsh_dfk_pair(
                ctx,
                mol,
                prep,
                config.df_k_aux.as_deref(),
                k_mix.omega,
                ooc_budget,
                crate::df_k::RiKPrecision::resolve(config.ri_k_precision, prep.nbasis()),
            )
        })?;
        (Some(RefCell::new(sr)), Some(RefCell::new(lr)))
    } else {
        (None, None)
    };
    Ok(ScfEnv {
        s,
        h,
        vnn,
        ooc_budget,
        cosmo_cavity,
        pcm_ctx,
        polarizable_site_basis,
        dfk_sr,
        dfk_lr,
    })
}

/// The pieces of a [`crate::rhf::PeriodicInjection`] as the drivers use
/// them: `(S, h, V_nn)` for [`prepare`], the J/K builders the iteration
/// drives, and the injected XC. All `None` on the molecular path.
pub(crate) type SplitInjection<'a> = (
    Option<(Array2<f64>, Array2<f64>, f64)>,
    Option<(Box<dyn JBuilder + 'a>, Box<dyn KBuilder + 'a>)>,
    Option<Box<dyn crate::rhf::XcBuilder + 'a>>,
);

/// Split an injection into what [`prepare`] takes and what the loop drives.
pub(crate) fn split_injection(
    inj: Option<crate::rhf::PeriodicInjection<'_>>,
) -> SplitInjection<'_> {
    match inj {
        Some(crate::rhf::PeriodicInjection {
            s,
            h,
            vnn,
            j,
            k,
            xc,
        }) => (Some((s, h, vnn)), Some((j, k)), xc),
        None => (None, None, None),
    }
}

/// The exact-exchange mix of a solve. Injected (periodic) KS: a global hybrid
/// scales the injected K by its exact-exchange fraction, which must lie in
/// `[0, 1]` (`solver` names the entry point in the error); the validators
/// guarantee no molecular XC beside it. Molecular path: `molecular()`, the
/// XC contribution's mix (the default, pure HF, without one).
pub(crate) fn resolve_k_mix(
    inj_xc: Option<&dyn crate::rhf::XcBuilder>,
    molecular: impl FnOnce() -> KMix,
    solver: &str,
) -> Result<KMix, FerricError> {
    match inj_xc {
        Some(x) => {
            let a = x.exact_exchange_fraction();
            if !(a.is_finite() && (0.0..=1.0).contains(&a)) {
                return Err(FerricError::General(format!(
                    "{solver}: XcBuilder exact-exchange fraction must be in [0, 1], got {a}"
                )));
            }
            Ok(KMix {
                sr: a,
                lr: a,
                omega: 0.0,
            })
        }
        None => Ok(molecular()),
    }
}

/// [`prepare`], timed as `setup:prepare`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_timed<'a>(
    phases: &mut PhaseAccumulator,
    ctx: &'a ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    config: &RhfConfig,
    k_mix: &KMix,
    pre: Option<(Array2<f64>, Array2<f64>, f64)>,
    xc_plan: ferric_dft::xc_trait::XcMemoryPlan,
) -> Result<ScfEnv<'a>, FerricError> {
    let prepare_t = phases.begin("setup:prepare");
    let env = prepare(ctx, mol, prep, config, k_mix, pre, xc_plan)?;
    phases.stop(prepare_t);
    Ok(env)
}

/// The ONE SR/LR exchange response every Hessian matvec in a solve uses —
/// Newton steps, TRAH steps and the post-SCF stability eigensolve alike. It
/// borrows the SAME two fitters `subtract_rsh_exchange` assembles the
/// converged Fock from and reads the SAME `k_mix`, so the Fock and the
/// Hessian cannot disagree about the kernel (the #314 defect). `None` at
/// omega == 0, where the matvecs keep their untouched plain-Coulomb path.
pub(crate) fn rsh_response<'r>(
    env: &'r ScfEnv<'_>,
    k_mix: KMix,
) -> Option<crate::rsh_response::RshResponse<'r>> {
    match (env.dfk_sr.as_ref(), env.dfk_lr.as_ref()) {
        (Some(sr), Some(lr)) => Some(crate::rsh_response::RshResponse::new(
            sr,
            lr,
            k_mix.sr,
            k_mix.lr,
            k_mix.omega,
        )),
        _ => None,
    }
}

/// The density-fitted J/K setup of every driver: the RI-J / RI-K fitters on
/// `jk_route`'s aux bases (timed as `setup:df_fitters`; one shared
/// `PreparedBasis` when the two names agree, see `build_df_jk`), and the
/// record of the builders that actually produced the energy, so the analytic
/// gradient differentiates this energy instead of re-deriving the aux
/// resolution. The record is `None` when nothing is fitted: an all-exact
/// run's gradient path is untouched. `xc_k_mix` is the functional's mix, or
/// `None` without a molecular functional (it picks the RI-J precision site).
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_df_fitters<'a>(
    phases: &mut PhaseAccumulator,
    ctx: &'a ParallelContext,
    mol: &Molecule,
    op: ferric_integrals::operator::Operator,
    prep: &PreparedBasis,
    config: &RhfConfig,
    jk_route: &crate::fock_assembly::JkRoute,
    env: &ScfEnv<'_>,
    xc_k_mix: Option<KMix>,
) -> Result<DfFitters<'a>, FerricError> {
    let df_fitters_t = phases.begin("setup:df_fitters");
    crate::raw_handoff::clear_slot(config);
    let (df_j, df_k) = crate::raw_handoff::with_metric_capture(config, || {
        crate::fock_assembly::build_df_jk(
            ctx,
            mol,
            op,
            prep,
            jk_route.j_aux.as_deref(),
            jk_route.k_aux.as_deref(),
            env.ooc_budget,
            config.integral_thresh,
            config.resolved_ri_j_precision(crate::df_j::ri_j_site(prep.nbasis(), xc_k_mix)),
            crate::df_k::RiKPrecision::resolve(config.ri_k_precision, prep.nbasis()),
        )
    })?;
    phases.stop(df_fitters_t);
    let df_jk_route = jk_route
        .record(
            op,
            env.ooc_budget,
            df_j.as_ref()
                .map_or(crate::df_j::RiJPrecision::F64, |j| j.precision()),
        )
        .map(|r| {
            // The precision the K tensor(s) were ACTUALLY stored at: the ω = 0
            // DfK, else the SR fitter of a range-separated pair.
            let stored = df_k
                .as_ref()
                .map(|k| k.precision())
                .or_else(|| env.dfk_sr.as_ref().map(|k| k.borrow().precision()))
                .unwrap_or(crate::df_k::RiKPrecision::F64);
            r.with_ri_k_precision(stored)
        });
    Ok((df_j, df_k, df_jk_route))
}

/// What [`build_df_fitters`] returns: RI-J, the ω = 0 RI-K, and the route
/// record.
pub(crate) type DfFitters<'a> = (
    Option<crate::split_rij::RiJ<'a>>,
    Option<DfK<'a>>,
    Option<crate::result::DfJkRoute>,
);

/// Mixed `dfj-pack`/`dfk-occ` builds only while the density moves, and an
/// f64 exit (see `crate::jk_ladder`); inert unless one of them is requested.
pub(crate) fn jk_ladder(df_j: bool, df_k: bool, env: &ScfEnv<'_>) -> crate::jk_ladder::JkLadder {
    crate::jk_ladder::JkLadder::new([df_j, df_k, env.dfk_sr.is_some(), env.dfk_lr.is_some()])
}

/// Periodic full-rebuild guard against f64 drift accumulating across many
/// incremental Fock updates (PySCF re-triggers via `direct_scf_tol`; Psi4 hard
/// resets every `INCFOCK_FULL_FOCK_EVERY = 5`). 8 is a compromise between
/// Psi4's 5 and amortizing the full-build cost, verified to hold the
/// incremental-vs-full energy well within the direct-path noise floor. A full
/// rebuild fires on iteration 1, whenever `iter % INCREMENTAL_FULL_REBUILD_EVERY
/// == 1`, and whenever no last-Fock density is held. One value for all three
/// drivers.
pub(crate) const INCREMENTAL_FULL_REBUILD_EVERY: usize = 8;

/// The J/K builders of an open-shell (UHF or ROHF) solve and the per-spin
/// buffers they fill: J[D_α + D_β], K[D_α] and K[D_β].
///
/// ROHF's Fock build is structurally the same three-matrix problem as UHF's;
/// the ROHF-specific work (the Roothaan coupling) acts on the ASSEMBLED
/// `F_α`/`F_β` downstream of [`Self::build_fock`] and is not touched here.
///
/// K is built per spin:
/// - range-separated (ω > 0): `K_σ = c_SR·K_SR[D_σ] + c_LR·K_LR[D_σ]` from
///   the SR/LR DfK pair [`prepare`] built;
/// - plain hybrid / pure HF (ω = 0): `K_σ = c_K·K[D_σ]` from the combined
///   DirectJK, a pluggable builder, DF-K, or DirectK;
/// - pure DFT (`c_K = 0`): no K.
///
/// J is the combined DirectJK, DF-J when an aux basis is named (any ω), or
/// DirectJ. Every builder lives for the whole solve: each lazily builds a
/// per-thread libint2 EnginePool on first use (ctors serialized behind a
/// global mutex), so a loop-local builder would pay that construction every
/// iteration.
pub(crate) struct OpenShellJk<'a> {
    /// The injected (periodic) J/K pair; pre-empts every molecular builder.
    pub(crate) inj_jk: Option<(Box<dyn JBuilder + 'a>, Box<dyn KBuilder + 'a>)>,
    pub(crate) df_j: Option<crate::split_rij::RiJ<'a>>,
    pub(crate) df_k: Option<DfK<'a>>,
    pub(crate) direct_jk: Option<crate::direct_jk::DirectJK<'a>>,
    pub(crate) direct_j: Option<crate::direct_j::DirectJ<'a>>,
    pub(crate) direct_k: Option<crate::direct_k::DirectK<'a>>,
    /// A pluggable exchange builder ("link" / "cosx"); ONE instance serves
    /// BOTH spins (see [`crate::fock_assembly::build_open_shell_pluggable_k`]).
    pub(crate) pluggable_k: Option<Box<dyn KBuilder + 'a>>,
    /// The COSX grid schedule (`None` without one).
    pub(crate) cosx_sched: Option<crate::cosx_schedule::ScheduleRun>,
    /// The in-core ERI store serving the combined DirectJK, if any.
    _incore: Option<std::sync::Arc<crate::incore_jk::IncoreEri>>,
    /// See [`Self::new`].
    incremental_direct: bool,
    /// The (α, β) densities that produced the CURRENT contents of the
    /// buffers. `None` until the first full build; reset to force one.
    pub(crate) d_last_fock: Option<(Array2<f64>, Array2<f64>)>,
    pub(crate) j_buf: Array2<f64>,
    pub(crate) k_a_buf: Array2<f64>,
    pub(crate) k_b_buf: Array2<f64>,
    pub(crate) total_quartets: usize,
}

/// What an open-shell driver hands [`OpenShellJk::new`].
pub(crate) struct OpenShellJkSetup<'a> {
    pub(crate) inj_jk: Option<(Box<dyn JBuilder + 'a>, Box<dyn KBuilder + 'a>)>,
    pub(crate) df_j: Option<crate::split_rij::RiJ<'a>>,
    pub(crate) df_k: Option<DfK<'a>>,
    pub(crate) started_k: crate::cosx_schedule::StartedK<'a>,
    /// Exact exchange reaches F (`c_K != 0` or ω > 0).
    pub(crate) need_k: bool,
    pub(crate) omega: f64,
}

impl<'a> OpenShellJk<'a> {
    /// Build the direct builders the route needs.
    ///
    /// Combined open-shell direct J+K (single quartet pass): when BOTH J and
    /// K come from the direct (non-DF) path with an ordinary ω = 0 kernel,
    /// one `DirectJK::build_uhf` pass produces J[D_α+D_β], K[D_α] and K[D_β]
    /// together, replacing three separate quartet traversals that each
    /// re-evaluated the same integrals and screened on the loose global
    /// max|D| scalar instead of the tight six-pairwise shell table. Gated to
    /// exactly the case where all three matrices are direct and share the
    /// Coulomb operator, with no pluggable K and no injection; every other
    /// combination keeps the per-matrix builders. `FERRIC_SCF_COMBINED_JK=0`
    /// (or `off`/`false`) restores the three-pass build for A/B checks.
    ///
    /// Incremental (differential) Fock build, combined DirectJK path only:
    /// the same scheme as the closed-shell path (see `rhf::RhfJk::new`), for
    /// the same reason restricted to the direct path (the DF/RI paths carry a
    /// naux-dependent fitted-Fock noise floor that ΔD accumulation would
    /// compound). It screens on |ΔD_α| + |ΔD_β| (see
    /// `DirectJK::build_uhf_incremental`) and is opt-IN, unlike the
    /// closed-shell path: `direct_jk::open_shell_incremental_enabled` holds
    /// the measurements (correct, ~1.00-1.05x). A pluggable K is never fed a
    /// ΔD: the combined DirectJK requires `pluggable_k.is_none()`, so there
    /// is no DirectJK (and no incremental path) whenever one is active. LinK
    /// would otherwise build its density-pair list from a difference density
    /// and COSX's overlap fit is defined against a true density.
    pub(crate) fn new(
        (ctx, prep, bounds, config): (
            &'a ParallelContext,
            &'a PreparedBasis,
            &'a crate::screening::SchwarzBounds,
            &RhfConfig,
        ),
        ooc_budget: usize,
        setup: OpenShellJkSetup<'a>,
    ) -> Result<Self, FerricError> {
        let OpenShellJkSetup {
            inj_jk,
            df_j,
            df_k,
            started_k: (cosx_sched, pluggable_k),
            need_k,
            omega,
        } = setup;
        let n = prep.nbasis();
        let injected = inj_jk.is_some();
        let combined_direct_jk = df_j.is_none()
            && df_k.is_none()
            && need_k
            && omega == 0.0
            && pluggable_k.is_none()
            && !injected
            && crate::direct_jk::combined_open_shell_jk_enabled();
        // `[scf] eri`: the in-core ERI store serves the combined DirectJK.
        let incore = crate::incore_jk::attach_for_scf(
            ctx,
            prep,
            bounds,
            config,
            ooc_budget,
            (combined_direct_jk, omega),
        )?;
        let thresh = config.integral_thresh;
        let direct_jk = combined_direct_jk
            .then(|| crate::direct_jk::DirectJK::new(ctx, prep, bounds, thresh, ooc_budget));
        // No molecular direct builders on the injected path (J/K come from
        // `inj_jk`).
        let direct_j = (df_j.is_none() && !combined_direct_jk && !injected)
            .then(|| crate::direct_j::DirectJ::new(ctx, prep, bounds, thresh, ooc_budget));
        let direct_k = (need_k
            && omega == 0.0
            && df_k.is_none()
            && !combined_direct_jk
            && pluggable_k.is_none()
            && !injected)
            .then(|| crate::direct_k::DirectK::new(ctx, prep, bounds, thresh, ooc_budget));
        let incremental_direct = direct_jk.is_some()
            && incore.is_none()
            && crate::direct_jk::open_shell_incremental_enabled();
        Ok(OpenShellJk {
            inj_jk,
            df_j,
            df_k,
            direct_jk,
            direct_j,
            direct_k,
            pluggable_k,
            cosx_sched,
            _incore: incore,
            incremental_direct,
            d_last_fock: None,
            j_buf: Array2::<f64>::zeros((n, n)),
            k_a_buf: Array2::<f64>::zeros((n, n)),
            k_b_buf: Array2::<f64>::zeros((n, n)),
            total_quartets: 0,
        })
    }

    /// Zero the buffers unless this iteration accumulates onto them: on an
    /// incremental iteration they KEEP the previous iteration's J/K_σ (the
    /// delta is added onto them). Returns whether it is incremental.
    pub(crate) fn begin_iteration(&mut self, iter: usize) -> bool {
        let direct_full_rebuild = self.incremental_direct
            && (self.d_last_fock.is_none() || iter % INCREMENTAL_FULL_REBUILD_EVERY == 1);
        let direct_incremental = self.incremental_direct && !direct_full_rebuild;
        if !direct_incremental {
            self.j_buf.fill(0.0);
            self.k_a_buf.fill(0.0);
            self.k_b_buf.fill(0.0);
        }
        direct_incremental
    }

    /// The Fock build stage: J (and, on the combined path, K_α/K_β) into the
    /// buffers, then `F_σ = H + J − K_σ_total` per spin, assembled in place:
    /// no K_σ_total clones for HF/plain hybrids and no zeros allocations for
    /// pure DFT. `F_σ += (−c)·K` is bit-identical to `F_σ −= c·K` (the sign
    /// flip is exact); the range-separated branch keeps the explicit SR/LR
    /// combination for the same reason. `d_occ` is each spin's bare C_occ
    /// for the DF-K half-transform (`None` takes the density factor); `c_k`
    /// the ω = 0 exchange coefficient.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn build_fock(
        &mut self,
        direct_incremental: bool,
        (d_a, d_b, d_total): (&Array2<f64>, &Array2<f64>, &Array2<f64>),
        (d_occ_a, d_occ_b): (Option<&Array2<f64>>, Option<&Array2<f64>>),
        env: &ScfEnv<'_>,
        (k_mix, need_k, c_k): (KMix, bool, f64),
        jk_ladder: &crate::jk_ladder::JkLadder,
        phases: &mut PhaseAccumulator,
    ) -> Result<(Array2<f64>, Array2<f64>), FerricError> {
        let jk_t = phases.begin("jk_build");
        let jk_scope = jk_ladder.begin_build(&[d_a, d_b]);
        self.build_j(direct_incremental, (d_a, d_b, d_total))?;
        let h = &env.h;
        let mut f_a: Array2<f64> = h + &self.j_buf;
        let mut f_b: Array2<f64> = h + &self.j_buf;
        if k_mix.omega > 0.0 {
            let mut dfk_sr = env
                .dfk_sr
                .as_ref()
                .expect("dfk_sr built when omega>0")
                .borrow_mut();
            let mut dfk_lr = env
                .dfk_lr
                .as_ref()
                .expect("dfk_lr built when omega>0")
                .borrow_mut();
            let (dfk_sr, dfk_lr) = (&mut *dfk_sr, &mut *dfk_lr);
            crate::fock_assembly::subtract_rsh_exchange(
                dfk_sr, dfk_lr, d_a, d_occ_a, 1.0, &mut f_a, k_mix.sr, k_mix.lr, 1.0,
            )?;
            crate::fock_assembly::subtract_rsh_exchange(
                dfk_sr, dfk_lr, d_b, d_occ_b, 1.0, &mut f_b, k_mix.sr, k_mix.lr, 1.0,
            )?;
        } else if need_k {
            self.build_k((d_a, d_b), (d_occ_a, d_occ_b))?;
            f_a.scaled_add(-c_k, &self.k_a_buf);
            f_b.scaled_add(-c_k, &self.k_b_buf);
        }
        drop(jk_scope);
        phases.stop(jk_t);
        Ok((f_a, f_b))
    }

    /// J from the total density: the injected J, the combined single-pass
    /// J + K_α + K_β (incremental on ΔD_σ when this iteration is), DF-J, or
    /// DirectJ. An injected (periodic) J pre-empts all of them (the injected
    /// validators guarantee no DF/pluggable builder beside it).
    fn build_j(
        &mut self,
        direct_incremental: bool,
        (d_a, d_b, d_total): (&Array2<f64>, &Array2<f64>, &Array2<f64>),
    ) -> Result<(), FerricError> {
        if let Some((ij, _)) = self.inj_jk.as_mut() {
            self.total_quartets += ij.build(d_total, &mut self.j_buf)?;
        } else if let Some(djk) = self.direct_jk.as_mut() {
            let (j, ka, kb) = (&mut self.j_buf, &mut self.k_a_buf, &mut self.k_b_buf);
            if direct_incremental {
                let (da_prev, db_prev) = self
                    .d_last_fock
                    .as_ref()
                    .expect("d_last_fock set on full rebuild");
                let delta_a = d_a - da_prev;
                let delta_b = d_b - db_prev;
                let delta_total = &delta_a + &delta_b;
                self.total_quartets +=
                    djk.build_uhf_incremental(&delta_total, &delta_a, &delta_b, j, ka, kb)?;
            } else {
                self.total_quartets += djk.build_uhf(d_total, d_a, d_b, j, ka, kb)?;
            }
            self.d_last_fock = Some((d_a.clone(), d_b.clone()));
        } else if let Some(dfj) = self.df_j.as_mut() {
            dfj.build(d_total, &mut self.j_buf)?;
        } else {
            let dj = self.direct_j.as_mut().expect("DirectJ built before loop");
            self.total_quartets += dj.build(d_total, &mut self.j_buf)?;
        }
        Ok(())
    }

    /// K_α/K_β at ω = 0: the injected K, the combined pass (already filled by
    /// [`Self::build_j`]), the pluggable builder, DF-K (the C_occ
    /// half-transform per spin when available, D_σ = C_occ,σ·C_occ,σᵀ; else
    /// the density factor), or DirectK. The injected and pluggable builders
    /// take a per-spin `update_density(D_σ)` + `build(D_σ)` from one instance
    /// (see [`crate::fock_assembly::build_open_shell_pluggable_k`]); an
    /// injected Madelung term (if any) is linear in D_σ, so no per-spin
    /// factor.
    fn build_k(
        &mut self,
        (d_a, d_b): (&Array2<f64>, &Array2<f64>),
        (d_occ_a, d_occ_b): (Option<&Array2<f64>>, Option<&Array2<f64>>),
    ) -> Result<(), FerricError> {
        let (ka, kb) = (&mut self.k_a_buf, &mut self.k_b_buf);
        if let Some((_, ik)) = self.inj_jk.as_mut() {
            self.total_quartets +=
                crate::fock_assembly::build_open_shell_pluggable_k(ik.as_mut(), d_a, d_b, ka, kb)?;
        } else if self.direct_jk.is_some() {
            // K_α/K_β already filled by the combined single-pass build.
        } else if let Some(pk) = self.pluggable_k.as_mut() {
            self.total_quartets +=
                crate::fock_assembly::build_open_shell_pluggable_k(pk.as_mut(), d_a, d_b, ka, kb)?;
        } else if let Some(dfk) = self.df_k.as_mut() {
            match d_occ_a {
                Some(c) => dfk.build_from_occ(c, ka)?,
                None => dfk.build_from_density_factor(d_a, ka)?,
            };
            match d_occ_b {
                Some(c) => dfk.build_from_occ(c, kb)?,
                None => dfk.build_from_density_factor(d_b, kb)?,
            };
        } else {
            let dk = self.direct_k.as_mut().expect("DirectK built before loop");
            use crate::direct_k::DirectK;
            self.total_quartets += <DirectK as KBuilder>::build(dk, d_a, ka)?;
            self.total_quartets += <DirectK as KBuilder>::build(dk, d_b, kb)?;
        }
        Ok(())
    }

    /// The Newton/TRAH/stability matvec responses of these builders
    /// (`uhf::scf_responses`).
    pub(crate) fn responses(&mut self) -> crate::uhf_newton::ScfResponses<'_> {
        crate::uhf::scf_responses(
            self.df_j.as_mut(),
            self.df_k.as_mut(),
            self.direct_j.as_mut(),
            self.direct_k.as_mut(),
        )
    }
}

/// The XC of one open-shell solve: the molecular UKS contribution or an
/// injected (periodic) polarized one (never both).
pub(crate) struct OpenShellXc<'a> {
    pub(crate) xc_contrib: Option<Box<dyn ferric_dft::xc_trait::UksXcContribution>>,
    pub(crate) inj_xc: Option<Box<dyn crate::rhf::XcBuilder + 'a>>,
}

impl OpenShellXc<'_> {
    /// Kohn-Sham: a functional is active (the SCF's own, or an injected one).
    pub(crate) fn active(&self) -> [bool; 2] {
        [self.xc_contrib.is_some(), self.inj_xc.is_some()]
    }
}

/// The divergence / stall early exit of the open-shell drivers (the shared
/// [`ScfMonitor`] tests; both are no-ops at the `None` defaults): an error
/// carrying the last energy.
pub(crate) fn bail_if_diverged_or_stalled(
    mon: &mut ScfMonitor,
    (energy, err_max, iter): (f64, f64, usize),
    config: &RhfConfig,
) -> Result<(), FerricError> {
    if mon.diverging(energy, config.divergence_tol) || mon.stalled(err_max, config.stall_window) {
        return Err(FerricError::ScfConvergence {
            iterations: iter,
            last_energy: mon.prev_e,
        });
    }
    Ok(())
}

/// The trust-region augmented-Hessian (TRAH) bookkeeping a driver carries
/// across iterations (opt-in; inert when `config.trah_trigger` is `None`).
///
/// `state` carries the radius and the pending ρ assessment across SCF
/// iterations (see `crate::trah::TrahState` for why ρ must span two).
/// `undo` holds the orbitals the pending step departed FROM (`U` is the
/// driver's undo payload), so a rejected step can be exactly undone. Both
/// stay `None` when `config.trah_trigger` is `None`, which is what makes the
/// disabled path allocation-free and bit-identical.
pub(crate) struct TrahLoop<U> {
    pub(crate) state: Option<crate::trah::TrahState>,
    pub(crate) undo: Option<U>,
    /// Set on the iteration a TRAH step is rejected; consumed at the next
    /// iteration's arming gate (`crate::trah::settle_engaged_latch`).
    last_rejected: bool,
    /// TRAH keeps control from its first step until it declines a step as
    /// worthless or its radius collapses -- see the drivers' arming gates.
    pub(crate) engaged: bool,
}

/// What [`TrahLoop::arm`] decided for this iteration.
pub(crate) struct TrahArm {
    /// A TRAH step runs this iteration.
    pub(crate) runs: bool,
    /// The previous iteration rejected a step (see `settle_engaged_latch`).
    pub(crate) after_rejection: bool,
    /// The trust radius has collapsed.
    pub(crate) collapsed: bool,
}

impl<U> TrahLoop<U> {
    pub(crate) fn new(config: &RhfConfig) -> Self {
        TrahLoop {
            state: config
                .trah_trigger
                .map(|_| crate::trah::TrahState::new(config.trah)),
            undo: None,
            last_rejected: false,
            engaged: false,
        }
    }

    /// A TRAH step runs only if TRAH is `armed` AND the radius has not
    /// collapsed. Both conditions are computed ONCE here, because the
    /// "discard a stale prediction" rule must key off exactly the same
    /// condition as "take a step": if the radius collapses while a
    /// prediction is pending, DIIS resumes, and a later re-arm would
    /// otherwise score that stale prediction against an energy produced by
    /// DIIS steps in between — a ρ built from two unrelated points.
    pub(crate) fn arm(&mut self, armed: bool) -> TrahArm {
        let collapsed = self.state.as_ref().is_some_and(|s| s.collapsed());
        let runs = armed && !collapsed;
        let after_rejection =
            crate::trah::settle_engaged_latch(&mut self.engaged, &mut self.last_rejected, runs);
        if let Some(st) = self.state.as_mut() {
            if !runs {
                if st.has_pending() {
                    crate::trah::note_trah_prediction_discarded();
                }
                st.clear_pending();
                self.undo = None;
            }
        }
        TrahArm {
            runs,
            after_rejection,
            collapsed,
        }
    }

    /// Phase 1: score the pending step against `energy`, the energy at the
    /// density that step produced. `on_rho` receives ρ when a verdict was
    /// reached.
    ///
    /// `assess` returns None when NOTHING was pending — the first armed
    /// iteration, and the iteration right after a rejection (which consumed
    /// its pending step and skipped Phase 2). That is not an acceptance:
    /// treating it as one and then printing the PERSISTENT `last_rho` made
    /// the trace show a frozen ρ repeated across a reject/"accept"
    /// alternation that was not happening. Keep the Option so "no verdict"
    /// stays distinguishable from "accepted".
    pub(crate) fn assess(
        &mut self,
        energy: f64,
        iter: usize,
        on_rho: impl FnOnce(f64),
    ) -> Option<crate::trah::TrahVerdict> {
        let verdict = self.state.as_mut().and_then(|st| st.assess(energy));
        let rho = self.state.as_ref().and_then(|s| s.last_rho);
        if let (Some(_), Some(rho)) = (verdict, rho) {
            on_rho(rho);
        }
        if crate::rhf::scf_trace() {
            if let (Some(v), Some(rho)) = (verdict, rho) {
                eprintln!(
                    "TRAH iter={iter}: rho={rho:.6} verdict={v:?} \
                     Delta={:.3e} acc={} rej={}",
                    self.state.as_ref().map(|s| s.radius()).unwrap_or(0.0),
                    self.state.as_ref().map(|s| s.accepted).unwrap_or(0),
                    self.state.as_ref().map(|s| s.rejected).unwrap_or(0),
                );
            }
        }
        verdict
    }

    /// A rejected step: count it, latch the rejection, and hand back the
    /// orbitals to restore. `engaged` is already set: the rejected step set
    /// it when it was taken, and only a hand-off or a decline (both of which
    /// clear the pending prediction) releases it. So the restored point
    /// stays armed whatever its err_max.
    pub(crate) fn reject(&mut self, iter: usize) -> Option<U> {
        crate::trah::note_trah_rejection();
        self.last_rejected = true;
        let undo = self.undo.take();
        if undo.is_some() && crate::rhf::scf_trace() {
            eprintln!(
                "TRAH iter={iter}: REJECTED (ρ={:.3e}), restoring orbitals, Δ→{:.3e}",
                self.state.as_ref().and_then(|s| s.last_rho).unwrap_or(0.0),
                self.state.as_ref().map(|s| s.radius()).unwrap_or(0.0)
            );
        }
        undo
    }
}

/// The per-iteration records of every driver, from values the loop already
/// computed (nothing is computed for them and nothing is read back, so the
/// energy is bit-identical with both on or off, `tests/runlog_bit_identity.rs`).
///
/// - The machine-readable run-log record (`crate::runlog`), streamed and
///   flushed NOW: a run killed at iteration 90 of 100 must leave the first
///   90 on disk. `label` names the method ("rhf", "uks", ...); `grad_rms` is
///   `None` where a driver forms only commutator maxima, never an RMS
///   (reporting a fabricated 0 would be worse than "not measured here").
/// - Live progress for a user watching a long-running job (opt-in via
///   `config.verbose`: CLI `--verbose`/`-v`, or TOML `[scf] verbose = true`),
///   printed to STDOUT as `<prefix> iter=...` (normal-operation progress,
///   not a warning), unlike the `FERRIC_SCF_TRACE` debug channel (stderr,
///   separately gated).
///
/// Both are rank-0-only under MPI, so ranks > 0 never emit duplicate lines
/// or interleave duplicate records into one file.
#[allow(clippy::too_many_arguments)]
pub(crate) fn log_scf_iter(
    ctx: &ParallelContext,
    config: &RhfConfig,
    (label, prefix): (&str, &str),
    iter: usize,
    (energy, de): (f64, f64),
    mon: &ScfMonitor,
    err_max: f64,
    grad_rms: Option<f64>,
) {
    if !ctx.is_root() {
        return;
    }
    if let Some(rl) = crate::runlog::log() {
        rl.scf_iter(
            label,
            crate::runlog::current_rung(),
            iter,
            energy,
            de,
            mon.dp_rms,
            mon.dp_max,
            err_max,
            grad_rms,
        );
    }
    if config.verbose {
        println!(
            "{prefix} iter={iter:4}  E={energy:.10}  dE={de:.3e}  dp_rms={:.3e}  err_max={err_max:.3e}",
            mon.dp_rms
        );
    }
}

/// Whether `config` gets the automatic meta-GGA level shift: a meta-GGA
/// functional, no level shift written, and the auto shift not switched off
/// (`meta_gga_auto_level_shift = false` is how an explicit `level_shift = 0`
/// is told apart from an unset one).
fn auto_level_shift_applies(config: &RhfConfig) -> bool {
    config.meta_gga_auto_level_shift
        && config.level_shift == 0.0
        && crate::rohf::xc_is_metagga(config.xc.as_deref())
}

/// The default virtual-block level shift: the user's value, or 0.5 for
/// meta-GGA when the user set none (meta-GGA SCF limit-cycles under plain
/// DIIS — see the solve_rhf comment where this convention originated).
pub(crate) fn effective_level_shift(config: &RhfConfig) -> f64 {
    if auto_level_shift_applies(config) {
        0.5
    } else {
        config.level_shift
    }
}

// The virtual-block level shift is ramped off with `e = max|DIIS error|` so the
// converged Fock matrix is the unshifted one and the converged energy does
// not depend on the ramp. There are two ramps, each defined once here, because
// neither is best for both families (measured 2026-11-19, release,
// single-thread, energies equal to 1e-10 Ha in every case; book 8.2):
//
// * closed shell, linear: water RHF/cc-pVDZ with shift 2.0 converges in 11
//   iterations with it and 16 with the rational ramp; it is never slower on
//   the shifted RHF/RKS cases measured (water SCAN 10 vs 11, CH4 r2SCAN 9 vs
//   11, COSe RHF shift 0.5 16 vs 17).
// * open shell, rational: the k-point ROKS PBE0 trap
//   (`kroks_hybrid_default_level_shift_converges_the_roothaan_trap`)
//   converges in 34 iterations with it and does NOT converge in 200 with the
//   linear ramp (max error parked at 7e-3, where linear keeps 7% of the shift
//   and rational 87%); two of the six Gamma ROKS PBE0 perturbed starts take
//   314 and 229 iterations with the linear ramp against 177 and 172. On easy
//   molecular open shells the linear ramp would save 0-7 iterations.

/// `max|e|` at and above which the closed-shell ramp applies the full shift.
const CLOSED_SHELL_RAMP_ERR: f64 = 0.1;
/// `max|e|` at which the open-shell ramp applies half the shift.
const OPEN_SHELL_RAMP_ERR: f64 = 1e-3;

/// RHF/RKS level shift in force at DIIS error `err_max`:
/// `shift · min(err_max / 0.1, 1)` (see the comment above for why it differs
/// from [`open_shell_level_shift_ramp`]).
pub fn closed_shell_level_shift_ramp(shift: f64, err_max: f64) -> f64 {
    shift * (err_max / CLOSED_SHELL_RAMP_ERR).min(1.0)
}

/// UHF/UKS, ROHF/ROKS and periodic (Gamma and k-point) ROKS level shift in
/// force at DIIS error `err_max`: `shift · err_max / (err_max + 1e-3)` (see
/// the comment above for why it differs from
/// [`closed_shell_level_shift_ramp`]).
pub fn open_shell_level_shift_ramp(shift: f64, err_max: f64) -> f64 {
    shift * err_max / (err_max + OPEN_SHELL_RAMP_ERR)
}

/// ΔP signals: (rms, max) of `d_new − d_old` — the primary convergence signal
/// every variant gates on (see `rhf::scf_converged`).
///
/// Zero-alloc: no `diff` temp. `d_new`/`d_old` are always standard (C-
/// contiguous) layout here — every call site builds its density via `.dot()`
/// on operands with row-stride > 1 (nocc > 1 in every SCF this runs on; the
/// `.dot()` column-major edge case only bites at nocc/nvir == 1, see
/// `ndarray-dot-forder-when-both-stride1` — not this density shape), so
/// `as_slice()` never fails and its element order is exactly the order
/// `(d_new - d_old).iter()` would have visited. `sum::<f64>()` on a plain
/// iterator is `std`'s left-to-right fold (`0.0 + a[0] + a[1] + ...`), so
/// summing `(a[i]-b[i])²` via a single left-to-right `fold` over the zipped
/// slices reproduces that exact association — bit-identical to the old
/// `(d_new - d_old).iter().map(|v| v*v).sum()`.
pub(crate) fn density_change(d_new: &Array2<f64>, d_old: &Array2<f64>) -> (f64, f64) {
    match (d_new.as_slice(), d_old.as_slice()) {
        (Some(a), Some(b)) => {
            let (sum_sq, max) =
                a.iter()
                    .zip(b.iter())
                    .fold((0.0f64, 0.0f64), |(sum_sq, max), (&x, &y)| {
                        let diff = x - y;
                        (sum_sq + diff * diff, f64::max(max, diff.abs()))
                    });
            let n2 = (a.len() as f64).max(1.0);
            ((sum_sq / n2).sqrt(), max)
        }
        // Defensive fallback for a non-standard-layout input (never hit at
        // current call sites, but correctness over performance if one ever
        // is): identical formula via the temp, matching the pre-existing
        // ndarray element order for a non-contiguous array.
        _ => {
            let diff = d_new - d_old;
            let n2 = (diff.len() as f64).max(1.0);
            let rms = (diff.iter().map(|v| v * v).sum::<f64>() / n2).sqrt();
            let max = diff.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
            (rms, max)
        }
    }
}

/// Observational-only warning for the DIIS-history memory footprint.
///
/// The DIIS drivers retain up to `diis_size` entries, each holding some number
/// of `n × n` `f64` matrices — see [`crate::diis::diis_history_bytes`], which
/// lives next to the buffers it counts.
///
/// # Why this used to be wrong
///
/// The projection was hardcoded as `diis_size × 2 × n² × 8` for every variant,
/// justified by the argument that "UHF's coupled driver keeps separate α/β
/// histories but the *combined* error vector is still one logical DIIS slot per
/// iteration, so the same projection is used uniformly across all three
/// variants."
///
/// That reasoning counts the wrong thing. What consumes memory is the number of
/// `n × n` **buffers**, not the number of logical slots: `step_pair`
/// (`diis.rs`) pushes `fock_hist_b` and `err_hist_b` in addition to
/// `fock_hist`/`err_hist`, so UHF holds **four** matrices per slot, not two.
/// ADIIS/EDIIS runs hold an [`crate::diis::EnergyDiis`]'s `fock_hist` +
/// `dens_hist` *alongside* the Pulay `Diis`, adding two more. The old figure
/// was therefore 2× low for UHF and up to 3× low for RHF with ADIIS/EDIIS
/// enabled — and it was called from RHF only, so UHF, the variant it
/// under-counted most, projected nothing at all.
///
/// At `nbf=2000, diis_size=8` the single-spin Pulay figure is ~0.5 GB, big
/// enough to matter against a small `ooc_budget` but usually negligible against
/// the 3-index tensor it sits alongside; the UHF+EnergyDiis figure is ~1.5 GB.
///
/// Emits ONE stderr warning when the projection exceeds `ooc_budget / 4` —
/// mirroring `ferric_core::memory::warn_if_rss_over`'s philosophy: purely
/// observational, NEVER an error. A working job must never start failing
/// because of this check; it only ever prints a heads-up line.
pub(crate) fn warn_if_diis_history_large(
    label: &str,
    n: usize,
    diis_size: usize,
    shape: crate::diis::DiisHistoryShape,
    energy_diis: bool,
    ooc_budget: usize,
) {
    let projected = crate::diis::diis_history_bytes(n, diis_size, shape, energy_diis);
    let threshold =
        ferric_core::memory::transient_share(ooc_budget, ferric_core::memory::Share::Quarter);
    if projected > threshold {
        eprintln!(
            "ferric WARNING [{label}]: projected DIIS history {:.2} GB (diis_size={diis_size}, \
             nbf={n}) exceeds 25% of the {:.2} GB memory budget (observability only — DIIS will \
             still run; if this recurs, lower [scf] diis_size or raise [memory] budget_gb / \
             FERRIC_MEM_BUDGET_GB)",
            projected as f64 / 1e9,
            ooc_budget as f64 / 1e9,
        );
    }
}

/// Stage-seam RSS safety net (mirrors the `ferric-rpa` stage-seam convention:
/// `ferric_core::memory::warn_if_rss_over(label, ooc_budget, 1.1)`).
///
/// Purely observational — never a hard error — this just names the SCF
/// variant and stage in the warning line so a recurring over-budget RSS is
/// traceable to a specific solver/point in the run. `variant` is e.g. "RHF",
/// "UHF", "ROHF"; `stage` is e.g. "setup" or "converged".
///
/// `ri_k_resident_bytes`: the in-core size of the ω = 0 RI-K (DF-K) tensor
/// when the run holds one. When the warning fires with a non-zero size, it
/// is followed by [`ri_k_memory_remedy`], since that tensor is a whole second
/// three-index allocation beside RI-J's for a hybrid functional and
/// `k_builder` (the knob a reader reaches for) does nothing while it is live.
pub(crate) fn warn_if_rss_over_at_stage(
    variant: &str,
    stage: &str,
    ooc_budget: usize,
    ri_k_resident_bytes: Option<usize>,
) {
    let fired =
        ferric_core::memory::warn_if_rss_over(&format!("{variant} {stage}"), ooc_budget, 1.1);
    note_ri_k_remedy(fired, ri_k_resident_bytes);
}

/// Prints [`ri_k_memory_remedy`] after an over-budget warning that `fired`
/// while the run holds a non-empty RI-K tensor.
fn note_ri_k_remedy(fired: bool, ri_k_resident_bytes: Option<usize>) {
    if let (true, Some(bytes)) = (fired, ri_k_resident_bytes.filter(|&b| b > 0)) {
        eprintln!("{}", ri_k_memory_remedy(bytes));
    }
}

/// The memory remedy for a run holding an ω = 0 RI-K tensor of `bytes`:
/// `df_k_aux = ""` selects exact four-centre K, which holds no three-index
/// tensor for exchange (and is the reference RI-K approximates). Pure so its
/// content is testable.
pub(crate) fn ri_k_memory_remedy(bytes: usize) -> String {
    format!(
        "ferric NOTE: density-fitted exchange (RI-K) holds {:.2} GB of this; \
         df_k_aux=\"\" (Python) / [scf] df_k_aux = \"\" selects exact four-centre K and \
         does not build that tensor (k_builder has no effect while RI-K is active)",
        bytes as f64 / 1e9
    )
}

/// Diagonalize a Fock matrix in the rectangular canonical-orthogonal basis
/// (X from `rhf::canonical_orthogonalizer`, shape n×m with m ≤ n): Fʹ = XᵀFX,
/// C = XVʹ, padded back to (n × n) with sentinel-energy (1e6) zero columns for
/// dropped near-singular modes. Shared by RHF and UHF (per spin). ROHF keeps
/// its symmetric S^{-1/2} transform — a distinct orthogonalization policy.
///
/// The eigensolver is LAPACK `dsyevd_` (divide and conquer); see
/// [`FockEigensolver`] for why that is safe here. A serial section: runs at the
/// serial BLAS thread count (`ferric_core::blas_threads::opt_in_blas_threads`,
/// default = the rayon pool size; 1 on a rayon worker such as the SAD
/// free-atom path).
pub(crate) fn diagonalize_rect(
    f: &Array2<f64>,
    x: &Array2<f64>,
) -> Result<(Vec<f64>, Array2<f64>), FerricError> {
    diagonalize_rect_with(f, x, FockEigensolver::DivideConquer)
}

/// LAPACK driver for the Fock eigenproblem in [`diagonalize_rect_with`].
///
/// The SCF uses divide and conquer. The two drivers return the same
/// eigenvalues but may pick different (equally valid) eigenvectors inside a
/// degenerate subspace. Every consumer of the SCF's C (the density, the DF-K
/// and XC occupied factors, the level-shift projector, MOM) is invariant to a
/// rotation inside a block that is wholly occupied or wholly virtual, so the
/// choice only reaches the SCF at rounding level: the `fock_eigensolver_tests`
/// anchor pins the occupied density of CH4 (t2) and benzene (e1g) to 1e-12.
///
/// History: dsyevd was first rejected here on 2026-07-24 (09961c814) because
/// `b3lyp_methane` stopped converging at rung 0. That was the explicit
/// LU inverse of the RI-J metric (fixed by a Cholesky solve in 2d23e9ed2),
/// whose forward error amplified any bit-level change into 1e-5 Ha energy
/// jitter: with dsyevd patched onto f66652b96 (just before that fix), the
/// CH4 cases climb to ladder rungs 2 and 3, while on the fixed code both
/// solvers converge at rung 0 in the same iteration count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FockEigensolver {
    /// `dsyev_` (QR algorithm), via `ndarray_linalg::Eigh`. Kept as the
    /// reference the anchor test compares against.
    #[cfg_attr(not(test), allow(dead_code))]
    Qr,
    /// `dsyevd_` (divide and conquer), via `ferric_core::linalg::eigh_dc`.
    DivideConquer,
}

/// [`diagonalize_rect`] with an explicit eigensolver.
pub(crate) fn diagonalize_rect_with(
    f: &Array2<f64>,
    x: &Array2<f64>,
    solver: FockEigensolver,
) -> Result<(Vec<f64>, Array2<f64>), FerricError> {
    let n = x.nrows();
    let m = x.ncols();
    let (evals, c_kept) = with_blas_threads(opt_in_blas_threads(), || {
        crate::blas_probe::record(&crate::blas_probe::FOCK_DIAG);
        let f_prime = x.t().dot(f).dot(x);
        let (evals, evecs) = match solver {
            FockEigensolver::Qr => {
                let (e, v) = f_prime
                    .eigh(ndarray_linalg::UPLO::Upper)
                    .map_err(|e| FerricError::Lapack(format!("F diag: {e}")))?;
                (e.to_vec(), v)
            }
            FockEigensolver::DivideConquer => {
                ferric_core::linalg::eigh_dc(&f_prime, ferric_core::linalg::Uplo::Upper)?
            }
        };
        let c_kept = x.dot(&evecs); // (n × m)
        Ok::<_, FerricError>((evals, c_kept))
    })?;
    if m == n {
        return Ok((evals, c_kept));
    }
    let mut c = Array2::<f64>::zeros((n, n));
    c.slice_mut(ndarray::s![.., ..m]).assign(&c_kept);
    let mut eps = evals;
    eps.resize(n, 1e6);
    Ok((eps, c))
}

/// Per-iteration COSMO + IEF-PCM reaction-field terms, built from the TOTAL
/// density and added identically to every spin Fock the policy carries
/// (the reaction field is a classical, spin-independent potential). Returns
/// `(e_cosmo, e_pcm)` — kept separate so each solver's energy expression
/// `e_elec + e_xc + e_cosmo + e_pcm + vnn` keeps its historical FP
/// association. No-op `(0.0, 0.0)` when both are `None` (vacuum), preserving
/// the byte-identical-to-vacuum regression contract.
#[allow(clippy::too_many_arguments)]
pub(crate) fn solvent_terms(
    mol: &Molecule,
    prep: &PreparedBasis,
    config: &RhfConfig,
    cosmo_cavity: Option<&crate::cosmo::CosmoCavity>,
    pcm_ctx: Option<&ferric_pcm::PcmContext>,
    polarizable_site_basis: Option<&ferric_integrals::site_basis::SiteBasis>,
    d_total: &Array2<f64>,
    focks: &mut [&mut Array2<f64>],
) -> Result<(f64, f64, f64, Option<Array2<f64>>), FerricError> {
    let e_cosmo = if let Some(cavity) = cosmo_cavity {
        let cosmo_cfg = config
            .cosmo
            .as_ref()
            .expect("cosmo cavity implies config.cosmo");
        let cr = crate::cosmo::cosmo_reaction_field(mol, prep, cavity, cosmo_cfg, d_total)?;
        for f in focks.iter_mut() {
            **f += &cr.v_reaction;
        }
        cr.e_cosmo
    } else {
        0.0
    };
    let e_pcm = if let Some(pctx) = pcm_ctx {
        let (v_pcm, e_pcm) = ferric_pcm::pcm_step(pctx, mol, prep, d_total)?;
        for f in focks.iter_mut() {
            **f += &v_pcm;
        }
        e_pcm
    } else {
        0.0
    };
    // Thole-damped polarizable embedding: re-solve the induced dipoles from
    // the CURRENT total density every iteration (same pattern as COSMO/PCM
    // above), fold the resulting Fock term into every spin Fock uniformly
    // (the induced-dipole potential is a classical, spin-independent
    // operator), and surface E_pol as a standalone energy term plus the
    // converged dipoles for `ScfResult::induced_dipoles`. `None`/empty-sites
    // (`polarizable_site_basis: None`) is a true no-op: (0.0, None) with the
    // focks untouched, exactly as COSMO/PCM's `None` branches are — see
    // `polarizable_none_is_bit_identical_to_plain_scf` and
    // `polarizable_empty_sites_is_bit_identical_to_plain_scf` in
    // `tests/qmmm_polarizable.rs`.
    let (e_pol, induced_dipoles) = if let Some(site_basis_p) = polarizable_site_basis {
        let pol_cfg = config
            .polarizable
            .as_ref()
            .expect("polarizable_site_basis implies config.polarizable");
        let ir = crate::polarizable::induce(
            mol,
            prep,
            config.external_potential.as_ref(),
            pol_cfg,
            site_basis_p,
            d_total,
        )?;
        for f in focks.iter_mut() {
            **f += &ir.v_induced;
        }
        (ir.e_pol, Some(ir.dipoles))
    } else {
        (0.0, None)
    };
    Ok((e_cosmo, e_pcm, e_pol, induced_dipoles))
}

/// Convergence bookkeeping shared by every SCF variant: previous energy, the
/// carried ΔP signals (INFINITY before the first density rebuild, so the gate
/// can never fire on iteration 1), the divergence streak, and the stall
/// history. One implementation of the exits that used to live only in
/// `solve_rhf` — through this, UHF/ROHF honor `stall_window` /
/// `divergence_tol` too (they silently ignored them before; the `None`
/// defaults are byte-identical to that old behavior).
pub(crate) struct ScfMonitor {
    pub prev_e: f64,
    pub dp_rms: f64,
    pub dp_max: f64,
    errmax_history: Vec<f64>,
    divergence_streak: usize,
    /// Set by [`Self::invalidate_deltas`] to the bits of the `prev_e` it
    /// invalidated: while `prev_e` still holds them, ΔE reads INFINITY. The
    /// next `note_energy` replaces `prev_e` and so lifts it (a new energy
    /// equal to the old to the bit keeps it one iteration longer: conservative).
    de_invalid_at: Option<u64>,
}

impl ScfMonitor {
    /// Create a new SCF convergence monitor with default initial state.
    pub fn new() -> Self {
        ScfMonitor {
            prev_e: 0.0,
            dp_rms: f64::INFINITY,
            dp_max: f64::INFINITY,
            errmax_history: Vec::new(),
            divergence_streak: 0,
            de_invalid_at: None,
        }
    }

    /// Discard the carried ΔP and ΔE: the next convergence decision sees both
    /// as INFINITY, so the gate cannot fire until a new density change and a
    /// new energy pair have been recorded. ΔP returns with the next
    /// `record_density_change`, ΔE once `note_energy` has replaced `prev_e`.
    /// Used by the J/K ladder's f64 redo (`crate::jk_ladder`).
    pub fn invalidate_deltas(&mut self) {
        self.dp_rms = f64::INFINITY;
        self.dp_max = f64::INFINITY;
        self.de_invalid_at = Some(self.prev_e.to_bits());
    }

    /// The settled signals for this iteration's convergence decision.
    pub fn signals(&self, energy: f64) -> ConvergenceSignals {
        ConvergenceSignals {
            de: self.delta_e(energy),
            dp_rms: self.dp_rms,
            dp_max: self.dp_max,
        }
    }

    /// `|energy − prev_e|`, or INFINITY while `prev_e` is invalidated.
    fn delta_e(&self, energy: f64) -> f64 {
        if self.de_invalid_at == Some(self.prev_e.to_bits()) {
            return f64::INFINITY;
        }
        (energy - self.prev_e).abs()
    }

    /// Record this iteration's ΔP from the old/new densities (call after the
    /// density rebuild; consumed by NEXT iteration's `signals`).
    pub fn record_density_change(&mut self, d_new: &Array2<f64>, d_old: &Array2<f64>) {
        let (rms, max) = density_change(d_new, d_old);
        self.dp_rms = rms;
        self.dp_max = max;
    }

    /// Divergence check (energy climbing > tol for 3 consecutive iterations).
    /// Call BEFORE `note_energy` — uses the signed change vs the previous
    /// iteration. `None` tol always returns false and records nothing.
    pub fn diverging(&mut self, energy: f64, tol: Option<f64>) -> bool {
        let Some(tol) = tol else { return false };
        if energy - self.prev_e > tol {
            self.divergence_streak += 1;
        } else {
            self.divergence_streak = 0;
        }
        self.divergence_streak >= 3
    }

    /// Stall check (running-min err_max stopped falling over `window`).
    /// Pushes to the history only when a window is configured, exactly like
    /// the original solve_rhf logic. `None` window always returns false.
    pub fn stalled(&mut self, err_max: f64, window: Option<usize>) -> bool {
        let Some(w) = window else { return false };
        self.errmax_history.push(err_max);
        crate::rhf::stall_detected(&self.errmax_history, w, err_max)
    }

    /// Bank this iteration's energy as `prev_e` for the next iteration.
    pub fn note_energy(&mut self, energy: f64) {
        self.prev_e = energy;
    }
}

#[cfg(test)]
mod monitor_tests {
    use super::ScfMonitor;
    use ndarray::Array2;

    /// Catches: an invalidation that leaves a finite ΔP or ΔE for the next
    /// decision, or one that never lifts (the gate could then never fire).
    #[test]
    fn invalidated_deltas_read_infinite_until_both_are_recorded_again() {
        let mut m = ScfMonitor::new();
        let (a, b) = (
            Array2::<f64>::zeros((2, 2)),
            Array2::from_elem((2, 2), 1e-3),
        );
        m.note_energy(-1.0);
        m.record_density_change(&b, &a);
        assert!(m.signals(-1.0 + 1e-9).de < 1e-8);
        m.invalidate_deltas();
        let s = m.signals(-1.0);
        assert!(s.de.is_infinite() && s.dp_rms.is_infinite() && s.dp_max.is_infinite());
        m.note_energy(-1.5);
        m.record_density_change(&a, &b);
        let s = m.signals(-1.5 + 1e-9);
        assert!(s.de < 1e-8 && s.dp_rms.is_finite() && s.dp_max.is_finite());
    }
}

#[cfg(test)]
mod ri_k_remedy_tests {
    use super::ri_k_memory_remedy;

    /// The over-budget RSS note names exact K (`df_k_aux = ""`) as the remedy
    /// for a held RI-K tensor, with its size, and says `k_builder` is inert
    /// while RI-K is active.
    ///
    /// MUTATION: drop the size or the `df_k_aux` spelling from the format
    /// string and this goes RED.
    #[test]
    fn ri_k_remedy_names_exact_k_and_the_tensor_size() {
        let msg = ri_k_memory_remedy(4_510_000_000);
        assert!(msg.contains("4.51 GB"), "{msg}");
        assert!(msg.contains("df_k_aux=\"\""), "{msg}");
        assert!(msg.contains("[scf] df_k_aux = \"\""), "{msg}");
        assert!(msg.contains("k_builder has no effect"), "{msg}");
    }
}

#[cfg(test)]
mod raw_budget_tests {
    use super::*;
    use ferric_dft::xc_trait::XcMemoryPlan;

    /// A fixed budget loses the cache whether or not a pool holds it, and only that.
    #[test]
    fn explicit_budget_moves_only_by_the_nlc_cache() {
        let plan = |held_bytes, nlc_bytes| XcMemoryPlan {
            held_bytes,
            nlc_bytes,
            resident_if_alone: 3000,
        };
        assert_eq!(raw_budget(10_000, plan(0, 0)), 10_000);
        assert_eq!(raw_budget(10_000, plan(0, 500)), 9_500, "no pool");
        assert_eq!(raw_budget(10_000, plan(500, 500)), 9_500, "pool");
    }

    #[test]
    fn a_live_budget_does_not_count_the_cache_twice() {
        // Live, no pool: the live figure has already fallen with the cache.
        assert_eq!(budget_beside_nlc(9_000, true, 500, None), 9_000);
        // Live, ample pool: the same.
        assert_eq!(budget_beside_nlc(9_000, true, 500, Some(10_000)), 9_000);
        // Live, tight pool: held to what the pool has left, once.
        assert_eq!(budget_beside_nlc(9_800, true, 500, Some(10_000)), 9_500);
        // Fixed: the cache comes off.
        assert_eq!(budget_beside_nlc(9_000, false, 500, None), 8_500);
        // a 10,000 cgroup sets an 8,000 auto budget that the cache does not lower
        assert_eq!(budget_beside_nlc(8_000, true, 3_000, Some(8_000)), 5_000);
        // MemAvailable already binds below that ceiling
        assert_eq!(budget_beside_nlc(4_000, true, 3_000, Some(8_000)), 4_000);
        // the planned AO blocks come off a cgroup ceiling before the cache does
        let ceiling = ceiling_beside_ao(Some(8_000), 2_000);
        assert_eq!(ceiling, Some(6_000));
        assert_eq!(budget_beside_nlc(8_000, true, 500, ceiling), 5_500);
        assert_eq!(ceiling_beside_ao(Some(1_000), 2_000), Some(0));
        assert_eq!(ceiling_beside_ao(Some(8_000), 0), Some(8_000));
        assert_eq!(ceiling_beside_ao(None, 2_000), None);
        assert_eq!(lowest_ceiling(Some(7), Some(5)), Some(5));
        assert_eq!(lowest_ceiling(Some(5), Some(7)), Some(5));
        assert_eq!(lowest_ceiling(None, Some(7)), Some(7));
        assert_eq!(lowest_ceiling(Some(7), None), Some(7));
        assert_eq!(lowest_ceiling(None, None), None);
        // No cache: nothing moves, however small the pool. A pool that cannot pay
        // for an in-core tensor must still refuse it, not quietly reroute it.
        assert_eq!(budget_beside_nlc(9_000, true, 0, Some(200)), 9_000);
        assert_eq!(budget_beside_nlc(9_000, false, 0, Some(200)), 9_000);
    }

    /// The auto budget gives up 0.8 of what XC will hold, as it would if the bytes
    /// were already allocated. Skipped where the live figure cannot show it: a fixed
    /// source, a cgroup limit below MemAvailable, or too little memory to subtract
    /// from. Other tests allocate between the two readings, so one close reading in
    /// a few is enough; without the subtraction none comes close.
    #[test]
    fn auto_budget_allows_for_planned_xc_memory() {
        let x = 1usize << 30;
        let res = ferric_core::memory::resolve_budget(None);
        let meminfo = ferric_core::memory::sys::available_ram_bytes();
        let live = ferric_core::memory::detect_available_bytes();
        if res.source != ferric_core::memory::BudgetSource::AutoDetected
            || meminfo.is_none_or(|m| m < 4 * x)
            || live.unwrap_or(0) + (1 << 26) < meminfo.unwrap_or(0)
        {
            return;
        }
        let plan = XcMemoryPlan {
            resident_if_alone: x,
            ..Default::default()
        };
        let want = 0.8 * x as f64;
        let drops: Vec<usize> = (0..5)
            .map(|_| raw_budget(0, XcMemoryPlan::default()).saturating_sub(raw_budget(0, plan)))
            .collect();
        assert!(
            drops.iter().any(|&d| (d as f64 - want).abs() < 0.25 * want),
            "budget fell by {drops:?}, want about {want}"
        );
    }

    #[test]
    fn a_lost_block_says_what_is_left_of_the_guess() {
        let block = LostGuessBlock {
            z: 8,
            route: "free-atom SCF",
            reason: "no memory".into(),
        };
        let some = guess_block_lost_text(&block, false);
        assert!(some.contains("failed for O (Z=8)"), "{some}");
        assert!(
            some.contains("partial MINAO guess, not from hcore"),
            "{some}"
        );
        let all = guess_block_lost_text(&block, true);
        assert!(all.contains("first Fock matrix is hcore"), "{all}");
        assert!(!all.contains("partial"), "{all}");
    }
}

#[cfg(test)]
mod fock_eigensolver_tests {
    //! Exactness anchor for the dsyevd (divide-and-conquer) Fock
    //! diagonalization: on a FIXED Fock matrix with a degenerate, fully
    //! occupied block, dsyevd and dsyev must give the same eigenvalues and the
    //! same occupied density, even though the eigenvectors inside the block
    //! differ. The split check (an occupation that cuts the block) is the
    //! built-in mutation: it must see the two eigenvector choices disagree,
    //! or the density assertion is not testing the invariance at all.
    use super::{diagonalize_rect_with, FockEigensolver};
    use crate::rhf::{canonical_orthogonalizer, solve_rhf, RhfConfig};
    use crate::screening::SchwarzBounds;
    use ferric_core::basis;
    use ferric_core::mol::Molecule;
    use ferric_core::parallel::ParallelContext;
    use ferric_integrals::basis_bridge::PreparedBasis;
    use ferric_integrals::oneelectron;
    use ferric_integrals::operator::Operator;
    use ndarray::{s, Array2};

    /// Converged AO Fock matrix, the canonical orthogonalizer and nocc.
    fn fixed_fock(
        xyz: &str,
        basis_name: &str,
        cfg: RhfConfig,
    ) -> (Array2<f64>, Array2<f64>, usize) {
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let bs = basis::bundled(basis_name).unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let res = solve_rhf(&ParallelContext::default(), &mol, &prep, op, &bounds, &cfg).unwrap();
        let x = canonical_orthogonalizer(&oneelectron::overlap(&prep)).unwrap();
        (res.fock_alpha, x, mol.nelec() as usize / 2)
    }

    fn density(c: &Array2<f64>, nocc: usize) -> Array2<f64> {
        let co = c.slice(s![.., ..nocc]);
        2.0 * co.dot(&co.t())
    }

    fn max_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(p, q)| (p - q).abs())
            .fold(0.0, f64::max)
    }

    /// `block` = first orbital of the degenerate occupied block (0-based; the
    /// block ends at nocc); `split` = an occupation count that cuts the block.
    fn check(
        label: &str,
        f: &Array2<f64>,
        x: &Array2<f64>,
        nocc: usize,
        block: usize,
        split: usize,
    ) {
        let (e_qr, c_qr) = diagonalize_rect_with(f, x, FockEigensolver::Qr).unwrap();
        let (e_dc, c_dc) = diagonalize_rect_with(f, x, FockEigensolver::DivideConquer).unwrap();
        // The case under test really is degenerate and fully occupied.
        let spread = e_qr[nocc - 1] - e_qr[block];
        let gap = e_qr[nocc] - e_qr[nocc - 1];
        assert!(
            spread < 1e-10,
            "{label}: block not degenerate (spread {spread:.2e})"
        );
        assert!(gap > 1e-2, "{label}: no HOMO-LUMO gap ({gap:.2e})");

        let de = e_qr
            .iter()
            .zip(&e_dc)
            .map(|(a, b)| (a - b).abs() / a.abs().max(1.0))
            .fold(0.0, f64::max);
        let dd = max_abs_diff(&density(&c_qr, nocc), &density(&c_dc, nocc));
        let dd_split = max_abs_diff(&density(&c_qr, split), &density(&c_dc, split));
        eprintln!(
            "[{label}] spread {spread:.2e}, max rel d_eps = {de:.2e}, max |dD| = {dd:.2e}, \
             split max |dD| = {dd_split:.2e}"
        );
        assert!(de <= 1e-12, "{label}: eigenvalues differ by {de:.2e}");
        assert!(dd <= 1e-12, "{label}: occupied density differs by {dd:.2e}");
        assert!(
            dd_split > 1e-6,
            "{label}: the two solvers chose the same vectors inside the block \
             (split |dD| {dd_split:.2e}), so the density check is vacuous"
        );
    }

    /// CH4/cc-pVDZ B3LYP with RI-JK: the system whose `b3lyp_methane`
    /// convergence check failed when dsyevd was first tried (2026-07-24).
    /// Occupied: 1a1, 2a1, 1t2 (indices 2..5).
    #[test]
    fn dsyevd_matches_dsyev_density_on_methane_t2() {
        let xyz = "5\nCH4\nC 0 0 0\nH 0.6276 0.6276 0.6276\nH -0.6276 -0.6276 0.6276\n\
                   H -0.6276 0.6276 -0.6276\nH 0.6276 -0.6276 -0.6276\n";
        let cfg = RhfConfig {
            xc: Some("B3LYP".into()),
            df_j_aux: Some("def2-universal-jkfit".into()),
            df_k_aux: Some("def2-universal-jkfit".into()),
            energy_conv: 1e-10,
            density_conv: 1e-8,
            ..Default::default()
        };
        let (f, x, nocc) = fixed_fock(xyz, "cc-pvdz", cfg);
        assert_eq!(nocc, 5);
        check("CH4/cc-pVDZ B3LYP", &f, &x, nocc, 2, 4);
    }

    /// Benzene (exact D6h) def2-SVP HF with RI-JK: the HOMO is the doubly
    /// degenerate 1e1g pair (occupied indices 19, 20). HF, not a functional:
    /// the XC grid is not D6h-invariant and splits the pair by ~1e-4 Ha.
    #[test]
    fn dsyevd_matches_dsyev_density_on_benzene_e1g() {
        let mut xyz = String::from("12\nbenzene\n");
        for (sym, r) in [("C", 1.396_f64), ("H", 2.479)] {
            for k in 0..6 {
                let t = std::f64::consts::FRAC_PI_3 * k as f64;
                xyz.push_str(&format!(
                    "{sym} {:.15} {:.15} 0\n",
                    r * t.sin(),
                    r * t.cos()
                ));
            }
        }
        let cfg = RhfConfig {
            df_j_aux: Some("def2-universal-jkfit".into()),
            df_k_aux: Some("def2-universal-jkfit".into()),
            // f64: f32 storage rounds symmetry-equivalent integrals (equal to
            // ~1e-15) to different f32 neighbours and splits this e1g block by
            // 9.5e-9 (measured), which is not what this eigensolver test probes.
            ri_j_precision: Some(crate::df_j::RiJPrecision::F64),
            ..Default::default()
        };
        let (f, x, nocc) = fixed_fock(&xyz, "def2-svp", cfg);
        assert_eq!(nocc, 21);
        check("benzene/def2-SVP HF", &f, &x, nocc, 19, 20);
    }
}

#[cfg(test)]
mod level_shift_ramp_tests {
    use super::{closed_shell_level_shift_ramp, open_shell_level_shift_ramp};

    fn close(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-15, "{a} vs {b}");
    }

    /// RHF/RKS: the full shift while `max|e| >= 0.1`, linear below.
    #[test]
    fn closed_shell_ramp_is_full_above_a_tenth_and_linear_below() {
        close(closed_shell_level_shift_ramp(0.5, 1.0), 0.5);
        close(closed_shell_level_shift_ramp(0.5, 0.1), 0.5);
        close(closed_shell_level_shift_ramp(0.5, 1e-2), 0.05);
        close(closed_shell_level_shift_ramp(0.5, 1e-3), 0.005);
        assert_eq!(closed_shell_level_shift_ramp(0.5, 0.0), 0.0);
    }

    /// UHF/ROHF and the periodic ROKS loops: half the shift at `max|e| =
    /// 1e-3`, most of it at 1e-2 (where the k-point ROKS trap needs it).
    #[test]
    fn open_shell_ramp_is_half_the_shift_at_1e_3() {
        close(open_shell_level_shift_ramp(0.5, 1e-3), 0.25);
        close(open_shell_level_shift_ramp(0.5, 1e-2), 0.5 * 1e-2 / 1.1e-2);
        close(open_shell_level_shift_ramp(0.5, 1.0), 0.5 / 1.001);
        assert_eq!(open_shell_level_shift_ramp(0.5, 0.0), 0.0);
    }
}
