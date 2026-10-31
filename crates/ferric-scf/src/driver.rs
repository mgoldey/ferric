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
use crate::rhf::{ConvergenceSignals, RhfConfig};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_core::FerricError;
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
/// - An NLC cache already charged to the pool leaves that much less for raw. Raw
///   picks `InCore` from this figure alone and refuses if the pool cannot pay, so
///   the figure has to tell it. Otherwise a budget just above the tensor is
///   refused where a smaller one, which gets a soft tier, runs.
///
/// Nothing here reads the thread count at an explicit budget.
pub(crate) fn raw_budget(config_bytes: usize, plan: ferric_dft::xc_trait::XcMemoryPlan) -> usize {
    let explicit = (config_bytes != 0).then_some(config_bytes);
    ferric_core::memory::resolve_budget_after(explicit, plan.resident_if_alone)
        .bytes
        .saturating_sub(plan.held_bytes)
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
    let main = config.dft_grid.clone().unwrap_or_default();
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
/// with the solver). The hcore always folds the ECP projector in via
/// `hcore_ecp_with_external`: for an all-electron basis `ecp_potential` returns
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
        Some(injected) => injected,
        None => {
            let s = oneelectron::overlap(prep);
            let h = oneelectron::hcore_ecp_with_external(
                prep,
                mol,
                prep.basis_set(),
                config.external_potential.as_ref(),
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
        let (sr, lr) = crate::fock_assembly::build_rsh_dfk_pair(
            ctx,
            mol,
            prep,
            config.df_k_aux.as_deref(),
            k_mix.omega,
            ooc_budget,
        )?;
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

/// The default virtual-block level shift: the user's value, or 0.5 for
/// meta-GGA when the user set none (meta-GGA SCF limit-cycles under plain
/// DIIS — see the solve_rhf comment where this convention originated).
pub(crate) fn effective_level_shift(config: &RhfConfig) -> f64 {
    if config.level_shift == 0.0 && crate::rohf::xc_is_metagga(config.xc.as_deref()) {
        0.5
    } else {
        config.level_shift
    }
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
pub(crate) fn warn_if_rss_over_at_stage(variant: &str, stage: &str, ooc_budget: usize) {
    ferric_core::memory::warn_if_rss_over(&format!("{variant} {stage}"), ooc_budget, 1.1);
}

/// Diagonalize a Fock matrix in the rectangular canonical-orthogonal basis
/// (X from `rhf::canonical_orthogonalizer`, shape n×m with m ≤ n): Fʹ = XᵀFX,
/// C = XVʹ, padded back to (n × n) with sentinel-energy (1e6) zero columns for
/// dropped near-singular modes. Shared by RHF and UHF (per spin). ROHF keeps
/// its symmetric S^{-1/2} transform — a distinct orthogonalization policy.
///
/// Runs under the opt-in BLAS raise (`FERRIC_BLAS_THREADS`, default 1 →
/// byte-identical to a bare call; the rayon-worker self-guard protects the
/// SAD/free-atom path, see the blas_threads docs).
pub(crate) fn diagonalize_rect(
    f: &Array2<f64>,
    x: &Array2<f64>,
) -> Result<(Vec<f64>, Array2<f64>), FerricError> {
    let n = x.nrows();
    let m = x.ncols();
    let (evals, c_kept) = with_blas_threads(opt_in_blas_threads(), || {
        let f_prime = x.t().dot(f).dot(x);
        // NOTE: this every-SCF-iteration Fock diagonalization deliberately keeps
        // the QR-algorithm `.eigh()` (dsyev_) rather than the faster
        // divide-and-conquer `ferric_core::linalg::eigh_dc` (dsyevd_). Both give
        // identical eigenvalues, but they pick DIFFERENT (equally valid)
        // eigenvectors inside degenerate subspaces (e.g. methane/Td triply-
        // degenerate MOs). Those eigenvectors feed the density matrix, and the
        // dsyevd choice pushes symmetric molecules into a fixed point where ΔP
        // limit-cycles just above the 1e-8 density-convergence gate (B3LYP/CH4:
        // energy still correct to 1.1e-8 Ha, but ladder_converged=false). The
        // eigenvector-consuming SCF convergence path is therefore NOT a safe
        // place to swap eigensolvers; eigenvalue-only consumers (RPA dielectric
        // sweeps) are, and use eigh_dc/eigvalsh_dc. See the linalg module doc.
        let (evals, evecs) = f_prime.eigh(ndarray_linalg::UPLO::Upper)?;
        let c_kept = x.dot(&evecs); // (n × m)
        Ok::<_, ndarray_linalg::error::LinalgError>((evals, c_kept))
    })
    .map_err(|e| FerricError::Lapack(format!("F diag: {e}")))?;
    if m == n {
        return Ok((evals.to_vec(), c_kept));
    }
    let mut c = Array2::<f64>::zeros((n, n));
    c.slice_mut(ndarray::s![.., ..m]).assign(&c_kept);
    let mut eps = evals.to_vec();
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
mod raw_budget_tests {
    use super::*;
    use ferric_dft::xc_trait::XcMemoryPlan;

    /// An explicit budget is a fixed number: only a held cache moves it.
    #[test]
    fn explicit_budget_moves_only_by_what_the_pool_holds() {
        let plan = |held_bytes, nlc_bytes| XcMemoryPlan {
            held_bytes,
            nlc_bytes,
            resident_if_alone: 3000,
        };
        assert_eq!(raw_budget(10_000, plan(0, 0)), 10_000);
        assert_eq!(raw_budget(10_000, plan(0, 500)), 10_000);
        assert_eq!(raw_budget(10_000, plan(500, 500)), 9_500);
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
        let meminfo = std::fs::read_to_string("/proc/meminfo")
            .ok()
            .and_then(|s| ferric_core::memory::parse_meminfo_available(&s));
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
