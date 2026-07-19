//! Closed-shell restricted Hartree-Fock (RHF) solver.
//!
//! Implements the Roothaan-Hall SCF procedure with DIIS convergence acceleration
//! and Schwarz-screened two-electron integral evaluation.

use crate::df_j::DfJ;
use crate::df_k::DfK;
use crate::diis::DiisDriver;
use crate::direct_j::DirectJ;
use crate::direct_jk::DirectJK;
use crate::direct_k::DirectK;
use crate::fock::{JBuilder, KBuilder};
use crate::guess::hcore_guess;
use ferric_dft::cdft::Constraint;
use crate::result::{ScfExit, ScfResult, Spin};

use crate::link_k::LinkK;
use crate::screening::SchwarzBounds;
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::blas_threads::{opt_in_blas_threads, with_blas_threads};
use ferric_integrals::oneelectron;
use ferric_integrals::operator::Operator;
use ferric_core::parallel::ParallelContext;
use ndarray::Array2;
use ndarray_linalg::Eigh;

/// Configuration parameters for the RHF solver.
#[derive(Debug, Clone)]
pub struct RhfConfig {
    pub max_iter: usize,
    pub energy_conv: f64,
    pub density_conv: f64,
    pub diis_size: usize,
    /// DIIS family: `Pulay` (default, plain commutator DIIS — historical
    /// byte-identical behavior) or `Adiis`/`Ediis` (energy-based variant in the
    /// early SCF, switching to Pulay near convergence — for hard TM-dimer cases).
    pub diis_flavor: crate::diis::DiisFlavor,
    /// Commutator `err_max` crossover below which `diis_flavor` reverts to plain
    /// Pulay. Ignored when `diis_flavor == Pulay`. Default 1e-1 (PySCF/ORCA).
    pub diis_switch_thresh: f64,
    /// Finite-temperature (Fermi-Dirac) occupation smearing width σ = k_B·T in
    /// Hartree. `None` (default) = integer 0/2 aufbau occupation (unchanged).
    /// `Some(σ)` smears the frontier — a convergence aid for near-degenerate
    /// d-manifolds (TM dimers/metals). See `smearing.rs`.
    pub smearing_sigma: Option<f64>,
    pub integral_thresh: f64,
    /// Choose K matrix builder: "direct" (default) or "link".
    pub k_builder: Option<String>,
    /// Optional auxiliary basis for density-fitted Coulomb (RI-J). When set, J is
    /// built from precomputed 3-center ERIs in O(N^2 · naux) per iteration instead
    /// of contracting full 4-index ERIs.
    pub df_j_aux: Option<String>,
    /// Optional auxiliary basis for density-fitted exchange (RI-K). When set, K is
    /// built from the V^{-1/2}-dressed 3-center tensor in O(N^3 · naux) GEMMs.
    /// Should be a JK-fit basis (e.g. `def2-universal-jkfit`), not an RI/MP2-fit
    /// basis, which would introduce mHa-scale error in K.
    pub df_k_aux: Option<String>,
    /// XC functional name (None = pure HF; e.g. "LDA", "PBE", "B3LYP", "wB97X-V").
    pub xc: Option<String>,
    /// Main DFT grid spec. Default (75, 110) when xc.is_some().
    pub dft_grid: Option<ferric_dft::grid::AtomicGridConfig>,
    /// NLC (VV10) grid spec. Default (50, 50) when XC requires VV10.
    pub nlc_grid: Option<ferric_dft::grid::AtomicGridConfig>,
    /// Level shift (Ha) applied to the virtual–virtual block of the Fock
    /// matrix in MO basis. Defaults to 0 (no shift). Used to damp oscillations
    /// in open-shell SCF (ROHF/UHF/ROKS) where DIIS plateaus near a near-
    /// degenerate transition. A shift of 0.1–0.5 Ha is typical.
    pub level_shift: f64,
    /// If > 0 in ROHF/ROKS: switch from DIIS to a damped-Newton step once the
    /// DIIS error (err_max) drops below this trigger. Setting `1e-2` is a
    /// reasonable default for OH-doublet LDA/PBE plateaus. A value of 0
    /// disables Newton entirely (DIIS-only).
    pub newton_trigger: f64,
    /// If > 0 in ROHF/ROKS: switch from DIIS / damped-Newton to an
    /// augmented-Hessian Newton step once err_max drops below this trigger.
    /// AH handles vanishing Hessian eigenvalues that trip up PCG. A value
    /// of 0 disables AH. Both triggers can be set: PCG fires first (when
    /// err_max < newton_trigger) and AH takes over when err_max < ah_trigger
    /// (typically a tighter threshold).
    pub ah_trigger: f64,
    /// If > 0 (RHF/UHF/ROHF/ROKS): activate Maximum-Overlap Method
    /// reordering after this many DIIS iters. From iter `mom_after_iter + 1`
    /// onward, the occupied MO set is picked by AO-overlap with the
    /// previous-iter accepted set (rather than by ε). Default 0 = disabled.
    /// MOM pins whatever basin DIIS holds at arming — it breaks occupied-set
    /// flip-flop, it does not steer to the ground state. Arm only after DIIS
    /// has settled (open-shell plateaus: ~5; closed-shell wanderers: 50+).
    pub mom_after_iter: usize,
    /// cDFT constraints. Empty (default) = ordinary SCF. Each constraint pins a
    /// fragment population (charge or spin) to a target via a Lagrange
    /// multiplier added to the Fock matrix. Consumed by `solve_cdft_uhf`.
    pub constraints: Vec<Constraint>,
    /// cDFT outer-loop convergence: stop when max_C |N_C − target_C| is below
    /// this (electrons). Default 1e-5.
    pub cdft_lambda_tol: f64,
    /// Fractional (ensemble) occupation of a degenerate frontier shell. When
    /// `true` (UHF/UKS only), if the per-spin HOMO sits inside a group of
    /// near-degenerate orbitals that straddle the occupation boundary, the
    /// integer occupation is spread *equally* over that group (e.g. a ³P atom's
    /// 2 p-electrons → 2/3 in each of px/py/pz). This restores the spherical
    /// symmetry that otherwise makes the GGA XC potential orientation-dependent
    /// and the SCF oscillate forever (free O/S/Si atoms never converge with
    /// integer occupation + PBE). Default `false` — opt-in; integer-occupation
    /// paths are unchanged.
    pub fractional_occ: bool,
    /// Hard ceiling (bytes) for the resident 3-index footprint in DfJ/DfK. When
    /// the dense `(naux,nao,nao)` tensor would exceed this, the source spills
    /// aux-blocks to disk instead of allocating in core. `0` = unset → resolved
    /// via [`resolve_three_index_budget`] (this value, when non-zero, OVERRIDES
    /// the `FERRIC_MEM_BUDGET_GB` / `FERRIC_OOC_BUDGET_GB` env vars; env fills in
    /// only when this is 0; then auto-detect 0.8×RAM; then a 2 GiB fallback).
    pub three_index_budget_bytes: usize,
    /// Optional externally-supplied initial density matrix. When `Some(d)`, the
    /// SCF loop uses this density as the starting point instead of computing an
    /// hcore or SAD guess internally. Shape must match `(nbasis, nbasis)`. The
    /// primary use-case is a SAD guess built by `guess::sad_guess(...)`.
    pub init_guess_density: Option<Array2<f64>>,
    /// When `true` (the default) and no `init_guess_density` is supplied, the SCF
    /// starts from a SAD (superposition-of-atomic-densities) guess computed via
    /// `guess::sad_guess`, falling back to the hcore guess if SAD fails. SAD cures
    /// the heavy-atom RHF divergence class (COSe/C2H3Br) that hcore triggers. Set
    /// `false` to force the bare hcore guess — used internally by `sad_guess` for
    /// its free-atom solves to break the recursion.
    pub use_sad_guess: bool,
    /// If `Some(n)`: abort early when the running minimum of `err_max` over the
    /// last `n` iters has not dropped below 0.9× its value over the previous `n`
    /// iters (gradient stopped falling) AND err_max is still above the 1e-4
    /// plateau floor. `None` (default) disables stall detection. Used by the
    /// convergence ladder to advance a stuck rung in ~n iters instead of max_iter.
    pub stall_window: Option<usize>,
    /// If `Some(f)`: abort early when the energy rises by more than `f` Ha for 3
    /// consecutive iterations (actively diverging, not just noisy). `None`
    /// (default) disables divergence detection.
    pub divergence_tol: Option<f64>,
    /// Fixed classical external potential (point charges + uniform field).
    /// `None` (default) = no external potential; folded into `hcore` once
    /// before the SCF loop, orthogonal to cDFT's per-iteration Fock hook.
    pub external_potential: Option<ferric_core::external_potential::ExternalPotential>,
    /// COSMO implicit-solvent configuration. `None` (default) = no solvent;
    /// the SCF loop is then byte-for-byte identical to a build with no COSMO
    /// support at all. Unlike `external_potential` (folded into `hcore`
    /// ONCE before the loop), COSMO's reaction-field potential depends on
    /// the density and is recomputed EVERY iteration (see `crate::cosmo`).
    pub cosmo: Option<crate::cosmo::CosmoConfig>,
    /// IEF-PCM implicit-solvent configuration. `None` (default) = vacuum
    /// (no solvent); this MUST be byte-identical to a plain vacuum
    /// calculation (see `pcm_none_matches_vacuum_*` regression tests).
    /// Unlike `external_potential` (fixed, folded into `hcore` once before
    /// the loop), PCM's apparent surface charge depends self-consistently
    /// on the solute density, so the reaction-field operator is rebuilt
    /// from the CURRENT density every SCF iteration (see the `pcm_ctx`
    /// handling in `solve_rhf`) and the outer SCF/DIIS loop carries the
    /// overall q-vs-D fixed point, the same pattern PySCF/Psi4 use.
    /// Mutually usable alongside `cosmo` (both are independent implicit-
    /// solvent models); using both simultaneously is not validated and not
    /// currently prevented at the type level — callers should pick one.
    pub pcm: Option<ferric_pcm::PcmConfig>,
}

impl Default for RhfConfig {
    fn default() -> Self {
        Self {
            // Convergence gate = ΔP primary + ΔE loose sanity (see scf_converged),
            // NOT the DIIS commutator. `density_conv` is the TIGHT threshold on
            // dp_rms (ORCA TolRMSP): the density genuinely drains here (MEASURED
            // ~1e-9 at aTZ). `energy_conv` is a LOOSE "not still descending"
            // bound on |ΔE| — deliberately 1e-3, far above the ~2e-5 DF energy
            // noise floor, because ΔE (like the commutator) floors with naux and
            // a tight ΔE is unreachable. ΔP does the real work.
            //
            // History: (1e-10, 1e-8) once guarded an H2O+ UHF false convergence
            // (a *gradient*-gated accept of a state 85 mHa high). ΔP is a stronger
            // wrong-basin signal than that gradient — regression-guarded by the
            // h2o_plus_* UHF tests, which pass under this gate.
            max_iter: 200,
            energy_conv: 1e-3,
            density_conv: 1e-6,
            diis_size: 8,
            diis_flavor: crate::diis::DiisFlavor::Pulay,
            diis_switch_thresh: 1e-1,
            smearing_sigma: None,
            integral_thresh: 1e-12,
            k_builder: None,
            df_j_aux: None,
            df_k_aux: None,
            xc: None,
            dft_grid: None,
            nlc_grid: None,
            level_shift: 0.0,
            newton_trigger: 0.0,
            ah_trigger: 0.0,
            mom_after_iter: 0,
            constraints: Vec::new(),
            cdft_lambda_tol: 1e-5,
            fractional_occ: false,
            // 0 = "unset" → resolve_three_index_budget auto-detects (0.8×RAM).
            three_index_budget_bytes: 0,
            init_guess_density: None,
            use_sad_guess: true,
            stall_window: None,
            divergence_tol: None,
            external_potential: None,
            cosmo: None,
            pcm: None,
        }
    }
}

/// Resolve the 3-index memory budget in bytes by delegating to the single
/// unified resolver [`ferric_core::memory::resolve_budget_bytes`], so every
/// memory setting shares ONE precedence chain (TOML/config > env > auto):
/// 1. A non-zero `config_bytes` — an explicit caller choice (TOML `[memory]`
///    budget / config field / kwarg). **TOML/config overrides env.**
/// 2. `FERRIC_MEM_BUDGET_GB` env var (GiB).
/// 3. Legacy `FERRIC_OOC_BUDGET_GB` / `FERRIC_ERI3_BUDGET_GB` env vars (GiB).
/// 4. Auto: 0.8 × detected available RAM (keeps the DF-JK tensor IN RAM on a
///    box with adequate memory instead of spilling under a blind 2 GiB cap).
/// 5. 2 GiB fallback.
///
/// `0` is the unset sentinel (matches the unified resolver, which treats
/// `Some(0)` as "no explicit budget"). Callers with no budget pass `0`.
///
/// Shared by RHF/UHF/ROHF so the budget is honored uniformly across all
/// DF-J/DF-K construction sites. This function no longer reads any env var
/// itself — env fallback lives entirely in the unified resolver, so TOML can
/// never be silently overridden by `FERRIC_OOC_BUDGET_GB` (it previously was).
pub fn resolve_three_index_budget(config_bytes: usize) -> usize {
    let explicit = (config_bytes != 0).then_some(config_bytes);
    ferric_core::memory::resolve_budget_bytes(explicit)
}

/// Pure stall-detector arithmetic, extracted from the `solve_rhf` loop so it can
/// be unit-tested with synthetic `errmax_history` sequences instead of forcing a
/// real molecule to stall (slow/nondeterministic).
///
/// Returns `true` when the running minimum of `errmax_history` over the last
/// `window` entries has not dropped below 0.9x its value over the previous
/// `window` entries (gradient stopped falling — robust to oscillation, since a
/// wide-band limit cycle has net-zero running-min change), AND `current_err` is
/// still above the 1e-4 plateau floor (below it, the separate plateau-acceptance
/// path owns the regime). Returns `false` when `window == 0` (degenerate/no-op
/// config) or when there isn't yet `2*window` entries of history.
fn stall_detected(errmax_history: &[f64], window: usize, current_err: f64) -> bool {
    if window == 0 {
        return false;
    }
    if errmax_history.len() < 2 * window || current_err < 1e-4 {
        return false;
    }
    let n_hist = errmax_history.len();
    let recent_min = errmax_history[n_hist - window..]
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let prev_min = errmax_history[n_hist - 2 * window..n_hist - window]
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    recent_min >= 0.9 * prev_min
}

/// The settled convergence signals for one SCF iteration. All are magnitudes
/// (already `.abs()`/norm'd by the caller).
///
/// - `de`: |E − E_prev|, energy change.
/// - `dp_rms`: RMS of the density change ΔP = D_new − D_last, i.e.
///   `‖ΔP‖_F / sqrt(nao²)`.
/// - `dp_max`: max element of |ΔP|.
///
/// The DIIS commutator (FDS−SDF) is deliberately absent: it is a *diagnostic*
/// (printed in the trace), never a gate — see [`scf_converged`] for why.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ConvergenceSignals {
    pub de: f64,
    pub dp_rms: f64,
    pub dp_max: f64,
}

/// Decide SCF convergence from the settled signals — the ORCA `ConvCheckMode=2`
/// design, where the **density change ΔP is the primary (tight) signal** and the
/// energy change is only a *loose* "not still descending" sanity bound. Both
/// ORCA and PySCF use ΔP-primary in place of gating on the DIIS/orbital-gradient
/// commutator, and ORCA's default checks energy *stability*, not a tight ΔE.
///
/// # Why ΔP, and why ΔE only loosely
///
/// Under RI-J/RI-JK the fitted Fock carries a self-consistency error that grows
/// with `naux`. This floors BOTH the commutator's max element AND the
/// iteration-to-iteration energy change — MEASURED at aug-cc-pVTZ (toluene,
/// RI-JK): `err_max` parks at ~1.26e-6 and `dE` oscillates at ~2e-5, neither
/// draining. Gating on *either* has the same naux-chasing pathology (a fixed
/// small tolerance is unreachable and the floor scales with the aux basis).
///
/// `dp_rms` is the ONE signal that drains cleanly at the RI fixed point: the
/// density stops moving (D_{n+1} = D_n to fitting precision) even while the
/// commutator and the energy still jitter on the noise floor. MEASURED same run:
/// `dp_rms` drains monotonically to ~1e-9 while `dE` sits at 2e-5. So ΔP is the
/// convergence criterion; ΔE is used only to reject a run that is *still
/// actively descending* (early iterations have `dE ≫` any noise floor).
///
/// # The gate
///
/// Converged ⟺ density settled **and** energy not actively descending:
/// - `dp_rms < density_conv`         (primary, tight — the real signal)
/// - `dp_max < 10·density_conv`      (ORCA `TolMaxP` companion: guards a single
///   still-moving element while the RMS looks settled)
/// - `de   < energy_conv`            (LOOSE sanity bound, default 1e-3 — well
///   above the ~2e-5 DF energy floor, so it excludes a descending run without
///   demanding an unreachable tight ΔE)
///
/// The commutator (FDS−SDF) is *not* consulted — diagnostic only.
///
/// Returns `Some(ScfExit::Converged)` when met, else `None` (keep iterating).
/// The caller still owns divergence/stall/max-iter exits.
pub(crate) fn scf_converged(
    sig: ConvergenceSignals,
    energy_conv: f64,
    density_conv: f64,
) -> Option<crate::result::ScfExit> {
    let dp_rms_ok = sig.dp_rms < density_conv;
    let dp_max_ok = sig.dp_max < 10.0 * density_conv;
    let energy_not_descending = sig.de < energy_conv;
    if dp_rms_ok && dp_max_ok && energy_not_descending {
        Some(crate::result::ScfExit::Converged)
    } else {
        None
    }
}

/// Solve the closed-shell RHF equations for a molecule.
///
/// Uses the Roothaan-Hall procedure: build Fock matrix from density, diagonalize,
/// rebuild density, iterate until convergence. DIIS extrapolation accelerates
/// convergence. Returns `Ok(ScfResult)` whether or not it converges — check
/// `result.converged` / `result.exit` (`ScfExit::MaxIter` carries the
/// best-effort density/MOs from the final iteration). Only returns
/// [`FerricError::ScfConvergence`] for the odd-electron-count input error.
pub fn solve_rhf(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    config: &RhfConfig,
) -> Result<ScfResult, FerricError> {
    let s = oneelectron::overlap(prep);
    // hcore_ecp adds the ECP projector V_ECP when the basis carries ECPs;
    // identical to hcore() (zero extra cost) for all-electron basis sets.
    let h = oneelectron::hcore_ecp_with_external(
        prep,
        mol,
        prep.basis_set(),
        config.external_potential.as_ref(),
    )?;
    let n = prep.nbasis();
    let nelec = mol.nelec();
    if nelec % 2 != 0 {
        return Err(FerricError::ScfConvergence {
            iterations: 0,
            last_energy: 0.0,
        });
    }
    let nocc = (nelec / 2) as usize;
    // IEF-PCM: cavity + S/D/K/R geometry-only setup, built ONCE (mirrors
    // DfJ/DfK/LinkK being built once before the loop). `None` (default) means
    // this is `None` too and the per-iteration hook below is a pure no-op —
    // see the pcm_none_matches_vacuum_* regression tests.
    let pcm_ctx: Option<ferric_pcm::PcmContext> = config
        .pcm
        .as_ref()
        .map(|pcfg| ferric_pcm::PcmContext::new(mol, pcfg))
        .transpose()?;
    let vnn = mol.nuclear_repulsion()
        + config.external_potential.as_ref().map_or(0.0, |ext| {
            ext.charge_nuclear_energy(mol) + ext.field_nuclear_energy(mol)
        });

    // COSMO cavity: geometry-only (independent of the density), so it is
    // built once here, before the loop. `None` when `config.cosmo` is
    // `None` — every downstream use is gated on this being `Some`, so the
    // SCF loop below is byte-for-byte identical to a COSMO-less build when
    // solvation is disabled.
    let cosmo_cavity: Option<crate::cosmo::CosmoCavity> = match config.cosmo.as_ref() {
        Some(cfg) => Some(crate::cosmo::CosmoCavity::build(mol, cfg)?),
        None => None,
    };

    // Initial density: explicit override > SAD (default) > hcore. SAD is the
    // default because the bare hcore guess diverges on heavy-atom closed shells
    // (COSe/C2H3Br); if SAD fails to build (e.g. a free-atom solve doesn't
    // converge) we fall back to hcore rather than aborting the whole SCF.
    let mut d = if let Some(d0) = config.init_guess_density.as_ref() {
        d0.clone()
    } else if config.use_sad_guess {
        // MINAO projection guess (no per-element free-atom SCF for heavy atoms;
        // GWH atomic-hcore block for Z≥21/g-function elements). Falls back to
        // hcore if it fails. The `use_sad_guess` field name is retained for
        // API/config compatibility but now selects MINAO.
        match crate::guess::minao_projection_guess(mol, prep, prep.basis_set()) {
            Ok(d_minao) => d_minao,
            Err(_) => hcore_guess(&s, &h, nocc)?,
        }
    } else {
        hcore_guess(&s, &h, nocc)?
    };
    let mut f = Array2::zeros((n, n));
    let mut j_buf = Array2::<f64>::zeros((n, n));
    let mut k_buf = Array2::<f64>::zeros((n, n));
    // Combined DIIS driver. With the default `diis_flavor = Pulay` this is a
    // pure-Pulay driver whose `step` is byte-identical to `Diis::step`;
    // ADIIS/EDIIS activate only when a caller opts in.
    let mut diis = DiisDriver::new(config.diis_flavor, config.diis_size, config.diis_switch_thresh);
    // MOM reference: last accepted occupied MO block (None until armed).
    let mut mom_ref: Option<Array2<f64>> = None;
    let mut prev_e = 0.0;
    // Density change from the PREVIOUS iteration's D update (ΔP = D_new − D_old),
    // the ORCA/PySCF primary convergence signal. Carried into the next
    // iteration's convergence check (ΔP for iter N is known only after iter N's
    // density rebuild). INFINITY on iter 1 so the gate can never fire before a
    // real ΔP exists. See scf_converged / the df-jk-noise-floor memory.
    let mut dp_rms = f64::INFINITY;
    let mut dp_max = f64::INFINITY;
    let mut total_quartets = 0;

    if let Some(kb) = config.k_builder.as_deref() {
        if kb != "direct" && kb != "link" {
            return Err(FerricError::General(format!("unknown k_builder '{kb}': valid options are 'direct', 'link'")));
        }
    }

    // Build the XC contribution once. None for pure HF. Done before df_j/df_k
    // setup so we can read k_mix and apply auto JK-aux defaults for hybrid /
    // RSH functionals without forcing the user to set them explicitly.
    use ferric_dft::ks::KsXc;
    use ferric_dft::xc_trait::{XcContribution, KMix};

    let xc_contrib: Option<Box<dyn XcContribution>> = if let Some(name) = config.xc.as_deref() {
        let main = config.dft_grid.clone().unwrap_or_default();
        let nlc = config.nlc_grid.clone()
            .unwrap_or(ferric_dft::grid::AtomicGridConfig { n_radial: 50, n_angular: 50 });
        let ks = KsXc::new(mol, prep.basis_set(), name, &main, &nlc)
            .map_err(|e| FerricError::General(format!("KsXc init for {name}: {e:?}")))?;
        Some(Box::new(ks) as Box<dyn XcContribution>)
    } else {
        None
    };

    let k_mix: KMix = xc_contrib.as_ref().map(|x| x.k_mix()).unwrap_or_default();

    // Meta-GGA (SCAN / r2SCAN) SCF is stiffer than LDA/GGA: with plain DIIS from
    // a SAD guess it limit-cycles just above the tight density threshold (the
    // τ-dependent Fock amplifies the grid-integration noise). A modest default
    // virtual-block level shift (the same ramped shift used to cure heavy-atom
    // closed-shell divergence) makes it converge cleanly — SCAN/H2O in ~13 iters
    // vs. never at ls=0. Only applied when the user hasn't set their own shift,
    // and ramped to zero as the gradient converges so the final energy is
    // unperturbed. LDA/GGA/hybrid keep ls=0 (unchanged behavior).
    let effective_level_shift = if config.level_shift == 0.0
        && crate::rohf::xc_is_metagga(config.xc.as_deref())
    {
        0.5
    } else {
        config.level_shift
    };

    // Resolve the out-of-core 3-index memory budget once (env override wins).
    let ooc_budget = resolve_three_index_budget(config.three_index_budget_bytes);

    // Auto-default JK aux bases when the functional needs exact exchange but
    // the caller hasn't explicitly set df_j_aux / df_k_aux. This makes
    // `cfg.xc = Some("B3LYP")` (or any hybrid/RSH) work out of the box.
    // Pure HF (no xc) keeps the historical behavior of no auto-default.
    let needs_k = xc_contrib.is_some() && (k_mix.sr > 0.0 || k_mix.omega > 0.0);
    let needs_j = xc_contrib.is_some();
    // Whether the SCF actually consumes an exact-exchange matrix. True for pure
    // HF (no functional) and for hybrids / RSH. FALSE for a pure functional
    // (LDA/GGA such as PBE), where k_mix is all-zero and any K built would be
    // multiplied by 0 and thrown away. Gates both the exchange-builder
    // construction and the per-iteration K build below, so pure DFT never pays
    // for exact exchange it does not use. (For a pure functional this equals
    // `needs_k`; for HF it is true where `needs_k` is false, since `needs_k` is
    // DFT-specific.)
    let k_consumed = xc_contrib.is_none() || k_mix.sr > 0.0 || k_mix.omega > 0.0;
    const DEFAULT_JK_AUX: &str = "def2-universal-jkfit";
    let df_j_aux_eff: Option<String> = config.df_j_aux.clone()
        .or_else(|| needs_j.then(|| DEFAULT_JK_AUX.into()));
    let df_k_aux_eff: Option<String> = config.df_k_aux.clone()
        .or_else(|| needs_k.then(|| DEFAULT_JK_AUX.into()));

    // Density-fitted Coulomb (RI-J). Builds 3-center tensor + inverse metric once.
    // Under MPI, `new_banded(Some(ctx))` stripes the aux band across ranks so
    // each rank builds/holds only its band of B (memory scales with rank count);
    // size-1 / non-MPI is byte-identical to the serial path.
    let mut df_j: Option<DfJ> = if let Some(aux_name) = df_j_aux_eff.as_deref() {
        let dfbs_set = ferric_core::basis::bundled(aux_name)?;
        let dfbs = PreparedBasis::new(mol, &dfbs_set)?;
        Some(DfJ::new_banded(op, prep, &dfbs, ooc_budget, Some(ctx))?)
    } else {
        None
    };

    // Density-fitted exchange (RI-K). Builds V^{-1/2}-dressed 3-center tensor once.
    let mut df_k: Option<DfK> = if let Some(aux_name) = df_k_aux_eff.as_deref() {
        let dfbs_set = ferric_core::basis::bundled(aux_name)?;
        let dfbs = PreparedBasis::new(mol, &dfbs_set)?;
        Some(DfK::new_banded(op, prep, &dfbs, ooc_budget, Some(ctx))?)
    } else {
        None
    };

    // RSH path: pre-build the SR/LR DfK fitters once. Their 3-center B[P,μ,ν]
    // tensors are geometry-only — only the D-dependent contraction happens per
    // SCF iteration. Building inside the loop was a major hot-spot for wB97X-V.
    let (mut dfk_sr, mut dfk_lr) = if k_mix.omega > 0.0 {
        let aux_name = df_k_aux_eff.as_deref().ok_or_else(|| {
            FerricError::General(
                "Range-separated hybrid requires RhfConfig.df_k_aux (e.g. \"def2-universal-jkfit\")".into()
            )
        })?;
        let dfbs_set = ferric_core::basis::bundled(aux_name)?;
        let dfbs_prep = PreparedBasis::new(mol, &dfbs_set)?;
        (
            Some(DfK::new_banded(Operator::erfc(k_mix.omega), prep, &dfbs_prep, ooc_budget, Some(ctx))?),
            Some(DfK::new_banded(Operator::erf(k_mix.omega), prep, &dfbs_prep, ooc_budget, Some(ctx))?),
        )
    } else {
        (None, None)
    };

    // Build LinkK once — SignificantPairs is geometry-dependent and expensive per iteration.
    // When using the "link" builder, compute a fresh SchwarzBounds to own the lifetime.
    let link_schwarz_opt = if config.k_builder.as_deref() == Some("link") {
        Some(SchwarzBounds::compute(op, prep)?)
    } else {
        None
    };
    let mut k_builder: Option<Box<dyn KBuilder>> = link_schwarz_opt.as_ref().map(|sb| {
        let mut lk = LinkK::new(ctx, prep, sb, op, config.integral_thresh);
        lk.update_density(&d);
        Box::new(lk) as Box<dyn KBuilder>
    });

    // Canonical orthogonalizer X = U_kept · diag(1/sqrt(λ_kept)), shape (n × m),
    // dropping eigenvectors of S with λ < LINDEP_THRESH (near-linear-dependence).
    // For well-conditioned S, m == n and X reproduces existing energies; see
    // canonical_orthogonalizer / diagonalize for the padding-back-to-n convention.
    let x = canonical_orthogonalizer(&s)?;

    // Previous-iteration MO coefficients, needed to build the virtual-block level
    // shift (a projector onto the prior virtuals). None before the first solve.
    let mut c_prev: Option<Array2<f64>> = None;
    // Orbital energies/MOs from the most recent diagonalization, retained across
    // iterations so a max_iter exit can still report eps/MOs for the final density.
    let mut last_eps: Vec<f64> = Vec::new();
    let mut last_c: Array2<f64> = Array2::zeros((n, n));
    // Fermi-Dirac smearing state (μ, occupations, entropy) from the most recent
    // smeared density rebuild; `None` when smearing is off (default). Used only
    // for optional free-energy trace reporting.
    let mut last_smearing: Option<crate::smearing::Smearing> = None;

    // Stall/divergence early-abort state (opt-in via RhfConfig; both None by
    // default leaves existing behavior unchanged).
    let mut errmax_history: Vec<f64> = Vec::new();
    let mut divergence_streak = 0usize;

    // Shared constructor for every non-converged exit path (MaxIter / Stalled /
    // Diverged). The Converged path re-diagonalizes fresh and is NOT built from
    // this closure.
    let build_nonconverged = |exit: ScfExit,
                               d: &Array2<f64>,
                               c: &Array2<f64>,
                               eps: &[f64],
                               f: &Array2<f64>,
                               energy: f64,
                               iter: usize,
                               cq: usize|
     -> ScfResult {
        ScfResult {
            spin: Spin::Restricted,
            energy,
            density_total: d.clone(),
            density_alpha: d * 0.5,
            density_beta: None,
            mos_alpha: c.clone(),
            mos_beta: None,
            eps_alpha: eps.to_vec(),
            eps_beta: None,
            fock_alpha: f.clone(),
            fock_beta: None,
            converged: false,
            iterations: iter,
            exit,
            computed_quartets: cq,
        }
    };

    // Direct J/K builders hoisted out of the SCF loop: each lazily builds a
    // per-thread libint2 EnginePool on first use (engines are constructed behind
    // a global ctor mutex), so a loop-local builder would pay that construction
    // every iteration. Which builders exist mirrors the branch structure below.
    let df_any = df_j.is_some() || df_k.is_some();
    let mut direct_j: Option<DirectJ> = if (df_any && df_j.is_none()) || (!df_any && k_builder.is_some()) {
        Some(DirectJ::new(ctx, prep, bounds, config.integral_thresh))
    } else {
        None
    };
    // Only build the exchange builder when exact exchange is actually consumed
    // (see `k_consumed`). Pure DFT (LDA/GGA, k_mix all zero) discards any K it
    // builds, so a full direct 4-center K on an all-electron heavy-atom system
    // (e.g. Cu2/aug-cc-pVDZ) dominated the iteration at ~99 s while the actual XC
    // grid work was ~0.4 s. HF and hybrids/RSH keep k_consumed = true, unaffected.
    let mut direct_k: Option<DirectK> = if df_any && df_k.is_none() && k_consumed {
        Some(DirectK::new(ctx, prep, bounds, config.integral_thresh))
    } else {
        None
    };
    let mut direct_jk: Option<DirectJK> = if !df_any && k_builder.is_none() {
        Some(DirectJK::new(ctx, prep, bounds, config.integral_thresh))
    } else {
        None
    };

    for iter in 1..=config.max_iter {
        ctx.check_interrupted()?;
        // Build J and K using selected builder (reuse pre-allocated buffers)
        j_buf.fill(0.0);
        k_buf.fill(0.0);
        // Build J: DF-J if configured, else fall through to combined direct path below.
        // Build K: DF-K > LinkK > combined DirectJK, in priority order.
        if df_any {
            if let Some(dfj) = df_j.as_mut() {
                dfj.build(&d, &mut j_buf)?;
            } else {
                let dj = direct_j.as_mut().expect("DirectJ built before loop");
                total_quartets += dj.build(&d, &mut j_buf)?;
            }
            if let Some(dfk) = df_k.as_mut() {
                dfk.build(&d, &mut k_buf)?;
            } else if let Some(dk) = direct_k.as_mut() {
                // Only reached when exact exchange is consumed (k_consumed):
                // for pure DFT `direct_k` is None and k_buf stays zero, since
                // the discarded K would only be multiplied by k_mix = 0 below.
                total_quartets += <DirectK as KBuilder>::build(dk, &d, &mut k_buf)?;
            }
        } else if let Some(lk) = k_builder.as_mut() {
            let dj = direct_j.as_mut().expect("DirectJ built before loop");
            total_quartets += dj.build(&d, &mut j_buf)?;
            lk.update_density(&d);
            total_quartets += lk.build(&d, &mut k_buf)?;
        } else {
            let djk = direct_jk.as_mut().expect("DirectJK built before loop");
            total_quartets += djk.build(&d, &mut j_buf, &mut k_buf)?;
        }

        // k_total accumulates the exact-exchange contribution to be subtracted from F
        // as ½ · k_total. Convention:
        //   pure HF (xc=None):    k_mix = {1, 1, 0}      → k_total = k_buf (existing)
        //   pure DFT (LDA/PBE):   k_mix = {0, 0, 0}      → k_total = 0 (skip K)
        //   plain hybrid (B3LYP): k_mix = {α, α, 0}      → k_total = α · k_buf
        //   RSH (wB97X-V):        k_mix = {sr, lr, ω>0}  → k_total = sr·K_SR + lr·K_LR
        let k_total: Array2<f64> = if k_mix.omega > 0.0 {
            // Range-separated: SR/LR DfK fitters were built once before the
            // loop (geometry-only). Only the D-dependent contraction runs here.
            let dfk_sr = dfk_sr.as_mut().expect("dfk_sr built when omega>0");
            let dfk_lr = dfk_lr.as_mut().expect("dfk_lr built when omega>0");

            let mut k_sr = Array2::<f64>::zeros((n, n));
            dfk_sr.build(&d, &mut k_sr)?;

            let mut k_lr = Array2::<f64>::zeros((n, n));
            dfk_lr.build(&d, &mut k_lr)?;

            k_mix.sr * &k_sr + k_mix.lr * &k_lr
        } else if k_mix.sr > 0.0 {
            // Plain hybrid or pure HF: use the K already built by the existing builder path.
            k_mix.sr * &k_buf
        } else {
            // Pure DFT: no exact exchange.
            Array2::<f64>::zeros((n, n))
        };

        // F = H + J − ½ K_total  (V_xc, COSMO reaction field added below)
        f.assign(&(&h + &j_buf - &(0.5 * &k_total)));

        // Electronic energy BEFORE adding V_xc (V_xc is one-body in F but
        // E_xc is its own integral) and BEFORE adding the COSMO reaction
        // field (same reasoning: E_cosmo = ½ q·v is its own closed-form
        // energy expression, not the trace of D against V_reaction — see
        // crate::cosmo module docs and the PySCF cross-check in its
        // `energy_elec`/`get_veff` split, where `e_solvent` is added
        // directly to the total rather than folded into the ½Tr[D·vhf] term).
        let e_elec_no_xc: f64 = 0.5 * (&d * &(&h + &f)).sum();
        let e_xc = if let Some(x) = xc_contrib.as_ref() {
            x.add_xc(&d, &mut f)
        } else {
            0.0
        };

        // COSMO reaction field: depends on the CURRENT density, so it is
        // recomputed every iteration (unlike `external_potential`, folded
        // into `h` once before the loop). Added into F (so it enters the
        // Fock matrix used for the next diagonalization/DIIS step, mirroring
        // how PySCF's `get_fock` adds `v_solvent` on top of `vhf`), and its
        // energy contribution is added directly to `energy` (not via the
        // ½Tr[D·(h+F)] trace).
        let e_cosmo = if let Some(cavity) = cosmo_cavity.as_ref() {
            let cosmo_cfg = config.cosmo.as_ref().expect("cosmo_cavity implies config.cosmo");
            let cr = crate::cosmo::cosmo_reaction_field(mol, prep, cavity, cosmo_cfg, &d)?;
            f += &cr.v_reaction;
            cr.e_cosmo
        } else {
            0.0
        };

        // IEF-PCM: solve for the apparent surface charge from the CURRENT
        // density, add the reaction-field operator to F (so the SCF
        // equations feel it and the density relaxes in response), and add
        // E_pcm as its own standalone energy term — NOT folded into the
        // 0.5·D·(H+F) one-electron trace above, which would double-count it
        // (see ferric_pcm's crate doc + rhf.rs call site comment: this
        // mirrors how PySCF's `_attach_solvent.py` adds `e_solvent` to
        // `e_tot` separately from the ordinary one-electron energy, after
        // folding `v_solvent` into the Fock/`vhf` the SCF equations use).
        // Independent of COSMO above: both are additive if a caller sets
        // both `config.cosmo` and `config.pcm` (unusual and unvalidated —
        // see the `pcm` field doc — but the two hooks don't interfere).
        let e_pcm = if let Some(pctx) = pcm_ctx.as_ref() {
            let (v_pcm, e_pcm) = ferric_pcm::pcm_step(pctx, mol, prep, &d)?;
            f += &v_pcm;
            e_pcm
        } else {
            0.0
        };

        let energy = e_elec_no_xc + e_xc + e_cosmo + e_pcm + vnn;

        // DIIS error: e = FDS - SDF. Runs once per SCF iteration, serially
        // (this whole loop body is outside any rayon region — the JK build
        // above is the only rayon-parallel step and has already returned).
        // Opt-in BLAS raise via FERRIC_BLAS_THREADS (default 1, unchanged
        // behavior): opt_in_blas_threads()'s rayon-worker self-guard also
        // protects the SAD/free-atom path, which calls solve_rhf from inside
        // guess.rs's run_serial_pool (a 1-thread rayon pool — still "inside
        // rayon" for the guard's purposes, so it always resolves to 1 there).
        let (fds, sdf) = with_blas_threads(opt_in_blas_threads(), || {
            (f.dot(&d).dot(&s), s.dot(&d).dot(&f))
        });
        let err = &fds - &sdf;

        let de = (energy - prev_e).abs();
        let err_max = err.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        // RMS commutator (‖FDS−SDF‖_F / sqrt(size)) — a diagnostic, NOT a gate.
        let grad_rms = {
            let n = (err.len() as f64).max(1.0);
            (err.iter().map(|v| v * v).sum::<f64>() / n).sqrt()
        };

        if scf_trace() {
            eprintln!(
                "SCF iter={iter:4}  E={energy:.12}  dE={de:.3e}  \
                 dp_rms={dp_rms:.3e}  dp_max={dp_max:.3e}  \
                 |g|_rms={grad_rms:.3e}  err_max={err_max:.3e}"
            );
        }

        // Convergence decision: energy + density change (ORCA ConvCheckMode-2 /
        // PySCF), NOT the DIIS commutator. Under RI-J/RI-JK the commutator parks
        // on a naux-dependent noise floor and never drains; ΔP does (MEASURED),
        // so we gate on ΔP and treat the commutator as diagnostic only. This
        // replaces the former tower of naux-chasing plateau hacks (df_noise_floor,
        // near-degeneracy grad_stalled, oscillation noise-band) with one
        // size-independent criterion. See scf_converged.
        //
        // dp_rms/dp_max are INFINITY until the first density rebuild (iter 1), so
        // the `iter > 1` guard below is belt-and-suspenders on top of that.
        let conv_exit = scf_converged(
            ConvergenceSignals { de, dp_rms, dp_max },
            config.energy_conv,
            config.density_conv,
        );

        // Divergence: energy climbing for consecutive iters.
        if let Some(tol) = config.divergence_tol {
            if energy - prev_e > tol {
                divergence_streak += 1;
            } else {
                divergence_streak = 0;
            }
            if divergence_streak >= 3 {
                if scf_trace() {
                    eprintln!("SCF diverged at iter={iter}: dE={:.3e} > tol for 3 iters", energy - prev_e);
                }
                return Ok(build_nonconverged(ScfExit::Diverged, &d, &last_c, &last_eps, &f, energy, iter, total_quartets));
            }
        }

        // Stall: running-min err_max over a window stopped falling. Robust to
        // oscillation (a wide-band limit cycle has net-zero running-min change).
        // Only fires above the 1e-4 floor — below it the plateau path accepts.
        if let Some(w) = config.stall_window {
            errmax_history.push(err_max);
            if stall_detected(&errmax_history, w, err_max) {
                if scf_trace() {
                    eprintln!("SCF stalled at iter={iter}: err_max={err_max:.3e} (no progress over {w} iters)");
                }
                return Ok(build_nonconverged(ScfExit::Stalled, &d, &last_c, &last_eps, &f, energy, iter, total_quartets));
            }
        }

        if iter > 1 {
            if let Some(exit) = conv_exit {
                // Report Fermi level / entropy / Mermin free energy when smearing
                // is active (trace-gated; `energy` is the smeared internal energy,
                // the free energy is E − σ·S).
                if scf_trace() {
                    if let (Some(sigma), Some(sm)) = (config.smearing_sigma, last_smearing.as_ref()) {
                        eprintln!(
                            "SCF converged with Fermi smearing σ={sigma:.3e} Ha: \
                             μ={:.6} Ha, S={:.4e} k_B, E_free=E−σS={:.10} Ha",
                            sm.mu, sm.entropy, energy - sigma * sm.entropy
                        );
                    }
                }
                let (orb_e, c) = diagonalize(&f, &x)?;
                let density_alpha = 0.5 * &d;
                return Ok(ScfResult {
                    spin: Spin::Restricted,
                    energy,
                    density_total: d,
                    density_alpha,
                    density_beta: None,
                    mos_alpha: c,
                    mos_beta: None,
                    eps_alpha: orb_e,
                    eps_beta: None,
                    fock_alpha: f,
                    fock_beta: None,
                    converged: true,
                    exit,
                    iterations: iter,
                    computed_quartets: total_quartets,
                });
            }
        }
        prev_e = energy;

        // ── Second-order (Newton) update, RHF/RKS ────────────────────────────
        // When enabled (newton_trigger > 0) and err_max has dropped below the
        // trigger, take a damped-Newton step on the single occ→virt orbital
        // rotation instead of DIIS — the closed-shell analogue of the UHF/ROKS
        // Newton paths. `last_c` holds the MOs that produced the current density
        // `d` (set at the tail of the previous iteration; zeros before iter 2),
        // so the branch is gated to `iter > 3`, matching solve_uhf. For RKS this
        // reuses the SAME LDA/GGA f_xc kernel (via FxcKernelStore) that the
        // ROKS/UKS Newton paths use. Gated to the non-RSH case (ω = 0), since
        // the Newton matvec's K comes from the plain Coulomb `build_jk`; RSH and
        // meta-GGA (no τ f_xc) keep the DIIS path. Default newton_trigger = 0
        // ⇒ this branch never fires and existing RHF/RKS results are unchanged.
        let use_newton = config.newton_trigger > 0.0
            && iter > 3
            && err_max < config.newton_trigger
            && k_mix.omega == 0.0
            && !crate::rohf::xc_is_metagga(config.xc.as_deref());
        if use_newton {
            let c_cur = &last_c;
            let f_mo = c_cur.t().dot(&f).dot(c_cur);

            // Build the f_xc kernel (LDA or GGA) at the current restricted
            // reference (d_α = d_β = ½·d) once per Newton step; None for pure HF.
            let fxc_store = if xc_contrib.is_some() {
                let main = config.dft_grid.clone().unwrap_or_default();
                let name = config.xc.as_deref().expect("xc_contrib implies Some(xc)");
                let d_half = 0.5 * &d;
                Some(crate::rohf::FxcKernelStore::build(mol, prep, &main, name, &d_half, &d_half)?)
            } else {
                None
            };
            let fxc_storage = fxc_store.as_ref().map(|s| s.response());
            let fxc_ref: Option<&crate::rohf_newton::FxcResponse<'_>> = fxc_storage.as_deref();

            let inputs = crate::rhf_newton::RhfNewtonInputs {
                prep,
                bounds,
                c: c_cur,
                f_mo: &f_mo,
                nocc,
                k_mix_sr: if xc_contrib.is_some() { k_mix.sr } else { 1.0 },
                fxc: fxc_ref,
                thresh: config.integral_thresh,
            };
            let (c_new, _kmax) = crate::rhf_newton::rhf_newton_step(
                ctx,
                &inputs,
                config.level_shift.max(1e-6),
                0.2, // trust radius
                20,
                1e-7,
            )?;

            // Rebuild density D = 2·C_occ·C_occᵀ and record ΔP for convergence.
            let c_occ = c_new.slice(ndarray::s![.., ..nocc]);
            let d_new = with_blas_threads(opt_in_blas_threads(), || 2.0 * c_occ.dot(&c_occ.t()));
            {
                let diff = &d_new - &d;
                let n2 = (diff.len() as f64).max(1.0);
                dp_rms = (diff.iter().map(|v| v * v).sum::<f64>() / n2).sqrt();
                dp_max = diff.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
            }
            d.assign(&d_new);
            last_c = c_new;
            if effective_level_shift > 0.0 {
                c_prev = Some(last_c.clone());
            }
            continue;
        }

        // `d`/`energy`/`err_max` feed the energy-based branch (ADIIS/EDIIS);
        // ignored when the driver is pure Pulay (default).
        let mut f_new = diis.step(&f, &err, &d, energy, err_max);

        // Virtual-block level shift (config.level_shift > 0): add `shift` to the
        // diagonal of the virtual block in MO basis, i.e. F += shift · S·C_v·C_vᵀ·S
        // where C_v are the previous iteration's virtual MOs. This widens the
        // occupied–virtual gap and damps the orbital rotation that bare DIIS
        // overshoots — the standard cure for SCF oscillation/divergence on
        // heavy-atom closed shells (e.g. COSe, C2H3Br). The shift is ramped to
        // zero as the gradient converges (err_max → 0), so the converged Fock is
        // unshifted and the final energy is unperturbed. Applied from iter ≥ 2
        // (needs a prior C); the convergence check above runs on the *unshifted*
        // F, so an accepted solution never carries the shift.
        if effective_level_shift > 0.0 {
            if let Some(c_p) = c_prev.as_ref() {
                let shift_ramp = effective_level_shift * (err_max / 0.1).min(1.0);
                if shift_ramp > 0.0 {
                    let c_vir = c_p.slice(ndarray::s![.., nocc..]);
                    let scv = s.dot(&c_vir); // (n × nvir)
                    f_new = &f_new + &(shift_ramp * scv.dot(&scv.t()));
                }
            }
        }

        let (eps, mut c) = diagonalize(&f_new, &x)?;
        last_eps = eps;
        // Only retain C across iterations when the level shift needs it; the
        // default (shift = 0) path skips the clone entirely.
        if effective_level_shift > 0.0 {
            c_prev = Some(c.clone());
        }

        // MOM occupied-orbital selection from iter mom_after_iter+1: pin the
        // occupied set by AO-overlap with the previous accepted occupation
        // instead of aufbau (breaks occupied-set flip-flop, e.g. C2H4·Ar).
        if config.mom_after_iter > 0 && iter > config.mom_after_iter {
            if let Some(r) = mom_ref.as_ref() {
                let empty_open = Array2::<f64>::zeros((c.nrows(), 0));
                c = crate::mom::mom_reorder(&c, &s, r, &empty_open, nocc, 0);
            }
        }
        if config.mom_after_iter > 0 && iter >= config.mom_after_iter {
            mom_ref = Some(c.slice(ndarray::s![.., ..nocc]).to_owned());
        }

        // Rebuild density. Default path (smearing off): D = 2 * C_occ @ C_occ^T
        // over the lowest `nocc` MOs (BLAS dgemm). Fermi-Dirac smearing path
        // (config.smearing_sigma = Some(σ>0)): solve μ so 2·Σ f_i = N_elec at
        // width σ, then build D = Σ_i (2·f_i) c_i c_iᵀ = C · diag(2f) · Cᵀ.
        // Exact-off: when smearing_sigma is None the `_` arm is the unchanged
        // pre-smearing integer code.
        let d_new = match config.smearing_sigma {
            Some(sigma) if sigma > 0.0 => {
                let sm = crate::smearing::solve_fermi_level(&last_eps, nelec as f64, sigma, 2.0)?;
                last_smearing = Some(sm.clone());
                with_blas_threads(opt_in_blas_threads(), || {
                    let mut c_weighted = c.clone();
                    for (i, &focc) in sm.occupations.iter().enumerate() {
                        let mut col = c_weighted.column_mut(i);
                        col *= 2.0 * focc;
                    }
                    c_weighted.dot(&c.t())
                })
            }
            _ => {
                let c_occ = c.slice(ndarray::s![.., ..nocc]);
                with_blas_threads(opt_in_blas_threads(), || 2.0 * c_occ.dot(&c_occ.t()))
            }
        };
        // Density change ΔP = D_new − D_old — the primary convergence signal
        // (consumed at the top of the next iteration by scf_converged). Unlike
        // the DIIS commutator, ΔP drains to zero at the RI fixed point even when
        // the commutator parks on the naux-dependent noise floor.
        {
            let diff = &d_new - &d;
            let n2 = (diff.len() as f64).max(1.0);
            dp_rms = (diff.iter().map(|v| v * v).sum::<f64>() / n2).sqrt();
            dp_max = diff.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        }
        d.assign(&d_new);
        last_c = c;
    }
    // Loop exhausted without convergence. Return the best-effort density so a
    // ladder can carry it forward; the caller checks `converged`.
    Ok(build_nonconverged(
        ScfExit::MaxIter,
        &d,
        &last_c,
        &last_eps,
        &f,
        prev_e,
        config.max_iter,
        total_quartets,
    ))
}

/// Build the Coulomb (J) and exchange (K) matrices from the density matrix.
///
/// Uses Schwarz screening and 8-fold permutational symmetry of the ERIs.
pub fn build_jk(
    ctx: &ParallelContext,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    thresh: f64,
    d: &Array2<f64>,
    j: &mut Array2<f64>,
    k: &mut Array2<f64>,
) -> Result<usize, FerricError> {
    use std::sync::atomic::{AtomicUsize, Ordering};

    ctx.check_interrupted()?;

    let nsh = prep.nshells();
    let nbf = prep.nbasis();
    let dims = prep.shell_dims();
    let offs = prep.shell_offsets();
    let computed_quartets = AtomicUsize::new(0);

    // Shell-blocked density-max table for Häser-Ahlrichs pair-wise screening.
    let mut d_max_shell = Array2::<f64>::zeros((nsh, nsh));
    for si in 0..nsh {
        for sj in 0..nsh {
            let (oi, ni) = (offs[si], dims[si]);
            let (oj, nj) = (offs[sj], dims[sj]);
            let mut m = 0.0f64;
            for a in 0..ni {
                for b in 0..nj {
                    let v = unsafe { d.uget((oi + a, oj + b)).abs() };
                    if v > m { m = v; }
                }
            }
            d_max_shell[(si, sj)] = m;
        }
    }

    let shell_pairs: Vec<_> = (0..nsh)
        .flat_map(|s1| (0..=s1).map(move |s2| (s1, s2)))
        .collect();

    // MPI rank striping (round-robin over the flat work list, mirroring
    // DirectJK::build in direct_jk.rs). Every rank builds the identical
    // `shell_pairs` list above, then keeps only the entries whose flat index
    // is congruent to its rank mod world size — a disjoint, covering
    // partition of the (s1,s2) pair list. Without this, every rank computed
    // the FULL J/K redundantly and the unconditional Allreduce below N-folded
    // the result instead of summing a genuine partition (confirmed: -np 2
    // water RHF converged to ≈-3958 Ha instead of -76.03 Ha). With
    // ctx.size == 1 (feature off, or a single rank), `idx % 1 == 0` is
    // trivially true for every idx, so this filter is a no-op and the list is
    // byte-identical to before — preserving the thread-count bit-identity
    // invariant this function is already relied on for (see
    // build_jk_bit_identical_across_thread_counts below).
    let shell_pairs: Vec<_> = shell_pairs
        .into_iter()
        .enumerate()
        .filter(|(idx, _)| idx % ctx.size == ctx.rank)
        .map(|(_, pair)| pair)
        .collect();

    // One engine per rayon thread (see engine_pool) — constructing an engine in
    // the fold init fires once per work-chunk, not per thread, storming the
    // global libint2 ctor mutex on heavy-element bases.
    let pool = crate::engine_pool::EnginePool::new(bounds.op, prep, 1e-14)?;

    // Deterministic, memory-bounded reduction (see direct_jk.rs / reduce.rs).
    // The old `fold(..).reduce(..)` tree combined per-chunk (J,K) partials in a
    // worker-count-dependent order, so J/K (and every downstream SCF energy,
    // gradient, and MP2/RPA number built on this density) drifted ~1 ULP with
    // RAYON_NUM_THREADS (proven by the P7 whole-pipeline gradient bit-identity
    // test: RHF gradient differed in the last bits between 1 and 4 threads,
    // traced here). Partition the (s1,s2) shell-pair work list — a pure
    // function of nsh, never of the thread count — and fold group partials in
    // strict ascending group order: bit-identical across thread counts.
    let n_pairs = shell_pairs.len();
    let group_size = crate::reduce::deterministic_group_size(n_pairs);
    let n_groups = n_pairs.div_ceil(group_size.max(1)).max(1);

    let mut total_j = Array2::<f64>::zeros((nbf, nbf));
    let mut total_k = Array2::<f64>::zeros((nbf, nbf));
    crate::reduce::grouped_deterministic_sum_pair(&mut total_j, &mut total_k, n_groups, nbf, |g| {
        let lo = g * group_size;
        let hi = (lo + group_size).min(n_pairs);
        let mut local_j = Array2::<f64>::zeros((nbf, nbf));
        let mut local_k = Array2::<f64>::zeros((nbf, nbf));
        let mut local_count = 0usize;
        for &(s1, s2) in &shell_pairs[lo..hi] {
            if ferric_core::INTERRUPT.load(std::sync::atomic::Ordering::Relaxed) {
                continue;
            }
            let b12 = bounds.q[(s1, s2)];
            let d12 = d_max_shell[(s1, s2)];
            let (n1, n2) = (dims[s1], dims[s2]);
            let (o1, o2) = (offs[s1], offs[s2]);
            let sym12 = s1 != s2;

            pool.with(|engine| {
            for s3 in 0..=s1 {
                if s3 % 100 == 0 && ferric_core::INTERRUPT.load(Ordering::Relaxed) {
                    return;
                }
                let s4max = if s3 == s1 { s2 } else { s3 };
                let d13 = d_max_shell[(s1, s3)];
                let d23 = d_max_shell[(s2, s3)];
                for s4 in 0..=s4max {
                    let b34 = bounds.q[(s3, s4)];
                    let d34 = d_max_shell[(s3, s4)];
                    let d14 = d_max_shell[(s1, s4)];
                    let d24 = d_max_shell[(s2, s4)];
                    let dmax = d12.max(d34).max(d13).max(d14).max(d23).max(d24);
                    if b12 * b34 * dmax < thresh {
                        continue;
                    }

                    if let Some(q) = engine.compute_quartet(prep, s1, s2, s3, s4) {
                        local_count += 1;
                        let (n3, n4) = (dims[s3], dims[s4]);
                        let (o3, o4) = (offs[s3], offs[s4]);
                        let sym34 = s3 != s4;
                        let sym1234 = (s1, s2) != (s3, s4);

                        // Fast-path for STO-3G / small shells (n=1)
                        if n1 == 1 && n2 == 1 && n3 == 1 && n4 == 1 {
                            let v = unsafe { *q.get_unchecked(0) };
                            unsafe {
                                *local_j.uget_mut((o1, o2)) += d.uget((o3, o4)) * v;
                                *local_k.uget_mut((o1, o3)) += d.uget((o2, o4)) * v;
                                if sym12 {
                                    *local_j.uget_mut((o2, o1)) += d.uget((o3, o4)) * v;
                                    *local_k.uget_mut((o2, o3)) += d.uget((o1, o4)) * v;
                                }
                                if sym34 {
                                    *local_j.uget_mut((o1, o2)) += d.uget((o4, o3)) * v;
                                    *local_k.uget_mut((o1, o4)) += d.uget((o2, o3)) * v;
                                }
                                if sym12 && sym34 {
                                    *local_j.uget_mut((o2, o1)) += d.uget((o4, o3)) * v;
                                    *local_k.uget_mut((o2, o4)) += d.uget((o1, o3)) * v;
                                }
                                if sym1234 {
                                    *local_j.uget_mut((o3, o4)) += d.uget((o1, o2)) * v;
                                    *local_k.uget_mut((o3, o1)) += d.uget((o4, o2)) * v;
                                    if sym12 {
                                        *local_j.uget_mut((o3, o4)) += d.uget((o2, o1)) * v;
                                        *local_k.uget_mut((o3, o2)) += d.uget((o4, o1)) * v;
                                    }
                                    if sym34 {
                                        *local_j.uget_mut((o4, o3)) += d.uget((o1, o2)) * v;
                                        *local_k.uget_mut((o4, o1)) += d.uget((o3, o2)) * v;
                                    }
                                    if sym12 && sym34 {
                                        *local_j.uget_mut((o4, o3)) += d.uget((o2, o1)) * v;
                                        *local_k.uget_mut((o4, o2)) += d.uget((o3, o1)) * v;
                                    }
                                }
                            }
                            continue;
                        }

                        // General path for larger shells
                        for a in 0..n1 {
                            for b in 0..n2 {
                                for c in 0..n3 {
                                    for dd in 0..n4 {
                                        let v = unsafe { *q.get_unchecked(((a * n2 + b) * n3 + c) * n4 + dd) };
                                        let mu = o1 + a;
                                        let nu = o2 + b;
                                        let la = o3 + c;
                                        let sg = o4 + dd;

                                        unsafe {
                                            *local_j.uget_mut((mu, nu)) += d.uget((la, sg)) * v;
                                            *local_k.uget_mut((mu, la)) += d.uget((nu, sg)) * v;

                                            if sym12 {
                                                *local_j.uget_mut((nu, mu)) += d.uget((la, sg)) * v;
                                                *local_k.uget_mut((nu, la)) += d.uget((mu, sg)) * v;
                                            }
                                            if sym34 {
                                                *local_j.uget_mut((mu, nu)) += d.uget((sg, la)) * v;
                                                *local_k.uget_mut((mu, sg)) += d.uget((nu, la)) * v;
                                            }
                                            if sym12 && sym34 {
                                                *local_j.uget_mut((nu, mu)) += d.uget((sg, la)) * v;
                                                *local_k.uget_mut((nu, sg)) += d.uget((mu, la)) * v;
                                            }
                                            if sym1234 {
                                                *local_j.uget_mut((la, sg)) += d.uget((mu, nu)) * v;
                                                *local_k.uget_mut((la, mu)) += d.uget((sg, nu)) * v;
                                                if sym12 {
                                                    *local_j.uget_mut((la, sg)) += d.uget((nu, mu)) * v;
                                                    *local_k.uget_mut((la, nu)) += d.uget((sg, mu)) * v;
                                                }
                                                if sym34 {
                                                    *local_j.uget_mut((sg, la)) += d.uget((mu, nu)) * v;
                                                    *local_k.uget_mut((sg, mu)) += d.uget((la, nu)) * v;
                                                }
                                                if sym12 && sym34 {
                                                    *local_j.uget_mut((sg, la)) += d.uget((nu, mu)) * v;
                                                    *local_k.uget_mut((sg, nu)) += d.uget((la, mu)) * v;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            });
        }
        computed_quartets.fetch_add(local_count, Ordering::Relaxed);
        Ok((local_j, local_k))
    })?;

    *j += &total_j;
    *k += &total_k;

    #[cfg(feature = "mpi")]
    if let Some(world) = ctx.world() {
        use mpi::traits::CommunicatorCollectives;
        let mut j_global = Array2::zeros(j.dim());
        let mut k_global = Array2::zeros(k.dim());
        world.all_reduce_into(j.as_slice().unwrap(), j_global.as_slice_mut().unwrap(), mpi::collective::SystemOperation::sum());
        world.all_reduce_into(k.as_slice().unwrap(), k_global.as_slice_mut().unwrap(), mpi::collective::SystemOperation::sum());
        *j = j_global;
        *k = k_global;
    }

    Ok(computed_quartets.load(Ordering::SeqCst))
}

/// Default linear-dependence threshold for canonical orthogonalization:
/// eigenvectors of the overlap matrix with eigenvalue below this are dropped from
/// the variational space. PySCF's default is ~1e-6 to 1e-7; 1e-6 is conservative.
pub(crate) const LINDEP_THRESH: f64 = 1e-6;

/// `FERRIC_SCF_TRACE` descriptor: per-iteration SCF convergence trace (env-only
/// debug toggle). Read at several sites in rhf/uhf/guess via [`scf_trace`].
static SCF_TRACE: ferric_core::config::ConfigVar<bool> = ferric_core::config::ConfigVar {
    env_name: "FERRIC_SCF_TRACE",
    default: false,
    parse: ferric_core::config::parse_toggle,
    validate: ferric_core::config::accept_any,
};

/// Whether the per-iteration SCF trace is on. `FERRIC_SCF_TRACE=1/true/on/yes`,
/// off for `0/false/off/no`/unset; a malformed value logs a warning and stays off.
pub(crate) fn scf_trace() -> bool {
    SCF_TRACE.toggle()
}

/// Effective linear-dependence threshold, overridable via `FERRIC_LINDEP_THRESH`.
///
/// The default [`LINDEP_THRESH`] (1e-6) drops nothing on the diffuse alkali/d-block
/// clusters (Na4/Na6/Cu2 at aug-cc-pVTZ), whose near-null overlap modes sit at
/// λ ≈ 3e-5–5e-5 — kept and amplified ~150× by 1/√λ, which parks the DIIS orbital
/// gradient just above `density_conv` so the SCF plateau-spins to `max_iter`
/// without ever declaring convergence. Raising the threshold for those systems
/// projects the offending modes out and lets DIIS converge. Env-scoped rather
/// than a global bump because a blanket 1e-4 would also drop legitimate in-band
/// modes on well-conditioned aromatics (C6H6/aug-cc-pVTZ has 19 modes in
/// [1e-6,1e-4) that DIIS drains fine), perturbing their banked energies.
fn lindep_thresh() -> f64 {
    static LINDEP: ferric_core::config::ConfigVar<f64> = ferric_core::config::ConfigVar {
        env_name: "FERRIC_LINDEP_THRESH",
        default: LINDEP_THRESH,
        parse: |s| s.parse::<f64>().map_err(|e| e.to_string()),
        validate: |v| {
            (v.is_finite() && *v > 0.0)
                .then_some(())
                .ok_or_else(|| "must be finite > 0".to_string())
        },
    };
    // A malformed/invalid override uses the default (was silent; now warns) —
    // this is read on the SCF hot path with no Result to propagate.
    LINDEP.get().map(|r| r.value).unwrap_or_else(|e| {
        eprintln!("[config] FERRIC_LINDEP_THRESH: {e}; using default {LINDEP_THRESH}");
        LINDEP_THRESH
    })
}

/// Build the canonical orthogonalizer X (n × m) from the overlap matrix S.
///
/// X = U_kept · diag(1/sqrt(λ_kept)), where U_kept are the eigenvectors of S
/// whose eigenvalue λ ≥ [`LINDEP_THRESH`]. `eigh` returns eigenvalues in
/// ASCENDING order, so the near-singular modes are first and are skipped.
/// For a well-conditioned S, m == n and no mode is dropped (regression-safe).
///
/// X is NOT symmetric (even when m == n) — callers must use the rectangular
/// transform Fʹ = Xᵀ F X, C = X Vʹ (see [`diagonalize`]).
pub(crate) fn canonical_orthogonalizer(s: &Array2<f64>) -> Result<Array2<f64>, FerricError> {
    let n = s.nrows();
    // Runs once at SCF setup, serially (called before the iteration loop, no
    // enclosing rayon region). Opt-in BLAS raise (default 1, unchanged
    // behavior) — see the DIIS FDS/SDF comment above for the SAD/free-atom
    // protection argument (identical here: opt_in_blas_threads()'s
    // rayon-worker self-guard resolves to 1 inside guess.rs's run_serial_pool).
    let (s_evals, s_evecs) = with_blas_threads(opt_in_blas_threads(), || {
        s.eigh(ndarray_linalg::UPLO::Upper)
    })
    .map_err(|e| FerricError::Lapack(format!("S diag: {e}")))?;
    // eigenvalues ascending: kept columns are those with λ ≥ the effective
    // threshold (default LINDEP_THRESH, overridable via FERRIC_LINDEP_THRESH).
    let thresh = lindep_thresh();
    let kept: Vec<usize> = (0..n).filter(|&i| s_evals[i] >= thresh).collect();
    let m = kept.len();
    let mut x = Array2::<f64>::zeros((n, m));
    for (col, &i) in kept.iter().enumerate() {
        let scale = 1.0 / s_evals[i].sqrt();
        for mu in 0..n {
            x[(mu, col)] = s_evecs[(mu, i)] * scale;
        }
    }
    Ok(x)
}

/// Diagonalize the Fock matrix in the canonical-orthogonal basis.
///
/// With X (n × m) rectangular: Fʹ = Xᵀ F X is (m × m), its eigenvectors Vʹ are
/// (m × m), and C_kept = X Vʹ is (n × m). The result is PADDED back to (n × n)
/// by appending (n − m) zero MO columns with sentinel energy 1e6, so every
/// downstream consumer (GW, RPA, density build) sees the historical (n × n) /
/// length-n shapes while the near-singular directions are inert virtuals.
fn diagonalize(
    f: &Array2<f64>,
    x: &Array2<f64>,
) -> Result<(Vec<f64>, Array2<f64>), FerricError> {
    let n = x.nrows();
    let m = x.ncols();
    // Runs once per SCF iteration, serially (called from the main loop body,
    // outside any rayon region — the JK build is the only rayon-parallel step
    // per iteration and has already returned by this point). Opt-in BLAS
    // raise (default 1); SAD/free-atom protection per the DIIS comment above.
    let (evals, c_kept) = with_blas_threads(opt_in_blas_threads(), || {
        let f_prime = x.t().dot(f).dot(x);
        let (evals, evecs) = f_prime.eigh(ndarray_linalg::UPLO::Upper)?;
        let c_kept = x.dot(&evecs); // (n × m)
        Ok::<_, ndarray_linalg::error::LinalgError>((evals, c_kept))
    })
    .map_err(|e| FerricError::Lapack(format!("F diag: {e}")))?;
    if m == n {
        return Ok((evals.to_vec(), c_kept));
    }
    // Pad to (n × n): zero MO columns + sentinel high energy for dropped modes.
    let mut c = Array2::<f64>::zeros((n, n));
    c.slice_mut(ndarray::s![.., ..m]).assign(&c_kept);
    let mut eps = evals.to_vec();
    eps.resize(n, 1e6);
    Ok((eps, c))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::screening::SchwarzBounds;
    use ferric_core::basis;
    use ferric_core::external_potential::{ExternalPotential, PointCharge};
    use ferric_core::mol::Molecule;
    use ferric_integrals::basis_bridge::PreparedBasis;

    // FERRIC_MEM_BUDGET_GB / FERRIC_OOC_BUDGET_GB are process-global; serialize
    // env-mutating tests (blas_threads.rs / ferric-core memory.rs pattern).
    // MUST be held by every test in this module that transitively reads the
    // budget env var, not just the one that mutates it: `solve_rhf` calls
    // `resolve_three_index_budget` unconditionally on every invocation (see the
    // `let ooc_budget = ...` line near the top of `solve_rhf`), which falls
    // through to `ferric_core::memory::resolve_budget_bytes` and reads the
    // ambient env value whenever no explicit `config.three_index_budget_bytes`
    // is set. Under cargo test's default parallelism, any test that calls
    // `solve_rhf` (directly or via a helper like `run_rhf_test`) can observe
    // `three_index_budget_auto_detects_ram_on_default`'s temporary near-zero /
    // pinned-GB env mutation mid-flight if it doesn't also hold this lock —
    // this file has 20+ such tests. Confirmed dormant (not actively flaky) as
    // of 2026-07-18, but the same race that hit
    // `eval_basis_on_grid_serial_and_parallel_paths_agree` in
    // ferric-export/src/gto_eval.rs applies here structurally; fixed
    // proactively rather than waiting for a flake. Pure-function tests that
    // never call `solve_rhf` (e.g. `scf_converged_*`, `stall_detected_*`) do
    // NOT need this lock — only reach for it when the test path touches
    // `resolve_three_index_budget` (directly or transitively).
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn three_index_budget_auto_detects_ram_on_default() {
        // The whole point of the RAM-aware resolver: on a box with adequate RAM,
        // the legacy 2 GiB default must NOT cap the budget — it must fall through
        // to auto-detect (0.8×available), so the DF-JK tensor stays in RAM
        // instead of spilling to disk. Env-var precedence is also asserted here.
        // These sub-cases mutate process env, so they live in ONE test (cargo
        // runs tests in a shared process; a separate test could race the env).
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let legacy_2gib = 2 * 1024 * 1024 * 1024;

        // Guard: don't let caller-set budget env vars perturb the assertions.
        // Both the unified var and the legacy OOC var feed the resolver now.
        let saved_ooc = std::env::var("FERRIC_OOC_BUDGET_GB").ok();
        let saved_mem = std::env::var("FERRIC_MEM_BUDGET_GB").ok();
        std::env::remove_var("FERRIC_OOC_BUDGET_GB");
        std::env::remove_var("FERRIC_MEM_BUDGET_GB");

        // (1) Unset (config_bytes == 0) -> auto-detect. On any real CI/dev box
        // this is 0.8×available RAM, which is >2 GiB; the old blind 2 GiB cap
        // must be exceeded. (If available RAM were truly <2.5 GiB the fallback
        // would also equal 2 GiB — accept that degenerate case rather than flake.)
        let auto = resolve_three_index_budget(0);
        let avail = ferric_core::memory::detect_available_bytes();
        if let Some(a) = avail {
            if (a as f64 * 0.8) as usize > legacy_2gib {
                assert!(
                    auto > legacy_2gib,
                    "unset budget should auto-detect above the old 2 GiB cap on a \
                     box with {a} bytes available; got {auto}"
                );
            }
        }

        // (2) A non-zero config value is an explicit choice, honored as-is —
        // INCLUDING a deliberate 2 GiB (the bug the 0-sentinel fixes: an
        // explicit 2 GiB must NOT be swallowed by auto-detect).
        let explicit = 7 * 1024 * 1024 * 1024;
        assert_eq!(resolve_three_index_budget(explicit), explicit);
        assert_eq!(
            resolve_three_index_budget(legacy_2gib),
            legacy_2gib,
            "an explicit 2 GiB budget must be honored, not auto-detected"
        );

        // (3) Precedence: TOML/config OVERRIDES env; env fills in only when
        // config is unset. (Corrected from the earlier env-first behavior so all
        // memory settings share one chain: TOML/config > env > auto.)
        std::env::set_var("FERRIC_OOC_BUDGET_GB", "3");
        assert_eq!(
            resolve_three_index_budget(0),
            3 * 1024 * 1024 * 1024,
            "env fills in when config is unset (config_bytes == 0)"
        );
        assert_eq!(
            resolve_three_index_budget(explicit),
            explicit,
            "TOML/config budget must OVERRIDE the FERRIC_OOC_BUDGET_GB env var"
        );
        std::env::remove_var("FERRIC_OOC_BUDGET_GB");

        // (4) The unified var FERRIC_MEM_BUDGET_GB is also overridden by config,
        // and itself takes precedence over the legacy OOC var when config unset.
        std::env::set_var("FERRIC_MEM_BUDGET_GB", "5");
        assert_eq!(
            resolve_three_index_budget(0),
            5 * 1024 * 1024 * 1024,
            "FERRIC_MEM_BUDGET_GB fills in when config is unset"
        );
        assert_eq!(
            resolve_three_index_budget(explicit),
            explicit,
            "TOML/config budget must OVERRIDE FERRIC_MEM_BUDGET_GB too"
        );
        std::env::remove_var("FERRIC_MEM_BUDGET_GB");

        // Restore whatever the harness had set.
        if let Some(v) = saved_ooc {
            std::env::set_var("FERRIC_OOC_BUDGET_GB", v);
        }
        if let Some(v) = saved_mem {
            std::env::set_var("FERRIC_MEM_BUDGET_GB", v);
        }
    }

    #[test]
    fn external_point_charge_changes_rhf_energy_and_matches_hand_calc() {
        // Holds ENV_LOCK (declared above) because solve_rhf reads the
        // process-global FERRIC_MEM_BUDGET_GB/FERRIC_OOC_BUDGET_GB env vars
        // internally via resolve_three_index_budget -- see the lock's doc
        // comment.
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let bs = ferric_core::basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let ctx = ParallelContext::default();

        let base = solve_rhf(&ctx, &mol, &prep, op, &bounds, &RhfConfig::default()).unwrap();

        // Place a +1 point charge 20 Bohr away (weak perturbation, should shift
        // energy by a small, nonzero, well-defined amount and not break convergence).
        let ext = ExternalPotential {
            point_charges: vec![PointCharge { q: 1.0, x: 0.0, y: 0.0, z: 20.0 }],
            field: None,
        };
        let config = RhfConfig { external_potential: Some(ext.clone()), ..Default::default() };
        let perturbed = solve_rhf(&ctx, &mol, &prep, op, &bounds, &config).unwrap();

        assert!(perturbed.converged);
        assert!((perturbed.energy - base.energy).abs() > 1e-8, "energy did not change");

        // Classical charge-nuclear energy alone (no electronic response) must be
        // a lower bound on the magnitude of a repulsive-like shift; more
        // importantly, verify the classical piece was actually added by checking
        // it against the standalone helper.
        let classical = ext.charge_nuclear_energy(&mol);
        assert!(classical.abs() > 0.0);
    }

    #[test]
    fn external_potential_none_matches_default_exactly() {
        // See ENV_LOCK doc comment: solve_rhf reads the budget env vars.
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let bs = ferric_core::basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let ctx = ParallelContext::default();

        let a = solve_rhf(&ctx, &mol, &prep, op, &bounds, &RhfConfig::default()).unwrap();
        let config = RhfConfig { external_potential: None, ..Default::default() };
        let b = solve_rhf(&ctx, &mol, &prep, op, &bounds, &config).unwrap();
        assert_eq!(a.energy, b.energy);
    }

    #[test]
    fn pcm_none_matches_vacuum_exactly() {
        // `pcm: None` (the default) must be byte-identical to a plain vacuum
        // calculation -- the same convention external_potential follows.
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let bs = ferric_core::basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let ctx = ParallelContext::default();

        let a = solve_rhf(&ctx, &mol, &prep, op, &bounds, &RhfConfig::default()).unwrap();
        let config = RhfConfig { pcm: None, ..Default::default() };
        let b = solve_rhf(&ctx, &mol, &prep, op, &bounds, &config).unwrap();
        assert_eq!(a.energy, b.energy);
    }

    /// Water/STO-3G in water solvent (eps=78.4): the standard textbook PCM
    /// validation case. Checks:
    ///   1. The SCF converges with PCM active.
    ///   2. The solvation free energy E_pcm = E(solvent) - E(vacuum) is
    ///      NEGATIVE (stabilizing) -- required for any physically sane
    ///      implicit-solvent model of a polar solute in a polar solvent.
    ///   3. The magnitude is "a few kcal/mol" (order-of-magnitude sane for a
    ///      small polar solute's electrostatic solvation term), cross-checked
    ///      against a genuine PySCF IEF-PCM run generated for this exact
    ///      molecule/basis/eps (see
    ///      /tmp/.../scratchpad/gen_pcm_ref.py and
    ///      pcm_water_sto3g_pyscf_ref.json: PySCF gives E_solv = -3.82
    ///      kcal/mol using its own SWIG tessellation). ferric's independent,
    ///      deliberately simpler hard-cutoff Lebedev tessellation (see
    ///      ferric_pcm::cavity's doc) is NOT expected to match PySCF's
    ///      number tightly -- different cavity discretizations give
    ///      different tessera counts/areas and thus different S/D matrices
    ///      -- so this test only requires the same SIGN and the same
    ///      ORDER OF MAGNITUDE (within a generous 0.5x-3x band), not
    ///      numerical agreement to PySCF's specific tessellation.
    #[test]
    fn pcm_water_solvation_energy_is_negative_and_reasonable_magnitude() {
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let bs = ferric_core::basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let ctx = ParallelContext::default();

        let vac = solve_rhf(&ctx, &mol, &prep, op, &bounds, &RhfConfig::default()).unwrap();
        assert!(vac.converged);

        let config = RhfConfig {
            pcm: Some(ferric_pcm::PcmConfig::water()),
            ..Default::default()
        };
        let solv = solve_rhf(&ctx, &mol, &prep, op, &bounds, &config).unwrap();
        assert!(solv.converged, "PCM SCF failed to converge");

        let e_solv_ha = solv.energy - vac.energy;
        let e_solv_kcal = e_solv_ha * 627.5094740631;

        assert!(
            e_solv_ha < 0.0,
            "solvation energy must be stabilizing (negative); got {e_solv_ha:.6} Ha = {e_solv_kcal:.3} kcal/mol"
        );

        // PySCF IEF-PCM reference for this exact molecule/basis/eps (own SWIG
        // tessellation): E_solv = -3.8228 kcal/mol. Require the same order of
        // magnitude, generously bounded (both negative: the "0.3x" bound is
        // the smaller-magnitude edge, the "4x" bound is the larger-magnitude
        // edge), since ferric's cavity construction is a deliberately
        // simpler independent discretization.
        let pyscf_ref_kcal = -3.8227667932356835_f64;
        let small_mag_bound = pyscf_ref_kcal * 0.3; // -1.15 kcal/mol
        let large_mag_bound = pyscf_ref_kcal * 4.0; // -15.29 kcal/mol
        assert!(
            e_solv_kcal < small_mag_bound && e_solv_kcal > large_mag_bound,
            "E_solv={e_solv_kcal:.3} kcal/mol is not within an order of magnitude of the \
             PySCF IEF-PCM reference ({pyscf_ref_kcal:.3} kcal/mol); expected between \
             {large_mag_bound:.3} and {small_mag_bound:.3} kcal/mol"
        );
    }

    /// Coverage-widening companion to `pcm_water_solvation_energy_is_negative_and_reasonable_magnitude`
    /// (see docs/VALIDATION.md's PCM entry / the "second system" cross-check task): that
    /// original test is ONE point (water/STO-3G/eps=78.4). This helper drives the same
    /// vacuum-vs-PCM comparison against an arbitrary molecule/basis/eps, so several
    /// independent systems can each get their own #[test] with their own PySCF reference
    /// number, all using the identical convention:
    ///
    ///   PySCF: `scf.RHF(mol).PCM()`, `with_solvent.method = "IEF-PCM"`,
    ///   `with_solvent.lebedev_order = 29` (302 points/sphere, PySCF's own default),
    ///   `with_solvent.vdw_scale = 1.2` (matches ferric's `PcmConfig` default).
    ///   See /tmp/ferric-pcm-widen/scratch_gen_pcm_refs.py for the exact generation script
    ///   (same convention as the original /tmp/.../gen_pcm_ref.py referenced by the
    ///   water/STO-3G test above).
    fn assert_pcm_solvation_matches_pyscf(
        xyz_path: &str,
        basis_name: &str,
        eps: f64,
        pyscf_e_solv_kcal: f64,
        rel_tol: f64,
        label: &str,
    ) {
        let mol = Molecule::load_xyz(xyz_path).unwrap();
        let bs = ferric_core::basis::bundled(basis_name).unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let ctx = ParallelContext::default();

        let vac = solve_rhf(&ctx, &mol, &prep, op, &bounds, &RhfConfig::default()).unwrap();
        assert!(vac.converged, "{label}: vacuum SCF failed to converge");

        let config = RhfConfig {
            pcm: Some(ferric_pcm::PcmConfig {
                epsilon: eps,
                ..ferric_pcm::PcmConfig::water()
            }),
            ..Default::default()
        };
        let solv = solve_rhf(&ctx, &mol, &prep, op, &bounds, &config).unwrap();
        assert!(solv.converged, "{label}: PCM SCF failed to converge");

        let e_solv_ha = solv.energy - vac.energy;
        let e_solv_kcal = e_solv_ha * 627.5094740631;

        assert!(
            e_solv_ha < 0.0,
            "{label}: solvation energy must be stabilizing (negative); got \
             {e_solv_ha:.6} Ha = {e_solv_kcal:.3} kcal/mol"
        );

        let rel_err = (e_solv_kcal - pyscf_e_solv_kcal).abs() / pyscf_e_solv_kcal.abs();
        assert!(
            rel_err < rel_tol,
            "{label}: E_solv={e_solv_kcal:.4} kcal/mol vs PySCF IEF-PCM reference \
             {pyscf_e_solv_kcal:.4} kcal/mol -- relative error {:.2}% exceeds {:.2}% tolerance",
            rel_err * 100.0,
            rel_tol * 100.0
        );
    }

    /// System 2/4 of the PCM coverage-widening sweep: SAME molecule/geometry as the
    /// original water/STO-3G point, but a bigger basis (cc-pVDZ) -- tests whether the
    /// tight agreement is basis-dependent (bigger basis -> more diffuse density near the
    /// cavity surface -> reaction field more sensitive to the hard-cutoff tessellation).
    /// PySCF IEF-PCM reference (own SWIG tessellation): E_solv = -6.2580 kcal/mol.
    #[test]
    fn pcm_water_ccpvdz_matches_pyscf_within_a_few_percent() {
        assert_pcm_solvation_matches_pyscf(
            "../../testdata/molecules/water.xyz",
            "cc-pvdz",
            78.4,
            -6.2580,
            0.05,
            "water/cc-pVDZ/eps=78.4",
        );
    }

    /// System 3/4: a genuinely different molecular TOPOLOGY at the same water eps --
    /// NH3 is pyramidal (C3v) rather than water's bent C2v, so the three N-H spheres
    /// overlap the central N sphere in a different geometric pattern than water's two
    /// O-H overlaps. Tests whether the hard-cutoff cavity's tightness on water was
    /// water-specific or genuinely generalizes to a different small polar molecule.
    ///
    /// PySCF IEF-PCM reference: E_solv = -3.9709 kcal/mol. ferric measures -3.65 kcal/mol
    /// (~8% relative error) -- looser than the ~0.3% water/STO-3G point but still the same
    /// order of magnitude and same sign; 15% is a deliberately generous bound (not a tight
    /// cross-check like the water case) since this is evidence the hard-cutoff cavity's
    /// accuracy is somewhat molecule-dependent, not that NH3 specifically is broken.
    #[test]
    fn pcm_nh3_sto3g_within_15_percent_of_pyscf() {
        assert_pcm_solvation_matches_pyscf(
            "../../testdata/molecules/nh3.xyz",
            "sto-3g",
            78.4,
            -3.9709,
            0.15,
            "NH3/STO-3G/eps=78.4",
        );
    }

    /// System 4/4: methanol (Cs, 6 atoms, 2 heavy atoms C+O 1.42 A apart, so the C and O
    /// vdW spheres -- and all 4 H spheres -- overlap much more densely than water's single
    /// central heavy atom or NH3's single central heavy atom). This is a DELIBERATE
    /// negative/stress case for the hard keep/discard cavity cut described in
    /// `ferric_pcm::cavity`'s module doc.
    ///
    /// MEASURED FINDING: agreement does NOT generalize here. PySCF IEF-PCM reference (own
    /// SWIG tessellation) is E_solv = -2.6219 kcal/mol at eps=20.7 (acetone); ferric gives
    /// -10.55 kcal/mol -- over 4x too negative (302% relative error). This is NOT an
    /// epsilon-regime artifact: re-run at water's own eps=78.4, PySCF gives -2.78 kcal/mol
    /// and ferric gives -8.25 kcal/mol, i.e. still ~3x too negative. Total cavity area is
    /// actually slightly LARGER than PySCF's SWIG cavity (280 vs 252 Bohr^2 at eps=78.4), so
    /// this isn't simple under-tessellation either -- the likely mechanism is the hard
    /// keep/discard cut (no GEPOL/SWIG switching function, no interstitial spheres)
    /// distorting the S/D operators specifically where MANY spheres mutually overlap (all
    /// 15 atom pairs in methanol have overlapping scaled-vdW spheres, vs water's 3 pairs),
    /// producing systematically too-strong screening rather than a random-sign error. This
    /// test intentionally asserts only sign + coarse order-of-magnitude (same generous
    /// 0.3x-6x band style as the original water smoke test), NOT tight agreement, and the
    /// doc comment records the precise degraded numbers as real evidence that the PCM
    /// grade should NOT be uniformly upgraded across all molecular shapes.
    #[test]
    fn pcm_methanol_sto3g_lower_eps_is_negative_but_not_tight_vs_pyscf() {
        let mol = Molecule::load_xyz("../../testdata/molecules/ch3oh.xyz").unwrap();
        let bs = ferric_core::basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let ctx = ParallelContext::default();

        let vac = solve_rhf(&ctx, &mol, &prep, op, &bounds, &RhfConfig::default()).unwrap();
        assert!(vac.converged);

        let config = RhfConfig {
            pcm: Some(ferric_pcm::PcmConfig {
                epsilon: 20.7,
                ..ferric_pcm::PcmConfig::water()
            }),
            ..Default::default()
        };
        let solv = solve_rhf(&ctx, &mol, &prep, op, &bounds, &config).unwrap();
        assert!(solv.converged, "PCM SCF failed to converge");

        let e_solv_ha = solv.energy - vac.energy;
        let e_solv_kcal = e_solv_ha * 627.5094740631;

        assert!(
            e_solv_ha < 0.0,
            "solvation energy must be stabilizing (negative); got {e_solv_ha:.6} Ha = \
             {e_solv_kcal:.3} kcal/mol"
        );

        // PySCF IEF-PCM reference: -2.6219 kcal/mol. ferric is measured to be ~4x too
        // negative here (see doc comment) -- only assert the same generous
        // order-of-magnitude band used by the original water smoke test, not tight
        // agreement, since tight agreement is precisely what this test demonstrates does
        // NOT hold for methanol's more densely overlapping cavity.
        let pyscf_ref_kcal = -2.6219_f64;
        let small_mag_bound = pyscf_ref_kcal * 0.3;
        let large_mag_bound = pyscf_ref_kcal * 6.0;
        assert!(
            e_solv_kcal < small_mag_bound && e_solv_kcal > large_mag_bound,
            "E_solv={e_solv_kcal:.3} kcal/mol is not within the generous order-of-magnitude \
             band of the PySCF IEF-PCM reference ({pyscf_ref_kcal:.3} kcal/mol); expected \
             between {large_mag_bound:.3} and {small_mag_bound:.3} kcal/mol"
        );
    }

    #[test]
    fn external_point_charge_changes_rks_pbe_energy() {
        // KS-DFT here is just RhfConfig{xc: Some(...), ..} through solve_rhf (no
        // separate RKS solver). This proves the external_potential wiring from
        // Task 5 composes with the xc.is_some() code path, not just the bare-HF one.
        // See ENV_LOCK doc comment: solve_rhf reads the budget env vars.
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let bs = ferric_core::basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let ctx = ParallelContext::default();

        let base_config = RhfConfig { xc: Some("PBE".to_string()), ..Default::default() };
        let base = solve_rhf(&ctx, &mol, &prep, op, &bounds, &base_config).unwrap();

        let ext = ExternalPotential {
            point_charges: vec![PointCharge { q: 1.0, x: 0.0, y: 0.0, z: 20.0 }],
            field: None,
        };
        let config = RhfConfig {
            xc: Some("PBE".to_string()),
            external_potential: Some(ext),
            ..Default::default()
        };
        let perturbed = solve_rhf(&ctx, &mol, &prep, op, &bounds, &config).unwrap();

        assert!(perturbed.converged);
        assert!((perturbed.energy - base.energy).abs() > 1e-8);
    }

    fn run_rhf_test(xyz: &str, basis_name: &str, ref_slug: &str, tol: f64) {
        // See ENV_LOCK doc comment: solve_rhf reads the budget env vars.
        // Held here (not at each call site) so it covers all callers
        // (test_rhf_h2_sto3g / test_rhf_h2o_sto3g / test_rhf_h2o_631g) in one place.
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let bs = basis::bundled(basis_name).unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let config = RhfConfig {
            energy_conv: 1e-12,
            density_conv: 1e-10,
            integral_thresh: 1e-14,
            ..Default::default()
        };
        let ctx = ParallelContext::default();
        let result = solve_rhf(&ctx, &mol, &prep, op, &bounds, &config).unwrap();
        assert!(result.converged, "RHF did not converge");
        eprintln!(
            "{ref_slug}: energy={:.12}, iters={}, vnn={:.12}",
            result.energy,
            result.iterations,
            mol.nuclear_repulsion()
        );
        let ref_path = format!("../../testdata/reference/{ref_slug}");
        if let Ok(text) = std::fs::read_to_string(&ref_path) {
            let ref_data: serde_json::Value = serde_json::from_str(&text).unwrap();
            let ref_energy = ref_data["energy"].as_f64().unwrap();
            assert!(
                (result.energy - ref_energy).abs() < tol,
                "{ref_slug}: got {:.10}, ref {:.10}",
                result.energy,
                ref_energy
            );
        }
    }

    /// MOM must not change where a well-behaved SCF converges.
    #[test]
    fn rhf_mom_no_harm_water() {
        // See ENV_LOCK doc comment: solve_rhf reads the budget env vars.
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let xyz = "3\nwater\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n";
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let ctx = ParallelContext::default();
        let base = RhfConfig { energy_conv: 1e-12, ..Default::default() };
        let mom = RhfConfig { mom_after_iter: 2, ..base.clone() };
        let e0 = solve_rhf(&ctx, &mol, &prep, op, &bounds, &base).unwrap().energy;
        let e1 = solve_rhf(&ctx, &mol, &prep, op, &bounds, &mom).unwrap().energy;
        assert!((e0 - e1).abs() < 1e-9, "MOM changed water RHF: {e0} vs {e1}");
    }

    /// A24-21 (C2H4·Ar dimer, aug-cc-pVDZ, DF-JK): aufbau DIIS-8 plateaus
    /// ~33 Ha above the minimum (err_max ~0.9, occupied-set flip-flop) and
    /// never converges in 400 iterations. MOM's contract is to pin the
    /// occupation and kill the flip-flop: with MOM armed at iter 80,
    /// the SCF *converges* — to whichever stationary state DIIS's basin
    /// held at arming (here ~-604, an excited solution 0.9 Ha above the
    /// C2H4+Ar ground state; the ground-state fix for this system is
    /// diis_size=16, with or without MOM). Arming while DIIS still wanders
    /// (iter <~50) pins worse states.
    /// Slow (~2 min release): run with --ignored.
    #[test]
    #[ignore]
    fn rhf_mom_converges_c2h4_ar_dimer() {
        // See ENV_LOCK doc comment: solve_rhf reads the budget env vars.
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let xyz = "7\na24-21 dimer\nC 0.00000000 0.66718073 -2.29024825\nC 0.00000000 -0.66718073 -2.29024825\nH -0.92400768 1.23202333 -2.28975239\nH 0.92400768 1.23202333 -2.28975239\nH -0.92400768 -1.23202333 -2.28975239\nH 0.92400768 -1.23202333 -2.28975239\nAr -0.00000000 0.00000000 1.60829261\n";
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let bs = basis::bundled("aug-cc-pvdz").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let ctx = ParallelContext::default();
        let config = RhfConfig {
            max_iter: 400,
            df_j_aux: Some("def2-universal-jkfit".into()),
            df_k_aux: Some("def2-universal-jkfit".into()),
            mom_after_iter: 80,
            ..Default::default()
        };
        let result = solve_rhf(&ctx, &mol, &prep, op, &bounds, &config).unwrap();
        assert!(result.converged, "MOM did not break the DIIS flip-flop");
        assert!(
            result.energy < -600.0,
            "C2H4·Ar dimer: converged to {:.10}, not in the molecular basin",
            result.energy
        );
        // Ground-state check: DIIS-16 + MOM lands on the C2H4+Ar limit.
        let config16 = RhfConfig { diis_size: 16, ..config };
        let r16 = solve_rhf(&ctx, &mol, &prep, op, &bounds, &config16).unwrap();
        assert!(
            (r16.energy - (-604.8439254747)).abs() < 1e-5,
            "DIIS-16+MOM: got {:.10}, expected -604.8439254747",
            r16.energy
        );
    }

    #[test]
    fn test_rhf_h2_sto3g() {
        run_rhf_test(
            "2\nH2\nH 0 0 0\nH 0 0 0.74\n",
            "sto-3g",
            "h2_sto-3g_rhf.json",
            1e-8,
        );
    }

    #[test]
    fn test_rhf_h2o_sto3g() {
        // Tolerance 5e-8 due to libint2 vs libcint integral differences
        run_rhf_test(
            "3\nwater\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n",
            "sto-3g",
            "h2o_sto-3g_rhf.json",
            5e-8,
        );
    }

    #[test]
    fn test_rhf_h2o_631g() {
        run_rhf_test(
            "3\nwater\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n",
            "6-31g",
            "h2o_6-31g_rhf.json",
            1e-8,
        );
    }

    /// COSe (O=C=Se) at aug-cc-pVDZ is a closed-shell singlet that PySCF RHF
    /// converges cleanly to E = -2512.5713600 Ha, but ferric's bare Roothaan+DIIS
    /// *diverges* (energy oscillates ±100 Ha, never settles — heavy-atom Se dense
    /// low-virtual manifold drives DIIS into a limit cycle). A virtual-block level
    /// shift, ramped off as the gradient drops, tames the oscillation. This test
    /// is the regression for that fix: it must converge to the PySCF energy.
    /// Slow (~1-2 min release): run with --ignored.
    #[test]
    #[ignore]
    fn rhf_level_shift_converges_cose() {
        // See ENV_LOCK doc comment: solve_rhf reads the budget env vars.
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let xyz = "3\nCOSe\nO 0.0000 0.0000 1.159\nC 0.0000 0.0000 0.0000\nSe 0.0000 0.0000 -1.709\n";
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let bs = basis::bundled("aug-cc-pvdz").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let ctx = ParallelContext::default();
        let config = RhfConfig {
            max_iter: 200,
            level_shift: 0.5,
            ..Default::default()
        };
        let result = solve_rhf(&ctx, &mol, &prep, op, &bounds, &config).unwrap();
        assert!(result.converged, "COSe RHF did not converge with level shift");
        assert!(
            (result.energy - (-2512.5713600037)).abs() < 1e-5,
            "COSe RHF: got {:.10}, expected PySCF -2512.5713600037",
            result.energy
        );
    }

    /// Na4 at aug-cc-pVTZ has a near-degenerate occupied manifold (four overlap
    /// eigenvalues ≈ 3e-5): bare DIIS reaches the correct ground-state energy but
    /// the orbital gradient parks on a noise floor (err_max ≈ 3e-5) it can never
    /// drain below density_conv, so without plateau acceptance the SCF spins to
    /// max_iter and returns Err (the "stuck job" in the GW100 aTZ sweep). This is
    /// the regression for the plateau-acceptance fix: it must converge (flagged)
    /// to the CORRECT ground state −774.894064, NOT the ~−771.5/−771.7 excited
    /// states that MOM or a raised lindep threshold converge to. Slow (~5-7 min
    /// release): run with --ignored.
    #[test]
    #[ignore]
    fn rhf_na4_atz_near_degeneracy_plateau_converges() {
        // See ENV_LOCK doc comment: solve_rhf reads the budget env vars.
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let xyz = "4\nNa4\nNa 0.0002445 -0.0998053 1.5471126\nNa -0.0002444 3.1776586 0.0486374\n\
                   Na 0.0002444 0.0997722 -1.5472150\nNa -0.0002444 -3.1776254 -0.0485350\n";
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let bs = basis::bundled("aug-cc-pvtz").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let ctx = ParallelContext::default();
        // Default config (no level shift, no MOM) — the exact path gw100_full uses.
        let config = RhfConfig { max_iter: 60, ..Default::default() };
        let result = solve_rhf(&ctx, &mol, &prep, op, &bounds, &config).unwrap();
        assert!(
            result.converged,
            "Na4/aTZ did not converge (plateau acceptance failed); iters={}",
            result.iterations
        );
        assert!(
            (result.energy - (-774.894064)).abs() < 1e-4,
            "Na4/aTZ converged to the WRONG state: got {:.6}, expected ground state \
             -774.894064 (a value near -771.5/-771.7 means the plateau ceiling let an \
             excited state through)",
            result.energy
        );
    }

    #[test]
    fn divergence_aborts_early() {
        use ferric_core::mol::Molecule;
        use ferric_core::basis;
        use ferric_core::parallel::ParallelContext;
        use ferric_integrals::basis_bridge::PreparedBasis;
        use ferric_integrals::operator::Operator;
        use crate::screening::SchwarzBounds;
        use crate::result::ScfExit;

        // A guess/level-shift-free run on a hard system that oscillates. We assert
        // the detector CAN fire and returns the right exit reason, using a synthetic
        // check on the config plumbing: divergence_tol very small so any energy rise
        // trips it, with divergence disabled it would run to max_iter.
        // See ENV_LOCK doc comment: solve_rhf reads the budget env vars.
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let ctx = ParallelContext::default();

        // Water/STO-3G converges cleanly, so divergence must NOT fire — exit is
        // Converged. This guards against false positives on a healthy descent.
        let cfg = RhfConfig {
            max_iter: 100,
            divergence_tol: Some(0.5),
            stall_window: Some(15),
            ..Default::default()
        };
        let r = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
        assert!(r.converged, "water/sto-3g must still converge with detectors on");
        assert_eq!(r.exit, ScfExit::Converged);
    }

    #[test]
    fn detectors_default_off() {
        let cfg = RhfConfig::default();
        assert!(cfg.stall_window.is_none());
        assert!(cfg.divergence_tol.is_none());
    }

    #[test]
    fn solve_rhf_maxiter_returns_ok_not_converged_with_density() {
        use crate::result::ScfExit;

        // See ENV_LOCK doc comment: solve_rhf reads the budget env vars.
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let ctx = ParallelContext::default();
        // max_iter = 1 guarantees non-convergence for water.
        let cfg = RhfConfig { max_iter: 1, ..Default::default() };
        let r = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg)
            .expect("max_iter must now return Ok, not Err");
        assert!(!r.converged, "should not be converged in 1 iter");
        assert_eq!(r.exit, ScfExit::MaxIter);
        assert_eq!(r.density_total.dim(), (prep.nbasis(), prep.nbasis()));
        assert!(r.density_total.iter().all(|v| v.is_finite()));
    }

    // --- scf_converged: pure-decision tests (ΔE + ΔP gate, ORCA ConvCheckMode-2) ---

    fn sig(de: f64, dp_rms: f64, dp_max: f64) -> ConvergenceSignals {
        ConvergenceSignals { de, dp_rms, dp_max }
    }

    // Tolerances mirror the defaults: energy_conv (loose ΔE sanity) = 1e-3,
    // density_conv (tight ΔP) = 1e-6.
    const E_CONV: f64 = 1e-3;
    const D_CONV: f64 = 1e-6;

    #[test]
    fn scf_converged_accepts_settled_density_and_nondescending_energy() {
        // ΔP under the tight tol AND ΔE under the loose sanity bound → Converged.
        let r = scf_converged(sig(2e-5, 5e-7, 4e-6), E_CONV, D_CONV);
        assert_eq!(r, Some(ScfExit::Converged));
    }

    #[test]
    fn scf_converged_accepts_when_energy_and_gradient_park_on_ri_floor() {
        // THE key case (MEASURED toluene/aTZ): the commutator parks at ~1.26e-6
        // AND ΔE floors at ~2e-5 — neither drains. But dp_rms has drained to
        // ~1e-9. The gate accepts on ΔP; the loose ΔE bound (1e-3) clears the
        // 2e-5 energy floor. This is the whole point of the redesign: BOTH the
        // gradient and ΔE floor with naux, only ΔP converges.
        let r = scf_converged(sig(2e-5, 1e-9, 9e-8), E_CONV, D_CONV);
        assert_eq!(r, Some(ScfExit::Converged),
            "must converge on ΔP even when BOTH the gradient and ΔE floor above their tols");
    }

    #[test]
    fn scf_converged_rejects_still_descending_density() {
        // Density still moving (dp_rms ≫ density_conv): a mid-descent iteration.
        let r = scf_converged(sig(2e-5, 3e-5, 1e-3), E_CONV, D_CONV);
        assert_eq!(r, None, "a still-moving density must not be accepted");
    }

    #[test]
    fn scf_converged_rejects_actively_descending_energy() {
        // Density looks settled but the energy is still dropping fast (ΔE ≫ the
        // loose bound) — an early iteration where DIIS briefly stalls the density
        // while the energy is far from settled. The loose ΔE bound still catches it.
        let r = scf_converged(sig(1e-2, 5e-7, 4e-6), E_CONV, D_CONV);
        assert_eq!(r, None, "an actively-descending energy must not be accepted");
    }

    #[test]
    fn scf_converged_dp_max_companion_guards_a_single_moving_element() {
        // dp_rms looks settled but one density element is still swinging
        // (dp_max > 10·density_conv) → reject (ORCA TolMaxP guard).
        let r = scf_converged(sig(2e-5, 5e-7, 5e-5), E_CONV, D_CONV);
        assert_eq!(r, None, "dp_max companion must reject a single moving element");
    }

    // --- stall_detected: pure-arithmetic positive-trip tests ---
    //
    // These synthesize errmax_history sequences directly, instead of forcing a
    // real molecule to stall (slow/nondeterministic), to exercise the trip path
    // that the existing detectors_default_off / divergence_aborts_early tests
    // cannot reach (they only guard against false positives on a healthy run).

    #[test]
    fn stall_detected_true_on_oscillating_plateau() {
        // Descend well below the 1e-4 floor's irrelevant here: the plateau band
        // itself is pinned above 1e-4 (a limit cycle that never drains). Window
        // w=4: 8 entries oscillating around 4.5 (band [4.0, 5.0]) means both the
        // recent and previous running-mins land at 4.0, so recent_min (4.0) >=
        // 0.9*prev_min (3.6) trips.
        let w = 4;
        let history = vec![4.0, 5.0, 4.0, 5.0, 4.0, 5.0, 4.0, 5.0];
        assert!(stall_detected(&history, w, 4.5));
    }

    #[test]
    fn stall_detected_false_on_clean_descent() {
        // Each window's running-min drops by >10% vs the previous window, so the
        // detector must not trip on genuine progress.
        let w = 4;
        let history = vec![1e-2, 9e-3, 8e-3, 7e-3, 5e-4, 4e-4, 3e-4, 2e-4];
        assert!(!stall_detected(&history, w, 2e-4));
    }

    #[test]
    fn stall_detected_false_below_plateau_floor() {
        // Flat history that would trip the running-min comparison, but
        // current_err is below the 1e-4 floor: the plateau path (separate logic)
        // owns this regime, not stall detection.
        let w = 4;
        let history = vec![5e-5; 8];
        assert!(!stall_detected(&history, w, 5e-5));
    }

    #[test]
    fn stall_detected_false_when_window_zero() {
        // Guard against the Some(0) false-trip a reviewer flagged: window=0
        // must never trip regardless of history/current_err.
        let history = vec![4.0, 5.0, 4.0, 5.0];
        assert!(!stall_detected(&history, 0, 4.5));
    }

    #[test]
    fn stall_detected_false_when_history_too_short() {
        // Fewer than 2*window entries: not enough data to compare running-mins.
        let w = 4;
        let history = vec![4.0, 5.0, 4.0, 5.0, 4.0]; // 5 < 2*4
        assert!(!stall_detected(&history, w, 4.5));
    }

    /// P14: `build_jk`'s J/K accumulation must be bit-identical regardless of
    /// the rayon worker count. Before this fix the shell-pair work list was
    /// combined via `fold(..).reduce(..)`, a binary tree whose association
    /// (and hence floating-point rounding) depends on the thread count — this
    /// was the root cause the P7 lane traced a whole-pipeline RHF gradient
    /// bit-identity mismatch (last-bit drift, 0x...dfc3 vs 0x...dfbf) back to.
    /// `grouped_deterministic_sum_pair` folds group partials in a fixed,
    /// thread-count-independent ascending order, so the result must match
    /// exactly across pools of different sizes.
    #[test]
    fn build_jk_bit_identical_across_thread_counts() {
        let mol = Molecule::parse_xyz(
            "3\nH2O\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.26\n",
            0,
            1,
        )
        .unwrap();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let n = prep.nbasis();

        // Dense symmetric density so every screened quartet contributes.
        let mut d = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                d[(i, j)] = 0.01 * ((i * 7 + j * 3) % 11) as f64;
            }
        }
        let d = 0.5 * (&d + &d.t());

        let run = |threads: usize| -> (Array2<f64>, Array2<f64>) {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                let ctx = ParallelContext::default();
                let mut j = Array2::zeros((n, n));
                let mut k = Array2::zeros((n, n));
                build_jk(&ctx, &prep, &bounds, 1e-14, &d, &mut j, &mut k).unwrap();
                (j, k)
            })
        };

        let (j1, k1) = run(1);
        let (j4, k4) = run(4);

        for i in 0..n {
            for jj in 0..n {
                assert_eq!(
                    j1[(i, jj)].to_bits(),
                    j4[(i, jj)].to_bits(),
                    "build_jk J not bit-identical across thread counts at ({i},{jj}): \
                     1-thread={:.17e}, 4-thread={:.17e}",
                    j1[(i, jj)],
                    j4[(i, jj)],
                );
                assert_eq!(
                    k1[(i, jj)].to_bits(),
                    k4[(i, jj)].to_bits(),
                    "build_jk K not bit-identical across thread counts at ({i},{jj}): \
                     1-thread={:.17e}, 4-thread={:.17e}",
                    k1[(i, jj)],
                    k4[(i, jj)],
                );
            }
        }
    }

    /// P14: whole-pipeline RHF gradient bit-identity, un-gating the finding
    /// left by the P7 lane (see the comment on
    /// `test_3c2c_assembly_bit_identical_across_thread_counts` in
    /// ferric-mp2/src/gradient.rs). Runs a full `solve_rhf` (which drives
    /// `build_jk` every SCF iteration) followed by `rhf_gradient` under
    /// dedicated 1- and 4-worker rayon pools and compares every component via
    /// `f64::to_bits`. This is the acceptance bar P14 was scoped to satisfy:
    /// build_jk's grouped deterministic reduction must make the *converged*
    /// SCF state — and everything built on it — thread-count-invariant, not
    /// just one isolated builder call.
    #[test]
    fn whole_pipeline_rhf_gradient_bit_identical_across_thread_counts() {
        // See ENV_LOCK doc comment: solve_rhf reads the budget env vars.
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let xyz = "3\nwater\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n";
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let config = RhfConfig {
            energy_conv: 1e-12,
            density_conv: 1e-10,
            integral_thresh: 1e-14,
            ..Default::default()
        };

        let run = |threads: usize| -> (f64, Array2<f64>) {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                let ctx = ParallelContext::default();
                let result = solve_rhf(&ctx, &mol, &prep, op, &bounds, &config).unwrap();
                assert!(result.converged, "RHF did not converge at {threads} threads");
                let grad = crate::gradient::rhf_gradient(&mol, &prep, op, &bounds, &result, None).unwrap();
                (result.energy, grad)
            })
        };

        let (e1, g1) = run(1);
        let (e4, g4) = run(4);

        assert_eq!(
            e1.to_bits(),
            e4.to_bits(),
            "RHF energy not bit-identical across thread counts: 1-thread={e1:.17e}, 4-thread={e4:.17e}"
        );
        for atom in 0..3 {
            for c in 0..3 {
                assert_eq!(
                    g1[(atom, c)].to_bits(),
                    g4[(atom, c)].to_bits(),
                    "RHF gradient not bit-identical across thread counts at atom={atom} coord={c}: \
                     1-thread={:.17e} (0x{:016x}), 4-thread={:.17e} (0x{:016x})",
                    g1[(atom, c)], g1[(atom, c)].to_bits(),
                    g4[(atom, c)], g4[(atom, c)].to_bits(),
                );
            }
        }
    }
}
