//! Closed-shell restricted Hartree-Fock (RHF) solver.
//!
//! Implements the Roothaan-Hall SCF procedure with DIIS convergence acceleration
//! and Schwarz-screened two-electron integral evaluation.

use crate::df_j::DfJ;
use crate::df_k::DfK;
use crate::diis::Diis;
use crate::direct_j::DirectJ;
use crate::direct_jk::DirectJK;
use crate::direct_k::DirectK;
use crate::fock::{JBuilder, KBuilder};
use crate::guess::hcore_guess;
use ferric_dft::cdft::Constraint;
use crate::result::{ScfResult, Spin};

use crate::link_k::LinkK;
use crate::screening::SchwarzBounds;
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
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
    /// aux-blocks to disk instead of allocating in core. Default 2 GB. The env
    /// var `FERRIC_OOC_BUDGET_GB` overrides this at runtime.
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
}

impl Default for RhfConfig {
    fn default() -> Self {
        Self {
            // Tightened from (1e-8, 1e-7, 100) after H2O+ false-convergence
            // diagnosis: with the looser tolerances, UHF would report
            // converged=true at a state 85 mHa above the true minimum.
            max_iter: 200,
            energy_conv: 1e-10,
            density_conv: 1e-8,
            diis_size: 8,
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
            three_index_budget_bytes: 2 * 1024 * 1024 * 1024,
            init_guess_density: None,
            use_sad_guess: true,
        }
    }
}

/// Resolve the 3-index memory budget in bytes via the unified resolver
/// [`ferric_core::memory::resolve_budget_bytes`]. Shared by RHF/UHF/ROHF so the
/// budget is honored uniformly across all DF-J/DF-K construction sites.
///
/// `config_bytes` is the TOML `[memory]` value (or the 2 GiB `RhfConfig`
/// default). Precedence: an env override (`FERRIC_MEM_BUDGET_GB`, or the legacy
/// `FERRIC_OOC_BUDGET_GB`/`FERRIC_ERI3_BUDGET_GB`) wins over `config_bytes`,
/// matching the legacy behavior where the env var beat the config field.
///
/// Note: because the `RhfConfig` default is a concrete 2 GiB (never 0), the
/// auto-detect branch of the resolver is only reached when a caller passes 0
/// explicitly; the SCF path therefore keeps its historical 2 GiB default when no
/// env var is set.
pub fn resolve_three_index_budget(config_bytes: usize) -> usize {
    use ferric_core::memory::{self, BudgetSource};
    // Try env-only first (explicit=None): if an env override exists it must beat
    // the config field, preserving the legacy env>config precedence.
    let env_res = memory::resolve_budget(None);
    match env_res.source {
        BudgetSource::UnifiedEnv
        | BudgetSource::LegacyOocEnv
        | BudgetSource::LegacyEri3Env => env_res.bytes,
        // No env override → honor the config field (or resolve auto/fallback if
        // the caller passed 0).
        _ => memory::resolve_budget_bytes(Some(config_bytes)),
    }
}

/// Solve the closed-shell RHF equations for a molecule.
///
/// Uses the Roothaan-Hall procedure: build Fock matrix from density, diagonalize,
/// rebuild density, iterate until convergence. DIIS extrapolation accelerates
/// convergence. Returns [`ScfResult`] on success or [`FerricError::ScfConvergence`]
/// if `max_iter` is exceeded.
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
    let h = oneelectron::hcore_ecp(prep, mol, prep.basis_set());
    let n = prep.nbasis();
    let nelec = mol.nelec();
    if nelec % 2 != 0 {
        return Err(FerricError::ScfConvergence {
            iterations: 0,
            last_energy: 0.0,
        });
    }
    let nocc = (nelec / 2) as usize;
    let vnn = mol.nuclear_repulsion();

    // Initial density: explicit override > SAD (default) > hcore. SAD is the
    // default because the bare hcore guess diverges on heavy-atom closed shells
    // (COSe/C2H3Br); if SAD fails to build (e.g. a free-atom solve doesn't
    // converge) we fall back to hcore rather than aborting the whole SCF.
    let mut d = if let Some(d0) = config.init_guess_density.as_ref() {
        d0.clone()
    } else if config.use_sad_guess {
        match crate::guess::sad_guess(mol, prep, prep.basis_set()) {
            Ok(d_sad) => d_sad,
            Err(_) => hcore_guess(&s, &h, nocc)?,
        }
    } else {
        hcore_guess(&s, &h, nocc)?
    };
    let mut f = Array2::zeros((n, n));
    let mut j_buf = Array2::<f64>::zeros((n, n));
    let mut k_buf = Array2::<f64>::zeros((n, n));
    let mut diis = Diis::new(config.diis_size);
    // MOM reference: last accepted occupied MO block (None until armed).
    let mut mom_ref: Option<Array2<f64>> = None;
    let mut prev_e = 0.0;
    // Previous iteration's err_max, for detecting a stalled DIIS gradient plateau
    // (near-degenerate manifolds park err_max on a noise floor it can't drain).
    let mut prev_err_max = f64::INFINITY;
    // Count of consecutive iterations where the energy is stationary but err_max
    // has stopped improving. Used only as a last-resort plateau acceptance.
    let mut plateau_streak = 0usize;
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

    // Resolve the out-of-core 3-index memory budget once (env override wins).
    let ooc_budget = resolve_three_index_budget(config.three_index_budget_bytes);

    // Auto-default JK aux bases when the functional needs exact exchange but
    // the caller hasn't explicitly set df_j_aux / df_k_aux. This makes
    // `cfg.xc = Some("B3LYP")` (or any hybrid/RSH) work out of the box.
    // Pure HF (no xc) keeps the historical behavior of no auto-default.
    let needs_k = xc_contrib.is_some() && (k_mix.sr > 0.0 || k_mix.omega > 0.0);
    let needs_j = xc_contrib.is_some();
    const DEFAULT_JK_AUX: &str = "def2-universal-jkfit";
    let df_j_aux_eff: Option<String> = config.df_j_aux.clone()
        .or_else(|| needs_j.then(|| DEFAULT_JK_AUX.into()));
    let df_k_aux_eff: Option<String> = config.df_k_aux.clone()
        .or_else(|| needs_k.then(|| DEFAULT_JK_AUX.into()));

    // Density-fitted Coulomb (RI-J). Builds 3-center tensor + inverse metric once.
    let mut df_j: Option<DfJ> = if let Some(aux_name) = df_j_aux_eff.as_deref() {
        let dfbs_set = ferric_core::basis::bundled(aux_name)?;
        let dfbs = PreparedBasis::new(mol, &dfbs_set)?;
        Some(DfJ::new(op, prep, &dfbs, ooc_budget)?)
    } else {
        None
    };

    // Density-fitted exchange (RI-K). Builds V^{-1/2}-dressed 3-center tensor once.
    let mut df_k: Option<DfK> = if let Some(aux_name) = df_k_aux_eff.as_deref() {
        let dfbs_set = ferric_core::basis::bundled(aux_name)?;
        let dfbs = PreparedBasis::new(mol, &dfbs_set)?;
        Some(DfK::new(op, prep, &dfbs, ooc_budget)?)
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
            Some(DfK::new(Operator::erfc(k_mix.omega), prep, &dfbs_prep, ooc_budget)?),
            Some(DfK::new(Operator::erf(k_mix.omega), prep, &dfbs_prep, ooc_budget)?),
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
    let mut direct_k: Option<DirectK> = if df_any && df_k.is_none() {
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
            } else {
                let dk = direct_k.as_mut().expect("DirectK built before loop");
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

        // F = H + J − ½ K_total  (V_xc added below)
        f.assign(&(&h + &j_buf - &(0.5 * &k_total)));

        // Electronic energy BEFORE adding V_xc (V_xc is one-body in F but
        // E_xc is its own integral).
        let e_elec_no_xc: f64 = 0.5 * (&d * &(&h + &f)).sum();
        let e_xc = if let Some(x) = xc_contrib.as_ref() {
            x.add_xc(&d, &mut f)
        } else {
            0.0
        };
        let energy = e_elec_no_xc + e_xc + vnn;

        // DIIS error: e = FDS - SDF
        let fds = f.dot(&d).dot(&s);
        let sdf = s.dot(&d).dot(&f);
        let err = &fds - &sdf;

        let de = (energy - prev_e).abs();
        let err_max = err.iter().map(|v| v.abs()).fold(0.0f64, f64::max);

        if std::env::var("FERRIC_SCF_TRACE").ok().as_deref() == Some("1") {
            eprintln!("SCF iter={:4}  E={:.12}  dE={:.3e}  err_max={:.3e}", iter, energy, de, err_max);
        }

        // DF builds introduce O(1e-6) Ha noise in the K matrix per iteration that
        // breaks strict energy variational convergence even when orbitals have
        // fully converged. When DF is active, accept on |FDS-SDF| (orbital gradient)
        // alone — the same criterion PySCF uses for DF-SCF.
        //
        // For large polyaromatic molecules with near-degenerate π orbitals the DF
        // noise floor can park err_max at ~1-5×density_conv indefinitely (H1
        // diagnosis: plateau, not oscillation). The fallback below accepts when
        // the gradient is within a 10× factor of the threshold AND the energy
        // change is below 1e-5 Ha (safely in the noise floor plateau, not still
        // descending toward the minimum).
        let df_active = df_j.is_some() || df_k.is_some();
        let energy_ok = de < config.energy_conv;
        let grad_ok = err_max < config.density_conv;
        let df_noise_floor_ok = df_active
            && err_max < 10.0 * config.density_conv
            && de < 1e-5;

        // Near-degeneracy plateau acceptance (both J/K paths). On diffuse
        // alkali/d-block clusters (Na4/Na6/Cu2 at aug-cc-pVTZ) the occupied
        // manifold is near-degenerate: DIIS reaches the correct ground-state
        // energy but the orbital gradient parks on a noise floor (err_max ≈ 3e-5)
        // it can never drain below density_conv, so bare `energy_ok && grad_ok`
        // never trips and the SCF spins to max_iter. We accept once the gradient
        // has demonstrably STALLED — improving <10% per iter for several iters —
        // while the energy is stationary (|dE| < 1e-6, safely at the noise floor,
        // NOT still descending). This preserves the basin bare DIIS already found
        // (verified lowest-energy for Na4: −774.894 vs the −771.5/−771.7 wrong
        // excited states that MOM / a raised lindep threshold converge to), and
        // cannot fire mid-descent (err_max would be dropping fast) or on an
        // oscillating run (err_max would not be monotonically stalled). The
        // result is flagged `converged` but the caller should treat a plateau
        // exit as gradient-loose; we surface it via the trace below.
        // err_max ceiling of 1e-4: the correct Na4 ground state plateaus at
        // err_max ≤ 3e-5, whereas the wrong excited states reached by MOM / a
        // raised lindep threshold plateau at ≥1e-3 — so 1e-4 accepts the former
        // with margin and excludes the latter. Never relax this above the value
        // that separates the two, or a genuinely-unconverged state could slip in.
        let grad_stalled = err_max <= prev_err_max && err_max > 0.9 * prev_err_max;
        if de < 1e-6 && grad_stalled && err_max < 1e-4 {
            plateau_streak += 1;
        } else {
            plateau_streak = 0;
        }
        let plateau_ok = plateau_streak >= 3;
        if plateau_ok && std::env::var("FERRIC_SCF_TRACE").ok().as_deref() == Some("1") {
            eprintln!(
                "SCF plateau accepted at iter={iter}: E={energy:.9} err_max={err_max:.3e} \
                 (gradient stalled on near-degeneracy noise floor)"
            );
        }
        prev_err_max = err_max;

        let strict = if df_active {
            grad_ok || df_noise_floor_ok
        } else {
            energy_ok && grad_ok
        };
        let converged = strict || plateau_ok;

        if iter > 1 && converged {
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
                iterations: iter,
                computed_quartets: total_quartets,
            });
        }
        prev_e = energy;

        let mut f_new = diis.step(&f, &err);

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
        if config.level_shift > 0.0 {
            if let Some(c_p) = c_prev.as_ref() {
                let shift_ramp = config.level_shift * (err_max / 0.1).min(1.0);
                if shift_ramp > 0.0 {
                    let c_vir = c_p.slice(ndarray::s![.., nocc..]);
                    let scv = s.dot(&c_vir); // (n × nvir)
                    f_new = &f_new + &(shift_ramp * scv.dot(&scv.t()));
                }
            }
        }

        let (_, mut c) = diagonalize(&f_new, &x)?;
        // Only retain C across iterations when the level shift needs it; the
        // default (shift = 0) path skips the clone entirely.
        if config.level_shift > 0.0 {
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

        // Rebuild density: D = 2 * C_occ @ C_occ^T  (BLAS dgemm)
        let c_occ = c.slice(ndarray::s![.., ..nocc]);
        d.assign(&(2.0 * c_occ.dot(&c_occ.t())));
    }
    Err(FerricError::ScfConvergence {
        iterations: config.max_iter,
        last_energy: prev_e,
    })
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
    use rayon::prelude::*;
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

    // One engine per rayon thread (see engine_pool) — constructing an engine in
    // the fold init fires once per work-chunk, not per thread, storming the
    // global libint2 ctor mutex on heavy-element bases.
    let pool = crate::engine_pool::EnginePool::new(bounds.op, prep, 1e-14)?;
    let total_jk = shell_pairs.into_par_iter().fold(
        || (
            Array2::zeros((nbf, nbf)),
            Array2::zeros((nbf, nbf)),
            0usize
        ),
        |(mut local_j, mut local_k, mut local_count), (s1, s2)| {
            if ferric_core::INTERRUPT.load(std::sync::atomic::Ordering::Relaxed) {
                return (local_j, local_k, local_count);
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
            (local_j, local_k, local_count)
        }
    ).map(|(j, k, count)| {
        computed_quartets.fetch_add(count, Ordering::Relaxed);
        (j, k)
    }).reduce(
        || (Array2::zeros((nbf, nbf)), Array2::zeros((nbf, nbf))),
        |mut acc, next| {
            acc.0 += &next.0;
            acc.1 += &next.1;
            acc
        }
    );

    *j += &total_jk.0;
    *k += &total_jk.1;

    #[cfg(feature = "mpi")]
    if let Some(world) = &ctx.world {
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
    std::env::var("FERRIC_LINDEP_THRESH")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
        .unwrap_or(LINDEP_THRESH)
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
    let (s_evals, s_evecs) = s
        .eigh(ndarray_linalg::UPLO::Upper)
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
    let f_prime = x.t().dot(f).dot(x);
    let (evals, evecs) = f_prime
        .eigh(ndarray_linalg::UPLO::Upper)
        .map_err(|e| FerricError::Lapack(format!("F diag: {e}")))?;
    let c_kept = x.dot(&evecs); // (n × m)
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
    use ferric_core::mol::Molecule;
    use ferric_integrals::basis_bridge::PreparedBasis;

    fn run_rhf_test(xyz: &str, basis_name: &str, ref_slug: &str, tol: f64) {
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
}
