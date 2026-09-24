//! Restricted Open-Shell Hartree-Fock (ROHF) solver.
//!
//! Spin-pure open-shell HF: a single set of MOs partitioned into doubly
//! occupied (closed), singly occupied (open, α-only), and virtual blocks.
//! ⟨S²⟩ is exact S(S+1) by construction.
//!
//! Coupling: **Guest-Saunders** (PySCF default, hard-coded — no knob).
//! Implementation mirrors PySCF's `get_roothaan_fock` (`pyscf/scf/rohf.py`):
//! the effective Fock is built via density-based projectors
//!   P_c = D_β · S,   P_o = (D_α − D_β) · S,   P_v = I − D_α · S
//! and then assembled from F_c = (F_α + F_β)/2, F_α, F_β according to the
//! Roothaan block table:
//! ```text
//! ========  ======== ====== =========
//! space      closed   open   virtual
//! ========  ======== ====== =========
//! closed       Fc      Fb     Fc
//! open         Fb      Fc     Fa
//! virtual      Fc      Fa     Fc
//! ========  ======== ====== =========
//! ```
//! Per Guest-Saunders (a_cc = a_oo = a_vv = 1/2, b = -1/2, c = 3/2 in the
//! a/b/c parametrisation), the diagonal blocks reduce to F_c — exactly what
//! the projector form above produces.

use crate::diis::Diis;
use crate::direct_j::DirectJ;
use crate::direct_k::DirectK;
use crate::fock::{JBuilder, KBuilder};
use crate::guess::hcore_guess;
use crate::result::{ScfResult, Spin};
use crate::rhf::RhfConfig;
use crate::screening::SchwarzBounds;

use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ndarray::Array2;
use ndarray_linalg::Eigh;

/// `FERRIC_ROHF_TRACE` descriptor: per-iteration ROHF plateau-dynamics trace
/// (env-only debug toggle). NOTE behavior change: previously read via `.is_ok()`,
/// so ANY value (incl. `=0`) enabled it; now `=0`/`false`/`off` disable it,
/// consistent with every other `FERRIC_*_TRACE` flag.
static ROHF_TRACE: ferric_core::config::ConfigVar<bool> = ferric_core::config::ConfigVar {
    env_name: "FERRIC_ROHF_TRACE",
    default: false,
    parse: ferric_core::config::parse_toggle,
    validate: ferric_core::config::accept_any,
};

/// Whether the per-iteration ROHF trace is on. Malformed value → warn + off.
fn rohf_trace() -> bool {
    ROHF_TRACE.toggle()
}

/// ROHF configuration mirrors RHF.
pub type RohfConfig = RhfConfig;

/// Test-only instrumentation: counts how many times a GGA-family (non-LDA)
/// f_xc kernel was built for a Newton/AH step. Lets the end-to-end ROKS/PBE
/// test prove the GGA Newton path actually engaged (rather than silently
/// falling back to DIIS). Not compiled into release/library consumers' hot
/// path beyond a single relaxed atomic increment.
#[doc(hidden)]
pub static GGA_FXC_KERNEL_BUILDS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Owns the f_xc kernel + its reference density for one Newton/AH step, and
/// hands out the `Fn(δD_α, δD_β) -> (δV_α, δV_β)` closure the ROHF Newton
/// solver consumes. Two variants: a purely local LDA kernel and the GGA kernel
/// (which adds the σ = |∇ρ|² coupling terms). Building this once per Newton step
/// and borrowing the closure from it keeps the (heavy) grid + AO evaluation out
/// of the per-matvec inner loop.
/// True when `xc_name` resolves to a meta-GGA functional (SCAN / r2SCAN / TPSS).
/// Meta-GGA is energy-only in Phase A: it has no τ-dependent f_xc kernel, so the
/// Newton/AH acceleration path must be skipped (fall back to DIIS). `None`
/// (pure HF/ROHF) and any resolver error are treated as "not meta-GGA".
pub(crate) fn xc_is_metagga(xc_name: Option<&str>) -> bool {
    match xc_name {
        Some(name) => match ferric_dft::libxc::xc_def_from_name(name) {
            Ok(def) => def
                .funcs
                .iter()
                .any(|f| f.family() == ferric_dft::libxc::FunctionalFamily::MetaGga),
            Err(_) => false,
        },
        None => false,
    }
}

pub(crate) enum FxcKernelStore {
    Lda {
        kernel: Box<ferric_dft::fxc::LdaFxcKernel>,
        rho_a0: Vec<f64>,
        rho_b0: Vec<f64>,
    },
    Gga {
        kernel: Box<ferric_dft::fxc::GgaFxcKernel>,
        ref_dens: Box<ferric_dft::density_on_grid::UksDensityGrid>,
    },
}

impl FxcKernelStore {
    /// Build the appropriate f_xc kernel for `xc_name` (LDA vs GGA-family) at
    /// the given reference densities. `xc_name` is the functional string that
    /// already produced a live `xc_contrib` upstream, so it is guaranteed
    /// LDA/GGA/hybrid/RSH (meta-GGA rejected at KsXcUks::new).
    pub(crate) fn build(
        mol: &Molecule,
        prep: &PreparedBasis,
        cfg: &ferric_dft::grid::AtomicGridConfig,
        xc_name: &str,
        d_a: &Array2<f64>,
        d_b: &Array2<f64>,
    ) -> Result<Self, FerricError> {
        let xc_def = ferric_dft::libxc::xc_def_from_name_nspin(xc_name, 2)
            .map_err(|e| FerricError::General(format!("fxc def for {xc_name}: {e:?}")))?;
        let is_lda = xc_def
            .funcs
            .iter()
            .all(|f| f.family() == ferric_dft::libxc::FunctionalFamily::Lda);
        if is_lda {
            let kernel = ferric_dft::fxc::LdaFxcKernel::new(mol, prep.basis_set(), xc_def, cfg)
                .map_err(|e| FerricError::General(format!("LdaFxcKernel: {e}")))?;
            let (rho_a0, rho_b0) = kernel.reference_density(d_a, d_b);
            Ok(FxcKernelStore::Lda {
                kernel: Box::new(kernel),
                rho_a0,
                rho_b0,
            })
        } else {
            let kernel = ferric_dft::fxc::GgaFxcKernel::new(mol, prep.basis_set(), xc_def, cfg)
                .map_err(|e| FerricError::General(format!("GgaFxcKernel: {e}")))?;
            let ref_dens = kernel.reference_density(d_a, d_b);
            GGA_FXC_KERNEL_BUILDS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(FxcKernelStore::Gga {
                kernel: Box::new(kernel),
                ref_dens: Box::new(ref_dens),
            })
        }
    }

    /// The `Fn(δD_α, δD_β) -> (δV_α, δV_β)` response closure, borrowing `self`.
    #[allow(clippy::type_complexity)]
    pub(crate) fn response(
        &self,
    ) -> Box<dyn Fn(&Array2<f64>, &Array2<f64>) -> (Array2<f64>, Array2<f64>) + Sync + '_> {
        match self {
            FxcKernelStore::Lda {
                kernel,
                rho_a0,
                rho_b0,
            } => Box::new(move |dd_a: &Array2<f64>, dd_b: &Array2<f64>| {
                kernel.apply_with_ref(rho_a0, rho_b0, dd_a, dd_b)
            }),
            FxcKernelStore::Gga { kernel, ref_dens } => {
                Box::new(move |dd_a: &Array2<f64>, dd_b: &Array2<f64>| {
                    kernel.apply_with_ref(ref_dens, dd_a, dd_b)
                })
            }
        }
    }
}

/// Solve restricted open-shell Hartree-Fock equations.
///
/// Uses `mol.charge` and `mol.multiplicity` to determine doubly/singly
/// occupied orbital counts:
///   - nocc_open   = mult − 1                (singly α-occupied)
///   - nocc_double = (nelec − nocc_open) / 2 (doubly occupied)
pub fn solve_rohf(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    config: &RohfConfig,
) -> Result<ScfResult, FerricError> {
    let r = solve_rohf_best_effort(ctx, mol, prep, op, bounds, config)?;
    if r.converged {
        Ok(r)
    } else {
        Err(FerricError::ScfConvergence {
            iterations: r.iterations,
            last_energy: r.energy,
        })
    }
}

/// ROHF/ROKS that returns its best-effort state instead of erroring on failure.
///
/// Returns `Ok` with `converged: false` when the SCF does not converge, keeping the
/// final density and MOs so a convergence ladder can restart from them.
/// **Callers must check `converged`.** Prefer [`solve_rohf`] unless implementing
/// restart/escalation logic.
pub fn solve_rohf_best_effort(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    _op: Operator,
    bounds: &SchwarzBounds,
    config: &RohfConfig,
) -> Result<ScfResult, FerricError> {
    use ferric_dft::ks::KsXcUks;
    use ferric_dft::xc_trait::{KMix, UksXcContribution};

    // Build UKS XC contribution once. None for pure ROHF.
    let xc_contrib: Option<Box<dyn UksXcContribution>> = if let Some(name) = config.xc.as_deref() {
        let main = config.dft_grid.clone().unwrap_or_default();
        let nlc = config
            .nlc_grid
            .clone()
            .unwrap_or(ferric_dft::grid::AtomicGridConfig {
                n_radial: 50,
                n_angular: 50,
                ..Default::default()
            });
        // Thread the caller's `[memory] budget_gb` into the grid AO cache --
        // the largest single allocation in a DFT job. This used to call the
        // UNbudgeted `new_with_omega`, which resolves from env/auto-detect
        // and so silently DISCARDED `config.three_index_budget_bytes`. The
        // budgeted constructor existed for exactly this and had ZERO
        // production callers; its own doc records the symptom ("Setting
        // `budget_gb = 4` on a 64 GB box still sized the grid cache against
        // ~51 GB"), i.e. the documented primary knob did nothing while
        // FERRIC_MEM_BUDGET_GB worked. 0 means unset, matching
        // `rhf::resolve_three_index_budget`.
        let ks = KsXcUks::new_with_omega_budgeted(
            mol,
            prep.basis_set(),
            name,
            &main,
            &nlc,
            config.xc_omega,
            (config.three_index_budget_bytes != 0).then_some(config.three_index_budget_bytes),
        )
        .map_err(|e| FerricError::General(format!("KsXcUks init for {name}: {e:?}")))?;
        Some(Box::new(ks) as Box<dyn UksXcContribution>)
    } else {
        None
    };

    let k_mix: KMix = xc_contrib.as_ref().map(|x| x.k_mix()).unwrap_or_default();
    let c_k: f64 = if xc_contrib.is_some() { k_mix.sr } else { 1.0 };

    // Meta-GGA (SCAN / r2SCAN) ROKS is stiffer than LDA/GGA and limit-cycles
    // under plain DIIS; apply the same modest default virtual-block level shift
    // the closed-shell RKS path uses (see solve_rhf) when the user hasn't set
    // one. Ramped to zero as the gradient converges. Meta-GGA falls back to DIIS
    // (no fxc Newton kernel), so this shift is the relevant stabilizer.
    let effective_level_shift = crate::driver::effective_level_shift(config);

    // RSH path (ω > 0): per-spin K from c_SR · K[erfc(ω)] + c_LR · K[erf(ω)]
    // via two DfK fitters (geometry-only — built once, contracted per spin
    // per iter). Mirrors solve_uhf's RSH path.
    // Shared geometry-only environment: S, hcore(+external+ECP — V_ECP folds
    // into hcore once, byte-identical to plain hcore for all-electron bases),
    // V_nn(+external), COSMO/PCM contexts, resolved memory budget, RSH
    // fitters. One construction serving all six SCF variants (crate::driver).
    let crate::driver::ScfEnv {
        s,
        h,
        vnn,
        ooc_budget,
        cosmo_cavity,
        pcm_ctx,
        polarizable_site_basis,
        mut dfk_sr,
        mut dfk_lr,
    } = crate::driver::prepare(ctx, mol, prep, config, &k_mix)?;
    let n = prep.nbasis();
    let nelec = mol.nelec() as i64;
    let mult = mol.multiplicity as i64;
    if mult < 1 {
        return Err(FerricError::General(
            "ROHF: multiplicity must be >= 1".into(),
        ));
    }
    let two_s = mult - 1; // 2S = number of singly-occupied (α) orbitals
    if (nelec - two_s) % 2 != 0 || nelec < two_s {
        return Err(FerricError::General(format!(
            "ROHF: incompatible nelec={nelec} and multiplicity={mult}"
        )));
    }
    let nocc_open = two_s as usize;
    let nocc_double = ((nelec - two_s) / 2) as usize;
    let nocc_a = nocc_double + nocc_open;
    let nocc_b = nocc_double;
    if nocc_a + nocc_b != nelec as usize {
        return Err(FerricError::General(
            "ROHF: nocc_a + nocc_b != nelec".into(),
        ));
    }
    // S^{-1/2}
    let (s_evals, s_evecs) = s
        .eigh(ndarray_linalg::UPLO::Upper)
        .map_err(|e| FerricError::Lapack(format!("S diag: {e}")))?;
    let mut u_scaled = s_evecs.clone();
    for i in 0..n {
        let scale = 1.0 / s_evals[i].sqrt();
        for mu in 0..n {
            u_scaled[(mu, i)] *= scale;
        }
    }
    let s_inv_sqrt = u_scaled.dot(&s_evecs.t());

    // Initial guess MOs: the CONFIGURED density guess (MINAO by default), else
    // hcore.
    //
    // # Why this is not just hcore any more
    //
    // Until 2026-09-17 this block ALWAYS used hcore: it called `hcore_guess`
    // purely as a "sanity check it succeeds", threw the density away with
    // `let _ =`, and diagonalized bare `h`. `RhfConfig::init_guess_density` and
    // `use_sad_guess` — which `rhf.rs` honours, and which default to the MINAO
    // projection — were referenced NOWHERE in this file, so a caller who
    // explicitly asked for a better open-shell guess silently got hcore. This
    // is the ROHF half of the defect fixed for UHF at `b687c394`; the two
    // solvers carried the identical pattern.
    //
    // MEASURED, `tests/rohf_state_selection.rs`, over 28 rows (15 chemical
    // systems, 13 of them at two bases; charged and neutral, doublet and
    // triplet, diatomic and polyatomic) against PySCF 2.13.0 ROHF at the same
    // geometry and basis. Each reference was cross-checked across five PySCF
    // `init_guess` settings and, for the wider sweep, 24 randomized starts, so
    // a guess-dependent reference is not mistaken for a converged one.
    //
    //                      above ref by >1e-3 eV    did not converge
    //   hcore  (pre-fix)        4 / 28                  2 / 28
    //   MINAO  (post-fix)       1 / 28                  0 / 28
    //
    // Repaired: OH/6-31G +4.3027 → 0, F₂⁺/6-31G +3.0201 → 0, NH₂/6-31G
    // +1.7818 → 0, HeNe⁺/def2-SVP +0.1303 → 0, plus HeNe⁺/6-31G and
    // CN/cc-pVDZ, which previously exhausted 400 iterations without converging.
    //
    // NOT FREE: CN is made WORSE at both bases — CN/6-31G +0.5753 eV (it
    // reached the reference from hcore) and CN/cc-pVDZ +0.3865 eV (which trades
    // a non-answer for a high answer). Those are genuine ROHF stationary points
    // of the same operator, not wrong energies: PySCF itself converges to
    // ferric's −92.1186236 from 5 of 40 randomized starts. See
    // `cn_is_the_system_the_guess_fix_costs`, which pins both magnitudes.
    //
    // The remaining rows are unchanged, several of them BIT-identically,
    // because those systems have one basin and the guess cannot matter.
    //
    // # No stability descent here, deliberately
    //
    // The UHF fix has a SECOND half: an opt-in `scf_stability_descent` that
    // follows a downhill orbital-Hessian eigenvector when the guess is not
    // enough. That is UNAVAILABLE to ROHF and this function does not fake it.
    // `crate::stability` implements exactly two operators (UHF and RHF), and
    // [`crate::stability::StabilitySkip::Rohf`] exists specifically to record
    // that the Roothaan open-shell Hessian is a THIRD one — one MO set with
    // closed/open/virtual blocks and Roothaan coupling, not a special case of
    // either. Running a UHF descent on ROHF MOs would follow an eigenvector of
    // an operator these orbitals are not a stationary point of. The convergence
    // exit below prints that skip reason when `check_stability` is set, so a
    // ROHF solution that this guess does not repair is DETECTABLE rather than
    // silently reported as the ground state.
    let mut c = match rohf_guess_mos(
        ctx,
        mol,
        prep,
        bounds,
        config,
        &h,
        &s,
        &s_inv_sqrt,
        nocc_double,
        nocc_open,
    )? {
        Some(c_guess) => c_guess,
        None => {
            // hcore guess: diagonalize bare h, exactly as this path did
            // unconditionally before.
            let _ = hcore_guess(&s, &h, nocc_a.max(1))?; // sanity check it succeeds
            let h_prime = s_inv_sqrt.dot(&h).dot(&s_inv_sqrt);
            let (_, c_prime) = h_prime
                .eigh(ndarray_linalg::UPLO::Upper)
                .map_err(|e| FerricError::Lapack(format!("H' diag: {e}")))?;
            s_inv_sqrt.dot(&c_prime)
        }
    };

    // ROHF densities (AO):
    //   D_c (closed/doubly-occupied) = 2 Σ_i C_i C_i^T  (i = 0..nocc_double)
    //   D_o (open/singly-α-occupied) = Σ_j C_j C_j^T    (j = nocc_double..nocc_a)
    // D_α = D_c/2 + D_o,  D_β = D_c/2  → (D_α + D_β) = D_c + D_o (total).
    // We track D_α and D_β internally to feed J/K builders (matching UHF JK).
    let (mut d_a, mut d_b) = build_rohf_densities(&c, nocc_double, nocc_open);

    let mut j_buf = Array2::<f64>::zeros((n, n));
    let mut k_a_buf = Array2::<f64>::zeros((n, n));
    let mut k_b_buf = Array2::<f64>::zeros((n, n));

    let mut diis = Diis::new(config.diis_size);
    // ROHF extrapolates the effective Fock with `step` (single-spin), so two
    // n×n matrices per entry despite the α/β densities it tracks internally.
    crate::driver::warn_if_diis_history_large(
        "ROHF",
        n,
        config.diis_size,
        crate::diis::DiisHistoryShape::SingleSpin,
        false,
        ooc_budget,
    );
    // Convergence bookkeeping (prev energy, ΔP signals, divergence streak,
    // stall history) — shared driver::ScfMonitor.
    // Most recent Thole-damped polarizable-embedding dipoles (None when
    // `config.polarizable` is off/empty) — threaded into both ScfResult
    // constructors below (converged early-exit and MaxIter).
    let mut last_induced_dipoles: Option<Array2<f64>> = None;
    let mut mon = crate::driver::ScfMonitor::new();
    // MOM reference: (closed-MO AO block, open-MO AO block) from the most
    // recently accepted iter. None until iter `config.mom_after_iter`.
    let mut mom_ref: Option<(Array2<f64>, Array2<f64>)> = None;
    let mut total_quartets = 0usize;
    // Last effective Fock, retained so the non-converged exit can report a
    // well-formed ScfResult (see solve_rohf_best_effort's tail).
    let mut f_eff_last = Array2::<f64>::zeros((n, n));
    // The spin Focks behind `f_eff_last`, kept for the gradient's
    // energy-weighted density (see `ScfResult::rohf_spin_focks`).
    let mut spin_focks_last: Option<(Array2<f64>, Array2<f64>)> = None;
    // Previous iteration's total density, for the ΔP convergence signal (shared
    // with solve_rhf via rhf::scf_converged). None on iter 1 → dp = INFINITY, so
    // the gate can't fire before a real density change exists.
    let mut prev_d_total: Option<Array2<f64>> = None;

    // K built per spin (same convention as solve_uhf's RSH path). Builders
    // hoisted out of the loop: each lazily builds a per-thread libint2
    // EnginePool on first use (ctors serialized behind a global mutex), so a
    // loop-local builder would pay that construction every iteration.
    let need_k = c_k != 0.0 || k_mix.omega > 0.0;
    let coulomb_op = bounds.op;
    let j_aux_eff = if k_mix.omega == 0.0 {
        config.df_j_aux.as_deref()
    } else {
        None
    };
    let k_aux_eff = if need_k && k_mix.omega == 0.0 {
        config.df_k_aux.as_deref()
    } else {
        None
    };
    let (mut df_j, mut df_k) = crate::fock_assembly::build_df_jk(
        ctx, mol, coulomb_op, prep, j_aux_eff, k_aux_eff, ooc_budget,
    )?;
    // Record the builders that produced the energy (same effective names as
    // `build_df_jk` above and `driver::prepare`'s RSH pair) so the analytic
    // gradient differentiates THIS energy. `None` when nothing is fitted.
    let df_jk_route = crate::result::DfJkRoute::from_scf(
        j_aux_eff,
        k_aux_eff,
        (k_mix.omega > 0.0).then(|| {
            (
                config
                    .df_k_aux
                    .as_deref()
                    .unwrap_or(crate::fock_assembly::DEFAULT_JK_AUX),
                k_mix.omega,
            )
        }),
        coulomb_op,
        ooc_budget,
    );
    // Combined open-shell direct J+K + incremental Fock — identical scheme and
    // identical gating to `solve_uhf`; see the block comments there (and on
    // `DirectJK::build_uhf` / `build_uhf_incremental`) for the rationale and for
    // the |ΔD_α| + |ΔD_β| screening-key choice. ROHF's Fock build is
    // structurally the same three-matrix (J[D_total], K[D_α], K[D_β]) problem as
    // UHF — the ROHF-specific work is the Roothaan coupling applied to the
    // ASSEMBLED f_a/f_b further below, which this does not touch.
    // Pluggable exchange builder ("link" / "cosx") — identical semantics,
    // warnings, hard errors and per-spin `update_density` contract to
    // `solve_uhf`; see the block comment there and
    // `fock_assembly::resolve_k_builder` / `build_pluggable_k`. Until
    // 2026-09-08 `solve_rohf` never read `config.k_builder`. The ROHF-specific
    // work (Roothaan coupling of the ASSEMBLED f_a/f_b) is downstream of K and
    // is untouched by this.
    let pluggable_k_kind = crate::fock_assembly::resolve_k_builder(
        config.k_builder.as_deref(),
        df_j.is_some() || df_k.is_some(),
        df_k.is_some(),
        need_k,
        k_mix.omega,
    )?;
    let pluggable_k_kind =
        crate::fock_assembly::narrow_k_builder_to_supported(pluggable_k_kind, need_k, k_mix.omega);
    // Same `LinkBound::SchwarzRef` adapter and same rationale as `solve_uhf`'s
    // — see the comment there. No table on `bounds` (the default) makes this
    // byte-identical to passing `bounds` directly.
    let link_bound = crate::screening::LinkBound::SchwarzRef(bounds);
    let mut pluggable_k: Option<Box<dyn KBuilder>> = crate::fock_assembly::build_pluggable_k(
        pluggable_k_kind,
        ctx,
        mol,
        prep,
        &link_bound,
        coulomb_op,
        &config.cosx,
        config.integral_thresh,
        ooc_budget,
    )?;

    let combined_direct_jk = df_j.is_none()
        && df_k.is_none()
        && need_k
        && k_mix.omega == 0.0
        && pluggable_k.is_none()
        && crate::direct_jk::combined_open_shell_jk_enabled();
    let mut direct_jk: Option<crate::direct_jk::DirectJK> = if combined_direct_jk {
        Some(crate::direct_jk::DirectJK::new(
            ctx,
            prep,
            bounds,
            config.integral_thresh,
            ooc_budget,
        ))
    } else {
        None
    };
    let mut direct_j: Option<DirectJ> = if df_j.is_none() && !combined_direct_jk {
        Some(DirectJ::new(
            ctx,
            prep,
            bounds,
            config.integral_thresh,
            ooc_budget,
        ))
    } else {
        None
    };
    let mut direct_k: Option<DirectK> = if need_k
        && k_mix.omega == 0.0
        && df_k.is_none()
        && !combined_direct_jk
        && pluggable_k.is_none()
    {
        Some(DirectK::new(
            ctx,
            prep,
            bounds,
            config.integral_thresh,
            ooc_budget,
        ))
    } else {
        None
    };
    // NOTE: opt-IN (default off), unlike the closed-shell path. See
    // `direct_jk::open_shell_incremental_enabled` for the measurements behind
    // that choice — the scheme is correct but measured ~1.00-1.05x here.
    // A pluggable K is never fed a ΔD: `combined_direct_jk` is gated on
    // `pluggable_k.is_none()`, so `direct_jk` (and hence the incremental path)
    // is off whenever one is active — see the fuller note in `solve_uhf`.
    let incremental_direct =
        direct_jk.is_some() && crate::direct_jk::open_shell_incremental_enabled();
    let mut d_last_fock: Option<(Array2<f64>, Array2<f64>)> = None;
    const INCREMENTAL_FULL_REBUILD_EVERY: usize = 8;

    for iter in 1..=config.max_iter {
        ctx.check_interrupted()?;
        let direct_full_rebuild = incremental_direct
            && (d_last_fock.is_none() || iter % INCREMENTAL_FULL_REBUILD_EVERY == 1);
        let direct_incremental = incremental_direct && !direct_full_rebuild;
        if !direct_incremental {
            j_buf.fill(0.0);
            k_a_buf.fill(0.0);
            k_b_buf.fill(0.0);
        }
        let d_total = &d_a + &d_b;
        // ΔP vs the previous iteration's total density — the primary convergence
        // signal (see rhf::scf_converged). The monitor stays at INFINITY until
        // the first recorded change (iter 1).
        if let Some(prev) = prev_d_total.as_ref() {
            mon.record_density_change(&d_total, prev);
        }
        prev_d_total = Some(d_total.clone());

        // Combined single-pass J + K_α + K_β when enabled; otherwise J alone here.
        if let Some(djk) = direct_jk.as_mut() {
            if direct_incremental {
                let (da_prev, db_prev) = d_last_fock
                    .as_ref()
                    .expect("d_last_fock set on full rebuild");
                let delta_a = &d_a - da_prev;
                let delta_b = &d_b - db_prev;
                let delta_total = &delta_a + &delta_b;
                total_quartets += djk.build_uhf_incremental(
                    &delta_total,
                    &delta_a,
                    &delta_b,
                    &mut j_buf,
                    &mut k_a_buf,
                    &mut k_b_buf,
                )?;
            } else {
                total_quartets +=
                    djk.build_uhf(&d_total, &d_a, &d_b, &mut j_buf, &mut k_a_buf, &mut k_b_buf)?;
            }
            d_last_fock = Some((d_a.clone(), d_b.clone()));
        } else if let Some(dfj) = df_j.as_mut() {
            // J from D_total — DF-J if configured, else direct.
            dfj.build(&d_total, &mut j_buf)?;
        } else {
            let dj = direct_j.as_mut().expect("DirectJ built before loop");
            total_quartets += dj.build(&d_total, &mut j_buf)?;
        }
        // F_σ = H + J − K_σ_total  (then + V^σ_xc below for ROKS), assembled
        // in place — no k_σ_total clones for HF/plain hybrids and no zeros
        // allocations for pure DFT. `f_σ += (−c)·K` is bit-identical to the
        // former `f_σ −= c·&K` clone path (sign flip is exact); the RSH branch
        // keeps the explicit SR/LR combination for the same reason.
        let mut f_a: Array2<f64> = &h + &j_buf;
        let mut f_b: Array2<f64> = &h + &j_buf;
        if k_mix.omega > 0.0 {
            let dfk_sr = dfk_sr.as_mut().expect("dfk_sr built when omega>0");
            let dfk_lr = dfk_lr.as_mut().expect("dfk_lr built when omega>0");
            // occ-path DISABLED (always None) — see the matching note in
            // rhf.rs: the DF-K half-transform's B-tensor reassociation differs
            // from the density path at the f64 floor, which measurably
            // prevented RHF from settling on benzene/def2-svp. Reverted to
            // always using the exact density contraction here too.
            crate::fock_assembly::subtract_rsh_exchange(
                dfk_sr, dfk_lr, &d_a, None, 1.0, &mut f_a, k_mix.sr, k_mix.lr, 1.0,
            )?;
            crate::fock_assembly::subtract_rsh_exchange(
                dfk_sr, dfk_lr, &d_b, None, 1.0, &mut f_b, k_mix.sr, k_mix.lr, 1.0,
            )?;
        } else if need_k {
            if direct_jk.is_some() {
                // K_α/K_β already filled by the combined single-pass build above.
            } else if let Some(kb) = pluggable_k.as_mut() {
                // Per-spin `update_density(D_σ)` + `build(D_σ)` from one shared
                // instance (see fock_assembly::build_open_shell_pluggable_k).
                total_quartets += crate::fock_assembly::build_open_shell_pluggable_k(
                    kb.as_mut(),
                    &d_a,
                    &d_b,
                    &mut k_a_buf,
                    &mut k_b_buf,
                )?;
            } else if let Some(dfk) = df_k.as_mut() {
                dfk.build(&d_a, &mut k_a_buf)?;
                dfk.build(&d_b, &mut k_b_buf)?;
            } else {
                let dk = direct_k.as_mut().expect("DirectK built before loop");
                total_quartets += <DirectK as KBuilder>::build(dk, &d_a, &mut k_a_buf)?;
                total_quartets += <DirectK as KBuilder>::build(dk, &d_b, &mut k_b_buf)?;
            }
            f_a.scaled_add(-c_k, &k_a_buf);
            f_b.scaled_add(-c_k, &k_b_buf);
        }

        // Pre-XC electronic energy.
        let e_elec_no_xc: f64 = 0.5 * ((&(&h + &f_a) * &d_a).sum() + (&(&h + &f_b) * &d_b).sum());
        let e_xc = if let Some(x) = xc_contrib.as_ref() {
            x.add_xc_uks(&d_a, &d_b, &mut f_a, &mut f_b)
        } else {
            0.0
        };

        // COSMO + IEF-PCM reaction fields, built from the total density
        // D_a + D_b and added identically to BOTH spin Focks (before the
        // Roothaan effective-Fock combination, so the coupled single-MO-set
        // ROHF equations feel them); energies are standalone terms — see
        // driver::solvent_terms and rhf::solve_rhf for the derivation notes.
        let (e_cosmo, e_pcm, e_pol, iter_induced_dipoles) = crate::driver::solvent_terms(
            mol,
            prep,
            config,
            cosmo_cavity.as_ref(),
            pcm_ctx.as_ref(),
            polarizable_site_basis.as_ref(),
            &d_total,
            &mut [&mut f_a, &mut f_b],
        )?;
        last_induced_dipoles = iter_induced_dipoles;

        let energy = e_elec_no_xc + e_xc + e_cosmo + e_pcm + e_pol + vnn;

        // Build Roothaan effective Fock (Guest-Saunders, via PySCF projector form).
        let f_eff = roothaan_fock(&f_a, &f_b, &d_a, &d_b, &s);
        f_eff_last = f_eff.clone();
        spin_focks_last = Some((f_a.clone(), f_b.clone()));

        // DIIS error: the proper ROHF orbital-rotation gradient (PySCF
        // `get_grad`). In MO basis the gradient has only three nonzero
        // off-diagonal blocks — the *unique* orbital rotations:
        //   g[v,c] = f_α[v,c] + f_β[v,c]    (closed → virtual)
        //   g[v,o] = f_α[v,o]               (open → virtual; only α occupies open)
        //   g[o,c] = f_β[o,c]               (closed → open; only β leaves open)
        // We then antisymmetrize (g - g^T) and project back to AO basis so
        // DIIS still operates on (n × n) error matrices. This eliminates the
        // within-class-rotation ambiguity that causes the LDA/PBE doublet-OH
        // plateau in the FDS-SDF formulation.
        let f_a_mo: Array2<f64> = c.t().dot(&f_a).dot(&c);
        let f_b_mo: Array2<f64> = c.t().dot(&f_b).dot(&c);
        let mut g_mo: Array2<f64> = Array2::zeros((n, n));
        // closed → virtual block: rows = virtual, cols = closed
        for p in nocc_a..n {
            for q in 0..nocc_double {
                g_mo[(p, q)] = f_a_mo[(p, q)] + f_b_mo[(p, q)];
            }
        }
        // open → virtual block: rows = virtual, cols = open
        for p in nocc_a..n {
            for q in nocc_double..nocc_a {
                g_mo[(p, q)] = f_a_mo[(p, q)];
            }
        }
        // closed → open block: rows = open, cols = closed
        for p in nocc_double..nocc_a {
            for q in 0..nocc_double {
                g_mo[(p, q)] = f_b_mo[(p, q)];
            }
        }
        // Antisymmetrize in MO basis, then transform back to AO.
        let g_mo_anti: Array2<f64> = &g_mo - &g_mo.t();
        let err: Array2<f64> = s.dot(&c).dot(&g_mo_anti).dot(&c.t()).dot(&s);

        let sig = mon.signals(energy);
        let de = sig.de;
        let err_max = err.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        // Converge on ΔP + loose ΔE, the same gate as solve_rhf/solve_uhf — the
        // ROHF cation path (used as a gw100 fallback) hits the same RI energy/
        // gradient noise floor at aTZ, so the old `err_max < density_conv` gate
        // would grind to MaxIter there too. See rhf::scf_converged.
        let conv_exit = crate::rhf::scf_converged(sig, config.energy_conv, config.density_conv);
        let converged = conv_exit.is_some();

        // Divergence / stall early exits (shared driver::ScfMonitor; both are
        // no-ops at the None defaults — ROHF previously ignored these knobs).
        if mon.diverging(energy, config.divergence_tol) || mon.stalled(err_max, config.stall_window)
        {
            return Err(FerricError::ScfConvergence {
                iterations: iter,
                last_energy: mon.prev_e,
            });
        }

        // FERRIC_ROHF_TRACE=1: per-iter diagnostic of plateau dynamics.
        // Logs (iter, energy, ΔE, err_max, per-block gradient max,
        // eigenvalues straddling SOMO, and the occupied-α↔virt overlap).
        if rohf_trace() {
            let (eps_now, _c_now) = diagonalize(&f_eff, &s_inv_sqrt)?;
            // Eigenvalues around the SOMO. With nocc_double β-pairs and
            // nocc_open singly-α-occupied orbitals, the SOMO index range is
            // [nocc_double .. nocc_double+nocc_open) and the LUMO starts at
            // nocc_a = nocc_double + nocc_open.
            let nlow = nocc_double.saturating_sub(2);
            let nhi = (nocc_a + 2).min(n);
            let eps_window: Vec<String> = (nlow..nhi)
                .map(|i| {
                    let tag = if i < nocc_double {
                        " D "
                    } else if i < nocc_a {
                        " S "
                    } else {
                        " V "
                    };
                    format!("[{i}{tag}{:.4}]", eps_now[i])
                })
                .collect();
            // Per-block gradient maxima from g_mo (pre-antisymmetrize).
            let mut g_vc_max = 0.0f64;
            let mut g_vo_max = 0.0f64;
            let mut g_oc_max = 0.0f64;
            for p in nocc_a..n {
                for q in 0..nocc_double {
                    g_vc_max = g_vc_max.max(g_mo[(p, q)].abs());
                }
                for q in nocc_double..nocc_a {
                    g_vo_max = g_vo_max.max(g_mo[(p, q)].abs());
                }
            }
            for p in nocc_double..nocc_a {
                for q in 0..nocc_double {
                    g_oc_max = g_oc_max.max(g_mo[(p, q)].abs());
                }
            }
            eprintln!(
                "ROHFTRACE it={iter:>3} E={energy:.10} dE={de:.3e} err={err_max:.3e} |g|vc={g_vc_max:.3e} |g|vo={g_vo_max:.3e} |g|oc={g_oc_max:.3e}  eps:{}",
                eps_window.join(" ")
            );
        }

        // Live per-iteration progress (see solve_rhf's identical block for the
        // full rationale). STDOUT, opt-in via `config.verbose`, rank-0-only.
        // Uses dp_rms/dp_max (same ΔP quantities `scf_converged` gates on
        // above) rather than the ROHFTRACE-specific per-block gradients, for a
        // format consistent with the RHF/UHF verbose line.
        if config.verbose && ctx.is_root() {
            println!(
                "ROHF iter={iter:4}  E={energy:.10}  dE={de:.3e}  dp_rms={:.3e}  err_max={err_max:.3e}",
                mon.dp_rms
            );
        }

        // Machine-readable per-iteration record, streamed and flushed NOW. See
        // the identical block in `solve_rhf` for the full rationale; every
        // value is one the loop already computed for the gate above.
        // `grad_rms` is `None`: this loop forms only `err_max` from the
        // antisymmetrized MO gradient, never an RMS.
        if ctx.is_root() {
            if let Some(rl) = crate::runlog::log() {
                rl.scf_iter(
                    if xc_contrib.is_some() { "roks" } else { "rohf" },
                    crate::runlog::current_rung(),
                    iter,
                    energy,
                    de,
                    mon.dp_rms,
                    mon.dp_max,
                    err_max,
                    None,
                );
            }
        }

        if iter > 1 && converged {
            let (eps, c_f) = diagonalize(&f_eff, &s_inv_sqrt)?;
            let (d_a_f, d_b_f) = build_rohf_densities(&c_f, nocc_double, nocc_open);
            let density_total = &d_a_f + &d_b_f;
            // Stability analysis is NOT available for a ROHF/ROKS reference:
            // the Roothaan orbital Hessian is a third operator (one MO set with
            // closed/open/virtual blocks and Roothaan coupling), not a special
            // case of either implemented one, so running `uhf_internal_stability`
            // or `rhf_internal_stability` here would analyse a Hessian these MOs
            // are not a stationary point of. Skipped with a printed reason
            // rather than silently returning a wrong-operator verdict; the
            // result field stays `None`, which is documented as "not checked".
            if config.check_stability {
                eprintln!(
                    "SCF stability: check requested but SKIPPED — {}. \
                     ScfResult::stability is None (not checked), which does NOT mean stable.",
                    crate::stability::StabilitySkip::Rohf.reason()
                );
            }
            return Ok(ScfResult {
                spin: Spin::RestrictedOpen,
                energy,
                density_total,
                density_alpha: d_a_f,
                density_beta: Some(d_b_f),
                mos_alpha: c_f,
                mos_beta: None,
                eps_alpha: eps,
                eps_beta: None,
                fock_alpha: f_eff,
                fock_beta: None,
                converged: true,
                exit: crate::result::ScfExit::Converged,
                iterations: iter,
                computed_quartets: total_quartets,
                induced_dipoles: last_induced_dipoles,
                stability: None,
                df_jk: df_jk_route.clone(),
                rohf_spin_focks: spin_focks_last.clone(),
            });
        }
        mon.note_energy(energy);

        // Pick update strategy. Newton step (if enabled and below trigger)
        // uses per-spin diagonal Fock entries to precondition a PCG solve of
        // H·κ = −g, then rotates C via the Cayley unitary.
        //   - HF (xc=None): full Newton with HF orbital Hessian only.
        //   - LDA ROKS: Newton + LDA f_xc kernel response (LdaFxcKernel).
        //   - GGA / hybrid / RSH-GGA ROKS: Newton + GGA f_xc kernel response
        //     (GgaFxcKernel — adds the σ = |∇ρ|² coupling terms). The hybrid /
        //     RSH exact-exchange fraction is folded in via `k_mix_sr`/the K
        //     builders exactly as in the Fock build, so the fxc kernel only
        //     ever supplies the semilocal DFT second-derivative response.
        //     Meta-GGA is still excluded (rejected upstream — no τ kernel).
        //
        // `xc_supports_newton_fxc` gates whether an f_xc-accelerated Newton/AH
        // step is available for this functional at all. LDA/GGA/hybrid/RSH all
        // have an f_xc kernel (LdaFxcKernel / GgaFxcKernel). Meta-GGA (SCAN /
        // r2SCAN) is energy-only in Phase A — there is no τ-dependent f_xc
        // kernel — so it falls back to plain DIIS here. Pure ROHF (xc=None)
        // keeps the HF-only Hessian path.
        let xc_supports_newton_fxc = xc_contrib.is_some() && !xc_is_metagga(config.xc.as_deref());
        // Augmented-Hessian Newton — used when err_max is below ah_trigger.
        // Handles vanishing Hessian eigenvalues (e.g., doublet OH at LDA with
        // near-degenerate SOMO/HOMO) that PCG can't resolve.
        let use_ah = config.ah_trigger > 0.0
            && iter > 3
            && err_max < config.ah_trigger
            && (xc_contrib.is_none() || xc_supports_newton_fxc)
            && k_mix.omega == 0.0;
        if use_ah {
            let (_, c_now) = diagonalize(&f_eff, &s_inv_sqrt)?;
            let f_a_mo = c_now.t().dot(&f_a).dot(&c_now);
            let f_b_mo = c_now.t().dot(&f_b).dot(&c_now);

            let fxc_store = if xc_supports_newton_fxc {
                let main = config.dft_grid.clone().unwrap_or_default();
                let name = config
                    .xc
                    .as_deref()
                    .expect("xc_supports_newton_fxc implies Some(xc)");
                Some(FxcKernelStore::build(mol, prep, &main, name, &d_a, &d_b)?)
            } else {
                None
            };
            let fxc_storage = fxc_store.as_ref().map(|s| s.response());
            let fxc_ref: Option<&crate::rohf_newton::FxcResponse<'_>> = fxc_storage.as_deref();

            let inputs = crate::rohf_newton::RohfNewtonInputs {
                prep,
                bounds,
                c: &c_now,
                f_a_mo: &f_a_mo,
                f_b_mo: &f_b_mo,
                nocc_double,
                nocc_open,
                k_mix_sr: if k_mix.omega > 0.0 { 0.0 } else { c_k },
                fxc: fxc_ref,
                thresh: config.integral_thresh,
                ooc_budget,
            };
            let ah_inputs = crate::rohf_ah::RohfAhInputs { base: &inputs };
            let (c_new, _kmax) = crate::rohf_ah::rohf_ah_step(
                ctx, &ah_inputs, /*max_step=*/ 0.2, /*davidson_conv=*/ 1e-7,
                /*davidson_max_vecs=*/ 50,
            )?;
            c = c_new;
            let (da_n, db_n) = build_rohf_densities(&c, nocc_double, nocc_open);
            d_a = da_n;
            d_b = db_n;
            continue;
        }
        let use_newton = config.newton_trigger > 0.0
            && iter > 3
            && err_max < config.newton_trigger
            && (xc_contrib.is_none() || xc_supports_newton_fxc);
        if use_newton {
            let (_, c_now) = diagonalize(&f_eff, &s_inv_sqrt)?;
            let f_a_mo = c_now.t().dot(&f_a).dot(&c_now);
            let f_b_mo = c_now.t().dot(&f_b).dot(&c_now);

            // Build the f_xc kernel (LDA or GGA) + reference density (closure
            // target). Kept here so the response closure borrows it; the heavy
            // grid/AO evaluation happens once per Newton step, not per matvec.
            let fxc_store = if xc_supports_newton_fxc {
                let main = config.dft_grid.clone().unwrap_or_default();
                let name = config
                    .xc
                    .as_deref()
                    .expect("xc_supports_newton_fxc implies Some(xc)");
                Some(FxcKernelStore::build(mol, prep, &main, name, &d_a, &d_b)?)
            } else {
                None
            };
            let fxc_storage = fxc_store.as_ref().map(|s| s.response());
            let fxc_ref: Option<&crate::rohf_newton::FxcResponse<'_>> = fxc_storage.as_deref();

            let inputs = crate::rohf_newton::RohfNewtonInputs {
                prep,
                bounds,
                c: &c_now,
                f_a_mo: &f_a_mo,
                f_b_mo: &f_b_mo,
                nocc_double,
                nocc_open,
                k_mix_sr: if k_mix.omega > 0.0 { 0.0 } else { c_k },
                fxc: fxc_ref,
                thresh: config.integral_thresh,
                ooc_budget,
            };
            let (c_new, _kmax) = crate::rohf_newton::rohf_newton_step(
                ctx,
                &inputs,
                config.level_shift.max(1e-6),
                0.1, // trust radius (conservative — ROKS hessians are stiff)
                20,
                1e-7,
            )?;
            c = c_new;
            let (da_n, db_n) = build_rohf_densities(&c, nocc_double, nocc_open);
            d_a = da_n;
            d_b = db_n;
        } else {
            // DIIS extrapolate effective Fock, then optionally level-shift the
            // virtual–virtual block in MO basis to damp open-shell oscillations.
            let mut f_new = diis.step(&f_eff, &err);
            if effective_level_shift > 0.0 && iter > 1 {
                const SHIFT_DAMP_ERR: f64 = 1e-3;
                let damp = err_max / (err_max + SHIFT_DAMP_ERR);
                let shift_eff = effective_level_shift * damp;
                if shift_eff > 1e-10 {
                    let c_virt = c.slice(ndarray::s![.., nocc_a..]);
                    let p_virt: Array2<f64> = c_virt.dot(&c_virt.t());
                    let shift_term: Array2<f64> = shift_eff * s.dot(&p_virt).dot(&s);
                    f_new += &shift_term;
                }
            }
            let (_, c_new) = diagonalize(&f_new, &s_inv_sqrt)?;
            let c_after_mom = if config.mom_after_iter > 0 && iter > config.mom_after_iter {
                match mom_ref.as_ref() {
                    Some((ref_closed, ref_open)) => crate::mom::mom_reorder(
                        &c_new,
                        &s,
                        ref_closed,
                        ref_open,
                        nocc_double,
                        nocc_open,
                    ),
                    None => c_new,
                }
            } else {
                c_new
            };
            c = c_after_mom;
            if config.mom_after_iter > 0 && iter >= config.mom_after_iter {
                let ref_closed = c.slice(ndarray::s![.., ..nocc_double]).to_owned();
                let ref_open = c
                    .slice(ndarray::s![.., nocc_double..nocc_double + nocc_open])
                    .to_owned();
                mom_ref = Some((ref_closed, ref_open));
            }
            let (da_n, db_n) = build_rohf_densities(&c, nocc_double, nocc_open);
            d_a = da_n;
            d_b = db_n;
        }
    }
    // Max iterations reached. Build the best-effort state and let the caller decide.
    //
    // Same split as solve_uhf/solve_uhf_best_effort: `solve_rohf` keeps its historical
    // Err contract (existing callers unwrap or `?` it), while
    // `solve_rohf_best_effort` returns the converged-so-far density and MOs so a
    // convergence LADDER can carry them into the next rung.
    let (eps_last, c_last) =
        diagonalize(&f_eff_last, &s_inv_sqrt).unwrap_or_else(|_| (vec![0.0; c.ncols()], c.clone()));
    let density_total = &d_a + &d_b;
    Ok(ScfResult {
        spin: Spin::RestrictedOpen,
        energy: mon.prev_e,
        density_total,
        density_alpha: d_a,
        density_beta: Some(d_b),
        mos_alpha: c_last,
        mos_beta: None,
        eps_alpha: eps_last,
        eps_beta: None,
        fock_alpha: f_eff_last,
        fock_beta: None,
        converged: false,
        exit: crate::result::ScfExit::MaxIter,
        iterations: config.max_iter,
        computed_quartets: total_quartets,
        induced_dipoles: last_induced_dipoles,
        stability: None,
        df_jk: df_jk_route,
        rohf_spin_focks: spin_focks_last,
    })
}

/// Build ROHF α/β densities from MO coefficients:
///   D_β = Σ_{i<nocc_double} C_i C_i^T
///   D_α = D_β + Σ_{j∈open} C_j C_j^T
fn build_rohf_densities(
    c: &Array2<f64>,
    nocc_double: usize,
    nocc_open: usize,
) -> (Array2<f64>, Array2<f64>) {
    let n = c.nrows();
    let mut d_b = Array2::<f64>::zeros((n, n));
    if nocc_double > 0 {
        let cd = c.slice(ndarray::s![.., ..nocc_double]);
        d_b = cd.dot(&cd.t());
    }
    let mut d_a = d_b.clone();
    if nocc_open > 0 {
        let co = c.slice(ndarray::s![.., nocc_double..nocc_double + nocc_open]);
        d_a = &d_a + &co.dot(&co.t());
    }
    (d_a, d_b)
}

/// Roothaan effective Fock (Guest-Saunders coupling).
/// Mirrors `pyscf.scf.rohf.get_roothaan_fock`.
/// Build the ROHF initial MOs from the CONFIGURED density guess.
///
/// Returns `Ok(None)` — meaning "use hcore" — when the caller asked for the bare
/// hcore guess (`use_sad_guess = false` with no explicit density), and also
/// whenever the configured guess cannot be built. A guess that fails is never
/// fatal: hcore is what this path did unconditionally until 2026-09-17, so
/// falling back to it can only reproduce the old behavior, never break a system
/// that used to work. Mirrors `uhf::uhf_guess_mos`.
///
/// # How a density becomes ROHF MOs
///
/// The SCF is MO-driven, so a guess DENSITY has to become occupied orbitals.
/// That is done the only way it can be: build the Fock AT the guess density and
/// diagonalize it. Two choices here are ROHF-flavoured rather than copied from
/// `uhf_guess_mos`, and BOTH are recorded below with what measurement actually
/// supports them — which is less than the obvious argument would suggest.
///
/// 1. **The spin split is by OCCUPATION.** Both `sad_guess` and the MINAO
///    projection return a spin-summed `D_total`. UHF splits it evenly
///    (`D_α = D_β = D/2`); ROHF has `nocc_α ≠ nocc_β` by construction, so
///    here `D_σ = D_total · nocc_σ / nelec`, which preserves both
///    `tr(D_α S) = nocc_α` and `tr(D_β S) = nocc_β` and reduces to the UHF
///    even split exactly when `nocc_α == nocc_β`.
/// 2. **The matrix diagonalized is the ROOTHAAN EFFECTIVE Fock**, not a spin
///    Fock — `roothaan_fock(F_α, F_β, D_α, D_β, S)`, the same function the
///    SCF loop uses. ROHF has ONE MO set, so this is the operator the iteration
///    goes on to use.
///
/// ## MEASURED: neither choice changes any converged state in the test suite
///
/// This is stated because the plausible argument for each ("an even split
/// throws the open-shell character away"; "diagonalizing `F_α` is a different
/// operator") is NOT what the data shows, and an unverified justification in a
/// docstring is worse than none. Both were mutation-tested in the foreground
/// against the full 28-row sweep in `tests/rohf_state_selection.rs`:
///
/// * Replacing the occupation split with UHF's `D/2` — **all 8 tests pass**.
///   Every converged energy is identical to 1–2 ulp (e.g. HeNe⁺/6-31G
///   −130.60332290752973 vs …967).
/// * Replacing the Roothaan effective Fock with a bare `F_α` — **all 8 tests
///   pass**, same states throughout.
///
/// So on every system measured, what selects the basin is the guess DENSITY;
/// the details of how that density is turned into orbitals do not matter. Both
/// choices are kept because they are the internally consistent ones — they
/// match this file's own occupation convention and its own Fock combination,
/// so a reader is not left wondering why the guess uses a different operator
/// from the loop — but NO accuracy claim rests on either, and a future change
/// to either is a refactor, not a correctness fix. If a system is ever found
/// where they DO differ, that system belongs in the sweep.
///
/// # Why the guess Fock is built with plain Coulomb J/K even under RSH/DFT
///
/// This is a GUESS. `build_jk` uses `bounds.op`, the operator the whole SCF was
/// set up with, and adds no XC. Under a range-separated or DFT reference the
/// guess Fock is therefore not the converged Fock — which is fine and is what
/// every SAD-style guess in every code does — but it means this function must
/// never be mistaken for a converged-Fock builder.
#[allow(clippy::too_many_arguments)]
fn rohf_guess_mos(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    config: &RohfConfig,
    h: &Array2<f64>,
    s: &Array2<f64>,
    s_inv_sqrt: &Array2<f64>,
    nocc_double: usize,
    nocc_open: usize,
) -> Result<Option<Array2<f64>>, FerricError> {
    let n = prep.nbasis();

    // Which density? An explicit one wins; otherwise the MINAO projection,
    // which is exactly what `rhf.rs` and `uhf.rs` resolve for the same two
    // config fields.
    let d_total = if let Some(d0) = config.init_guess_density.as_ref() {
        if d0.dim() != (n, n) {
            return Err(FerricError::General(format!(
                "ROHF: init_guess_density shape {:?} != ({n},{n})",
                d0.dim()
            )));
        }
        d0.clone()
    } else if config.use_sad_guess {
        match crate::guess::minao_projection_guess(mol, prep, prep.basis_set()) {
            Ok(d) => d,
            Err(e) => {
                if crate::rhf::scf_trace() {
                    eprintln!("ROHF guess: MINAO projection failed ({e:?}); falling back to hcore");
                }
                return Ok(None);
            }
        }
    } else {
        return Ok(None);
    };

    // Split the spin-summed guess density BY OCCUPATION (see the doc above).
    let nocc_a = nocc_double + nocc_open;
    let nocc_b = nocc_double;
    let nelec = nocc_a + nocc_b;
    if nelec == 0 {
        return Ok(None);
    }
    let d_a = &d_total * (nocc_a as f64 / nelec as f64);
    let d_b = &d_total * (nocc_b as f64 / nelec as f64);

    // F_σ = h + J[D_α + D_β] − K[D_σ], matching this file's own assembly.
    let mut j = Array2::<f64>::zeros((n, n));
    let mut k_scratch = Array2::<f64>::zeros((n, n));
    if let Err(e) = crate::rhf::build_jk(
        ctx,
        prep,
        bounds,
        config.integral_thresh,
        &d_total,
        &mut j,
        &mut k_scratch,
    ) {
        if crate::rhf::scf_trace() {
            eprintln!("ROHF guess: J build at the guess density failed ({e:?}); using hcore");
        }
        return Ok(None);
    }
    let mut k_a = Array2::<f64>::zeros((n, n));
    if let Err(e) = crate::rhf::build_jk(
        ctx,
        prep,
        bounds,
        config.integral_thresh,
        &d_a,
        &mut j.clone(),
        &mut k_a,
    ) {
        if crate::rhf::scf_trace() {
            eprintln!("ROHF guess: K_alpha build failed ({e:?}); using hcore");
        }
        return Ok(None);
    }
    let mut k_b = Array2::<f64>::zeros((n, n));
    if let Err(e) = crate::rhf::build_jk(
        ctx,
        prep,
        bounds,
        config.integral_thresh,
        &d_b,
        &mut j.clone(),
        &mut k_b,
    ) {
        if crate::rhf::scf_trace() {
            eprintln!("ROHF guess: K_beta build failed ({e:?}); using hcore");
        }
        return Ok(None);
    }
    let f_a = h + &j - &k_a;
    let f_b = h + &j - &k_b;

    // ONE MO set, from the Roothaan effective Fock — the same combination the
    // SCF loop below uses.
    let f_eff = roothaan_fock(&f_a, &f_b, &d_a, &d_b, s);
    match diagonalize(&f_eff, s_inv_sqrt) {
        Ok((_, c)) => Ok(Some(c)),
        Err(e) => {
            if crate::rhf::scf_trace() {
                eprintln!("ROHF guess: diagonalizing the guess Fock failed ({e:?}); using hcore");
            }
            Ok(None)
        }
    }
}

fn roothaan_fock(
    f_a: &Array2<f64>,
    f_b: &Array2<f64>,
    d_a: &Array2<f64>,
    d_b: &Array2<f64>,
    s: &Array2<f64>,
) -> Array2<f64> {
    let n = s.shape()[0];
    let f_c = 0.5 * (f_a + f_b);
    // Projectors: P_c = D_β S, P_o = (D_α − D_β) S, P_v = I − D_α S
    let p_c = d_b.dot(s);
    let do_diff: Array2<f64> = d_a - d_b;
    let p_o = do_diff.dot(s);
    let mut p_v = Array2::<f64>::eye(n);
    p_v = &p_v - &d_a.dot(s);

    // Upper-triangle pieces (PySCF builds half then symmetrises by F + F^T).
    let p_c_t = p_c.t();
    let p_o_t = p_o.t();
    let p_v_t = p_v.t();

    let mut f = 0.5 * p_c_t.dot(&f_c).dot(&p_c);
    f = &f + &(0.5 * p_o_t.dot(&f_c).dot(&p_o));
    f = &f + &(0.5 * p_v_t.dot(&f_c).dot(&p_v));
    f = &f + &p_o_t.dot(f_b).dot(&p_c);
    f = &f + &p_o_t.dot(f_a).dot(&p_v);
    f = &f + &p_v_t.dot(&f_c).dot(&p_c);
    let f_sym = &f + &f.t();
    f_sym
}

fn diagonalize(
    f: &Array2<f64>,
    s_inv_sqrt: &Array2<f64>,
) -> Result<(Vec<f64>, Array2<f64>), FerricError> {
    let f_prime = s_inv_sqrt.dot(f).dot(s_inv_sqrt);
    let (evals, evecs) = f_prime
        .eigh(ndarray_linalg::UPLO::Upper)
        .map_err(|e| FerricError::Lapack(format!("F diag: {e}")))?;
    let c = s_inv_sqrt.dot(&evecs);
    Ok((evals.to_vec(), c))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::basis;

    #[test]
    fn test_rohf_h_atom_sto3g() {
        // Single H atom — trivial 1-electron case.
        let mol = Molecule::parse_xyz("1\nH\nH 0 0 0\n", 0, 2).unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let cfg = RohfConfig {
            energy_conv: 1e-10,
            density_conv: 1e-9,
            ..Default::default()
        };
        let ctx = ParallelContext::default();
        let res = solve_rohf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
        assert!(res.converged);
        assert!(
            (res.energy + 0.466581850).abs() < 1e-5,
            "H atom energy = {}",
            res.energy
        );
    }
}
