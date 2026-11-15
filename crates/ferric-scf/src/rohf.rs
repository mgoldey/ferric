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
use crate::fock::{JBuilder, KBuilder};
use crate::guess::hcore_guess;
use crate::result::{ScfResult, Spin};
use crate::rhf::{validate_injected, InjectedConfigError, PeriodicInjection, RhfConfig};
use crate::screening::SchwarzBounds;
use ferric_integrals::blas_threads::{opt_in_blas_threads, with_blas_threads};

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
pub(crate) fn rohf_trace() -> bool {
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

/// Test-only instrumentation: counts τ-only meta-GGA f_xc kernel builds
/// ([`FxcKernelStore::Mgga`]), so the meta-GGA Newton / stability tests can
/// prove the τ kernel was the one applied.
#[doc(hidden)]
pub static MGGA_FXC_KERNEL_BUILDS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Test-only override of the F6 swap witness's restart budget
/// (`crate::rohf_occupation::MAX_AUFBAU_RESTARTS`); `usize::MAX` (the
/// default) means no override. Process-global, so a test that sets it must
/// live in its own test binary — see `tests/rohf_witness_give_up.rs`.
#[doc(hidden)]
pub static ROHF_MAX_AUFBAU_RESTARTS_OVERRIDE: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(usize::MAX);

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

thread_local! {
    /// Dense [`FxcKernelStore`] builds on this thread (test seam).
    static DENSE_FXC_BUILDS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// Screened f_xc responses ([`ferric_dft::xc_batch::ScreenedFxc`]) built
    /// for a Newton step on this thread (test seam).
    static SCREENED_FXC_BUILDS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Dense [`FxcKernelStore`] builds (each a fresh grid and a full χ/∇χ cache)
/// made on the calling thread so far. A test seam: Newton steps run on their
/// caller's thread, so the count is race-free under the parallel harness.
#[doc(hidden)]
pub fn dense_fxc_builds_on_this_thread() -> usize {
    DENSE_FXC_BUILDS.with(std::cell::Cell::get)
}

/// Screened f_xc responses built for Newton steps on the calling thread so
/// far (test seam; see [`dense_fxc_builds_on_this_thread`]).
#[doc(hidden)]
pub fn screened_fxc_builds_on_this_thread() -> usize {
    SCREENED_FXC_BUILDS.with(std::cell::Cell::get)
}

/// The functional `xc_name` resolved with `nspin = 2` (libxc's polarized
/// second derivatives) and the SCF's ω override, as every f_xc kernel needs
/// it (see [`FxcKernelStore::build`] for why the override matters).
pub(crate) fn fxc_xc_def(
    xc_name: &str,
    omega_override: Option<f64>,
) -> Result<ferric_dft::libxc::XcDef, FerricError> {
    match omega_override {
        Some(w) => ferric_dft::libxc::xc_def_from_name_nspin_omega(xc_name, 2, w),
        None => ferric_dft::libxc::xc_def_from_name_nspin(xc_name, 2),
    }
    .map_err(|e| FerricError::General(format!("fxc def for {xc_name}: {e:?}")))
}

/// Record a screened f_xc build for a Newton step: the thread-local seam, and
/// the family counters the dense builds also bump ([`GGA_FXC_KERNEL_BUILDS`],
/// [`MGGA_FXC_KERNEL_BUILDS`]), so "the GGA / τ kernel engaged" stays
/// observable whichever path built it.
pub(crate) fn note_screened_fxc_build(xc_def: &ferric_dft::libxc::XcDef) {
    use ferric_dft::libxc::FunctionalFamily;
    SCREENED_FXC_BUILDS.with(|c| c.set(c.get() + 1));
    let fam = |f: FunctionalFamily| xc_def.funcs.iter().any(|x| x.family() == f);
    if fam(FunctionalFamily::MetaGga) {
        MGGA_FXC_KERNEL_BUILDS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    } else if xc_def
        .funcs
        .iter()
        .any(|x| x.family() != FunctionalFamily::Lda)
    {
        GGA_FXC_KERNEL_BUILDS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

/// The bytes a no-pool f_xc kernel cache may take: the resolved budget
/// (`three_index_budget_bytes` when set, else `FERRIC_MEM_BUDGET_GB` / auto)
/// less this process's live RSS. Read only when no global pool is installed;
/// with a pool, the screened cache is charged to the pool instead.
pub(crate) fn fxc_no_pool_budget(config: &RhfConfig) -> usize {
    ferric_core::memory::available_budget_now(ferric_core::memory::resolve_budget_bytes(
        (config.three_index_budget_bytes != 0).then_some(config.three_index_budget_bytes),
    ))
}

/// The restricted f_xc response for an RKS Newton step at the closed-shell
/// density `d` (d_α = d_β = ½·d), as the `FxcResponse` closure the Hessian
/// matvec calls with `(δD_single, δD_single)`.
///
/// When the SCF's XC evaluator `x` has a screened grid
/// ([`ferric_dft::xc_trait::XcContribution::restricted_fxc`]), the response
/// runs there: the SCF's own batches, screening and AO cache, libxc's second
/// derivatives evaluated once here, one spin per product. Otherwise it falls
/// back to the dense [`FxcKernelStore`] on the configured grid.
///
/// The screened closure requires equal spin perturbations (the restricted
/// singlet response is one spin block, added once to the restricted Fock) and
/// returns that block for both spins. Public for the anchor tests
/// (`tests/newton_fxc_screened.rs`), which must reach this selection.
#[doc(hidden)]
pub fn restricted_fxc_response<'s>(
    x: &'s dyn ferric_dft::xc_trait::XcContribution,
    (mol, prep, config, name): (&Molecule, &PreparedBasis, &RhfConfig, &str),
    d: &Array2<f64>,
) -> Result<Box<crate::rohf_newton::FxcResponse<'s>>, FerricError> {
    let xc_pol = fxc_xc_def(name, config.xc_omega)?;
    if let Some(built) = x.restricted_fxc(&xc_pol, d, &|| fxc_no_pool_budget(config)) {
        let sf = built.map_err(|e| FerricError::General(format!("screened f_xc: {e}")))?;
        note_screened_fxc_build(&xc_pol);
        return Ok(Box::new(move |a: &Array2<f64>, b: &Array2<f64>| {
            assert!(
                std::ptr::eq(a, b) || a == b,
                "the restricted f_xc response needs equal α and β perturbations"
            );
            let v = sf.apply_restricted(a);
            (v.clone(), v)
        }));
    }
    let grid = config
        .dft_grid
        .clone()
        .unwrap_or_else(ferric_dft::grid::AtomicGridConfig::production);
    let d_half = 0.5 * d;
    let store = FxcKernelStore::build(mol, prep, &grid, name, config.xc_omega, &d_half, &d_half)?;
    Ok(Box::new(move |a: &Array2<f64>, b: &Array2<f64>| {
        store.response()(a, b)
    }))
}

/// [`restricted_fxc_response`] for a spin-polarized reference `(d_a, d_b)`
/// (UKS / ROKS Newton, AH and TRAH steps): the SCF's own screened grid when
/// its XC evaluator has one
/// ([`ferric_dft::xc_trait::UksXcContribution::polarized_fxc`]), else the
/// dense [`FxcKernelStore`]. `dense_gate` = `(label, budget)`: on the dense
/// fallback only, `budget()` gives `(needed, available)` bytes, checked with
/// `check_alloc` before that path builds its duplicate χ/∇χ cache (the
/// screened path builds none and never evaluates `budget`; its own kernel
/// cache is charged to the pool, or checked against [`fxc_no_pool_budget`]
/// when there is none).
#[doc(hidden)]
pub fn polarized_fxc_response<'s>(
    x: &'s dyn ferric_dft::xc_trait::UksXcContribution,
    (mol, prep, config, name): (&Molecule, &PreparedBasis, &RhfConfig, &str),
    (d_a, d_b): (&Array2<f64>, &Array2<f64>),
    dense_gate: Option<(&str, &dyn Fn() -> (usize, usize))>,
) -> Result<Box<crate::rohf_newton::FxcResponse<'s>>, FerricError> {
    let xc_pol = fxc_xc_def(name, config.xc_omega)?;
    if let Some(built) = x.polarized_fxc(&xc_pol, d_a, d_b, &|| fxc_no_pool_budget(config)) {
        let sf = built.map_err(|e| FerricError::General(format!("screened f_xc: {e}")))?;
        note_screened_fxc_build(&xc_pol);
        return Ok(Box::new(move |a: &Array2<f64>, b: &Array2<f64>| {
            sf.apply_polarized(a, b)
        }));
    }
    if let Some((label, budget)) = dense_gate {
        let (needed, avail) = budget();
        ferric_core::memory::check_alloc(label, needed, avail)?;
    }
    let grid = config
        .dft_grid
        .clone()
        .unwrap_or_else(ferric_dft::grid::AtomicGridConfig::production);
    let store = FxcKernelStore::build(mol, prep, &grid, name, config.xc_omega, d_a, d_b)?;
    Ok(Box::new(move |a: &Array2<f64>, b: &Array2<f64>| {
        store.response()(a, b)
    }))
}

/// The spin-polarized exchange-correlation kernel f_xc at a reference
/// density, as the Newton steps use it (LDA, GGA, hybrid and range-separated
/// GGA, and τ-only meta-GGA; VV10 has no kernel here and must be refused by
/// the caller).
pub enum FxcKernelStore {
    /// Every component of the functional is LDA.
    Lda {
        /// The LDA kernel on the integration grid.
        kernel: Box<ferric_dft::fxc::LdaFxcKernel>,
        /// Reference α density on the grid points.
        rho_a0: Vec<f64>,
        /// Reference β density on the grid points.
        rho_b0: Vec<f64>,
    },
    /// GGA, global hybrid or range-separated GGA (any non-LDA, non-meta-GGA
    /// functional).
    Gga {
        /// The GGA kernel on the integration grid.
        kernel: Box<ferric_dft::fxc::GgaFxcKernel>,
        /// Reference per-spin densities and gradients on the grid.
        ref_dens: Box<ferric_dft::density_on_grid::UksDensityGrid>,
    },
    /// τ-only meta-GGA (SCAN / r2SCAN / TPSS): the RKS/UKS Newton steps,
    /// RKS/UKS internal stability, the RKS/UKS/ROKS Z-vector and the Python
    /// response blocks apply it. ROKS Newton/AH and TRAH keep meta-GGA on DIIS.
    Mgga {
        /// The meta-GGA kernel on the integration grid.
        kernel: Box<ferric_dft::fxc::MggaFxcKernel>,
        /// Reference-density quantities the kernel is evaluated at.
        reference: Box<ferric_dft::fxc::MggaFxcReference>,
    },
}

impl FxcKernelStore {
    /// Build the appropriate f_xc kernel for `xc_name` (LDA, GGA-family or
    /// τ-only meta-GGA) at the given reference densities. `xc_name` is the
    /// functional string that already produced a live `xc_contrib` upstream.
    ///
    /// `omega_override` is `RhfConfig::xc_omega`, and threading it is NOT
    /// cosmetic. The SCF's own `xc_contrib` is built with the override (see
    /// `KsXcUks::new`'s `config.xc_omega` argument), so without it here the
    /// response kernel would be the one for the functional's PUBLISHED ω while
    /// the converged density came from the overridden one — a wrong-operator
    /// Hessian of exactly the kind the stability module exists to prevent, and
    /// the defect `zvector_ks::ks_hessian_unsupported_reason` still refuses an
    /// `xc_omega` run over. Measured on N2⁺/def2-SVP/ωB97X-V: omitting the
    /// override put λ_min's zero crossing at ω ≈ 0.468 instead of PySCF's
    /// ≈ 0.531, i.e. a sign DISAGREEMENT at ω = 0.53 while the energy agreed
    /// to 1e-9. `None` means "use the functional's published ω", which is
    /// byte-identical to the former behaviour for every caller that has no
    /// override set.
    pub fn build(
        mol: &Molecule,
        prep: &PreparedBasis,
        cfg: &ferric_dft::grid::AtomicGridConfig,
        xc_name: &str,
        omega_override: Option<f64>,
        d_a: &Array2<f64>,
        d_b: &Array2<f64>,
    ) -> Result<Self, FerricError> {
        DENSE_FXC_BUILDS.with(|c| c.set(c.get() + 1));
        let xc_def = fxc_xc_def(xc_name, omega_override)?;
        let is_lda = xc_def
            .funcs
            .iter()
            .all(|f| f.family() == ferric_dft::libxc::FunctionalFamily::Lda);
        let is_mgga = xc_def
            .funcs
            .iter()
            .any(|f| f.family() == ferric_dft::libxc::FunctionalFamily::MetaGga);
        if is_mgga {
            return Self::build_mgga(mol, prep, cfg, xc_def, (d_a, d_b));
        }
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

    /// The τ-only meta-GGA kernel at `(d_a, d_b)`.
    fn build_mgga(
        mol: &Molecule,
        prep: &PreparedBasis,
        cfg: &ferric_dft::grid::AtomicGridConfig,
        xc_def: ferric_dft::libxc::XcDef,
        (d_a, d_b): (&Array2<f64>, &Array2<f64>),
    ) -> Result<Self, FerricError> {
        let err = |e: String| FerricError::General(format!("MggaFxcKernel: {e}"));
        let kernel =
            ferric_dft::fxc::MggaFxcKernel::new(mol, prep.basis_set(), xc_def, cfg).map_err(err)?;
        let reference = kernel.reference(d_a, d_b).map_err(err)?;
        MGGA_FXC_KERNEL_BUILDS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(FxcKernelStore::Mgga {
            kernel: Box::new(kernel),
            reference: Box::new(reference),
        })
    }

    /// The `Fn(δD_α, δD_β) -> (δV_α, δV_β)` response closure, borrowing `self`.
    #[allow(clippy::type_complexity)]
    pub fn response(
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
            FxcKernelStore::Mgga { kernel, reference } => {
                Box::new(move |a: &Array2<f64>, b: &Array2<f64>| {
                    kernel.apply_with_ref(reference, a, b)
                })
            }
        }
    }
}

/// Trust radius (componentwise |κ| clip) of the ROHF/ROKS PCG Newton step,
/// also used for AH's Newton fallback. Conservative: ROKS Hessians are stiff.
const ROHF_NEWTON_MAX_STEP: f64 = 0.1;

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
    crate::symmetric::require_symmetry_off(config, "solve_rohf")?;
    solve_rohf_routed(ctx, mol, prep, bounds, config, None)
}

/// [`solve_rohf_best_effort`] started from caller-supplied MOs `c0`
/// (`(n, n)`, columns doubly occupied | singly occupied | virtual) instead of
/// the default guess; the secondary-basis guess does not apply. The geometry
/// drivers' warm start (`crate::warm_start`) is the caller.
pub(crate) fn solve_rohf_best_effort_with_guess(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    config: &RohfConfig,
    c0: &Array2<f64>,
) -> Result<ScfResult, FerricError> {
    crate::symmetric::require_symmetry_off(config, "solve_rohf")?;
    solve_rohf_impl_recorded(ctx, mol, prep, bounds, config, Some(c0), None, None, None)
}

/// The molecular ROHF solve, started from the secondary-basis guess MOs when
/// [`RhfConfig::guess_basis`] applies. A Fock `hook` acts on the target solve
/// only, never on the secondary-basis SCF or the Fock projection (see
/// `rhf::solve_rhf_routed`).
fn solve_rohf_routed(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    config: &RohfConfig,
    hook: Option<crate::fock::SpinFockHook<'_>>,
) -> Result<ScfResult, FerricError> {
    if !crate::basis2::applies(config, false) {
        return solve_rohf_impl_recorded(ctx, mol, prep, bounds, config, None, None, None, hook);
    }
    let kind = crate::basis2::ScfKind::Rohf;
    let g = crate::basis2::open_shell_guess(ctx, mol, prep, bounds, config, kind)?;
    let c0 = g.initial_mos().map(|(c, _)| c);
    let mut r = solve_rohf_impl_recorded(ctx, mol, prep, bounds, &g.config, c0, None, None, hook)?;
    r.guess_basis = Some(g.report);
    Ok(r)
}

/// Reject every `RohfConfig` feature [`solve_rohf_injected`] cannot honour,
/// naming the first offending field: everything [`validate_injected`] refuses
/// for RHF (DF/pluggable K, `xc`, Newton/TRAH/AURORA, the MINAO guess without
/// an explicit density, external potential, COSMO/PCM/polarizable,
/// `check_stability`), then the ROHF-specific ones:
///
/// * `scf_stability_descent` — ROHF has no stability descent at all
///   ([`crate::stability::StabilitySkip::Rohf`]); the molecular solver ignores
///   the flag, the injected path refuses it rather than ignore it silently.
/// * `ah_trigger` — the augmented-Hessian step (`rohf_ah` over
///   `rohf_newton::RohfNewtonInputs { prep, bounds, .. }`) builds its orbital
///   Hessian response from MOLECULAR J/K integrals, which would be wrong for
///   the injected (periodic) operator. (`newton_trigger`, the other
///   `rohf_newton` user, is already refused by [`validate_injected`].) The
///   injected path's second-order solver is
///   [`crate::rohf_trah::solve_rohf_injected_second_order`] instead.
///
/// Checked after the RHF list, so a config that sets both reports the RHF
/// field.
pub fn validate_injected_rohf(config: &RohfConfig) -> Result<(), InjectedConfigError> {
    validate_injected(config)?;
    if config.scf_stability_descent {
        return Err(InjectedConfigError {
            field: "scf_stability_descent",
            reason: "ROHF has no stability descent (the Roothaan open-shell Hessian is not \
                     implemented); the injected path refuses the flag instead of ignoring it",
        });
    }
    if config.ah_trigger > 0.0 {
        return Err(InjectedConfigError {
            field: "ah_trigger",
            reason: "the ROHF augmented-Hessian step rebuilds molecular J/K response \
                     integrals from prep/bounds",
        });
    }
    Ok(())
}

/// Restricted open-shell HF / KS on caller-supplied `(S, h, V_nn)` and J/K
/// builders — the Gamma-point periodic ROHF/ROKS entry (FINDINGS "Iteration
/// 10": ROKS = the UKS energy functional on one spatial MO set, Roothaan
/// effective Fock). Mirrors [`crate::uhf::solve_uhf_injected`]:
///
/// * `mol` supplies only the electron count, charge and multiplicity; `prep`
///   only `nbasis`; `op`/`bounds` are never read for integrals on this path.
/// * J = `inj.j` on `D_α + D_β`; K_σ = the SAME `inj.k`, called per spin as
///   `update_density(D_σ)` then `build(D_σ)`. F_σ = h + J − a·K_σ (+ V_σ),
///   then the unchanged Guest-Saunders Roothaan combination. Any Madelung
///   term lives in the K builder, linear in D_σ: NO per-spin ½, and with an
///   injected `xc` it rides on the `a·K` part only (a = the builder's
///   exact-exchange fraction; a = 0 never builds K).
/// * `xc = Some(x)` must report [`crate::rhf::XcBuilder::supports_polarized`];
///   `(E_xc, V_α, V_β) = x.build_polarized(D_α, D_β)`, E_xc outside the trace
///   (the molecular `add_xc_uks` convention).
/// * Guess: `initial_mos` (one `(nbasis, nbasis)` MO set; the first
///   `nocc_double + nocc_open` columns are occupied) > `config.init_guess_density`
///   (spin-summed; split by occupation, guess Fock `h + J[D] − K[D_σ]` from the
///   INJECTED builders, Roothaan-combined, no XC) > the core guess
///   (diagonalize the injected `h`, the molecular `use_sad_guess = false`
///   path). The molecular MINAO / `rhf::build_jk` guess is never called.
///
/// Errors (typed, by name) on any field [`validate_injected_rohf`] refuses, on
/// an `xc` without spin-polarized support or with `a` outside [0, 1], on
/// wrong-shape `s`/`h`/`initial_mos` or non-finite `vnn`, and — like
/// [`solve_rohf`] — when the SCF does not converge. Ewald-trap caveat as for
/// UHF: `ferric_pbc::rohf` checks the per-spin gaps against `a·v_M`.
#[allow(clippy::too_many_arguments)]
pub fn solve_rohf_injected<'a>(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    _op: Operator,
    bounds: &SchwarzBounds,
    config: &RohfConfig,
    inj: PeriodicInjection<'a>,
    initial_mos: Option<&Array2<f64>>,
) -> Result<ScfResult, FerricError> {
    validate_injected_rohf(config).map_err(|e| {
        FerricError::General(format!(
            "solve_rohf_injected: RhfConfig.{} is not supported with injected S/h/J/K: {}",
            e.field, e.reason
        ))
    })?;
    crate::rhf::check_injection_shapes("solve_rohf_injected", &inj, prep.nbasis())?;
    if let Some(x) = inj.xc.as_ref() {
        if !x.supports_polarized() {
            return Err(FerricError::General(
                "solve_rohf_injected: PeriodicInjection.xc does not support spin-polarized \
                 evaluation (XcBuilder::supports_polarized() is false): the injected ROKS path \
                 needs XcBuilder::build_polarized (e.g. ferric_pbc::dft::PeriodicXc)"
                    .into(),
            ));
        }
    }
    let r = solve_rohf_impl_recorded(
        ctx,
        mol,
        prep,
        bounds,
        config,
        initial_mos,
        Some(inj),
        None,
        None,
    )?;
    if r.converged {
        Ok(r)
    } else {
        Err(FerricError::ScfConvergence {
            iterations: r.iterations,
            last_energy: r.energy,
        })
    }
}

/// ROHF/ROKS with one extra density-dependent term,
/// `(E_hook, V_α, V_β) = hook(D_α, D_β)`: V_σ is added to F_σ before the
/// Roothaan effective-Fock combination and E_hook to the energy outside the
/// trace, so the converged energy is stationary under ROHF orbital rotations
/// when V_σ = ∂E_hook/∂D_σ. Settings the hook would bypass are refused by
/// [`crate::rhf::validate_fock_hook`]; a non-converged SCF is an error, as
/// for [`solve_rohf`].
pub fn solve_rohf_with_hook(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    config: &RohfConfig,
    hook: crate::fock::SpinFockHook<'_>,
) -> Result<ScfResult, FerricError> {
    crate::rhf::validate_fock_hook(config)?;
    let r = solve_rohf_routed(ctx, mol, prep, bounds, config, Some(hook))?;
    if r.converged {
        Ok(r)
    } else {
        Err(FerricError::ScfConvergence {
            iterations: r.iterations,
            last_energy: r.energy,
        })
    }
}

/// [`solve_rohf`] (`hook = None`) or [`solve_rohf_with_hook`] started from
/// per-spin AO densities `(D_α, D_β)` instead of the configured guess
/// (`rohf_spin_density_mos`): the first iteration runs at `(D_α, D_β)`
/// itself when it is a ROHF pair (`D_β` and `D_α − D_β` idempotent and
/// S-orthogonal, as at convergence), and at its projection onto one
/// closed / open / virtual MO set otherwise. `guess_basis` does not apply.
pub fn solve_rohf_from_spin_densities(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    config: &RohfConfig,
    densities: (&Array2<f64>, &Array2<f64>),
    hook: Option<crate::fock::SpinFockHook<'_>>,
) -> Result<ScfResult, FerricError> {
    crate::symmetric::require_symmetry_off(config, "solve_rohf")?;
    if hook.is_some() {
        crate::rhf::validate_fock_hook(config)?;
    }
    let (nocc_alpha, nocc_double) = crate::uhf::nocc_ab(mol)?;
    let s = ferric_integrals::oneelectron::overlap(prep);
    let c0 = rohf_spin_density_mos(&s, densities, nocc_alpha, nocc_double)?;
    let r = solve_rohf_impl_recorded(ctx, mol, prep, bounds, config, Some(&c0), None, None, hook)?;
    if r.converged {
        Ok(r)
    } else {
        Err(FerricError::ScfConvergence {
            iterations: r.iterations,
            last_energy: r.energy,
        })
    }
}

/// One ROHF MO set from per-spin densities, columns closed | open | virtual:
/// the closed shell is the `nocc_double` leading natural orbitals of `D_β`
/// (`crate::uhf::spin_natural_orbitals`), and the open and virtual orbitals
/// are the natural orbitals of `D_α` within the S-orthogonal complement of
/// the closed shell, largest occupation first. For `D_β = C_c C_cᵀ` and `D_α = C_c C_cᵀ + C_o C_oᵀ` (S-orthonormal
/// `[C_c C_o]`) the closed and open columns rebuild both densities to
/// round-off. The retained overlap rank must hold all `nocc_alpha`
/// (closed + open) occupied orbitals.
pub(crate) fn rohf_spin_density_mos(
    s: &Array2<f64>,
    densities: (&Array2<f64>, &Array2<f64>),
    nocc_alpha: usize,
    nocc_double: usize,
) -> Result<Array2<f64>, FerricError> {
    let n = s.nrows();
    for d in [densities.0, densities.1] {
        if d.dim() != (n, n) {
            return Err(FerricError::General(format!(
                "ROHF: spin density shape {:?} != ({n},{n})",
                d.dim()
            )));
        }
    }
    let x = crate::rhf::canonical_orthogonalizer(s)?;
    let m = x.ncols();
    if nocc_double > m || nocc_alpha > m {
        return Err(FerricError::General(format!(
            "ROHF: {nocc_alpha} alpha-occupied orbitals ({nocc_double} doubly occupied) \
             exceed the {m} orthogonalized functions the overlap retains"
        )));
    }
    let xs = x.t().dot(s);
    let in_orthonormal = |d: &Array2<f64>| xs.dot(d).dot(&xs.t());
    let (_, v) = crate::uhf::descending_eigh(&in_orthonormal(densities.1))?;
    let rest = v.slice(ndarray::s![.., nocc_double..]).to_owned();
    let (_, w) =
        crate::uhf::descending_eigh(&rest.t().dot(&in_orthonormal(densities.0)).dot(&rest))?;
    let mut u = Array2::<f64>::zeros((m, m));
    u.slice_mut(ndarray::s![.., ..nocc_double])
        .assign(&v.slice(ndarray::s![.., ..nocc_double]));
    u.slice_mut(ndarray::s![.., nocc_double..])
        .assign(&rest.dot(&w));
    Ok(crate::uhf::padded(&x.dot(&u), n))
}

/// The blocked solve of [`crate::symmetric_open::solve_rohf_symmetric`]: `mol`,
/// `prep` and `bounds` are already in the symmetry frame.
pub(crate) fn solve_rohf_blocked(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    config: &RohfConfig,
    spec: &crate::symmetric::SymmetrySpec,
) -> Result<ScfResult, FerricError> {
    let r = solve_rohf_impl_recorded(ctx, mol, prep, bounds, config, None, None, Some(spec), None)?;
    if r.converged {
        Ok(r)
    } else {
        Err(FerricError::ScfConvergence {
            iterations: r.iterations,
            last_energy: r.energy,
        })
    }
}

/// [`solve_rohf_impl`] plus its precision provenance record
/// ([`ScfResult::precision`]): the per-kernel counter delta over this solve.
/// The record observes only; it changes no number.
#[allow(clippy::too_many_arguments)]
fn solve_rohf_impl_recorded(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    config: &RohfConfig,
    initial_mos: Option<&Array2<f64>>,
    inj: Option<PeriodicInjection<'_>>,
    sym: Option<&crate::symmetric::SymmetrySpec>,
    hook: Option<crate::fock::SpinFockHook<'_>>,
) -> Result<ScfResult, FerricError> {
    let probe = ferric_core::gpu::provenance::PrecisionProbe::start();
    let xc = crate::result::xc_record(
        config.xc.as_deref(),
        inj.as_ref().is_some_and(|i| i.xc.is_some()),
    );
    let mut r = solve_rohf_impl(ctx, mol, prep, bounds, config, initial_mos, inj, sym, hook)?;
    r.precision = probe.finish(ferric_core::gpu::provenance::SCF_KERNELS);
    r.xc = xc;
    r.relativistic = config.relativistic;
    Ok(r)
}

/// Shared body of [`solve_rohf_best_effort`] (`inj = None`, `initial_mos =
/// None`: the molecular solver, byte-identical to the pre-injection code —
/// every injected-path branch is gated on `inj_jk`/`inj_xc` being `Some`) and
/// [`solve_rohf_injected`] (config already validated).
///
/// The solve runs as named stages that own their state, the counterparts of
/// `uhf::solve_uhf_impl`'s:
///
/// - setup: the XC contribution and exchange mix, the shared environment
///   ([`crate::driver::prepare`]), the occupation ([`rohf_nocc`]), the
///   orthogonalizer, the guess ([`rohf_initial_mos`]), the J/K route and
///   fitters, the pluggable K and the builders
///   ([`crate::driver::OpenShellJk`]), the F6 occupation guard;
/// - one iteration: the Fock build ([`crate::driver::OpenShellJk::build_fock`]),
///   the energy ([`RohfState::energy`]), the Roothaan effective Fock and its
///   orbital gradient ([`RohfState::measure`]), then the orbital update
///   ([`RohfState::update`]): augmented-Hessian or Newton, else DIIS;
/// - the convergence and ladder decision ([`RohfState::gate`]) and the F6
///   gradient guard and swap witness ([`RohfState::converged`]);
/// - finalization: [`RohfState::max_iter`].
///
/// Opt-in per-phase timing (`FERRIC_TIMING=1`; inert otherwise), with the
/// RHF table's phase names (see `rhf::solve_rhf_impl` for the placement
/// rules). On the open-shell path `jk_build` also holds the per-spin exchange
/// fold `F_σ −= c·K_σ` (the K builds sit inside that assembly). Untimed: the
/// Newton/AH steps, the occupation witness and the converged-exit extras.
#[allow(clippy::too_many_arguments)]
fn solve_rohf_impl(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    config: &RohfConfig,
    initial_mos: Option<&Array2<f64>>,
    inj: Option<PeriodicInjection<'_>>,
    sym: Option<&crate::symmetric::SymmetrySpec>,
    mut hook: Option<crate::fock::SpinFockHook<'_>>,
) -> Result<ScfResult, FerricError> {
    crate::mom::check_reference(config, "ROHF", false)?;
    crate::incore_jk::begin_scf(bounds);

    // Split the injection: (S, h, V_nn) go to the shared env builder, the J/K
    // (and XC) builders are driven in the loop.
    let (pre_env, mut inj_jk, inj_xc) = crate::driver::split_injection(inj);
    let injected = inj_jk.is_some();

    // Build UKS XC contribution once. None for pure ROHF.
    let mut xc_contrib = crate::driver::uks_xc_contribution(mol, prep, config)?;
    // Injected (periodic) ROKS: validate_injected_rohf guarantees
    // `xc_contrib` is None there.
    let k_mix = crate::driver::resolve_k_mix(
        inj_xc.as_deref(),
        || xc_contrib.as_ref().map(|x| x.k_mix()).unwrap_or_default(),
        "solve_rohf_injected",
    )?;
    let c_k: f64 = if xc_contrib.is_some() || inj_xc.is_some() {
        k_mix.sr
    } else {
        1.0
    };

    let mut phases = ferric_core::timing::PhaseAccumulator::new().report_on_drop(
        crate::rhf::phase_table_title_for(Spin::RestrictedOpen, config, inj_xc.is_some()),
    );
    // Shared geometry-only environment: S, hcore(+external+ECP — V_ECP folds
    // into hcore once, byte-identical to plain hcore for all-electron bases),
    // V_nn(+external), COSMO/PCM contexts, resolved memory budget, RSH
    // fitters. One construction serving all six SCF variants (crate::driver).
    let env = crate::driver::prepare_timed(
        &mut phases,
        ctx,
        mol,
        prep,
        config,
        &k_mix,
        pre_env,
        crate::driver::xc_plan(xc_contrib.as_deref()),
    )?;
    let rsh_response = crate::driver::rsh_response(&env, k_mix);
    let n = prep.nbasis();
    let (nocc_double, nocc_open) = rohf_nocc(mol)?;
    let nocc_a = nocc_double + nocc_open;
    // S^{-1/2}, or the retained-rank X when the overlap is near-singular.
    let orth_t = phases.begin("setup:orthogonalizer");
    let s_inv_sqrt = rohf_orthogonalizer(&env.s)?;
    if nocc_a > s_inv_sqrt.ncols() {
        return Err(FerricError::General(format!(
            "ROHF: {nocc_a} alpha-occupied orbitals exceed the {} orthogonalized \
             functions the overlap retains",
            s_inv_sqrt.ncols()
        )));
    }
    // Fock diagonalizer: global by default, per-irrep under point-group blocking.
    let diag = crate::symmetric::FockDiag::new(
        sym,
        &env.s,
        crate::symmetric::PlainDiag::InvSqrt(&s_inv_sqrt),
    )?;
    phases.stop(orth_t);
    let c = rohf_initial_mos(
        &mut phases,
        (ctx, mol, prep, bounds, config),
        (&env, &s_inv_sqrt, &diag),
        (nocc_double, nocc_open),
        (initial_mos, &mut inj_jk),
    )?;

    // ROHF extrapolates the effective Fock with `step` (single-spin), so two
    // n×n matrices per entry despite the α/β densities it tracks internally.
    crate::driver::warn_if_diis_history_large(
        "ROHF",
        n,
        config.diis_size,
        crate::diis::DiisHistoryShape::SingleSpin,
        false,
        env.ooc_budget,
    );

    let need_k = c_k != 0.0 || k_mix.omega > 0.0;
    let coulomb_op = bounds.op;
    // The J/K route: one decision for all three solvers
    // (`fock_assembly::resolve_jk_route`, whose module doc tabulates every
    // case). The open-shell solvers fit only over a NAMED aux basis (no auto
    // default), J fits for a range-separated hybrid exactly as for any
    // other functional, and with `k_builder = "cosx"` COSX replaces the
    // ω = 0 DF-K (a named `df_k_aux` alongside it is refused).
    let jk_route = crate::fock_assembly::resolve_jk_route(
        config,
        crate::fock_assembly::JkExchange {
            functional: xc_contrib.is_some(),
            k_mix,
        },
        Spin::RestrictedOpen,
    )?;
    let (df_j, df_k, df_jk_route) = crate::driver::build_df_fitters(
        &mut phases,
        ctx,
        mol,
        coulomb_op,
        prep,
        config,
        &jk_route,
        &env,
        xc_contrib.as_ref().map(|_| k_mix),
    )?;
    let link_bound = crate::screening::LinkBound::SchwarzRef(bounds);
    let pluggable_k = rohf_pluggable_k(
        (ctx, mol, prep, config),
        (&env, &link_bound, coulomb_op),
        (df_j.is_some(), jk_route.k_df_routed, df_k.is_some()),
        (need_k, k_mix.omega),
    )?;
    let mut jk = crate::driver::OpenShellJk::new(
        (ctx, prep, bounds, config),
        env.ooc_budget,
        crate::driver::OpenShellJkSetup {
            inj_jk,
            df_j,
            df_k,
            started_k: (None, pluggable_k),
            need_k,
            omega: k_mix.omega,
        },
    )?;

    // F6 occupation guard (see `crate::rohf_occupation` for the measured
    // defect and each remedy). OFF under explicit MOM, which deliberately
    // holds non-aufbau (ΔSCF) states, and under `rohf_occupation_guard =
    // false`, which restores the pre-F6 loop exactly.
    let guard = crate::rohf_occupation::OccupationGuard::new(
        !config.mom_enabled() && config.rohf_occupation_guard,
        nocc_double,
        nocc_open,
    )
    // Injected (periodic) path: no continuity lock (it can hold a periodic
    // hole state the unrelaxed witness cannot refute); molecular path unchanged.
    .with_continuity_lock(!injected);
    crate::driver::commit_xc(xc_contrib.as_deref_mut())?;
    if let Some(x) = xc_contrib.as_ref() {
        x.start_scf();
    }
    // Mixed `dfj-pack`/`dfk-occ` builds only while the density moves, and an
    // f64 exit (see `crate::jk_ladder`); inert unless one of them is requested.
    let jk_ladder = crate::driver::jk_ladder(jk.df_j.is_some(), jk.df_k.is_some(), &env);

    let inp = RohfInputs {
        ctx,
        mol,
        prep,
        bounds,
        config,
        env: &env,
        rsh_response: rsh_response.as_ref(),
        s_inv_sqrt: &s_inv_sqrt,
        diag: &diag,
        nocc: (nocc_double, nocc_open),
        k_mix,
        c_k,
        injected,
        effective_level_shift: crate::driver::effective_level_shift(config),
        df_jk_route,
    };
    let mut xc = crate::driver::OpenShellXc { xc_contrib, inj_xc };
    let mut st = RohfState::new(&inp, c, guard);

    let mut iter = 0usize;
    loop {
        iter += 1;
        if iter > st.guard.iteration_cap(config.max_iter) {
            break;
        }
        if let Some(c_probe) = st.guard.take_pending_orbitals() {
            // A witness probe: build this pass's Fock from a single-swap
            // neighbour of the converged state (full rebuild, no increment).
            st.c = c_probe;
            (st.d_a, st.d_b) = build_rohf_densities(&st.c, nocc_double, nocc_open);
            jk.d_last_fock = None;
        }
        ctx.check_interrupted()?;
        let direct_incremental = jk.begin_iteration(iter);
        let d_total = &st.d_a + &st.d_b;
        st.record_density_change(&d_total);
        // ROHF caches no occupied MO block (its densities are built from the
        // integer closed/open blocks), so the DF-K and RSH exchange take the
        // density contraction (`d_occ = None`).
        let (f_a, f_b) = jk.build_fock(
            direct_incremental,
            (&st.d_a, &st.d_b, &d_total),
            (None, None),
            &env,
            (k_mix, need_k, c_k),
            &jk_ladder,
            &mut phases,
        )?;
        (st.f_a, st.f_b) = (f_a, f_b);
        let energy = st.energy(&inp, &mut xc, &d_total, hook.as_mut(), &mut phases)?;
        // Swap witness: while a probe is outstanding, `energy` is the probe
        // determinant's unrelaxed energy (see OccupationGuard::on_energy).
        match st
            .guard
            .on_energy(energy, iter, &mut st.diis, &mut st.mon, ctx.is_root())
        {
            crate::rohf_occupation::GuardStep::Proceed => {}
            crate::rohf_occupation::GuardStep::Continue => continue,
            crate::rohf_occupation::GuardStep::Return(r)
            | crate::rohf_occupation::GuardStep::Stop(r) => {
                return Ok(keep_for_gradient(&inp, &mut jk, *r));
            }
        }
        let m = st.measure(&inp, energy, &mut phases);
        let Some(converged) = st.gate(&inp, &xc, (iter, &jk_ladder), energy, &m)? else {
            continue;
        };
        // Gradient guard (F6): a stagnated DIIS extrapolation settles ΔP and
        // ΔE without being a stationary point (OH ROKS/PBE on CI, 0.58 Ha high).
        if iter > 1
            && st
                .guard
                .accept(converged, m.err_max, iter, &mut st.diis, ctx.is_root())
        {
            if let Some(r) = st.converged(&inp, &xc, &jk, (energy, iter), m.f_eff)? {
                return Ok(keep_for_gradient(&inp, &mut jk, r));
            }
            continue;
        }
        st.mon.note_energy(energy);
        st.update(&inp, &xc, &mut jk, iter, m, &mut phases)?;
    }
    Ok(st.max_iter(&inp, &jk))
}

/// (nocc_double, nocc_open) from the molecule's charge and multiplicity.
fn rohf_nocc(mol: &Molecule) -> Result<(usize, usize), FerricError> {
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
    Ok((nocc_double, nocc_open))
}

/// The guess stage, timed as `setup:guess`: the starting MOs. Caller-supplied
/// MOs if provided, else the CONFIGURED density guess (MINAO by default),
/// else hcore.
///
/// # Why this is not just hcore
///
/// MEASURED, `tests/rohf_state_selection.rs`, over 28 rows (15 chemical
/// systems, 13 of them at two bases; charged and neutral, doublet and
/// triplet, diatomic and polyatomic) against PySCF 2.13.0 ROHF at the same
/// geometry and basis. Each reference was cross-checked across five PySCF
/// `init_guess` settings and, for the wider sweep, 24 randomized starts, so
/// a guess-dependent reference is not mistaken for a converged one.
///
/// ```text
///                      above ref by >1e-3 eV    did not converge
///   hcore                   4 / 28                  2 / 28
///   MINAO                   1 / 28                  0 / 28
/// ```
///
/// MINAO repairs OH/6-31G +4.3027 → 0, F₂⁺/6-31G +3.0201 → 0, NH₂/6-31G
/// +1.7818 → 0, HeNe⁺/def2-SVP +0.1303 → 0, plus HeNe⁺/6-31G and
/// CN/cc-pVDZ, which exhaust 400 iterations from hcore without converging.
///
/// NOT FREE: CN is WORSE from MINAO at both bases — CN/6-31G +0.5753 eV (it
/// reaches the reference from hcore) and CN/cc-pVDZ +0.3865 eV (which trades
/// a non-answer for a high answer). Those are genuine ROHF stationary points
/// of the same operator, not wrong energies: PySCF itself converges to
/// ferric's −92.1186236 from 5 of 40 randomized starts. See
/// `cn_is_the_system_the_guess_fix_costs`, which pins both magnitudes.
///
/// The remaining rows are the same from either guess, several of them
/// BIT-identically, because those systems have one basin and the guess
/// cannot matter.
///
/// # No stability descent here, deliberately
///
/// UHF has an opt-in `scf_stability_descent` that follows a downhill
/// orbital-Hessian eigenvector when the guess is not enough. That is
/// UNAVAILABLE to ROHF and this solver does not fake it.
/// `crate::stability` implements exactly two operators (UHF and RHF), and
/// [`crate::stability::StabilitySkip::Rohf`] exists specifically to record
/// that the Roothaan open-shell Hessian is a THIRD one — one MO set with
/// closed/open/virtual blocks and Roothaan coupling, not a special case of
/// either. Running a UHF descent on ROHF MOs would follow an eigenvector of
/// an operator these orbitals are not a stationary point of. The converged
/// exit prints that skip reason when `check_stability` is set, so a ROHF
/// solution that this guess does not repair is DETECTABLE rather than
/// silently reported as the ground state. What IS available without a
/// Hessian is the F6 swap witness (`crate::rohf_occupation`): at
/// convergence the best one-electron neighbours are evaluated, and a state
/// one of them lowers is not returned.
#[allow(clippy::type_complexity)]
fn rohf_initial_mos(
    phases: &mut ferric_core::timing::PhaseAccumulator,
    (ctx, mol, prep, bounds, config): (
        &ParallelContext,
        &Molecule,
        &PreparedBasis,
        &SchwarzBounds,
        &RohfConfig,
    ),
    (env, s_inv_sqrt, diag): (
        &crate::driver::ScfEnv<'_>,
        &Array2<f64>,
        &crate::symmetric::FockDiag<'_>,
    ),
    (nocc_double, nocc_open): (usize, usize),
    (initial_mos, inj_jk): (
        Option<&Array2<f64>>,
        &mut Option<(Box<dyn JBuilder + '_>, Box<dyn KBuilder + '_>)>,
    ),
) -> Result<Array2<f64>, FerricError> {
    let (s, h) = (&env.s, &env.h);
    let n = prep.nbasis();
    let nocc_a = nocc_double + nocc_open;
    let guess_t = phases.begin("setup:guess");
    let c = if let Some(c0) = initial_mos {
        // Caller-supplied MOs: the injected path's `initial_mos`, or the
        // secondary-basis guess (`solve_rohf_routed`) on the molecular path.
        if c0.dim() != (n, n) {
            return Err(FerricError::General(format!(
                "ROHF: initial MO shape {:?} != ({n},{n})",
                c0.dim()
            )));
        }
        c0.clone()
    } else if let Some((ij, ik)) = inj_jk.as_mut() {
        // Injected (periodic) path: never the molecular build_jk / MINAO. An
        // explicit density builds the guess Fock from the INJECTED J/K;
        // otherwise the core guess, identical to the molecular hcore branch.
        match config.init_guess_density.as_ref() {
            Some(d0) => injected_rohf_guess_mos(
                ij.as_mut(),
                ik.as_mut(),
                d0,
                h,
                s,
                s_inv_sqrt,
                nocc_double,
                nocc_open,
            )?,
            None => {
                let _ = hcore_guess(s, h, nocc_a.max(1))?;
                core_guess_mos(h, s_inv_sqrt)?
            }
        }
    } else {
        match rohf_guess_mos(
            ctx,
            mol,
            prep,
            bounds,
            config,
            h,
            s,
            diag,
            nocc_double,
            nocc_open,
        )? {
            Some(c_guess) => c_guess,
            None => {
                // hcore guess: diagonalize bare h.
                let _ = hcore_guess(s, h, nocc_a.max(1))?; // sanity check it succeeds
                diag.run(h)?.1
            }
        }
    };
    phases.stop(guess_t);
    Ok(c)
}

/// The pluggable exchange builder ("link" / "cosx"): the semantics,
/// warnings, hard errors and per-spin `update_density` contract of
/// `solve_uhf`'s (see `fock_assembly::resolve_k_builder` /
/// `build_pluggable_k`). ROHF runs no COSX grid schedule, and the COSX
/// final-grid pass is RHF/UHF-only: an explicit request is refused, the
/// default one skipped with a warning. The ROHF-specific work (the Roothaan
/// coupling of the ASSEMBLED F_α/F_β) is downstream of K.
#[allow(clippy::type_complexity)]
fn rohf_pluggable_k<'a>(
    (ctx, mol, prep, config): (
        &'a ParallelContext,
        &'a Molecule,
        &'a PreparedBasis,
        &RohfConfig,
    ),
    (env, link_bound, op): (
        &crate::driver::ScfEnv<'_>,
        &'a crate::screening::LinkBound<'a>,
        ferric_integrals::operator::Operator,
    ),
    (df_j, k_df_routed, df_k): (bool, bool, bool),
    (need_k, omega): (bool, f64),
) -> Result<Option<Box<dyn KBuilder + 'a>>, FerricError> {
    let pluggable_k_kind = crate::fock_assembly::resolve_k_builder(
        config.k_builder.as_deref(),
        df_j,
        k_df_routed,
        df_k,
        need_k,
        omega,
    )?;
    let pluggable_k_kind =
        crate::fock_assembly::narrow_k_builder_to_supported(pluggable_k_kind, need_k, omega);
    crate::cosx_schedule::refuse(pluggable_k_kind, &config.cosx, "ROHF/ROKS")?;
    if pluggable_k_kind == Some("cosx") && config.cosx.final_grid.is_some() {
        // The COSX final-grid pass is implemented for RHF/RKS and UHF/UKS
        // only. An explicit request is refused; the default one is skipped,
        // so the reported energy is the SCF-grid energy.
        if config.cosx.final_pass_explicit {
            return Err(FerricError::General(
                "COSX final-grid pass (cosx final_grid) is implemented for RHF/RKS and UHF/UKS, \
                 not ROHF/ROKS; unset the final grid for ROHF/ROKS runs"
                    .into(),
            ));
        }
        eprintln!(
            "[ferric] COSX final-grid pass skipped: not implemented for ROHF/ROKS; the energy is \
             the SCF-grid energy"
        );
    }
    // `LinkBound::SchwarzRef` (built by the caller) applies whichever
    // screening refinement `bounds` carries, as in `solve_uhf`; with no table
    // it is byte-identical to passing `bounds` directly.
    crate::fock_assembly::build_pluggable_k(
        pluggable_k_kind,
        ctx,
        mol,
        prep,
        link_bound,
        op,
        &config.cosx,
        config.integral_thresh,
        env.ooc_budget,
    )
}

/// Hand a converged result's fitted J to the analytic gradient
/// (`raw_handoff::keep_for_gradient`) and return it.
fn keep_for_gradient(
    inp: &RohfInputs<'_, '_>,
    jk: &mut crate::driver::OpenShellJk<'_>,
    r: ScfResult,
) -> ScfResult {
    if r.converged {
        let (nocc_double, nocc_open) = inp.nocc;
        crate::raw_handoff::keep_for_gradient(
            inp.config,
            inp.ctx,
            inp.mol,
            inp.prep,
            &mut jk.df_j,
            inp.df_jk_route.as_ref(),
            &[nocc_double + nocc_open, nocc_double],
        );
    }
    r
}

/// What one ROHF solve reads and never changes.
struct RohfInputs<'r, 'e> {
    ctx: &'r ParallelContext,
    mol: &'r Molecule,
    prep: &'r PreparedBasis,
    bounds: &'r SchwarzBounds,
    config: &'r RohfConfig,
    env: &'r crate::driver::ScfEnv<'e>,
    /// See [`crate::driver::rsh_response`].
    rsh_response: Option<&'r crate::rsh_response::RshResponse<'r>>,
    /// S^{-1/2}, or the retained-rank X when the overlap is near-singular.
    s_inv_sqrt: &'r Array2<f64>,
    diag: &'r crate::symmetric::FockDiag<'r>,
    /// (nocc_double, nocc_open).
    nocc: (usize, usize),
    k_mix: ferric_dft::xc_trait::KMix,
    /// The exchange coefficient on the full Coulomb K, per spin.
    c_k: f64,
    /// The injected (periodic) path: no Newton or AH step.
    injected: bool,
    /// Meta-GGA (SCAN / r2SCAN) ROKS is stiffer than LDA/GGA and limit-cycles
    /// under plain DIIS; this is the same modest default virtual-block level
    /// shift the closed-shell RKS path uses when the user hasn't set one
    /// (`driver::effective_level_shift`), damped to zero as the gradient
    /// converges. Meta-GGA ROKS takes DIIS (no Newton f_xc step), so this
    /// shift is the relevant stabilizer.
    effective_level_shift: f64,
    /// The builders that produced the energy, for the analytic gradient.
    df_jk_route: Option<crate::result::DfJkRoute>,
}

/// One iteration's Roothaan effective Fock and orbital gradient.
struct RohfMeasure {
    f_eff: Array2<f64>,
    f_a_mo: Array2<f64>,
    f_b_mo: Array2<f64>,
    g_mo: Array2<f64>,
    /// The antisymmetrized MO gradient, back in the AO basis (DIIS's error).
    err: Array2<f64>,
    err_max: f64,
    sig: crate::rhf::ConvergenceSignals,
}

/// The iteration state of one ROHF solve.
struct RohfState {
    c: Array2<f64>,
    /// ROHF densities (AO):
    ///   D_c (closed/doubly-occupied) = 2 Σ_i C_i C_i^T  (i = 0..nocc_double)
    ///   D_o (open/singly-α-occupied) = Σ_j C_j C_j^T    (j = nocc_double..nocc_a)
    /// D_α = D_c/2 + D_o,  D_β = D_c/2  → (D_α + D_β) = D_c + D_o (total).
    /// D_α and D_β feed the J/K builders, as in UHF.
    d_a: Array2<f64>,
    d_b: Array2<f64>,
    /// This iteration's assembled spin Focks.
    f_a: Array2<f64>,
    f_b: Array2<f64>,
    diis: Diis,
    /// Convergence bookkeeping (prev energy, ΔP signals, divergence streak,
    /// stall history) — shared driver::ScfMonitor.
    mon: crate::driver::ScfMonitor,
    /// Previous iteration's total density, for the ΔP convergence signal
    /// (shared with solve_rhf via rhf::scf_converged). None on iter 1 → dp =
    /// INFINITY, so the gate can't fire before a real density change exists.
    prev_d_total: Option<Array2<f64>>,
    /// Most recent Thole-damped polarizable-embedding dipoles (None when
    /// `config.polarizable` is off/empty) — threaded into both results.
    last_induced_dipoles: Option<Array2<f64>>,
    /// MOM reference: (closed-MO AO block, open-MO AO block) from the most
    /// recently accepted iter. None until iter `config.mom_after_iter`.
    mom_ref: Option<(Array2<f64>, Array2<f64>)>,
    /// Last effective Fock, retained so the non-converged exit can report a
    /// well-formed ScfResult (see solve_rohf_best_effort's tail).
    f_eff_last: Array2<f64>,
    /// The spin Focks behind `f_eff_last`, kept for the gradient's
    /// energy-weighted density (see `ScfResult::rohf_spin_focks`).
    spin_focks_last: Option<(Array2<f64>, Array2<f64>)>,
    /// The F6 occupation guard.
    guard: crate::rohf_occupation::OccupationGuard,
}

impl RohfState {
    fn new(
        inp: &RohfInputs<'_, '_>,
        c: Array2<f64>,
        guard: crate::rohf_occupation::OccupationGuard,
    ) -> Self {
        let (nocc_double, nocc_open) = inp.nocc;
        let n = inp.prep.nbasis();
        let (d_a, d_b) = build_rohf_densities(&c, nocc_double, nocc_open);
        RohfState {
            c,
            d_a,
            d_b,
            f_a: Array2::zeros((0, 0)),
            f_b: Array2::zeros((0, 0)),
            diis: Diis::new(inp.config.diis_size),
            mon: crate::driver::ScfMonitor::new(),
            prev_d_total: None,
            last_induced_dipoles: None,
            mom_ref: None,
            f_eff_last: Array2::<f64>::zeros((n, n)),
            spin_focks_last: None,
            guard,
        }
    }

    /// ΔP vs the previous iteration's total density — the primary
    /// convergence signal (see rhf::scf_converged). The monitor stays at
    /// INFINITY until the first recorded change (iter 1).
    fn record_density_change(&mut self, d_total: &Array2<f64>) {
        if let Some(prev) = self.prev_d_total.as_ref() {
            self.mon.record_density_change(d_total, prev);
        }
        self.prev_d_total = Some(d_total.clone());
    }

    /// The energy stage: the electronic energy of the assembled F_σ (before
    /// V_xc), then V_xc, the hook and the solvent terms folded into F_σ with
    /// their energies — all before the Roothaan combination, so the coupled
    /// single-MO-set ROHF equations feel them.
    fn energy(
        &mut self,
        inp: &RohfInputs<'_, '_>,
        xc: &mut crate::driver::OpenShellXc<'_>,
        d_total: &Array2<f64>,
        hook: Option<&mut crate::fock::SpinFockHook<'_>>,
        phases: &mut ferric_core::timing::PhaseAccumulator,
    ) -> Result<f64, FerricError> {
        let env = inp.env;
        let h = &env.h;
        let (d_a, d_b) = (&self.d_a, &self.d_b);
        let (f_a, f_b) = (&mut self.f_a, &mut self.f_b);
        // Pre-XC electronic energy.
        let energy_t = phases.begin("energy_trace");
        let e_elec_no_xc: f64 = 0.5 * ((&(h + &*f_a) * d_a).sum() + (&(h + &*f_b) * d_b).sum());
        phases.stop(energy_t);
        let xc_t = phases.begin("xc_digest");
        let e_xc = if let Some(x) = xc.xc_contrib.as_ref() {
            x.add_xc_uks(d_a, d_b, f_a, f_b)
        } else if let Some(ix) = xc.inj_xc.as_mut() {
            // Injected (periodic) ROKS: V_σ added after the exchange term and
            // before the Roothaan combination; E_xc outside the trace.
            let (e, v_a, v_b) = ix.build_polarized(d_a, d_b)?;
            if v_a.dim() != f_a.dim() || v_b.dim() != f_b.dim() {
                return Err(FerricError::General(format!(
                    "solve_rohf_injected: XcBuilder::build_polarized returned V_xc of shapes \
                     {:?}/{:?}, expected {:?}",
                    v_a.dim(),
                    v_b.dim(),
                    f_a.dim()
                )));
            }
            *f_a += &v_a;
            *f_b += &v_b;
            e
        } else {
            0.0
        };
        phases.stop(xc_t);
        // Caller-supplied term (`solve_rohf_with_hook`): V_σ into F_σ before
        // the Roothaan combination, its energy outside the trace.
        let e_hook =
            crate::fock::apply_spin_fock_hook(hook, "solve_rohf_with_hook", (d_a, d_b), f_a, f_b)?;

        // COSMO + IEF-PCM reaction fields, built from the total density
        // D_a + D_b and added identically to BOTH spin Focks (before the
        // Roothaan effective-Fock combination); energies are standalone terms
        // — see driver::solvent_terms and rhf::RhfState::energy.
        let solvent_t = phases.begin("solvent_terms");
        let (e_cosmo, e_pcm, e_pol, iter_induced_dipoles) = crate::driver::solvent_terms(
            inp.mol,
            inp.prep,
            inp.config,
            env.cosmo_cavity.as_ref(),
            env.pcm_ctx.as_ref(),
            env.polarizable_site_basis.as_ref(),
            d_total,
            &mut [f_a, f_b],
        )?;
        self.last_induced_dipoles = iter_induced_dipoles;
        phases.stop(solvent_t);

        Ok(e_elec_no_xc + e_xc + e_hook + e_cosmo + e_pcm + e_pol + env.vnn)
    }

    /// The gradient stage: the Roothaan effective Fock (Guest-Saunders, via
    /// PySCF's projector form) and the DIIS error, the proper ROHF
    /// orbital-rotation gradient (PySCF `get_grad`). In the MO basis the
    /// gradient has only three nonzero off-diagonal blocks — the *unique*
    /// orbital rotations:
    ///   g[v,c] = f_α[v,c] + f_β[v,c]    (closed → virtual)
    ///   g[v,o] = f_α[v,o]               (open → virtual; only α occupies open)
    ///   g[o,c] = f_β[o,c]               (closed → open; only β leaves open)
    /// It is antisymmetrized (g − gᵀ) and projected back to the AO basis so
    /// DIIS still operates on (n × n) error matrices. This eliminates the
    /// within-class-rotation ambiguity that causes the LDA/PBE doublet-OH
    /// plateau in the FDS-SDF formulation.
    fn measure(
        &mut self,
        inp: &RohfInputs<'_, '_>,
        energy: f64,
        phases: &mut ferric_core::timing::PhaseAccumulator,
    ) -> RohfMeasure {
        let (nocc_double, nocc_open) = inp.nocc;
        let s = &inp.env.s;
        let (f_a, f_b, c) = (&self.f_a, &self.f_b, &self.c);
        let f_eff = roothaan_fock(f_a, f_b, &self.d_a, &self.d_b, s);
        self.f_eff_last = f_eff.clone();
        self.spin_focks_last = Some((f_a.clone(), f_b.clone()));
        // Serial section (outside any rayon region): serial BLAS thread count.
        let commutator_t = phases.begin("diis_commutator");
        let (f_a_mo, f_b_mo, g_mo, err): (Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>) =
            with_blas_threads(opt_in_blas_threads(), || {
                crate::blas_probe::record(&crate::blas_probe::ROHF_DIIS_ERROR);
                let f_a_mo: Array2<f64> = c.t().dot(f_a).dot(c);
                let f_b_mo: Array2<f64> = c.t().dot(f_b).dot(c);
                let g_mo = rohf_gradient_mo(&f_a_mo, &f_b_mo, nocc_double, nocc_open);
                // Antisymmetrize in MO basis, then transform back to AO.
                let g_mo_anti: Array2<f64> = &g_mo - &g_mo.t();
                let err = s.dot(c).dot(&g_mo_anti).dot(&c.t()).dot(s);
                (f_a_mo, f_b_mo, g_mo, err)
            });
        phases.stop(commutator_t);
        let sig = self.mon.signals(energy);
        let err_max = err.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        RohfMeasure {
            f_eff,
            f_a_mo,
            f_b_mo,
            g_mo,
            err,
            err_max,
            sig,
        }
    }

    /// The convergence and ladder decision: ΔP + loose ΔE, the same gate as
    /// solve_rhf/solve_uhf (the ROHF cation path, a gw100 fallback, hits the
    /// same RI energy/gradient noise floor at aTZ, so an `err_max <
    /// density_conv` gate would grind to MaxIter there; see
    /// rhf::scf_converged). A divergence or stall is an error. Also the
    /// per-iteration trace, verbose line and run-log record. `None`: redo
    /// this iteration in f64.
    fn gate(
        &mut self,
        inp: &RohfInputs<'_, '_>,
        xc: &crate::driver::OpenShellXc<'_>,
        (iter, jk_ladder): (usize, &crate::jk_ladder::JkLadder),
        energy: f64,
        m: &RohfMeasure,
    ) -> Result<Option<bool>, FerricError> {
        let config = inp.config;
        let (nocc_double, nocc_open) = inp.nocc;
        let conv_exit = crate::rhf::scf_converged(m.sig, config.energy_conv, config.density_conv);
        // A converged exit after a mixed J/K build is refused: this iteration
        // is redone in f64 at the same density, from a clean DIIS history and
        // with ΔP/ΔE invalidated (see `crate::jk_ladder::JkExit`).
        let conv_exit = match jk_ladder.veto_mixed_exit(conv_exit) {
            crate::jk_ladder::JkExit::Proceed(c) => c,
            crate::jk_ladder::JkExit::RedoInF64 => {
                self.diis.reset();
                self.mon.invalidate_deltas();
                self.prev_d_total = None;
                return Ok(None);
            }
        };
        let converged = conv_exit.is_some();
        crate::driver::bail_if_diverged_or_stalled(
            &mut self.mon,
            (energy, m.err_max, iter),
            config,
        )?;

        // FERRIC_ROHF_TRACE=1: per-iter diagnostic of plateau dynamics.
        // Logs (iter, energy, ΔE, err_max, per-block gradient max,
        // eigenvalues straddling SOMO, and the occupied-α↔virt overlap).
        let (de, err_max) = (m.sig.de, m.err_max);
        if rohf_trace() {
            let n = inp.prep.nbasis();
            let nocc_a = nocc_double + nocc_open;
            let (eps_now, _c_now) = diagonalize(&m.f_eff, inp.s_inv_sqrt)?;
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
            let (g_vc_max, g_vo_max, g_oc_max) =
                gradient_block_maxima(&m.g_mo, nocc_double, nocc_open);
            eprintln!(
                "ROHFTRACE it={iter:>3} E={energy:.10} dE={de:.3e} err={err_max:.3e} |g|vc={g_vc_max:.3e} |g|vo={g_vo_max:.3e} |g|oc={g_oc_max:.3e}  eps:{}",
                eps_window.join(" ")
            );
        }

        // `grad_rms` is `None`: this loop forms only `err_max` from the
        // antisymmetrized MO gradient, never an RMS.
        let label = if xc.active().contains(&true) {
            "roks"
        } else {
            "rohf"
        };
        crate::driver::log_scf_iter(
            inp.ctx,
            config,
            (label, "ROHF"),
            iter,
            (energy, de),
            &self.mon,
            err_max,
            None,
        );
        Ok(Some(converged))
    }

    /// The converged exit the F6 gradient guard accepted: the f64 V_xc, the
    /// exit eigenpairs in the converged occupation, the stability skip
    /// notice, and the result — returned now, or held by the swap witness
    /// (`None`) while its best single-swap neighbours are evaluated on the
    /// next passes.
    fn converged(
        &mut self,
        inp: &RohfInputs<'_, '_>,
        xc: &crate::driver::OpenShellXc<'_>,
        jk: &crate::driver::OpenShellJk<'_>,
        (energy, iter): (f64, usize),
        f_eff: Array2<f64>,
    ) -> Result<Option<ScfResult>, FerricError> {
        let (nocc_double, nocc_open) = inp.nocc;
        let s = &inp.env.s;
        let refined = refine_exit_focks(
            xc.xc_contrib.as_deref(),
            &mut self.f_a,
            &mut self.f_b,
            &self.d_a,
            &self.d_b,
            s,
        );
        let f_eff = match refined {
            Some(fe) => {
                self.spin_focks_last = Some((self.f_a.clone(), self.f_b.clone()));
                fe
            }
            None => f_eff,
        };
        let (eps, c_f) = inp.diag.run(&f_eff)?;
        // Keep the converged occupation (see `exit_labels`): index aufbau
        // here would hand back the other determinant of a ROKS state whose
        // effective-Fock spectrum is not in aufbau order.
        let (eps, c_f) = self.guard.exit_labels(&self.c, eps, c_f, s);
        let (d_a_f, d_b_f) = build_rohf_densities(&c_f, nocc_double, nocc_open);
        let density_total = &d_a_f + &d_b_f;
        // Stability analysis is NOT available for a ROHF/ROKS reference: the
        // Roothaan orbital Hessian is a third operator (one MO set with
        // closed/open/virtual blocks and Roothaan coupling), not a special
        // case of either implemented one, so running `uhf_internal_stability`
        // or `rhf_internal_stability` here would analyse a Hessian these MOs
        // are not a stationary point of. Skipped with a printed reason rather
        // than silently returning a wrong-operator verdict; the result field
        // stays `None`, which is documented as "not checked".
        if inp.config.check_stability {
            eprintln!(
                "SCF stability: check requested but SKIPPED — {}. \
                 ScfResult::stability is None (not checked), which does NOT mean stable.",
                crate::stability::StabilitySkip::Rohf.reason()
            );
        }
        let result = ScfResult {
            guess_basis: None,
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
            // Counts the quartets that produced THIS state; a later witness
            // probe's builds are not added to a result that stands.
            computed_quartets: jk.total_quartets,
            induced_dipoles: self.last_induced_dipoles.clone(),
            stability: None,
            stability_external: None,
            df_jk: inp.df_jk_route.clone(),
            rohf_spin_focks: self.spin_focks_last.clone(),
            xc: None,
            cosx_final: None,
            cosx_schedule: None,
            convergence: None,
            precision: Default::default(),
            relativistic: None,
        };
        // Swap witness (F6): returned now, or held while its best single-swap
        // neighbours are evaluated on the next passes.
        Ok(self.guard.on_converged(result, &self.f_a, &self.f_b))
    }

    /// The orbital update, in precedence order: the augmented-Hessian Newton
    /// step, the PCG Newton step, else DIIS.
    ///
    /// Both second-order steps solve H·κ = −g from the current orbitals `c`
    /// (H = the exact ROHF/ROKS orbital Hessian, `rohf_newton::hessian_matvec`;
    /// the per-spin diagonal Fock gap is only its preconditioner), then
    /// rotate C via the Cayley unitary.
    ///   - HF (xc=None): full Newton with HF orbital Hessian only.
    ///   - LDA ROKS: Newton + LDA f_xc kernel response (LdaFxcKernel).
    ///   - GGA / hybrid / RSH-GGA ROKS: Newton + GGA f_xc kernel response
    ///     (GgaFxcKernel — adds the σ = |∇ρ|² coupling terms). The hybrid /
    ///     RSH exact-exchange fraction is folded in via `k_mix_sr`/the K
    ///     builders exactly as in the Fock build, so the fxc kernel only
    ///     ever supplies the semilocal DFT second-derivative response.
    ///   - τ-only meta-GGA ROKS keeps DIIS. The τ kernel exists
    ///     (`FxcKernelStore::Mgga`, used by the ROKS Z-vector), but the ROKS
    ///     Newton step on it has not been measured since the 2026-11 fix
    ///     below (before it: NH2/6-31G r2SCAN, newton_trigger 1e-2, not
    ///     converged in 200 iterations vs DIIS 10).
    ///   - MEASURED 2026-11-11 (tests/roks_newton_convergence.rs; energy_conv
    ///     1e-9, density_conv 1e-7, max_iter 200; B3LYP with RI-J + DF-K
    ///     def2-universal-jkfit, PBE direct), iterations (HVPs), all at the
    ///     DIIS energy within 1e-7 Ha (AH HVP counts vary by a few between
    ///     runs: last-bit thread-count drift moves the Davidson's stop).
    ///     One HVP costs about one Fock build (its J/K + f_xc), so
    ///     iterations + HVPs is the Fock-equivalent cost:
    ///
    ///     ```text
    ///                         DIIS  Newton(1e-2)  soscf(1e-3)  AH(1e-2)
    ///       NH2/6-31G PBE       8      6 (6)        6 (6)        6 (25)
    ///       NH2/cc-pVDZ B3LYP   9      7 (8)        7 (4)        7 (30)
    ///       OH/cc-pVDZ B3LYP   18      8 (18)       9 (18)       8 (35-38)
    ///     ```
    ///
    ///     Degenerate ²Π, PBE/6-31G (energy 1e-10 / density 1e-8): OH on-axis
    ///     20 / 9 (21) / 11 (20); 1e-5 Å off-axis OH 18 / 10 (24) / 10 (14) /
    ///     AH 10 (46), CH 22 / 8 (25) / 10 (22) / AH not converged, NO 26 /
    ///     12 (20) / 13 (19) / AH 26 (251). Off-axis, which hole
    ///     orientation a run lands in is box dependent: here AH takes the
    ///     other one, on CI Newton/soscf did too (OH, NO). They are
    ///     degenerate up to the XC grid's rotational variance (up to
    ///     3.4e-6 Ha on OH/6-31G, #172), so off-axis no mode is held to
    ///     DIIS's energy closer than 1e-5 Ha.
    ///     Before the fix: NH2 PBE 136 (254) / 21 (31) / 140 (93), NH2 B3LYP
    ///     not converged (544) / 136 (309) / 169 (150), OH B3LYP not
    ///     converged (1270) / 167 (695) / not converged (193). Both branches
    ///     stepped from `eig(F_eff)` of the old-density Fock, where the
    ///     gradient is ≈ 0 by construction (an undamped Roothaan step); the
    ///     Hessian's Fock part was the diagonal gap only (decisive for OH's
    ///     soft π mode); and the AH Davidson returned κ = 0 (see `rohf_ah`).
    ///   - PCG cap 20 and the 0.1 componentwise clip: over the 37 Newton
    ///     steps of that test file PCG needed at most 9 iterations (never
    ///     the cap), and the clip bound in 2 steps, both on CH's near-flat
    ///     π-orientation mode (|κ|max 7.9, 0.76), where it is what keeps the
    ///     step bounded (FERRIC_ROHF_TRACE prints both per step).
    ///
    /// `xc_supports_newton_fxc` gates whether an f_xc-accelerated Newton/AH
    /// step is taken for this functional. Pure ROHF (xc=None) keeps the
    /// HF-only Hessian path.
    fn update(
        &mut self,
        inp: &RohfInputs<'_, '_>,
        xc: &crate::driver::OpenShellXc<'_>,
        jk: &mut crate::driver::OpenShellJk<'_>,
        iter: usize,
        m: RohfMeasure,
        phases: &mut ferric_core::timing::PhaseAccumulator,
    ) -> Result<(), FerricError> {
        let config = inp.config;
        let (nocc_double, nocc_open) = inp.nocc;
        let err_max = m.err_max;
        let xc_contrib = xc.xc_contrib.as_deref();
        let xc_supports_newton_fxc = xc_contrib.is_some() && !xc_is_metagga(config.xc.as_deref());
        // Augmented-Hessian Newton — used when err_max is below ah_trigger.
        // Handles vanishing Hessian eigenvalues (e.g., doublet OH at LDA with
        // near-degenerate SOMO/HOMO) that PCG can't resolve.
        let use_ah = config.ah_trigger > 0.0
            && !inp.injected
            && iter > 3
            && err_max < config.ah_trigger
            && (xc_contrib.is_none() || xc_supports_newton_fxc)
            && inp.k_mix.omega == 0.0;
        let use_newton = config.newton_trigger > 0.0
            && !inp.injected
            && iter > 3
            && err_max < config.newton_trigger
            && (xc_contrib.is_none() || xc_supports_newton_fxc);
        if !use_ah && !use_newton {
            return self.diis_step(inp, iter, m, phases);
        }
        // Step from the current orbitals `c`, the determinant whose densities
        // built f_a/f_b, so `f_a_mo`/`f_b_mo` carry the true gradient, as the
        // UHF/RHF Newton steps do. Replacing C by `eig(F_eff)` of the same
        // old-density Fock and stepping from there would put the gradient at
        // ≈ 0 by construction (|g| ~25x too small on NH2 B3LYP), so each
        // "Newton" step would be an undamped Roothaan step, which
        // limit-cycles on these radicals. Occupation labels need no
        // `guard.relabel`: `c` already carries the iterate's own closed /
        // open / virtual labels, which the step only rotates.
        //
        // The f_xc kernel (LDA or GGA) at the reference density, once per
        // step on the SCF's own screened grid (dense fallback); the response
        // closure borrows it.
        let fxc_storage = xc_contrib
            .filter(|_| xc_supports_newton_fxc)
            .map(|x| {
                let name = config
                    .xc
                    .as_deref()
                    .expect("xc_supports_newton_fxc implies Some(xc)");
                polarized_fxc_response(
                    x,
                    (inp.mol, inp.prep, config, name),
                    (&self.d_a, &self.d_b),
                    None,
                )
            })
            .transpose()?;
        let fxc_ref: Option<&crate::rohf_newton::FxcResponse<'_>> = fxc_storage.as_deref();

        // The response J/K come from this SCF's own RI-J / DF-K (the energy
        // being converged is the fitted one); `None` keeps direct.
        let responses = jk.responses();
        let inputs = crate::rohf_newton::RohfNewtonInputs {
            prep: inp.prep,
            bounds: inp.bounds,
            c: &self.c,
            f_a_mo: &m.f_a_mo,
            f_b_mo: &m.f_b_mo,
            nocc_double,
            nocc_open,
            // The SR/LR response comes from `rsh`; this scalar is read only
            // on the omega == 0 path.
            k_mix_sr: inp.c_k,
            rsh: inp.rsh_response,
            fxc: fxc_ref,
            thresh: config.integral_thresh,
            ooc_budget: inp.env.ooc_budget,
            j_response: responses.j(),
            k_response: responses.k(),
        };
        let (c_new, _kmax) = if use_ah {
            let ah_inputs = crate::rohf_ah::RohfAhInputs {
                base: &inputs,
                newton_level_shift: config.level_shift.max(1e-6),
                newton_max_step: ROHF_NEWTON_MAX_STEP,
            };
            crate::rohf_ah::rohf_ah_step(
                inp.ctx,
                &ah_inputs,
                /*max_step=*/ 0.2,
                // Below this |g| the AH step is zero (ΔP = 0, so the SCF
                // stops), as with the Newton branch's `cg_conv`: a fixed 1e-7
                // reported a tighter `density_conv` run converged at |g| up
                // to 1e-7 (2.6e-10 at density_conv 1e-12 on OH/6-31G).
                // At such tolerances the Davidson can stall short of it, and
                // `rohf_ah_step` then takes the PCG Newton step (`rohf_ah`).
                /*davidson_conv=*/
                config.density_conv.min(1e-7),
                /*davidson_max_vecs=*/ 50,
            )?
        } else {
            crate::rohf_newton::rohf_newton_step(
                inp.ctx,
                &inputs,
                config.level_shift.max(1e-6),
                ROHF_NEWTON_MAX_STEP,
                20,
                // Below cg_conv the step is zero (ΔP = 0), so a fixed 1e-7
                // would end a tighter `density_conv` run at |g| ~ 1e-7.
                config.density_conv.min(1e-7),
            )?
        };
        self.c = c_new;
        (self.d_a, self.d_b) = build_rohf_densities(&self.c, nocc_double, nocc_open);
        Ok(())
    }

    /// The DIIS orbital update: extrapolate the effective Fock, optionally
    /// level-shift the virtual–virtual block in the MO basis to damp
    /// open-shell oscillations, diagonalize, then the F6 hole-swap lock and
    /// MOM, and the densities.
    fn diis_step(
        &mut self,
        inp: &RohfInputs<'_, '_>,
        iter: usize,
        m: RohfMeasure,
        phases: &mut ferric_core::timing::PhaseAccumulator,
    ) -> Result<(), FerricError> {
        let config = inp.config;
        let (nocc_double, nocc_open) = inp.nocc;
        let nocc_a = nocc_double + nocc_open;
        let s = &inp.env.s;
        let err_max = m.err_max;
        let extrapolate_t = phases.begin("diis_extrapolate");
        let mut f_new = self.diis.step(&m.f_eff, &m.err);
        phases.stop(extrapolate_t);
        if inp.effective_level_shift > 0.0 && iter > 1 {
            const SHIFT_DAMP_ERR: f64 = 1e-3;
            let damp = err_max / (err_max + SHIFT_DAMP_ERR);
            let shift_eff = inp.effective_level_shift * damp;
            if shift_eff > 1e-10 {
                let c = &self.c;
                let shift_term: Array2<f64> = with_blas_threads(opt_in_blas_threads(), || {
                    crate::blas_probe::record(&crate::blas_probe::ROHF_LEVEL_SHIFT);
                    let c_virt = c.slice(ndarray::s![.., nocc_a..]);
                    let p_virt: Array2<f64> = c_virt.dot(&c_virt.t());
                    shift_eff * s.dot(&p_virt).dot(s)
                });
                f_new += &shift_term;
            }
        }
        let diagonalize_t = phases.begin("diagonalize");
        let (eps_new, c_new) = inp.diag.run(&f_new)?;
        phases.stop(diagonalize_t);
        // F6 hole-swap lock (see crate::rohf_occupation): a no-op (the same
        // `c_new`) unless the open shell has swapped several iterations
        // running, after which the labels follow orbital continuity.
        let announce = rohf_trace() || (config.verbose && inp.ctx.is_root());
        let (eps_new, c_new) =
            self.guard
                .select(&self.c, eps_new, c_new, s, &mut self.diis, announce);
        // ROHF builds its densities from the integer closed/open blocks, so
        // nothing below reads ε; it is still permuted with C (mom_reorder
        // keeps the pair consistent) and then dropped.
        let c_after_mom = if config.mom_after_iter > 0 && iter > config.mom_after_iter {
            match self.mom_ref.as_ref() {
                Some((ref_closed, ref_open)) => {
                    crate::mom::mom_reorder(
                        &c_new,
                        &eps_new,
                        s,
                        ref_closed,
                        ref_open,
                        nocc_double,
                        nocc_open,
                    )
                    .0
                }
                None => c_new,
            }
        } else {
            c_new
        };
        self.c = c_after_mom;
        if config.mom_after_iter > 0 && iter >= config.mom_after_iter {
            let ref_closed = self.c.slice(ndarray::s![.., ..nocc_double]).to_owned();
            let ref_open = self
                .c
                .slice(ndarray::s![.., nocc_double..nocc_double + nocc_open])
                .to_owned();
            self.mom_ref = Some((ref_closed, ref_open));
        }
        (self.d_a, self.d_b) = build_rohf_densities(&self.c, nocc_double, nocc_open);
        Ok(())
    }

    /// Finalization when the iteration cap is reached: the best-effort state,
    /// which the caller decides what to do with. As with
    /// `solve_uhf`/`solve_uhf_best_effort`, `solve_rohf` keeps its Err
    /// contract (existing callers unwrap or `?` it), while
    /// `solve_rohf_best_effort` returns the converged-so-far density and MOs
    /// so a convergence LADDER can carry them into the next rung.
    fn max_iter(self, inp: &RohfInputs<'_, '_>, jk: &crate::driver::OpenShellJk<'_>) -> ScfResult {
        let RohfState {
            c,
            d_a,
            d_b,
            mon,
            last_induced_dipoles,
            f_eff_last,
            spin_focks_last,
            guard,
            ..
        } = self;
        let (eps_last, c_last) = diagonalize(&f_eff_last, inp.s_inv_sqrt)
            .unwrap_or_else(|_| (vec![0.0; c.ncols()], c.clone()));
        let (eps_last, c_last) = guard.exit_labels(&c, eps_last, c_last, &inp.env.s);
        let density_total = &d_a + &d_b;
        ScfResult {
            guess_basis: None,
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
            iterations: inp.config.max_iter,
            computed_quartets: jk.total_quartets,
            induced_dipoles: last_induced_dipoles,
            stability: None,
            stability_external: None,
            df_jk: inp.df_jk_route.clone(),
            rohf_spin_focks: spin_focks_last,
            xc: None,
            cosx_final: None,
            cosx_schedule: None,
            convergence: None,
            precision: Default::default(),
            relativistic: None,
        }
    }
}

/// The ROHF orbital-rotation gradient in the MO basis (PySCF `get_grad`),
/// before antisymmetrization: only the three unique off-diagonal blocks
///   g[v,c] = f_α[v,c] + f_β[v,c]   (closed → virtual)
///   g[v,o] = f_α[v,o]              (open → virtual; only α occupies open)
///   g[o,c] = f_β[o,c]              (closed → open; only β leaves open)
/// are nonzero. Moved verbatim out of `solve_rohf_best_effort` (same element
/// order, same arithmetic).
fn rohf_gradient_mo(
    f_a_mo: &Array2<f64>,
    f_b_mo: &Array2<f64>,
    nocc_double: usize,
    nocc_open: usize,
) -> Array2<f64> {
    let n = f_a_mo.nrows();
    let nocc_a = nocc_double + nocc_open;
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
    g_mo
}

/// Per-block maxima `(|g|vc, |g|vo, |g|oc)` of the pre-antisymmetrized MO
/// gradient, for the FERRIC_ROHF_TRACE line. Moved verbatim out of
/// `solve_rohf_best_effort`.
fn gradient_block_maxima(
    g_mo: &Array2<f64>,
    nocc_double: usize,
    nocc_open: usize,
) -> (f64, f64, f64) {
    let n = g_mo.nrows();
    let nocc_a = nocc_double + nocc_open;
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
    (g_vc_max, g_vo_max, g_oc_max)
}

/// Build ROHF α/β densities from MO coefficients:
///   D_β = Σ_{i<nocc_double} C_i C_i^T
///   D_α = D_β + Σ_{j∈open} C_j C_j^T
pub(crate) fn build_rohf_densities(
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
    diag: &crate::symmetric::FockDiag<'_>,
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
        match crate::driver::minao_or_report("ROHF", mol, prep) {
            Some(d) => d,
            None => return Ok(None),
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
    match diag.run(&f_eff) {
        Ok((_, c)) => Ok(Some(c)),
        Err(e) => {
            if crate::rhf::scf_trace() {
                eprintln!("ROHF guess: diagonalizing the guess Fock failed ({e:?}); using hcore");
            }
            Ok(None)
        }
    }
}

/// Injected-path counterpart of [`rohf_guess_mos`] for an explicit
/// spin-summed guess density: split BY OCCUPATION (`D_σ = D · nocc_σ / N`, as
/// the molecular helper), guess Fock `F_σ = h + J[D] − K[D_σ]` from the
/// INJECTED builders (per spin `update_density` + `build`, so any Madelung
/// term is included consistently; no XC — it is a guess), Roothaan-combined
/// and diagonalized. Unlike the molecular helper a failing build is an error,
/// not a silent hcore fallback: on the injected path the builders ARE the
/// integrals.
#[allow(clippy::too_many_arguments)]
pub(crate) fn injected_rohf_guess_mos(
    j: &mut dyn JBuilder,
    k: &mut dyn KBuilder,
    d_total: &Array2<f64>,
    h: &Array2<f64>,
    s: &Array2<f64>,
    s_inv_sqrt: &Array2<f64>,
    nocc_double: usize,
    nocc_open: usize,
) -> Result<Array2<f64>, FerricError> {
    let n = h.nrows();
    if d_total.dim() != (n, n) {
        return Err(FerricError::General(format!(
            "solve_rohf_injected: init_guess_density shape {:?} != ({n},{n})",
            d_total.dim()
        )));
    }
    let nocc_a = nocc_double + nocc_open;
    let nocc_b = nocc_double;
    let nelec = (nocc_a + nocc_b).max(1);
    let d_a = d_total * (nocc_a as f64 / nelec as f64);
    let d_b = d_total * (nocc_b as f64 / nelec as f64);
    let mut jm = Array2::<f64>::zeros((n, n));
    j.build(d_total, &mut jm)?;
    let mut k_a = Array2::<f64>::zeros((n, n));
    let mut k_b = Array2::<f64>::zeros((n, n));
    crate::fock_assembly::build_open_shell_pluggable_k(k, &d_a, &d_b, &mut k_a, &mut k_b)?;
    let f_a = &(h + &jm) - &k_a;
    let f_b = &(h + &jm) - &k_b;
    let f_eff = roothaan_fock(&f_a, &f_b, &d_a, &d_b, s);
    let (_, c) = diagonalize(&f_eff, s_inv_sqrt)?;
    Ok(c)
}

/// At a converged exit: if the last XC build was mixed precision, swap its
/// V_xc in `f_a`/`f_b` for the f64 one (see
/// `UksXcContribution::refine_last_to_f64_uks`) and return the Roothaan
/// effective Fock rebuilt from them; `None` when nothing changed.
fn refine_exit_focks(
    xc: Option<&dyn ferric_dft::xc_trait::UksXcContribution>,
    f_a: &mut Array2<f64>,
    f_b: &mut Array2<f64>,
    d_a: &Array2<f64>,
    d_b: &Array2<f64>,
    s: &Array2<f64>,
) -> Option<Array2<f64>> {
    let refined = xc.is_some_and(|x| x.refine_last_to_f64_uks(f_a, f_b));
    refined.then(|| roothaan_fock(f_a, f_b, d_a, d_b, s))
}

pub(crate) fn roothaan_fock(
    f_a: &Array2<f64>,
    f_b: &Array2<f64>,
    d_a: &Array2<f64>,
    d_b: &Array2<f64>,
    s: &Array2<f64>,
) -> Array2<f64> {
    // Serial section (outside any rayon region): serial BLAS thread count.
    with_blas_threads(opt_in_blas_threads(), || {
        crate::blas_probe::record(&crate::blas_probe::ROOTHAAN_FOCK);
        roothaan_fock_inner(f_a, f_b, d_a, d_b, s)
    })
}

fn roothaan_fock_inner(
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

/// The ROHF orthogonalizer. When every overlap eigenvalue is at or above
/// the linear-dependence threshold (`rhf::lindep_thresh`) it is the
/// symmetric `S^{-1/2}` (`n x n`); otherwise it is the canonical `X`
/// (`n x m`, `m` the retained rank) of `rhf::canonical_orthogonalizer`, as
/// in RHF and UHF, so a near-null overlap mode is dropped rather than
/// scaled by `1/sqrt(λ)`. [`diagonalize`] and [`core_guess_mos`] accept
/// either shape.
pub(crate) fn rohf_orthogonalizer(s: &Array2<f64>) -> Result<Array2<f64>, FerricError> {
    // Serial section (outside any rayon region): serial BLAS thread count.
    with_blas_threads(opt_in_blas_threads(), || {
        crate::blas_probe::record(&crate::blas_probe::ROHF_ORTHOGONALIZER);
        let n = s.nrows();
        let (s_evals, s_evecs) = s
            .eigh(ndarray_linalg::UPLO::Upper)
            .map_err(|e| FerricError::Lapack(format!("S diag: {e}")))?;
        let thresh = crate::rhf::lindep_thresh();
        if s_evals.iter().any(|&l| !(l >= thresh)) {
            return crate::rhf::canonical_orthogonalizer(s);
        }
        let mut u_scaled = s_evecs.clone();
        for i in 0..n {
            let scale = 1.0 / s_evals[i].sqrt();
            for mu in 0..n {
                u_scaled[(mu, i)] *= scale;
            }
        }
        Ok(u_scaled.dot(&s_evecs.t()))
    })
}

/// Eigenvectors of the core Hamiltonian `h` in the orthonormal basis of `x`
/// ([`rohf_orthogonalizer`]), in the AO basis and ascending. A rectangular
/// `x` gives `n` columns, the last `n − m` zero (as
/// `driver::diagonalize_rect`).
pub(crate) fn core_guess_mos(h: &Array2<f64>, x: &Array2<f64>) -> Result<Array2<f64>, FerricError> {
    if x.ncols() != x.nrows() {
        return Ok(crate::driver::diagonalize_rect(h, x)?.1);
    }
    let h_prime = x.dot(h).dot(x);
    let (_, c_prime) = h_prime
        .eigh(ndarray_linalg::UPLO::Upper)
        .map_err(|e| FerricError::Lapack(format!("H' diag: {e}")))?;
    Ok(x.dot(&c_prime))
}

/// Eigenpairs of `f` in the orthonormal basis of `s_inv_sqrt`
/// ([`rohf_orthogonalizer`]): the symmetric transform when it is square,
/// `driver::diagonalize_rect` (padded to `n` columns, padded eigenvalues
/// `1e6`) when it is the rectangular retained-rank `X`.
pub(crate) fn diagonalize(
    f: &Array2<f64>,
    s_inv_sqrt: &Array2<f64>,
) -> Result<(Vec<f64>, Array2<f64>), FerricError> {
    if s_inv_sqrt.ncols() != s_inv_sqrt.nrows() {
        return crate::driver::diagonalize_rect(f, s_inv_sqrt);
    }
    // Serial section (outside any rayon region): serial BLAS thread count.
    ferric_integrals::blas_threads::with_blas_threads(
        ferric_integrals::blas_threads::opt_in_blas_threads(),
        || {
            let f_prime = s_inv_sqrt.dot(f).dot(s_inv_sqrt);
            let (evals, evecs) = f_prime
                .eigh(ndarray_linalg::UPLO::Upper)
                .map_err(|e| FerricError::Lapack(format!("F diag: {e}")))?;
            let c = s_inv_sqrt.dot(&evecs);
            Ok((evals.to_vec(), c))
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::basis;

    /// Overlap of rank 2 on 3 functions: one doubly occupied orbital fits,
    /// but 3 alpha-occupied orbitals (1 closed + 2 open) do not, so the
    /// start is refused instead of padding the open block with zero columns.
    #[test]
    fn spin_density_start_refuses_more_alpha_occupied_than_the_kept_rank() {
        let s = ndarray::arr2(&[[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        let d = Array2::<f64>::zeros((3, 3));
        let msg = rohf_spin_density_mos(&s, (&d, &d), 3, 1)
            .unwrap_err()
            .to_string();
        assert!(
            msg.contains("3 alpha-occupied")
                && msg.contains("1 doubly")
                && msg.contains("2 orthogonalized"),
            "{msg}"
        );
        assert!(rohf_spin_density_mos(&s, (&d, &d), 2, 1).is_ok());
    }

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
