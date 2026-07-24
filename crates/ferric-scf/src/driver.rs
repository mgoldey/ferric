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
    /// RSH SR/LR exchange fitters (ω > 0 only) — see `fock_assembly`.
    pub dfk_sr: Option<DfK<'a>>,
    pub dfk_lr: Option<DfK<'a>>,
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
) -> Result<ScfEnv<'a>, FerricError> {
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
    let cosmo_cavity = match config.cosmo.as_ref() {
        Some(cfg) => Some(crate::cosmo::CosmoCavity::build(mol, cfg)?),
        None => None,
    };
    let pcm_ctx = config
        .pcm
        .as_ref()
        .map(|pcfg| ferric_pcm::PcmContext::new(mol, pcfg))
        .transpose()?;
    let ooc_budget = crate::rhf::resolve_three_index_budget(config.three_index_budget_bytes);
    let (dfk_sr, dfk_lr) = if k_mix.omega > 0.0 {
        let (sr, lr) = crate::fock_assembly::build_rsh_dfk_pair(
            ctx,
            mol,
            prep,
            config.df_k_aux.as_deref(),
            k_mix.omega,
            ooc_budget,
        )?;
        (Some(sr), Some(lr))
    } else {
        (None, None)
    };
    Ok(ScfEnv { s, h, vnn, ooc_budget, cosmo_cavity, pcm_ctx, dfk_sr, dfk_lr })
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
            let (sum_sq, max) = a.iter().zip(b.iter()).fold((0.0f64, 0.0f64), |(sum_sq, max), (&x, &y)| {
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
/// The DIIS drivers retain up to `diis_size` (Fock, error) matrix pairs, each
/// `n × n` `f64` — `diis_size × 2 × n² × 8` bytes total (RHF/UHF/ROHF all
/// share this shape; UHF's coupled driver keeps separate α/β histories but
/// the *combined* error vector is still one logical DIIS slot per iteration,
/// so the same projection is used uniformly across all three variants rather
/// than chasing an exact per-variant byte count — this is an observability
/// heads-up, not a precise accounting). At `nbf=2000, diis_size=8` this is
/// ~0.5 GB, big enough to matter against a small `ooc_budget` but usually
/// negligible against the 3-index tensor it sits alongside.
///
/// Emits ONE stderr warning when the projection exceeds `ooc_budget / 4` —
/// mirroring `ferric_core::memory::warn_if_rss_over`'s philosophy: purely
/// observational, NEVER an error. A working job must never start failing
/// because of this check; it only ever prints a heads-up line.
pub(crate) fn warn_if_diis_history_large(label: &str, n: usize, diis_size: usize, ooc_budget: usize) {
    let projected = diis_size
        .saturating_mul(2)
        .saturating_mul(n)
        .saturating_mul(n)
        .saturating_mul(std::mem::size_of::<f64>());
    let threshold = ooc_budget / 4;
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
    ferric_core::memory::warn_if_rss_over(
        &format!("{variant} {stage}"),
        ooc_budget,
        1.1,
    );
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
    d_total: &Array2<f64>,
    focks: &mut [&mut Array2<f64>],
) -> Result<(f64, f64), FerricError> {
    let e_cosmo = if let Some(cavity) = cosmo_cavity {
        let cosmo_cfg = config.cosmo.as_ref().expect("cosmo cavity implies config.cosmo");
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
    Ok((e_cosmo, e_pcm))
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
}

impl ScfMonitor {
    pub fn new() -> Self {
        ScfMonitor {
            prev_e: 0.0,
            dp_rms: f64::INFINITY,
            dp_max: f64::INFINITY,
            errmax_history: Vec::new(),
            divergence_streak: 0,
        }
    }

    /// The settled signals for this iteration's convergence decision.
    pub fn signals(&self, energy: f64) -> ConvergenceSignals {
        ConvergenceSignals {
            de: (energy - self.prev_e).abs(),
            dp_rms: self.dp_rms,
            dp_max: self.dp_max,
        }
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
