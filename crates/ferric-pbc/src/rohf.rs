//! Stage 5b: Gamma-point restricted open-shell HF and KS (`gamma_rohf`,
//! `gamma_roks`), ported from `reference/pbc/pbc_uks.py::roks` (FINDINGS
//! "Iteration 10 (Python, Gamma UKS)": ROKS ≡ PySCF `pbc.dft.ROKS` to
//! ≤ 3.5e-12 on the triclinic 4H s+p triplet).
//!
//! The SCF is [`ferric_scf::rohf::solve_rohf_injected`]: the UHF/UKS
//! per-spin Fock, then the (unchanged) Guest-Saunders Roothaan combination on
//! one spatial MO set,
//!
//! ```text
//! F_σ = h + J[D_α + D_β] − a·(K[D_σ] + v_M S D_σ S) + V_σ[ρ_α, ρ_β]
//! ```
//!
//! with `a = 1`, `V_σ = 0` for ROHF and `a` = the functional's global
//! exact-exchange fraction for ROKS. The Madelung term lives in the injected
//! K builder (linear in D_σ: coefficient 1 per spin, FINDINGS Iteration 6)
//! and so rides on the exact-exchange part only (Iteration 8). At the same
//! density `E_ewald − E_none = −a v_M (N_α + N_β)/2`.
//!
//! # Ewald trap, staged start, gap report
//!
//! As for UHF/UKS ([`crate::uhf`] module doc): at Gamma `v_M S D_σ S` is
//! `v_M` × the σ-occupied projector, so `none` and `ewald` share stationary
//! densities while ewald lowers σ-occupied levels by `a·v_M`.
//! [`EwaldStart::Staged`] (default when `a > 0`) converges with
//! `exxdiv = none` first and restarts ewald from those MOs.
//!
//! ROHF has ONE MO set and no per-spin eigenvalues, so the per-spin gap is
//! built from the ACTUAL occupations ([`rohf_occupation_gaps`]): for each
//! ROHF MO `c_i`, `n_i^σ = c_iᵀ S D_σ S c_i` and `ε_i^σ = c_iᵀ F_σ c_i`
//! (the spin Fock, not the Roothaan one), `gap_σ = min ε^σ(unocc) −
//! max ε^σ(occ)`. Because `c_iᵀ S D_σ S c_i ∈ {0, 1}` for ROHF MOs, the ewald
//! gap is EXACTLY the none gap + `a·v_M` per spin, as for UHF. The check
//! `gap_σ ≥ a·v_M` is the UHF single-swap criterion transplanted: a
//! diagnostic, NOT derived for the Roothaan-coupled Hessian.
//!
//! Units: Bohr and Hartree.

use crate::dense_aft::ExxDiv;
use crate::dft::{
    refuse_molecular_grid_knobs, resolve_periodic_functional, DynXcRef, GammaUksConfig,
    GammaUksGridInfo, PeriodicDftError, PeriodicGridConfig, PeriodicXcConfig,
};
use crate::ewald::madelung_constant;
use crate::hcore::PeriodicHcore;
use crate::lattice::Cell;
use crate::uhf::{nocc_ab, EwaldStart, GammaUhfIntegrals, SpinGapReport};
use ferric_core::parallel::ParallelContext;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::result::ScfResult;
use ferric_scf::rhf::{PeriodicInjection, RhfConfig, XcBuilder};
use ferric_scf::rohf::{solve_rohf_injected, validate_injected_rohf};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;

/// Configuration for [`gamma_rohf`].
#[derive(Debug, Clone)]
pub struct GammaRohfConfig {
    /// Exchange divergence treatment applied by the K builder.
    pub exxdiv: ExxDiv,
    /// Start strategy for `exxdiv = ewald` (module doc).
    pub ewald_start: EwaldStart,
    /// SCF knobs; validated by `ferric_scf::rohf::validate_injected_rohf`.
    pub scf: RhfConfig,
    /// Optional starting MOs, one `(nao, nao)` set (the first
    /// `N_β + 2S` columns occupied). Used by the first stage only.
    pub initial_mos: Option<Array2<f64>>,
}

impl Default for GammaRohfConfig {
    fn default() -> Self {
        Self {
            exxdiv: ExxDiv::Ewald,
            ewald_start: EwaldStart::Staged,
            scf: RhfConfig {
                use_sad_guess: false,
                density_conv: 1e-10,
                max_iter: 200,
                ..Default::default()
            },
            initial_mos: None,
        }
    }
}

/// Result of [`gamma_rohf`].
#[derive(Debug, Clone)]
#[must_use]
pub struct GammaRohfResult {
    /// The final SCF (requested `exxdiv`); `mos_beta` is `None` (one MO set).
    pub scf: ScfResult,
    /// The `exxdiv = none` first stage, when the staged start ran.
    pub none_stage: Option<ScfResult>,
    /// `v_M` of the cell (applied iff `exxdiv = ewald`).
    pub madelung: f64,
    /// `(N_α, N_β)` from the cell's charge and multiplicity.
    pub nocc: (usize, usize),
    /// `⟨S²⟩` of the ROHF determinant from its MOs and the lattice overlap
    /// (= `S(S+1)` up to MO orthonormality).
    pub s2: f64,
    /// Occupation-aware per-spin gaps against `v_M_applied`
    /// ([`rohf_occupation_gaps`]).
    pub gaps: SpinGapReport,
}

/// Configuration for [`gamma_roks`] / [`gamma_roks_with_xc`] (the
/// [`GammaUksConfig`] fields, with one MO set as the optional guess).
#[derive(Debug, Clone)]
pub struct GammaRoksConfig {
    /// LDA / GGA / global-hybrid name (ignored by [`gamma_roks_with_xc`]).
    pub functional: String,
    /// The periodic grid (ignored by [`gamma_roks_with_xc`]).
    pub grid: PeriodicGridConfig,
    /// Exchange-divergence treatment of the injected K (consumed only by the
    /// `a·K` part).
    pub exxdiv: ExxDiv,
    /// Start strategy for `exxdiv = ewald` with `a > 0` (default staged;
    /// ignored for `a = 0`, where no K is built).
    pub ewald_start: EwaldStart,
    /// AO truncation and budget for the XC AO cache (ignored by
    /// [`gamma_roks_with_xc`]).
    pub xc: PeriodicXcConfig,
    /// SCF knobs, validated by `validate_injected_rohf` plus the
    /// molecular-grid fields refused here. [`GammaRoksConfig::new`] sets
    /// `level_shift` = [`ROKS_HYBRID_LEVEL_SHIFT`] and `max_iter` =
    /// [`ROKS_HYBRID_MAX_ITER`] for a functional with exact exchange (see
    /// there); every field is the caller's to override.
    pub scf: RhfConfig,
    /// Optional starting MOs, one `(nao, nao)` set; first stage only.
    pub initial_mos: Option<Array2<f64>>,
}

/// Default `scf.level_shift` (Hartree) that [`GammaRoksConfig::new`] sets for
/// a functional with exact exchange (`a > 0`). See [`GammaRoksConfig::new`].
pub const ROKS_HYBRID_LEVEL_SHIFT: f64 = 0.05;
/// Default `scf.max_iter` that [`GammaRoksConfig::new`] sets for a functional
/// with exact exchange (`a > 0`), per SCF stage.
pub const ROKS_HYBRID_MAX_ITER: usize = 600;

impl GammaRoksConfig {
    /// Defaults (grid (75, 302) SSF, exxdiv ewald, staged start, hcore
    /// guess, tight convergence) for `functional` — those of
    /// [`GammaUksConfig::new`], plus a DIIS robustness default for hybrids.
    ///
    /// # Level shift for `a > 0` (DIIS robustness, not a different answer)
    ///
    /// When `functional` resolves ([`resolve_periodic_functional`], the same
    /// name check [`gamma_roks`] runs) to an exact-exchange fraction
    /// `a > 0`, `scf.level_shift` = [`ROKS_HYBRID_LEVEL_SHIFT`] (0.05 Ha) and
    /// `scf.max_iter` = [`ROKS_HYBRID_MAX_ITER`] (600). `a = 0` (LDA/GGA)
    /// keeps the [`GammaUksConfig::new`] SCF defaults (shift 0, 200
    /// iterations), and so does a name that does not resolve (`gamma_roks`
    /// refuses it by name later; [`gamma_roks_with_xc`] ignores the name, so
    /// a caller-supplied hybrid builder must set the shift itself).
    ///
    /// Why (FINDINGS "ROKS PBE0 CI non-convergence (Python diagnosis) —
    /// 2026-09-25", measured on a numpy replica of the injected ROHF loop,
    /// `reference/pbc/roks_replica.py` / `run_roks_trap.py`, tri 4H s+p
    /// triplet PBE0 `exxdiv = none`): the ROKS minimum REPELS the undamped
    /// Roothaan map there, and DIIS (history 8) captures only starts within
    /// ~1e-5 of it, so convergence is start- and LAPACK-dependent (CI hit
    /// max_iter at −1.4348827058, a non-stationary snapshot of a chaotic
    /// wander). Over core-guess starts rotated by `exp(tK)`:
    ///
    /// * shift 0, 200 iterations: 2 of 19 reach the pin −1.465458280448;
    /// * shift 0.05, 600 iterations: 37 of 37 (6 seeds), 97-515 iterations,
    ///   median ~170; 0.1 also 37/37 (median ~320); 0.02 / 0.03 / 0.25 reach
    ///   only 9 / 18 / 18 of 19 within 600 (too weak to contract / crawling);
    /// * MOM (`mom_after_iter`) and the continuity lock instead land in the
    ///   non-aufbau stationary state −1.4434745673 (22 mHa above the pin).
    ///
    /// The shift acts on the virtual block only and is ramped by the solver
    /// as `ls·err/(err + 1e-3)` (the molecular ROHF ramp; `level_shift` is
    /// accepted by `validate_injected_rohf`), so it vanishes at convergence
    /// and cannot move the converged state: in the replica LDA/PBE/PBE0 (none
    /// and staged ewald) reproduce their pins to 1e-13 with it. The replica
    /// is not bit-identical to ferric, so the rates above are estimates;
    /// only one cell and one hybrid (PBE0) were measured. A second-order
    /// injected step would be the real cure; this is the robustness default
    /// until then.
    pub fn new(functional: &str) -> Self {
        let u = GammaUksConfig::new(functional);
        let mut scf = u.scf;
        let hybrid = resolve_periodic_functional(functional)
            .map(|(_, a)| a > 0.0)
            .unwrap_or(false);
        if hybrid {
            scf.level_shift = ROKS_HYBRID_LEVEL_SHIFT;
            scf.max_iter = ROKS_HYBRID_MAX_ITER;
        }
        Self {
            functional: u.functional,
            grid: u.grid,
            exxdiv: u.exxdiv,
            ewald_start: u.ewald_start,
            xc: u.xc,
            scf,
            initial_mos: None,
        }
    }
}

/// Result of [`gamma_roks`] / [`gamma_roks_with_xc`].
#[derive(Debug, Clone)]
#[must_use]
pub struct GammaRoksResult {
    /// The final SCF (requested `exxdiv`; energy includes E_xc and E_nn).
    pub scf: ScfResult,
    /// The `exxdiv = none` first stage, when the staged start ran.
    pub none_stage: Option<ScfResult>,
    /// `v_M` of the cell (applied, times `a`, iff `exxdiv = ewald`).
    pub madelung: f64,
    /// The functional's global exact-exchange fraction `a`.
    pub exact_exchange_fraction: f64,
    /// `(N_α, N_β)`.
    pub nocc: (usize, usize),
    /// `⟨S²⟩` from the MOs (= `S(S+1)` up to orthonormality).
    pub s2: f64,
    /// Occupation-aware per-spin gaps against `a·v_M_applied`.
    pub gaps: SpinGapReport,
    /// Semilocal `E_xc` at the converged `(D_α, D_β)`.
    pub e_xc: f64,
    /// Grid diagnostics ([`gamma_roks`] only).
    pub grid: Option<GammaUksGridInfo>,
    /// Driver stages as [`crate::dft::GammaRksResult::timings`]
    /// ([`gamma_roks`] only; empty from [`gamma_roks_with_xc`]).
    pub timings: crate::timing::PbcTimings,
}

/// `⟨S²⟩` of a single-determinant ROHF result from its ONE MO set:
/// `S_z(S_z+1) + N_β − Σ_{i<N_α, j<N_β} (c_iᵀ S c_j)²`. For S-orthonormal
/// MOs the sum is exactly `N_β`, so this measures `S(S+1)` plus the MO
/// orthonormality error (the ROHF spin purity is structural).
pub fn rohf_spin_square(r: &ScfResult, s: &Array2<f64>, na: usize, nb: usize) -> f64 {
    let sz = 0.5 * (na as f64 - nb as f64);
    if na == 0 || nb == 0 {
        return sz * (sz + 1.0) + nb as f64;
    }
    let c = &r.mos_alpha;
    let ca = c.slice(ndarray::s![.., ..na]);
    let cb = c.slice(ndarray::s![.., ..nb]);
    let ov = ca.t().dot(s).dot(&cb);
    let sum: f64 = ov.iter().map(|v| v * v).sum();
    sz * (sz + 1.0) + nb as f64 - sum
}

/// OCCUPATION-AWARE per-spin gaps of a converged ROHF/ROKS result against
/// `shift` (`a·v_M` applied; 0 for `exxdiv = none`) — module doc.
///
/// Per spin σ, over the ROHF MOs `c_i` (`r.mos_alpha`): occupation
/// `n_i^σ = c_iᵀ S D_σ S c_i` (> ½ occupied) and level `ε_i^σ = c_iᵀ F_σ c_i`
/// with `F_σ` from `r.rohf_spin_focks` (the spin Focks of the last iteration;
/// at convergence they agree with the final density to the SCF tolerance).
/// `gap_σ = min ε^σ(unoccupied) − max ε^σ(occupied)`: a hole below the Fermi
/// level gives a NEGATIVE gap. `None` for a spin with no occupied or no
/// unoccupied MO, or when the result carries no spin Focks.
pub fn rohf_occupation_gaps(r: &ScfResult, s: &Array2<f64>, shift: f64) -> SpinGapReport {
    fn gap(c: &Array2<f64>, f: &Array2<f64>, d: &Array2<f64>, s: &Array2<f64>) -> Option<f64> {
        if c.nrows() != s.nrows() || f.dim() != s.dim() || d.dim() != s.dim() {
            return None;
        }
        let sds = s.dot(d).dot(s);
        let fc = f.dot(c);
        let (mut occ_max, mut vir_min) = (f64::NEG_INFINITY, f64::INFINITY);
        for i in 0..c.ncols() {
            let ci = c.column(i);
            if ci.iter().all(|v| *v == 0.0) {
                continue;
            }
            let n_i = ci.dot(&sds.dot(&ci));
            let e_i = ci.dot(&fc.column(i));
            if n_i > 0.5 {
                occ_max = occ_max.max(e_i);
            } else {
                vir_min = vir_min.min(e_i);
            }
        }
        (occ_max.is_finite() && vir_min.is_finite()).then_some(vir_min - occ_max)
    }
    let (gap_alpha, gap_beta) = match (r.rohf_spin_focks.as_ref(), r.density_beta.as_ref()) {
        (Some((fa, fb)), Some(db)) => (
            gap(&r.mos_alpha, fa, &r.density_alpha, s),
            gap(&r.mos_alpha, fb, db, s),
        ),
        _ => (None, None),
    };
    let min_gap = [gap_alpha, gap_beta].into_iter().flatten().reduce(f64::min);
    SpinGapReport {
        gap_alpha,
        gap_beta,
        madelung_applied: shift,
        margin: min_gap.map(|g| g - shift),
    }
}

/// Output of the shared staged driver.
struct Staged {
    scf: ScfResult,
    none_stage: Option<ScfResult>,
    a: f64,
    v_m: f64,
    nocc: (usize, usize),
    s2: f64,
    gaps: SpinGapReport,
}

/// Shared body of [`gamma_rohf`] (`xc = None`, `a = 1`) and
/// [`gamma_roks_with_xc`]: the injected ROHF/ROKS, staged for ewald when
/// `a > 0`, then `⟨S²⟩` and the occupation-aware gap check.
#[allow(clippy::too_many_arguments)]
fn run_staged(
    who: &str,
    cell: &Cell,
    prep: &PreparedBasis,
    hc: &PeriodicHcore,
    ints: GammaUhfIntegrals<'_>,
    mut xc: Option<&mut dyn XcBuilder>,
    exxdiv: ExxDiv,
    ewald_start: EwaldStart,
    scf_cfg: &RhfConfig,
    initial_mos: Option<&Array2<f64>>,
) -> Result<Staged, FerricError> {
    validate_injected_rohf(scf_cfg).map_err(FerricError::from)?;
    let mol = cell.mol();
    let (na, nb) = nocc_ab(mol)?;
    let a = match xc.as_deref() {
        Some(x) => {
            if !x.supports_polarized() {
                return Err(PeriodicDftError::Unsupported {
                    feature: "closed-shell XcBuilder",
                    reason: format!(
                        "{who} needs XcBuilder::build_polarized (supports_polarized() is false)"
                    ),
                }
                .into());
            }
            x.exact_exchange_fraction()
        }
        None => 1.0,
    };
    if !(a.is_finite() && (0.0..=1.0).contains(&a)) {
        return Err(FerricError::General(format!(
            "{who}: exact-exchange fraction must be in [0, 1], got {a}"
        )));
    }
    let v_m = madelung_constant(cell)?;
    let applied = match exxdiv {
        ExxDiv::None => 0.0,
        ExxDiv::Ewald => v_m,
    };
    let ctx = ParallelContext::default();
    // Required by the solver signature; never read for integrals here.
    let bounds = SchwarzBounds::compute(Operator::coulomb(), prep)?;
    let mut run = |vm: f64, init: Option<&Array2<f64>>| {
        let (j, k) = crate::uhf::builders(ints, vm);
        let inj = PeriodicInjection {
            s: hc.s.clone(),
            h: hc.h.clone(),
            vnn: hc.enn,
            j,
            k,
            xc: match xc.as_mut() {
                Some(x) => Some(Box::new(DynXcRef(&mut **x))),
                None => None,
            },
        };
        solve_rohf_injected(
            &ctx,
            mol,
            prep,
            Operator::coulomb(),
            &bounds,
            scf_cfg,
            inj,
            init,
        )
    };
    let staged = exxdiv == ExxDiv::Ewald && ewald_start == EwaldStart::Staged && a > 0.0;
    let (scf, none_stage) = if staged {
        let first = run(0.0, initial_mos)?;
        let c0 = first.mos_alpha.clone();
        let second = run(applied, Some(&c0))?;
        (second, Some(first))
    } else {
        (run(applied, initial_mos)?, None)
    };
    let shift = a * applied;
    let gaps = rohf_occupation_gaps(&scf, &hc.s, shift);
    if !gaps.satisfied() {
        eprintln!(
            "{who} WARNING: occupation-aware per-spin gap below a*v_M (gap_alpha {:?}, \
             gap_beta {:?}, a*v_M {shift:.6}, margin {:?}). Under exxdiv=none this state has a \
             hole below the Fermi level: possibly the Gamma Ewald trap (FINDINGS Iterations 6, \
             10). Use EwaldStart::Staged or a better initial_mos.",
            gaps.gap_alpha, gaps.gap_beta, gaps.margin
        );
    }
    let s2 = rohf_spin_square(&scf, &hc.s, na, nb);
    Ok(Staged {
        scf,
        none_stage,
        a,
        v_m,
        nocc: (na, nb),
        s2,
        gaps,
    })
}

/// Gamma-point ROHF of `cell` (spin state from `cell.mol()`'s charge and
/// multiplicity) on the lattice one-electron terms `hc` and the J/K of
/// `ints`. `prep` must be the basis `hc`/`ints` were built in, from
/// `cell.mol()`.
///
/// Errors (by name) on config fields the injected ROHF refuses
/// (`validate_injected_rohf`: e.g. `ah_trigger`, `newton_trigger`,
/// `scf_stability_descent`, `xc` — use [`gamma_roks`]), shape mismatches and
/// a non-converged SCF stage. Warns when a per-spin gap is below `v_M`.
pub fn gamma_rohf(
    cell: &Cell,
    prep: &PreparedBasis,
    hc: &PeriodicHcore,
    ints: GammaUhfIntegrals<'_>,
    cfg: &GammaRohfConfig,
) -> Result<GammaRohfResult, FerricError> {
    let r = run_staged(
        "gamma_rohf",
        cell,
        prep,
        hc,
        ints,
        None,
        cfg.exxdiv,
        cfg.ewald_start,
        &cfg.scf,
        cfg.initial_mos.as_ref(),
    )?;
    Ok(GammaRohfResult {
        scf: r.scf,
        none_stage: r.none_stage,
        madelung: r.v_m,
        nocc: r.nocc,
        s2: r.s2,
        gaps: r.gaps,
    })
}

/// Gamma-point ROKS with [`crate::dft::PeriodicXc`] built from `cfg.functional` on
/// `cfg.grid` (spin-polarized evaluation, as [`crate::dft::gamma_uks`]).
///
/// Errors (by name) on refused functionals (RSH / meta-GGA / VV10 / double
/// hybrids), molecular-grid knobs (`scf.xc_omega`, `scf.dft_grid`,
/// `scf.nlc_grid`), SCF features `validate_injected_rohf` refuses, a
/// neighbour cutoff below the covering bound, budget overruns and a
/// non-converged SCF stage.
pub fn gamma_roks(
    cell: &Cell,
    prep: &PreparedBasis,
    hc: &PeriodicHcore,
    ints: GammaUhfIntegrals<'_>,
    cfg: &GammaRoksConfig,
) -> Result<GammaRoksResult, FerricError> {
    validate_injected_rohf(&cfg.scf).map_err(FerricError::from)?;
    refuse_molecular_grid_knobs(&cfg.scf)?;
    nocc_ab(cell.mol())?;
    // Cheap name checks before the grid is built.
    resolve_periodic_functional(&cfg.functional)?;
    let total = crate::timing::StageClock::start();
    let (grid, mut pxc, mut timings) =
        crate::dft::ks_grid_and_xc(cell, prep, &cfg.functional, &cfg.grid, &cfg.xc)?;
    let mut out = gamma_roks_with_xc(cell, prep, hc, ints, &mut pxc, cfg)?;
    let electrons_on_grid = pxc.integrate_density(&out.scf.density_total)?;
    out.grid = Some(GammaUksGridInfo {
        n_grid_points: grid.len(),
        neighbour_cutoff: grid.neighbour_cutoff(),
        electrons_on_grid,
    });
    timings.add_stage(&pxc.eval_timing());
    timings.finish(&total);
    out.timings = timings;
    Ok(out)
}

/// [`gamma_roks`] with a caller-supplied spin-polarized [`XcBuilder`]
/// (`cfg.functional`, `cfg.grid`, `cfg.xc` ignored). A builder with `a = 1`
/// and zero `E_xc`/`V_σ` is exactly [`gamma_rohf`].
pub fn gamma_roks_with_xc(
    cell: &Cell,
    prep: &PreparedBasis,
    hc: &PeriodicHcore,
    ints: GammaUhfIntegrals<'_>,
    xc: &mut dyn XcBuilder,
    cfg: &GammaRoksConfig,
) -> Result<GammaRoksResult, FerricError> {
    refuse_molecular_grid_knobs(&cfg.scf)?;
    let r = run_staged(
        "gamma_roks",
        cell,
        prep,
        hc,
        ints,
        Some(&mut *xc),
        cfg.exxdiv,
        cfg.ewald_start,
        &cfg.scf,
        cfg.initial_mos.as_ref(),
    )?;
    let d_b = r.scf.density_beta.as_ref().ok_or_else(|| {
        FerricError::General("gamma_roks: the SCF returned no beta density".into())
    })?;
    let (e_xc, _, _) = xc.build_polarized(&r.scf.density_alpha, d_b)?;
    Ok(GammaRoksResult {
        madelung: r.v_m,
        exact_exchange_fraction: r.a,
        nocc: r.nocc,
        s2: r.s2,
        gaps: r.gaps,
        e_xc,
        grid: None,
        none_stage: r.none_stage,
        scf: r.scf,
        timings: crate::timing::PbcTimings::default(),
    })
}
