//! Stage 7: Gamma-point closed-shell direct RPA (dRPA) on the periodic
//! integrals (Rust port of `reference/pbc/pbc_rpa.py`, FINDINGS "Iteration 4
//! (Python, Gamma dRPA)" and its "For the Rust port (stage 7)" section).
//!
//! ```text
//! Π_PQ(iω) = 4 Σ_ia B^P_ia B^Q_ia e_ia / (ω² + e_ia²),     e_ia = ε_a − ε_i > 0
//! E_c      = (1/2π) ∫_0^∞ dω { ln det[1 + Π(iω)] − tr Π(iω) }
//! ```
//!
//! Real arithmetic (Gamma orbitals are real). `(ia|jb)` lives in the SAME
//! G = 0-dropped kernel as the HF ERI; ov pair densities are neutral
//! (`C_oᵀ S C_v = 0`), so no G = 0 correction enters.
//!
//! # Integral sources ([`GammaDrpaIntegrals`])
//!
//! * [`GammaDrpaIntegrals::RsGdf`] — production. The SCF's own metric-dressed
//!   RS-GDF `B[k, μν]` → `B[k, ia]` ([`crate::mp2::b_ov_from_ao_b`], shared
//!   with MP2) → a [`ferric_mp2::rimp2::RpaIntermediates`] → ferric-rpa's
//!   [`ferric_rpa::run_pdep_rpa_from_parts`]: the molecular PDEP-RPA default
//!   pipeline (full-rank Lanczos eigensolve of ε̃(0), log-det frequency
//!   summands, the same energy integrator) without any molecular basis
//!   object. Per the prototype:
//!   - `b_ov` is ALREADY metric-dressed (`B = s^{-1/2} Uᵀ J3ᵀ`); no further
//!     `V^{-1/2}` is applied;
//!   - `naux` = the KEPT count after the lindep cut (it sizes the eigensolve);
//!   - `v_inv_sqrt` = `U_kept s_kept^{-1/2}` ([`RsGdf::metric_inv_sqrt`],
//!     `(naux_ao, naux_kept)`), used only to back-transform eigenpotentials;
//!   - `trunc_thresh = 0` (full rank; also the production rule) and
//!     `chi0_sparsity = Dense`. ferric-rpa's `run_pdep_rpa_from_intermediates`
//!     is NOT used: its atom seed sizes itself from `dfbs.nbasis()`, which is
//!     wrong after lindep drops, and it reads ε from `rhf.eps_r()`, which
//!     cannot carry the Madelung convention.
//! * [`GammaDrpaIntegrals::DenseAft`] — TEST/ORACLE ONLY: exact dense
//!   `(ia|jb) = Wᵀ I W` from the pure-AFT tensor and the PLASMON formula
//!   ([`drpa_plasmon`]) — no aux, no frequency quadrature, no eigensolve of
//!   ε̃: a different ERI AND a different energy construction from the B path.
//!
//! # Denominators
//!
//! Exactly the MP2 story ([`crate::mp2`] module doc): the reference's exxdiv
//! must be stated, and [`crate::mp2::occupied_shift`] turns it into the
//! requested [`Mp2Denominators`] convention (the type is shared; the name is
//! historical). The prototype measured shifted dRPA converging to the
//! molecule as a⁻³ with a coefficient predicted from molecular quantities
//! (H2/STO-3G c3 = 0.922741), unshifted as 1/a.
//!
//! # Quadrature
//!
//! Gauss–Legendre mapped by `ω = u0 (1+x)/(1−x)`, `u0 = 0.5` (the prototype's
//! `gl_quadrature(n, 0.5)`, ferric's `gauss_legendre_nodes`). Default
//! [`DEFAULT_GAMMA_DRPA_QUAD_POINTS`] = 40, NOT ferric-rpa's 20: the prototype
//! measured the 20-point grid at 2.3e-6 Ha on H2O/6-31G (core e_ia ≈ 21 Ha ≫
//! u0) and 7.4e-10 at 40. Validated strictly to
//! `[MIN_GAMMA_DRPA_QUAD_POINTS, MAX_GAMMA_DRPA_QUAD_POINTS]`.
//!
//! # Not measured (do not extrapolate)
//!
//! Anything > 16 AOs, frozen core on a periodic cell, real solids (a⁻³ is the
//! isolated-molecule law), PDEP truncation on a periodic B, cost.
//!
//! # Memory
//!
//! Every buffer ferric-pbc allocates or hands to ferric-rpa is reserved on a
//! [`crate::budget`] ledger against [`GammaDrpaConfig::budget_bytes`] before
//! the first allocation; the per-worker quadrature scratch is counted ONCE
//! (a gate must not read the thread count — ferric-rpa's own ceiling check,
//! which does, runs afterwards and narrows its panels instead).
//!
//! Units: Bohr and Hartree.

use crate::budget::{bytes_of, Ledger};
use crate::dense_aft::{DenseAftEri, ExxDiv};
use crate::ewald::madelung_constant;
use crate::lattice::Cell;
use crate::mp2::{b_ov_from_ao_b, occupied_shift, ovov_from_dense, Mp2Denominators};
use crate::rsgdf::RsGdf;
use ferric_core::FerricError;
use ferric_mp2::rimp2::{active_occ, RpaIntermediates};
use ferric_rpa::config::{QuadratureConfig, QuadratureScheme};
use ferric_rpa::{
    run_pdep_rpa_from_parts, Chi0Backend, Chi0Sparsity, Eigensolver, PdepRpaConfig, PdepRpaResult,
};
use ferric_scf::result::{ScfResult, Spin};
use ndarray::{s, Array2};
use ndarray_linalg::{Eigh, UPLO};
use std::f64::consts::PI;

/// Default number of imaginary-frequency points (module doc).
pub const DEFAULT_GAMMA_DRPA_QUAD_POINTS: usize = 40;
/// Smallest accepted grid (below it the error is ≫ 1e-6 Ha on any system
/// with a core or high virtual; prototype H2O/6-31G n = 10: 1.9e-5).
pub const MIN_GAMMA_DRPA_QUAD_POINTS: usize = 8;
/// Largest accepted grid (the log-det summand's large-ω cancellation is
/// measurable by n = 1024: prototype 1.9e-10 drift).
pub const MAX_GAMMA_DRPA_QUAD_POINTS: usize = 1024;
/// Gauss–Legendre map scale u0 (Hartree).
pub const GAMMA_DRPA_QUAD_U0: f64 = 0.5;

/// Strict check of a quadrature size (no clamping, no default).
pub fn validate_quad_points(n: usize) -> Result<(), FerricError> {
    if (MIN_GAMMA_DRPA_QUAD_POINTS..=MAX_GAMMA_DRPA_QUAD_POINTS).contains(&n) {
        Ok(())
    } else {
        Err(FerricError::General(format!(
            "Gamma dRPA: quad_points must lie in [{MIN_GAMMA_DRPA_QUAD_POINTS}, \
             {MAX_GAMMA_DRPA_QUAD_POINTS}], got {n}"
        )))
    }
}

/// Settings for [`gamma_drpa`]. Deliberately no `Default`: the reference
/// exxdiv and the denominator convention must be stated.
#[derive(Debug, Clone, Copy)]
pub struct GammaDrpaConfig {
    /// Core orbitals excluded from correlation (validated by `active_occ`).
    pub frozen_core: usize,
    /// The `exxdiv` the RHF REFERENCE was converged with.
    pub reference_exxdiv: ExxDiv,
    /// Denominator convention (shared with MP2).
    pub denominators: Mp2Denominators,
    /// Frequency points (validated by [`validate_quad_points`]; unused by the
    /// dense plasmon oracle but still validated).
    pub quad_points: usize,
    /// Memory budget in bytes (`None` = ferric's unified budget).
    pub budget_bytes: Option<usize>,
}

impl GammaDrpaConfig {
    /// Madelung-shifted denominators, no frozen core, 40 frequency points,
    /// ferric's unified budget.
    pub fn shifted(reference_exxdiv: ExxDiv) -> Self {
        Self {
            frozen_core: 0,
            reference_exxdiv,
            denominators: Mp2Denominators::MadelungShifted,
            quad_points: DEFAULT_GAMMA_DRPA_QUAD_POINTS,
            budget_bytes: None,
        }
    }
}

/// Where `(ia|jb)` comes from.
#[derive(Debug, Clone, Copy)]
pub enum GammaDrpaIntegrals<'a> {
    /// Production: the fitted periodic `B` (the same one the SCF used).
    RsGdf(&'a RsGdf),
    /// TEST/ORACLE ONLY: exact dense pure-AFT `(ia|jb)` + plasmon formula.
    DenseAft(&'a DenseAftEri),
}

/// Gamma-point dRPA result.
#[derive(Debug, Clone)]
#[must_use]
pub struct GammaDrpaResult {
    /// dRPA correlation energy.
    pub drpa_corr: f64,
    /// `rhf.energy + drpa_corr` (the RHF energy is the reference's own; see
    /// [`crate::mp2::GammaMp2Result::total_energy`]).
    pub total_energy: f64,
    /// Gamma-point Madelung constant `v_M` of the cell.
    pub madelung: f64,
    /// Shift actually added to every active occupied ε (0, −v_M or +v_M).
    pub occ_shift: f64,
    /// Correlated occupied / virtual counts.
    pub nocc_active: usize,
    pub nvir: usize,
    /// Rows of the fitted B (`None` for the dense oracle).
    pub naux: Option<usize>,
    /// Frequency points used (`None` for the plasmon oracle).
    pub quad_points: Option<usize>,
    /// Static dielectric eigenvalues λ(0) (`None` for the plasmon oracle).
    pub eigenvalues_static: Option<Vec<f64>>,
}

/// The ferric-rpa configuration this driver always uses (module doc).
pub fn pdep_config(quad_points: usize, budget_bytes: usize) -> PdepRpaConfig {
    PdepRpaConfig {
        frozen_core: 0,
        trunc_thresh: 0.0,
        quadrature: QuadratureConfig {
            scheme: QuadratureScheme::GaussLegendre,
            n_points: quad_points,
            u0: GAMMA_DRPA_QUAD_U0,
        },
        eigensolver: Eigensolver::Lanczos,
        chi0_backend: Chi0Backend::Dense,
        chi0_sparsity: Chi0Sparsity::Dense,
        memory_budget_bytes: Some(budget_bytes),
        need_inv_dielectric_freq: false,
        need_eigenvalues_freq: false,
        run_diagnostics: false,
        ..Default::default()
    }
}

/// `e_ia` in `i·nvir + a` order; every entry must be finite and > 0.
fn excitation_energies(eps_occ: &[f64], eps_vir: &[f64]) -> Result<Vec<f64>, FerricError> {
    let mut e = Vec::with_capacity(eps_occ.len() * eps_vir.len());
    for &ei in eps_occ {
        for &ea in eps_vir {
            let d = ea - ei;
            if !(d.is_finite() && d > 0.0) {
                return Err(FerricError::General(format!(
                    "Gamma dRPA: non-positive or non-finite e_ia = {d} (ε_i {ei}, ε_a {ea}); the \
                     dRPA frequency integral is undefined"
                )));
            }
            e.push(d);
        }
    }
    Ok(e)
}

/// dRPA from an already-transformed `b_ov` `(naux, nocc·nvir)` and the ACTIVE
/// denominators, through ferric-rpa's [`run_pdep_rpa_from_parts`] with
/// [`pdep_config`]. `v_inv_sqrt = None` uses the identity (eigenpotentials
/// then stay in the dressed basis; the energy does not depend on it).
/// Exposed for tests (e.g. scaling `b_ov` by √λ scales the kernel by λ).
pub fn drpa_from_b_ov(
    b_ov: Array2<f64>,
    v_inv_sqrt: Option<Array2<f64>>,
    eps_occ: &[f64],
    eps_vir: &[f64],
    quad_points: usize,
    budget_bytes: Option<usize>,
) -> Result<PdepRpaResult, FerricError> {
    let nocc = eps_occ.len();
    let inter = intermediates(b_ov, v_inv_sqrt, nocc, eps_vir.len(), nocc, 0)?;
    drpa_from_intermediates(&inter, eps_occ, eps_vir, quad_points, budget_bytes)
}

fn intermediates(
    b_ov: Array2<f64>,
    v_inv_sqrt: Option<Array2<f64>>,
    nocc: usize,
    nvir: usize,
    nocc_total: usize,
    first_occ: usize,
) -> Result<RpaIntermediates, FerricError> {
    let naux = b_ov.nrows();
    if b_ov.ncols() != nocc * nvir {
        return Err(FerricError::General(format!(
            "Gamma dRPA: b_ov is {:?}, expected (naux, {nocc}·{nvir})",
            b_ov.dim()
        )));
    }
    let v_inv_sqrt = v_inv_sqrt.unwrap_or_else(|| Array2::eye(naux));
    Ok(RpaIntermediates {
        b_ov,
        v_inv_sqrt,
        nocc,
        nvir,
        nocc_total,
        first_occ,
        naux,
    })
}

fn drpa_from_intermediates(
    inter: &RpaIntermediates,
    eps_occ: &[f64],
    eps_vir: &[f64],
    quad_points: usize,
    budget_bytes: Option<usize>,
) -> Result<PdepRpaResult, FerricError> {
    validate_quad_points(quad_points)?;
    let _ = excitation_energies(eps_occ, eps_vir)?;
    let cfg = pdep_config(quad_points, crate::budget::resolve(budget_bytes));
    run_pdep_rpa_from_parts(inter, eps_occ, eps_vir, &cfg)
}

/// TEST/ORACLE: the O(V²) term of the log expansion,
/// `−(1/2π) ∫_0^∞ dω tr Π(iω)² / 2`, on the same GL grid (`u0 = 0.5`).
/// Analytically equal to the DIRECT (Coulomb-only) MP2 energy
/// `2 Σ (ia|jb)² / (ε_i + ε_j − ε_a − ε_b)`. Written independently of
/// ferric-rpa (only the grid is shared); dense `Π`, test scale.
pub fn drpa_second_order_from_b_ov(
    b_ov: &Array2<f64>,
    eps_occ: &[f64],
    eps_vir: &[f64],
    quad_points: usize,
) -> Result<f64, FerricError> {
    if !(1..=4 * MAX_GAMMA_DRPA_QUAD_POINTS).contains(&quad_points) {
        return Err(FerricError::General(format!(
            "drpa_second_order_from_b_ov: quad_points {quad_points} out of range"
        )));
    }
    let e = excitation_energies(eps_occ, eps_vir)?;
    let (naux, nov) = b_ov.dim();
    if nov != e.len() {
        return Err(FerricError::General(format!(
            "drpa_second_order_from_b_ov: b_ov is {:?}, expected {} columns",
            b_ov.dim(),
            e.len()
        )));
    }
    let (freqs, weights) =
        ferric_rpa::quadrature::gauss_legendre_nodes(quad_points, GAMMA_DRPA_QUAD_U0);
    let mut total = 0.0_f64;
    for (&w, &wk) in freqs.iter().zip(&weights) {
        let mut bs = b_ov.to_owned();
        for (p, &ep) in e.iter().enumerate() {
            let f = (4.0 * ep / (w * w + ep * ep)).sqrt();
            bs.column_mut(p).mapv_inplace(|x| x * f);
        }
        // Same nonzero spectrum from the smaller Gram matrix; tr Π² = ‖G‖_F².
        let g = if naux <= nov {
            bs.dot(&bs.t())
        } else {
            bs.t().dot(&bs)
        };
        let tr_pi2: f64 = g.iter().map(|x| x * x).sum();
        total += wk * (-0.5 * tr_pi2);
    }
    Ok(total / (2.0 * PI))
}

/// TEST/ORACLE: singlet direct RPA from an exact `(ia|jb)` matrix
/// `(nov, nov)` (row/column `i·nvir + a`) by the plasmon formula
/// (`A = D + 2K`, `B = 2K`):
/// `Ω² = eig[D^{1/2} (D + 4K) D^{1/2}]`, `E_c = ½ (Σ Ω − tr A)`.
/// No quadrature, no aux. Errors on a non-positive Ω² (dRPA instability).
pub fn drpa_plasmon(
    ovov: &Array2<f64>,
    eps_occ: &[f64],
    eps_vir: &[f64],
) -> Result<f64, FerricError> {
    let e = excitation_energies(eps_occ, eps_vir)?;
    let nov = e.len();
    if ovov.dim() != (nov, nov) {
        return Err(FerricError::General(format!(
            "drpa_plasmon: (ia|jb) is {:?}, expected ({nov}, {nov})",
            ovov.dim()
        )));
    }
    let sd: Vec<f64> = e.iter().map(|x| x.sqrt()).collect();
    let mut m = Array2::<f64>::zeros((nov, nov));
    for p in 0..nov {
        for q in 0..nov {
            // Symmetrised K (Wᵀ I W is symmetric only to roundoff).
            let k = 0.5 * (ovov[(p, q)] + ovov[(q, p)]);
            m[(p, q)] = 4.0 * sd[p] * k * sd[q];
        }
        m[(p, p)] += e[p] * e[p];
    }
    let (lam, _) = m
        .eigh(UPLO::Upper)
        .map_err(|err| FerricError::Lapack(format!("drpa_plasmon eigh: {err}")))?;
    let lmin = lam.iter().copied().fold(f64::INFINITY, f64::min);
    if !(lmin > 0.0) {
        return Err(FerricError::General(format!(
            "drpa_plasmon: dRPA instability (smallest Ω² = {lmin:e})"
        )));
    }
    let sum_omega: f64 = lam.iter().map(|x| x.sqrt()).sum();
    let tr_a: f64 = (0..nov).map(|p| e[p] + 2.0 * ovov[(p, p)]).sum();
    Ok(0.5 * (sum_omega - tr_a))
}

/// Gamma-point closed-shell dRPA from a converged Gamma RHF `rhf` on `cell`.
/// See the module doc for the integral sources and the conventions.
pub fn gamma_drpa(
    cell: &Cell,
    rhf: &ScfResult,
    ints: GammaDrpaIntegrals<'_>,
    cfg: &GammaDrpaConfig,
) -> Result<GammaDrpaResult, FerricError> {
    validate_quad_points(cfg.quad_points)?;
    if !matches!(rhf.spin, Spin::Restricted) {
        return Err(FerricError::General(
            "gamma_drpa: closed-shell RHF reference required (got a non-restricted result)".into(),
        ));
    }
    if !rhf.converged {
        return Err(FerricError::General(
            "gamma_drpa: the RHF reference did not converge".into(),
        ));
    }
    let nelec = cell.mol().nelec();
    if nelec <= 0 || nelec % 2 != 0 {
        return Err(FerricError::General(format!(
            "gamma_drpa: closed shell needs an even, positive electron count (got {nelec})"
        )));
    }
    let nocc_total = (nelec / 2) as usize;
    let nocc = active_occ(nocc_total, cfg.frozen_core)?;
    let first_occ = cfg.frozen_core;
    let c = rhf.mos_r();
    let eps_in = rhf.eps_r();
    let (nao, nmo) = c.dim();
    if eps_in.len() != nmo || nmo <= nocc_total {
        return Err(FerricError::General(format!(
            "gamma_drpa: {nmo} MOs / {} orbital energies with {nocc_total} occupied: no virtuals \
             or inconsistent reference",
            eps_in.len()
        )));
    }
    let nvir = nmo - nocc_total;
    let madelung = madelung_constant(cell)?;
    let occ_shift = occupied_shift(cfg.reference_exxdiv, cfg.denominators, madelung);
    let eps_occ: Vec<f64> = eps_in[first_occ..nocc_total]
        .iter()
        .map(|e| e + occ_shift)
        .collect();
    let eps_vir: Vec<f64> = eps_in[nocc_total..].to_vec();
    let _ = excitation_energies(&eps_occ, &eps_vir)?;

    let c_occ = c.slice(s![.., first_occ..nocc_total]);
    let c_vir = c.slice(s![.., nocc_total..]);
    let nov = nocc * nvir;
    let budget = crate::budget::resolve(cfg.budget_bytes);
    let mut ledger = Ledger::new(budget);

    let (drpa_corr, naux, quad_points, eigenvalues_static) = match ints {
        GammaDrpaIntegrals::RsGdf(gdf) => {
            let b = gdf.b();
            let naux = b.nrows();
            let w = gdf.metric_inv_sqrt();
            if b.ncols() != nao * nao || w.ncols() != naux {
                return Err(FerricError::General(format!(
                    "gamma_drpa: RS-GDF B {:?} / metric map {:?} inconsistent with nao = {nao}",
                    b.dim(),
                    w.dim()
                )));
            }
            let naux_ao = w.nrows();
            let (nx, na) = (naux as u64, naux_ao as u64);
            ledger.reserve(
                &format!("Gamma dRPA half-transformed B[k,μ,a] (naux = {naux}, nao = {nao}, nvir = {nvir})"),
                bytes_of(nx.saturating_mul(nao as u64), nvir.saturating_mul(8)),
            )?;
            ledger.reserve(
                &format!("Gamma dRPA B[k,ia] (naux = {naux}, nocc·nvir = {nov})"),
                bytes_of(nx, nov.saturating_mul(8)),
            )?;
            ledger.reserve(
                &format!("Gamma dRPA metric map copy + eigenpotentials (naux_ao = {naux_ao}, naux = {naux})"),
                bytes_of(na.saturating_mul(nx), 16),
            )?;
            ledger.reserve(
                &format!("Gamma dRPA static dielectric + eigenvectors (naux = {naux})"),
                bytes_of(nx.saturating_mul(nx), 16),
            )?;
            ledger.reserve(
                &format!("Gamma dRPA projection y = Uᵀ B (naux = {naux}, nocc·nvir = {nov})"),
                bytes_of(nx, nov.saturating_mul(8)),
            )?;
            ledger.reserve(
                &format!(
                    "Gamma dRPA per-worker quadrature scratch (naux = {naux}, nocc·nvir = {nov})"
                ),
                bytes_of(nx, nov.saturating_add(naux).saturating_mul(8)),
            )?;
            let b_ov = b_ov_from_ao_b(b, nao, c_occ, c_vir)?;
            let inter = intermediates(b_ov, Some(w.to_owned()), nocc, nvir, nocc_total, first_occ)?;
            let r = drpa_from_intermediates(
                &inter,
                &eps_occ,
                &eps_vir,
                cfg.quad_points,
                // The whole budget, like the MP2 kernel call: ferric-rpa's own
                // ceiling check and panel widths read it (they count per-worker
                // scratch and narrow panels rather than fail).
                Some(budget),
            )?;
            if !r.eigensolver_converged {
                return Err(FerricError::General(
                    "gamma_drpa: static dielectric eigensolve reported non-convergence".into(),
                ));
            }
            (
                r.e_rpa,
                Some(naux),
                Some(cfg.quad_points),
                Some(r.eigenvalues_static),
            )
        }
        GammaDrpaIntegrals::DenseAft(eri) => {
            let n2 = nao * nao;
            ledger.reserve(
                &format!("Gamma dRPA dense W = C_o⊗C_v + I·W (nao = {nao}, nocc·nvir = {nov})"),
                bytes_of(n2 as u64, nov.saturating_mul(16)),
            )?;
            ledger.reserve(
                &format!(
                    "Gamma dRPA dense (ia|jb) + plasmon matrix + eigenvectors (nocc·nvir = {nov})"
                ),
                bytes_of(nov as u64, nov.saturating_mul(24)),
            )?;
            let g = ovov_from_dense(eri.eri(), nao, c_occ, c_vir)?;
            (drpa_plasmon(&g, &eps_occ, &eps_vir)?, None, None, None)
        }
    };
    if !drpa_corr.is_finite() {
        return Err(FerricError::General(format!(
            "gamma_drpa: non-finite correlation energy {drpa_corr}"
        )));
    }
    Ok(GammaDrpaResult {
        drpa_corr,
        total_energy: rhf.energy + drpa_corr,
        madelung,
        occ_shift,
        nocc_active: nocc,
        nvir,
        naux,
        quad_points,
        eigenvalues_static,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quad_points_are_strictly_validated() {
        for ok in [8, 40, 1024] {
            assert!(validate_quad_points(ok).is_ok(), "{ok}");
        }
        for bad in [0, 1, 7, 1025] {
            assert!(validate_quad_points(bad).is_err(), "{bad}");
        }
    }

    /// nov = 1 closed form: E = ½ [√(D(D + 4K)) − D − 2K].
    #[test]
    fn plasmon_matches_the_single_excitation_closed_form() {
        let (ei, ea, k) = (-0.6_f64, 0.7_f64, 0.3_f64);
        let d = ea - ei;
        let want = 0.5 * ((d * (d + 4.0 * k)).sqrt() - d - 2.0 * k);
        let got = drpa_plasmon(&Array2::from_elem((1, 1), k), &[ei], &[ea]).unwrap();
        assert!((got - want).abs() < 1e-15, "{got} vs {want}");
        // A gapless reference is refused, not integrated.
        assert!(drpa_plasmon(&Array2::from_elem((1, 1), k), &[0.1], &[0.1]).is_err());
    }
}
