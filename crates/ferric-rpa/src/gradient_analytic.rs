//! Analytic closed-shell RI-dRPA nuclear gradient on a Hartree-Fock reference.
//!
//! Differentiates the full-rank `n`-point quadrature sum `run_pdep_rpa`
//! evaluates (`book/src/15-correlated-gradients/05-rpa-gradient.md#rpa-gradient`):
//! `E_c = (1/2π) Σ_k w_k [ln det(1 + Π_k) − tr Π_k]`, `Π_k = B g_k(Δ) Bᵀ`,
//! `g_k(e) = 4e / (u_k² + e²)` (`u_k` the imaginary frequency). The frequency loop here builds the three-index
//! density `Γ` and the orbital-energy densities `P_oo`, `P_vv` (Daleckii-Krein
//! divided differences, so the energy is taken in its occupied/virtual
//! rotation-invariant form); the Z-vector, the relaxed-density Lagrangian and
//! the 3-centre/2-centre RI response are the RI-MP2 gradient's
//! ([`ferric_mp2::gradient::relaxed_ri_gradient`]).
//!
//! Scope: closed shell, RHF reference on the reference SCF's J/K route,
//! Coulomb operator, dense full-rank χ₀ (`trunc_thresh = 0`, `Chi0Backend::Dense`,
//! `Chi0Sparsity::Dense`). Frozen core is supported (the core-active block of
//! the RI-MP2 gradient,
//! `book/src/15-correlated-gradients/05-rpa-gradient.md#rpa-gradient-frozen-core`). The
//! finite-difference route
//! ([`crate::gradient`]) stays the oracle and the `[rpa] gradient = "fd"` path.

use ferric_core::basis::BasisSet;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_core::timing::PhaseAccumulator;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::blas_threads::{opt_in_blas_threads, with_blas_threads};
use ferric_integrals::operator::{Operator, OperatorKind};
use ferric_mp2::rimp2::{compute_rpa_intermediates, Mp2Intermediates, RiMp2Config};
use ferric_scf::result::ScfResult;
use ferric_scf::rhf::RhfConfig;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::warm_start::{solve_rhf_warm, ScfWarmStart};
use ndarray::{s, Array1, Array2, ArrayView2};
use ndarray_linalg::{Eigh, UPLO};

use crate::config::Chi0Backend;
use crate::energy::trace_log_summand_term;
use crate::quadrature::build_quadrature;
use crate::{Chi0Sparsity, PdepRpaConfig};

/// Which term of `S(Π) = ln det(1 + Π) − tr Π` is differentiated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RingOrder {
    /// The full dRPA summand.
    Full,
    /// Its second-order term `−½ tr Π²` alone: the quadrature form of the
    /// direct-ring MP2 energy `2 E_OS` (the exactness anchor against the
    /// SCS-MP2 gradient at `(c_os, c_ss) = (2, 0)`).
    SecondOrder,
}

/// First variation of the n-point RI-dRPA correlation energy, in the
/// normalization of [`Mp2Intermediates::from_ri_densities`].
#[derive(Debug, Clone)]
pub struct RpaRiDensities {
    /// `E_c`, the quadrature sum (Hartree).
    pub e_c: f64,
    /// `x[P, ia] = ¼ ∂E_c/∂B^P_ia`, `(naux, nocc·nvir)`.
    pub x_ov: Array2<f64>,
    /// `½ ∂E_c/∂F_ij` over the active occupied block (symmetric).
    pub p_oo: Array2<f64>,
    /// `½ ∂E_c/∂F_ab` (symmetric).
    pub p_vv: Array2<f64>,
}

/// Divided difference of `g(e) = 4e/(u² + e²)` (`u` the imaginary frequency):
/// `(g(x) − g(y))/(x − y) = 4(u² − xy)/((u² + x²)(u² + y²))`, equal to `g'(x)`
/// at `x = y` with no cancellation.
pub fn g_divided_difference(x: f64, y: f64, omega: f64) -> f64 {
    let w2 = omega * omega;
    4.0 * (w2 - x * y) / ((w2 + x * x) * (w2 + y * y))
}

/// `Γ`, `P_oo`, `P_vv` and `E_c` of the quadrature sum over `(freqs, weights)`
/// for `b_ov = B^P_ia` (`(naux, nocc·nvir)`, `ia = i·nvir + a`) and canonical
/// orbital energies (`book/src/15-correlated-gradients/05-rpa-gradient.md#rpa-gradient-densities`).
pub fn rpa_ri_densities(
    b_ov: &Array2<f64>,
    eps_occ: &[f64],
    eps_vir: &[f64],
    freqs: &[f64],
    weights: &[f64],
    order: RingOrder,
) -> Result<RpaRiDensities, FerricError> {
    let (no, nv) = (eps_occ.len(), eps_vir.len());
    let naux = b_ov.nrows();
    if b_ov.ncols() != no * nv || freqs.len() != weights.len() {
        return Err(FerricError::General(format!(
            "rpa_ri_densities: b_ov is {:?} for nocc {no} x nvir {nv}, {} nodes vs {} weights",
            b_ov.dim(),
            freqs.len(),
            weights.len()
        )));
    }
    let gaps = ov_gaps(eps_occ, eps_vir);
    let b_by_vir = occ_inner_layout(b_ov.view(), no, nv);
    let mut out = RpaRiDensities {
        e_c: 0.0,
        x_ov: Array2::zeros((naux, no * nv)),
        p_oo: Array2::zeros((no, no)),
        p_vv: Array2::zeros((nv, nv)),
    };
    for (&omega, &wk) in freqs.iter().zip(weights) {
        let g = gaps.mapv(|e| 4.0 * e / (omega * omega + e * e));
        let (s_k, q_b) = frequency_response(b_ov, &g, order)?;
        out.e_c += wk * s_k / (2.0 * std::f64::consts::PI);
        // x = Γ/4, Γ^P_ia = (1/π) Σ_k w_k g_ia (Q B)^P_ia.
        let xs = wk / (4.0 * std::f64::consts::PI);
        out.x_ov.zip_mut_with(&(&q_b * &g), |x, v| *x += xs * v);
        // P/2 with P_ab = (1/2π) w Σ_i M_ia,ib g1, P_ij = −(1/2π) w Σ_a M_ia,ja g1.
        let ps = wk / (4.0 * std::f64::consts::PI);
        accumulate_p_vv(
            &mut out.p_vv,
            b_ov.view(),
            q_b.view(),
            &gaps,
            omega,
            ps,
            no,
            nv,
        );
        let qb_by_vir = occ_inner_layout(q_b.view(), no, nv);
        accumulate_p_oo(
            &mut out.p_oo,
            &b_by_vir,
            &qb_by_vir,
            &gaps,
            omega,
            -ps,
            no,
            nv,
        );
    }
    Ok(out)
}

/// `e_ia = ε_a − ε_i`, flat `i·nvir + a`.
fn ov_gaps(eps_occ: &[f64], eps_vir: &[f64]) -> Array1<f64> {
    eps_occ
        .iter()
        .flat_map(|&ei| eps_vir.iter().map(move |&ea| ea - ei))
        .collect()
}

/// `(naux, nocc·nvir)` → `(naux, nvir·nocc)` with the occupied index inner,
/// so the columns of one virtual are contiguous.
fn occ_inner_layout(m: ArrayView2<f64>, no: usize, nv: usize) -> Array2<f64> {
    let naux = m.nrows();
    let mut out = Array2::zeros((naux, nv * no));
    for i in 0..no {
        for a in 0..nv {
            out.column_mut(a * no + i).assign(&m.column(i * nv + a));
        }
    }
    out
}

/// One frequency: `S = ln det(1 + Π) − tr Π` (or `−½ tr Π²`) and `Q·B` with
/// `Q = (1 + Π)⁻¹ − 1` (or `−Π`), `Π = B diag(g) Bᵀ`. `1 + Π` is diagonalized
/// so `S` is the eigenvalue sum of `energy::trace_log_summand_term`, the
/// summand of the energy route.
fn frequency_response(
    b_ov: &Array2<f64>,
    g: &Array1<f64>,
    order: RingOrder,
) -> Result<(f64, Array2<f64>), FerricError> {
    let pi = with_blas_threads(opt_in_blas_threads(), || (b_ov * g).dot(&b_ov.t()));
    let (s, q) = match order {
        RingOrder::SecondOrder => {
            let s = -0.5 * pi.iter().map(|v| v * v).sum::<f64>();
            (s, -pi)
        }
        RingOrder::Full => {
            let (lam, u) = pi.eigh(UPLO::Lower).map_err(|e| {
                FerricError::General(format!("dRPA gradient: eigh of Π failed: {e}"))
            })?;
            if let Some(&bad) = lam.iter().find(|&&l| l.is_nan() || l <= -1.0) {
                return Err(FerricError::General(format!(
                    "dRPA gradient: 1 + Π has eigenvalue {} <= 0",
                    1.0 + bad
                )));
            }
            let s = lam.iter().map(|&l| trace_log_summand_term(1.0 + l)).sum();
            let d = lam.mapv(|l| -l / (1.0 + l));
            let q = with_blas_threads(opt_in_blas_threads(), || (&u * &d).dot(&u.t()));
            (s, q)
        }
    };
    let qb = with_blas_threads(opt_in_blas_threads(), || q.dot(b_ov));
    Ok((s, qb))
}

/// `p_vv[a, b] += scale · Σ_i (B_iᵀ (QB)_i)[a, b] · g1(e_ia, e_ib)`.
#[allow(clippy::too_many_arguments)]
fn accumulate_p_vv(
    p_vv: &mut Array2<f64>,
    b_ov: ArrayView2<f64>,
    qb: ArrayView2<f64>,
    gaps: &Array1<f64>,
    omega: f64,
    scale: f64,
    no: usize,
    nv: usize,
) {
    for i in 0..no {
        let cols = s![.., i * nv..(i + 1) * nv];
        let m = b_ov.slice(cols).t().dot(&qb.slice(cols));
        let e = gaps.slice(s![i * nv..(i + 1) * nv]);
        for ((a, b), p) in p_vv.indexed_iter_mut() {
            *p += scale * m[(a, b)] * g_divided_difference(e[a], e[b], omega);
        }
    }
}

/// `p_oo[i, j] += scale · Σ_a (B_aᵀ (QB)_a)[i, j] · g1(e_ia, e_ja)`, on the
/// occupied-inner layouts.
#[allow(clippy::too_many_arguments)]
fn accumulate_p_oo(
    p_oo: &mut Array2<f64>,
    b_by_vir: &Array2<f64>,
    qb_by_vir: &Array2<f64>,
    gaps: &Array1<f64>,
    omega: f64,
    scale: f64,
    no: usize,
    nv: usize,
) {
    for a in 0..nv {
        let cols = s![.., a * no..(a + 1) * no];
        let m = b_by_vir.slice(cols).t().dot(&qb_by_vir.slice(cols));
        for ((i, j), p) in p_oo.indexed_iter_mut() {
            let (ei, ej) = (gaps[i * nv + a], gaps[j * nv + a]);
            *p += scale * m[(i, j)] * g_divided_difference(ei, ej, omega);
        }
    }
}

/// Energy and analytic gradient of `E_RHF + E_c^RPA`.
#[derive(Debug, Clone)]
pub struct RpaAnalyticGradient {
    /// `E_RHF + E_c` at the reference geometry (Hartree).
    pub energy: f64,
    /// `E_c`, the full-rank n-point quadrature sum.
    pub e_c: f64,
    /// `dE/dR`, `(natoms, 3)`, Hartree/Bohr.
    pub gradient: Array2<f64>,
}

/// Refuse every configuration whose energy is not the full-rank dense
/// RI-dRPA sum this gradient differentiates.
pub fn check_analytic_scope(cfg: &PdepRpaConfig, op: Operator) -> Result<(), FerricError> {
    let refuse = |why: &str| {
        Err(FerricError::General(format!(
            "analytic RPA gradient: {why}; use the finite-difference gradient ([rpa] gradient = \"fd\")"
        )))
    };
    if cfg.trunc_thresh != 0.0 {
        return refuse(&format!(
            "it differentiates the full-rank energy, but trunc_thresh = {} truncates the PDEP basis \
             (set trunc_thresh = 0)",
            cfg.trunc_thresh
        ));
    }
    if !matches!(cfg.chi0_backend, Chi0Backend::Dense) {
        return refuse("only the dense χ₀ backend is differentiated");
    }
    if !matches!(cfg.chi0_sparsity, Chi0Sparsity::Dense) {
        return refuse("only dense χ₀ sparsity is differentiated");
    }
    if !matches!(op.kind, OperatorKind::Coulomb) {
        return refuse("only the Coulomb operator is differentiated");
    }
    Ok(())
}

/// [`rpa_analytic_gradient`] with the reference RHF solved through `warm`
/// (warm-started from the carrier's previous geometry).
pub fn rpa_analytic_gradient_warm(
    mol: &Molecule,
    obs_basis: &BasisSet,
    aux_basis: &BasisSet,
    op: Operator,
    cfg: &PdepRpaConfig,
    warm: &mut ScfWarmStart,
) -> Result<RpaAnalyticGradient, FerricError> {
    check_analytic_scope(cfg, op)?;
    let ctx = ParallelContext::default();
    let obs = PreparedBasis::new(mol, obs_basis)?;
    let dfbs = PreparedBasis::new(mol, aux_basis)?;
    let bounds = SchwarzBounds::compute(op, &obs)?;
    let rhf_cfg = RhfConfig {
        density_conv: 1e-9,
        ..Default::default()
    };
    let rhf = solve_rhf_warm(&ctx, mol, &obs, op, &bounds, &rhf_cfg, warm)?;
    if !rhf.converged {
        return Err(FerricError::ScfConvergence {
            iterations: rhf.iterations,
            last_energy: rhf.energy,
        });
    }
    rpa_analytic_gradient_on(mol, &obs, &dfbs, op, &bounds, &rhf, cfg, RingOrder::Full)
}

/// Energy and analytic gradient of `E_RHF + E_c^RPA` with a fresh reference
/// RHF (`density_conv` 1e-9, the finite-difference route's setting).
pub fn rpa_analytic_gradient(
    mol: &Molecule,
    obs_basis: &BasisSet,
    aux_basis: &BasisSet,
    op: Operator,
    cfg: &PdepRpaConfig,
) -> Result<RpaAnalyticGradient, FerricError> {
    rpa_analytic_gradient_warm(
        mol,
        obs_basis,
        aux_basis,
        op,
        cfg,
        &mut ScfWarmStart::new(false),
    )
}

/// The gradient on a converged reference `rhf` (exact or RI-JK route, as
/// solved). `order` selects the full dRPA summand or its second-order term.
#[allow(clippy::too_many_arguments)]
pub fn rpa_analytic_gradient_on(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    rhf: &ScfResult,
    cfg: &PdepRpaConfig,
    order: RingOrder,
) -> Result<RpaAnalyticGradient, FerricError> {
    check_analytic_scope(cfg, op)?;
    let mut acc = PhaseAccumulator::new().report_on_drop("RPA analytic gradient");
    let mp2_cfg = RiMp2Config {
        frozen_core: cfg.frozen_core,
        memory_budget_bytes: cfg.memory_budget_bytes,
        ..Default::default()
    };
    let ri = acc.time("intermediates (b_ov)", || {
        compute_rpa_intermediates(mol, obs, dfbs, op, rhf, &mp2_cfg)
    })?;
    let _charge = ferric_core::memory::pool::reserve_global(
        "RPA analytic gradient: x, QB, BΓ scratch (3 naux·nov) + Π, eigenvectors, Q (3 naux²)",
        3 * (ri.naux * ri.nocc * ri.nvir + ri.naux * ri.naux) * 8,
    )?;
    let eps = rhf.eps_r();
    let (first, nocc_total) = (ri.first_occ, ri.nocc_total);
    let (freqs, weights) = build_quadrature(&cfg.quadrature);
    let dens = acc.time("frequency loop (Γ, P_oo, P_vv)", || {
        rpa_ri_densities(
            &ri.b_ov,
            &eps[first..nocc_total],
            &eps[nocc_total..],
            &freqs,
            &weights,
            order,
        )
    })?;
    let e_c = dens.e_c;
    let inter = Mp2Intermediates::from_ri_densities(ri, dens.x_ov, dens.p_oo, dens.p_vv, e_c)?;
    let budget = ferric_core::memory::resolve_budget_bytes(cfg.memory_budget_bytes);
    let gradient = ferric_mp2::gradient::relaxed_ri_gradient(
        mol, obs, dfbs, op, bounds, rhf, &inter, budget, &mut acc,
    )?;
    Ok(RpaAnalyticGradient {
        energy: rhf.energy + e_c,
        e_c,
        gradient,
    })
}
