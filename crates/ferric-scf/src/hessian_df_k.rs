//! Analytic RHF Hessian pieces for density-fitted EXCHANGE (RI-K).
//!
//! The energy this differentiates is the closed-shell RI-K energy of
//! [`crate::df_k::DfK`] / [`crate::df_gradient`]:
//!
//! ```text
//!   E_K = α Σ_ij L_ij · M L_ij ,   α = −¼,   D = X Xᵀ (X = √2 C_occ, m columns)
//!   L_{P,ij} = Σ_μν X_μi (P|μν) X_νj ,   M = V^{-1/2} V^{-1/2},   V_PQ = (P|Q)
//! ```
//!
//! which equals `−¼ Σ D_μλ D_νσ (μν|λσ)_fit` with
//! `(μν|λσ)_fit = (μν|P) M_PQ (Q|λσ)`. The exchange Fock matrix is
//! `F_K = ∂E_K/∂D = 2α K[D]`, `K[D]_μλ = Σ_νσ D_νσ (μν|λσ)_fit` (`−½K`).
//!
//! # Scope: no dropped metric modes
//!
//! `DfK` drops metric modes below `DFK_LINDEP_THRESH`. With none dropped,
//! `M = V⁻¹` exactly (the product of the two `V^{-1/2}` factors), so every
//! derivative below follows from `∂M = −M V_x M`. A metric with a dropped mode
//! is refused with a typed error naming the count (the energy is then a
//! spectral function of `V` and its second derivative needs the
//! Daleckii–Krein formula to second order, which is not implemented). Modes in
//! `[1e-10, 1e-8)` are kept and warned about.
//!
//! # Derivation (fixed `D`, i.e. fixed `X`; derivatives of integrals only)
//!
//! Index `ij` is carried along; `⟨a, b⟩ = Σ_{P,ij} a_{P,ij} b_{P,ij}`.
//! `Γ = M L` (`naux × m²`), `L_x = ∂L/∂x` through the integrals (the orbital
//! functions AND the auxiliary function `P` move with their atoms).
//!
//! ```text
//!   E_x  = α [ 2⟨L_x, Γ⟩ − ⟨Γ, V_x Γ⟩ ]
//!   ∂_y Γ = M Z_y ,         Z_y := L_y − V_y Γ
//!   E_xy = α [ 2⟨L_xy, Γ⟩ − ⟨Γ, V_xy Γ⟩ + 2⟨Z_x, M Z_y⟩ ]
//! ```
//!
//! (`∂_y` of `2⟨L_x,Γ⟩ = 2⟨L_xy,Γ⟩ + 2⟨L_x, M Z_y⟩` and of
//! `−⟨Γ, V_x Γ⟩ = −⟨Γ, V_xy Γ⟩ − 2⟨M Z_y, V_x Γ⟩`; the two cross pieces
//! combine to `2⟨(L_x − V_x Γ), M Z_y⟩`. This is the RI-J result
//! `d_xyᵀc − ½cᵀV_xy c + Z_xᵀ M Z_y` with `d → L` and `½ → α`.) The cross term
//! is symmetric in `x ↔ y`.
//!
//! The second derivatives enter only through two contractions, never as
//! tensors, both from [`crate::hessian_df_stencil`]:
//!
//! * `2α⟨L_xy, Γ⟩ = Σ_{P,μν} W_{Pμν} ∂²(P|μν)` with the per-auxiliary-function
//!   weight `W_P = 2α X Γ_P Xᵀ` (`Γ_P` is the symmetric `m × m` matrix of row
//!   `P`; `L` is symmetric in `ij`, hence so is `Γ`, which lets the swapped
//!   order of an off-diagonal shell pair carry the factor 2);
//! * `−α⟨Γ, V_xy Γ⟩ = Σ_PQ (−α Γ Γᵀ)_PQ ∂²(P|Q)`.
//!
//! ## First-order Fock matrix `K^x` (AO, fixed `D`)
//!
//! With the fit coefficients `C_{P,μσ} = M_PQ (Q|μσ)`,
//!
//! ```text
//!   K^x_μλ = Σ_{P,ν} ∂_x(P|μν) T_{P,νλ} + (μ↔λ) − Σ_PQ (V_x)_PQ Σ_j Ĉ_{P,μj} Ĉ_{Q,λj}
//!   Ĉ_{P,λj} = Σ_σ C_{P,λσ} X_σj = [M B]_{P,λj},  B_{P,λj} = Σ_σ (P|λσ) X_σj
//!   T_{P,νλ} = Σ_j X_νj Ĉ_{P,λj} = Σ_σ D_νσ C_{P,λσ}
//! ```
//!
//! (differentiate `(μν|P) M (Q|λσ) D_νσ`: the two integral derivatives give the
//! first two terms, `∂M = −M V_x M` the third.) The Fock contribution is
//! `2α K^x`. The matrix-valued piece the exact path does not have is the third
//! term (the aux-centre / metric response of the fit in the orbital basis); the
//! first piece contains the auxiliary-centre derivative of `(P|μν)` too.
//!
//! ## CPHF
//!
//! `G(D1) = J(D1) − ½K(D1)` is linear in `D1`; the response uses the SCF's own
//! `DfK` (the dressed `V^{-1/2}(Q|μν)` source) through its density path
//! `K = Σ_P B_P D1 B_P`. That path is written for a general matrix: it never
//! factors `D1`, so the non-positive-semidefinite response density
//! `D1 = C_v U C_oᵀ + h.c.` and the occupied-block density are handled with no
//! change (the occupied-orbital path `build_from_occ` is NOT used, it needs
//! `D = C Cᵀ`).
//!
//! ## Factorization
//!
//! `M` is built as `W·W` with `W = U Λ^{-1/2} Uᵀ` from the same `eigh` as
//! `DfK::v_inv_sqrt_lindep`, i.e. the SCF's own construction (no explicit
//! `V⁻¹` from LU). With nothing dropped it agrees with `V⁻¹` to `~ε·cond(V)`.
//!
//! ## What the tests establish (and what they cannot see)
//!
//! * No exactly-spanning auxiliary basis exists for exchange: the pair
//!   densities `φ_i φ_j` of orbitals on different atoms are two-centre
//!   products, and with both on one atom every integral is one-centre (zero
//!   Hessian), so the RI-J exact-span anchor has no K analogue.
//! * For ONE occupied column `E_K = 2α E_J^DF`: every skeleton piece equals
//!   `2α` times the RI-J piece through an independent code path (blind to the
//!   pair index `ij`, i.e. to `m > 1`).
//! * `K^x` against the SCF's own `DfK` differenced; the skeleton pieces and the
//!   whole skeleton against differences of the first-derivative pieces and of
//!   `df_jk_gradient` at fixed density; the total against the differenced DF
//!   gradient with the SCF re-converged; PySCF's analytic DF Hessian.
//!
//! ## Memory and cost
//!
//! With `A = naux`, `n = nbf`, `m` = occupied count, `3N` coordinates, the
//! resident arrays are `B, Ĉ` (`2·A·n·m`), `L, Γ` (`2·A·m²`), `L_x → Z_x` and
//! `∂_xΓ` (`2·3N·A·m²`), the dense `V_x` (`3N·A²`), plus the `K^x` matrices
//! (`3·3N·n²`) and `chunk` per-aux-shell partials of one `3N·n²` matrix. All of
//! it is charged to the global pool up front (`rik_peak_bytes`); a typed error
//! when it does not fit. Peak: 3.4 MB for water cc-pVDZ / def2-universal-jkfit
//! (3 atoms, `A = 113`, `m = 5`); 10.5 GB for 30 atoms with 300 basis
//! functions, `m = 60`, `A = 1500` (of which `L_x` and `∂_xΓ` are 3.9 GB each
//! and `V_x` 1.6 GB). Everything is built once per Hessian, not per CPHF
//! iteration; the integral work is one streamed `(P|μν)` pass, one
//! first-derivative 3c pass (`L_x` and `K1` together), one 2c pass, and the two
//! stencil contractions (about 30 times the engine calls of a gradient pass).
//! The `K^x` metric term costs `3N · (A²·n·m + A·m·n²)` flops (the sparsity of
//! `V_x` in the moved atom is not exploited).
//!
//! ## Conditioning
//!
//! The Cholesky-free `M` amplifies integral noise by `cond(V)`. Measured (water
//! 6-31G, Hessian vs the Richardson finite difference of ferric's own DF-K
//! gradient, fitting bases with no metric mode in the gradient's noise band):
//! cond 2.5e5 (cc-pVDZ-RI) 1.4e-10, 2.4e6 (aug-cc-pVDZ-RIFIT) 1.6e-10, 9.1e6
//! (aug-cc-pVTZ-RIFIT) 2.5e-10, 2.5e7 (def2-universal-jkfit) 1.3e-10, 2.4e7
//! (def2-QZVPP-RIFIT) 4.4e-10, **2.0e8 (aug-cc-pVQZ-RIFIT) 6.7e-9**: the error
//! begins to grow near cond 1e8, hence the warning above `RIK_COND_WARN`. A
//! metric with a mode below the DF-K cut is refused (see Scope). The
//! reference gradient itself treats kept modes in `[1e-10, 1e-7)` as dropped
//! (its noise floor), so a basis with such a mode has no valid finite-difference
//! reference and yields no accuracy datum.

use crate::df_gradient::aux_basis;
use crate::hessian_df::{check_metric_conditioning_for, chunk_len, MetricUse};
use crate::hessian_df_stencil::{
    eri2_deriv2_contract, eri3_deriv2_contract, weight_matrix, StencilConfig,
};
use crate::result::DfJkRoute;
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::operator::Operator;
use ferric_integrals::three_index_source::ThreeIndexSource;
use ferric_integrals::threeindex::coulomb_metric_2c;
use ndarray::{s, Array1, Array2, Array3, Axis};
use ndarray_linalg::{Eigh, Inverse, UPLO};
use rayon::prelude::*;

/// Condition number of the exchange metric above which the Hessian warns: the
/// error against the finite difference of the DF-K gradient (water 6-31G) is
/// 1.3e-10 .. 4.4e-10 for cond 2.5e5 .. 2.5e7 and 6.7e-9 at cond 2e8.
pub const RIK_COND_WARN: f64 = 1e8;

/// Closed-shell exchange weight `α` of `E_K = α Σ_ij L_ij · M L_ij`.
pub const RIK_ALPHA: f64 = -0.25;

/// Deliberately broken constructions, wired through the production code path
/// for the mutation tests of `tests/df_hessian_rik.rs`. `Default` is the
/// production construction. NOT public API.
#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
pub struct RikMutants {
    /// Drop `−α⟨Γ, V_xy Γ⟩` (the 2c second derivative).
    pub drop_metric_second: bool,
    /// Drop the cross terms of the skeleton (`Z_x := L_x`).
    pub drop_cross: bool,
    /// Drop the whole metric-response term of `K^x` (`−V_x Ĉ Ĉ`).
    pub drop_kx_metric: bool,
    /// Drop `K^x` altogether (the fixed-density exchange Fock derivative).
    pub drop_kx: bool,
    /// Omit the auxiliary-centre derivative of `(P|μν)` in `K^x`.
    pub omit_aux_centre_kx: bool,
    /// Use the transposed pair `T_{P,λν}` in `K^x` (the `(ij)` pair swapped).
    pub transpose_pair_kx: bool,
    /// Use an erfc(0.4) metric for `V`, `V_x`, `V_xy`, `M`, with Coulomb 3c.
    pub swap_metric: bool,
    /// Apply an explicit `V⁻¹` (LU) where the SCF used `V^{-1/2}V^{-1/2}`.
    pub explicit_inverse: bool,
    /// Multiplier on the exchange weight `α` everywhere (1 = production).
    pub alpha_factor: f64,
    /// Stencil construction (step, order, shell displacement).
    pub stencil: StencilConfig,
    /// Skip the conditioning refusal (measurement only).
    pub allow_ill_conditioned: bool,
}

impl Default for RikMutants {
    fn default() -> Self {
        RikMutants {
            drop_metric_second: false,
            drop_cross: false,
            drop_kx_metric: false,
            drop_kx: false,
            omit_aux_centre_kx: false,
            transpose_pair_kx: false,
            swap_metric: false,
            explicit_inverse: false,
            alpha_factor: 1.0,
            stencil: StencilConfig::default(),
            allow_ill_conditioned: false,
        }
    }
}

/// The RI-K pieces of the Hessian.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct RikTerms {
    /// `∂²E_K^DF/∂x∂y` at fixed `D`, `3N × 3N` (symmetrized).
    pub skeleton: Array2<f64>,
    /// `max |S − Sᵀ|` of the skeleton before symmetrization.
    pub skeleton_asymmetry: f64,
    /// `2α⟨L_xy, Γ⟩`.
    pub three_centre: Array2<f64>,
    /// `−α⟨Γ, V_xy Γ⟩`.
    pub metric_second: Array2<f64>,
    /// `2α⟨Z_x, M Z_y⟩`.
    pub cross: Array2<f64>,
    /// `∂K[D]/∂x` (unscaled), one `n × n` AO matrix per coordinate.
    pub first_order_k: Vec<Array2<f64>>,
    /// `2α ∂K[D]/∂x`: what is ADDED to the Fock derivative (`−½K^x`).
    pub first_order_fock: Vec<Array2<f64>>,
}

/// First-derivative quantities of the fit for the piece tests.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct RikFirst {
    /// `Γ = M L` at this geometry, `naux × m²`.
    pub gamma: Array2<f64>,
    /// `⟨Γ*, L_x⟩` for each `x` (`Γ*` = own `Γ` or the fixed one).
    pub lg: Array1<f64>,
    /// `⟨Γ*, V_x Γ*⟩` for each `x`.
    pub gvg: Array1<f64>,
    /// `Z_x` for the own `Γ`, `3N × (naux·m²)`.
    pub z: Array2<f64>,
}

/// Metric, `M` and the operators of one exchange fit.
struct Fit {
    dfbs: PreparedBasis,
    m: Array2<f64>,
    op: Operator,
    metric_op: Operator,
    budget: usize,
}

impl Fit {
    fn new(
        dfbs: PreparedBasis,
        op: Operator,
        budget: usize,
        mutants: &RikMutants,
    ) -> Result<Self, FerricError> {
        let metric_op = if mutants.swap_metric {
            Operator::erfc(0.4)
        } else {
            op
        };
        let v = coulomb_metric_2c(metric_op, &dfbs)?;
        if !mutants.allow_ill_conditioned {
            check_metric_conditioning_for(&v, MetricUse::ExchangeFit)?;
        }
        let m = if mutants.explicit_inverse {
            v.inv()
                .map_err(|e| FerricError::Lapack(format!("analytic RI-K Hessian: inverse: {e}")))?
        } else {
            // The SCF's own construction (DfK::v_inv_sqrt_lindep) with nothing
            // dropped: W = U Λ^{-1/2} Uᵀ, M = W W.
            let (lam, u) = v
                .eigh(UPLO::Upper)
                .map_err(|e| FerricError::Lapack(format!("analytic RI-K Hessian: eigh: {e}")))?;
            let mut us = u.clone();
            for k in 0..lam.len() {
                let sc = 1.0 / lam[k].sqrt();
                us.column_mut(k).mapv_inplace(|x| x * sc);
            }
            let w = us.dot(&u.t());
            w.dot(&w)
        };
        Ok(Fit {
            dfbs,
            m: m.as_standard_layout().to_owned(),
            op,
            metric_op,
            budget,
        })
    }
}

/// Bytes the Hessian pieces keep resident beyond the budget-bounded raw
/// tensor (see the module doc); charged to the pool before allocation.
#[doc(hidden)]
pub fn rik_peak_bytes(natoms: usize, nbf: usize, nocc: usize, naux: usize) -> usize {
    let n3 = 3 * natoms;
    let m2 = nocc * nocc;
    let doubles = 2 * naux * nbf * nocc
        + 3 * naux * m2
        + 2 * n3 * naux * m2
        + n3 * naux * naux
        + 3 * naux * naux
        + (3 + chunk_len(n3, nbf)) * n3 * nbf * nbf;
    8usize.saturating_mul(doubles)
}

/// The RI-K Hessian pieces at the occupied factor `x` (`D = X Xᵀ`).
#[doc(hidden)]
pub fn rik_terms(
    mol: &Molecule,
    obs: &PreparedBasis,
    route: &DfJkRoute,
    x: &Array2<f64>,
    mutants: &RikMutants,
) -> Result<RikTerms, FerricError> {
    let name = route.k_aux.as_deref().ok_or_else(|| {
        FerricError::General("analytic RI-K Hessian: the SCF route has no RI-K basis".into())
    })?;
    rik_terms_with_aux(
        mol.atoms.len(),
        obs,
        aux_basis(mol, name)?,
        route.op,
        route.budget_bytes,
        x,
        mutants,
    )
}

/// Quantities shared by the production terms and the piece tests.
struct Core {
    fit: Fit,
    gamma: Array2<f64>,
    /// `L_x`, `3N × (A·m²)`.
    lx: Array2<f64>,
    /// `V_x`, `3N × A × A`.
    vx: Array3<f64>,
    /// `Ĉ` as `(A, m, n)`, the layout `K^x`'s metric term contracts.
    chat: Array3<f64>,
    /// `K1_x = Σ ∂_x(P|μν) T_{P,νλ}`, `3N × n²` (not symmetrized).
    k1: Array2<f64>,
}

#[allow(clippy::too_many_arguments)]
fn build_core(
    natoms: usize,
    obs: &PreparedBasis,
    fit: Fit,
    x: &Array2<f64>,
    mutants: &RikMutants,
    with_k1: bool,
) -> Result<Core, FerricError> {
    let (n, naux, m) = (obs.nbasis(), fit.dfbs.nbasis(), x.ncols());
    let m2 = m * m;

    // Pass 1: L_P = Xᵀ (P|··) X and B_{P,λj} = Σ_σ (P|λσ) X_σj.
    let mut source = ThreeIndexSource::build(fit.op, obs, &fit.dfbs, fit.budget)?;
    let mut l = Array2::<f64>::zeros((naux, m2));
    let mut b = Array3::<f64>::zeros((naux, n, m));
    source.for_each_block(|blk| {
        let nb = blk.data.shape()[0];
        let rows: Vec<(Array2<f64>, Array2<f64>)> = (0..nb)
            .into_par_iter()
            .map(|q| {
                let t = blk.data.index_axis(Axis(0), q).dot(x);
                let lq = x.t().dot(&t);
                (t, lq)
            })
            .collect();
        for (q, (t, lq)) in rows.into_iter().enumerate() {
            b.index_axis_mut(Axis(0), blk.p0 + q).assign(&t);
            for (dst, src) in l.row_mut(blk.p0 + q).iter_mut().zip(lq.iter()) {
                *dst = *src;
            }
        }
        Ok(())
    })?;
    drop(source);

    let gamma = fit.m.dot(&l).as_standard_layout().to_owned();
    drop(l);
    let b_flat = b
        .into_shape_with_order((naux, n * m))
        .map_err(|e| FerricError::General(format!("B reshape: {e}")))?;
    let chat_flat = fit.m.dot(&b_flat).as_standard_layout().to_owned();
    drop(b_flat);
    let chat = chat_flat
        .into_shape_with_order((naux, n, m))
        .map_err(|e| FerricError::General(format!("Chat reshape: {e}")))?;

    // (A, n, m) -> (A, m, n)
    let chat_p = chat
        .permuted_axes([0, 2, 1])
        .as_standard_layout()
        .to_owned();

    // Pass 2: first-derivative 3c blocks: L_x always, K1_x on request.
    let (lx, k1) = first_derivative_3c(obs, &fit, natoms, x, &chat_p, mutants, with_k1)?;
    let vx = first_derivative_2c(&fit, natoms)?;
    Ok(Core {
        fit,
        gamma,
        lx,
        vx,
        chat: chat_p,
        k1,
    })
}

/// [`rik_terms`] with an explicit auxiliary basis (the exactness anchors hand
/// it a basis that spans the pair densities exactly).
#[doc(hidden)]
pub fn rik_terms_with_aux(
    natoms: usize,
    obs: &PreparedBasis,
    dfbs: PreparedBasis,
    op: Operator,
    budget: usize,
    x: &Array2<f64>,
    mutants: &RikMutants,
) -> Result<RikTerms, FerricError> {
    let fit = Fit::new(dfbs, op, budget, mutants)?;
    let (n, naux, m) = (obs.nbasis(), fit.dfbs.nbasis(), x.ncols());
    let n3 = 3 * natoms;
    let m2 = m * m;
    let alpha = RIK_ALPHA * mutants.alpha_factor;
    let _charge = ferric_core::memory::pool::reserve_global(
        "RI-K Hessian fit tensors, L^x/Z^x, V^x and K^x matrices",
        rik_peak_bytes(natoms, n, m, naux),
    )?;

    let mut core = build_core(natoms, obs, fit, x, mutants, true)?;
    let gamma = core.gamma.clone();

    // Z_x = L_x - V_x Γ (in place), then ∂_xΓ = M Z_x.
    if !mutants.drop_cross {
        for xi in 0..n3 {
            let vg = core.vx.index_axis(Axis(0), xi).dot(&gamma.view());
            let mut row = core.lx.row_mut(xi);
            let flat = row.as_slice_mut().expect("contiguous");
            for (dst, src) in flat.iter_mut().zip(vg.iter()) {
                *dst -= *src;
            }
        }
    }
    let z = core.lx;
    let mut gamma_x = Array2::<f64>::zeros((n3, naux * m2));
    for xi in 0..n3 {
        let zx = z
            .row(xi)
            .to_owned()
            .into_shape_with_order((naux, m2))
            .map_err(|e| FerricError::General(format!("Z reshape: {e}")))?;
        let gx = core.fit.m.dot(&zx);
        for (dst, src) in gamma_x.row_mut(xi).iter_mut().zip(gx.iter()) {
            *dst = *src;
        }
    }
    let cross = (z.dot(&gamma_x.t())) * (2.0 * alpha);
    drop(gamma_x);
    drop(z);

    // Skeleton second derivatives through the stencil.
    let cfg = &mutants.stencil;
    let fit = &core.fit;
    let gam_ref = &gamma;
    let three_centre = eri3_deriv2_contract(obs, &fit.dfbs, fit.op, cfg, |sp, s1, s2, w| {
        weight_gamma(obs, &fit.dfbs, gam_ref, x, 2.0 * alpha, (sp, s1, s2), w)
    })?;
    let metric_second = if mutants.drop_metric_second {
        Array2::zeros((n3, n3))
    } else {
        let m2w = gamma.dot(&gamma.t()) * (-alpha);
        eri2_deriv2_contract(&fit.dfbs, fit.metric_op, cfg, |sp, sq, w| {
            weight_matrix(&fit.dfbs, &m2w, (sp, sq), w)
        })?
    };
    let raw = &three_centre + &metric_second + &cross;
    let skeleton_asymmetry = raw
        .iter()
        .zip(raw.t().iter())
        .fold(0.0f64, |mx, (a, b)| mx.max((a - b).abs()));
    let skeleton = (&raw + &raw.t()) * 0.5;

    // K^x = K1 + K1ᵀ - Σ_j Ĉ_jᵀ V_x Ĉ_j
    let chat_flat = core
        .chat
        .view()
        .into_shape_with_order((naux, m * n))
        .map_err(|e| FerricError::General(format!("Chat flat: {e}")))?
        .to_owned();
    let chat_rows = core
        .chat
        .view()
        .into_shape_with_order((naux * m, n))
        .map_err(|e| FerricError::General(format!("Chat rows: {e}")))?
        .to_owned();
    let mut first_order_k = Vec::with_capacity(n3);
    let mut first_order_fock = Vec::with_capacity(n3);
    for xi in 0..n3 {
        let k1 = core
            .k1
            .row(xi)
            .to_owned()
            .into_shape_with_order((n, n))
            .map_err(|e| FerricError::General(format!("K1 reshape: {e}")))?;
        let mut kx = &k1 + &k1.t();
        if !mutants.drop_kx_metric {
            let w = core
                .vx
                .index_axis(Axis(0), xi)
                .dot(&chat_flat)
                .as_standard_layout()
                .to_owned();
            let w_rows = w
                .into_shape_with_order((naux * m, n))
                .map_err(|e| FerricError::General(format!("W reshape: {e}")))?;
            kx -= &chat_rows.t().dot(&w_rows);
        }
        let scale = if mutants.drop_kx { 0.0 } else { 2.0 * alpha };
        first_order_fock.push(&kx * scale);
        first_order_k.push(kx);
    }
    Ok(RikTerms {
        skeleton,
        skeleton_asymmetry,
        three_centre,
        metric_second,
        cross,
        first_order_k,
        first_order_fock,
    })
}

/// First-derivative pieces at the occupied factor `x` for the per-term finite
/// difference tests: `⟨Γ*, L_x⟩`, `⟨Γ*, V_x Γ*⟩` (with `Γ*` the fixed
/// `gamma_fixed` when given, else this geometry's own) and `Z_x`.
#[doc(hidden)]
pub fn rik_first(
    mol: &Molecule,
    obs: &PreparedBasis,
    route: &DfJkRoute,
    x: &Array2<f64>,
    gamma_fixed: Option<&Array2<f64>>,
) -> Result<RikFirst, FerricError> {
    let mutants = RikMutants::default();
    let name = route.k_aux.as_deref().ok_or_else(|| {
        FerricError::General("analytic RI-K Hessian: the SCF route has no RI-K basis".into())
    })?;
    let natoms = mol.atoms.len();
    let fit = Fit::new(
        aux_basis(mol, name)?,
        route.op,
        route.budget_bytes,
        &mutants,
    )?;
    let (naux, m) = (fit.dfbs.nbasis(), x.ncols());
    let n3 = 3 * natoms;
    let core = build_core(natoms, obs, fit, x, &mutants, false)?;
    let g_star = gamma_fixed.unwrap_or(&core.gamma);
    let mut lg = Array1::<f64>::zeros(n3);
    let mut gvg = Array1::<f64>::zeros(n3);
    let mut z = core.lx.clone();
    for xi in 0..n3 {
        lg[xi] = core
            .lx
            .row(xi)
            .iter()
            .zip(g_star.iter())
            .map(|(a, b)| a * b)
            .sum();
        let vg_star = core.vx.index_axis(Axis(0), xi).dot(g_star);
        gvg[xi] = vg_star.iter().zip(g_star.iter()).map(|(a, b)| a * b).sum();
        let vg_own = core.vx.index_axis(Axis(0), xi).dot(&core.gamma);
        for (dst, src) in z.row_mut(xi).iter_mut().zip(vg_own.iter()) {
            *dst -= *src;
        }
    }
    let _ = (naux, m);
    Ok(RikFirst {
        gamma: core.gamma,
        lg,
        gvg,
        z,
    })
}

/// Block weights `w_{P,μν} = scale · (X Γ_P Xᵀ)_{μν}` for the 3c stencil;
/// `Γ_P` is symmetric, so an off-diagonal shell pair carries both orders as 2×.
fn weight_gamma(
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    gamma: &Array2<f64>,
    x: &Array2<f64>,
    scale: f64,
    shells: (usize, usize, usize),
    w: &mut Vec<f64>,
) -> bool {
    let (sp, s1, s2) = shells;
    let m = x.ncols();
    let (od, oo) = (obs.shell_dims(), obs.shell_offsets());
    let (dd, doff) = (dfbs.shell_dims(), dfbs.shell_offsets());
    let x1 = x.slice(s![oo[s1]..oo[s1] + od[s1], ..]);
    let x2 = x.slice(s![oo[s2]..oo[s2] + od[s2], ..]);
    let fac = if s1 != s2 { 2.0 * scale } else { scale };
    for p in 0..dd[sp] {
        let g = gamma
            .slice(s![doff[sp] + p, ..])
            .into_shape_with_order((m, m))
            .expect("contiguous gamma row");
        let blk = x1.dot(&g).dot(&x2.t());
        w.extend(blk.iter().map(|v| fac * v));
    }
    true
}

/// `L_x` (`3N × A·m²`) and optionally `K1_x` (`3N × n²`) from the
/// first-derivative three-centre blocks, one auxiliary shell per task.
fn first_derivative_3c(
    obs: &PreparedBasis,
    fit: &Fit,
    natoms: usize,
    x: &Array2<f64>,
    chat: &Array3<f64>,
    mutants: &RikMutants,
    with_k1: bool,
) -> Result<(Array2<f64>, Array2<f64>), FerricError> {
    let n3 = 3 * natoms;
    let (n, naux, m) = (obs.nbasis(), fit.dfbs.nbasis(), x.ncols());
    let m2 = m * m;
    let dfbs = &fit.dfbs;
    Engine::new_3center_deriv(fit.op, obs, dfbs, 1e-14)?;
    let mut lx = Array2::<f64>::zeros((n3, naux * m2));
    let mut k1 = Array2::<f64>::zeros((n3, if with_k1 { n * n } else { 0 }));
    let shells: Vec<usize> = (0..dfbs.nshells()).collect();
    for chunk in shells.chunks(chunk_len(n3, n)) {
        let parts: Vec<(usize, Array2<f64>, Array2<f64>)> = chunk
            .par_iter()
            .map_init(
                || {
                    Engine::new_3center_deriv(fit.op, obs, dfbs, 1e-14)
                        .expect("3-center deriv engine (pre-validated)")
                },
                |eng, &sp| shell_partial(eng, obs, dfbs, sp, x, chat, n3, with_k1, mutants),
            )
            .collect();
        for (sp, lxp, k1p) in parts {
            let p0 = dfbs.shell_offsets()[sp];
            let width = lxp.ncols();
            lx.slice_mut(s![.., p0 * m2..p0 * m2 + width]).assign(&lxp);
            if with_k1 {
                k1 += &k1p;
            }
        }
    }
    Ok((lx, k1))
}

/// One auxiliary shell's `L_x` columns and `K1_x` partial.
#[allow(clippy::too_many_arguments)]
fn shell_partial(
    eng: &mut Engine,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    sp: usize,
    x: &Array2<f64>,
    chat: &Array3<f64>,
    n3: usize,
    with_k1: bool,
    mutants: &RikMutants,
) -> (usize, Array2<f64>, Array2<f64>) {
    let n = obs.nbasis();
    let m = x.ncols();
    let m2 = m * m;
    let (od, oo, o_at) = (obs.shell_dims(), obs.shell_offsets(), obs.shell_to_atom());
    let np = dfbs.shell_dims()[sp];
    let pf0 = dfbs.shell_offsets()[sp];
    let atom_p = dfbs.shell_to_atom()[sp];
    let mut lxp = Array2::<f64>::zeros((n3, np * m2));
    let mut k1p = Array2::<f64>::zeros((n3, if with_k1 { n * n } else { 0 }));
    // T_p[ν, λ] = Σ_j X_νj Ĉ_{p,λ,j}  (Ĉ stored (A, m, n): chat[p, j, λ]).
    let t: Vec<Array2<f64>> = if with_k1 {
        (0..np)
            .map(|p| {
                let cp = chat.index_axis(Axis(0), pf0 + p); // (m, n): [j, λ]
                let tp = x.dot(&cp);
                if mutants.transpose_pair_kx {
                    tp.t().as_standard_layout().to_owned()
                } else {
                    tp
                }
            })
            .collect()
    } else {
        Vec::new()
    };
    for s1 in 0..obs.nshells() {
        for s2 in 0..=s1 {
            let Some(deriv) = eng.compute_eri3_deriv(obs, dfbs, sp, s1, s2) else {
                continue;
            };
            let (n1, n2) = (od[s1], od[s2]);
            let bs = np * n1 * n2;
            let atoms = [atom_p, o_at[s1], o_at[s2]];
            let x1 = x.slice(s![oo[s1]..oo[s1] + n1, ..]);
            let x2 = x.slice(s![oo[s2]..oo[s2] + n2, ..]);
            for (centre, &atom) in atoms.iter().enumerate() {
                for q in 0..3 {
                    let xi = 3 * atom + q;
                    for p in 0..np {
                        let off = (3 * centre + q) * bs + p * n1 * n2;
                        let dm =
                            ndarray::ArrayView2::from_shape((n1, n2), &deriv[off..off + n1 * n2])
                                .expect("block shape");
                        // L_x[P] += X1ᵀ Dm X2 (+ transpose for s1 != s2)
                        let y = x1.t().dot(&dm).dot(&x2);
                        let mut row = lxp.row_mut(xi);
                        let seg =
                            &mut row.as_slice_mut().expect("contiguous")[p * m2..(p + 1) * m2];
                        for i in 0..m {
                            for j in 0..m {
                                let v = if s1 != s2 {
                                    y[(i, j)] + y[(j, i)]
                                } else {
                                    y[(i, j)]
                                };
                                seg[i * m + j] += v;
                            }
                        }
                        if with_k1 && !(mutants.omit_aux_centre_kx && centre == 0) {
                            let tp = &t[p];
                            let r1 = dm.dot(&tp.slice(s![oo[s2]..oo[s2] + n2, ..]));
                            let mut k1row = k1p.row_mut(xi);
                            let kseg = k1row.as_slice_mut().expect("contiguous");
                            for a in 0..n1 {
                                let base = (oo[s1] + a) * n;
                                for lam in 0..n {
                                    kseg[base + lam] += r1[(a, lam)];
                                }
                            }
                            if s1 != s2 {
                                let r2 = dm.t().dot(&tp.slice(s![oo[s1]..oo[s1] + n1, ..]));
                                for b2 in 0..n2 {
                                    let base = (oo[s2] + b2) * n;
                                    for lam in 0..n {
                                        kseg[base + lam] += r2[(b2, lam)];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    (sp, lxp, k1p)
}

/// Dense `V_x = ∂(P|Q)/∂x` (`3N × A × A`) from the first-derivative 2c blocks.
fn first_derivative_2c(fit: &Fit, natoms: usize) -> Result<Array3<f64>, FerricError> {
    let n3 = 3 * natoms;
    let dfbs = &fit.dfbs;
    let naux = dfbs.nbasis();
    Engine::new_2center_deriv(fit.metric_op, dfbs, 1e-14)?;
    let (dims, offs, at) = (
        dfbs.shell_dims().to_vec(),
        dfbs.shell_offsets().to_vec(),
        dfbs.shell_to_atom().to_vec(),
    );
    let mut vx = Array3::<f64>::zeros((n3, naux, naux));
    let shells: Vec<usize> = (0..dfbs.nshells()).collect();
    for chunk in shells.chunks(32) {
        type Blocks = Vec<(usize, usize, Array3<f64>)>;
        let parts: Vec<Blocks> = chunk
            .par_iter()
            .map_init(
                || {
                    Engine::new_2center_deriv(fit.metric_op, dfbs, 1e-14)
                        .expect("2-center deriv engine (pre-validated)")
                },
                |eng, &sp| {
                    let mut out = Vec::new();
                    for sq in 0..=sp {
                        let Some(deriv) = eng.compute_eri2_deriv(dfbs, sp, sq) else {
                            continue;
                        };
                        let (np, nq) = (dims[sp], dims[sq]);
                        let bs = np * nq;
                        let mut blk = Array3::<f64>::zeros((n3, np, nq));
                        for (centre, &atom) in [at[sp], at[sq]].iter().enumerate() {
                            for k in 0..3 {
                                for p in 0..np {
                                    for q in 0..nq {
                                        blk[(3 * atom + k, p, q)] +=
                                            deriv[(3 * centre + k) * bs + p * nq + q];
                                    }
                                }
                            }
                        }
                        out.push((sp, sq, blk));
                    }
                    out
                },
            )
            .collect();
        for blocks in parts {
            for (sp, sq, blk) in blocks {
                let (np, nq) = (dims[sp], dims[sq]);
                for xi in 0..n3 {
                    for p in 0..np {
                        for q in 0..nq {
                            let v = blk[(xi, p, q)];
                            vx[(xi, offs[sp] + p, offs[sq] + q)] = v;
                            vx[(xi, offs[sq] + q, offs[sp] + p)] = v;
                        }
                    }
                }
            }
        }
    }
    Ok(vx)
}
