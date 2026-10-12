//! Analytic RHF Hessian for density-fitted Coulomb (RI-J) with EXACT exchange.
//!
//! The SCF energy this differentiates is
//!
//! ```text
//!   E = Tr[D h] + E_J^DF[D] + E_K[D] + V_nn,          D = 2 C_o C_oᵀ
//!   E_J^DF = ½ dᵀ M d,   d_P = Σ_μν (P|μν) D_μν,   M = V⁻¹,   V_PQ = (P|Q)
//!   E_K    = −¼ Σ D_μλ D_νσ (μν|λσ)                (exact four-centre)
//! ```
//!
//! # What changes against the exact-integral Hessian, and what does not
//!
//! The exact Hessian ([`crate::hessian`]) is `skeleton (fixed D and W) +
//! orbital response`. The response derivation uses three facts only: the
//! Fock matrix is `F = ∂E/∂D` and satisfies `F C = S C ε`; the orbitals stay
//! orthonormal; and `D → G(D) = ∂F/∂D · D` is the linear map the CPHF
//! iterates. All three hold for the fitted functional: `F_μν = ∂E_J^DF/∂D_μν
//! = (P|μν) c_P` (`c = M d`) is exactly what `DfJ` builds, and `G(D1) =
//! J_DF(D1) − ½ K(D1)` is linear in `D1`. So the nuclear term, the
//! one-electron skeleton, the overlap/`W` term, `K`'s skeleton and `K^x`, the
//! CPHF solver (`pcg`, `solve_cphf_one`, `assemble_response`) and the whole
//! frequency layer are REUSED UNCHANGED. Three pieces are replaced:
//!
//! 1. the J part of the two-electron skeleton (`Σ Γ_J ∂²(μν|λσ)` →
//!    `RijTerms::skeleton`); the exact skeleton keeps only
//!    `Γ_K = −¼ D_μλ D_νσ`;
//! 2. the J part of the fixed-density first-order Fock matrix `F^x`
//!    (`RijTerms::first_order_j`); the exact `K^x` stays;
//! 3. the Coulomb part of `G` inside the CPHF, which becomes the very
//!    `DfJ` builder the SCF used (the exchange part stays the exact
//!    four-centre `K`).
//!
//! # Derivation (fixed `D`, derivatives of integrals only)
//!
//! Write `d_x = ∂d/∂x` for the derivative of `d_P = Σ D_μν (P|μν)` through the
//! integrals at fixed `D` (the orbital basis functions AND the auxiliary
//! function `P` move with their atoms), `V_x = ∂V/∂x`, `V_xy`, `d_xy`
//! likewise. `∂M = −M V_x M`, hence with `c = M d`
//!
//! ```text
//!   ∂_x c = M (d_x − V_x c) = M Z_x,            Z_x := d_x − V_x c
//!   E_x   = d_xᵀ c − ½ cᵀ V_x c
//!   E_xy  = d_xyᵀ c − ½ cᵀ V_xy c                          (3c and 2c second derivatives)
//!         + d_xᵀ M d_y − d_xᵀ M V_y c − d_yᵀ M V_x c + cᵀ V_y M V_x c
//!         = d_xyᵀ c − ½ cᵀ V_xy c + Z_xᵀ M Z_y.
//! ```
//!
//! (`E_x` is the gradient `df_gradient::df_jk_gradient` contracts; `E_xy`
//! follows from differentiating it once more, using `∂_y c = M Z_y` in
//! `d_xᵀ c` and in `cᵀ V_x c`:
//! `∂_y(d_xᵀ c) = d_xyᵀ c + d_xᵀ M Z_y` and
//! `∂_y(½ cᵀ V_x c) = Z_yᵀ M V_x c + ½ cᵀ V_xy c`, whose sum with
//! `d_xᵀ M Z_y` reorganizes to the line above.) The cross terms are
//! symmetric in `x ↔ y` (`Z_xᵀ M Z_y`), so swapping `d_x d` with `d_y d` in
//! them is an algebraic identity, not a mutant.
//!
//! Second derivatives enter only through two CONTRACTIONS with `c` and `D`,
//! never as tensors: `Σ_P c_P Σ D_μν ∂²(P|μν)` (3c stencil, weight
//! `c_P D_μν`) and `−½ Σ c_P c_Q ∂²(P|Q)` (2c stencil, weight
//! `−½ c_P c_Q`), both from [`crate::hessian_df_stencil`].
//!
//! ## First-order Fock matrix
//!
//! `J_μν(D) = Σ_P (P|μν) c_P`, so at fixed `D`
//!
//! ```text
//!   J^x_μν = ∂_x(P|μν) c_P + (P|μν) [M Z_x]_P            (the second piece is the
//!                                                         auxiliary-fit response)
//! ```
//!
//! The first piece is the first-derivative three-centre block weighted by
//! `c`; the second is a plain `DfJ`-style contraction with the vector
//! `y_x = M Z_x` in place of `c`. It carries the aux-centre dependence of
//! the fit (the derivative of `c` through `V` and through the movement of
//! `P`); omitting it is not a small error (`RijMutants::omit_aux_jx`).
//!
//! ## Cholesky versus explicit inverse
//!
//! `M` is never formed. Every application `M v` is a back-substitution
//! against the Cholesky factor of `V` built exactly as `DfJ` builds it
//! (`coulomb_metric_2c` then `factorizec(Lower)`), the factorization the SCF's
//! `c` came from (the "explicit inverse jitter" defect: `V⁻¹` from LU has a
//! forward error `~ ε·cond(V)` that is not a smooth function of `D`).
//! Second derivatives of `M` are never needed in closed form because the
//! algebra above only uses `c` and `y_x = M Z_x`.
//!
//! ## Conditioning
//!
//! The 2c stencil and the Cholesky solves amplify noise by `cond(V)`.
//! RI-J uses the FULL inverse, so there are no dropped modes. Measured
//! accuracy against the cond(V) of the fit (6-31G water, own-gradient FD):
//! cond 2.5e5 (cc-pVDZ-RI), 2.5e7 (def2-universal-jkfit), 2.4e6 / 9.1e6 / 2.0e8
//! (aug-cc-pV{D,T,Q}Z-RIFIT) all agree to 1.6e-10 .. 2.0e-10, i.e. no
//! degradation; an exactly-spanning fit stays exact (skeleton 2e-13, `J^x`
//! 6e-13) up to cond 2e14 (a benign case: the data are consistent with the
//! metric). Between cond 2e8 and the refusal threshold (about 3e12 for
//! `λ_max` ~ 3e2) a generic density is unmeasured, hence a warning below
//! `RIJ_METRIC_WARN`. A metric whose smallest eigenvalue is below the DF-K cut
//! (`crate::df_k::DFK_LINDEP_THRESH`) is numerically singular for this
//! purpose and is refused with a typed error naming the count of such modes.
//!
//! ## Cost
//!
//! One streamed `(P|μν)` pass for `d`, one first-derivative 3c pass (`d_x`
//! and the `c`-weighted AO matrices `J^x`), one 2c first-derivative pass
//! (`V_x c`), one more streamed pass for the `y_x`-weighted `J^x`, and the
//! two stencil contractions (about 30 times the engine calls of a gradient
//! pass). Memory beyond the budget-bounded raw tensor, charged to the global
//! pool up front (a typed error when it does not fit): inside `rij_terms`,
//! `j1`, `j2` and the output `J^x` are alive together (`3 · 8 · 3N · nbf²`
//! bytes) plus `chunk` per-aux-shell partials of the same size, where
//! `chunk = clamp(256 MB / (8 · 3N · nbf²), 1, 32)` (so it is 1 beyond about
//! 40 atoms with 400 basis functions: the first-derivative 3c pass then runs
//! one aux shell at a time), and `8 · (3 · 3N · naux + 2 · naux²)` bytes of
//! metric-sized work arrays. After it returns, `J^x` stays resident next to
//! the exact path's first-order Fock matrices and the `G^x` accumulators
//! (about three more `3N · nbf²` matrices), which are not charged here.

use crate::df_gradient::aux_basis;
use crate::df_k::DFK_LINDEP_THRESH;
use crate::hessian_df_stencil::{
    eri2_deriv2_contract, eri3_deriv2_contract, weight_c_times_d, weight_matrix, StencilConfig,
};
use crate::result::DfJkRoute;
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::operator::Operator;
use ferric_integrals::three_index_source::ThreeIndexSource;
use ferric_integrals::threeindex::coulomb_metric_2c;
use ndarray::{s, Array1, Array2};
use ndarray_linalg::cholesky::{CholeskyFactorized, FactorizeC, SolveC};
use ndarray_linalg::{Eigh, Inverse, UPLO};
use rayon::prelude::*;

/// Largest number of auxiliary shells whose per-shell partials are alive at
/// once in the first-derivative passes (the reduction is chunked and ordered,
/// so the result does not depend on the thread count).
const MAX_CHUNK: usize = 32;
/// Bytes one chunk of per-shell `3N × nbf²` partials may occupy.
const CHUNK_BYTES: usize = 256 << 20;

/// Deliberately broken constructions, wired through the production code path
/// for the mutation tests of `tests/df_hessian_rij.rs`. `Default` is the
/// production construction. NOT public API.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default)]
pub struct RijMutants {
    /// Drop `−½ cᵀ V_xy c` (the 2c second derivative).
    pub drop_metric_second: bool,
    /// Drop the 3c-2c cross terms of the skeleton (`Z_x := d_x`).
    pub drop_cross: bool,
    /// Drop the auxiliary-fit part `(P|μν) [M Z_x]_P` of `J^x`.
    pub omit_aux_jx: bool,
    /// Use an erfc(0.4) metric for `V`, `V_x`, `V_xy`, `M`, with Coulomb 3c.
    pub swap_metric: bool,
    /// Apply an explicit `V⁻¹` where the SCF used the Cholesky solve.
    pub explicit_inverse: bool,
    /// Stencil construction (step, order, shell displacement).
    pub stencil: StencilConfig,
    /// Skip the conditioning refusal (measurement of the accuracy of a
    /// deliberately ill-conditioned fit; never set in production).
    pub allow_ill_conditioned: bool,
}

/// The RI-J pieces of the Hessian.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct RijTerms {
    /// `∂²E_J^DF/∂x∂y` at fixed `D`, `3N × 3N` (symmetrized).
    pub skeleton: Array2<f64>,
    /// `max |S − Sᵀ|` of the skeleton before symmetrization.
    pub skeleton_asymmetry: f64,
    /// `Σ_P c_P Σ D_μν ∂²(P|μν)`.
    pub three_centre: Array2<f64>,
    /// `−½ cᵀ V_xy c`.
    pub metric_second: Array2<f64>,
    /// `Z_xᵀ M Z_y`.
    pub cross: Array2<f64>,
    /// `J^x`, one `nbf × nbf` AO matrix per coordinate `x = 3·atom + axis`.
    pub first_order_j: Vec<Array2<f64>>,
    /// The first-derivative vectors the skeleton was assembled from.
    pub first: RijFirst,
}

/// Fixed-density first-derivative quantities of the fit (`3N` rows each).
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct RijFirst {
    /// `c = M d`.
    pub c: Array1<f64>,
    /// `d_x`, `3N × naux`.
    pub dx: Array2<f64>,
    /// `V_x c`, `3N × naux`.
    pub vc: Array2<f64>,
    /// `y_x = M (d_x − V_x c)`, `3N × naux` (so `∂_x c`).
    pub y: Array2<f64>,
}

/// Metric, its Cholesky factor and the operators of one fit.
struct Fit {
    dfbs: PreparedBasis,
    chol: CholeskyFactorized<ndarray::OwnedRepr<f64>>,
    v_inv: Option<Array2<f64>>,
    op: Operator,
    metric_op: Operator,
    budget: usize,
}

impl Fit {
    fn new(
        dfbs: PreparedBasis,
        op: Operator,
        budget: usize,
        mutants: &RijMutants,
    ) -> Result<Self, FerricError> {
        let metric_op = if mutants.swap_metric {
            Operator::erfc(0.4)
        } else {
            op
        };
        let v = coulomb_metric_2c(metric_op, &dfbs)?;
        if !mutants.allow_ill_conditioned {
            check_metric_conditioning(&v)?;
        }
        let chol = v.factorizec(UPLO::Lower).map_err(|e| {
            FerricError::Lapack(format!(
                "analytic RI-J Hessian: Cholesky of the Coulomb metric failed: {e}"
            ))
        })?;
        let v_inv = if mutants.explicit_inverse {
            Some(v.inv().map_err(|e| {
                FerricError::Lapack(format!("analytic RI-J Hessian: explicit inverse: {e}"))
            })?)
        } else {
            None
        };
        Ok(Fit {
            dfbs,
            chol,
            v_inv,
            op,
            metric_op,
            budget,
        })
    }

    /// `M v` through the factorization the SCF's `DfJ` uses.
    fn solve(&self, rhs: &Array1<f64>) -> Result<Array1<f64>, FerricError> {
        if let Some(inv) = &self.v_inv {
            return Ok(inv.dot(rhs));
        }
        self.chol
            .solvec(rhs)
            .map_err(|e| FerricError::Lapack(format!("analytic RI-J Hessian: metric solve: {e}")))
    }
}

/// Metric eigenvalue below which the fit is still accepted but warned about:
/// the full inverse the energy uses amplifies integral noise by `1/λ_min`.
/// Measured accuracy (module doc): no degradation up to cond(V) = 2e8 for real
/// fitting bases, and none up to cond(V) = 2e14 for an exactly-spanning fit,
/// so this is a precaution, not a measured failure.
pub const RIJ_METRIC_WARN: f64 = 1e-8;

/// Refuse a Coulomb metric with eigenvalues below the DF-K cut (a typed error
/// naming the count) and warn in the band below [`RIJ_METRIC_WARN`]. Zero
/// refusals for every healthy fitting basis.
fn check_metric_conditioning(v: &Array2<f64>) -> Result<(), FerricError> {
    let (lam, _) = v
        .eigh(UPLO::Upper)
        .map_err(|e| FerricError::Lapack(format!("analytic RI-J Hessian: V eigh: {e}")))?;
    let modes = lam.iter().filter(|&&l| l < DFK_LINDEP_THRESH).count();
    if modes > 0 {
        return Err(FerricError::General(format!(
            "analytic RI-J Hessian: the Coulomb metric of the fitting basis has {modes} \
             eigenvalue(s) below {DFK_LINDEP_THRESH:.0e}; the full inverse the energy uses is \
             numerically singular, so its second derivative is refused (use finite differences)"
        )));
    }
    if lam[0] < RIJ_METRIC_WARN {
        eprintln!(
            "[ferric] warning: analytic RI-J Hessian: the smallest eigenvalue of the Coulomb \
             metric is {:.2e} (cond {:.1e}); the full inverse amplifies integral noise by \
             1/lambda_min and the Hessian accuracy is unmeasured in this band",
            lam[0],
            lam[lam.len() - 1] / lam[0]
        );
    }
    Ok(())
}

/// The pre-SCF probe's conditioning check: the metric depends on the geometry
/// and the fitting basis only, so it can be decided before any SCF.
#[doc(hidden)]
pub fn probe_metric(dfbs: &PreparedBasis, op: Operator) -> Result<(), FerricError> {
    check_metric_conditioning(&coulomb_metric_2c(op, dfbs)?)
}

/// Chunk width for the per-aux-shell partial reductions.
fn chunk_len(n3: usize, nbf: usize) -> usize {
    let per = 8 * n3 * nbf * nbf;
    (CHUNK_BYTES / per.max(1)).clamp(1, MAX_CHUNK)
}

/// The RI-J Hessian pieces at the converged density `d` (total, `2 C_o C_oᵀ`).
#[doc(hidden)]
pub fn rij_terms(
    mol: &Molecule,
    obs: &PreparedBasis,
    route: &DfJkRoute,
    d: &Array2<f64>,
    mutants: &RijMutants,
) -> Result<RijTerms, FerricError> {
    let name = route.j_aux.as_deref().ok_or_else(|| {
        FerricError::General("analytic RI-J Hessian: the SCF route has no RI-J basis".into())
    })?;
    let dfbs = aux_basis(mol, name)?;
    rij_terms_with_aux(
        mol.atoms.len(),
        obs,
        dfbs,
        route.op,
        route.budget_bytes,
        d,
        mutants,
    )
}

/// [`rij_terms`] with an explicit auxiliary basis (the exactness anchors hand
/// it a basis that spans the density's products exactly).
#[doc(hidden)]
pub fn rij_terms_with_aux(
    natoms: usize,
    obs: &PreparedBasis,
    dfbs: PreparedBasis,
    op: Operator,
    budget: usize,
    d: &Array2<f64>,
    mutants: &RijMutants,
) -> Result<RijTerms, FerricError> {
    let fit = Fit::new(dfbs, op, budget, mutants)?;
    let n3 = 3 * natoms;
    let nbf = obs.nbasis();
    // Peak inside this function: j1, j2 and the output `first_order_j` are
    // alive together (3 x 3N nbf^2), plus the chunk of per-shell partials;
    // the metric-sized work arrays are d_x, V_x c, y_x and the 2c stencil
    // weight (naux^2). Charged to the global budget up front: a typed error
    // when it does not fit.
    let naux = fit.dfbs.nbasis();
    let per_matrix = 8 * n3 * nbf * nbf;
    let _charge = ferric_core::memory::pool::reserve_global(
        "RI-J Hessian J^x matrices, first-derivative partials and metric work",
        per_matrix
            .saturating_mul(3 + chunk_len(n3, nbf))
            .saturating_add(8 * (3 * n3 * naux + 2 * naux * naux)),
    )?;
    let mut source = ThreeIndexSource::build(fit.op, obs, &fit.dfbs, fit.budget)?;
    let d_p = d_vector(&mut source, d, fit.dfbs.nbasis())?;
    let c = fit.solve(&d_p)?;
    let (dx, j1) = first_derivative_3c(obs, &fit, natoms, d, &c)?;
    let vc = first_derivative_2c(&fit, natoms, &c)?;
    let z = &dx - &vc;
    let y = solve_rows(&fit, &z)?;
    let j2 = aux_weighted_ao(&mut source, &y, nbf)?;

    let (z_skel, y_skel) = if mutants.drop_cross {
        (dx.clone(), solve_rows(&fit, &dx)?)
    } else {
        (z.clone(), y.clone())
    };
    let cross = z_skel.dot(&y_skel.t());
    let cfg = &mutants.stencil;
    let cs = c.as_slice().expect("contiguous c");
    let three_centre = eri3_deriv2_contract(obs, &fit.dfbs, fit.op, cfg, |sp, s1, s2, w| {
        weight_c_times_d(obs, &fit.dfbs, cs, d, (sp, s1, s2), w)
    })?;
    let metric_second = if mutants.drop_metric_second {
        Array2::zeros((n3, n3))
    } else {
        let m2 = Array2::from_shape_fn((c.len(), c.len()), |(p, q)| -0.5 * c[p] * c[q]);
        eri2_deriv2_contract(&fit.dfbs, fit.metric_op, cfg, |sp, sq, w| {
            weight_matrix(&fit.dfbs, &m2, (sp, sq), w)
        })?
    };
    let raw = &three_centre + &metric_second + &cross;
    let skeleton_asymmetry = raw
        .iter()
        .zip(raw.t().iter())
        .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
    let skeleton = (&raw + &raw.t()) * 0.5;

    let mut first_order_j = Vec::with_capacity(n3);
    for x in 0..n3 {
        let mut jx = j1.row(x).to_owned();
        if !mutants.omit_aux_jx {
            jx += &j2.row(x);
        }
        first_order_j.push(
            jx.into_shape_with_order((nbf, nbf))
                .map_err(|e| FerricError::General(format!("J^x reshape: {e}")))?,
        );
    }
    Ok(RijTerms {
        skeleton,
        skeleton_asymmetry,
        three_centre,
        metric_second,
        cross,
        first_order_j,
        first: RijFirst { c, dx, vc, y },
    })
}

/// First-derivative quantities only (no second derivatives, no `J^x`): the
/// fit vector `c`, `d_x`, `V_x c` and `∂_x c`. Used by the per-term finite
/// difference tests at displaced geometries. With `c_fixed` the vector
/// `V_x c` is formed with that `c` instead of the geometry's own (the
/// fixed-`c` pieces of the skeleton).
#[doc(hidden)]
pub fn rij_first(
    mol: &Molecule,
    obs: &PreparedBasis,
    route: &DfJkRoute,
    d: &Array2<f64>,
    c_fixed: Option<&Array1<f64>>,
) -> Result<RijFirst, FerricError> {
    let mutants = RijMutants::default();
    let name = route.j_aux.as_deref().ok_or_else(|| {
        FerricError::General("analytic RI-J Hessian: the SCF route has no RI-J basis".into())
    })?;
    let fit = Fit::new(
        aux_basis(mol, name)?,
        route.op,
        route.budget_bytes,
        &mutants,
    )?;
    let natoms = mol.atoms.len();
    let mut source = ThreeIndexSource::build(fit.op, obs, &fit.dfbs, fit.budget)?;
    let d_p = d_vector(&mut source, d, fit.dfbs.nbasis())?;
    let c = fit.solve(&d_p)?;
    let (dx, _) = first_derivative_3c(obs, &fit, natoms, d, &c)?;
    let vc = first_derivative_2c(&fit, natoms, c_fixed.unwrap_or(&c))?;
    let y = solve_rows(&fit, &(&dx - &vc))?;
    Ok(RijFirst { c, dx, vc, y })
}

/// `d_P = Σ (P|μν) D_μν`.
fn d_vector(
    source: &mut ThreeIndexSource,
    d: &Array2<f64>,
    naux: usize,
) -> Result<Array1<f64>, FerricError> {
    let mut d_p = Array1::<f64>::zeros(naux);
    source.for_each_block(|blk| {
        for q in 0..blk.data.shape()[0] {
            d_p[blk.p0 + q] = blk
                .data
                .index_axis(ndarray::Axis(0), q)
                .iter()
                .zip(d.iter())
                .map(|(a, b)| a * b)
                .sum::<f64>();
        }
        Ok(())
    })?;
    Ok(d_p)
}

/// `M` applied to every row of `z` (`3N × naux`).
fn solve_rows(fit: &Fit, z: &Array2<f64>) -> Result<Array2<f64>, FerricError> {
    let mut y = Array2::<f64>::zeros(z.dim());
    for x in 0..z.nrows() {
        y.row_mut(x).assign(&fit.solve(&z.row(x).to_owned())?);
    }
    Ok(y)
}

/// `Σ_P y[x, P] (P|μν)` as a `3N × nbf²` matrix: the `DfJ` contraction with
/// every row of `y` in place of `c`.
fn aux_weighted_ao(
    source: &mut ThreeIndexSource,
    y: &Array2<f64>,
    nbf: usize,
) -> Result<Array2<f64>, FerricError> {
    let mut out = Array2::<f64>::zeros((y.nrows(), nbf * nbf));
    source.for_each_block(|blk| {
        let b = blk.data.shape()[0];
        let flat = blk
            .data
            .to_shape((b, nbf * nbf))
            .map_err(|e| FerricError::General(format!("3c block reshape: {e}")))?;
        out += &y.slice(s![.., blk.p0..blk.p0 + b]).dot(&flat);
        Ok(())
    })?;
    Ok(out)
}

/// `d_x` (`3N × naux`) and the `c`-weighted AO matrices
/// `Σ_P c_P ∂_x(P|μν)` (`3N × nbf²`) from the first-derivative 3c blocks.
fn first_derivative_3c(
    obs: &PreparedBasis,
    fit: &Fit,
    natoms: usize,
    d: &Array2<f64>,
    c: &Array1<f64>,
) -> Result<(Array2<f64>, Array2<f64>), FerricError> {
    let n3 = 3 * natoms;
    let (nbf, naux) = (obs.nbasis(), fit.dfbs.nbasis());
    let dfbs = &fit.dfbs;
    Engine::new_3center_deriv(fit.op, obs, dfbs, 1e-14)?;
    let mut dx = Array2::<f64>::zeros((n3, naux));
    let mut j1 = Array2::<f64>::zeros((n3, nbf * nbf));
    let shells: Vec<usize> = (0..dfbs.nshells()).collect();
    for chunk in shells.chunks(chunk_len(n3, nbf)) {
        let parts: Vec<(usize, Array2<f64>, Array2<f64>)> = chunk
            .par_iter()
            .map_init(
                || {
                    Engine::new_3center_deriv(fit.op, obs, dfbs, 1e-14)
                        .expect("3-center deriv engine (pre-validated)")
                },
                |eng, &sp| shell_partial_3c(eng, obs, dfbs, sp, d, c, n3),
            )
            .collect();
        for (sp, dxp, j1p) in parts {
            let p0 = dfbs.shell_offsets()[sp];
            dx.slice_mut(s![.., p0..p0 + dxp.ncols()]).assign(&dxp);
            j1 += &j1p;
        }
    }
    Ok((dx, j1))
}

/// One auxiliary shell's `d_x` block and `c`-weighted AO partial.
fn shell_partial_3c(
    eng: &mut Engine,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    sp: usize,
    d: &Array2<f64>,
    c: &Array1<f64>,
    n3: usize,
) -> (usize, Array2<f64>, Array2<f64>) {
    let nbf = obs.nbasis();
    let (od, oo, o_at) = (obs.shell_dims(), obs.shell_offsets(), obs.shell_to_atom());
    let np = dfbs.shell_dims()[sp];
    let pf0 = dfbs.shell_offsets()[sp];
    let atom_p = dfbs.shell_to_atom()[sp];
    let mut dxp = Array2::<f64>::zeros((n3, np));
    let mut j1p = Array2::<f64>::zeros((n3, nbf * nbf));
    for s1 in 0..obs.nshells() {
        for s2 in 0..=s1 {
            let Some(deriv) = eng.compute_eri3_deriv(obs, dfbs, sp, s1, s2) else {
                continue;
            };
            let (n1, n2) = (od[s1], od[s2]);
            let bs = np * n1 * n2;
            let atoms = [atom_p, o_at[s1], o_at[s2]];
            for p in 0..np {
                let cp = c[pf0 + p];
                for i in 0..n1 {
                    for j in 0..n2 {
                        let (mu, nu) = (oo[s1] + i, oo[s2] + j);
                        let idx = (p * n1 + i) * n2 + j;
                        let dsym = if s1 != s2 {
                            d[(mu, nu)] + d[(nu, mu)]
                        } else {
                            d[(mu, nu)]
                        };
                        for (centre, &atom) in atoms.iter().enumerate() {
                            for q in 0..3 {
                                let v = deriv[(3 * centre + q) * bs + idx];
                                let x = 3 * atom + q;
                                dxp[(x, p)] += v * dsym;
                                j1p[(x, mu * nbf + nu)] += v * cp;
                                if s1 != s2 {
                                    j1p[(x, nu * nbf + mu)] += v * cp;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    (sp, dxp, j1p)
}

/// `V_x c` (`3N × naux`) from the first-derivative 2c blocks.
fn first_derivative_2c(
    fit: &Fit,
    natoms: usize,
    c: &Array1<f64>,
) -> Result<Array2<f64>, FerricError> {
    let n3 = 3 * natoms;
    let dfbs = &fit.dfbs;
    let naux = dfbs.nbasis();
    Engine::new_2center_deriv(fit.metric_op, dfbs, 1e-14)?;
    let mut vc = Array2::<f64>::zeros((n3, naux));
    let shells: Vec<usize> = (0..dfbs.nshells()).collect();
    for chunk in shells.chunks(MAX_CHUNK) {
        let parts: Vec<Array2<f64>> = chunk
            .par_iter()
            .map_init(
                || {
                    Engine::new_2center_deriv(fit.metric_op, dfbs, 1e-14)
                        .expect("2-center deriv engine (pre-validated)")
                },
                |eng, &sp| shell_partial_2c(eng, dfbs, sp, c, n3),
            )
            .collect();
        for p in parts {
            vc += &p;
        }
    }
    Ok(vc)
}

fn shell_partial_2c(
    eng: &mut Engine,
    dfbs: &PreparedBasis,
    sp: usize,
    c: &Array1<f64>,
    n3: usize,
) -> Array2<f64> {
    let (dims, offs, at) = (
        dfbs.shell_dims(),
        dfbs.shell_offsets(),
        dfbs.shell_to_atom(),
    );
    let mut out = Array2::<f64>::zeros((n3, dfbs.nbasis()));
    for sq in 0..=sp {
        let Some(deriv) = eng.compute_eri2_deriv(dfbs, sp, sq) else {
            continue;
        };
        let (np, nq) = (dims[sp], dims[sq]);
        let bs = np * nq;
        let atoms = [at[sp], at[sq]];
        for p in 0..np {
            for q in 0..nq {
                let (pf, qf) = (offs[sp] + p, offs[sq] + q);
                for (centre, &atom) in atoms.iter().enumerate() {
                    for k in 0..3 {
                        let v = deriv[(3 * centre + k) * bs + p * nq + q];
                        let x = 3 * atom + k;
                        out[(x, pf)] += v * c[qf];
                        if sp != sq {
                            out[(x, qf)] += v * c[pf];
                        }
                    }
                }
            }
        }
    }
    out
}
