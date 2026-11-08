//! Nuclear gradients of density-fitted (RI) Coulomb and exchange energies.
//!
//! The SCF builds J with [`crate::df_j::DfJ`] (RI-J by default in every KS-DFT
//! run), ω = 0 exchange with [`crate::df_k::DfK`] when `df_k_aux` resolves to
//! a basis, and range-separated exchange ALWAYS through an erfc/erf `DfK`
//! pair. The energy those builders produce is a different function of the
//! nuclear coordinates than the exact four-centre energy, so its gradient
//! must differentiate the fitted integrals, the auxiliary basis centres and
//! the fitting metric. Pairing an RI energy with the exact four-centre
//! gradient (what every gradient entry point did before this module) leaves
//! the optimizer converging to a point that is not stationary on the energy
//! it reports.
//!
//! ## Energies differentiated
//!
//! With `(P|μν)` the raw three-centre integrals and `V_PQ = (P|Q)` the
//! two-centre metric of the SAME operator the energy used:
//!
//! * RI-J (Coulomb metric, full Cholesky inverse, as in `DfJ`):
//!   `E_J = ½ dᵀ V⁻¹ d`, `d_P = Σ_μν (P|μν) D_μν`. With `c = V⁻¹ d`,
//!   `∂E_J = Σ_P c_P Σ_μν D_μν ∂(P|μν) − ½ cᵀ ∂V c`.
//! * RI-K, one channel per density `D_s = X_s X_sᵀ` with energy weight `α_s`
//!   (`α = −c_K/4` for a closed-shell total density, `−c_K/2` per spin for
//!   UHF/ROHF): `E_K = Σ_s α_s Σ_ij L^s_ij · M L^s_ij`,
//!   `L^s_{P,ij} = Σ_μν X_μi (P|μν) X_νj`, `M` the metric (pseudo-)inverse
//!   `DfK` applies (`V^{-1/2}·V^{-1/2}` with its eigenvalue cut). The
//!   three-centre weight is `2 α_s X_s Γ^s_P X_sᵀ`, `Γ^s = M L^s`; the
//!   two-centre weight is the derivative of `M` contracted with
//!   `Y = Σ_s α_s Σ_ij L^s_ij L^s_ijᵀ`.
//!
//! These are the standard DF-SCF gradient expressions (Weigend, Häser, Patzelt
//! & Ahlrichs, Chem. Phys. Lett. 294, 143 (1998) for RI-J; Weigend, Kattannek
//! & Ahlrichs, J. Chem. Phys. 130, 164106 (2009) for RI-K), the orbital
//! response being carried as usual by the energy-weighted density in the
//! one-electron Pulay term. A numpy prototype of exactly these contractions
//! reproduces `pyscf.df.grad.rhf`/`uhf` to 2e-11 / 1.4e-10 Ha/Bohr.
//!
//! ## The truncated exchange metric
//!
//! `DfK` builds `V^{-1/2}` from `eigh(V)` and DROPS modes with eigenvalue
//! below the cut of `crate::df_k::metric_lindep_cut` (nominally 1e-10, moved
//! into a spectral gap when an eigenvalue lies within rounding of it; this
//! gradient applies the same rule to the same `eigh`); the long-range erf metric
//! with a JK-fit basis always has such modes (17 of 113 for water /
//! def2-universal-jkfit at ω = 0.3). The energy is then `E(M)` with the
//! spectral function `M = f(V)`, `f(λ) = 1/λ` above the cut, `0` below. Its
//! exact derivative is the Daleckii–Krein formula
//! `∂M = U (F ∘ Uᵀ ∂V U) Uᵀ` with the divided differences
//! `F_kl = (f(λ_k) − f(λ_l)) / (λ_k − λ_l)` (`−1/(λ_k λ_l)` when both modes are
//! kept, which reduces to the familiar `−M ∂V M`), i.e. it includes the
//! rotation of kept modes into dropped ones, which the plain `−M ∂V M` misses.
//! Prototype, water/6-31G, fixed density, with a threshold that sits in a
//! clean spectral gap: Daleckii–Krein matches finite differences to 2e-8,
//! the plain formula is 5e-5 off.
//!
//! In floating point, however, modes whose eigenvalue sits just above the
//! cut are noise-dominated: `Γ_k = (Uᵀ L)_k / λ_k` carries the integrals'
//! absolute error divided by λ_k (1e-14 / 1e-10 at libint's precision), so an
//! exactly-evaluated derivative of the 1e-10-truncated energy is off by ~5e-6
//! Ha/Bohr (prototype, erf ω = 0.3, water 6-31G and cc-pVDZ) even though the
//! ENERGY is smooth to ~1e-10 Ha, because those modes carry almost no energy
//! (the whole [1e-10, 1e-7) band holds 1.0e-9 Ha there). The gradient
//! therefore applies the Daleckii–Krein formula with a noise floor
//! `DFK_GRADIENT_NOISE_FLOOR` (1e-7): modes the energy kept below that floor
//! are differentiated as if dropped. The residual is exactly the derivative
//! of the energy held in that band. MEASURED (prototype, central FD of the
//! 1e-10-truncated energy): floor 1e-10 → 5.0e-6 / 4.1e-6, floor 1e-7 →
//! 1.2e-8 / 2.7e-9 Ha/Bohr (6-31G / cc-pVDZ). The band energy is computed on
//! every call and a warning is printed when it exceeds
//! `NOISE_BAND_ENERGY_WARN`, so a system where the band is NOT negligible
//! cannot pass silently. The Coulomb and erfc metrics of these systems have
//! no eigenvalue below ~1e-5, so for them the floor changes nothing and the
//! formula is the exact `−M ∂V M` one.
//!
//! RI-J uses the full Cholesky inverse (no truncation), exactly as `DfJ`.
//!
//! ## Memory
//!
//! The raw `(P|μν)` tensor is read once, so it is never stored: it is
//! computed one aux-shell-aligned block at a time (at most
//! [`GRADIENT_3C_BLOCK_CAP_BYTES`](crate::df_gradient::GRADIENT_3C_BLOCK_CAP_BYTES)
//! and a quarter of the budget), contracted into `d_P` / `L^s`, and freed.
//! Resident beside that block: the `(naux, m_s²)` exchange intermediates per
//! channel (`m_s` = density rank, i.e. the occupied count), a handful of
//! `(naux, naux)` metric matrices, and per rayon worker one `(n_P, nao, nao)`
//! weight slab for the current auxiliary shell. All are charged to the global
//! memory pool before allocation.
//!
//! ## MPI
//!
//! The SCF stripes the aux band of `DfJ`/`DfK` across ranks; this gradient
//! does NOT. Every rank computes the full, identical gradient from the full
//! aux range (a gradient is evaluated once per geometry, not per SCF
//! iteration), so it is correct under MPI but its memory does not shrink with
//! rank count.

use crate::df_j::RiJPrecision;
use crate::df_k::metric_lindep_cut;
use crate::result::DfJkRoute;
use crate::screening::SchwarzBounds;
use ferric_core::linalg::par_gemm;
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::blas_threads::{opt_in_blas_threads, with_blas_threads};
use ferric_integrals::engine::Engine;
use ferric_integrals::operator::Operator;
use ferric_integrals::threeindex::coulomb_metric_2c;
use ndarray::{Array1, Array2, Array3, Axis};
use ndarray_linalg::cholesky::{FactorizeC, SolveC};
use ndarray_linalg::{Eigh, UPLO};
use rayon::prelude::*;

/// Metric eigenvalue below which the exchange GRADIENT treats a mode as
/// dropped, even when the energy (cut by `crate::df_k::metric_lindep_cut`) kept it. See
/// the module doc: the modes in `[1e-10, 1e-7)` are noise-dominated in the
/// derivative but hold ~1e-9 Ha of energy.
pub const DFK_GRADIENT_NOISE_FLOOR: f64 = 1e-7;

/// Warn when the exchange energy held by metric modes in the noise band
/// `[energy cut, DFK_GRADIENT_NOISE_FLOOR)` exceeds this (Ha): the
/// gradient's residual against the energy is the derivative of that band.
pub const NOISE_BAND_ENERGY_WARN: f64 = 1e-6;

/// Relative cut on the eigenvalues of an AO density when factoring it as
/// `D = X Xᵀ` (a density from an SCF is `C n Cᵀ` with `n ≥ 0`).
const DENSITY_FACTOR_REL_CUT: f64 = 1e-12;

/// One density-fitted exchange channel: an AO density `D_s` and the weight
/// `α_s` of `Σ_μνλσ D_μν D_λσ (μλ|νσ)_fit` in the energy.
pub struct DfKChannel<'a> {
    pub density: &'a Array2<f64>,
    pub alpha: f64,
}

/// Factor a positive-semidefinite AO density as `D = X Xᵀ`.
///
/// Works for any SCF density (`C n Cᵀ`, `n ≥ 0`), including fractional
/// (smeared) occupations, where the occupied MOs alone would not reproduce
/// `D`. A clearly negative eigenvalue means `D` is not an SCF density and is a
/// hard error rather than a silently wrong exchange gradient.
pub(crate) fn density_factor(d: &Array2<f64>) -> Result<Array2<f64>, FerricError> {
    let n = d.nrows();
    let (evals, evecs) = d
        .eigh(UPLO::Upper)
        .map_err(|e| FerricError::Lapack(format!("DF-K gradient: density eigh failed: {e}")))?;
    let scale = evals.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
    if scale == 0.0 {
        return Ok(Array2::zeros((n, 0)));
    }
    let cut = DENSITY_FACTOR_REL_CUT * scale;
    if let Some(&neg) = evals.iter().find(|&&v| v < -1e-8 * scale) {
        return Err(FerricError::General(format!(
            "DF-K gradient: density has eigenvalue {neg:.3e} (max |λ| {scale:.3e}); \
             an SCF density is positive semidefinite"
        )));
    }
    let keep: Vec<usize> = (0..evals.len()).filter(|&k| evals[k] > cut).collect();
    let mut x = Array2::<f64>::zeros((n, keep.len()));
    for (col, &k) in keep.iter().enumerate() {
        let s = evals[k].sqrt();
        for r in 0..n {
            x[(r, col)] = evecs[(r, k)] * s;
        }
    }
    Ok(x)
}

/// Gradient of `½ Tr[D K_fit(D)]` at fixed D for ANY symmetric `d`, under
/// the fit [`df_jk_gradient`] differentiates.
///
/// The exchange channels need positive-semidefinite densities (they are
/// factored as `X Xᵀ`); a response or Z-vector density is indefinite. With
/// the eigen-split `d = P − N` (`P, N ⪰ 0`) and `Q` the quadratic form,
/// `Q(P − N) = 2Q(P) + 2Q(N) − Q(P + N)`, and `P`, `N`, `P + N = |d|` are
/// all PSD, so the energy is three channels with weights `1, 1, −½`. A PSD
/// `d` (no eigenvalue below `−1e-8·max|λ|`, the factorization's own
/// tolerance) stays the single channel `(d, ½)`, bit-identical to calling
/// [`df_jk_gradient`] with it.
pub fn df_k_gradient_any(
    natoms: usize,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    d: &Array2<f64>,
    budget_bytes: usize,
) -> Result<Array2<f64>, FerricError> {
    let (evals, evecs) = d
        .eigh(UPLO::Upper)
        .map_err(|e| FerricError::Lapack(format!("DF-K gradient: density eigh failed: {e}")))?;
    let scale = evals.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
    if evals.iter().all(|&v| v >= -1e-8 * scale) {
        let ch = [DfKChannel {
            density: d,
            alpha: 0.5,
        }];
        return df_jk_gradient(natoms, obs, dfbs, op, None, &ch, budget_bytes);
    }
    let part = |keep: &dyn Fn(f64) -> bool| -> Array2<f64> {
        let mut m = Array2::<f64>::zeros(d.dim());
        for (k, &lam) in evals.iter().enumerate() {
            if keep(lam) {
                let v = evecs.column(k);
                let vv = v.insert_axis(ndarray::Axis(1));
                m.scaled_add(lam.abs(), &vv.dot(&vv.t()));
            }
        }
        m
    };
    let (p, n) = (part(&|l| l > 0.0), part(&|l| l < 0.0));
    let abs = &p + &n;
    let ch = [
        DfKChannel {
            density: &p,
            alpha: 1.0,
        },
        DfKChannel {
            density: &n,
            alpha: 1.0,
        },
        DfKChannel {
            density: &abs,
            alpha: -0.5,
        },
    ];
    df_jk_gradient(natoms, obs, dfbs, op, None, &ch, budget_bytes)
}

/// Largest raw `(P|μν)` block the gradient's one-pass contraction holds.
///
/// The block is transient (computed, contracted, freed), so width only trades
/// load balance across the aux shells of one `eri3_block` call against
/// memory; the values do not depend on it. 512 MiB is 127 aux rows at
/// nao = 724.
pub const GRADIENT_3C_BLOCK_CAP_BYTES: usize = 512 * 1024 * 1024;

/// Aux rows per one-pass block: [`GRADIENT_3C_BLOCK_CAP_BYTES`] or a quarter
/// of the budget, whichever is smaller, and at least one row (a whole shell
/// is always admitted by [`aux_shell_blocks`]).
fn gradient_block_rows(nao: usize, budget_bytes: usize) -> usize {
    let row = nao.saturating_mul(nao).saturating_mul(8).max(1);
    ((budget_bytes / 4).min(GRADIENT_3C_BLOCK_CAP_BYTES) / row).max(1)
}

/// Split the aux functions into contiguous `[p0, p1)` blocks of whole shells,
/// each at most `max_rows` rows unless a single shell is wider. Never splitting
/// a shell keeps `eri3_block` from computing a straddling shell twice.
fn aux_shell_blocks(offsets: &[usize], dims: &[usize], max_rows: usize) -> Vec<(usize, usize)> {
    let mut blocks = Vec::new();
    let mut start: Option<usize> = None;
    let mut end = 0;
    for (&off, &dim) in offsets.iter().zip(dims) {
        match start {
            Some(p0) if off + dim - p0 > max_rows => {
                blocks.push((p0, end));
                start = Some(off);
            }
            None => start = Some(off),
            _ => {}
        }
        end = off + dim;
    }
    if let Some(p0) = start {
        blocks.push((p0, end));
    }
    blocks
}

/// Divided difference of the truncated inverse `f(λ) = 1/λ` (λ kept), `0`
/// (dropped) — the Daleckii–Krein kernel of `∂M` (see the module doc).
#[inline]
fn truncated_inverse_divided_difference(lk: f64, kk: bool, ll: f64, kl: bool) -> f64 {
    match (kk, kl) {
        (true, true) => -1.0 / (lk * ll),
        (true, false) => 1.0 / (lk * (lk - ll)),
        (false, true) => 1.0 / (ll * (ll - lk)),
        (false, false) => 0.0,
    }
}

/// Gradient of `E_J^DF(D_J) + Σ_s α_s E_K^DF(D_s)` with respect to the nuclei,
/// at FIXED AO densities (the orbital response is the caller's Pulay term).
///
/// `op` must be the operator the energy's fit used, for both the three-centre
/// integrals and the metric (Coulomb for RI-J; erfc/erf for RSH exchange).
/// `j_density` is the TOTAL density for RI-J, or `None`; channels with
/// `alpha == 0` are skipped. Returns `(natoms, 3)`.
///
/// Deterministic: both derivative passes map over auxiliary shells, collect in
/// shell order, and fold serially, so the result is bit-identical across
/// thread counts.
///
/// The RI-J fitted density `d_P` is formed from f64 integrals: the gradient of
/// an SCF whose RI-J tensor was f64 ([`df_jk_gradient_at_precision`] with
/// `RiJPrecision::F64`, bit-identical).
#[allow(clippy::too_many_arguments)]
pub fn df_jk_gradient(
    natoms: usize,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    j_density: Option<&Array2<f64>>,
    k_channels: &[DfKChannel<'_>],
    budget_bytes: usize,
) -> Result<Array2<f64>, FerricError> {
    df_jk_gradient_at_precision(
        natoms,
        obs,
        dfbs,
        op,
        j_density,
        k_channels,
        budget_bytes,
        RiJPrecision::F64,
    )
}

/// [`df_jk_gradient`] for an SCF whose RI-J tensor was stored at
/// `j_precision`. With f32 the RI-J fitted density `d̂_P = Σ f64(fl32(B))·D`
/// is formed from the SAME rounded integrals the SCF's `DfJ` stored (each
/// f64 integral rounded to nearest, then contracted in f64), so `ĉ = V⁻¹d̂`
/// is the SCF's own fit coefficient vector and the gradient is that of the
/// rounded-B functional `E'_J = ½d̂ᵀV⁻¹d̂` the SCF minimised (derivative
/// integrals and the metric derivative stay f64: `fl32(B(R))` is treated as
/// `B(R)(1+δ)` with `δ` fixed, so its derivative is `∂B` to `u32` relative).
/// Exchange channels are unaffected (DF-K is always f64). The pair screen of
/// the SCF tensor is not reproduced (it drops pairs below `integral_thresh`,
/// as for an f64 tensor).
#[allow(clippy::too_many_arguments)]
pub fn df_jk_gradient_at_precision(
    natoms: usize,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    j_density: Option<&Array2<f64>>,
    k_channels: &[DfKChannel<'_>],
    budget_bytes: usize,
    j_precision: RiJPrecision,
) -> Result<Array2<f64>, FerricError> {
    let j_f32 = j_precision == RiJPrecision::F32;
    let n = obs.nbasis();
    let naux = dfbs.nbasis();
    let mut grad = Array2::<f64>::zeros((natoms, 3));

    let factors: Vec<(Array2<f64>, f64)> = k_channels
        .iter()
        .filter(|c| c.alpha != 0.0)
        .map(|c| Ok((density_factor(c.density)?, c.alpha)))
        .collect::<Result<Vec<_>, FerricError>>()?
        .into_iter()
        .filter(|(x, _)| x.ncols() > 0)
        .collect();
    if j_density.is_none() && factors.is_empty() {
        return Ok(grad);
    }

    // --- memory plane: everything resident beyond the budget-bounded source.
    // (naux, m²) per channel for L/Lt and then Γ (two live at a time), plus
    // five (naux, naux) metric matrices (V, U, Ỹ, H and one GEMM temporary).
    let k_bytes: usize = factors
        .iter()
        .map(|(x, _)| 2 * naux * x.ncols() * x.ncols() * 8)
        .sum();
    let _charge = ferric_core::memory::pool::reserve_global(
        "DF-JK gradient metric + exchange intermediates",
        k_bytes.saturating_add(5 * naux * naux * 8),
    )?;

    let t_st = ferric_core::timing::Stage::start("dfg:metric2c");
    let v = coulomb_metric_2c(op, dfbs)?;
    t_st.end();
    let t_st = ferric_core::timing::Stage::start("dfg:3c-pass");

    // ---- one pass over the raw (P|μν): d_P for J, L^s for K ----
    // The tensor is read exactly once, so it is computed one aux-shell-aligned
    // block at a time and never stored: storing it (the old
    // `ThreeIndexSource`, in core when it fit the budget) cost the same
    // integrals plus a resident copy of the whole tensor — 7.6 GB packed for
    // danuglipron/def2-SVP (nao 724, naux 3635), the job's peak. Each row's
    // values are the same `eri3_block` writes (write-once per element), and
    // every row is contracted independently, so the result does not depend
    // on the blocking.
    let blocks = aux_shell_blocks(
        dfbs.shell_offsets(),
        dfbs.shell_dims(),
        gradient_block_rows(n, budget_bytes),
    );
    let block_bytes = blocks
        .iter()
        .map(|&(p0, p1)| p1 - p0)
        .max()
        .unwrap_or(0)
        .saturating_mul(n)
        .saturating_mul(n)
        .saturating_mul(8);
    let block_charge = ferric_core::memory::pool::reserve_global(
        "DF-JK gradient 3-index block (computed once, not stored)",
        block_bytes,
    )?;
    let mut d_p = Array1::<f64>::zeros(naux);
    let mut ls: Vec<Array2<f64>> = factors
        .iter()
        .map(|(x, _)| Array2::<f64>::zeros((naux, x.ncols() * x.ncols())))
        .collect();
    for &(p0, p1) in &blocks {
        let block = ferric_integrals::threeindex::eri3_block(op, obs, dfbs, p0, p1)?;
        let b = p1 - p0;
        let data = &block;
        if let Some(dj) = j_density {
            // Disjoint writes per aux row: order-independent, bit-identical.
            let vals: Vec<f64> = (0..b)
                .into_par_iter()
                .map(|q| {
                    let row = data.index_axis(Axis(0), q);
                    if j_f32 {
                        row.iter()
                            .zip(dj.iter())
                            .map(|(&a, c)| f64::from(a as f32) * c)
                            .sum::<f64>()
                    } else {
                        row.iter().zip(dj.iter()).map(|(a, c)| a * c).sum::<f64>()
                    }
                })
                .collect();
            for (q, val) in vals.into_iter().enumerate() {
                d_p[p0 + q] = val;
            }
        }
        for ((x, _), l) in factors.iter().zip(ls.iter_mut()) {
            // L_P = Xᵀ (P|··) X, one (m, m) matrix per aux row.
            let rows: Vec<Array2<f64>> = (0..b)
                .into_par_iter()
                .map(|q| {
                    let bp = data.index_axis(Axis(0), q);
                    let t = bp.dot(x);
                    x.t().dot(&t)
                })
                .collect();
            for (q, lpq) in rows.into_iter().enumerate() {
                // `iter()` is logical (row-major) order whatever the memory
                // layout `dot` produced.
                for (dst, src) in l.row_mut(p0 + q).iter_mut().zip(lpq.iter()) {
                    *dst = *src;
                }
            }
        }
    }
    drop(block_charge);
    t_st.end();
    let t_st = ferric_core::timing::Stage::start("dfg:metric-weights");

    // ---- two-centre weight H (Σ_PQ H_PQ ∂V_PQ) and J coefficients ----
    let mut h2 = Array2::<f64>::zeros((naux, naux));
    let c_j: Option<Array1<f64>> = if j_density.is_some() {
        // Same Cholesky solve as DfJ: the energy used the exact inverse.
        let chol = v.factorizec(UPLO::Lower).map_err(|e| {
            FerricError::Lapack(format!(
                "DF-J gradient: Cholesky of the Coulomb metric failed (V is not \
                 numerically positive definite): {e}"
            ))
        })?;
        let c = chol
            .solvec(&d_p)
            .map_err(|e| FerricError::Lapack(format!("DF-J gradient: metric solve failed: {e}")))?;
        for p in 0..naux {
            for q in 0..naux {
                h2[(p, q)] -= 0.5 * c[p] * c[q];
            }
        }
        Some(c)
    } else {
        None
    };

    // Exchange: Γ^s (naux, m²) in the ORIGINAL aux basis, plus the metric term.
    let mut gammas: Vec<Array2<f64>> = Vec::with_capacity(factors.len());
    if !factors.is_empty() {
        // Divide and conquer (`dsyevd`), the same solver as DfK's
        // `v_inv_sqrt_lindep` (`dsyev` MEASURED 3.5 s on one core against
        // 0.65 s for this call at naux = 1538, sulfamethoxazole,
        // def2-universal-jkfit). The gradient depends on the eigenbasis only
        // through which modes sit above or below the energy's cut /
        // DFK_GRADIENT_NOISE_FLOOR, and the energy's cut (at most
        // LINDEP_CUT_WINDOW · 1e-10 = 1e-8, below the floor) enters the
        // gradient only through the band-energy diagnostic.
        let t_e = ferric_core::timing::Stage::start("dfg:metric-eigh");
        let (lam, u) = with_blas_threads(opt_in_blas_threads(), || {
            ferric_core::linalg::eigh_dc(&v, ferric_core::linalg::Uplo::Upper)
        })
        .map_err(|e| FerricError::Lapack(format!("DF-K gradient: V eigh failed: {e}")))?;
        t_e.end();
        let t_e = ferric_core::timing::Stage::start("dfg:metric-gemms");
        // The energy's cut, by the rule DfK applies (`metric_lindep_cut`).
        let cut_e = metric_lindep_cut(&lam);
        let tau_g = cut_e.cut.max(DFK_GRADIENT_NOISE_FLOOR);
        let keep_e: Vec<bool> = lam.iter().map(|&l| cut_e.keeps(l)).collect();
        let keep_g: Vec<bool> = lam.iter().map(|&l| l >= tau_g).collect();

        // These four products are O(naux² m²) and O(naux³) (~76 GFLOP for a
        // 28-atom def2-SVP hybrid) and run outside any rayon region, so they
        // are column-blocked over the rayon pool (`par_gemm`), one-thread BLAS
        // per block.
        let mut y_t = Array2::<f64>::zeros((naux, naux));
        let mut band_energy = 0.0f64;
        for ((_, alpha), l) in factors.iter().zip(ls.drain(..)) {
            // Lt = Uᵀ L: the fitted integrals in the metric eigenbasis.
            let mut lt = Array2::<f64>::zeros(l.dim());
            par_gemm(1.0, &u.t(), &l.view(), 0.0, &mut lt);
            drop(l);
            for k in 0..naux {
                if keep_e[k] && !keep_g[k] {
                    band_energy += alpha * lt.row(k).iter().map(|v| v * v).sum::<f64>() / lam[k];
                }
            }
            // Ỹ += α Lt Ltᵀ
            par_gemm(*alpha, &lt.view(), &lt.t(), 1.0, &mut y_t);
            // Γ = U diag(w) Lt, w_k = 1/λ_k on gradient-kept modes, else 0.
            let mut wlt = lt;
            for k in 0..naux {
                let w = if keep_g[k] { 1.0 / lam[k] } else { 0.0 };
                wlt.row_mut(k).mapv_inplace(|v| v * w);
            }
            let mut gam = Array2::<f64>::zeros(wlt.dim());
            par_gemm(1.0, &u.view(), &wlt.view(), 0.0, &mut gam);
            gammas.push(gam);
        }
        if band_energy.abs() > NOISE_BAND_ENERGY_WARN {
            eprintln!(
                "[ferric] warning: DF-K gradient: {band_energy:.3e} Ha of fitted exchange sits in \
                 auxiliary-metric modes with eigenvalue in [{:.2e}, \
                 {DFK_GRADIENT_NOISE_FLOOR:.0e}); the gradient treats them as dropped, so it \
                 differs from the derivative of the reported energy by the derivative of that \
                 band",
                cut_e.cut
            );
        }
        // H_K = U (F ∘ Ỹ) Uᵀ
        for k in 0..naux {
            for l in 0..naux {
                y_t[(k, l)] *=
                    truncated_inverse_divided_difference(lam[k], keep_g[k], lam[l], keep_g[l]);
            }
        }
        let mut uy = Array2::<f64>::zeros((naux, naux));
        par_gemm(1.0, &u.view(), &y_t.view(), 0.0, &mut uy);
        drop(y_t);
        // h2 += (U F̃) Uᵀ, accumulated in place.
        par_gemm(1.0, &uy.view(), &u.t(), 1.0, &mut h2);
        t_e.end();
    }

    t_st.end();
    let t_st = ferric_core::timing::Stage::start("dfg:3c-deriv-pass");
    // ---- three-centre derivative pass, one auxiliary shell per task ----
    let nsh_obs = obs.nshells();
    let nsh_df = dfbs.nshells();
    let dims_obs = obs.shell_dims();
    let offs_obs = obs.shell_offsets();
    let dims_df = dfbs.shell_dims();
    let offs_df = dfbs.shell_offsets();
    let sh2at_obs = obs.shell_to_atom();
    let sh2at_df = dfbs.shell_to_atom();

    let max_np = dims_df.iter().copied().max().unwrap_or(0);
    let workers = rayon::current_num_threads().max(1);
    let _slab_charge = ferric_core::memory::pool::reserve_global(
        "DF-JK gradient per-worker 3c weight slabs",
        workers
            .saturating_mul(max_np)
            .saturating_mul(n)
            .saturating_mul(n)
            .saturating_mul(8),
    )?;

    // Surface an engine-construction error serially before the fan-out.
    Engine::new_3center_deriv(op, obs, dfbs, 1e-14)?;
    let partials: Vec<Array2<f64>> = (0..nsh_df)
        .into_par_iter()
        .map_init(
            || {
                Engine::new_3center_deriv(op, obs, dfbs, 1e-14)
                    .expect("3-center deriv engine (pre-validated)")
            },
            |eng, sp| {
                let np = dims_df[sp];
                let pf0 = offs_df[sp];
                let mut local = Array2::<f64>::zeros((natoms, 3));
                // G[p, μ, ν] = ∂E/∂(P|μν) for this shell's aux functions.
                let mut slab = Array3::<f64>::zeros((np, n, n));
                for p in 0..np {
                    let pf = pf0 + p;
                    let mut g = slab.index_axis_mut(Axis(0), p);
                    if let (Some(c), Some(dj)) = (c_j.as_ref(), j_density) {
                        g.scaled_add(c[pf], dj);
                    }
                    for ((x, alpha), gam) in factors.iter().zip(gammas.iter()) {
                        let m = x.ncols();
                        let gp = Array2::from_shape_fn((m, m), |(i, j)| gam[(pf, i * m + j)]);
                        let xg = x.dot(&gp);
                        let contrib = xg.dot(&x.t());
                        g.scaled_add(2.0 * alpha, &contrib);
                    }
                }
                let atom_p = sh2at_df[sp];
                for s1 in 0..nsh_obs {
                    for s2 in 0..=s1 {
                        let Some(deriv) = eng.compute_eri3_deriv(obs, dfbs, sp, s1, s2) else {
                            continue;
                        };
                        let n1 = dims_obs[s1];
                        let n2 = dims_obs[s2];
                        let block_sz = np * n1 * n2;
                        let sym12 = s1 != s2;
                        let atom_1 = sh2at_obs[s1];
                        let atom_2 = sh2at_obs[s2];
                        for p in 0..np {
                            for i in 0..n1 {
                                let mu = offs_obs[s1] + i;
                                for j in 0..n2 {
                                    let nu = offs_obs[s2] + j;
                                    let idx = (p * n1 + i) * n2 + j;
                                    let gval = if sym12 {
                                        slab[(p, mu, nu)] + slab[(p, nu, mu)]
                                    } else {
                                        slab[(p, mu, nu)]
                                    };
                                    if gval == 0.0 {
                                        continue;
                                    }
                                    // 9 blocks: [dP_xyz, d1_xyz, d2_xyz].
                                    for coord in 0..3 {
                                        local[(atom_p, coord)] +=
                                            gval * deriv[coord * block_sz + idx];
                                        local[(atom_1, coord)] +=
                                            gval * deriv[(3 + coord) * block_sz + idx];
                                        local[(atom_2, coord)] +=
                                            gval * deriv[(6 + coord) * block_sz + idx];
                                    }
                                }
                            }
                        }
                    }
                }
                local
            },
        )
        .collect();
    for p in &partials {
        grad += p;
    }
    t_st.end();
    let t_st = ferric_core::timing::Stage::start("dfg:2c-deriv-pass");

    // ---- two-centre metric derivative pass ----
    Engine::new_2center_deriv(op, dfbs, 1e-14)?;
    let partials: Vec<Array2<f64>> = (0..nsh_df)
        .into_par_iter()
        .map_init(
            || {
                Engine::new_2center_deriv(op, dfbs, 1e-14)
                    .expect("2-center deriv engine (pre-validated)")
            },
            |eng, sp| {
                let mut local = Array2::<f64>::zeros((natoms, 3));
                for sq in 0..=sp {
                    let Some(deriv) = eng.compute_eri2_deriv(dfbs, sp, sq) else {
                        continue;
                    };
                    let np = dims_df[sp];
                    let nq = dims_df[sq];
                    let block_sz = np * nq;
                    let sym = sp != sq;
                    let atom_p = sh2at_df[sp];
                    let atom_q = sh2at_df[sq];
                    for p in 0..np {
                        let pf = offs_df[sp] + p;
                        for q in 0..nq {
                            let qf = offs_df[sq] + q;
                            let idx = p * nq + q;
                            let gval = if sym {
                                h2[(pf, qf)] + h2[(qf, pf)]
                            } else {
                                h2[(pf, qf)]
                            };
                            // 6 blocks: [dP_xyz, dQ_xyz].
                            for coord in 0..3 {
                                local[(atom_p, coord)] += gval * deriv[coord * block_sz + idx];
                                local[(atom_q, coord)] +=
                                    gval * deriv[(3 + coord) * block_sz + idx];
                            }
                        }
                    }
                }
                local
            },
        )
        .collect();
    for p in &partials {
        grad += p;
    }
    t_st.end();
    Ok(grad)
}

/// The densities entering the two-electron energy.
#[derive(Clone, Copy)]
pub enum TwoElectronDensity<'a> {
    /// Closed shell: the TOTAL density `D = 2 C_occ C_occᵀ`.
    Closed(&'a Array2<f64>),
    /// Open shell (UHF/ROHF/UKS/ROKS): per-spin densities.
    Open {
        alpha: &'a Array2<f64>,
        beta: &'a Array2<f64>,
    },
}

/// Exact-exchange content of the energy: `c_k` scales ω = 0 exchange; `rsh =
/// Some((c_sr, c_lr, ω))` replaces it with `c_sr·K[erfc(ω)] + c_lr·K[erf(ω)]`.
#[derive(Clone, Copy, Debug)]
pub struct ExchangeMix {
    pub c_k: f64,
    pub rsh: Option<(f64, f64, f64)>,
}

impl ExchangeMix {
    /// Hartree-Fock: full ω = 0 exchange.
    pub fn hartree_fock() -> Self {
        ExchangeMix {
            c_k: 1.0,
            rsh: None,
        }
    }
    /// From a functional's exchange mix (same convention as the SCF's F assembly).
    pub fn from_k_mix(k_mix: &ferric_dft::xc_trait::KMix) -> Self {
        ExchangeMix {
            c_k: k_mix.sr,
            rsh: (k_mix.omega > 0.0).then_some((k_mix.sr, k_mix.lr, k_mix.omega)),
        }
    }
}

/// Exchange channels of the SCF energy convention: a closed-shell total
/// density carries `−c/4 Σ D D (μλ|νσ)`, each open-shell spin density
/// `−c/2 Σ D_σ D_σ (μλ|νσ)` (the Γ of the four-centre loops, exactly).
fn exchange_channels<'a>(dens: TwoElectronDensity<'a>, scale: f64) -> Vec<DfKChannel<'a>> {
    match dens {
        TwoElectronDensity::Closed(d) => vec![DfKChannel {
            density: d,
            alpha: -0.25 * scale,
        }],
        TwoElectronDensity::Open { alpha, beta } => vec![
            DfKChannel {
                density: alpha,
                alpha: -0.5 * scale,
            },
            DfKChannel {
                density: beta,
                alpha: -0.5 * scale,
            },
        ],
    }
}

pub(crate) fn aux_basis(mol: &Molecule, name: &str) -> Result<PreparedBasis, FerricError> {
    let bs = ferric_core::basis::bundled(name)?;
    PreparedBasis::new(mol, &bs)
}

/// RI-J gradient `∂E_J^DF(D)/∂R` alone, with the aux basis, operator and
/// budget the SCF recorded in `route` — the Coulomb half of a RIJCOSX
/// gradient (exchange from COSX, see `crate::cosx_gradient`). Errors if the
/// route has no RI-J (the caller must then use the four-centre J gradient).
pub(crate) fn df_j_gradient(
    mol: &Molecule,
    prep: &PreparedBasis,
    route: &DfJkRoute,
    d: &Array2<f64>,
) -> Result<Array2<f64>, FerricError> {
    let ja = route.j_aux.as_deref().ok_or_else(|| {
        FerricError::General("df_j_gradient: the SCF route has no RI-J aux basis".into())
    })?;
    let dfbs = aux_basis(mol, ja)?;
    df_jk_gradient_at_precision(
        mol.atoms.len(),
        prep,
        &dfbs,
        route.op,
        Some(d),
        &[],
        route.budget_bytes,
        route.ri_j_precision,
    )
}

/// Two-electron (J + exact-exchange) gradient of the energy an SCF computed,
/// routed by the builders it recorded in [`crate::result::ScfResult::df_jk`].
///
/// Every term that was density-fitted is differentiated through
/// [`df_jk_gradient`] with the recorded aux basis, operator and (for RSH) ω;
/// every term that was exact goes through the existing four-centre
/// derivative loops. Only called when a route is present: an all-exact SCF
/// keeps its gradient's original code path untouched.
#[allow(clippy::too_many_arguments)]
pub fn routed_two_electron_gradient(
    mol: &Molecule,
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    dens: TwoElectronDensity<'_>,
    route: &DfJkRoute,
    mix: ExchangeMix,
) -> Result<Array2<f64>, FerricError> {
    use crate::ks_gradient::{
        twoelectron_gradient_scaled_k, twoelectron_gradient_uhf_scaled_k, twoelectron_k_gradient,
        twoelectron_k_gradient_uhf,
    };
    let natoms = mol.atoms.len();
    let d_total: Array2<f64> = match dens {
        TwoElectronDensity::Closed(d) => d.clone(),
        TwoElectronDensity::Open { alpha, beta } => alpha + beta,
    };
    let channels = |scale: f64| exchange_channels(dens, scale);
    // Exact four-centre J (+ c·K) and K-only pieces, same Γ as before.
    let exact_jk = |c: f64| -> Result<Array2<f64>, FerricError> {
        match dens {
            TwoElectronDensity::Closed(d) => twoelectron_gradient_scaled_k(prep, op, bounds, d, c),
            TwoElectronDensity::Open { alpha, beta } => {
                twoelectron_gradient_uhf_scaled_k(prep, op, bounds, &d_total, alpha, beta, c)
            }
        }
    };
    let exact_k = |kop: Operator, c: f64| -> Result<Array2<f64>, FerricError> {
        match dens {
            TwoElectronDensity::Closed(d) => twoelectron_k_gradient(prep, kop, bounds, d, c),
            TwoElectronDensity::Open { alpha, beta } => {
                twoelectron_k_gradient_uhf(prep, kop, bounds, alpha, beta, c)
            }
        }
    };

    let j_df = route.j_aux.as_deref();
    let k0_needed = mix.rsh.is_none() && mix.c_k != 0.0;
    let k0_df = if k0_needed {
        route.k_aux.as_deref()
    } else {
        None
    };
    let k0_exact = k0_needed && k0_df.is_none();

    let mut grad = Array2::<f64>::zeros((natoms, 3));
    match (j_df.is_some(), k0_exact) {
        (false, true) => grad += &exact_jk(mix.c_k)?,
        (false, false) => grad += &exact_jk(0.0)?,
        (true, true) => grad += &exact_k(op, mix.c_k)?,
        (true, false) => {}
    }

    // ω = 0 fitted pieces; one pass when J and K share the aux basis.
    let budget = route.budget_bytes;
    match (j_df, k0_df) {
        (Some(ja), Some(ka)) if ja == ka => {
            let dfbs = aux_basis(mol, ja)?;
            grad += &df_jk_gradient_at_precision(
                natoms,
                prep,
                &dfbs,
                route.op,
                Some(&d_total),
                &channels(mix.c_k),
                budget,
                route.ri_j_precision,
            )?;
        }
        _ => {
            if let Some(ja) = j_df {
                let dfbs = aux_basis(mol, ja)?;
                grad += &df_jk_gradient_at_precision(
                    natoms,
                    prep,
                    &dfbs,
                    route.op,
                    Some(&d_total),
                    &[],
                    budget,
                    route.ri_j_precision,
                )?;
            }
            if let Some(ka) = k0_df {
                let dfbs = aux_basis(mol, ka)?;
                grad += &df_jk_gradient(
                    natoms,
                    prep,
                    &dfbs,
                    route.op,
                    None,
                    &channels(mix.c_k),
                    budget,
                )?;
            }
        }
    }

    // Range-separated exchange: fitted with the SCF's own aux basis and ω.
    if let Some((c_sr, c_lr, omega)) = mix.rsh {
        match route.rsh_k.as_ref() {
            Some((aux, omega_fit)) => {
                let dfbs = aux_basis(mol, aux)?;
                grad += &df_jk_gradient(
                    natoms,
                    prep,
                    &dfbs,
                    Operator::erfc(*omega_fit),
                    None,
                    &channels(c_sr),
                    budget,
                )?;
                grad += &df_jk_gradient(
                    natoms,
                    prep,
                    &dfbs,
                    Operator::erf(*omega_fit),
                    None,
                    &channels(c_lr),
                    budget,
                )?;
            }
            None => {
                grad += &exact_k(Operator::erfc(omega), c_sr)?;
                grad += &exact_k(Operator::erf(omega), c_lr)?;
            }
        }
    }
    Ok(grad)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `D = X Xᵀ` must hold for a rank-deficient PSD density (the RHF case).
    /// FAILS if the factor drops or mis-scales occupied directions.
    #[test]
    fn density_factor_reproduces_density() {
        let c = ndarray::array![[0.8, 0.1], [0.3, -0.5], [0.2, 0.7], [-0.4, 0.2]];
        let d = 2.0 * c.dot(&c.t());
        let x = density_factor(&d).unwrap();
        assert_eq!(x.ncols(), 2, "rank-2 density must give a rank-2 factor");
        let err = (&x.dot(&x.t()) - &d)
            .iter()
            .fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(err < 1e-13, "X Xᵀ − D = {err:.3e}");
    }

    #[test]
    fn density_factor_rejects_indefinite_matrix() {
        let d = ndarray::array![[1.0, 0.0], [0.0, -0.5]];
        assert!(density_factor(&d).is_err());
    }

    /// With every mode kept the kernel is `−1/(λ_k λ_l)`, i.e. `∂M = −M ∂V M`.
    /// FAILS if the kept/kept branch is replaced by the rotation form.
    #[test]
    fn divided_difference_reduces_to_minus_m_dv_m_when_nothing_dropped() {
        let v = truncated_inverse_divided_difference(2.0, true, 0.5, true);
        assert!((v - (-1.0)).abs() < 1e-15);
        // Kept/dropped: (1/λk − 0)/(λk − λl).
        let v = truncated_inverse_divided_difference(2.0, true, 0.5, false);
        assert!((v - 1.0 / (2.0 * 1.5)).abs() < 1e-15);
        assert_eq!(
            truncated_inverse_divided_difference(1.0, false, 2.0, false),
            0.0
        );
    }

    /// Blocks tile `[0, naux)` exactly, in order, never split a shell, and stay
    /// within `max_rows` except for a single shell wider than it. FAILS if a
    /// boundary row is dropped or duplicated (the gradient would then miss or
    /// double-count `d_P` rows) or if a shell is split.
    #[test]
    fn aux_shell_blocks_tile_whole_shells() {
        let dims = [1usize, 3, 5, 1, 7, 9, 3, 1];
        let offsets: Vec<usize> = dims
            .iter()
            .scan(0, |acc, &d| {
                let o = *acc;
                *acc += d;
                Some(o)
            })
            .collect();
        let naux: usize = dims.iter().sum();
        for max_rows in [1, 4, 8, 9, 10, 16, naux, usize::MAX] {
            let blocks = aux_shell_blocks(&offsets, &dims, max_rows);
            assert_eq!(blocks.first().unwrap().0, 0, "max_rows {max_rows}");
            assert_eq!(blocks.last().unwrap().1, naux, "max_rows {max_rows}");
            for w in blocks.windows(2) {
                assert_eq!(w[0].1, w[1].0, "gap/overlap at max_rows {max_rows}");
            }
            for &(p0, p1) in &blocks {
                assert!(offsets.contains(&p0), "block starts mid-shell");
                assert!(p1 == naux || offsets.contains(&p1), "block ends mid-shell");
                let one_shell = offsets.iter().filter(|&&o| o >= p0 && o < p1).count() == 1;
                assert!(p1 - p0 <= max_rows || one_shell, "oversized block");
            }
        }
        assert_eq!(
            aux_shell_blocks(&offsets, &dims, usize::MAX),
            vec![(0, naux)]
        );
        assert!(aux_shell_blocks(&[], &[], 4).is_empty());
    }
}
