//! Second-order (trust-region augmented-Hessian) ROHF / ROKS on the EXACT
//! orbital gradient and Hessian, stepping from the CURRENT orbitals.
//!
//! Port of `reference/pbc/pbc_rohf_newton.py` (FINDINGS "Iteration 24
//! (Python, second-order ROHF on the injected path) — 2026-09-27"). Built for
//! the injected (periodic) path ([`solve_rohf_injected_second_order`](crate::rohf_trah::solve_rohf_injected_second_order)), where
//! the DIIS loop of [`crate::rohf::solve_rohf_injected`] cannot converge the
//! triclinic 4H s+p ROHF triplet: at PySCF's minimum (`exxdiv = none`) a
//! virtual lies BELOW the second open orbital in the Roothaan `F_eff` order,
//! so that minimum is not a fixed point of an aufbau-by-`F_eff` loop, whatever
//! DIIS or level shift is used. This solver never diagonalizes `F_eff` and
//! never chooses an occupation after the guess: `C ← C·U(κ)` with κ from a
//! trust-region AH step on `E(C e^κ)`.
//!
//! # Parametrization and derivatives
//!
//! Identical to [`crate::rohf_newton`] / [`crate::rohf_ah`]: `U` = the Cayley
//! unitary of an antisymmetric MO-basis κ with ONE free parameter per
//! non-redundant pair `(p, q)`, `κ[p,q] = x`, `κ[q,p] = −x`, `p` the less
//! occupied index; packing order vc (virt × closed, row-major), vo (virt ×
//! open), oc (open × closed). Occupations `n_α = 1` on closed + open,
//! `n_β = 1` on closed. With `f_σ = Cᵀ F_σ C` and `N_σ = diag(n_σ)`:
//!
//! ```text
//! gradient     G1 = Σ_σ (N_σ f_σ − f_σ N_σ)
//! Hessian·x    G2 = Σ_σ ½[N_σ, A_σ] + ½(X_σ f_σ − f_σ X_σ) + [N_σ, δF_σ]
//!              A_σ = f_σ X − X f_σ,  X_σ = X N_σ − N_σ X,  X = unpack(x)
//!              δF_σ = Cᵀ δF_σ^AO C,  (δF_α^AO, δF_β^AO) = response(C X_α Cᵀ, C X_β Cᵀ)
//! read-off     out[(p, q)] = G[q, p] − G[p, q]
//! ```
//!
//! The true gradient is EXACTLY 2 × `rohf_newton::gradient_blocks`
//! in each of the three blocks (measured to 1e-8 by FD in the prototype, and
//! pinned by this module's tests); the true Hessian is 2 × ferric's packed
//! 2e response plus the full Fock-commutator terms, O(n³) per product and no
//! extra J/K. Both are fed to [`crate::trah::solve_trust_region`] UNSCALED,
//! so its `predicted` is a true energy change and ρ → 1 at the tail.
//!
//! # The response callback
//!
//! Everything integral-specific is behind [`RohfOrbitalModel`](crate::rohf_trah::RohfOrbitalModel): the energy and
//! spin Focks at a density, and the linear response ([`OrbitalResponse`](crate::rohf_trah::OrbitalResponse))
//!
//! ```text
//! δF_σ = J[δD_α + δD_β] − a·K[δD_σ] + δV_xc^σ
//! ```
//!
//! where `K` is the SAME exchange builder the Fock uses (so a Madelung term
//! `v_M S δD_σ S` in an injected K rides along), and `δV_xc` is a central FD
//! of `XcBuilder::build_polarized` (2 XC builds per product; step
//! `fxc_fd_step / max|δD|`). δD is symmetric but INDEFINITE and not
//! idempotent: the builder must be a plain linear contraction (the ferric-pbc
//! dense-AFT and RS-GDF J/K are; a density-magnitude screen that assumes a
//! PSD `D`, e.g. LinK's, would not be, which is one reason the molecular path
//! is not routed here).
//!
//! # Differences from the prototype's driver
//!
//! * The AH solve is [`crate::trah::solve_trust_region`] (a fresh block
//!   Davidson per α), not the prototype's one-subspace-for-all-α solver. To
//!   keep that affordable every Hessian product within a macro iteration goes
//!   through `SpanCache`: a product is formed only for the component of a
//!   request outside the span of the directions already applied, so the α
//!   search, the re-solve after a rejected step and the `predicted` product
//!   reuse products instead of rebuilding J/K. It is exact up to rounding
//!   (the Hessian is linear; the FD XC response is linear to its truncation).
//! * Rejection, radius update and the re-solve from the SAME point follow the
//!   prototype (ρ < `rho_reject` rejects and re-solves at once; the radius
//!   grows only for a step on the boundary), not [`crate::trah::TrahState`]'s
//!   two-iteration bookkeeping — the trial point's Fock is built inside the
//!   step, so ρ is always formed against the point the step was taken from.
//! * The Davidson tolerance is set per macro iteration from the gradient,
//!   `max(inner_tol_floor, inner_tol · min(1, ‖g‖) · ‖g‖)` (the prototype's
//!   κ-residual tolerance), overriding `trah.davidson_conv`.
//!
//! Units: Hartree.

use crate::fock::{JBuilder, KBuilder};
use crate::result::{ScfExit, ScfResult, Spin};
use crate::rhf::{PeriodicInjection, RhfConfig, XcBuilder};
use crate::stability::{davidson_lowest, StabilityConfig};
use crate::trah::{solve_trust_region, TrahConfig};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ndarray::Array2;
use ndarray_linalg::{Eigh, Inverse, UPLO};
use std::cell::{Cell, RefCell};

/// Linear response of the spin Focks to a density perturbation, at a fixed
/// reference density: `(δD_α, δD_β) ↦ (δF_α, δF_β)`, all AO. `δD_σ` is
/// symmetric but indefinite. See the module doc for the formula.
pub type OrbitalResponse<'a> =
    dyn Fn(&Array2<f64>, &Array2<f64>) -> Result<(Array2<f64>, Array2<f64>), FerricError> + 'a;

/// What the second-order ROHF/ROKS driver needs from the integrals: the
/// energy and spin Focks at a density, and the Fock response at a reference
/// density. `&self` methods: an implementation with mutable builders uses
/// interior mutability (the driver never calls the two re-entrantly).
pub trait RohfOrbitalModel {
    /// AO overlap (for dimensions and the result; the rotation keeps `CᵀSC`).
    fn overlap(&self) -> &Array2<f64>;
    /// `(E_total, F_α, F_β)` at `(D_α, D_β)`; `F_σ` INCLUDES `V_xc^σ`.
    #[allow(clippy::type_complexity)]
    fn energy_and_focks(
        &self,
        d_a: &Array2<f64>,
        d_b: &Array2<f64>,
    ) -> Result<(f64, Array2<f64>, Array2<f64>), FerricError>;
    /// `(δF_α, δF_β)` for `(δD_α, δD_β)` at the reference `(D_α, D_β)`.
    fn response(
        &self,
        d_a: &Array2<f64>,
        d_b: &Array2<f64>,
        dd_a: &Array2<f64>,
        dd_b: &Array2<f64>,
    ) -> Result<(Array2<f64>, Array2<f64>), FerricError>;
}

/// Controls for [`rohf_second_order`] / [`solve_rohf_injected_second_order`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RohfTrahConfig {
    /// Trust region (radius0 0.4, Fletcher 0.7 / 1.2 at ρ 0.25 / 0.75,
    /// reject at ρ < 0, α on log α in [1, 1000]). `davidson_conv` is
    /// overridden per macro iteration (module doc).
    pub trah: TrahConfig,
    /// Converged when max |packed gradient| (ferric's `gradient_blocks`
    /// scale = ½ the true gradient) is below this. Default 1e-9.
    pub grad_tol: f64,
    /// Macro-iteration (Fock-point) cap. Default 100.
    pub max_macro: usize,
    /// Relative inner (Davidson) tolerance factor. Default 1e-2.
    pub inner_tol: f64,
    /// Floor of the inner tolerance. Default 1e-11.
    pub inner_tol_floor: f64,
    /// Central-FD XC response step, scaled by `1 / max|δD|`. Default 1e-4
    /// (the prototype's; H symmetric to 1e-7..1e-10 with it).
    pub fxc_fd_step: f64,
    /// At convergence, compute the lowest exact orbital-Hessian eigenvalue
    /// ([`RohfSecondOrderInfo::lowest_hessian_eigenvalue`]; > 0 ⇔ a true
    /// local minimum). Costs a Davidson on the Hessian. Default false.
    pub check_minimum: bool,
    /// Memory budget for the product cache and the n×n working set; `None`
    /// = the caller's `RhfConfig::three_index_budget_bytes` if set, else
    /// ferric's unified default.
    pub budget_bytes: Option<usize>,
}

impl Default for RohfTrahConfig {
    fn default() -> Self {
        Self {
            trah: TrahConfig::default(),
            grad_tol: 1e-9,
            max_macro: 100,
            inner_tol: 1e-2,
            inner_tol_floor: 1e-11,
            fxc_fd_step: 1e-4,
            check_minimum: false,
            budget_bytes: None,
        }
    }
}

/// Diagnostics of one second-order run.
#[derive(Debug, Clone, PartialEq)]
pub struct RohfSecondOrderInfo {
    /// Whether max |packed gradient| < `grad_tol` was reached.
    pub converged: bool,
    /// Macro iterations (points at which the gradient was tested).
    pub macro_iterations: usize,
    /// Fock builds (guess point + every trial point, accepted or not).
    pub fock_builds: usize,
    /// Hessian products that called the response (cache misses).
    pub hessian_products: usize,
    /// Accepted steps.
    pub accepted: usize,
    /// Steps rejected by the ρ test (each re-solved from the same point).
    pub rejected: usize,
    /// ρ of the last trial step.
    pub last_rho: Option<f64>,
    /// max |packed gradient| at the returned point.
    pub final_gradient_max: f64,
    /// Trust radius at exit.
    pub final_radius: f64,
    /// Lowest exact orbital-Hessian eigenvalue (TRUE scale) at the returned
    /// point, when `check_minimum` was set and the Davidson converged.
    pub lowest_hessian_eigenvalue: Option<f64>,
}

/// One determinant: MOs (closed | open | virtual), densities, spin Focks
/// (AO, incl. XC) and their MO transforms, total energy.
#[derive(Debug, Clone)]
pub struct RohfPoint {
    pub c: Array2<f64>,
    pub d_a: Array2<f64>,
    pub d_b: Array2<f64>,
    pub f_a: Array2<f64>,
    pub f_b: Array2<f64>,
    pub f_a_mo: Array2<f64>,
    pub f_b_mo: Array2<f64>,
    pub energy: f64,
}

impl RohfPoint {
    /// Energy and Focks at `c` (one Fock build).
    pub fn at(
        model: &dyn RohfOrbitalModel,
        c: Array2<f64>,
        nocc_double: usize,
        nocc_open: usize,
    ) -> Result<Self, FerricError> {
        let (d_a, d_b) = crate::rohf::build_rohf_densities(&c, nocc_double, nocc_open);
        let (energy, f_a, f_b) = model.energy_and_focks(&d_a, &d_b)?;
        let f_a_mo = c.t().dot(&f_a).dot(&c);
        let f_b_mo = c.t().dot(&f_b).dot(&c);
        Ok(Self {
            c,
            d_a,
            d_b,
            f_a,
            f_b,
            f_a_mo,
            f_b_mo,
            energy,
        })
    }
}

/// Number of non-redundant ROHF rotations: `nv·nd + nv·no + no·nd`.
pub fn rohf_rotation_count(n: usize, nocc_double: usize, nocc_open: usize) -> usize {
    let na = nocc_double + nocc_open;
    let nv = n.saturating_sub(na);
    nv * nocc_double + nv * nocc_open + nocc_open * nocc_double
}

/// The packed pairs `(p, q)` in ferric's order: vc, vo, oc (row-major).
fn pairs(n: usize, nd: usize, no: usize) -> Vec<(usize, usize)> {
    let na = nd + no;
    let mut out = Vec::with_capacity(rohf_rotation_count(n, nd, no));
    for p in na..n {
        for q in 0..nd {
            out.push((p, q));
        }
    }
    for p in na..n {
        for q in nd..na {
            out.push((p, q));
        }
    }
    for p in nd..na {
        for q in 0..nd {
            out.push((p, q));
        }
    }
    out
}

/// Antisymmetric κ from the packed vector.
fn unpack(x: &[f64], n: usize, nd: usize, no: usize) -> Array2<f64> {
    let mut k = Array2::<f64>::zeros((n, n));
    for (&v, (p, q)) in x.iter().zip(pairs(n, nd, no)) {
        k[(p, q)] = v;
        k[(q, p)] = -v;
    }
    k
}

/// The TRUE orbital gradient `dE/dx` (= 2 × ferric's packed
/// `gradient_blocks` in every block): vc `2(f_α + f_β)[p,q]`, vo
/// `2 f_α[p,q]`, oc `2 f_β[p,q]`.
pub fn rohf_exact_gradient(
    f_a_mo: &Array2<f64>,
    f_b_mo: &Array2<f64>,
    nocc_double: usize,
    nocc_open: usize,
) -> Vec<f64> {
    let n = f_a_mo.nrows();
    let (nd, na) = (nocc_double, nocc_double + nocc_open);
    pairs(n, nocc_double, nocc_open)
        .into_iter()
        .map(|(p, q)| {
            if p >= na && q < nd {
                2.0 * (f_a_mo[(p, q)] + f_b_mo[(p, q)])
            } else if p >= na {
                2.0 * f_a_mo[(p, q)]
            } else {
                2.0 * f_b_mo[(p, q)]
            }
        })
        .collect()
}

/// Hessian-diagonal approximation (TRUE scale, 2 × ferric's per-spin gaps),
/// used only as the Davidson preconditioner.
pub fn rohf_hessian_diagonal(
    f_a_mo: &Array2<f64>,
    f_b_mo: &Array2<f64>,
    nocc_double: usize,
    nocc_open: usize,
) -> Vec<f64> {
    let n = f_a_mo.nrows();
    let (nd, na) = (nocc_double, nocc_double + nocc_open);
    let fa = |i: usize| f_a_mo[(i, i)];
    let fb = |i: usize| f_b_mo[(i, i)];
    pairs(n, nocc_double, nocc_open)
        .into_iter()
        .map(|(p, q)| {
            if p >= na && q < nd {
                2.0 * ((fa(p) + fb(p)) - (fa(q) + fb(q)))
            } else if p >= na {
                2.0 * (fa(p) - fa(q))
            } else {
                2.0 * (fb(p) - fb(q))
            }
        })
        .collect()
}

/// The EXACT orbital Hessian (TRUE scale) applied to the packed `x` at the
/// determinant `c` with MO spin Focks `f_a_mo`/`f_b_mo` (module doc for the
/// formula). One `response` call; the rest is O(n³) MO algebra.
#[allow(clippy::too_many_arguments)]
pub fn rohf_exact_hessian_matvec(
    c: &Array2<f64>,
    f_a_mo: &Array2<f64>,
    f_b_mo: &Array2<f64>,
    nocc_double: usize,
    nocc_open: usize,
    response: &OrbitalResponse<'_>,
    x: &[f64],
) -> Result<Vec<f64>, FerricError> {
    let n = c.ncols();
    let (nd, na) = (nocc_double, nocc_double + nocc_open);
    let np = rohf_rotation_count(n, nd, nocc_open);
    if x.len() != np {
        return Err(FerricError::General(format!(
            "rohf_exact_hessian_matvec: vector length {} != {np} rotations",
            x.len()
        )));
    }
    let occ_a: Vec<f64> = (0..n).map(|i| if i < na { 1.0 } else { 0.0 }).collect();
    let occ_b: Vec<f64> = (0..n).map(|i| if i < nd { 1.0 } else { 0.0 }).collect();
    let xk = unpack(x, n, nd, nocc_open);
    // X_σ = X N_σ − N_σ X: element (i, j) = X[i,j] (n_j − n_i).
    let xs = |occ: &[f64]| {
        let mut m = xk.clone();
        for ((i, j), v) in m.indexed_iter_mut() {
            *v *= occ[j] - occ[i];
        }
        m
    };
    let xs_a = xs(occ_a.as_slice());
    let xs_b = xs(occ_b.as_slice());
    let dd_a = c.dot(&xs_a).dot(&c.t());
    let dd_b = c.dot(&xs_b).dot(&c.t());
    let (df_a, df_b) = response(&dd_a, &dd_b)?;
    let mut g = Array2::<f64>::zeros((n, n));
    for (occ, f, xsig, df_ao) in [
        (&occ_a, f_a_mo, &xs_a, &df_a),
        (&occ_b, f_b_mo, &xs_b, &df_b),
    ] {
        let df = c.t().dot(df_ao).dot(c);
        let a = f.dot(&xk) - xk.dot(f);
        let b = xsig.dot(f) - f.dot(xsig);
        for ((i, j), v) in g.indexed_iter_mut() {
            let dn = occ[i] - occ[j];
            *v += 0.5 * dn * a[(i, j)] + 0.5 * b[(i, j)] + dn * df[(i, j)];
        }
    }
    Ok(pairs(n, nd, nocc_open)
        .into_iter()
        .map(|(p, q)| g[(q, p)] - g[(p, q)])
        .collect())
}

/// `C · U`, `U` the Cayley unitary `(I − κ/2)⁻¹ (I + κ/2)` of the packed κ.
pub fn rohf_cayley_rotate(
    c: &Array2<f64>,
    kappa: &[f64],
    nocc_double: usize,
    nocc_open: usize,
) -> Result<Array2<f64>, FerricError> {
    let n = c.ncols();
    let half = 0.5 * &unpack(kappa, n, nocc_double, nocc_open);
    let eye = Array2::<f64>::eye(n);
    let inv = (&eye - &half)
        .inv()
        .map_err(|e| FerricError::Lapack(format!("ROHF second-order Cayley inverse: {e}")))?;
    Ok(c.dot(&inv.dot(&(&eye + &half))))
}

/// Product cache over one macro iteration (module doc). Keeps an
/// orthonormal basis `Q` of the directions already applied and `H Q`; a
/// request `v = Q c + r` costs one real product only for `r / ‖r‖` when
/// `‖r‖ > 1e-12 ‖v‖`, and none otherwise. Once `cap` directions are held,
/// further out-of-span requests are applied directly and not stored.
struct SpanCache {
    q: Vec<Vec<f64>>,
    hq: Vec<Vec<f64>>,
    cap: usize,
    misses: usize,
}

impl SpanCache {
    /// Relative out-of-span threshold below which a request is served from
    /// the cache (orthogonalization noise is ~1e-16 · √m).
    const REL_TOL: f64 = 1e-12;

    fn new(cap: usize) -> Self {
        Self {
            q: Vec::new(),
            hq: Vec::new(),
            cap,
            misses: 0,
        }
    }

    fn apply(
        &mut self,
        v: &[f64],
        raw: &dyn Fn(&[f64]) -> Result<Vec<f64>, FerricError>,
    ) -> Result<Vec<f64>, FerricError> {
        let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
        let vn = dot(v, v).sqrt();
        let m = v.len();
        if vn == 0.0 {
            return Ok(vec![0.0; m]);
        }
        let mut r = v.to_vec();
        let mut coef = vec![0.0f64; self.q.len()];
        for _ in 0..2 {
            for (ci, qi) in coef.iter_mut().zip(&self.q) {
                let c = dot(qi, &r);
                *ci += c;
                for (rk, qk) in r.iter_mut().zip(qi) {
                    *rk -= c * qk;
                }
            }
        }
        let rn = dot(&r, &r).sqrt();
        if rn > Self::REL_TOL * vn && self.q.len() >= self.cap {
            self.misses += 1;
            return raw(v);
        }
        let mut out = vec![0.0f64; m];
        for (ci, hqi) in coef.iter().zip(&self.hq) {
            for (ok, hk) in out.iter_mut().zip(hqi) {
                *ok += ci * hk;
            }
        }
        if rn > Self::REL_TOL * vn {
            let qn: Vec<f64> = r.iter().map(|x| x / rn).collect();
            let hqn = raw(&qn)?;
            self.misses += 1;
            for (ok, hk) in out.iter_mut().zip(&hqn) {
                *ok += rn * hk;
            }
            self.q.push(qn);
            self.hq.push(hqn);
        }
        Ok(out)
    }
}

/// Outcome of one macro step.
enum Step {
    Accepted(Box<RohfPoint>),
    Collapsed,
}

/// Trust-region AH ROHF/ROKS from `c0` (occupied = the first
/// `nocc_double + nocc_open` columns, S-orthonormal) on `model`. Returns the
/// last point (the converged one when `info.converged`) and diagnostics.
/// Does not error on non-convergence; errors on a failed Fock/response build
/// or AH solve.
pub fn rohf_second_order(
    ctx: &ParallelContext,
    model: &dyn RohfOrbitalModel,
    c0: &Array2<f64>,
    nocc_double: usize,
    nocc_open: usize,
    cfg: &RohfTrahConfig,
) -> Result<(RohfPoint, RohfSecondOrderInfo), FerricError> {
    let (nd, no) = (nocc_double, nocc_open);
    let n = c0.ncols();
    if c0.nrows() != model.overlap().nrows() || n != model.overlap().nrows() {
        return Err(FerricError::General(format!(
            "rohf_second_order: MO shape {:?} does not match the overlap {:?}",
            c0.dim(),
            model.overlap().dim()
        )));
    }
    if nd + no > n {
        return Err(FerricError::General(format!(
            "rohf_second_order: {} occupied orbitals exceed {n} MOs",
            nd + no
        )));
    }
    let np = rohf_rotation_count(n, nd, no);
    // Memory: the n×n working set (two points + Hessian temporaries, ~40
    // matrices) is checked; the product cache is SIZED from what is left of
    // a quarter of the budget (it only saves work, so it shrinks, never
    // errors). Davidson's own subspace inside crate::trah is bounded by
    // `trah.davidson_max_vecs` vectors of length np + 1 and is included.
    let budget = ferric_core::memory::resolve_budget_bytes(cfg.budget_bytes);
    let dav_bytes = 2 * cfg.trah.davidson_max_vecs.max(4) * (np + 1) * 8;
    let work_bytes = 40 * n * n * 8 + dav_bytes;
    ferric_core::memory::check_alloc(
        &format!(
            "ROHF second-order working set (n = {n}: ~40 n×n matrices + the AH Davidson \
             subspace, {np} rotations)"
        ),
        work_bytes,
        budget,
    )?;
    let per_dir = 2 * np.max(1) * 8;
    let cache_share = (budget / 4).saturating_sub(work_bytes);
    let cap = np.min(256).min(cache_share / per_dir);

    let trace = crate::rohf::rohf_trace();
    let t = &cfg.trah;
    let mut info = RohfSecondOrderInfo {
        converged: false,
        macro_iterations: 0,
        fock_builds: 0,
        hessian_products: 0,
        accepted: 0,
        rejected: 0,
        last_rho: None,
        final_gradient_max: f64::INFINITY,
        final_radius: t.radius0,
        lowest_hessian_eigenvalue: None,
    };
    let mut pt = RohfPoint::at(model, c0.clone(), nd, no)?;
    info.fock_builds += 1;
    let mut radius = t.radius0;

    for macro_it in 1..=cfg.max_macro {
        ctx.check_interrupted()?;
        info.macro_iterations = macro_it;
        let g = rohf_exact_gradient(&pt.f_a_mo, &pt.f_b_mo, nd, no);
        let gmax = 0.5 * g.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        info.final_gradient_max = gmax;
        if trace {
            eprintln!(
                "ROHF-TRAH macro {macro_it:3} E {:.12} max|g_packed| {gmax:.3e} radius {radius:.3e} \
                 hvp {} rejects {}",
                pt.energy, info.hessian_products, info.rejected
            );
        }
        if np == 0 || gmax < cfg.grad_tol {
            info.converged = true;
            break;
        }
        let gn = g.iter().map(|v| v * v).sum::<f64>().sqrt();
        let mut tcfg = *t;
        tcfg.davidson_conv = (cfg.inner_tol * gn.min(1.0) * gn).max(cfg.inner_tol_floor);
        let diag = rohf_hessian_diagonal(&pt.f_a_mo, &pt.f_b_mo, nd, no);

        let outcome = {
            let cache = RefCell::new(SpanCache::new(cap));
            let (d_a, d_b) = (&pt.d_a, &pt.d_b);
            let resp = |a: &Array2<f64>, b: &Array2<f64>| model.response(d_a, d_b, a, b);
            let raw = |v: &[f64]| {
                rohf_exact_hessian_matvec(&pt.c, &pt.f_a_mo, &pt.f_b_mo, nd, no, &resp, v)
            };
            let matvec = |v: &[f64]| cache.borrow_mut().apply(v, &raw);
            let out = loop {
                let step = solve_trust_region(&g, &matvec, &diag, radius, &tcfg)?;
                let c_new = rohf_cayley_rotate(&pt.c, &step.kappa, nd, no)?;
                let new = RohfPoint::at(model, c_new, nd, no)?;
                info.fock_builds += 1;
                let act = new.energy - pt.energy;
                let rho = if step.predicted.abs() < t.predicted_min {
                    1.0
                } else if step.predicted < 0.0 {
                    act / step.predicted
                } else {
                    f64::NEG_INFINITY
                };
                info.last_rho = Some(rho);
                if trace {
                    eprintln!(
                        "ROHF-TRAH   step |k| {:.3e} alpha {:.2} mu {:+.3e} pred {:+.3e} act {act:+.3e} \
                         rho {rho:+.4} shifts {}",
                        step.norm, step.alpha, step.level_shift, step.predicted, step.shift_iterations
                    );
                }
                if rho < t.rho_reject {
                    info.rejected += 1;
                    radius *= t.shrink;
                    if radius < t.radius_min {
                        break Step::Collapsed;
                    }
                    continue;
                }
                info.accepted += 1;
                if rho <= t.rho_poor {
                    radius *= t.shrink;
                } else if rho > t.rho_good && step.on_boundary {
                    radius = (radius * t.grow).min(t.radius_max);
                }
                radius = radius.max(t.radius_min);
                break Step::Accepted(Box::new(new));
            };
            info.hessian_products += cache.borrow().misses;
            out
        };
        info.final_radius = radius;
        match outcome {
            Step::Accepted(new) => pt = *new,
            Step::Collapsed => {
                if trace {
                    eprintln!(
                        "ROHF-TRAH trust radius collapsed below {:.1e}",
                        t.radius_min
                    );
                }
                return Ok((pt, info));
            }
        }
        if macro_it == cfg.max_macro {
            // The point just accepted was never tested; test it for the record.
            let g = rohf_exact_gradient(&pt.f_a_mo, &pt.f_b_mo, nd, no);
            info.final_gradient_max = 0.5 * g.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            info.converged = info.final_gradient_max < cfg.grad_tol;
        }
    }

    if info.converged && cfg.check_minimum && np > 0 {
        let cache = RefCell::new(SpanCache::new(cap));
        let (d_a, d_b) = (&pt.d_a, &pt.d_b);
        let resp = |a: &Array2<f64>, b: &Array2<f64>| model.response(d_a, d_b, a, b);
        let raw =
            |v: &[f64]| rohf_exact_hessian_matvec(&pt.c, &pt.f_a_mo, &pt.f_b_mo, nd, no, &resp, v);
        let diag = rohf_hessian_diagonal(&pt.f_a_mo, &pt.f_b_mo, nd, no);
        let dcfg = StabilityConfig {
            conv_thresh: 1e-8,
            ..StabilityConfig::default()
        };
        let pair = davidson_lowest(
            np,
            |v: &[f64]| cache.borrow_mut().apply(v, &raw),
            &diag,
            &dcfg,
        )?;
        info.hessian_products += cache.borrow().misses;
        if pair.converged {
            info.lowest_hessian_eigenvalue = Some(pair.eigenvalue);
        }
    }
    Ok((pt, info))
}

/// Rotate within the closed, open and virtual blocks so each diagonalizes
/// the Roothaan `F_eff` (energy- and density-invariant). Returns
/// `(C, eps, F_eff)`; `eps` follows the column order (closed | open |
/// virtual, ascending within each block), NOT globally sorted.
fn semicanonicalize(
    pt: &RohfPoint,
    s: &Array2<f64>,
    nd: usize,
    no: usize,
) -> Result<(Array2<f64>, Vec<f64>, Array2<f64>), FerricError> {
    let n = pt.c.ncols();
    let f_eff = crate::rohf::roothaan_fock(&pt.f_a, &pt.f_b, &pt.d_a, &pt.d_b, s);
    let f_mo = pt.c.t().dot(&f_eff).dot(&pt.c);
    let mut c = pt.c.clone();
    let mut eps = vec![0.0f64; n];
    for (lo, hi) in [(0, nd), (nd, nd + no), (nd + no, n)] {
        if hi <= lo {
            continue;
        }
        let blk = f_mo.slice(ndarray::s![lo..hi, lo..hi]).to_owned();
        let blk = 0.5 * (&blk + &blk.t());
        let (w, v) = blk
            .eigh(UPLO::Upper)
            .map_err(|e| FerricError::Lapack(format!("ROHF semicanonical eigh: {e}")))?;
        let rot = pt.c.slice(ndarray::s![.., lo..hi]).dot(&v);
        c.slice_mut(ndarray::s![.., lo..hi]).assign(&rot);
        for (k, wk) in w.iter().enumerate() {
            eps[lo + k] = *wk;
        }
    }
    Ok((c, eps, f_eff))
}

/// [`RohfOrbitalModel`] over a [`PeriodicInjection`] (module doc): the same
/// Fock assembly as [`crate::rohf::solve_rohf_injected`] (`F_σ = h + J −
/// a·K_σ + V_σ`, `E = ½Σ_σ Tr[D_σ(h + F_σ^noxc)] + E_xc + V_nn`) and the
/// response from the SAME builders, `δV_xc` by central FD.
struct InjectedRohfModel<'a> {
    s: Array2<f64>,
    h: Array2<f64>,
    vnn: f64,
    j: RefCell<Box<dyn JBuilder + 'a>>,
    k: RefCell<Box<dyn KBuilder + 'a>>,
    xc: Option<RefCell<Box<dyn XcBuilder + 'a>>>,
    /// Exact-exchange coefficient (1 for ROHF, the builder's `a` for ROKS).
    c_k: f64,
    fxc_step: f64,
    quartets: Cell<usize>,
}

impl RohfOrbitalModel for InjectedRohfModel<'_> {
    fn overlap(&self) -> &Array2<f64> {
        &self.s
    }

    fn energy_and_focks(
        &self,
        d_a: &Array2<f64>,
        d_b: &Array2<f64>,
    ) -> Result<(f64, Array2<f64>, Array2<f64>), FerricError> {
        let n = self.h.nrows();
        let d_total = d_a + d_b;
        let mut jm = Array2::<f64>::zeros((n, n));
        let mut q = self.j.borrow_mut().build(&d_total, &mut jm)?;
        let mut f_a: Array2<f64> = &self.h + &jm;
        let mut f_b: Array2<f64> = &self.h + &jm;
        if self.c_k != 0.0 {
            let mut k_a = Array2::<f64>::zeros((n, n));
            let mut k_b = Array2::<f64>::zeros((n, n));
            q += crate::fock_assembly::build_open_shell_pluggable_k(
                &mut **self.k.borrow_mut(),
                d_a,
                d_b,
                &mut k_a,
                &mut k_b,
            )?;
            f_a.scaled_add(-self.c_k, &k_a);
            f_b.scaled_add(-self.c_k, &k_b);
        }
        self.quartets.set(self.quartets.get() + q);
        let e_elec_no_xc: f64 =
            0.5 * ((&(&self.h + &f_a) * d_a).sum() + (&(&self.h + &f_b) * d_b).sum());
        let e_xc = match self.xc.as_ref() {
            Some(x) => {
                let (e, v_a, v_b) = x.borrow_mut().build_polarized(d_a, d_b)?;
                if v_a.dim() != f_a.dim() || v_b.dim() != f_b.dim() {
                    return Err(FerricError::General(format!(
                        "solve_rohf_injected_second_order: XcBuilder::build_polarized returned \
                         V_xc of shapes {:?}/{:?}, expected {:?}",
                        v_a.dim(),
                        v_b.dim(),
                        f_a.dim()
                    )));
                }
                f_a += &v_a;
                f_b += &v_b;
                e
            }
            None => 0.0,
        };
        Ok((e_elec_no_xc + e_xc + self.vnn, f_a, f_b))
    }

    fn response(
        &self,
        d_a: &Array2<f64>,
        d_b: &Array2<f64>,
        dd_a: &Array2<f64>,
        dd_b: &Array2<f64>,
    ) -> Result<(Array2<f64>, Array2<f64>), FerricError> {
        let n = self.h.nrows();
        let dd_t = dd_a + dd_b;
        let mut dj = Array2::<f64>::zeros((n, n));
        let mut q = self.j.borrow_mut().build(&dd_t, &mut dj)?;
        let mut df_a = dj.clone();
        let mut df_b = dj;
        if self.c_k != 0.0 {
            let mut dk_a = Array2::<f64>::zeros((n, n));
            let mut dk_b = Array2::<f64>::zeros((n, n));
            q += crate::fock_assembly::build_open_shell_pluggable_k(
                &mut **self.k.borrow_mut(),
                dd_a,
                dd_b,
                &mut dk_a,
                &mut dk_b,
            )?;
            df_a.scaled_add(-self.c_k, &dk_a);
            df_b.scaled_add(-self.c_k, &dk_b);
        }
        self.quartets.set(self.quartets.get() + q);
        if let Some(x) = self.xc.as_ref() {
            let m = dd_a
                .iter()
                .chain(dd_b.iter())
                .fold(0.0f64, |acc, v| acc.max(v.abs()));
            if m > 0.0 {
                let t = self.fxc_step / m;
                let mut x = x.borrow_mut();
                let (_, vp_a, vp_b) =
                    x.build_polarized(&(d_a + &(t * dd_a)), &(d_b + &(t * dd_b)))?;
                let (_, vm_a, vm_b) =
                    x.build_polarized(&(d_a - &(t * dd_a)), &(d_b - &(t * dd_b)))?;
                let inv = 0.5 / t;
                df_a.scaled_add(inv, &(&vp_a - &vm_a));
                df_b.scaled_add(inv, &(&vp_b - &vm_b));
            }
        }
        Ok((df_a, df_b))
    }
}

/// Second-order ROHF/ROKS on caller-supplied `(S, h, V_nn)` and J/K (+ XC)
/// builders: the injected-path counterpart of
/// [`crate::rohf::solve_rohf_injected`] with the SAME energy functional,
/// config validation ([`crate::rohf::validate_injected_rohf`]) and guess
/// (`initial_mos` > `config.init_guess_density` > the core guess), but the
/// trust-region AH step of [`rohf_second_order`] instead of the DIIS /
/// Roothaan-diagonalization loop.
///
/// `RhfConfig` fields that only steer the DIIS loop are NOT used here:
/// `max_iter`, `density_conv`, `energy_conv`, `diis_size`, `level_shift`,
/// `mom_after_iter`, `rohf_occupation_guard` (no F6 witness is run: the
/// solver certifies a local minimum only via `cfg.check_minimum`).
/// Convergence is `cfg.grad_tol` on the packed orbital gradient.
///
/// The returned MOs are semicanonical in the closed / open / virtual blocks
/// of the Roothaan `F_eff` (`eps_alpha` in that column order, NOT globally
/// sorted — at `exxdiv = none` on the tri 4H cell a virtual lies below the
/// second open orbital); `fock_alpha` = `F_eff`; `rohf_spin_focks` = the
/// spin Focks of the returned density. Errors like [`crate::rohf::solve_rohf_injected`],
/// and with [`FerricError::ScfConvergence`] when the gradient test is not met
/// (macro cap or trust-radius collapse).
#[allow(clippy::too_many_arguments)]
pub fn solve_rohf_injected_second_order<'a>(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    config: &RhfConfig,
    inj: PeriodicInjection<'a>,
    initial_mos: Option<&Array2<f64>>,
    cfg: &RohfTrahConfig,
) -> Result<(ScfResult, RohfSecondOrderInfo), FerricError> {
    let who = "solve_rohf_injected_second_order";
    crate::rohf::validate_injected_rohf(config).map_err(|e| {
        FerricError::General(format!(
            "{who}: RhfConfig.{} is not supported with injected S/h/J/K: {}",
            e.field, e.reason
        ))
    })?;
    let n = prep.nbasis();
    crate::rhf::check_injection_shapes(who, &inj, n)?;
    let PeriodicInjection {
        s,
        h,
        vnn,
        mut j,
        mut k,
        xc,
    } = inj;
    let c_k = match xc.as_ref() {
        Some(x) => {
            if !x.supports_polarized() {
                return Err(FerricError::General(format!(
                    "{who}: PeriodicInjection.xc does not support spin-polarized evaluation \
                     (XcBuilder::supports_polarized() is false)"
                )));
            }
            let a = x.exact_exchange_fraction();
            if !(a.is_finite() && (0.0..=1.0).contains(&a)) {
                return Err(FerricError::General(format!(
                    "{who}: XcBuilder exact-exchange fraction must be in [0, 1], got {a}"
                )));
            }
            a
        }
        None => 1.0,
    };

    let nelec = mol.nelec() as i64;
    let mult = mol.multiplicity as i64;
    if mult < 1 {
        return Err(FerricError::General(
            "ROHF: multiplicity must be >= 1".into(),
        ));
    }
    let two_s = mult - 1;
    if (nelec - two_s) % 2 != 0 || nelec < two_s {
        return Err(FerricError::General(format!(
            "ROHF: incompatible nelec={nelec} and multiplicity={mult}"
        )));
    }
    let nocc_open = two_s as usize;
    let nocc_double = ((nelec - two_s) / 2) as usize;
    let nocc_a = nocc_double + nocc_open;
    if nocc_a > n {
        return Err(FerricError::General(format!(
            "{who}: {nocc_a} occupied orbitals exceed {n} basis functions"
        )));
    }

    // Guess, as solve_rohf_injected: initial_mos > init_guess_density (via the
    // INJECTED builders) > the core guess (symmetric S^{-1/2}).
    let (s_evals, s_evecs) = s
        .eigh(UPLO::Upper)
        .map_err(|e| FerricError::Lapack(format!("S diag: {e}")))?;
    let mut u_scaled = s_evecs.clone();
    for i in 0..n {
        let scale = 1.0 / s_evals[i].sqrt();
        for mu in 0..n {
            u_scaled[(mu, i)] *= scale;
        }
    }
    let s_inv_sqrt = u_scaled.dot(&s_evecs.t());
    let c0 = if let Some(c0) = initial_mos {
        if c0.dim() != (n, n) {
            return Err(FerricError::General(format!(
                "{who}: initial MO shape {:?} != ({n},{n})",
                c0.dim()
            )));
        }
        c0.clone()
    } else if let Some(d0) = config.init_guess_density.as_ref() {
        crate::rohf::injected_rohf_guess_mos(
            j.as_mut(),
            k.as_mut(),
            d0,
            &h,
            &s,
            &s_inv_sqrt,
            nocc_double,
            nocc_open,
        )?
    } else {
        let _ = crate::guess::hcore_guess(&s, &h, nocc_a.max(1))?;
        let h_prime = s_inv_sqrt.dot(&h).dot(&s_inv_sqrt);
        let (_, c_prime) = h_prime
            .eigh(UPLO::Upper)
            .map_err(|e| FerricError::Lapack(format!("H' diag: {e}")))?;
        s_inv_sqrt.dot(&c_prime)
    };

    let mut run_cfg = *cfg;
    if run_cfg.budget_bytes.is_none() && config.three_index_budget_bytes != 0 {
        run_cfg.budget_bytes = Some(config.three_index_budget_bytes);
    }
    let model = InjectedRohfModel {
        s,
        h,
        vnn,
        j: RefCell::new(j),
        k: RefCell::new(k),
        xc: xc.map(RefCell::new),
        c_k,
        fxc_step: cfg.fxc_fd_step,
        quartets: Cell::new(0),
    };
    let (pt, info) = rohf_second_order(ctx, &model, &c0, nocc_double, nocc_open, &run_cfg)?;
    if !info.converged {
        return Err(FerricError::ScfConvergence {
            iterations: info.macro_iterations,
            last_energy: pt.energy,
        });
    }
    let (c, eps, f_eff) = semicanonicalize(&pt, &model.s, nocc_double, nocc_open)?;
    let result = ScfResult {
        spin: Spin::RestrictedOpen,
        energy: pt.energy,
        density_total: &pt.d_a + &pt.d_b,
        density_alpha: pt.d_a.clone(),
        density_beta: Some(pt.d_b.clone()),
        mos_alpha: c,
        mos_beta: None,
        eps_alpha: eps,
        eps_beta: None,
        fock_alpha: f_eff,
        fock_beta: None,
        converged: true,
        exit: ScfExit::Converged,
        iterations: info.macro_iterations,
        computed_quartets: model.quartets.get(),
        induced_dipoles: None,
        stability: None,
        stability_external: None,
        df_jk: None,
        cosx_final: None,
        cosx_schedule: None,
        rohf_spin_focks: Some((pt.f_a.clone(), pt.f_b.clone())),
    };
    Ok((result, info))
}

#[cfg(test)]
mod tests {
    //! Derivative anchors on a synthetic, integral-free model (S = 1):
    //! `J[D] = Σ_P L_P tr(L_P D)`, `K[D] = Σ_P L_P D L_P` (a DF-like 8-fold
    //! symmetric ERI) plus a smooth non-linear "XC" `E_nl = β Σ_μ (D_t)_μμ³`
    //! with analytic response. The gradient is checked against a central FD
    //! of the energy along random κ, the Hessian against the second FD
    //! (Cayley and exp agree through second order), symmetrized over random
    //! pairs by polarization. A diagonal-Fock-only Hessian (ferric's
    //! `rohf_newton`) or a packed-scale gradient fails these by O(|f_offdiag|)
    //! and by a factor 2.
    use super::*;

    struct Toy {
        s: Array2<f64>,
        h: Array2<f64>,
        l: Vec<Array2<f64>>,
        c_k: f64,
        beta: f64,
    }

    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (*seed >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }

    fn sym(n: usize, amp: f64, seed: &mut u64) -> Array2<f64> {
        let mut m = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..=i {
                let v = amp * lcg(seed);
                m[(i, j)] = v;
                m[(j, i)] = v;
            }
        }
        m
    }

    impl Toy {
        fn new(n: usize, c_k: f64, beta: f64) -> Self {
            let mut seed = 0x1234_5678_9abc_def0u64;
            let mut h = sym(n, 0.5, &mut seed);
            for i in 0..n {
                h[(i, i)] += i as f64 * 0.7 - 1.0;
            }
            let l = (0..4).map(|_| sym(n, 0.4, &mut seed)).collect();
            Self {
                s: Array2::eye(n),
                h,
                l,
                c_k,
                beta,
            }
        }
        fn j(&self, d: &Array2<f64>) -> Array2<f64> {
            let mut out = Array2::<f64>::zeros(d.dim());
            for lp in &self.l {
                out.scaled_add((lp * d).sum(), lp);
            }
            out
        }
        fn k(&self, d: &Array2<f64>) -> Array2<f64> {
            let mut out = Array2::<f64>::zeros(d.dim());
            for lp in &self.l {
                out += &lp.dot(d).dot(lp);
            }
            out
        }
    }

    impl RohfOrbitalModel for Toy {
        fn overlap(&self) -> &Array2<f64> {
            &self.s
        }
        fn energy_and_focks(
            &self,
            d_a: &Array2<f64>,
            d_b: &Array2<f64>,
        ) -> Result<(f64, Array2<f64>, Array2<f64>), FerricError> {
            let dt = d_a + d_b;
            let jm = self.j(&dt);
            let mut f_a = &self.h + &jm;
            let mut f_b = &self.h + &jm;
            f_a.scaled_add(-self.c_k, &self.k(d_a));
            f_b.scaled_add(-self.c_k, &self.k(d_b));
            let e0 = 0.5 * ((&(&self.h + &f_a) * d_a).sum() + (&(&self.h + &f_b) * d_b).sum());
            let mut e_nl = 0.0;
            for i in 0..dt.nrows() {
                let x = dt[(i, i)];
                e_nl += self.beta * x * x * x;
                f_a[(i, i)] += 3.0 * self.beta * x * x;
                f_b[(i, i)] += 3.0 * self.beta * x * x;
            }
            Ok((e0 + e_nl, f_a, f_b))
        }
        fn response(
            &self,
            d_a: &Array2<f64>,
            d_b: &Array2<f64>,
            dd_a: &Array2<f64>,
            dd_b: &Array2<f64>,
        ) -> Result<(Array2<f64>, Array2<f64>), FerricError> {
            let dt = d_a + d_b;
            let ddt = dd_a + dd_b;
            let jm = self.j(&ddt);
            let mut df_a = jm.clone();
            let mut df_b = jm;
            df_a.scaled_add(-self.c_k, &self.k(dd_a));
            df_b.scaled_add(-self.c_k, &self.k(dd_b));
            for i in 0..dt.nrows() {
                let v = 6.0 * self.beta * dt[(i, i)] * ddt[(i, i)];
                df_a[(i, i)] += v;
                df_b[(i, i)] += v;
            }
            Ok((df_a, df_b))
        }
    }

    const N: usize = 6;
    const ND: usize = 1;
    const NO: usize = 2;

    /// Unit-norm pseudo-random direction (keeps FD truncation O(1)).
    fn random_vec(len: usize, seed: u64) -> Vec<f64> {
        let mut s = seed;
        let v: Vec<f64> = (0..len).map(|_| lcg(&mut s)).collect();
        let nv = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        v.into_iter().map(|x| x / nv).collect()
    }

    /// A generic (non-stationary) orthonormal determinant: Cayley of a random
    /// full antisymmetric matrix.
    fn start_c() -> Array2<f64> {
        let mut seed = 42u64;
        let mut k = Array2::<f64>::zeros((N, N));
        for i in 0..N {
            for j in 0..i {
                let v = 0.6 * lcg(&mut seed);
                k[(i, j)] = v;
                k[(j, i)] = -v;
            }
        }
        let eye = Array2::<f64>::eye(N);
        let half = 0.5 * &k;
        (&eye - &half).inv().unwrap().dot(&(&eye + &half))
    }

    fn energy_along(toy: &Toy, c: &Array2<f64>, x: &[f64], t: f64) -> f64 {
        let xs: Vec<f64> = x.iter().map(|v| v * t).collect();
        let cr = rohf_cayley_rotate(c, &xs, ND, NO).unwrap();
        RohfPoint::at(toy, cr, ND, NO).unwrap().energy
    }

    fn hvp(toy: &Toy, pt: &RohfPoint, x: &[f64]) -> Vec<f64> {
        let resp = |a: &Array2<f64>, b: &Array2<f64>| toy.response(&pt.d_a, &pt.d_b, a, b);
        rohf_exact_hessian_matvec(&pt.c, &pt.f_a_mo, &pt.f_b_mo, ND, NO, &resp, x).unwrap()
    }

    fn dot(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    #[test]
    fn exact_gradient_matches_a_finite_difference_and_is_twice_the_packed_one() {
        for (c_k, beta) in [(1.0, 0.0), (0.25, 0.3)] {
            let toy = Toy::new(N, c_k, beta);
            let c = start_c();
            let pt = RohfPoint::at(&toy, c.clone(), ND, NO).unwrap();
            let g = rohf_exact_gradient(&pt.f_a_mo, &pt.f_b_mo, ND, NO);
            let np = rohf_rotation_count(N, ND, NO);
            assert_eq!(g.len(), np);
            assert_eq!(np, 3 + 6 + 2);
            for seed in 1..4u64 {
                let x = random_vec(np, seed);
                let h = 1e-5;
                let fd =
                    (energy_along(&toy, &c, &x, h) - energy_along(&toy, &c, &x, -h)) / (2.0 * h);
                let an = dot(&g, &x);
                assert!(
                    (fd - an).abs() < 1e-6 * an.abs().max(1.0),
                    "c_k {c_k} beta {beta}: FD {fd:.12} vs g.x {an:.12}"
                );
            }
            // The packed (ferric gradient_blocks) gradient is exactly half.
            let (vc, vo, oc) =
                crate::rohf_newton::gradient_blocks_from(&pt.f_a_mo, &pt.f_b_mo, ND, NO);
            let packed: Vec<f64> = vc
                .iter()
                .chain(vo.iter())
                .chain(oc.iter())
                .copied()
                .collect();
            for (a, b) in g.iter().zip(&packed) {
                assert_eq!(*a, 2.0 * b, "true gradient must be exactly 2x packed");
            }
        }
    }

    #[test]
    fn exact_hessian_matches_second_finite_differences_and_is_symmetric() {
        for (c_k, beta) in [(1.0, 0.0), (0.25, 0.3)] {
            let toy = Toy::new(N, c_k, beta);
            let c = start_c();
            let pt = RohfPoint::at(&toy, c.clone(), ND, NO).unwrap();
            let np = rohf_rotation_count(N, ND, NO);
            let e0 = pt.energy;
            let h = 2e-3;
            let quad = |v: &[f64]| {
                (energy_along(&toy, &c, v, h) + energy_along(&toy, &c, v, -h) - 2.0 * e0) / (h * h)
            };
            for seed in 10..13u64 {
                let x = random_vec(np, seed);
                let y = random_vec(np, seed + 100);
                let hx = hvp(&toy, &pt, &x);
                let hy = hvp(&toy, &pt, &y);
                let (xhx, yhx, xhy) = (dot(&x, &hx), dot(&y, &hx), dot(&x, &hy));
                let fd_xx = quad(&x);
                let xp: Vec<f64> = x.iter().zip(&y).map(|(a, b)| a + b).collect();
                let xm: Vec<f64> = x.iter().zip(&y).map(|(a, b)| a - b).collect();
                let fd_xy = 0.25 * (quad(&xp) - quad(&xm));
                let scale = xhx.abs().max(1.0);
                assert!(
                    (fd_xx - xhx).abs() < 1e-4 * scale,
                    "c_k {c_k} beta {beta}: x.Hx {xhx:.10} vs FD {fd_xx:.10}"
                );
                assert!(
                    (fd_xy - yhx).abs() < 1e-4 * scale,
                    "c_k {c_k} beta {beta}: y.Hx {yhx:.10} vs FD {fd_xy:.10}"
                );
                assert!(
                    (yhx - xhy).abs() < 1e-11 * scale,
                    "Hessian not symmetric: {yhx:e} vs {xhy:e}"
                );
            }
        }
    }

    #[test]
    fn the_span_cache_reproduces_the_uncached_product() {
        let toy = Toy::new(N, 0.25, 0.3);
        let pt = RohfPoint::at(&toy, start_c(), ND, NO).unwrap();
        let np = rohf_rotation_count(N, ND, NO);
        let raw = |v: &[f64]| -> Result<Vec<f64>, FerricError> { Ok(hvp(&toy, &pt, v)) };
        let mut cache = SpanCache::new(np);
        let a = random_vec(np, 7);
        let b = random_vec(np, 8);
        let ab: Vec<f64> = a.iter().zip(&b).map(|(x, y)| 0.3 * x - 1.7 * y).collect();
        let _ = cache.apply(&a, &raw).unwrap();
        let _ = cache.apply(&b, &raw).unwrap();
        assert_eq!(cache.misses, 2);
        let got = cache.apply(&ab, &raw).unwrap();
        assert_eq!(
            cache.misses, 2,
            "an in-span request must not call the response"
        );
        let want = hvp(&toy, &pt, &ab);
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() < 1e-12 * w.abs().max(1.0), "{g} vs {w}");
        }
        // Capped cache: out-of-span requests go straight through.
        let mut capped = SpanCache::new(0);
        let got = capped.apply(&a, &raw).unwrap();
        assert_eq!(got, hvp(&toy, &pt, &a));
    }
}
