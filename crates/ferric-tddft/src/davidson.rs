//! Block Davidson eigensolvers for the TDA and Casida response problems, with
//! a caller-supplied matrix-free product.
//!
//! # TDA
//!
//! `A X = ω X`, A symmetric. Rayleigh–Ritz on an orthonormal subspace `V`
//! (`H = Vᵀ A V`), residual `r = A x − θ x` for each target Ritz pair, and
//! Olsen's diagonal correction `t = P r − ε P x`, `P = (θ − D)⁻¹` with
//! `D_ia = ε_a − ε_i` and `ε` making `t ⊥ x` (see `diagonal_correction`).
//!
//! # Casida
//!
//! `(A+B)(X+Y) = Ω (X−Y)`, `(A−B)(X−Y) = Ω (X+Y)`, both `A ± B` symmetric and
//! `A − B` positive definite for a stable reference. One subspace `V` serves
//! both vectors. With `M± = Vᵀ (A±B) V`, the Galerkin conditions
//! `M+ u = Ω v`, `M− v = Ω u` give `M− M+ u = Ω² u`; with `L = M−^½` and
//! `u = L z`, that is the symmetric `L M+ L z = Ω² z` — the subspace image of
//! the dense `(A−B)^½ (A+B) (A−B)^½` problem. For unit `z`,
//! `u = Ω^{-½} L z` carries `(X+Y)·(X−Y) = 1`, the dense normalization.
//! Residuals `r+ = (A+B)(X+Y) − Ω (X−Y)` and `r− = (A−B)(X−Y) − Ω (X+Y)`;
//! the corrections solve the diagonal model `[[D, −Ω], [−Ω, D]] δ = −r`
//! per element with Olsen's projection (see `casida_correction`), and both
//! `δ+` and `δ−` join the subspace.
//!
//! The reported Casida residual is `√((‖r+‖² + ‖r−‖²)/2)`: the residual of
//! the full `[[A, B], [B, A]] [X; Y] = Ω [X; −Y]` problem in the
//! `|X|² − |Y|² = 1` normalization (`r_X = (r+ + r−)/2`, `r_Y = (r+ − r−)/2`).
//!
//! # Roots
//!
//! Targets are the lowest `n_roots` Ritz values above [`POSITIVE_ROOT`], the
//! dense path's root selection. Guess vectors sit on the lowest diagonal
//! elements, at least `max(2·n_roots, n_roots + 8)` of them and every
//! diagonal element tied with the last one taken (a degenerate pair of
//! orbital differences is never split). Each also carries a small
//! deterministic dense part, so the subspace reaches every symmetry block
//! (see `guess_vectors`).
//!
//! An unconverged root after `max_iter` iterations is an error, never a
//! returned number, and so is a negative Casida `Ω²` (an instability of the
//! reference in the run's spin channel).

use ferric_core::FerricError;
use ndarray::{s, Array1, Array2, Axis};
use ndarray_linalg::{Eigh, UPLO};

/// Ritz values at or below this are not roots (the dense path's cut).
pub const POSITIVE_ROOT: f64 = 1e-10;

/// A correction whose norm after two Gram–Schmidt passes is below this
/// fraction of its norm before is treated as already spanned (two passes
/// leave a spanned vector at ~1e-16·√m relative size).
const NULL_DIRECTION_REL: f64 = 1e-10;

/// Smallest `|θ − D|` the diagonal preconditioner divides by.
const PRECOND_FLOOR: f64 = 1e-8;

/// Iteration controls of a Davidson solve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DavidsonSettings {
    /// Converged when every target root's residual norm is at or below this
    /// (Hartree; see the module docs for the Casida residual). The energy
    /// error of a converged root is second order, `≲ ‖r‖² / gap`.
    pub conv_tol: f64,
    /// Iteration cap; reaching it with an unconverged root is an error.
    pub max_iter: usize,
    /// Largest subspace before a collapse onto the current Ritz vectors.
    /// `None`: `max(n_guess + 10·n_roots, 50)`. Capped at the dimension.
    pub max_space: Option<usize>,
}

impl Default for DavidsonSettings {
    fn default() -> Self {
        Self {
            conv_tol: 1e-6,
            max_iter: 100,
            max_space: None,
        }
    }
}

/// Converged roots of a Davidson solve.
#[derive(Debug, Clone)]
pub struct DavidsonOutcome {
    /// The lowest `n_roots` positive eigenvalues (ω for TDA, Ω for Casida),
    /// ascending.
    pub eigenvalues: Array1<f64>,
    /// Column n: TDA `X_n` (unit norm); Casida `(X+Y)_n` with
    /// `(X+Y)·(X−Y) = 1`.
    pub vectors: Array2<f64>,
    /// Final residual norm of each root.
    pub residuals: Vec<f64>,
    /// Iterations taken.
    pub iterations: usize,
    /// Vectors the product was applied to, in total.
    pub products: usize,
}

/// Number of guess vectors for `n_roots` roots in `dim` dimensions with
/// diagonal `diag`, and their indices (lowest diagonal first, ties at the
/// cut included).
fn guess_indices(diag: &Array1<f64>, n_roots: usize) -> Vec<usize> {
    let dim = diag.len();
    let mut order: Vec<usize> = (0..dim).collect();
    order.sort_by(|&a, &b| diag[a].total_cmp(&diag[b]).then(a.cmp(&b)));
    let mut n = (2 * n_roots).max(n_roots + 8).min(dim);
    while n < dim && (diag[order[n]] - diag[order[n - 1]]).abs() <= 1e-8 {
        n += 1;
    }
    order.truncate(n);
    order
}

/// Relative size of the dense part of each guess vector (see [`guess_vectors`]).
const GUESS_SPREAD: f64 = 0.1;

/// Orthonormalized unit vectors on `idx`, each first given a deterministic
/// dense part (elements uniform in `±GUESS_SPREAD/√dim`, norm about
/// `GUESS_SPREAD/√3`).
///
/// Neither the products nor the diagonal preconditioner mix point-group
/// irreps, so a subspace grown from pure unit vectors never reaches an irrep
/// none of them belongs to, and a low root there is skipped while every
/// returned root converges. The dense part gives each guess a component in
/// every irrep.
fn guess_vectors(dim: usize, idx: &[usize]) -> Array2<f64> {
    let amp = GUESS_SPREAD / (dim as f64).sqrt();
    let mut v = Array2::from_shape_fn((dim, idx.len()), |(i, k)| {
        amp * splitmix_unit(((k as u64) << 32) | i as u64)
    });
    for (k, &i) in idx.iter().enumerate() {
        v[(i, k)] += 1.0;
    }
    orthonormalize_against(&Array2::zeros((dim, 0)), v)
}

/// A deterministic value in [−1, 1) from `seed` (one SplitMix64 step).
fn splitmix_unit(seed: u64) -> f64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
}

fn resolve_space(s: &DavidsonSettings, n_guess: usize, n_roots: usize, dim: usize) -> usize {
    s.max_space
        .unwrap_or_else(|| (n_guess + 10 * n_roots).max(50))
        .max(n_guess + n_roots)
        .min(dim)
}

/// Orthonormalize the columns of `t` against the orthonormal columns of `v`
/// and among themselves (two classical Gram–Schmidt passes); columns that are
/// already spanned are dropped. Returns the surviving columns.
fn orthonormalize_against(v: &Array2<f64>, t: Array2<f64>) -> Array2<f64> {
    let mut kept: Vec<Array1<f64>> = Vec::new();
    for col in t.columns() {
        let mut c = col.to_owned();
        let n0 = c.dot(&c).sqrt();
        if n0 == 0.0 || !n0.is_finite() {
            continue;
        }
        c /= n0;
        for _ in 0..2 {
            if v.ncols() > 0 {
                let proj = v.t().dot(&c);
                c -= &v.dot(&proj);
            }
            for k in &kept {
                let p = k.dot(&c);
                c.scaled_add(-p, k);
            }
        }
        let n1 = c.dot(&c).sqrt();
        if n1 > NULL_DIRECTION_REL {
            c /= n1;
            kept.push(c);
        }
    }
    let dim = v.nrows();
    let mut out = Array2::<f64>::zeros((dim, kept.len()));
    for (k, c) in kept.into_iter().enumerate() {
        out.column_mut(k).assign(&c);
    }
    out
}

fn append_columns(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    ndarray::concatenate(Axis(1), &[a.view(), b.view()]).expect("same row count")
}

fn symmetric_eigh(m: &Array2<f64>, what: &str) -> Result<(Array1<f64>, Array2<f64>), FerricError> {
    let sym = 0.5 * (m + &m.t());
    sym.eigh(UPLO::Lower)
        .map_err(|e| FerricError::General(format!("Davidson: {what} diagonalization failed: {e}")))
}

/// Indices (into ascending `vals`) of the first `n_roots` values above
/// [`POSITIVE_ROOT`].
fn targets(vals: &Array1<f64>, n_roots: usize) -> Vec<usize> {
    (0..vals.len())
        .filter(|&k| vals[k] > POSITIVE_ROOT)
        .take(n_roots)
        .collect()
}

fn residual_list(residuals: &[f64]) -> String {
    let listed: Vec<String> = residuals.iter().map(|r| format!("{r:.2e}")).collect();
    listed.join(", ")
}

fn stalled_error(kind: &str, m: usize, dim: usize, residuals: &[f64], tol: f64) -> FerricError {
    FerricError::General(format!(
        "{kind} Davidson: the subspace ({m} of {dim} dimensions) cannot grow, with root \
         residuals [{}] above conv_tol {tol:e}",
        residual_list(residuals)
    ))
}

fn unconverged_error(kind: &str, iters: usize, residuals: &[f64], tol: f64) -> FerricError {
    FerricError::General(format!(
        "{kind} Davidson did not converge in {iters} iterations: root residuals [{}] \
         against conv_tol {tol:e}. Raise the iteration cap, or use the dense solver",
        residual_list(residuals)
    ))
}

/// `1 / (θ − D)` per element, `|θ − D|` floored at [`PRECOND_FLOOR`].
fn inverse_shifted_diag(theta: f64, diag: &Array1<f64>) -> Array1<f64> {
    diag.mapv(|d| {
        let den = theta - d;
        1.0 / if den.abs() < PRECOND_FLOOR {
            PRECOND_FLOOR.copysign(den)
        } else {
            den
        }
    })
}

/// Olsen's correction `t = P r − ε P x`, `P = (θ − D)⁻¹`, with
/// `ε = xᵀ P r / xᵀ P x` so that `xᵀ t = 0`.
///
/// The plain `P r` stalls when the root's eigenvector lives on a block of
/// equal orbital differences whose value is the eigenvalue itself
/// (`A u = D u = λ u`, e.g. a coupling-free combination of a degenerate
/// π → π* set): there `P r ≈ −x` plus the leakage amplified by
/// `1/(θ − λ)` on that same block, so almost every new direction is
/// already spanned. Removing the `x` part first keeps the step in the
/// complement. `ε` is skipped when `xᵀ P x` is not finite or vanishes.
fn diagonal_correction(
    r: &Array1<f64>,
    x: &ndarray::ArrayView1<f64>,
    theta: f64,
    diag: &Array1<f64>,
) -> Array1<f64> {
    let p = inverse_shifted_diag(theta, diag);
    let pr = &p * r;
    let px = &p * x;
    let den = x.dot(&px);
    if den.is_finite() && den != 0.0 {
        let eps = x.dot(&pr) / den;
        if eps.is_finite() {
            return pr - eps * px;
        }
    }
    pr
}

/// The lowest `n_roots` positive TDA roots of the symmetric operator whose
/// product is `apply_a` (columns of the argument → columns of `A x`), with
/// diagonal preconditioner `diag`.
///
/// # Errors
///
/// An unconverged root after `settings.max_iter` iterations; a subspace that
/// can grow no further with a root above tolerance; a failed product.
pub fn davidson_tda<F>(
    mut apply_a: F,
    diag: &Array1<f64>,
    n_roots: usize,
    settings: &DavidsonSettings,
) -> Result<DavidsonOutcome, FerricError>
where
    F: FnMut(&Array2<f64>) -> Result<Array2<f64>, FerricError>,
{
    let dim = diag.len();
    let n_roots = n_roots.min(dim);
    let guess = guess_indices(diag, n_roots);
    let max_space = resolve_space(settings, guess.len(), n_roots, dim);
    let mut v = guess_vectors(dim, &guess);
    let mut av = apply_a(&v)?;
    let mut products = v.ncols();
    let mut residuals = Vec::new();
    for iter in 1..=settings.max_iter {
        let (theta, y) = symmetric_eigh(&v.t().dot(&av), "TDA subspace")?;
        let tg = targets(&theta, n_roots);
        let yt = y.select(Axis(1), &tg);
        let x = v.dot(&yt);
        let ax = av.dot(&yt);
        let mut corr = Array2::<f64>::zeros((dim, 0));
        residuals.clear();
        for (k, &j) in tg.iter().enumerate() {
            let r = &ax.column(k) - &(theta[j] * &x.column(k));
            let rn = r.dot(&r).sqrt();
            residuals.push(rn);
            if rn > settings.conv_tol {
                let t = diagonal_correction(&r, &x.column(k), theta[j], diag);
                corr = append_columns(&corr, &t.insert_axis(Axis(1)));
            }
        }
        if tg.len() == n_roots && corr.ncols() == 0 {
            return Ok(DavidsonOutcome {
                eigenvalues: tg.iter().map(|&j| theta[j]).collect(),
                vectors: x,
                residuals,
                iterations: iter,
                products,
            });
        }
        if v.ncols() + corr.ncols() > max_space {
            // Collapse onto the lowest Ritz vectors (orthonormal, since Y is),
            // reusing the products: no new A applications.
            let keep = guess.len().max(n_roots).min(theta.len());
            let yk = y.slice(s![.., ..keep]);
            v = v.dot(&yk);
            av = av.dot(&yk);
        }
        let new = orthonormalize_against(&v, corr);
        if new.ncols() == 0 {
            return Err(stalled_error(
                "TDA",
                v.ncols(),
                dim,
                &residuals,
                settings.conv_tol,
            ));
        }
        let anew = apply_a(&new)?;
        products += new.ncols();
        v = append_columns(&v, &new);
        av = append_columns(&av, &anew);
    }
    Err(unconverged_error(
        "TDA",
        settings.max_iter,
        &residuals,
        settings.conv_tol,
    ))
}

/// Ritz pairs of the projected Casida problem: Ω and the subspace
/// coefficients `u` (of X+Y) and `v` (of X−Y) for each positive root, the
/// first `n_roots` of them.
fn casida_ritz(
    mp: &Array2<f64>,
    mm: &Array2<f64>,
    n_roots: usize,
) -> Result<(Vec<f64>, Array2<f64>, Array2<f64>), FerricError> {
    let (lam, w) = symmetric_eigh(mm, "Casida (A-B) subspace")?;
    if let Some(&bad) = lam.iter().find(|&&l| l <= 0.0) {
        return Err(FerricError::General(format!(
            "TDDFT: (A-B) has a non-positive eigenvalue {bad:.6e} on the Davidson subspace \
             — RPA instability"
        )));
    }
    let l_half = {
        let mut ws = w.clone();
        for (mut col, &l) in ws.columns_mut().into_iter().zip(&lam) {
            col *= l.sqrt();
        }
        ws.dot(&w.t())
    };
    let h = l_half.dot(mp).dot(&l_half);
    let (om2, z) = symmetric_eigh(&h, "Casida subspace")?;
    // A negative subspace Ω² means Vᵀ(A+B)V, hence A+B, is indefinite: an
    // instability of the reference, refused as the dense path does.
    if let Some(&w2) = om2.iter().find(|&&w| w < -crate::tddft::NEGATIVE_OMEGA_SQ) {
        return Err(crate::tddft::negative_omega_sq_error(w2));
    }
    let tg = targets(&om2, n_roots);
    let m = mp.nrows();
    let mut u = Array2::<f64>::zeros((m, tg.len()));
    let mut v = Array2::<f64>::zeros((m, tg.len()));
    let mut omegas = Vec::with_capacity(tg.len());
    for (k, &j) in tg.iter().enumerate() {
        let om = om2[j].sqrt();
        let uk = l_half.dot(&z.column(j)) / om.sqrt();
        let vk = mp.dot(&uk) / om;
        u.column_mut(k).assign(&uk);
        v.column_mut(k).assign(&vk);
        omegas.push(om);
    }
    Ok((omegas, u, v))
}

/// The lowest `n_roots` positive Casida roots. `apply_pm` maps the columns of
/// its argument to the columns of `((A+B) x, (A−B) x)`.
///
/// # Errors
///
/// As [`davidson_tda`], plus an `A − B` that is not positive definite on the
/// subspace (an RPA instability, as the dense path).
pub fn davidson_casida<F>(
    mut apply_pm: F,
    diag: &Array1<f64>,
    n_roots: usize,
    settings: &DavidsonSettings,
) -> Result<DavidsonOutcome, FerricError>
where
    F: FnMut(&Array2<f64>) -> Result<(Array2<f64>, Array2<f64>), FerricError>,
{
    let dim = diag.len();
    let n_roots = n_roots.min(dim);
    let guess = guess_indices(diag, n_roots);
    let max_space = resolve_space(settings, guess.len(), n_roots, dim);
    let mut b = guess_vectors(dim, &guess);
    let (mut pb, mut mb) = apply_pm(&b)?;
    let mut products = b.ncols();
    let mut residuals = Vec::new();
    for iter in 1..=settings.max_iter {
        let (omegas, u, w) = casida_ritz(&b.t().dot(&pb), &b.t().dot(&mb), n_roots)?;
        let (xpy, xmy) = (b.dot(&u), b.dot(&w));
        let (p_xpy, m_xmy) = (pb.dot(&u), mb.dot(&w));
        let mut corr = Array2::<f64>::zeros((dim, 0));
        residuals.clear();
        for (k, &om) in omegas.iter().enumerate() {
            let rp = &p_xpy.column(k) - &(om * &xmy.column(k));
            let rm = &m_xmy.column(k) - &(om * &xpy.column(k));
            let rn = (0.5 * (rp.dot(&rp) + rm.dot(&rm))).sqrt();
            residuals.push(rn);
            if rn > settings.conv_tol {
                let (dp, dm) =
                    casida_correction((&rp, &rm), (&xpy.column(k), &xmy.column(k)), om, diag);
                corr = append_columns(&corr, &dp.insert_axis(Axis(1)));
                corr = append_columns(&corr, &dm.insert_axis(Axis(1)));
            }
        }
        if omegas.len() == n_roots && corr.ncols() == 0 {
            return Ok(DavidsonOutcome {
                eigenvalues: Array1::from(omegas),
                vectors: xpy,
                residuals,
                iterations: iter,
                products,
            });
        }
        if b.ncols() + corr.ncols() > max_space {
            // Collapse onto span{X+Y, X−Y} of the current roots plus the
            // lowest guess-count Ritz directions, reusing the products.
            let keep = collapse_basis(&u, &w);
            b = b.dot(&keep);
            pb = pb.dot(&keep);
            mb = mb.dot(&keep);
        }
        let new = orthonormalize_against(&b, corr);
        if new.ncols() == 0 {
            return Err(stalled_error(
                "Casida",
                b.ncols(),
                dim,
                &residuals,
                settings.conv_tol,
            ));
        }
        let (pn, mn) = apply_pm(&new)?;
        products += new.ncols();
        b = append_columns(&b, &new);
        pb = append_columns(&pb, &pn);
        mb = append_columns(&mb, &mn);
    }
    Err(unconverged_error(
        "Casida",
        settings.max_iter,
        &residuals,
        settings.conv_tol,
    ))
}

/// Orthonormal subspace coefficients spanning the current roots' `u` and `v`
/// (the collapse target of [`davidson_casida`]).
fn collapse_basis(u: &Array2<f64>, w: &Array2<f64>) -> Array2<f64> {
    let m = u.nrows();
    let empty = Array2::<f64>::zeros((m, 0));
    orthonormalize_against(&empty, append_columns(u, w))
}

/// `K⁻¹ [a; b]` for the per-element model `K = [[D, −Ω], [−Ω, D]]`
/// (`|det|` floored at [`PRECOND_FLOOR`]).
fn casida_model_solve(
    a: &Array1<f64>,
    b: &ndarray::ArrayView1<f64>,
    om: f64,
    diag: &Array1<f64>,
) -> (Array1<f64>, Array1<f64>) {
    let n = diag.len();
    let mut p = Array1::<f64>::zeros(n);
    let mut m = Array1::<f64>::zeros(n);
    for k in 0..n {
        let d = diag[k];
        let det = d * d - om * om;
        let det = if det.abs() < PRECOND_FLOOR {
            PRECOND_FLOOR.copysign(det)
        } else {
            det
        };
        p[k] = (d * a[k] + om * b[k]) / det;
        m[k] = (om * a[k] + d * b[k]) / det;
    }
    (p, m)
}

/// Olsen's correction for the Casida model: `δ = −K⁻¹ r + ε K⁻¹ q`, with
/// `r = [r+; r−]`, `q = [X−Y; X+Y]` (the metric partner of the current
/// `[X+Y; X−Y]`, so `H z = Ω q` at convergence) and
/// `ε = qᵀ K⁻¹ r / qᵀ K⁻¹ q`, which makes `qᵀ δ = 0`. As for TDA (see
/// [`diagonal_correction`]), the plain `−K⁻¹ r` stalls on a root whose
/// eigenvector sits on a block of equal orbital differences equal to `Ω`.
/// `ε` is skipped when `qᵀ K⁻¹ q` is not finite or vanishes.
fn casida_correction(
    (rp, rm): (&Array1<f64>, &Array1<f64>),
    (xpy, xmy): (&ndarray::ArrayView1<f64>, &ndarray::ArrayView1<f64>),
    om: f64,
    diag: &Array1<f64>,
) -> (Array1<f64>, Array1<f64>) {
    let (kp, km) = casida_model_solve(rp, &rm.view(), om, diag);
    let (qp, qm) = casida_model_solve(&xmy.to_owned(), xpy, om, diag);
    let den = xmy.dot(&qp) + xpy.dot(&qm);
    let num = xmy.dot(&kp) + xpy.dot(&km);
    let eps = num / den;
    if den.is_finite() && den != 0.0 && eps.is_finite() {
        (-kp + eps * qp, -km + eps * qm)
    } else {
        (-kp, -km)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Xorshift64(u64);
    impl Xorshift64 {
        fn next_f64(&mut self) -> f64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            ((x >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
        }
    }

    const DIM: usize = 90;
    const N_ROOTS: usize = 5;

    /// A response-like problem: `D_k` spread over [0.3, 2.3] with a few
    /// near-degenerate pairs, plus symmetric couplings of size `c`.
    fn diag_and_coupling(seed: u64, c: f64) -> (Array1<f64>, Array2<f64>) {
        let mut rng = Xorshift64(seed);
        let d = Array1::from_shape_fn(DIM, |k| {
            0.3 + 2.0 * (k / 2) as f64 / DIM as f64 + 1e-3 * (k % 2) as f64
        });
        let mut e = Array2::<f64>::zeros((DIM, DIM));
        for i in 0..DIM {
            for j in i..DIM {
                let v = c * rng.next_f64();
                e[(i, j)] = v;
                e[(j, i)] = v;
            }
        }
        (d, e)
    }

    fn with_diag(d: &Array1<f64>, e: &Array2<f64>) -> Array2<f64> {
        let mut a = e.clone();
        for k in 0..d.len() {
            a[(k, k)] += d[k];
        }
        a
    }

    fn spectral_norm(m: &Array2<f64>) -> f64 {
        let (w, _) = m.eigh(UPLO::Lower).unwrap();
        w.iter().fold(0.0_f64, |x, &y| x.max(y.abs()))
    }

    /// Distance from `w[k]` to the nearest other eigenvalue.
    fn gap(w: &Array1<f64>, k: usize) -> f64 {
        (0..w.len())
            .filter(|&j| j != k)
            .map(|j| (w[j] - w[k]).abs())
            .fold(f64::INFINITY, f64::min)
    }

    fn settings(max_space: Option<usize>) -> DavidsonSettings {
        DavidsonSettings {
            conv_tol: 1e-9,
            max_iter: 200,
            max_space,
        }
    }

    /// TDA roots and vectors equal the dense `eigh`, both with the subspace
    /// allowed to reach `dim` and at the default (restarting) subspace size.
    ///
    /// Bars (derived): a Ritz pair (θ, x) with residual r has an eigenvalue
    /// within `‖r‖²/gap` (quadratic residual bound for a symmetric matrix)
    /// and `sin∠(x, x_k) ≤ ‖r‖/gap` (Davis–Kahan); the dense reference is
    /// backward stable, so it adds `dim·ε·‖A‖₂` to the eigenvalue and that
    /// over the gap to the angle.
    ///
    /// Mutation (applied and reverted, 2026-11-13): returning `θ − D` instead
    /// of dividing the residual by it in `diagonal_correction` stalls the
    /// subspace (the "cannot grow" error). A single Gram–Schmidt pass passes
    /// here and fails `root_in_a_block_no_guess_touches_is_found` (Casida
    /// does not converge in 200 iterations).
    #[test]
    fn tda_matches_dense_eigh() {
        let (d, e) = diag_and_coupling(0x9E3779B97F4A7C15, 0.02);
        let a = with_diag(&d, &e);
        let (w, v) = a.eigh(UPLO::Lower).unwrap();
        let floor = DIM as f64 * f64::EPSILON * spectral_norm(&a);
        for max_space in [Some(DIM), None] {
            let out = davidson_tda(|x| Ok(a.dot(x)), &d, N_ROOTS, &settings(max_space)).unwrap();
            assert_eq!(out.eigenvalues.len(), N_ROOTS);
            for k in 0..N_ROOTS {
                let r = out.residuals[k];
                assert!(r <= 1e-9, "root {k}: residual {r:e}");
                let g = gap(&w, k);
                let de = (out.eigenvalues[k] - w[k]).abs();
                assert!(
                    de <= r * r / g + floor,
                    "root {k}: |dw| {de:e}, bar {:e}",
                    r * r / g + floor
                );
                // ‖x − (x·v)v‖, not √(1 − c²): the latter floors at √(2ε).
                let (x, vk) = (out.vectors.column(k), v.column(k));
                let perp = &x - &(x.dot(&vk) * &vk);
                let sin = perp.dot(&perp).sqrt();
                assert!(sin <= (r + floor) / g, "root {k}: sin {sin:e}");
            }
        }
    }

    /// Casida roots equal the dense `(A−B)^½ (A+B) (A−B)^½` solve, and the
    /// returned `X+Y` carries `(X+Y)·(X−Y) = 1`.
    ///
    /// Bars (derived): with `X−Y := (A+B)(X+Y)/Ω` and
    /// `z = (A−B)^½ (X−Y)`, `H z − Ω² z = r_H` for the symmetric
    /// `H = (A−B)^½ (A+B) (A−B)^½`. The Rayleigh quotient ρ of z is within
    /// `‖Hz − ρz‖²/(‖z‖² gap_H)` of an eigenvalue and Ω² is within
    /// `|zᵀ r_H|/‖z‖²` of ρ; the dense reference adds `dim·ε·‖H‖₂`. The sum,
    /// over `Ω + Ω_dense`, bounds |ΔΩ|. Normalization:
    /// `(X+Y)ᵀ(A+B)(X+Y)/Ω − 1 = (X+Y)·r+/Ω`, bounded by `√2 ‖X+Y‖ res/Ω`.
    ///
    /// Mutation (applied and reverted, 2026-11-13): dropping the `1/√Ω` in
    /// `casida_ritz`'s `u` fails the normalization check (−0.76 for root 0);
    /// using `M+` in place of `M−` for `L` stalls the subspace here and makes
    /// `non_positive_a_minus_b_is_an_rpa_instability` miss the instability.
    #[test]
    fn casida_matches_dense_solve() {
        let (d, e1) = diag_and_coupling(0x243F6A8885A308D3, 0.02);
        let (_, e2) = diag_and_coupling(0x13198A2E03707344, 0.01);
        let a = with_diag(&d, &e1);
        let b = e2;
        let apb = &a + &b;
        let amb = &a - &b;
        let (lam, u) = amb.eigh(UPLO::Lower).unwrap();
        assert!(lam.iter().all(|&l| l > 0.0));
        let mut us = u.clone();
        for (mut col, &l) in us.columns_mut().into_iter().zip(&lam) {
            col *= l.sqrt();
        }
        let amb_half = us.dot(&u.t());
        let h = amb_half.dot(&apb).dot(&amb_half);
        let (w2, _) = h.eigh(UPLO::Lower).unwrap();
        let floor = DIM as f64 * f64::EPSILON * spectral_norm(&h);
        for max_space in [Some(DIM), None] {
            let out = davidson_casida(
                |x| Ok((apb.dot(x), amb.dot(x))),
                &d,
                N_ROOTS,
                &settings(max_space),
            )
            .unwrap();
            for k in 0..N_ROOTS {
                let om = out.eigenvalues[k];
                let xpy = out.vectors.column(k).to_owned();
                let res = out.residuals[k];
                assert!(res <= 1e-9, "root {k}: residual {res:e}");
                let xmy = apb.dot(&xpy) / om;
                let z = amb_half.dot(&xmy);
                let zz = z.dot(&z);
                let hz = h.dot(&z);
                let rho = z.dot(&hz) / zz;
                let rq = &hz - &(rho * &z);
                let rh = &hz - &(om * om * &z);
                let bar2 = z.dot(&rh).abs() / zz + rq.dot(&rq) / (zz * gap(&w2, k)) + floor;
                let om_d = w2[k].sqrt();
                let de = (om - om_d).abs();
                assert!(
                    de <= bar2 / (om + om_d),
                    "root {k}: |dOmega| {de:e}, bar {:e}",
                    bar2 / (om + om_d)
                );
                let norm = xpy.dot(&apb.dot(&xpy)) / om - 1.0;
                let nbar = 2f64.sqrt() * xpy.dot(&xpy).sqrt() * res / om + floor;
                assert!(norm.abs() <= nbar, "root {k}: (X+Y).(X-Y) - 1 = {norm:e}");
            }
        }
    }

    /// A low root in a block no guess index belongs to is still found.
    ///
    /// The last two basis functions form their own block (a stand-in for an
    /// irrep): diagonal 3.0, above every guess index's, coupled by −2.9, so
    /// that block's lowest eigenvalue 0.1 is the lowest of the whole matrix.
    /// Products never mix the blocks, so pure unit-vector guesses converge on
    /// five roots of the other block and return no error. Bar: a symmetric
    /// matrix has an eigenvalue within `‖r‖` of a Ritz value with unit vector
    /// (Bauer–Fike); with B = 0, `X+Y = X−Y` is a unit vector and the Casida
    /// residual is `‖A x − Ω x‖`. The dense reference adds `dim·ε·‖A‖₂`.
    ///
    /// Mutation (applied and reverted, 2026-11-13): with `GUESS_SPREAD` = 0
    /// the TDA solve converges with 0.2374 as its lowest root (dense: 0.1).
    #[test]
    fn root_in_a_block_no_guess_touches_is_found() {
        let (d0, e) = diag_and_coupling(0x5851F42D4C957F2D, 0.02);
        let mut a = with_diag(&d0, &e);
        let mut d = d0.clone();
        let (p, q) = (DIM - 2, DIM - 1);
        for k in 0..DIM {
            for j in [p, q] {
                a[(k, j)] = 0.0;
                a[(j, k)] = 0.0;
            }
        }
        a[(p, p)] = 3.0;
        a[(q, q)] = 3.0;
        a[(p, q)] = -2.9;
        a[(q, p)] = -2.9;
        d[p] = 3.0;
        d[q] = 3.0;
        let (w, _) = a.eigh(UPLO::Lower).unwrap();
        assert!((w[0] - 0.1).abs() < 1e-12, "construction: lowest {}", w[0]);
        let floor = DIM as f64 * f64::EPSILON * spectral_norm(&a);
        let tda = davidson_tda(|x| Ok(a.dot(x)), &d, N_ROOTS, &settings(None)).unwrap();
        // B = 0: the Casida roots are the TDA roots.
        let cas =
            davidson_casida(|x| Ok((a.dot(x), a.dot(x))), &d, N_ROOTS, &settings(None)).unwrap();
        for (label, out) in [("TDA", &tda), ("Casida", &cas)] {
            for k in 0..N_ROOTS {
                let r = out.residuals[k];
                let bar = r + floor;
                let de = (out.eigenvalues[k] - w[k]).abs();
                assert!(
                    de <= bar,
                    "{label} root {k}: {} vs dense {} (bar {bar:e})",
                    out.eigenvalues[k],
                    w[k]
                );
            }
        }
    }

    #[test]
    fn unconverged_root_is_an_error() {
        let (d, e) = diag_and_coupling(7, 0.02);
        let a = with_diag(&d, &e);
        let s = DavidsonSettings {
            conv_tol: 1e-12,
            max_iter: 1,
            max_space: None,
        };
        let err = davidson_tda(|x| Ok(a.dot(x)), &d, N_ROOTS, &s).unwrap_err();
        assert!(format!("{err}").contains("did not converge"), "{err}");
    }

    #[test]
    fn non_positive_a_minus_b_is_an_rpa_instability() {
        let (d, e) = diag_and_coupling(11, 0.02);
        let a = with_diag(&d, &e);
        // B = A + I makes A - B = -I.
        let b = &a + &Array2::<f64>::eye(DIM);
        let (apb, amb) = (&a + &b, &a - &b);
        let err = davidson_casida(
            |x| Ok((apb.dot(x), amb.dot(x))),
            &d,
            N_ROOTS,
            &DavidsonSettings::default(),
        )
        .unwrap_err();
        assert!(format!("{err}").contains("RPA instability"), "{err}");
    }
}
