//! Block Davidson for the lowest right eigenpairs of a real non-symmetric
//! operator with a caller-supplied product (the EOM-CCSD similarity-
//! transformed Hamiltonian).
//!
//! Rayleigh–Ritz on an orthonormal subspace `V`: `G = Vᵀ A V` is a general
//! real matrix, diagonalized by LAPACK `dgeev`. A target Ritz pair `(θ, y)`
//! gives `x = V y` (unit norm) and the residual `r = A x − θ x`. The
//! correction is Olsen's `t = P r − ε P x`, `P = (θ − D)⁻¹`, `ε = xᵀ P r /
//! xᵀ P x` (so `xᵀ t = 0`), the same step as the TDDFT solver
//! (`ferric_tddft::davidson`), with `D` the caller's diagonal.
//!
//! For a non-symmetric operator the eigenvalue error of a pair with residual
//! `r` is first order in `‖r‖` (`|Δθ| ≲ ‖r‖ ‖l‖ / |lᵀx|` with `l` the left
//! eigenvector), not second order as in the symmetric case, so `conv_tol` is
//! a bound on the energy error up to that condition factor.
//!
//! Targets are the lowest `n_roots` Ritz values by real part above
//! [`POSITIVE_ROOT`](crate::eom_davidson::POSITIVE_ROOT). A target with a
//! non-negligible imaginary part (a complex
//! pair of the projected matrix, which a real symmetric operator never
//! produces) is unconverged: both its real and imaginary Ritz vectors join the
//! subspace, and a pair still complex at `max_iter` is an error.
//!
//! Guess vectors sit on the lowest diagonal elements (ties at the cut
//! included), each with a small deterministic dense part so the subspace
//! reaches every point-group irrep. A collapse keeps the lowest Ritz vectors
//! and reuses their products.

use ferric_core::FerricError;
use ndarray::{Array1, Array2, Axis};
use ndarray_linalg::Eig;

/// Ritz values at or below this are not roots.
pub const POSITIVE_ROOT: f64 = 1e-10;

/// A Ritz value is complex when `|Im θ|` exceeds this times `max(1, |Re θ|)`.
const COMPLEX_REL: f64 = 1e-10;

/// A correction whose norm after two Gram–Schmidt passes is below this
/// fraction of its norm before is already spanned.
const NULL_DIRECTION_REL: f64 = 1e-10;

/// Smallest `|θ − D|` the diagonal preconditioner divides by.
const PRECOND_FLOOR: f64 = 1e-8;

/// Relative size of the dense part of each guess vector.
const GUESS_SPREAD: f64 = 0.1;

/// Iteration controls of a non-symmetric Davidson solve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NonsymDavidsonSettings {
    /// Converged when every target root's residual norm `‖A x − θ x‖`
    /// (unit `x`) is at or below this (Hartree).
    pub conv_tol: f64,
    /// Iteration cap; reaching it with an unconverged root is an error.
    pub max_iter: usize,
    /// Largest subspace before a collapse. `None`: `max(n_guess + 10·n_roots,
    /// 50)`. Capped at the dimension.
    pub max_space: Option<usize>,
}

impl Default for NonsymDavidsonSettings {
    fn default() -> Self {
        Self {
            conv_tol: 1e-6,
            max_iter: 100,
            max_space: None,
        }
    }
}

/// Converged roots of a non-symmetric Davidson solve.
#[derive(Debug, Clone)]
pub struct NonsymDavidsonOutcome {
    /// The lowest `n_roots` eigenvalues above [`POSITIVE_ROOT`], ascending.
    pub eigenvalues: Vec<f64>,
    /// Column n: the unit-norm right eigenvector of root n.
    pub vectors: Array2<f64>,
    /// Final residual norm of each root.
    pub residuals: Vec<f64>,
    /// Iterations taken.
    pub iterations: usize,
    /// Vectors the product was applied to, in total.
    pub products: usize,
}

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

/// A deterministic value in [−1, 1) from `seed` (one SplitMix64 step).
fn splitmix_unit(seed: u64) -> f64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
}

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

/// Orthonormalize the columns of `t` against the orthonormal columns of `v`
/// and among themselves (two classical Gram–Schmidt passes), dropping
/// columns already spanned.
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
    let mut out = Array2::<f64>::zeros((v.nrows(), kept.len()));
    for (k, c) in kept.into_iter().enumerate() {
        out.column_mut(k).assign(&c);
    }
    out
}

fn append_columns(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    ndarray::concatenate(Axis(1), &[a.view(), b.view()]).expect("same row count")
}

/// Olsen's correction `t = P r − ε P x`, `P = (θ − D)⁻¹`, `xᵀ t = 0`.
fn olsen_correction(
    r: &Array1<f64>,
    x: &ndarray::ArrayView1<f64>,
    theta: f64,
    diag: &Array1<f64>,
) -> Array1<f64> {
    let p = diag.mapv(|d| {
        let den = theta - d;
        1.0 / if den.abs() < PRECOND_FLOOR {
            PRECOND_FLOOR.copysign(den)
        } else {
            den
        }
    });
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

/// One Ritz pair of the projected matrix, real part of the coefficients and
/// (for a complex value) the imaginary part.
struct Ritz {
    theta: f64,
    complex: bool,
    re: Array1<f64>,
    im: Array1<f64>,
}

/// Ritz pairs of `G`, ordered by real part, keeping those above
/// [`POSITIVE_ROOT`].
fn ritz_pairs(g: &Array2<f64>) -> Result<Vec<Ritz>, FerricError> {
    let (w, y) = g.eig().map_err(|e| {
        FerricError::General(format!(
            "EOM Davidson: subspace diagonalization failed: {e}"
        ))
    })?;
    let mut order: Vec<usize> = (0..w.len()).collect();
    order.sort_by(|&a, &b| w[a].re.total_cmp(&w[b].re).then(a.cmp(&b)));
    let mut out = Vec::new();
    for k in order {
        if w[k].re <= POSITIVE_ROOT {
            continue;
        }
        // Of a conjugate pair keep the member with Im > 0 (its vector carries
        // both real directions).
        let complex = w[k].im.abs() > COMPLEX_REL * w[k].re.abs().max(1.0);
        if complex && w[k].im < 0.0 {
            continue;
        }
        let col = y.column(k);
        out.push(Ritz {
            theta: w[k].re,
            complex,
            re: col.mapv(|c| c.re),
            im: col.mapv(|c| c.im),
        });
    }
    Ok(out)
}

fn residual_list(residuals: &[f64]) -> String {
    let listed: Vec<String> = residuals.iter().map(|r| format!("{r:.2e}")).collect();
    listed.join(", ")
}

/// The lowest `n_roots` right eigenpairs (by real part, above
/// [`POSITIVE_ROOT`]) of the operator whose product is `apply` (columns of
/// the argument → columns of `A x`), preconditioned by `diag`.
///
/// # Errors
///
/// An unconverged or still-complex root after `settings.max_iter`
/// iterations; a subspace that can grow no further with a root above
/// tolerance; fewer than `n_roots` roots above [`POSITIVE_ROOT`] in the full
/// space; a failed product or subspace diagonalization.
pub fn davidson_nonsymmetric<F>(
    mut apply: F,
    diag: &Array1<f64>,
    n_roots: usize,
    settings: &NonsymDavidsonSettings,
) -> Result<NonsymDavidsonOutcome, FerricError>
where
    F: FnMut(&Array2<f64>) -> Result<Array2<f64>, FerricError>,
{
    let dim = diag.len();
    let n_roots = n_roots.min(dim);
    let guess = guess_indices(diag, n_roots);
    let max_space = settings
        .max_space
        .unwrap_or_else(|| (guess.len() + 10 * n_roots).max(50))
        .max(guess.len() + 2 * n_roots)
        .min(dim);
    let mut v = guess_vectors(dim, &guess);
    let mut av = apply(&v)?;
    let mut products = v.ncols();
    let mut residuals: Vec<f64> = Vec::new();
    for iter in 1..=settings.max_iter {
        let ritz = ritz_pairs(&v.t().dot(&av))?;
        let tg = &ritz[..ritz.len().min(n_roots)];
        let mut corr = Array2::<f64>::zeros((dim, 0));
        let mut x_all = Array2::<f64>::zeros((dim, tg.len()));
        residuals.clear();
        let mut any_complex = false;
        for (k, p) in tg.iter().enumerate() {
            let x0 = v.dot(&p.re);
            let nx = x0.dot(&x0).sqrt();
            let x = &x0 / nx;
            let ax = av.dot(&p.re) / nx;
            let r = &ax - &(p.theta * &x);
            let rn = r.dot(&r).sqrt();
            x_all.column_mut(k).assign(&x);
            if p.complex {
                any_complex = true;
                residuals.push(f64::INFINITY);
                let xi = v.dot(&p.im);
                corr = append_columns(&corr, &xi.insert_axis(Axis(1)));
            } else {
                residuals.push(rn);
            }
            if p.complex || rn > settings.conv_tol {
                let t = olsen_correction(&r, &x.view(), p.theta, diag);
                corr = append_columns(&corr, &t.insert_axis(Axis(1)));
            }
        }
        let full_space = v.ncols() == dim;
        if corr.ncols() == 0 && (tg.len() == n_roots || full_space) && !any_complex {
            if tg.len() < n_roots {
                return Err(FerricError::General(format!(
                    "EOM Davidson: only {} of {n_roots} roots lie above {POSITIVE_ROOT:e} in the \
                     {dim}-dimensional space",
                    tg.len()
                )));
            }
            return Ok(NonsymDavidsonOutcome {
                eigenvalues: tg.iter().map(|p| p.theta).collect(),
                vectors: x_all,
                residuals,
                iterations: iter,
                products,
            });
        }
        if v.ncols() + corr.ncols() > max_space {
            // Collapse onto the lowest Ritz vectors (real and imaginary parts
            // of complex ones), reusing the products: Q = V C with C = Vᵀ Q
            // exact because Q lies in span(V).
            let keep = guess.len().max(n_roots).min(ritz.len());
            let mut coef = Array2::<f64>::zeros((v.ncols(), 0));
            for p in &ritz[..keep] {
                coef = append_columns(&coef, &p.re.clone().insert_axis(Axis(1)));
                if p.complex {
                    coef = append_columns(&coef, &p.im.clone().insert_axis(Axis(1)));
                }
            }
            let q = orthonormalize_against(&Array2::zeros((dim, 0)), v.dot(&coef));
            let c = v.t().dot(&q);
            av = av.dot(&c);
            v = q;
        }
        let new = orthonormalize_against(&v, corr);
        if new.ncols() == 0 {
            return Err(FerricError::General(format!(
                "EOM Davidson: the subspace ({} of {dim} dimensions) cannot grow, with root \
                 residuals [{}] above conv_tol {:e}",
                v.ncols(),
                residual_list(&residuals),
                settings.conv_tol
            )));
        }
        let anew = apply(&new)?;
        products += new.ncols();
        v = append_columns(&v, &new);
        av = append_columns(&av, &anew);
    }
    Err(FerricError::General(format!(
        "EOM Davidson did not converge in {} iterations: root residuals [{}] against conv_tol \
         {:e} (inf = a complex Ritz pair)",
        settings.max_iter,
        residual_list(&residuals),
        settings.conv_tol
    )))
}

/// All eigenvalues of the dense matrix `a` with real part above
/// [`POSITIVE_ROOT`], ascending by real part, as `(re, im)` pairs. The
/// oracle the Davidson roots are tested against.
///
/// # Errors
///
/// A failed LAPACK `dgeev`.
pub fn dense_eigenvalues(a: &Array2<f64>) -> Result<Vec<(f64, f64)>, FerricError> {
    let (w, _) = a
        .eig()
        .map_err(|e| FerricError::General(format!("dense eig failed: {e}")))?;
    let mut out: Vec<(f64, f64)> = w
        .iter()
        .filter(|c| c.re > POSITIVE_ROOT)
        .map(|c| (c.re, c.im))
        .collect();
    out.sort_by(|a, b| a.0.total_cmp(&b.0));
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A non-symmetric matrix with known real spectrum: S diag(λ) S⁻¹ with S
    /// upper unit-triangular, so S⁻¹ is exact by back substitution.
    fn similar_to_diagonal(lams: &[f64]) -> Array2<f64> {
        let n = lams.len();
        let mut s = Array2::<f64>::eye(n);
        for i in 0..n {
            for j in i + 1..n {
                s[(i, j)] = 0.05 * splitmix_unit((i * n + j) as u64);
            }
        }
        // S⁻¹ of a unit upper-triangular matrix, column by column.
        let mut sinv = Array2::<f64>::eye(n);
        for j in 0..n {
            for i in (0..j).rev() {
                let mut acc = 0.0;
                for k in i + 1..=j {
                    acc += s[(i, k)] * sinv[(k, j)];
                }
                sinv[(i, j)] = -acc;
            }
        }
        let d = Array2::from_diag(&Array1::from(lams.to_vec()));
        s.dot(&d).dot(&sinv)
    }

    #[test]
    fn finds_the_lowest_roots_of_a_non_symmetric_matrix() {
        let n = 120;
        let lams: Vec<f64> = (0..n).map(|k| 0.3 + 0.01 * k as f64).collect();
        let a = similar_to_diagonal(&lams);
        let asym = (&a - &a.t()).iter().map(|x| x.abs()).fold(0.0, f64::max);
        assert!(asym > 1e-3, "test matrix is not non-symmetric enough");
        let diag = a.diag().to_owned();
        let out = davidson_nonsymmetric(
            |x| Ok(a.dot(x)),
            &diag,
            4,
            &NonsymDavidsonSettings {
                conv_tol: 1e-10,
                max_iter: 200,
                max_space: Some(20),
            },
        )
        .unwrap();
        for (k, &e) in out.eigenvalues.iter().enumerate() {
            assert!((e - lams[k]).abs() < 1e-9, "root {k}: {e} vs {}", lams[k]);
        }
        let dense = dense_eigenvalues(&a).unwrap();
        assert!((dense[0].0 - lams[0]).abs() < 1e-12);
    }
}
