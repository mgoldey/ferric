//! CI-specific block(=1)-Davidson eigensolver for the lowest CAS-CI root.
//!
//! Structurally modeled on `ferric_rpa::davidson` (subspace expansion,
//! Rayleigh–Ritz on the small projected matrix, residual/convergence check,
//! restart), but with a caller-supplied CI matvec (`sigma = H c`; the driver
//! passes [`crate::sigma::DirectSigma::apply`]) and the standard CI diagonal
//! preconditioner `t = r / (H_ii - lambda)`.
//!
//! Phase A targets a single lowest root, so this is a plain (block size 1)
//! Davidson. Multiple roots (block Davidson) are a later phase.

use ferric_core::FerricError;
use ndarray::{Array1, Array2};
use ndarray_linalg::{Eigh, UPLO};

/// Result of the CI Davidson solve for the lowest root.
#[derive(Debug, Clone, PartialEq)]
pub struct CiDavidsonResult {
    /// Lowest eigenvalue (total active-space energy incl. e_core, since e_core
    /// is folded into the Hamiltonian diagonal).
    pub eigenvalue: f64,
    /// Corresponding normalized CI eigenvector, length n_det.
    pub eigenvector: Vec<f64>,
    /// Iterations taken.
    pub iterations: usize,
    /// Whether the residual norm fell below `conv_thresh`.
    pub converged: bool,
}

/// A correction whose norm after orthogonalization is below this fraction of
/// its norm before is treated as already spanned by the subspace. Two
/// Gram–Schmidt passes leave a spanned vector at round-off relative size
/// (~1e-16·√m), so 1e-10 separates it from a genuine new direction with a
/// wide margin on both sides.
const NULL_DIRECTION_REL: f64 = 1e-10;

/// Find the lowest eigenpair of the CI Hamiltonian defined implicitly by the
/// `matvec` closure (`sigma = H c`) and the diagonal `diag` (H_ii, used for
/// preconditioning and the initial guess).
///
/// * `ndet` — dimension of the CI space.
/// * `matvec` — computes `H c` for a coefficient vector `c` (length ndet).
/// * `diag` — the Hamiltonian diagonal, length ndet.
/// * `conv_thresh` — residual-norm convergence threshold.
/// * `max_subspace` — maximum subspace size before a collapse/restart.
/// * `max_iter` — hard iteration cap.
pub fn davidson_lowest<F>(
    ndet: usize,
    matvec: F,
    diag: &[f64],
    conv_thresh: f64,
    max_subspace: usize,
    max_iter: usize,
) -> Result<CiDavidsonResult, FerricError>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    if ndet == 0 {
        return Err(FerricError::General(
            "CI Davidson: empty determinant space".to_string(),
        ));
    }
    // Trivial 1x1 space.
    if ndet == 1 {
        let e = diag[0];
        return Ok(CiDavidsonResult {
            eigenvalue: e,
            eigenvector: vec![1.0],
            iterations: 0,
            converged: true,
        });
    }

    let max_sub = max_subspace.max(2).min(ndet);

    // Initial guess: unit vector on the lowest-diagonal determinant.
    let mut lowest = 0usize;
    for (i, &d) in diag.iter().enumerate() {
        if d < diag[lowest] {
            lowest = i;
        }
    }

    // Subspace of trial vectors (columns), and their H-images.
    let mut v_basis: Vec<Array1<f64>> = Vec::new();
    let mut hv_basis: Vec<Array1<f64>> = Vec::new();

    let mut guess = Array1::<f64>::zeros(ndet);
    guess[lowest] = 1.0;
    v_basis.push(guess);

    let mut theta = diag[lowest];
    let mut best_vec = {
        let mut v = vec![0.0f64; ndet];
        v[lowest] = 1.0;
        v
    };
    let mut converged = false;
    let mut iters = 0usize;

    for iter in 0..max_iter {
        iters = iter + 1;
        // Ensure H-images exist for every basis vector.
        while hv_basis.len() < v_basis.len() {
            let idx = hv_basis.len();
            let hv = matvec(v_basis[idx].as_slice().unwrap());
            hv_basis.push(Array1::from(hv));
        }

        let m = v_basis.len();
        // Small projected matrix G = Vᵀ H V (symmetric).
        let mut g = Array2::<f64>::zeros((m, m));
        for i in 0..m {
            for j in i..m {
                let val = v_basis[i].dot(&hv_basis[j]);
                g[(i, j)] = val;
                g[(j, i)] = val;
            }
        }
        let (evals, evecs) = g
            .eigh(UPLO::Upper)
            .map_err(|e| FerricError::General(format!("CI Davidson eigh failed: {e}")))?;
        // Lowest Ritz pair (eigh returns ascending).
        theta = evals[0];
        let y = evecs.column(0);

        // Ritz vector x = sum_i y_i v_i ; Hx = sum_i y_i (H v_i).
        let mut x = Array1::<f64>::zeros(ndet);
        let mut hx = Array1::<f64>::zeros(ndet);
        for i in 0..m {
            x.scaled_add(y[i], &v_basis[i]);
            hx.scaled_add(y[i], &hv_basis[i]);
        }

        // Residual r = Hx - theta x.
        let resid: Array1<f64> = &hx - &(theta * &x);
        let resid_norm = resid.dot(&resid).sqrt();

        best_vec = x.to_vec();

        if resid_norm < conv_thresh {
            converged = true;
            break;
        }

        // Diagonal (Davidson) preconditioner: t_i = r_i / (H_ii - theta),
        // guarded away from zero denominators.
        let mut t = Array1::<f64>::zeros(ndet);
        for i in 0..ndet {
            let denom = diag[i] - theta;
            let denom = if denom.abs() < 1e-8 {
                if denom < 0.0 {
                    -1e-8
                } else {
                    1e-8
                }
            } else {
                denom
            };
            t[i] = resid[i] / denom;
        }

        // Orthogonalize t against the current subspace (modified Gram–Schmidt,
        // two passes). The correction's size follows the residual, so "already
        // spanned" is judged relative to its norm before orthogonalization: an
        // absolute cut would stop the solve once the residual itself reached it.
        let t0norm = t.dot(&t).sqrt();
        for _ in 0..2 {
            for v in &v_basis {
                let overlap = t.dot(v);
                t.scaled_add(-overlap, v);
            }
        }
        let tnorm = t.dot(&t).sqrt();
        if !(tnorm > NULL_DIRECTION_REL * t0norm) {
            // New direction is null (already spanned): no further progress.
            // The residual was checked against `conv_thresh` above and missed
            // it, so the solve stops unconverged.
            break;
        }
        t.mapv_inplace(|x| x / tnorm);

        // Restart/collapse if the subspace is full: rebuild from the current
        // Ritz vector plus the new correction.
        if m + 1 > max_sub {
            let xnorm = x.dot(&x).sqrt();
            let x_unit = &x / xnorm;
            v_basis.clear();
            hv_basis.clear();
            v_basis.push(x_unit);
            v_basis.push(t);
        } else {
            v_basis.push(t);
        }
    }

    // Normalize the returned eigenvector.
    let nrm = best_vec.iter().map(|x| x * x).sum::<f64>().sqrt();
    if nrm > 0.0 {
        for x in best_vec.iter_mut() {
            *x /= nrm;
        }
    }

    Ok(CiDavidsonResult {
        eigenvalue: theta,
        eigenvector: best_vec,
        iterations: iters,
        converged,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray_linalg::Eigh;

    /// Diagonal 0..n plus a small symmetric coupling: the correction
    /// r/(H_ii − θ) is about as small as the residual, so a solve asked for a
    /// residual below 1e-10 must keep going once the residual reaches it.
    fn spread_matrix(n: usize) -> Array2<f64> {
        let mut h = Array2::<f64>::zeros((n, n));
        let mut st = 0x2545_f491_4f6c_dd1du64;
        for i in 0..n {
            h[(i, i)] = i as f64;
            for j in 0..i {
                st = st
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let x = 1e-2 * (((st >> 11) as f64) / (1u64 << 53) as f64 - 0.5);
                h[(i, j)] = x;
                h[(j, i)] = x;
            }
        }
        h
    }

    /// `‖Hx − θx‖` of a returned eigenpair.
    fn residual(h: &Array2<f64>, r: &CiDavidsonResult) -> f64 {
        let x = Array1::from(r.eigenvector.clone());
        let res = &h.dot(&x) - &(r.eigenvalue * &x);
        res.dot(&res).sqrt()
    }

    /// The null-direction exit (the correction is already spanned, so the
    /// subspace cannot grow) stops an unconverged solve. The operator is
    /// `H + E` with `E` antisymmetric of size 1e-6, so no Ritz pair has a
    /// residual near round-off: a solve asked for 1e-300 grows the subspace
    /// to the full space and leaves through that exit at a floor `r0` of the
    /// order of `E`. Thresholds from `r0/5` to `r0/2` are below that floor
    /// and must not be reported met.
    #[test]
    fn null_direction_exit_is_not_converged() {
        let n = 16;
        let h = spread_matrix(n);
        let mut a = h.clone();
        for i in 0..n {
            for j in 0..i {
                let e = 1e-6 * (((i * 7 + j * 3) % 11) as f64 - 5.0) / 5.0;
                a[(i, j)] += e;
                a[(j, i)] -= e;
            }
        }
        let diag: Vec<f64> = (0..n).map(|i| a[(i, i)]).collect();
        let mv = |c: &[f64]| a.dot(&Array1::from(c.to_vec())).to_vec();
        let floor = davidson_lowest(n, mv, &diag, 1e-300, n, 100).unwrap();
        assert!(!floor.converged, "1e-300 reported met");
        let r0 = residual(&a, &floor);
        assert!(r0 > 1e-9, "floor residual {r0:e} is not above round-off");
        for f in [0.2, 0.3, 0.5] {
            let thresh = f * r0;
            let r = davidson_lowest(n, mv, &diag, thresh, n, 100).unwrap();
            assert!(
                !r.converged,
                "converged at threshold {thresh:e}, residual {:e} (floor {r0:e})",
                residual(&a, &r)
            );
        }
    }

    #[test]
    fn reaches_a_residual_below_the_correction_size() {
        let n = 16;
        let h = spread_matrix(n);
        let diag: Vec<f64> = (0..n).map(|i| h[(i, i)]).collect();
        let mv = |c: &[f64]| h.dot(&Array1::from(c.to_vec())).to_vec();
        let thresh = 1e-13;
        let r = davidson_lowest(n, mv, &diag, thresh, n, 100).unwrap();
        assert!(
            r.converged,
            "not converged after {} iterations",
            r.iterations
        );
        let x = Array1::from(r.eigenvector.clone());
        let res = &h.dot(&x) - &(r.eigenvalue * &x);
        assert!(res.dot(&res).sqrt() < thresh);
        let (evals, _) = h.eigh(UPLO::Upper).unwrap();
        assert!((r.eigenvalue - evals[0]).abs() < 1e-12);
    }
}
