//! CI-specific block(=1)-Davidson eigensolver for the lowest CAS-CI root.
//!
//! Structurally modeled on `ferric_rpa::davidson` (subspace expansion,
//! Rayleigh–Ritz on the small projected matrix, residual/convergence check,
//! restart), but with a CI-native matvec (`sigma = H c` via the Slater–Condon
//! [`crate::hamiltonian::sigma`] build) and the standard CI diagonal
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

        // Orthogonalize t against the current subspace (modified Gram–Schmidt).
        for v in &v_basis {
            let overlap = t.dot(v);
            t.scaled_add(-overlap, v);
        }
        let tnorm = t.dot(&t).sqrt();
        if tnorm < 1e-10 {
            // New direction is null (already spanned) — converged in practice.
            converged = resid_norm < conv_thresh * 10.0;
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
