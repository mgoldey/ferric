//! Davidson subspace eigensolver for PDEP dielectric eigenpotentials.

use ferric_core::FerricError;
use ndarray::{Array1, Array2};
use ndarray_linalg::{Eigh, QR, UPLO};

pub struct DavidsonResult {
    /// Converged eigenvalues λ_α, sorted descending (most significant first).
    pub eigenvalues: Vec<f64>,
    /// Eigenpotentials, shape (naux, n_converged). Columns = V_α.
    pub eigenvectors: Array2<f64>,
}

/// Run Davidson at ω=0 to find the leading eigenpotentials of ε̃(0).
///
/// `dielectric_fn(v_mat, omega)` returns the projected dielectric matrix
/// ε̃_αβ(iω) for trial vectors V (columns of v_mat).
///
/// `m0`: initial subspace dimension (= naux for RI-seed)
/// `n_desired`: number of eigenpairs to extract
pub fn run_davidson_static<F>(
    m0: usize,
    dielectric_fn: F,
    conv_thresh: f64,
    max_vecs: usize,
    n_desired: usize,
) -> Result<DavidsonResult, FerricError>
where
    F: Fn(&Array2<f64>, f64) -> Array2<f64>,
{
    // Seed: identity subspace (unit vectors in aux space)
    let seed = Array2::eye(m0);
    run_davidson_seeded(seed, dielectric_fn, conv_thresh, max_vecs, n_desired)
}

/// Run Davidson with an explicit seed matrix.
///
/// `seed`: initial trial subspace, shape (naux, n_seed). Columns are orthonormal
///   trial vectors in the dressed aux basis. Davidson will grow this subspace
///   as needed until `n_desired` eigenpairs converge.
///
/// `dielectric_fn(v_mat, omega)` returns the projected dielectric matrix
/// ε̃_αβ(iω) for trial vectors V (columns of v_mat).
///
/// `n_desired`: number of eigenpairs to extract
pub fn run_davidson_seeded<F>(
    seed: Array2<f64>,
    dielectric_fn: F,
    conv_thresh: f64,
    max_vecs: usize,
    n_desired: usize,
) -> Result<DavidsonResult, FerricError>
where
    F: Fn(&Array2<f64>, f64) -> Array2<f64>,
{
    // Orthonormalize the seed to get the initial subspace
    let mut v_mat = if seed.ncols() <= seed.nrows() {
        qr_orthonormalize(seed)?
    } else {
        seed
    };

    let max_iter = 200;
    for _iter in 0..max_iter {
        let m = v_mat.ncols();

        // Form projected dielectric
        let eps_proj = dielectric_fn(&v_mat, 0.0);

        // Diagonalize (symmetric)
        let (evals, evecs) = eps_proj.eigh(UPLO::Upper)
            .map_err(|e| FerricError::General(format!("Davidson diagonalization failed: {e}")))?;

        // Ritz vectors in original space: V @ evecs (naux, m)
        let ritz = v_mat.dot(&evecs);

        // Compute residuals for the n_desired largest eigenvalues
        let mut max_resid = 0.0f64;
        let mut new_vecs: Vec<Array1<f64>> = Vec::new();
        let m_check = m.min(n_desired + 2);
        for k in (m.saturating_sub(m_check))..m {
            let lk = evals[k];
            let vk = ritz.column(k);

            // residual = ε̃ @ vk − lk * vk
            let eps_vk = v_mat.dot(&eps_proj.dot(&evecs.column(k)));
            let resid: Array1<f64> = &eps_vk - &(lk * &vk);
            let resid_norm = resid.dot(&resid).sqrt();
            max_resid = max_resid.max(resid_norm);

            if resid_norm > conv_thresh {
                // Davidson preconditioner: divide by shift (ε̃ ≥ 1)
                let denom = (lk - 1.0).abs().max(0.1);
                let t: Array1<f64> = resid.mapv(|x| x / denom);
                new_vecs.push(t);
            }
        }

        // All checked eigenpairs have converged.
        if max_resid < conv_thresh || new_vecs.is_empty() {
            if m >= n_desired {
                // Full convergence: we have enough eigenpairs.
                let n_keep = n_desired.min(m);
                let start = m - n_keep;
                let eigenvalues: Vec<f64> = evals.slice(ndarray::s![start..]).iter().copied().rev().collect();
                let eigenvectors = ritz.slice(ndarray::s![.., start..]).to_owned();
                // Reverse columns so largest eigenvalue is first
                let eigenvectors = eigenvectors.slice(ndarray::s![.., ..;-1]).to_owned();
                return Ok(DavidsonResult { eigenvalues, eigenvectors });
            }

            // m < n_desired: current Ritz vectors all converged, but subspace is too small.
            // This happens when the seed spans only a subset of the eigenpairs.
            // Expand with unit vectors orthogonal to the current subspace.
            let naux = v_mat.nrows();
            let budget = max_vecs.saturating_sub(m);
            if budget == 0 || m >= naux {
                // No room to grow: return what we have (truncation step will handle it).
                let eigenvalues: Vec<f64> = evals.iter().copied().rev().collect();
                let eigenvectors = ritz.slice(ndarray::s![.., ..;-1]).to_owned();
                return Ok(DavidsonResult { eigenvalues, eigenvectors });
            }
            // Add a batch of orthogonal unit vectors to bootstrap coverage of the missing space.
            let n_to_add = (n_desired - m).min(budget).min(naux - m);
            let mut expanded = Array2::zeros((naux, m + n_to_add));
            expanded.slice_mut(ndarray::s![.., ..m]).assign(&v_mat);
            let mut added = 0;
            for ei in 0..naux {
                if added >= n_to_add { break; }
                // Gram-Schmidt: project unit vector against all current columns
                let mut e = Array1::zeros(naux);
                e[ei] = 1.0;
                for col_idx in 0..(m + added) {
                    let col = expanded.column(col_idx);
                    let proj = col.dot(&e);
                    let col_owned = col.to_owned();
                    e = e - proj * &col_owned;
                }
                let norm = e.dot(&e).sqrt();
                if norm > 1e-10 {
                    e.mapv_inplace(|x| x / norm);
                    expanded.column_mut(m + added).assign(&e);
                    added += 1;
                }
            }
            if added > 0 {
                v_mat = expanded.slice(ndarray::s![.., ..(m + added)]).to_owned();
            }
            // No need to re-QR here since we built ortho columns incrementally.
            continue;
        }

        if v_mat.ncols() + new_vecs.len() > max_vecs {
            // Restart: keep only current Ritz vectors
            let n_keep = n_desired.min(m);
            let start = m - n_keep;
            v_mat = ritz.slice(ndarray::s![.., start..]).to_owned();
        } else {
            // Expand subspace
            let naux = v_mat.nrows();
            let n_new = new_vecs.len();
            let mut expanded = Array2::zeros((naux, v_mat.ncols() + n_new));
            expanded.slice_mut(ndarray::s![.., ..v_mat.ncols()]).assign(&v_mat);
            for (j, t) in new_vecs.iter().enumerate() {
                expanded.slice_mut(ndarray::s![.., v_mat.ncols() + j]).assign(t);
            }
            v_mat = expanded;
        }

        // Orthonormalize via QR
        v_mat = qr_orthonormalize(v_mat)?;
    }

    Err(FerricError::General("Davidson did not converge".into()))
}

/// Orthonormalize columns of mat via QR decomposition.
fn qr_orthonormalize(mat: Array2<f64>) -> Result<Array2<f64>, FerricError> {
    let (q, _r) = mat.qr()
        .map_err(|e| FerricError::General(format!("QR failed: {e}")))?;
    Ok(q)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn davidson_recovers_known_eigenvalues() {
        // 2D dielectric: ε̃ = [[2.0, 0.5], [0.5, 3.0]]
        // Eigenvalues: ~1.809 and ~3.191
        // We'll test that Davidson finds them given the explicit matrix.
        use ndarray::array;

        let result = run_davidson_static(
            2,      // naux = m0
            |v_mat: &Array2<f64>, _omega: f64| -> Array2<f64> {
                let fixed = array![[2.0f64, 0.5], [0.5, 3.0]];
                v_mat.t().dot(&fixed.dot(v_mat))
            },
            1e-6,  // conv_thresh
            20,    // max_vecs
            2,     // n_desired
        ).unwrap();

        let mut evals = result.eigenvalues.clone();
        evals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // For 2x2 [[2,0.5],[0.5,3]]: trace=5, det=6-0.25=5.75
        // λ = (5 ± sqrt(2)) / 2
        let expected_lo = (5.0 - 2.0f64.sqrt()) / 2.0;
        let expected_hi = (5.0 + 2.0f64.sqrt()) / 2.0;
        assert!((evals[0] - expected_lo).abs() < 1e-4,
            "λ_0={} expected {}", evals[0], expected_lo);
        assert!((evals[1] - expected_hi).abs() < 1e-4,
            "λ_1={} expected {}", evals[1], expected_hi);
    }
}
