use ferric_core::FerricError;
use ndarray::Array2;
use ndarray_linalg::Eigh;

pub fn hcore_guess(
    s: &Array2<f64>,
    h: &Array2<f64>,
    nocc: usize,
) -> Result<Array2<f64>, FerricError> {
    let n = s.nrows();
    // S^{-1/2} via eigendecomposition
    let (s_evals, s_evecs) = s
        .eigh(ndarray_linalg::UPLO::Upper)
        .map_err(|e| FerricError::Lapack(format!("S diag: {e}")))?;
    let mut s_inv_sqrt = Array2::zeros((n, n));
    for i in 0..n {
        let val = 1.0 / s_evals[i].sqrt();
        for mu in 0..n {
            for nu in 0..n {
                s_inv_sqrt[(mu, nu)] += s_evecs[(mu, i)] * val * s_evecs[(nu, i)];
            }
        }
    }
    // H' = S^{-1/2} H S^{-1/2}
    let h_prime = s_inv_sqrt.dot(h).dot(&s_inv_sqrt);
    let (_, c_prime) = h_prime
        .eigh(ndarray_linalg::UPLO::Upper)
        .map_err(|e| FerricError::Lapack(format!("H' diag: {e}")))?;
    // C = S^{-1/2} C'
    let c = s_inv_sqrt.dot(&c_prime);
    // D = 2 C_occ C_occ^T
    let mut d = Array2::zeros((n, n));
    for mu in 0..n {
        for nu in 0..n {
            let mut sum = 0.0;
            for i in 0..nocc {
                sum += c[(mu, i)] * c[(nu, i)];
            }
            d[(mu, nu)] = 2.0 * sum;
        }
    }
    Ok(d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::basis;
    use ferric_core::mol::Molecule;
    use ferric_integrals::basis_bridge::PreparedBasis;
    use ferric_integrals::oneelectron;

    #[test]
    fn test_hcore_guess_water_trace() {
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let s = oneelectron::overlap(&prep);
        let h = oneelectron::hcore(&prep);
        let d = hcore_guess(&s, &h, 5).unwrap();
        let n = prep.nbasis();
        // D symmetric
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (d[(i, j)] - d[(j, i)]).abs() < 1e-10,
                    "D not symmetric at ({i},{j})"
                );
            }
        }
        // tr(DS) = nelec = 10
        let tr: f64 = (0..n)
            .map(|i| (0..n).map(|j| d[(i, j)] * s[(i, j)]).sum::<f64>())
            .sum();
        assert!((tr - 10.0).abs() < 1e-6, "tr(DS) = {tr}, expected 10");
    }
}
