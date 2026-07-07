//! Initial density matrix guess for the SCF procedure.

use ferric_core::FerricError;
use ndarray::Array2;
use ndarray_linalg::Eigh;

/// Generate an initial density matrix from the core Hamiltonian eigenvectors.
///
/// Diagonalizes H in the canonically-orthogonalized basis and occupies the
/// lowest `nocc` orbitals to form D = 2 * C_occ * C_occ^T. Uses the same
/// linear-dependence filtering as the SCF loop's orthogonalizer, so a
/// near-singular overlap (e.g. aug bases on clustered atoms) is dropped from
/// the guess instead of seeding the SCF with an Inf/NaN density.
pub fn hcore_guess(
    s: &Array2<f64>,
    h: &Array2<f64>,
    nocc: usize,
) -> Result<Array2<f64>, FerricError> {
    let x = crate::rhf::canonical_orthogonalizer(s)?; // (n, m), m ≤ n
    let m = x.ncols();
    if nocc > m {
        return Err(FerricError::General(format!(
            "nocc = {nocc} exceeds the orthogonalized basis dimension {m} (nbasis = {}) — check charge and basis set",
            s.nrows()
        )));
    }
    // H' = Xᵀ H X
    let h_prime = x.t().dot(h).dot(&x);
    let (_, c_prime) = h_prime
        .eigh(ndarray_linalg::UPLO::Upper)
        .map_err(|e| FerricError::Lapack(format!("H' diag: {e}")))?;
    // C = X C'
    let c = x.dot(&c_prime);
    // D = 2 C_occ C_occ^T
    let c_occ = c.slice(ndarray::s![.., ..nocc]);
    let d = c_occ.dot(&c_occ.t()) * 2.0;
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
    fn hcore_guess_near_singular_overlap_is_finite() {
        // Overlap with an exactly-zero eigenvalue (perfect linear dependence).
        // The guess must drop the singular mode like the SCF loop's canonical
        // orthogonalizer — not seed the SCF with an Inf/NaN density.
        let s = ndarray::array![[1.0, 1.0], [1.0, 1.0]];
        let h = ndarray::array![[-1.0, -0.5], [-0.5, -1.0]];
        let d = hcore_guess(&s, &h, 1).unwrap();
        assert!(
            d.iter().all(|v| v.is_finite()),
            "guess density has non-finite entries: {d:?}"
        );
    }

    #[test]
    fn hcore_guess_nocc_exceeding_basis_is_an_error() {
        // More occupied orbitals than basis functions (e.g. a charge typo in
        // the input) must be a clean Err, not a slice panic.
        let s = ndarray::array![[1.0, 0.0], [0.0, 1.0]];
        let h = ndarray::array![[-1.0, 0.0], [0.0, -1.0]];
        let res = hcore_guess(&s, &h, 3);
        assert!(res.is_err(), "nocc > nbasis must be an error, got {res:?}");
    }

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
