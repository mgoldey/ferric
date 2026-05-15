//! Boys orbital localization.
//!
//! Maximizes the Boys functional F = Σ_i (⟨i|x|i⟩² + ⟨i|y|i⟩² + ⟨i|z|i⟩²)
//! via 2×2 Jacobi rotations (Foster & Boys 1960).

use ndarray::Array2;

/// Result of Boys localization.
pub struct BoysResult {
    /// Localized MO coefficients, shape (nbas, nocc).
    pub c_loc: Array2<f64>,
    /// Boys centers [nocc × 3]: ⟨i|r|i⟩ for each localized orbital.
    pub centers: Array2<f64>,
    /// Number of sweeps performed.
    pub iterations: usize,
    /// Whether the localization converged.
    pub converged: bool,
}

/// Boys localization of occupied orbitals.
///
/// `c_occ`: (nbas, nocc) — canonical occupied MO coefficient matrix.
/// `dip`: [x_mat, y_mat, z_mat] — dipole integral matrices (nbas × nbas), AO basis.
/// `max_iter`: maximum number of sweeps over all pairs.
///
/// Returns localized orbitals and Boys centers.
pub fn boys_localize(
    c_occ: &Array2<f64>,
    dip: &[Array2<f64>; 3],
    max_iter: usize,
) -> BoysResult {
    let nbas = c_occ.nrows();
    let nocc = c_occ.ncols();

    // Working copy of the coefficient matrix.
    let mut c = c_occ.to_owned();

    // Compute MO dipole matrices in the occupied subspace: d[alpha]_ij = C^T @ dip_alpha @ C
    // We keep these as nocc×nocc matrices and update them in-place upon each rotation.
    let mut d: [Array2<f64>; 3] = std::array::from_fn(|alpha| {
        // d_ij = sum_mu sum_nu C_mu_i * dip_alpha[mu,nu] * C_nu_j
        let dip_c: Array2<f64> = dip[alpha].dot(&c); // (nbas, nocc)
        c.t().dot(&dip_c) // (nocc, nocc)
    });

    let mut converged = false;
    let mut iterations = 0;

    'outer: for _sweep in 0..max_iter {
        iterations += 1;
        let mut max_theta: f64 = 0.0;

        for i in 0..nocc {
            for j in (i + 1)..nocc {
                // For each coordinate alpha, compute:
                //   d_alpha = d[alpha][i,i] - d[alpha][j,j]   (diagonal difference)
                //   o_alpha = d[alpha][i,j]                   (off-diagonal)
                // A = Σ_alpha (o_alpha^2 - d_alpha^2/4)
                // B = Σ_alpha d_alpha * o_alpha / 2
                // theta = atan2(B, -A) / 4
                // PySCF Boys formula:
                // A = Σ_α [ (q_ii - q_jj)²/4 - q_ij² ]
                // B = Σ_α q_ij * (q_ii - q_jj)
                // θ = atan2(B, -A) / 4  (maximizes Boys functional)
                let mut big_a = 0.0_f64;
                let mut big_b = 0.0_f64;
                for alpha in 0..3 {
                    let da = d[alpha][(i, i)] - d[alpha][(j, j)]; // q_ii - q_jj
                    let oa = d[alpha][(i, j)];                     // q_ij
                    big_a += da * da / 4.0 - oa * oa;
                    big_b += oa * da;
                }

                // Optimal rotation angle maximizing Boys functional.
                let theta = f64::atan2(big_b, -big_a) / 4.0;
                if theta.abs() > max_theta {
                    max_theta = theta.abs();
                }

                if theta.abs() < 1e-12 {
                    continue;
                }

                let cos_t = theta.cos();
                let sin_t = theta.sin();

                // Rotate coefficient columns i and j.
                for mu in 0..nbas {
                    let ci = c[(mu, i)];
                    let cj = c[(mu, j)];
                    c[(mu, i)] = cos_t * ci - sin_t * cj;
                    c[(mu, j)] = sin_t * ci + cos_t * cj;
                }

                // Update MO dipole matrices for all alpha.
                // For each alpha, update row/col i and j of d[alpha].
                // New d[alpha]_pi = cos(t)*d_pi_old - sin(t)*d_pj_old  for all p
                // New d[alpha]_pj = sin(t)*d_pi_old + cos(t)*d_pj_old  for all p
                // Then similarly for rows.
                for alpha in 0..3 {
                    // Update columns i and j.
                    for p in 0..nocc {
                        let dpi = d[alpha][(p, i)];
                        let dpj = d[alpha][(p, j)];
                        d[alpha][(p, i)] = cos_t * dpi - sin_t * dpj;
                        d[alpha][(p, j)] = sin_t * dpi + cos_t * dpj;
                    }
                    // Update rows i and j.
                    for p in 0..nocc {
                        let dip_val = d[alpha][(i, p)];
                        let djp = d[alpha][(j, p)];
                        d[alpha][(i, p)] = cos_t * dip_val - sin_t * djp;
                        d[alpha][(j, p)] = sin_t * dip_val + cos_t * djp;
                    }
                }
            }
        }

        if max_theta < 1e-8 {
            converged = true;
            break 'outer;
        }
    }

    // Extract Boys centers: ⟨i|r_alpha|i⟩ = d[alpha][i,i]
    let mut centers = Array2::zeros((nocc, 3));
    for i in 0..nocc {
        for alpha in 0..3 {
            centers[(i, alpha)] = d[alpha][(i, i)];
        }
    }

    BoysResult {
        c_loc: c,
        centers,
        iterations,
        converged,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::basis;
    use ferric_core::mol::Molecule;
    use ferric_core::parallel::ParallelContext;
    use ferric_integrals::basis_bridge::PreparedBasis;
    use ferric_integrals::oneelectron::dipole;
    use ferric_integrals::operator::Operator;
    use ferric_scf::rhf::{solve_rhf, RhfConfig};
    use ferric_scf::screening::SchwarzBounds;
    use ndarray::s;

    #[test]
    fn test_boys_water_sto3g() {
        // Run RHF on water, then localize occupied orbitals.
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();

        let ctx = ParallelContext::default();
        let config = RhfConfig::default();
        let rhf = solve_rhf(&ctx, &mol, &prep, op, &bounds, &config).unwrap();

        let nocc = mol.nelec() as usize / 2;
        // c_occ: (nbas, nocc)
        let c_occ = rhf.mos.slice(s![.., ..nocc]).to_owned();

        // Compute dipole integrals at origin (0,0,0)
        let dip = dipole(&prep, [0.0, 0.0, 0.0]);

        let result = boys_localize(&c_occ, &dip, 200);

        println!(
            "Boys localization: {} iterations, converged={}",
            result.iterations, result.converged
        );
        println!("Centers:\n{:.4}", result.centers);

        assert!(
            result.converged,
            "Boys localization did not converge in {} iterations",
            result.iterations
        );
        assert!(
            result.iterations <= 200,
            "Too many iterations: {}",
            result.iterations
        );

        // The localized orbital centers should lie near water molecule atoms.
        // Water has 5 occupied orbitals in STO-3G: 1 O core, 2 O lone pairs, 2 O-H bonds.
        assert_eq!(result.centers.nrows(), nocc);
        assert_eq!(result.centers.ncols(), 3);

        // Check that the localized orbitals are still orthonormal via C^T S C = I
        use ferric_integrals::oneelectron::overlap;
        let s = overlap(&prep);
        let sc = s.dot(&result.c_loc);
        let cts_c = result.c_loc.t().dot(&sc);
        for i in 0..nocc {
            for j in 0..nocc {
                let expected = if i == j { 1.0 } else { 0.0 };
                let diff = (cts_c[(i, j)] - expected).abs();
                assert!(
                    diff < 1e-8,
                    "Orthonormality violated at ({i},{j}): got {:.2e}",
                    diff
                );
            }
        }
    }
}
