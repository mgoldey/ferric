//! Boys orbital localization and LMP2 domain construction.
//!
//! Maximizes the Boys functional F = Σ_i (⟨i|x|i⟩² + ⟨i|y|i⟩² + ⟨i|z|i⟩²)
//! via 2×2 Jacobi rotations (Foster & Boys 1960).
//!
//! Domain construction assigns each Boys-localized orbital i a set of AO basis
//! functions whose shells lie within a distance cutoff of its Boys center.  The
//! union of domains for a given μ index gives the sparsity pattern of the
//! Laplace pseudo-densities P(t) and Q(t).

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

/// LMP2 domain for a set of Boys-localized orbitals.
///
/// `ao_mask[μ]` is true if basis function μ belongs to at least one orbital's domain.
/// `orbital_domains[i]` is the sorted list of AO indices in orbital i's domain.
pub struct LmpDomains {
    /// For each orbital i: sorted AO indices within cutoff_bohr of its Boys center.
    pub orbital_domains: Vec<Vec<usize>>,
    /// Union mask over all orbitals: ao_mask[μ] = true if μ is in any domain.
    pub ao_mask: Vec<bool>,
    /// Sorted list of AO indices that appear in any domain.
    pub active_aos: Vec<usize>,
}

/// Build LMP2 domains from Boys centers and shell spatial information.
///
/// Each shell is assigned to an orbital's domain if the shell center (atom position)
/// lies within `cutoff_bohr` of the orbital's Boys center.  All AO functions of a
/// qualifying shell are included (shells are not split).
///
/// `centers`: (nocc × 3) Boys centers in Bohr.
/// `shell_centers`: per-shell Cartesian centers (Bohr), length nshells.
/// `shell_offsets`: cumulative AO index for each shell, length nshells+1.
pub fn build_domains(
    centers: &Array2<f64>,
    shell_centers: &[[f64; 3]],
    shell_offsets: &[usize],
    cutoff_bohr: f64,
) -> LmpDomains {
    let nocc = centers.nrows();
    let nshells = shell_centers.len();
    let nbas = *shell_offsets.last().unwrap();
    let cutoff_sq = cutoff_bohr * cutoff_bohr;

    let mut orbital_domains: Vec<Vec<usize>> = vec![Vec::new(); nocc];

    for i in 0..nocc {
        let cx = centers[(i, 0)];
        let cy = centers[(i, 1)];
        let cz = centers[(i, 2)];
        for s in 0..nshells {
            let sc = shell_centers[s];
            let dx = sc[0] - cx;
            let dy = sc[1] - cy;
            let dz = sc[2] - cz;
            if dx*dx + dy*dy + dz*dz <= cutoff_sq {
                let start = shell_offsets[s];
                let end = shell_offsets[s + 1];
                for mu in start..end {
                    orbital_domains[i].push(mu);
                }
            }
        }
        orbital_domains[i].sort_unstable();
        orbital_domains[i].dedup();
    }

    // Union mask
    let mut ao_mask = vec![false; nbas];
    for dom in &orbital_domains {
        for &mu in dom {
            ao_mask[mu] = true;
        }
    }
    let active_aos: Vec<usize> = (0..nbas).filter(|&mu| ao_mask[mu]).collect();

    LmpDomains { orbital_domains, ao_mask, active_aos }
}

/// Build a sparse occupied pseudo-density restricted to the LMP2 domains.
///
/// `P(t)_{μν} = Σ_i C_{μi}^can exp(+t ε_i) C_{νi}^can`
///
/// Uses canonical MO coefficients and orbital energies.  The Boys domain
/// structure determines *which* (μ,ν) pairs to compute: element (μ,ν) is
/// included if both μ and ν lie in the domain of at least one orbital i.
///
/// `c_can`: canonical occupied MOs (nbas × nocc).
/// `eps_occ`: canonical occupied orbital energies, length nocc.
/// `domains`: orbital domains from Boys localization (for sparsity only).
pub fn build_pseudo_density_occ_sparse(
    c_can: &Array2<f64>,
    eps_occ: &[f64],
    t: f64,
    domains: &LmpDomains,
) -> Array2<f64> {
    let nbas = c_can.nrows();
    let nocc = c_can.ncols();
    assert_eq!(eps_occ.len(), nocc);
    let mut p = Array2::zeros((nbas, nbas));
    for i in 0..nocc {
        let factor = (t * eps_occ[i]).exp();
        for &mu in &domains.orbital_domains[i] {
            let c_mu_i = c_can[(mu, i)] * factor;
            for &nu in &domains.orbital_domains[i] {
                p[(mu, nu)] += c_mu_i * c_can[(nu, i)];
            }
        }
    }
    p
}

/// Build a sparse virtual pseudo-density restricted to the active AO union.
///
/// `Q(t)_{μν} = Σ_a C_{μa} exp(-t ε_a) C_{νa}` for μ,ν ∈ active_aos only.
/// (Virtuals are not localized — use full virtual space but restrict AO indices.)
pub fn build_pseudo_density_vir_sparse(
    c_vir: &Array2<f64>,
    eps: &[f64],
    t: f64,
    nocc_total: usize,
    domains: &LmpDomains,
) -> Array2<f64> {
    let nbas = c_vir.nrows();
    let nvir = c_vir.ncols();
    let mut q = Array2::zeros((nbas, nbas));
    for a in 0..nvir {
        let factor = (-t * eps[nocc_total + a]).exp();
        for &mu in &domains.active_aos {
            let c_mu_a = c_vir[(mu, a)] * factor;
            for &nu in &domains.active_aos {
                q[(mu, nu)] += c_mu_a * c_vir[(nu, a)];
            }
        }
    }
    q
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
        let c_occ = rhf.mos_r().slice(s![.., ..nocc]).to_owned();

        // Compute dipole integrals at origin (0,0,0)
        let dip = dipole(&prep, [0.0, 0.0, 0.0]).unwrap();

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
