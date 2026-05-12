use crate::diis::Diis;
use crate::guess::hcore_guess;
use crate::screening::SchwarzBounds;
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::oneelectron;
use ferric_integrals::operator::Operator;
use ndarray::Array2;
use ndarray_linalg::Eigh;

#[derive(Debug, Clone)]
pub struct RhfConfig {
    pub max_iter: usize,
    pub energy_conv: f64,
    pub density_conv: f64,
    pub diis_size: usize,
    pub integral_thresh: f64,
}

impl Default for RhfConfig {
    fn default() -> Self {
        Self {
            max_iter: 100,
            energy_conv: 1e-8,
            density_conv: 1e-7,
            diis_size: 8,
            integral_thresh: 1e-12,
        }
    }
}

#[derive(Debug)]
pub struct RhfResult {
    pub energy: f64,
    pub density: Array2<f64>,
    pub mos: Array2<f64>,
    pub orbital_energies: Vec<f64>,
    pub fock: Array2<f64>,
    pub converged: bool,
    pub iterations: usize,
}

pub fn solve_rhf(
    mol: &Molecule,
    prep: &PreparedBasis,
    _op: Operator,
    bounds: &SchwarzBounds,
    config: &RhfConfig,
) -> Result<RhfResult, FerricError> {
    let s = oneelectron::overlap(prep);
    let h = oneelectron::hcore(prep);
    let n = prep.nbasis();
    let nelec = mol.nelec();
    if nelec % 2 != 0 {
        return Err(FerricError::ScfConvergence {
            iterations: 0,
            last_energy: 0.0,
        });
    }
    let nocc = (nelec / 2) as usize;
    let vnn = mol.nuclear_repulsion();

    let mut d = hcore_guess(&s, &h, nocc)?;
    let mut f = Array2::zeros((n, n));
    let mut diis = Diis::new(config.diis_size);
    let mut prev_e = 0.0;

    // Precompute S^{-1/2} for diagonalization
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

    for iter in 1..=config.max_iter {
        // Build J and K
        let mut j = Array2::zeros((n, n));
        let mut k = Array2::zeros((n, n));
        build_jk(prep, bounds, config.integral_thresh, &d, &mut j, &mut k)?;

        // F = H + J - 0.5*K
        f.assign(&(&h + &j - &(0.5 * &k)));

        // Electronic energy: E_elec = 0.5 * tr(D * (H + F))
        let e_elec: f64 = (0..n)
            .flat_map(|i| (0..n).map(move |j| (i, j)))
            .map(|(i, j)| 0.5 * d[(i, j)] * (h[(i, j)] + f[(i, j)]))
            .sum();
        let energy = e_elec + vnn;

        // DIIS error: e = FDS - SDF
        let fds = f.dot(&d).dot(&s);
        let sdf = s.dot(&d).dot(&f);
        let err = &fds - &sdf;

        let de = (energy - prev_e).abs();
        let err_max = err.iter().map(|v| v.abs()).fold(0.0f64, f64::max);

        if iter > 1 && de < config.energy_conv && err_max < config.density_conv {
            let (orb_e, c) = diagonalize(&f, &s_inv_sqrt)?;
            return Ok(RhfResult {
                energy,
                density: d,
                mos: c,
                orbital_energies: orb_e,
                fock: f,
                converged: true,
                iterations: iter,
            });
        }
        prev_e = energy;

        let f_new = diis.step(&f, &err);
        let (_, c) = diagonalize(&f_new, &s_inv_sqrt)?;

        // Rebuild density
        d.fill(0.0);
        for mu in 0..n {
            for nu in 0..n {
                let mut sum = 0.0;
                for i in 0..nocc {
                    sum += c[(mu, i)] * c[(nu, i)];
                }
                d[(mu, nu)] = 2.0 * sum;
            }
        }
    }
    Err(FerricError::ScfConvergence {
        iterations: config.max_iter,
        last_energy: prev_e,
    })
}

fn build_jk(
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    thresh: f64,
    d: &Array2<f64>,
    j: &mut Array2<f64>,
    k: &mut Array2<f64>,
) -> Result<(), FerricError> {
    let nsh = prep.nshells();
    let dims = prep.shell_dims();
    let offs = prep.shell_offsets();
    let max_d = d.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    let mut engine = Engine::new_2e(bounds.op, prep, 1e-14)?;

    // Loop over canonical shell quartets: s1>=s2, s3>=s4, (s1,s2)>=(s3,s4)
    // For each integral (mu nu | la sg), we have up to 8 equivalent integrals.
    // Degeneracy factors based on which shell symmetries are broken:
    //   f12 = if s1==s2 { 1 } else { 2 }  — bra permutation symmetry
    //   f34 = if s3==s4 { 1 } else { 2 }  — ket permutation symmetry
    //   f1234 = if (s1,s2)==(s3,s4) { 1 } else { 2 }  — bra-ket symmetry
    //
    // We accumulate into J and K by enumerating all equivalent permutations explicitly.

    for s1 in 0..nsh {
        for s2 in 0..=s1 {
            let b12 = bounds.q[(s1, s2)];
            for s3 in 0..=s1 {
                let s4max = if s3 == s1 { s2 } else { s3 };
                for s4 in 0..=s4max {
                    let b34 = bounds.q[(s3, s4)];
                    if b12 * b34 * max_d < thresh {
                        continue;
                    }
                    let quartet = engine.compute_quartet(prep, s1, s2, s3, s4);
                    if let Some(q) = quartet {
                        let (n1, n2, n3, n4) = (dims[s1], dims[s2], dims[s3], dims[s4]);
                        let (o1, o2, o3, o4) = (offs[s1], offs[s2], offs[s3], offs[s4]);
                        let sym12 = s1 != s2;
                        let sym34 = s3 != s4;
                        let sym1234 = (s1, s2) != (s3, s4);
                        for a in 0..n1 {
                            for b in 0..n2 {
                                for c in 0..n3 {
                                    for dd in 0..n4 {
                                        let v = q[((a * n2 + b) * n3 + c) * n4 + dd];
                                        let mu = o1 + a;
                                        let nu = o2 + b;
                                        let la = o3 + c;
                                        let sg = o4 + dd;

                                        // (mu nu | la sg) — always present
                                        // J: J[mu,nu] += D[la,sg] * v
                                        // K: K[mu,la] += D[nu,sg] * v
                                        j[(mu, nu)] += d[(la, sg)] * v;
                                        k[(mu, la)] += d[(nu, sg)] * v;

                                        if sym12 {
                                            // (nu mu | la sg)
                                            // J: J[nu,mu] += D[la,sg] * v
                                            // K: K[nu,la] += D[mu,sg] * v
                                            j[(nu, mu)] += d[(la, sg)] * v;
                                            k[(nu, la)] += d[(mu, sg)] * v;
                                        }

                                        if sym34 {
                                            // (mu nu | sg la)
                                            // J: J[mu,nu] += D[sg,la] * v  (same as D[la,sg] for symmetric D)
                                            // K: K[mu,sg] += D[nu,la] * v
                                            j[(mu, nu)] += d[(sg, la)] * v;
                                            k[(mu, sg)] += d[(nu, la)] * v;
                                        }

                                        if sym12 && sym34 {
                                            // (nu mu | sg la)
                                            // J: J[nu,mu] += D[sg,la] * v
                                            // K: K[nu,sg] += D[mu,la] * v
                                            j[(nu, mu)] += d[(sg, la)] * v;
                                            k[(nu, sg)] += d[(mu, la)] * v;
                                        }

                                        if sym1234 {
                                            // (la sg | mu nu)
                                            // J: J[la,sg] += D[mu,nu] * v
                                            // K: K[la,mu] += D[sg,nu] * v
                                            j[(la, sg)] += d[(mu, nu)] * v;
                                            k[(la, mu)] += d[(sg, nu)] * v;

                                            if sym34 {
                                                // (sg la | mu nu)
                                                // J: J[sg,la] += D[mu,nu] * v
                                                // K: K[sg,mu] += D[la,nu] * v
                                                j[(sg, la)] += d[(mu, nu)] * v;
                                                k[(sg, mu)] += d[(la, nu)] * v;
                                            }

                                            if sym12 {
                                                // (la sg | nu mu)
                                                // J: J[la,sg] += D[nu,mu] * v
                                                // K: K[la,nu] += D[sg,mu] * v
                                                j[(la, sg)] += d[(nu, mu)] * v;
                                                k[(la, nu)] += d[(sg, mu)] * v;
                                            }

                                            if sym12 && sym34 {
                                                // (sg la | nu mu)
                                                // J: J[sg,la] += D[nu,mu] * v
                                                // K: K[sg,nu] += D[la,mu] * v
                                                j[(sg, la)] += d[(nu, mu)] * v;
                                                k[(sg, nu)] += d[(la, mu)] * v;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

fn diagonalize(
    f: &Array2<f64>,
    s_inv_sqrt: &Array2<f64>,
) -> Result<(Vec<f64>, Array2<f64>), FerricError> {
    let f_prime = s_inv_sqrt.dot(f).dot(s_inv_sqrt);
    let (evals, evecs) = f_prime
        .eigh(ndarray_linalg::UPLO::Upper)
        .map_err(|e| FerricError::Lapack(format!("F diag: {e}")))?;
    let c = s_inv_sqrt.dot(&evecs);
    Ok((evals.to_vec(), c))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::screening::SchwarzBounds;
    use ferric_core::basis;
    use ferric_core::mol::Molecule;
    use ferric_integrals::basis_bridge::PreparedBasis;

    fn run_rhf_test(xyz: &str, basis_name: &str, ref_slug: &str, tol: f64) {
        let mol = Molecule::parse_xyz(xyz).unwrap();
        let bs = basis::bundled(basis_name).unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let config = RhfConfig {
            energy_conv: 1e-12,
            density_conv: 1e-10,
            integral_thresh: 1e-14,
            ..Default::default()
        };
        let result = solve_rhf(&mol, &prep, op, &bounds, &config).unwrap();
        assert!(result.converged, "RHF did not converge");
        eprintln!(
            "{ref_slug}: energy={:.12}, iters={}, vnn={:.12}",
            result.energy,
            result.iterations,
            mol.nuclear_repulsion()
        );
        let ref_path = format!("../../testdata/reference/{ref_slug}");
        if let Ok(text) = std::fs::read_to_string(&ref_path) {
            let ref_data: serde_json::Value = serde_json::from_str(&text).unwrap();
            let ref_energy = ref_data["energy"].as_f64().unwrap();
            assert!(
                (result.energy - ref_energy).abs() < tol,
                "{ref_slug}: got {:.10}, ref {:.10}",
                result.energy,
                ref_energy
            );
        }
    }

    #[test]
    fn test_rhf_h2_sto3g() {
        run_rhf_test(
            "2\nH2\nH 0 0 0\nH 0 0 0.74\n",
            "sto-3g",
            "h2_sto-3g_rhf.json",
            1e-8,
        );
    }

    #[test]
    fn test_rhf_h2o_sto3g() {
        // Tolerance 5e-8 due to libint2 vs libcint integral differences
        run_rhf_test(
            "3\nwater\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n",
            "sto-3g",
            "h2o_sto-3g_rhf.json",
            5e-8,
        );
    }

    #[test]
    fn test_rhf_h2o_631g() {
        run_rhf_test(
            "3\nwater\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n",
            "6-31g",
            "h2o_6-31g_rhf.json",
            1e-8,
        );
    }
}
