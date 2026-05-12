use crate::basis_bridge::PreparedBasis;
use crate::engine::Engine;
use crate::operator::Operator;
use ferric_core::FerricError;
use ndarray::{Array2, Array3};

/// Build the 2-center Coulomb metric (P|Q), shape (naux, naux).
pub fn coulomb_metric_2c(op: Operator, dfbs: &PreparedBasis) -> Result<Array2<f64>, FerricError> {
    let naux = dfbs.nbasis();
    let nsh = dfbs.nshells();
    let dims = dfbs.shell_dims();
    let offs = dfbs.shell_offsets();
    let mut eng = Engine::new_2center(op, dfbs, 1e-14)?;
    let mut v = Array2::zeros((naux, naux));
    for sp in 0..nsh {
        for sq in 0..=sp {
            let block = eng.compute_eri2(dfbs, sp, sq);
            let np = dims[sp];
            let nq = dims[sq];
            for p in 0..np {
                for q in 0..nq {
                    let val = block[p * nq + q];
                    v[(offs[sp] + p, offs[sq] + q)] = val;
                    v[(offs[sq] + q, offs[sp] + p)] = val;
                }
            }
        }
    }
    Ok(v)
}

/// Build 3-center integrals (P|mn), shape (naux, nbasis, nbasis).
pub fn eri3_tensor(op: Operator, obs: &PreparedBasis, dfbs: &PreparedBasis) -> Result<Array3<f64>, FerricError> {
    let naux = dfbs.nbasis();
    let nbas = obs.nbasis();
    let nsh_obs = obs.nshells();
    let nsh_df = dfbs.nshells();
    let dims_obs = obs.shell_dims();
    let offs_obs = obs.shell_offsets();
    let dims_df = dfbs.shell_dims();
    let offs_df = dfbs.shell_offsets();
    let mut eng = Engine::new_3center(op, obs, dfbs, 1e-14)?;
    let mut eri = Array3::zeros((naux, nbas, nbas));
    for sp in 0..nsh_df {
        for s1 in 0..nsh_obs {
            for s2 in 0..=s1 {
                if let Some(block) = eng.compute_eri3(obs, dfbs, sp, s1, s2) {
                    let np = dims_df[sp];
                    let n1 = dims_obs[s1];
                    let n2 = dims_obs[s2];
                    for p in 0..np {
                        for i in 0..n1 {
                            for j in 0..n2 {
                                let val = block[(p * n1 + i) * n2 + j];
                                eri[(offs_df[sp] + p, offs_obs[s1] + i, offs_obs[s2] + j)] = val;
                                eri[(offs_df[sp] + p, offs_obs[s2] + j, offs_obs[s1] + i)] = val;
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(eri)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::basis;
    use ferric_core::mol::Molecule;

    #[test]
    fn test_coulomb_metric_2c_symmetric() {
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n").unwrap();
        let dfbs_set = basis::bundled("cc-pvdz-ri").unwrap();
        let dfbs = PreparedBasis::new(&mol, &dfbs_set).unwrap();
        let v = coulomb_metric_2c(Operator::coulomb(), &dfbs).unwrap();
        let n = dfbs.nbasis();
        for i in 0..n {
            for j in 0..n {
                assert!((v[(i, j)] - v[(j, i)]).abs() < 1e-12,
                    "(P|Q) not symmetric at ({i},{j})");
            }
        }
        // Diagonal should be positive
        for i in 0..n { assert!(v[(i, i)] > 0.0, "(P|P) should be positive"); }
    }

    #[test]
    fn test_eri3_symmetric_in_mu_nu() {
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n").unwrap();
        let obs_set = basis::bundled("sto-3g").unwrap();
        let dfbs_set = basis::bundled("cc-pvdz-ri").unwrap();
        let obs = PreparedBasis::new(&mol, &obs_set).unwrap();
        let dfbs = PreparedBasis::new(&mol, &dfbs_set).unwrap();
        let eri = eri3_tensor(Operator::coulomb(), &obs, &dfbs).unwrap();
        let naux = dfbs.nbasis();
        let nbas = obs.nbasis();
        for p in 0..naux {
            for i in 0..nbas {
                for j in 0..nbas {
                    assert!((eri[(p, i, j)] - eri[(p, j, i)]).abs() < 1e-12,
                        "ERI3 not symmetric at P={p},i={i},j={j}");
                }
            }
        }
    }
}
