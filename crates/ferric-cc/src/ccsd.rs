use super::{CcConfig, CcResult};
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::RhfResult;
use ferric_mp2::mo_transform::{transform_3center_oo, transform_3center_ov, transform_3center_vv};
use ferric_mp2::rimp2::cholesky_inverse_sqrt;
use ndarray::{Array2, Array4};

pub fn ccsd(
    _mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &RhfResult,
    cfg: &CcConfig,
) -> Result<CcResult, FerricError> {
    let nbas = obs.nbasis();
    let nocc_total = rhf.eps_r().iter().filter(|&&e| e < 0.0).count();
    let nocc = nocc_total - cfg.frozen_core;
    let nvir = nbas - nocc_total;
    let naux = dfbs.nbasis();

    let eps = rhf.eps_r();
    let c = rhf.mos_r();
    let c_occ = c.slice(ndarray::s![.., cfg.frozen_core..nocc_total]).to_owned();
    let c_vir = c.slice(ndarray::s![.., nocc_total..]).to_owned();

    // 1. Get RI amplitudes B^P in MO basis blocks
    let v2c = ferric_integrals::threeindex::coulomb_metric_2c(op, dfbs)?;
    let v_inv_sqrt = cholesky_inverse_sqrt(&v2c)?;
    let eri3_ao = ferric_integrals::threeindex::eri3_tensor(op, obs, dfbs)?;
    
    let b_ao = eri3_ao;
    let b_ov = transform_3center_ov(&b_ao, &c_occ, &c_vir);
    let b_oo = transform_3center_oo(&b_ao, &c_occ);
    let b_vv = transform_3center_vv(&b_ao, &c_vir);

    // Contract with V^-1/2
    let mut b_ia = ndarray::Array3::zeros((naux, nocc, nvir));
    let mut b_ij = ndarray::Array3::zeros((naux, nocc, nocc));
    let mut b_ab = ndarray::Array3::zeros((naux, nvir, nvir));

    for p in 0..naux {
        for q in 0..naux {
            let f = v_inv_sqrt[(p, q)];
            b_ia.slice_mut(ndarray::s![p, .., ..]).scaled_add(f, &b_ov.slice(ndarray::s![q, .., ..]));
            b_ij.slice_mut(ndarray::s![p, .., ..]).scaled_add(f, &b_oo.slice(ndarray::s![q, .., ..]));
            b_ab.slice_mut(ndarray::s![p, .., ..]).scaled_add(f, &b_vv.slice(ndarray::s![q, .., ..]));
        }
    }

    // 2. Initialize T1, T2 (Guess: T1=0, T2=MP2)
    let mut t1 = Array2::zeros((nocc, nvir));
    let mut t2 = Array4::zeros((nocc, nvir, nocc, nvir));
    // MP2 guess for T2...
    for i in 0..nocc {
        for j in 0..nocc {
            for a in 0..nvir {
                for b in 0..nvir {
                    let mut eri_iajb = 0.0;
                    for p in 0..naux {
                        eri_iajb += b_ia[(p, i, a)] * b_ia[(p, j, b)];
                    }
                    let d_ijab = eps[cfg.frozen_core + i] + eps[cfg.frozen_core + j] 
                               - eps[nocc_total + a] - eps[nocc_total + b];
                    t2[(i, a, j, b)] = eri_iajb / d_ijab;
                }
            }
        }
    }

    // 3. CCSD Iteration
    let mut e_old: f64 = 0.0;
    for iter in 0..cfg.max_iter {
        // A. Energy
        let mut e_corr: f64 = 0.0;
        for i in 0..nocc {
            for j in 0..nocc {
                for a in 0..nvir {
                    for b in 0..nvir {
                        let mut eri_iajb = 0.0;
                        for p in 0..naux {
                            eri_iajb += b_ia[(p, i, a)] * b_ia[(p, j, b)];
                        }
                        
                        // Term: (2 * T2_ijab + 2 * T1_ia * T1_jb - T2_jiab - T1_ja * T1_ib) * (ia|jb)
                        let t_comp = 2.0 * (t2[(i, a, j, b)] + t1[(i, a)] * t1[(j, b)])
                                   - (t2[(j, a, i, b)] + t1[(j, a)] * t1[(i, b)]);
                        e_corr += t_comp * eri_iajb;
                    }
                }
            }
        }

        let d_e: f64 = (e_corr - e_old).abs();
        if iter > 0 && d_e < 1e-10 {
            println!("CCSD converged in {} iterations. E_corr = {:.10}", iter, e_corr);
            return Ok(CcResult { correlation_energy: e_corr, t1: Some(t1), t2 });
        }
        e_old = e_corr;

        // B. Residuals R1, R2
        let r1: Array2<f64> = Array2::zeros((nocc, nvir));
        let mut r2: Array4<f64> = Array4::zeros((nocc, nvir, nocc, nvir));

        for i in 0..nocc {
            for j in 0..nocc {
                for a in 0..nvir {
                    for b in 0..nvir {
                        let mut eri_iajb = 0.0;
                        for p in 0..naux {
                            eri_iajb += b_ia[(p, i, a)] * b_ia[(p, j, b)];
                        }
                        r2[(i, a, j, b)] = eri_iajb;
                    }
                }
            }
        }

        // Add non-linear ladder terms (DRY helpers)
        let pp_ladder = crate::helpers::contract_pp_ladder(&b_ab, &t2);
        let hh_ladder = crate::helpers::contract_hh_ladder(&b_ij, &t2);
        r2 = r2 + pp_ladder + hh_ladder;

        let mut max_d_t: f64 = 0.0;
        for i in 0..nocc {
            for a in 0..nvir {
                let d_ia = eps[cfg.frozen_core + i] - eps[nocc_total + a];
                let delta: f64 = r1[(i, a)] / d_ia;
                t1[(i, a)] = t1[(i, a)] + delta;
                max_d_t = max_d_t.max(delta.abs());
            }
        }
        for i in 0..nocc {
            for j in 0..nocc {
                for a in 0..nvir {
                    for b in 0..nvir {
                        let d_ijab = eps[cfg.frozen_core + i] + eps[cfg.frozen_core + j] 
                                   - eps[nocc_total + a] - eps[nocc_total + b];
                        let delta = r2[(i, a, j, b)] / d_ijab;
                        t2[(i, a, j, b)] = t2[(i, a, j, b)] + delta;
                        max_d_t = max_d_t.max(delta.abs());
                    }
                }
            }
        }
    }

    Err(FerricError::Convergence("CCSD did not converge".into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::basis;
    use ferric_core::mol::Molecule;
    use ferric_integrals::basis_bridge::PreparedBasis;
    use ferric_integrals::operator::Operator;
    use ferric_scf::rhf::{solve_rhf, RhfConfig};
    use ferric_scf::screening::SchwarzBounds;
    use ferric_core::parallel::ParallelContext;

    #[test]
    fn test_ccsd_h2_sto3g() {
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
        let obs_set = basis::bundled("sto-3g").unwrap();
        let dfbs_set = basis::bundled("cc-pvdz-ri").unwrap();
        let obs = PreparedBasis::new(&mol, &obs_set).unwrap();
        let dfbs = PreparedBasis::new(&mol, &dfbs_set).unwrap();
        let op = Operator::coulomb();
        
        let ctx = ParallelContext::default();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf_config = RhfConfig::default();
        let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &rhf_config).unwrap();
        
        let cc_cfg = CcConfig {
            frozen_core: 0,
            max_iter: 50,
            energy_conv: 1e-8,
            ..Default::default()
        };
        
        let result = ccsd(&mol, &obs, &dfbs, op, &rhf, &cc_cfg).unwrap();
        
        println!("CCSD correlation energy: {:.10}", result.correlation_energy);
        // H2/STO-3G CCSD correlation energy is ~ -0.018 Hartree (similar to CCD for H2)
        assert!((result.correlation_energy - (-0.018)).abs() < 1e-2);
    }
}
