use super::{CcConfig, CcResult};
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::ScfResult;
use ferric_mp2::mo_transform::{transform_3center_oo, transform_3center_ov, transform_3center_vv};
use ferric_mp2::rimp2::cholesky_inverse_sqrt;
use ferric_tensors::{einsum, Axis, Tensor};
use ndarray::{Array4, ArrayD, IxDyn};

/// Compute CCD correlation energy.
pub fn ccd(
    _mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    cfg: &CcConfig,
) -> Result<CcResult, FerricError> {
    let nbas = obs.nbasis();
    let nocc_total = rhf.eps_r().iter().filter(|&&e| e < 0.0).count();
    let nocc = nocc_total - cfg.frozen_core;
    let nvir = nbas - nocc_total;

    let eps = rhf.eps_r();
    let c = rhf.mos_r();
    let c_occ = c.slice(ndarray::s![.., cfg.frozen_core..nocc_total]).to_owned();
    let c_vir = c.slice(ndarray::s![.., nocc_total..]).to_owned();

    // 1. Get RI amplitudes B^P in MO basis blocks
    let v2c = ferric_integrals::threeindex::coulomb_metric_2c(op, dfbs)?;
    let v_inv_sqrt = cholesky_inverse_sqrt(&v2c)?;
    let eri3_ao = ferric_integrals::threeindex::eri3_tensor(op, obs, dfbs)?;
    
    let b_ao = eri3_ao; // (naux, nbas, nbas)
    let b_ov = transform_3center_ov(&b_ao, &c_occ, &c_vir);
    let b_oo = transform_3center_oo(&b_ao, &c_occ);
    let b_vv = transform_3center_vv(&b_ao, &c_vir);

    // Contract with V^-1/2: B^P_{xy} = sum_Q V^{-1/2}_{PQ} (Q|xy).
    // Reshape each (naux, d1, d2) block to (naux, d1*d2), apply the naux×naux
    // matmul via BLAS2 .dot(), reshape back.
    let dress = |b: &ndarray::Array3<f64>| -> ndarray::Array3<f64> {
        let (na, d1, d2) = b.dim();
        let flat = b.view().into_shape_with_order((na, d1 * d2)).unwrap();
        v_inv_sqrt
            .dot(&flat)
            .into_shape_with_order((na, d1, d2))
            .unwrap()
    };
    let b_ia = dress(&b_ov); // (naux, nocc, nvir)
    let b_ij = dress(&b_oo); // (naux, nocc, nocc)
    let b_ab = dress(&b_vv); // (naux, nvir, nvir)

    // Dressed RI blocks as Tensors for einsum!, wrapped ONCE here. These are
    // loop-invariant (geometry/orbital-only); the ladder helpers borrow them
    // each iteration instead of re-cloning the (potentially hundreds-of-MB)
    // B^P_ab / B^P_ij arrays per call.
    let b_ia_t = Tensor::new(b_ia.into_dyn(), [Axis::Aux, Axis::O, Axis::V]);
    let b_ab_t = Tensor::new(b_ab.into_dyn(), [Axis::Aux, Axis::V, Axis::V]);
    let b_ij_t = Tensor::new(b_ij.into_dyn(), [Axis::Aux, Axis::O, Axis::O]);

    // The chemist (ia|jb) integral g[i,a,j,b] = sum_P B^P_ia B^P_jb, built ONCE
    // via einsum! and reused for the MP2 guess, the energy, and the (ab|ij)
    // residual term (previously computed three times with scalar sum_P loops).
    let g_iajb: ArrayD<f64> = einsum!("Pia,Pjb->iajb", &b_ia_t, &b_ia_t);
    // (ab|ij) residual term is exactly g[i,a,j,b]; keep an Ix4 view-free copy
    // once (loop-invariant) instead of cloning g every iteration.
    let g_iajb_4: ndarray::Array4<f64> = g_iajb
        .clone()
        .into_dimensionality::<ndarray::Ix4>()
        .unwrap();

    // 2. Form initial T2 guess (MP2): t2[i,a,j,b] = g[i,a,j,b] / D_ijab.
    let mut t2 = Array4::zeros((nocc, nvir, nocc, nvir));
    for i in 0..nocc {
        for j in 0..nocc {
            for a in 0..nvir {
                for b in 0..nvir {
                    let d_ijab = eps[cfg.frozen_core + i] + eps[cfg.frozen_core + j]
                               - eps[nocc_total + a] - eps[nocc_total + b];
                    t2[(i, a, j, b)] = g_iajb[[i, a, j, b]] / d_ijab;
                }
            }
        }
    }

    // gx[i,a,j,b] = 2 g[i,a,j,b] - g[i,b,j,a]; the exchange reindex (ib|ja) is
    // g permuted on a<->b = axes [0,3,2,1]. Used in the energy contraction.
    let g_ibja = g_iajb
        .clone()
        .permuted_axes(IxDyn(&[0, 3, 2, 1]))
        .as_standard_layout()
        .into_owned();
    let gx = 2.0 * &g_iajb - &g_ibja;
    let gx_t = Tensor::new(gx, [Axis::O, Axis::V, Axis::O, Axis::V]);

    // 3. Iteration loop
    let mut e_old = 0.0;
    for iter in 0..cfg.max_iter {
        // A. Compute correlation energy: e_corr = sum_iajb t2 * (2 g_iajb - g_ibja).
        let t2_t = Tensor::new(t2.clone().into_dyn(), [Axis::O, Axis::V, Axis::O, Axis::V]);
        let e_corr: f64 = einsum!("iajb,iajb->", &t2_t, &gx_t);

        let d_e = (e_corr - e_old).abs();
        if iter > 0 && d_e < 1e-10 {
            println!("CCD converged in {} iterations. E_corr = {:.10}", iter, e_corr);
            return Ok(CcResult { correlation_energy: e_corr, t1: None, t2 });
        }
        e_old = e_corr;

        // B. Compute residuals R_iajb = (ab|ij) + L_iajb + H_iajb.
        // The (ab|ij) term is the loop-invariant g_iajb_4; the ladder helpers
        // borrow the pre-wrapped, loop-invariant B tensors (no per-iter clone).
        let pp_ladder = crate::helpers::contract_pp_ladder(&b_ab_t, &t2_t);
        let hh_ladder = crate::helpers::contract_hh_ladder(&b_ij_t, &t2_t);
        let r2 = &g_iajb_4 + &pp_ladder + &hh_ladder;

        // Update T2: T = T + R / D
        for i in 0..nocc {
            for j in 0..nocc {
                for a in 0..nvir {
                    for b in 0..nvir {
                        let d_ijab = eps[cfg.frozen_core + i] + eps[cfg.frozen_core + j] 
                                   - eps[nocc_total + a] - eps[nocc_total + b];
                        let delta = r2[(i, a, j, b)] / d_ijab;
                        t2[(i, a, j, b)] += delta;
                    }
                }
            }
        }
    }

    Err(FerricError::Convergence("CCD did not converge".into()))
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
    fn test_ccd_h2_sto3g() {
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
        
        let result = ccd(&mol, &obs, &dfbs, op, &rhf, &cc_cfg).unwrap();
        
        println!("CCD correlation energy: {:.10}", result.correlation_energy);
        // H2/STO-3G CCD correlation energy is ~ -0.018 Hartree
        assert!((result.correlation_energy - (-0.018)).abs() < 1e-2);
    }

    #[test]
    fn ccd_h2_sto3g_energy_pinned() {
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("sto-3g").unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        let op = Operator::coulomb();
        let ctx = ParallelContext::default();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
        let cfg = CcConfig { frozen_core: 0, max_iter: 50, energy_conv: 1e-8, ..Default::default() };
        let r = ccd(&mol, &obs, &dfbs, op, &rhf, &cfg).unwrap();
        assert!((r.correlation_energy - (-0.0239287831)).abs() < 1e-9, "got {:.10}", r.correlation_energy);
    }
}
