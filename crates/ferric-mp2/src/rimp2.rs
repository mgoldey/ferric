use crate::mo_transform::transform_3center_ov;
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_integrals::threeindex;
use ferric_scf::rhf::RhfResult;
use ndarray::Array2;
use ndarray_linalg::{Cholesky, UPLO};

#[derive(Debug, Clone)]
pub struct RiMp2Config {
    pub frozen_core: usize,
}

impl Default for RiMp2Config {
    fn default() -> Self {
        Self { frozen_core: 0 }
    }
}

#[derive(Debug)]
pub struct RiMp2Result {
    pub mp2_corr: f64,
    pub total_energy: f64,
}

pub fn ri_mp2(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &RhfResult,
    config: &RiMp2Config,
) -> Result<RiMp2Result, FerricError> {
    let nbas = obs.nbasis();
    let nelec = mol.nelec() as usize;
    let nocc_total = nelec / 2;
    let nocc = nocc_total - config.frozen_core;
    let first_occ = config.frozen_core;
    let nvir = nbas - nocc_total;
    let naux = dfbs.nbasis();
    let eps = &rhf.orbital_energies;
    let c = &rhf.mos;

    let c_occ = c.slice(ndarray::s![.., first_occ..first_occ + nocc]).to_owned();
    let c_vir = c.slice(ndarray::s![.., nocc_total..]).to_owned();

    // (P|Q) metric and V^{-1/2}
    let v2c = threeindex::coulomb_metric_2c(op, dfbs)?;
    let v2c_inv_sqrt = cholesky_inverse_sqrt(&v2c)?;

    // (P|mu nu) 3-center integrals
    let eri3_ao = threeindex::eri3_tensor(op, obs, dfbs)?;

    // MO transform -> (P|ia)
    let eri3_mo = transform_3center_ov(&eri3_ao, &c_occ, &c_vir);

    // B_ia^P = sum_Q (P|Q)^{-1/2} (Q|ia)
    let eri3_flat = eri3_mo
        .into_shape_with_order((naux, nocc * nvir))
        .unwrap();
    let b_flat = v2c_inv_sqrt.dot(&eri3_flat); // (naux, nocc*nvir)

    // MP2 energy
    let mut e_mp2 = 0.0;
    for i in 0..nocc {
        for j in 0..nocc {
            for a in 0..nvir {
                for b in 0..nvir {
                    let ia = i * nvir + a;
                    let jb = j * nvir + b;
                    let ib = i * nvir + b;
                    let ja = j * nvir + a;
                    let eri_iajb: f64 =
                        (0..naux).map(|p| b_flat[(p, ia)] * b_flat[(p, jb)]).sum();
                    let eri_ibja: f64 =
                        (0..naux).map(|p| b_flat[(p, ib)] * b_flat[(p, ja)]).sum();
                    let denom = eps[first_occ + i] + eps[first_occ + j]
                        - eps[nocc_total + a]
                        - eps[nocc_total + b];
                    e_mp2 += eri_iajb * (2.0 * eri_iajb - eri_ibja) / denom;
                }
            }
        }
    }

    Ok(RiMp2Result {
        mp2_corr: e_mp2,
        total_energy: rhf.energy + e_mp2,
    })
}

/// V^{-1/2} via Cholesky: V = L L^T, then V^{-1/2} = L^{-1}
pub fn cholesky_inverse_sqrt(v: &Array2<f64>) -> Result<Array2<f64>, FerricError> {
    let l = v
        .cholesky(UPLO::Lower)
        .map_err(|e| FerricError::Lapack(format!("Cholesky on (P|Q): {e}")))?;
    let n = l.nrows();
    // Forward-substitution to invert lower-triangular L
    let mut l_inv = Array2::zeros((n, n));
    for i in 0..n {
        l_inv[(i, i)] = 1.0 / l[(i, i)];
        for j in (0..i).rev() {
            let mut sum = 0.0;
            for k in j..i {
                sum += l[(i, k)] * l_inv[(k, j)];
            }
            l_inv[(i, j)] = -sum / l[(i, i)];
        }
    }
    // V^{-1/2} = L^{-1} (so that B = L^{-1} (Q|ia) and B^T B = (ia|P) V^{-1} (Q|jb))
    Ok(l_inv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::basis;
    use ferric_core::mol::Molecule;
    use ferric_integrals::basis_bridge::PreparedBasis;
    use ferric_scf::rhf::{solve_rhf, RhfConfig};
    use ferric_scf::screening::SchwarzBounds;

    fn run_ri_mp2(xyz: &str, basis_name: &str, aux_name: &str) -> (RhfResult, RiMp2Result) {
        let mol = Molecule::parse_xyz(xyz).unwrap();
        let bs = basis::bundled(basis_name).unwrap();
        let obs = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(
            &mol,
            &obs,
            op,
            &bounds,
            &RhfConfig {
                energy_conv: 1e-10,
                ..Default::default()
            },
        )
        .unwrap();
        let aux_bs = basis::bundled(aux_name).unwrap();
        let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
        let mp2 = ri_mp2(&mol, &obs, &dfbs, op, &rhf, &RiMp2Config::default()).unwrap();
        (rhf, mp2)
    }

    #[test]
    fn test_rimp2_h2o_ccpvdz() {
        let xyz = "3\nwater\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n";
        let (rhf, mp2) = run_ri_mp2(xyz, "cc-pvdz", "cc-pvdz-ri");
        eprintln!(
            "H2O/cc-pVDZ RHF energy: {:.10}, RI-MP2 corr: {:.10}, total: {:.10}",
            rhf.energy, mp2.mp2_corr, mp2.total_energy
        );
        // PySCF RI-MP2 (cc-pvdz-ri): corr = -0.2040334729
        assert!(
            (mp2.mp2_corr - (-0.2040334729)).abs() < 1e-6,
            "RI-MP2 corr: got {:.10}, ref -0.2040334729",
            mp2.mp2_corr
        );
    }

    #[test]
    fn test_rimp2_h2_ccpvdz() {
        let xyz = "2\nH2\nH 0 0 0\nH 0 0 0.74\n";
        let (rhf, mp2) = run_ri_mp2(xyz, "cc-pvdz", "cc-pvdz-ri");
        eprintln!(
            "H2/cc-pVDZ RHF energy: {:.10}, RI-MP2 corr: {:.10}, total: {:.10}",
            rhf.energy, mp2.mp2_corr, mp2.total_energy
        );
        // PySCF canonical MP2: -0.0263715576 (RI should be close)
        assert!(
            (mp2.mp2_corr - (-0.0263715576)).abs() < 1e-4,
            "H2 RI-MP2 corr: {:.10}",
            mp2.mp2_corr
        );
    }
}
