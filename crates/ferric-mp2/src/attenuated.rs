//! Attenuated MP2 methods (Goldey & Head-Gordon, JPCL 2012).
//!
//! Replace 1/r12 in the MP2 correlation integrals with an attenuated operator
//! (erfc or terfc), keeping only short-range correlation. The erfc operator is
//! supported natively by libint2; the terfc/erfc equivalence holds when
//! omega = 1/(r0 * sqrt(2)).

use crate::rimp2::{ri_mp2_spin_components, RiMp2Config, SpinComponents};
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::{Operator, OperatorKind};
use ferric_scf::rhf::RhfResult;

/// Configuration for attenuated MP2.
#[derive(Debug, Clone)]
pub struct AttenuatedMp2Config {
    /// Attenuation distance r0 in Bohr. Typical: 1.984 Bohr (1.05 Angstrom) for aDZ.
    pub r0: f64,
    /// Optional scaling factor for the correlation energy.
    pub scaling: f64,
    /// Frozen core orbitals.
    pub frozen_core: usize,
}

impl Default for AttenuatedMp2Config {
    fn default() -> Self {
        Self {
            r0: 1.984,    // 1.05 Angstrom in Bohr (aDZ optimal)
            scaling: 1.0,
            frozen_core: 0,
        }
    }
}

/// Result from attenuated MP2.
#[derive(Debug)]
pub struct AttenuatedMp2Result {
    pub mp2_corr: f64,
    pub total_energy: f64,
    pub spin_components: SpinComponents,
}

/// Compute attenuated RI-MP2 using erfc(omega*r)/r operator.
///
/// The attenuation parameter omega is derived from r0 via the curvature
/// constraint: omega = 1/(r0 * sqrt(2)). For large r0, erfc approaches
/// the unit step function and the result converges to standard MP2.
pub fn attenuated_ri_mp2(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    rhf: &RhfResult,
    config: &AttenuatedMp2Config,
) -> Result<AttenuatedMp2Result, FerricError> {
    let omega = 1.0 / (config.r0 * std::f64::consts::SQRT_2);
    let op = Operator {
        kind: OperatorKind::ErfcCoulomb,
        omega,
        distance: config.r0,
    };
    let ri_config = RiMp2Config { frozen_core: config.frozen_core };
    let (sc, _) = ri_mp2_spin_components(mol, obs, dfbs, op, rhf, &ri_config)?;
    let scaled_corr = config.scaling * sc.e_total;
    Ok(AttenuatedMp2Result {
        mp2_corr: scaled_corr,
        total_energy: rhf.energy + scaled_corr,
        spin_components: sc,
    })
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

    fn setup_h2() -> (Molecule, PreparedBasis, PreparedBasis, RhfResult) {
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n").unwrap();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let obs = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(
            &mol, &obs, op, &bounds,
            &RhfConfig { energy_conv: 1e-10, ..Default::default() },
        ).unwrap();
        let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
        (mol, obs, dfbs, rhf)
    }

    #[test]
    fn test_attenuated_mp2_smaller_than_full() {
        let (mol, obs, dfbs, rhf) = setup_h2();
        let full = crate::rimp2::ri_mp2(
            &mol, &obs, &dfbs, Operator::coulomb(), &rhf,
            &crate::rimp2::RiMp2Config::default(),
        ).unwrap();
        let att = attenuated_ri_mp2(&mol, &obs, &dfbs, &rhf, &AttenuatedMp2Config::default()).unwrap();
        eprintln!("Full RI-MP2 corr: {:.10}, Attenuated corr: {:.10}", full.mp2_corr, att.mp2_corr);
        assert!(
            att.mp2_corr.abs() < full.mp2_corr.abs(),
            "attenuated |{}| should be < full |{}|", att.mp2_corr, full.mp2_corr
        );
    }

    #[test]
    fn test_attenuated_mp2_large_r0_approaches_full() {
        let (mol, obs, dfbs, rhf) = setup_h2();
        let full = crate::rimp2::ri_mp2(
            &mol, &obs, &dfbs, Operator::coulomb(), &rhf,
            &crate::rimp2::RiMp2Config::default(),
        ).unwrap();
        let config = AttenuatedMp2Config { r0: 50.0, ..Default::default() };
        let att = attenuated_ri_mp2(&mol, &obs, &dfbs, &rhf, &config).unwrap();
        eprintln!(
            "Full RI-MP2 corr: {:.10}, Large-r0 attenuated corr: {:.10}, diff: {:.2e}",
            full.mp2_corr, att.mp2_corr, (att.mp2_corr - full.mp2_corr).abs()
        );
        assert!(
            (att.mp2_corr - full.mp2_corr).abs() < 1e-4,
            "large r0 attenuated ({}) should approach full ({})", att.mp2_corr, full.mp2_corr
        );
    }
}
