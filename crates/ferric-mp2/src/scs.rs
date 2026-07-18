//! Spin-component scaled MP2 variants.
//!
//! - SCS-MP2: E = c_OS * E_OS + c_SS * E_SS (Grimme, JCP 2003)
//! - SCS-MP2(2terfc): dual-attenuated SCS (Goldey, Dutoi, Head-Gordon, PCCP 2013)
//!   E = c_OS * E_OS(r0_1) + c_SS * [E_SS(r0_2) - E_SS(r0_1)], using the EXACT
//!   `terfc` operator (2D interpolation tables), not the erfc approximation.

use crate::rimp2::{ri_mp2_spin_components, RiMp2Config};
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::ScfResult;

/// Angstrom to Bohr conversion factor.
const ANGSTROM_TO_BOHR: f64 = 1.8897259886;


/// Standard SCS-MP2 configuration (Grimme, JCP 2003).
#[derive(Debug, Clone)]
pub struct ScsMp2Config {
    /// Opposite-spin scaling coefficient (Grimme default: 6/5).
    pub c_os: f64,
    /// Same-spin scaling coefficient (Grimme default: 1/3).
    pub c_ss: f64,
    /// Frozen core orbitals.
    pub frozen_core: usize,
    /// Optional resident-bytes ceiling for the 3-index MO transform, propagated
    /// into the internal `RiMp2Config`. `None` → resolved via
    /// [`ferric_core::memory::resolve_budget_bytes`].
    pub memory_budget_bytes: Option<usize>,
}

impl Default for ScsMp2Config {
    fn default() -> Self {
        Self { c_os: 6.0 / 5.0, c_ss: 1.0 / 3.0, frozen_core: 0, memory_budget_bytes: None }
    }
}



/// Result from SCS-MP2 or SCS-MP2(2terfc).
#[derive(Debug)]
pub struct ScsMp2Result {
    /// Total energy: E_RHF + scs_corr.
    pub total_energy: f64,
    /// SCS correlation energy.
    pub scs_corr: f64,
    /// Opposite-spin component (possibly scaled/attenuated).
    pub e_os: f64,
    /// Same-spin component (possibly scaled/attenuated).
    pub e_ss: f64,
}

/// Standard SCS-MP2: E = c_OS * E_OS + c_SS * E_SS.
///
/// Uses full Coulomb operator (no attenuation). With c_OS=1, c_SS=1 this
/// recovers standard MP2.
pub fn scs_mp2(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    rhf: &ScfResult,
    config: &ScsMp2Config,
) -> Result<ScsMp2Result, FerricError> {
    let ri_config = RiMp2Config { frozen_core: config.frozen_core, memory_budget_bytes: config.memory_budget_bytes };
    let (sc, _) = ri_mp2_spin_components(mol, obs, dfbs, Operator::coulomb(), rhf, &ri_config)?;
    let scs_corr = config.c_os * sc.e_os + config.c_ss * sc.e_ss;
    Ok(ScsMp2Result {
        total_energy: rhf.energy + scs_corr,
        scs_corr,
        e_os: sc.e_os,
        e_ss: sc.e_ss,
    })
}

/// SCS-MP2(2terfc) configuration (Goldey, Dutoi, Head-Gordon, PCCP 2013; thesis Eq 5.6).
#[derive(Debug, Clone)]
pub struct ScsMp2TerfcConfig {
    /// Bonded attenuation distance r0(1) in Bohr.
    pub r0_bonded: f64,
    /// Non-bonded attenuation distance r0(2) in Bohr (must be > r0(1)).
    pub r0_nonbonded: f64,
    /// Opposite-spin scaling coefficient.
    pub c_os: f64,
    /// Same-spin scaling coefficient.
    pub c_ss: f64,
    /// Frozen core orbitals.
    pub frozen_core: usize,
    /// Optional resident-bytes ceiling for the 3-index MO transform, propagated
    /// into the internal `RiMp2Config`. `None` → resolved via
    /// [`ferric_core::memory::resolve_budget_bytes`].
    pub memory_budget_bytes: Option<usize>,
}

impl Default for ScsMp2TerfcConfig {
    fn default() -> Self {
        // Optimal parameters from thesis: r0(1)=0.75 A, r0(2)=1.05 A, c_OS=1.27, c_SS=4.05.
        Self {
            r0_bonded: 0.75 * ANGSTROM_TO_BOHR,
            r0_nonbonded: 1.05 * ANGSTROM_TO_BOHR,
            c_os: 1.27,
            c_ss: 4.05,
            frozen_core: 0,
            memory_budget_bytes: None,
        }
    }
}

/// SCS-MP2(2terfc): E = c_OS * E_OS(r0_1) + c_SS * [E_SS(r0_2) - E_SS(r0_1)].
///
/// Dual-attenuated SCS-MP2 from Goldey, Dutoi, Head-Gordon (PCCP 2013). Calls
/// `ri_mp2_spin_components` twice with the EXACT `terfc` operator at the bonded
/// (r0_1) and non-bonded (r0_2) attenuation distances. Unlike the deprecated
/// spike, this uses the interpolation-table terfc integrals, not an erfc fit.
pub fn scs_mp2_2terfc(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    rhf: &ScfResult,
    config: &ScsMp2TerfcConfig,
) -> Result<ScsMp2Result, FerricError> {
    assert!(config.r0_nonbonded > config.r0_bonded, "r0(2) must be > r0(1)");
    let ri_config = RiMp2Config { frozen_core: config.frozen_core, memory_budget_bytes: config.memory_budget_bytes };

    // Spin components at r0(1) (bonded, shorter range) via exact terfc.
    let (sc1, _) =
        ri_mp2_spin_components(mol, obs, dfbs, Operator::terfc(config.r0_bonded), rhf, &ri_config)?;
    // Spin components at r0(2) (non-bonded, longer range).
    let (sc2, _) = ri_mp2_spin_components(
        mol, obs, dfbs, Operator::terfc(config.r0_nonbonded), rhf, &ri_config,
    )?;

    // Thesis Eq 5.6: E = c_OS * E_OS(r0_1) + c_SS * [E_SS(r0_2) - E_SS(r0_1)].
    let e_ss = sc2.e_ss - sc1.e_ss;
    let scs_corr = config.c_os * sc1.e_os + config.c_ss * e_ss;
    Ok(ScsMp2Result {
        total_energy: rhf.energy + scs_corr,
        scs_corr,
        e_os: sc1.e_os,
        e_ss,
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

    fn setup_h2() -> (Molecule, PreparedBasis, PreparedBasis, ScfResult) {
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let obs = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(
            &ferric_core::parallel::ParallelContext::default(),
            &mol, &obs, op, &bounds,
            &RhfConfig { energy_conv: 1e-10, ..Default::default() },
        ).unwrap();
        let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
        (mol, obs, dfbs, rhf)
    }

    #[test]
    fn test_scs_mp2_unit_scaling_equals_standard() {
        let (mol, obs, dfbs, rhf) = setup_h2();
        let full = crate::rimp2::ri_mp2(
            &mol, &obs, &dfbs, Operator::coulomb(), &rhf,
            &RiMp2Config::default(),
        ).unwrap();
        let scs = scs_mp2(
            &mol, &obs, &dfbs, &rhf,
            &ScsMp2Config { c_os: 1.0, c_ss: 1.0, frozen_core: 0, memory_budget_bytes: None },
        ).unwrap();
        eprintln!(
            "SCS (c_OS=1, c_SS=1) corr: {:.10}, standard RI-MP2 corr: {:.10}",
            scs.scs_corr, full.mp2_corr
        );
        assert!(
            (scs.scs_corr - full.mp2_corr).abs() < 1e-10,
            "SCS with c_OS=c_SS=1 ({}) should equal standard ({})",
            scs.scs_corr, full.mp2_corr
        );
    }

}
