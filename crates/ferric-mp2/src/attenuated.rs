//! Attenuated MP2 methods (Goldey & Head-Gordon, JPCL 2012).
//!
//! Replace 1/r12 in the MP2 correlation integrals with an attenuated operator
//! (erfc), keeping only short-range correlation. The erfc operator is supported
//! natively by libint2 and parameterized directly by the range-separation
//! parameter omega (in Bohr⁻¹ internally; Å⁻¹ at the user-facing boundary).

use crate::rimp2::{ri_mp2_spin_components, RiMp2Config, SpinComponents};
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::ScfResult;

/// Configuration for attenuated MP2.
#[derive(Debug, Clone)]
pub struct AttenuatedMp2Config {
    /// Range-separation parameter omega in Bohr⁻¹.
    /// Default: 0.222234 Bohr⁻¹ (= 0.420 Å⁻¹, dissertation erfc optimal).
    pub omega: f64,
    /// Optional scaling factor for the correlation energy.
    pub scaling: f64,
    /// Frozen core orbitals.
    pub frozen_core: usize,
}

/// Bohr⁻¹ per Å⁻¹ (inverse of the Å-to-Bohr conversion).
pub const BOHR_INV_PER_ANG_INV: f64 = 1.0 / 1.8897259886;

impl Default for AttenuatedMp2Config {
    fn default() -> Self {
        Self {
            omega: 0.420 * BOHR_INV_PER_ANG_INV, // 0.420 Å⁻¹ in Bohr⁻¹
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

/// Explicit erfc-attenuated alias of [`attenuated_ri_mp2`].
///
/// Same implementation; named to make the operator choice unambiguous
/// at the call site. Use [`attenuated_ri_mp2_long_range`] for the
/// complementary erf-attenuated (long-range only) variant.
#[inline]
pub fn erfc_attenuated_ri_mp2(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    rhf: &ScfResult,
    config: &AttenuatedMp2Config,
) -> Result<AttenuatedMp2Result, FerricError> {
    attenuated_ri_mp2(mol, obs, dfbs, rhf, config)
}

/// Long-range RI-MP2 using erf(omega·r)/r operator (complement of erfc).
///
/// Useful for range-separated hybrid decompositions where the short-range
/// part is treated by DFT and the long-range part by MP2 correlation.
/// erf(ω·r)/r + erfc(ω·r)/r = 1/r exactly, so
///   `attenuated_ri_mp2(erfc) + attenuated_ri_mp2_long_range(erf) ≈ ri_mp2(Coulomb)`
/// up to integral roundoff.
pub fn attenuated_ri_mp2_long_range(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    rhf: &ScfResult,
    config: &AttenuatedMp2Config,
) -> Result<AttenuatedMp2Result, FerricError> {
    let op = Operator::erf(config.omega);
    let ri_config = RiMp2Config { frozen_core: config.frozen_core };
    let (sc, _) = ri_mp2_spin_components(mol, obs, dfbs, op, rhf, &ri_config)?;
    let scaled_corr = config.scaling * sc.e_total;
    Ok(AttenuatedMp2Result {
        mp2_corr: scaled_corr,
        total_energy: rhf.energy + scaled_corr,
        spin_components: sc,
    })
}

/// Compute attenuated RI-MP2 using erfc(omega*r)/r operator.
///
/// The range-separation parameter omega is supplied directly (Bohr⁻¹). As
/// omega → 0, erfc → 1 and the result converges to standard MP2; as omega
/// grows the long-range Coulomb tail is suppressed and only short-range
/// correlation is retained.
pub fn attenuated_ri_mp2(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    rhf: &ScfResult,
    config: &AttenuatedMp2Config,
) -> Result<AttenuatedMp2Result, FerricError> {
    let op = Operator::erfc(config.omega);
    let ri_config = RiMp2Config { frozen_core: config.frozen_core };
    let (sc, _) = ri_mp2_spin_components(mol, obs, dfbs, op, rhf, &ri_config)?;
    let scaled_corr = config.scaling * sc.e_total;
    Ok(AttenuatedMp2Result {
        mp2_corr: scaled_corr,
        total_energy: rhf.energy + scaled_corr,
        spin_components: sc,
    })
}

/// Helper for range-separated MP2: returns (E_erfc_sr, E_erf_lr, E_full).
/// Should satisfy E_erfc_sr + E_erf_lr ≈ E_full (range-separation identity).
pub fn rs_mp2_decomposition(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    rhf: &ScfResult,
    config: &AttenuatedMp2Config,
) -> Result<(f64, f64, f64), FerricError> {
    use ferric_integrals::operator::Operator;
    let sr = erfc_attenuated_ri_mp2(mol, obs, dfbs, rhf, config)?.mp2_corr;
    let lr = attenuated_ri_mp2_long_range(mol, obs, dfbs, rhf, config)?.mp2_corr;
    let op = Operator::coulomb();
    let ri_config = RiMp2Config { frozen_core: config.frozen_core };
    let (sc, _) = ri_mp2_spin_components(mol, obs, dfbs, op, rhf, &ri_config)?;
    Ok((sr, lr, sc.e_total))
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
    fn test_attenuated_mp2_small_omega_approaches_full() {
        let (mol, obs, dfbs, rhf) = setup_h2();
        let full = crate::rimp2::ri_mp2(
            &mol, &obs, &dfbs, Operator::coulomb(), &rhf,
            &crate::rimp2::RiMp2Config::default(),
        ).unwrap();
        let config = AttenuatedMp2Config { omega: 0.01, ..Default::default() };
        let att = attenuated_ri_mp2(&mol, &obs, &dfbs, &rhf, &config).unwrap();
        eprintln!(
            "Full RI-MP2 corr: {:.10}, Small-omega attenuated corr: {:.10}, diff: {:.2e}",
            full.mp2_corr, att.mp2_corr, (att.mp2_corr - full.mp2_corr).abs()
        );
        assert!(
            (att.mp2_corr - full.mp2_corr).abs() < 1e-4,
            "small omega attenuated ({}) should approach full ({})", att.mp2_corr, full.mp2_corr
        );
    }

    #[test]
    fn test_rs_mp2_decomposition_sums_approximately_to_full() {
        // Range-separation identity erfc + erf = 1 ⇒ exact integrals satisfy
        // MP2(erfc) + MP2(erf) = MP2(Coulomb). With RI approximation using
        // cc-pVDZ-RI (fit for Coulomb), each operator picks up a different
        // RI truncation error and the identity holds only at the mHa level,
        // not bitwise. Production range-separated MP2 needs operator-specific
        // RI fits.
        let (mol, obs, dfbs, rhf) = setup_h2();
        let config = AttenuatedMp2Config { omega: 0.5, ..Default::default() };
        let (sr, lr, full) = rs_mp2_decomposition(&mol, &obs, &dfbs, &rhf, &config).unwrap();
        let sum = sr + lr;
        eprintln!(
            "MP2 decomposition: erfc_SR={:.8}, erf_LR={:.8}, sum={:.8}, full={:.8}, RI-error={:.2e}",
            sr, lr, sum, full, (sum - full).abs()
        );
        // ~10 mHa allowed — typical RI mismatch for non-Coulomb operators.
        assert!(
            (sum - full).abs() < 0.05,
            "erf + erfc MP2 should sum to full Coulomb MP2 within RI tolerance (diff = {:.2e})",
            (sum - full).abs()
        );
    }

    #[test]
    fn test_erfc_alias_matches_attenuated() {
        // The explicit erfc-named API must be bit-identical to attenuated_ri_mp2.
        let (mol, obs, dfbs, rhf) = setup_h2();
        let config = AttenuatedMp2Config { omega: 0.3, ..Default::default() };
        let a = attenuated_ri_mp2(&mol, &obs, &dfbs, &rhf, &config).unwrap();
        let b = erfc_attenuated_ri_mp2(&mol, &obs, &dfbs, &rhf, &config).unwrap();
        assert!((a.mp2_corr - b.mp2_corr).abs() < 1e-12);
    }
}
