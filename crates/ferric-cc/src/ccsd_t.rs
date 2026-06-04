//! CCSD(T) perturbative triples correction.

use super::{CcConfig, CcResult};
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::ScfResult;

/// Compute the (T) triples correction to CCSD.
///
/// NOT YET IMPLEMENTED — returns 0.0. The perturbative-triples energy is the
/// non-iterative O(N^7) correction
/// `E_T = Σ_{ijkabc} (4 W + W_perm) (V − V_perm) / D_ijkabc`
/// built from the converged T1/T2 amplitudes and the integrals. The CCSD driver
/// now produces correct spin-orbital T1/T2 (see [`crate::ccsd::ccsd`]), so the
/// inputs are in place; the triples contraction itself remains to be written.
/// Returning 0.0 means `ccsd_t` currently reports E_CCSD, not E_CCSD(T).
pub fn ccsd_t(
    _mol: &Molecule,
    _obs: &PreparedBasis,
    _dfbs: &PreparedBasis,
    _op: Operator,
    _rhf: &ScfResult,
    cc: &CcResult,
    _cfg: &CcConfig,
) -> Result<f64, FerricError> {
    // Require T1 to be present (CCSD must have run), then return the unimplemented
    // (zero) triples correction. The previous dead O(N^7) loop computed nothing
    // and indexed orbital energies with amplitude dimensions, which panics now
    // that T2 is stored in the spin-orbital basis.
    let _t1 = cc
        .t1
        .as_ref()
        .ok_or_else(|| FerricError::General("CCSD(T) requires T1 amplitudes".into()))?;
    Ok(0.0)
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
    use crate::ccsd::ccsd;

    #[test]
    fn test_ccsd_t_h2_sto3g() {
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
        
        let ccsd_res = ccsd(&mol, &obs, &dfbs, op, &rhf, &cc_cfg).unwrap();
        let t_corr = ccsd_t(&mol, &obs, &dfbs, op, &rhf, &ccsd_res, &cc_cfg).unwrap();
        
        assert_eq!(t_corr, 0.0); // Current stub behavior
    }
}
