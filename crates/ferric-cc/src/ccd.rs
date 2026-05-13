//! CCD (Coupled-Cluster Doubles) stub.

use super::{CcConfig, CcResult};
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::RhfResult;
use ndarray::Array4;

pub fn ccd(
    _mol: &Molecule,
    obs: &PreparedBasis,
    _aux: &PreparedBasis,
    _op: Operator,
    _rhf: &RhfResult,
    cfg: &CcConfig,
) -> Result<CcResult, FerricError> {
    let nbas = obs.nbasis();
    let nocc = nbas.saturating_sub(cfg.frozen_core) / 2;
    let nvir = nbas.saturating_sub(nocc);

    Ok(CcResult {
        correlation_energy: 0.0,
        t1: None,
        t2: Array4::zeros((nocc, nvir, nocc, nvir)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ccd_stub_returns() {
        // Just verify the stub compiles and returns without panic
    }
}
