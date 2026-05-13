//! CCSD = CCD + Singles stub.

use super::{ccd, CcConfig, CcResult};
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::RhfResult;
use ndarray::Array2;

pub fn ccsd(
    mol: &Molecule,
    obs: &PreparedBasis,
    aux: &PreparedBasis,
    op: Operator,
    rhf: &RhfResult,
    cfg: &CcConfig,
) -> Result<CcResult, FerricError> {
    let ccd_res = ccd::ccd(mol, obs, aux, op, rhf, cfg)?;

    let nbas = obs.nbasis();
    let nocc_total = rhf.orbital_energies.len() / 2;
    let nocc = nocc_total.saturating_sub(cfg.frozen_core);
    let nvir = nbas.saturating_sub(nocc_total);
    let t1 = Array2::<f64>::zeros((nocc, nvir));

    Ok(CcResult {
        correlation_energy: ccd_res.correlation_energy,
        t1: Some(t1),
        t2: ccd_res.t2,
    })
}
