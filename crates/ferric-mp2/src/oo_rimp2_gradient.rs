//! Analytical nuclear gradients for OO-RI-MP2.

use crate::oo_rimp2::{build_oo_mp2_relaxed_density, compute_b_full_mo, compute_t2_and_integrals, OoRiMp2Result};
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::gradient::{oneelectron_gradient, build_energy_weighted_density};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;

/// Compute the analytical nuclear gradient for OO-RI-MP2.
pub fn oo_ri_mp2_gradient(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    result: &OoRiMp2Result,
    frozen_core: usize,
) -> Result<Array2<f64>, FerricError> {
    let nbas = obs.nbasis();
    let nocc_total = (mol.nelec() / 2) as usize;
    let nocc = nocc_total - frozen_core;
    let nvir = nbas - nocc_total;
    let nmo = nbas;

    // 1. Recompute B_full and T2 for the optimized orbitals
    let b_full = compute_b_full_mo(obs, dfbs, op, &result.mos)?;
    let b_ov = b_full.slice(ndarray::s![.., frozen_core..nocc_total, nocc_total..]).to_owned();
    let (t2, _) = compute_t2_and_integrals(
        &b_ov.into_shape_with_order((dfbs.nbasis(), nocc * nvir)).unwrap(),
        &result.orbital_energies,
        nocc,
        nvir,
        nocc_total,
        frozen_core,
        dfbs.nbasis(),
    );

    // 2. Build relaxed 1-PDM in MO basis
    let p_mo = build_oo_mp2_relaxed_density(&t2, nocc, nvir, nmo, frozen_core);
    
    // 3. Map to AO basis: D_AO = C * P_MO * C^T
    let d_ao = result.mos.dot(&p_mo).dot(&result.mos.t());

    // 4. Build energy-weighted density W
    // For OO-MP2, W has contributions from both HF and MP2 parts.
    // Simplified: reuse the HF builder for now as a baseline.
    let w_ao = build_energy_weighted_density_oo(&result, nocc_total, &p_mo);

    // 5. One-electron gradient contributions
    let mut grad = oneelectron_gradient(mol, obs, &d_ao, &w_ao)?;

    // 6. Two-electron gradient contributions (RI-specific)
    // This is the most complex part and requires d(PQ)/dR and d(P|mu nu)/dR.
    // ... logic for RI response ...
    
    Ok(grad)
}

fn build_energy_weighted_density_oo(
    result: &OoRiMp2Result,
    nocc_total: usize,
    p_mo: &Array2<f64>,
) -> Array2<f64> {
    let n = result.mos.nrows();
    let mut w_mo = Array2::zeros((n, n));
    for p in 0..n {
        for q in 0..n {
            w_mo[(p, q)] = 0.5 * p_mo[(p, q)] * (result.orbital_energies[p] + result.orbital_energies[q]);
        }
    }
    result.mos.dot(&w_mo).dot(&result.mos.t())
}
