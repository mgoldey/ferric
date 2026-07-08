//! Analytical nuclear gradients for OO-RI-MP2 (stub).
//!
//! The OO-RI-MP2 gradient is more complex than standard RI-MP2 because
//! the orbitals are optimized and the response equations differ.
//! This module provides the interface; full implementation is pending.

use crate::rimp2::active_occ;
use crate::oo_rimp2::{compute_b_full_mo, compute_t2_and_integrals, build_mp2_density, OoRiMp2Result};
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::gradient::oneelectron_gradient;
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;

/// Compute the analytical nuclear gradient for OO-RI-MP2 (partial).
pub fn oo_ri_mp2_gradient(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    _bounds: &SchwarzBounds,
    result: &OoRiMp2Result,
    frozen_core: usize,
) -> Result<Array2<f64>, FerricError> {
    let nbas = obs.nbasis();
    let nocc_total = (mol.nelec() / 2) as usize;
    let nocc = active_occ(nocc_total, frozen_core)?;
    let nvir = nbas - nocc_total;
    let naux = dfbs.nbasis();

    let b_full = compute_b_full_mo(obs, dfbs, op, &result.mos)?;
    let b_ov_3d = b_full.slice(ndarray::s![.., frozen_core..nocc_total, nocc_total..]);
    let b_ov = b_ov_3d.to_owned()
        .into_shape_with_order((naux, nocc * nvir))
        .unwrap();

    let (t2, _) = compute_t2_and_integrals(
        &b_ov, &result.orbital_energies, nocc, nvir, nocc_total, frozen_core, naux,
    );

    let (p_oo, p_vv) = build_mp2_density(&t2, nocc, nvir);

    // Build relaxed 1-PDM in MO basis (simplified: no Z-vector for OO case)
    let nmo = nbas;
    let mut p_mo = Array2::zeros((nmo, nmo));
    for i in 0..nocc {
        let i_mo = frozen_core + i;
        p_mo[(i_mo, i_mo)] += 2.0;
        for j in 0..nocc {
            let j_mo = frozen_core + j;
            p_mo[(i_mo, j_mo)] += p_oo[(i, j)];
        }
    }
    for a in 0..nvir {
        let a_mo = nocc_total + a;
        for b in 0..nvir {
            let b_mo = nocc_total + b;
            p_mo[(a_mo, b_mo)] += p_vv[(a, b)];
        }
    }

    let d_ao = result.mos.dot(&p_mo).dot(&result.mos.t());

    // Simplified W: W_pq = eps_p * P_pq (diagonal approximation)
    let mut w_mo = Array2::zeros((nmo, nmo));
    for p in 0..nmo {
        for q in 0..nmo {
            w_mo[(p, q)] = 0.5 * p_mo[(p, q)] * (result.orbital_energies[p] + result.orbital_energies[q]);
        }
    }
    let w_ao = result.mos.dot(&w_mo).dot(&result.mos.t());

    let grad = oneelectron_gradient(mol, obs, &d_ao, &w_ao)?;

    Ok(grad)
}
