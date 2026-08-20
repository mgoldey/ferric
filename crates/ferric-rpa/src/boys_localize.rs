//! Boys-localized occupied orbital wrapper for PDEP-RPA screening.
//!
//! Thin convenience layer on top of `ferric_mp2::boys::boys_localize`:
//! computes Foster-Boys localized occupied MOs and Boys centroids for the
//! active-occupied block of an RHF result.
//!
//! Closed-shell only (uses `eps_r()/mos_r()`).

use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron::dipole;
use ferric_mp2::boys::boys_localize;
use ferric_scf::ScfResult;
use ndarray::{s, Array2};

/// Result of localizing the active occupied block.
#[derive(Debug, Clone)]
pub struct BoysOccupied {
    /// Localized occupied MO coefficients, shape (nbas, nocc_active).
    pub c_loc: Array2<f64>,
    /// Per-orbital Boys centroid ⟨i_loc | r | i_loc⟩, [nocc_active][3] (Bohr).
    pub centroids: Vec<[f64; 3]>,
    /// Per-orbital Boys "spread" (Σ_α ⟨i|r_α²|i⟩ − ⟨i|r_α|i⟩² is the canonical
    /// definition, but here we report ‖centroid − centroid_mean‖ as a cheap
    /// diagnostic — true spreads need r² integrals which we skip).
    pub spreads: Vec<f64>,
    /// Number of Foster-Boys Jacobi sweeps performed.
    pub iterations: usize,
    /// Whether Foster-Boys converged.
    pub converged: bool,
}

/// Boys-localize the active occupied orbital block of an RHF result.
///
/// `first_occ` is the number of frozen-core orbitals to skip. Active occupied
/// block runs `[first_occ, first_occ + nocc_active)` along the MO axis.
pub fn boys_localize_occupied(
    rhf: &ScfResult,
    obs: &PreparedBasis,
    first_occ: usize,
    nocc_active: usize,
) -> Result<BoysOccupied, FerricError> {
    let c_can = rhf.mos_r();
    let c_occ_active = c_can.slice(s![.., first_occ..first_occ + nocc_active]).to_owned();

    // Dipole AO integrals at origin; localization rotation is gauge-invariant
    // for the choice of origin (only diagonal differences and off-diagonals are
    // used in the Foster-Boys 2×2 functional).
    let dip = dipole(obs, [0.0, 0.0, 0.0])?;

    let boys = boys_localize(&c_occ_active, &dip, 200);

    let mut centroids: Vec<[f64; 3]> = Vec::with_capacity(nocc_active);
    for i in 0..nocc_active {
        centroids.push([
            boys.centers[(i, 0)],
            boys.centers[(i, 1)],
            boys.centers[(i, 2)],
        ]);
    }

    // Cheap diagnostic: distance of each centroid from the mean (Bohr).
    let mut mean = [0.0f64; 3];
    for c in &centroids {
        for a in 0..3 { mean[a] += c[a]; }
    }
    for a in 0..3 { mean[a] /= nocc_active.max(1) as f64; }
    let spreads: Vec<f64> = centroids
        .iter()
        .map(|c| {
            let dx = c[0] - mean[0];
            let dy = c[1] - mean[1];
            let dz = c[2] - mean[2];
            (dx * dx + dy * dy + dz * dz).sqrt()
        })
        .collect();

    Ok(BoysOccupied {
        c_loc: boys.c_loc,
        centroids,
        spreads,
        iterations: boys.iterations,
        converged: boys.converged,
    })
}
