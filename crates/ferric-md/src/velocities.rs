//! Velocity initialisation and kinetic-energy helpers (atomic units).

use crate::rng::Xoshiro256pp;
use crate::units::BOLTZMANN_HARTREE_PER_K;
use ndarray::Array2;

/// Draw Maxwell-Boltzmann velocities at `temperature_k` for masses in
/// electron masses: each component is `sqrt(k_B T / m) * xi`, `xi ~ N(0,1)`,
/// drawn atom by atom in x, y, z order.
pub fn maxwell_boltzmann(
    masses_me: &[f64],
    temperature_k: f64,
    rng: &mut Xoshiro256pp,
) -> Array2<f64> {
    let kt = BOLTZMANN_HARTREE_PER_K * temperature_k;
    let mut v = Array2::zeros((masses_me.len(), 3));
    for (mut row, &m) in v.rows_mut().into_iter().zip(masses_me) {
        let sigma = (kt / m).sqrt();
        for x in row.iter_mut() {
            *x = sigma * rng.next_gaussian();
        }
    }
    v
}

/// Subtract the centre-of-mass velocity in place.
pub fn remove_com_momentum(masses_me: &[f64], v: &mut Array2<f64>) {
    let total_mass: f64 = masses_me.iter().sum();
    let mut p = [0.0; 3];
    for (row, &m) in v.rows().into_iter().zip(masses_me) {
        for k in 0..3 {
            p[k] += m * row[k];
        }
    }
    for mut row in v.rows_mut() {
        for k in 0..3 {
            row[k] -= p[k] / total_mass;
        }
    }
}

/// Kinetic energy `sum m v^2 / 2` in Hartree.
pub fn kinetic_energy(masses_me: &[f64], v: &Array2<f64>) -> f64 {
    v.rows()
        .into_iter()
        .zip(masses_me)
        .map(|(row, &m)| 0.5 * m * row.dot(&row))
        .sum()
}

/// Temperature in K from kinetic energy and degrees of freedom,
/// `T = 2 E_kin / (n_dof k_B)`; zero when `n_dof == 0`.
pub fn temperature(kinetic: f64, n_dof: usize) -> f64 {
    if n_dof == 0 {
        return 0.0;
    }
    2.0 * kinetic / (n_dof as f64 * BOLTZMANN_HARTREE_PER_K)
}
