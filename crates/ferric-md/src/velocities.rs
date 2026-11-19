//! Velocity initialisation and kinetic-energy helpers (atomic units).

use crate::rng::Xoshiro256pp;
use ndarray::Array2;

/// Draw Maxwell-Boltzmann velocities at `temperature_k` for masses in
/// electron masses.
pub fn maxwell_boltzmann(
    masses_me: &[f64],
    temperature_k: f64,
    rng: &mut Xoshiro256pp,
) -> Array2<f64> {
    let _ = (masses_me, temperature_k, rng);
    todo!()
}

/// Subtract the centre-of-mass velocity in place.
pub fn remove_com_momentum(masses_me: &[f64], v: &mut Array2<f64>) {
    let _ = (masses_me, v);
    todo!()
}

/// Kinetic energy `sum m v^2 / 2` in Hartree.
pub fn kinetic_energy(masses_me: &[f64], v: &Array2<f64>) -> f64 {
    let _ = (masses_me, v);
    todo!()
}

/// Temperature in K from kinetic energy and degrees of freedom.
pub fn temperature(kinetic: f64, n_dof: usize) -> f64 {
    let _ = (kinetic, n_dof);
    todo!()
}
