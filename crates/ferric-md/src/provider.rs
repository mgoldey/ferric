//! The [`ForceProvider`] abstraction: anything that yields energy and gradient.

use ferric_core::error::FerricError;
use ndarray::Array2;

/// Source of potential energies and nuclear gradients.
pub trait ForceProvider {
    /// Number of atoms the provider expects.
    fn n_atoms(&self) -> usize;
    /// `coords` is `(n, 3)` in Bohr; returns `(E` in Hartree, `dE/dR` in
    /// Hartree/Bohr with shape `(n, 3))`.
    fn energy_gradient(&mut self, coords: &Array2<f64>) -> Result<(f64, Array2<f64>), FerricError>;
    /// Short human-readable label of the method (used in certificates).
    fn label(&self) -> String;
}

/// A [`ForceProvider`] backed by a closure.
pub struct FnProvider<F> {
    /// Number of atoms.
    pub n_atoms: usize,
    /// Label reported by [`ForceProvider::label`].
    pub label: String,
    /// The energy/gradient closure.
    pub f: F,
}

impl<F: FnMut(&Array2<f64>) -> Result<(f64, Array2<f64>), FerricError>> ForceProvider
    for FnProvider<F>
{
    fn n_atoms(&self) -> usize {
        self.n_atoms
    }

    fn energy_gradient(&mut self, coords: &Array2<f64>) -> Result<(f64, Array2<f64>), FerricError> {
        (self.f)(coords)
    }

    fn label(&self) -> String {
        self.label.clone()
    }
}
