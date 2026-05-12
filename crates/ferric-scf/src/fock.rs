//! Fock matrix builder traits and composite builder.
//!
//! The [`JBuilder`] and [`KBuilder`] traits abstract the Coulomb and exchange
//! matrix construction, allowing future pluggable implementations (e.g., LinK, CFMM).

use ferric_core::FerricError;
use ndarray::Array2;

/// Trait for building the Coulomb matrix J from a density matrix.
pub trait JBuilder {
    fn build(&mut self, d: &Array2<f64>, j: &mut Array2<f64>) -> Result<(), FerricError>;
    fn reset(&mut self);
}

/// Trait for building the exchange matrix K from a density matrix.
pub trait KBuilder {
    fn build(&mut self, d: &Array2<f64>, k: &mut Array2<f64>) -> Result<(), FerricError>;
    fn update_density(&mut self, d: &Array2<f64>);
    fn reset(&mut self);
}

/// Composite Fock builder: F = H_core + J - 0.5*K.
pub struct FockBuilder {
    pub hcore: Array2<f64>,
    pub j: Box<dyn JBuilder>,
    pub k: Box<dyn KBuilder>,
}

impl FockBuilder {
    pub fn build(&mut self, d: &Array2<f64>, f: &mut Array2<f64>) -> Result<(), FerricError> {
        let n = d.nrows();
        let mut j = Array2::zeros((n, n));
        let mut k = Array2::zeros((n, n));
        self.j.build(d, &mut j)?;
        self.k.build(d, &mut k)?;
        f.assign(&(&self.hcore + &j - &(0.5 * &k)));
        Ok(())
    }
}
