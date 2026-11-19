//! The molecular system being propagated.

use ferric_core::error::FerricError;
use ferric_core::mol::Molecule;
use ndarray::Array2;

/// Atoms, masses and starting positions of an MD run.
#[derive(Debug, Clone)]
pub struct MdSystem {
    /// Element symbols, one per atom.
    pub symbols: Vec<String>,
    /// Atomic masses in amu.
    pub masses_amu: Vec<f64>,
    /// Starting coordinates, `(n, 3)`, Bohr.
    pub coords: Array2<f64>,
}

impl MdSystem {
    /// Build a system, validating lengths, shape and positive finite masses.
    pub fn new(
        symbols: Vec<String>,
        masses_amu: Vec<f64>,
        coords: Array2<f64>,
    ) -> Result<Self, FerricError> {
        let _ = (&symbols, &masses_amu, &coords);
        todo!()
    }

    /// Build from a [`Molecule`]; masses default to standard atomic masses.
    /// Errors on ghost atoms, and on Z > 54 unless `masses_amu` overrides.
    pub fn from_molecule(mol: &Molecule, masses_amu: Option<&[f64]>) -> Result<Self, FerricError> {
        let _ = (mol, masses_amu);
        todo!()
    }
}
