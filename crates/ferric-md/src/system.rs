//! The molecular system being propagated.

use ferric_core::elements::atomic_mass;
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
        let n = symbols.len();
        let err = |m: String| FerricError::General(format!("invalid MD system: {m}"));
        if n == 0 {
            return Err(err("no atoms".into()));
        }
        if masses_amu.len() != n || coords.dim() != (n, 3) {
            return Err(err(format!(
                "{n} symbols, {} masses and coordinates of shape {:?} (expected ({n}, 3))",
                masses_amu.len(),
                coords.dim()
            )));
        }
        if let Some((i, m)) = masses_amu
            .iter()
            .enumerate()
            .find(|(_, m)| !(m.is_finite() && **m > 0.0))
        {
            return Err(err(format!(
                "mass of atom {i} must be finite and positive, got {m}"
            )));
        }
        if !coords.iter().all(|x| x.is_finite()) {
            return Err(err("coordinates contain non-finite values".into()));
        }
        Ok(Self {
            symbols,
            masses_amu,
            coords,
        })
    }

    /// Number of atoms.
    pub fn n_atoms(&self) -> usize {
        self.symbols.len()
    }

    /// Build from a [`Molecule`]; masses default to standard atomic masses.
    /// Errors on ghost atoms, and on Z > 54 unless `masses_amu` overrides.
    pub fn from_molecule(mol: &Molecule, masses_amu: Option<&[f64]>) -> Result<Self, FerricError> {
        if let Some(i) = mol.atoms.iter().position(|a| a.ghost) {
            return Err(FerricError::unsupported(
                "molecular dynamics with ghost atoms",
                format!("atom {i} is a ghost (basis-only) centre with no mass; remove it"),
            ));
        }
        let masses = match masses_amu {
            Some(m) => m.to_vec(),
            None => default_masses(mol)?,
        };
        let symbols = mol.atoms.iter().map(|a| a.symbol.clone()).collect();
        let mut coords = Array2::zeros((mol.atoms.len(), 3));
        for (mut row, a) in coords.rows_mut().into_iter().zip(&mol.atoms) {
            row.assign(&ndarray::arr1(&[a.x, a.y, a.zpos]));
        }
        Self::new(symbols, masses, coords)
    }
}

/// Standard atomic weights; errors outside the tabulated range.
fn default_masses(mol: &Molecule) -> Result<Vec<f64>, FerricError> {
    mol.atoms
        .iter()
        .enumerate()
        .map(|(i, a)| {
            atomic_mass(a.z).ok_or_else(|| {
                FerricError::General(format!(
                    "no standard atomic mass for atom {i} ({}, Z = {}); pass masses_amu",
                    a.symbol, a.z
                ))
            })
        })
        .collect()
}
