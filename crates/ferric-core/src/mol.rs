//! Molecular geometry: atoms, XYZ parser, and nuclear repulsion energy.

use crate::basis::BasisSet;
use crate::elements::symbol_to_z;
use crate::FerricError;
use std::fs;

const ANGSTROM_TO_BOHR: f64 = 1.0 / 0.529_177_210_92;

/// A single atom with element symbol, atomic number, and Cartesian coordinates in Bohr.
#[derive(Debug, Clone)]
pub struct Atom {
    pub symbol: String,
    pub z: i32,
    pub x: f64,
    pub y: f64,
    pub zpos: f64,
    /// Number of core electrons replaced by an effective core potential (ECP).
    /// Zero for all-electron atoms. Set by [`Molecule::apply_ecp`] when an ECP
    /// basis is loaded. The effective nuclear charge seen by the electrons is
    /// `z - n_core_ecp`, and the valence electron count is reduced accordingly.
    pub n_core_ecp: i32,
}

impl Atom {
    /// Effective nuclear charge: `z - n_core_ecp`. Equals `z` for all-electron
    /// atoms. This is the point charge the nuclear-attraction operator and the
    /// nuclear-repulsion energy must use for ECP atoms.
    #[inline]
    pub fn effective_z(&self) -> i32 {
        self.z - self.n_core_ecp
    }
}

/// A collection of atoms forming a molecule.
///
/// Coordinates are stored internally in Bohr. The XYZ parser converts from Angstroms.
#[derive(Debug, Clone)]
pub struct Molecule {
    pub atoms: Vec<Atom>,
    pub charge: i32,
    pub multiplicity: usize,
}

impl Molecule {
    /// Load a molecule from an XYZ file on disk, assuming neutral singlet.
    ///
    /// Coordinates in the file are expected in Angstroms and are converted to Bohr.
    pub fn load_xyz(path: &str) -> Result<Self, FerricError> {
        let text = fs::read_to_string(path).map_err(FerricError::Io)?;
        Self::parse_xyz(&text, 0, 1)
    }

    /// Load a molecule from an XYZ file with explicit charge and multiplicity.
    pub fn load_xyz_with_charge(path: &str, charge: i32, mult: usize) -> Result<Self, FerricError> {
        let text = fs::read_to_string(path).map_err(FerricError::Io)?;
        Self::parse_xyz(&text, charge, mult)
    }

    /// Parse a molecule from an XYZ-format string, with explicit charge and multiplicity.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferric_core::mol::Molecule;
    ///
    /// let xyz = "2\nH2\nH 0 0 0\nH 0 0 0.74\n";
    /// let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    /// assert_eq!(mol.atoms.len(), 2);
    /// ```
    pub fn parse_xyz(text: &str, charge: i32, multiplicity: usize) -> Result<Self, FerricError> {
        let mut lines = text.lines();
        let n: usize = lines
            .next()
            .ok_or_else(|| FerricError::XyzParse("empty file".into()))?
            .trim()
            .parse()
            .map_err(|e| FerricError::XyzParse(format!("bad atom count: {e}")))?;
        lines.next(); // comment line
        let mut atoms = Vec::with_capacity(n);
        for i in 0..n {
            let line = lines
                .next()
                .ok_or_else(|| FerricError::XyzParse(format!("expected {n} atoms, got {i}")))?;
            let fields: Vec<&str> = line.split_whitespace().collect();
            if fields.len() < 4 {
                return Err(FerricError::XyzParse(format!(
                    "atom {i}: expected 4 fields, got {}",
                    fields.len()
                )));
            }
            let sym = fields[0];
            let z = symbol_to_z(sym).ok_or_else(|| {
                FerricError::XyzParse(format!("unknown element {sym:?} at atom {i}"))
            })?;
            let x: f64 = fields[1].parse().map_err(|e| FerricError::XyzParse(format!("atom {i} x: {e}")))?;
            let y: f64 = fields[2].parse().map_err(|e| FerricError::XyzParse(format!("atom {i} y: {e}")))?;
            let zpos: f64 = fields[3].parse().map_err(|e| FerricError::XyzParse(format!("atom {i} z: {e}")))?;
            atoms.push(Atom {
                symbol: sym.to_string(),
                z,
                x: x * ANGSTROM_TO_BOHR,
                y: y * ANGSTROM_TO_BOHR,
                zpos: zpos * ANGSTROM_TO_BOHR,
                n_core_ecp: 0,
            });
        }
        Ok(Molecule { atoms, charge, multiplicity })
    }

    /// Compute the classical nuclear repulsion energy in Hartree.
    pub fn nuclear_repulsion(&self) -> f64 {
        let mut v = 0.0;
        for i in 0..self.atoms.len() {
            for j in (i + 1)..self.atoms.len() {
                let a = &self.atoms[i];
                let b = &self.atoms[j];
                let dx = a.x - b.x;
                let dy = a.y - b.y;
                let dz = a.zpos - b.zpos;
                let r = (dx * dx + dy * dy + dz * dz).sqrt();
                v += (a.effective_z() as f64) * (b.effective_z() as f64) / r;
            }
        }
        v
    }

    /// Total number of (explicitly treated) electrons:
    /// `Σ z − Σ n_core_ecp − charge`. For ECP atoms the core electrons replaced
    /// by the potential are excluded, so this is the valence electron count.
    pub fn nelec(&self) -> i32 {
        let z_sum: i32 = self.atoms.iter().map(|a| a.effective_z()).sum();
        z_sum - self.charge
    }

    /// Populate each atom's `n_core_ecp` from an ECP-carrying basis set.
    ///
    /// For every atom whose element has an ECP definition in `bs.ecps`, set its
    /// `n_core_ecp` to that definition's `n_core`. Atoms without an ECP are left
    /// untouched (`n_core_ecp` stays 0). This is the single point where the
    /// reduced electron count and effective nuclear charge enter the molecule;
    /// `nelec()` and `nuclear_repulsion()` both read `n_core_ecp` afterward.
    ///
    /// No-op when `bs.ecps` is empty (the all-electron path).
    pub fn apply_ecp(&mut self, bs: &BasisSet) {
        if bs.ecps.is_empty() {
            return;
        }
        for atom in &mut self.atoms {
            if let Some(def) = bs.ecp_for_element(atom.z) {
                atom.n_core_ecp = def.n_core;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_water() {
        let xyz = "3\nwater\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n";
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        assert_eq!(mol.atoms.len(), 3);
        assert_eq!(mol.atoms[0].z, 8);
        assert_eq!(mol.atoms[1].z, 1);
        assert_eq!(mol.nelec(), 10);
    }

    #[test]
    fn test_nuclear_repulsion_water() {
        let xyz = "3\nwater optimized HF/cc-pVDZ\nO   0.000000   0.000000   0.117790\nH   0.000000   0.755453  -0.471161\nH   0.000000  -0.755453  -0.471161\n";
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let vnn = mol.nuclear_repulsion();
        assert!((vnn - 9.189193229309746).abs() < 1e-6, "Vnn = {vnn}, expected 9.189193...");
    }

    #[test]
    fn test_load_xyz_file() {
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        assert_eq!(mol.atoms.len(), 3);
        assert_eq!(mol.nelec(), 10);
    }

    #[test]
    fn test_nelec_i2_def2ecp() {
        // I2 with def2-ECP: each iodine has its 28-electron core replaced, leaving
        // 53 − 28 = 25 explicit electrons each → 50 total, not 2*53 = 106.
        // (Verified against PySCF: gto.M(...,ecp='def2-svp').nelectron == 50,
        // atom_charge == 25. The "7 valence" chemical count is NOT the def2-ECP
        // explicit electron count — def2-ECP removes only [Kr]4d10 = 28.)
        use crate::basis;
        let xyz = "2\nI2\nI 0.0 0.0 0.0\nI 0.0 0.0 2.666\n";
        let mut mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        assert_eq!(mol.nelec(), 106, "before apply_ecp: all-electron count");
        let bs = basis::bundled("def2-svp").unwrap();
        mol.apply_ecp(&bs);
        assert_eq!(mol.atoms[0].n_core_ecp, 28);
        assert_eq!(mol.atoms[0].effective_z(), 25);
        assert_eq!(mol.nelec(), 50, "I2 explicit electrons (25 each)");
    }

    #[test]
    fn test_apply_ecp_noop_without_ecp() {
        // A basis set with no ECP block must leave n_core_ecp untouched.
        use crate::basis;
        let mol_xyz = "3\nwater\nO 0 0 0.117790\nH 0 0.755453 -0.471161\nH 0 -0.755453 -0.471161\n";
        let mut mol = Molecule::parse_xyz(mol_xyz, 0, 1).unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        mol.apply_ecp(&bs);
        assert!(mol.atoms.iter().all(|a| a.n_core_ecp == 0));
        assert_eq!(mol.nelec(), 10);
    }
}
