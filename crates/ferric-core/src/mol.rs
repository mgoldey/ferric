//! Molecular geometry: atoms, XYZ parser, and nuclear repulsion energy.

use crate::elements::symbol_to_z;
use crate::FerricError;
use std::fs;

const ANGSTROM_TO_BOHR: f64 = 1.0 / 0.529_177_210_92;

/// A single atom with element symbol, atomic number, and Cartesian coordinates in Bohr.
///
/// Ghost atoms (XYZ symbol prefixed with `@`, e.g. `@O`) carry the full basis of
/// their element but contribute **zero nuclear charge** and **zero electrons**.
/// They are used for counterpoise (CP) corrections.
///
/// Invariant: `z` is always the physical atomic number of the element (even for
/// ghosts), so basis lookup via `for_element(z)` works unchanged.  The `ghost`
/// flag suppresses contributions to `nelec()`, `nuclear_repulsion()`, and the
/// nuclear-attraction one-electron integrals.
#[derive(Debug, Clone)]
pub struct Atom {
    pub symbol: String,
    pub z: i32,
    pub x: f64,
    pub y: f64,
    pub zpos: f64,
    /// `true` if this is a ghost (basis-only) center: zero nuclear charge, zero electrons.
    pub ghost: bool,
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
            let raw_sym = fields[0];
            // Ghost atoms are denoted by a leading '@' (Q-Chem convention).
            let (ghost, sym) = if let Some(stripped) = raw_sym.strip_prefix('@') {
                (true, stripped)
            } else {
                (false, raw_sym)
            };
            let z = symbol_to_z(sym).ok_or_else(|| {
                FerricError::XyzParse(format!("unknown element {raw_sym:?} at atom {i}"))
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
                ghost,
            });
        }
        Ok(Molecule { atoms, charge, multiplicity })
    }

    /// Compute the classical nuclear repulsion energy in Hartree.
    ///
    /// Ghost atoms (zero nuclear charge) contribute nothing to this sum.
    pub fn nuclear_repulsion(&self) -> f64 {
        let mut v = 0.0;
        for i in 0..self.atoms.len() {
            if self.atoms[i].ghost { continue; }
            for j in (i + 1)..self.atoms.len() {
                if self.atoms[j].ghost { continue; }
                let a = &self.atoms[i];
                let b = &self.atoms[j];
                let dx = a.x - b.x;
                let dy = a.y - b.y;
                let dz = a.zpos - b.zpos;
                let r = (dx * dx + dy * dy + dz * dz).sqrt();
                v += (a.z as f64) * (b.z as f64) / r;
            }
        }
        v
    }

    /// Total number of electrons (sum of atomic numbers of real atoms minus charge).
    ///
    /// Ghost atoms (zero nuclear charge) contribute 0 electrons.
    pub fn nelec(&self) -> i32 {
        let z_sum: i32 = self.atoms.iter().filter(|a| !a.ghost).map(|a| a.z).sum();
        z_sum - self.charge
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

    // ── Ghost atom tests ──────────────────────────────────────────────────────

    /// Parse a 4-atom XYZ with an `@O` ghost: verify ghost flag, z, nelec, Vnn.
    #[test]
    fn test_ghost_parse_at_o() {
        // Water (3 real atoms) + a ghost O far away at (0, 0, 100 Å)
        let xyz = "4\nwater + ghost O\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n@O 0.000000 0.000000 100.000000\n";
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        assert_eq!(mol.atoms.len(), 4);

        // Ghost atom properties
        let ghost = &mol.atoms[3];
        assert!(ghost.ghost, "last atom should be a ghost");
        assert_eq!(ghost.z, 8, "ghost O should still have z=8 for basis assignment");
        assert_eq!(ghost.symbol, "O");

        // Real atoms are not ghosts
        for i in 0..3 {
            assert!(!mol.atoms[i].ghost, "atom {i} should not be a ghost");
        }

        // nelec: ghost O contributes no electrons
        let water_xyz = "3\nwater\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n";
        let water = Molecule::parse_xyz(water_xyz, 0, 1).unwrap();
        assert_eq!(mol.nelec(), water.nelec(), "nelec should equal water alone (ghost contributes 0 electrons)");

        // nuclear_repulsion: ghost O at 100 Å contributes zero
        let vnn_ghost = mol.nuclear_repulsion();
        let vnn_water = water.nuclear_repulsion();
        assert!(
            (vnn_ghost - vnn_water).abs() < 1e-12,
            "Vnn with far ghost ({vnn_ghost}) should equal water Vnn ({vnn_water})"
        );
    }

    /// Case-insensitive ghost: `@o` (lowercase) should work the same as `@O`.
    #[test]
    fn test_ghost_lowercase() {
        let xyz = "1\nghost He lowercase\n@he 0.0 0.0 0.0\n";
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        assert!(mol.atoms[0].ghost);
        assert_eq!(mol.atoms[0].z, 2); // He
        assert_eq!(mol.nelec(), -0); // 0 real electrons, charge 0
    }
}
