//! Molecular geometry: atoms, XYZ parser, and nuclear repulsion energy.

use crate::basis::BasisSet;
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
    /// Number of core electrons replaced by an effective core potential (ECP).
    /// Zero for all-electron atoms. Set by [`Molecule::apply_ecp`] when an ECP
    /// basis is loaded. The effective nuclear charge seen by the electrons is
    /// `z - n_core_ecp`, and the valence electron count is reduced accordingly.
    pub n_core_ecp: i32,
}

impl Atom {
    /// Effective nuclear charge: `0` for a ghost (basis-only) center, else
    /// `z - n_core_ecp`. Equals `z` for an ordinary all-electron atom. This is
    /// the point charge the nuclear-attraction operator and the nuclear-repulsion
    /// energy must use.
    #[inline]
    pub fn effective_z(&self) -> i32 {
        if self.ghost { 0 } else { self.z - self.n_core_ecp }
    }
}

/// Validate that `nelec` (total electron count) and `multiplicity` (2S+1) are
/// consistent: `n_alpha = (nelec + multiplicity - 1) / 2` must be a
/// non-negative integer.
///
/// Derivation: `n_alpha - n_beta = multiplicity - 1` (2S = multiplicity - 1
/// unpaired electrons) and `n_alpha + n_beta = nelec`, so
/// `n_alpha = (nelec + multiplicity - 1) / 2`. This fails (non-integer) when
/// `nelec + multiplicity - 1` is odd, i.e. `nelec` and `multiplicity` have
/// the same parity (both even or both odd) — an odd electron count requires
/// an even multiplicity (doublet, quartet, ...) and vice versa. It also fails
/// (negative `n_beta`) when `multiplicity - 1 > nelec`, i.e. more unpaired
/// spin than there are electrons to supply.
///
/// Without this check, an inconsistent combination (e.g. an odd-electron
/// molecule declared as a closed-shell singlet) sails through molecule
/// construction and only fails deep in the SCF loop as a misleading
/// `"SCF did not converge after 0 iterations"` error.
fn validate_electron_multiplicity_parity(nelec: i32, multiplicity: usize) -> Result<(), FerricError> {
    let two_s = multiplicity as i64 - 1; // multiplicity - 1 = 2S = n_alpha - n_beta
    let numerator = nelec as i64 + two_s;
    // n_alpha = numerator / 2 must be a non-negative integer, AND the implied
    // n_beta = nelec - n_alpha must also be non-negative (numerator <= 2*nelec,
    // i.e. two_s <= nelec) -- otherwise multiplicity demands more unpaired
    // electrons than the molecule has electrons at all (e.g. 1 electron,
    // multiplicity=4 would need 3 unpaired electrons from a single electron).
    if numerator < 0 || numerator % 2 != 0 || numerator > 2 * nelec as i64 {
        return Err(FerricError::General(format!(
            "inconsistent charge/multiplicity: {nelec} electrons with multiplicity {multiplicity} \
             implies n_alpha = ({nelec} + {multiplicity} - 1) / 2 = {numerator}/2, which is not a \
             non-negative integer no greater than the electron count. An odd electron count needs \
             an even multiplicity (2, 4, ...) and vice versa; multiplicity - 1 (unpaired electrons) \
             must also not exceed the electron count."
        )));
    }
    Ok(())
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
        let text = fs::read_to_string(path)
            .map_err(|e| FerricError::General(format!("cannot read xyz file {path:?}: {e}")))?;
        Self::parse_xyz(&text, 0, 1)
    }

    /// Load a molecule from an XYZ file with explicit charge and multiplicity.
    pub fn load_xyz_with_charge(path: &str, charge: i32, mult: usize) -> Result<Self, FerricError> {
        let text = fs::read_to_string(path)
            .map_err(|e| FerricError::General(format!("cannot read xyz file {path:?}: {e}")))?;
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
                n_core_ecp: 0,
            });
        }
        let mol = Molecule { atoms, charge, multiplicity };
        validate_electron_multiplicity_parity(mol.nelec(), multiplicity)?;
        Ok(mol)
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
                v += (a.effective_z() as f64) * (b.effective_z() as f64) / r;
            }
        }
        v
    }

    /// Total number of (explicitly treated) electrons:
    /// `Σ effective_z − charge`. Ghost atoms contribute 0 (basis-only centers);
    /// for ECP atoms the core electrons replaced by the potential are excluded
    /// (via `effective_z = z − n_core_ecp`), so this is the valence electron count.
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

    // ── Ghost atom tests ──────────────────────────────────────────────────────

    // (ghost) Parse a 4-atom XYZ with an `@O` ghost: verify ghost flag, z, nelec, Vnn.
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

    // ── ECP tests ─────────────────────────────────────────────────────────────

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

    // ── Electron-count / multiplicity parity ────────────────────────────────

    /// The exact broken case from the usability audit: an odd-electron
    /// molecule (H atom, 1 electron) declared as a closed-shell singlet
    /// (multiplicity=1). This must be a clear, immediate error at parse time,
    /// not a silent pass-through that later fails deep in the SCF loop as a
    /// misleading "SCF did not converge after 0 iterations" message.
    #[test]
    fn odd_electron_singlet_is_rejected_at_parse_time() {
        let xyz = "1\nH atom\nH 0.0 0.0 0.0\n";
        let err = Molecule::parse_xyz(xyz, 0, 1).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("inconsistent") && msg.contains("multiplicity"),
            "expected a clear parity error, got: {msg}"
        );
    }

    #[test]
    fn odd_electron_doublet_is_accepted() {
        // H atom, 1 electron, multiplicity 2 (doublet) is the physically
        // consistent combination and must parse fine.
        let xyz = "1\nH atom\nH 0.0 0.0 0.0\n";
        let mol = Molecule::parse_xyz(xyz, 0, 2).unwrap();
        assert_eq!(mol.nelec(), 1);
    }

    #[test]
    fn even_electron_singlet_is_accepted() {
        // Water, 10 electrons, multiplicity 1 (closed-shell singlet): the
        // ordinary, common case must be unaffected by the new check.
        let xyz = "3\nwater\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n";
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        assert_eq!(mol.nelec(), 10);
    }

    #[test]
    fn even_electron_doublet_is_rejected() {
        // Water (10 electrons) declared as a doublet (multiplicity=2) is
        // the reverse parity mismatch and must also be rejected.
        let xyz = "3\nwater\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n";
        let err = Molecule::parse_xyz(xyz, 0, 2).unwrap_err();
        assert!(err.to_string().contains("inconsistent"));
    }

    #[test]
    fn multiplicity_exceeding_electron_count_is_rejected() {
        // multiplicity - 1 (unpaired electrons) cannot exceed nelec.
        let xyz = "1\nH atom\nH 0.0 0.0 0.0\n";
        let err = Molecule::parse_xyz(xyz, 0, 4).unwrap_err(); // needs 3 unpaired e- but only has 1
        assert!(err.to_string().contains("inconsistent"));
    }

    #[test]
    fn charged_molecule_parity_uses_effective_electron_count() {
        // H2+ : 2 protons - 1 charge = 1 electron -> must be a doublet, not a singlet.
        let xyz = "2\nH2+\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\n";
        assert!(Molecule::parse_xyz(xyz, 1, 1).is_err());
        let mol = Molecule::parse_xyz(xyz, 1, 2).unwrap();
        assert_eq!(mol.nelec(), 1);
    }
}
