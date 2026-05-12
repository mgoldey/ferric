use crate::elements::symbol_to_z;
use crate::FerricError;
use std::fs;

const ANGSTROM_TO_BOHR: f64 = 1.0 / 0.529_177_210_92;

#[derive(Debug, Clone)]
pub struct Atom {
    pub symbol: String,
    pub z: i32,
    pub x: f64,
    pub y: f64,
    pub zpos: f64,
}

#[derive(Debug, Clone)]
pub struct Molecule {
    pub atoms: Vec<Atom>,
}

impl Molecule {
    pub fn load_xyz(path: &str) -> Result<Self, FerricError> {
        let text = fs::read_to_string(path).map_err(FerricError::Io)?;
        Self::parse_xyz(&text)
    }

    pub fn parse_xyz(text: &str) -> Result<Self, FerricError> {
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
            });
        }
        Ok(Molecule { atoms })
    }

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
                v += (a.z as f64) * (b.z as f64) / r;
            }
        }
        v
    }

    pub fn nelec(&self) -> i32 {
        self.atoms.iter().map(|a| a.z).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_water() {
        let xyz = "3\nwater\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n";
        let mol = Molecule::parse_xyz(xyz).unwrap();
        assert_eq!(mol.atoms.len(), 3);
        assert_eq!(mol.atoms[0].z, 8);
        assert_eq!(mol.atoms[1].z, 1);
        assert_eq!(mol.nelec(), 10);
    }

    #[test]
    fn test_nuclear_repulsion_water() {
        let xyz = "3\nwater optimized HF/cc-pVDZ\nO   0.000000   0.000000   0.117790\nH   0.000000   0.755453  -0.471161\nH   0.000000  -0.755453  -0.471161\n";
        let mol = Molecule::parse_xyz(xyz).unwrap();
        let vnn = mol.nuclear_repulsion();
        assert!((vnn - 9.189193229309746).abs() < 1e-6, "Vnn = {vnn}, expected 9.189193...");
    }

    #[test]
    fn test_load_xyz_file() {
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        assert_eq!(mol.atoms.len(), 3);
        assert_eq!(mol.nelec(), 10);
    }
}
