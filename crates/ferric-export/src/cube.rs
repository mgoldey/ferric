use ferric_core::mol::Molecule;
use ndarray::Array3;
use std::fs::File;
use std::io::{BufWriter, Write};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ExportError {
    #[error("IO Error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Export error: {0}")]
    Other(String),
}

/// A 3D Cartesian grid specification for volumetric data.
#[derive(Debug, Clone)]
pub struct GridSpec {
    pub origin: [f64; 3],
    pub n_x: usize,
    pub n_y: usize,
    pub n_z: usize,
    pub step_x: [f64; 3],
    pub step_y: [f64; 3],
    pub step_z: [f64; 3],
}

impl GridSpec {
    /// Constructs a uniform orthogonal grid tightly bounding the molecule plus a margin.
    pub fn bounding_box(mol: &Molecule, margin_bohr: f64, spacing_bohr: f64) -> Self {
        let mut min = [f64::MAX; 3];
        let mut max = [f64::MIN; 3];

        for atom in &mol.atoms {
            let coords = [atom.x, atom.y, atom.zpos];
            for i in 0..3 {
                if coords[i] < min[i] { min[i] = coords[i]; }
                if coords[i] > max[i] { max[i] = coords[i]; }
            }
        }

        let origin = [
            min[0] - margin_bohr,
            min[1] - margin_bohr,
            min[2] - margin_bohr,
        ];

        let lengths = [
            (max[0] - min[0]) + 2.0 * margin_bohr,
            (max[1] - min[1]) + 2.0 * margin_bohr,
            (max[2] - min[2]) + 2.0 * margin_bohr,
        ];

        let n_x = (lengths[0] / spacing_bohr).ceil() as usize;
        let n_y = (lengths[1] / spacing_bohr).ceil() as usize;
        let n_z = (lengths[2] / spacing_bohr).ceil() as usize;

        GridSpec {
            origin,
            n_x,
            n_y,
            n_z,
            step_x: [spacing_bohr, 0.0, 0.0],
            step_y: [0.0, spacing_bohr, 0.0],
            step_z: [0.0, 0.0, spacing_bohr],
        }
    }
}

/// Writes a 3D array of values to a standard Gaussian Cube file.
pub fn export_cube(
    path: &str,
    mol: &Molecule,
    grid: &GridSpec,
    data: &Array3<f64>,
    comment: &str,
) -> Result<(), ExportError> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    // Cube header: 2 comment lines
    writeln!(writer, "Ferric generated cube file")?;
    writeln!(writer, "{}", comment)?;

    // Number of atoms and origin
    writeln!(
        writer,
        "{:5} {:12.6} {:12.6} {:12.6}",
        mol.atoms.len(),
        grid.origin[0],
        grid.origin[1],
        grid.origin[2]
    )?;

    // Grid vectors
    writeln!(writer, "{:5} {:12.6} {:12.6} {:12.6}", grid.n_x, grid.step_x[0], grid.step_x[1], grid.step_x[2])?;
    writeln!(writer, "{:5} {:12.6} {:12.6} {:12.6}", grid.n_y, grid.step_y[0], grid.step_y[1], grid.step_y[2])?;
    writeln!(writer, "{:5} {:12.6} {:12.6} {:12.6}", grid.n_z, grid.step_z[0], grid.step_z[1], grid.step_z[2])?;

    // Atoms
    for atom in &mol.atoms {
        let z_float = atom.z as f64;
        writeln!(
            writer,
            "{:5} {:12.6} {:12.6} {:12.6} {:12.6}",
            atom.z, z_float, atom.x, atom.y, atom.zpos
        )?;
    }

    // Volumetric data: outer loop x, middle y, inner z
    // formatted 6 values per line
    let mut count = 0;
    for ix in 0..grid.n_x {
        for iy in 0..grid.n_y {
            for iz in 0..grid.n_z {
                write!(writer, " {:12.5E}", data[[ix, iy, iz]])?;
                count += 1;
                if count % 6 == 0 {
                    writeln!(writer)?;
                }
            }
            if count % 6 != 0 {
                writeln!(writer)?;
                count = 0;
            }
        }
    }

    Ok(())
}

/// Helper function placeholder: evaluates MOs and exports multiple cubes
pub fn export_mo_cubes(_mol: &Molecule, _path_prefix: &str) -> Result<(), ExportError> {
    // Requires AO evaluator implementation in ferric-integrals
    Err(ExportError::Other("AO grid evaluation not yet implemented".into()))
}
