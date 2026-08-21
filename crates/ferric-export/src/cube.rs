use ferric_core::basis::BasisSet;
use ferric_core::mol::Molecule;
use ndarray::{Array2, Array3};
use std::fs::File;
use std::io::{BufWriter, Write};
use thiserror::Error;

/// Errors from cube-file or NPZ export.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum ExportError {
    #[error("IO Error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Export error: {0}")]
    Other(String),
}

impl From<ExportError> for ferric_core::error::FerricError {
    fn from(e: ExportError) -> Self {
        match e {
            ExportError::Io(io) => Self::Io(io),
            ExportError::Other(s) => Self::General(s),
        }
    }
}

pub use ferric_integrals::ao_grid::GridSpec;

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

/// Evaluates a batch of molecular orbitals on a grid and exports one Gaussian
/// cube file per orbital.
///
/// `mo_coeffs` is the AO×MO coefficient matrix in the standard ferric
/// convention used throughout `ferric-scf`/`ferric-export::ml` (e.g.
/// `ScfResult::mos_r()`/`mos_alpha`): shape `(nbasis, n_mo)`, rows are AO
/// basis functions (in the order matching `eval_basis_on_grid`/`bs`),
/// columns are MOs (density = `C · Cᵀ`, see `rhf.rs`). `mo_indices` selects
/// which MO columns (0-based) to export.
///
/// The AO grid (`eval_basis_on_grid`) is evaluated once and reused for every
/// requested MO — one shared `O(nbasis × npts)` cost instead of one per MO.
/// Each MO's grid values are then a single GEMV:
/// `mo_vals = mo_coeffs.column(i)ᵀ · chi`.
///
/// Output files are named `{path_prefix}_mo{i:03}.cube` for each `i` in
/// `mo_indices`.
pub fn export_mo_cubes(
    mol: &Molecule,
    bs: &BasisSet,
    mo_coeffs: &Array2<f64>,
    mo_indices: &[usize],
    grid: &GridSpec,
    path_prefix: &str,
) -> Result<(), ExportError> {
    let chi = crate::gto_eval::eval_basis_on_grid(mol, bs, grid)
        .map_err(|e| ExportError::Other(e.to_string()))?;
    let nbf = chi.nrows();

    if mo_coeffs.nrows() != nbf {
        return Err(ExportError::Other(format!(
            "mo_coeffs row count {} does not match basis size {}",
            mo_coeffs.nrows(),
            nbf
        )));
    }

    for &i in mo_indices {
        if i >= mo_coeffs.ncols() {
            return Err(ExportError::Other(format!(
                "MO index {} out of range (mo_coeffs has {} columns)",
                i,
                mo_coeffs.ncols()
            )));
        }

        // mo_vals[g] = sum_mu mo_coeffs[mu, i] * chi[mu, g]  (single GEMV)
        let mo_vals = mo_coeffs.column(i).t().dot(&chi); // shape (npts,)

        let mut data = Array3::<f64>::zeros((grid.n_x, grid.n_y, grid.n_z));
        for ix in 0..grid.n_x {
            for iy in 0..grid.n_y {
                for iz in 0..grid.n_z {
                    let g = (ix * grid.n_y + iy) * grid.n_z + iz;
                    data[[ix, iy, iz]] = mo_vals[g];
                }
            }
        }

        let path = format!("{path_prefix}_mo{i:03}.cube");
        let comment = format!("MO {i} density (psi), ferric export_mo_cubes");
        export_cube(&path, mol, grid, &data, &comment)?;
    }

    Ok(())
}

#[cfg(test)]
mod mo_cube_tests {
    use super::*;
    use ferric_core::basis;
    use ferric_core::parallel::ParallelContext;
    use ferric_integrals::basis_bridge::PreparedBasis;
    use ferric_integrals::operator::Operator;
    use ferric_scf::rhf::{solve_rhf, RhfConfig};
    use ferric_scf::screening::SchwarzBounds;

    fn h2_mol() -> Molecule {
        Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 1.4\n", 0, 1).unwrap()
    }

    #[test]
    fn export_mo_cubes_writes_parseable_nonempty_cube_files() {
        let mol = h2_mol();
        let bs = basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let cfg = RhfConfig::default();
        let ctx = ParallelContext::default();
        let rhf = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();

        let grid = GridSpec {
            origin: [-1.5, -1.5, -1.5],
            n_x: 6,
            n_y: 6,
            n_z: 8,
            step_x: [0.5, 0.0, 0.0],
            step_y: [0.0, 0.5, 0.0],
            step_z: [0.0, 0.0, 0.5],
        };

        let tmp_dir = std::env::temp_dir();
        let prefix = tmp_dir.join(format!(
            "ferric_export_mo_cubes_test_{}",
            std::process::id()
        ));
        let prefix_str = prefix.to_str().unwrap();

        // H2/STO-3G has 2 MOs (bonding sigma_g = index 0, antibonding
        // sigma_u* = index 1); export both.
        let indices = [0usize, 1usize];
        export_mo_cubes(&mol, &bs, rhf.mos_r(), &indices, &grid, prefix_str).unwrap();

        for &i in &indices {
            let path = format!("{prefix_str}_mo{i:03}.cube");
            let contents = std::fs::read_to_string(&path).unwrap();
            let lines: Vec<&str> = contents.lines().collect();
            // 2 comment lines + 1 atom-count/origin line + 3 grid-vector
            // lines + 2 atom lines = 8 header lines, then volumetric data.
            assert!(lines.len() > 8, "cube file {path} too short: {} lines", lines.len());
            assert!(lines[0].contains("Ferric"));
            // atom-count line: first token should be "2" (2 H atoms)
            let natoms: usize = lines[2].split_whitespace().next().unwrap().parse().unwrap();
            assert_eq!(natoms, 2);
            // Non-empty numeric payload: at least one data value parses as f64.
            let has_numeric_data = lines[8..]
                .iter()
                .flat_map(|l| l.split_whitespace())
                .any(|tok| tok.parse::<f64>().is_ok());
            assert!(has_numeric_data, "cube file {path} has no parseable volumetric data");

            std::fs::remove_file(&path).ok();
        }

        // H2/STO-3G bonding MO (index 0): symmetric molecule along z with
        // both H at equal |z| from the bond midpoint (z=0.7), so |C_H1| ==
        // |C_H2| for the lowest (totally symmetric) MO, matching the
        // AO-level symmetry check in gto_eval.rs's normalization test.
        let c0 = rhf.mos_r().column(0);
        assert!((c0[0].abs() - c0[1].abs()).abs() < 1e-8, "bonding MO not symmetric: {c0:?}");
    }
}
