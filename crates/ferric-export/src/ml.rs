use ndarray::{Array1, Array2};
use ndarray_npy::NpzWriter;
use std::fs::File;
use crate::cube::ExportError;

/// Exports key tensors and metadata for Machine Learning (e.g. Diffusion models)
/// into a compressed NPZ archive.
pub fn export_npz(
    path: &str,
    mo_coeffs: Option<&Array2<f64>>,
    orbital_energies: Option<&[f64]>,
    pdep_eigenvectors: Option<&Array2<f64>>,
    boys_coeffs: Option<&Array2<f64>>,
    coords: Option<&Array2<f64>>,
    atomic_numbers: Option<&[usize]>,
) -> Result<(), ExportError> {
    let file = File::create(path)?;
    let mut writer = NpzWriter::new(file);

    if let Some(c) = mo_coeffs {
        writer.add_array("mo_coeffs", c).map_err(|e| ExportError::Other(e.to_string()))?;
    }
    
    if let Some(e) = orbital_energies {
        let e_arr = Array1::from_vec(e.to_vec());
        writer.add_array("orbital_energies", &e_arr).map_err(|e| ExportError::Other(e.to_string()))?;
    }

    if let Some(v) = pdep_eigenvectors {
        writer.add_array("pdep_eigenvectors", v).map_err(|e| ExportError::Other(e.to_string()))?;
    }

    if let Some(bc) = boys_coeffs {
        writer.add_array("boys_coeffs", bc).map_err(|e| ExportError::Other(e.to_string()))?;
    }

    if let Some(c) = coords {
        writer.add_array("coords", c).map_err(|e| ExportError::Other(e.to_string()))?;
    }

    if let Some(z) = atomic_numbers {
        let z_arr = Array1::from_vec(z.iter().map(|&x| x as i64).collect());
        writer.add_array("atomic_numbers", &z_arr).map_err(|e| ExportError::Other(e.to_string()))?;
    }

    writer.finish().map_err(|e| ExportError::Other(e.to_string()))?;
    Ok(())
}
