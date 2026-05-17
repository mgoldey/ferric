use ndarray::{Array1, Array2, Array3};
use ndarray_npy::NpzWriter;
use std::fs::File;
use crate::cube::ExportError;

/// Exports key tensors and metadata for Machine Learning (e.g. Diffusion models)
/// into a compressed NPZ archive.
#[allow(clippy::too_many_arguments)]
pub fn export_npz(
    path: &str,
    mo_coeffs: Option<&Array2<f64>>,
    orbital_energies: Option<&[f64]>,
    pdep_eigenvectors: Option<&Array2<f64>>,
    boys_coeffs: Option<&Array2<f64>>,
    coords: Option<&Array2<f64>>,
    atomic_numbers: Option<&[usize]>,
    esp_atoms: Option<&[f64]>,
    alpha_tensor: Option<&[[f64; 3]; 3]>,
    electric_field: Option<&[[f64; 3]]>,
    density_matrix: Option<&Array2<f64>>,
    alpha_atomic: Option<&[[[f64; 3]; 3]]>,
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

    if let Some(v) = esp_atoms {
        let v_arr = Array1::from_vec(v.to_vec());
        writer.add_array("esp_atoms", &v_arr).map_err(|e| ExportError::Other(e.to_string()))?;
    }

    if let Some(a) = alpha_tensor {
        let flat: Vec<f64> = a.iter().flat_map(|row| row.iter().copied()).collect();
        let a_arr = Array2::from_shape_vec((3, 3), flat).unwrap();
        writer.add_array("alpha_tensor", &a_arr).map_err(|e| ExportError::Other(e.to_string()))?;
    }

    if let Some(ef) = electric_field {
        let n = ef.len();
        let flat: Vec<f64> = ef.iter().flat_map(|row| row.iter().copied()).collect();
        let ef_arr = Array2::from_shape_vec((n, 3), flat).unwrap();
        writer.add_array("electric_field", &ef_arr)
            .map_err(|e| ExportError::Other(e.to_string()))?;
    }

    if let Some(dm) = density_matrix {
        writer.add_array("density_matrix", dm)
            .map_err(|e| ExportError::Other(e.to_string()))?;
    }

    if let Some(aa) = alpha_atomic {
        let n = aa.len();
        let mut flat: Vec<f64> = Vec::with_capacity(n * 9);
        for a in aa {
            for row in a {
                for v in row {
                    flat.push(*v);
                }
            }
        }
        let arr = Array3::from_shape_vec((n, 3, 3), flat).unwrap();
        writer
            .add_array("alpha_atomic", &arr)
            .map_err(|e| ExportError::Other(e.to_string()))?;
    }

    writer.finish().map_err(|e| ExportError::Other(e.to_string()))?;
    Ok(())
}
