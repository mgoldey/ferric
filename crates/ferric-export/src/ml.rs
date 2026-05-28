use ndarray::{Array1, Array2, Array3, Array4};
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
    hirshfeld_charges: Option<&[f64]>,
    lowdin_charges: Option<&[f64]>,
    c6_freqs: Option<&[f64]>,
    c6_weights: Option<&[f64]>,
    alpha_atomic_dynamic: Option<&[Vec<[[f64; 3]; 3]>]>,
    c6_iso: Option<&Array2<f64>>,
    c6_aniso: Option<&[Vec<[[f64; 3]; 3]>]>,
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

    if let Some(q) = hirshfeld_charges {
        let q_arr = Array1::from_vec(q.to_vec());
        writer
            .add_array("hirshfeld_charges", &q_arr)
            .map_err(|e| ExportError::Other(e.to_string()))?;
    }

    if let Some(q) = lowdin_charges {
        let q_arr = Array1::from_vec(q.to_vec());
        writer
            .add_array("lowdin_charges", &q_arr)
            .map_err(|e| ExportError::Other(e.to_string()))?;
    }

    if let Some(f) = c6_freqs {
        let a = Array1::from_vec(f.to_vec());
        writer.add_array("c6_freqs", &a).map_err(|e| ExportError::Other(e.to_string()))?;
    }

    if let Some(w) = c6_weights {
        let a = Array1::from_vec(w.to_vec());
        writer.add_array("c6_weights", &a).map_err(|e| ExportError::Other(e.to_string()))?;
    }

    if let Some(ad) = alpha_atomic_dynamic {
        let natoms = ad.len();
        let nfreq = if natoms > 0 { ad[0].len() } else { 0 };
        let mut flat: Vec<f64> = Vec::with_capacity(natoms * nfreq * 9);
        for atom in ad {
            for t in atom {
                for row in t {
                    for v in row {
                        flat.push(*v);
                    }
                }
            }
        }
        let arr = Array4::from_shape_vec((natoms, nfreq, 3, 3), flat).unwrap();
        writer
            .add_array("alpha_atomic_dynamic", &arr)
            .map_err(|e| ExportError::Other(e.to_string()))?;
    }

    if let Some(c) = c6_iso {
        writer.add_array("c6_iso", c).map_err(|e| ExportError::Other(e.to_string()))?;
    }

    if let Some(ca) = c6_aniso {
        let n = ca.len();
        let mut flat: Vec<f64> = Vec::with_capacity(n * n * 9);
        for row in ca {
            for t in row {
                for r in t {
                    for v in r {
                        flat.push(*v);
                    }
                }
            }
        }
        let arr = Array4::from_shape_vec((n, n, 3, 3), flat).unwrap();
        writer.add_array("c6_aniso", &arr).map_err(|e| ExportError::Other(e.to_string()))?;
    }

    writer.finish().map_err(|e| ExportError::Other(e.to_string()))?;
    Ok(())
}
