use ndarray::{Array1, Array2, Array3, Array4};
use ndarray_npy::NpzWriter;
use std::fs::File;
use crate::cube::ExportError;

/// Exports key tensors and metadata for Machine Learning (e.g. Diffusion models)
/// into a compressed NPZ archive.
///
/// CONSUMER WARNING — `c6_iso`/`c6_aniso` (open-work-triage item #9 / S9
/// spike, 2026-07-17): these two arrays are the per-atom PAIR Casimir-Polder
/// tensors (`C6Result::c6_iso_pair`/`c6_aniso_pair` from
/// `ferric_rpa::dispersion::casimir_polder_c6`), NOT the molecular C6 total.
/// **`c6_iso.sum()` is NOT the molecular C6** and diverges from the correct,
/// DOSD-comparable value by roughly -20% to -58% in measured cases (water/
/// aug-cc-pVDZ/RPA@PBE: Becke -57.6%, Hirshfeld -19.5% — see the bounded
/// regression test `bounded_divergence_pair_sum_vs_molecular_c6_water` in
/// `crates/ferric-rpa/tests/s9_per_atom_c6_consistency.rs` and the
/// CONSUMER WARNING on `dispersion::C6Result` for why: the per-atom pair
/// tensors use an atom-centred operator that excludes inter-atomic
/// charge-transfer/coupling that the molecular response includes). The
/// correct DOSD-comparable molecular C6 is `C6Result::c6_molecular_iso`,
/// which the CLI prints to stdout as `molecular C6 = X a.u.` but is
/// currently NOT itself written to this NPZ file — a consumer who wants
/// "the" molecular C6 must read it from CLI stdout (or call
/// `casimir_polder_c6` directly), not sum this array. See also
/// `docs/dosd-c6-rpa-vs-ts.md`'s "Numerical notes" for the analogous H2 case
/// (6.88 pair-sum vs 9.22 correct).
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
    // Per-atom PAIR Casimir-Polder tensors — NOT the molecular C6 total.
    // `c6_iso.sum()` != the molecular C6; see the CONSUMER WARNING on this
    // function's doc comment before using these to approximate a molecular
    // total.
    c6_iso: Option<&Array2<f64>>,
    c6_aniso: Option<&[Vec<[[f64; 3]; 3]>]>,
    dipole: Option<&[f64; 3]>,
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

    if let Some(mu) = dipole {
        let mu_arr = Array1::from_vec(mu.to_vec());
        writer.add_array("dipole", &mu_arr).map_err(|e| ExportError::Other(e.to_string()))?;
    }

    writer.finish().map_err(|e| ExportError::Other(e.to_string()))?;
    Ok(())
}
