use ndarray::{Array1, Array2, Array3, Array4};
use ndarray_npy::NpzWriter;
use std::fs::File;
use crate::cube::ExportError;

/// Atomic partial-charge schemes for the NPZ bundle. Grouping these (rather
/// than flat positional `Option<&[f64]>` params) mirrors `PdepRpaConfig`'s
/// nested-sub-struct precedent (`crates/ferric-rpa/src/config.rs`) and is
/// the natural landing spot for any future charge scheme (CHELPG/RESP/NPA/
/// ...) without growing `export_npz`'s own parameter count again.
#[derive(Default, Clone, Copy)]
pub struct ChargeSchemes<'a> {
    pub hirshfeld: Option<&'a [f64]>,
    pub lowdin: Option<&'a [f64]>,
    pub mulliken: Option<&'a [f64]>,
    /// CHELPG (ESP-fitted) atomic charges — structurally different from the
    /// three population-partition schemes above (see
    /// `ferric_rpa::properties::chelpg_charges`).
    pub chelpg: Option<&'a [f64]>,
    /// RESP (restrained ESP-fitted) atomic charges (see
    /// `ferric_rpa::properties::resp_charges`).
    pub resp: Option<&'a [f64]>,
}

/// Static and per-atom polarizability/field-response outputs.
#[derive(Default, Clone, Copy)]
pub struct PolarizabilityBundle<'a> {
    pub esp_atoms: Option<&'a [f64]>,
    pub alpha_tensor: Option<&'a [[f64; 3]; 3]>,
    pub electric_field: Option<&'a [[f64; 3]]>,
    pub alpha_atomic: Option<&'a [[[f64; 3]; 3]]>,
}

/// Dispersion (C6) outputs.
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
#[derive(Default, Clone, Copy)]
pub struct DispersionBundle<'a> {
    pub c6_freqs: Option<&'a [f64]>,
    pub c6_weights: Option<&'a [f64]>,
    pub alpha_atomic_dynamic: Option<&'a [Vec<[[f64; 3]; 3]>]>,
    pub c6_iso: Option<&'a Array2<f64>>,
    pub c6_aniso: Option<&'a [Vec<[[f64; 3]; 3]>]>,
}

/// Everything `export_npz` can write, grouped by category. See
/// `ChargeSchemes`/`PolarizabilityBundle`/`DispersionBundle` for the
/// per-category fields and their CONSUMER WARNINGs.
#[derive(Default, Clone, Copy)]
pub struct NpzBundle<'a> {
    pub mo_coeffs: Option<&'a Array2<f64>>,
    pub orbital_energies: Option<&'a [f64]>,
    pub pdep_eigenvectors: Option<&'a Array2<f64>>,
    pub boys_coeffs: Option<&'a Array2<f64>>,
    pub coords: Option<&'a Array2<f64>>,
    pub atomic_numbers: Option<&'a [usize]>,
    pub density_matrix: Option<&'a Array2<f64>>,
    pub dipole: Option<&'a [f64; 3]>,
    pub charges: ChargeSchemes<'a>,
    pub polarizability: PolarizabilityBundle<'a>,
    pub dispersion: DispersionBundle<'a>,
}

/// Exports key tensors and metadata for Machine Learning (e.g. Diffusion models)
/// into a compressed NPZ archive. See `NpzBundle` and its nested
/// `ChargeSchemes`/`PolarizabilityBundle`/`DispersionBundle` sub-structs for
/// what can be written and the CONSUMER WARNINGs on the C6 fields.
pub fn export_npz(path: &str, bundle: &NpzBundle) -> Result<(), ExportError> {
    let file = File::create(path)?;
    let mut writer = NpzWriter::new(file);

    if let Some(c) = bundle.mo_coeffs {
        writer.add_array("mo_coeffs", c).map_err(|e| ExportError::Other(e.to_string()))?;
    }

    if let Some(e) = bundle.orbital_energies {
        let e_arr = Array1::from_vec(e.to_vec());
        writer.add_array("orbital_energies", &e_arr).map_err(|e| ExportError::Other(e.to_string()))?;
    }

    if let Some(v) = bundle.pdep_eigenvectors {
        writer.add_array("pdep_eigenvectors", v).map_err(|e| ExportError::Other(e.to_string()))?;
    }

    if let Some(bc) = bundle.boys_coeffs {
        writer.add_array("boys_coeffs", bc).map_err(|e| ExportError::Other(e.to_string()))?;
    }

    if let Some(c) = bundle.coords {
        writer.add_array("coords", c).map_err(|e| ExportError::Other(e.to_string()))?;
    }

    if let Some(z) = bundle.atomic_numbers {
        let z_arr = Array1::from_vec(z.iter().map(|&x| x as i64).collect());
        writer.add_array("atomic_numbers", &z_arr).map_err(|e| ExportError::Other(e.to_string()))?;
    }

    if let Some(v) = bundle.polarizability.esp_atoms {
        let v_arr = Array1::from_vec(v.to_vec());
        writer.add_array("esp_atoms", &v_arr).map_err(|e| ExportError::Other(e.to_string()))?;
    }

    if let Some(a) = bundle.polarizability.alpha_tensor {
        let flat: Vec<f64> = a.iter().flat_map(|row| row.iter().copied()).collect();
        let a_arr = Array2::from_shape_vec((3, 3), flat).unwrap();
        writer.add_array("alpha_tensor", &a_arr).map_err(|e| ExportError::Other(e.to_string()))?;
    }

    if let Some(ef) = bundle.polarizability.electric_field {
        let n = ef.len();
        let flat: Vec<f64> = ef.iter().flat_map(|row| row.iter().copied()).collect();
        let ef_arr = Array2::from_shape_vec((n, 3), flat).unwrap();
        writer.add_array("electric_field", &ef_arr)
            .map_err(|e| ExportError::Other(e.to_string()))?;
    }

    if let Some(dm) = bundle.density_matrix {
        writer.add_array("density_matrix", dm)
            .map_err(|e| ExportError::Other(e.to_string()))?;
    }

    if let Some(aa) = bundle.polarizability.alpha_atomic {
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

    if let Some(q) = bundle.charges.hirshfeld {
        let q_arr = Array1::from_vec(q.to_vec());
        writer
            .add_array("hirshfeld_charges", &q_arr)
            .map_err(|e| ExportError::Other(e.to_string()))?;
    }

    if let Some(q) = bundle.charges.lowdin {
        let q_arr = Array1::from_vec(q.to_vec());
        writer
            .add_array("lowdin_charges", &q_arr)
            .map_err(|e| ExportError::Other(e.to_string()))?;
    }

    if let Some(q) = bundle.charges.mulliken {
        let q_arr = Array1::from_vec(q.to_vec());
        writer
            .add_array("mulliken_charges", &q_arr)
            .map_err(|e| ExportError::Other(e.to_string()))?;
    }

    if let Some(q) = bundle.charges.chelpg {
        let q_arr = Array1::from_vec(q.to_vec());
        writer
            .add_array("chelpg_charges", &q_arr)
            .map_err(|e| ExportError::Other(e.to_string()))?;
    }

    if let Some(q) = bundle.charges.resp {
        let q_arr = Array1::from_vec(q.to_vec());
        writer
            .add_array("resp_charges", &q_arr)
            .map_err(|e| ExportError::Other(e.to_string()))?;
    }

    if let Some(f) = bundle.dispersion.c6_freqs {
        let a = Array1::from_vec(f.to_vec());
        writer.add_array("c6_freqs", &a).map_err(|e| ExportError::Other(e.to_string()))?;
    }

    if let Some(w) = bundle.dispersion.c6_weights {
        let a = Array1::from_vec(w.to_vec());
        writer.add_array("c6_weights", &a).map_err(|e| ExportError::Other(e.to_string()))?;
    }

    if let Some(ad) = bundle.dispersion.alpha_atomic_dynamic {
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

    if let Some(c) = bundle.dispersion.c6_iso {
        writer.add_array("c6_iso", c).map_err(|e| ExportError::Other(e.to_string()))?;
    }

    if let Some(ca) = bundle.dispersion.c6_aniso {
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

    if let Some(mu) = bundle.dipole {
        let mu_arr = Array1::from_vec(mu.to_vec());
        writer.add_array("dipole", &mu_arr).map_err(|e| ExportError::Other(e.to_string()))?;
    }

    writer.finish().map_err(|e| ExportError::Other(e.to_string()))?;
    Ok(())
}
