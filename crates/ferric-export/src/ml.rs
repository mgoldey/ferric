use crate::cube::ExportError;
use ndarray::{Array1, Array2, Array3, Array4};
use ndarray_npy::NpzWriter;
use std::fs::File;

/// Atomic partial-charge schemes for the NPZ bundle. Grouping these (rather
/// than flat positional `Option<&[f64]>` params) mirrors `PdepRpaConfig`'s
/// nested-sub-struct precedent (`crates/ferric-rpa/src/config.rs`) and is
/// the natural landing spot for any future charge scheme (CHELPG/RESP/NPA/
/// ...) without growing `export_npz`'s own parameter count again.
#[derive(Debug, Default, Clone, Copy)]
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
#[derive(Debug, Default, Clone, Copy)]
pub struct PolarizabilityBundle<'a> {
    pub esp_atoms: Option<&'a [f64]>,
    /// ESP evaluated at ARBITRARY points (a vdW/solvent-accessible surface,
    /// say), in Hartree atomic units. Must be exported together with
    /// [`Self::esp_points`] — values without their coordinates are unusable,
    /// so the writer rejects one without the other rather than emitting a
    /// half-specified array.
    ///
    /// NOTE this is a DIFFERENT quantity from `esp_atoms`: it includes the
    /// nuclear Z/r term and so diverges at a nucleus, whereas `esp_atoms`
    /// excludes the self-term. See `ferric_scf::properties::esp_at_points`.
    pub esp_surface: Option<&'a [f64]>,
    /// The (N, 3) Cartesian points, in **Bohr**, at which `esp_surface` was
    /// evaluated.
    pub esp_points: Option<&'a Array2<f64>>,
    pub alpha_tensor: Option<&'a [[f64; 3]; 3]>,
    pub electric_field: Option<&'a [[f64; 3]]>,
    pub alpha_atomic: Option<&'a [[[f64; 3]; 3]]>,
}

/// How a per-atom C6/α decomposition was produced: which α(iω) source, and
/// which atomic partition carved the molecular density into atoms.
///
/// This is NOT decoration. A per-atom C6 is a **partition convention**, not a
/// physical observable — Becke and Hirshfeld decompositions of the SAME
/// molecule disagree by up to ~10× on the per-atom magnitudes (see
/// `crates/ferric-rpa/tests/s9_per_atom_c6_consistency.rs`'s
/// `partition_dependence_becke_vs_hirshfeld_water`). A bare per-atom number
/// with no partition attached is therefore not interpretable, which is why
/// [`C6Export`] carries this as a non-`Option` field: the writer cannot emit
/// `c6_iso`/`c6_aniso`/`alpha_atomic_dynamic` without it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C6Provenance<'a> {
    /// Atomic partition, e.g. `"becke"` or `"hirshfeld"`. Mirrors
    /// `ferric_rpa::dispersion::DispersionPartition`; kept as a `&str` here so
    /// `ferric-export` does not take a dependency on `ferric-rpa` (the
    /// dependency runs the other way).
    pub partition: &'a str,
    /// α(iω) source, e.g. `"ts"`, `"pdep"` or `"mbd"`. Mirrors
    /// `ferric_rpa::dispersion::C6Source`. Equally load-bearing: a TS
    /// single-pole per-atom α and a PDEP-RPA one are different objects even
    /// at the same partition.
    pub source: &'a str,
}

/// Dispersion (C6) outputs — per-atom arrays plus the provenance that makes
/// them interpretable and the molecular total that is actually observable.
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
/// charge-transfer/coupling that the molecular response includes).
///
/// The correct DOSD-comparable molecular C6 is [`Self::c6_molecular_iso`],
/// which is now written to the NPZ as the `c6_molecular_iso` scalar — a
/// consumer wanting "the" molecular C6 reads THAT key, and never sums
/// `c6_iso`. See also `docs/dosd-c6-rpa-vs-ts.md`'s "Numerical notes" for the
/// analogous H2 case (6.88 pair-sum vs 9.22 correct).
///
/// STRUCTURAL NOTE (2026-09-16): the per-atom arrays and their
/// [`C6Provenance`] are grouped in ONE struct behind ONE `Option` precisely
/// so a caller cannot supply the arrays and leave the provenance out. An
/// `Option<partition>` sitting beside `Option<c6_iso>` would recreate the
/// untagged-export problem this grouping exists to prevent.
#[derive(Debug, Clone, Copy)]
pub struct C6Export<'a> {
    /// Which partition/source produced the per-atom arrays. MANDATORY.
    pub provenance: C6Provenance<'a>,
    /// Imaginary-frequency quadrature nodes ω_k, a.u.
    pub c6_freqs: &'a [f64],
    /// Casimir-Polder quadrature weights w_k.
    pub c6_weights: &'a [f64],
    /// Per-atom dynamic polarizability α^A_{ij}(iω_k), `[natoms][nfreq]` 3×3.
    /// PARTITION-DEPENDENT — see [`C6Provenance`].
    pub alpha_atomic_dynamic: &'a [Vec<[[f64; 3]; 3]>],
    /// Per-atom-PAIR isotropic C6^{AB}, (N, N). PARTITION-DEPENDENT, and
    /// `c6_iso.sum()` is NOT the molecular C6 — see the struct warning.
    pub c6_iso: &'a Array2<f64>,
    /// Per-atom-PAIR anisotropic C6^{AB}_{ij}, `[N][N]` 3×3.
    /// PARTITION-DEPENDENT.
    pub c6_aniso: &'a [Vec<[[f64; 3]; 3]>],
    /// The molecular isotropic C6 (`C6Result::c6_molecular_iso`), a.u. This
    /// is the DOSD-comparable OBSERVABLE, computed from the global-origin
    /// molecular response — partition-INDEPENDENT, unlike everything else in
    /// this struct.
    pub c6_molecular_iso: f64,
}

/// Dispersion (C6) outputs. All-or-nothing: see [`C6Export`].
#[derive(Debug, Default, Clone, Copy)]
pub struct DispersionBundle<'a> {
    pub c6: Option<C6Export<'a>>,
}

/// Everything `export_npz` can write, grouped by category. See
/// `ChargeSchemes`/`PolarizabilityBundle`/`DispersionBundle` for the
/// per-category fields and their CONSUMER WARNINGs.
#[derive(Debug, Default, Clone, Copy)]
pub struct NpzBundle<'a> {
    pub mo_coeffs: Option<&'a Array2<f64>>,
    pub orbital_energies: Option<&'a [f64]>,
    pub pdep_eigenvectors: Option<&'a Array2<f64>>,
    pub boys_coeffs: Option<&'a Array2<f64>>,
    pub coords: Option<&'a Array2<f64>>,
    pub atomic_numbers: Option<&'a [usize]>,
    pub density_matrix: Option<&'a Array2<f64>>,
    /// Per-MO centroids (n, 3) Bohr — pairs with `orbital_spreads`.
    pub orbital_centers: Option<&'a Array2<f64>>,
    /// Per-MO spatial spreads `sigma = sqrt(<r^2> - |<r>|^2)`, Bohr.
    pub orbital_spreads: Option<&'a [f64]>,
    /// Electronic-density second-moment tensor (3, 3) about the origin.
    pub density_second_moment: Option<&'a Array2<f64>>,
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
        writer
            .add_array("mo_coeffs", c)
            .map_err(|e| ExportError::Other(e.to_string()))?;
    }

    if let Some(e) = bundle.orbital_energies {
        let e_arr = Array1::from_vec(e.to_vec());
        writer
            .add_array("orbital_energies", &e_arr)
            .map_err(|e| ExportError::Other(e.to_string()))?;
    }

    if let Some(v) = bundle.pdep_eigenvectors {
        writer
            .add_array("pdep_eigenvectors", v)
            .map_err(|e| ExportError::Other(e.to_string()))?;
    }

    if let Some(bc) = bundle.boys_coeffs {
        writer
            .add_array("boys_coeffs", bc)
            .map_err(|e| ExportError::Other(e.to_string()))?;
    }

    if let Some(oc) = bundle.orbital_centers {
        writer
            .add_array("orbital_centers", oc)
            .map_err(|e| ExportError::Other(e.to_string()))?;
    }
    if let Some(os) = bundle.orbital_spreads {
        let a = Array1::from_vec(os.to_vec());
        writer
            .add_array("orbital_spreads", &a)
            .map_err(|e| ExportError::Other(e.to_string()))?;
    }
    if let Some(m2) = bundle.density_second_moment {
        writer
            .add_array("density_second_moment", m2)
            .map_err(|e| ExportError::Other(e.to_string()))?;
    }

    if let Some(c) = bundle.coords {
        writer
            .add_array("coords", c)
            .map_err(|e| ExportError::Other(e.to_string()))?;
    }

    if let Some(z) = bundle.atomic_numbers {
        let z_arr = Array1::from_vec(z.iter().map(|&x| x as i64).collect());
        writer
            .add_array("atomic_numbers", &z_arr)
            .map_err(|e| ExportError::Other(e.to_string()))?;
    }

    if let Some(v) = bundle.polarizability.esp_atoms {
        let v_arr = Array1::from_vec(v.to_vec());
        writer
            .add_array("esp_atoms", &v_arr)
            .map_err(|e| ExportError::Other(e.to_string()))?;
    }

    // Surface ESP: values and coordinates travel together or not at all.
    match (
        bundle.polarizability.esp_surface,
        bundle.polarizability.esp_points,
    ) {
        (Some(v), Some(pts)) => {
            if pts.ncols() != 3 || pts.nrows() != v.len() {
                return Err(ExportError::Other(format!(
                    "esp_surface has {} values but esp_points has shape {:?}; \
                     expected ({}, 3)",
                    v.len(),
                    pts.shape(),
                    v.len()
                )));
            }
            let v_arr = Array1::from_vec(v.to_vec());
            writer
                .add_array("esp_surface", &v_arr)
                .map_err(|e| ExportError::Other(e.to_string()))?;
            writer
                .add_array("esp_points", pts)
                .map_err(|e| ExportError::Other(e.to_string()))?;
        }
        (Some(_), None) => {
            return Err(ExportError::Other(
                "esp_surface was supplied without esp_points; the values are \
                 meaningless without the coordinates they were evaluated at"
                    .into(),
            ))
        }
        (None, Some(_)) => {
            return Err(ExportError::Other(
                "esp_points was supplied without esp_surface".into(),
            ))
        }
        (None, None) => {}
    }

    if let Some(a) = bundle.polarizability.alpha_tensor {
        let flat: Vec<f64> = a.iter().flat_map(|row| row.iter().copied()).collect();
        let a_arr = Array2::from_shape_vec((3, 3), flat).unwrap();
        writer
            .add_array("alpha_tensor", &a_arr)
            .map_err(|e| ExportError::Other(e.to_string()))?;
    }

    if let Some(ef) = bundle.polarizability.electric_field {
        let n = ef.len();
        let flat: Vec<f64> = ef.iter().flat_map(|row| row.iter().copied()).collect();
        let ef_arr = Array2::from_shape_vec((n, 3), flat).unwrap();
        writer
            .add_array("electric_field", &ef_arr)
            .map_err(|e| ExportError::Other(e.to_string()))?;
    }

    if let Some(dm) = bundle.density_matrix {
        writer
            .add_array("density_matrix", dm)
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

    // Dispersion: ALL-OR-NOTHING. The per-atom arrays and the provenance that
    // makes them interpretable are written from one `Option`, so an NPZ can
    // never contain an untagged per-atom C6 (see `C6Export`/`C6Provenance`).
    if let Some(c6) = bundle.dispersion.c6 {
        // Provenance strings as UTF-8 byte arrays (`|u1`). `ndarray-npy` 0.9
        // implements `WritableElement` only for numeric primitives and
        // `bool` — there is no numpy-string element type available — so a
        // `u8` array is the one encoding that (a) actually round-trips
        // through this writer and (b) is trivially decodable by the intended
        // Python/numpy consumer:
        //     np.load(f)["c6_partition"].tobytes().decode()  -> "hirshfeld"
        // The alternative (an integer enum code) would require the consumer
        // to carry ferric's discriminant table out-of-band, which is exactly
        // the kind of undocumented convention this tagging exists to remove.
        let part = Array1::from_vec(c6.provenance.partition.as_bytes().to_vec());
        writer
            .add_array("c6_partition", &part)
            .map_err(|e| ExportError::Other(e.to_string()))?;
        let src = Array1::from_vec(c6.provenance.source.as_bytes().to_vec());
        writer
            .add_array("c6_source", &src)
            .map_err(|e| ExportError::Other(e.to_string()))?;

        let a = Array1::from_vec(c6.c6_freqs.to_vec());
        writer
            .add_array("c6_freqs", &a)
            .map_err(|e| ExportError::Other(e.to_string()))?;

        let a = Array1::from_vec(c6.c6_weights.to_vec());
        writer
            .add_array("c6_weights", &a)
            .map_err(|e| ExportError::Other(e.to_string()))?;

        let ad = c6.alpha_atomic_dynamic;
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
        let arr = Array4::from_shape_vec((natoms, nfreq, 3, 3), flat)
            .map_err(|e| ExportError::Other(e.to_string()))?;
        writer
            .add_array("alpha_atomic_dynamic", &arr)
            .map_err(|e| ExportError::Other(e.to_string()))?;

        writer
            .add_array("c6_iso", c6.c6_iso)
            .map_err(|e| ExportError::Other(e.to_string()))?;

        let ca = c6.c6_aniso;
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
        let arr = Array4::from_shape_vec((n, n, 3, 3), flat)
            .map_err(|e| ExportError::Other(e.to_string()))?;
        writer
            .add_array("c6_aniso", &arr)
            .map_err(|e| ExportError::Other(e.to_string()))?;

        // The OBSERVABLE. Written as a length-1 f64 array (npz has no scalar
        // type); `np.load(f)["c6_molecular_iso"][0]` is the molecular C6.
        // This is the key a consumer should read instead of `c6_iso.sum()`.
        let mol_c6 = Array1::from_vec(vec![c6.c6_molecular_iso]);
        writer
            .add_array("c6_molecular_iso", &mol_c6)
            .map_err(|e| ExportError::Other(e.to_string()))?;
    }

    if let Some(mu) = bundle.dipole {
        let mu_arr = Array1::from_vec(mu.to_vec());
        writer
            .add_array("dipole", &mu_arr)
            .map_err(|e| ExportError::Other(e.to_string()))?;
    }

    writer
        .finish()
        .map_err(|e| ExportError::Other(e.to_string()))?;
    Ok(())
}
