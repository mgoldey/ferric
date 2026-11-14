//! `run_pdep_rpa` property outputs: the kwargs that mirror the `[rpa]` TOML
//! keys, the collecting reporter, and the numpy views of the bundle.
//!
//! The computation is `ferric_rpa::property_bundle`, shared with the CLI, so
//! the arrays are the ones the CLI writes to its NPZ.

use super::*;
use ferric_core::basis::BasisSet;
use ferric_rpa::property_bundle::{PdepPropertyRequest, PropertyReporter};
use numpy::PyArrayMethods;

/// Collects the property warnings (raised as `RuntimeWarning`s once the
/// bundle is computed) and drops the CLI's progress lines: the values are on
/// the result.
#[derive(Default)]
pub(crate) struct CollectingReporter {
    pub(crate) warnings: Vec<String>,
}

impl PropertyReporter for CollectingReporter {
    fn info(&mut self, _msg: &str) {}
    fn warn(&mut self, msg: &str) {
        self.warnings.push(msg.to_string());
    }
}

/// The property-related kwargs of `run_pdep_rpa`, named as the `[rpa]` keys.
#[derive(Default)]
pub(crate) struct PropertyKwargs {
    pub(crate) properties: bool,
    pub(crate) export_npz: bool,
    pub(crate) compute_esp: Option<bool>,
    pub(crate) compute_esp_surface: Option<bool>,
    pub(crate) esp_surface_vdw_scale: Option<f64>,
    pub(crate) esp_surface_n_angular: Option<usize>,
    pub(crate) compute_polarizability: Option<bool>,
    pub(crate) compute_electric_field: Option<bool>,
    pub(crate) compute_alpha_atomic: Option<bool>,
    pub(crate) compute_density_matrix: Option<bool>,
    pub(crate) compute_dipole: Option<bool>,
    pub(crate) compute_lowdin_charges: Option<bool>,
    pub(crate) compute_hirshfeld_charges: Option<bool>,
    pub(crate) compute_mulliken_charges: Option<bool>,
    pub(crate) compute_chelpg_charges: Option<bool>,
    pub(crate) compute_resp_charges: Option<bool>,
    pub(crate) compute_c6: Option<bool>,
    pub(crate) c6_source: Option<String>,
    pub(crate) c6_partition: Option<String>,
}

impl PropertyKwargs {
    /// Whether any individual property switch was passed.
    fn any_switch(&self) -> bool {
        self.compute_esp.is_some()
            || self.compute_esp_surface.is_some()
            || self.esp_surface_vdw_scale.is_some()
            || self.esp_surface_n_angular.is_some()
            || self.compute_polarizability.is_some()
            || self.compute_electric_field.is_some()
            || self.compute_alpha_atomic.is_some()
            || self.compute_density_matrix.is_some()
            || self.compute_dipole.is_some()
            || self.compute_lowdin_charges.is_some()
            || self.compute_hirshfeld_charges.is_some()
            || self.compute_mulliken_charges.is_some()
            || self.compute_chelpg_charges.is_some()
            || self.compute_resp_charges.is_some()
            || self.compute_c6.is_some()
            || self.c6_source.is_some()
            || self.c6_partition.is_some()
    }

    /// The request, or `None` when properties are off. Property switches
    /// without `properties=True` / `export_npz=` are an error (they would be
    /// silently ignored); unknown `c6_source` / `c6_partition` strings are a
    /// `ValueError` here, before any SCF runs.
    pub(crate) fn request(&self) -> PyResult<Option<PdepPropertyRequest>> {
        use ferric_rpa::dispersion::{C6Source, DispersionPartition};
        let on = self.properties || self.export_npz;
        if !on {
            if self.any_switch() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "compute_* / esp_surface_* / c6_source / c6_partition select properties \
                     of the property bundle; pass properties=True (or export_npz=...) to \
                     compute it",
                ));
            }
            return Ok(None);
        }
        C6Source::parse_config_str(self.c6_source.as_deref())
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        DispersionPartition::parse_config_str(self.c6_partition.as_deref())
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        let d = PdepPropertyRequest::default();
        Ok(Some(PdepPropertyRequest {
            esp: self.compute_esp.unwrap_or(d.esp),
            esp_surface: self.compute_esp_surface.unwrap_or(d.esp_surface),
            esp_surface_vdw_scale: self
                .esp_surface_vdw_scale
                .unwrap_or(d.esp_surface_vdw_scale),
            esp_surface_n_angular: self
                .esp_surface_n_angular
                .unwrap_or(d.esp_surface_n_angular),
            polarizability: self.compute_polarizability.unwrap_or(d.polarizability),
            electric_field: self.compute_electric_field.unwrap_or(d.electric_field),
            alpha_atomic: self.compute_alpha_atomic.unwrap_or(d.alpha_atomic),
            density_matrix: self.compute_density_matrix.unwrap_or(d.density_matrix),
            dipole: self.compute_dipole.unwrap_or(d.dipole),
            lowdin: self.compute_lowdin_charges.unwrap_or(d.lowdin),
            hirshfeld: self.compute_hirshfeld_charges.unwrap_or(d.hirshfeld),
            mulliken: self.compute_mulliken_charges.unwrap_or(d.mulliken),
            chelpg: self.compute_chelpg_charges.unwrap_or(d.chelpg),
            resp: self.compute_resp_charges.unwrap_or(d.resp),
            c6: self.compute_c6.unwrap_or(d.c6),
            c6_source: self.c6_source.clone(),
            c6_partition: self.c6_partition.clone(),
        }))
    }
}

/// A row-major flattening of nested tensors, reshaped by the caller.
pub(crate) fn flat3(t: &[[f64; 3]; 3]) -> Vec<f64> {
    t.iter().flat_map(|r| r.iter().copied()).collect()
}

/// An array of shape `shape` from `flat` (row-major).
pub(crate) fn nd<'py>(
    py: Python<'py>,
    flat: Vec<f64>,
    shape: &[usize],
) -> PyResult<Bound<'py, PyAny>> {
    Ok(PyArray1::from_vec(py, flat)
        .reshape(shape.to_vec())?
        .into_any())
}

/// The eigenpotential-cube kwargs as `(prefix, spec)`. `export_eigpot_count`,
/// `cube_spacing` and `cube_margin` without a prefix are an error (they would
/// be silently ignored); a non-positive spacing is a `ValueError`.
pub(crate) fn cube_spec(
    prefix: Option<&str>,
    count: Option<usize>,
    spacing: Option<f64>,
    margin: Option<f64>,
) -> PyResult<Option<(String, ferric_rpa::property_bundle::EigpotCubeSpec)>> {
    use ferric_rpa::property_bundle::EigpotCubeSpec;
    let Some(prefix) = prefix else {
        if count.is_some() || spacing.is_some() || margin.is_some() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "export_eigpot_count / cube_spacing / cube_margin need export_eigpot_prefix",
            ));
        }
        return Ok(None);
    };
    let d = EigpotCubeSpec::default();
    let spec = EigpotCubeSpec {
        spacing: spacing.unwrap_or(d.spacing),
        margin: margin.unwrap_or(d.margin),
        count: count.unwrap_or(d.count),
    };
    if !(spec.spacing > 0.0 && spec.spacing.is_finite()) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "cube_spacing must be positive, got {}",
            spec.spacing
        )));
    }
    Ok(Some((prefix.to_string(), spec)))
}

/// Everything the bundle / export step reads from `run_pdep_rpa`.
pub(crate) struct BundleJob<'a> {
    pub(crate) ctx: &'a ParallelContext,
    pub(crate) mol: &'a Molecule,
    pub(crate) bs: &'a BasisSet,
    pub(crate) aux_bs: &'a BasisSet,
    pub(crate) prep: &'a PreparedBasis,
    pub(crate) dfbs: &'a PreparedBasis,
    pub(crate) rhf: &'a ScfResult,
    pub(crate) op: Operator,
    pub(crate) rpa_cfg: &'a ferric_rpa::PdepRpaConfig,
    pub(crate) scf_config: &'a RhfConfig,
    pub(crate) request: Option<PdepPropertyRequest>,
    pub(crate) export_npz: Option<&'a str>,
    pub(crate) allow_partial_npz: bool,
    pub(crate) cubes: Option<(String, ferric_rpa::property_bundle::EigpotCubeSpec)>,
}

/// What `run_pdep_rpa` attaches to its result beyond the energies.
#[derive(Default)]
pub(crate) struct BundleOutput {
    pub(crate) props: Option<ferric_rpa::property_bundle::PdepProperties>,
    pub(crate) warnings: Vec<String>,
    pub(crate) written: Vec<String>,
    pub(crate) density_matrix: Option<Array2<f64>>,
    pub(crate) mo_coeffs: Option<Array2<f64>>,
    pub(crate) orbital_energies: Option<Vec<f64>>,
}

/// Compute the property bundle, write the eigenpotential cubes and the NPZ.
///
/// The NPZ is written even when a requested property failed (a partial bundle
/// is salvageable); the call then raises `RuntimeError` listing the gaps
/// unless `allow_partial_npz=True`, the Python form of the CLI's nonzero exit.
pub(crate) fn run_property_bundle(
    py: Python<'_>,
    job: &BundleJob,
    rpa_result: &ferric_rpa::PdepRpaResult,
    cert_json: &str,
) -> PyResult<BundleOutput> {
    use ferric_rpa::property_bundle::{
        compute_pdep_properties, export_eigenpotential_cubes, write_pdep_npz, PropertyInputs,
    };
    let mut rep = CollectingReporter::default();
    let mut out = BundleOutput::default();
    if let Some((prefix, spec)) = job.cubes.as_ref() {
        out.written = export_eigenpotential_cubes(
            prefix,
            spec,
            job.mol,
            &job.aux_bs.name,
            job.aux_bs,
            rpa_result,
            &mut rep,
        );
    }
    for w in &rep.warnings {
        crate::scf::emit_runtime_warning(py, w.clone());
    }
    let n_cube_warnings = rep.warnings.len();
    if let Some(request) = job.request.as_ref() {
        let proatom =
            ferric_scf::properties::scf_proatom_provider(job.ctx, job.bs, job.op, job.scf_config);
        let inputs = PropertyInputs {
            ctx: job.ctx,
            mol: job.mol,
            prep: job.prep,
            bs: job.bs,
            dfbs: job.dfbs,
            result: job.rhf,
            op: job.op,
            rpa_cfg: job.rpa_cfg,
            rhf_config: job.scf_config,
            proatom: &proatom,
        };
        let props = compute_pdep_properties(&inputs, request, &mut rep)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        for w in &rep.warnings[n_cube_warnings..] {
            crate::scf::emit_runtime_warning(py, w.clone());
        }
        let mut write_err = None;
        if let Some(path) = job.export_npz {
            match write_pdep_npz(path, &inputs, &props, rpa_result, cert_json) {
                Ok(()) => out.written.push(path.to_string()),
                Err(e) => write_err = Some(format!("failed to write {path}: {e}")),
            }
        }
        if job.export_npz.is_some() {
            check_complete(&props.gaps, write_err, job.allow_partial_npz)?;
        }
        out.density_matrix = props
            .density_matrix
            .then(|| job.rhf.density_total().clone());
        if job.rhf.spin == ferric_scf::result::Spin::Restricted {
            out.mo_coeffs = Some(job.rhf.mos_r().clone());
            out.orbital_energies = Some(job.rhf.eps_r().to_vec());
        }
        out.props = Some(props);
    }
    out.warnings = rep.warnings;
    Ok(out)
}

/// Refuse to report success on a bundle missing something the caller asked
/// for (only when it was exported: a returned object carries `property_gaps`).
fn check_complete(gaps: &[String], write_err: Option<String>, allow_partial: bool) -> PyResult<()> {
    if let Some(e) = write_err {
        return Err(make_err(e));
    }
    if gaps.is_empty() || allow_partial {
        return Ok(());
    }
    Err(make_err(format!(
        "the NPZ bundle is incomplete -- {} requested propert{} failed: {}; \
         pass allow_partial_npz=True to accept the gaps",
        gaps.len(),
        if gaps.len() == 1 { "y" } else { "ies" },
        gaps.join("; ")
    )))
}
