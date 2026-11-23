//! RPA bindings: moved verbatim from `lib.rs`.

use super::*;
use crate::pdep_properties::{
    cube_spec, flat3, nd, run_property_bundle, BundleJob, PropertyKwargs,
};

// ── RS-MP2-RPA (SR-MP2 + LR-dRPA, Δ-form B or coupled-rings T) ──

#[pyclass(frozen)]
#[pyo3(name = "RsMp2RpaResult")]
pub(crate) struct PyRsMp2RpaResult {
    #[pyo3(get)]
    pub(crate) total_energy: f64,
    /// The reference SCF: "RHF" for a singlet, "UHF" for multiplicity > 1
    /// (then `rhf_energy` is the UHF energy and `e_dmp2_lr` is
    /// `E_ab + (D_aa + D_bb)/2`, the second-order term of the U-dRPA).
    #[pyo3(get)]
    pub(crate) reference: String,
    #[pyo3(get)]
    pub(crate) rhf_energy: f64,
    #[pyo3(get)]
    pub(crate) e_corr: f64,
    /// Diagnostic naive sum E_MP2[erfc] + E_dRPA[erf] (formulation A).
    /// Only available when `formulation="delta-lr"`; None for coupled-rings.
    #[pyo3(get)]
    pub(crate) e_corr_naive: Option<f64>,
    #[pyo3(get)]
    pub(crate) e_mp2_full: f64,
    #[pyo3(get)]
    pub(crate) e_sr_mp2: f64,
    #[pyo3(get)]
    pub(crate) e_lr_mp2: f64,
    #[pyo3(get)]
    pub(crate) e_dmp2_lr: f64,
    /// E_dRPA[erf] (DeltaLr only; None for CoupledRings).
    #[pyo3(get)]
    pub(crate) e_drpa_lr: Option<f64>,
    /// ΔdRPA[Coulomb] = E_dRPA[Coulomb] − 2·E_OS[Coulomb] (CoupledRings only).
    #[pyo3(get)]
    pub(crate) e_delta_drpa_full: Option<f64>,
    /// ΔdRPA[erfc] = E_dRPA[erfc] − 2·E_OS[erfc] (CoupledRings only).
    #[pyo3(get)]
    pub(crate) e_delta_drpa_sr: Option<f64>,
    /// How the reference SCF built J/K: "exact", "rijk" or "rijcosx".
    #[pyo3(get)]
    pub(crate) scf_jk: String,
    /// The printout's SCF J/K echo (`ScfJk::label`: names a non-default
    /// exchange builder on the exact path).
    pub(crate) jk_label: String,
    /// The SCF's JK-fitting basis (`None` for "exact").
    #[pyo3(get)]
    pub(crate) jk_auxbasis: Option<String>,
    pub(crate) certificate_data: ferric_core::certificate::Certificate,
}

stored_method_result!(PyRsMp2RpaResult, total_energy: total_energy, certificate: certificate_data);

#[pymethods]
impl PyRsMp2RpaResult {
    #[doc = method_certificate_doc!()]
    #[getter]
    fn certificate(&self, py: Python<'_>) -> PyResult<PyObject> {
        crate::method_result::certificate_getter(py, self)
    }
    fn __repr__(&self) -> String {
        format!(
            "RsMp2RpaResult(total_energy={:.10}, e_corr={:.10})",
            self.total_energy, self.e_corr,
        )
    }
    fn __str__(&self) -> String {
        format!(
            "RS-MP2-RPA Total Energy: {:.10} Ha (RHF: {:.10}, corr: {:.10}, SR-MP2: {:.10}, LR-MP2: {:.10}) [SCF J/K: {}]",
            self.total_energy, self.rhf_energy, self.e_corr, self.e_sr_mp2, self.e_lr_mp2,
            self.jk_label,
        )
    }
}

/// SR-MP2 + LR-dRPA on an RHF reference, or a UHF one for multiplicity > 1
/// (spin-summed U-MP2 and U-dRPA) (`formulation` "delta-lr" or
/// "coupled-rings"; `attenuator` "erf" with `omega` in A^-1, or "terf" with
/// `r0` in A).
///
/// `scf_jk` / `jk_auxbasis` work as in `run_rimp2`, BUT THE DEFAULT DIFFERS:
/// this entry point has always used RI-JK, so `scf_jk` defaults to "rijk"
/// (on "def2-universal-jkfit") here, not "exact". Pass `scf_jk="exact"` for
/// four-centre J/K. A `k_builder` under "rijk" is a ValueError (it used to
/// be ignored with a warning). The mode actually used is echoed as
/// `result.scf_jk` / `result.jk_auxbasis`.
///
/// The analytic gradient of `E_RHF + E_c^RPA` on the run's own converged RHF
/// (full rank, dense, Coulomb: `refuse_pdep_rpa_gradient` has screened the
/// settings), the function behind the CLI's pdep-rpa `task = "optimize"`.
fn pdep_rpa_total_gradient(
    mol: &ferric_core::mol::Molecule,
    prep: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    rhf: &ferric_scf::result::ScfResult,
    cfg: &ferric_rpa::PdepRpaConfig,
) -> Result<Array2<f64>, ferric_core::FerricError> {
    ferric_rpa::gradient_analytic::rpa_analytic_gradient_on(
        mol,
        prep,
        dfbs,
        op,
        bounds,
        rhf,
        cfg,
        ferric_rpa::gradient_analytic::RingOrder::Full,
    )
    .map(|g| g.gradient)
}

/// What `run_pdep_rpa(with_gradient=True)` must refuse: every setting that
/// makes the reported energy something other than the full-rank dense
/// Coulomb RI-dRPA on an exact closed-shell RHF reference, the one
/// `ferric_rpa::gradient_analytic` differentiates.
struct GradientScope<'a> {
    multiplicity: usize,
    trunc_thresh: Option<f64>,
    chi0_sparsity: Option<&'a str>,
    k_builder: Option<&'a str>,
    /// `df_j_aux` / `df_k_aux` given (an RI-JK reference).
    df_aux: bool,
    /// Point charges, an external field or a solvent in the reference SCF.
    environment: bool,
    /// A property bundle, an NPZ export or eigenpotential cubes.
    properties: bool,
    run_diagnostics: bool,
}

/// Raise `ValueError` for any setting the analytic RPA gradient does not
/// differentiate, before any integral is computed. The list is
/// `gradient_analytic::check_analytic_scope` (full rank, dense chi0, dense
/// sparsity, Coulomb) plus the reference-SCF settings the gradient does not
/// see (it takes none of the external potential, PCM or COSX terms).
fn refuse_pdep_rpa_gradient(s: &GradientScope<'_>) -> PyResult<()> {
    let refuse = |what: String| {
        Err(crate::errors::input_err(format!(
            "run_pdep_rpa: with_gradient=True {what}"
        )))
    };
    if s.multiplicity > 1 {
        return refuse(format!(
            "is closed-shell only (there is no open-shell RPA gradient), but the molecule has \
             multiplicity {}",
            s.multiplicity
        ));
    }
    if let Some(t) = s.trunc_thresh {
        if t != 0.0 {
            return refuse(format!(
                "differentiates the full-rank energy, but trunc_thresh={t} truncates the PDEP \
                 basis: omit trunc_thresh (it is 0 with with_gradient) or pass 0"
            ));
        }
    }
    if !matches!(
        ferric_rpa::config::Chi0Sparsity::parse_config_str(s.chi0_sparsity),
        Ok(ferric_rpa::config::Chi0Sparsity::Dense)
    ) {
        return refuse("differentiates dense chi0 only: omit chi0_sparsity".into());
    }
    // The spelling as passed (the SCF config parses it after this check).
    if s.k_builder
        .is_some_and(ferric_scf::options::KBuilderKind::spells_cosx)
    {
        return refuse("has no COSX exchange response: use an exact k_builder or omit it".into());
    }
    if s.df_aux {
        return refuse(
            "differentiates the exact four-centre J/K reference: omit df_j_aux and df_k_aux".into(),
        );
    }
    if s.environment {
        return refuse(
            "does not differentiate the point-charge / field / solvent terms of the reference: \
             remove point_charges, external_field and solvent"
                .into(),
        );
    }
    if s.properties {
        return refuse(
            "returns the gradient only: remove properties, export_npz, compute_* and \
             export_eigpot_*"
                .into(),
        );
    }
    if s.run_diagnostics {
        return refuse("does not run the diagnostics pass: remove run_diagnostics".into());
    }
    Ok(())
}

/// `scf_max_iter`, `scf_energy_conv`, `scf_density_conv`: controls of the internal reference SCF,
/// with the names and meaning of `run_rhf` (defaults unchanged when omitted).
///
/// With an ECP basis set the ECP is applied to the molecule, so the reference SCF is the one
/// `run_rhf` (`run_uhf` for an open-shell molecule) solves.
/// `frozen_core` counts orbitals of the valence occupied space, after the ECP has removed its core.
#[pyfunction]
#[pyo3(signature = (mol, basis_set, auxbasis, omega=None, frozen_core=None, k_builder=None, formulation=None, attenuator=None, r0=None, terf_omega=None, memory_budget_gb=None, scf_jk=None, jk_auxbasis=None, scf_ri_j_precision=None, scf_max_iter=None, scf_energy_conv=None, scf_density_conv=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_rs_mp2_rpa(
    py: Python<'_>,
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    auxbasis: &PyBasisSet,
    omega: Option<&Bound<'_, PyAny>>,
    frozen_core: Option<usize>,
    k_builder: Option<&str>,
    formulation: Option<&Bound<'_, PyAny>>,
    attenuator: Option<&str>,
    r0: Option<f64>,
    terf_omega: Option<&Bound<'_, PyAny>>,
    memory_budget_gb: Option<f64>,
    scf_jk: Option<&str>,
    jk_auxbasis: Option<&str>,
    scf_ri_j_precision: Option<&str>,
    scf_max_iter: Option<usize>,
    scf_energy_conv: Option<f64>,
    scf_density_conv: Option<f64>,
) -> PyResult<PyObject> {
    // The default differs from the other MP2 entry points: this one has
    // always used RI-JK on def2-universal-jkfit (same convention as pdep-rpa).
    let (cfg_rhf, jk) = mp2_reference_config(
        "run_rs_mp2_rpa",
        k_builder,
        memory_budget_gb,
        scf_jk,
        jk_auxbasis,
        ferric_scf::scf_jk::ScfJkMode::RiJk,
        ScfConv::new(scf_max_iter, scf_energy_conv, scf_density_conv),
        scf_ri_j_precision,
    )?;
    let sys = prepare(py, mol, basis_set)?;
    let dfbs = prepare_aux(py, &sys, auxbasis)?;
    let rhf = sys.solve_reference(&cfg_rhf).map_err(make_err)?;
    let (mol, prep) = (sys.mol(), sys.basis());
    let (forms, forms_seq) = rs_scan_axis::<String>(formulation, "formulation")?;
    let forms = forms
        .unwrap_or_else(|| vec!["delta-lr".to_string()])
        .iter()
        .map(|f| match f.as_str() {
            "delta-lr" => Ok(ferric_rpa::RsMp2RpaFormulation::DeltaLr),
            "coupled-rings" => Ok(ferric_rpa::RsMp2RpaFormulation::CoupledRings),
            other => Err(crate::errors::input_err(format!(
                "unknown formulation \"{other}\"; expected \"delta-lr\" or \"coupled-rings\""
            ))),
        })
        .collect::<PyResult<Vec<_>>>()?;
    // attenuator: "erf" (default, ω in Å⁻¹) or "terf" (r0 in Å, ω derived).
    let atten = match attenuator.unwrap_or("erf") {
        "erf" => ferric_rpa::rs_mp2_rpa::Attenuator::Erf,
        "terf" => ferric_rpa::rs_mp2_rpa::Attenuator::Terf,
        other => {
            return Err(crate::errors::input_err(format!(
                "unknown attenuator \"{other}\"; expected \"erf\" or \"terf\""
            )))
        }
    };
    let terf = matches!(atten, ferric_rpa::rs_mp2_rpa::Attenuator::Terf);
    // The scan axis is `omega` for erf and `terf_omega` for terf; a list on
    // the other one has no meaning and is refused.
    let (omegas, omega_seq) = rs_scan_axis::<f64>(omega, "omega")?;
    let (terf_omegas, terf_seq) = rs_scan_axis::<f64>(terf_omega, "terf_omega")?;
    if omega_seq && terf {
        return Err(crate::errors::input_err(
            "omega is ignored with attenuator=\"terf\"; scan the seam sharpness with a list-valued terf_omega",
        ));
    }
    if terf_seq && !terf {
        return Err(crate::errors::input_err(
            "terf_omega needs attenuator=\"terf\"; scan an erf split with a list-valued omega",
        ));
    }
    let is_scan = forms_seq || omega_seq || terf_seq;
    for (name, v) in [("omega", &omegas), ("terf_omega", &terf_omegas)] {
        if let Some(w) = v
            .as_ref()
            .and_then(|v| v.iter().find(|w| !(w.is_finite() && **w > 0.0)))
        {
            return Err(crate::errors::input_err(format!(
                "{name} must be finite and > 0 (got {w})"
            )));
        }
    }
    if forms
        .iter()
        .enumerate()
        .any(|(i, f)| forms[..i].contains(f))
    {
        return Err(crate::errors::input_err(
            "formulation lists the same formulation twice",
        ));
    }
    // omega is supplied in Å⁻¹; convert to Bohr⁻¹ for the operator. r0 is
    // supplied in Å and converted to Bohr for RsMp2RpaConfig (Bohr-native).
    const ANG2BOHR_R0: f64 = ANGSTROM_TO_BOHR;
    let to_bohr = ferric_mp2::attenuated::BOHR_INV_PER_ANG_INV;
    let mut cfg = ferric_rpa::RsMp2RpaConfig {
        omega: omegas.as_ref().map_or(0.420, |v| v[0]) * to_bohr,
        attenuator: atten,
        r0: r0.unwrap_or(3.18 / ANG2BOHR_R0) * ANG2BOHR_R0,
        // terf_omega in Å⁻¹ (same convention as omega); None keeps the
        // Dutoi curvature link ω = 1/(r0·√2). Terf-only.
        terf_omega: terf_omegas.as_ref().map(|v| v[0] * to_bohr),
        frozen_core: frozen_core.unwrap_or(0),
        formulation: forms[0],
        ..Default::default()
    };
    cfg.drpa.memory_budget_bytes = budget_bytes_from_gb(memory_budget_gb);
    // One (config, result) per (point, formulation), point-major.
    let mut points: Vec<(ferric_rpa::RsMp2RpaConfig, ferric_rpa::RsMp2RpaResult)> = Vec::new();
    if is_scan {
        let axis: Vec<f64> = if terf {
            terf_omegas.clone().ok_or_else(|| {
                crate::errors::input_err(
                    "scanning formulations with attenuator=\"terf\" needs terf_omega",
                )
            })?
        } else {
            omegas.clone().unwrap_or_else(|| vec![0.420])
        };
        let bohr: Vec<f64> = axis.iter().map(|w| w * to_bohr).collect();
        let rs = ferric_rpa::rs_mp2_lr_rpa_scan(mol, prep, &dfbs, &rhf, &cfg, &bohr, &forms)
            .map_err(crate::make_err)?;
        for (k, r) in rs.into_iter().enumerate() {
            let mut c = cfg.clone();
            c.formulation = forms[k % forms.len()];
            if terf {
                c.terf_omega = Some(bohr[k / forms.len()]);
            } else {
                c.omega = bohr[k / forms.len()];
            }
            points.push((c, r));
        }
    } else {
        let r = ferric_rpa::rs_mp2_lr_rpa(mol, prep, &dfbs, &rhf, &cfg).map_err(crate::make_err)?;
        points.push((cfg.clone(), r));
    }
    let base_cert = reference_certificate(mol, &rhf, &cfg_rhf, &basis_set.inner);
    let mut out = Vec::with_capacity(points.len());
    for (c, r) in points {
        let certificate_data =
            ferric_rpa::certificate::rs_mp2_rpa_certificate(&base_cert, &auxbasis.inner.name, &c);
        out.push(PyRsMp2RpaResult {
            total_energy: r.total_energy,
            reference: crate::mp2::reference_name(&rhf),
            rhf_energy: rhf.energy,
            e_corr: r.e_corr,
            e_corr_naive: r.e_corr_naive,
            e_mp2_full: r.e_mp2_full,
            e_sr_mp2: r.e_sr_mp2,
            e_lr_mp2: r.e_lr_mp2,
            e_dmp2_lr: r.e_dmp2_lr,
            e_drpa_lr: r.e_drpa_lr,
            e_delta_drpa_full: r.e_delta_drpa_full,
            e_delta_drpa_sr: r.e_delta_drpa_sr,
            scf_jk: jk.mode.as_str().to_string(),
            jk_label: jk.label(cfg_rhf.k_builder),
            jk_auxbasis: jk.aux.clone(),
            certificate_data,
        });
    }
    if is_scan {
        Ok(out
            .into_iter()
            .map(|r| Py::new(py, r))
            .collect::<PyResult<Vec<_>>>()?
            .into_pyobject(py)?
            .into_any()
            .unbind())
    } else {
        let r = out.pop().expect("one result");
        Ok(Py::new(py, r)?.into_any())
    }
}

/// A scalar-or-sequence keyword: `(values, was_sequence)`; `None` is absent.
/// A sequence must be non-empty.
fn rs_scan_axis<'py, T: FromPyObject<'py>>(
    obj: Option<&Bound<'py, PyAny>>,
    name: &str,
) -> PyResult<(Option<Vec<T>>, bool)> {
    let Some(o) = obj.filter(|o| !o.is_none()) else {
        return Ok((None, false));
    };
    if let Ok(x) = o.extract::<T>() {
        return Ok((Some(vec![x]), false));
    }
    let v: Vec<T> = o.extract().map_err(|_| {
        crate::errors::input_err(format!(
            "{name} must be a scalar or a sequence of the same type"
        ))
    })?;
    if v.is_empty() {
        return Err(crate::errors::input_err(format!(
            "{name} must not be an empty list"
        )));
    }
    Ok((Some(v), true))
}

// ── PDEP-RPA ──

#[pyclass(frozen)]
#[pyo3(name = "PdepRpaResult")]
pub(crate) struct PyPdepRpaResult {
    #[pyo3(get)]
    pub(crate) rhf_energy: f64,
    #[pyo3(get)]
    pub(crate) e_rpa: f64,
    #[pyo3(get)]
    pub(crate) total_energy: f64,
    #[pyo3(get)]
    pub(crate) n_eigenpotentials: usize,
    #[pyo3(get)]
    pub(crate) e_rpa_dft_diag: Option<f64>,
    /// Whether the static-dielectric eigensolve (Davidson or Lanczos) met its
    /// residual-norm convergence tolerance. `false` means `eigenvalues_static`
    /// / `eigenpotentials` are the eigensolver's best-effort Ritz pairs after
    /// exhausting its iteration budget, not verified eigenpairs.
    #[pyo3(get)]
    pub(crate) eigensolver_converged: bool,
    pub(crate) eigenvalues_static: Vec<f64>,
    pub(crate) eigenpotentials: Array2<f64>,
    pub(crate) quad_freqs: Vec<f64>,
    pub(crate) quad_weights: Vec<f64>,
    pub(crate) eigenvalues_freq: Array2<f64>,
    pub(crate) certificate_data: ferric_core::certificate::Certificate,
    /// The property bundle (`properties=True` / `export_npz=`), else `None`.
    pub(crate) props: Option<ferric_rpa::property_bundle::PdepProperties>,
    /// Warnings of the property computation (also printed to stderr).
    pub(crate) property_warnings: Vec<String>,
    /// Paths of the files `export_npz` / `export_eigpot_prefix` wrote.
    pub(crate) written_files: Vec<String>,
    /// Reference AO density / MO coefficients / orbital energies, kept with the
    /// property bundle (the NPZ's `density_matrix`, `mo_coeffs`,
    /// `orbital_energies`).
    pub(crate) density_matrix: Option<Array2<f64>>,
    pub(crate) mo_coeffs: Option<Array2<f64>>,
    pub(crate) orbital_energies: Option<Vec<f64>>,
    /// `dE_total/dR` (natoms x 3, Hartree/Bohr) when solved with `with_gradient=True`.
    pub(crate) gradient_data: Option<Array2<f64>>,
}

/// Getter body: `Some(array)` from an optional property, else `None`.
macro_rules! prop_array {
    ($self:ident, $py:ident, $field:ident, |$v:ident| $flat:expr, $shape:expr) => {
        match $self.props.as_ref().and_then(|p| p.$field.as_ref()) {
            Some($v) => Ok(Some(nd($py, $flat, &$shape)?)),
            None => Ok(None),
        }
    };
}

stored_method_result!(PyPdepRpaResult, total_energy: total_energy, certificate: certificate_data);

#[pymethods]
impl PyPdepRpaResult {
    #[doc = method_certificate_doc!()]
    #[getter]
    fn certificate(&self, py: Python<'_>) -> PyResult<PyObject> {
        crate::method_result::certificate_getter(py, self)
    }
    /// Analytic nuclear gradient of `total_energy` (RHF + RI-dRPA) as an
    /// (natoms, 3) array in Hartree/Bohr, rows in the Molecule's atom order;
    /// `None` unless the result came from `run_pdep_rpa(..., with_gradient=True)`.
    fn gradient<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
        self.gradient_data
            .as_ref()
            .map(|g| PyArray2::from_array(py, g))
    }
    /// Static dielectric eigenvalues λ_α(0), length M (sorted descending).
    /// Plot `λ - 1` vs α on a log scale for a scree plot of the PDEP basis.
    #[getter]
    fn eigenvalues_static<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.eigenvalues_static)
    }
    /// PDEP eigenpotential coefficients in the RI auxiliary basis, shape (naux, M).
    /// Column α gives the density ρ_α(r) = Σ_P c_α^P χ_P(r), whose Coulomb
    /// potential is the eigenpotential (see `response_modes`). Orthonormal in
    /// the aux metric V = (P|Q) (cᵀVc = I), with cᵀ(V+Π)c = diag(λ_α(0)).
    #[getter]
    fn eigenpotentials<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        PyArray2::from_owned_array(py, self.eigenpotentials.clone())
    }
    /// Imaginary-frequency quadrature points ω_k.
    #[getter]
    fn quad_freqs<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.quad_freqs)
    }
    /// Imaginary-frequency quadrature weights w_k.
    #[getter]
    fn quad_weights<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.quad_weights)
    }
    /// λ_α(iω_k) tensor, shape (N_quad, M).
    #[getter]
    fn eigenvalues_freq<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        PyArray2::from_owned_array(py, self.eigenvalues_freq.clone())
    }

    // ── property bundle (run_pdep_rpa(properties=True) / export_npz=) ──
    // Names are the NPZ keys. `None` = not requested, or failed (see
    // `property_gaps`).

    /// Reference AO density matrix, shape (nao, nao) (property bundle only,
    /// unless `compute_density_matrix=False`).
    #[getter]
    fn density_matrix<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
        self.density_matrix
            .as_ref()
            .map(|d| PyArray2::from_owned_array(py, d.clone()))
    }
    /// Reference MO coefficients, shape (nao, nmo) (property bundle only).
    #[getter]
    fn mo_coeffs<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
        self.mo_coeffs
            .as_ref()
            .map(|d| PyArray2::from_owned_array(py, d.clone()))
    }
    /// Reference orbital energies, shape (nmo,), Hartree (property bundle only).
    #[getter]
    fn orbital_energies<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.orbital_energies
            .as_ref()
            .map(|e| PyArray1::from_slice(py, e))
    }
    /// Requested properties that failed to compute (empty if none or if the
    /// bundle was not requested).
    #[getter]
    fn property_gaps(&self) -> Vec<String> {
        self.props
            .as_ref()
            .map(|p| p.gaps.clone())
            .unwrap_or_default()
    }
    /// Warnings raised while computing the property bundle (TS C6 skipped,
    /// a property failed, ...), as printed to stderr.
    #[getter]
    fn property_warnings(&self) -> Vec<String> {
        self.property_warnings.clone()
    }
    /// Files written by `export_npz=` and `export_eigpot_prefix=`.
    #[getter]
    fn written_files(&self) -> Vec<String> {
        self.written_files.clone()
    }
    /// ESP at the nuclei, shape (N,), a.u. (own nucleus excluded).
    #[getter]
    fn esp_atoms<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        prop_array!(self, py, esp_atoms, |v| v.clone(), [v.len()])
    }
    /// ESP on the vdW surface shell (`compute_esp_surface=True`), shape (npts,).
    #[getter]
    fn esp_surface<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        prop_array!(self, py, esp_surface, |v| v.clone(), [v.len()])
    }
    /// Points of `esp_surface`, shape (npts, 3), Bohr.
    #[getter]
    fn esp_points<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        prop_array!(
            self,
            py,
            esp_points,
            |v| v.iter().copied().collect(),
            [v.nrows(), 3]
        )
    }
    /// Electric field at the nuclei, shape (N, 3), a.u.
    #[getter]
    fn electric_field<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        prop_array!(
            self,
            py,
            electric_field,
            |v| v.iter().flatten().copied().collect(),
            [v.len(), 3]
        )
    }
    /// Molecular static PDEP-RPA polarizability, shape (3, 3), a.u.
    #[getter]
    fn alpha_tensor<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        prop_array!(self, py, alpha_tensor, |v| flat3(v), [3, 3])
    }
    /// Per-atom intrinsic Becke polarizability, shape (N, 3, 3), a.u. Charge
    /// transfer excluded; `alpha_ct` is the remainder.
    #[getter]
    fn alpha_atomic<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        prop_array!(
            self,
            py,
            alpha_atomic,
            |v| v.iter().flat_map(flat3).collect(),
            [v.len(), 3, 3]
        )
    }
    /// Charge-transfer remainder `alpha_tensor - alpha_atomic.sum(0)`, (3, 3).
    #[getter]
    fn alpha_ct<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        prop_array!(self, py, alpha_ct, |v| flat3(v), [3, 3])
    }
    /// Molecular dipole, shape (3,), a.u.
    #[getter]
    fn dipole<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        prop_array!(self, py, dipole, |v| v.to_vec(), [3])
    }
    /// Löwdin charges, shape (N,).
    #[getter]
    fn lowdin_charges<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        prop_array!(self, py, lowdin, |v| v.clone(), [v.len()])
    }
    /// Hirshfeld charges (same-basis free-atom SCF proatoms), shape (N,).
    #[getter]
    fn hirshfeld_charges<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        prop_array!(self, py, hirshfeld, |v| v.clone(), [v.len()])
    }
    /// Mulliken charges, shape (N,).
    #[getter]
    fn mulliken_charges<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        prop_array!(self, py, mulliken, |v| v.clone(), [v.len()])
    }
    /// CHELPG charges, shape (N,).
    #[getter]
    fn chelpg_charges<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        prop_array!(self, py, chelpg, |v| v.clone(), [v.len()])
    }
    /// RESP charges, shape (N,).
    #[getter]
    fn resp_charges<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        prop_array!(self, py, resp, |v| v.clone(), [v.len()])
    }
    /// Orbital centroids, shape (nmo, 3), Bohr.
    #[getter]
    fn orbital_centers<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        prop_array!(
            self,
            py,
            orbital_centers,
            |v| v.iter().copied().collect(),
            [v.nrows(), 3]
        )
    }
    /// Orbital spreads, shape (nmo,), Bohr.
    #[getter]
    fn orbital_spreads<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        prop_array!(self, py, orbital_spreads, |v| v.clone(), [v.len()])
    }
    /// Electronic-density second moment about the origin, shape (3, 3).
    #[getter]
    fn density_second_moment<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        prop_array!(
            self,
            py,
            density_second_moment,
            |v| v.iter().copied().collect(),
            [3, 3]
        )
    }
    /// Imaginary-frequency nodes of the C6 quadrature, shape (nfreq,).
    #[getter]
    fn c6_freqs<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        prop_array!(self, py, c6, |c| c.freqs.clone(), [c.freqs.len()])
    }
    /// Quadrature weights matching `c6_freqs`.
    #[getter]
    fn c6_weights<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        prop_array!(self, py, c6, |c| c.weights.clone(), [c.weights.len()])
    }
    /// Per-atom-PAIR isotropic C6, shape (N, N). A partition convention (see
    /// `c6_partition`); `c6_iso.sum()` is NOT the molecular C6.
    #[getter]
    fn c6_iso<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        prop_array!(
            self,
            py,
            c6,
            |c| c.c6_iso.iter().copied().collect(),
            [c.c6_iso.nrows(), c.c6_iso.ncols()]
        )
    }
    /// Per-atom-pair anisotropic C6, shape (N, N, 3, 3).
    #[getter]
    fn c6_aniso<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        prop_array!(
            self,
            py,
            c6,
            |c| c.c6_aniso.iter().flatten().flat_map(flat3).collect(),
            [c.c6_aniso.len(), c.c6_aniso.len(), 3, 3]
        )
    }
    /// Per-atom dynamic polarizability α^A(iω), shape (N, nfreq, 3, 3).
    #[getter]
    fn alpha_atomic_dynamic<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        prop_array!(
            self,
            py,
            c6,
            |c| c
                .alpha_atomic_dynamic
                .iter()
                .flatten()
                .flat_map(flat3)
                .collect(),
            [c.alpha_atomic_dynamic.len(), c.freqs.len(), 3, 3]
        )
    }
    /// Dynamic charge-transfer remainder, shape (nfreq, 3, 3) (`c6_source="pdep"` only).
    #[getter]
    fn alpha_ct_dynamic<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        match self
            .props
            .as_ref()
            .and_then(|p| p.c6.as_ref())
            .and_then(|c| c.alpha_ct_dynamic.as_ref())
        {
            Some(v) => Ok(Some(nd(
                py,
                v.iter().flat_map(flat3).collect(),
                &[v.len(), 3, 3],
            )?)),
            None => Ok(None),
        }
    }
    /// Molecular isotropic C6 (a.u.): the DOSD-comparable observable,
    /// partition-independent.
    #[getter]
    fn c6_molecular_iso(&self) -> Option<f64> {
        self.props.as_ref()?.c6.as_ref().map(|c| c.c6_molecular_iso)
    }
    /// Atomic partition behind the per-atom C6 arrays ("becke" / "hirshfeld").
    #[getter]
    fn c6_partition(&self) -> Option<&'static str> {
        self.props.as_ref()?.c6.as_ref().map(|c| c.partition)
    }
    /// α(iω) source behind the per-atom C6 arrays ("ts" / "pdep" / "mbd").
    #[getter]
    fn c6_source(&self) -> Option<&'static str> {
        self.props.as_ref()?.c6.as_ref().map(|c| c.source)
    }

    /// Write a scree plot of the static dielectric eigenvalues to `path` (PNG).
    ///
    /// Plots `|λ_α(0) - 1|` on a log y-axis against eigenpotential index α.
    /// The shape of this curve diagnoses how aggressively PDEP can be truncated.
    /// Pure-Python implementation: dispatches to matplotlib via the GIL.
    #[pyo3(signature = (path, title=None))]
    fn save_scree_plot(&self, py: Python<'_>, path: &str, title: Option<&str>) -> PyResult<()> {
        let mpl = py.import("matplotlib")?;
        mpl.call_method1("use", ("Agg",))?;
        let plt = py.import("matplotlib.pyplot")?;

        let fig = plt.call_method0("figure")?;
        let _ax = fig.call_method0("gca")?;
        let alphas: Vec<usize> = (0..self.eigenvalues_static.len()).collect();
        let deviations: Vec<f64> = self
            .eigenvalues_static
            .iter()
            .map(|&l| (l - 1.0).abs())
            .collect();
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("marker", "o")?;
        kwargs.set_item("markersize", 3)?;
        kwargs.set_item("linewidth", 1)?;
        plt.call_method("semilogy", (alphas, deviations), Some(&kwargs))?;
        plt.call_method1("xlabel", ("Eigenpotential index α",))?;
        plt.call_method1("ylabel", ("|λ_α(0) − 1|",))?;
        let default_title = format!(
            "PDEP scree: {} eigenpotentials, E_c(RPA) = {:.6} Ha",
            self.n_eigenpotentials, self.e_rpa,
        );
        plt.call_method1("title", (title.unwrap_or(&default_title),))?;
        plt.call_method0("grid")?;
        let save_kwargs = pyo3::types::PyDict::new(py);
        save_kwargs.set_item("dpi", 120)?;
        save_kwargs.set_item("bbox_inches", "tight")?;
        plt.call_method("savefig", (path,), Some(&save_kwargs))?;
        plt.call_method0("close")?;
        Ok(())
    }
    fn __repr__(&self) -> String {
        format!(
            "PdepRpaResult(total_energy={:.10}, e_rpa={:.10}, n_eigenpotentials={})",
            self.total_energy, self.e_rpa, self.n_eigenpotentials,
        )
    }
    fn __str__(&self) -> String {
        format!(
            "PDEP-RPA Total Energy: {:.10} Ha (RHF: {:.10}, E_RPA: {:.10}, {} eigenpotentials, converged: {})",
            self.total_energy, self.rhf_energy, self.e_rpa, self.n_eigenpotentials, self.eigensolver_converged,
        )
    }
}

/// `scf_max_iter`, `scf_energy_conv`, `scf_density_conv`: controls of the internal reference SCF,
/// with the names and meaning of `run_rhf` (defaults unchanged when omitted).
///
/// With an ECP basis set the ECP is applied to the molecule, so the reference SCF is the one
/// `run_rhf` solves.
/// `frozen_core` counts orbitals of the valence occupied space, after the ECP has removed its core.
#[pyfunction]
#[pyo3(signature = (
    mol, basis_set, auxbasis,
    frozen_core=None, n_quad=None, quadrature=None, u0=None,
    trunc_thresh=None, eigensolver_conv_thresh=None,
    run_diagnostics=false, k_builder=None, chi0_sparsity=None,
    memory_budget_gb=None, point_charges=None, external_field=None,
    solvent=None, pcm_lebedev_order=None,
    scf_max_iter=None, scf_energy_conv=None, scf_density_conv=None,
    properties=false, export_npz=None, allow_partial_npz=false,
    compute_esp=None, compute_esp_surface=None, esp_surface_vdw_scale=None,
    esp_surface_n_angular=None, compute_polarizability=None,
    compute_electric_field=None, compute_alpha_atomic=None,
    compute_density_matrix=None, compute_dipole=None,
    compute_lowdin_charges=None, compute_hirshfeld_charges=None,
    compute_mulliken_charges=None, compute_chelpg_charges=None,
    compute_resp_charges=None, compute_c6=None, c6_source=None,
    c6_partition=None, df_j_aux=None, df_k_aux=None,
    export_eigpot_prefix=None, export_eigpot_count=None,
    cube_spacing=None, cube_margin=None, with_gradient=false,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_pdep_rpa(
    py: Python<'_>,
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    auxbasis: &PyBasisSet,
    frozen_core: Option<usize>,
    n_quad: Option<usize>,
    quadrature: Option<&str>,
    u0: Option<f64>,
    trunc_thresh: Option<f64>,
    eigensolver_conv_thresh: Option<f64>,
    run_diagnostics: bool,
    k_builder: Option<&str>,
    chi0_sparsity: Option<&str>,
    memory_budget_gb: Option<f64>,
    point_charges: Option<Vec<(f64, f64, f64, f64)>>,
    external_field: Option<(f64, f64, f64)>,
    solvent: Option<&Bound<'_, PyAny>>,
    pcm_lebedev_order: Option<usize>,
    scf_max_iter: Option<usize>,
    scf_energy_conv: Option<f64>,
    scf_density_conv: Option<f64>,
    properties: bool,
    export_npz: Option<&str>,
    allow_partial_npz: bool,
    compute_esp: Option<bool>,
    compute_esp_surface: Option<bool>,
    esp_surface_vdw_scale: Option<f64>,
    esp_surface_n_angular: Option<usize>,
    compute_polarizability: Option<bool>,
    compute_electric_field: Option<bool>,
    compute_alpha_atomic: Option<bool>,
    compute_density_matrix: Option<bool>,
    compute_dipole: Option<bool>,
    compute_lowdin_charges: Option<bool>,
    compute_hirshfeld_charges: Option<bool>,
    compute_mulliken_charges: Option<bool>,
    compute_chelpg_charges: Option<bool>,
    compute_resp_charges: Option<bool>,
    compute_c6: Option<bool>,
    c6_source: Option<&str>,
    c6_partition: Option<&str>,
    df_j_aux: Option<&str>,
    df_k_aux: Option<&str>,
    export_eigpot_prefix: Option<&str>,
    export_eigpot_count: Option<usize>,
    cube_spacing: Option<f64>,
    cube_margin: Option<f64>,
    with_gradient: bool,
) -> PyResult<PyPdepRpaResult> {
    use ferric_rpa::config::{QuadratureConfig, QuadratureScheme, SternheimerConfig};
    let prop_kwargs = PropertyKwargs {
        properties,
        export_npz: export_npz.is_some(),
        compute_esp,
        compute_esp_surface,
        esp_surface_vdw_scale,
        esp_surface_n_angular,
        compute_polarizability,
        compute_electric_field,
        compute_alpha_atomic,
        compute_density_matrix,
        compute_dipole,
        compute_lowdin_charges,
        compute_hirshfeld_charges,
        compute_mulliken_charges,
        compute_chelpg_charges,
        compute_resp_charges,
        compute_c6,
        c6_source: c6_source.map(str::to_string),
        c6_partition: c6_partition.map(str::to_string),
    };
    // Validated before the SCF: a bad c6_source / c6_partition or a property
    // switch without the bundle requested raises ValueError up front.
    let prop_request = prop_kwargs.request()?;
    let cubes = cube_spec(
        export_eigpot_prefix,
        export_eigpot_count,
        cube_spacing,
        cube_margin,
    )?;
    if with_gradient {
        refuse_pdep_rpa_gradient(&GradientScope {
            multiplicity: mol.inner.multiplicity,
            trunc_thresh,
            chi0_sparsity,
            k_builder,
            df_aux: df_j_aux.is_some() || df_k_aux.is_some(),
            environment: point_charges.is_some() || external_field.is_some() || solvent.is_some(),
            properties: prop_request.is_some() || cubes.is_some(),
            run_diagnostics,
        })?;
    }

    use ferric_rpa::{run_pdep_rpa_on, PdepRpaConfig};

    let sys = prepare(py, mol, basis_set)?;
    let dfbs = prepare_aux(py, &sys, auxbasis)?;
    let (mol, prep, op, bounds) = (sys.mol(), sys.basis(), sys.op(), sys.bounds());
    // The reference SCF must see the ENVIRONMENT, or the dielectric response
    // computed from it is that of an isolated molecule. Screening a
    // ligand-pocket interaction needs the response of the ligand AS IT SITS
    // IN THE POCKET, so the point charges have to reach this SCF.
    // Parsed before the reference config, which parses `k_builder` last.
    let pcm = build_pcm_config(solvent, pcm_lebedev_order)?;
    let mut scf_config = rhf_config_conv(
        k_builder,
        budget_bytes_from_gb(memory_budget_gb),
        ScfConv::new(scf_max_iter, scf_energy_conv, scf_density_conv),
    )?;
    if with_gradient && scf_density_conv.is_none() {
        // The analytic gradient's reference setting (`rpa_analytic_gradient`,
        // the CLI optimize path): an energy-grade density threshold leaves
        // an SCF residual that the Z-vector does not remove.
        scf_config.density_conv = 1e-9;
    }
    // Same opt-out spellings as run_rhf; unset keeps the exact four-centre J/K
    // (the CLI's pdep-rpa reference defaults to RI-JK on def2-universal-jkfit).
    scf_config.df_j_aux = df_j_aux.map(ferric_scf::rhf::normalize_df_aux);
    scf_config.df_k_aux = df_k_aux.map(ferric_scf::rhf::normalize_df_aux);
    scf_config.external_potential = build_external_potential(point_charges, external_field);
    scf_config.pcm = pcm;
    let rhf = sys
        .solve_rhf(&scf_config)
        .and_then(ScfResult::into_converged)
        .map_err(make_err)?;

    // Canonical parser (shared with the CLI): unknown schemes error rather than
    // silently running Gauss-Legendre.
    let scheme = QuadratureScheme::parse_config_str(quadrature)
        .map_err(|e| crate::errors::input_err(format!("quadrature: {e}")))?;
    if u0.is_some() && !scheme.honours_u0() {
        eprintln!(
            "warning: u0 is ignored by quadrature='minimax' (it derives u0 from \
             n_quad); pass quadrature='gauss-legendre' or 'chebyshev-tan' to use it"
        );
    }
    let cfg = PdepRpaConfig {
        frozen_core: frozen_core.unwrap_or(0),
        // The analytic gradient differentiates the full-rank energy.
        trunc_thresh: trunc_thresh.unwrap_or(if with_gradient { 0.0 } else { 1e-4 }),
        eigensolver_max_vecs: 0,
        eigensolver_conv_thresh: lanczos_conv_thresh_or(eigensolver_conv_thresh, 1e-6),
        quadrature: QuadratureConfig {
            scheme,
            n_points: n_quad.unwrap_or(20),
            u0: u0.unwrap_or(0.5),
        },
        sternheimer: SternheimerConfig::default(),
        run_diagnostics,
        eigensolver: ferric_rpa::Eigensolver::default(),
        chi0_backend: ferric_rpa::config::Chi0Backend::default(),
        chi0_sparsity: ferric_rpa::config::Chi0Sparsity::parse_config_str(chi0_sparsity)
            .map_err(make_err)?,
        memory_budget_bytes: budget_bytes_from_gb(memory_budget_gb),
        // Python run_pdep_rpa is energy-only; the GW/property paths that consume
        // the inverse-dielectric stack have their own entry points (M9 gate).
        need_inv_dielectric_freq: false,
        need_eigenvalues_freq: true,
        verbose: false,
    };
    let r = run_pdep_rpa_on(&dfbs, &rhf, &cfg).map_err(make_err)?;
    if !r.eigensolver_converged {
        eprintln!(
            "warning: PDEP-RPA eigensolver did not fully converge (best-effort Ritz pairs; \
             eigenvalues_static/eigenpotentials are not verified to residual tolerance)"
        );
    }
    let gradient_data = if with_gradient {
        Some(
            py.allow_threads(|| pdep_rpa_total_gradient(mol, prep, &dfbs, op, bounds, &rhf, &cfg))
                .map_err(make_err)?,
        )
    } else {
        None
    };
    let certificate_data = ferric_mp2::certificate::with_anchor_envelope(
        ferric_rpa::certificate::pdep_rpa_certificate(
            &reference_certificate(mol, &rhf, &scf_config, &basis_set.inner),
            &auxbasis.inner.name,
            &cfg,
            &r,
        ),
        "PDEP-RPA",
        mol,
        &basis_set.inner,
        Some(&auxbasis.inner),
    );
    let cert_json = certificate_data.to_json().to_string();
    let out = run_property_bundle(
        py,
        &BundleJob {
            ctx: sys.ctx(),
            mol,
            bs: &basis_set.inner,
            aux_bs: &auxbasis.inner,
            prep,
            dfbs: &dfbs,
            rhf: &rhf,
            op,
            rpa_cfg: &cfg,
            scf_config: &scf_config,
            request: prop_request,
            export_npz,
            allow_partial_npz,
            cubes,
        },
        &r,
        &cert_json,
    )?;
    Ok(PyPdepRpaResult {
        certificate_data,
        props: out.props,
        property_warnings: out.warnings,
        written_files: out.written,
        density_matrix: out.density_matrix,
        mo_coeffs: out.mo_coeffs,
        orbital_energies: out.orbital_energies,
        gradient_data,
        rhf_energy: rhf.energy,
        e_rpa: r.e_rpa,
        total_energy: rhf.energy + r.e_rpa,
        n_eigenpotentials: r.n_eigenpotentials,
        e_rpa_dft_diag: r.e_rpa_dft_diag,
        eigensolver_converged: r.eigensolver_converged,
        eigenvalues_static: r.eigenvalues_static,
        eigenpotentials: r.eigenpotentials,
        quad_freqs: r.quad_freqs,
        quad_weights: r.quad_weights,
        eigenvalues_freq: r.eigenvalues_freq,
    })
}
