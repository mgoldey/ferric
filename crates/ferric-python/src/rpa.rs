//! RPA bindings: moved verbatim from `lib.rs`.

use super::*;

// ── RS-MP2-RPA (SR-MP2 + LR-dRPA, Δ-form B or coupled-rings T) ──

#[pyclass]
#[pyo3(name = "RsMp2RpaResult")]
pub(crate) struct PyRsMp2RpaResult {
    #[pyo3(get)]
    pub(crate) total_energy: f64,
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

#[pymethods]
impl PyRsMp2RpaResult {
    /// Validation certificate of this result (see `RhfResult.certificate`):
    /// the method's grade from the validation page capped by the reference
    /// SCF's, its settings as `approximations`, the reference's as
    /// `scf <name>`, and its warnings.
    #[getter]
    fn certificate(&self, py: Python<'_>) -> PyResult<PyObject> {
        certificate_dict(py, &self.certificate_data)
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

/// SR-MP2 + LR-dRPA on an RHF reference (`formulation` "delta-lr" or
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
/// `scf_max_iter`, `scf_energy_conv`, `scf_density_conv`: controls of the internal reference SCF,
/// with the names and meaning of `run_rhf` (defaults unchanged when omitted).
#[pyfunction]
#[pyo3(signature = (mol, basis_set, auxbasis, omega=None, frozen_core=None, k_builder=None, formulation=None, attenuator=None, r0=None, terf_omega=None, memory_budget_gb=None, scf_jk=None, jk_auxbasis=None, scf_ri_j_precision=None, scf_max_iter=None, scf_energy_conv=None, scf_density_conv=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_rs_mp2_rpa(
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    auxbasis: &PyBasisSet,
    omega: Option<f64>,
    frozen_core: Option<usize>,
    k_builder: Option<&str>,
    formulation: Option<&str>,
    attenuator: Option<&str>,
    r0: Option<f64>,
    terf_omega: Option<f64>,
    memory_budget_gb: Option<f64>,
    scf_jk: Option<&str>,
    jk_auxbasis: Option<&str>,
    scf_ri_j_precision: Option<&str>,
    scf_max_iter: Option<usize>,
    scf_energy_conv: Option<f64>,
    scf_density_conv: Option<f64>,
) -> PyResult<PyRsMp2RpaResult> {
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
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    let dfbs = PreparedBasis::new(&mol.inner, &auxbasis.inner).map_err(make_err)?;
    let rhf = converged_rhf(&mol.inner, &prep, &cfg_rhf)?;
    // Map formulation string to enum.
    let form = match formulation.unwrap_or("delta-lr") {
        "delta-lr" => ferric_rpa::RsMp2RpaFormulation::DeltaLr,
        "coupled-rings" => ferric_rpa::RsMp2RpaFormulation::CoupledRings,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown formulation \"{other}\"; expected \"delta-lr\" or \"coupled-rings\""
            )))
        }
    };
    // attenuator: "erf" (default, ω in Å⁻¹) or "terf" (r0 in Å, ω derived).
    let atten = match attenuator.unwrap_or("erf") {
        "erf" => ferric_rpa::rs_mp2_rpa::Attenuator::Erf,
        "terf" => ferric_rpa::rs_mp2_rpa::Attenuator::Terf,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown attenuator \"{other}\"; expected \"erf\" or \"terf\""
            )))
        }
    };
    // omega is supplied in Å⁻¹; convert to Bohr⁻¹ for the operator. r0 is
    // supplied in Å (2026-07-21: fixed from Bohr, matching r0_bonded/
    // r0_nonbonded's existing Å convention elsewhere in this file); convert
    // to Bohr for RsMp2RpaConfig, which stays Bohr-native.
    const ANG2BOHR_R0: f64 = ANGSTROM_TO_BOHR;
    let mut cfg = ferric_rpa::RsMp2RpaConfig {
        omega: omega.unwrap_or(0.420) * ferric_mp2::attenuated::BOHR_INV_PER_ANG_INV,
        attenuator: atten,
        r0: r0.unwrap_or(3.18 / ANG2BOHR_R0) * ANG2BOHR_R0,
        // terf_omega in Å⁻¹ (same convention as omega); None keeps the
        // Dutoi curvature link ω = 1/(r0·√2). Terf-only.
        terf_omega: terf_omega.map(|w| w * ferric_mp2::attenuated::BOHR_INV_PER_ANG_INV),
        frozen_core: frozen_core.unwrap_or(0),
        formulation: form,
        ..Default::default()
    };
    cfg.drpa.memory_budget_bytes = budget_bytes_from_gb(memory_budget_gb);
    let r = ferric_rpa::rs_mp2_lr_rpa(&mol.inner, &prep, &dfbs, &rhf, &cfg).map_err(make_err)?;
    let certificate_data = ferric_rpa::certificate::rs_mp2_rpa_certificate(
        &reference_certificate(&mol.inner, &rhf, &cfg_rhf, &basis_set.inner),
        &auxbasis.inner.name,
        &cfg,
    );
    Ok(PyRsMp2RpaResult {
        total_energy: r.total_energy,
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
        jk_label: jk.label(),
        jk_auxbasis: jk.aux,
        certificate_data,
    })
}

// ── PDEP-RPA ──

#[pyclass]
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
}

#[pymethods]
impl PyPdepRpaResult {
    /// Validation certificate of this result (see `RhfResult.certificate`):
    /// the method's grade from the validation page capped by the reference
    /// SCF's, its settings as `approximations`, the reference's as
    /// `scf <name>`, and its warnings.
    #[getter]
    fn certificate(&self, py: Python<'_>) -> PyResult<PyObject> {
        certificate_dict(py, &self.certificate_data)
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
#[pyfunction]
#[pyo3(signature = (
    mol, basis_set, auxbasis,
    frozen_core=None, n_quad=None, quadrature=None, u0=None,
    trunc_thresh=None, eigensolver_conv_thresh=None,
    run_diagnostics=false, k_builder=None, chi0_sparsity=None,
    memory_budget_gb=None, point_charges=None, external_field=None,
    solvent=None, pcm_lebedev_order=None,
    scf_max_iter=None, scf_energy_conv=None, scf_density_conv=None,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_pdep_rpa(
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
) -> PyResult<PyPdepRpaResult> {
    use ferric_rpa::config::{QuadratureConfig, QuadratureScheme, SternheimerConfig};
    use ferric_rpa::{run_pdep_rpa as run_pdep_rpa_inner, PdepRpaConfig};

    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    let dfbs = PreparedBasis::new(&mol.inner, &auxbasis.inner).map_err(make_err)?;
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).map_err(make_err)?;
    let ctx = ParallelContext::default();
    // The reference SCF must see the ENVIRONMENT, or the dielectric response
    // computed from it is that of an isolated molecule. Screening a
    // ligand-pocket interaction needs the response of the ligand AS IT SITS
    // IN THE POCKET, so the point charges have to reach this SCF.
    let mut scf_config = rhf_config_conv(
        k_builder,
        budget_bytes_from_gb(memory_budget_gb),
        ScfConv::new(scf_max_iter, scf_energy_conv, scf_density_conv),
    );
    scf_config.external_potential = build_external_potential(point_charges, external_field);
    scf_config.pcm = build_pcm_config(solvent, pcm_lebedev_order)?;
    let rhf = solve_rhf(&ctx, &mol.inner, &prep, op, &bounds, &scf_config).map_err(make_err)?;
    if !rhf.converged {
        return Err(make_err(ferric_core::FerricError::ScfConvergence {
            iterations: rhf.iterations,
            last_energy: rhf.energy,
        }));
    }

    // Canonical parser (shared with the CLI): unknown schemes error rather than
    // silently running Gauss-Legendre.
    let scheme = QuadratureScheme::parse_config_str(quadrature)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("quadrature: {e}")))?;
    if u0.is_some() && !scheme.honours_u0() {
        eprintln!(
            "warning: u0 is ignored by quadrature='minimax' (it derives u0 from \
             n_quad); pass quadrature='gauss-legendre' or 'chebyshev-tan' to use it"
        );
    }
    let cfg = PdepRpaConfig {
        frozen_core: frozen_core.unwrap_or(0),
        trunc_thresh: trunc_thresh.unwrap_or(1e-4),
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
    let r = run_pdep_rpa_inner(&mol.inner, &prep, &dfbs, op, &rhf, &cfg).map_err(make_err)?;
    if !r.eigensolver_converged {
        eprintln!(
            "warning: PDEP-RPA eigensolver did not fully converge (best-effort Ritz pairs; \
             eigenvalues_static/eigenpotentials are not verified to residual tolerance)"
        );
    }
    let certificate_data = ferric_rpa::certificate::pdep_rpa_certificate(
        &reference_certificate(&mol.inner, &rhf, &scf_config, &basis_set.inner),
        &auxbasis.inner.name,
        &cfg,
        &r,
    );
    Ok(PyPdepRpaResult {
        certificate_data,
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
