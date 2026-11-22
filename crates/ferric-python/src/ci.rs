//! `ferric.run_casci`: CAS-CI (`ferric_ci::run_cas_ci`) with spin-summed
//! 1- and 2-RDMs, and `ferric.run_casscf`: two-step CASSCF
//! (`ferric_ci::run_casscf`), both on a closed-shell exact-J/K RHF reference,
//! and `ferric.run_nevpt2`: SC-NEVPT2 (`ferric_ci::sc_nevpt2`) on either.
//! `run_casscf` and `run_nevpt2` run `ferric_ci::driver`, as the CLI kinds
//! `casscf` and `nevpt2` do, and carry those kinds' certificates.

use crate::{budget_bytes_from_gb, certificate_dict, make_err, prepare, PyBasisSet, PyMolecule};
use ferric_ci::driver::{rhf_start, ScfStart};
use ferric_core::certificate::{BuildStamp, Certificate};
use ndarray::{Array2, Array4};
use numpy::{IntoPyArray, PyArray2, PyArray4};
use pyo3::prelude::*;

/// Library capability name on the certificate (no CLI `method.kind`).
const CAPABILITY: &str = "casci";

/// The build this extension was compiled from.
fn build_stamp() -> BuildStamp {
    BuildStamp::new(
        ferric_build_info::VERSION,
        ferric_build_info::COMMIT,
        ferric_build_info::DIRTY,
    )
}

#[pyclass]
#[pyo3(name = "CasCiResult")]
struct PyCasCiResult {
    #[pyo3(get)]
    e_total: f64,
    #[pyo3(get)]
    e_core: f64,
    #[pyo3(get)]
    e_active: f64,
    #[pyo3(get)]
    rhf_energy: f64,
    #[pyo3(get)]
    n_determinants: usize,
    #[pyo3(get)]
    converged: bool,
    #[pyo3(get)]
    iterations: usize,
    #[pyo3(get)]
    active_start: usize,
    n_alpha_str: usize,
    n_beta_str: usize,
    ci: Vec<f64>,
    dm1: Option<Array2<f64>>,
    dm2: Option<Array4<f64>>,
    h_active: Array2<f64>,
    eri_active: Array4<f64>,
    mo_coeff_active: Array2<f64>,
    certificate_data: Certificate,
}

#[pymethods]
impl PyCasCiResult {
    /// CI coefficients, shape `(n_alpha_strings, n_beta_strings)`; strings in
    /// ascending bitmask order (bit `p` = active orbital `p` occupied).
    #[getter]
    fn ci_vector<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let a = Array2::from_shape_vec((self.n_alpha_str, self.n_beta_str), self.ci.clone())
            .map_err(make_err)?;
        Ok(a.into_pyarray(py))
    }
    /// `dm1[p,q] = <E_pq>` over the active orbitals, or `None` when
    /// `compute_rdms=False`.
    #[getter]
    fn dm1<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
        self.dm1.clone().map(|a| a.into_pyarray(py))
    }
    /// `dm2[p,q,r,s] = Σ_στ <a†_{pσ} a†_{rτ} a_{sτ} a_{qσ}>` (PySCF
    /// `make_rdm12` convention), or `None`.
    #[getter]
    fn dm2<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray4<f64>>> {
        self.dm2.clone().map(|a| a.into_pyarray(py))
    }
    /// Active one-electron integrals including the inactive-core potential.
    #[getter]
    fn h_active<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.h_active.clone().into_pyarray(py)
    }
    /// Active two-electron integrals `(pq|rs)`, chemist notation.
    #[getter]
    fn eri_active<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray4<f64>> {
        self.eri_active.clone().into_pyarray(py)
    }
    /// Active RHF orbitals, shape `(nbasis, n_active)`.
    #[getter]
    fn mo_coeff_active<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.mo_coeff_active.clone().into_pyarray(py)
    }
    /// Validation certificate (schema of `RhfResult.certificate`).
    #[getter]
    fn certificate(&self, py: Python<'_>) -> PyResult<PyObject> {
        certificate_dict(py, &self.certificate_data)
    }
    fn __repr__(&self) -> String {
        format!(
            "CasCiResult(e_total={:.10}, n_determinants={}, converged={})",
            self.e_total, self.n_determinants, self.converged
        )
    }
}

/// CAS-CI on the canonical RHF orbitals `active_start .. active_start +
/// n_active`, lowest root of the given `(n_alpha, n_beta)` sector.
#[pyfunction]
#[pyo3(signature = (mol, basis_set, n_active, n_elec_active, active_start=None, compute_rdms=true, conv_thresh=1e-9, max_subspace=50, max_iter=200, memory_budget_gb=None, scf_max_iter=None, scf_energy_conv=None, scf_density_conv=None))]
#[allow(clippy::too_many_arguments)]
fn run_casci(
    py: Python<'_>,
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    n_active: usize,
    n_elec_active: (usize, usize),
    active_start: Option<usize>,
    compute_rdms: bool,
    conv_thresh: f64,
    max_subspace: usize,
    max_iter: usize,
    memory_budget_gb: Option<f64>,
    scf_max_iter: Option<usize>,
    scf_energy_conv: Option<f64>,
    scf_density_conv: Option<f64>,
) -> PyResult<PyCasCiResult> {
    let sys = prepare(mol, basis_set)?;
    let mol = sys.mol();
    let nelec = mol.nelec() as usize;
    let n_act_el = n_elec_active.0 + n_elec_active.1;
    let active_start = match active_start {
        Some(s) => s,
        None => {
            if n_act_el > nelec || !(nelec - n_act_el).is_multiple_of(2) {
                return Err(make_err(format!(
                    "run_casci: {n_act_el} active electrons do not leave an even inactive \
                     count out of {nelec}"
                )));
            }
            (nelec - n_act_el) / 2
        }
    };
    let budget = budget_bytes_from_gb(memory_budget_gb);
    let start = ScfStart {
        max_iter: scf_max_iter,
        energy_conv: scf_energy_conv,
        density_conv: scf_density_conv,
    };
    let rhf = py
        .allow_threads(|| rhf_start(&sys, start, budget))
        .map_err(make_err)?;
    let cfg = ferric_ci::CasCiConfig {
        n_active,
        n_elec_active,
        active_start,
        conv_thresh,
        max_subspace,
        max_iter,
        memory_budget_bytes: budget,
        compute_rdms,
    };
    let r = py
        .allow_threads(|| ferric_ci::driver::casci_on_rhf(&sys, &rhf, &cfg))
        .map_err(make_err)?;

    let mut cert = Certificate::ungraded(CAPABILITY).with_build(build_stamp());
    if !r.converged {
        cert.push_warning(format!(
            "CAS-CI Davidson did not reach conv_thresh {conv_thresh:e} in {} iterations",
            r.iterations
        ));
    }
    let n = n_active;
    let mos = rhf.mos_r();
    let mo_coeff_active = mos
        .slice(ndarray::s![.., active_start..active_start + n])
        .to_owned();
    let eri_active =
        Array4::from_shape_vec((n, n, n, n), r.integrals.eri.clone()).map_err(make_err)?;
    let (dm1, dm2) = match r.rdms {
        Some(d) => (Some(d.dm1), Some(d.dm2)),
        None => (None, None),
    };
    Ok(PyCasCiResult {
        e_total: r.e_total,
        e_core: r.e_core,
        e_active: r.e_active,
        rhf_energy: rhf.energy,
        n_determinants: r.n_determinants,
        converged: r.converged,
        iterations: r.iterations,
        active_start,
        n_alpha_str: ferric_ci::strings::enumerate_strings(n, n_elec_active.0).len(),
        n_beta_str: ferric_ci::strings::enumerate_strings(n, n_elec_active.1).len(),
        ci: r.ci_vector,
        dm1,
        dm2,
        h_active: r.integrals.h.clone(),
        eri_active,
        mo_coeff_active,
        certificate_data: cert,
    })
}

#[pyclass]
#[pyo3(name = "CasScfResult")]
struct PyCasScfResult {
    #[pyo3(get)]
    e_total: f64,
    #[pyo3(get)]
    e_core: f64,
    #[pyo3(get)]
    rhf_energy: f64,
    #[pyo3(get)]
    n_determinants: usize,
    #[pyo3(get)]
    iterations: usize,
    #[pyo3(get)]
    grad_norm: f64,
    #[pyo3(get)]
    converged: bool,
    #[pyo3(get)]
    ci_residual_norm: f64,
    #[pyo3(get)]
    n_core: usize,
    /// Integral route the run used: "dense", "direct" or "ri".
    #[pyo3(get)]
    integrals: String,
    /// Auxiliary basis of the RI route, `None` on the exact routes.
    #[pyo3(get)]
    auxbasis: Option<String>,
    /// Lowest eigenvalue of the CI-relaxed orbital Hessian at `mo_coeff`
    /// (Ha); `None` with `casscf_stability="off"` or an unconverged run.
    #[pyo3(get)]
    stability_lowest_eigenvalue: Option<f64>,
    /// Verdict of that check: "STABLE", "UNSTABLE" (a saddle), "MARGINAL"
    /// or "INDETERMINATE ..."; `None` without a check.
    #[pyo3(get)]
    stability_verdict: Option<String>,
    /// Steps taken along negative modes (`casscf_stability="follow"`).
    #[pyo3(get)]
    stability_follow_steps: usize,
    n_alpha_str: usize,
    n_beta_str: usize,
    ci: Vec<f64>,
    dm1: Array2<f64>,
    dm2: Array4<f64>,
    mo_coeff: Array2<f64>,
    mo_energy: Option<Vec<f64>>,
    certificate_data: Certificate,
}

#[pymethods]
impl PyCasScfResult {
    /// CI coefficients, shape `(n_alpha_strings, n_beta_strings)`, in the
    /// optimized active orbitals.
    #[getter]
    fn ci_vector<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let a = Array2::from_shape_vec((self.n_alpha_str, self.n_beta_str), self.ci.clone())
            .map_err(make_err)?;
        Ok(a.into_pyarray(py))
    }
    /// Spin-summed 1-RDM over the active orbitals.
    #[getter]
    fn dm1<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.dm1.clone().into_pyarray(py)
    }
    /// Spin-summed 2-RDM (PySCF `make_rdm12` convention).
    #[getter]
    fn dm2<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray4<f64>> {
        self.dm2.clone().into_pyarray(py)
    }
    /// Optimized orbitals, shape `(nbasis, nmo)`: core, active, virtual.
    /// Canonical when `run_casscf(canonicalize=True)`.
    #[getter]
    fn mo_coeff<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.mo_coeff.clone().into_pyarray(py)
    }
    /// Diagonal of `F^I + F^A` in `mo_coeff` (Ha) when canonicalized, else
    /// `None`.
    #[getter]
    fn mo_energy<'py>(&self, py: Python<'py>) -> Option<Bound<'py, numpy::PyArray1<f64>>> {
        self.mo_energy.clone().map(|v| v.into_pyarray(py))
    }
    /// Validation certificate (schema of `RhfResult.certificate`).
    #[getter]
    fn certificate(&self, py: Python<'_>) -> PyResult<PyObject> {
        certificate_dict(py, &self.certificate_data)
    }
    fn __repr__(&self) -> String {
        format!(
            "CasScfResult(e_total={:.10}, iterations={}, grad_norm={:.1e}, converged={})",
            self.e_total, self.iterations, self.grad_norm, self.converged
        )
    }
}

/// Two-step CASSCF from the canonical RHF orbitals: `n_active` active
/// orbitals above `n_core` doubly occupied ones, single root of the
/// `(n_alpha, n_beta)` sector, optionally canonical orbitals on return.
#[pyfunction]
#[pyo3(signature = (mol, basis_set, n_active, n_elec_active, n_core=None, casscf_grad_tol=1e-7, casscf_energy_tol=1e-12, casscf_max_iter=300, casscf_max_step=0.2, ci_conv_thresh=1e-10, ci_max_subspace=50, ci_max_iter=200, memory_budget_gb=None, scf_max_iter=None, scf_energy_conv=None, scf_density_conv=None, canonicalize=false, casscf_integrals="auto", casscf_integral_thresh=None, casscf_optimizer="ah", casscf_auxbasis=None, casscf_stability="off"))]
#[allow(clippy::too_many_arguments)]
fn run_casscf(
    py: Python<'_>,
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    n_active: usize,
    n_elec_active: (usize, usize),
    n_core: Option<usize>,
    casscf_grad_tol: f64,
    casscf_energy_tol: f64,
    casscf_max_iter: usize,
    casscf_max_step: f64,
    ci_conv_thresh: f64,
    ci_max_subspace: usize,
    ci_max_iter: usize,
    memory_budget_gb: Option<f64>,
    scf_max_iter: Option<usize>,
    scf_energy_conv: Option<f64>,
    scf_density_conv: Option<f64>,
    canonicalize: bool,
    casscf_integrals: &str,
    casscf_integral_thresh: Option<f64>,
    casscf_optimizer: &str,
    casscf_auxbasis: Option<&str>,
    casscf_stability: &str,
) -> PyResult<PyCasScfResult> {
    let integrals = ferric_ci::casscf::CasScfIntegrals::parse_config_str(
        casscf_integrals,
        casscf_integral_thresh,
        casscf_auxbasis,
    )
    .map_err(|e| make_err(format!("run_casscf: casscf_{e}")))?;
    let optimizer = ferric_ci::CasScfOptimizer::parse_config_str(casscf_optimizer)
        .map_err(|e| make_err(format!("run_casscf: casscf_{e}")))?;
    let stability = ferric_ci::CasScfStabilityMode::parse_config_str(casscf_stability)
        .map_err(|e| make_err(format!("run_casscf: casscf_{e}")))?;
    let sys = prepare(mol, basis_set)?;
    let mol = sys.mol();
    let n_core = match n_core {
        Some(s) => s,
        None => {
            ferric_ci::driver::default_n_core(mol.nelec() as usize, n_elec_active, "run_casscf")
                .map_err(make_err)?
        }
    };
    let budget = budget_bytes_from_gb(memory_budget_gb);
    let start = ScfStart {
        max_iter: scf_max_iter,
        energy_conv: scf_energy_conv,
        density_conv: scf_density_conv,
    };
    let rhf = py
        .allow_threads(|| rhf_start(&sys, start, budget))
        .map_err(make_err)?;
    let cfg = ferric_ci::CasScfConfig {
        n_active,
        n_elec_active,
        n_core,
        grad_tol: casscf_grad_tol,
        energy_tol: casscf_energy_tol,
        max_iter: casscf_max_iter,
        max_step: casscf_max_step,
        ci_conv_thresh,
        ci_max_subspace,
        ci_max_iter,
        memory_budget_bytes: budget,
        canonicalize,
        integrals,
        optimizer,
        stability,
        ..Default::default()
    };
    let r = py
        .allow_threads(|| ferric_ci::driver::casscf_on_rhf(&sys, &rhf, &cfg))
        .map_err(make_err)?;

    let cert = ferric_mp2::certificate::with_anchor_envelope(
        ferric_ci::certificate::casscf_certificate_prefixed(&r, &cfg, "casscf_")
            .with_build(build_stamp()),
        "CASSCF",
        mol,
        &basis_set.inner,
        None,
    );
    if !r.converged {
        crate::emit_runtime_warning(
            py,
            ferric_ci::certificate::casscf_unconverged_warning_prefixed(&r, &cfg, "casscf_"),
        );
    }
    if let Some(st) = r.stability.as_ref() {
        if st.verdict() == ferric_scf::stability::StabilityVerdict::Unstable {
            crate::emit_runtime_warning(
                py,
                format!(
                    "CASSCF point is a saddle: {} (casscf_stability=\"follow\" steps down it)",
                    st.summary()
                ),
            );
        }
    }
    Ok(PyCasScfResult {
        e_total: r.e_total,
        e_core: r.e_core,
        rhf_energy: rhf.energy,
        n_determinants: r.n_determinants,
        iterations: r.iterations,
        grad_norm: r.grad_norm,
        converged: r.converged,
        ci_residual_norm: r.ci_residual_norm,
        n_core,
        integrals: r.integrals.as_str().to_string(),
        auxbasis: r.integrals.aux().map(str::to_string),
        stability_lowest_eigenvalue: r.stability.as_ref().map(|s| s.lowest_eigenvalue),
        stability_verdict: r
            .stability
            .as_ref()
            .map(|s| s.verdict().label().to_string()),
        stability_follow_steps: r.stability.as_ref().map_or(0, |s| s.follow_steps),
        n_alpha_str: ferric_ci::strings::enumerate_strings(n_active, n_elec_active.0).len(),
        n_beta_str: ferric_ci::strings::enumerate_strings(n_active, n_elec_active.1).len(),
        ci: r.ci_vector,
        dm1: r.rdms.dm1,
        dm2: r.rdms.dm2,
        mo_coeff: r.mo_coeff,
        mo_energy: r.mo_energy,
        certificate_data: cert,
    })
}

#[pyclass]
#[pyo3(name = "Nevpt2Result")]
struct PyNevpt2Result {
    #[pyo3(get)]
    e_corr: f64,
    #[pyo3(get)]
    e_total: f64,
    #[pyo3(get)]
    e_reference: f64,
    #[pyo3(get)]
    reference: String,
    #[pyo3(get)]
    rhf_energy: f64,
    #[pyo3(get)]
    reference_iterations: usize,
    #[pyo3(get)]
    reference_grad_norm: Option<f64>,
    /// Lowest CI-relaxed orbital Hessian eigenvalue of the CASSCF reference
    /// (Ha); `None` on a CAS-CI reference or with `casscf_stability="off"`.
    #[pyo3(get)]
    reference_stability_lowest_eigenvalue: Option<f64>,
    #[pyo3(get)]
    ci_residual_norm: f64,
    #[pyo3(get)]
    n_core: usize,
    #[pyo3(get)]
    n_active: usize,
    #[pyo3(get)]
    n_virtual: usize,
    energies: ferric_ci::Nevpt2Classes,
    norms: ferric_ci::Nevpt2Classes,
    mo_coeff: Array2<f64>,
    mo_energy: Vec<f64>,
    certificate_data: Certificate,
}

fn classes_dict<'py>(
    py: Python<'py>,
    c: &ferric_ci::Nevpt2Classes,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let d = pyo3::types::PyDict::new(py);
    for (name, v) in ferric_ci::Nevpt2Classes::NAMES.iter().zip(c.to_array()) {
        d.set_item(*name, v)?;
    }
    Ok(d)
}

#[pymethods]
impl PyNevpt2Result {
    /// Class energies (Ha), keyed by PySCF's names `Sr`, `Si`, `Sijrs`,
    /// `Sijr`, `Srsi`, `Srs`, `Sij`, `Sir`.
    #[getter]
    fn energies<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        classes_dict(py, &self.energies)
    }
    /// Class norms (sum of the perturber norms over every label).
    #[getter]
    fn norms<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        classes_dict(py, &self.norms)
    }
    /// Orbitals used: canonical core and virtual columns, the reference's
    /// active columns.
    #[getter]
    fn mo_coeff<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.mo_coeff.clone().into_pyarray(py)
    }
    /// Diagonal of `F^I + F^A` in `mo_coeff` (Ha).
    #[getter]
    fn mo_energy<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<f64>> {
        self.mo_energy.clone().into_pyarray(py)
    }
    /// Validation certificate (schema of `RhfResult.certificate`).
    #[getter]
    fn certificate(&self, py: Python<'_>) -> PyResult<PyObject> {
        certificate_dict(py, &self.certificate_data)
    }
    fn __repr__(&self) -> String {
        format!(
            "Nevpt2Result(e_total={:.10}, e_corr={:.10}, reference={})",
            self.e_total, self.e_corr, self.reference
        )
    }
}

/// SC-NEVPT2 on a CASSCF (default) or CAS-CI reference from the canonical
/// RHF orbitals. The reference must converge; an unconverged one raises.
#[pyfunction]
#[pyo3(signature = (mol, basis_set, n_active, n_elec_active, n_core=None, reference="casscf", casscf_grad_tol=1e-7, casscf_energy_tol=1e-12, casscf_max_iter=300, casscf_max_step=0.2, ci_conv_thresh=1e-10, ci_max_subspace=50, ci_max_iter=200, memory_budget_gb=None, scf_max_iter=None, scf_energy_conv=None, scf_density_conv=None, casscf_integrals="auto", casscf_integral_thresh=None, casscf_optimizer="ah", casscf_auxbasis=None, casscf_stability="off"))]
#[allow(clippy::too_many_arguments)]
fn run_nevpt2(
    py: Python<'_>,
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    n_active: usize,
    n_elec_active: (usize, usize),
    n_core: Option<usize>,
    reference: &str,
    casscf_grad_tol: f64,
    casscf_energy_tol: f64,
    casscf_max_iter: usize,
    casscf_max_step: f64,
    ci_conv_thresh: f64,
    ci_max_subspace: usize,
    ci_max_iter: usize,
    memory_budget_gb: Option<f64>,
    scf_max_iter: Option<usize>,
    scf_energy_conv: Option<f64>,
    scf_density_conv: Option<f64>,
    casscf_integrals: &str,
    casscf_integral_thresh: Option<f64>,
    casscf_optimizer: &str,
    casscf_auxbasis: Option<&str>,
    casscf_stability: &str,
) -> PyResult<PyNevpt2Result> {
    let integrals = ferric_ci::casscf::CasScfIntegrals::parse_config_str(
        casscf_integrals,
        casscf_integral_thresh,
        casscf_auxbasis,
    )
    .map_err(|e| make_err(format!("run_nevpt2: casscf_{e}")))?;
    let optimizer = ferric_ci::CasScfOptimizer::parse_config_str(casscf_optimizer)
        .map_err(|e| make_err(format!("run_nevpt2: casscf_{e}")))?;
    let stability = ferric_ci::CasScfStabilityMode::parse_config_str(casscf_stability)
        .map_err(|e| make_err(format!("run_nevpt2: casscf_{e}")))?;
    let on = ferric_ci::driver::Nevpt2On::parse_config_str(reference)
        .map_err(|e| make_err(format!("run_nevpt2: {e}")))?;
    let sys = prepare(mol, basis_set)?;
    let mol = sys.mol();
    let n_core = match n_core {
        Some(s) => s,
        None => {
            ferric_ci::driver::default_n_core(mol.nelec() as usize, n_elec_active, "run_nevpt2")
                .map_err(make_err)?
        }
    };
    let budget = budget_bytes_from_gb(memory_budget_gb);
    let start = ScfStart {
        max_iter: scf_max_iter,
        energy_conv: scf_energy_conv,
        density_conv: scf_density_conv,
    };
    let rhf = py
        .allow_threads(|| rhf_start(&sys, start, budget))
        .map_err(make_err)?;
    let cfg = ferric_ci::CasScfConfig {
        n_active,
        n_elec_active,
        n_core,
        grad_tol: casscf_grad_tol,
        energy_tol: casscf_energy_tol,
        max_iter: casscf_max_iter,
        max_step: casscf_max_step,
        ci_conv_thresh,
        ci_max_subspace,
        ci_max_iter,
        memory_budget_bytes: budget,
        integrals,
        optimizer,
        stability,
        ..Default::default()
    };
    let run = py
        .allow_threads(|| ferric_ci::driver::nevpt2_on_rhf(&sys, &rhf, on, &cfg))
        .map_err(make_err)?;
    let cert = ferric_mp2::certificate::with_anchor_envelope(
        ferric_ci::certificate::nevpt2_certificate(on, &cfg, run.reference_stability.as_ref())
            .with_build(build_stamp()),
        "SC-NEVPT2",
        mol,
        &basis_set.inner,
        None,
    );
    let stability_lambda = run
        .reference_stability
        .as_ref()
        .map(|s| s.lowest_eigenvalue);
    let (r, iterations, grad_norm) = (
        run.result,
        run.reference_iterations,
        run.reference_grad_norm,
    );
    Ok(PyNevpt2Result {
        e_corr: r.e_corr,
        e_total: r.e_reference + r.e_corr,
        e_reference: r.e_reference,
        reference: on.as_str().to_string(),
        rhf_energy: rhf.energy,
        reference_iterations: iterations,
        reference_grad_norm: grad_norm,
        reference_stability_lowest_eigenvalue: stability_lambda,
        ci_residual_norm: r.ci_residual_norm,
        n_core: r.n_core,
        n_active: r.n_active,
        n_virtual: r.n_virtual,
        energies: r.energies,
        norms: r.norms,
        mo_coeff: r.mo_coeff,
        mo_energy: r.mo_energy,
        certificate_data: cert,
    })
}

/// Register `run_casci`, `run_casscf`, `run_nevpt2` and their result classes.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_casci, m)?)?;
    m.add_class::<PyCasCiResult>()?;
    m.add_function(wrap_pyfunction!(run_casscf, m)?)?;
    m.add_class::<PyCasScfResult>()?;
    m.add_function(wrap_pyfunction!(run_nevpt2, m)?)?;
    m.add_class::<PyNevpt2Result>()
}
