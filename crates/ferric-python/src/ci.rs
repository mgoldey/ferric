//! `ferric.run_casci`: CAS-CI (`ferric_ci::run_cas_ci`) with spin-summed
//! 1- and 2-RDMs, and `ferric.run_casscf`: two-step CASSCF
//! (`ferric_ci::run_casscf`), both on a closed-shell exact-J/K RHF reference.

use crate::{
    budget_bytes_from_gb, certificate_dict, make_err, rhf_config_conv, PyBasisSet, PyMolecule,
    ScfConv,
};
use ferric_core::certificate::{BuildStamp, Certificate};
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::solve_rhf;
use ferric_scf::screening::SchwarzBounds;
use ndarray::{Array2, Array4};
use numpy::{IntoPyArray, PyArray2, PyArray4};
use pyo3::prelude::*;

/// Library capability name on the certificate (no CLI `method.kind`).
const CAPABILITY: &str = "casci";

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
    // The valence count and V_ECP of an ECP basis, as the SCF bindings use.
    let mut emol = mol.inner.clone();
    emol.apply_ecp(&basis_set.inner);
    let mol = &emol;
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
    let prep = PreparedBasis::new(mol, &basis_set.inner).map_err(make_err)?;
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).map_err(make_err)?;
    let rhf_config = rhf_config_conv(
        None,
        budget,
        ScfConv::new(scf_max_iter, scf_energy_conv, scf_density_conv),
    );
    let rhf = solve_rhf(
        &ParallelContext::default(),
        mol,
        &prep,
        op,
        &bounds,
        &rhf_config,
    )
    .map_err(make_err)?;
    if !rhf.converged {
        return Err(make_err(ferric_core::FerricError::ScfConvergence {
            iterations: rhf.iterations,
            last_energy: rhf.energy,
        }));
    }
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
    let r = ferric_ci::run_cas_ci(mol, &prep, &rhf, cfg).map_err(make_err)?;

    let mut cert = Certificate::ungraded(CAPABILITY).with_build(BuildStamp::new(
        ferric_build_info::VERSION,
        ferric_build_info::COMMIT,
        ferric_build_info::DIRTY,
    ));
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

/// Library capability name of `run_casscf` on the certificate.
const CASSCF_CAPABILITY: &str = "casscf";

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
#[pyo3(signature = (mol, basis_set, n_active, n_elec_active, n_core=None, casscf_grad_tol=1e-7, casscf_energy_tol=1e-12, casscf_max_iter=300, casscf_max_step=0.2, ci_conv_thresh=1e-10, ci_max_subspace=50, ci_max_iter=200, memory_budget_gb=None, scf_max_iter=None, scf_energy_conv=None, scf_density_conv=None, canonicalize=false))]
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
) -> PyResult<PyCasScfResult> {
    // The valence count and V_ECP of an ECP basis, as the SCF bindings use.
    let mut emol = mol.inner.clone();
    emol.apply_ecp(&basis_set.inner);
    let mol = &emol;
    let nelec = mol.nelec() as usize;
    let n_act_el = n_elec_active.0 + n_elec_active.1;
    let n_core = match n_core {
        Some(s) => s,
        None => {
            if n_act_el > nelec || !(nelec - n_act_el).is_multiple_of(2) {
                return Err(make_err(format!(
                    "run_casscf: {n_act_el} active electrons do not leave an even core \
                     count out of {nelec}"
                )));
            }
            (nelec - n_act_el) / 2
        }
    };
    let budget = budget_bytes_from_gb(memory_budget_gb);
    let prep = PreparedBasis::new(mol, &basis_set.inner).map_err(make_err)?;
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).map_err(make_err)?;
    let rhf_config = rhf_config_conv(
        None,
        budget,
        ScfConv::new(scf_max_iter, scf_energy_conv, scf_density_conv),
    );
    let rhf = solve_rhf(
        &ParallelContext::default(),
        mol,
        &prep,
        op,
        &bounds,
        &rhf_config,
    )
    .map_err(make_err)?;
    if !rhf.converged {
        return Err(make_err(ferric_core::FerricError::ScfConvergence {
            iterations: rhf.iterations,
            last_energy: rhf.energy,
        }));
    }
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
        ..Default::default()
    };
    let r = ferric_ci::run_casscf(mol, &prep, &rhf, &cfg).map_err(make_err)?;

    let mut cert = Certificate::ungraded(CASSCF_CAPABILITY).with_build(BuildStamp::new(
        ferric_build_info::VERSION,
        ferric_build_info::COMMIT,
        ferric_build_info::DIRTY,
    ));
    if !r.converged {
        let msg = format!(
            "CASSCF did not converge in {} macro iterations (|g| = {:.2e}, \
             casscf_grad_tol {casscf_grad_tol:e}, casscf_energy_tol {casscf_energy_tol:e})",
            r.iterations, r.grad_norm
        );
        crate::emit_runtime_warning(py, msg.clone());
        cert.push_warning(msg);
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

/// Register `run_casci`, `run_casscf` and their result classes.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_casci, m)?)?;
    m.add_class::<PyCasCiResult>()?;
    m.add_function(wrap_pyfunction!(run_casscf, m)?)?;
    m.add_class::<PyCasScfResult>()
}
