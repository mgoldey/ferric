//! Python bindings for ferric (pyo3).
//!
//! Exposes the engine to Python: `Molecule` / `BasisSet` constructors plus
//! `run_rhf`, `run_rimp2`, `run_attenuated_rimp2`, `run_scs_mp2`, the Laplace and
//! coupled-cluster drivers, and geometry optimization. Each binding wraps the
//! corresponding Rust driver and returns a result object with energies/components.
//! Build with `uv run maturin develop --release` (see the README for the venv
//! caveat).

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::attenuated::{attenuated_ri_mp2, AttenuatedMp2Config};
use ferric_mp2::laplace::laplace_ri_mp2;
use ferric_mp2::rimp2::{ri_mp2, RiMp2Config};
use ferric_mp2::scs::{scs_mp2, ScsMp2Config};
use ferric_scf::ks_gradient::ks_gradient_closed;
use ferric_scf::optimize::{optimize_geometry, OptimizeConfig};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_cc::ccd::ccd as run_ccd_inner;
use ferric_cc::ccsd::ccsd as run_ccsd_inner;
use ferric_cc::ccsd_t::ccsd_t as run_ccsd_t_inner;
use ferric_cc::CcConfig;
use ndarray::Array2;
use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;

// ── Molecule ──

#[pyclass]
#[pyo3(name = "Molecule")]
struct PyMolecule { inner: Molecule }

#[pymethods]
impl PyMolecule {
    #[staticmethod]
    fn from_xyz(path: &str) -> PyResult<Self> {
        let mol = Molecule::load_xyz(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{e}")))?;
        Ok(PyMolecule { inner: mol })
    }
    #[staticmethod]
    #[pyo3(signature = (s, charge=0, multiplicity=1))]
    fn from_xyz_string(s: &str, charge: i32, multiplicity: usize) -> PyResult<Self> {
        let mol = Molecule::parse_xyz(s, charge, multiplicity)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))?;
        Ok(PyMolecule { inner: mol })
    }
    fn nuclear_repulsion(&self) -> f64 { self.inner.nuclear_repulsion() }
    fn natoms(&self) -> usize { self.inner.atoms.len() }
    fn nelec(&self) -> i32 { self.inner.nelec() }
}

// ── BasisSet ──

#[pyclass]
#[pyo3(name = "BasisSet")]
struct PyBasisSet { inner: ferric_core::basis::BasisSet }

#[pymethods]
impl PyBasisSet {
    #[staticmethod]
    fn bundled(name: &str) -> PyResult<Self> {
        let bs = basis::bundled(name)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))?;
        Ok(PyBasisSet { inner: bs })
    }
}

// ── Helper ──

fn make_err(e: impl std::fmt::Display) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(format!("{e}"))
}

fn rhf_config(k_builder: Option<&str>) -> RhfConfig {
    RhfConfig { k_builder: k_builder.map(|s| s.to_string()), ..Default::default() }
}

// ── RHF ──

#[pyclass]
#[pyo3(name = "RhfResult")]
struct PyRhfResult {
    #[pyo3(get)] energy: f64,
    #[pyo3(get)] converged: bool,
    #[pyo3(get)] iterations: usize,
    #[pyo3(get)] computed_quartets: usize,
    density_data: Array2<f64>,
    orbital_energies_data: Vec<f64>,
}

#[pymethods]
impl PyRhfResult {
    fn density<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        PyArray2::from_array(py, &self.density_data)
    }
    fn orbital_energies<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.orbital_energies_data.clone())
    }
}

#[pyfunction]
#[pyo3(signature = (mol, basis_set, max_iter=None, energy_conv=None, k_builder=None))]
fn run_rhf(
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    max_iter: Option<usize>,
    energy_conv: Option<f64>,
    k_builder: Option<&str>,
) -> PyResult<PyRhfResult> {
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).map_err(make_err)?;
    let mut config = RhfConfig {
        max_iter: max_iter.unwrap_or(100),
        energy_conv: energy_conv.unwrap_or(1e-8),
        ..Default::default()
    };
    if let Some(kb) = k_builder {
        config.k_builder = Some(kb.to_string());
    }
    let ctx = ParallelContext::default();
    let r = solve_rhf(&ctx, &mol.inner, &prep, op, &bounds, &config).map_err(make_err)?;
    Ok(PyRhfResult {
        energy: r.energy, converged: r.converged, iterations: r.iterations,
        computed_quartets: r.computed_quartets,
        density_data: r.density_total, orbital_energies_data: r.eps_alpha,
    })
}

// ── Optimize ──

#[pyclass]
#[pyo3(name = "OptimizeResult")]
struct PyOptimizeResult {
    #[pyo3(get)] energy: f64,
    #[pyo3(get)] converged: bool,
    #[pyo3(get)] steps: usize,
}

#[pyfunction]
#[pyo3(signature = (mol, basis_name, max_steps=None, e_conv=None))]
fn run_optimize(mol: &PyMolecule, basis_name: &str,
                max_steps: Option<usize>, e_conv: Option<f64>) -> PyResult<PyOptimizeResult> {
    let ctx = ParallelContext::default();
    let r = optimize_geometry(&ctx, &mol.inner, basis_name, Operator::coulomb(),
                              &RhfConfig::default(),
                              &OptimizeConfig {
                                  max_steps: max_steps.unwrap_or(100),
                                  e_conv: e_conv.unwrap_or(1e-6),
                                  ..Default::default()
                              }).map_err(make_err)?;
    Ok(PyOptimizeResult { energy: r.energy, converged: r.converged, steps: r.steps })
}

// ── RI-MP2 ──

#[pyclass]
#[pyo3(name = "RiMp2Result")]
struct PyRiMp2Result {
    #[pyo3(get)] total_energy: f64,
    #[pyo3(get)] rhf_energy: f64,
    #[pyo3(get)] mp2_corr: f64,
}

#[pyfunction]
#[pyo3(signature = (mol, basis_set, auxbasis, frozen_core=None, k_builder=None))]
fn run_rimp2(mol: &PyMolecule, basis_set: &PyBasisSet, auxbasis: &PyBasisSet,
             frozen_core: Option<usize>, k_builder: Option<&str>) -> PyResult<PyRiMp2Result> {
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    let dfbs = PreparedBasis::new(&mol.inner, &auxbasis.inner).map_err(make_err)?;
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).map_err(make_err)?;
    let ctx = ParallelContext::default();
    let rhf = solve_rhf(&ctx, &mol.inner, &prep, op, &bounds, &rhf_config(k_builder)).map_err(make_err)?;
    let mp2 = ri_mp2(&mol.inner, &prep, &dfbs, op, &rhf,
                      &RiMp2Config { frozen_core: frozen_core.unwrap_or(0) }).map_err(make_err)?;
    Ok(PyRiMp2Result { total_energy: mp2.total_energy, rhf_energy: rhf.energy, mp2_corr: mp2.mp2_corr })
}

// ── Laplace RI-MP2 ──

#[pyclass]
#[pyo3(name = "LaplaceMp2Result")]
struct PyLaplaceMp2Result {
    #[pyo3(get)] total_energy: f64,
    #[pyo3(get)] mp2_corr: f64,
    #[pyo3(get)] e_os: f64,
    #[pyo3(get)] e_ss: f64,
}

#[pyfunction]
#[pyo3(signature = (mol, basis_set, auxbasis, n_quad=None, frozen_core=None, k_builder=None))]
fn run_laplace_mp2(mol: &PyMolecule, basis_set: &PyBasisSet, auxbasis: &PyBasisSet,
                   n_quad: Option<usize>, frozen_core: Option<usize>,
                   k_builder: Option<&str>) -> PyResult<PyLaplaceMp2Result> {
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    let dfbs = PreparedBasis::new(&mol.inner, &auxbasis.inner).map_err(make_err)?;
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).map_err(make_err)?;
    let ctx = ParallelContext::default();
    let rhf = solve_rhf(&ctx, &mol.inner, &prep, op, &bounds, &rhf_config(k_builder)).map_err(make_err)?;
    let r = laplace_ri_mp2(&mol.inner, &prep, &dfbs, op, &rhf,
                           n_quad.unwrap_or(7), frozen_core.unwrap_or(0)).map_err(make_err)?;
    Ok(PyLaplaceMp2Result { 
        total_energy: r.total_energy, 
        mp2_corr: r.mp2_corr,
        e_os: r.e_os,
        e_ss: r.e_ss,
    })
}

// ── Attenuated RI-MP2 ──

#[pyclass]
#[pyo3(name = "AttenuatedMp2Result")]
struct PyAttenuatedMp2Result {
    #[pyo3(get)] total_energy: f64,
    #[pyo3(get)] rhf_energy: f64,
    #[pyo3(get)] mp2_corr: f64,
    #[pyo3(get)] e_os: f64,
    #[pyo3(get)] e_ss: f64,
}

#[pyfunction]
#[pyo3(signature = (mol, basis_set, auxbasis, omega=None, frozen_core=None, k_builder=None))]
fn run_attenuated_rimp2(mol: &PyMolecule, basis_set: &PyBasisSet, auxbasis: &PyBasisSet,
                        omega: Option<f64>, frozen_core: Option<usize>,
                        k_builder: Option<&str>) -> PyResult<PyAttenuatedMp2Result> {
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    let dfbs = PreparedBasis::new(&mol.inner, &auxbasis.inner).map_err(make_err)?;
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).map_err(make_err)?;
    let ctx = ParallelContext::default();
    let rhf = solve_rhf(&ctx, &mol.inner, &prep, op, &bounds, &rhf_config(k_builder)).map_err(make_err)?;
    // omega is supplied in Å⁻¹; convert to Bohr⁻¹ for the operator.
    let cfg = AttenuatedMp2Config {
        omega: omega.unwrap_or(0.420) * ferric_mp2::attenuated::BOHR_INV_PER_ANG_INV,
        scaling: 1.0, frozen_core: frozen_core.unwrap_or(0),
        screen_thresh: None,
    };
    let r = attenuated_ri_mp2(&mol.inner, &prep, &dfbs, &rhf, &cfg).map_err(make_err)?;
    Ok(PyAttenuatedMp2Result {
        total_energy: r.total_energy, rhf_energy: rhf.energy, mp2_corr: r.mp2_corr,
        e_os: r.spin_components.e_os, e_ss: r.spin_components.e_ss,
    })
}

// ── SCS-MP2 ──

#[pyclass]
#[pyo3(name = "ScsMp2Result")]
struct PyScsMp2Result {
    #[pyo3(get)] total_energy: f64,
    #[pyo3(get)] rhf_energy: f64,
    #[pyo3(get)] scs_corr: f64,
    #[pyo3(get)] e_os: f64,
    #[pyo3(get)] e_ss: f64,
}

#[pyfunction]
#[pyo3(signature = (mol, basis_set, auxbasis, c_os=None, c_ss=None, frozen_core=None, k_builder=None))]
fn run_scs_mp2(mol: &PyMolecule, basis_set: &PyBasisSet, auxbasis: &PyBasisSet,
               c_os: Option<f64>, c_ss: Option<f64>, frozen_core: Option<usize>,
               k_builder: Option<&str>) -> PyResult<PyScsMp2Result> {
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    let dfbs = PreparedBasis::new(&mol.inner, &auxbasis.inner).map_err(make_err)?;
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).map_err(make_err)?;
    let ctx = ParallelContext::default();
    let rhf = solve_rhf(&ctx, &mol.inner, &prep, op, &bounds, &rhf_config(k_builder)).map_err(make_err)?;
    let cfg = ScsMp2Config {
        c_os: c_os.unwrap_or(6.0 / 5.0), c_ss: c_ss.unwrap_or(1.0 / 3.0),
        frozen_core: frozen_core.unwrap_or(0),
    };
    let r = scs_mp2(&mol.inner, &prep, &dfbs, &rhf, &cfg).map_err(make_err)?;
    Ok(PyScsMp2Result {
        total_energy: r.total_energy, rhf_energy: rhf.energy, scs_corr: r.scs_corr,
        e_os: r.e_os, e_ss: r.e_ss,
    })
}



// ── RS-MP2-RPA (SR-MP2 + LR-dRPA, Δ-form B or coupled-rings T) ──

#[pyclass]
#[pyo3(name = "RsMp2RpaResult")]
struct PyRsMp2RpaResult {
    #[pyo3(get)] total_energy: f64,
    #[pyo3(get)] rhf_energy: f64,
    #[pyo3(get)] e_corr: f64,
    /// Diagnostic naive sum E_MP2[erfc] + E_dRPA[erf] (formulation A).
    /// Only available when `formulation="delta-lr"`; None for coupled-rings.
    #[pyo3(get)] e_corr_naive: Option<f64>,
    #[pyo3(get)] e_mp2_full: f64,
    #[pyo3(get)] e_sr_mp2: f64,
    #[pyo3(get)] e_lr_mp2: f64,
    #[pyo3(get)] e_dmp2_lr: f64,
    /// E_dRPA[erf] (DeltaLr only; None for CoupledRings).
    #[pyo3(get)] e_drpa_lr: Option<f64>,
    /// ΔdRPA[Coulomb] = E_dRPA[Coulomb] − 2·E_OS[Coulomb] (CoupledRings only).
    #[pyo3(get)] e_delta_drpa_full: Option<f64>,
    /// ΔdRPA[erfc] = E_dRPA[erfc] − 2·E_OS[erfc] (CoupledRings only).
    #[pyo3(get)] e_delta_drpa_sr: Option<f64>,
}

#[pyfunction]
#[pyo3(signature = (mol, basis_set, auxbasis, omega=None, frozen_core=None, k_builder=None, formulation=None))]
fn run_rs_mp2_rpa(mol: &PyMolecule, basis_set: &PyBasisSet, auxbasis: &PyBasisSet,
                  omega: Option<f64>, frozen_core: Option<usize>,
                  k_builder: Option<&str>,
                  formulation: Option<&str>) -> PyResult<PyRsMp2RpaResult> {
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    let dfbs = PreparedBasis::new(&mol.inner, &auxbasis.inner).map_err(make_err)?;
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).map_err(make_err)?;
    let ctx = ParallelContext::default();
    let mut cfg_rhf = rhf_config(k_builder);
    // Default RI-J/RI-K to def2-universal-jkfit (same convention as pdep-rpa and
    // run_dft); keeps SCF aux separate from the SR-MP2+LR-RPA correlation aux
    // (see ferric-jk-aux-convention). k_builder from the caller is preserved.
    cfg_rhf.df_j_aux = Some("def2-universal-jkfit".to_string());
    cfg_rhf.df_k_aux = Some("def2-universal-jkfit".to_string());
    let rhf = solve_rhf(&ctx, &mol.inner, &prep, op, &bounds, &cfg_rhf).map_err(make_err)?;
    // Map formulation string to enum.
    let form = match formulation.unwrap_or("delta-lr") {
        "delta-lr" => ferric_rpa::RsMp2RpaFormulation::DeltaLr,
        "coupled-rings" => ferric_rpa::RsMp2RpaFormulation::CoupledRings,
        other => return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown formulation \"{other}\"; expected \"delta-lr\" or \"coupled-rings\""
        ))),
    };
    // omega is supplied in Å⁻¹; convert to Bohr⁻¹ for the operator.
    let cfg = ferric_rpa::RsMp2RpaConfig {
        omega: omega.unwrap_or(0.420) * ferric_mp2::attenuated::BOHR_INV_PER_ANG_INV,
        frozen_core: frozen_core.unwrap_or(0),
        formulation: form,
        ..Default::default()
    };
    let r = ferric_rpa::rs_mp2_lr_rpa(&mol.inner, &prep, &dfbs, &rhf, &cfg)
        .map_err(make_err)?;
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
    })
}

// ── KS-DFT ──

#[pyclass]
#[pyo3(name = "DftResult")]
struct PyDftResult {
    #[pyo3(get)] total_energy: f64,
    vxc_data: Array2<f64>,
    /// Analytic nuclear gradient (natoms × 3) in Ha/Bohr, when computed
    /// (i.e. `with_gradient=True` was passed to `run_ksdft`). Closed-shell
    /// LDA / GGA / hybrid / RSH only; VV10 nonlocal piece is excluded.
    gradient_data: Option<Array2<f64>>,
}

#[pymethods]
impl PyDftResult {
    fn vxc<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        PyArray2::from_array(py, &self.vxc_data)
    }

    /// Return the cached analytic nuclear gradient as an (natoms, 3) array.
    /// Returns `None` if the result was produced without `with_gradient=True`.
    fn gradient<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
        self.gradient_data.as_ref().map(|g| PyArray2::from_array(py, g))
    }
}

#[pyfunction]
#[pyo3(signature = (mol, basis_set, functional=None, k_builder=None, with_gradient=false))]
fn run_dft(mol: &PyMolecule, basis_set: &PyBasisSet,
           functional: Option<&str>, k_builder: Option<&str>,
           with_gradient: bool) -> PyResult<PyDftResult> {
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).map_err(make_err)?;
    let ctx = ParallelContext::default();
    let mut cfg = rhf_config(k_builder);
    let xc_name = functional.unwrap_or("LDA").to_string();
    cfg.xc = Some(xc_name.clone());
    // RI-J always on (matches PySCF density_fit reference convention).
    cfg.df_j_aux = Some("def2-universal-jkfit".to_string());
    // RI-K only matters for hybrid/RSH; harmless for pure DFT (path is bypassed
    // when k_mix.sr == 0 and k_mix.omega == 0).
    cfg.df_k_aux = Some("def2-universal-jkfit".to_string());
    let rhf = solve_rhf(&ctx, &mol.inner, &prep, op, &bounds, &cfg).map_err(make_err)?;
    let nbf = rhf.mos_alpha.nrows();
    let gradient_data = if with_gradient {
        Some(
            ks_gradient_closed(&mol.inner, &prep, &basis_set.inner, op, &bounds, &xc_name, &rhf)
                .map_err(make_err)?
        )
    } else {
        None
    };
    Ok(PyDftResult {
        total_energy: rhf.energy,
        vxc_data: Array2::<f64>::zeros((nbf, nbf)),
        gradient_data,
    })
}

/// Alias under the spec's canonical name. Same surface as `run_dft`.
#[pyfunction]
#[pyo3(signature = (mol, basis_set, functional=None, k_builder=None, with_gradient=false))]
fn run_ksdft(mol: &PyMolecule, basis_set: &PyBasisSet,
             functional: Option<&str>, k_builder: Option<&str>,
             with_gradient: bool) -> PyResult<PyDftResult> {
    run_dft(mol, basis_set, functional, k_builder, with_gradient)
}

// ── CC (stub) ──

#[pyclass]
#[pyo3(name = "CcResult")]
struct PyCcResult {
    #[pyo3(get)] correlation_energy: f64,
    #[pyo3(get)] t_correction: Option<f64>,
}

#[pyfunction]
#[pyo3(signature = (mol, basis_set, auxbasis, frozen_core=None, k_builder=None))]
fn run_ccd(mol: &PyMolecule, basis_set: &PyBasisSet, auxbasis: &PyBasisSet,
           frozen_core: Option<usize>, k_builder: Option<&str>) -> PyResult<PyCcResult> {
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    let dfbs = PreparedBasis::new(&mol.inner, &auxbasis.inner).map_err(make_err)?;
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).map_err(make_err)?;
    let ctx = ParallelContext::default();
    let rhf = solve_rhf(&ctx, &mol.inner, &prep, op, &bounds, &rhf_config(k_builder)).map_err(make_err)?;
    let cfg = CcConfig { frozen_core: frozen_core.unwrap_or(0), ..Default::default() };
    let r = run_ccd_inner(&mol.inner, &prep, &dfbs, op, &rhf, &cfg).map_err(make_err)?;
    Ok(PyCcResult { correlation_energy: r.correlation_energy, t_correction: None })
}

#[pyfunction]
#[pyo3(signature = (mol, basis_set, auxbasis, frozen_core=None, k_builder=None))]
fn run_ccsd(mol: &PyMolecule, basis_set: &PyBasisSet, auxbasis: &PyBasisSet,
            frozen_core: Option<usize>, k_builder: Option<&str>) -> PyResult<PyCcResult> {
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    let dfbs = PreparedBasis::new(&mol.inner, &auxbasis.inner).map_err(make_err)?;
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).map_err(make_err)?;
    let ctx = ParallelContext::default();
    let rhf = solve_rhf(&ctx, &mol.inner, &prep, op, &bounds, &rhf_config(k_builder)).map_err(make_err)?;
    let cfg = CcConfig { frozen_core: frozen_core.unwrap_or(0), ..Default::default() };
    let r = run_ccsd_inner(&mol.inner, &prep, &dfbs, op, &rhf, &cfg).map_err(make_err)?;
    Ok(PyCcResult { correlation_energy: r.correlation_energy, t_correction: None })
}

#[pyfunction]
#[pyo3(signature = (mol, basis_set, auxbasis, frozen_core=None, k_builder=None))]
fn run_ccsd_t(mol: &PyMolecule, basis_set: &PyBasisSet, auxbasis: &PyBasisSet,
              frozen_core: Option<usize>, k_builder: Option<&str>) -> PyResult<PyCcResult> {
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    let dfbs = PreparedBasis::new(&mol.inner, &auxbasis.inner).map_err(make_err)?;
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).map_err(make_err)?;
    let ctx = ParallelContext::default();
    let rhf = solve_rhf(&ctx, &mol.inner, &prep, op, &bounds, &rhf_config(k_builder)).map_err(make_err)?;
    let cfg = CcConfig { frozen_core: frozen_core.unwrap_or(0), ..Default::default() };
    let r = run_ccsd_inner(&mol.inner, &prep, &dfbs, op, &rhf, &cfg).map_err(make_err)?;
    let e_t = run_ccsd_t_inner(&mol.inner, &prep, &dfbs, op, &rhf, &r, &cfg).map_err(make_err)?;
    Ok(PyCcResult { correlation_energy: r.correlation_energy, t_correction: Some(e_t) })
}

// ── PDEP-RPA ──

#[pyclass]
#[pyo3(name = "PdepRpaResult")]
struct PyPdepRpaResult {
    #[pyo3(get)] rhf_energy: f64,
    #[pyo3(get)] e_rpa: f64,
    #[pyo3(get)] total_energy: f64,
    #[pyo3(get)] n_eigenpotentials: usize,
    #[pyo3(get)] e_rpa_dft_diag: Option<f64>,
    eigenvalues_static: Vec<f64>,
    eigenpotentials: Array2<f64>,
    quad_freqs: Vec<f64>,
    quad_weights: Vec<f64>,
    eigenvalues_freq: Array2<f64>,
}

#[pymethods]
impl PyPdepRpaResult {
    /// Static dielectric eigenvalues λ_α(0), length M (sorted descending).
    /// Plot `λ - 1` vs α on a log scale for a scree plot of the PDEP basis.
    #[getter]
    fn eigenvalues_static<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.eigenvalues_static)
    }
    /// PDEP eigenpotential coefficients in the RI auxiliary basis, shape (naux, M).
    /// Column α gives c_α^P such that V_α(r) = Σ_P c_α^P χ_P(r).
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
        let deviations: Vec<f64> = self.eigenvalues_static.iter()
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
}

#[pyfunction]
#[pyo3(signature = (
    mol, basis_set, auxbasis,
    frozen_core=None, n_quad=None, quadrature=None, u0=None,
    trunc_thresh=None, davidson_conv_thresh=None,
    run_diagnostics=false, k_builder=None,
))]
#[allow(clippy::too_many_arguments)]
fn run_pdep_rpa(
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    auxbasis: &PyBasisSet,
    frozen_core: Option<usize>,
    n_quad: Option<usize>,
    quadrature: Option<&str>,
    u0: Option<f64>,
    trunc_thresh: Option<f64>,
    davidson_conv_thresh: Option<f64>,
    run_diagnostics: bool,
    k_builder: Option<&str>,
) -> PyResult<PyPdepRpaResult> {
    use ferric_rpa::config::{QuadratureConfig, QuadratureScheme, SternheimerConfig};
    use ferric_rpa::{run_pdep_rpa as run_pdep_rpa_inner, PdepRpaConfig};

    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    let dfbs = PreparedBasis::new(&mol.inner, &auxbasis.inner).map_err(make_err)?;
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).map_err(make_err)?;
    let ctx = ParallelContext::default();
    let rhf = solve_rhf(&ctx, &mol.inner, &prep, op, &bounds, &rhf_config(k_builder)).map_err(make_err)?;

    let scheme = match quadrature.unwrap_or("gauss-legendre") {
        "minimax" | "mm" => QuadratureScheme::MiniMax,
        _ => QuadratureScheme::GaussLegendre,
    };
    let cfg = PdepRpaConfig {
        frozen_core: frozen_core.unwrap_or(0),
        trunc_thresh: trunc_thresh.unwrap_or(1e-4),
        davidson_max_vecs: 0,
        davidson_conv_thresh: davidson_conv_thresh.unwrap_or(1e-6),
        quadrature: QuadratureConfig {
            scheme,
            n_points: n_quad.unwrap_or(40),
            u0: u0.unwrap_or(0.5),
        },
        sternheimer: SternheimerConfig::default(),
        run_diagnostics,
        eigensolver: ferric_rpa::Eigensolver::default(),
        chi0_backend: ferric_rpa::config::Chi0Backend::default(),
        chi0_sparsity: ferric_rpa::config::Chi0Sparsity::default(),
    };
    let r = run_pdep_rpa_inner(&mol.inner, &prep, &dfbs, op, &rhf, &cfg).map_err(make_err)?;
    Ok(PyPdepRpaResult {
        rhf_energy: rhf.energy,
        e_rpa: r.e_rpa,
        total_energy: rhf.energy + r.e_rpa,
        n_eigenpotentials: r.n_eigenpotentials,
        e_rpa_dft_diag: r.e_rpa_dft_diag,
        eigenvalues_static: r.eigenvalues_static,
        eigenpotentials: r.eigenpotentials,
        quad_freqs: r.quad_freqs,
        quad_weights: r.quad_weights,
        eigenvalues_freq: r.eigenvalues_freq,
    })
}

// ── Module ──

#[pymodule]
fn ferric(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register global signal handler for Ctrl+C
    let _ = ctrlc::set_handler(move || {
        eprintln!("\n[Ferric] Interrupt signal caught! Bailing out of kernels...");
        ferric_core::INTERRUPT.store(true, std::sync::atomic::Ordering::SeqCst);
    });

    m.add_class::<PyMolecule>()?;
    m.add_class::<PyBasisSet>()?;
    m.add_class::<PyRhfResult>()?;
    m.add_class::<PyOptimizeResult>()?;
    m.add_class::<PyRiMp2Result>()?;
    m.add_class::<PyAttenuatedMp2Result>()?;
    m.add_class::<PyScsMp2Result>()?;
    m.add_class::<PyLaplaceMp2Result>()?;
    m.add_class::<PyDftResult>()?;
    m.add_class::<PyCcResult>()?;
    m.add_class::<PyPdepRpaResult>()?;
    m.add_class::<PyRsMp2RpaResult>()?;
    m.add_function(wrap_pyfunction!(run_rhf, m)?)?;
    m.add_function(wrap_pyfunction!(run_optimize, m)?)?;
    m.add_function(wrap_pyfunction!(run_rimp2, m)?)?;
    m.add_function(wrap_pyfunction!(run_attenuated_rimp2, m)?)?;
    m.add_function(wrap_pyfunction!(run_scs_mp2, m)?)?;

    m.add_function(wrap_pyfunction!(run_laplace_mp2, m)?)?;
    m.add_function(wrap_pyfunction!(run_dft, m)?)?;
    m.add_function(wrap_pyfunction!(run_ksdft, m)?)?;
    m.add_function(wrap_pyfunction!(run_ccd, m)?)?;
    m.add_function(wrap_pyfunction!(run_ccsd, m)?)?;
    m.add_function(wrap_pyfunction!(run_ccsd_t, m)?)?;
    m.add_function(wrap_pyfunction!(run_pdep_rpa, m)?)?;
    m.add_function(wrap_pyfunction!(run_rs_mp2_rpa, m)?)?;
    Ok(())
}
