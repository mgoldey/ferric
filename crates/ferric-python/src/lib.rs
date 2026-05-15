use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::attenuated::{attenuated_ri_mp2, AttenuatedMp2Config};
use ferric_mp2::laplace::laplace_ri_mp2;
use ferric_mp2::rimp2::{ri_mp2, RiMp2Config};
use ferric_mp2::scs::{scs_mp2, scs_mp2_2terfc, ScsMp2Config, ScsMp2TerfcConfig};
use ferric_scf::optimize::{optimize_geometry, OptimizeConfig};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_dft::{dft, DftConfig};
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
    fn from_xyz_string(s: &str) -> PyResult<Self> {
        let mol = Molecule::parse_xyz(s)
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
        density_data: r.density, orbital_energies_data: r.orbital_energies,
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
#[pyo3(signature = (mol, basis_set, auxbasis, r0=None, frozen_core=None, k_builder=None))]
fn run_attenuated_rimp2(mol: &PyMolecule, basis_set: &PyBasisSet, auxbasis: &PyBasisSet,
                        r0: Option<f64>, frozen_core: Option<usize>,
                        k_builder: Option<&str>) -> PyResult<PyAttenuatedMp2Result> {
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    let dfbs = PreparedBasis::new(&mol.inner, &auxbasis.inner).map_err(make_err)?;
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).map_err(make_err)?;
    let ctx = ParallelContext::default();
    let rhf = solve_rhf(&ctx, &mol.inner, &prep, op, &bounds, &rhf_config(k_builder)).map_err(make_err)?;
    let cfg = AttenuatedMp2Config {
        r0: r0.unwrap_or(1.05) * 1.8897259886,
        scaling: 1.0, frozen_core: frozen_core.unwrap_or(0),
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

#[pyfunction]
#[pyo3(signature = (mol, basis_set, auxbasis, r0_bonded=None, r0_nonbonded=None, c_os=None, c_ss=None, frozen_core=None, k_builder=None))]
fn run_scs_mp2_2terfc(mol: &PyMolecule, basis_set: &PyBasisSet, auxbasis: &PyBasisSet,
                      r0_bonded: Option<f64>, r0_nonbonded: Option<f64>,
                      c_os: Option<f64>, c_ss: Option<f64>,
                      frozen_core: Option<usize>, k_builder: Option<&str>) -> PyResult<PyScsMp2Result> {
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    let dfbs = PreparedBasis::new(&mol.inner, &auxbasis.inner).map_err(make_err)?;
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).map_err(make_err)?;
    let ctx = ParallelContext::default();
    let rhf = solve_rhf(&ctx, &mol.inner, &prep, op, &bounds, &rhf_config(k_builder)).map_err(make_err)?;
    let a2b = 1.8897259886;
    let cfg = ScsMp2TerfcConfig {
        r0_bonded: r0_bonded.unwrap_or(0.75) * a2b,
        r0_nonbonded: r0_nonbonded.unwrap_or(1.05) * a2b,
        c_os: c_os.unwrap_or(1.27), c_ss: c_ss.unwrap_or(4.05),
        frozen_core: frozen_core.unwrap_or(0),
    };
    let r = scs_mp2_2terfc(&mol.inner, &prep, &dfbs, &rhf, &cfg).map_err(make_err)?;
    Ok(PyScsMp2Result {
        total_energy: r.total_energy, rhf_energy: rhf.energy, scs_corr: r.scs_corr,
        e_os: r.e_os, e_ss: r.e_ss,
    })
}

// ── DFT (stub) ──

#[pyclass]
#[pyo3(name = "DftResult")]
struct PyDftResult {
    #[pyo3(get)] total_energy: f64,
    vxc_data: Array2<f64>,
}

#[pymethods]
impl PyDftResult {
    fn vxc<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        PyArray2::from_array(py, &self.vxc_data)
    }
}

#[pyfunction]
#[pyo3(signature = (mol, basis_set, functional=None, k_builder=None))]
fn run_dft(mol: &PyMolecule, basis_set: &PyBasisSet,
           functional: Option<&str>, k_builder: Option<&str>) -> PyResult<PyDftResult> {
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).map_err(make_err)?;
    let ctx = ParallelContext::default();
    let rhf = solve_rhf(&ctx, &mol.inner, &prep, op, &bounds, &rhf_config(k_builder)).map_err(make_err)?;
    let cfg = DftConfig { functional: functional.unwrap_or("LDA_X").to_string(), grid_spacing: 0.2 };
    let r = dft(&rhf.density, &cfg).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;
    Ok(PyDftResult { total_energy: rhf.energy + r.total_energy, vxc_data: r.vxc })
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
    m.add_function(wrap_pyfunction!(run_rhf, m)?)?;
    m.add_function(wrap_pyfunction!(run_optimize, m)?)?;
    m.add_function(wrap_pyfunction!(run_rimp2, m)?)?;
    m.add_function(wrap_pyfunction!(run_attenuated_rimp2, m)?)?;
    m.add_function(wrap_pyfunction!(run_scs_mp2, m)?)?;
    m.add_function(wrap_pyfunction!(run_scs_mp2_2terfc, m)?)?;
    m.add_function(wrap_pyfunction!(run_laplace_mp2, m)?)?;
    m.add_function(wrap_pyfunction!(run_dft, m)?)?;
    m.add_function(wrap_pyfunction!(run_ccd, m)?)?;
    m.add_function(wrap_pyfunction!(run_ccsd, m)?)?;
    m.add_function(wrap_pyfunction!(run_ccsd_t, m)?)?;
    Ok(())
}
