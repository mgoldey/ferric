use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::attenuated::{attenuated_ri_mp2, AttenuatedMp2Config};
use ferric_mp2::rimp2::{ri_mp2, RiMp2Config};
use ferric_mp2::scs::{scs_mp2, scs_mp2_2terfc, ScsMp2Config, ScsMp2TerfcConfig};
use ferric_scf::optimize::{optimize_geometry, OptimizeConfig};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;
use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;

#[pyclass]
#[pyo3(name = "Molecule")]
struct PyMolecule {
    inner: Molecule,
}

#[pymethods]
impl PyMolecule {
    #[staticmethod]
    fn from_xyz(path: &str) -> PyResult<Self> {
        let mol = Molecule::load_xyz(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{e}")))?;
        Ok(PyMolecule { inner: mol })
    }

    fn nuclear_repulsion(&self) -> f64 {
        self.inner.nuclear_repulsion()
    }

    fn natoms(&self) -> usize {
        self.inner.atoms.len()
    }

    fn nelec(&self) -> i32 {
        self.inner.nelec()
    }
}

#[pyclass]
#[pyo3(name = "BasisSet")]
struct PyBasisSet {
    inner: ferric_core::basis::BasisSet,
}

#[pymethods]
impl PyBasisSet {
    #[staticmethod]
    fn bundled(name: &str) -> PyResult<Self> {
        let bs = basis::bundled(name)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))?;
        Ok(PyBasisSet { inner: bs })
    }
}

#[pyclass]
#[pyo3(name = "RhfResult")]
struct PyRhfResult {
    #[pyo3(get)]
    energy: f64,
    #[pyo3(get)]
    converged: bool,
    #[pyo3(get)]
    iterations: usize,
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
#[pyo3(signature = (mol, basis_set, max_iter=None, energy_conv=None))]
fn run_rhf(
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    max_iter: Option<usize>,
    energy_conv: Option<f64>,
) -> PyResult<PyRhfResult> {
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;
    let config = RhfConfig {
        max_iter: max_iter.unwrap_or(100),
        energy_conv: energy_conv.unwrap_or(1e-8),
        ..Default::default()
    };
    let ctx = ParallelContext::default();
    let result = solve_rhf(&ctx, &mol.inner, &prep, op, &bounds, &config)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;
    Ok(PyRhfResult {
        energy: result.energy,
        converged: result.converged,
        iterations: result.iterations,
        density_data: result.density,
        orbital_energies_data: result.orbital_energies,
    })
}

#[pyclass]
#[pyo3(name = "OptimizeResult")]
struct PyOptimizeResult {
    #[pyo3(get)]
    energy: f64,
    #[pyo3(get)]
    converged: bool,
    #[pyo3(get)]
    steps: usize,
}

#[pyfunction]
#[pyo3(signature = (mol, basis_name, max_steps=None, e_conv=None))]
fn run_optimize(
    mol: &PyMolecule,
    basis_name: &str,
    max_steps: Option<usize>,
    e_conv: Option<f64>,
) -> PyResult<PyOptimizeResult> {
    let op = Operator::coulomb();
    let rhf_config = RhfConfig::default();
    let opt_config = OptimizeConfig {
        max_steps: max_steps.unwrap_or(100),
        e_conv: e_conv.unwrap_or(1e-6),
        ..Default::default()
    };
    let ctx = ParallelContext::default();
    let result = optimize_geometry(&ctx, &mol.inner, basis_name, op, &rhf_config, &opt_config)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;
    Ok(PyOptimizeResult {
        energy: result.energy,
        converged: result.converged,
        steps: result.steps,
    })
}

#[pyclass]
#[pyo3(name = "RiMp2Result")]
struct PyRiMp2Result {
    #[pyo3(get)]
    total_energy: f64,
    #[pyo3(get)]
    rhf_energy: f64,
    #[pyo3(get)]
    mp2_corr: f64,
}

#[pyfunction]
#[pyo3(signature = (mol, basis_set, auxbasis, frozen_core=None))]
fn run_rimp2(
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    auxbasis: &PyBasisSet,
    frozen_core: Option<usize>,
) -> PyResult<PyRiMp2Result> {
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;
    let dfbs = PreparedBasis::new(&mol.inner, &auxbasis.inner)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;
    let ctx = ParallelContext::default();
    let rhf_config = RhfConfig::default();
    let rhf_result = solve_rhf(&ctx, &mol.inner, &prep, op, &bounds, &rhf_config)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;
    let mp2_config = RiMp2Config {
        frozen_core: frozen_core.unwrap_or(0),
    };
    let mp2_result = ri_mp2(&mol.inner, &prep, &dfbs, op, &rhf_result, &mp2_config)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;
    Ok(PyRiMp2Result {
        total_energy: mp2_result.total_energy,
        rhf_energy: rhf_result.energy,
        mp2_corr: mp2_result.mp2_corr,
    })
}

#[pyclass]
#[pyo3(name = "AttenuatedMp2Result")]
struct PyAttenuatedMp2Result {
    #[pyo3(get)]
    total_energy: f64,
    #[pyo3(get)]
    rhf_energy: f64,
    #[pyo3(get)]
    mp2_corr: f64,
    #[pyo3(get)]
    e_os: f64,
    #[pyo3(get)]
    e_ss: f64,
}

#[pyfunction]
#[pyo3(signature = (mol, basis_set, auxbasis, r0=None, frozen_core=None))]
fn run_attenuated_rimp2(
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    auxbasis: &PyBasisSet,
    r0: Option<f64>,
    frozen_core: Option<usize>,
) -> PyResult<PyAttenuatedMp2Result> {
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;
    let dfbs = PreparedBasis::new(&mol.inner, &auxbasis.inner)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;
    let ctx = ParallelContext::default();
    let rhf_result = solve_rhf(&ctx, &mol.inner, &prep, op, &bounds, &RhfConfig::default())
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;
    let angstrom_to_bohr = 1.8897259886;
    let config = AttenuatedMp2Config {
        r0: r0.unwrap_or(1.05) * angstrom_to_bohr,
        scaling: 1.0,
        frozen_core: frozen_core.unwrap_or(0),
    };
    let result = attenuated_ri_mp2(&mol.inner, &prep, &dfbs, &rhf_result, &config)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;
    Ok(PyAttenuatedMp2Result {
        total_energy: result.total_energy,
        rhf_energy: rhf_result.energy,
        mp2_corr: result.mp2_corr,
        e_os: result.spin_components.e_os,
        e_ss: result.spin_components.e_ss,
    })
}

#[pyclass]
#[pyo3(name = "ScsMp2Result")]
struct PyScsMp2Result {
    #[pyo3(get)]
    total_energy: f64,
    #[pyo3(get)]
    rhf_energy: f64,
    #[pyo3(get)]
    scs_corr: f64,
    #[pyo3(get)]
    e_os: f64,
    #[pyo3(get)]
    e_ss: f64,
}

#[pyfunction]
#[pyo3(signature = (mol, basis_set, auxbasis, c_os=None, c_ss=None, frozen_core=None))]
fn run_scs_mp2(
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    auxbasis: &PyBasisSet,
    c_os: Option<f64>,
    c_ss: Option<f64>,
    frozen_core: Option<usize>,
) -> PyResult<PyScsMp2Result> {
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;
    let dfbs = PreparedBasis::new(&mol.inner, &auxbasis.inner)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;
    let ctx = ParallelContext::default();
    let rhf_result = solve_rhf(&ctx, &mol.inner, &prep, op, &bounds, &RhfConfig::default())
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;
    let config = ScsMp2Config {
        c_os: c_os.unwrap_or(6.0 / 5.0),
        c_ss: c_ss.unwrap_or(1.0 / 3.0),
        frozen_core: frozen_core.unwrap_or(0),
    };
    let result = scs_mp2(&mol.inner, &prep, &dfbs, &rhf_result, &config)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;
    Ok(PyScsMp2Result {
        total_energy: result.total_energy,
        rhf_energy: rhf_result.energy,
        scs_corr: result.scs_corr,
        e_os: result.e_os,
        e_ss: result.e_ss,
    })
}

#[pyfunction]
#[pyo3(signature = (mol, basis_set, auxbasis, r0_bonded=None, r0_nonbonded=None, c_os=None, c_ss=None, frozen_core=None))]
fn run_scs_mp2_2terfc(
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    auxbasis: &PyBasisSet,
    r0_bonded: Option<f64>,
    r0_nonbonded: Option<f64>,
    c_os: Option<f64>,
    c_ss: Option<f64>,
    frozen_core: Option<usize>,
) -> PyResult<PyScsMp2Result> {
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;
    let dfbs = PreparedBasis::new(&mol.inner, &auxbasis.inner)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;
    let ctx = ParallelContext::default();
    let rhf_result = solve_rhf(&ctx, &mol.inner, &prep, op, &bounds, &RhfConfig::default())
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;
    let angstrom_to_bohr = 1.8897259886;
    let config = ScsMp2TerfcConfig {
        r0_bonded: r0_bonded.unwrap_or(0.75) * angstrom_to_bohr,
        r0_nonbonded: r0_nonbonded.unwrap_or(1.05) * angstrom_to_bohr,
        c_os: c_os.unwrap_or(1.27),
        c_ss: c_ss.unwrap_or(4.05),
        frozen_core: frozen_core.unwrap_or(0),
    };
    let result = scs_mp2_2terfc(&mol.inner, &prep, &dfbs, &rhf_result, &config)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;
    Ok(PyScsMp2Result {
        total_energy: result.total_energy,
        rhf_energy: rhf_result.energy,
        scs_corr: result.scs_corr,
        e_os: result.e_os,
        e_ss: result.e_ss,
    })
}

#[pymodule]
fn ferric(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMolecule>()?;
    m.add_class::<PyBasisSet>()?;
    m.add_class::<PyRhfResult>()?;
    m.add_class::<PyOptimizeResult>()?;
    m.add_class::<PyRiMp2Result>()?;
    m.add_class::<PyAttenuatedMp2Result>()?;
    m.add_class::<PyScsMp2Result>()?;
    m.add_function(wrap_pyfunction!(run_rhf, m)?)?;
    m.add_function(wrap_pyfunction!(run_optimize, m)?)?;
    m.add_function(wrap_pyfunction!(run_rimp2, m)?)?;
    m.add_function(wrap_pyfunction!(run_attenuated_rimp2, m)?)?;
    m.add_function(wrap_pyfunction!(run_scs_mp2, m)?)?;
    m.add_function(wrap_pyfunction!(run_scs_mp2_2terfc, m)?)?;
    Ok(())
}
