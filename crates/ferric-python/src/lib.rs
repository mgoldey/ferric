use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
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
    let result = solve_rhf(&mol.inner, &prep, op, &bounds, &config)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;
    Ok(PyRhfResult {
        energy: result.energy,
        converged: result.converged,
        iterations: result.iterations,
        density_data: result.density,
        orbital_energies_data: result.orbital_energies,
    })
}

#[pymodule]
fn ferric(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMolecule>()?;
    m.add_class::<PyBasisSet>()?;
    m.add_class::<PyRhfResult>()?;
    m.add_function(wrap_pyfunction!(run_rhf, m)?)?;
    Ok(())
}
