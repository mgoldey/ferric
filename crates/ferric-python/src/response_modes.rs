//! `ferric.response_modes`: eigenmodes of the closed-shell RPA dielectric as
//! auxiliary-basis densities (`ferric_rpa::response_modes`), and
//! `ferric.write_response_mode_cube` to put one on a cube grid.

use crate::{
    budget_bytes_from_gb, cube_grid_spec, make_err, DensitySource, PyBasisSet, PyMolecule,
};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::*;

/// Leading eigenmodes of the RPA dielectric at one imaginary frequency.
#[pyclass]
#[pyo3(name = "ResponseModes")]
pub(crate) struct PyResponseModes {
    inner: ferric_rpa::response_modes::ResponseModes,
}

#[pymethods]
impl PyResponseModes {
    /// Imaginary frequency ω (Hartree); 0 is the static dielectric.
    #[getter]
    fn omega(&self) -> f64 {
        self.inner.omega
    }
    /// Number of auxiliary basis functions.
    #[getter]
    fn naux(&self) -> usize {
        self.inner.naux
    }
    /// Number of modes kept.
    #[getter]
    fn n_modes(&self) -> usize {
        self.inner.eigenvalues.len()
    }
    /// Dielectric eigenvalues λ_α(iω) >= 1, descending.
    #[getter]
    fn eigenvalues<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.eigenvalues)
    }
    /// Aux coefficients (naux, M): column α is the density
    /// ρ_α(r) = Σ_P c_Pα χ_P(r), with unit Coulomb self-energy (cᵀVc = I).
    #[getter]
    fn coefficients<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.coefficients.clone().into_pyarray(py)
    }
    /// ‖ρ_α‖₂ = sqrt(∫ρ_α² dr), length M.
    #[getter]
    fn l2_norms<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.l2_norms)
    }
    /// Field couplings s (M, 3): Coulomb overlap of each mode with the bare
    /// induced density of a unit field along x, y, z.
    #[getter]
    fn field_couplings<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.field_couplings.clone().into_pyarray(py)
    }
    /// Uncoupled polarizability 4 μᵀ D_ω μ (3, 3), atomic units.
    #[getter]
    fn alpha_bare<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        to_3x3(py, self.inner.alpha_bare)
    }
    /// alpha_bare − Σ_α s_α s_αᵀ / λ_α (3, 3): the RPA polarizability at iω
    /// when every mode was kept, its truncation otherwise.
    fn polarizability_from_modes<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        to_3x3(py, self.inner.polarizability_from_modes())
    }
    fn __repr__(&self) -> String {
        format!(
            "ResponseModes(n_modes={}, naux={}, omega={})",
            self.inner.eigenvalues.len(),
            self.inner.naux,
            self.inner.omega
        )
    }
}

fn to_3x3(py: Python<'_>, a: [[f64; 3]; 3]) -> Bound<'_, PyArray2<f64>> {
    ndarray::Array2::from_shape_fn((3, 3), |(i, j)| a[i][j]).into_pyarray(py)
}

/// Leading `n_modes` eigenmodes of the closed-shell RPA dielectric
/// ε(iω) = 1 − v χ₀(iω) built on `result` (an RHF or RKS result for `mol` in
/// `basis_set`), with `auxbasis` the RI fitting basis. `omega` is an imaginary
/// frequency in Hartree (0 = static). Each mode is a charge density in the aux
/// basis; see `ResponseModes`.
#[pyfunction]
#[pyo3(signature = (mol, basis_set, auxbasis, result, n_modes=10, omega=0.0, memory_budget_gb=None))]
#[allow(clippy::too_many_arguments)]
fn response_modes(
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    auxbasis: &PyBasisSet,
    result: DensitySource,
    n_modes: usize,
    omega: f64,
    memory_budget_gb: Option<f64>,
) -> PyResult<PyResponseModes> {
    if n_modes == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "n_modes must be >= 1",
        ));
    }
    if !(omega.is_finite() && omega >= 0.0) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "omega must be a finite imaginary frequency >= 0 (Hartree), got {omega}"
        )));
    }
    let obs = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    let dfbs = PreparedBasis::new(&mol.inner, &auxbasis.inner).map_err(make_err)?;
    let scf = result.scf();
    if scf.mos_r().nrows() != obs.nbasis() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "result has {} AOs but basis_set places {} on mol: pass the basis the SCF ran in",
            scf.mos_r().nrows(),
            obs.nbasis()
        )));
    }
    let inner = ferric_rpa::response_modes::response_modes(
        &mol.inner,
        &obs,
        &dfbs,
        scf,
        Operator::coulomb(),
        n_modes,
        omega,
        budget_bytes_from_gb(memory_budget_gb),
    )
    .map_err(make_err)?;
    Ok(PyResponseModes { inner })
}

/// Write mode `index` of `modes` (from `response_modes` with this `auxbasis`)
/// as a Gaussian cube file. `normalization` is "l2" (∫ρ² dr = 1, isovalues
/// comparable to an orbital's) or "coulomb" (the stored coefficients, unit
/// Coulomb self-energy). Grid arguments as in `write_cube` (Bohr). Returns the
/// (origin, shape) used.
#[pyfunction]
#[pyo3(signature = (path, mol, auxbasis, modes, index, normalization="l2", spacing=0.3, padding=4.0, origin=None, shape=None))]
#[allow(clippy::too_many_arguments)]
fn write_response_mode_cube(
    path: &str,
    mol: &PyMolecule,
    auxbasis: &PyBasisSet,
    modes: &PyResponseModes,
    index: usize,
    normalization: &str,
    spacing: f64,
    padding: f64,
    origin: Option<[f64; 3]>,
    shape: Option<[usize; 3]>,
) -> PyResult<([f64; 3], [usize; 3])> {
    let m = &modes.inner;
    if index >= m.eigenvalues.len() {
        return Err(pyo3::exceptions::PyIndexError::new_err(format!(
            "mode index {index} out of range ({} modes)",
            m.eigenvalues.len()
        )));
    }
    let scale = match normalization {
        "l2" => 1.0 / m.l2_norms[index],
        "coulomb" => 1.0,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown normalization {other:?}: expected \"l2\" or \"coulomb\""
            )))
        }
    };
    let grid = cube_grid_spec(mol, spacing, padding, origin, shape)?;
    let chi =
        ferric_export::eval_basis_on_grid(&mol.inner, &auxbasis.inner, &grid).map_err(make_err)?;
    if chi.nrows() != m.naux {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "auxbasis places {} functions on mol but the modes have naux = {}",
            chi.nrows(),
            m.naux
        )));
    }
    let vals = m.coefficients.column(index).dot(&chi) * scale;
    let data = ndarray::Array3::from_shape_vec((grid.n_x, grid.n_y, grid.n_z), vals.to_vec())
        .map_err(make_err)?;
    let comment = format!(
        "RPA response mode {index} lambda(i*{:.4})={:.6} ({normalization}-normalised density), Bohr",
        m.omega, m.eigenvalues[index]
    );
    ferric_export::export_cube(path, &mol.inner, &grid, &data, &comment).map_err(make_err)?;
    Ok((grid.origin, [grid.n_x, grid.n_y, grid.n_z]))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyResponseModes>()?;
    m.add_function(wrap_pyfunction!(response_modes, m)?)?;
    m.add_function(wrap_pyfunction!(write_response_mode_cube, m)?)
}
