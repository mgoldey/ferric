//! Building blocks: ferric's AO integrals and J/K builds as numpy arrays.
//!
//! The `run_*` drivers return finished results. `AoIntegrals` instead hands a
//! Python author the pieces a new method is assembled from (one-electron
//! matrices, 4-, 3- and 2-index ERIs under any supported operator, and the
//! production direct J/K build), so a method can be prototyped in numpy on
//! top of the same engine the drivers use.
//!
//! `einsum(spec, a, b)` exposes the contraction engine ferric's own
//! correlation codes use (`einsum!`), with the same spec rules.
//!
//! Conventions match the drivers: Bohr, Hartree, AOs in ferric's shell order
//! (`shell_offsets()`), and `jk()` takes the TOTAL density, so the closed-shell
//! Fock matrix is `hcore() + J - 0.5 * K`.

use super::{make_err, PyBasisSet, PyMolecule};
use ferric_core::basis::BasisSet;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::engine_pool::eri_precision;
use ferric_integrals::oneelectron;
use ferric_integrals::operator::Operator;
use ferric_integrals::threeindex::{coulomb_metric_2c, eri3_tensor};
use ferric_scf::screening::SchwarzBounds;
use ndarray::{Array2, Array3, Array4};
use numpy::{PyArray2, PyArray3, PyArray4, PyArrayDyn, PyReadonlyArray2, PyReadonlyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Default cap on the dense `eri()` tensor, in GiB.
const DEFAULT_ERI_MAX_GB: f64 = 2.0;

/// Largest `max |D - D^T|` `jk()` accepts. The J/K build folds the
/// permutational symmetry of the ERIs, which is only valid for a symmetric
/// density; an asymmetric one would give a silently wrong K.
const DENSITY_SYMMETRY_TOL: f64 = 1e-10;

/// Parse the `operator=` / `omega=` pair strictly: `"coulomb"` takes no omega,
/// `"erf"` and `"erfc"` require a positive one (Bohr⁻¹). Anything else is a
/// `ValueError`, never a silent default.
fn parse_operator(operator: &str, omega: Option<f64>) -> PyResult<Operator> {
    match (operator, omega) {
        ("coulomb", None) => Ok(Operator::coulomb()),
        ("coulomb", Some(_)) => Err(PyValueError::new_err(
            "operator='coulomb' takes no omega; use operator='erf' or 'erfc'",
        )),
        ("erf" | "erfc", Some(w)) if w.is_finite() && w > 0.0 => Ok(if operator == "erf" {
            Operator::erf(w)
        } else {
            Operator::erfc(w)
        }),
        ("erf" | "erfc", _) => Err(PyValueError::new_err(format!(
            "operator='{operator}' requires a positive finite omega (Bohr^-1), got {omega:?}"
        ))),
        _ => Err(PyValueError::new_err(format!(
            "unknown operator '{operator}'; expected 'coulomb', 'erf' or 'erfc'"
        ))),
    }
}

/// Dense (μν|λσ) over every shell quartet, scattered by 8-fold symmetry.
///
/// Serial on purpose: this is the nao⁴ reference tensor for small systems,
/// not a production path (`jk()` is the production path).
fn dense_eri(op: Operator, prep: &PreparedBasis) -> PyResult<Array4<f64>> {
    let n = prep.nbasis();
    let nsh = prep.nshells();
    let dims = prep.shell_dims();
    let offs = prep.shell_offsets();
    let mut eng = Engine::new_2e(op, prep, eri_precision()).map_err(make_err)?;
    let mut eri = Array4::<f64>::zeros((n, n, n, n));
    for s1 in 0..nsh {
        for s2 in 0..=s1 {
            let p12 = s1 * (s1 + 1) / 2 + s2;
            for s3 in 0..nsh {
                for s4 in 0..=s3 {
                    if s3 * (s3 + 1) / 2 + s4 > p12 {
                        continue;
                    }
                    let block = match eng.compute_quartet_checked(prep, s1, s2, s3, s4) {
                        Ok(Some(b)) => b,
                        Ok(None) => continue,
                        Err(status) => {
                            return Err(make_err(format!(
                                "ERI engine refused shell quartet ({s1},{s2}|{s3},{s4}): status {status}"
                            )))
                        }
                    };
                    let (n1, n2, n3, n4) = (dims[s1], dims[s2], dims[s3], dims[s4]);
                    for a in 0..n1 {
                        for b in 0..n2 {
                            for c in 0..n3 {
                                for d in 0..n4 {
                                    let v = block[((a * n2 + b) * n3 + c) * n4 + d];
                                    let (i, j) = (offs[s1] + a, offs[s2] + b);
                                    let (k, l) = (offs[s3] + c, offs[s4] + d);
                                    for (p, q) in [(i, j), (j, i)] {
                                        for (r, s) in [(k, l), (l, k)] {
                                            eri[[p, q, r, s]] = v;
                                            eri[[r, s, p, q]] = v;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(eri)
}

/// AO integrals and J/K builds for one molecule in one basis.
///
/// Construct once per geometry; every method reuses the prepared basis. ECPs
/// carried by the basis are applied exactly as `run_rhf` applies them
/// (`hcore()` includes V_ECP; `nelec` counts valence electrons).
#[pyclass]
#[pyo3(name = "AoIntegrals")]
struct PyAoIntegrals {
    mol: Molecule,
    bs: BasisSet,
    prep: PreparedBasis,
}

impl PyAoIntegrals {
    fn aux_prep(&self, aux: &PyBasisSet) -> PyResult<PreparedBasis> {
        PreparedBasis::new(&self.mol, &aux.inner).map_err(make_err)
    }
}

#[pymethods]
impl PyAoIntegrals {
    #[new]
    fn new(mol: &PyMolecule, basis_set: &PyBasisSet) -> PyResult<Self> {
        let mut emol = mol.inner.clone();
        emol.apply_ecp(&basis_set.inner);
        let prep = PreparedBasis::new(&emol, &basis_set.inner).map_err(make_err)?;
        Ok(PyAoIntegrals {
            mol: emol,
            bs: basis_set.inner.clone(),
            prep,
        })
    }
    /// Number of AO basis functions.
    #[getter]
    fn nao(&self) -> usize {
        self.prep.nbasis()
    }
    /// Electron count after ECP core removal (what the SCF occupies).
    #[getter]
    fn nelec(&self) -> i32 {
        self.mol.nelec()
    }
    /// Nuclear repulsion energy (Hartree), with ECP-effective charges.
    fn nuclear_repulsion(&self) -> f64 {
        self.mol.nuclear_repulsion()
    }
    /// AO offset of each shell plus a trailing `nao` (length nshells + 1,
    /// like PySCF's `ao_loc`): shell `i` is AOs `offs[i]..offs[i + 1]`.
    fn shell_offsets(&self) -> Vec<usize> {
        self.prep.shell_offsets().to_vec()
    }
    /// Atom index of each shell.
    fn shell_atoms(&self) -> Vec<usize> {
        self.prep.shell_to_atom().to_vec()
    }
    /// Atom index of each AO.
    fn ao_atoms(&self) -> Vec<usize> {
        let mut out = Vec::with_capacity(self.prep.nbasis());
        for (&atom, &dim) in self.prep.shell_to_atom().iter().zip(self.prep.shell_dims()) {
            out.extend(std::iter::repeat_n(atom, dim));
        }
        out
    }
    /// Overlap S (nao × nao).
    fn overlap<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        PyArray2::from_owned_array(py, oneelectron::overlap(&self.prep))
    }
    /// Kinetic energy T (nao × nao).
    fn kinetic<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        PyArray2::from_owned_array(py, oneelectron::kinetic(&self.prep))
    }
    /// Nuclear attraction V (nao × nao), ECP-effective charges, no V_ECP.
    fn nuclear<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        PyArray2::from_owned_array(py, oneelectron::nuclear(&self.prep))
    }
    /// The ECP projector V_ECP (nao × nao), or `None` without an ECP basis.
    fn ecp<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
        oneelectron::ecp_potential(&self.mol, &self.bs).map(|v| PyArray2::from_owned_array(py, v))
    }
    /// Core Hamiltonian T + V (+ V_ECP), the matrix `run_rhf` starts from.
    fn hcore<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        PyArray2::from_owned_array(py, oneelectron::hcore_ecp(&self.prep, &self.mol, &self.bs))
    }
    /// Dipole integrals ⟨μ|(r - origin)|ν⟩, shape (3, nao, nao); `origin` in
    /// Bohr. The electronic dipole is `-einsum('xij,ij->x', d, D)`.
    #[pyo3(signature = (origin=(0.0, 0.0, 0.0)))]
    fn dipole<'py>(
        &self,
        py: Python<'py>,
        origin: (f64, f64, f64),
    ) -> PyResult<Bound<'py, PyArray3<f64>>> {
        let [x, y, z] =
            oneelectron::dipole(&self.prep, [origin.0, origin.1, origin.2]).map_err(make_err)?;
        let n = self.prep.nbasis();
        let mut out = Array3::<f64>::zeros((3, n, n));
        for (axis, m) in [x, y, z].iter().enumerate() {
            out.index_axis_mut(ndarray::Axis(0), axis).assign(m);
        }
        Ok(PyArray3::from_owned_array(py, out))
    }
    /// Dense 4-index ERIs (μν|λσ) in chemists' notation, shape (nao,)*4.
    ///
    /// nao⁴ memory: raises `ValueError` above `max_gb` GiB (default 2) rather
    /// than exhausting RAM. For anything but small systems use `jk()`, or
    /// `eri3()` + `eri2()` for density fitting.
    #[pyo3(signature = (operator="coulomb", omega=None, max_gb=DEFAULT_ERI_MAX_GB))]
    fn eri<'py>(
        &self,
        py: Python<'py>,
        operator: &str,
        omega: Option<f64>,
        max_gb: f64,
    ) -> PyResult<Bound<'py, PyArray4<f64>>> {
        let op = parse_operator(operator, omega)?;
        let n = self.prep.nbasis() as f64;
        let gb = n.powi(4) * 8.0 / f64::from(1u32 << 30);
        if gb > max_gb {
            return Err(PyValueError::new_err(format!(
                "dense ERI tensor needs {gb:.2} GiB > max_gb={max_gb}; use jk() or eri3()/eri2(), \
                 or raise max_gb"
            )));
        }
        let prep = &self.prep;
        let eri = py.allow_threads(|| dense_eri(op, prep))?;
        Ok(PyArray4::from_owned_array(py, eri))
    }
    /// 3-index ERIs (P|μν) against the auxiliary basis `aux`, shape
    /// (naux, nao, nao). Raw integrals: no metric is applied.
    #[pyo3(signature = (aux, operator="coulomb", omega=None))]
    fn eri3<'py>(
        &self,
        py: Python<'py>,
        aux: &PyBasisSet,
        operator: &str,
        omega: Option<f64>,
    ) -> PyResult<Bound<'py, PyArray3<f64>>> {
        let op = parse_operator(operator, omega)?;
        let dfbs = self.aux_prep(aux)?;
        let obs = &self.prep;
        let t = py
            .allow_threads(|| eri3_tensor(op, obs, &dfbs))
            .map_err(make_err)?;
        Ok(PyArray3::from_owned_array(py, t))
    }
    /// 2-index ERIs (P|Q) of the auxiliary basis `aux` (the fitting metric
    /// when `operator` is the fitted kernel), shape (naux, naux).
    #[pyo3(signature = (aux, operator="coulomb", omega=None))]
    fn eri2<'py>(
        &self,
        py: Python<'py>,
        aux: &PyBasisSet,
        operator: &str,
        omega: Option<f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let op = parse_operator(operator, omega)?;
        let dfbs = self.aux_prep(aux)?;
        let m = py
            .allow_threads(|| coulomb_metric_2c(op, &dfbs))
            .map_err(make_err)?;
        Ok(PyArray2::from_owned_array(py, m))
    }
    /// Coulomb and exchange matrices `(J, K)` for a symmetric AO density,
    /// built by the same Schwarz-screened direct J/K the SCF uses.
    ///
    /// `J_μν = Σ (μν|λσ) D_λσ`, `K_μν = Σ (μλ|νσ) D_λσ`; pass the TOTAL
    /// density and the closed-shell Fock matrix is `hcore() + J - 0.5 * K`.
    /// `thresh` is the Schwarz screening threshold (the SCF's
    /// `integral_thresh`, default 1e-12).
    #[pyo3(signature = (density, operator="coulomb", omega=None, thresh=1e-12))]
    fn jk<'py>(
        &self,
        py: Python<'py>,
        density: PyReadonlyArray2<'py, f64>,
        operator: &str,
        omega: Option<f64>,
        thresh: f64,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
        let op = parse_operator(operator, omega)?;
        let n = self.prep.nbasis();
        let d = density.as_array().to_owned();
        if d.dim() != (n, n) {
            return Err(PyValueError::new_err(format!(
                "density has shape {:?}, expected ({n}, {n})",
                d.dim()
            )));
        }
        // Finite first: `f64::max` drops NaN, so a NaN density would
        // otherwise pass the symmetry check below.
        if !d.iter().all(|v| v.is_finite()) {
            return Err(PyValueError::new_err("density contains NaN or inf"));
        }
        let asym = d
            .iter()
            .zip(d.t().iter())
            .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
        if asym > DENSITY_SYMMETRY_TOL {
            return Err(PyValueError::new_err(format!(
                "density is not symmetric (max |D - D^T| = {asym:.3e}); jk() folds ERI \
                 permutational symmetry and needs a symmetric density"
            )));
        }
        let prep = &self.prep;
        let (j, k) = py.allow_threads(|| -> PyResult<(Array2<f64>, Array2<f64>)> {
            let bounds = SchwarzBounds::compute(op, prep).map_err(make_err)?;
            let mut j = Array2::<f64>::zeros((n, n));
            let mut k = Array2::<f64>::zeros((n, n));
            ferric_scf::rhf::build_jk(
                &ParallelContext::default(),
                prep,
                &bounds,
                thresh,
                &d,
                &mut j,
                &mut k,
            )
            .map_err(make_err)?;
            Ok((j, k))
        })?;
        Ok((
            PyArray2::from_owned_array(py, j),
            PyArray2::from_owned_array(py, k),
        ))
    }
    fn __repr__(&self) -> String {
        format!(
            "AoIntegrals(basis={:?}, natoms={}, nao={})",
            self.bs.name,
            self.mol.atoms.len(),
            self.prep.nbasis()
        )
    }
}

/// Binary tensor contraction with ferric's GEMM engine (the one behind the
/// `einsum!` macro in ferric's MP2/CC code), returning `scale * result`.
///
/// Spec rules (stricter than `numpy.einsum`, and an error, not a fallback,
/// when broken): exactly two operands and an explicit `->`; an index shared
/// by both operands is summed unless it is also in the output, in which case
/// it is a batch (element-wise) index; no index may repeat within one term;
/// the output lists batch indices, then the left operand's free indices,
/// then the right's (any order within each group). A scalar spec
/// (`"ij,ij->"`) returns a float.
#[pyfunction]
#[pyo3(signature = (spec, a, b, scale=1.0))]
fn einsum(
    py: Python<'_>,
    spec: &str,
    a: PyReadonlyArrayDyn<'_, f64>,
    b: PyReadonlyArrayDyn<'_, f64>,
    scale: f64,
) -> PyResult<PyObject> {
    let (av, bv) = (a.as_array(), b.as_array());
    let out = py
        .allow_threads(|| ferric_tensors::einsum_dyn(spec, av, bv, scale))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    if out.ndim() == 0 {
        let v = *out
            .iter()
            .next()
            .expect("0-d einsum result holds one element");
        return Ok(v.into_pyobject(py)?.into_any().unbind());
    }
    Ok(PyArrayDyn::from_owned_array(py, out).into_any().unbind())
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyAoIntegrals>()?;
    m.add_function(wrap_pyfunction!(einsum, m)?)
}
