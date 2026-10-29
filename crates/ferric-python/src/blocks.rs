//! Building blocks: ferric's AO integrals and J/K builds as numpy arrays.
//!
//! The `run_*` drivers return finished results. `AoIntegrals` instead hands a
//! Python author the pieces a new method is assembled from (one-electron
//! matrices, 4-, 3- and 2-index ERIs under any supported operator, and the
//! production direct J/K build), so a method can be prototyped in numpy on
//! top of the same engine the drivers use.
//!
//! `AoIntegrals.df_jk(aux)` returns a `DfJk`: the production RI-J/RI-K
//! fitters, built once and reused for every density.
//!
//! `einsum(spec, a, b)` exposes the contraction engine ferric's own
//! correlation codes use (`einsum!`), with the same spec rules.
//!
//! `PyFockHook` adapts a Python callable `hook(D) -> (E, V)` to
//! `ferric_scf::rhf::RhfFockHook`, for `run_rhf(fock_hook=)` and
//! `run_dft(fock_hook=)`.
//!
//! Conventions match the drivers: Bohr, Hartree, AOs in ferric's shell order
//! (`shell_offsets()`), and `jk()` takes the TOTAL density, so the closed-shell
//! Fock matrix is `hcore() + J - 0.5 * K`.

use super::{make_err, PyBasisSet, PyMolecule};
use ferric_core::basis::BasisSet;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::engine_pool::eri_precision;
use ferric_integrals::oneelectron;
use ferric_integrals::operator::Operator;
use ferric_integrals::three_index_source::ThreeIndexSource;
use ferric_integrals::threeindex::{coulomb_metric_2c, eri3_tensor};
use ferric_scf::df_j::DfJ;
use ferric_scf::df_k::DfK;
use ferric_scf::fock::{JBuilder, KBuilder};
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

/// Max `|A - Aᵀ|` of a finite square matrix; `NaN`/`inf` entries are an
/// error first, because `f64::max` drops NaN and would hide them.
fn asymmetry(a: &Array2<f64>, what: &str) -> PyResult<f64> {
    if !a.iter().all(|v| v.is_finite()) {
        return Err(PyValueError::new_err(format!("{what} contains NaN or inf")));
    }
    Ok(a.iter()
        .zip(a.t().iter())
        .fold(0.0f64, |m, (x, y)| m.max((x - y).abs())))
}

/// Copy a density and check it is `(n, n)`, finite and symmetric: the J/K
/// builds fold ERI permutational symmetry, so an asymmetric D would give a
/// silently wrong K.
fn checked_density(density: PyReadonlyArray2<'_, f64>, n: usize) -> PyResult<Array2<f64>> {
    let d = density.as_array().to_owned();
    if d.dim() != (n, n) {
        return Err(PyValueError::new_err(format!(
            "density has shape {:?}, expected ({n}, {n})",
            d.dim()
        )));
    }
    let asym = asymmetry(&d, "density")?;
    if asym > DENSITY_SYMMETRY_TOL {
        return Err(PyValueError::new_err(format!(
            "density is not symmetric (max |D - D^T| = {asym:.3e}); the J/K builds fold ERI \
             permutational symmetry and need a symmetric density"
        )));
    }
    Ok(d)
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
        let d = checked_density(density, n)?;
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
    /// Density-fitted J/K builders against the auxiliary basis `aux`: the
    /// RI-J and RI-K fitters `run_rhf(df_j_aux=, df_k_aux=)` uses, built
    /// once (one raw (P|μν) pass shared by both) and reused on every call.
    ///
    /// `memory_budget_gb` bounds the resident 3-index tensors like
    /// `run_rhf`'s kwarg (default: auto-detected); above it they spill.
    #[pyo3(signature = (aux, operator="coulomb", omega=None, memory_budget_gb=None))]
    fn df_jk(
        &self,
        py: Python<'_>,
        aux: &PyBasisSet,
        operator: &str,
        omega: Option<f64>,
        memory_budget_gb: Option<f64>,
    ) -> PyResult<PyDfJk> {
        let op = parse_operator(operator, omega)?;
        let budget = ferric_scf::rhf::resolve_three_index_budget(
            super::budget_bytes_from_gb(memory_budget_gb).unwrap_or(0),
        );
        let dfbs = self.aux_prep(aux)?;
        let obs = &self.prep;
        let (j, k) = py
            .allow_threads(|| -> Result<_, FerricError> {
                let mut raw = ThreeIndexSource::build(op, obs, &dfbs, budget)?;
                let k = DfK::from_full_raw(&mut raw, obs, &dfbs, op, budget, None)?;
                let j = DfJ::from_source(raw, op, &dfbs, budget, None)?;
                Ok((j, k))
            })
            .map_err(make_err)?;
        Ok(PyDfJk {
            j,
            k,
            nao: obs.nbasis(),
            naux: dfbs.nbasis(),
        })
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

/// RI-J and RI-K for one orbital/auxiliary basis pair, from
/// `AoIntegrals.df_jk(aux)`.
///
/// `J ≈ Σ_PQ (μν|P) V⁻¹_PQ (Q|λσ) D_λσ` and the matching K, with V the
/// (P|Q) metric under the same operator; pass the TOTAL density, so the
/// closed-shell Fock matrix is `hcore() + J - 0.5 * K`, as for `jk()`.
#[pyclass]
#[pyo3(name = "DfJk")]
struct PyDfJk {
    j: DfJ<'static>,
    k: DfK<'static>,
    nao: usize,
    naux: usize,
}

impl PyDfJk {
    fn build_j(&mut self, d: &Array2<f64>) -> PyResult<Array2<f64>> {
        let mut j = Array2::<f64>::zeros((self.nao, self.nao));
        self.j.build(d, &mut j).map_err(make_err)?;
        Ok(j)
    }
    fn build_k(&mut self, d: &Array2<f64>) -> PyResult<Array2<f64>> {
        let mut k = Array2::<f64>::zeros((self.nao, self.nao));
        self.k.update_density(d);
        self.k.build(d, &mut k).map_err(make_err)?;
        Ok(k)
    }
}

#[pymethods]
impl PyDfJk {
    /// Number of auxiliary functions.
    #[getter]
    fn naux(&self) -> usize {
        self.naux
    }
    /// Fitted Coulomb matrix J for a symmetric AO density.
    fn j<'py>(
        &mut self,
        py: Python<'py>,
        density: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let d = checked_density(density, self.nao)?;
        Ok(PyArray2::from_owned_array(py, self.build_j(&d)?))
    }
    /// Fitted exchange matrix K for a symmetric AO density.
    fn k<'py>(
        &mut self,
        py: Python<'py>,
        density: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let d = checked_density(density, self.nao)?;
        Ok(PyArray2::from_owned_array(py, self.build_k(&d)?))
    }
    /// `(J, K)` for a symmetric AO density.
    fn jk<'py>(
        &mut self,
        py: Python<'py>,
        density: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
        let d = checked_density(density, self.nao)?;
        let j = self.build_j(&d)?;
        let k = self.build_k(&d)?;
        Ok((
            PyArray2::from_owned_array(py, j),
            PyArray2::from_owned_array(py, k),
        ))
    }
    fn __repr__(&self) -> String {
        format!("DfJk(nao={}, naux={})", self.nao, self.naux)
    }
}

/// Largest `max |V - V^T|` a Fock hook may return. An asymmetric V would
/// make the Fock matrix non-Hermitian and the eigensolve meaningless.
const HOOK_SYMMETRY_TOL: f64 = 1e-10;

/// A Python `hook(D) -> (E, V)` as an `RhfFockHook`. The SCF runs with the
/// GIL released, so each call re-acquires it. A Python exception, or a
/// return value of the wrong kind, is stored and re-raised unchanged by
/// [`PyFockHook::finish`] once the SCF has unwound.
pub(crate) struct PyFockHook {
    callable: Py<PyAny>,
    nao: usize,
    err: Option<PyErr>,
}

impl PyFockHook {
    pub(crate) fn new(callable: &Bound<'_, PyAny>, nao: usize) -> PyResult<Self> {
        if !callable.is_callable() {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "fock_hook must be callable as hook(D) -> (E, V)",
            ));
        }
        Ok(PyFockHook {
            callable: callable.clone().unbind(),
            nao,
            err: None,
        })
    }

    fn eval(&self, py: Python<'_>, d: &Array2<f64>) -> PyResult<(f64, Array2<f64>)> {
        let out = self.callable.call1(py, (PyArray2::from_array(py, d),))?;
        let (e, v): (f64, PyReadonlyArray2<'_, f64>) = out.extract(py).map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(
                "fock_hook must return (E: float, V: 2-D float array)",
            )
        })?;
        let v = v.as_array().to_owned();
        let n = self.nao;
        if v.dim() != (n, n) {
            return Err(PyValueError::new_err(format!(
                "fock_hook returned V of shape {:?}, expected ({n}, {n})",
                v.dim()
            )));
        }
        if !e.is_finite() {
            return Err(PyValueError::new_err(format!(
                "fock_hook returned a non-finite energy {e}"
            )));
        }
        let asym = asymmetry(&v, "fock_hook's V")?;
        if asym > HOOK_SYMMETRY_TOL {
            return Err(PyValueError::new_err(format!(
                "fock_hook returned a non-symmetric V (max |V - V^T| = {asym:.3e})"
            )));
        }
        Ok((e, v))
    }

    /// The `FnMut` the SCF calls.
    pub(crate) fn call(&mut self, d: &Array2<f64>) -> Result<(f64, Array2<f64>), FerricError> {
        Python::with_gil(|py| self.eval(py, d)).map_err(|e| {
            self.err = Some(e);
            FerricError::General("fock_hook failed".into())
        })
    }

    /// The hook's own error if it raised, otherwise `scf`'s error as usual.
    pub(crate) fn finish<T>(self, scf: Result<T, FerricError>) -> PyResult<T> {
        match (self.err, scf) {
            (Some(e), _) => Err(e),
            (None, r) => r.map_err(make_err),
        }
    }
}

/// `solve_rhf_with_hook` with a Python `hook(D) -> (E, V)`, GIL released
/// for the SCF. Unsupported settings are a `ValueError` before any work.
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_rhf_hooked(
    py: Python<'_>,
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    config: &ferric_scf::rhf::RhfConfig,
    hook: &Bound<'_, PyAny>,
) -> PyResult<ferric_scf::result::ScfResult> {
    ferric_scf::rhf::validate_fock_hook(config)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let mut h = PyFockHook::new(hook, prep.nbasis())?;
    let r = py.allow_threads(|| {
        let mut f = |d: &Array2<f64>| h.call(d);
        ferric_scf::rhf::solve_rhf_with_hook(ctx, mol, prep, op, bounds, config, &mut f)
    });
    h.finish(r)
}

/// `run_dft`'s SCF: the convergence ladder, or with a `fock_hook` its first
/// rung only (later rungs switch on SOSCF, whose orbital Hessian would not
/// include the hook). `hooked` carries the hook with `run_dft`'s
/// `with_gradient` and `dispersion is not None` flags: neither the analytic
/// gradient nor the dispersion terms would include the hook, so both are
/// refused before the SCF.
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_ks_ladder(
    py: Python<'_>,
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    ladder: &[ferric_scf::ladder::Rung],
    hooked: Option<(&Bound<'_, PyAny>, bool, bool)>,
) -> PyResult<ferric_scf::result::ScfResult> {
    let refuse = |what: &str| {
        Err(PyValueError::new_err(format!(
            "run_dft: fock_hook cannot be combined with {what}"
        )))
    };
    match hooked {
        None => py
            .allow_threads(|| {
                ferric_scf::ladder::solve_rhf_ladder(ctx, mol, prep, op, bounds, ladder)
            })
            .map(|lr| lr.result)
            .map_err(make_err),
        Some((_, true, _)) => refuse("with_gradient=True"),
        Some((_, _, true)) => refuse("dispersion"),
        Some((hook, false, false)) => {
            solve_rhf_hooked(py, ctx, mol, prep, op, bounds, &ladder[0].config, hook)
        }
    }
}

/// `run_dft(symmetry=True)` refusals: the existing ones, plus a `fock_hook`
/// (the point-group-blocked solve does not call it).
pub(crate) fn refuse_symmetric_dft(
    symmetry: bool,
    with_gradient: bool,
    dispersion: bool,
    hooked: bool,
) -> PyResult<()> {
    if hooked {
        return Err(PyValueError::new_err(
            "run_dft: fock_hook cannot be combined with symmetry=True / irrep_nelec",
        ));
    }
    super::refuse_dft_symmetry_extras(symmetry, with_gradient, dispersion)
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
    m.add_class::<PyDfJk>()?;
    m.add_function(wrap_pyfunction!(einsum, m)?)
}
