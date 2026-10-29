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
//! MO transforms: `ao2mo(c1, c2, c3, c4)` (integral-direct (pq|rs), never
//! holding the nao⁴ AO tensor) and `df_mo(aux, c1, c2)` (fitted B^P_pq).
//!
//! `AoIntegrals.xc_functional(name)` returns an `XcFunctional`: E_xc and V_xc
//! (closed or spin-polarized) and the XC gradient on the SCF's own grid.
//!
//! Response: `XcFunctional.response(D, D1)` (the f_xc kernel), and
//! `AoIntegrals.cphf(...)` / `ucphf(...)`, the orbital-Hessian product and
//! CPHF/CPKS solve a Z-vector or a static property needs.
//!
//! Gradient pieces, each `(natoms, 3)` at fixed densities:
//! `grad_nuclear_repulsion()`, `grad_hcore(D)`, `grad_overlap(W)` and
//! `grad_jk(D, aux=None)` (exact, or density-fitted with `aux`); an SCF
//! gradient is assembled from them in numpy.
//!
//! `einsum(spec, a, b)` exposes the contraction engine ferric's own
//! correlation codes use (`einsum!`), with the same spec rules.
//!
//! `PyFockHook` adapts a Python callable `hook(D) -> (E, V)` to
//! `ferric_scf::rhf::RhfFockHook`, for `run_rhf(fock_hook=)` and
//! `run_dft(fock_hook=)`, and `hook(D_a, D_b) -> (E, V_a, V_b)` to
//! `ferric_scf::fock::SpinFockHook`, for `run_uhf` / `run_rohf(fock_hook=)`.
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
use ndarray::{Array2, Array3, Array4, Axis};
use numpy::{PyArray2, PyArray3, PyArray4, PyArrayDyn, PyReadonlyArray2, PyReadonlyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

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

/// Copy a matrix and check it is `(n, n)` and finite.
fn checked_square(m: PyReadonlyArray2<'_, f64>, n: usize, what: &str) -> PyResult<Array2<f64>> {
    let m = m.as_array().to_owned();
    if m.dim() != (n, n) {
        return Err(PyValueError::new_err(format!(
            "{what} has shape {:?}, expected ({n}, {n})",
            m.dim()
        )));
    }
    asymmetry(&m, what)?;
    Ok(m)
}

/// Copy a density and check it is `(n, n)`, finite and symmetric: the J/K
/// builds (and the gradient contractions) fold ERI permutational symmetry,
/// so an asymmetric D would give a silently wrong result.
fn checked_density(density: PyReadonlyArray2<'_, f64>, n: usize) -> PyResult<Array2<f64>> {
    let d = checked_square(density, n, "density")?;
    let asym = asymmetry(&d, "density")?;
    if asym > DENSITY_SYMMETRY_TOL {
        return Err(PyValueError::new_err(format!(
            "density is not symmetric (max |D - D^T| = {asym:.3e}); the J/K builds fold ERI \
             permutational symmetry and need a symmetric density"
        )));
    }
    Ok(d)
}

/// Copy MO coefficients and check they are `(nao, k)` and finite.
fn checked_coeffs(c: PyReadonlyArray2<'_, f64>, nao: usize, what: &str) -> PyResult<Array2<f64>> {
    let c = c.as_array().to_owned();
    if c.nrows() != nao {
        return Err(PyValueError::new_err(format!(
            "{what} has shape {:?}, expected ({nao}, k)",
            c.dim()
        )));
    }
    if !c.iter().all(|v| v.is_finite()) {
        return Err(PyValueError::new_err(format!("{what} contains NaN or inf")));
    }
    Ok(c)
}

/// (μν|rs) for every AO pair μν and the ket MO pair rs, from one shell pair
/// (s1, s2) of the bra: all kets (s3 ≥ s4) are computed, scattered into
/// `(n1·n2, nao, nao)`, and transformed with `c3`/`c4`. Returns the block as
/// `(n1·n2, n3·n4)` rows in (a, b) order within the shell pair.
fn half_transformed_pair(
    eng: &mut Engine,
    prep: &PreparedBasis,
    (s1, s2): (usize, usize),
    c3: &Array2<f64>,
    c4: &Array2<f64>,
) -> Result<Array2<f64>, String> {
    let n = prep.nbasis();
    let dims = prep.shell_dims();
    let offs = prep.shell_offsets();
    let (n1, n2) = (dims[s1], dims[s2]);
    let mut g = Array3::<f64>::zeros((n1 * n2, n, n));
    for s3 in 0..prep.nshells() {
        for s4 in 0..=s3 {
            let block = match eng.compute_quartet_checked(prep, s1, s2, s3, s4) {
                Ok(Some(b)) => b,
                Ok(None) => continue,
                Err(status) => {
                    return Err(format!(
                        "ERI engine refused shell quartet ({s1},{s2}|{s3},{s4}): status {status}"
                    ))
                }
            };
            let (n3, n4) = (dims[s3], dims[s4]);
            for ab in 0..n1 * n2 {
                for c in 0..n3 {
                    for d in 0..n4 {
                        let v = block[(ab * n3 + c) * n4 + d];
                        let (k, l) = (offs[s3] + c, offs[s4] + d);
                        g[[ab, k, l]] = v;
                        g[[ab, l, k]] = v;
                    }
                }
            }
        }
    }
    let (m3, m4) = (c3.ncols(), c4.ncols());
    // (ab·λ, σ) · C4 -> (ab, λ, s); then C3ᵀ per ab -> (ab, r, s).
    let gs = g
        .into_shape_with_order((n1 * n2 * n, n))
        .expect("contiguous")
        .dot(c4)
        .into_shape_with_order((n1 * n2, n, m4))
        .expect("contiguous");
    let mut out = Array2::<f64>::zeros((n1 * n2, m3 * m4));
    for (ab, slab) in gs.axis_iter(Axis(0)).enumerate() {
        let rs = c3.t().dot(&slab);
        out.row_mut(ab)
            .assign(&rs.into_shape_with_order(m3 * m4).expect("contiguous"));
    }
    Ok(out)
}

/// (pq|rs) = Σ C1_μp C2_νq C3_λr C4_σs (μν|λσ), integral-direct: one pass
/// over bra shell pairs (parallel, 2-fold bra symmetry, 2-fold ket symmetry)
/// builds the half-transformed `(nao, nao, n3·n4)`, then two GEMMs finish it.
fn ao2mo_direct(
    op: Operator,
    prep: &PreparedBasis,
    c: [&Array2<f64>; 4],
) -> Result<Array4<f64>, String> {
    let n = prep.nbasis();
    let offs = prep.shell_offsets();
    let dims = prep.shell_dims();
    let (m1, m2, m3, m4) = (c[0].ncols(), c[1].ncols(), c[2].ncols(), c[3].ncols());
    let pairs: Vec<(usize, usize)> = (0..prep.nshells())
        .flat_map(|s1| (0..=s1).map(move |s2| (s1, s2)))
        .collect();
    let blocks = pairs
        .par_iter()
        .map_init(
            || Engine::new_2e(op, prep, eri_precision()),
            |eng, &pair| -> Result<_, String> {
                let eng = eng.as_mut().map_err(|e| e.to_string())?;
                Ok((pair, half_transformed_pair(eng, prep, pair, c[2], c[3])?))
            },
        )
        .collect::<Result<Vec<_>, String>>()?;
    let mut half = Array3::<f64>::zeros((n, n, m3 * m4));
    for ((s1, s2), blk) in blocks {
        for a in 0..dims[s1] {
            for b in 0..dims[s2] {
                let row = blk.row(a * dims[s2] + b);
                let (mu, nu) = (offs[s1] + a, offs[s2] + b);
                half.slice_mut(ndarray::s![mu, nu, ..]).assign(&row);
                half.slice_mut(ndarray::s![nu, mu, ..]).assign(&row);
            }
        }
    }
    // C1ᵀ over μ, then C2ᵀ over ν per p.
    let t1 = c[0]
        .t()
        .dot(
            &half
                .into_shape_with_order((n, n * m3 * m4))
                .expect("contiguous"),
        )
        .into_shape_with_order((m1, n, m3 * m4))
        .expect("contiguous");
    let mut out = Array3::<f64>::zeros((m1, m2, m3 * m4));
    for (p, slab) in t1.axis_iter(Axis(0)).enumerate() {
        out.index_axis_mut(Axis(0), p).assign(&c[1].t().dot(&slab));
    }
    Ok(out
        .into_shape_with_order((m1, m2, m3, m4))
        .expect("contiguous"))
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
    /// `veff`'s J / exchange-mix K maps, built on first use per
    /// (exchange mix, aux basis name).
    veff_maps: std::sync::Mutex<VeffMaps>,
}

/// `(c_sr, c_lr, omega)` as bits, and the aux basis name (`None` = exact).
type VeffKey = ([u64; 3], Option<String>);

type VeffMaps = Vec<(VeffKey, std::sync::Arc<FockResponse>)>;

fn lock_maps(m: &std::sync::Mutex<VeffMaps>) -> std::sync::MutexGuard<'_, VeffMaps> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl PyAoIntegrals {
    fn aux_prep(&self, aux: &PyBasisSet) -> PyResult<PreparedBasis> {
        PreparedBasis::new(&self.mol, &aux.inner).map_err(make_err)
    }
    /// Refuse an `XcFunctional` built for another molecule or basis.
    fn same_system(&self, xc: &PyXcFunctional) -> PyResult<()> {
        let same_atoms = xc.mol.atoms.len() == self.mol.atoms.len()
            && xc
                .mol
                .atoms
                .iter()
                .zip(&self.mol.atoms)
                .all(|(a, b)| a.z == b.z && [a.x, a.y, a.zpos] == [b.x, b.y, b.zpos]);
        if !same_atoms || xc.bs.name != self.bs.name || xc.prep.nbasis() != self.prep.nbasis() {
            return Err(PyValueError::new_err(
                "xc was built by another AoIntegrals (different molecule or basis)",
            ));
        }
        Ok(())
    }

    /// The cached J / exchange-mix K map for `(mix, aux)`.
    fn veff_map(
        &self,
        py: Python<'_>,
        mix: (f64, f64, f64),
        aux: Option<&PyBasisSet>,
        memory_budget_gb: Option<f64>,
    ) -> PyResult<std::sync::Arc<FockResponse>> {
        let key: VeffKey = (
            [mix.0.to_bits(), mix.1.to_bits(), mix.2.to_bits()],
            aux.map(|a| a.inner.name.clone()),
        );
        if let Some((_, m)) = lock_maps(&self.veff_maps).iter().find(|(k, _)| *k == key) {
            return Ok(m.clone());
        }
        let aux = response_aux(aux, memory_budget_gb);
        let spec = XcSpec {
            name: String::new(),
            grid: Default::default(),
            mix,
        };
        let zero = Array2::<f64>::zeros((self.prep.nbasis(), self.prep.nbasis()));
        let (mol, bs) = (&self.mol, &self.bs);
        let map = py
            .allow_threads(|| {
                FockResponse::new(mol, bs, Some(&spec), (&zero, &zero), aux_ref(&aux), false)
            })
            .map_err(make_err)?;
        let map = std::sync::Arc::new(map);
        lock_maps(&self.veff_maps).push((key, map.clone()));
        Ok(map)
    }

    /// dV_nn/dR: `oneelectron_gradient` at D = W = 0 is exactly its
    /// nuclear-repulsion term (every other term is linear in D or W).
    fn nuclear_gradient(&self) -> Result<Array2<f64>, FerricError> {
        let zero = Array2::<f64>::zeros((self.prep.nbasis(), self.prep.nbasis()));
        ferric_scf::gradient::oneelectron_gradient(&self.mol, &self.prep, &zero, &zero, None)
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
            veff_maps: std::sync::Mutex::new(Vec::new()),
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
    /// MO-basis ERIs (pq|rs) = Σ C1_μp C2_νq C3_λr C4_σs (μν|λσ), shape
    /// (k1, k2, k3, k4), chemists' notation.
    ///
    /// Integral-direct: the nao⁴ AO tensor is never stored; peak memory is
    /// the half-transformed nao²·k3·k4 plus the result, and above `max_gb`
    /// GiB it raises `ValueError`. Put the smaller orbital sets in `c3`/`c4`.
    #[pyo3(signature = (c1, c2, c3, c4, operator="coulomb", omega=None, max_gb=DEFAULT_ERI_MAX_GB))]
    #[allow(clippy::too_many_arguments)]
    fn ao2mo<'py>(
        &self,
        py: Python<'py>,
        c1: PyReadonlyArray2<'py, f64>,
        c2: PyReadonlyArray2<'py, f64>,
        c3: PyReadonlyArray2<'py, f64>,
        c4: PyReadonlyArray2<'py, f64>,
        operator: &str,
        omega: Option<f64>,
        max_gb: f64,
    ) -> PyResult<Bound<'py, PyArray4<f64>>> {
        let op = parse_operator(operator, omega)?;
        let n = self.prep.nbasis();
        let c = [
            checked_coeffs(c1, n, "c1")?,
            checked_coeffs(c2, n, "c2")?,
            checked_coeffs(c3, n, "c3")?,
            checked_coeffs(c4, n, "c4")?,
        ];
        let k: Vec<f64> = c.iter().map(|m| m.ncols() as f64).collect();
        let words = (n * n) as f64 * k[2] * k[3] + k.iter().product::<f64>();
        let gb = words * 8.0 / f64::from(1u32 << 30);
        if gb > max_gb {
            return Err(PyValueError::new_err(format!(
                "ao2mo needs {gb:.2} GiB > max_gb={max_gb} (half-transformed nao^2*k3*k4 plus \
                 the result); use df_mo(), or raise max_gb"
            )));
        }
        let prep = &self.prep;
        let out = py
            .allow_threads(|| ao2mo_direct(op, prep, [&c[0], &c[1], &c[2], &c[3]]))
            .map_err(make_err)?;
        Ok(PyArray4::from_owned_array(py, out))
    }
    /// Fitted 3-index MO tensor B^P_pq = Σ_Q (V^-1/2)_PQ (Q|μν) C1_μp C2_νq,
    /// shape (naux, k1, k2), so that (pq|rs) ≈ Σ_P B^P_pq B^P_rs. V is the
    /// (P|Q) metric under the same operator (RI-MP2's `metric_inverse_sqrt`:
    /// Cholesky, or a regularized eigh for erf).
    #[pyo3(signature = (aux, c1, c2, operator="coulomb", omega=None))]
    fn df_mo<'py>(
        &self,
        py: Python<'py>,
        aux: &PyBasisSet,
        c1: PyReadonlyArray2<'py, f64>,
        c2: PyReadonlyArray2<'py, f64>,
        operator: &str,
        omega: Option<f64>,
    ) -> PyResult<Bound<'py, PyArray3<f64>>> {
        let op = parse_operator(operator, omega)?;
        let n = self.prep.nbasis();
        let (c1, c2) = (checked_coeffs(c1, n, "c1")?, checked_coeffs(c2, n, "c2")?);
        let dfbs = self.aux_prep(aux)?;
        let obs = &self.prep;
        let b = py
            .allow_threads(|| -> Result<Array3<f64>, FerricError> {
                let raw = eri3_tensor(op, obs, &dfbs)?;
                let mo = ferric_mp2::mo_transform::transform_3center_ov(&raw, &c1, &c2);
                let v = coulomb_metric_2c(op, &dfbs)?;
                let v_is = ferric_mp2::rimp2::metric_inverse_sqrt(&v, op)?;
                Ok(ferric_mp2::mo_transform::dress_3index(&mo, &v_is))
            })
            .map_err(make_err)?;
        Ok(PyArray3::from_owned_array(py, b))
    }
    /// Nuclear-repulsion gradient dV_nn/dR, shape (natoms, 3), Hartree/Bohr.
    fn grad_nuclear_repulsion<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let g = py
            .allow_threads(|| self.nuclear_gradient())
            .map_err(make_err)?;
        Ok(PyArray2::from_owned_array(py, g))
    }
    /// Σ D_μν ∂h_μν/∂R at fixed D, shape (natoms, 3): kinetic, nuclear
    /// attraction (basis-centre and operator derivatives) and V_ECP. `D` is
    /// the TOTAL density. The overlap (Pulay) term is `grad_overlap(W)`.
    fn grad_hcore<'py>(
        &self,
        py: Python<'py>,
        density: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let d = checked_density(density, self.prep.nbasis())?;
        let g = py
            .allow_threads(|| -> Result<Array2<f64>, FerricError> {
                let zero = Array2::<f64>::zeros(d.dim());
                let (mol, prep) = (&self.mol, &self.prep);
                let mut g = ferric_scf::gradient::oneelectron_gradient(mol, prep, &d, &zero, None)?;
                g -= &self.nuclear_gradient()?;
                g += &ferric_scf::gradient::ecp_gradient(mol, prep, &d)?;
                Ok(g)
            })
            .map_err(make_err)?;
        Ok(PyArray2::from_owned_array(py, g))
    }
    /// Σ W_μν ∂S_μν/∂R, shape (natoms, 3), for any (n, n) `W`, used as given
    /// (no symmetrization, no sign). For an SCF, W is the energy-weighted
    /// density and the term enters the gradient with a minus sign.
    fn grad_overlap<'py>(
        &self,
        py: Python<'py>,
        w: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let w = checked_square(w, self.prep.nbasis(), "W")?;
        let prep = &self.prep;
        let g = py
            .allow_threads(|| ferric_scf::gradient::overlap_deriv_contract(prep, &w))
            .map_err(make_err)?;
        Ok(PyArray2::from_owned_array(py, g))
    }
    /// Two-electron integral-derivative terms at fixed D, each (natoms, 3):
    /// `(gJ, gK)` with gJ = ∂/∂R ½Tr[D J(D)] and gK = ∂/∂R ½Tr[D K(D)].
    ///
    /// Closed-shell HF: gJ - 0.5 gK (D total). UHF: gJ(D_a + D_b) - gK(D_a)
    /// - gK(D_b). `D` may be any symmetric matrix, so a bilinear term
    /// follows by polarization, E(D1, D2) = [E(D1 + D2) - E(D1 - D2)] / 4
    /// (response / Z-vector densities are indefinite).
    ///
    /// `aux` differentiates the fits `df_jk(aux, operator, omega)` builds
    /// instead (the energy of an SCF with `df_j_aux=df_k_aux=aux`): the
    /// 3-index integrals, the auxiliary-basis centres and the metric (RI-J:
    /// Cholesky inverse; RI-K: DfK's truncated V^-1/2, see
    /// ferric_scf::df_gradient).
    #[pyo3(signature = (density, operator="coulomb", omega=None, aux=None, memory_budget_gb=None))]
    fn grad_jk<'py>(
        &self,
        py: Python<'py>,
        density: PyReadonlyArray2<'py, f64>,
        operator: &str,
        omega: Option<f64>,
        aux: Option<&PyBasisSet>,
        memory_budget_gb: Option<f64>,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
        let op = parse_operator(operator, omega)?;
        let d = checked_density(density, self.prep.nbasis())?;
        let prep = &self.prep;
        let (gj, gk) = match aux {
            None => py.allow_threads(|| -> Result<_, FerricError> {
                let bounds = SchwarzBounds::compute(op, prep)?;
                let gj = ferric_scf::gradient::twoelectron_j_gradient(prep, op, &bounds, &d)?;
                // twoelectron_gradient = gJ - ½gK (Γ = ½DD - ¼DD exchange).
                let g2 = ferric_scf::gradient::twoelectron_gradient(prep, op, &bounds, &d)?;
                let gk = 2.0 * (&gj - &g2);
                Ok((gj, gk))
            }),
            Some(aux) => {
                let budget = ferric_scf::rhf::resolve_three_index_budget(
                    super::budget_bytes_from_gb(memory_budget_gb).unwrap_or(0),
                );
                let dfbs = self.aux_prep(aux)?;
                let natoms = self.mol.atoms.len();
                py.allow_threads(|| -> Result<_, FerricError> {
                    use ferric_scf::df_gradient::{df_jk_gradient, df_k_gradient_any};
                    let gj = df_jk_gradient(natoms, prep, &dfbs, op, Some(&d), &[], budget)?;
                    let gk = df_k_gradient_any(natoms, prep, &dfbs, op, &d, budget)?;
                    Ok((gj, gk))
                })
            }
        }
        .map_err(make_err)?;
        Ok((
            PyArray2::from_owned_array(py, gj),
            PyArray2::from_owned_array(py, gk),
        ))
    }
    /// The exchange-correlation functional `functional` on the grid
    /// `run_dft` uses (same `grid_radial` / `grid_angular` / `grid_prune`
    /// kwargs and defaults), built once and reused per density.
    #[pyo3(signature = (functional, grid_radial=None, grid_angular=None, grid_prune=None))]
    fn xc_functional(
        &self,
        py: Python<'_>,
        functional: &str,
        grid_radial: Option<usize>,
        grid_angular: Option<usize>,
        grid_prune: Option<&str>,
    ) -> PyResult<PyXcFunctional> {
        use ferric_dft::xc_trait::XcContribution;
        let main =
            super::resolve_dft_grid(grid_radial, grid_angular, grid_prune)?.unwrap_or_default();
        let grids = ferric_scf::ks_gradient::KsGradGrids {
            main,
            nlc: ferric_dft::grid::AtomicGridConfig::default_nlc(),
        };
        let (mol, bs) = (&self.mol, &self.bs);
        let closed = py
            .allow_threads(|| {
                ferric_dft::ks::KsXc::new(mol, bs, functional, &grids.main, &grids.nlc)
            })
            .map_err(|e| PyValueError::new_err(format!("functional '{functional}': {e}")))?;
        let k = closed.k_mix();
        Ok(PyXcFunctional {
            name: functional.to_string(),
            mol: self.mol.clone(),
            bs: self.bs.clone(),
            prep: PreparedBasis::new(&self.mol, &self.bs).map_err(make_err)?,
            grids,
            closed,
            open: None,
            exchange_mix: (k.sr, k.lr, k.omega),
        })
    }
    /// The two-electron (+ XC) potential for a symmetric TOTAL density:
    /// `(E, V)` with `V = J - 0.5 * K` (Hartree-Fock, `xc=None`) or
    /// `V = J - 0.5 * (c_sr K_erfc + c_lr K_erf) + V_xc` (an `XcFunctional`
    /// of this molecule), and `E = 0.5 Tr[D (J - 0.5 mix K)] (+ E_xc)`, so
    /// the Fock matrix is `hcore() + V` and the electronic energy
    /// `Tr[D hcore()] + E` (PySCF's `get_veff`).
    ///
    /// `aux` fits J and K as an SCF with `df_j_aux=df_k_aux=aux` does
    /// (range-separated exchange under erfc and erf); `None` is exact. The
    /// J/K builders are kept per (exchange mix, aux).
    #[pyo3(signature = (density, xc=None, aux=None, memory_budget_gb=None))]
    fn veff<'py>(
        &self,
        py: Python<'py>,
        density: PyReadonlyArray2<'py, f64>,
        xc: Option<PyRef<'py, PyXcFunctional>>,
        aux: Option<&PyBasisSet>,
        memory_budget_gb: Option<f64>,
    ) -> PyResult<(f64, Bound<'py, PyArray2<f64>>)> {
        if let Some(x) = &xc {
            self.same_system(x)?;
        }
        let d = checked_density(density, self.prep.nbasis())?;
        let mix = xc.as_ref().map_or((1.0, 1.0, 0.0), |x| x.exchange_mix);
        let map = self.veff_map(py, mix, aux, memory_budget_gb)?;
        let half = 0.5 * &d;
        let [g, _] = py
            .allow_threads(|| map.fock([&half, &half], true))
            .map_err(make_err)?;
        let mut e = 0.5 * (&d * &g).sum();
        let mut v = g;
        if let Some(x) = &xc {
            let x: &PyXcFunctional = x;
            let (e_xc, v_xc) = py.allow_threads(|| x.xc_closed(&d));
            e += e_xc;
            v += &v_xc;
        }
        Ok((e, PyArray2::from_owned_array(py, v)))
    }
    /// Spin-polarized `veff`: `(E, V_a, V_b)` with
    /// `V_s = J(D_a + D_b) - mix K(D_s) (+ V_xc,s)` and
    /// `E = 0.5 Σ_s Tr[D_s (J - mix K(D_s))] (+ E_xc)`.
    #[pyo3(signature = (d_a, d_b, xc=None, aux=None, memory_budget_gb=None))]
    fn veff_polarized<'py>(
        &self,
        py: Python<'py>,
        d_a: PyReadonlyArray2<'py, f64>,
        d_b: PyReadonlyArray2<'py, f64>,
        xc: Option<PyRefMut<'py, PyXcFunctional>>,
        aux: Option<&PyBasisSet>,
        memory_budget_gb: Option<f64>,
    ) -> PyResult<(f64, Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
        let mut xc = xc;
        if let Some(x) = &xc {
            self.same_system(x)?;
        }
        let n = self.prep.nbasis();
        let (da, db) = (checked_density(d_a, n)?, checked_density(d_b, n)?);
        let mix = xc.as_ref().map_or((1.0, 1.0, 0.0), |x| x.exchange_mix);
        let map = self.veff_map(py, mix, aux, memory_budget_gb)?;
        let [mut va, mut vb] = py
            .allow_threads(|| map.fock([&da, &db], false))
            .map_err(make_err)?;
        let mut e = 0.5 * ((&da * &va).sum() + (&db * &vb).sum());
        if let Some(x) = xc.as_mut() {
            let (e_xc, xa, xb) = x.xc_open(&da, &db)?;
            e += e_xc;
            va += &xa;
            vb += &xb;
        }
        Ok((
            e,
            PyArray2::from_owned_array(py, va),
            PyArray2::from_owned_array(py, vb),
        ))
    }
    /// Closed-shell CPHF (`xc=None`) or CPKS (`xc` an `XcFunctional` of
    /// this molecule) for converged MOs `mo_coeff` (nao, nmo) with energies
    /// `mo_energy` and `nocc` doubly occupied orbitals. The XC kernel is
    /// built once at `D = 2 C_occ C_occᵀ`. See `Cphf`.
    #[pyo3(signature = (mo_coeff, mo_energy, nocc, xc=None, aux=None, memory_budget_gb=None))]
    #[allow(clippy::too_many_arguments)]
    fn cphf(
        &self,
        py: Python<'_>,
        mo_coeff: PyReadonlyArray2<'_, f64>,
        mo_energy: Vec<f64>,
        nocc: usize,
        xc: Option<PyRef<'_, PyXcFunctional>>,
        aux: Option<&PyBasisSet>,
        memory_budget_gb: Option<f64>,
    ) -> PyResult<PyCphf> {
        let c = checked_coeffs(mo_coeff, self.prep.nbasis(), "mo_coeff")?;
        let frame = SpinFrame::new(&c, &mo_energy, nocc, "cphf")?;
        let spec = xc.as_ref().map(|x| x.spec());
        let half = frame.c_occ.dot(&frame.c_occ.t());
        let (mol, bs) = (&self.mol, &self.bs);
        let aux = response_aux(aux, memory_budget_gb);
        let fock = py
            .allow_threads(|| {
                FockResponse::new(mol, bs, spec.as_ref(), (&half, &half), aux_ref(&aux), true)
            })
            .map_err(make_err)?;
        Ok(PyCphf {
            resp: Response {
                fock,
                frames: vec![frame],
            },
        })
    }
    /// Open-shell UCPHF/UCPKS for converged per-spin MOs; see `Ucphf`.
    #[pyo3(signature = (
        mo_coeff_a, mo_coeff_b, mo_energy_a, mo_energy_b, nocc_a, nocc_b,
        xc=None, aux=None, memory_budget_gb=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn ucphf(
        &self,
        py: Python<'_>,
        mo_coeff_a: PyReadonlyArray2<'_, f64>,
        mo_coeff_b: PyReadonlyArray2<'_, f64>,
        mo_energy_a: Vec<f64>,
        mo_energy_b: Vec<f64>,
        nocc_a: usize,
        nocc_b: usize,
        xc: Option<PyRef<'_, PyXcFunctional>>,
        aux: Option<&PyBasisSet>,
        memory_budget_gb: Option<f64>,
    ) -> PyResult<PyUcphf> {
        let n = self.prep.nbasis();
        let ca = checked_coeffs(mo_coeff_a, n, "mo_coeff_a")?;
        let cb = checked_coeffs(mo_coeff_b, n, "mo_coeff_b")?;
        let fa = SpinFrame::new(&ca, &mo_energy_a, nocc_a, "ucphf alpha")?;
        let fb = SpinFrame::new(&cb, &mo_energy_b, nocc_b, "ucphf beta")?;
        let spec = xc.as_ref().map(|x| x.spec());
        let (da, db) = (fa.c_occ.dot(&fa.c_occ.t()), fb.c_occ.dot(&fb.c_occ.t()));
        let (mol, bs) = (&self.mol, &self.bs);
        let aux = response_aux(aux, memory_budget_gb);
        let fock = py
            .allow_threads(|| {
                FockResponse::new(mol, bs, spec.as_ref(), (&da, &db), aux_ref(&aux), true)
            })
            .map_err(make_err)?;
        Ok(PyUcphf {
            resp: Response {
                fock,
                frames: vec![fa, fb],
            },
        })
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
                // RI-J needs a Cholesky of the metric, which the erf metric of
                // a Coulomb-fitting basis does not admit (DfK regularizes
                // instead): keep K usable and report J's failure on use.
                let j = DfJ::from_source(raw, op, &dfbs, budget, None).map_err(|e| e.to_string());
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
    /// `Err` (the Cholesky failure) when the metric admits no RI-J.
    j: Result<DfJ<'static>, String>,
    k: DfK<'static>,
    nao: usize,
    naux: usize,
}

impl PyDfJk {
    fn build_j(&mut self, d: &Array2<f64>) -> PyResult<Array2<f64>> {
        let dfj = self.j.as_mut().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "RI-J is unavailable for this operator/aux basis ({e}); k() still works"
            ))
        })?;
        let mut j = Array2::<f64>::zeros((self.nao, self.nao));
        dfj.build(d, &mut j).map_err(make_err)?;
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

/// One exchange-correlation functional on one molecule's grid, from
/// `AoIntegrals.xc_functional(name)`. `eval`/`eval_polarized` give the
/// semilocal (+ VV10) E_xc and V_xc the SCF adds; `grad`/`grad_polarized`
/// the matching XC gradient at fixed density (grid response included).
/// Exact exchange is NOT included anywhere here: `exchange_mix` says how
/// much to add (see its doc).
#[pyclass]
#[pyo3(name = "XcFunctional")]
struct PyXcFunctional {
    name: String,
    mol: Molecule,
    bs: BasisSet,
    prep: PreparedBasis,
    grids: ferric_scf::ks_gradient::KsGradGrids,
    closed: ferric_dft::ks::KsXc,
    /// Built on the first spin-polarized call.
    open: Option<ferric_dft::ks::KsXcUks>,
    exchange_mix: (f64, f64, f64),
}

impl PyXcFunctional {
    fn spec(&self) -> XcSpec {
        XcSpec {
            name: self.name.clone(),
            grid: self.grids.main.clone(),
            mix: self.exchange_mix,
        }
    }
    /// `(E_xc, V_xc)` at a checked total density, full f64 (fresh ladder).
    fn xc_closed(&self, d: &Array2<f64>) -> (f64, Array2<f64>) {
        use ferric_dft::xc_trait::XcContribution;
        self.closed.start_scf();
        let mut v = Array2::<f64>::zeros(d.dim());
        let e = self.closed.add_xc(d, &mut v);
        (e, v)
    }
    /// `(E_xc, V_a, V_b)` at checked per-spin densities.
    fn xc_open(
        &mut self,
        d_a: &Array2<f64>,
        d_b: &Array2<f64>,
    ) -> PyResult<(f64, Array2<f64>, Array2<f64>)> {
        use ferric_dft::xc_trait::UksXcContribution;
        let xc = self.open()?;
        xc.start_scf();
        let (mut v_a, mut v_b) = (Array2::zeros(d_a.dim()), Array2::zeros(d_b.dim()));
        let e = xc.add_xc_uks(d_a, d_b, &mut v_a, &mut v_b);
        Ok((e, v_a, v_b))
    }
    fn open(&mut self) -> PyResult<&ferric_dft::ks::KsXcUks> {
        if self.open.is_none() {
            let g = &self.grids;
            let uks =
                ferric_dft::ks::KsXcUks::new(&self.mol, &self.bs, &self.name, &g.main, &g.nlc)
                    .map_err(|e| make_err(FerricError::from(e)))?;
            self.open = Some(uks);
        }
        Ok(self.open.as_ref().expect("built above"))
    }
}

#[pymethods]
impl PyXcFunctional {
    /// The functional name as given.
    #[getter]
    fn name(&self) -> &str {
        &self.name
    }
    /// Exact-exchange admixture `(c_sr, c_lr, omega)`. The closed-shell
    /// Fock matrix is `h + J - 0.5 * (c_sr * K_erfc + c_lr * K_erf) + V_xc`
    /// with K under erfc(omega)/erf(omega) (`jk(D, "erfc", omega=...)`); for
    /// omega == 0 that is `h + J - 0.5 * c_sr * K + V_xc` (c_sr == c_lr).
    /// Per spin: `h + J - (c_sr K_erfc[D_s] + c_lr K_erf[D_s]) + V_s`.
    #[getter]
    fn exchange_mix(&self) -> (f64, f64, f64) {
        self.exchange_mix
    }
    /// `(E_xc, V_xc)` for a symmetric TOTAL density (VV10 included).
    fn eval<'py>(
        &self,
        py: Python<'py>,
        density: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<(f64, Bound<'py, PyArray2<f64>>)> {
        let d = checked_density(density, self.prep.nbasis())?;
        let (e, v) = py.allow_threads(|| self.xc_closed(&d));
        Ok((e, PyArray2::from_owned_array(py, v)))
    }
    /// `(E_xc, V_a, V_b)` for symmetric per-spin densities.
    fn eval_polarized<'py>(
        &mut self,
        py: Python<'py>,
        d_a: PyReadonlyArray2<'py, f64>,
        d_b: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<(f64, Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
        let n = self.prep.nbasis();
        let (d_a, d_b) = (checked_density(d_a, n)?, checked_density(d_b, n)?);
        self.open()?;
        let (e, v_a, v_b) = py.allow_threads(|| self.xc_open(&d_a, &d_b))?;
        Ok((
            e,
            PyArray2::from_owned_array(py, v_a),
            PyArray2::from_owned_array(py, v_b),
        ))
    }
    /// XC gradient at fixed total density, (natoms, 3): semilocal with grid
    /// response, plus VV10 (the `run_dft(with_gradient=True)` XC term).
    fn grad<'py>(
        &self,
        py: Python<'py>,
        density: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let d = checked_density(density, self.prep.nbasis())?;
        let g = py
            .allow_threads(|| {
                ferric_scf::ks_gradient::xc_gradient_closed(
                    &self.mol,
                    &self.prep,
                    &self.bs,
                    &self.name,
                    &d,
                    &self.grids,
                )
            })
            .map_err(make_err)?;
        Ok(PyArray2::from_owned_array(py, g))
    }
    /// XC gradient at fixed per-spin densities, (natoms, 3).
    fn grad_polarized<'py>(
        &self,
        py: Python<'py>,
        d_a: PyReadonlyArray2<'py, f64>,
        d_b: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let n = self.prep.nbasis();
        let (d_a, d_b) = (checked_density(d_a, n)?, checked_density(d_b, n)?);
        let g = py
            .allow_threads(|| {
                ferric_scf::ks_gradient::xc_gradient_open(
                    &self.mol,
                    &self.prep,
                    &self.bs,
                    &self.name,
                    (&d_a, &d_b),
                    &self.grids,
                )
            })
            .map_err(make_err)?;
        Ok(PyArray2::from_owned_array(py, g))
    }
    /// Closed-shell kernel response V1 = d/dt V_xc(D + t D1) at t = 0, for
    /// symmetric `density` (total) and `d1`. LDA / GGA / hybrid / RSH-GGA
    /// only: meta-GGA and VV10 have no kernel here and raise ValueError.
    /// Builds the kernel at `density` on each call (`cphf` caches it).
    fn response<'py>(
        &self,
        py: Python<'py>,
        density: PyReadonlyArray2<'py, f64>,
        d1: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let n = self.prep.nbasis();
        let (d, d1) = (checked_density(density, n)?, checked_density(d1, n)?);
        let spec = self.spec();
        let (h, h1) = (0.5 * &d, 0.5 * &d1);
        let [v, _] = py
            .allow_threads(|| -> Result<_, FerricError> {
                let k = spec.kernel(&self.mol, &self.prep, (&h, &h))?;
                Ok(apply_kernel(&k, (&h1, &h1)))
            })
            .map_err(make_err)?;
        Ok(PyArray2::from_owned_array(py, v))
    }
    /// Spin-polarized kernel response `(V1_a, V1_b)` at `(d_a, d_b)` along
    /// `(d1_a, d1_b)`; same functional limits as `response`.
    fn response_polarized<'py>(
        &self,
        py: Python<'py>,
        d_a: PyReadonlyArray2<'py, f64>,
        d_b: PyReadonlyArray2<'py, f64>,
        d1_a: PyReadonlyArray2<'py, f64>,
        d1_b: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
        let n = self.prep.nbasis();
        let (d_a, d_b) = (checked_density(d_a, n)?, checked_density(d_b, n)?);
        let (d1_a, d1_b) = (checked_density(d1_a, n)?, checked_density(d1_b, n)?);
        let spec = self.spec();
        let [va, vb] = py
            .allow_threads(|| -> Result<_, FerricError> {
                let k = spec.kernel(&self.mol, &self.prep, (&d_a, &d_b))?;
                Ok(apply_kernel(&k, (&d1_a, &d1_b)))
            })
            .map_err(make_err)?;
        Ok((
            PyArray2::from_owned_array(py, va),
            PyArray2::from_owned_array(py, vb),
        ))
    }
    fn __repr__(&self) -> String {
        format!("XcFunctional({:?}, nao={})", self.name, self.prep.nbasis())
    }
}

/// What a response object needs from an `XcFunctional`: the name, its grid
/// and the exact-exchange mix `(c_sr, c_lr, omega)`.
#[derive(Clone)]
struct XcSpec {
    name: String,
    grid: ferric_dft::grid::AtomicGridConfig,
    mix: (f64, f64, f64),
}

impl XcSpec {
    /// The f_xc kernel at `(d_a, d_b)`. Meta-GGA (no τ kernel) and VV10 (no
    /// nonlocal kernel) are refused: the response would silently miss them.
    fn kernel(
        &self,
        mol: &Molecule,
        prep: &PreparedBasis,
        (d_a, d_b): (&Array2<f64>, &Array2<f64>),
    ) -> Result<ferric_scf::rohf::FxcKernelStore, FerricError> {
        let def = ferric_dft::libxc::xc_def_from_name(&self.name)
            .map_err(|e| FerricError::General(format!("libxc: {e:?}")))?;
        let mgga = def
            .funcs
            .iter()
            .any(|f| f.family() == ferric_dft::libxc::FunctionalFamily::MetaGga);
        if mgga || def.vv10.is_some() {
            return Err(FerricError::General(format!(
                "functional '{}': no exchange-correlation kernel for {} (LDA, GGA, hybrid and \
                 range-separated GGA only)",
                self.name,
                if mgga { "meta-GGA" } else { "VV10" }
            )));
        }
        ferric_scf::rohf::FxcKernelStore::build(mol, prep, &self.grid, &self.name, None, d_a, d_b)
    }
}

fn apply_kernel(
    k: &ferric_scf::rohf::FxcKernelStore,
    (d1_a, d1_b): (&Array2<f64>, &Array2<f64>),
) -> [Array2<f64>; 2] {
    let (va, vb) = (k.response())(d1_a, d1_b);
    [va, vb]
}

/// The exchange operator of one K build in the response.
#[derive(Clone, Copy)]
enum KOp {
    Coulomb,
    Erfc,
    Erf,
}

/// Density-fitted response J/K against one auxiliary basis: RI-J, plus
/// RI-K under each operator the exchange mix needs. The response densities
/// are indefinite; the fitters take them on their density path as they are
/// (as the analytic Hessian's CPHF does).
struct DfResponse {
    j: std::sync::Mutex<DfJ<'static>>,
    /// `[Coulomb]` (global hybrid / HF), `[erfc, erf]` (range-separated), or
    /// empty (no exact exchange).
    k: Vec<std::sync::Mutex<DfK<'static>>>,
}

impl DfResponse {
    fn new(
        obs: &PreparedBasis,
        dfbs: &PreparedBasis,
        mix: (f64, f64, f64),
        budget: usize,
    ) -> Result<Self, FerricError> {
        let coulomb = Operator::coulomb();
        let mut raw = ThreeIndexSource::build(coulomb, obs, dfbs, budget)?;
        let mut k = Vec::new();
        if mix.2 > 0.0 {
            for op in [Operator::erfc(mix.2), Operator::erf(mix.2)] {
                k.push(std::sync::Mutex::new(DfK::new(op, obs, dfbs, budget)?));
            }
        } else if mix.0 != 0.0 {
            k.push(std::sync::Mutex::new(DfK::from_full_raw(
                &mut raw, obs, dfbs, coulomb, budget, None,
            )?));
        }
        let j = std::sync::Mutex::new(DfJ::from_source(raw, coulomb, dfbs, budget, None)?);
        Ok(DfResponse { j, k })
    }

    fn coulomb(&self, d: &Array2<f64>) -> Result<Array2<f64>, FerricError> {
        let mut out = Array2::zeros(d.dim());
        let mut j = self.j.lock().unwrap_or_else(|e| e.into_inner());
        j.build(d, &mut out)?;
        Ok(out)
    }

    fn exchange(&self, op: KOp, d: &Array2<f64>) -> Result<Array2<f64>, FerricError> {
        let idx = match op {
            KOp::Coulomb | KOp::Erfc => 0,
            KOp::Erf => 1,
        };
        let mut out = Array2::zeros(d.dim());
        let mut k = self.k[idx].lock().unwrap_or_else(|e| e.into_inner());
        k.update_density(d);
        k.build(d, &mut out)?;
        Ok(out)
    }
}

/// The linear Fock response per spin to symmetric `(D1_a, D1_b)`:
/// `F1_s = J(D1_a + D1_b) - [c_sr K_erfc(D1_s) + c_lr K_erf(D1_s)] + V1xc_s`
/// (omega = 0: `c_sr K(D1_s)`; Hartree-Fock: c_sr = 1, no kernel). Exact
/// four-centre J/K, or RI-J/RI-K with `df`.
struct FockResponse {
    prep: PreparedBasis,
    coulomb: SchwarzBounds,
    /// `(erfc, erf)` bounds of a range-separated functional.
    rsh: Option<(SchwarzBounds, SchwarzBounds)>,
    mix: (f64, f64, f64),
    kernel: Option<ferric_scf::rohf::FxcKernelStore>,
    df: Option<DfResponse>,
}

impl FockResponse {
    /// `aux` (with its resolved budget) selects RI-J/RI-K; `with_kernel`
    /// adds the f_xc kernel at `(d_a, d_b)` (the response), without it this
    /// is the plain J / exchange-mix K map (`XcFunctional.veff`).
    fn new(
        mol: &Molecule,
        bs: &BasisSet,
        xc: Option<&XcSpec>,
        (d_a, d_b): (&Array2<f64>, &Array2<f64>),
        aux: Option<(&BasisSet, usize)>,
        with_kernel: bool,
    ) -> Result<Self, FerricError> {
        let prep = PreparedBasis::new(mol, bs)?;
        let coulomb = SchwarzBounds::compute(Operator::coulomb(), &prep)?;
        // Hartree-Fock (no functional): full exact exchange, no kernel.
        let mix = xc.map_or((1.0, 1.0, 0.0), |x| x.mix);
        let rsh = if mix.2 > 0.0 {
            Some((
                SchwarzBounds::compute(Operator::erfc(mix.2), &prep)?,
                SchwarzBounds::compute(Operator::erf(mix.2), &prep)?,
            ))
        } else {
            None
        };
        let kernel = match xc {
            Some(x) if with_kernel => Some(x.kernel(mol, &prep, (d_a, d_b))?),
            _ => None,
        };
        let df = match aux {
            Some((aux_bs, budget)) => {
                let dfbs = PreparedBasis::new(mol, aux_bs)?;
                Some(DfResponse::new(&prep, &dfbs, mix, budget)?)
            }
            None => None,
        };
        Ok(FockResponse {
            prep,
            coulomb,
            rsh,
            mix,
            kernel,
            df,
        })
    }

    fn jk(&self, bounds: &SchwarzBounds, d: &Array2<f64>) -> Result<[Array2<f64>; 2], FerricError> {
        let n = self.prep.nbasis();
        let (mut j, mut k) = (Array2::zeros((n, n)), Array2::zeros((n, n)));
        ferric_scf::rhf::build_jk(
            &ParallelContext::default(),
            &self.prep,
            bounds,
            1e-12,
            d,
            &mut j,
            &mut k,
        )?;
        Ok([j, k])
    }

    fn coulomb(&self, d: &Array2<f64>) -> Result<Array2<f64>, FerricError> {
        match &self.df {
            Some(df) => df.coulomb(d),
            None => Ok(self.jk(&self.coulomb, d)?[0].clone()),
        }
    }

    fn k_of(&self, op: KOp, d: &Array2<f64>) -> Result<Array2<f64>, FerricError> {
        if let Some(df) = &self.df {
            return df.exchange(op, d);
        }
        let bounds = match (op, &self.rsh) {
            (KOp::Coulomb, _) => &self.coulomb,
            (KOp::Erfc, Some((sr, _))) => sr,
            (KOp::Erf, Some((_, lr))) => lr,
            (_, None) => unreachable!("erf/erfc exchange only for a range-separated mix"),
        };
        let [_, k] = self.jk(bounds, d)?;
        Ok(k)
    }

    /// `c_sr K_erfc(D) + c_lr K_erf(D)`, or `c_sr K(D)` at omega = 0.
    fn exchange(&self, d: &Array2<f64>) -> Result<Array2<f64>, FerricError> {
        let (c_sr, c_lr, _) = self.mix;
        Ok(match &self.rsh {
            Some(_) => c_sr * &self.k_of(KOp::Erfc, d)? + c_lr * &self.k_of(KOp::Erf, d)?,
            None if c_sr != 0.0 => c_sr * &self.k_of(KOp::Coulomb, d)?,
            None => Array2::zeros(d.dim()),
        })
    }

    /// `[F1_a, F1_b]`; `same_spin` (closed shell, D1_a == D1_b) builds the
    /// exchange and the kernel once.
    fn fock(
        &self,
        d1: [&Array2<f64>; 2],
        same_spin: bool,
    ) -> Result<[Array2<f64>; 2], FerricError> {
        let j = self.coulomb(&(d1[0] + d1[1]))?;
        let ka = self.exchange(d1[0])?;
        let kb = if same_spin {
            ka.clone()
        } else {
            self.exchange(d1[1])?
        };
        let mut f = [&j - &ka, &j - &kb];
        if let Some(k) = &self.kernel {
            let [va, vb] = apply_kernel(k, (d1[0], d1[1]));
            f[0] += &va;
            f[1] += &vb;
        }
        Ok(f)
    }
}

/// One spin's MO frame: occupied and virtual coefficients and the
/// `(nvir, nocc)` denominators `e_a - e_i`.
struct SpinFrame {
    c_occ: Array2<f64>,
    c_vir: Array2<f64>,
    denom: Array2<f64>,
}

impl SpinFrame {
    fn new(c: &Array2<f64>, e: &[f64], nocc: usize, what: &str) -> PyResult<Self> {
        let nmo = c.ncols();
        if e.len() != nmo || nocc == 0 || nocc >= nmo {
            return Err(PyValueError::new_err(format!(
                "{what}: need mo_energy of length {nmo} (got {}) and 0 < nocc < {nmo} (got {nocc})",
                e.len()
            )));
        }
        let denom = Array2::from_shape_fn((nmo - nocc, nocc), |(a, i)| e[nocc + a] - e[i]);
        if denom.iter().any(|&x| !(x > 0.0)) {
            return Err(PyValueError::new_err(format!(
                "{what}: e_a - e_i must be positive for every virtual/occupied pair (not an \
                 aufbau reference, or orbital energies out of order)"
            )));
        }
        Ok(SpinFrame {
            c_occ: c.slice(ndarray::s![.., ..nocc]).to_owned(),
            c_vir: c.slice(ndarray::s![.., nocc..]).to_owned(),
            denom,
        })
    }
    /// `C_v U C_oᵀ + C_o Uᵀ C_vᵀ`.
    fn density(&self, u: &Array2<f64>) -> Array2<f64> {
        let x = self.c_vir.dot(u).dot(&self.c_occ.t());
        &x + &x.t()
    }
    fn vo(&self, f: &Array2<f64>) -> Array2<f64> {
        self.c_vir.t().dot(f).dot(&self.c_occ)
    }
}

/// The orbital-Hessian (A + B) action and its preconditioned-CG solve for
/// one or two spin frames. Closed shell: one frame, occupation 2.
struct Response {
    fock: FockResponse,
    frames: Vec<SpinFrame>,
}

impl Response {
    fn closed(&self) -> bool {
        self.frames.len() == 1
    }
    /// Per-spin AO response densities of rotations `u` (closed: the TOTAL
    /// `2 (C_v U C_oᵀ + h.c.)`).
    fn densities(&self, u: &[Array2<f64>]) -> Vec<Array2<f64>> {
        let occ = if self.closed() { 2.0 } else { 1.0 };
        self.frames
            .iter()
            .zip(u)
            .map(|(f, x)| occ * &f.density(x))
            .collect()
    }
    fn apply(&self, u: &[Array2<f64>]) -> Result<Vec<Array2<f64>>, FerricError> {
        let d1 = self.densities(u);
        let f1 = if self.closed() {
            let h = 0.5 * &d1[0];
            let [fa, _] = self.fock.fock([&h, &h], true)?;
            vec![fa]
        } else {
            self.fock.fock([&d1[0], &d1[1]], false)?.to_vec()
        };
        Ok(self
            .frames
            .iter()
            .zip(u)
            .zip(&f1)
            .map(|((fr, x), f)| &fr.denom * x + fr.vo(f))
            .collect())
    }
    /// Preconditioned conjugate gradient on (A + B) U = rhs from U = 0,
    /// stopping when max |residual| < tol. A non-positive curvature means
    /// the reference is not a minimum (A + B not positive definite).
    fn solve(
        &self,
        rhs: &[Array2<f64>],
        tol: f64,
        max_iter: usize,
    ) -> Result<(Vec<Array2<f64>>, usize, f64), FerricError> {
        let dot = |a: &[Array2<f64>], b: &[Array2<f64>]| -> f64 {
            a.iter().zip(b).map(|(x, y)| (x * y).sum()).sum()
        };
        let max_abs = |a: &[Array2<f64>]| {
            a.iter()
                .flat_map(|x| x.iter())
                .fold(0.0f64, |m, v| m.max(v.abs()))
        };
        let precond = |r: &[Array2<f64>]| -> Vec<Array2<f64>> {
            self.frames
                .iter()
                .zip(r)
                .map(|(f, x)| x / &f.denom)
                .collect()
        };
        let mut x: Vec<Array2<f64>> = rhs.iter().map(|b| Array2::zeros(b.dim())).collect();
        let mut r: Vec<Array2<f64>> = rhs.to_vec();
        let mut resid = max_abs(&r);
        if resid < tol {
            return Ok((x, 0, resid));
        }
        let mut z = precond(&r);
        let mut p = z.clone();
        let mut rz = dot(&r, &z);
        for it in 1..=max_iter {
            let ap = self.apply(&p)?;
            let pap = dot(&p, &ap);
            if !(pap > 0.0) {
                return Err(FerricError::Convergence(format!(
                    "CPHF: non-positive curvature p.(A+B)p = {pap:.3e} at iteration {it}; the \
                     reference is not a minimum of the orbital Hessian"
                )));
            }
            let alpha = rz / pap;
            for k in 0..x.len() {
                x[k].scaled_add(alpha, &p[k]);
                r[k].scaled_add(-alpha, &ap[k]);
            }
            resid = max_abs(&r);
            if resid < tol {
                return Ok((x, it, resid));
            }
            z = precond(&r);
            let rz_new = dot(&r, &z);
            let beta = rz_new / rz;
            rz = rz_new;
            for k in 0..p.len() {
                p[k] = &z[k] + &(beta * &p[k]);
            }
        }
        Err(FerricError::Convergence(format!(
            "CPHF did not converge in {max_iter} iterations (max |residual| {resid:.3e} > tol \
             {tol:.1e})"
        )))
    }
}

/// `cphf`/`ucphf`'s `aux` kwarg with its resolved memory budget.
fn response_aux(
    aux: Option<&PyBasisSet>,
    memory_budget_gb: Option<f64>,
) -> Option<(BasisSet, usize)> {
    aux.map(|a| {
        let budget = ferric_scf::rhf::resolve_three_index_budget(
            super::budget_bytes_from_gb(memory_budget_gb).unwrap_or(0),
        );
        (a.inner.clone(), budget)
    })
}

fn aux_ref(aux: &Option<(BasisSet, usize)>) -> Option<(&BasisSet, usize)> {
    aux.as_ref().map(|(b, n)| (b, *n))
}

/// Check a list of rotations against the frames' `(nvir, nocc)` shapes.
fn checked_rotations(
    frames: &[SpinFrame],
    u: Vec<PyReadonlyArray2<'_, f64>>,
) -> PyResult<Vec<Array2<f64>>> {
    if u.len() != frames.len() {
        return Err(PyValueError::new_err(format!(
            "expected {} rotation matrix(es), got {}",
            frames.len(),
            u.len()
        )));
    }
    frames
        .iter()
        .zip(u)
        .enumerate()
        .map(|(sp, (f, x))| {
            let x = x.as_array().to_owned();
            if x.dim() != f.denom.dim() {
                return Err(PyValueError::new_err(format!(
                    "rotation {sp} has shape {:?}, expected (nvir, nocc) = {:?}",
                    x.dim(),
                    f.denom.dim()
                )));
            }
            if !x.iter().all(|v| v.is_finite()) {
                return Err(PyValueError::new_err("rotation contains NaN or inf"));
            }
            Ok(x)
        })
        .collect()
}

/// Closed-shell CPHF/CPKS for one SCF solution, from `AoIntegrals.cphf`.
///
/// Rotations `U` are `(nvir, nocc)` (virtual rows). `matvec(U)` is the
/// real orbital-Hessian action `(A + B) U = (e_a - e_i) U + C_vᵀ F1(D1) C_o`
/// with `D1 = density(U) = 2 (C_v U C_oᵀ + C_o Uᵀ C_vᵀ)` and F1 the
/// linear Fock response (J - ½·exchange mix·K + f_xc). `solve(rhs)` returns
/// U with `(A + B) U = rhs`: for a one-electron perturbation h1 use
/// `rhs = -C_vᵀ h1 C_o`, then `D1 = density(U)`; for a Z-vector, the
/// Lagrangian. Exact four-centre J/K, or RI-J/RI-K with `cphf(aux=...)`.
#[pyclass]
#[pyo3(name = "Cphf")]
struct PyCphf {
    resp: Response,
}

/// Open-shell (UHF/UKS) twin of `Cphf`, from `AoIntegrals.ucphf`: rotations
/// are the pair `(U_a, U_b)`, `density` gives `D1_s = C_vs U_s C_osᵀ + h.c.`
/// (unit occupation) and `F1_s = J(D1_a + D1_b) - exchange mix·K(D1_s) +
/// f_xc_s`.
#[pyclass]
#[pyo3(name = "Ucphf")]
struct PyUcphf {
    resp: Response,
}

/// Shared pymethod bodies of `Cphf` / `Ucphf`.
fn response_matvec<'py>(
    py: Python<'py>,
    resp: &Response,
    u: Vec<PyReadonlyArray2<'py, f64>>,
) -> PyResult<Vec<Bound<'py, PyArray2<f64>>>> {
    let u = checked_rotations(&resp.frames, u)?;
    let out = py.allow_threads(|| resp.apply(&u)).map_err(make_err)?;
    Ok(out
        .into_iter()
        .map(|a| PyArray2::from_owned_array(py, a))
        .collect())
}

fn response_solve<'py>(
    py: Python<'py>,
    resp: &Response,
    rhs: Vec<PyReadonlyArray2<'py, f64>>,
    tol: f64,
    max_iter: usize,
) -> PyResult<Vec<Bound<'py, PyArray2<f64>>>> {
    let rhs = checked_rotations(&resp.frames, rhs)?;
    let (x, _, _) = py
        .allow_threads(|| resp.solve(&rhs, tol, max_iter))
        .map_err(make_err)?;
    Ok(x.into_iter()
        .map(|a| PyArray2::from_owned_array(py, a))
        .collect())
}

fn response_densities<'py>(
    py: Python<'py>,
    resp: &Response,
    u: Vec<PyReadonlyArray2<'py, f64>>,
) -> PyResult<Vec<Bound<'py, PyArray2<f64>>>> {
    let u = checked_rotations(&resp.frames, u)?;
    Ok(resp
        .densities(&u)
        .into_iter()
        .map(|a| PyArray2::from_owned_array(py, a))
        .collect())
}

#[pymethods]
impl PyCphf {
    /// `(A + B) U`, shape (nvir, nocc).
    fn matvec<'py>(
        &self,
        py: Python<'py>,
        u: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(response_matvec(py, &self.resp, vec![u])?.remove(0))
    }
    /// U with `(A + B) U = rhs` (preconditioned CG, max |residual| < tol).
    #[pyo3(signature = (rhs, tol=1e-9, max_iter=100))]
    fn solve<'py>(
        &self,
        py: Python<'py>,
        rhs: PyReadonlyArray2<'py, f64>,
        tol: f64,
        max_iter: usize,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(response_solve(py, &self.resp, vec![rhs], tol, max_iter)?.remove(0))
    }
    /// AO response density `2 (C_v U C_oᵀ + C_o Uᵀ C_vᵀ)` (total).
    fn density<'py>(
        &self,
        py: Python<'py>,
        u: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(response_densities(py, &self.resp, vec![u])?.remove(0))
    }
}

#[pymethods]
impl PyUcphf {
    /// `((A + B) U)_a, ((A + B) U)_b`.
    fn matvec<'py>(
        &self,
        py: Python<'py>,
        u_a: PyReadonlyArray2<'py, f64>,
        u_b: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
        let mut v = response_matvec(py, &self.resp, vec![u_a, u_b])?;
        let b = v.pop().expect("two spins");
        Ok((v.pop().expect("two spins"), b))
    }
    /// `(U_a, U_b)` with `(A + B) U = (rhs_a, rhs_b)`.
    #[pyo3(signature = (rhs_a, rhs_b, tol=1e-9, max_iter=100))]
    fn solve<'py>(
        &self,
        py: Python<'py>,
        rhs_a: PyReadonlyArray2<'py, f64>,
        rhs_b: PyReadonlyArray2<'py, f64>,
        tol: f64,
        max_iter: usize,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
        let mut v = response_solve(py, &self.resp, vec![rhs_a, rhs_b], tol, max_iter)?;
        let b = v.pop().expect("two spins");
        Ok((v.pop().expect("two spins"), b))
    }
    /// Per-spin AO response densities `C_vs U_s C_osᵀ + h.c.`.
    fn density<'py>(
        &self,
        py: Python<'py>,
        u_a: PyReadonlyArray2<'py, f64>,
        u_b: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
        let mut v = response_densities(py, &self.resp, vec![u_a, u_b])?;
        let b = v.pop().expect("two spins");
        Ok((v.pop().expect("two spins"), b))
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
                "fock_hook must be callable (hook(D) -> (E, V) closed-shell, \
                 hook(D_a, D_b) -> (E, V_a, V_b) open-shell)",
            ));
        }
        Ok(PyFockHook {
            callable: callable.clone().unbind(),
            nao,
            err: None,
        })
    }

    /// Copy one returned potential and check it is `(nao, nao)` and
    /// symmetric.
    fn potential(&self, v: PyReadonlyArray2<'_, f64>, name: &str) -> PyResult<Array2<f64>> {
        let v = v.as_array().to_owned();
        let n = self.nao;
        if v.dim() != (n, n) {
            return Err(PyValueError::new_err(format!(
                "fock_hook returned {name} of shape {:?}, expected ({n}, {n})",
                v.dim()
            )));
        }
        let asym = asymmetry(&v, &format!("fock_hook's {name}"))?;
        if asym > HOOK_SYMMETRY_TOL {
            return Err(PyValueError::new_err(format!(
                "fock_hook returned a non-symmetric {name} (max |{name} - {name}^T| = {asym:.3e})"
            )));
        }
        Ok(v)
    }

    fn energy(e: f64) -> PyResult<f64> {
        if e.is_finite() {
            Ok(e)
        } else {
            Err(PyValueError::new_err(format!(
                "fock_hook returned a non-finite energy {e}"
            )))
        }
    }

    fn eval(&self, py: Python<'_>, d: &Array2<f64>) -> PyResult<(f64, Array2<f64>)> {
        let out = self.callable.call1(py, (PyArray2::from_array(py, d),))?;
        let (e, v): (f64, PyReadonlyArray2<'_, f64>) = out.extract(py).map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(
                "fock_hook must return (E: float, V: 2-D float array)",
            )
        })?;
        Ok((Self::energy(e)?, self.potential(v, "V")?))
    }

    fn eval_spin(
        &self,
        py: Python<'_>,
        d_a: &Array2<f64>,
        d_b: &Array2<f64>,
    ) -> PyResult<(f64, Array2<f64>, Array2<f64>)> {
        let args = (PyArray2::from_array(py, d_a), PyArray2::from_array(py, d_b));
        let out = self.callable.call1(py, args)?;
        type Ret<'a> = (f64, PyReadonlyArray2<'a, f64>, PyReadonlyArray2<'a, f64>);
        let (e, v_a, v_b): Ret<'_> = out.extract(py).map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(
                "an open-shell fock_hook must return (E: float, V_a, V_b: 2-D float arrays)",
            )
        })?;
        Ok((
            Self::energy(e)?,
            self.potential(v_a, "V_a")?,
            self.potential(v_b, "V_b")?,
        ))
    }

    fn stash<T>(&mut self, r: PyResult<T>) -> Result<T, FerricError> {
        r.map_err(|e| {
            self.err = Some(e);
            FerricError::General("fock_hook failed".into())
        })
    }

    /// The closed-shell `FnMut` the SCF calls.
    pub(crate) fn call(&mut self, d: &Array2<f64>) -> Result<(f64, Array2<f64>), FerricError> {
        let r = Python::with_gil(|py| self.eval(py, d));
        self.stash(r)
    }

    /// The open-shell `FnMut` the SCF calls.
    pub(crate) fn call_spin(
        &mut self,
        d_a: &Array2<f64>,
        d_b: &Array2<f64>,
    ) -> Result<(f64, Array2<f64>, Array2<f64>), FerricError> {
        let r = Python::with_gil(|py| self.eval_spin(py, d_a, d_b));
        self.stash(r)
    }

    /// The hook's own error if it raised, otherwise `scf`'s error as usual.
    pub(crate) fn finish<T>(self, scf: Result<T, FerricError>) -> PyResult<T> {
        match (self.err, scf) {
            (Some(e), _) => Err(e),
            (None, r) => r.map_err(make_err),
        }
    }
}

/// Which open-shell solver [`solve_open_shell`] runs.
#[derive(Clone, Copy)]
pub(crate) enum OpenShell {
    Uhf,
    Rohf,
}

/// `run_uhf` / `run_rohf`'s SCF: the plain solve, or with a Python
/// `hook(D_a, D_b) -> (E, V_a, V_b)` the hooked one
/// (`solve_uhf_with_hook` / `solve_rohf_with_hook`). Unsupported settings
/// are a `ValueError` before any work. GIL released for the SCF.
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_open_shell(
    py: Python<'_>,
    kind: OpenShell,
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    config: &ferric_scf::rhf::RhfConfig,
    fock_hook: Option<&Bound<'_, PyAny>>,
) -> PyResult<ferric_scf::result::ScfResult> {
    use ferric_scf::{rohf, uhf};
    let Some(hook) = fock_hook else {
        return py
            .allow_threads(|| match kind {
                OpenShell::Uhf => uhf::solve_uhf(ctx, mol, prep, bounds, config),
                OpenShell::Rohf => rohf::solve_rohf(ctx, mol, prep, op, bounds, config),
            })
            .map_err(make_err);
    };
    ferric_scf::rhf::validate_fock_hook(config)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let mut h = PyFockHook::new(hook, prep.nbasis())?;
    let r = py.allow_threads(|| {
        let mut f = |d_a: &Array2<f64>, d_b: &Array2<f64>| h.call_spin(d_a, d_b);
        match kind {
            OpenShell::Uhf => uhf::solve_uhf_with_hook(ctx, mol, prep, bounds, config, &mut f),
            OpenShell::Rohf => rohf::solve_rohf_with_hook(ctx, mol, prep, bounds, config, &mut f),
        }
    });
    h.finish(r)
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
    m.add_class::<PyXcFunctional>()?;
    m.add_class::<PyCphf>()?;
    m.add_class::<PyUcphf>()?;
    m.add_function(wrap_pyfunction!(einsum, m)?)
}
