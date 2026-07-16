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
use ferric_mp2::scs::{scs_mp2, scs_mp2_2terfc, ScsMp2Config, ScsMp2TerfcConfig};
use ferric_scf::ks_gradient::ks_gradient_closed;
use ferric_scf::optimize::{optimize_geometry, OptimizeConfig};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::uhf::{solve_uhf, UhfConfig};
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

/// Convert an optional `memory_budget_gb` kwarg (GiB) to the explicit
/// `Option<usize>` bytes expected by the method configs / the unified resolver.
/// `None` (unset) stays `None` → the resolver auto-detects. A non-positive value
/// also maps to `None` (treated as "unset").
fn budget_bytes_from_gb(memory_budget_gb: Option<f64>) -> Option<usize> {
    memory_budget_gb.and_then(|g| {
        let b = ferric_core::memory::gib_to_bytes(g);
        if b == 0 { None } else { Some(b) }
    })
}

fn rhf_config(k_builder: Option<&str>) -> RhfConfig {
    RhfConfig { k_builder: k_builder.map(|s| s.to_string()), ..Default::default() }
}

/// Convert the Python `point_charges` / `external_field` kwargs into an
/// `ExternalPotential`. Returns `None` when both are unset (no perturbation),
/// matching `RhfConfig`'s "None = no external potential" convention.
fn build_external_potential(
    point_charges: Option<Vec<(f64, f64, f64, f64)>>,
    external_field: Option<(f64, f64, f64)>,
) -> Option<ferric_core::external_potential::ExternalPotential> {
    let pcs: Vec<ferric_core::external_potential::PointCharge> = point_charges
        .unwrap_or_default()
        .into_iter()
        .map(|(q, x, y, z)| ferric_core::external_potential::PointCharge { q, x, y, z })
        .collect();
    let field = external_field.map(|(ex, ey, ez)| [ex, ey, ez]);
    if pcs.is_empty() && field.is_none() {
        None
    } else {
        Some(ferric_core::external_potential::ExternalPotential { point_charges: pcs, field })
    }
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

/// Closed-shell RHF (also UHF/ROHF convergence aids apply to the open-shell
/// drivers). Exposes the full SCF knob set, matching the CLI `[scf]` TOML section
/// for parity — same names, same defaults, same units.
///
/// Convergence aids:
///   level_shift     virtual-virtual block shift in Hartree (open-shell; 0.2 is a
///                   useful default for OH-like doublets at LDA/PBE). 0 = off.
///   mom_after_iter  Maximum-Overlap Method: pin the occupied set by AO-overlap
///                   after this many DIIS iters. 0 = aufbau throughout. Fixes
///                   occupied-set flip-flop non-convergence.
/// Fock builders:
///   k_builder       "direct" (default) or "link" (linear-scaling exchange).
///   df_j_aux        auxiliary basis name for density-fitted Coulomb (RI-J).
///   df_k_aux        auxiliary basis name for density-fitted exchange (RI-K);
///                   should be a JK-fit basis, not an MP2-fit basis.
/// External perturbation:
///   point_charges   list of (q, x, y, z) classical point charges (Hartree
///                   atomic units; coordinates in Bohr) added to the one-electron
///                   Hamiltonian and nuclear repulsion. None/empty = no charges.
///   external_field  uniform (Ex, Ey, Ez) electric field in Hartree atomic units.
///                   None = no field.
#[pyfunction]
#[pyo3(signature = (
    mol, basis_set,
    max_iter=None, energy_conv=None, density_conv=None, diis_size=None,
    integral_thresh=None, k_builder=None, df_j_aux=None, df_k_aux=None,
    level_shift=None, mom_after_iter=None,
    point_charges=None, external_field=None,
))]
#[allow(clippy::too_many_arguments)]
fn run_rhf(
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    max_iter: Option<usize>,
    energy_conv: Option<f64>,
    density_conv: Option<f64>,
    diis_size: Option<usize>,
    integral_thresh: Option<f64>,
    k_builder: Option<&str>,
    df_j_aux: Option<&str>,
    df_k_aux: Option<&str>,
    level_shift: Option<f64>,
    mom_after_iter: Option<usize>,
    point_charges: Option<Vec<(f64, f64, f64, f64)>>,
    external_field: Option<(f64, f64, f64)>,
) -> PyResult<PyRhfResult> {
    // Apply ECP core-electron counts (no-op without an ECP basis) so nelec()
    // gives the valence count; the effective nuclear charge is set inside
    // PreparedBasis::new from basis_set.ecps.
    let mut emol = mol.inner.clone();
    emol.apply_ecp(&basis_set.inner);
    let prep = PreparedBasis::new(&emol, &basis_set.inner).map_err(make_err)?;
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).map_err(make_err)?;
    // Defaults mirror the CLI `[scf]` section (config.rs) so CLI and Python agree.
    let config = RhfConfig {
        max_iter: max_iter.unwrap_or(100),
        energy_conv: energy_conv.unwrap_or(1e-3),
        density_conv: density_conv.unwrap_or(1e-6),
        diis_size: diis_size.unwrap_or(8),
        integral_thresh: integral_thresh.unwrap_or(1e-12),
        k_builder: k_builder.map(|s| s.to_string()),
        df_j_aux: df_j_aux.map(|s| s.to_string()),
        df_k_aux: df_k_aux.map(|s| s.to_string()),
        level_shift: level_shift.unwrap_or(0.0),
        mom_after_iter: mom_after_iter.unwrap_or(0),
        external_potential: build_external_potential(point_charges, external_field),
        ..Default::default()
    };
    let ctx = ParallelContext::default();
    let r = solve_rhf(&ctx, &emol, &prep, op, &bounds, &config).map_err(make_err)?;
    Ok(PyRhfResult {
        energy: r.energy, converged: r.converged, iterations: r.iterations,
        computed_quartets: r.computed_quartets,
        density_data: r.density_total, orbital_energies_data: r.eps_alpha,
    })
}

// ── UHF (open-shell) ──

#[pyclass]
#[pyo3(name = "UhfResult")]
struct PyUhfResult {
    #[pyo3(get)] energy: f64,
    #[pyo3(get)] converged: bool,
    #[pyo3(get)] iterations: usize,
    #[pyo3(get)] computed_quartets: usize,
    density_alpha_data: Array2<f64>,
    density_beta_data: Array2<f64>,
    eps_alpha_data: Vec<f64>,
    eps_beta_data: Vec<f64>,
}

#[pymethods]
impl PyUhfResult {
    fn density_alpha<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        PyArray2::from_array(py, &self.density_alpha_data)
    }
    fn density_beta<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        PyArray2::from_array(py, &self.density_beta_data)
    }
    fn orbital_energies_alpha<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.eps_alpha_data.clone())
    }
    fn orbital_energies_beta<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.eps_beta_data.clone())
    }
}

/// Unrestricted Hartree-Fock (open-shell). α/β electron counts come from the
/// molecule's `charge` and `multiplicity` (set via `Molecule.from_xyz_string`).
/// Exposes the same SCF knob set as `run_rhf`; the convergence aids `level_shift`
/// and `mom_after_iter` are especially useful for open-shell doublets / radicals
/// where DIIS plateaus or the occupied set flip-flops.
#[pyfunction]
#[pyo3(signature = (
    mol, basis_set,
    max_iter=None, energy_conv=None, density_conv=None, diis_size=None,
    integral_thresh=None, k_builder=None, df_j_aux=None, df_k_aux=None,
    level_shift=None, mom_after_iter=None,
    point_charges=None, external_field=None,
))]
#[allow(clippy::too_many_arguments)]
fn run_uhf(
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    max_iter: Option<usize>,
    energy_conv: Option<f64>,
    density_conv: Option<f64>,
    diis_size: Option<usize>,
    integral_thresh: Option<f64>,
    k_builder: Option<&str>,
    df_j_aux: Option<&str>,
    df_k_aux: Option<&str>,
    level_shift: Option<f64>,
    mom_after_iter: Option<usize>,
    point_charges: Option<Vec<(f64, f64, f64, f64)>>,
    external_field: Option<(f64, f64, f64)>,
) -> PyResult<PyUhfResult> {
    let mut emol = mol.inner.clone();
    emol.apply_ecp(&basis_set.inner);
    let prep = PreparedBasis::new(&emol, &basis_set.inner).map_err(make_err)?;
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).map_err(make_err)?;
    // Same defaults as run_rhf / the CLI [scf] section.
    let config = UhfConfig {
        max_iter: max_iter.unwrap_or(100),
        energy_conv: energy_conv.unwrap_or(1e-3),
        density_conv: density_conv.unwrap_or(1e-6),
        diis_size: diis_size.unwrap_or(8),
        integral_thresh: integral_thresh.unwrap_or(1e-12),
        k_builder: k_builder.map(|s| s.to_string()),
        df_j_aux: df_j_aux.map(|s| s.to_string()),
        df_k_aux: df_k_aux.map(|s| s.to_string()),
        level_shift: level_shift.unwrap_or(0.0),
        mom_after_iter: mom_after_iter.unwrap_or(0),
        external_potential: build_external_potential(point_charges, external_field),
        ..Default::default()
    };
    let ctx = ParallelContext::default();
    let r = solve_uhf(&ctx, &emol, &prep, &bounds, &config).map_err(make_err)?;
    // density_beta / mos_beta / eps_beta are always populated for the UHF path.
    Ok(PyUhfResult {
        energy: r.energy,
        converged: r.converged,
        iterations: r.iterations,
        computed_quartets: r.computed_quartets,
        density_alpha_data: r.density_alpha,
        density_beta_data: r.density_beta.unwrap_or_else(|| r.density_total.clone()),
        eps_alpha_data: r.eps_alpha,
        eps_beta_data: r.eps_beta.unwrap_or_default(),
    })
}

// ── Optimize ──

#[pyclass]
#[pyo3(name = "OptimizeResult")]
struct PyOptimizeResult {
    #[pyo3(get)] energy: f64,
    #[pyo3(get)] converged: bool,
    #[pyo3(get)] steps: usize,
    mol_data: Molecule,
}

#[pymethods]
impl PyOptimizeResult {
    /// The optimized geometry as a new `Molecule`.
    fn mol(&self) -> PyMolecule {
        PyMolecule { inner: self.mol_data.clone() }
    }
}

#[pyfunction]
#[pyo3(signature = (mol, basis_name, max_steps=None, e_conv=None, point_charges=None, external_field=None))]
fn run_optimize(mol: &PyMolecule, basis_name: &str,
                max_steps: Option<usize>, e_conv: Option<f64>,
                point_charges: Option<Vec<(f64, f64, f64, f64)>>,
                external_field: Option<(f64, f64, f64)>) -> PyResult<PyOptimizeResult> {
    let ctx = ParallelContext::default();
    let rhf_config = RhfConfig {
        external_potential: build_external_potential(point_charges, external_field),
        ..Default::default()
    };
    let r = optimize_geometry(&ctx, &mol.inner, basis_name, Operator::coulomb(),
                              &rhf_config,
                              &OptimizeConfig {
                                  max_steps: max_steps.unwrap_or(100),
                                  e_conv: e_conv.unwrap_or(1e-6),
                                  ..Default::default()
                              }).map_err(make_err)?;
    Ok(PyOptimizeResult { energy: r.energy, converged: r.converged, steps: r.steps, mol_data: r.mol })
}

// ── Electronic properties (ESP / Hirshfeld / Löwdin charges) ──

/// Accepts either an `RhfResult` or a `DftResult` — both carry a converged
/// closed-shell density, which is all these property functions need. Lets
/// `esp_at_atoms`/`hirshfeld_charges`/`lowdin_charges` work on any energy
/// result rather than being RHF-only.
enum DensitySource<'py> {
    Rhf(PyRef<'py, PyRhfResult>),
    Dft(PyRef<'py, PyDftResult>),
}

impl<'py> DensitySource<'py> {
    fn density(&self) -> &Array2<f64> {
        match self {
            DensitySource::Rhf(r) => &r.density_data,
            DensitySource::Dft(d) => &d.density_data,
        }
    }
}

impl<'py> FromPyObject<'py> for DensitySource<'py> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(r) = ob.extract::<PyRef<'py, PyRhfResult>>() {
            return Ok(DensitySource::Rhf(r));
        }
        if let Ok(d) = ob.extract::<PyRef<'py, PyDftResult>>() {
            return Ok(DensitySource::Dft(d));
        }
        Err(pyo3::exceptions::PyTypeError::new_err(
            "expected an RhfResult or DftResult",
        ))
    }
}

/// Electrostatic potential at each nucleus, in Hartree atomic units (e/Bohr).
/// `result` is an `RhfResult` or `DftResult` from a converged SCF.
#[pyfunction]
fn esp_at_atoms(mol: &PyMolecule, basis_set: &PyBasisSet, result: DensitySource) -> PyResult<Vec<f64>> {
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    ferric_rpa::properties::esp_at_atoms(&mol.inner, &prep, result.density()).map_err(make_err)
}

/// Hirshfeld partial charges (units of e), using the default free-atom
/// (single-exponential Slater) proatom reference. `result` is an `RhfResult`
/// or `DftResult` from a converged SCF.
#[pyfunction]
fn hirshfeld_charges(mol: &PyMolecule, basis_set: &PyBasisSet, result: DensitySource) -> PyResult<Vec<f64>> {
    ferric_rpa::properties::hirshfeld_charges(&mol.inner, &basis_set.inner, result.density(), None)
        .map_err(make_err)
}

/// Löwdin (symmetric-orthogonalization) partial charges (units of e).
/// Closed-shell only. `result` is an `RhfResult` or `DftResult` from a
/// converged SCF.
#[pyfunction]
fn lowdin_charges(mol: &PyMolecule, basis_set: &PyBasisSet, result: DensitySource) -> PyResult<Vec<f64>> {
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    ferric_rpa::properties::lowdin_charges(&mol.inner, &prep, result.density()).map_err(make_err)
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
#[pyo3(signature = (mol, basis_set, auxbasis, frozen_core=None, k_builder=None, memory_budget_gb=None))]
fn run_rimp2(mol: &PyMolecule, basis_set: &PyBasisSet, auxbasis: &PyBasisSet,
             frozen_core: Option<usize>, k_builder: Option<&str>,
             memory_budget_gb: Option<f64>) -> PyResult<PyRiMp2Result> {
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    let dfbs = PreparedBasis::new(&mol.inner, &auxbasis.inner).map_err(make_err)?;
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).map_err(make_err)?;
    let ctx = ParallelContext::default();
    let rhf = solve_rhf(&ctx, &mol.inner, &prep, op, &bounds, &rhf_config(k_builder)).map_err(make_err)?;
    if !rhf.converged {
        return Err(make_err(ferric_core::FerricError::ScfConvergence { iterations: rhf.iterations, last_energy: rhf.energy }));
    }
    let mp2 = ri_mp2(&mol.inner, &prep, &dfbs, op, &rhf,
                      &RiMp2Config { frozen_core: frozen_core.unwrap_or(0), memory_budget_bytes: budget_bytes_from_gb(memory_budget_gb) }).map_err(make_err)?;
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
    if !rhf.converged {
        return Err(make_err(ferric_core::FerricError::ScfConvergence { iterations: rhf.iterations, last_energy: rhf.energy }));
    }
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
#[pyo3(signature = (mol, basis_set, auxbasis, omega=None, frozen_core=None, k_builder=None, memory_budget_gb=None))]
fn run_attenuated_rimp2(mol: &PyMolecule, basis_set: &PyBasisSet, auxbasis: &PyBasisSet,
                        omega: Option<f64>, frozen_core: Option<usize>,
                        k_builder: Option<&str>,
                        memory_budget_gb: Option<f64>) -> PyResult<PyAttenuatedMp2Result> {
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    let dfbs = PreparedBasis::new(&mol.inner, &auxbasis.inner).map_err(make_err)?;
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).map_err(make_err)?;
    let ctx = ParallelContext::default();
    let rhf = solve_rhf(&ctx, &mol.inner, &prep, op, &bounds, &rhf_config(k_builder)).map_err(make_err)?;
    if !rhf.converged {
        return Err(make_err(ferric_core::FerricError::ScfConvergence { iterations: rhf.iterations, last_energy: rhf.energy }));
    }
    // omega is supplied in Å⁻¹; convert to Bohr⁻¹ for the operator.
    let cfg = AttenuatedMp2Config {
        omega: omega.unwrap_or(0.420) * ferric_mp2::attenuated::BOHR_INV_PER_ANG_INV,
        scaling: 1.0, frozen_core: frozen_core.unwrap_or(0),
        screen_thresh: None,
        memory_budget_bytes: budget_bytes_from_gb(memory_budget_gb),
    };
    let r = attenuated_ri_mp2(&mol.inner, &prep, &dfbs, &rhf, &cfg).map_err(make_err)?;
    Ok(PyAttenuatedMp2Result {
        total_energy: r.total_energy, rhf_energy: rhf.energy, mp2_corr: r.mp2_corr,
        e_os: r.spin_components.e_os, e_ss: r.spin_components.e_ss,
    })
}

// ── terfc-attenuated RI-MP2 (exact terfc via interpolation tables) ──

/// MP2(terfc): RI-MP2 with the EXACT tempered-erfc operator (Dutoi/Goldey
/// interpolation tables), a single cutoff `r0` (Å). SCF stays full-Coulomb; only
/// the MP2 correlation is attenuated. Requires the terfc tables on disk
/// (FERRIC_TERF_TABLE_DIR). Paper aDZ-optimal r0 = 1.05 Å.
#[pyfunction]
#[pyo3(signature = (mol, basis_set, auxbasis, r0=None, frozen_core=None, k_builder=None, memory_budget_gb=None))]
fn run_terfc_rimp2(mol: &PyMolecule, basis_set: &PyBasisSet, auxbasis: &PyBasisSet,
                   r0: Option<f64>, frozen_core: Option<usize>,
                   k_builder: Option<&str>,
                   memory_budget_gb: Option<f64>) -> PyResult<PyRiMp2Result> {
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    let dfbs = PreparedBasis::new(&mol.inner, &auxbasis.inner).map_err(make_err)?;
    let coul = Operator::coulomb();
    let bounds = SchwarzBounds::compute(coul, &prep).map_err(make_err)?;
    let ctx = ParallelContext::default();
    let rhf = solve_rhf(&ctx, &mol.inner, &prep, coul, &bounds, &rhf_config(k_builder)).map_err(make_err)?;
    if !rhf.converged {
        return Err(make_err(ferric_core::FerricError::ScfConvergence { iterations: rhf.iterations, last_energy: rhf.energy }));
    }
    // r0 supplied in Å; convert to Bohr for the operator.
    let r0_bohr = r0.unwrap_or(1.05) * 1.8897259886;
    let mp2 = ri_mp2(&mol.inner, &prep, &dfbs, Operator::terfc(r0_bohr), &rhf,
                      &RiMp2Config { frozen_core: frozen_core.unwrap_or(0), memory_budget_bytes: budget_bytes_from_gb(memory_budget_gb) }).map_err(make_err)?;
    Ok(PyRiMp2Result { total_energy: mp2.total_energy, rhf_energy: rhf.energy, mp2_corr: mp2.mp2_corr })
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
#[pyo3(signature = (mol, basis_set, auxbasis, c_os=None, c_ss=None, frozen_core=None, k_builder=None, memory_budget_gb=None))]
fn run_scs_mp2(mol: &PyMolecule, basis_set: &PyBasisSet, auxbasis: &PyBasisSet,
               c_os: Option<f64>, c_ss: Option<f64>, frozen_core: Option<usize>,
               k_builder: Option<&str>,
               memory_budget_gb: Option<f64>) -> PyResult<PyScsMp2Result> {
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    let dfbs = PreparedBasis::new(&mol.inner, &auxbasis.inner).map_err(make_err)?;
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).map_err(make_err)?;
    let ctx = ParallelContext::default();
    let rhf = solve_rhf(&ctx, &mol.inner, &prep, op, &bounds, &rhf_config(k_builder)).map_err(make_err)?;
    if !rhf.converged {
        return Err(make_err(ferric_core::FerricError::ScfConvergence { iterations: rhf.iterations, last_energy: rhf.energy }));
    }
    let cfg = ScsMp2Config {
        c_os: c_os.unwrap_or(6.0 / 5.0), c_ss: c_ss.unwrap_or(1.0 / 3.0),
        frozen_core: frozen_core.unwrap_or(0),
        memory_budget_bytes: budget_bytes_from_gb(memory_budget_gb),
    };
    let r = scs_mp2(&mol.inner, &prep, &dfbs, &rhf, &cfg).map_err(make_err)?;
    Ok(PyScsMp2Result {
        total_energy: r.total_energy, rhf_energy: rhf.energy, scs_corr: r.scs_corr,
        e_os: r.e_os, e_ss: r.e_ss,
    })
}

/// SCS-MP2(2terfc): dual-attenuated SCS-MP2 (Goldey, Dutoi, Head-Gordon, PCCP
/// 2013) using the EXACT terfc operator at two cutoffs `r0_bonded` < `r0_nonbonded`
/// (Å). E = c_OS·E_OS(r0_1) + c_SS·[E_SS(r0_2) − E_SS(r0_1)]. Requires the terfc
/// tables (FERRIC_TERF_TABLE_DIR). Paper defaults: r0=0.75/1.05 Å, c_OS=1.27, c_SS=4.05.
#[pyfunction]
#[pyo3(signature = (mol, basis_set, auxbasis, r0_bonded=None, r0_nonbonded=None, c_os=None, c_ss=None, frozen_core=None, k_builder=None, memory_budget_gb=None))]
#[allow(clippy::too_many_arguments)]
fn run_scs_mp2_2terfc(mol: &PyMolecule, basis_set: &PyBasisSet, auxbasis: &PyBasisSet,
                      r0_bonded: Option<f64>, r0_nonbonded: Option<f64>,
                      c_os: Option<f64>, c_ss: Option<f64>, frozen_core: Option<usize>,
                      k_builder: Option<&str>,
                      memory_budget_gb: Option<f64>) -> PyResult<PyScsMp2Result> {
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    let dfbs = PreparedBasis::new(&mol.inner, &auxbasis.inner).map_err(make_err)?;
    let coul = Operator::coulomb();
    let bounds = SchwarzBounds::compute(coul, &prep).map_err(make_err)?;
    let ctx = ParallelContext::default();
    let rhf = solve_rhf(&ctx, &mol.inner, &prep, coul, &bounds, &rhf_config(k_builder)).map_err(make_err)?;
    if !rhf.converged {
        return Err(make_err(ferric_core::FerricError::ScfConvergence { iterations: rhf.iterations, last_energy: rhf.energy }));
    }
    const ANG2BOHR: f64 = 1.8897259886;
    let cfg = ScsMp2TerfcConfig {
        r0_bonded: r0_bonded.unwrap_or(0.75) * ANG2BOHR,
        r0_nonbonded: r0_nonbonded.unwrap_or(1.05) * ANG2BOHR,
        c_os: c_os.unwrap_or(1.27), c_ss: c_ss.unwrap_or(4.05),
        frozen_core: frozen_core.unwrap_or(0),
        memory_budget_bytes: budget_bytes_from_gb(memory_budget_gb),
    };
    if cfg.r0_nonbonded <= cfg.r0_bonded {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "r0_nonbonded must be > r0_bonded",
        ));
    }
    let r = scs_mp2_2terfc(&mol.inner, &prep, &dfbs, &rhf, &cfg).map_err(make_err)?;
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
#[pyo3(signature = (mol, basis_set, auxbasis, omega=None, frozen_core=None, k_builder=None, formulation=None, memory_budget_gb=None))]
#[allow(clippy::too_many_arguments)]
fn run_rs_mp2_rpa(mol: &PyMolecule, basis_set: &PyBasisSet, auxbasis: &PyBasisSet,
                  omega: Option<f64>, frozen_core: Option<usize>,
                  k_builder: Option<&str>,
                  formulation: Option<&str>,
                  memory_budget_gb: Option<f64>) -> PyResult<PyRsMp2RpaResult> {
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
    if !rhf.converged {
        return Err(make_err(ferric_core::FerricError::ScfConvergence { iterations: rhf.iterations, last_energy: rhf.energy }));
    }
    // Map formulation string to enum.
    let form = match formulation.unwrap_or("delta-lr") {
        "delta-lr" => ferric_rpa::RsMp2RpaFormulation::DeltaLr,
        "coupled-rings" => ferric_rpa::RsMp2RpaFormulation::CoupledRings,
        other => return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown formulation \"{other}\"; expected \"delta-lr\" or \"coupled-rings\""
        ))),
    };
    // omega is supplied in Å⁻¹; convert to Bohr⁻¹ for the operator.
    let mut cfg = ferric_rpa::RsMp2RpaConfig {
        omega: omega.unwrap_or(0.420) * ferric_mp2::attenuated::BOHR_INV_PER_ANG_INV,
        frozen_core: frozen_core.unwrap_or(0),
        formulation: form,
        ..Default::default()
    };
    cfg.drpa.memory_budget_bytes = budget_bytes_from_gb(memory_budget_gb);
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
    #[pyo3(get)] converged: bool,
    vxc_data: Array2<f64>,
    density_data: Array2<f64>,
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

    fn density<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        PyArray2::from_array(py, &self.density_data)
    }

    /// Return the cached analytic nuclear gradient as an (natoms, 3) array.
    /// Returns `None` if the result was produced without `with_gradient=True`.
    fn gradient<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
        self.gradient_data.as_ref().map(|g| PyArray2::from_array(py, g))
    }
}

/// Kohn-Sham DFT (closed-shell). `functional` is an XC name (LDA/PBE/B3LYP/
/// wB97X-V/…). Convergence aids match the CLI `[scf]` section: `level_shift`
/// (Ha) and `mom_after_iter` help difficult DFT SCFs converge.
#[pyfunction]
#[pyo3(signature = (
    mol, basis_set, functional=None, k_builder=None, with_gradient=false,
    max_iter=None, energy_conv=None, density_conv=None,
    level_shift=None, mom_after_iter=None,
    point_charges=None, external_field=None,
))]
#[allow(clippy::too_many_arguments)]
fn run_dft(mol: &PyMolecule, basis_set: &PyBasisSet,
           functional: Option<&str>, k_builder: Option<&str>,
           with_gradient: bool,
           max_iter: Option<usize>, energy_conv: Option<f64>, density_conv: Option<f64>,
           level_shift: Option<f64>, mom_after_iter: Option<usize>,
           point_charges: Option<Vec<(f64, f64, f64, f64)>>,
           external_field: Option<(f64, f64, f64)>) -> PyResult<PyDftResult> {
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).map_err(make_err)?;
    let ctx = ParallelContext::default();
    let mut cfg = rhf_config(k_builder);
    if let Some(v) = max_iter { cfg.max_iter = v; }
    if let Some(v) = energy_conv { cfg.energy_conv = v; }
    if let Some(v) = density_conv { cfg.density_conv = v; }
    if let Some(v) = level_shift { cfg.level_shift = v; }
    if let Some(v) = mom_after_iter { cfg.mom_after_iter = v; }
    cfg.external_potential = build_external_potential(point_charges, external_field);
    let xc_name = functional.unwrap_or("LDA").to_string();
    cfg.xc = Some(xc_name.clone());
    // RI-J always on (matches PySCF density_fit reference convention).
    cfg.df_j_aux = Some("def2-universal-jkfit".to_string());
    // RI-K only matters for hybrid/RSH; harmless for pure DFT (path is bypassed
    // when k_mix.sr == 0 and k_mix.omega == 0).
    cfg.df_k_aux = Some("def2-universal-jkfit".to_string());
    // Run through the level-shift ladder (KS-DFT = solve_rhf with cfg.xc set),
    // so a hybrid on a hard system that DIIS-limit-cycles at level_shift=0
    // escalates the virtual-block shift instead of erroring out at max_iter.
    // The base cfg carries xc / grid / DF-JK aux into every rung. Only surface
    // a convergence error if the whole ladder fails to converge.
    let ladder = ferric_scf::ladder::ksdft_ladder(&cfg);
    let lr = ferric_scf::ladder::solve_rhf_ladder(&ctx, &mol.inner, &prep, op, &bounds, &ladder)
        .map_err(make_err)?;
    let rhf = lr.result;
    if !rhf.converged {
        return Err(make_err(ferric_core::FerricError::ScfConvergence { iterations: rhf.iterations, last_energy: rhf.energy }));
    }
    let nbf = rhf.mos_alpha.nrows();
    let gradient_data = if with_gradient {
        Some(
            ks_gradient_closed(&mol.inner, &prep, &basis_set.inner, op, &bounds, &xc_name, &rhf,
                                cfg.external_potential.as_ref())
                .map_err(make_err)?
        )
    } else {
        None
    };
    Ok(PyDftResult {
        total_energy: rhf.energy,
        converged: rhf.converged,
        vxc_data: Array2::<f64>::zeros((nbf, nbf)),
        density_data: rhf.density_total,
        gradient_data,
    })
}

/// Alias under the spec's canonical name. Same surface as `run_dft`.
#[pyfunction]
#[pyo3(signature = (
    mol, basis_set, functional=None, k_builder=None, with_gradient=false,
    max_iter=None, energy_conv=None, density_conv=None,
    level_shift=None, mom_after_iter=None,
    point_charges=None, external_field=None,
))]
#[allow(clippy::too_many_arguments)]
fn run_ksdft(mol: &PyMolecule, basis_set: &PyBasisSet,
             functional: Option<&str>, k_builder: Option<&str>,
             with_gradient: bool,
             max_iter: Option<usize>, energy_conv: Option<f64>, density_conv: Option<f64>,
             level_shift: Option<f64>, mom_after_iter: Option<usize>,
             point_charges: Option<Vec<(f64, f64, f64, f64)>>,
             external_field: Option<(f64, f64, f64)>) -> PyResult<PyDftResult> {
    run_dft(mol, basis_set, functional, k_builder, with_gradient,
            max_iter, energy_conv, density_conv, level_shift, mom_after_iter,
            point_charges, external_field)
}

// ── CC (stub) ──

#[pyclass]
#[pyo3(name = "CcResult")]
struct PyCcResult {
    #[pyo3(get)] correlation_energy: f64,
    #[pyo3(get)] t_correction: Option<f64>,
}

#[pyfunction]
#[pyo3(signature = (mol, basis_set, auxbasis, frozen_core=None, k_builder=None, memory_budget_gb=None))]
fn run_ccd(mol: &PyMolecule, basis_set: &PyBasisSet, auxbasis: &PyBasisSet,
           frozen_core: Option<usize>, k_builder: Option<&str>,
           memory_budget_gb: Option<f64>) -> PyResult<PyCcResult> {
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    let dfbs = PreparedBasis::new(&mol.inner, &auxbasis.inner).map_err(make_err)?;
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).map_err(make_err)?;
    let ctx = ParallelContext::default();
    let rhf = solve_rhf(&ctx, &mol.inner, &prep, op, &bounds, &rhf_config(k_builder)).map_err(make_err)?;
    if !rhf.converged {
        return Err(make_err(ferric_core::FerricError::ScfConvergence { iterations: rhf.iterations, last_energy: rhf.energy }));
    }
    let cfg = CcConfig { frozen_core: frozen_core.unwrap_or(0), memory_budget_bytes: budget_bytes_from_gb(memory_budget_gb), ..Default::default() };
    let r = run_ccd_inner(&mol.inner, &prep, &dfbs, op, &rhf, &cfg).map_err(make_err)?;
    Ok(PyCcResult { correlation_energy: r.correlation_energy, t_correction: None })
}

#[pyfunction]
#[pyo3(signature = (mol, basis_set, auxbasis, frozen_core=None, k_builder=None, memory_budget_gb=None))]
fn run_ccsd(mol: &PyMolecule, basis_set: &PyBasisSet, auxbasis: &PyBasisSet,
            frozen_core: Option<usize>, k_builder: Option<&str>,
            memory_budget_gb: Option<f64>) -> PyResult<PyCcResult> {
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    let dfbs = PreparedBasis::new(&mol.inner, &auxbasis.inner).map_err(make_err)?;
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).map_err(make_err)?;
    let ctx = ParallelContext::default();
    let rhf = solve_rhf(&ctx, &mol.inner, &prep, op, &bounds, &rhf_config(k_builder)).map_err(make_err)?;
    if !rhf.converged {
        return Err(make_err(ferric_core::FerricError::ScfConvergence { iterations: rhf.iterations, last_energy: rhf.energy }));
    }
    let cfg = CcConfig { frozen_core: frozen_core.unwrap_or(0), memory_budget_bytes: budget_bytes_from_gb(memory_budget_gb), ..Default::default() };
    let r = run_ccsd_inner(&mol.inner, &prep, &dfbs, op, &rhf, &cfg).map_err(make_err)?;
    Ok(PyCcResult { correlation_energy: r.correlation_energy, t_correction: None })
}

#[pyfunction]
#[pyo3(signature = (mol, basis_set, auxbasis, frozen_core=None, k_builder=None, memory_budget_gb=None))]
fn run_ccsd_t(mol: &PyMolecule, basis_set: &PyBasisSet, auxbasis: &PyBasisSet,
              frozen_core: Option<usize>, k_builder: Option<&str>,
              memory_budget_gb: Option<f64>) -> PyResult<PyCcResult> {
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    let dfbs = PreparedBasis::new(&mol.inner, &auxbasis.inner).map_err(make_err)?;
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).map_err(make_err)?;
    let ctx = ParallelContext::default();
    let rhf = solve_rhf(&ctx, &mol.inner, &prep, op, &bounds, &rhf_config(k_builder)).map_err(make_err)?;
    if !rhf.converged {
        return Err(make_err(ferric_core::FerricError::ScfConvergence { iterations: rhf.iterations, last_energy: rhf.energy }));
    }
    let cfg = CcConfig { frozen_core: frozen_core.unwrap_or(0), memory_budget_bytes: budget_bytes_from_gb(memory_budget_gb), ..Default::default() };
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
    /// Whether the static-dielectric eigensolve (Davidson or Lanczos) met its
    /// residual-norm convergence tolerance. `false` means `eigenvalues_static`
    /// / `eigenpotentials` are the eigensolver's best-effort Ritz pairs after
    /// exhausting its iteration budget, not verified eigenpairs.
    #[pyo3(get)] eigensolver_converged: bool,
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
    trunc_thresh=None, eigensolver_conv_thresh=None,
    run_diagnostics=false, k_builder=None, chi0_sparsity=None,
    memory_budget_gb=None,
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
    eigensolver_conv_thresh: Option<f64>,
    run_diagnostics: bool,
    k_builder: Option<&str>,
    chi0_sparsity: Option<&str>,
    memory_budget_gb: Option<f64>,
) -> PyResult<PyPdepRpaResult> {
    use ferric_rpa::config::{QuadratureConfig, QuadratureScheme, SternheimerConfig};
    use ferric_rpa::{run_pdep_rpa as run_pdep_rpa_inner, PdepRpaConfig};

    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    let dfbs = PreparedBasis::new(&mol.inner, &auxbasis.inner).map_err(make_err)?;
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).map_err(make_err)?;
    let ctx = ParallelContext::default();
    let rhf = solve_rhf(&ctx, &mol.inner, &prep, op, &bounds, &rhf_config(k_builder)).map_err(make_err)?;
    if !rhf.converged {
        return Err(make_err(ferric_core::FerricError::ScfConvergence { iterations: rhf.iterations, last_energy: rhf.energy }));
    }

    // Canonical parser (shared with the CLI): unknown schemes error rather than
    // silently running Gauss-Legendre.
    let scheme = QuadratureScheme::parse_config_str(quadrature)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("quadrature: {e}")))?;
    if u0.is_some() && !scheme.honours_u0() {
        eprintln!(
            "warning: u0 is ignored by quadrature='minimax' (it derives u0 from \
             n_quad); pass quadrature='gauss-legendre' or 'chebyshev-tan' to use it"
        );
    }
    let cfg = PdepRpaConfig {
        frozen_core: frozen_core.unwrap_or(0),
        trunc_thresh: trunc_thresh.unwrap_or(1e-4),
        eigensolver_max_vecs: 0,
        eigensolver_conv_thresh: eigensolver_conv_thresh.unwrap_or(1e-6),
        quadrature: QuadratureConfig {
            scheme,
            n_points: n_quad.unwrap_or(40),
            u0: u0.unwrap_or(0.5),
        },
        sternheimer: SternheimerConfig::default(),
        run_diagnostics,
        eigensolver: ferric_rpa::Eigensolver::default(),
        chi0_backend: ferric_rpa::config::Chi0Backend::default(),
        chi0_sparsity: ferric_rpa::config::Chi0Sparsity::parse_config_str(chi0_sparsity)
            .map_err(make_err)?,
        memory_budget_bytes: budget_bytes_from_gb(memory_budget_gb),
        // Python run_pdep_rpa is energy-only; the GW/property paths that consume
        // the inverse-dielectric stack have their own entry points (M9 gate).
        need_inv_dielectric_freq: false,
    };
    let r = run_pdep_rpa_inner(&mol.inner, &prep, &dfbs, op, &rhf, &cfg).map_err(make_err)?;
    if !r.eigensolver_converged {
        eprintln!(
            "warning: PDEP-RPA eigensolver did not fully converge (best-effort Ritz pairs; \
             eigenvalues_static/eigenpotentials are not verified to residual tolerance)"
        );
    }
    Ok(PyPdepRpaResult {
        rhf_energy: rhf.energy,
        e_rpa: r.e_rpa,
        total_energy: rhf.energy + r.e_rpa,
        n_eigenpotentials: r.n_eigenpotentials,
        e_rpa_dft_diag: r.e_rpa_dft_diag,
        eigensolver_converged: r.eigensolver_converged,
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
    // Safe-by-default threading: pin OpenBLAS to 1 thread (rayon owns ferric's
    // parallelism) unless the host set OPENBLAS_NUM_THREADS. Without this, an
    // `import ferric` in a host process inherits OpenBLAS's nproc default and
    // oversubscribes rayon × BLAS (3–5× slowdown, possible stack overflow).
    ferric_integrals::blas_threads::init_threading();
    // Register global signal handler for Ctrl+C
    let _ = ctrlc::set_handler(move || {
        eprintln!("\n[Ferric] Interrupt signal caught! Bailing out of kernels...");
        ferric_core::INTERRUPT.store(true, std::sync::atomic::Ordering::SeqCst);
    });

    m.add_class::<PyMolecule>()?;
    m.add_class::<PyBasisSet>()?;
    m.add_class::<PyRhfResult>()?;
    m.add_class::<PyUhfResult>()?;
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
    m.add_function(wrap_pyfunction!(run_uhf, m)?)?;
    m.add_function(wrap_pyfunction!(run_optimize, m)?)?;
    m.add_function(wrap_pyfunction!(esp_at_atoms, m)?)?;
    m.add_function(wrap_pyfunction!(hirshfeld_charges, m)?)?;
    m.add_function(wrap_pyfunction!(lowdin_charges, m)?)?;
    m.add_function(wrap_pyfunction!(run_rimp2, m)?)?;
    m.add_function(wrap_pyfunction!(run_attenuated_rimp2, m)?)?;
    m.add_function(wrap_pyfunction!(run_terfc_rimp2, m)?)?;
    m.add_function(wrap_pyfunction!(run_scs_mp2, m)?)?;
    m.add_function(wrap_pyfunction!(run_scs_mp2_2terfc, m)?)?;

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
