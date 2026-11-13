//! Python bindings for ferric (pyo3).
//!
//! Exposes the engine to Python: `Molecule` / `BasisSet` constructors plus
//! `run_rhf`, `run_uhf`, `run_rohf`, `run_rimp2`, `run_oo_rimp2`,
//! `run_attenuated_rimp2`, `run_scs_mp2`, the Laplace and coupled-cluster
//! drivers, and geometry optimization. Each binding wraps the corresponding
//! Rust driver and returns a result object with energies/components.
//!
//! Also exposes the conformer-ensemble machinery (`ConformerEnsemble`,
//! `boltzmann_weights`, `weighted_stats*`, `EnsembleDiagnostics`) so an RDKit
//! `EmbedMultipleConfs` conformer set can be turned into a Boltzmann-weighted
//! ferric property with a spread. Geometry input there is **Ångström**,
//! matching RDKit's `GetPositions()` and ferric's XYZ readers.
//!
//! Build with `uv run maturin develop --release` from the repository root; the
//! venv caveat is under "Python bindings from source" in
//! `site/src/using/installation.md`.

use ferric_cc::ccd::ccd as run_ccd_inner;
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::att_vv10::{
    att_mp2_vv10, att_mp2_vv10_scan, AttVv10Attenuator, AttVv10Config, AttVv10ScanArm,
    AttVv10SpinComponents,
};
use ferric_mp2::attenuated::{attenuated_ri_mp2, AttenuatedMp2Config};
use ferric_mp2::double_hybrid::{mp2_double_hybrid, DoubleHybridKind};
use ferric_mp2::laplace::{
    laplace_ri_mp2, laplace_sos_mp2_with_basis, SosFormulation, SosMp2Config,
};
use ferric_mp2::mp3::mp3_energy;
use ferric_mp2::oo_rimp2::{oo_ri_mp2, OoRiMp2Config};
use ferric_mp2::rimp2::{ri_mp2, RiMp2Config};
use ferric_mp2::scs::{scs_mp2, scs_mp2_2terfc, ScsMp2Config, ScsMp2TerfcConfig};
use ferric_scf::ks_gradient::{ks_gradient_closed_with_exchange_on_grids, KsGradGrids};
use ferric_scf::noise_floor::SolverFamily;
use ferric_scf::optimize::{optimize_geometry, optimize_geometry_uhf, OptimizeConfig};
use ferric_scf::result::ScfResult;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::rohf::RohfConfig;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::{solve_uhf, UhfConfig};
// NOTE: the spin-orbital `ferric_cc::ccsd::ccsd` is deliberately NOT imported
// here. `solve_rhf` always yields a restricted reference, so both `run_ccsd`
// and `run_ccsd_t` take the spin-adapted solver — the latter by expanding its
// amplitudes into the spin-orbital convention. An open-shell CC entry point
// would need to import it again.
use ferric_cc::ccsd_closed_shell::ccsd_closed_shell as run_ccsd_cs_inner;
use ferric_cc::ccsd_t_closed_shell::ccsd_t_closed_shell as run_ccsd_t_cs_inner;
use ferric_cc::CcConfig;
use ndarray::{Array2, Array3};
use numpy::{PyArray1, PyArray2, PyArray3};
use pyo3::prelude::*;

mod blocks;
mod cc;
mod cdft;
mod ci;
mod conformers;
mod dft;
mod gamma;
mod geometry;
mod gw;
mod integrals;
mod mp2;
mod pbc;
mod properties;
mod qmmm;
mod response_modes;
mod rpa;
mod scf;
mod tddft;

use cc::*;
use cdft::*;
use conformers::*;
use dft::*;
use gamma::*;
use geometry::*;
use gw::*;
use integrals::*;
use mp2::*;
use properties::*;
use qmmm::*;
use rpa::*;
use scf::*;
use tddft::*;

// ── Molecule ──

/// A molecular geometry (atoms, charge, multiplicity). Coordinates are stored
/// internally in Bohr; XYZ input is in Ångström.
#[pyclass]
#[pyo3(name = "Molecule")]
struct PyMolecule {
    inner: Molecule,
}

#[pymethods]
impl PyMolecule {
    /// Load a molecule from an XYZ file on disk.
    #[staticmethod]
    #[pyo3(signature = (path, charge=0, multiplicity=1))]
    fn from_xyz(path: &str, charge: i32, multiplicity: usize) -> PyResult<Self> {
        let mol = Molecule::load_xyz_with_charge(path, charge, multiplicity)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{e}")))?;
        Ok(PyMolecule { inner: mol })
    }
    /// Parse a molecule from an XYZ-format string, with explicit `charge` and
    /// `multiplicity` (2S+1). Use this for open-shell or charged molecules.
    #[staticmethod]
    #[pyo3(signature = (s, charge=0, multiplicity=1))]
    fn from_xyz_string(s: &str, charge: i32, multiplicity: usize) -> PyResult<Self> {
        let mol = Molecule::parse_xyz(s, charge, multiplicity)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))?;
        Ok(PyMolecule { inner: mol })
    }
    /// Build a molecule from arrays: `elements` holds one element symbol
    /// (`"O"`) or atomic number (`8`) per atom, `coords` the matching
    /// `(x, y, z)` rows in `units` (`"angstrom"`, the default, or `"bohr"`),
    /// `ghost` optional per-atom basis-only flags. The result equals
    /// `from_xyz_string` of the same atoms written with `repr()` coordinates.
    /// Raises `ValueError` on an unknown element or unit, mismatched lengths,
    /// a row that is not three finite numbers, a symbol carrying the XYZ `@`
    /// ghost marker (use `ghost=`) or an impossible charge/multiplicity.
    #[staticmethod]
    #[pyo3(signature = (elements, coords, units="angstrom", charge=0, multiplicity=1, ghost=None))]
    fn from_arrays(
        elements: Vec<Bound<'_, PyAny>>,
        coords: Vec<Vec<f64>>,
        units: &str,
        charge: i32,
        multiplicity: usize,
        ghost: Option<Vec<bool>>,
    ) -> PyResult<Self> {
        let inner = molecule_from_arrays(&elements, &coords, units, charge, multiplicity, ghost)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyMolecule { inner })
    }
    /// Classical nuclear repulsion energy in Hartree.
    fn nuclear_repulsion(&self) -> f64 {
        self.inner.nuclear_repulsion()
    }
    /// Number of atoms.
    fn natoms(&self) -> usize {
        self.inner.atoms.len()
    }
    /// Total electron count (accounts for `charge` and any ECP core electrons).
    fn nelec(&self) -> i32 {
        self.inner.nelec()
    }
    /// Cartesian coordinates of every atom in **Ångström**, one `(x, y, z)`
    /// tuple per atom in `symbols()` order. Internally ferric stores Bohr;
    /// this divides by the same Å→Bohr constant the XYZ parser multiplied
    /// by (see `ConformerEnsemble.coordinates()` for why divide, not
    /// multiply by the reciprocal), so a round trip is accurate to ~1 ulp.
    fn coords(&self) -> Vec<(f64, f64, f64)> {
        self.inner
            .atoms
            .iter()
            .map(|a| {
                (
                    a.x / ANGSTROM_TO_BOHR,
                    a.y / ANGSTROM_TO_BOHR,
                    a.zpos / ANGSTROM_TO_BOHR,
                )
            })
            .collect()
    }
    /// Cartesian coordinates in **Bohr** — the internal values, untouched.
    /// These are the units `run_rhf(..., point_charges=...)` and
    /// `QmmmSystem.point_charges()` use.
    fn coords_bohr(&self) -> Vec<(f64, f64, f64)> {
        self.inner
            .atoms
            .iter()
            .map(|a| (a.x, a.y, a.zpos))
            .collect()
    }
    /// Element symbols in atom order (ghost atoms keep their plain symbol;
    /// the `@` marker is not round-tripped).
    fn symbols(&self) -> Vec<String> {
        self.inner.atoms.iter().map(|a| a.symbol.clone()).collect()
    }
    /// Atomic numbers in atom order.
    ///
    /// Reports each element's true `Z`; it is **not** reduced by ECP core
    /// electrons even when an ECP-carrying basis has been applied (`nelec()` is
    /// the accessor that accounts for those).
    fn atomic_numbers(&self) -> Vec<i32> {
        self.inner.atoms.iter().map(|a| a.z).collect()
    }
    /// Per-atom ghost (basis-only center) flags, in atom order.
    ///
    /// A ghost carries basis functions but zero nuclear charge and zero
    /// electrons, so it appears in `symbols()`/`coords()` while contributing
    /// nothing to `nuclear_repulsion()` or `nelec()`. `symbols()` returns the
    /// plain element symbol for a ghost, so this flag is the only way to tell
    /// one apart.
    fn is_ghost(&self) -> Vec<bool> {
        self.inner.atoms.iter().map(|a| a.ghost).collect()
    }
    /// XYZ-format string (Ångström), ready to write to a file or feed back to
    /// `from_xyz_string`.
    ///
    /// The comment line records charge and multiplicity, which the XYZ format
    /// itself cannot carry. `from_xyz_string` does **not** parse them back, so
    /// pass them explicitly on any round-trip involving an ion or an open-shell
    /// species — otherwise the geometry survives and the electron count
    /// silently reverts to neutral singlet.
    fn to_xyz_string(&self) -> String {
        molecule_to_xyz_string(&self.inner)
    }
}

/// `Molecule.from_arrays`: the per-atom element (symbol or atomic number),
/// coordinates in `units`, ghost flags, then [`Molecule::from_atoms`]. A
/// symbol is stored as written (as the XYZ parser stores it); an atomic
/// number gets its canonical symbol. Ångström rows are multiplied by
/// `ANGSTROM_TO_BOHR`, the XYZ parser's conversion.
fn molecule_from_arrays(
    elements: &[Bound<'_, PyAny>],
    coords: &[Vec<f64>],
    units: &str,
    charge: i32,
    multiplicity: usize,
    ghost: Option<Vec<bool>>,
) -> Result<Molecule, String> {
    let scale = match units.to_ascii_lowercase().as_str() {
        "angstrom" => ANGSTROM_TO_BOHR,
        "bohr" => 1.0,
        _ => {
            return Err(format!(
                "Molecule.from_arrays: units must be \"angstrom\" or \"bohr\", got {units:?}"
            ))
        }
    };
    let n = elements.len();
    if coords.len() != n {
        return Err(format!(
            "Molecule.from_arrays: {n} elements but {} coordinate rows",
            coords.len()
        ));
    }
    let ghost = ghost.unwrap_or_else(|| vec![false; n]);
    if ghost.len() != n {
        return Err(format!(
            "Molecule.from_arrays: {n} elements but {} ghost flags",
            ghost.len()
        ));
    }
    let mut atoms = Vec::with_capacity(n);
    for (i, ((el, row), &is_ghost)) in elements.iter().zip(coords).zip(&ghost).enumerate() {
        let (symbol, z) = element_of(el, i)?;
        let [x, y, zpos] = <[f64; 3]>::try_from(row.as_slice()).map_err(|_| {
            format!(
                "Molecule.from_arrays: coordinate row {i} has {} values, expected 3",
                row.len()
            )
        })?;
        if !(x.is_finite() && y.is_finite() && zpos.is_finite()) {
            return Err(format!(
                "Molecule.from_arrays: coordinate row {i} is not finite: ({x}, {y}, {zpos})"
            ));
        }
        atoms.push(ferric_core::mol::Atom {
            symbol,
            z,
            x: x * scale,
            y: y * scale,
            zpos: zpos * scale,
            ghost: is_ghost,
            n_core_ecp: 0,
        });
    }
    Molecule::from_atoms(atoms, charge, multiplicity).map_err(|e| e.to_string())
}

/// One `from_arrays` element: a symbol (kept as written) or an atomic number
/// (given its canonical symbol), with its atomic number.
fn element_of(el: &Bound<'_, PyAny>, i: usize) -> Result<(String, i32), String> {
    use ferric_core::elements::{symbol_to_z, z_to_symbol};
    if let Ok(sym) = el.extract::<String>() {
        if sym.starts_with('@') {
            return Err(format!(
                "Molecule.from_arrays: element {i} is {sym:?}; mark ghost atoms with ghost=, \
                 not the XYZ '@' prefix"
            ));
        }
        let z = symbol_to_z(&sym)
            .ok_or_else(|| format!("Molecule.from_arrays: unknown element {sym:?} at atom {i}"))?;
        return Ok((sym, z));
    }
    let z = el.extract::<i32>().map_err(|_| {
        format!("Molecule.from_arrays: element {i} is neither a symbol nor an atomic number")
    })?;
    let sym = z_to_symbol(z)
        .ok_or_else(|| format!("Molecule.from_arrays: unknown atomic number {z} at atom {i}"))?;
    Ok((sym.to_string(), z))
}

/// Render a molecule as an XYZ-format string in Ångström.
///
/// A free function rather than an inherent method so the `#[pymethods]` block
/// stays a thin dispatch layer — its maintainability index is gated by
/// `scripts/complexity_gate.py` — and so the formatting is reusable from any
/// future binding that needs to emit a geometry.
fn molecule_to_xyz_string(mol: &Molecule) -> String {
    let mut s = format!(
        "{}\ncharge={} multiplicity={}\n",
        mol.atoms.len(),
        mol.charge,
        mol.multiplicity
    );
    for a in &mol.atoms {
        s.push_str(&format!(
            "{:<3} {:18.12} {:18.12} {:18.12}\n",
            a.symbol,
            a.x / ANGSTROM_TO_BOHR,
            a.y / ANGSTROM_TO_BOHR,
            a.zpos / ANGSTROM_TO_BOHR,
        ));
    }
    s
}

// ── BasisSet ──

/// A Gaussian basis set (orbital or auxiliary/RI-fitting).
#[pyclass]
#[pyo3(name = "BasisSet")]
struct PyBasisSet {
    inner: ferric_core::basis::BasisSet,
}

#[pymethods]
impl PyBasisSet {
    /// Load one of ferric's bundled basis sets by name (e.g. `"sto-3g"`,
    /// `"cc-pvdz"`, `"def2-svp"`, or an RI/JK auxiliary set like
    /// `"cc-pvdz-ri"`/`"def2-universal-jkfit"`). Raises `ValueError` if `name`
    /// is not a bundled set.
    ///
    /// d shells follow Gaussian's convention: the 6-31G family (6-31G
    /// through 6-31++G**) is Cartesian 6D; the 6-311G family is spherical 5D
    /// on every element (Na-Ar included); all other sets are spherical.
    /// PySCF defaults to 5D, so compare 6-31G-family numbers with
    /// `gto.M(..., cart=True)` (water/6-31G*: about 1.4 mHa otherwise).
    #[staticmethod]
    fn bundled(name: &str) -> PyResult<Self> {
        let bs = basis::bundled(name)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))?;
        Ok(PyBasisSet { inner: bs })
    }
    /// Primary names of every bundled basis set (aliases such as
    /// `"6-31g(d)"` or `"def2-tzvp/c"` are also accepted by `bundled`).
    #[staticmethod]
    fn bundled_names() -> Vec<&'static str> {
        basis::bundled_names()
    }
    /// Atomic numbers this basis has shells for, ascending.
    fn elements(&self) -> Vec<i32> {
        self.inner.elements()
    }
    /// The contracted shells this basis places on element `z`, in the order
    /// they enter the AO basis on every atom of that element: one
    /// `(l, pure, exponents, coefficients)` tuple per shell. Exponents are in
    /// Bohr^-2. Coefficients are the contraction coefficients as stored (after
    /// the load-time renormalisation to unit self-overlap); libint2 multiplies
    /// in each primitive's normalisation when it builds the shell, so they are
    /// coefficients of NORMALISED primitives. Raises `ValueError` when the
    /// basis has no shells for `z`.
    fn shells(&self, z: i32) -> PyResult<Vec<(i32, bool, Vec<f64>, Vec<f64>)>> {
        let shells = self.inner.for_element(z).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "{}",
                self.inner.missing_element_error(z)
            ))
        })?;
        Ok(shells
            .iter()
            .map(|s| (s.l, s.pure, s.exponents.clone(), s.coefficients.clone()))
            .collect())
    }
    /// Core electrons this basis's ECP replaces on element `z`; 0 when the
    /// basis carries no ECP for `z` (as the SCF applies it to non-ghost atoms).
    fn ecp_core_electrons(&self, z: i32) -> i32 {
        self.inner.ecp_for_element(z).map_or(0, |d| d.n_core)
    }
    /// Load a basis set from a Basis Set Exchange JSON file (the format of
    /// ferric's bundled sets). Contractions are renormalised to unit
    /// self-overlap exactly as for the bundled sets. Raises `ValueError` on
    /// an unreadable or malformed file.
    ///
    /// Use it to pin a basis to another code's exact digits: e.g. PySCF's
    /// `sto-3g` carries fewer digits than ferric's bundled BSE copy, which
    /// is visible at the 1e-8 Ha level.
    #[staticmethod]
    fn from_bse_json(path: &str) -> PyResult<Self> {
        let bs = basis::load_bse_json(path)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))?;
        Ok(PyBasisSet { inner: bs })
    }
    /// Number of basis functions this set places on `mol` (spherical or
    /// Cartesian per shell, as the SCF builds them; ghost atoms included).
    /// For an orbital basis it is the `nbf` of the run; for an auxiliary
    /// (RI/JK-fit) basis it is `naux`, which sizes the density-fitting
    /// tensors: a packed `(P|mn)` tensor holds at most
    /// `naux * nbf * (nbf + 1) / 2` elements (8 bytes each at f64, 4 at f32;
    /// see `run_dft` for which tensors a run holds and when one is smaller or
    /// square). Builds no integrals.
    /// Raises `ValueError` when the basis does not cover an element of `mol`.
    fn n_functions(&self, mol: &PyMolecule) -> PyResult<usize> {
        let prep = PreparedBasis::new(&mol.inner, &self.inner)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))?;
        Ok(prep.nbasis())
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
        if b == 0 {
            None
        } else {
            Some(b)
        }
    })
}

/// The SCF convergence keywords shared by every `run_*` driver that solves a
/// reference SCF internally, with the meaning of `run_rhf`'s `max_iter`,
/// `energy_conv`, `density_conv`. Naming rule: a driver whose method IS the SCF
/// (`run_rhf`/`run_uhf`/`run_rohf`/`run_dft`/`run_qmmm`, ...) spells them bare;
/// every other driver (post-SCF methods, geometry/path drivers) spells them
/// `scf_max_iter`, `scf_energy_conv`, `scf_density_conv`, so `max_iter` never
/// has to guess which stage it caps. `None` leaves the `RhfConfig` default
/// untouched, so omitting all three is byte-identical to a call that never
/// had them.
#[derive(Clone, Copy, Default)]
struct ScfConv {
    max_iter: Option<usize>,
    energy_conv: Option<f64>,
    density_conv: Option<f64>,
}

impl ScfConv {
    fn new(max_iter: Option<usize>, energy_conv: Option<f64>, density_conv: Option<f64>) -> Self {
        Self {
            max_iter,
            energy_conv,
            density_conv,
        }
    }

    /// Overwrite the given fields of `cfg`; unset ones keep `cfg`'s values.
    fn apply(self, cfg: &mut RhfConfig) {
        cfg.max_iter = self.max_iter.unwrap_or(cfg.max_iter);
        cfg.energy_conv = self.energy_conv.unwrap_or(cfg.energy_conv);
        cfg.density_conv = self.density_conv.unwrap_or(cfg.density_conv);
    }
}

fn rhf_config(k_builder: Option<&str>) -> RhfConfig {
    rhf_config_budgeted(k_builder, None)
}

/// [`rhf_config`] plus the caller's resolved memory budget.
///
/// # Why this exists
///
/// Every correlated Python entry point that accepts a `memory_budget_gb` kwarg
/// (`run_rimp2`, `run_ccsd`, `run_pdep_rpa`, …) runs a *reference SCF* first,
/// and every one of them built that SCF's config with a bare
/// `rhf_config(k_builder)` — leaving `three_index_budget_bytes` at its
/// `0`-means-auto default. So `run_ccsd(..., memory_budget_gb=2)` bounded the
/// CC step and let the SCF's DF-JK size itself against 80% of whatever the box
/// happened to have free. The kwarg did not actually bound the process peak
/// for any Python caller, which matters most in Python: an over-allocation
/// there OOM-kills the host interpreter, the risk `check_alloc`'s own docs name
/// as its motivation.
///
/// `RhfConfig::three_index_budget_bytes` is a bare `usize` whose `0` sentinel
/// means "unset → auto" (see `ferric_scf::rhf::resolve_three_index_budget`),
/// so an unset kwarg maps to `0` and preserves the previous behaviour exactly.
fn rhf_config_budgeted(k_builder: Option<&str>, budget_bytes: Option<usize>) -> RhfConfig {
    RhfConfig {
        k_builder: k_builder.map(|s| s.to_string()),
        three_index_budget_bytes: budget_bytes.unwrap_or(0),
        ..Default::default()
    }
}

/// [`rhf_config_budgeted`] with the caller's SCF convergence keywords applied.
fn rhf_config_conv(
    k_builder: Option<&str>,
    budget_bytes: Option<usize>,
    conv: ScfConv,
) -> RhfConfig {
    let mut cfg = rhf_config_budgeted(k_builder, budget_bytes);
    conv.apply(&mut cfg);
    cfg
}

/// Resolve a `df_*_aux` kwarg into an `RhfConfig` auxiliary-basis setting.
///
/// `None` keeps the caller-facing DEFAULT (density fitting on, which is what
/// every ferric DFT entry point has always done). An explicit `""`, `"none"`
/// or `"exact"` selects conventional four-centre Coulomb instead.
///
/// Why this exists: `run_dft` hardcoded RI-J with no override, so a caller
/// comparing ferric against an exact-Coulomb reference measured the RI-J
/// fitting error and read it as a ferric defect. It grows with system size --
/// MEASURED at PBE/STO-3G against conventional J: water 0.28, benzene 1.16,
/// and a 71-atom drug molecule 9.5 kcal/mol.
///
/// An unrecognised basis NAME is passed through unchanged so the SCF layer
/// reports it, rather than being silently replaced by the default (which
/// would turn a typo into a quiet change of method).
fn resolve_df_aux(requested: Option<&str>, default_aux: &str) -> Option<String> {
    match requested {
        None => Some(default_aux.to_string()),
        // Every opt-out spelling becomes Some(""), the SENTINEL the SCF layer
        // reads as "explicitly do not density-fit". Returning None here would
        // mean "unset", which the auto-default then fills back in. The
        // spelling list is shared with run_rhf/run_uhf/run_rohf and the CLI
        // (`ferric_scf::rhf::normalize_df_aux`).
        Some(v) => Some(ferric_scf::rhf::normalize_df_aux(v)),
    }
}

/// One COSX grid kwarg: `(radial, angular)` (flat) or `(radial, angular,
/// prune)` with `prune` in `"none"` / `"sgx"` / `"nwchem"` (the CLI's
/// `cosx_grid = { radial, angular, prune }` table). Strict.
fn parse_cosx_grid(
    name: &str,
    v: &Bound<'_, PyAny>,
) -> PyResult<ferric_dft::grid::AtomicGridConfig> {
    let (r, a, p) = if let Ok((r, a)) = v.extract::<(usize, usize)>() {
        (r, a, None)
    } else if let Ok((r, a, p)) = v.extract::<(usize, usize, String)>() {
        (r, a, Some(p))
    } else {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{name} must be (radial, angular) or (radial, angular, prune)"
        )));
    };
    ferric_scf::cosx_k::grid_from_parts(r, a, p.as_deref())
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{name}: {e}")))
}

/// The `cosx_*` kwargs of `run_rhf` / `run_uhf` / `run_dft` (same names and
/// rules as the CLI `[scf] cosx_*` keys): `cosx_grid`, `cosx_final_pass`,
/// `cosx_final_grid`, `cosx_overlap_fit`. Any of them with `k_builder` other
/// than `"cosx"` is a `ValueError` (a knob that would silently do nothing).
fn resolve_cosx_kwargs(
    k_builder: Option<&str>,
    cosx_grid: Option<&Bound<'_, PyAny>>,
    cosx_final_pass: Option<bool>,
    cosx_final_grid: Option<&Bound<'_, PyAny>>,
    cosx_overlap_fit: Option<bool>,
) -> PyResult<ferric_scf::cosx_k::CosxConfig> {
    let any = cosx_grid.is_some()
        || cosx_final_pass.is_some()
        || cosx_final_grid.is_some()
        || cosx_overlap_fit.is_some();
    if any && k_builder != Some("cosx") {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "cosx_grid / cosx_final_pass / cosx_final_grid / cosx_overlap_fit are only read with \
             k_builder=\"cosx\" (got k_builder={k_builder:?})"
        )));
    }
    let mut cfg = ferric_scf::cosx_k::CosxConfig::default();
    let grid = cosx_grid
        .map(|v| parse_cosx_grid("cosx_grid", v))
        .transpose()?;
    let final_grid = cosx_final_grid
        .map(|v| parse_cosx_grid("cosx_final_grid", v))
        .transpose()?;
    ferric_scf::cosx_k::apply_grid_knobs(&mut cfg, grid, cosx_final_pass, final_grid)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    if let Some(fit) = cosx_overlap_fit {
        cfg.overlap_fit = fit;
    }
    Ok(cfg)
}

/// A precision provenance record (`ferric_core::gpu::provenance`) as a
/// Python dict, through the same JSON the CLI run log writes, so the two
/// schemas cannot drift: `{"precision": "f64" | "mixed", "kernels": [...]}`.
fn precision_dict(
    py: Python<'_>,
    r: &ferric_core::gpu::provenance::PrecisionRecord,
) -> PyResult<PyObject> {
    let json = py.import("json")?;
    Ok(json
        .call_method1("loads", (r.to_json().to_string(),))?
        .unbind())
}

/// The SCF certificate (`ferric_scf::certificate::scf_certificate`) stamped
/// with this build.
fn stamped_scf_certificate(
    mol: &Molecule,
    r: &ScfResult,
    config: &RhfConfig,
) -> ferric_core::certificate::Certificate {
    ferric_scf::certificate::scf_certificate(mol, r, config).with_build(
        ferric_core::certificate::BuildStamp::new(
            ferric_build_info::VERSION,
            ferric_build_info::COMMIT,
            ferric_build_info::DIRTY,
        ),
    )
}

/// The certificate an SCF-kind entry point returns (`run_rhf`, `run_uhf`,
/// `run_rohf`, `run_dft`, `run_ksdft`): [`stamped_scf_certificate`] with a
/// warning for each way the run on `basis` lies outside the solver's energy
/// anchor (`ferric_scf::certificate::with_scf_envelope`, the helper the CLI
/// uses); a post-SCF method's reference carries the same warnings
/// ([`reference_certificate`]). `fock_hooked`: the solve ran with a `fock_hook`
/// (`ferric_scf::certificate::with_fock_hook`).
fn scf_kind_certificate(
    mol: &Molecule,
    r: &ScfResult,
    config: &RhfConfig,
    basis: &ferric_core::basis::BasisSet,
    fock_hooked: bool,
) -> ferric_core::certificate::Certificate {
    let c = ferric_scf::certificate::with_scf_envelope(
        stamped_scf_certificate(mol, r, config),
        mol,
        r,
        config,
        basis,
    );
    if fock_hooked {
        ferric_scf::certificate::with_fock_hook(c)
    } else {
        c
    }
}

/// The reference certificate a post-SCF method folds in
/// (`Certificate::with_reference`): [`stamped_scf_certificate`] with the
/// envelope warnings of the SCF on `basis`
/// (`ferric_scf::certificate::with_scf_envelope`), so a reference outside its
/// solver's anchor shows up on the post-SCF certificate.
fn reference_certificate(
    mol: &Molecule,
    r: &ScfResult,
    config: &RhfConfig,
    basis: &ferric_core::basis::BasisSet,
) -> ferric_core::certificate::Certificate {
    ferric_scf::certificate::with_scf_envelope(
        stamped_scf_certificate(mol, r, config),
        mol,
        r,
        config,
        basis,
    )
}

/// The method.kind a geometry/frequency run on `spin` with `config` is graded
/// under, and the stamped certificate of the SCF at `mol` (one extra SCF: the
/// geometry and frequency drivers do not return theirs). `Err` carries the
/// warning text when that SCF fails; the run itself is not failed by it.
fn task_reference_certificate(
    mol: &Molecule,
    basis_name: &str,
    config: &RhfConfig,
    spin: ferric_scf::Spin,
) -> (
    &'static str,
    Result<ferric_core::certificate::Certificate, String>,
) {
    let kind = match (spin, config.xc.is_some()) {
        (ferric_scf::Spin::Restricted, false) => "rhf",
        (ferric_scf::Spin::Restricted, true) => "ksdft",
        (ferric_scf::Spin::Unrestricted, _) => "uhf",
        (ferric_scf::Spin::RestrictedOpen, _) => "rohf",
    };
    let r = ferric_scf::certificate::final_point_scf(
        &ParallelContext::default(),
        mol,
        basis_name,
        Operator::coulomb(),
        config,
        spin,
    )
    .map(|(scf, used)| stamped_scf_certificate(mol, &scf, &used))
    .map_err(|e| format!("the final-point SCF for this certificate failed: {e}"));
    (kind, r)
}

/// A certificate as a Python dict, through the same JSON the CLI run log
/// writes.
fn certificate_dict(
    py: Python<'_>,
    c: &ferric_core::certificate::Certificate,
) -> PyResult<PyObject> {
    let json = py.import("json")?;
    Ok(json
        .call_method1("loads", (c.to_json().to_string(),))?
        .unbind())
}

/// The COSX final-grid pass record as a dict, or `None` when no pass ran.
fn cosx_final_dict(
    py: Python<'_>,
    f: Option<ferric_scf::cosx_k::CosxFinalPass>,
) -> PyResult<Option<Py<pyo3::types::PyDict>>> {
    let Some(f) = f else {
        return Ok(None);
    };
    let d = pyo3::types::PyDict::new(py);
    d.set_item("e_scf_grid", f.e_scf_grid)?;
    d.set_item("e_final", f.e_final)?;
    d.set_item("npts_scf_grid", f.npts_scf)?;
    d.set_item("npts_final_grid", f.npts_final)?;
    d.set_item("gradient_differentiates", "e_scf_grid")?;
    Ok(Some(d.into()))
}

/// Resolve the `grid_radial` / `grid_angular` / `grid_prune` /
/// `grid_weight_cut` kwargs into the main KS grid config (`RhfConfig::dft_grid`).
///
/// All `None` returns `None`, which every KS path resolves to
/// `AtomicGridConfig::production()` (75x110, nwchem-radial, 1e-14 weight
/// cutoff). Any one set materialises an explicit config: unset sizes from that
/// same grid, and an unset `grid_prune` / `grid_weight_cut` keeps the default
/// scheme / cutoff (`AtomicGridConfig::production_sized`); `"none"` / `0.0`
/// turn them off. It touches the MAIN grid only: the VV10/NLC grid keeps its
/// own flat 50x50 default, where pruning has no valid table (same scoping as
/// the CLI's `[dft] grid_prune`).
///
/// `grid_weight_cut` is ORCA's `WeightCut` (`AtomicGridConfig::weight_cut`):
/// main-grid points whose final weight has `|w| < grid_weight_cut` are dropped.
///
/// Everything is validated HERE, as a `ValueError`, rather than downstream:
/// `lebedev()` PANICS on an unsupported angular order, and a prune scheme with
/// no table at the requested order (`n_angular = 50`) would otherwise surface
/// only as a generic SCF error after the GIL is released. `grid_prune` goes
/// through the same strict parser as the CLI (`PruneScheme::parse_config_str`),
/// so an unknown value is an error, never a silent flat grid; a negative, NaN
/// or infinite `grid_weight_cut` is a `ValueError`, mirroring the CLI's
/// `[dft] grid_weight_cut`.
fn resolve_dft_grid_with_cut(
    grid_radial: Option<usize>,
    grid_angular: Option<usize>,
    grid_prune: Option<&str>,
    grid_weight_cut: Option<f64>,
) -> PyResult<Option<ferric_dft::grid::AtomicGridConfig>> {
    use ferric_dft::prune::{validate_xc_scheme, PruneScheme, SUPPORTED_LEBEDEV_ORDERS};
    let prune = match grid_prune {
        None => None,
        Some(s) => PruneScheme::parse_config_str(s)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("grid_prune: {e}")))?,
    };
    let weight_cut = match grid_weight_cut {
        None => None,
        Some(c) if c.is_finite() && c >= 0.0 => (c > 0.0).then_some(c),
        Some(c) => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "grid_weight_cut = {c} is invalid: it must be a finite number >= 0 \
                 (e.g. 1e-14; 0 or None = no cutoff)"
            )))
        }
    };
    // Absent keywords mean the production grid (`None`, which every KS path
    // resolves to `AtomicGridConfig::production`). An explicit size keeps the
    // default scheme and cutoff unless grid_prune / grid_weight_cut say
    // otherwise ("none", 0), exactly as the CLI's `[dft]` section does.
    if grid_radial.is_none()
        && grid_angular.is_none()
        && grid_prune.is_none()
        && grid_weight_cut.is_none()
    {
        return Ok(None);
    }
    let default = ferric_dft::grid::AtomicGridConfig::production();
    let n_radial = grid_radial.unwrap_or(default.n_radial);
    let n_angular = grid_angular.unwrap_or(default.n_angular);
    let sized = ferric_dft::grid::AtomicGridConfig::production_sized(n_radial, n_angular);
    let prune = if grid_prune.is_some() {
        prune
    } else {
        sized.prune
    };
    let weight_cut = if grid_weight_cut.is_some() {
        weight_cut
    } else {
        sized.weight_cut
    };
    if n_radial == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "grid_radial must be > 0",
        ));
    }
    if !SUPPORTED_LEBEDEV_ORDERS.contains(&n_angular) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "grid_angular = {n_angular} is not a supported Lebedev order \
             (supported: {SUPPORTED_LEBEDEV_ORDERS:?})"
        )));
    }
    if prune == Some(PruneScheme::Sgx) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "grid_prune = \"sgx\" is the COSX exchange grid's scheme (cosx_grid=); the XC \
             grid takes \"none\", \"nwchem\", \"nwchem-radial\" or \"defgrid1/2/3\"",
        ));
    }
    if matches!(prune, Some(PruneScheme::DefGrid(_)))
        && (grid_radial.is_some() || grid_angular.is_some())
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "grid_prune = \"defgrid1/2/3\" fixes the radial and angular grid; \
             remove grid_radial / grid_angular",
        ));
    }
    if let Some(scheme) = prune {
        validate_xc_scheme(scheme, n_angular)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("grid_prune: {e}")))?;
    }
    Ok(Some(ferric_dft::grid::AtomicGridConfig {
        n_radial,
        n_angular,
        prune,
        weight_cut,
    }))
}

/// Convert the Python `solvent` kwarg into a `PcmConfig`.
///
/// Implicit solvation is the difference between an electrostatic interaction
/// term and something that can be compared to an affinity. A vacuum
/// calculation lets both partners leave water for free, so burying an ionized
/// group costs nothing -- MEASURED on danuglipron/GLP-1R, that overshoots the
/// experimental binding free energy by 4.5x.
///
/// Accepts a dielectric constant directly (`solvent=78.4`) or a named solvent.
/// A name that is not recognised ERRORS rather than silently defaulting to
/// vacuum, which would look like a successful run with no solvent.
fn build_pcm_config(
    solvent: Option<&Bound<'_, PyAny>>,
    lebedev_order: Option<usize>,
) -> PyResult<Option<ferric_pcm::PcmConfig>> {
    let Some(obj) = solvent else {
        return Ok(None);
    };
    // The solvent table, the dielectric validation and the Lebedev-order set
    // all live in `ferric_pcm::PcmConfig` (`for_solvent` / `with_epsilon` /
    // `with_lebedev_order`), shared with the CLI's `[pcm]` section so the two
    // surfaces cannot drift apart.
    // The library reports these as `FerricError::General`; drop the variant's
    // "General error: " prefix so the ValueError reads as a kwarg message.
    let msg = |e: ferric_core::FerricError| match e {
        ferric_core::FerricError::General(m) => m,
        other => other.to_string(),
    };
    let cfg = if let Ok(eps) = obj.extract::<f64>() {
        ferric_pcm::PcmConfig::with_epsilon(eps)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("solvent: {}", msg(e))))?
    } else if let Ok(name) = obj.extract::<String>() {
        ferric_pcm::PcmConfig::for_solvent(&name).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "solvent: {} (e.g. solvent=78.4)",
                msg(e)
            ))
        })?
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "solvent must be a dielectric constant (float) or a solvent name (str)",
        ));
    };
    let cfg = match lebedev_order {
        // Checked here so a bad order is a ValueError naming the kwarg, not a
        // RuntimeError from deep inside RHF setup.
        Some(order) => cfg.with_lebedev_order(order).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("pcm_lebedev_order: {}", msg(e)))
        })?,
        None => cfg,
    };
    Ok(Some(cfg))
}

/// Parse the `diis` kwarg into a `DiisFlavor` (strict — unknown values error).
/// None = Pulay (plain DIIS, the default). Delegates to the parser the CLI's
/// `[scf] diis` uses, so the two surfaces accept exactly the same spellings.
fn parse_diis_flavor(diis: Option<&str>) -> PyResult<ferric_scf::diis::DiisFlavor> {
    use ferric_scf::diis::DiisFlavor;
    match diis {
        None => Ok(DiisFlavor::Pulay),
        Some(s) => DiisFlavor::parse_config_str(s)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string())),
    }
}

/// Parse the `ri_j_precision` / `scf_ri_j_precision` kwarg (strict: `"f32"`
/// or `"f64"`, anything else is a `ValueError`; `None` = the default: f32,
/// f64 on the geometry entry points, see [`geometry_task_scf`]).
fn parse_ri_j_precision(s: Option<&str>) -> PyResult<Option<ferric_scf::df_j::RiJPrecision>> {
    s.map(ferric_scf::df_j::RiJPrecision::parse_config_str)
        .transpose()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

/// The geometry/frequency entry points' (`run_optimize*`, `run_frequencies`,
/// `run_saddle`, `run_irc`) one call on their SCF config: mark it a geometry
/// task, so an unset `scf_ri_j_precision` resolves to f64 (the f32 RI-J
/// energy surface is jagged between nearby geometries), and apply an explicit
/// `scf_ri_j_precision`, which always wins. The certificate's `ri-j` entry
/// records the precision and why.
fn geometry_task_scf(cfg: &mut RhfConfig, scf_ri_j_precision: Option<&str>) -> PyResult<()> {
    cfg.ri_j_task = ferric_scf::df_j::RiJTask::Geometry;
    cfg.ri_j_precision = parse_ri_j_precision(scf_ri_j_precision)?;
    Ok(())
}

/// Parse the `guess` kwarg into `RhfConfig::use_sad_guess` (strict — unknown
/// values error; before this any string but "hcore" silently ran MINAO).
/// None = MINAO. Shares `ferric_scf::guess::InitialGuess` with the CLI.
fn parse_guess(guess: Option<&str>) -> PyResult<bool> {
    match guess {
        None => Ok(true),
        Some(s) => ferric_scf::guess::InitialGuess::parse_config_str(s)
            .map(|g| g.use_sad_guess())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string())),
    }
}

/// Parse the `guess_basis` / `guess_projection` kwargs (strict, shared with
/// the CLI's `[scf] guess_basis`): an unknown basis or projection, or a
/// projection without a basis, is a `ValueError`. Both `None` = no
/// secondary-basis guess.
fn parse_guess_basis(
    guess_basis: Option<&str>,
    guess_projection: Option<&str>,
) -> PyResult<Option<ferric_scf::basis2::GuessBasis>> {
    let bad = |e: String| pyo3::exceptions::PyValueError::new_err(e);
    match (guess_basis, guess_projection) {
        (None, None) => Ok(None),
        (None, Some(_)) => Err(bad("guess_projection is set but guess_basis is not".into())),
        (Some(b), p) => ferric_scf::basis2::GuessBasis::bundled(b, p.unwrap_or("fock"))
            .map(Some)
            .map_err(|e| bad(e.to_string())),
    }
}

/// `(n_alpha, n_beta)` from the molecule's electron count (after
/// `apply_ecp`) and multiplicity; an impossible pair is a `ValueError`
/// naming `who`.
fn spin_counts(mol: &Molecule, who: &str) -> PyResult<(usize, usize)> {
    let nelec = mol.nelec() as i64;
    let two_s = mol.multiplicity as i64 - 1;
    if two_s < 0 || nelec < two_s || (nelec - two_s) % 2 != 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{who}: {nelec} electrons cannot have multiplicity {}",
            mol.multiplicity
        )));
    }
    Ok((
        ((nelec + two_s) / 2) as usize,
        ((nelec - two_s) / 2) as usize,
    ))
}

/// Refuse `dm0=` next to `guess=` or `guess_basis=`: each names a starting
/// point, and the SCF would honour only one of them (`dm0` outranks both).
fn refuse_dm0_with_guess(
    who: &str,
    has_dm0: bool,
    guess: Option<&str>,
    guess_basis: Option<&str>,
) -> PyResult<()> {
    let other = match (guess, guess_basis) {
        _ if !has_dm0 => return Ok(()),
        (Some(_), _) => "guess",
        (None, Some(_)) => "guess_basis",
        (None, None) => return Ok(()),
    };
    Err(pyo3::exceptions::PyValueError::new_err(format!(
        "{who}: dm0 and {other} both set the starting point; pass one of them"
    )))
}

/// `run_uhf`'s `mom_reference=`: `None` is `"previous"`. `"initial"` applies
/// MOM from the first iteration, so it is refused next to `mom_after_iter > 0`
/// (a later start) and `stability_descent` (which leaves the start's state).
fn parse_mom_reference(
    value: Option<&str>,
    mom_after_iter: Option<usize>,
    stability_descent: Option<bool>,
) -> PyResult<ferric_scf::mom::MomReference> {
    let reference = value
        .map(ferric_scf::mom::MomReference::parse_config_str)
        .transpose()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
        .unwrap_or_default();
    let conflicts = mom_after_iter.unwrap_or(0) > 0 || stability_descent.unwrap_or(false);
    if reference == ferric_scf::mom::MomReference::Initial && conflicts {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "run_uhf: mom_reference=\"initial\" applies MOM from the first iteration and \
             holds the starting occupation; pass neither mom_after_iter nor stability_descent",
        ));
    }
    Ok(reference)
}

/// The secondary-basis guess report as a dict (`basis`, `projection`,
/// `iterations`, `energy`, `nbasis_guess`, `nbasis_target`, `converged`,
/// `unconverged_reason`, `used_as_guess`), or `None` when no
/// secondary-basis guess ran.
fn guess_basis_dict(
    py: Python<'_>,
    r: Option<&ferric_scf::basis2::GuessBasisReport>,
) -> PyResult<Option<Py<pyo3::types::PyDict>>> {
    let Some(r) = r else {
        return Ok(None);
    };
    let d = pyo3::types::PyDict::new(py);
    d.set_item("basis", &r.basis)?;
    d.set_item("projection", r.projection.as_str())?;
    d.set_item("iterations", r.iterations)?;
    d.set_item("energy", r.energy)?;
    d.set_item("nbasis_guess", r.nbasis_guess)?;
    d.set_item("nbasis_target", r.nbasis_target)?;
    d.set_item("converged", r.converged)?;
    d.set_item("unconverged_reason", r.unconverged_reason.as_deref())?;
    d.set_item("used_as_guess", r.used_as_guess)?;
    Ok(Some(d.into()))
}

/// Convert the Python `point_charges` / `external_field` kwargs into an
/// `ExternalPotential`. Returns `None` when both are unset (no perturbation),
/// matching `RhfConfig`'s "None = no external potential" convention.
fn build_external_potential(
    point_charges: Option<Vec<(f64, f64, f64, f64)>>,
    external_field: Option<(f64, f64, f64)>,
) -> Option<ferric_core::external_potential::ExternalPotential> {
    build_external_potential_with_smeared(point_charges, None, external_field)
}

/// Like [`build_external_potential`], plus an optional `smeared_charges`
/// list of `(q, x, y, z, width)` tuples (Bohr / Bohr). Returns `None` only
/// when point charges, smeared charges, AND field are all unset.
fn build_external_potential_with_smeared(
    point_charges: Option<Vec<(f64, f64, f64, f64)>>,
    smeared_charges: Option<Vec<(f64, f64, f64, f64, f64)>>,
    external_field: Option<(f64, f64, f64)>,
) -> Option<ferric_core::external_potential::ExternalPotential> {
    let pcs: Vec<ferric_core::external_potential::PointCharge> = point_charges
        .unwrap_or_default()
        .into_iter()
        .map(|(q, x, y, z)| ferric_core::external_potential::PointCharge { q, x, y, z })
        .collect();
    let scs: Vec<ferric_core::external_potential::SmearedCharge> = smeared_charges
        .unwrap_or_default()
        .into_iter()
        .map(
            |(q, x, y, z, width)| ferric_core::external_potential::SmearedCharge {
                q,
                x,
                y,
                z,
                width,
            },
        )
        .collect();
    let field = external_field.map(|(ex, ey, ez)| [ex, ey, ez]);
    if pcs.is_empty() && scs.is_empty() && field.is_none() {
        None
    } else {
        Some(ferric_core::external_potential::ExternalPotential {
            point_charges: pcs,
            smeared_charges: scs,
            field,
        })
    }
}

// ── Module ──

/// One direction of an IRC walk.
#[pyclass(name = "IrcBranch")]
#[derive(Clone)]
struct PyIrcBranch {
    /// Symbols at the endpoint.
    #[pyo3(get)]
    symbols: Vec<String>,
    /// Endpoint coordinates in ANGSTROM.
    #[pyo3(get)]
    coords: Vec<(f64, f64, f64)>,
    #[pyo3(get)]
    energy: f64,
    #[pyo3(get)]
    steps: usize,
    /// `True` if the gradient threshold was met; `False` if the step budget
    /// ran out first.
    ///
    /// A `False` here means the endpoint is NOT a basin and this branch
    /// identifies no minimum -- it is a partial path, and the pair of
    /// endpoints must not be read as "the reaction".
    #[pyo3(get)]
    converged: bool,
}

/// Both directions of an IRC, plus the saddle they came from.
#[pyclass(name = "IrcResult")]
#[derive(Clone)]
struct PyIrcResult {
    /// The `+mode` direction.
    #[pyo3(get)]
    forward: PyIrcBranch,
    /// The `-mode` direction.
    #[pyo3(get)]
    reverse: PyIrcBranch,
    /// Energy at the saddle, for a barrier against each endpoint.
    #[pyo3(get)]
    saddle_energy: f64,
}

#[pymethods]
impl PyIrcResult {
    /// Barrier from the forward endpoint up to the saddle, in Hartree.
    fn forward_barrier(&self) -> f64 {
        self.saddle_energy - self.forward.energy
    }

    /// Barrier from the reverse endpoint up to the saddle, in Hartree.
    fn reverse_barrier(&self) -> f64 {
        self.saddle_energy - self.reverse.energy
    }

    /// Did BOTH directions reach a basin?
    ///
    /// Only then do the two endpoints answer "which minima does this saddle
    /// connect?". One unconverged branch leaves one side unidentified, and the
    /// result must not be read as a reaction.
    fn both_converged(&self) -> bool {
        self.forward.converged && self.reverse.converged
    }
}

/// Outcome of a P-RFO transition-state search.
#[pyclass(name = "SaddleResult")]
#[derive(Clone)]
struct PySaddleResult {
    /// Symbols of the geometry the search arrived at.
    #[pyo3(get)]
    symbols: Vec<String>,
    /// Coordinates in ANGSTROM, matching `symbols`. Valid to inspect even when
    /// `converged` is false -- it is where the search stopped.
    #[pyo3(get)]
    coords: Vec<(f64, f64, f64)>,
    #[pyo3(get)]
    energy: f64,
    #[pyo3(get)]
    steps: usize,
    /// Whether the GRADIENT converged. This alone does NOT mean a transition
    /// state was found -- every stationary point satisfies it. See
    /// `n_imaginary` and `is_transition_state`.
    #[pyo3(get)]
    converged: bool,
    /// Negative eigenvalues of the projected Hessian at the final geometry.
    /// **1** is a first-order transition state; 0 is a minimum; >1 is a
    /// higher-order saddle.
    #[pyo3(get)]
    n_imaginary: usize,
    /// The single negative mode's eigenvector in Cartesian coordinates (3N),
    /// when there is exactly one; `None` otherwise. A caller MUST check this
    /// points along the intended reaction coordinate -- one imaginary
    /// frequency means first-order saddle, not "the saddle you meant".
    #[pyo3(get)]
    imaginary_mode: Option<Vec<f64>>,
    #[pyo3(get)]
    lowest_eigenvalue: f64,
    /// Which construction produced the Hessian of the FINAL geometry (the one
    /// behind `n_imaginary`): "finite-difference", "analytic" or (under
    /// `hessian="auto"` with `xc=`) "analytic-moving-grid".
    #[pyo3(get)]
    hessian_source: String,
}

#[pymethods]
impl PySaddleResult {
    /// A first-order transition state: gradient converged AND exactly one
    /// imaginary frequency. Both halves are required.
    fn is_transition_state(&self) -> bool {
        self.converged && self.n_imaginary == 1
    }

    fn __repr__(&self) -> String {
        format!(
            "SaddleResult(converged={}, steps={}, n_imaginary={}, E={:.8}, TS={})",
            self.converged,
            self.steps,
            self.n_imaginary,
            self.energy,
            self.is_transition_state()
        )
    }
}

/// Parse `run_saddle`'s `hessian=` and refuse `delta=` with a non-FD method.
///
/// `delta` is the central-difference displacement; an analytic Hessian has
/// none. Refusing the combination keeps a silently ignored `delta=` from
/// returning (#214 review: that is what pinning to FD was meant to prevent).
fn parse_saddle_hessian(
    hessian: &str,
    delta: Option<f64>,
) -> PyResult<ferric_scf::frequencies::HessianMethod> {
    use ferric_scf::frequencies::HessianMethod;
    let method = HessianMethod::parse_config_str(hessian)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    if method == HessianMethod::AnalyticMovingGrid {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "hessian=\"analytic-moving-grid\" is not available in run_saddle as an explicit \
             value; \"auto\" selects the moving-grid Kohn-Sham Hessian wherever it applies \
             (use \"fd\", \"auto\" or \"analytic\")",
        ));
    }
    if delta.is_some() && method != HessianMethod::FiniteDifference {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "delta= is the finite-difference displacement and is ignored by an \
             analytic Hessian; it is only valid with hessian=\"fd\" (got \
             hessian={hessian:?})"
        )));
    }
    Ok(method)
}

/// Search for a first-order saddle point by P-RFO.
///
/// Costs `2 * (6N + 1) + (n_steps + 1)` gradient evaluations (MEASURED): two
/// Hessians -- each 6N DISPLACED central-difference evaluations plus one at the
/// undisplaced geometry -- and one gradient per step plus one before the first.
/// At N = 20 the Hessians dominate until n_steps exceeds ~240, which is why
/// they are built twice and carried by a Bofill update in between.
///
/// Raises if the starting geometry has NO negative projected mode -- P-RFO from
/// a minimum's basin has nothing to climb, and returning a result from there
/// would be a minimum labelled as a transition state.
///
/// `hessian`: "fd" (default, unchanged behaviour) builds every Hessian by
/// central differences of the analytic gradient, honouring `delta`. "analytic"
/// uses the analytic closed-shell RHF Hessian (one SCF plus CPHF per Hessian,
/// no displaced gradients) and raises, naming the reason, when it does not
/// apply (KS `xc`, point charges / external field, ECP, a basis above the
/// libint second-derivative limit). "auto" uses the analytic Hessian where it
/// applies (with `xc=`, the analytic moving-grid Kohn-Sham Hessian of
/// `run_frequencies`) and otherwise falls back to "fd"; `.hessian_source` says
/// which ran.
/// `delta` is only meaningful with "fd": combining it with "auto" or
/// "analytic" raises rather than being silently ignored.
///
/// `scf_warm_start` (default True): each step's SCF starts from the previous
/// step's converged orbitals (`ferric_scf::warm_start`); False re-guesses
/// every step's SCF. The finite-difference Hessians stay cold (as in
/// `run_frequencies`, warm FD Hessians are noisier at equal convergence).
///
/// `scf_ri_j_precision` (`"f32"` or `"f64"`; RI-J runs with `xc`): unset is
/// `"f64"`, as for every geometry task (the f32 RI-J surface jitters between
/// nearby geometries); an explicit `"f32"` is honoured.
#[pyfunction]
#[pyo3(signature = (
    mol, basis_name, xc=None, multiplicity=None,
    max_steps=None, trust_radius=None, follow_mode=None, delta=None,
    point_charges=None, external_field=None, hessian="fd",
    scf_max_iter=None, scf_energy_conv=None, scf_density_conv=None,
    scf_warm_start=None, scf_ri_j_precision=None,
))]
#[allow(clippy::too_many_arguments)]
fn run_saddle(
    mol: &PyMolecule,
    basis_name: &str,
    xc: Option<&str>,
    multiplicity: Option<u32>,
    max_steps: Option<usize>,
    trust_radius: Option<f64>,
    follow_mode: Option<usize>,
    delta: Option<f64>,
    point_charges: Option<Vec<(f64, f64, f64, f64)>>,
    external_field: Option<(f64, f64, f64)>,
    hessian: &str,
    scf_max_iter: Option<usize>,
    scf_energy_conv: Option<f64>,
    scf_density_conv: Option<f64>,
    scf_warm_start: Option<bool>,
    scf_ri_j_precision: Option<&str>,
) -> PyResult<PySaddleResult> {
    use ferric_scf::frequencies::{harmonic_frequencies, FrequencyConfig, FrequencyReference};
    use ferric_scf::gradient::rhf_gradient;
    use ferric_scf::saddle::{find_saddle, SaddleConfig};
    use ferric_scf::screening::SchwarzBounds;
    use ferric_scf::warm_start::solve_rhf_warm;

    if let Some(t) = trust_radius {
        if !(t.is_finite() && t > 0.0) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "trust_radius must be finite and > 0 (got {t})"
            )));
        }
    }

    let hessian_method = parse_saddle_hessian(hessian, delta)?;

    let mut m = mol.inner.clone();
    if let Some(mult) = multiplicity {
        m.multiplicity = mult as usize;
    }
    // BOTH callbacks below are RESTRICTED -- `solve_rhf` plus either
    // `rhf_gradient` or `ks_gradient_closed`, and `FrequencyReference::Rhf`.
    // Accepting a multiplicity > 1 would run a doublet through a closed-shell
    // reference and return a confident answer for a state that does not exist.
    // Refuse rather than silently choosing the wrong physics.
    if m.multiplicity != 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "run_saddle currently supports closed-shell (multiplicity = 1) \
             references only; got multiplicity = {}. Both the gradient and the \
             Hessian callbacks are restricted, so an open-shell request would \
             be answered with a closed-shell reference rather than refused.",
            m.multiplicity
        )));
    }
    // ECP must be applied BEFORE the first SCF: it changes the electron count,
    // so a search started without it optimizes a different molecule than the
    // one the caller asked about.
    {
        let bs = ferric_core::basis::bundled(basis_name).map_err(make_err)?;
        m.apply_ecp(&bs);
    }
    // The MM field. Both closures below ALREADY read
    // `scf_g.external_potential` and hand it to `rhf_gradient` /
    // `ks_gradient_closed`, and `harmonic_frequencies` threads it too -- the
    // only thing that was missing was a way to SET it from Python, so the
    // catalyst branch of the QM/MM workflow was gas-phase only while
    // `run_optimize` next door already took these kwargs.
    //
    // A saddle is the case where this matters most: an MM field can REMOVE the
    // stationary point a gas-phase search would find (a symmetry-breaking
    // charge pair makes planar NH3 non-stationary), so searching in vacuum and
    // hoping the barrier transfers is not an approximation, it is a different
    // question.
    let mut scf_cfg = RhfConfig {
        xc: xc.map(|s| s.to_string()),
        external_potential: build_external_potential(point_charges, external_field),
        ..Default::default()
    };
    ScfConv::new(scf_max_iter, scf_energy_conv, scf_density_conv).apply(&mut scf_cfg);
    geometry_task_scf(&mut scf_cfg, scf_ri_j_precision)?;
    let mut cfg = SaddleConfig::default();
    if let Some(v) = max_steps {
        cfg.max_steps = v;
    }
    if let Some(v) = trust_radius {
        cfg.trust_radius = v;
    }
    if let Some(v) = follow_mode {
        cfg.follow_mode = v;
    }

    let op = Operator::coulomb();
    let basis = basis_name.to_string();
    let ctx = ParallelContext::default();

    let scf_g = scf_cfg.clone();
    let basis_g = basis.clone();
    // Each step's SCF starts from the previous step's converged orbitals
    // (`ferric_scf::warm_start`). The Hessians below stay cold
    // (`FrequencyConfig::scf_warm_start` default): a warm FD Hessian is
    // noisier at equal convergence, and the final one decides n_imaginary.
    let mut warm = ferric_scf::warm_start::ScfWarmStart::new(scf_warm_start.unwrap_or(true));
    let energy_gradient = move |mm: &ferric_core::mol::Molecule| {
        let bs = ferric_core::basis::bundled(&basis_g)?;
        let prep = PreparedBasis::new(mm, &bs)?;
        let bounds = SchwarzBounds::compute(op, &prep)?;
        let res = solve_rhf_warm(&ctx, mm, &prep, op, &bounds, &scf_g, &mut warm)?;
        // An unconverged SCF is not a point on any surface. Letting one
        // through hands the optimizer an energy and a gradient from a density
        // that never settled, and the search walks on noise while reporting
        // ordinary-looking numbers.
        if !res.converged {
            return Err(ferric_core::FerricError::General(
                "saddle search: the SCF did not converge at this geometry, so \
                 its energy and gradient are not points on the potential \
                 surface. Loosen the geometry, raise scf_max_iter, or start closer \
                 to the barrier."
                    .to_string(),
            ));
        }
        // `solve_rhf` returns a KS energy when `xc` is set, so the gradient
        // must be the KS one too. Calling `rhf_gradient` here would pair a
        // KS-DFT energy with a Hartree-Fock gradient -- two different surfaces,
        // and the search would converge to a stationary point of neither.
        let g = if let Some(xc_name) = scf_g.xc.as_deref() {
            ks_gradient_closed_with_exchange_on_grids(
                mm,
                &prep,
                &bs,
                op,
                &bounds,
                xc_name,
                &res,
                scf_g.external_potential.as_ref(),
                None,
                &KsGradGrids::from_config(&scf_g),
            )?
        } else {
            rhf_gradient(
                mm,
                &prep,
                op,
                &bounds,
                &res,
                scf_g.external_potential.as_ref(),
            )?
        };
        Ok((res.energy, ndarray::Array1::from_iter(g.iter().copied())))
    };

    let scf_h = scf_cfg.clone();
    let basis_h = basis.clone();
    // Records which construction produced the most recent Hessian, so the
    // result can report it (the last call is the final-geometry Hessian).
    let source_cell: std::rc::Rc<std::cell::RefCell<String>> =
        std::rc::Rc::new(std::cell::RefCell::new(String::new()));
    let source_w = source_cell.clone();
    let hessian_cb = move |mm: &ferric_core::mol::Molecule| {
        // `FrequencyReference::Rhf` already means "closed-shell RHF, or
        // closed-shell KS-DFT when RhfConfig::xc is set" -- see its docstring.
        // So this one variant covers both, and the Hessian follows `xc`
        // automatically. (The GRADIENT callback above is not so lucky: it has
        // to pick between rhf_gradient and ks_gradient_closed by hand.)
        // Default is finite differences (the #214 decision: `delta=` is
        // honoured and the documented cost model holds). The analytic Hessian
        // is OPT-IN via `hessian=`; "analytic" errors when unsupported and
        // "auto" falls back to FD, reported through `hessian_source` ("auto"
        // with `xc` is the moving-grid KS Hessian where it applies).
        let mut fc = FrequencyConfig {
            reference: FrequencyReference::Rhf,
            hessian: hessian_method,
            ..Default::default()
        };
        if let Some(d) = delta {
            fc.delta = d;
        }
        let fr = harmonic_frequencies(&ParallelContext::default(), mm, &basis_h, op, &scf_h, &fc)?;
        *source_w.borrow_mut() = fr.hessian_source.label().to_string();
        Ok(fr.cartesian_hessian)
    };

    let r = find_saddle(&m, &cfg, energy_gradient, hessian_cb).map_err(make_err)?;
    let hessian_source = source_cell.borrow().clone();

    const BOHR_TO_ANGSTROM: f64 = ferric_core::units::BOHR_TO_ANGSTROM;
    Ok(PySaddleResult {
        symbols: r.mol.atoms.iter().map(|a| a.symbol.clone()).collect(),
        coords: r
            .mol
            .atoms
            .iter()
            .map(|a| {
                (
                    a.x * BOHR_TO_ANGSTROM,
                    a.y * BOHR_TO_ANGSTROM,
                    a.zpos * BOHR_TO_ANGSTROM,
                )
            })
            .collect(),
        energy: r.energy,
        steps: r.steps,
        converged: r.converged,
        n_imaginary: r.n_imaginary,
        imaginary_mode: r.imaginary_mode.map(|v| v.to_vec()),
        lowest_eigenvalue: r.lowest_eigenvalue,
        hessian_source,
    })
}

/// Follow the intrinsic reaction coordinate off a saddle, both directions.
///
/// Answers the question `run_saddle` cannot: **which two minima does this
/// saddle connect?** One imaginary mode proves a geometry is a first-order
/// saddle; it does not prove it is the saddle you meant. A methyl rotor gives
/// one imaginary mode exactly as a bond-breaking coordinate does.
///
/// `mode` is the 3N Cartesian eigenvector -- pass `SaddleResult.imaginary_mode`
/// straight through. It is mass-weighted and normalised internally, so its
/// input scale does not matter; its DIRECTION does, and `forward` is `+mode`.
///
/// Cost: MEASURED ~71 gradients per branch on NH3 inversion, so ~142 for the
/// pair. That is over half a TS search again -- budget the two together.
///
/// Closed-shell only, matching `run_saddle`: the callbacks here are the same
/// restricted ones, so an open-shell request is refused rather than answered
/// with the wrong reference.
///
/// `scf_warm_start` (default True): each point's SCF starts from the previous
/// point's converged orbitals (`ferric_scf::warm_start`); False re-guesses
/// every SCF.
///
/// `scf_ri_j_precision` (`"f32"` or `"f64"`; RI-J runs with `xc`): unset is
/// `"f64"`, as for every geometry task (the f32 RI-J surface jitters between
/// nearby geometries); an explicit `"f32"` is honoured.
#[pyfunction]
#[pyo3(signature = (
    mol, basis_name, mode, xc=None, multiplicity=None,
    step=None, max_steps=None, g_max_thresh=None, initial_displacement=None,
    point_charges=None, external_field=None,
    scf_max_iter=None, scf_energy_conv=None, scf_density_conv=None,
    scf_warm_start=None, scf_ri_j_precision=None,
))]
#[allow(clippy::too_many_arguments)]
fn run_irc(
    mol: &PyMolecule,
    basis_name: &str,
    mode: Vec<f64>,
    xc: Option<&str>,
    multiplicity: Option<u32>,
    step: Option<f64>,
    max_steps: Option<usize>,
    g_max_thresh: Option<f64>,
    initial_displacement: Option<f64>,
    point_charges: Option<Vec<(f64, f64, f64, f64)>>,
    external_field: Option<(f64, f64, f64)>,
    scf_max_iter: Option<usize>,
    scf_energy_conv: Option<f64>,
    scf_density_conv: Option<f64>,
    scf_warm_start: Option<bool>,
    scf_ri_j_precision: Option<&str>,
) -> PyResult<PyIrcResult> {
    use ferric_scf::gradient::rhf_gradient;
    use ferric_scf::irc::{follow_irc, IrcConfig};
    use ferric_scf::screening::SchwarzBounds;
    use ferric_scf::warm_start::{solve_rhf_warm, ScfWarmStart};

    let mut m = mol.inner.clone();
    if let Some(mult) = multiplicity {
        m.multiplicity = mult as usize;
    }
    // Same restriction, and the same reason, as `run_saddle`: the callback
    // below is `solve_rhf` plus a closed-shell gradient, so an open-shell
    // request would be answered with a closed-shell reference.
    if m.multiplicity != 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "run_irc currently supports closed-shell (multiplicity = 1) \
             references only; got multiplicity = {}.",
            m.multiplicity
        )));
    }
    // ECP before the first SCF -- it changes the electron count, so a path
    // walked without it is a path on a different molecule.
    {
        let bs = ferric_core::basis::bundled(basis_name).map_err(make_err)?;
        m.apply_ecp(&bs);
    }

    // The MM field, same as `run_saddle`. An IRC started from an embedded
    // saddle MUST be walked in the same field: the path is a property of the
    // surface, and a gas-phase walk from an embedded saddle descends a
    // different surface than the one the saddle sits on.
    let mut scf_cfg = RhfConfig {
        xc: xc.map(|s| s.to_string()),
        external_potential: build_external_potential(point_charges, external_field),
        ..Default::default()
    };
    ScfConv::new(scf_max_iter, scf_energy_conv, scf_density_conv).apply(&mut scf_cfg);
    geometry_task_scf(&mut scf_cfg, scf_ri_j_precision)?;
    let mut cfg = IrcConfig::default();
    if let Some(v) = step {
        cfg.step = v;
    }
    if let Some(v) = max_steps {
        cfg.max_steps = v;
    }
    if let Some(v) = g_max_thresh {
        cfg.g_max_thresh = v;
    }
    if let Some(v) = initial_displacement {
        cfg.initial_displacement = v;
    }

    let op = Operator::coulomb();
    let basis = basis_name.to_string();
    // Each point's SCF starts from the previous point's converged orbitals
    // (`ferric_scf::warm_start`). The jump from the end of the forward branch
    // back to the saddle region is caught by its overlap test (cold guess).
    let mut warm = ScfWarmStart::new(scf_warm_start.unwrap_or(true));
    let eg = move |mm: &ferric_core::mol::Molecule| {
        let bs = ferric_core::basis::bundled(&basis)?;
        let prep = PreparedBasis::new(mm, &bs)?;
        let bounds = SchwarzBounds::compute(op, &prep)?;
        let res = solve_rhf_warm(
            &ParallelContext::default(),
            mm,
            &prep,
            op,
            &bounds,
            &scf_cfg,
            &mut warm,
        )?;
        if !res.converged {
            return Err(ferric_core::FerricError::General(
                "IRC: the SCF did not converge at this point on the path, so its \
                 energy and gradient are not on the potential surface."
                    .to_string(),
            ));
        }
        // KS energy needs the KS gradient. Pairing `solve_rhf`'s XC energy
        // with `rhf_gradient` would walk one surface while reporting another.
        let g = if let Some(xc_name) = scf_cfg.xc.as_deref() {
            ks_gradient_closed_with_exchange_on_grids(
                mm,
                &prep,
                &bs,
                op,
                &bounds,
                xc_name,
                &res,
                scf_cfg.external_potential.as_ref(),
                None,
                &KsGradGrids::from_config(&scf_cfg),
            )?
        } else {
            rhf_gradient(
                mm,
                &prep,
                op,
                &bounds,
                &res,
                scf_cfg.external_potential.as_ref(),
            )?
        };
        Ok((res.energy, ndarray::Array1::from_iter(g.iter().copied())))
    };

    let r = follow_irc(&m, &ndarray::Array1::from_vec(mode), &cfg, eg).map_err(make_err)?;

    // Same as run_saddle: Bohr (ferric's internal unit) to the Angstrom the
    // Python surface uses.
    const BOHR_TO_ANGSTROM: f64 = ferric_core::units::BOHR_TO_ANGSTROM;
    let to_branch = |b: ferric_scf::irc::IrcBranch| PyIrcBranch {
        symbols: b.mol.atoms.iter().map(|a| a.symbol.clone()).collect(),
        coords: b
            .mol
            .atoms
            .iter()
            .map(|a| {
                (
                    a.x * BOHR_TO_ANGSTROM,
                    a.y * BOHR_TO_ANGSTROM,
                    a.zpos * BOHR_TO_ANGSTROM,
                )
            })
            .collect(),
        energy: b.energy,
        steps: b.steps,
        converged: b.converged,
    };
    Ok(PyIrcResult {
        forward: to_branch(r.forward),
        reverse: to_branch(r.reverse),
        saddle_energy: r.saddle_energy,
    })
}

/// Python bindings for ferric (pyo3).
///
/// Exposes the engine to Python: `Molecule` / `BasisSet` constructors plus
/// `run_rhf`, `run_uhf`, `run_rohf`, `run_rimp2`, `run_oo_rimp2`,
/// `run_attenuated_rimp2`, `run_scs_mp2`, the Laplace and coupled-cluster
/// drivers, and geometry optimization. Each binding wraps the corresponding
/// Rust driver and returns a result object with energies/components.
///
/// Also exposes the conformer-ensemble machinery (`ConformerEnsemble`,
/// `boltzmann_weights`, `weighted_stats*`, `EnsembleDiagnostics`) so an RDKit
/// `EmbedMultipleConfs` conformer set can be turned into a Boltzmann-weighted
/// ferric property with a spread. Geometry input there is **Ångström**,
/// matching RDKit's `GetPositions()` and ferric's XYZ readers.
///
/// Build with `uv run maturin develop --release` from the repository root; the
/// venv caveat is under "Python bindings from source" in
/// `site/src/using/installation.md`.
///
/// NOTE: this docstring is duplicated (not just referenced) from the file-level
/// `//!` comment at the top of this file. pyo3 only picks up a `///` doc
/// comment placed directly above `#[pymodule] fn ferric`, not a file-level
/// `//!` — without this, `help(ferric)` / `ferric.__doc__` return empty even
/// though the `//!` content renders fine in `cargo doc`. Keep both in sync.
/// Registers the MP2-V research bindings (the one-SCF multi-arm scan and the
/// VV10 weight variants), kept out of `ferric()` so the module initialiser
/// stays under its complexity baseline.
fn register_mp2v_research(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_mp2_v_scan, m)?)?;
    m.add_function(wrap_pyfunction!(run_vv10_variants, m)?)?;
    Ok(())
}

#[pymodule]
fn ferric(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Threading model (see ferric_core::blas_threads): thread stacks large
    // enough for threaded LAPACK on any thread, and OpenBLAS pinned to 1
    // thread in parallel regions unless the host set OPENBLAS_NUM_THREADS.
    // Serial sections raise it per `ferric.set_blas_threads`.
    ferric_integrals::blas_threads::init_threading();
    // Register global signal handler for Ctrl+C
    let _ = ctrlc::set_handler(move || {
        eprintln!("\n[Ferric] Interrupt signal caught! Bailing out of kernels...");
        ferric_core::INTERRUPT.store(true, std::sync::atomic::Ordering::SeqCst);
    });

    m.add_class::<PyMolecule>()?;
    m.add_class::<PyBasisSet>()?;
    add_rhf_result_classes(m)?;
    m.add_class::<PyUhfResult>()?;
    m.add_class::<PyCdftConstraint>()?;
    m.add_class::<PyCdftResult>()?;
    m.add_class::<PyCdftCouplingResult>()?;
    m.add_class::<PyOptimizeResult>()?;
    m.add_class::<PyQmmmSystem>()?;
    m.add_class::<PyQmmmResult>()?;
    m.add_class::<PyQmmmOptimizeResult>()?;
    m.add_class::<PyMmTopology>()?;
    m.add_class::<PyFrequencyResult>()?;
    m.add_class::<PyRiMp2Result>()?;
    m.add_class::<PyOoRiMp2Result>()?;
    m.add_class::<PyMp3Result>()?;
    m.add_class::<PyAttenuatedMp2Result>()?;
    m.add_class::<PyScsMp2Result>()?;
    m.add_class::<PyMp2VResult>()?;
    m.add_class::<PyLaplaceMp2Result>()?;
    m.add_class::<PySosMp2Result>()?;
    m.add_class::<PyDftResult>()?;
    m.add_class::<PyCcResult>()?;
    m.add_class::<PyPdepRpaResult>()?;
    m.add_class::<PyRsMp2RpaResult>()?;
    m.add_class::<PyGwResult>()?;
    m.add_class::<PyUGwResult>()?;
    m.add_class::<PyBseResult>()?;
    m.add_class::<PyTdhfStaticPolarizabilityResult>()?;
    m.add_class::<PyTddftResult>()?;
    m.add_class::<PyDoubleHybridResult>()?;
    m.add_class::<PyConformerEnsemble>()?;
    m.add_class::<PyBoltzmannWeights>()?;
    m.add_class::<PyWeightedStats>()?;
    m.add_class::<PyEnsembleDiagnostics>()?;
    // Returned by `boys_localize`; unregistered, `ferric.BoysResult` did not exist.
    m.add_class::<PyBoysResult>()?;
    // Conformer-ensemble constants, so callers need not hardcode them.
    m.add(
        "DEFAULT_TEMPERATURE_K",
        ferric_core::conformers::DEFAULT_TEMPERATURE_K,
    )?;
    m.add(
        "BOLTZMANN_HARTREE_PER_K",
        ferric_core::conformers::BOLTZMANN_HARTREE_PER_K,
    )?;
    m.add_function(wrap_pyfunction!(boltzmann_weights, m)?)?;
    m.add_function(wrap_pyfunction!(weighted_stats, m)?)?;
    m.add_function(wrap_pyfunction!(weighted_stats_vector, m)?)?;
    m.add_function(wrap_pyfunction!(weighted_stats_tensor, m)?)?;
    // Underscore-prefixed: not part of the documented Python API, only the
    // `ferric` console script's entry point wires to this by name -- see
    // pyproject.toml [project.scripts].
    m.add_function(wrap_pyfunction!(_cli_main, m)?)?;
    add_build_identity(m)?;
    m.add_function(wrap_pyfunction!(run_rhf, m)?)?;
    m.add_function(wrap_pyfunction!(run_rhf_gamma, m)?)?;
    m.add_class::<PyGammaRhfResult>()?;
    pbc::register(m)?;
    ci::register(m)?;
    response_modes::register(m)?;
    pbc::register_omega_chooser(m)?;
    m.add_function(wrap_pyfunction!(run_uhf, m)?)?;
    m.add_function(wrap_pyfunction!(run_rohf, m)?)?;
    m.add_function(wrap_pyfunction!(run_cdft, m)?)?;
    m.add_function(wrap_pyfunction!(cdft_coupling, m)?)?;
    m.add_function(wrap_pyfunction!(run_optimize, m)?)?;
    m.add_function(wrap_pyfunction!(run_optimize_uhf, m)?)?;
    m.add_function(wrap_pyfunction!(run_qmmm, m)?)?;
    m.add_function(wrap_pyfunction!(run_optimize_qmmm, m)?)?;
    m.add_function(wrap_pyfunction!(run_frequencies, m)?)?;
    m.add_function(wrap_pyfunction!(run_xtb_energy_gradient, m)?)?;
    m.add_function(wrap_pyfunction!(run_xtb_optimize, m)?)?;
    m.add_function(wrap_pyfunction!(run_xtb_frequencies, m)?)?;
    m.add_function(wrap_pyfunction!(run_saddle, m)?)?;
    m.add_function(wrap_pyfunction!(run_irc, m)?)?;
    m.add_class::<PySaddleResult>()?;
    m.add_class::<PyIrcBranch>()?;
    m.add_class::<PyIrcResult>()?;
    m.add_function(wrap_pyfunction!(esp_at_atoms, m)?)?;
    m.add_function(wrap_pyfunction!(esp_at_points, m)?)?;
    m.add_function(wrap_pyfunction!(esp_on_surface, m)?)?;
    m.add_function(wrap_pyfunction!(hirshfeld_charges, m)?)?;
    m.add_function(wrap_pyfunction!(lowdin_charges, m)?)?;
    m.add_function(wrap_pyfunction!(run_drpa, m)?)?;
    m.add_function(wrap_pyfunction!(run_drpa_scan, m)?)?;
    m.add_function(wrap_pyfunction!(run_linlccd, m)?)?;
    m.add_function(wrap_pyfunction!(tune_omega, m)?)?;
    m.add_function(wrap_pyfunction!(orbital_moments, m)?)?;
    m.add_function(wrap_pyfunction!(density_second_moment, m)?)?;
    m.add_function(wrap_pyfunction!(eval_ao_on_points, m)?)?;
    m.add_function(wrap_pyfunction!(eval_density_on_points, m)?)?;
    m.add_function(wrap_pyfunction!(eval_mo_on_points, m)?)?;
    m.add_function(wrap_pyfunction!(write_cube, m)?)?;
    m.add_function(wrap_pyfunction!(mulliken_charges, m)?)?;
    m.add_function(wrap_pyfunction!(chelpg_charges, m)?)?;
    m.add_function(wrap_pyfunction!(resp_charges, m)?)?;
    m.add_function(wrap_pyfunction!(hirshfeld_polarizability, m)?)?;
    m.add_function(wrap_pyfunction!(run_rimp2, m)?)?;
    m.add_function(wrap_pyfunction!(run_oo_rimp2, m)?)?;
    m.add_function(wrap_pyfunction!(run_mp3, m)?)?;
    m.add_function(wrap_pyfunction!(run_attenuated_rimp2, m)?)?;
    m.add_function(wrap_pyfunction!(run_terfc_rimp2, m)?)?;
    m.add_function(wrap_pyfunction!(run_scs_mp2, m)?)?;
    m.add_function(wrap_pyfunction!(run_scs_mp2_2terfc, m)?)?;
    m.add_function(wrap_pyfunction!(run_mp2_v, m)?)?;
    m.add_function(wrap_pyfunction!(run_double_hybrid, m)?)?;

    m.add_function(wrap_pyfunction!(run_laplace_mp2, m)?)?;
    m.add_function(wrap_pyfunction!(run_laplace_sos_mp2, m)?)?;
    m.add_function(wrap_pyfunction!(run_dft, m)?)?;
    m.add_function(wrap_pyfunction!(cosx_grid_point_count, m)?)?;
    m.add_function(wrap_pyfunction!(run_ksdft, m)?)?;
    m.add_function(wrap_pyfunction!(dft_grid_point_count, m)?)?;
    m.add_function(wrap_pyfunction!(d3bj_energy, m)?)?;
    m.add_function(wrap_pyfunction!(mbd_rsscs_energy, m)?)?;
    m.add_class::<PyMbdRsscsResult>()?;
    m.add_function(wrap_pyfunction!(run_ccd, m)?)?;
    m.add_function(wrap_pyfunction!(run_ccsd, m)?)?;
    m.add_function(wrap_pyfunction!(run_ccsd_t, m)?)?;
    m.add_function(wrap_pyfunction!(run_pdep_rpa, m)?)?;
    m.add_function(wrap_pyfunction!(run_rs_mp2_rpa, m)?)?;
    m.add_function(wrap_pyfunction!(run_gw, m)?)?;
    m.add_function(wrap_pyfunction!(run_u_gw, m)?)?;
    m.add_function(wrap_pyfunction!(run_bse_tda, m)?)?;
    m.add_function(wrap_pyfunction!(run_tdhf_static_polarizability, m)?)?;
    m.add_function(wrap_pyfunction!(run_tddft, m)?)?;
    m.add_function(wrap_pyfunction!(compute_eri3, m)?)?;
    m.add_function(wrap_pyfunction!(compute_eri3_mo, m)?)?;
    m.add_function(wrap_pyfunction!(compute_metric_2c, m)?)?;
    m.add_function(wrap_pyfunction!(boys_localize, m)?)?;
    register_device_and_geometry(m)?;
    // Tail call (no `?`): the module initialiser sits at its complexity baseline.
    register_mp2v_research(m)
}

/// Registration tail of the module: shell geometry and the CUDA backend
/// (kept out of `ferric` so the module function's complexity stays bounded).
/// `configure_gpu` (device placement) and `configure_precision` (kernels).
fn register_configure(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(configure_gpu, m)?)?;
    m.add_function(wrap_pyfunction!(configure_precision, m)?)?;
    Ok(())
}

fn register_device_and_geometry(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(shell_info, m)?)?;
    m.add_function(wrap_pyfunction!(gpu_status, m)?)?;
    register_configure(m)?;
    // `AoIntegrals` + `einsum`: the building blocks for writing a method in
    // Python (see blocks.rs).
    blocks::register(m)
}

/// `eigensolver_conv_thresh`, or `default` when unset, warning if it was
/// passed: the bindings always run `Eigensolver::Lanczos`, whose full-rank PDEP
/// eigensolve is one dense eigh that reads no convergence threshold (see
/// `Eigensolver::honours_iteration_knobs`).
fn lanczos_conv_thresh_or(set: Option<f64>, default: f64) -> f64 {
    if set.is_some() && !ferric_rpa::Eigensolver::default().honours_iteration_knobs() {
        eprintln!(
            "warning: eigensolver_conv_thresh is ignored: the Lanczos eigensolver solves \
             the full-rank PDEP problem with one dense eigh, which has no convergence \
             threshold; drop the argument"
        );
    }
    set.unwrap_or(default)
}
