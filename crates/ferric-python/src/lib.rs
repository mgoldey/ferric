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
//! Build with `uv run maturin develop --release` (see the README for the venv
//! caveat).

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::attenuated::{attenuated_ri_mp2, AttenuatedMp2Config};
use ferric_mp2::laplace::{laplace_ri_mp2, laplace_sos_mp2, SosFormulation, SosMp2Config};
use ferric_mp2::mp3::mp3_energy;
use ferric_mp2::oo_rimp2::{oo_ri_mp2, OoRiMp2Config};
use ferric_mp2::rimp2::{ri_mp2, RiMp2Config};
use ferric_mp2::att_vv10::{att_mp2_vv10, AttVv10Attenuator, AttVv10Config, AttVv10SpinComponents};
use ferric_mp2::scs::{scs_mp2, scs_mp2_2terfc, ScsMp2Config, ScsMp2TerfcConfig};
use ferric_mp2::double_hybrid::{mp2_double_hybrid, DoubleHybridKind};
use ferric_scf::ks_gradient::ks_gradient_closed;
use ferric_scf::optimize::{optimize_geometry, OptimizeConfig};
use ferric_scf::result::ScfResult;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::rohf::{solve_rohf, RohfConfig};
use ferric_scf::uhf::{solve_uhf, UhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_cc::ccd::ccd as run_ccd_inner;
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

// ── Molecule ──

/// A molecular geometry (atoms, charge, multiplicity). Coordinates are stored
/// internally in Bohr; XYZ input is in Ångström.
#[pyclass]
#[pyo3(name = "Molecule")]
struct PyMolecule { inner: Molecule }

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
    /// Classical nuclear repulsion energy in Hartree.
    fn nuclear_repulsion(&self) -> f64 { self.inner.nuclear_repulsion() }
    /// Number of atoms.
    fn natoms(&self) -> usize { self.inner.atoms.len() }
    /// Total electron count (accounts for `charge` and any ECP core electrons).
    fn nelec(&self) -> i32 { self.inner.nelec() }
    /// Cartesian coordinates of every atom in **Ångström**, one `(x, y, z)`
    /// tuple per atom in `symbols()` order. Internally ferric stores Bohr;
    /// this divides by the same Å→Bohr constant the XYZ parser multiplied
    /// by (see `ConformerEnsemble.coordinates()` for why divide, not
    /// multiply by the reciprocal), so a round trip is accurate to ~1 ulp.
    fn coords(&self) -> Vec<(f64, f64, f64)> {
        self.inner
            .atoms
            .iter()
            .map(|a| (a.x / ANGSTROM_TO_BOHR, a.y / ANGSTROM_TO_BOHR, a.zpos / ANGSTROM_TO_BOHR))
            .collect()
    }
    /// Cartesian coordinates in **Bohr** — the internal values, untouched.
    /// These are the units `run_rhf(..., point_charges=...)` and
    /// `QmmmSystem.point_charges()` use.
    fn coords_bohr(&self) -> Vec<(f64, f64, f64)> {
        self.inner.atoms.iter().map(|a| (a.x, a.y, a.zpos)).collect()
    }
    /// Element symbols in atom order (ghost atoms keep their plain symbol;
    /// the `@` marker is not round-tripped).
    fn symbols(&self) -> Vec<String> {
        self.inner.atoms.iter().map(|a| a.symbol.clone()).collect()
    }
}

// ── BasisSet ──

/// A Gaussian basis set (orbital or auxiliary/RI-fitting).
#[pyclass]
#[pyo3(name = "BasisSet")]
struct PyBasisSet { inner: ferric_core::basis::BasisSet }

#[pymethods]
impl PyBasisSet {
    /// Load one of ferric's bundled basis sets by name (e.g. `"sto-3g"`,
    /// `"cc-pvdz"`, `"def2-svp"`, or an RI/JK auxiliary set like
    /// `"cc-pvdz-ri"`/`"def2-universal-jkfit"`). Raises `ValueError` if `name`
    /// is not a bundled set.
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

/// Parse the `diis` kwarg into a `DiisFlavor` (strict — unknown values error).
/// None = Pulay (plain DIIS, the default).
fn parse_diis_flavor(diis: Option<&str>) -> PyResult<ferric_scf::diis::DiisFlavor> {
    use ferric_scf::diis::DiisFlavor;
    match diis {
        None | Some("pulay") => Ok(DiisFlavor::Pulay),
        Some("adiis") => Ok(DiisFlavor::Adiis),
        Some("ediis") => Ok(DiisFlavor::Ediis),
        Some(other) => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "diis = '{other}' not recognized (use 'pulay', 'adiis', or 'ediis')"
        ))),
    }
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

/// Result of a closed-shell RHF (or `run_ksdft` KS-DFT) calculation.
#[pyclass]
#[pyo3(name = "RhfResult")]
struct PyRhfResult {
    /// Total SCF energy in Hartree (electronic + nuclear repulsion).
    #[pyo3(get)] energy: f64,
    /// Whether the SCF met both the energy and density convergence thresholds.
    #[pyo3(get)] converged: bool,
    /// Number of SCF iterations run.
    #[pyo3(get)] iterations: usize,
    /// Number of unique two-electron integral quartets actually computed
    /// (reflects Schwarz/QQR screening; lower than the naive N^4 count).
    #[pyo3(get)] computed_quartets: usize,
    density_data: Array2<f64>,
    orbital_energies_data: Vec<f64>,
    scf_data: ScfResult,
}

#[pymethods]
impl PyRhfResult {
    /// The converged AO-basis density matrix as a 2D numpy array (n_bf × n_bf).
    fn density<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        PyArray2::from_array(py, &self.density_data)
    }
    /// Molecular orbital energies (Hartree) as a 1D numpy array, ascending order.
    fn orbital_energies<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.orbital_energies_data.clone())
    }
    /// MO coefficient matrix C (n_bf × n_mo), column k = MO k in ascending
    /// energy order — the same matrix the Rust post-HF drivers consume.
    fn mo_coefficients<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        PyArray2::from_array(py, &self.scf_data.mos_alpha)
    }
    fn __repr__(&self) -> String {
        format!("RhfResult(energy={:.10}, converged={})", self.energy, self.converged)
    }
    fn __str__(&self) -> String {
        format!("RHF Energy: {:.10} Ha (converged: {}, {} iterations)", self.energy, self.converged, self.iterations)
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
    guess=None, diis=None, smearing_sigma=None, soscf=None,
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
    guess: Option<&str>,
    diis: Option<&str>,
    smearing_sigma: Option<f64>,
    soscf: Option<bool>,
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
        diis_flavor: parse_diis_flavor(diis)?,
        smearing_sigma,
        newton_trigger: if soscf.unwrap_or(false) { 1e-3 } else { 0.0 },
        use_sad_guess: !matches!(guess, Some("hcore")),
        external_potential: build_external_potential(point_charges, external_field),
        ..Default::default()
    };
    let ctx = ParallelContext::default();
    let r = solve_rhf(&ctx, &emol, &prep, op, &bounds, &config).map_err(make_err)?;
    Ok(PyRhfResult {
        energy: r.energy, converged: r.converged, iterations: r.iterations,
        computed_quartets: r.computed_quartets,
        density_data: r.density_total.clone(), orbital_energies_data: r.eps_alpha.clone(),
        scf_data: r,
    })
}

// ── QM/MM ──

/// A QM/MM partition: a full structure (ligand + pocket, say) split into a
/// quantum region and fixed MM point charges, with optional link-atom capping
/// of cut bonds and a boundary-charge scheme. Thin wrapper over
/// `ferric_scf::qmmm::QmmmSystem`; no new physics.
///
/// Units: this constructor takes **Ångström** (like every other geometry
/// entry point on the Python side); `point_charges()` and `run_rhf`'s
/// `point_charges=` kwarg are in **Bohr**. `qm_molecule()` /
/// `point_charges()` feed the existing `run_rhf` / `run_uhf` / `run_optimize`
/// unchanged, or use `run_qmmm` for energy + gradients in one call.
#[pyclass]
#[pyo3(name = "QmmmSystem")]
#[derive(Clone)]
struct PyQmmmSystem { inner: ferric_scf::qmmm::QmmmSystem }

#[pymethods]
impl PyQmmmSystem {
    /// `symbols`/`coords_angstrom`/`charges` describe the FULL structure, one
    /// entry per atom. `charges` are the MM partial charges (e); an atom's
    /// charge is ignored if it lands in the QM region. Symbols that are not
    /// elements (e.g. `"X"` for a bare charge site) are allowed in the MM
    /// region only.
    ///
    /// Select the QM region with EITHER `qm_indices` OR `qm_seeds` +
    /// `qm_radius_angstrom` (every atom within that distance of any seed,
    /// plus the seeds — the "ligand plus everything within R" pocket case;
    /// selection is by atom, with no residue completion).
    #[new]
    #[pyo3(signature = (symbols, coords_angstrom, charges, qm_indices=None, qm_seeds=None, qm_radius_angstrom=None, charge=0, multiplicity=1))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        symbols: Vec<String>,
        coords_angstrom: Vec<Vec<f64>>,
        charges: Vec<f64>,
        qm_indices: Option<Vec<usize>>,
        qm_seeds: Option<Vec<usize>>,
        qm_radius_angstrom: Option<f64>,
        charge: i32,
        multiplicity: usize,
    ) -> PyResult<Self> {
        use ferric_scf::qmmm::{QmSelection, QmmmAtom, QmmmSystem};
        if symbols.len() != coords_angstrom.len() || symbols.len() != charges.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "symbols ({}), coords_angstrom ({}) and charges ({}) must have the same length",
                symbols.len(), coords_angstrom.len(), charges.len()
            )));
        }
        if let Some((i, bad)) = coords_angstrom.iter().enumerate().find(|(_, c)| c.len() != 3) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "coords_angstrom[{i}] has {} components, expected 3", bad.len()
            )));
        }
        let atoms: Vec<QmmmAtom> = symbols
            .iter()
            .zip(coords_angstrom.iter())
            .zip(charges.iter())
            .map(|((sym, c), &q)| {
                let zn = ferric_core::elements::symbol_to_z(sym).unwrap_or(0);
                QmmmAtom::new(sym.clone(), zn, c[0] * ANGSTROM_TO_BOHR, c[1] * ANGSTROM_TO_BOHR, c[2] * ANGSTROM_TO_BOHR, q)
            })
            .collect();
        let selection = match (qm_indices, qm_seeds, qm_radius_angstrom) {
            (Some(idx), None, None) => QmSelection::Indices(idx),
            (None, Some(seeds), Some(r)) => QmSelection::WithinRadius { seeds, radius: r * ANGSTROM_TO_BOHR },
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "select the QM region with EITHER qm_indices OR qm_seeds + qm_radius_angstrom",
                ))
            }
        };
        let inner = QmmmSystem::new(&atoms, selection, charge, multiplicity).map_err(make_err)?;
        if let Some(&bad) = inner.qm_indices.iter().find(|&&i| atoms[i].z == 0) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "QM atom {bad} has symbol {:?}, which is not an element", atoms[bad].symbol
            )));
        }
        Ok(Self { inner })
    }

    /// Cap every bond in `bonds` (index pairs into the full structure) that
    /// the partition cuts with a scaled-position link hydrogen at `scale` of
    /// the bond length from the QM atom (default 1.09/1.53, the C–C case).
    /// Returns a new system; `self` is unchanged.
    #[pyo3(signature = (bonds, scale=None))]
    fn with_link_atoms(&self, bonds: Vec<(usize, usize)>, scale: Option<f64>) -> PyResult<Self> {
        let scale = scale.unwrap_or(ferric_scf::qmmm::DEFAULT_LINK_SCALE);
        Ok(Self { inner: self.inner.clone().with_link_atoms(&bonds, scale).map_err(make_err)? })
    }

    /// Apply a boundary-charge scheme to the MM host atom of each cut bond:
    /// `"keep"` (untreated, default), `"delete-host"` (Z1), `"rc"`
    /// (redistributed charge onto M1–M2 bond midpoints; conserves charge) or
    /// `"rcd"` (also conserves the M1–M2 dipole). Needs the full bond list to
    /// find the M2 shell. Returns a new system.
    fn with_boundary_charges(&self, bonds: Vec<(usize, usize)>, scheme: &str) -> PyResult<Self> {
        let scheme = ferric_scf::qmmm::BoundaryChargeScheme::parse_config_str(scheme)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner: self.inner.clone().with_boundary_charges(&bonds, scheme).map_err(make_err)? })
    }

    /// The QM region (plus link hydrogens, appended last) as a `Molecule`.
    fn qm_molecule(&self) -> PyMolecule { PyMolecule { inner: self.inner.to_qm_molecule() } }
    /// Every embedding charge as `(q, x, y, z)` in **Bohr** — directly
    /// usable as `point_charges=` for `run_rhf`/`run_uhf`/`run_optimize`.
    /// Atom-centred charges first, then any RC/RCD midpoint charges.
    fn point_charges(&self) -> Vec<(f64, f64, f64, f64)> {
        self.inner
            .to_external_potential()
            .map(|ep| ep.point_charges.iter().map(|pc| (pc.q, pc.x, pc.y, pc.z)).collect())
            .unwrap_or_default()
    }
    /// Full-structure indices of the QM atoms (ascending; link atoms excluded).
    fn qm_indices(&self) -> Vec<usize> { self.inner.qm_indices.clone() }
    /// Full-structure indices of the MM atoms (ascending).
    fn mm_indices(&self) -> Vec<usize> { self.inner.mm_indices.clone() }
    /// Number of real QM atoms (link hydrogens occupy `qm_molecule()` indices
    /// `qm_atom_count()..`).
    fn qm_atom_count(&self) -> usize { self.inner.qm_atom_count() }
    /// Number of atoms in the full structure.
    fn natoms(&self) -> usize { self.inner.atoms.len() }
    /// Link hydrogen positions in **Ångström**, in `qm_molecule()` order.
    fn link_atom_positions(&self) -> Vec<(f64, f64, f64)> {
        self.inner
            .link_atoms
            .iter()
            .map(|l| (l.position[0] / ANGSTROM_TO_BOHR, l.position[1] / ANGSTROM_TO_BOHR, l.position[2] / ANGSTROM_TO_BOHR))
            .collect()
    }
    /// Shortest link-hydrogen-to-MM-charge distance in **Ångström**, or
    /// `None` without link atoms / charges. A diagnostic: under ~0.5–1 Å you
    /// want `with_boundary_charges("rcd")`.
    fn min_link_to_charge_distance(&self) -> Option<f64> {
        self.inner.min_link_to_charge_distance().map(|d| d / ANGSTROM_TO_BOHR)
    }
    /// The boundary scheme currently applied, as its config string.
    fn boundary_scheme(&self) -> &'static str { self.inner.boundary_scheme.config_str() }
    fn __repr__(&self) -> String {
        format!(
            "QmmmSystem(n_qm={}, n_mm={}, n_link={}, scheme={:?})",
            self.inner.qm_atom_count(), self.inner.mm_indices.len(), self.inner.link_atoms.len(),
            self.inner.boundary_scheme.config_str()
        )
    }
}

/// Result of `run_qmmm`: the embedded SCF energy and the three gradient
/// views. All gradients are `dE/dR` in Hartree/Bohr except `mm_forces()`,
/// which is the FORCE (`−dE/dR`) on each embedding charge, because that is
/// what an MM integrator consumes.
#[pyclass]
#[pyo3(name = "QmmmResult")]
struct PyQmmmResult {
    #[pyo3(get)] energy: f64,
    #[pyo3(get)] converged: bool,
    #[pyo3(get)] iterations: usize,
    qm_gradient_data: Array2<f64>,
    mm_forces_data: Vec<[f64; 3]>,
    full_gradient_data: Array2<f64>,
}

#[pymethods]
impl PyQmmmResult {
    /// `dE/dR` on the QM molecule as solved: real QM atoms then link
    /// hydrogens, `(n_qm + n_link, 3)`.
    fn qm_gradient<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        PyArray2::from_array(py, &self.qm_gradient_data)
    }
    /// Force on each embedding charge, `(n_charges, 3)`, in
    /// `QmmmSystem.point_charges()` order.
    fn mm_forces<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let n = self.mm_forces_data.len();
        let mut a = Array2::<f64>::zeros((n, 3));
        for (i, f) in self.mm_forces_data.iter().enumerate() {
            for k in 0..3 { a[(i, k)] = f[k]; }
        }
        PyArray2::from_array(py, &a)
    }
    /// `dE/dR` over every REAL atom of the full structure, `(natoms, 3)`:
    /// link-atom rows chain-ruled onto their hosts, MM forces mapped onto the
    /// atoms carrying the charges (midpoint charges split half/half).
    fn full_gradient<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        PyArray2::from_array(py, &self.full_gradient_data)
    }
    fn __repr__(&self) -> String {
        format!("QmmmResult(energy={:.10}, converged={}, n_full={})", self.energy, self.converged, self.full_gradient_data.nrows())
    }
}

/// Embedded SCF energy plus gradients for a `QmmmSystem` in one call.
///
/// `method` is `"rhf"` (default) or `"uhf"`. The SCF knobs mirror
/// `run_rhf`. No MM force field is involved: the energy is `E_QM` in the
/// field of the fixed charges (plus the classical charge–nuclear term), and
/// the gradients are its exact derivatives.
#[pyfunction]
#[pyo3(signature = (system, basis_name, method=None, max_iter=None, energy_conv=None, density_conv=None, level_shift=None, mom_after_iter=None, guess=None))]
#[allow(clippy::too_many_arguments)]
fn run_qmmm(
    system: &PyQmmmSystem,
    basis_name: &str,
    method: Option<&str>,
    max_iter: Option<usize>,
    energy_conv: Option<f64>,
    density_conv: Option<f64>,
    level_shift: Option<f64>,
    mom_after_iter: Option<usize>,
    guess: Option<&str>,
) -> PyResult<PyQmmmResult> {
    use ferric_scf::gradient::{rhf_gradient, uhf_gradient};
    use ferric_scf::qmmm::{full_gradient, mm_forces};

    let sys = &system.inner;
    let bs = basis::bundled(basis_name).map_err(make_err)?;
    let mut mol = sys.to_qm_molecule();
    mol.apply_ecp(&bs);
    let prep = PreparedBasis::new(&mol, &bs).map_err(make_err)?;
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).map_err(make_err)?;
    let config = RhfConfig {
        max_iter: max_iter.unwrap_or(100),
        energy_conv: energy_conv.unwrap_or(1e-3),
        density_conv: density_conv.unwrap_or(1e-6),
        level_shift: level_shift.unwrap_or(0.0),
        mom_after_iter: mom_after_iter.unwrap_or(0),
        use_sad_guess: !matches!(guess, Some("hcore")),
        external_potential: sys.to_external_potential(),
        ..Default::default()
    };
    let ctx = ParallelContext::default();
    let ext = config.external_potential.as_ref();
    let (r, qm_grad) = match method.unwrap_or("rhf") {
        "rhf" => {
            let r = solve_rhf(&ctx, &mol, &prep, op, &bounds, &config).map_err(make_err)?;
            let g = rhf_gradient(&mol, &prep, op, &bounds, &r, ext).map_err(make_err)?;
            (r, g)
        }
        "uhf" => {
            let r = solve_uhf(&ctx, &mol, &prep, &bounds, &config).map_err(make_err)?;
            let g = uhf_gradient(&mol, &prep, op, &bounds, &r, ext).map_err(make_err)?;
            (r, g)
        }
        m => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "run_qmmm: method must be \"rhf\" or \"uhf\", got {m:?}"
            )))
        }
    };
    let forces = mm_forces(sys, &mol, &prep, r.density_total()).map_err(make_err)?;
    let full = full_gradient(sys, &qm_grad, &forces).map_err(make_err)?;
    Ok(PyQmmmResult {
        energy: r.energy,
        converged: r.converged,
        iterations: r.iterations,
        qm_gradient_data: qm_grad,
        mm_forces_data: forces,
        full_gradient_data: full,
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
    fn __repr__(&self) -> String {
        format!("UhfResult(energy={:.10}, converged={})", self.energy, self.converged)
    }
    fn __str__(&self) -> String {
        format!("UHF Energy: {:.10} Ha (converged: {}, {} iterations)", self.energy, self.converged, self.iterations)
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

// ── ROHF (open-shell, spin-pure) ──

/// Restricted Open-Shell Hartree-Fock. α/β occupations come from the
/// molecule's `charge` and `multiplicity` (set via `Molecule.from_xyz_string`):
/// `nocc_open = multiplicity - 1` singly-occupied (α) orbitals, the remainder
/// doubly occupied. Coupling is Guest-Saunders (PySCF default; not a knob).
/// Returns the same `UhfResult` shape as `run_uhf` — ROHF's `alpha`/`beta`
/// density and orbital-energy accessors carry the doubly + singly occupied
/// α set and the doubly-occupied-only β set respectively (single set of
/// spin-pure MOs, so `mos_alpha`/`mos_beta` coincide internally but the
/// occupied-orbital semantics differ from UHF). Exposes the same SCF knob
/// set as `run_rhf`/`run_uhf`.
#[pyfunction]
#[pyo3(signature = (
    mol, basis_set,
    max_iter=None, energy_conv=None, density_conv=None, diis_size=None,
    integral_thresh=None, k_builder=None, df_j_aux=None, df_k_aux=None,
    level_shift=None, mom_after_iter=None,
    point_charges=None, external_field=None,
))]
#[allow(clippy::too_many_arguments)]
fn run_rohf(
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
    // Same defaults as run_rhf / run_uhf / the CLI [scf] section.
    let config: RohfConfig = RhfConfig {
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
    let r = solve_rohf(&ctx, &emol, &prep, op, &bounds, &config).map_err(make_err)?;
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
    fn __repr__(&self) -> String {
        format!("OptimizeResult(energy={:.10}, converged={}, steps={})", self.energy, self.converged, self.steps)
    }
    fn __str__(&self) -> String {
        format!("Optimized Energy: {:.10} Ha (converged: {}, {} steps)", self.energy, self.converged, self.steps)
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

#[pyclass]
#[pyo3(name = "FrequencyResult")]
struct PyFrequencyResult {
    /// Vibrational wavenumbers in cm^-1, ascending. `3N-6` entries (`3N-5` if
    /// linear). A NEGATIVE entry is an imaginary frequency, by the usual
    /// convention -- it means a mode with a negative force constant, i.e. the
    /// geometry is not a minimum.
    #[pyo3(get)] frequencies: Vec<f64>,
    /// The projected-out translation/rotation modes, cm^-1. Should be ~0;
    /// retained as a diagnostic.
    #[pyo3(get)] trans_rot_frequencies: Vec<f64>,
    #[pyo3(get)] is_linear: bool,
    /// Largest |H_ij - H_ji| in the raw Cartesian Hessian, Hartree/Bohr^2.
    /// Zero in exact arithmetic, so this is a direct read on whether `delta`
    /// and the SCF thresholds suit the system. A large value invalidates the
    /// frequencies.
    #[pyo3(get)] asymmetry: f64,
    #[pyo3(get)] n_gradient_evaluations: usize,
    /// Electronic energy at the undisplaced geometry.
    #[pyo3(get)] energy: f64,
}

#[pymethods]
impl PyFrequencyResult {
    fn __repr__(&self) -> String {
        format!(
            "FrequencyResult(energy={:.10}, n_modes={}, is_linear={})",
            self.energy, self.frequencies.len(), self.is_linear,
        )
    }
    fn __str__(&self) -> String {
        let n_imag = self.frequencies.iter().filter(|&&f| f < 0.0).count();
        format!(
            "Vibrational Frequencies: {} modes, {} imaginary, energy={:.10} Ha, asymmetry={:.2e}",
            self.frequencies.len(), n_imag, self.energy, self.asymmetry,
        )
    }
}

/// Harmonic vibrational frequencies, by finite difference of the ANALYTIC
/// gradient (6N gradient evaluations), mass-weighted with translations and
/// rotations projected out.
///
/// `reference` selects the SCF: "rhf" (default), "uhf", or "rohf". Setting
/// `xc` promotes it to the matching KS variant (RKS/UKS/ROKS).
///
/// `delta` is the central-difference displacement in Bohr (default 5e-4). It is
/// a real accuracy knob that degrades silently -- too large adds truncation
/// error, too small amplifies SCF noise. Check `.asymmetry` rather than
/// assuming.
#[pyfunction]
#[pyo3(signature = (mol, basis_name, reference=None, xc=None, delta=None, multiplicity=None))]
fn run_frequencies(
    mol: &PyMolecule,
    basis_name: &str,
    reference: Option<&str>,
    xc: Option<&str>,
    delta: Option<f64>,
    multiplicity: Option<u32>,
) -> PyResult<PyFrequencyResult> {
    use ferric_scf::frequencies::{harmonic_frequencies, FrequencyConfig, FrequencyReference};

    // Strict, like the CLI: an unrecognized reference must ERROR rather than
    // silently running RHF and handing back frequencies for the wrong system.
    let refr = match reference {
        None | Some("rhf") => FrequencyReference::Rhf,
        Some("uhf") => FrequencyReference::Uhf,
        Some("rohf") => FrequencyReference::Rohf,
        Some(other) => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown reference \"{other}\"; expected \"rhf\", \"uhf\" or \"rohf\""
            )))
        }
    };
    if let Some(d) = delta {
        if !(d.is_finite() && d > 0.0) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "delta must be finite and > 0 (got {d})"
            )));
        }
    }

    let mut m = mol.inner.clone();
    if let Some(mult) = multiplicity {
        m.multiplicity = mult as usize;
    }
    let scf_cfg = RhfConfig {
        xc: xc.map(|s| s.to_string()),
        ..Default::default()
    };
    let mut fcfg = FrequencyConfig { reference: refr, ..Default::default() };
    if let Some(d) = delta {
        fcfg.delta = d;
    }

    let ctx = ParallelContext::default();
    let r = harmonic_frequencies(&ctx, &m, basis_name, Operator::coulomb(), &scf_cfg, &fcfg)
        .map_err(make_err)?;
    Ok(PyFrequencyResult {
        frequencies: r.frequencies,
        trans_rot_frequencies: r.trans_rot_frequencies,
        is_linear: r.is_linear,
        asymmetry: r.asymmetry,
        n_gradient_evaluations: r.n_gradient_evaluations,
        energy: r.energy,
    })
}

// ── Conformer ensembles (Boltzmann-weighted property averaging) ──
//
// Thin wrapper over `ferric_core::conformers`. Nothing is reimplemented here:
// the invariant checks, the E_min-shifted weights, and the shifted-form
// weighted variance all live in core. This layer only converts Python shapes
// (lists of coordinate arrays from RDKit, lists of energies) into core's
// types, and converts core's typed errors into Python exceptions rather than
// letting a Rust panic cross the pyo3 boundary (which would abort the host
// interpreter).
//
// UNITS. RDKit's `Chem.Conformer.GetPositions()` returns **Ångström**.
// ferric's `Molecule` stores **Bohr**. `Molecule::parse_xyz` (and therefore
// `Molecule.from_xyz` / `from_xyz_string`, and `conformers::load_multi_xyz`)
// take Ångström input and convert. To stay consistent with every other
// geometry entry point in this module, `ConformerEnsemble.from_coordinates`
// also takes **Ångström** and converts internally — so an RDKit
// `GetPositions()` array can be passed straight through with no scaling by
// the caller. `ConformerEnsemble.coordinates()` returns Ångström again (a
// ~1-ulp round-trip, not bit-exact: the Å->Bohr multiply has no exact
// inverse); `coordinates_bohr()` returns the stored Bohr values verbatim for
// callers who want the raw internal representation.

/// Map a `ConformerError` to a Python exception.
///
/// Every variant is a caller/input error, not an internal failure, so
/// `ValueError` is the right class throughout — the caller can catch it and
/// fix the input (re-order the conformers, drop the unconverged one). The
/// message is core's, which already names the offending conformer/atom index.
fn conformer_err(e: ferric_core::conformers::ConformerError) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(format!("{e}"))
}

/// Build a `Molecule` from a coordinate array plus a shared element list.
///
/// `coords` is `natoms x 3` in **Ångström**. Rather than constructing `Atom`s
/// by hand (which would silently bypass core's element lookup, ghost-prefix
/// handling, and the electron-count/multiplicity parity check), this formats a
/// standard XYZ frame and hands it to `Molecule::parse_xyz` — the exact same
/// path `Molecule.from_xyz_string` takes, so the unit convention and all
/// validation are shared by construction rather than by convention.
///
/// The `{:?}` float formatting is Rust's shortest round-trip representation:
/// `parse::<f64>()` recovers the identical bit pattern, so this is exact, not
/// an approximation. (Verified over 2e6 random f64 bit patterns.)
fn molecule_from_coords(
    coords: &[Vec<f64>],
    symbols: &[String],
    charge: i32,
    multiplicity: usize,
    conformer_index: usize,
) -> PyResult<Molecule> {
    if coords.len() != symbols.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "conformer {conformer_index}: got {} coordinate rows but {} element symbols; \
             the element list is shared by all conformers and must match each geometry's \
             atom count",
            coords.len(),
            symbols.len()
        )));
    }
    let mut xyz = format!("{}\nconformer {conformer_index}\n", coords.len());
    for (atom_index, (row, sym)) in coords.iter().zip(symbols).enumerate() {
        if row.len() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "conformer {conformer_index}, atom {atom_index}: expected 3 Cartesian \
                 coordinates, got {}",
                row.len()
            )));
        }
        if !row.iter().all(|v| v.is_finite()) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "conformer {conformer_index}, atom {atom_index}: non-finite coordinate \
                 {row:?} (NaN/inf geometries produce a plausible-looking but meaningless \
                 energy, so they are rejected here)"
            )));
        }
        xyz.push_str(&format!("{} {:?} {:?} {:?}\n", sym, row[0], row[1], row[2]));
    }
    Molecule::parse_xyz(&xyz, charge, multiplicity).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("conformer {conformer_index}: {e}"))
    })
}

/// A weighted mean with its spread. Returned by every ensemble-averaging call.
///
/// The standard deviation is not optional: an ensemble average without a
/// spread cannot distinguish "one dominant conformer, the ensemble was
/// unnecessary" from "fifty comparable conformers, any single-conformer number
/// is simply wrong". `min`/`max` are the unweighted range, for context.
#[pyclass]
#[pyo3(name = "WeightedStats")]
#[derive(Clone, Copy)]
struct PyWeightedStats {
    /// Weighted mean `Σ w_i x_i`.
    #[pyo3(get)] mean: f64,
    /// Weighted population standard deviation `sqrt(Σ w_i (x_i - mean)²)`.
    /// Exactly 0.0 for a one-conformer ensemble.
    #[pyo3(get)] std_dev: f64,
    /// Smallest value across conformers (unweighted).
    #[pyo3(get)] min: f64,
    /// Largest value across conformers (unweighted).
    #[pyo3(get)] max: f64,
}

impl From<ferric_core::conformers::WeightedStats> for PyWeightedStats {
    fn from(s: ferric_core::conformers::WeightedStats) -> Self {
        PyWeightedStats { mean: s.mean, std_dev: s.std_dev, min: s.min, max: s.max }
    }
}

#[pymethods]
impl PyWeightedStats {
    fn __repr__(&self) -> String {
        format!(
            "WeightedStats(mean={:.10e}, std_dev={:.10e}, min={:.10e}, max={:.10e})",
            self.mean, self.std_dev, self.min, self.max
        )
    }
}

/// Population-structure readout for a Boltzmann-weighted ensemble.
///
/// Read `verdict` first: it states in plain language whether the ensemble
/// mattered. `effective_n_conformers` is the inverse participation ratio
/// `1 / Σ w_i²` — 1.0 when one conformer carries everything, N when N are
/// degenerate.
#[pyclass]
#[pyo3(name = "EnsembleDiagnostics")]
struct PyEnsembleDiagnostics {
    #[pyo3(get)] n_conformers: usize,
    /// Conformers within `kT` of the minimum (always >= 1: the minimum itself).
    #[pyo3(get)] n_within_kt: usize,
    /// Conformers within `2kT` of the minimum.
    #[pyo3(get)] n_within_2kt: usize,
    /// Conformers within `5kT` of the minimum.
    #[pyo3(get)] n_within_5kt: usize,
    /// Largest single Boltzmann population.
    #[pyo3(get)] max_weight: f64,
    /// Index of the conformer carrying `max_weight`.
    #[pyo3(get)] max_weight_index: usize,
    /// Inverse participation ratio `1 / Σ w_i²`.
    #[pyo3(get)] effective_n_conformers: f64,
    #[pyo3(get)] temperature_k: f64,
    /// Plain-language verdict on whether the ensemble was needed.
    #[pyo3(get)] verdict: String,
    text: String,
}

#[pymethods]
impl PyEnsembleDiagnostics {
    /// `True` when one conformer carries at least `threshold` of the
    /// population (pass 0.95 for the usual sense) — the ensemble added nothing.
    #[pyo3(signature = (threshold=0.95))]
    fn is_single_conformer_dominated(&self, threshold: f64) -> bool {
        self.max_weight >= threshold
    }
    /// Multi-line human-readable summary (same text as the Rust `Display`).
    fn __str__(&self) -> String { self.text.clone() }
    fn __repr__(&self) -> String {
        format!(
            "EnsembleDiagnostics(n_conformers={}, max_weight={:.6}, effective_n_conformers={:.3}, T={:.2} K)",
            self.n_conformers, self.max_weight, self.effective_n_conformers, self.temperature_k
        )
    }
}

impl From<ferric_core::conformers::EnsembleDiagnostics> for PyEnsembleDiagnostics {
    fn from(d: ferric_core::conformers::EnsembleDiagnostics) -> Self {
        PyEnsembleDiagnostics {
            n_conformers: d.n_conformers,
            n_within_kt: d.n_within_kt,
            n_within_2kt: d.n_within_2kt,
            n_within_5kt: d.n_within_5kt,
            max_weight: d.max_weight,
            max_weight_index: d.max_weight_index,
            effective_n_conformers: d.effective_n_conformers,
            temperature_k: d.temperature_k,
            verdict: d.verdict().to_string(),
            text: d.to_string(),
        }
    }
}

/// Boltzmann populations of an ensemble at one temperature.
#[pyclass]
#[pyo3(name = "BoltzmannWeights")]
struct PyBoltzmannWeights {
    /// Normalized populations in ensemble order. Sum to 1.
    #[pyo3(get)] weights: Vec<f64>,
    /// Energies relative to the ensemble minimum, in Hartree (all >= 0).
    #[pyo3(get)] relative_energies: Vec<f64>,
    #[pyo3(get)] temperature_k: f64,
    /// `kT` at that temperature, in Hartree.
    #[pyo3(get)] kt_hartree: f64,
    /// Index of the lowest-energy conformer.
    #[pyo3(get)] min_index: usize,
    /// `Z = Σ exp(-(E_i - E_min)/kT)`, always >= 1.
    #[pyo3(get)] partition_function: f64,
    inner: ferric_core::conformers::BoltzmannWeights,
}

#[pymethods]
impl PyBoltzmannWeights {
    /// Population-structure diagnostics for this weighting.
    fn diagnostics(&self) -> PyEnsembleDiagnostics { self.inner.diagnostics().into() }
    fn __len__(&self) -> usize { self.weights.len() }
    fn __repr__(&self) -> String {
        format!(
            "BoltzmannWeights(n={}, T={:.2} K, max_weight={:.6})",
            self.weights.len(),
            self.temperature_k,
            self.inner.diagnostics().max_weight
        )
    }
}

impl From<ferric_core::conformers::BoltzmannWeights> for PyBoltzmannWeights {
    fn from(w: ferric_core::conformers::BoltzmannWeights) -> Self {
        PyBoltzmannWeights {
            weights: w.weights.clone(),
            relative_energies: w.relative_energies.clone(),
            temperature_k: w.temperature_k,
            kt_hartree: w.kt_hartree,
            min_index: w.min_index,
            partition_function: w.partition_function,
            inner: w,
        }
    }
}

/// A set of conformers of one chemical species, sharing atom ordering,
/// composition, charge and multiplicity.
///
/// The shared-ordering invariant is enforced at construction and cannot be
/// bypassed. This matters: averaging a per-atom property (charges, per-atom
/// C6) across differently-ordered geometries produces a plausible-looking
/// number that is simply wrong, with no runtime symptom. A violation raises
/// `ValueError` naming the offending conformer and atom.
///
/// # Units
///
/// `from_coordinates` takes **Ångström**, matching RDKit's
/// `Conformer.GetPositions()` and ferric's own XYZ entry points. Coordinates
/// are stored internally in Bohr (as everywhere in ferric); `coordinates()`
/// converts back to Ångström (to ~1 ulp — the conversion is a float multiply
/// with no exact inverse), `coordinates_bohr()` returns the stored Bohr values
/// verbatim.
///
/// # Typical use (the RDKit path)
///
/// ```python
/// from rdkit import Chem
/// from rdkit.Chem import AllChem
/// import ferric
///
/// m = Chem.AddHs(Chem.MolFromSmiles("CCO"))
/// AllChem.EmbedMultipleConfs(m, numConfs=20, randomSeed=0xf00d)
/// symbols = [a.GetSymbol() for a in m.GetAtoms()]
/// coords  = [c.GetPositions() for c in m.GetConformers()]   # Ångström
///
/// ens = ferric.ConformerEnsemble.from_coordinates(coords, symbols)
///
/// basis = ferric.BasisSet.bundled("sto-3g")
/// energies, dipoles = [], []
/// for mol in ens.molecules():
///     scf = ferric.run_rhf(mol, basis)
///     energies.append(scf.energy)
///     dipoles.append(some_dipole(mol, basis, scf))
///
/// w = ens.boltzmann_weights(energies)          # 298.15 K by default
/// print(w.diagnostics())                       # did the ensemble matter?
/// mu = ferric.weighted_stats_vector(dipoles, w.weights)
/// print(mu[0].mean, "+/-", mu[0].std_dev)
/// ```
#[pyclass]
#[pyo3(name = "ConformerEnsemble")]
struct PyConformerEnsemble {
    inner: ferric_core::conformers::ConformerEnsemble,
}

#[pymethods]
impl PyConformerEnsemble {
    /// Build an ensemble from a list of `natoms x 3` coordinate arrays plus one
    /// shared element list — exactly the shape RDKit gives you:
    ///
    /// ```python
    /// coords  = [c.GetPositions() for c in mol.GetConformers()]  # Ångström
    /// symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    /// ens = ferric.ConformerEnsemble.from_coordinates(coords, symbols)
    /// ```
    ///
    /// **`coordinates` are in Ångström**, matching RDKit's `GetPositions()`
    /// and ferric's XYZ readers; they are converted to Bohr internally. Do not
    /// pre-scale.
    ///
    /// `elements` may be symbols (`"C"`, `"@O"` for a ghost center) or atomic
    /// numbers — pass whichever you have; both are accepted. It is a single
    /// list, shared by all conformers, because a conformer set is one species
    /// by definition.
    ///
    /// Raises `ValueError` on: an empty conformer list, a coordinate/element
    /// count mismatch, a row that is not 3 Cartesian components, a non-finite
    /// coordinate, an unknown element, an inconsistent electron-count /
    /// multiplicity parity, or (from core's invariant check) any conformer
    /// disagreeing with conformer 0 on composition, ordering, charge or
    /// multiplicity.
    #[staticmethod]
    #[pyo3(signature = (coordinates, elements, charge=0, multiplicity=1, energies=None))]
    fn from_coordinates(
        coordinates: Vec<Vec<Vec<f64>>>,
        elements: Bound<'_, PyAny>,
        charge: i32,
        multiplicity: usize,
        energies: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        if coordinates.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "conformer ensemble is empty (need at least one conformer); \
                 did EmbedMultipleConfs return zero conformers?",
            ));
        }
        let symbols = extract_element_symbols(&elements)?;
        let molecules: Vec<Molecule> = coordinates
            .iter()
            .enumerate()
            .map(|(i, c)| molecule_from_coords(c, &symbols, charge, multiplicity, i))
            .collect::<PyResult<_>>()?;

        let inner = match energies {
            Some(e) => {
                if e.len() != molecules.len() {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "got {} energies for {} conformers",
                        e.len(),
                        molecules.len()
                    )));
                }
                ferric_core::conformers::ConformerEnsemble::from_molecules_and_energies(
                    molecules, &e,
                )
            }
            None => ferric_core::conformers::ConformerEnsemble::from_molecules(molecules),
        }
        .map_err(conformer_err)?;
        Ok(PyConformerEnsemble { inner })
    }

    /// Build an ensemble from a **multi-frame** XYZ file (concatenated XYZ
    /// blocks — what RDKit/OpenBabel write for a conformer set).
    ///
    /// `Molecule.from_xyz` reads only the FIRST frame of such a file and
    /// silently ignores the rest; this reads every frame. Coordinates in the
    /// file are Ångström, as in any XYZ.
    ///
    /// Energies are **not** parsed from the comment lines: the formats vary
    /// (`E = -155.4`, bare floats, `MMFF94 -12.3`) and guessing would silently
    /// mis-assign weights. Pass `energies=` explicitly, or compute them with
    /// ferric and use `boltzmann_weights(energies)`.
    #[staticmethod]
    #[pyo3(signature = (path, charge=0, multiplicity=1, energies=None))]
    fn from_multi_xyz(
        path: &str,
        charge: i32,
        multiplicity: usize,
        energies: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        let molecules = ferric_core::conformers::load_multi_xyz(path, charge, multiplicity)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))?;
        let inner = match energies {
            Some(e) => {
                if e.len() != molecules.len() {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "got {} energies for {} frames in {path:?}",
                        e.len(),
                        molecules.len()
                    )));
                }
                ferric_core::conformers::ConformerEnsemble::from_molecules_and_energies(
                    molecules, &e,
                )
            }
            None => ferric_core::conformers::ConformerEnsemble::from_molecules(molecules),
        }
        .map_err(conformer_err)?;
        Ok(PyConformerEnsemble { inner })
    }

    /// Number of conformers (always >= 1).
    fn __len__(&self) -> usize { self.inner.len() }
    /// Number of conformers (always >= 1).
    fn n_conformers(&self) -> usize { self.inner.len() }
    /// Number of atoms per conformer (shared by construction).
    fn n_atoms(&self) -> usize { self.inner.n_atoms() }

    /// The conformer geometries as `Molecule` objects, ready to hand to
    /// `run_rhf` / `run_ksdft` / any other driver.
    fn molecules(&self) -> Vec<PyMolecule> {
        self.inner
            .conformers()
            .iter()
            .map(|c| PyMolecule { inner: c.molecule.clone() })
            .collect()
    }

    /// Geometry of conformer `index` as a `Molecule`.
    fn molecule(&self, index: usize) -> PyResult<PyMolecule> {
        self.inner
            .conformers()
            .get(index)
            .map(|c| PyMolecule { inner: c.molecule.clone() })
            .ok_or_else(|| {
                pyo3::exceptions::PyIndexError::new_err(format!(
                    "conformer index {index} out of range (ensemble has {})",
                    self.inner.len()
                ))
            })
    }

    /// Element symbols, shared by all conformers, in atom order. Ghost centers
    /// come back without the `@` prefix (use `is_ghost()` to distinguish them).
    fn elements(&self) -> Vec<String> {
        self.inner.conformers()[0]
            .molecule
            .atoms
            .iter()
            .map(|a| a.symbol.clone())
            .collect()
    }

    /// Atomic numbers, shared by all conformers, in atom order.
    fn atomic_numbers(&self) -> Vec<i32> {
        self.inner.conformers()[0].molecule.atoms.iter().map(|a| a.z).collect()
    }

    /// Per-atom ghost flags (basis-only centers), shared by all conformers.
    fn is_ghost(&self) -> Vec<bool> {
        self.inner.conformers()[0].molecule.atoms.iter().map(|a| a.ghost).collect()
    }

    /// All conformer geometries as a list of `natoms x 3` numpy arrays in
    /// **Ångström** — the same units `from_coordinates` accepts, so this
    /// round-trips to ~1 ulp (< 1e-14 Å). It is NOT bit-exact: the Å->Bohr
    /// conversion on the way in is a floating-point multiply, which has no
    /// exact inverse. Compare with a tolerance, not `==`.
    fn coordinates<'py>(&self, py: Python<'py>) -> Vec<Bound<'py, PyArray2<f64>>> {
        self.coords_arrays(py, false)
    }

    /// All conformer geometries as a list of `natoms x 3` numpy arrays in
    /// **Bohr** — ferric's internal storage units, returned verbatim with no
    /// arithmetic applied.
    fn coordinates_bohr<'py>(&self, py: Python<'py>) -> Vec<Bound<'py, PyArray2<f64>>> {
        self.coords_arrays(py, true)
    }

    /// Conformer energies in Hartree, in ensemble order.
    ///
    /// Raises `ValueError` if any conformer has no energy set (or a non-finite
    /// one) — an unconverged SCF must not be silently averaged in.
    fn energies(&self) -> PyResult<Vec<f64>> {
        self.inner.energies().map_err(conformer_err)
    }

    /// Set the energy (Hartree) of one conformer, e.g. as SCF results come
    /// back. Raises `ValueError` on an out-of-range index or a non-finite
    /// energy.
    fn set_energy(&mut self, index: usize, energy: f64) -> PyResult<()> {
        self.inner.set_energy(index, energy).map_err(conformer_err)
    }

    /// Boltzmann weights at `temperature_k` Kelvin (default 298.15 K,
    /// thermochemical standard).
    ///
    /// `energies` (Hartree) may be supplied here, or omitted to use energies
    /// already attached to the ensemble (via `from_coordinates(energies=...)`
    /// or `set_energy`). Supplying them here does NOT mutate the ensemble.
    ///
    /// Weights are `w_i = exp(-(E_i - E_min)/kT) / Z`. The `E_min` shift is
    /// what makes this safe on absolute electronic energies: raw
    /// `exp(-E_i/kT)` at E ~ -10^2 Ha and kT ~ 9.4e-4 Ha overflows instantly.
    ///
    /// Raises `ValueError` on a non-positive/non-finite temperature, a
    /// non-finite energy, or an energy-count mismatch.
    #[pyo3(signature = (energies=None, temperature_k=ferric_core::conformers::DEFAULT_TEMPERATURE_K))]
    fn boltzmann_weights(
        &self,
        energies: Option<Vec<f64>>,
        temperature_k: f64,
    ) -> PyResult<PyBoltzmannWeights> {
        let e = match energies {
            Some(e) => {
                if e.len() != self.inner.len() {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "got {} energies for {} conformers",
                        e.len(),
                        self.inner.len()
                    )));
                }
                e
            }
            None => self.inner.energies().map_err(conformer_err)?,
        };
        ferric_core::conformers::boltzmann_weights(&e, temperature_k)
            .map(Into::into)
            .map_err(conformer_err)
    }

    /// Population-structure diagnostics at `temperature_k`. Convenience for
    /// `boltzmann_weights(...).diagnostics()`.
    #[pyo3(signature = (energies=None, temperature_k=ferric_core::conformers::DEFAULT_TEMPERATURE_K))]
    fn diagnostics(
        &self,
        energies: Option<Vec<f64>>,
        temperature_k: f64,
    ) -> PyResult<PyEnsembleDiagnostics> {
        Ok(self.boltzmann_weights(energies, temperature_k)?.inner.diagnostics().into())
    }

    fn __repr__(&self) -> String {
        format!(
            "ConformerEnsemble(n_conformers={}, n_atoms={})",
            self.inner.len(),
            self.inner.n_atoms()
        )
    }
}

impl PyConformerEnsemble {
    /// Shared body of `coordinates()` / `coordinates_bohr()`: emit one
    /// `natoms x 3` array per conformer from the stored Bohr values.
    ///
    /// `bohr = true` returns them untouched. `bohr = false` converts to
    /// Ångström by DIVIDING by `ANGSTROM_TO_BOHR` — the same constant
    /// `Molecule::parse_xyz` multiplied by — which is the closer inverse than
    /// multiplying by the reciprocal literal (see `ANGSTROM_TO_BOHR`'s doc).
    fn coords_arrays<'py>(
        &self,
        py: Python<'py>,
        bohr: bool,
    ) -> Vec<Bound<'py, PyArray2<f64>>> {
        let conv = |v: f64| if bohr { v } else { v / ANGSTROM_TO_BOHR };
        self.inner
            .conformers()
            .iter()
            .map(|c| {
                let n = c.molecule.atoms.len();
                let mut a = Array2::<f64>::zeros((n, 3));
                for (i, at) in c.molecule.atoms.iter().enumerate() {
                    a[[i, 0]] = conv(at.x);
                    a[[i, 1]] = conv(at.y);
                    a[[i, 2]] = conv(at.zpos);
                }
                PyArray2::from_array(py, &a)
            })
            .collect()
    }
}

/// The Ångström->Bohr factor `ferric_core::mol` applies when parsing XYZ
/// (its `ANGSTROM_TO_BOHR`, which is private, hence the duplicate literal —
/// keep the two in sync).
///
/// `coordinates()` inverts the conversion by **dividing** by this constant
/// rather than multiplying by the literal 0.529_177_210_92. Both are correct
/// to 1 ulp, but the divide is the better inverse: measured over 3e6 random
/// coordinates in -20..20 Å, `x * A2B / A2B != x` for 5.0% of values whereas
/// `x * A2B * 0.529_177_210_92 != x` for 15.4%; worst-case error is 3.6e-15 Å
/// either way. Floating-point multiplication is not exactly invertible, so a
/// round-trip through `from_coordinates` -> `coordinates()` is accurate to
/// ~1 ulp (< 1e-14 Å), NOT bit-exact — do not assert equality on it.
const ANGSTROM_TO_BOHR: f64 = 1.0 / 0.529_177_210_92;

/// Accept either element symbols (`"C"`, `"@O"`) or atomic numbers (`6`) for
/// the shared element list — RDKit hands you `GetSymbol()` naturally, but
/// `GetAtomicNum()` is just as common, and silently accepting only one of them
/// is a trap. Mixed lists are accepted too (each entry is resolved on its own).
fn extract_element_symbols(elements: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    let items: Vec<Bound<'_, PyAny>> = elements.extract().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(
            "elements must be a sequence of element symbols (e.g. ['O','H','H']) \
             or atomic numbers (e.g. [8,1,1])",
        )
    })?;
    if items.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "elements list is empty (need one entry per atom)",
        ));
    }
    items
        .iter()
        .enumerate()
        .map(|(i, item)| {
            if let Ok(s) = item.extract::<String>() {
                return Ok(s);
            }
            // `bool` is a subclass of `int` in Python; extracting an atomic
            // number from `True` would silently give hydrogen. Reject it.
            if item.is_instance_of::<pyo3::types::PyBool>() {
                return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                    "element {i} is a bool, not an element symbol or atomic number"
                )));
            }
            if let Ok(z) = item.extract::<i32>() {
                return ferric_core::elements::z_to_symbol(z).map(str::to_string).ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "element {i}: atomic number {z} is not a known element"
                    ))
                });
            }
            Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "element {i}: expected an element symbol (str) or atomic number (int), \
                 got {}",
                item.get_type().name().map(|n| n.to_string()).unwrap_or_else(|_| "?".into())
            )))
        })
        .collect()
}

/// Boltzmann weights from a bare list of energies (Hartree) at
/// `temperature_k` Kelvin (default 298.15 K).
///
/// Free-function form for callers holding energies without a
/// `ConformerEnsemble` — e.g. weighting results that came from somewhere else.
/// `w_i = exp(-(E_i - E_min)/kT) / Z`, computed with the `E_min` shift so
/// absolute electronic energies do not overflow.
///
/// Raises `ValueError` on an empty list, a non-finite energy, or a
/// non-positive/non-finite temperature.
#[pyfunction]
#[pyo3(signature = (energies, temperature_k=ferric_core::conformers::DEFAULT_TEMPERATURE_K))]
fn boltzmann_weights(energies: Vec<f64>, temperature_k: f64) -> PyResult<PyBoltzmannWeights> {
    ferric_core::conformers::boltzmann_weights(&energies, temperature_k)
        .map(Into::into)
        .map_err(conformer_err)
}

/// Weighted mean and standard deviation of a **scalar** property across
/// conformers.
///
/// `values[i]` is the property computed on conformer `i`; `weights[i]` its
/// Boltzmann population (from `boltzmann_weights`). Returns a `WeightedStats`
/// carrying `mean`, `std_dev`, `min`, `max`.
///
/// The variance is computed in the shifted form `Σ w_i (x_i - mean)²`, not
/// `E[x²] - E[x]²`: the latter loses every significant digit when the values
/// are large and their spread is small — exactly the regime of absolute
/// electronic energies.
///
/// Raises `ValueError` on a length mismatch, an empty list, or a non-finite
/// value.
#[pyfunction]
fn weighted_stats(values: Vec<f64>, weights: Vec<f64>) -> PyResult<PyWeightedStats> {
    ferric_core::conformers::weighted_stats(&values, &weights)
        .map(Into::into)
        .map_err(conformer_err)
}

/// Weighted mean and standard deviation of a **vector-valued** property
/// (a dipole, a per-atom charge array), applied component-wise.
///
/// `values[i]` is conformer `i`'s property vector; all must be the same
/// length. Returns one `WeightedStats` per component.
///
/// Component-wise is the right default and the reason the spread is
/// mandatory: a symmetric ensemble's dipole components can cancel to near
/// zero while every individual conformer is strongly polar. The mean alone
/// would report "non-polar"; the per-component `std_dev` exposes the truth.
/// For the average *magnitude* instead, compute per-conformer magnitudes in
/// Python and pass them to `weighted_stats` — the two are different physical
/// questions and this API refuses to conflate them.
///
/// Raises `ValueError` on a length mismatch or ragged input.
#[pyfunction]
fn weighted_stats_vector(
    values: Vec<Vec<f64>>,
    weights: Vec<f64>,
) -> PyResult<Vec<PyWeightedStats>> {
    ferric_core::conformers::weighted_stats_vector(&values, &weights)
        .map(|v| v.into_iter().map(Into::into).collect())
        .map_err(conformer_err)
}

/// Weighted mean and standard deviation of a **rank-2 tensor** property
/// (a 3x3 polarizability, a quadrupole), applied element-wise.
///
/// `values[i]` is conformer `i`'s `nrow x ncol` tensor; all conformers must
/// supply the same shape. Returns an `nrow x ncol` nested list of
/// `WeightedStats`.
///
/// Raises `ValueError` on a length mismatch or inconsistent shapes.
#[pyfunction]
fn weighted_stats_tensor(
    values: Vec<Vec<Vec<f64>>>,
    weights: Vec<f64>,
) -> PyResult<Vec<Vec<PyWeightedStats>>> {
    ferric_core::conformers::weighted_stats_tensor(&values, &weights)
        .map(|rows| rows.into_iter().map(|r| r.into_iter().map(Into::into).collect()).collect())
        .map_err(conformer_err)
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

    fn scf(&self) -> &ScfResult {
        match self {
            DensitySource::Rhf(r) => &r.scf_data,
            DensitySource::Dft(d) => &d.scf_data,
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

/// Electrostatic potential at ARBITRARY points, in Hartree atomic units
/// (e/Bohr). `points` is an (N, 3) array of Cartesian positions in **Bohr**.
///
/// This is the primitive behind CHELPG/RESP and is what you want for an ESP on
/// a molecular surface (vdW shell, solvent-accessible grid, ...). Prefer it
/// over reconstructing the surface potential from partial charges plus induced
/// dipoles: that is a point-multipole approximation to a quantity computed
/// here exactly from the density.
///
/// `result` is an `RhfResult` or `DftResult` from a converged SCF.
#[pyfunction]
fn esp_at_points(
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    result: DensitySource,
    points: numpy::PyReadonlyArray2<f64>,
) -> PyResult<Vec<f64>> {
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    let arr = points.as_array();
    if arr.ncols() != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "points must have shape (N, 3); got (_, {})",
            arr.ncols()
        )));
    }
    let pts: Vec<[f64; 3]> = arr
        .rows()
        .into_iter()
        .map(|r| [r[0], r[1], r[2]])
        .collect();
    ferric_scf::properties::esp_at_points(&mol.inner, &prep, result.density(), &pts)
        .map_err(make_err)
}

/// Hirshfeld partial charges (units of e), using the default free-atom
/// (single-exponential Slater) proatom reference. `result` is an `RhfResult`
/// or `DftResult` from a converged SCF.
#[pyfunction]
fn hirshfeld_charges(mol: &PyMolecule, basis_set: &PyBasisSet, result: DensitySource) -> PyResult<Vec<f64>> {
    ferric_rpa::properties::hirshfeld_charges(&mol.inner, &basis_set.inner, result.density(), None)
        .map_err(make_err)
}


/// Per-orbital centroids <p|r|p> (Bohr, list of [x,y,z]) and spatial spreads
/// sigma_p = sqrt(<r^2> - |<r>|^2) (Bohr) for the converged restricted MOs.
/// Mirrors Q-Chem 7's "second moments of orbitals" property; the spreads are
/// the pair-gate inputs of the amplitude-threshold local-correlation family.
#[pyfunction]
fn orbital_moments(
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    result: DensitySource,
) -> PyResult<(Vec<[f64; 3]>, Vec<f64>)> {
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    let scf = result.scf();
    if scf.spin != ferric_scf::result::Spin::Restricted {
        return Err(make_err(ferric_core::FerricError::General(
            "orbital_moments: restricted (RHF/RKS) results only".into(),
        )));
    }
    let (centers, spreads) =
        ferric_integrals::oneelectron::orbital_moments(&prep, scf.mos_r()).map_err(make_err)?;
    let c: Vec<[f64; 3]> = (0..centers.nrows())
        .map(|p| [centers[(p, 0)], centers[(p, 1)], centers[(p, 2)]])
        .collect();
    Ok((c, spreads))
}

/// Electronic-density second-moment tensor (3x3 nested list, Bohr^2) about
/// the origin: M_xy = sum_uv D_uv <u|x y|v>; trace = <r^2> of the density.
#[pyfunction]
fn density_second_moment(
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    result: DensitySource,
) -> PyResult<[[f64; 3]; 3]> {
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    ferric_integrals::oneelectron::density_second_moment(&prep, result.density(), [0.0; 3])
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

/// Mulliken partial charges (units of e), the standard textbook population
/// analysis. Known to be more basis-set-sensitive than Löwdin (prefer
/// `lowdin_charges` for a more basis-stable partition); included as the
/// standard baseline every QC package provides. Closed-shell only. `result`
/// is an `RhfResult` or `DftResult` from a converged SCF.
#[pyfunction]
fn mulliken_charges(mol: &PyMolecule, basis_set: &PyBasisSet, result: DensitySource) -> PyResult<Vec<f64>> {
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    ferric_rpa::properties::mulliken_charges(&mol.inner, &prep, result.density()).map_err(make_err)
}

/// CHELPG (CHarges from Electrostatic Potentials, Grid-based) partial
/// charges (units of e). Structurally different from
/// `hirshfeld_charges`/`lowdin_charges`/`mulliken_charges`: those split the
/// electron density directly among atoms (population partition); CHELPG
/// instead fits atom-centered point charges to best reproduce the molecular
/// electrostatic potential on a grid around the molecule (Breneman & Wiberg,
/// J. Comput. Chem. 11, 361 (1990)). Closed-shell only. `result` is an
/// `RhfResult` or `DftResult` from a converged SCF.
#[pyfunction]
fn chelpg_charges(mol: &PyMolecule, basis_set: &PyBasisSet, result: DensitySource) -> PyResult<Vec<f64>> {
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    ferric_rpa::properties::chelpg_charges(&mol.inner, &prep, result.density()).map_err(make_err)
}

/// RESP (Restrained ElectroStatic Potential) partial charges (units of e).
/// Same ESP grid-fitting as `chelpg_charges`, plus a hyperbolic restraint
/// damping non-hydrogen atomic charges toward zero (Bayly, Cieplak,
/// Cornell, Kollman, J. Phys. Chem. 97, 10269 (1993)). This is a
/// single-stage restrained fit with the standard literature weight/
/// tightness parameters — NOT the full multi-stage/multi-conformer RESP
/// averaging procedure. Closed-shell only. `result` is an `RhfResult` or
/// `DftResult` from a converged SCF.
#[pyfunction]
fn resp_charges(mol: &PyMolecule, basis_set: &PyBasisSet, result: DensitySource) -> PyResult<Vec<f64>> {
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    ferric_rpa::properties::resp_charges(&mol.inner, &prep, result.density()).map_err(make_err)
}

/// Per-atom Hirshfeld-partitioned static dipole polarizability tensors
/// (Bohr^3), via true PDEP-RPA α(ω=0) on a real-space Becke/Hirshfeld grid.
/// Closed-shell (Restricted) only — `result` must come from `run_rhf` or
/// `run_dft` on a closed-shell molecule. `auxbasis` is the RI auxiliary basis
/// (e.g. a `*-rifit`/`*-ri` bundled set, NOT the SCF's own `df_j_aux`).
///
/// This is materially more expensive than `esp_at_atoms`/`hirshfeld_charges`/
/// `lowdin_charges`: it builds RI intermediates plus an AO-value grid over
/// the whole molecule (`FERRIC_HIRSHFELD_SPACING`/`FERRIC_HIRSHFELD_MARGIN`
/// env vars control resolution vs. memory; the grid is budget-checked
/// against `memory_budget_gb`, auto-resolved to 80% of available RAM when
/// omitted — see `ferric_core::memory::resolve_budget_bytes`).
#[pyfunction]
#[pyo3(signature = (mol, basis_set, auxbasis, result, memory_budget_gb=None))]
fn hirshfeld_polarizability<'py>(
    py: Python<'py>,
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    auxbasis: &PyBasisSet,
    result: DensitySource,
    memory_budget_gb: Option<f64>,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    let dfbs = PreparedBasis::new(&mol.inner, &auxbasis.inner).map_err(make_err)?;
    let op = Operator::coulomb();
    let cfg = ferric_rpa::PdepRpaConfig {
        memory_budget_bytes: budget_bytes_from_gb(memory_budget_gb),
        ..Default::default()
    };
    let alpha = ferric_rpa::properties::pdep_polarizability_hirshfeld(
        &mol.inner, &prep, &basis_set.inner, &dfbs, result.scf(), op, &cfg, None,
    ).map_err(make_err)?;
    let natoms = alpha.len();
    let mut arr = Array3::<f64>::zeros((natoms, 3, 3));
    for (a, tensor) in alpha.iter().enumerate() {
        for i in 0..3 {
            for j in 0..3 {
                arr[(a, i, j)] = tensor[i][j];
            }
        }
    }
    Ok(PyArray3::from_owned_array(py, arr))
}

// ── RI-MP2 ──

/// Result of a `run_rimp2` calculation.
#[pyclass]
#[pyo3(name = "RiMp2Result")]
struct PyRiMp2Result {
    /// RHF + MP2 correlation energy, Hartree.
    #[pyo3(get)] total_energy: f64,
    /// The converged reference RHF energy alone, Hartree.
    #[pyo3(get)] rhf_energy: f64,
    /// MP2 correlation energy alone (always negative), Hartree.
    #[pyo3(get)] mp2_corr: f64,
}

#[pymethods]
impl PyRiMp2Result {
    fn __repr__(&self) -> String {
        format!("RiMp2Result(total_energy={:.10}, mp2_corr={:.10})", self.total_energy, self.mp2_corr)
    }
    fn __str__(&self) -> String {
        format!(
            "RI-MP2 Total Energy: {:.10} Ha (RHF: {:.10}, corr: {:.10})",
            self.total_energy, self.rhf_energy, self.mp2_corr,
        )
    }
}

/// Resolution-of-identity (density-fitted) MP2 on a closed-shell RHF
/// reference. Runs its own internal RHF first (via `k_builder`, same
/// convention as `run_rhf`'s `k_builder` kwarg), then the RI-MP2 correlation
/// energy using `auxbasis` as the fitting basis.
///
/// `auxbasis` is the RI auxiliary basis (e.g. a bundled `*-ri`/`*-rifit` set
/// such as `"cc-pvdz-ri"` for orbital basis `"cc-pvdz"`, or `"def2-svp-rifit"`
/// for `"def2-svp"`) — NOT the SCF's own `df_j_aux`/`df_k_aux`. See
/// `docs/quickstart.md`'s basis/auxiliary-basis pairing table.
///
/// `frozen_core` (default 0) excludes that many lowest-energy occupied
/// orbitals from the correlation treatment.
///
/// Raises if the internal RHF does not converge.
///
/// Returns a [`RiMp2Result`](PyRiMp2Result) with `total_energy` (RHF + MP2
/// correlation), `rhf_energy` (the reference energy), and `mp2_corr` (the
/// correlation energy alone, always negative).
#[pyfunction]
#[pyo3(signature = (mol, basis_set, auxbasis, frozen_core=None, k_builder=None, memory_budget_gb=None, kappa=None))]
fn run_rimp2(mol: &PyMolecule, basis_set: &PyBasisSet, auxbasis: &PyBasisSet,
             frozen_core: Option<usize>, k_builder: Option<&str>,
             memory_budget_gb: Option<f64>, kappa: Option<f64>) -> PyResult<PyRiMp2Result> {
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
                      &RiMp2Config { frozen_core: frozen_core.unwrap_or(0), memory_budget_bytes: budget_bytes_from_gb(memory_budget_gb), kappa, ..Default::default() }).map_err(make_err)?;
    Ok(PyRiMp2Result { total_energy: mp2.total_energy, rhf_energy: rhf.energy, mp2_corr: mp2.mp2_corr })
}


/// Amplitude-threshold local MP2 (WSHG23 single-threshold; closed-shell).
/// `eps = 0` reproduces `run_rimp2` exactly (library anchor <= 1e-9); the
/// default `eps = 1e-4` carries a one-sided ~linear-in-eps truncation error.
/// Returns a dict with e_corr, e_corr_canonical_ri, total_energy and the
/// sparsity counters (keep/pair fractions, domain sizes, CG iterations).
#[pyfunction]
#[pyo3(signature = (mol, basis_set, auxbasis, eps=None, frozen_core=None, k_builder=None, memory_budget_gb=None, compute_reference=None))]
fn run_lmp2(py: Python<'_>, mol: &PyMolecule, basis_set: &PyBasisSet, auxbasis: &PyBasisSet,
            eps: Option<f64>, frozen_core: Option<usize>, k_builder: Option<&str>,
            memory_budget_gb: Option<f64>, compute_reference: Option<bool>) -> PyResult<Py<pyo3::types::PyDict>> {
    use ferric_mp2::lmp2_amplitude::{amplitude_lmp2, AmplitudeLmp2Config};
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    let dfbs = PreparedBasis::new(&mol.inner, &auxbasis.inner).map_err(make_err)?;
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).map_err(make_err)?;
    let ctx = ParallelContext::default();
    let rhf = solve_rhf(&ctx, &mol.inner, &prep, op, &bounds, &rhf_config(k_builder)).map_err(make_err)?;
    if !rhf.converged {
        return Err(make_err(ferric_core::FerricError::ScfConvergence { iterations: rhf.iterations, last_energy: rhf.energy }));
    }
    let r = amplitude_lmp2(&mol.inner, &prep, &basis_set.inner, &dfbs, op, &rhf,
        &AmplitudeLmp2Config {
            eps: eps.unwrap_or(1e-4),
            frozen_core: frozen_core.unwrap_or(0),
            eri3_budget_bytes: budget_bytes_from_gb(memory_budget_gb),
            compute_reference: compute_reference.unwrap_or(true),
            ..Default::default()
        }).map_err(make_err)?;
    let d = pyo3::types::PyDict::new(py);
    d.set_item("e_corr", r.e_corr)?;
    d.set_item("e_corr_canonical_ri", r.e_corr_canonical_ri)?;
    d.set_item("total_energy", r.e_total)?;
    d.set_item("rhf_energy", rhf.energy)?;
    d.set_item("keep_fraction", r.keep_fraction)?;
    d.set_item("pair_fraction", r.pair_fraction)?;
    d.set_item("dom_mean", r.dom_mean)?;
    d.set_item("dom_max", r.dom_max)?;
    d.set_item("cg_iterations", r.cg_iterations)?;
    Ok(d.into())
}


/// Build an `AmplitudeDrpaConfig` from the shared `run_drpa`/`run_drpa_scan`
/// kwargs. `diis`/`eps_rtol_factor` default ON at the BINDING level
/// (diis=8, eps_rtol_factor=0.1) — the Rust library default stays
/// None/off; these are the calibrated values from
/// wiki/amplitude-threshold-drpa.md's 2026-08-17 DIIS+eps-link session
/// (70-79 -> 8-11 iterations, energies within the measured subdominance
/// bound, ~10% of truncation error). `diis=0` maps to `None` (disables
/// DIIS); `eps_rtol_factor=0.0` maps to `None` (disables the eps-link,
/// which is also the eps=0 behavior regardless — see the Rust doc).
#[allow(clippy::too_many_arguments)]
fn drpa_config(
    eps: Option<f64>,
    frozen_core: Option<usize>,
    memory_budget_gb: Option<f64>,
    compute_reference: Option<bool>,
    diis: Option<usize>,
    eps_rtol_factor: Option<f64>,
) -> ferric_mp2::drpa_amplitude::AmplitudeDrpaConfig {
    use ferric_mp2::drpa_amplitude::AmplitudeDrpaConfig;
    let diis_subspace = diis.unwrap_or(8);
    let eps_rtol = eps_rtol_factor.unwrap_or(0.1);
    AmplitudeDrpaConfig {
        eps: eps.unwrap_or(1e-4),
        frozen_core: frozen_core.unwrap_or(0),
        eri3_budget_bytes: budget_bytes_from_gb(memory_budget_gb),
        compute_reference: compute_reference.unwrap_or(true),
        diis: if diis_subspace == 0 { None } else { Some(diis_subspace) },
        eps_rtol_factor: if eps_rtol == 0.0 { None } else { Some(eps_rtol) },
        ..Default::default()
    }
}

fn drpa_result_to_dict(
    py: Python<'_>,
    r: &ferric_mp2::drpa_amplitude::AmplitudeDrpaResult,
    rhf_energy: f64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let d = pyo3::types::PyDict::new(py);
    d.set_item("e_corr", r.e_corr)?;
    d.set_item("e_corr_plasmon_canonical", r.e_corr_plasmon_canonical)?;
    d.set_item("total_energy", r.e_total)?;
    d.set_item("rhf_energy", rhf_energy)?;
    d.set_item("keep_fraction", r.keep_fraction)?;
    d.set_item("pair_fraction", r.pair_fraction)?;
    d.set_item("iterations", r.iterations)?;
    d.set_item("relres", r.relres)?;
    d.set_item("converged", r.converged)?;
    Ok(d.into())
}

/// Amplitude-threshold direct RPA (drCCD Riccati, ragged direct-assembly
/// path; proof notebook 12). eps = 0 anchors on the canonical
/// semicanonicalized plasmon formula; finite eps carries a ~linear
/// one-sided threshold error (dRPA is non-variational). Closed-shell.
///
/// `diis` (default 8): Pulay/DIIS subspace size accelerating the Riccati
/// fixed point (measured 70-79 -> 25-28 iterations); pass `diis=0` to
/// disable and recover the legacy unaccelerated solve.
/// `eps_rtol_factor` (default 0.1): links the fixed-point stopping
/// tolerance to eps (effective rtol = max(fp_rtol, eps_rtol_factor*eps));
/// combined with DIIS this reaches 8-11 iterations. Pass
/// `eps_rtol_factor=0.0` to disable and use the tight fp_rtol=1e-12
/// always. At `eps=0` this is a no-op regardless of the factor, so the
/// eps=0 exactness anchor is unaffected either way.
///
/// Both knobs change WHICH POINT on the same truncated-equation solution
/// manifold is returned, not which equation is solved: energies at finite
/// eps can shift versus the legacy tight-rtol/no-DIIS solve, but the shift
/// is calibrated to stay within ~10% of eps's own truncation error (see
/// wiki/amplitude-threshold-drpa.md, "Subdominance calibration").
#[pyfunction]
#[pyo3(signature = (mol, basis_set, auxbasis, eps=None, frozen_core=None, k_builder=None, memory_budget_gb=None, compute_reference=None, diis=None, eps_rtol_factor=None))]
#[allow(clippy::too_many_arguments)]
fn run_drpa(py: Python<'_>, mol: &PyMolecule, basis_set: &PyBasisSet, auxbasis: &PyBasisSet,
            eps: Option<f64>, frozen_core: Option<usize>, k_builder: Option<&str>,
            memory_budget_gb: Option<f64>, compute_reference: Option<bool>,
            diis: Option<usize>, eps_rtol_factor: Option<f64>) -> PyResult<Py<pyo3::types::PyDict>> {
    use ferric_mp2::drpa_amplitude::amplitude_drpa;
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    let dfbs = PreparedBasis::new(&mol.inner, &auxbasis.inner).map_err(make_err)?;
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).map_err(make_err)?;
    let ctx = ParallelContext::default();
    let rhf = solve_rhf(&ctx, &mol.inner, &prep, op, &bounds, &rhf_config(k_builder)).map_err(make_err)?;
    if !rhf.converged {
        return Err(make_err(ferric_core::FerricError::ScfConvergence { iterations: rhf.iterations, last_energy: rhf.energy }));
    }
    let cfg = drpa_config(eps, frozen_core, memory_budget_gb, compute_reference, diis, eps_rtol_factor);
    let r = amplitude_drpa(&mol.inner, &prep, &basis_set.inner, &dfbs, op, &rhf, &cfg).map_err(make_err)?;
    drpa_result_to_dict(py, &r, rhf.energy)
}

/// Amplitude-threshold direct RPA over a LIST of eps values, reusing ONE
/// RHF solve + ONE localized-basis assembly (RHF, Boys localization,
/// VV-HV, Fock blocks, unwhitened RI B — everything eps-INDEPENDENT)
/// across every point; only the eps-dependent ragged assembly + Riccati
/// solve reruns per eps. Returns a list of dicts, each with the SAME keys
/// as `run_drpa`, plus `wall_s` (that point's ragged-assembly+solve wall
/// time) and `prefix_wall_s` (the ONE shared RHF+localization+assembly
/// wall time, repeated on every dict for convenience — it is not
/// re-paid per point). Per-point energies are byte-identical (<1e-12) to
/// calling `run_drpa` separately at each eps with the same kwargs.
#[pyfunction]
#[pyo3(signature = (mol, basis_set, auxbasis, eps_list, frozen_core=None, k_builder=None, memory_budget_gb=None, compute_reference=None, diis=None, eps_rtol_factor=None))]
#[allow(clippy::too_many_arguments)]
fn run_drpa_scan(py: Python<'_>, mol: &PyMolecule, basis_set: &PyBasisSet, auxbasis: &PyBasisSet,
                  eps_list: Vec<f64>, frozen_core: Option<usize>, k_builder: Option<&str>,
                  memory_budget_gb: Option<f64>, compute_reference: Option<bool>,
                  diis: Option<usize>, eps_rtol_factor: Option<f64>) -> PyResult<Py<pyo3::types::PyList>> {
    use ferric_mp2::drpa_amplitude::amplitude_drpa_scan_timed;
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    let dfbs = PreparedBasis::new(&mol.inner, &auxbasis.inner).map_err(make_err)?;
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).map_err(make_err)?;
    let ctx = ParallelContext::default();
    let rhf = solve_rhf(&ctx, &mol.inner, &prep, op, &bounds, &rhf_config(k_builder)).map_err(make_err)?;
    if !rhf.converged {
        return Err(make_err(ferric_core::FerricError::ScfConvergence { iterations: rhf.iterations, last_energy: rhf.energy }));
    }
    let base_cfg = drpa_config(None, frozen_core, memory_budget_gb, compute_reference, diis, eps_rtol_factor);
    let (results, prefix_wall_s, per_point_walls) = amplitude_drpa_scan_timed(
        &mol.inner, &prep, &basis_set.inner, &dfbs, op, &rhf, &base_cfg, &eps_list,
    ).map_err(make_err)?;
    let out = pyo3::types::PyList::empty(py);
    for ((r, eps), wall_s) in results.iter().zip(&eps_list).zip(&per_point_walls) {
        let d = drpa_result_to_dict(py, r, rhf.energy)?;
        let d_ref = d.bind(py);
        d_ref.set_item("eps", *eps)?;
        d_ref.set_item("wall_s", *wall_s)?;
        d_ref.set_item("prefix_wall_s", prefix_wall_s)?;
        out.append(d_ref)?;
    }
    Ok(out.into())
}

/// Amplitude-threshold LinLCCD (ragged path, proof notebook 13).
/// `variant`: "drivers" (== RI-MP2), "hh" (LinLCCD(hh)), "full".
/// eps = 0 anchors on the canonical spin-orbital linlccd. Closed-shell.
#[pyfunction]
#[pyo3(signature = (mol, basis_set, auxbasis, variant=None, eps=None, frozen_core=None, k_builder=None, memory_budget_gb=None))]
fn run_linlccd_amplitude(py: Python<'_>, mol: &PyMolecule, basis_set: &PyBasisSet,
                         auxbasis: &PyBasisSet, variant: Option<&str>, eps: Option<f64>,
                         frozen_core: Option<usize>, k_builder: Option<&str>,
                         memory_budget_gb: Option<f64>) -> PyResult<Py<pyo3::types::PyDict>> {
    use ferric_cc::linlccd::LadderVariant;
    use ferric_cc::linlccd_amplitude::{amplitude_linlccd, AmplitudeLinLccdConfig};
    let var = match variant.unwrap_or("hh") {
        "drivers" => LadderVariant::DriversOnly,
        "hh" => LadderVariant::Hh,
        "full" => LadderVariant::Full,
        other => {
            return Err(make_err(ferric_core::FerricError::General(format!(
                "unknown LinLCCD variant {other:?}; expected \"drivers\", \"hh\", or \"full\""
            ))))
        }
    };
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    let dfbs = PreparedBasis::new(&mol.inner, &auxbasis.inner).map_err(make_err)?;
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).map_err(make_err)?;
    let ctx = ParallelContext::default();
    let rhf = solve_rhf(&ctx, &mol.inner, &prep, op, &bounds, &rhf_config(k_builder)).map_err(make_err)?;
    if !rhf.converged {
        return Err(make_err(ferric_core::FerricError::ScfConvergence { iterations: rhf.iterations, last_energy: rhf.energy }));
    }
    let r = amplitude_linlccd(&mol.inner, &prep, &basis_set.inner, &dfbs, op, &rhf,
        &AmplitudeLinLccdConfig {
            eps: eps.unwrap_or(1e-4),
            frozen_core: frozen_core.unwrap_or(0),
            eri3_budget_bytes: budget_bytes_from_gb(memory_budget_gb),
            ..Default::default()
        }, var).map_err(make_err)?;
    let d = pyo3::types::PyDict::new(py);
    d.set_item("e_corr", r.e_corr)?;
    d.set_item("total_energy", r.e_total)?;
    d.set_item("rhf_energy", rhf.energy)?;
    d.set_item("keep_fraction", r.keep_fraction)?;
    d.set_item("cg_iterations", r.cg_iterations)?;
    Ok(d.into())
}

/// Optimal (Baer/Kronik IP-based) tuning of the range-separation omega for
/// an RSH functional. Returns {omega, j, converged, evals: [(omega,
/// eps_homo, ip, j), ...]}. Closed-shell neutral + doublet cation.
#[pyfunction]
#[pyo3(signature = (mol, basis_set, functional, omega_lo=None, omega_hi=None, omega_tol=None, max_evals=None))]
fn tune_omega(py: Python<'_>, mol: &PyMolecule, basis_set: &PyBasisSet, functional: &str,
              omega_lo: Option<f64>, omega_hi: Option<f64>, omega_tol: Option<f64>,
              max_evals: Option<usize>) -> PyResult<Py<pyo3::types::PyDict>> {
    use ferric_scf::omega_tuning::{tune_omega as tune, OmegaTuneConfig};
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).map_err(make_err)?;
    let ctx = ParallelContext::default();
    let cfg = OmegaTuneConfig {
        functional: functional.to_string(),
        omega_lo: omega_lo.unwrap_or(0.1),
        omega_hi: omega_hi.unwrap_or(1.0),
        omega_tol: omega_tol.unwrap_or(5e-3),
        max_evals: max_evals.unwrap_or(24),
        ..Default::default()
    };
    let r = tune(&ctx, &mol.inner, &prep, &bounds, &cfg).map_err(make_err)?;
    let d = pyo3::types::PyDict::new(py);
    d.set_item("omega", r.omega)?;
    d.set_item("j", r.j)?;
    d.set_item("converged", r.converged)?;
    let evals: Vec<(f64, f64, f64, f64)> =
        r.evals.iter().map(|e| (e.omega, e.eps_homo, e.ip_delta_scf, e.j)).collect();
    d.set_item("evals", evals)?;
    Ok(d.into())
}

// ── OO-RI-MP2 (orbital-optimized) ──

#[pyclass]
#[pyo3(name = "OoRiMp2Result")]
struct PyOoRiMp2Result {
    #[pyo3(get)] total_energy: f64,
    #[pyo3(get)] hf_energy: f64,
    #[pyo3(get)] mp2_corr: f64,
    #[pyo3(get)] converged: bool,
    #[pyo3(get)] iterations: usize,
    #[pyo3(get)] grad_norm: f64,
}

#[pymethods]
impl PyOoRiMp2Result {
    fn __repr__(&self) -> String {
        format!(
            "OoRiMp2Result(total_energy={:.10}, converged={}, iterations={})",
            self.total_energy, self.converged, self.iterations,
        )
    }
    fn __str__(&self) -> String {
        format!(
            "OO-RI-MP2 Total Energy: {:.10} Ha (HF: {:.10}, corr: {:.10}, converged: {}, {} iterations, grad_norm: {:.2e})",
            self.total_energy, self.hf_energy, self.mp2_corr, self.converged, self.iterations, self.grad_norm,
        )
    }
}

/// Orbital-optimized RI-MP2: jointly minimizes E_HF + E_MP2 over orbital
/// rotations (level-shifted approximate Newton + DIIS + Cayley rotation).
/// Starts from a converged RHF reference (same convention as `run_rimp2`).
/// `max_iter`/`grad_conv`/`level_shift`/`diis_size` control the orbital
/// rotation loop; the rest of `OoRiMp2Config` (energy_conv, step_size,
/// use_diis) stays at its library default, matching the CLI's `oo-rimp2`
/// arm which only threads `frozen_core` + `memory_budget_bytes` through.
#[pyfunction]
#[pyo3(signature = (
    mol, basis_set, auxbasis, frozen_core=None, k_builder=None,
    max_iter=None, grad_conv=None, level_shift=None, diis_size=None,
    memory_budget_gb=None,
))]
#[allow(clippy::too_many_arguments)]
fn run_oo_rimp2(mol: &PyMolecule, basis_set: &PyBasisSet, auxbasis: &PyBasisSet,
                frozen_core: Option<usize>, k_builder: Option<&str>,
                max_iter: Option<usize>, grad_conv: Option<f64>,
                level_shift: Option<f64>, diis_size: Option<usize>,
                memory_budget_gb: Option<f64>) -> PyResult<PyOoRiMp2Result> {
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    let dfbs = PreparedBasis::new(&mol.inner, &auxbasis.inner).map_err(make_err)?;
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).map_err(make_err)?;
    let ctx = ParallelContext::default();
    let rhf = solve_rhf(&ctx, &mol.inner, &prep, op, &bounds, &rhf_config(k_builder)).map_err(make_err)?;
    if !rhf.converged {
        return Err(make_err(ferric_core::FerricError::ScfConvergence { iterations: rhf.iterations, last_energy: rhf.energy }));
    }
    // Orbital-rotation loop knobs default to OoRiMp2Config::default() (same
    // library default the CLI's oo-rimp2 arm uses); frozen_core/memory_budget
    // follow the run_rimp2 convention.
    let default_cfg = OoRiMp2Config::default();
    let cfg = OoRiMp2Config {
        max_iter: max_iter.unwrap_or(default_cfg.max_iter),
        grad_conv: grad_conv.unwrap_or(default_cfg.grad_conv),
        level_shift: level_shift.unwrap_or(default_cfg.level_shift),
        diis_size: diis_size.unwrap_or(default_cfg.diis_size),
        frozen_core: frozen_core.unwrap_or(0),
        memory_budget_bytes: budget_bytes_from_gb(memory_budget_gb),
        ..default_cfg
    };
    // `run_oo_rimp2` has no point_charges/external_field kwargs today (unlike
    // run_rhf/run_uhf/run_optimize) — this Python entry point never builds an
    // ExternalPotential, so `ext` is always None here. Adding QM/MM support to
    // this binding is out of Lane F1's scope; this just keeps behavior
    // unchanged after `oo_ri_mp2`'s hcore-bug fix.
    let r = oo_ri_mp2(&mol.inner, &prep, &dfbs, op, &bounds, &rhf, &cfg, None).map_err(make_err)?;
    Ok(PyOoRiMp2Result {
        total_energy: r.total_energy,
        hf_energy: r.hf_energy,
        mp2_corr: r.mp2_corr,
        converged: r.converged,
        iterations: r.iterations,
        grad_norm: r.grad_norm,
    })
}

// ── MP3 (spin-orbital, via einsum!) ──

#[pyclass]
#[pyo3(name = "Mp3Result")]
struct PyMp3Result {
    #[pyo3(get)] e_hf: f64,
    #[pyo3(get)] e_mp2: f64,
    #[pyo3(get)] e_mp3: f64,
    #[pyo3(get)] e_corr: f64,
    #[pyo3(get)] e_total: f64,
}

#[pymethods]
impl PyMp3Result {
    fn __repr__(&self) -> String {
        format!("Mp3Result(e_total={:.10}, e_corr={:.10})", self.e_total, self.e_corr)
    }
    fn __str__(&self) -> String {
        format!(
            "MP3 Total Energy: {:.10} Ha (HF: {:.10}, MP2: {:.10}, MP3: {:.10})",
            self.e_total, self.e_hf, self.e_mp2, self.e_mp3,
        )
    }
}

#[pyfunction]
#[pyo3(signature = (mol, basis_set, auxbasis, frozen_core=None, k_builder=None))]
fn run_mp3(mol: &PyMolecule, basis_set: &PyBasisSet, auxbasis: &PyBasisSet,
           frozen_core: Option<usize>, k_builder: Option<&str>) -> PyResult<PyMp3Result> {
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    let dfbs = PreparedBasis::new(&mol.inner, &auxbasis.inner).map_err(make_err)?;
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).map_err(make_err)?;
    let ctx = ParallelContext::default();
    let rhf = solve_rhf(&ctx, &mol.inner, &prep, op, &bounds, &rhf_config(k_builder)).map_err(make_err)?;
    if !rhf.converged {
        return Err(make_err(ferric_core::FerricError::ScfConvergence { iterations: rhf.iterations, last_energy: rhf.energy }));
    }
    // mp3_energy has no memory-budget parameter (unlike ri_mp2/laplace_ri_mp2);
    // its internal VVVV size guard resolves the budget itself via
    // ferric_core::memory::resolve_budget_bytes(None) (env/auto-detect only).
    let mp3 = mp3_energy(&mol.inner, &prep, &dfbs, op, &rhf, frozen_core.unwrap_or(0)).map_err(make_err)?;
    Ok(PyMp3Result { e_hf: mp3.e_hf, e_mp2: mp3.e_mp2, e_mp3: mp3.e_mp3, e_corr: mp3.e_corr, e_total: mp3.e_total })
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

#[pymethods]
impl PyLaplaceMp2Result {
    fn __repr__(&self) -> String {
        format!("LaplaceMp2Result(total_energy={:.10}, mp2_corr={:.10})", self.total_energy, self.mp2_corr)
    }
    fn __str__(&self) -> String {
        format!(
            "Laplace RI-MP2 Total Energy: {:.10} Ha (corr: {:.10}, OS: {:.10}, SS: {:.10})",
            self.total_energy, self.mp2_corr, self.e_os, self.e_ss,
        )
    }
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

// ── Laplace SOS-MP2 ──

#[pyclass]
#[pyo3(name = "SosMp2Result")]
struct PySosMp2Result {
    #[pyo3(get)] total_energy: f64,
    #[pyo3(get)] rhf_energy: f64,
    /// The SCALED correlation energy, `c_os * e_os`.
    #[pyo3(get)] sos_corr: f64,
    /// The UNSCALED opposite-spin correlation energy. Directly comparable
    /// against `run_rimp2(..)`'s opposite-spin component.
    #[pyo3(get)] e_os: f64,
    /// The `c_os` actually applied, echoed for provenance.
    #[pyo3(get)] c_os: f64,
    /// Quadrature points actually used.
    #[pyo3(get)] n_quad: usize,
    /// `"mo"` or `"ao"`, echoed so a caller can record which algebra ran.
    #[pyo3(get)] formulation: String,
}

#[pymethods]
impl PySosMp2Result {
    fn __repr__(&self) -> String {
        format!(
            "SosMp2Result(total_energy={:.10}, sos_corr={:.10}, formulation=\"{}\")",
            self.total_energy, self.sos_corr, self.formulation,
        )
    }
    fn __str__(&self) -> String {
        format!(
            "SOS-MP2 Total Energy: {:.10} Ha (RHF: {:.10}, SOS corr: {:.10}, c_os={}, n_quad={}, formulation={})",
            self.total_energy, self.rhf_energy, self.sos_corr, self.c_os, self.n_quad, self.formulation,
        )
    }
}

/// Laplace-transform SOS-MP2: `E = c_os * E_OS`.
///
/// `formulation` selects the algebra:
///   - `"mo"` (default) — τ-weighted `(P|ia)` amplitudes.
///   - `"ao"` — occupied/virtual pseudo-densities. EXACT: same quantity as
///     `"mo"`, agreeing to round-off. Dense here, so not a scaling win.
///   - `"ao-sparse"` — Boys-localized, domain-restricted AO. The one
///     APPROXIMATE variant: it discards AO pairs outside every orbital domain,
///     converging to `"ao"` as `domain_cutoff_bohr` grows. Requires that
///     argument; passing it with `"mo"`/`"ao"` raises rather than being ignored.
///
/// An unrecognized value raises rather than silently running the default.
///
/// There is deliberately no `c_ss`: SOS-MP2 *is* the `c_ss = 0` limit, which is
/// exactly what lets the Laplace denominator factorize. `c_os = 1.0` recovers
/// the bare opposite-spin MP2 energy (the tests' hard internal reference).
///
/// `memory_budget_gb` caps the resident 3-index tensors, same meaning as
/// elsewhere in this module; the AO path fails fast with a clear error rather
/// than overshooting it.
#[pyfunction]
#[pyo3(signature = (mol, basis_set, auxbasis, c_os=None, n_quad=None, frozen_core=None,
                    formulation=None, domain_cutoff_bohr=None, k_builder=None,
                    memory_budget_gb=None))]
#[allow(clippy::too_many_arguments)]
fn run_laplace_sos_mp2(mol: &PyMolecule, basis_set: &PyBasisSet, auxbasis: &PyBasisSet,
                       c_os: Option<f64>, n_quad: Option<usize>, frozen_core: Option<usize>,
                       formulation: Option<&str>, domain_cutoff_bohr: Option<f64>,
                       k_builder: Option<&str>,
                       memory_budget_gb: Option<f64>) -> PyResult<PySosMp2Result> {
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    let dfbs = PreparedBasis::new(&mol.inner, &auxbasis.inner).map_err(make_err)?;
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).map_err(make_err)?;
    let ctx = ParallelContext::default();
    let rhf = solve_rhf(&ctx, &mol.inner, &prep, op, &bounds, &rhf_config(k_builder)).map_err(make_err)?;
    if !rhf.converged {
        return Err(make_err(ferric_core::FerricError::ScfConvergence { iterations: rhf.iterations, last_energy: rhf.energy }));
    }
    let form = SosFormulation::parse_config_str(formulation, domain_cutoff_bohr)
        .map_err(make_err)?;
    let cfg = SosMp2Config {
        c_os: c_os.unwrap_or(1.3),
        frozen_core: frozen_core.unwrap_or(0),
        n_quad: n_quad.unwrap_or(7),
        memory_budget_bytes: budget_bytes_from_gb(memory_budget_gb),
        domain_cutoff_bohr,
    };
    let r = laplace_sos_mp2(&mol.inner, &prep, &dfbs, op, &rhf, &cfg, form).map_err(make_err)?;
    Ok(PySosMp2Result {
        total_energy: r.total_energy,
        rhf_energy: rhf.energy,
        sos_corr: r.sos_corr,
        e_os: r.e_os,
        c_os: r.c_os,
        n_quad: r.n_quad,
        formulation: match form {
            SosFormulation::Mo => "mo".to_string(),
            SosFormulation::Ao => "ao".to_string(),
            SosFormulation::AoSparse(_) => "ao-sparse".to_string(),
        },
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

#[pymethods]
impl PyAttenuatedMp2Result {
    fn __repr__(&self) -> String {
        format!("AttenuatedMp2Result(total_energy={:.10}, mp2_corr={:.10})", self.total_energy, self.mp2_corr)
    }
    fn __str__(&self) -> String {
        format!(
            "Attenuated RI-MP2 Total Energy: {:.10} Ha (RHF: {:.10}, corr: {:.10}, OS: {:.10}, SS: {:.10})",
            self.total_energy, self.rhf_energy, self.mp2_corr, self.e_os, self.e_ss,
        )
    }
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
                      &RiMp2Config { frozen_core: frozen_core.unwrap_or(0), memory_budget_bytes: budget_bytes_from_gb(memory_budget_gb), ..Default::default() }).map_err(make_err)?;
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

#[pymethods]
impl PyScsMp2Result {
    fn __repr__(&self) -> String {
        format!("ScsMp2Result(total_energy={:.10}, scs_corr={:.10})", self.total_energy, self.scs_corr)
    }
    fn __str__(&self) -> String {
        format!(
            "SCS-MP2 Total Energy: {:.10} Ha (RHF: {:.10}, SCS corr: {:.10}, OS: {:.10}, SS: {:.10})",
            self.total_energy, self.rhf_energy, self.scs_corr, self.e_os, self.e_ss,
        )
    }
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

// ── MP2-V (attenuated MP2 + Eq.-11-damped VV10, JCTC 11, 4159 (2015)) ──

#[pyclass]
#[pyo3(name = "Mp2VResult")]
struct PyMp2VResult {
    /// E_HF + E_c^attMP2 + E_nl^VV10.
    #[pyo3(get)] total_energy: f64,
    #[pyo3(get)] rhf_energy: f64,
    /// Attenuated MP2 correlation energy (terfc/erfc at r0, optionally
    /// decoupled omega).
    #[pyo3(get)] att_mp2_corr: f64,
    /// VV10 nonlocal correlation, damped per `vv10_damping` (Eq. 11 by
    /// default). NOTE: the damping makes this LESS negative, not more — the
    /// dispersion it supplies is a difference effect (dimer minus monomers).
    #[pyo3(get)] vv10_e_nl: f64,
    #[pyo3(get)] e_os: f64,
    #[pyo3(get)] e_ss: f64,
    /// Grid points in the VV10 nonlocal integration (tells a suspiciously
    /// small E_nl from a suspiciously small grid).
    #[pyo3(get)] n_nlc_points: usize,
}

#[pymethods]
impl PyMp2VResult {
    fn __repr__(&self) -> String {
        format!(
            "Mp2VResult(total_energy={:.10}, att_mp2_corr={:.10}, vv10_e_nl={:.10})",
            self.total_energy, self.att_mp2_corr, self.vv10_e_nl,
        )
    }
    fn __str__(&self) -> String {
        format!(
            "MP2-V Total Energy: {:.10} Ha (RHF: {:.10}, att-MP2 corr: {:.10}, VV10 E_nl: {:.10})",
            self.total_energy, self.rhf_energy, self.att_mp2_corr, self.vv10_e_nl,
        )
    }
}

/// MP2-V (Goldey/Belzunces/Head-Gordon, JCTC 11, 4159 (2015)):
/// E = E_HF + E_c^attMP2(r0[, omega]) + E_nl^VV10 damped by 1 − terfc(R; r0[, omega])².
///
/// Closed-shell RHF references only (the CLI `kind = "mp2-v"` also handles
/// open shell). Units at this boundary: `r0` in Å (default 1.00, the published
/// value), `omega` in Å⁻¹ (default None = the Dutoi curvature link
/// ω = 1/(r0√2), the published method; setting it decouples the terfc seam
/// sharpness and reaches BOTH halves of Eq. 11 in lockstep — unparameterized
/// until b is refit). `attenuator`: "terfc" (published; needs
/// FERRIC_TERF_TABLE_DIR) or "erfc" (table-free control; rejects `omega`).
/// `vv10_damping`: "terfc" (published Eq. 11) or "none" (bare VV10 —
/// double-counts short range, measurement-only). b defaults 11.0, c 0.0089.
#[pyfunction]
#[pyo3(signature = (mol, basis_set, auxbasis, r0=None, b=None, c=None, omega=None, attenuator=None, vv10_damping=None, frozen_core=None, k_builder=None, memory_budget_gb=None))]
#[allow(clippy::too_many_arguments)]
fn run_mp2_v(mol: &PyMolecule, basis_set: &PyBasisSet, auxbasis: &PyBasisSet,
             r0: Option<f64>, b: Option<f64>, c: Option<f64>, omega: Option<f64>,
             attenuator: Option<&str>, vv10_damping: Option<&str>,
             frozen_core: Option<usize>, k_builder: Option<&str>,
             memory_budget_gb: Option<f64>) -> PyResult<PyMp2VResult> {
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    let dfbs = PreparedBasis::new(&mol.inner, &auxbasis.inner).map_err(make_err)?;
    let coul = Operator::coulomb();
    let bounds = SchwarzBounds::compute(coul, &prep).map_err(make_err)?;
    let ctx = ParallelContext::default();
    let rhf = solve_rhf(&ctx, &mol.inner, &prep, coul, &bounds, &rhf_config(k_builder)).map_err(make_err)?;
    if !rhf.converged {
        return Err(make_err(ferric_core::FerricError::ScfConvergence { iterations: rhf.iterations, last_energy: rhf.energy }));
    }

    // Start from the published MP2-V(terfc, aTZ) parameterization and override
    // only what the caller set (same construction order as the CLI's
    // build_att_vv10_config: damping BEFORE r0, so from_r0_angstrom sees the
    // final damping variant and keeps Eq. 11's two halves in lockstep).
    let mut cfg = AttVv10Config::mp2_v_terfc_atz();
    cfg.attenuator = match attenuator.unwrap_or("terfc") {
        "terfc" => AttVv10Attenuator::Terfc,
        "erfc" => AttVv10Attenuator::Erfc,
        other => return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown attenuator \"{other}\"; expected \"terfc\" (published) or \"erfc\" (control)"
        ))),
    };
    cfg.vv10_damping = match vv10_damping.unwrap_or("terfc") {
        "terfc" => ferric_dft::vv10::Vv10Damping::Terfc {
            r0_bohr: cfg.r0_bohr,
            // Seam sharpness derives from `cfg.omega` at evaluation time
            // (effective_vv10_damping) — never duplicated here.
            omega_bohr_inv: None,
        },
        "none" => ferric_dft::vv10::Vv10Damping::None,
        other => return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown vv10_damping \"{other}\"; expected \"terfc\" (published, Eq. 11) or \"none\" (bare VV10, double-counts short range)"
        ))),
    };
    if let Some(r0_ang) = r0 {
        if !r0_ang.is_finite() || r0_ang <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "r0 must be finite and > 0 (got {r0_ang} A)"
            )));
        }
        cfg = cfg.from_r0_angstrom(r0_ang);
    }
    if let Some(b) = b { cfg.vv10.b = b; }
    if let Some(c) = c { cfg.vv10.c = c; }
    // omega is supplied in Å⁻¹ (same boundary convention as run_rs_mp2_rpa's
    // terf_omega); Bohr⁻¹ internally. Validation (finite/positive, terfc-only)
    // is the library's, so the error text cannot drift from the CLI's.
    cfg.omega = omega.map(|w| w * ferric_mp2::attenuated::BOHR_INV_PER_ANG_INV);
    cfg.frozen_core = frozen_core.unwrap_or(0);
    cfg.memory_budget_bytes = budget_bytes_from_gb(memory_budget_gb);

    let r = att_mp2_vv10(&mol.inner, &prep, &basis_set.inner, &dfbs, &rhf, &cfg)
        .map_err(make_err)?;
    let (e_os, e_ss) = match &r.spin_components {
        AttVv10SpinComponents::Restricted(s) => (s.e_os, s.e_ss),
        AttVv10SpinComponents::Unrestricted(_) => unreachable!("closed-shell entry point"),
    };
    Ok(PyMp2VResult {
        total_energy: r.total,
        rhf_energy: r.e_hf,
        att_mp2_corr: r.e_c_att_mp2,
        vv10_e_nl: r.e_nl_vv10,
        e_os,
        e_ss,
        n_nlc_points: r.n_nlc_points,
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

#[pymethods]
impl PyRsMp2RpaResult {
    fn __repr__(&self) -> String {
        format!(
            "RsMp2RpaResult(total_energy={:.10}, e_corr={:.10})",
            self.total_energy, self.e_corr,
        )
    }
    fn __str__(&self) -> String {
        format!(
            "RS-MP2-RPA Total Energy: {:.10} Ha (RHF: {:.10}, corr: {:.10}, SR-MP2: {:.10}, LR-MP2: {:.10})",
            self.total_energy, self.rhf_energy, self.e_corr, self.e_sr_mp2, self.e_lr_mp2,
        )
    }
}

#[pyfunction]
#[pyo3(signature = (mol, basis_set, auxbasis, omega=None, frozen_core=None, k_builder=None, formulation=None, attenuator=None, r0=None, terf_omega=None, memory_budget_gb=None))]
#[allow(clippy::too_many_arguments)]
fn run_rs_mp2_rpa(mol: &PyMolecule, basis_set: &PyBasisSet, auxbasis: &PyBasisSet,
                  omega: Option<f64>, frozen_core: Option<usize>,
                  k_builder: Option<&str>,
                  formulation: Option<&str>,
                  attenuator: Option<&str>, r0: Option<f64>,
                  terf_omega: Option<f64>,
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
    // attenuator: "erf" (default, ω in Å⁻¹) or "terf" (r0 in Å, ω derived).
    let atten = match attenuator.unwrap_or("erf") {
        "erf" => ferric_rpa::rs_mp2_rpa::Attenuator::Erf,
        "terf" => ferric_rpa::rs_mp2_rpa::Attenuator::Terf,
        other => return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown attenuator \"{other}\"; expected \"erf\" or \"terf\""
        ))),
    };
    // omega is supplied in Å⁻¹; convert to Bohr⁻¹ for the operator. r0 is
    // supplied in Å (2026-07-21: fixed from Bohr, matching r0_bonded/
    // r0_nonbonded's existing Å convention elsewhere in this file); convert
    // to Bohr for RsMp2RpaConfig, which stays Bohr-native.
    const ANG2BOHR_R0: f64 = 1.8897259886;
    let mut cfg = ferric_rpa::RsMp2RpaConfig {
        omega: omega.unwrap_or(0.420) * ferric_mp2::attenuated::BOHR_INV_PER_ANG_INV,
        attenuator: atten,
        r0: r0.unwrap_or(3.18 / ANG2BOHR_R0) * ANG2BOHR_R0,
        // terf_omega in Å⁻¹ (same convention as omega); None keeps the
        // Dutoi curvature link ω = 1/(r0·√2). Terf-only.
        terf_omega: terf_omega.map(|w| w * ferric_mp2::attenuated::BOHR_INV_PER_ANG_INV),
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
    scf_data: ScfResult,
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
    fn __repr__(&self) -> String {
        format!("DftResult(total_energy={:.10}, converged={})", self.total_energy, self.converged)
    }
    fn __str__(&self) -> String {
        format!("KS-DFT Energy: {:.10} Ha (converged: {})", self.total_energy, self.converged)
    }
}

// ---------------------------------------------------------------------------
// MP2-based double hybrids (B2PLYP, DSD-PBEP86)
// ---------------------------------------------------------------------------

#[pyclass]
struct PyDoubleHybridResult {
    #[pyo3(get)] total_energy: f64,
    #[pyo3(get)] e_ks: f64,
    #[pyo3(get)] e_corr_scaled: f64,
    #[pyo3(get)] e_os: f64,
    #[pyo3(get)] e_ss: f64,
    #[pyo3(get)] c_os: f64,
    #[pyo3(get)] c_ss: f64,
}

#[pymethods]
impl PyDoubleHybridResult {
    fn __repr__(&self) -> String {
        format!("DoubleHybridResult(total_energy={:.10}, e_ks={:.10}, e_corr={:.10})",
            self.total_energy, self.e_ks, self.e_corr_scaled)
    }
    fn __str__(&self) -> String {
        format!("Double Hybrid Energy: {:.10} Ha (KS: {:.10}, scaled MP2: {:.10})",
            self.total_energy, self.e_ks, self.e_corr_scaled)
    }
}

/// Run an MP2-based double hybrid (B2PLYP or DSD-PBEP86).
///
/// `kind`: "b2plyp" or "dsd-pbep86".
/// Converges the appropriate DFT reference, then adds scaled MP2 correlation.
#[pyfunction]
#[pyo3(signature = (mol, basis_set, auxbasis, kind="b2plyp", frozen_core=None, k_builder=None, memory_budget_gb=None))]
fn run_double_hybrid(mol: &PyMolecule, basis_set: &PyBasisSet, auxbasis: &PyBasisSet,
                     kind: &str, frozen_core: Option<usize>,
                     k_builder: Option<&str>, memory_budget_gb: Option<f64>) -> PyResult<PyDoubleHybridResult> {
    let dh_kind = match kind.to_lowercase().replace(['-', '_'], "").as_str() {
        "b2plyp" => DoubleHybridKind::B2plyp,
        "dsdpbep86" => DoubleHybridKind::DsdPbep86,
        _ => return Err(pyo3::exceptions::PyValueError::new_err(
            format!("unknown double hybrid kind '{kind}'; expected 'b2plyp' or 'dsd-pbep86'")
        )),
    };

    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    let dfbs = PreparedBasis::new(&mol.inner, &auxbasis.inner).map_err(make_err)?;
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).map_err(make_err)?;
    let ctx = ParallelContext::default();

    let mut cfg = rhf_config(k_builder);
    cfg.xc = Some(dh_kind.xc_name().to_string());
    cfg.df_j_aux = Some("def2-universal-jkfit".to_string());
    cfg.df_k_aux = Some("def2-universal-jkfit".to_string());

    let ladder = ferric_scf::ladder::ksdft_ladder(&cfg);
    let lr = ferric_scf::ladder::solve_rhf_ladder(&ctx, &mol.inner, &prep, op, &bounds, &ladder)
        .map_err(make_err)?;
    let ks = lr.result;
    if !ks.converged {
        return Err(make_err(ferric_core::FerricError::ScfConvergence {
            iterations: ks.iterations, last_energy: ks.energy,
        }));
    }

    let mut mp2_cfg = dh_kind.mp2_config();
    if let Some(fc) = frozen_core { mp2_cfg.frozen_core = fc; }
    mp2_cfg.memory_budget_bytes = budget_bytes_from_gb(memory_budget_gb);

    let r = mp2_double_hybrid(&mol.inner, &prep, &dfbs, &ks, &mp2_cfg).map_err(make_err)?;

    Ok(PyDoubleHybridResult {
        total_energy: r.total_energy,
        e_ks: r.e_ks,
        e_corr_scaled: r.e_corr_scaled,
        e_os: r.spin_components.e_os,
        e_ss: r.spin_components.e_ss,
        c_os: r.c_os,
        c_ss: r.c_ss,
    })
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
        density_data: rhf.density_total.clone(),
        gradient_data,
        scf_data: rhf,
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

#[pymethods]
impl PyCcResult {
    fn __repr__(&self) -> String {
        match self.t_correction {
            Some(t) => format!(
                "CcResult(correlation_energy={:.10}, t_correction={:.10})",
                self.correlation_energy, t,
            ),
            None => format!(
                "CcResult(correlation_energy={:.10})",
                self.correlation_energy,
            ),
        }
    }
    fn __str__(&self) -> String {
        match self.t_correction {
            Some(t) => format!(
                "CC Correlation Energy: {:.10} Ha, (T) correction: {:.10} Ha, total corr+T: {:.10} Ha",
                self.correlation_energy, t, self.correlation_energy + t,
            ),
            None => format!(
                "CC Correlation Energy: {:.10} Ha",
                self.correlation_energy,
            ),
        }
    }
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
    // `solve_rhf` always yields a restricted reference, so this is the
    // spin-adapted path unconditionally: same CCSD energy, but over spatial
    // orbitals (no/nv) rather than spin orbitals (2no/2nv), so the O(N^6) VVVV
    // block is 16x smaller. Measured on water/aug-cc-pVDZ: 24.6 s -> 1.10 s
    // (22x) and peak RSS 1.82 GB -> 0.48 GB.
    //
    // NOTE `run_ccsd_t` below deliberately does NOT switch — see its comment.
    let r = run_ccsd_cs_inner(&mol.inner, &prep, &dfbs, op, &rhf, &cfg).map_err(make_err)?;
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
    // `solve_rhf` always yields a restricted reference, so both halves take the
    // spin-adapted path: `ccsd_closed_shell` for the amplitudes and
    // `ccsd_t_closed_shell` for the triples. Both work over spatial orbitals
    // (no/nv) instead of spin orbitals (2no/2nv).
    //
    // The amplitudes flow straight through — no `expand_amplitudes_to_spin_orbital`
    // round-trip, since the spin-adapted (T) consumes the spatial convention
    // natively (and hard-rejects spin-orbital amplitudes with a typed error).
    //
    // Measured on water/aug-cc-pVDZ: CCSD 24.8 -> 0.70 s and (T) 1.87 -> 0.14 s,
    // i.e. ~26.6 s -> under a second for the pair. The two (T) implementations
    // agree to 1.9e-16 when fed identical amplitudes, and to 7.8e-10..6.9e-7
    // against PySCF's `ccsd_t()` — inside the ~5e-6 Ha RI floor.
    let r_cs = run_ccsd_cs_inner(&mol.inner, &prep, &dfbs, op, &rhf, &cfg).map_err(make_err)?;
    let e_t = run_ccsd_t_cs_inner(&mol.inner, &prep, &dfbs, op, &rhf, &r_cs, &cfg)
        .map_err(make_err)?;
    Ok(PyCcResult { correlation_energy: r_cs.correlation_energy, t_correction: Some(e_t) })
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
    fn __repr__(&self) -> String {
        format!(
            "PdepRpaResult(total_energy={:.10}, e_rpa={:.10}, n_eigenpotentials={})",
            self.total_energy, self.e_rpa, self.n_eigenpotentials,
        )
    }
    fn __str__(&self) -> String {
        format!(
            "PDEP-RPA Total Energy: {:.10} Ha (RHF: {:.10}, E_RPA: {:.10}, {} eigenpotentials, converged: {})",
            self.total_energy, self.rhf_energy, self.e_rpa, self.n_eigenpotentials, self.eigensolver_converged,
        )
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
        need_eigenvalues_freq: true,
        verbose: false,
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

// ── GW (closed-shell: G0W0 / COHSEX / evGW0 / evGW) ──
//
// `run_u_gw` is wired below; `bse.rs`'s `run_bse_tda` is wired further down
// (see `run_cis_tda`/`run_bse_c6`/`run_bse_c6_ks`'s still-open note there) —
// see docs/open-work-triage-2026-07-14-open.md #54.

#[pyclass]
#[pyo3(name = "GwResult")]
struct PyGwResult {
    #[pyo3(get)] ref_energy: f64,
    /// MO indices (absolute) for which QP energies were computed.
    #[pyo3(get)] mo_indices: Vec<usize>,
    /// Mean-field (input) orbital energies for those MOs, Ha.
    eps_mf: Vec<f64>,
    /// QP energies (final), Ha.
    eps_qp: Vec<f64>,
    /// Exchange self-energy Σ_x (diagonal MO), Ha.
    sigma_x: Vec<f64>,
    /// Correlation self-energy Σ_c at the converged QP energy, Ha.
    sigma_c: Vec<f64>,
    /// Z-factor (renormalization), dimensionless.
    z_factor: Vec<f64>,
    /// Per-state QP Newton-solve convergence flag, aligned with `mo_indices`.
    /// `false` ⇒ that MO's `eps_qp`/`sigma_c`/`z_factor` is the Newton
    /// solver's last iterate, not a converged root. Always all-`true` for
    /// COHSEX (closed-form).
    #[pyo3(get)] qp_converged: Vec<bool>,
    /// evGW/evGW0 outer eigenvalue self-consistency iteration count (0 for
    /// G0W0/COHSEX).
    #[pyo3(get)] n_ev_iter: usize,
    /// Whether the evGW/evGW0 outer loop met `ev_conv_thresh` within
    /// `max_ev_iter`. Always `true` for G0W0/COHSEX.
    #[pyo3(get)] outer_converged: bool,
}

#[pymethods]
impl PyGwResult {
    #[getter]
    fn eps_mf<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.eps_mf)
    }
    #[getter]
    fn eps_qp<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.eps_qp)
    }
    #[getter]
    fn sigma_x<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.sigma_x)
    }
    #[getter]
    fn sigma_c<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.sigma_c)
    }
    #[getter]
    fn z_factor<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.z_factor)
    }
    fn __repr__(&self) -> String {
        format!(
            "GwResult(ref_energy={:.10}, n_qp={}, outer_converged={})",
            self.ref_energy, self.mo_indices.len(), self.outer_converged,
        )
    }
    fn __str__(&self) -> String {
        let n_unconverged = self.qp_converged.iter().filter(|&&c| !c).count();
        format!(
            "GW: ref_energy={:.10} Ha, {} QP states, {} unconverged QP, outer_converged={}, {} ev iterations",
            self.ref_energy, self.mo_indices.len(), n_unconverged, self.outer_converged, self.n_ev_iter,
        )
    }
}

/// Closed-shell G0W0/COHSEX/evGW0/evGW on an RHF or RKS reference.
///
/// `method`: "g0w0" (default) | "cohsex" | "evgw0" | "evgw" (case-insensitive;
/// unknown values are a hard `ValueError`, matching this repo's strict-config
/// convention elsewhere). `qp_mos`: optional `(lo, hi)` absolute-MO range;
/// unset uses the library default `{HOMO-2..LUMO+2}`. `xc`: `None` (default)
/// runs an HF reference; setting it (e.g. `"pbe"`) runs the closed-shell
/// KS-DFT solver first and folds Σx−vxc into the QP self-consistency (see
/// `ferric_gw::vxc_mo::vxc_diagonal_mo`).
#[pyfunction]
#[pyo3(signature = (
    mol, basis_set, auxbasis,
    method=None, xc=None, qp_mos=None,
    max_ev_iter=None, ev_conv_thresh=None, pade_npts=None, qp_newton_damp=None,
    frozen_core=None, n_quad=None, quadrature=None, u0=None,
    trunc_thresh=None, eigensolver_conv_thresh=None,
    k_builder=None, chi0_sparsity=None, memory_budget_gb=None,
))]
#[allow(clippy::too_many_arguments)]
fn run_gw(
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    auxbasis: &PyBasisSet,
    method: Option<&str>,
    xc: Option<&str>,
    qp_mos: Option<(usize, usize)>,
    max_ev_iter: Option<usize>,
    ev_conv_thresh: Option<f64>,
    pade_npts: Option<usize>,
    qp_newton_damp: Option<f64>,
    frozen_core: Option<usize>,
    n_quad: Option<usize>,
    quadrature: Option<&str>,
    u0: Option<f64>,
    trunc_thresh: Option<f64>,
    eigensolver_conv_thresh: Option<f64>,
    k_builder: Option<&str>,
    chi0_sparsity: Option<&str>,
    memory_budget_gb: Option<f64>,
) -> PyResult<PyGwResult> {
    use ferric_gw::{run_gw as run_gw_inner, vxc_mo::vxc_diagonal_mo, GwConfig, GwMethod};
    use ferric_rpa::config::{QuadratureConfig, QuadratureScheme, SternheimerConfig};
    use ferric_rpa::PdepRpaConfig;

    let gw_method = match method.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
        None | Some("g0w0") => GwMethod::G0W0,
        Some("cohsex") => GwMethod::Cohsex,
        Some("evgw0") => GwMethod::EvGw0,
        Some("evgw") => GwMethod::EvGw,
        Some(other) => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "method: unknown value \"{other}\"; expected \"g0w0\", \"cohsex\", \"evgw0\", or \"evgw\""
            )));
        }
    };

    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    let dfbs = PreparedBasis::new(&mol.inner, &auxbasis.inner).map_err(make_err)?;
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).map_err(make_err)?;
    let ctx = ParallelContext::default();

    let mut cfg = rhf_config(k_builder);
    let vxc_diag = if let Some(xc_name) = xc {
        // KS reference: RI-J/RI-K default (matches run_dft/run_ksdft), run
        // through the level-shift ladder for the same DIIS-oscillation
        // fallback KS-DFT gets elsewhere.
        cfg.xc = Some(xc_name.to_string());
        cfg.df_j_aux = Some("def2-universal-jkfit".to_string());
        cfg.df_k_aux = Some("def2-universal-jkfit".to_string());
        let ladder = ferric_scf::ladder::ksdft_ladder(&cfg);
        let lr = ferric_scf::ladder::solve_rhf_ladder(&ctx, &mol.inner, &prep, op, &bounds, &ladder)
            .map_err(make_err)?;
        let scf = lr.result;
        if !scf.converged {
            return Err(make_err(ferric_core::FerricError::ScfConvergence {
                iterations: scf.iterations, last_energy: scf.energy,
            }));
        }
        let (diag, _beta) = vxc_diagonal_mo(&mol.inner, &basis_set.inner, xc_name, &scf).map_err(make_err)?;
        (scf, Some(diag))
    } else {
        let scf = solve_rhf(&ctx, &mol.inner, &prep, op, &bounds, &cfg).map_err(make_err)?;
        if !scf.converged {
            return Err(make_err(ferric_core::FerricError::ScfConvergence {
                iterations: scf.iterations, last_energy: scf.energy,
            }));
        }
        (scf, None)
    };
    let (scf, vxc_diag) = vxc_diag;

    let scheme = QuadratureScheme::parse_config_str(quadrature)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("quadrature: {e}")))?;
    let fc = frozen_core.unwrap_or(0);
    let pdep_cfg = PdepRpaConfig {
        frozen_core: fc,
        trunc_thresh: trunc_thresh.unwrap_or(1e-4),
        eigensolver_max_vecs: 0,
        eigensolver_conv_thresh: eigensolver_conv_thresh.unwrap_or(1e-6),
        quadrature: QuadratureConfig {
            scheme,
            n_points: n_quad.unwrap_or(20),
            u0: u0.unwrap_or(0.5),
        },
        sternheimer: SternheimerConfig::default(),
        run_diagnostics: false,
        eigensolver: ferric_rpa::Eigensolver::default(),
        chi0_backend: ferric_rpa::config::Chi0Backend::default(),
        chi0_sparsity: ferric_rpa::config::Chi0Sparsity::parse_config_str(chi0_sparsity)
            .map_err(make_err)?,
        memory_budget_bytes: budget_bytes_from_gb(memory_budget_gb),
        // run_gw forces this on internally; set explicitly for clarity too.
        need_inv_dielectric_freq: true,
        need_eigenvalues_freq: true,
        verbose: false,
    };
    let gw_cfg = GwConfig {
        method: gw_method,
        qp_mos: qp_mos.map(|(lo, hi)| lo..hi),
        max_ev_iter: max_ev_iter.unwrap_or(20),
        ev_conv_thresh: ev_conv_thresh.unwrap_or(1e-4),
        pade_npts: pade_npts.unwrap_or(0),
        qp_newton_damp: qp_newton_damp.unwrap_or(1.0),
        frozen_core: fc,
        memory_budget_bytes: budget_bytes_from_gb(memory_budget_gb),
        verbose: false,
    };

    let r = run_gw_inner(&mol.inner, &prep, &dfbs, op, &scf, &pdep_cfg, &gw_cfg, vxc_diag.as_ref())
        .map_err(make_err)?;
    if !r.outer_converged {
        eprintln!(
            "warning: {:?} eigenvalue self-consistency did NOT converge in {} \
             iterations (thresh {:.1e}); QP energies are the last sweep",
            gw_cfg.method, r.n_ev_iter, gw_cfg.ev_conv_thresh
        );
    }
    let bad: Vec<usize> = r.qp_converged.iter().enumerate()
        .filter(|(_, &c)| !c).map(|(i, _)| r.mo_indices[i]).collect();
    if !bad.is_empty() {
        eprintln!(
            "warning: QP Newton solve did not converge for MO(s) {bad:?}; \
             those QP energies are best-effort"
        );
    }
    Ok(PyGwResult {
        ref_energy: scf.energy,
        mo_indices: r.mo_indices,
        eps_mf: r.eps_mf.to_vec(),
        eps_qp: r.eps_qp.to_vec(),
        sigma_x: r.sigma_x.to_vec(),
        sigma_c: r.sigma_c.to_vec(),
        z_factor: r.z_factor.to_vec(),
        qp_converged: r.qp_converged,
        n_ev_iter: r.n_ev_iter,
        outer_converged: r.outer_converged,
    })
}

// ── U-GW (open-shell: U-G0W0 / U-COHSEX / U-evGW0 / U-evGW) ──
//
// BSE-TDA (`run_bse_tda`) is closed-shell only and wired further down
// (see the "── BSE-TDA ──" section) — see docs/open-work-triage-2026-07-14-open.md #54.

#[pyclass]
#[pyo3(name = "UGwResult")]
struct PyUGwResult {
    #[pyo3(get)] ref_energy: f64,
    /// MO indices (absolute) for which QP energies were computed, shared by
    /// both spin channels.
    #[pyo3(get)] mo_indices: Vec<usize>,
    eps_mf_a: Vec<f64>,
    eps_qp_a: Vec<f64>,
    sigma_x_a: Vec<f64>,
    sigma_c_a: Vec<f64>,
    z_factor_a: Vec<f64>,
    eps_mf_b: Vec<f64>,
    eps_qp_b: Vec<f64>,
    sigma_x_b: Vec<f64>,
    sigma_c_b: Vec<f64>,
    z_factor_b: Vec<f64>,
    /// Per-state QP Newton-solve convergence flags, aligned with `mo_indices`
    /// (see `PyGwResult::qp_converged` for the per-flag meaning). Always
    /// all-`true` for COHSEX.
    #[pyo3(get)] qp_converged_a: Vec<bool>,
    #[pyo3(get)] qp_converged_b: Vec<bool>,
    /// evGW/evGW0 outer eigenvalue self-consistency iteration count (0 for
    /// G0W0/COHSEX).
    #[pyo3(get)] n_ev_iter: usize,
    /// Whether the U-evGW/U-evGW0 outer loop met `ev_conv_thresh` within
    /// `max_ev_iter`. Always `true` for U-G0W0/U-COHSEX.
    #[pyo3(get)] outer_converged: bool,
}

#[pymethods]
impl PyUGwResult {
    #[getter]
    fn eps_mf_a<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.eps_mf_a)
    }
    #[getter]
    fn eps_qp_a<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.eps_qp_a)
    }
    #[getter]
    fn sigma_x_a<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.sigma_x_a)
    }
    #[getter]
    fn sigma_c_a<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.sigma_c_a)
    }
    #[getter]
    fn z_factor_a<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.z_factor_a)
    }
    #[getter]
    fn eps_mf_b<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.eps_mf_b)
    }
    #[getter]
    fn eps_qp_b<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.eps_qp_b)
    }
    #[getter]
    fn sigma_x_b<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.sigma_x_b)
    }
    #[getter]
    fn sigma_c_b<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.sigma_c_b)
    }
    #[getter]
    fn z_factor_b<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.z_factor_b)
    }
    fn __repr__(&self) -> String {
        format!(
            "UGwResult(ref_energy={:.10}, n_qp={}, outer_converged={})",
            self.ref_energy, self.mo_indices.len(), self.outer_converged,
        )
    }
    fn __str__(&self) -> String {
        let n_unconverged_a = self.qp_converged_a.iter().filter(|&&c| !c).count();
        let n_unconverged_b = self.qp_converged_b.iter().filter(|&&c| !c).count();
        format!(
            "U-GW: ref_energy={:.10} Ha, {} QP states, unconverged QP: {} alpha/{} beta, outer_converged={}, {} ev iterations",
            self.ref_energy, self.mo_indices.len(), n_unconverged_a, n_unconverged_b, self.outer_converged, self.n_ev_iter,
        )
    }
}

/// Open-shell U-G0W0/U-COHSEX/U-evGW0/U-evGW on a UHF/UKS or ROHF reference.
///
/// `reference`: "uhf" (default) | "rohf" (case-insensitive; unknown values
/// are a hard `ValueError`). `method`/`xc`/`qp_mos`/... mirror `run_gw`'s
/// kwarg shape exactly; `xc` set runs the open-shell KS-DFT ladder (UKS) and
/// applies the Σx−vxc correction per spin channel via
/// `UGwResult::apply_kohn_sham_correction` (run_u_gw itself doesn't thread
/// vxc_diag through — see its doc in `ferric_gw::run_u_gw`).
#[pyfunction]
#[pyo3(signature = (
    mol, basis_set, auxbasis,
    reference=None, method=None, xc=None, qp_mos=None,
    max_ev_iter=None, ev_conv_thresh=None, pade_npts=None, qp_newton_damp=None,
    frozen_core=None, n_quad=None, quadrature=None, u0=None,
    trunc_thresh=None, eigensolver_conv_thresh=None,
    k_builder=None, chi0_sparsity=None, memory_budget_gb=None,
))]
#[allow(clippy::too_many_arguments)]
fn run_u_gw(
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    auxbasis: &PyBasisSet,
    reference: Option<&str>,
    method: Option<&str>,
    xc: Option<&str>,
    qp_mos: Option<(usize, usize)>,
    max_ev_iter: Option<usize>,
    ev_conv_thresh: Option<f64>,
    pade_npts: Option<usize>,
    qp_newton_damp: Option<f64>,
    frozen_core: Option<usize>,
    n_quad: Option<usize>,
    quadrature: Option<&str>,
    u0: Option<f64>,
    trunc_thresh: Option<f64>,
    eigensolver_conv_thresh: Option<f64>,
    k_builder: Option<&str>,
    chi0_sparsity: Option<&str>,
    memory_budget_gb: Option<f64>,
) -> PyResult<PyUGwResult> {
    use ferric_gw::{run_u_gw as run_u_gw_inner, vxc_mo::vxc_diagonal_mo, GwConfig, GwMethod};
    use ferric_rpa::config::{QuadratureConfig, QuadratureScheme, SternheimerConfig};
    use ferric_rpa::PdepRpaConfig;
    use ferric_scf::rohf::solve_rohf;

    let gw_method = match method.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
        None | Some("g0w0") => GwMethod::G0W0,
        Some("cohsex") => GwMethod::Cohsex,
        Some("evgw0") => GwMethod::EvGw0,
        Some("evgw") => GwMethod::EvGw,
        Some(other) => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "method: unknown value \"{other}\"; expected \"g0w0\", \"cohsex\", \"evgw0\", or \"evgw\""
            )));
        }
    };
    let reference = match reference.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
        None | Some("uhf") => "uhf",
        Some("rohf") => "rohf",
        Some(other) => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "reference: unknown value \"{other}\"; expected \"uhf\" or \"rohf\""
            )));
        }
    };

    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    let dfbs = PreparedBasis::new(&mol.inner, &auxbasis.inner).map_err(make_err)?;
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).map_err(make_err)?;
    let ctx = ParallelContext::default();

    let mut cfg = rhf_config(k_builder);
    // MOM after 5 DIIS iters prevents orbital reordering on open-shell atoms
    // (same precedent as the CLI's "pdep-rpa"/"gw" open-shell dispatch).
    cfg.mom_after_iter = 5;
    let vxc_diag = if let Some(xc_name) = xc {
        cfg.xc = Some(xc_name.to_string());
        cfg.df_j_aux = Some("def2-universal-jkfit".to_string());
        cfg.df_k_aux = Some("def2-universal-jkfit".to_string());
        let scf = if reference == "rohf" {
            solve_rohf(&ctx, &mol.inner, &prep, op, &bounds, &cfg).map_err(make_err)?
        } else {
            solve_uhf(&ctx, &mol.inner, &prep, &bounds, &cfg).map_err(make_err)?
        };
        if !scf.converged {
            return Err(make_err(ferric_core::FerricError::ScfConvergence {
                iterations: scf.iterations, last_energy: scf.energy,
            }));
        }
        let (diag_a, diag_b) = vxc_diagonal_mo(&mol.inner, &basis_set.inner, xc_name, &scf).map_err(make_err)?;
        (scf, Some((diag_a, diag_b)))
    } else {
        let scf = if reference == "rohf" {
            solve_rohf(&ctx, &mol.inner, &prep, op, &bounds, &cfg).map_err(make_err)?
        } else {
            solve_uhf(&ctx, &mol.inner, &prep, &bounds, &cfg).map_err(make_err)?
        };
        if !scf.converged {
            return Err(make_err(ferric_core::FerricError::ScfConvergence {
                iterations: scf.iterations, last_energy: scf.energy,
            }));
        }
        (scf, None)
    };
    let (scf, vxc_diag) = vxc_diag;

    let scheme = QuadratureScheme::parse_config_str(quadrature)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("quadrature: {e}")))?;
    let fc = frozen_core.unwrap_or(0);
    let pdep_cfg = PdepRpaConfig {
        frozen_core: fc,
        trunc_thresh: trunc_thresh.unwrap_or(1e-4),
        eigensolver_max_vecs: 0,
        eigensolver_conv_thresh: eigensolver_conv_thresh.unwrap_or(1e-6),
        quadrature: QuadratureConfig {
            scheme,
            n_points: n_quad.unwrap_or(20),
            u0: u0.unwrap_or(0.5),
        },
        sternheimer: SternheimerConfig::default(),
        run_diagnostics: false,
        eigensolver: ferric_rpa::Eigensolver::default(),
        chi0_backend: ferric_rpa::config::Chi0Backend::default(),
        chi0_sparsity: ferric_rpa::config::Chi0Sparsity::parse_config_str(chi0_sparsity)
            .map_err(make_err)?,
        memory_budget_bytes: budget_bytes_from_gb(memory_budget_gb),
        // run_u_gw forces this on internally; set explicitly for clarity too.
        need_inv_dielectric_freq: true,
        need_eigenvalues_freq: true,
        verbose: false,
    };
    let gw_cfg = GwConfig {
        method: gw_method,
        qp_mos: qp_mos.map(|(lo, hi)| lo..hi),
        max_ev_iter: max_ev_iter.unwrap_or(20),
        ev_conv_thresh: ev_conv_thresh.unwrap_or(1e-4),
        pade_npts: pade_npts.unwrap_or(0),
        qp_newton_damp: qp_newton_damp.unwrap_or(1.0),
        frozen_core: fc,
        memory_budget_bytes: budget_bytes_from_gb(memory_budget_gb),
        verbose: false,
    };

    let mut r = run_u_gw_inner(&mol.inner, &prep, &dfbs, op, &scf, &pdep_cfg, &gw_cfg)
        .map_err(make_err)?;
    if let Some((diag_a, diag_b)) = vxc_diag.as_ref() {
        r.apply_kohn_sham_correction(diag_a, diag_b);
    }
    if !r.outer_converged {
        eprintln!(
            "warning: U-{:?} eigenvalue self-consistency did NOT converge in {} \
             iterations (thresh {:.1e}); QP energies are the last sweep",
            gw_cfg.method, r.n_ev_iter, gw_cfg.ev_conv_thresh
        );
    }
    for (spin_label, flags) in [("alpha", &r.qp_converged_a), ("beta", &r.qp_converged_b)] {
        let bad: Vec<usize> = flags.iter().enumerate()
            .filter(|(_, &c)| !c).map(|(i, _)| r.mo_indices[i]).collect();
        if !bad.is_empty() {
            eprintln!(
                "warning: QP Newton solve did not converge for {spin_label} MO(s) {bad:?}; \
                 those QP energies are best-effort"
            );
        }
    }
    Ok(PyUGwResult {
        ref_energy: scf.energy,
        mo_indices: r.mo_indices,
        eps_mf_a: r.eps_mf_a.to_vec(),
        eps_qp_a: r.eps_qp_a.to_vec(),
        sigma_x_a: r.sigma_x_a.to_vec(),
        sigma_c_a: r.sigma_c_a.to_vec(),
        z_factor_a: r.z_factor_a.to_vec(),
        eps_mf_b: r.eps_mf_b.to_vec(),
        eps_qp_b: r.eps_qp_b.to_vec(),
        sigma_x_b: r.sigma_x_b.to_vec(),
        sigma_c_b: r.sigma_c_b.to_vec(),
        z_factor_b: r.z_factor_b.to_vec(),
        qp_converged_a: r.qp_converged_a,
        qp_converged_b: r.qp_converged_b,
        n_ev_iter: r.n_ev_iter,
        outer_converged: r.outer_converged,
    })
}

// ── BSE-TDA (closed-shell singlet excitation energies on a G0W0@HF reference) ──
//
// `run_cis_tda`/`run_bse_c6`/`run_bse_c6_ks` are NOT wired here — see
// docs/open-work-triage-2026-07-14-open.md #54. `run_cis_tda` is a
// diagnostic-only assembly cross-check (not a production entry point, per
// its own doc comment in `ferric_gw::bse`); the C6/dispersion variants are
// lower-priority per the G9 task brief and still open.

#[pyclass]
#[pyo3(name = "BseResult")]
struct PyBseResult {
    /// Number of occupied / virtual orbitals in the BSE (ia) window
    /// (frozen-core aware).
    #[pyo3(get)] nocc: usize,
    #[pyo3(get)] nvir: usize,
    /// Singlet excitation energies Ω_n (Hartree), ascending.
    omega: Vec<f64>,
    /// GW quasiparticle energies used for the diagonal (active block, Ha).
    eps_qp: Vec<f64>,
    /// Length-gauge oscillator strengths f_n (dimensionless), same ordering
    /// as `omega`. See `ferric_gw::bse::tda_oscillator_strengths` for the
    /// convention (PySCF-cross-checked, see
    /// `crates/ferric-gw/tests/bse_oscillator_strength.rs`).
    oscillator_strength: Vec<f64>,
}

#[pymethods]
impl PyBseResult {
    #[getter]
    fn omega<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.omega)
    }
    #[getter]
    fn eps_qp<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.eps_qp)
    }
    #[getter]
    fn oscillator_strength<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.oscillator_strength)
    }
    /// Lowest singlet excitation energy in eV.
    fn lowest_ev(&self) -> f64 {
        self.omega[0] * 27.211_386_245_988
    }
    /// Oscillator strength of the lowest singlet.
    fn lowest_oscillator_strength(&self) -> f64 {
        self.oscillator_strength[0]
    }
    fn __repr__(&self) -> String {
        format!(
            "BseResult(nocc={}, nvir={}, n_excitations={})",
            self.nocc, self.nvir, self.omega.len(),
        )
    }
    fn __str__(&self) -> String {
        let lowest_ev = if self.omega.is_empty() {
            "N/A".to_string()
        } else {
            format!("{:.4}", self.omega[0] * 27.211_386_245_988)
        };
        format!(
            "BSE-TDA: {} excitations, lowest={} eV, nocc={}, nvir={}",
            self.omega.len(), lowest_ev, self.nocc, self.nvir,
        )
    }
}

/// BSE-TDA singlet excitation energies on a closed-shell (RHF) reference.
///
/// Computes G0W0@HF quasiparticle energies internally for every MO (so every
/// particle–hole pair in the active window has a real QP energy), builds the
/// static-screened TDA matrix, and returns its eigenvalues. Closed-shell
/// only — `ferric_gw::bse::run_bse_tda` hard-errors on a non-restricted
/// reference; there is no `reference`/UHF kwarg (mirrors `run_gw`'s HF-only
/// default path, not `run_u_gw`). Kwargs mirror `run_gw`'s `[rpa]`-equivalent
/// shape (n_quad/quadrature/u0/trunc_thresh/eigensolver_conv_thresh/
/// k_builder/chi0_sparsity/memory_budget_gb) plus `frozen_core`, which is
/// threaded to both the PDEP (W) build and the internal GW self-energy build
/// for consistency, exactly like the CLI's `"bse-tda"` dispatch arm.
#[pyfunction]
#[pyo3(signature = (
    mol, basis_set, auxbasis,
    frozen_core=None, n_quad=None, quadrature=None, u0=None,
    trunc_thresh=None, eigensolver_conv_thresh=None,
    k_builder=None, chi0_sparsity=None, memory_budget_gb=None,
))]
#[allow(clippy::too_many_arguments)]
fn run_bse_tda(
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    auxbasis: &PyBasisSet,
    frozen_core: Option<usize>,
    n_quad: Option<usize>,
    quadrature: Option<&str>,
    u0: Option<f64>,
    trunc_thresh: Option<f64>,
    eigensolver_conv_thresh: Option<f64>,
    k_builder: Option<&str>,
    chi0_sparsity: Option<&str>,
    memory_budget_gb: Option<f64>,
) -> PyResult<PyBseResult> {
    use ferric_gw::bse::run_bse_tda as run_bse_tda_inner;
    use ferric_rpa::config::{QuadratureConfig, QuadratureScheme, SternheimerConfig};
    use ferric_rpa::PdepRpaConfig;

    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    let dfbs = PreparedBasis::new(&mol.inner, &auxbasis.inner).map_err(make_err)?;
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).map_err(make_err)?;
    let ctx = ParallelContext::default();

    let cfg = rhf_config(k_builder);
    let scf = solve_rhf(&ctx, &mol.inner, &prep, op, &bounds, &cfg).map_err(make_err)?;
    if !scf.converged {
        return Err(make_err(ferric_core::FerricError::ScfConvergence {
            iterations: scf.iterations, last_energy: scf.energy,
        }));
    }

    let scheme = QuadratureScheme::parse_config_str(quadrature)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("quadrature: {e}")))?;
    let fc = frozen_core.unwrap_or(0);
    let pdep_cfg = PdepRpaConfig {
        frozen_core: fc,
        trunc_thresh: trunc_thresh.unwrap_or(1e-4),
        eigensolver_max_vecs: 0,
        eigensolver_conv_thresh: eigensolver_conv_thresh.unwrap_or(1e-6),
        quadrature: QuadratureConfig {
            scheme,
            n_points: n_quad.unwrap_or(20),
            u0: u0.unwrap_or(0.5),
        },
        sternheimer: SternheimerConfig::default(),
        run_diagnostics: false,
        eigensolver: ferric_rpa::Eigensolver::default(),
        chi0_backend: ferric_rpa::config::Chi0Backend::default(),
        chi0_sparsity: ferric_rpa::config::Chi0Sparsity::parse_config_str(chi0_sparsity)
            .map_err(make_err)?,
        memory_budget_bytes: budget_bytes_from_gb(memory_budget_gb),
        // run_bse_tda's internal GW build forces this on regardless; set it
        // explicitly for clarity at the call site too (matches run_gw).
        need_inv_dielectric_freq: true,
        need_eigenvalues_freq: true,
        verbose: false,
    };

    let r = run_bse_tda_inner(&mol.inner, &prep, &dfbs, op, &scf, &pdep_cfg, fc)
        .map_err(make_err)?;
    Ok(PyBseResult {
        nocc: r.nocc,
        nvir: r.nvir,
        omega: r.omega,
        eps_qp: r.eps_qp,
        oscillator_strength: r.oscillator_strength,
    })
}

// ── TDHF/RPAx static polarizability (closed-shell, KS reference) ──
//
// `run_bse_c6`/`run_bse_c6_ks` (the dynamic alpha(iw)/C6 variants of this same
// kernel) are deliberately NOT wired here. docs/VALIDATION.md's "Correlation /
// response (RPA, GW, BSE)" table records a validated negative result: C6 from
// alpha(iw) on this kernel stays ~63% low regardless of the HOMO-LUMO gap,
// worse than ferric's production dRPA/PDEP C6 pipeline. Only the STATIC
// (omega=0) polarizability is exposed as a production entry point.

#[pyclass]
#[pyo3(name = "TdhfStaticPolarizabilityResult")]
struct PyTdhfStaticPolarizabilityResult {
    #[pyo3(get)] nocc: usize,
    #[pyo3(get)] nvir: usize,
    /// Isotropic average (1/3) Tr(alpha), a.u.
    #[pyo3(get)] iso: f64,
    tensor: [[f64; 3]; 3],
}

#[pymethods]
impl PyTdhfStaticPolarizabilityResult {
    /// Cartesian alpha_ij(0) tensor (3x3, a.u.), i,j in {x,y,z}.
    #[getter]
    fn tensor<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let flat: Vec<f64> = self.tensor.iter().flat_map(|row| row.iter().copied()).collect();
        let arr = Array2::from_shape_vec((3, 3), flat).expect("3x3 tensor is always well-shaped");
        PyArray2::from_array(py, &arr)
    }
    fn __repr__(&self) -> String {
        format!(
            "TdhfStaticPolarizabilityResult(iso={:.6}, nocc={}, nvir={})",
            self.iso, self.nocc, self.nvir,
        )
    }
    fn __str__(&self) -> String {
        format!(
            "TDHF Static Polarizability: iso={:.6} a.u., nocc={}, nvir={}",
            self.iso, self.nocc, self.nvir,
        )
    }
}

/// RPAx@KS **static** polarizability (omega=0) on a closed-shell KS reference.
///
/// **Scope: static polarizability only.** Do not use this method, or its
/// output, for C6/dispersion — see the module doc above and
/// `ferric_gw::bse::run_rpax_static_polarizability`'s doc comment for the
/// full negative-result caveat (validated in `docs/VALIDATION.md`: C6 built
/// from this same kernel's dynamic alpha(iw) stays ~63% low regardless of
/// the HOMO-LUMO gap, worse than ferric's production dRPA/PDEP C6 pipeline).
///
/// `xc` is REQUIRED (e.g. `"pbe"`) — this method's validated accuracy
/// (static alpha ~= DOSD water, 9.24 vs 9.64 a.u.) is specifically a
/// KS-reference result; the HF-reference variant of this same kernel gives a
/// much worse static alpha (~5.24 a.u.), so there is no HF-default fallback
/// here (unlike `run_gw`'s `xc=None` HF default). `scissor` (Hartree, default
/// 0.0) is added to every virtual orbital energy before assembling the
/// diagonal — a cheap proxy for widening the KS gap toward a GW-level gap.
/// Other kwargs mirror `run_bse_tda`'s `[rpa]`-equivalent shape (n_quad/
/// quadrature/u0/trunc_thresh/eigensolver_conv_thresh/k_builder/
/// chi0_sparsity/memory_budget_gb) plus `frozen_core`.
#[pyfunction]
#[pyo3(signature = (
    mol, basis_set, auxbasis, xc,
    scissor=None, frozen_core=None, n_quad=None, quadrature=None, u0=None,
    trunc_thresh=None, eigensolver_conv_thresh=None,
    k_builder=None, chi0_sparsity=None, memory_budget_gb=None,
))]
#[allow(clippy::too_many_arguments)]
fn run_tdhf_static_polarizability(
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    auxbasis: &PyBasisSet,
    xc: &str,
    scissor: Option<f64>,
    frozen_core: Option<usize>,
    n_quad: Option<usize>,
    quadrature: Option<&str>,
    u0: Option<f64>,
    trunc_thresh: Option<f64>,
    eigensolver_conv_thresh: Option<f64>,
    k_builder: Option<&str>,
    chi0_sparsity: Option<&str>,
    memory_budget_gb: Option<f64>,
) -> PyResult<PyTdhfStaticPolarizabilityResult> {
    use ferric_gw::bse::run_rpax_static_polarizability;
    use ferric_rpa::config::{QuadratureConfig, QuadratureScheme, SternheimerConfig};
    use ferric_rpa::PdepRpaConfig;

    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    let dfbs = PreparedBasis::new(&mol.inner, &auxbasis.inner).map_err(make_err)?;
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).map_err(make_err)?;
    let ctx = ParallelContext::default();

    // KS reference (required — see doc comment above), same ladder path
    // run_gw's xc branch uses.
    let mut cfg = rhf_config(k_builder);
    cfg.xc = Some(xc.to_string());
    cfg.df_j_aux = Some("def2-universal-jkfit".to_string());
    cfg.df_k_aux = Some("def2-universal-jkfit".to_string());
    let ladder = ferric_scf::ladder::ksdft_ladder(&cfg);
    let lr = ferric_scf::ladder::solve_rhf_ladder(&ctx, &mol.inner, &prep, op, &bounds, &ladder)
        .map_err(make_err)?;
    let scf = lr.result;
    if !scf.converged {
        return Err(make_err(ferric_core::FerricError::ScfConvergence {
            iterations: scf.iterations, last_energy: scf.energy,
        }));
    }

    let scheme = QuadratureScheme::parse_config_str(quadrature)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("quadrature: {e}")))?;
    let fc = frozen_core.unwrap_or(0);
    let pdep_cfg = PdepRpaConfig {
        frozen_core: fc,
        trunc_thresh: trunc_thresh.unwrap_or(1e-4),
        eigensolver_max_vecs: 0,
        eigensolver_conv_thresh: eigensolver_conv_thresh.unwrap_or(1e-6),
        quadrature: QuadratureConfig {
            scheme,
            n_points: n_quad.unwrap_or(20),
            u0: u0.unwrap_or(0.5),
        },
        sternheimer: SternheimerConfig::default(),
        run_diagnostics: false,
        eigensolver: ferric_rpa::Eigensolver::default(),
        chi0_backend: ferric_rpa::config::Chi0Backend::default(),
        chi0_sparsity: ferric_rpa::config::Chi0Sparsity::parse_config_str(chi0_sparsity)
            .map_err(make_err)?,
        memory_budget_bytes: budget_bytes_from_gb(memory_budget_gb),
        // No GW self-energy build in this path (static screening modes from
        // run_pdep_rpa only) — does not need the inverse-dielectric
        // frequency stack that run_bse_tda/run_gw force on.
        need_inv_dielectric_freq: false,
        need_eigenvalues_freq: true,
        verbose: false,
    };

    let r = run_rpax_static_polarizability(
        &mol.inner, &prep, &dfbs, op, &scf, &pdep_cfg, fc, scissor.unwrap_or(0.0),
    )
    .map_err(make_err)?;
    Ok(PyTdhfStaticPolarizabilityResult {
        nocc: r.nocc,
        nvir: r.nvir,
        iso: r.iso,
        tensor: r.tensor,
    })
}

// ── TDDFT (closed-shell linear response) ──

#[pyclass]
#[pyo3(name = "TddftResult")]
struct PyTddftResult {
    #[pyo3(get)] n_roots: usize,
    #[pyo3(get)] method: String,
    excitation_energies: Vec<f64>,
    oscillator_strengths: Vec<f64>,
}

#[pymethods]
impl PyTddftResult {
    #[getter]
    fn excitation_energies<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.excitation_energies)
    }
    #[getter]
    fn oscillator_strengths<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.oscillator_strengths)
    }
    fn lowest_ev(&self) -> f64 {
        if self.excitation_energies.is_empty() { 0.0 }
        else { self.excitation_energies[0] * 27.211_386_245_988 }
    }
    fn __repr__(&self) -> String {
        format!(
            "TddftResult(method={}, n_roots={}, lowest={:.4} eV)",
            self.method, self.n_roots, self.lowest_ev(),
        )
    }
    fn __str__(&self) -> String {
        let ha_to_ev = 27.211_386_245_988;
        let mut s = format!("TDDFT {} — {} roots:\n", self.method, self.n_roots);
        for (i, (&e, &f)) in self.excitation_energies.iter()
            .zip(&self.oscillator_strengths).enumerate()
        {
            s += &format!("  {}: {:.6} Ha ({:.4} eV)  f = {:.6}\n", i + 1, e, e * ha_to_ev, f);
        }
        s
    }
}

#[pyfunction]
#[pyo3(signature = (mol, basis_set, auxbasis, functional=None, n_roots=3, method="tda"))]
fn run_tddft(
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    auxbasis: &PyBasisSet,
    functional: Option<&str>,
    n_roots: usize,
    method: &str,
) -> PyResult<PyTddftResult> {
    use ferric_tddft::{TddftConfig, TddftMethod};

    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    let dfbs = PreparedBasis::new(&mol.inner, &auxbasis.inner).map_err(make_err)?;
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).map_err(make_err)?;
    let ctx = ParallelContext::default();

    let tddft_method = match method.to_lowercase().as_str() {
        "tda" | "cis" => TddftMethod::Tda,
        "casida" | "rpa" | "tddft" | "tdhf" => TddftMethod::Casida,
        _ => return Err(pyo3::exceptions::PyValueError::new_err(
            format!("Unknown TDDFT method '{method}'; expected 'tda' or 'casida'"),
        )),
    };

    let c_hf;
    let scf = if let Some(xc_name) = functional {
        let mut cfg = rhf_config(None);
        cfg.xc = Some(xc_name.to_string());
        cfg.df_j_aux = Some("def2-universal-jkfit".to_string());
        cfg.df_k_aux = Some("def2-universal-jkfit".to_string());
        let ladder = ferric_scf::ladder::ksdft_ladder(&cfg);
        let lr = ferric_scf::ladder::solve_rhf_ladder(&ctx, &mol.inner, &prep, op, &bounds, &ladder)
            .map_err(make_err)?;
        let xc_def = ferric_dft::libxc::xc_def_from_name(xc_name)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))?;
        let k_mix = ferric_dft::libxc::k_mix_from_xc_def(&xc_def);
        c_hf = k_mix.sr;
        lr.result
    } else {
        let cfg = rhf_config(None);
        let scf = solve_rhf(&ctx, &mol.inner, &prep, op, &bounds, &cfg).map_err(make_err)?;
        c_hf = 1.0;
        scf
    };
    if !scf.converged {
        return Err(make_err(ferric_core::FerricError::ScfConvergence {
            iterations: scf.iterations, last_energy: scf.energy,
        }));
    }

    let config = TddftConfig { n_roots, method: tddft_method };
    let r = ferric_tddft::run_tddft(&mol.inner, &prep, &dfbs, &scf, &config, c_hf)
        .map_err(make_err)?;

    Ok(PyTddftResult {
        n_roots: r.excitation_energies.len(),
        method: format!("{:?}", r.method),
        excitation_energies: r.excitation_energies,
        oscillator_strengths: r.oscillator_strengths,
    })
}

// ── Raw integral access (for Python-side method prototyping) ──

/// Raw 3-center Coulomb integrals (P|μν) as a numpy array of shape
/// (naux, n_bf, n_bf). Undressed — pair with `compute_metric_2c` and a
/// V^{-1/2} (or Cholesky L^{-1}) factor on the Python side to form RI
/// B-tensors. ECPs are applied to the molecule first, matching `run_rhf`.
#[pyfunction]
fn compute_eri3<'py>(
    py: Python<'py>,
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    aux_basis_set: &PyBasisSet,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let mut emol = mol.inner.clone();
    emol.apply_ecp(&basis_set.inner);
    let obs = PreparedBasis::new(&emol, &basis_set.inner).map_err(make_err)?;
    let dfbs = PreparedBasis::new(&emol, &aux_basis_set.inner).map_err(make_err)?;
    let eri3 = ferric_integrals::threeindex::eri3_tensor(Operator::coulomb(), &obs, &dfbs)
        .map_err(make_err)?;
    Ok(PyArray3::from_array(py, &eri3))
}

/// Blocked MO-basis 3-center Coulomb integrals (P|pq), shape
/// (naux, n_left, n_right), for arbitrary coefficient matrices `c_left` /
/// `c_right` of shape (n_bf, n_*) — e.g. localized occupieds × canonical
/// virtuals. Unlike `compute_eri3`, the naux·n_bf² AO tensor is NEVER
/// materialized: raw aux-row blocks are generated under `memory_budget_gb`
/// and transformed immediately (`eri3_mo_ov_blocked`, bit-identical to the
/// dense path). This is what makes C32+/cc-pVDZ domain prototyping possible
/// from Python — the dense AO tensor is ~13 GB there. ECPs applied first,
/// matching `run_rhf`.
/// Resolve a low-level operator spec: name + RAW Bohr-native parameters
/// (unlike the run_* wrappers, which take Angstrom units). omega in Bohr^-1,
/// r0 in Bohr. "terfc"/"terf" with omega=None use the Dutoi-linked
/// omega = 1/(r0*sqrt2); an explicit omega decouples them (tables cover
/// (omega*r0)^2 <= 80; needs FERRIC_TERF_TABLE_DIR).
fn resolve_operator(name: &str, omega: Option<f64>, r0: Option<f64>) -> PyResult<Operator> {
    let need = |o: Option<f64>, what: &str| {
        o.ok_or_else(|| pyo3::exceptions::PyValueError::new_err(format!("operator '{name}' requires {what}")))
    };
    Ok(match name {
        "coulomb" => Operator::coulomb(),
        "erf" => Operator::erf(need(omega, "omega (Bohr^-1)")?),
        "erfc" => Operator::erfc(need(omega, "omega (Bohr^-1)")?),
        "terfc" => match omega {
            Some(w) => Operator::terfc_with_omega(need(r0, "r0 (Bohr)")?, w),
            None => Operator::terfc(need(r0, "r0 (Bohr)")?),
        },
        "terf" => match omega {
            Some(w) => Operator::terf_with_omega(need(r0, "r0 (Bohr)")?, w),
            None => Operator::terf(need(r0, "r0 (Bohr)")?),
        },
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown operator '{other}'; expected coulomb|erf|erfc|terf|terfc"
            )))
        }
    })
}

#[pyfunction]
#[pyo3(signature = (mol, basis_set, aux_basis_set, c_left, c_right, memory_budget_gb=2.0, operator="coulomb", omega=None, r0=None))]
fn compute_eri3_mo<'py>(
    py: Python<'py>,
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    aux_basis_set: &PyBasisSet,
    c_left: numpy::PyReadonlyArray2<f64>,
    c_right: numpy::PyReadonlyArray2<f64>,
    memory_budget_gb: f64,
    operator: &str,
    omega: Option<f64>,
    r0: Option<f64>,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let mut emol = mol.inner.clone();
    emol.apply_ecp(&basis_set.inner);
    let obs = PreparedBasis::new(&emol, &basis_set.inner).map_err(make_err)?;
    let dfbs = PreparedBasis::new(&emol, &aux_basis_set.inner).map_err(make_err)?;
    let n_bf = obs.nbasis();
    let cl = c_left.as_array().to_owned();
    let cr = c_right.as_array().to_owned();
    if cl.nrows() != n_bf || cr.nrows() != n_bf {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "coefficient rows must equal n_bf={n_bf} (got c_left {}x{}, c_right {}x{})",
            cl.nrows(),
            cl.ncols(),
            cr.nrows(),
            cr.ncols()
        )));
    }
    let budget = (memory_budget_gb.max(0.1) * (1u64 << 30) as f64) as usize;
    let op = resolve_operator(operator, omega, r0)?;
    let mo = ferric_mp2::rimp2::eri3_mo_ov_blocked(op, &obs, &dfbs, &cl, &cr, budget)
        .map_err(make_err)?;
    Ok(PyArray3::from_owned_array(py, mo))
}

/// Result of Boys localization: localized coefficients, orbital centers,
/// and convergence info.
#[pyclass(name = "BoysResult")]
struct PyBoysResult {
    c_loc_data: ndarray::Array2<f64>,
    centers_data: ndarray::Array2<f64>,
    #[pyo3(get)]
    converged: bool,
    #[pyo3(get)]
    iterations: usize,
}

#[pymethods]
impl PyBoysResult {
    /// Localized MO coefficients, shape (n_bf, n_orb). Orthonormal (a unitary
    /// rotation of the input orbitals).
    fn c_loc<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        PyArray2::from_array(py, &self.c_loc_data)
    }
    /// Boys centers <i|r|i>, shape (n_orb, 3), in Bohr.
    fn centers<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        PyArray2::from_array(py, &self.centers_data)
    }
    fn __repr__(&self) -> String {
        format!(
            "BoysResult(converged={}, iterations={})",
            self.converged, self.iterations,
        )
    }
    fn __str__(&self) -> String {
        format!(
            "Boys Localization: {} orbitals, converged={}, {} iterations",
            self.c_loc_data.ncols(), self.converged, self.iterations,
        )
    }
}

/// Boys (Foster-Boys) localization of the given orbitals (columns of
/// `c_orbs`, shape (n_bf, n_orb) — typically the occupied block of
/// `mo_coefficients()`). Same Jacobi-sweep implementation the Rust probes
/// use. Returns BoysResult{c_loc(), centers(), converged, iterations}.
#[pyfunction]
#[pyo3(signature = (mol, basis_set, c_orbs, max_iter=200))]
fn boys_localize(
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    c_orbs: numpy::PyReadonlyArray2<f64>,
    max_iter: usize,
) -> PyResult<PyBoysResult> {
    let mut emol = mol.inner.clone();
    emol.apply_ecp(&basis_set.inner);
    let obs = PreparedBasis::new(&emol, &basis_set.inner).map_err(make_err)?;
    let c = c_orbs.as_array().to_owned();
    if c.nrows() != obs.nbasis() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "c_orbs rows must equal n_bf={} (got {})",
            obs.nbasis(),
            c.nrows()
        )));
    }
    let dip = ferric_integrals::oneelectron::dipole(&obs, [0.0, 0.0, 0.0]).map_err(make_err)?;
    let res = ferric_mp2::boys::boys_localize(&c, &dip, max_iter);
    Ok(PyBoysResult {
        c_loc_data: res.c_loc,
        centers_data: res.centers,
        converged: res.converged,
        iterations: res.iterations,
    })
}

/// Shell geometry of `basis_set` on `mol` (works for orbital AND auxiliary
/// sets): returns (centers, first_function_offsets, n_functions) with shapes
/// ((n_shells, 3) in Bohr, (n_shells,), (n_shells,)). Enough to build
/// geometric fitting domains from Python (aux functions within r_cut of an
/// orbital center).
#[pyfunction]
fn shell_info<'py>(
    py: Python<'py>,
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
)> {
    let prep = PreparedBasis::new(&mol.inner, &basis_set.inner).map_err(make_err)?;
    let centers = prep.shell_centers();
    let mut c = ndarray::Array2::<f64>::zeros((centers.len(), 3));
    for (i, xyz) in centers.iter().enumerate() {
        c[(i, 0)] = xyz[0];
        c[(i, 1)] = xyz[1];
        c[(i, 2)] = xyz[2];
    }
    let offs: Vec<i64> = prep.shell_offsets().iter().map(|&x| x as i64).collect();
    let dims: Vec<i64> = prep.shell_dims().iter().map(|&x| x as i64).collect();
    Ok((
        PyArray2::from_owned_array(py, c),
        PyArray1::from_vec(py, offs),
        PyArray1::from_vec(py, dims),
    ))
}

/// 2-center metric (P|w|Q) over the auxiliary basis, shape (naux, naux).
/// Default w = Coulomb; `operator`/`omega`/`r0` follow the same RAW
/// Bohr-native conventions as `compute_eri3_mo`.
#[pyfunction]
#[pyo3(signature = (mol, basis_set, aux_basis_set, operator="coulomb", omega=None, r0=None))]
fn compute_metric_2c<'py>(
    py: Python<'py>,
    mol: &PyMolecule,
    basis_set: &PyBasisSet,
    aux_basis_set: &PyBasisSet,
    operator: &str,
    omega: Option<f64>,
    r0: Option<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let mut emol = mol.inner.clone();
    emol.apply_ecp(&basis_set.inner);
    let dfbs = PreparedBasis::new(&emol, &aux_basis_set.inner).map_err(make_err)?;
    let op = resolve_operator(operator, omega, r0)?;
    let v2c = ferric_integrals::threeindex::coulomb_metric_2c(op, &dfbs)
        .map_err(make_err)?;
    Ok(PyArray2::from_array(py, &v2c))
}

// ── Module ──

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
/// Build with `uv run maturin develop --release` (see the README for the venv
/// caveat).
///
/// NOTE: this docstring is duplicated (not just referenced) from the file-level
/// `//!` comment at the top of this file. pyo3 only picks up a `///` doc
/// comment placed directly above `#[pymodule] fn ferric`, not a file-level
/// `//!` — without this, `help(ferric)` / `ferric.__doc__` return empty even
/// though the `//!` content renders fine in `cargo doc`. Keep both in sync.
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
    m.add_class::<PyQmmmSystem>()?;
    m.add_class::<PyQmmmResult>()?;
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
    m.add_class::<PyConformerEnsemble>()?;
    m.add_class::<PyBoltzmannWeights>()?;
    m.add_class::<PyWeightedStats>()?;
    m.add_class::<PyEnsembleDiagnostics>()?;
    // Conformer-ensemble constants, so callers need not hardcode them.
    m.add("DEFAULT_TEMPERATURE_K", ferric_core::conformers::DEFAULT_TEMPERATURE_K)?;
    m.add("BOLTZMANN_HARTREE_PER_K", ferric_core::conformers::BOLTZMANN_HARTREE_PER_K)?;
    m.add_function(wrap_pyfunction!(boltzmann_weights, m)?)?;
    m.add_function(wrap_pyfunction!(weighted_stats, m)?)?;
    m.add_function(wrap_pyfunction!(weighted_stats_vector, m)?)?;
    m.add_function(wrap_pyfunction!(weighted_stats_tensor, m)?)?;
    m.add_function(wrap_pyfunction!(run_rhf, m)?)?;
    m.add_function(wrap_pyfunction!(run_uhf, m)?)?;
    m.add_function(wrap_pyfunction!(run_rohf, m)?)?;
    m.add_function(wrap_pyfunction!(run_optimize, m)?)?;
    m.add_function(wrap_pyfunction!(run_qmmm, m)?)?;
    m.add_function(wrap_pyfunction!(run_frequencies, m)?)?;
    m.add_function(wrap_pyfunction!(esp_at_atoms, m)?)?;
    m.add_function(wrap_pyfunction!(esp_at_points, m)?)?;
    m.add_function(wrap_pyfunction!(hirshfeld_charges, m)?)?;
    m.add_function(wrap_pyfunction!(lowdin_charges, m)?)?;
    m.add_function(wrap_pyfunction!(run_lmp2, m)?)?;
    m.add_function(wrap_pyfunction!(run_drpa, m)?)?;
    m.add_function(wrap_pyfunction!(run_drpa_scan, m)?)?;
    m.add_function(wrap_pyfunction!(run_linlccd_amplitude, m)?)?;
    m.add_function(wrap_pyfunction!(tune_omega, m)?)?;
    m.add_function(wrap_pyfunction!(orbital_moments, m)?)?;
    m.add_function(wrap_pyfunction!(density_second_moment, m)?)?;
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
    m.add_function(wrap_pyfunction!(run_ksdft, m)?)?;
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
    m.add_function(wrap_pyfunction!(shell_info, m)?)?;
    Ok(())
}
