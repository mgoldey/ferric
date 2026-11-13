//! QM/MM bindings: moved verbatim from `lib.rs`.

use super::*;

// ── QM/MM ──

/// An explicit-parameter AMBER-form MM force field topology. Thin wrapper
/// over `ferric_mm::MmTopology`; this class assigns no parameters of its
/// own (see `ferric-mm`'s crate docs) — every number here is caller-supplied
/// data, typically produced by `scripts/validation/mm_topology.py::topology_from_openmm`
/// or a hand-built toy system.
///
/// Units: constructor takes AMBER-convention units (kcal/mol, Angstrom,
/// degrees), matching most published force fields; internally converted
/// once to Hartree/Bohr/radians.
#[pyclass]
#[pyo3(name = "MmTopology")]
#[derive(Clone)]
pub(crate) struct PyMmTopology {
    pub(crate) inner: ferric_mm::MmTopology,
}

#[pymethods]
impl PyMmTopology {
    /// `bonds`: list of `(i, j, k_kcal_per_mol_per_ang2, r0_angstrom)`.
    /// `angles`: list of `(i, j, k, k_theta_kcal_per_mol_per_rad2, theta0_degrees)`.
    /// `torsions`: list of `(i, j, k, l, periodicity, k_phi_kcal_per_mol, phase_degrees)`.
    /// `sigmas_angstrom`/`epsilons_kcal` are per-atom Lennard-Jones parameters
    /// (Lorentz-Berthelot mixed at evaluation time).
    #[staticmethod]
    #[pyo3(signature = (charges, sigmas_angstrom, epsilons_kcal, bonds, angles, torsions))]
    #[allow(clippy::too_many_arguments)]
    fn from_amber_units(
        charges: Vec<f64>,
        sigmas_angstrom: Vec<f64>,
        epsilons_kcal: Vec<f64>,
        bonds: Vec<(usize, usize, f64, f64)>,
        angles: Vec<(usize, usize, usize, f64, f64)>,
        torsions: Vec<(usize, usize, usize, usize, u32, f64, f64)>,
    ) -> PyResult<Self> {
        if sigmas_angstrom.len() != charges.len() || epsilons_kcal.len() != charges.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "MmTopology.from_amber_units: charges ({}), sigmas_angstrom ({}) and \
                 epsilons_kcal ({}) must have the same length",
                charges.len(),
                sigmas_angstrom.len(),
                epsilons_kcal.len()
            )));
        }
        let lj_angstrom_kcal: Vec<(f64, f64)> =
            sigmas_angstrom.into_iter().zip(epsilons_kcal).collect();
        let inner = ferric_mm::MmTopology::from_amber_units(
            charges,
            lj_angstrom_kcal,
            bonds,
            angles,
            torsions,
        )
        .map_err(make_err)?;
        Ok(Self { inner })
    }

    fn n_atoms(&self) -> usize {
        self.inner.n_atoms()
    }

    fn __repr__(&self) -> String {
        format!(
            "MmTopology(n_atoms={}, n_bonds={}, n_angles={}, n_torsions={})",
            self.inner.n_atoms(),
            self.inner.bonds.len(),
            self.inner.angles.len(),
            self.inner.torsions.len()
        )
    }
}

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
pub(crate) struct PyQmmmSystem {
    pub(crate) inner: ferric_scf::qmmm::QmmmSystem,
}

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
    /// selection is by atom, with no residue completion by default). Add
    /// `residue_ids` (one entry per atom, `PocketCharges.residue_ids` shape)
    /// alongside `qm_seeds`/`qm_radius_angstrom` to pull in whole residues
    /// instead — a residue joins if ANY of its atoms is within radius.
    /// `residue_ids` together with `qm_indices` is an error (an explicit
    /// index list has no notion of "whole residue" to complete).
    /// `widths_angstrom`, when given, must have one entry per atom (0.0 =
    /// point charge, matching [`ferric_scf::qmmm::QmmmAtom`]'s default).
    /// Ignored for atoms that end up in the QM region, exactly like `charges`.
    /// `polarizabilities_angstrom3`, when given, must have one entry per
    /// atom (0.0 = not a polarizable site) — Thole-damped induced-dipole
    /// polarizable embedding (see `run_qmmm`'s `thole_a` knob); ignored for
    /// atoms in the QM region, exactly like `charges`/`widths_angstrom`.
    #[new]
    #[pyo3(signature = (symbols, coords_angstrom, charges, qm_indices=None, qm_seeds=None, qm_radius_angstrom=None, residue_ids=None, charge=0, multiplicity=1, widths_angstrom=None, polarizabilities_angstrom3=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        symbols: Vec<String>,
        coords_angstrom: Vec<Vec<f64>>,
        charges: Vec<f64>,
        qm_indices: Option<Vec<usize>>,
        qm_seeds: Option<Vec<usize>>,
        qm_radius_angstrom: Option<f64>,
        residue_ids: Option<Vec<usize>>,
        charge: i32,
        multiplicity: usize,
        widths_angstrom: Option<Vec<f64>>,
        polarizabilities_angstrom3: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        use ferric_scf::qmmm::{QmSelection, QmmmAtom, QmmmSystem};
        if symbols.len() != coords_angstrom.len() || symbols.len() != charges.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "symbols ({}), coords_angstrom ({}) and charges ({}) must have the same length",
                symbols.len(),
                coords_angstrom.len(),
                charges.len()
            )));
        }
        if let Some(w) = &widths_angstrom {
            if w.len() != symbols.len() {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "widths_angstrom ({}) must have one entry per atom ({})",
                    w.len(),
                    symbols.len()
                )));
            }
        }
        if let Some(a) = &polarizabilities_angstrom3 {
            if a.len() != symbols.len() {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "polarizabilities_angstrom3 ({}) must have one entry per atom ({})",
                    a.len(),
                    symbols.len()
                )));
            }
        }
        if let Some((i, bad)) = coords_angstrom
            .iter()
            .enumerate()
            .find(|(_, c)| c.len() != 3)
        {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "coords_angstrom[{i}] has {} components, expected 3",
                bad.len()
            )));
        }
        let atoms: Vec<QmmmAtom> = symbols
            .iter()
            .zip(coords_angstrom.iter())
            .zip(charges.iter())
            .enumerate()
            .map(|(i, ((sym, c), &q))| {
                let zn = ferric_core::elements::symbol_to_z(sym).unwrap_or(0);
                let width_bohr = widths_angstrom
                    .as_ref()
                    .map(|w| w[i] * ANGSTROM_TO_BOHR)
                    .unwrap_or(0.0);
                let alpha_bohr3 = polarizabilities_angstrom3
                    .as_ref()
                    .map(|a| a[i] * ANGSTROM_TO_BOHR.powi(3))
                    .unwrap_or(0.0);
                QmmmAtom::new_smeared(
                    sym.clone(),
                    zn,
                    c[0] * ANGSTROM_TO_BOHR,
                    c[1] * ANGSTROM_TO_BOHR,
                    c[2] * ANGSTROM_TO_BOHR,
                    q,
                    width_bohr,
                )
                .with_alpha(alpha_bohr3)
            })
            .collect();
        let selection = match (qm_indices, qm_seeds, qm_radius_angstrom, residue_ids) {
            (Some(idx), None, None, None) => QmSelection::Indices(idx),
            (None, Some(seeds), Some(r), None) => QmSelection::WithinRadius {
                seeds,
                radius: r * ANGSTROM_TO_BOHR,
            },
            (None, Some(seeds), Some(r), Some(residue_ids)) => {
                QmSelection::WithinRadiusWholeResidues {
                    seeds,
                    radius: r * ANGSTROM_TO_BOHR,
                    residue_ids,
                }
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "select the QM region with EITHER qm_indices OR qm_seeds + \
                     qm_radius_angstrom (+ optional residue_ids)",
                ))
            }
        };
        let inner = QmmmSystem::new(&atoms, selection, charge, multiplicity).map_err(make_err)?;
        if let Some(&bad) = inner.qm_indices.iter().find(|&&i| atoms[i].z == 0) {
            // Say WHY it is there and what to do. A bare-charge site (symbol
            // "X", z = 0) is how a pocket point charge enters the structure --
            // it has no basis, so it can never be QM. The common way to hit
            // this is sizing a QM region by RADIUS: the sphere grows until it
            // reaches the pocket, and the message then has to distinguish "you
            // picked a bad atom" from "your radius is too large", which the
            // bare index did not.
            let n_bare = atoms.iter().filter(|a| a.z == 0).count();
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "QM atom {bad} has symbol {:?}, which is not an element, so it \
                 cannot be in the QM region -- a bare-charge site (z = 0) has no \
                 basis functions. This structure has {n_bare} such site(s), which \
                 is how MM point charges are represented. If you selected by \
                 radius, REDUCE qm_radius_angstrom until the sphere holds only \
                 real atoms, or list the QM atoms explicitly with qm_indices.",
                atoms[bad].symbol
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
        Ok(Self {
            inner: self
                .inner
                .clone()
                .with_link_atoms(&bonds, scale)
                .map_err(make_err)?,
        })
    }

    /// Apply a boundary-charge scheme to the MM host atom of each cut bond:
    /// `"keep"` (untreated, default), `"delete-host"` (Z1), `"rc"`
    /// (redistributed charge onto M1–M2 bond midpoints; conserves charge) or
    /// `"rcd"` (also conserves the M1–M2 dipole). Needs the full bond list to
    /// find the M2 shell. Returns a new system.
    fn with_boundary_charges(&self, bonds: Vec<(usize, usize)>, scheme: &str) -> PyResult<Self> {
        let scheme = ferric_scf::qmmm::BoundaryChargeScheme::parse_config_str(scheme)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            inner: self
                .inner
                .clone()
                .with_boundary_charges(&bonds, scheme)
                .map_err(make_err)?,
        })
    }

    /// The QM region (plus link hydrogens, appended last) as a `Molecule`.
    fn qm_molecule(&self) -> PyMolecule {
        PyMolecule {
            inner: self.inner.to_qm_molecule(),
        }
    }
    /// Every POINT embedding charge as `(q, x, y, z)` in **Bohr** — directly
    /// usable as `point_charges=` for `run_rhf`/`run_uhf`/`run_optimize`.
    /// Gaussian-smeared charges (`width > 0`) are NOT included here — see
    /// `smeared_charges()`.
    fn point_charges(&self) -> Vec<(f64, f64, f64, f64)> {
        self.inner
            .to_external_potential()
            .map(|ep| {
                ep.point_charges
                    .iter()
                    .map(|pc| (pc.q, pc.x, pc.y, pc.z))
                    .collect()
            })
            .unwrap_or_default()
    }
    /// Every Gaussian-smeared embedding charge as `(q, x, y, z, width_bohr)`
    /// — directly usable as `smeared_charges=` for `run_rhf`/`run_uhf`.
    /// Point charges (`width == 0`) are NOT included here — see
    /// `point_charges()`.
    fn smeared_charges(&self) -> Vec<(f64, f64, f64, f64, f64)> {
        self.inner
            .to_external_potential()
            .map(|ep| {
                ep.smeared_charges
                    .iter()
                    .map(|sc| (sc.q, sc.x, sc.y, sc.z, sc.width))
                    .collect()
            })
            .unwrap_or_default()
    }
    /// Positions (Bohr) of every embedding charge that actually enters the
    /// potential, in the SAME canonical order as `mm_forces()`'s rows:
    /// atom-centred charges (point AND Gaussian-smeared interleaved, in
    /// ascending full-structure atom index) first, then any RC/RCD midpoint
    /// charges. This is NOT the concatenation of `point_charges()` then
    /// `smeared_charges()` unless every charge in this system is one kind —
    /// use this method's order, not those two lists', to interpret
    /// `mm_forces()`.
    fn mm_charge_positions(&self) -> Vec<(f64, f64, f64)> {
        self.inner
            .mm_charge_positions()
            .into_iter()
            .map(|[x, y, z]| (x, y, z))
            .collect()
    }
    /// Full-structure indices of the QM atoms (ascending; link atoms excluded).
    fn qm_indices(&self) -> Vec<usize> {
        self.inner.qm_indices.clone()
    }
    /// Full-structure indices of the MM atoms (ascending).
    fn mm_indices(&self) -> Vec<usize> {
        self.inner.mm_indices.clone()
    }
    /// Number of real QM atoms (link hydrogens occupy `qm_molecule()` indices
    /// `qm_atom_count()..`).
    fn qm_atom_count(&self) -> usize {
        self.inner.qm_atom_count()
    }
    /// Number of atoms in the full structure.
    fn natoms(&self) -> usize {
        self.inner.atoms.len()
    }
    /// Every atom's current position in **Ångström**, in full-structure
    /// index order (the same ordering `qm_indices()`/`mm_indices()` index
    /// into) — regardless of QM/MM role. Useful for comparing a system
    /// before/after `run_optimize_qmmm` (e.g. to check whether a
    /// particular MM atom actually moved).
    fn atom_coords_angstrom(&self) -> Vec<(f64, f64, f64)> {
        self.inner
            .atoms
            .iter()
            .map(|a| {
                (
                    a.x / ANGSTROM_TO_BOHR,
                    a.y / ANGSTROM_TO_BOHR,
                    a.z_pos / ANGSTROM_TO_BOHR,
                )
            })
            .collect()
    }
    /// Link hydrogen positions in **Ångström**, in `qm_molecule()` order.
    fn link_atom_positions(&self) -> Vec<(f64, f64, f64)> {
        self.inner
            .link_atoms
            .iter()
            .map(|l| {
                (
                    l.position[0] / ANGSTROM_TO_BOHR,
                    l.position[1] / ANGSTROM_TO_BOHR,
                    l.position[2] / ANGSTROM_TO_BOHR,
                )
            })
            .collect()
    }
    /// Shortest link-hydrogen-to-MM-charge distance in **Ångström**, or
    /// `None` without link atoms / charges. A diagnostic: under ~0.5–1 Å you
    /// want `with_boundary_charges("rcd")`.
    fn min_link_to_charge_distance(&self) -> Option<f64> {
        self.inner
            .min_link_to_charge_distance()
            .map(|d| d / ANGSTROM_TO_BOHR)
    }
    /// The boundary scheme currently applied, as its config string.
    fn boundary_scheme(&self) -> &'static str {
        self.inner.boundary_scheme.config_str()
    }
    fn __repr__(&self) -> String {
        format!(
            "QmmmSystem(n_qm={}, n_mm={}, n_link={}, scheme={:?})",
            self.inner.qm_atom_count(),
            self.inner.mm_indices.len(),
            self.inner.link_atoms.len(),
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
pub(crate) struct PyQmmmResult {
    #[pyo3(get)]
    pub(crate) energy: f64,
    #[pyo3(get)]
    pub(crate) converged: bool,
    #[pyo3(get)]
    pub(crate) iterations: usize,
    /// Thole-damped polarizable-embedding polarisation energy (Hartree).
    /// `0.0` when no atom in the system carries a nonzero polarisability
    /// (`polarizabilities_angstrom3` was omitted or all-zero) — the exact
    /// non-polarizable code path, matching
    /// `PolarizableSites{sites: vec![], ..}`'s bit-identical-to-off anchor.
    #[pyo3(get)]
    pub(crate) e_pol: f64,
    pub(crate) qm_gradient_data: Array2<f64>,
    pub(crate) mm_forces_data: Vec<[f64; 3]>,
    pub(crate) full_gradient_data: Array2<f64>,
    pub(crate) mm_energy_data: ferric_mm::MmEnergy,
    pub(crate) induced_dipoles_data: Option<Array2<f64>>,
}

#[pymethods]
impl PyQmmmResult {
    /// `dE/dR` on the QM molecule as solved: real QM atoms then link
    /// hydrogens, `(n_qm + n_link, 3)`.
    fn qm_gradient<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        PyArray2::from_array(py, &self.qm_gradient_data)
    }
    /// Force on each embedding charge, `(n_charges, 3)`, in
    /// `QmmmSystem.mm_charge_positions()` order: atom-centred charges (point
    /// AND Gaussian-smeared interleaved, ascending full-structure atom
    /// index), then any RC/RCD midpoint charges. This is **NOT**
    /// `QmmmSystem.point_charges()` order once any charge is Gaussian-smeared
    /// (`widths_angstrom`/`smeared_charges()` nonzero) — `point_charges()`
    /// only lists the point subset, so zipping it with `mm_forces()` silently
    /// misaligns rows whenever a smeared charge is present. Zipping
    /// `point_charges()` with `mm_forces()` is only valid when the system has
    /// no smeared charges at all (`smeared_charges() == []`); otherwise read
    /// `mm_charge_positions()` alongside this array.
    fn mm_forces<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let n = self.mm_forces_data.len();
        let mut a = Array2::<f64>::zeros((n, 3));
        for (i, f) in self.mm_forces_data.iter().enumerate() {
            for k in 0..3 {
                a[(i, k)] = f[k];
            }
        }
        PyArray2::from_array(py, &a)
    }
    /// `dE/dR` over every REAL atom of the full structure, `(natoms, 3)`:
    /// link-atom rows chain-ruled onto their hosts, MM forces mapped onto the
    /// atoms carrying the charges (midpoint charges split half/half).
    fn full_gradient<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        PyArray2::from_array(py, &self.full_gradient_data)
    }
    /// The MM force-field energy components (Hartree), as a dict with keys
    /// `bond`/`angle`/`torsion`/`lj`/`coulomb`/`total`. All-zero when
    /// `run_qmmm` was not given `mm_topology=` (no MM force field at all —
    /// the exactness contract: omitting it and passing an all-zero
    /// topology are indistinguishable here).
    #[getter]
    fn mm_energy(&self, py: Python<'_>) -> PyResult<Py<pyo3::types::PyDict>> {
        let d = pyo3::types::PyDict::new(py);
        d.set_item("bond", self.mm_energy_data.bond)?;
        d.set_item("angle", self.mm_energy_data.angle)?;
        d.set_item("torsion", self.mm_energy_data.torsion)?;
        d.set_item("lj", self.mm_energy_data.lj)?;
        d.set_item("coulomb", self.mm_energy_data.coulomb)?;
        d.set_item("total", self.mm_energy_data.total)?;
        Ok(d.into())
    }
    /// Converged Thole-damped polarizable-embedding induced dipoles
    /// (`(n_sites, 3)`, a.u.), in the SAME atom order
    /// `QmmmSystem.mm_indices()` filtered to `alpha > 0` would produce
    /// (ascending full-structure index) — `None` when no site was
    /// polarizable (`e_pol == 0.0` in that case too).
    fn induced_dipoles<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
        self.induced_dipoles_data
            .as_ref()
            .map(|d| PyArray2::from_array(py, d))
    }
    fn __repr__(&self) -> String {
        format!(
            "QmmmResult(energy={:.10}, converged={}, e_pol={:.3e}, n_full={})",
            self.energy,
            self.converged,
            self.e_pol,
            self.full_gradient_data.nrows()
        )
    }
}

/// Embedded SCF energy plus gradients for a `QmmmSystem` in one call.
///
/// `method` is `"rhf"` (default), `"uhf"`, `"rks"` or `"uks"`. The SCF knobs
/// mirror `run_rhf`/`run_ksdft`: `xc` selects the functional for `"rks"`/
/// `"uks"` (required for those methods; RI-J/RI-K aux is set to
/// `def2-universal-jkfit` automatically, matching `run_ksdft`) and is
/// rejected for `"rhf"`/`"uhf"` (passing a functional to a method that
/// ignores it would silently produce a plain-HF result). No MM force field
/// is involved (unless `mm_topology=` is given): the energy is `E_QM` in
/// the field of the fixed charges (plus the classical charge-nuclear term),
/// and the gradients are its exact derivatives.
///
/// **Thole-damped polarizable embedding**: if `system` was built with
/// `polarizabilities_angstrom3=` (any nonzero entry among the MM atoms),
/// `thole_a` sets the Thole damping parameter (`None`/omitted = the
/// standard default `2.1304`; pass `0.0` to disable damping entirely — the
/// bare point-dipole tensor). `QmmmResult.e_pol`/`.induced_dipoles()`
/// report the polarisation energy and converged dipoles. The QM gradient
/// correctly includes the polarizable Fock-term contribution for ALL FOUR
/// methods (`"rhf"`/`"uhf"`/`"rks"`/`"uks"`, via `rhf_gradient_with_polarizable`/
/// `uhf_gradient_with_polarizable`/`ks_gradient_closed_with_polarizable`/
/// `ks_gradient_uks_with_polarizable` — see `ferric_scf::polarizable`'s Task
/// B3/B4 gradients; the term is a fixed-mu Hellmann-Feynman contraction
/// depending only on the total spin-summed density, so it applies
/// identically to every SCF variant, see `polarizable_gradient_term`'s
/// doc). **`mm_forces()`/`full_gradient()` DO include every
/// polarizable-embedding force term as of Lane B5**: a polarizable site's
/// own `dE_pol/dR_site` (`ferric_scf::qmmm::polarizable_site_gradient`) AND
/// the reaction force every embedding charge feels from the OTHER sites'
/// induced dipoles (`ferric_scf::qmmm::polarizable_charge_gradient_rows`,
/// wired via `ferric_scf::polarizable::charge_gradient_contribution` — see
/// that function's doc for why this is a distinct term from the site's own
/// row) are both added into `full_gradient()`'s output here, for every
/// method. Verified against central finite differences of the fully
/// reconverged total energy on a colocated charge+alpha MM atom (the
/// realistic force-field case) and a two-site mutual-induction case — see
/// `crates/ferric-scf/tests/qmmm_polarizable_forces.rs`. Boundary charges
/// (RC/RCD) combined with a polarizable site are NOT covered by an FD test
/// (documented gap in that same test file, not a silent one) — the
/// arithmetic (half the reaction force onto each host) is implemented and
/// follows the same convention `full_gradient` already uses for
/// `mm_forces`, but is unverified in combination.
#[pyfunction]
#[pyo3(signature = (system, basis_name, method=None, xc=None, max_iter=None, energy_conv=None, density_conv=None, level_shift=None, mom_after_iter=None, guess=None, mm_topology=None, thole_a=None, ri_j_precision=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_qmmm(
    system: &PyQmmmSystem,
    basis_name: &str,
    method: Option<&str>,
    xc: Option<&str>,
    max_iter: Option<usize>,
    energy_conv: Option<f64>,
    density_conv: Option<f64>,
    level_shift: Option<f64>,
    mom_after_iter: Option<usize>,
    guess: Option<&str>,
    mm_topology: Option<&PyMmTopology>,
    thole_a: Option<f64>,
    ri_j_precision: Option<&str>,
) -> PyResult<PyQmmmResult> {
    use ferric_scf::gradient::{
        rhf_gradient, rhf_gradient_with_polarizable, uhf_gradient, uhf_gradient_with_polarizable,
    };
    use ferric_scf::ks_gradient::{
        ks_gradient_closed_with_polarizable_on_grids, ks_gradient_uks_with_exchange_on_grids,
        ks_gradient_uks_with_polarizable_on_grids,
    };
    use ferric_scf::polarizable::{induce, PolarizableSites, DEFAULT_THOLE_A};
    use ferric_scf::qmmm::{full_gradient_with_mm, mm_forces, qmmm_mm_terms};

    let method = method.unwrap_or("rhf");
    let is_ks = matches!(method, "rks" | "uks");
    if xc.is_some() && !is_ks {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "run_qmmm: xc is only valid for method=\"rks\"/\"uks\", got method={method:?}"
        )));
    }
    if is_ks && xc.is_none() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "run_qmmm: method=\"rks\"/\"uks\" requires xc",
        ));
    }

    let sys = &system.inner;
    let bs = basis::bundled(basis_name).map_err(make_err)?;
    let mut mol = sys.to_qm_molecule();
    mol.apply_ecp(&bs);
    let prep = PreparedBasis::new(&mol, &bs).map_err(make_err)?;
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).map_err(make_err)?;
    let pol_sites = sys.to_polarizable_sites();
    let polarizable = if pol_sites.is_empty() {
        None
    } else {
        Some(PolarizableSites {
            sites: pol_sites,
            thole_a: match thole_a {
                None => Some(DEFAULT_THOLE_A),
                Some(a) if a > 0.0 => Some(a),
                Some(_) => None,
            },
            exclusions: Vec::new(),
            dipole_zeta: 1e4,
            max_sites_dense: 4000,
        })
    };
    let mut config = RhfConfig {
        max_iter: max_iter.unwrap_or_else(default_scf_max_iter),
        energy_conv: energy_conv.unwrap_or(1e-3),
        density_conv: density_conv.unwrap_or(1e-6),
        level_shift: level_shift.unwrap_or(0.0),
        mom_after_iter: mom_after_iter.unwrap_or(0),
        use_sad_guess: parse_guess(guess)?,
        external_potential: sys.to_external_potential(),
        polarizable,
        ..Default::default()
    };
    if is_ks {
        config.xc = xc.map(|s| s.to_string());
        // RI-J always on for KS-DFT (matches run_ksdft's density_fit
        // convention); RI-K only matters for hybrids but is harmless
        // otherwise (bypassed when k_mix.sr == 0 and k_mix.omega == 0).
        config.df_j_aux = Some("def2-universal-jkfit".to_string());
        config.ri_j_precision = parse_ri_j_precision(ri_j_precision)?;
        config.df_k_aux = Some("def2-universal-jkfit".to_string());
    }
    let ctx = ParallelContext::default();
    let ext = config.external_potential.as_ref();
    let (r, qm_grad) = match method {
        "rhf" => {
            let r = solve_rhf(&ctx, &mol, &prep, op, &bounds, &config).map_err(make_err)?;
            let g = if config.polarizable.is_some() {
                rhf_gradient_with_polarizable(
                    &mol,
                    &prep,
                    op,
                    &bounds,
                    &r,
                    ext,
                    config.polarizable.as_ref(),
                    r.induced_dipoles.as_ref(),
                )
                .map_err(make_err)?
            } else {
                rhf_gradient(&mol, &prep, op, &bounds, &r, ext).map_err(make_err)?
            };
            (r, g)
        }
        "uhf" => {
            let r = solve_uhf(&ctx, &mol, &prep, &bounds, &config).map_err(make_err)?;
            let g = if config.polarizable.is_some() {
                uhf_gradient_with_polarizable(
                    &mol,
                    &prep,
                    op,
                    &bounds,
                    &r,
                    ext,
                    config.polarizable.as_ref(),
                    r.induced_dipoles.as_ref(),
                )
                .map_err(make_err)?
            } else {
                uhf_gradient(&mol, &prep, op, &bounds, &r, ext).map_err(make_err)?
            };
            (r, g)
        }
        "rks" => {
            let r = solve_rhf(&ctx, &mol, &prep, op, &bounds, &config).map_err(make_err)?;
            let g = if config.polarizable.is_some() {
                ks_gradient_closed_with_polarizable_on_grids(
                    &mol,
                    &prep,
                    &bs,
                    op,
                    &bounds,
                    xc.unwrap(),
                    &r,
                    ext,
                    config.polarizable.as_ref(),
                    r.induced_dipoles.as_ref(),
                    &KsGradGrids::from_config(&config),
                )
                .map_err(make_err)?
            } else {
                ks_gradient_closed_with_exchange_on_grids(
                    &mol,
                    &prep,
                    &bs,
                    op,
                    &bounds,
                    xc.unwrap(),
                    &r,
                    ext,
                    None,
                    &KsGradGrids::from_config(&config),
                )
                .map_err(make_err)?
            };
            (r, g)
        }
        "uks" => {
            let r = solve_uhf(&ctx, &mol, &prep, &bounds, &config).map_err(make_err)?;
            let g = if config.polarizable.is_some() {
                ks_gradient_uks_with_polarizable_on_grids(
                    &mol,
                    &prep,
                    &bs,
                    op,
                    &bounds,
                    xc.unwrap(),
                    &r,
                    ext,
                    config.polarizable.as_ref(),
                    r.induced_dipoles.as_ref(),
                    &KsGradGrids::from_config(&config),
                )
                .map_err(make_err)?
            } else {
                ks_gradient_uks_with_exchange_on_grids(
                    &mol,
                    &prep,
                    &bs,
                    op,
                    &bounds,
                    xc.unwrap(),
                    &r,
                    ext,
                    None,
                    &KsGradGrids::from_config(&config),
                )
                .map_err(make_err)?
            };
            (r, g)
        }
        m => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "run_qmmm: method must be \"rhf\", \"uhf\", \"rks\" or \"uks\", got {m:?}"
            )))
        }
    };
    // Recompute e_pol standalone (ScfResult does not carry it directly —
    // only induced_dipoles) via a single induce() call at the now-converged
    // density; this reproduces the SAME dipoles the SCF loop's internal
    // call already used this iteration (see qmmm_polarizable.rs's
    // matches_pyscf_prototype_* tests, which cross-check this exact
    // pattern to 1e-7 Ha against the PySCF prototype).
    let e_pol = if let Some(pol) = config.polarizable.as_ref() {
        let site_xyz: Vec<[f64; 4]> = pol
            .sites
            .iter()
            .map(|s| [s.x, s.y, s.z, pol.dipole_zeta])
            .collect();
        let site_basis_p =
            ferric_integrals::site_basis::SiteBasis::new(&site_xyz, 1).map_err(make_err)?;
        let ir =
            induce(&mol, &prep, ext, pol, &site_basis_p, r.density_total()).map_err(make_err)?;
        (ir.e_pol, Some(ir.dipoles))
    } else {
        (0.0, None)
    };
    let (e_pol, induced_dipoles_data) = e_pol;

    let forces = mm_forces(sys, &mol, &prep, r.density_total()).map_err(make_err)?;

    let (mm_energy, mut full) = match mm_topology {
        Some(top) => {
            let n = sys.atoms.len();
            let mut coords_full = Array2::<f64>::zeros((n, 3));
            for (i, a) in sys.atoms.iter().enumerate() {
                coords_full[(i, 0)] = a.x;
                coords_full[(i, 1)] = a.y;
                coords_full[(i, 2)] = a.z_pos;
            }
            let (e_mm, g_mm) = qmmm_mm_terms(sys, &top.inner, &coords_full).map_err(make_err)?;
            let full = full_gradient_with_mm(sys, &qm_grad, &forces, &g_mm).map_err(make_err)?;
            (e_mm, full)
        }
        None => {
            let full = ferric_scf::qmmm::full_gradient(sys, &qm_grad, &forces).map_err(make_err)?;
            (ferric_mm::MmEnergy::default(), full)
        }
    };

    // Thole-damped polarizable embedding's own force terms — a polarizable
    // site's OWN dE_pol/dR_site, plus the reaction force every embedding
    // charge feels from the OTHER sites' induced dipoles
    // (`charge_gradient_contribution`) — reusing the SAME dipoles `e_pol`
    // above just computed (same converged density, same call pattern this
    // module already documents). See `full_gradient_with_polarizable`'s doc
    // in `ferric_scf::qmmm` for the full derivation of both terms; this
    // closes the "mm_forces()/full_gradient() do NOT yet include a
    // polarizable site's own force" gap this function's doc comment used to
    // describe.
    if let Some(pol) = config.polarizable.as_ref() {
        let dipoles = induced_dipoles_data
            .as_ref()
            .expect("polarizable config always produces dipoles");
        let site_rows = ferric_scf::qmmm::polarizable_site_gradient(
            sys,
            &mol,
            &prep,
            r.density_total(),
            ext,
            pol,
            dipoles,
        )
        .map_err(make_err)?;
        for (full_idx, g) in site_rows {
            for k in 0..3 {
                full[(full_idx, k)] += g[k];
            }
        }
        if let Some(ext) = ext {
            let charge_rows =
                ferric_scf::qmmm::polarizable_charge_gradient_rows(sys, ext, pol, dipoles);
            for (full_idx, g) in charge_rows {
                for k in 0..3 {
                    full[(full_idx, k)] += g[k];
                }
            }
        }
    }

    Ok(PyQmmmResult {
        energy: r.energy,
        converged: r.converged,
        iterations: r.iterations,
        e_pol,
        qm_gradient_data: qm_grad,
        mm_forces_data: forces,
        full_gradient_data: full,
        mm_energy_data: mm_energy,
        induced_dipoles_data,
    })
}

/// Result of `run_optimize_qmmm`: the relaxed partition plus its energy
/// trajectory.
#[pyclass]
#[pyo3(name = "QmmmOptimizeResult")]
pub(crate) struct PyQmmmOptimizeResult {
    #[pyo3(get)]
    pub(crate) energy: f64,
    #[pyo3(get)]
    pub(crate) converged: bool,
    #[pyo3(get)]
    pub(crate) steps: usize,
    pub(crate) system_data: ferric_scf::qmmm::QmmmSystem,
    pub(crate) energies_data: Vec<f64>,
}

#[pymethods]
impl PyQmmmOptimizeResult {
    /// The partition at the final (optimized) geometry.
    fn system(&self) -> PyQmmmSystem {
        PyQmmmSystem {
            inner: self.system_data.clone(),
        }
    }
    /// Total energy (Hartree) at every step, in order (length `steps + 1`,
    /// the first entry being the starting geometry).
    fn energies(&self) -> Vec<f64> {
        self.energies_data.clone()
    }
    fn __repr__(&self) -> String {
        format!(
            "QmmmOptimizeResult(energy={:.10}, converged={}, steps={})",
            self.energy, self.converged, self.steps
        )
    }
    fn __str__(&self) -> String {
        format!(
            "Optimized QM/MM Energy: {:.10} Ha (converged: {}, {} steps)",
            self.energy, self.converged, self.steps
        )
    }
}

/// Optimize a `QmmmSystem`'s geometry: real QM atoms always move; MM atoms
/// move per `move_mm`.
///
/// `move_mm`: `"none"` (default — only QM atoms move), `"all"` (every MM
/// atom moves too), `("within", radius_angstrom)` (MM atoms within that
/// distance of any QM atom, measured once at the starting geometry), or
/// `("residues", [residue_id, ...])` (requires the system to have been built
/// with `residue_ids=`, i.e. whole-residue QM selection — `ValueError`
/// otherwise). Any `move_mm` other than `"none"` requires `mm_topology`
/// (moving MM atoms with no force field holding their own geometry together
/// is meaningless).
///
/// `method`/`xc` follow `run_qmmm`'s convention: `"rhf"` (default), `"uhf"`,
/// `"rks"`, `"uks"`; `xc` required for the KS variants, rejected otherwise.
#[pyfunction]
#[pyo3(signature = (system, basis_name, method=None, xc=None, move_mm=None, mm_topology=None, max_steps=None, e_conv=None, scf_max_iter=None, scf_energy_conv=None, scf_density_conv=None, scf_warm_start=None, scf_ri_j_precision=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_optimize_qmmm(
    py: Python<'_>,
    system: &PyQmmmSystem,
    basis_name: &str,
    method: Option<&str>,
    xc: Option<&str>,
    move_mm: Option<Bound<'_, PyAny>>,
    mm_topology: Option<&PyMmTopology>,
    max_steps: Option<usize>,
    e_conv: Option<f64>,
    scf_max_iter: Option<usize>,
    scf_energy_conv: Option<f64>,
    scf_density_conv: Option<f64>,
    scf_warm_start: Option<bool>,
    scf_ri_j_precision: Option<&str>,
) -> PyResult<PyQmmmOptimizeResult> {
    use ferric_scf::qmmm::{optimize_qmmm, QmmmMethod, QmmmOptimizeConfig};

    let method_name = method.unwrap_or("rhf");
    let is_ks = matches!(method_name, "rks" | "uks");
    if xc.is_some() && !is_ks {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "run_optimize_qmmm: xc is only valid for method=\"rks\"/\"uks\", got method={method_name:?}"
        )));
    }
    if is_ks && xc.is_none() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "run_optimize_qmmm: method=\"rks\"/\"uks\" requires xc",
        ));
    }
    let qmmm_method = match method_name {
        "rhf" => QmmmMethod::Rhf,
        "uhf" => QmmmMethod::Uhf,
        "rks" => QmmmMethod::Rks(xc.unwrap().to_string()),
        "uks" => QmmmMethod::Uks(xc.unwrap().to_string()),
        m => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "run_optimize_qmmm: method must be \"rhf\", \"uhf\", \"rks\" or \"uks\", got {m:?}"
            )))
        }
    };

    let move_mm = parse_move_mm(py, move_mm)?;

    let mut scf = RhfConfig::default();
    geometry_task_scf(&mut scf, scf_ri_j_precision)?;
    ScfConv::new(scf_max_iter, scf_energy_conv, scf_density_conv).apply(&mut scf);
    let opt = OptimizeConfig {
        max_steps: max_steps.unwrap_or(100),
        e_conv: e_conv.unwrap_or(1e-6),
        scf_warm_start: scf_warm_start.unwrap_or(true),
        ..Default::default()
    };
    let cfg = QmmmOptimizeConfig {
        method: qmmm_method,
        move_mm,
        opt,
        mm_topology: mm_topology.map(|t| t.inner.clone()),
        scf,
    };

    let ctx = ParallelContext::default();
    let r = optimize_qmmm(&ctx, &system.inner, basis_name, &cfg).map_err(make_err)?;
    Ok(PyQmmmOptimizeResult {
        energy: r.energy,
        converged: r.converged,
        steps: r.steps,
        system_data: r.system,
        energies_data: r.energies,
    })
}

/// Parse the `move_mm=` argument: `None`/`"none"` -> `MoveMm::None`, `"all"`
/// -> `MoveMm::All`, `("within", radius_angstrom)` -> `MoveMm::WithinRadius`
/// (converted to Bohr), `("residues", [ids])` -> `MoveMm::Residues`. Any
/// other value is a `ValueError`, per the repo's config-honesty convention
/// (never silently default to `"none"` on a typo).
pub(crate) fn parse_move_mm(
    py: Python<'_>,
    move_mm: Option<Bound<'_, PyAny>>,
) -> PyResult<ferric_scf::qmmm::MoveMm> {
    use ferric_scf::qmmm::MoveMm;
    let Some(obj) = move_mm else {
        return Ok(MoveMm::None);
    };
    if let Ok(s) = obj.extract::<String>() {
        return match s.as_str() {
            "none" => Ok(MoveMm::None),
            "all" => Ok(MoveMm::All),
            other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "run_optimize_qmmm: unknown move_mm string {other:?}; expected \"none\", \"all\", \
                 (\"within\", radius_angstrom), or (\"residues\", [ids])"
            ))),
        };
    }
    if let Ok((tag, arg)) = obj.extract::<(String, Bound<'_, PyAny>)>() {
        match tag.as_str() {
            "within" => {
                let r: f64 = arg.extract().map_err(|_| {
                    pyo3::exceptions::PyValueError::new_err(
                        "run_optimize_qmmm: (\"within\", radius_angstrom) needs a float radius",
                    )
                })?;
                return Ok(MoveMm::WithinRadius(r * ANGSTROM_TO_BOHR));
            }
            "residues" => {
                let ids: Vec<usize> = arg.extract().map_err(|_| {
                    pyo3::exceptions::PyValueError::new_err(
                        "run_optimize_qmmm: (\"residues\", [ids]) needs a list of int residue ids",
                    )
                })?;
                return Ok(MoveMm::Residues(ids));
            }
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "run_optimize_qmmm: unknown move_mm tag {other:?}; expected \"within\" or \"residues\""
                )))
            }
        }
    }
    let _ = py;
    Err(pyo3::exceptions::PyValueError::new_err(
        "run_optimize_qmmm: move_mm must be \"none\", \"all\", (\"within\", radius_angstrom), \
         or (\"residues\", [ids])",
    ))
}
