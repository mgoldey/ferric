//! Conformer-ensemble bindings: moved verbatim from `lib.rs`.

use super::*;

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
pub(crate) fn conformer_err(e: ferric_core::conformers::ConformerError) -> PyErr {
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
pub(crate) fn molecule_from_coords(
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
pub(crate) struct PyWeightedStats {
    /// Weighted mean `Σ w_i x_i`.
    #[pyo3(get)]
    pub(crate) mean: f64,
    /// Weighted population standard deviation `sqrt(Σ w_i (x_i - mean)²)`.
    /// Exactly 0.0 for a one-conformer ensemble.
    #[pyo3(get)]
    pub(crate) std_dev: f64,
    /// Smallest value across conformers (unweighted).
    #[pyo3(get)]
    pub(crate) min: f64,
    /// Largest value across conformers (unweighted).
    #[pyo3(get)]
    pub(crate) max: f64,
}

impl From<ferric_core::conformers::WeightedStats> for PyWeightedStats {
    fn from(s: ferric_core::conformers::WeightedStats) -> Self {
        PyWeightedStats {
            mean: s.mean,
            std_dev: s.std_dev,
            min: s.min,
            max: s.max,
        }
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
pub(crate) struct PyEnsembleDiagnostics {
    #[pyo3(get)]
    pub(crate) n_conformers: usize,
    /// Conformers within `kT` of the minimum (always >= 1: the minimum itself).
    #[pyo3(get)]
    pub(crate) n_within_kt: usize,
    /// Conformers within `2kT` of the minimum.
    #[pyo3(get)]
    pub(crate) n_within_2kt: usize,
    /// Conformers within `5kT` of the minimum.
    #[pyo3(get)]
    pub(crate) n_within_5kt: usize,
    /// Largest single Boltzmann population.
    #[pyo3(get)]
    pub(crate) max_weight: f64,
    /// Index of the conformer carrying `max_weight`.
    #[pyo3(get)]
    pub(crate) max_weight_index: usize,
    /// Inverse participation ratio `1 / Σ w_i²`.
    #[pyo3(get)]
    pub(crate) effective_n_conformers: f64,
    #[pyo3(get)]
    pub(crate) temperature_k: f64,
    /// Plain-language verdict on whether the ensemble was needed.
    #[pyo3(get)]
    pub(crate) verdict: String,
    pub(crate) text: String,
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
    fn __str__(&self) -> String {
        self.text.clone()
    }
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
pub(crate) struct PyBoltzmannWeights {
    /// Normalized populations in ensemble order. Sum to 1.
    #[pyo3(get)]
    pub(crate) weights: Vec<f64>,
    /// Energies relative to the ensemble minimum, in Hartree (all >= 0).
    #[pyo3(get)]
    pub(crate) relative_energies: Vec<f64>,
    #[pyo3(get)]
    pub(crate) temperature_k: f64,
    /// `kT` at that temperature, in Hartree.
    #[pyo3(get)]
    pub(crate) kt_hartree: f64,
    /// Index of the lowest-energy conformer.
    #[pyo3(get)]
    pub(crate) min_index: usize,
    /// `Z = Σ exp(-(E_i - E_min)/kT)`, always >= 1.
    #[pyo3(get)]
    pub(crate) partition_function: f64,
    pub(crate) inner: ferric_core::conformers::BoltzmannWeights,
}

#[pymethods]
impl PyBoltzmannWeights {
    /// Population-structure diagnostics for this weighting.
    fn diagnostics(&self) -> PyEnsembleDiagnostics {
        self.inner.diagnostics().into()
    }
    fn __len__(&self) -> usize {
        self.weights.len()
    }
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
pub(crate) struct PyConformerEnsemble {
    pub(crate) inner: ferric_core::conformers::ConformerEnsemble,
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
    fn __len__(&self) -> usize {
        self.inner.len()
    }
    /// Number of conformers (always >= 1).
    fn n_conformers(&self) -> usize {
        self.inner.len()
    }
    /// Number of atoms per conformer (shared by construction).
    fn n_atoms(&self) -> usize {
        self.inner.n_atoms()
    }

    /// The conformer geometries as `Molecule` objects, ready to hand to
    /// `run_rhf` / `run_ksdft` / any other driver.
    fn molecules(&self) -> Vec<PyMolecule> {
        self.inner
            .conformers()
            .iter()
            .map(|c| PyMolecule {
                inner: c.molecule.clone(),
            })
            .collect()
    }

    /// Geometry of conformer `index` as a `Molecule`.
    fn molecule(&self, index: usize) -> PyResult<PyMolecule> {
        self.inner
            .conformers()
            .get(index)
            .map(|c| PyMolecule {
                inner: c.molecule.clone(),
            })
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
        self.inner.conformers()[0]
            .molecule
            .atoms
            .iter()
            .map(|a| a.z)
            .collect()
    }

    /// Per-atom ghost flags (basis-only centers), shared by all conformers.
    fn is_ghost(&self) -> Vec<bool> {
        self.inner.conformers()[0]
            .molecule
            .atoms
            .iter()
            .map(|a| a.ghost)
            .collect()
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
        Ok(self
            .boltzmann_weights(energies, temperature_k)?
            .inner
            .diagnostics()
            .into())
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
    fn coords_arrays<'py>(&self, py: Python<'py>, bohr: bool) -> Vec<Bound<'py, PyArray2<f64>>> {
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

/// The Ångström->Bohr factor `ferric_core::mol` applies when parsing XYZ:
/// ferric's one conversion, [`ferric_core::units::ANGSTROM_TO_BOHR`].
///
/// `coordinates()` inverts the conversion by **dividing** by this constant
/// rather than multiplying by `BOHR_TO_ANGSTROM`. Both are correct
/// to 1 ulp, but the divide is the better inverse: measured over 3e6 random
/// coordinates in -20..20 Å, `x * A2B / A2B != x` for 5.0% of values whereas
/// `x * A2B * BOHR_TO_ANGSTROM != x` for 15.4%; worst-case error is 3.6e-15 Å
/// either way. Floating-point multiplication is not exactly invertible, so a
/// round-trip through `from_coordinates` -> `coordinates()` is accurate to
/// ~1 ulp (< 1e-14 Å), NOT bit-exact — do not assert equality on it.
pub(crate) use ferric_core::units::ANGSTROM_TO_BOHR;

/// Accept either element symbols (`"C"`, `"@O"`) or atomic numbers (`6`) for
/// the shared element list — RDKit hands you `GetSymbol()` naturally, but
/// `GetAtomicNum()` is just as common, and silently accepting only one of them
/// is a trap. Mixed lists are accepted too (each entry is resolved on its own).
pub(crate) fn extract_element_symbols(elements: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
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
                return ferric_core::elements::z_to_symbol(z)
                    .map(str::to_string)
                    .ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err(format!(
                            "element {i}: atomic number {z} is not a known element"
                        ))
                    });
            }
            Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "element {i}: expected an element symbol (str) or atomic number (int), \
                 got {}",
                item.get_type()
                    .name()
                    .map(|n| n.to_string())
                    .unwrap_or_else(|_| "?".into())
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
pub(crate) fn boltzmann_weights(
    energies: Vec<f64>,
    temperature_k: f64,
) -> PyResult<PyBoltzmannWeights> {
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
pub(crate) fn weighted_stats(values: Vec<f64>, weights: Vec<f64>) -> PyResult<PyWeightedStats> {
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
pub(crate) fn weighted_stats_vector(
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
pub(crate) fn weighted_stats_tensor(
    values: Vec<Vec<Vec<f64>>>,
    weights: Vec<f64>,
) -> PyResult<Vec<Vec<PyWeightedStats>>> {
    ferric_core::conformers::weighted_stats_tensor(&values, &weights)
        .map(|rows| {
            rows.into_iter()
                .map(|r| r.into_iter().map(Into::into).collect())
                .collect()
        })
        .map_err(conformer_err)
}
