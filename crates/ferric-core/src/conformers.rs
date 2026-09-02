//! Conformer ensembles and Boltzmann-weighted property averaging.
//!
//! Every property ferric computes — a dipole, an ESP, a polarizability — is a
//! property of **one geometry**. For a flexible molecule the physical observable
//! is a thermal average over the accessible conformers:
//!
//! ```text
//!   <A> = Σ_i w_i A_i,     w_i = exp(-(E_i - E_min)/kT) / Z
//! ```
//!
//! A single-conformer number answers a different question than the one asked.
//! This module supplies the averaging machinery: the ensemble container (with
//! the atom-ordering invariant enforced at construction), the Boltzmann weights,
//! weighted mean **and standard deviation** of scalar/tensor properties, and
//! ensemble diagnostics that say plainly whether the ensemble was needed at all.
//!
//! # Scope
//!
//! Conformer **generation** is deliberately NOT implemented here. RDKit does
//! that well, and the `esp-conditioning` consumer already runs
//! SMILES → RDKit → ferric. Conformers are taken as *input*: either a
//! `Vec<Molecule>` or a multi-frame XYZ file ([`crate::conformers::parse_multi_xyz`]).
//!
//! # Why the standard deviation is not optional
//!
//! A conformer-averaged number without a spread is misleading: it hides whether
//! the average came from one dominant conformer (in which case the ensemble was
//! unnecessary) or from fifty comparable ones (in which case any
//! single-conformer result is simply wrong). [`crate::conformers::WeightedStats`] always carries
//! both, and [`crate::conformers::EnsembleDiagnostics`] reports the population structure that
//! produced them.
//!
//! # Example
//!
//! ```
//! use ferric_core::conformers::{ConformerEnsemble, DEFAULT_TEMPERATURE_K};
//! use ferric_core::mol::Molecule;
//!
//! let xyz = "1\nH atom\nH 0.0 0.0 0.0\n";
//! let a = Molecule::parse_xyz(xyz, 0, 2).unwrap();
//! let b = a.clone();
//!
//! // Two degenerate conformers -> 50/50.
//! let ens = ConformerEnsemble::from_molecules_and_energies(
//!     vec![a, b],
//!     &[-0.5, -0.5],
//! ).unwrap();
//! let w = ens.boltzmann_weights(DEFAULT_TEMPERATURE_K).unwrap();
//! assert!((w.weights[0] - 0.5).abs() < 1e-15);
//! ```

use crate::mol::Molecule;
use crate::FerricError;

/// Boltzmann constant in Hartree per Kelvin.
///
/// `k_B = 1.380649e-23 J/K` (exact, SI 2019 redefinition) divided by
/// `E_h = 4.3597447222071e-18 J` (CODATA 2018 Hartree energy).
pub const BOLTZMANN_HARTREE_PER_K: f64 = 3.166_811_563_455_608e-6;

/// Default averaging temperature: 298.15 K (25 °C, thermochemical standard).
pub const DEFAULT_TEMPERATURE_K: f64 = 298.15;

// ─────────────────────────────────────────────────────────────────────────────
// Errors
// ─────────────────────────────────────────────────────────────────────────────

/// Typed failures specific to conformer-ensemble construction and averaging.
///
/// These are separated from the free-form [`FerricError::General`] because the
/// composition/ordering mismatch is the single most likely way this module gets
/// misused, and a caller may reasonably want to match on it (e.g. to re-order
/// the incoming RDKit conformers rather than abort).
// `PartialEq` only, not `Eq`: `BadTemperature` carries an `f64` (which may be
// NaN — a NaN temperature is one of the inputs this variant reports).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ConformerError {
    /// Fewer than one conformer supplied.
    Empty,
    /// Conformer `index` has a different atom count than the reference (index 0).
    AtomCountMismatch { index: usize, expected: usize, found: usize },
    /// Conformer `index` disagrees with the reference on atom `atom`: element,
    /// ghost flag, or ECP core count. Averaging across these silently corrupts
    /// every property, so it is a hard error.
    CompositionMismatch {
        index: usize,
        atom: usize,
        expected: String,
        found: String,
    },
    /// Conformer `index` has a different total charge or spin multiplicity than
    /// the reference. These are different electronic states, not conformers.
    StateMismatch {
        index: usize,
        expected: (i32, usize),
        found: (i32, usize),
    },
    /// The number of energies does not match the number of conformers.
    EnergyCountMismatch { n_conformers: usize, n_energies: usize },
    /// A supplied energy is not finite (NaN/inf), typically an unconverged SCF
    /// that was not checked before being pushed into the ensemble.
    NonFiniteEnergy { index: usize },
    /// Temperature is not strictly positive and finite.
    BadTemperature(f64),
    /// A per-conformer property vector has the wrong length, or the individual
    /// property values have inconsistent shapes.
    PropertyShapeMismatch { expected: usize, found: usize },
}

impl std::fmt::Display for ConformerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConformerError::Empty => write!(f, "conformer ensemble is empty (need at least one conformer)"),
            ConformerError::AtomCountMismatch { index, expected, found } => write!(
                f,
                "conformer {index} has {found} atoms but the reference conformer (0) has {expected}; \
                 conformers must share atom ordering and composition"
            ),
            ConformerError::CompositionMismatch { index, atom, expected, found } => write!(
                f,
                "conformer {index}, atom {atom}: expected {expected} but found {found}; \
                 conformers must share atom ordering and composition (a permuted or \
                 re-elemented geometry silently corrupts every averaged property)"
            ),
            ConformerError::StateMismatch { index, expected, found } => write!(
                f,
                "conformer {index} has charge/multiplicity {found:?} but the reference conformer (0) \
                 has {expected:?}; these are different electronic states, not conformers of one species"
            ),
            ConformerError::EnergyCountMismatch { n_conformers, n_energies } => write!(
                f,
                "got {n_energies} energies for {n_conformers} conformers"
            ),
            ConformerError::NonFiniteEnergy { index } => write!(
                f,
                "conformer {index} has a non-finite energy (NaN or infinity) — \
                 check the SCF converged before adding it to the ensemble"
            ),
            ConformerError::BadTemperature(t) => write!(
                f,
                "temperature {t} K is not a positive finite number"
            ),
            ConformerError::PropertyShapeMismatch { expected, found } => write!(
                f,
                "property vector has {found} entries but the ensemble has {expected} conformers \
                 (or the per-conformer property shapes disagree)"
            ),
        }
    }
}

impl std::error::Error for ConformerError {}

impl From<ConformerError> for FerricError {
    fn from(e: ConformerError) -> Self {
        FerricError::General(e.to_string())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Conformer
// ─────────────────────────────────────────────────────────────────────────────

/// One geometry of a flexible molecule, optionally with its computed energy.
///
/// The energy is `Option` because the ensemble is usefully constructible before
/// the SCF runs (geometry in, energy filled later). Every weighting operation
/// requires all energies to be present and errors clearly otherwise.
#[derive(Debug, Clone, PartialEq)]
pub struct Conformer {
    /// The geometry. Coordinates in Bohr, as everywhere else in ferric.
    pub molecule: Molecule,
    /// Total electronic energy in Hartree, once computed.
    pub energy: Option<f64>,
    /// Free-form label (e.g. RDKit conformer id, "anti", "gauche+").
    pub label: Option<String>,
}

impl Conformer {
    /// A conformer with no energy yet.
    pub fn new(molecule: Molecule) -> Self {
        Conformer { molecule, energy: None, label: None }
    }

    /// A conformer with a known total energy in Hartree.
    pub fn with_energy(molecule: Molecule, energy: f64) -> Self {
        Conformer { molecule, energy: Some(energy), label: None }
    }

    /// Attach a label (builder style).
    pub fn labeled(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
}

/// Compact per-atom identity used for the composition-invariant check.
fn atom_identity(a: &crate::mol::Atom) -> String {
    format!(
        "{}{} (Z={}, n_core_ecp={})",
        if a.ghost { "@" } else { "" },
        a.symbol,
        a.z,
        a.n_core_ecp
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// ConformerEnsemble
// ─────────────────────────────────────────────────────────────────────────────

/// A set of conformers of **one** chemical species, sharing atom ordering,
/// composition, charge and multiplicity.
///
/// The shared-ordering invariant is checked once at construction and cannot be
/// bypassed: the `conformers` field is private and mutation goes through
/// [`ConformerEnsemble::push`], which re-checks. This matters because averaging
/// a per-atom property (charges, per-atom C6, ESP grids referenced to atom
/// centers) across differently-ordered geometries produces a plausible-looking
/// number that is simply wrong, with no runtime symptom.
#[derive(Debug, Clone, PartialEq)]
pub struct ConformerEnsemble {
    conformers: Vec<Conformer>,
}

impl ConformerEnsemble {
    /// Build an ensemble from conformers, validating the shared-composition
    /// invariant against conformer 0.
    ///
    /// Errors with [`ConformerError::Empty`], [`ConformerError::AtomCountMismatch`],
    /// [`ConformerError::CompositionMismatch`], or [`ConformerError::StateMismatch`].
    pub fn new(conformers: Vec<Conformer>) -> Result<Self, ConformerError> {
        if conformers.is_empty() {
            return Err(ConformerError::Empty);
        }
        let reference = &conformers[0];
        for (index, c) in conformers.iter().enumerate().skip(1) {
            check_compatible(reference, c, index)?;
        }
        Ok(ConformerEnsemble { conformers })
    }

    /// Build an ensemble from bare geometries (energies unset).
    pub fn from_molecules(molecules: Vec<Molecule>) -> Result<Self, ConformerError> {
        Self::new(molecules.into_iter().map(Conformer::new).collect())
    }

    /// Build an ensemble from geometries plus their total energies in Hartree.
    ///
    /// This is the usual entry point after running SCF on each RDKit conformer.
    pub fn from_molecules_and_energies(
        molecules: Vec<Molecule>,
        energies: &[f64],
    ) -> Result<Self, ConformerError> {
        if molecules.len() != energies.len() {
            return Err(ConformerError::EnergyCountMismatch {
                n_conformers: molecules.len(),
                n_energies: energies.len(),
            });
        }
        Self::new(
            molecules
                .into_iter()
                .zip(energies.iter())
                .map(|(m, &e)| Conformer::with_energy(m, e))
                .collect(),
        )
    }

    /// Append a conformer, re-checking the shared-composition invariant.
    pub fn push(&mut self, conformer: Conformer) -> Result<(), ConformerError> {
        let index = self.conformers.len();
        check_compatible(&self.conformers[0], &conformer, index)?;
        self.conformers.push(conformer);
        Ok(())
    }

    /// Number of conformers. Always >= 1.
    pub fn len(&self) -> usize {
        self.conformers.len()
    }

    /// Always `false` — an ensemble cannot be constructed empty. Present for
    /// clippy/API convention only.
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Number of atoms per conformer (shared by construction).
    pub fn n_atoms(&self) -> usize {
        self.conformers[0].molecule.atoms.len()
    }

    /// Read-only view of the conformers.
    pub fn conformers(&self) -> &[Conformer] {
        &self.conformers
    }

    /// Set the energy of conformer `index` (e.g. as SCF results come back).
    pub fn set_energy(&mut self, index: usize, energy: f64) -> Result<(), ConformerError> {
        if index >= self.conformers.len() {
            return Err(ConformerError::PropertyShapeMismatch {
                expected: self.conformers.len(),
                found: index + 1,
            });
        }
        if !energy.is_finite() {
            return Err(ConformerError::NonFiniteEnergy { index });
        }
        self.conformers[index].energy = Some(energy);
        Ok(())
    }

    /// All conformer energies in Hartree, in ensemble order.
    ///
    /// Errors if any conformer lacks an energy or carries a non-finite one.
    pub fn energies(&self) -> Result<Vec<f64>, ConformerError> {
        self.conformers
            .iter()
            .enumerate()
            .map(|(index, c)| match c.energy {
                Some(e) if e.is_finite() => Ok(e),
                _ => Err(ConformerError::NonFiniteEnergy { index }),
            })
            .collect()
    }

    /// Boltzmann weights at `temperature_k` Kelvin.
    ///
    /// `w_i = exp(-(E_i - E_min)/kT) / Z`. The `E_min` shift is what makes this
    /// numerically safe: raw `exp(-E_i/kT)` on absolute electronic energies
    /// (order -10^2 Ha, `kT ≈ 9.4e-4` Ha) overflows to infinity immediately.
    /// With the shift the largest exponent is exactly 0, so the largest term is
    /// exactly 1 and `Z >= 1`.
    pub fn boltzmann_weights(&self, temperature_k: f64) -> Result<BoltzmannWeights, ConformerError> {
        let energies = self.energies()?;
        boltzmann_weights(&energies, temperature_k)
    }

    /// Boltzmann weights at the standard 298.15 K.
    pub fn boltzmann_weights_default(&self) -> Result<BoltzmannWeights, ConformerError> {
        self.boltzmann_weights(DEFAULT_TEMPERATURE_K)
    }
}

fn check_compatible(
    reference: &Conformer,
    candidate: &Conformer,
    index: usize,
) -> Result<(), ConformerError> {
    let r = &reference.molecule;
    let c = &candidate.molecule;
    if r.atoms.len() != c.atoms.len() {
        return Err(ConformerError::AtomCountMismatch {
            index,
            expected: r.atoms.len(),
            found: c.atoms.len(),
        });
    }
    if r.charge != c.charge || r.multiplicity != c.multiplicity {
        return Err(ConformerError::StateMismatch {
            index,
            expected: (r.charge, r.multiplicity),
            found: (c.charge, c.multiplicity),
        });
    }
    for (atom, (ra, ca)) in r.atoms.iter().zip(c.atoms.iter()).enumerate() {
        // Element identity, ghost status and ECP core count must all match.
        // Symbol alone is not enough: a ghost `@O` and a real `O` share a
        // symbol but contribute differently to every energy and property.
        if ra.z != ca.z || ra.ghost != ca.ghost || ra.n_core_ecp != ca.n_core_ecp {
            return Err(ConformerError::CompositionMismatch {
                index,
                atom,
                expected: atom_identity(ra),
                found: atom_identity(ca),
            });
        }
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Boltzmann weights
// ─────────────────────────────────────────────────────────────────────────────

/// Boltzmann populations of an ensemble at a given temperature, plus the
/// diagnostics needed to judge whether the ensemble mattered.
#[derive(Debug, Clone, PartialEq)]
pub struct BoltzmannWeights {
    /// Normalized populations, in ensemble order. Sum to 1.
    pub weights: Vec<f64>,
    /// Energies relative to the minimum, in Hartree (all >= 0).
    pub relative_energies: Vec<f64>,
    /// Temperature used, in Kelvin.
    pub temperature_k: f64,
    /// `kT` at that temperature, in Hartree.
    pub kt_hartree: f64,
    /// Index of the lowest-energy conformer.
    pub min_index: usize,
    /// Partition function `Z = Σ exp(-(E_i - E_min)/kT)`, relative to `E_min`.
    /// Always `>= 1` (the minimum contributes exactly 1).
    pub partition_function: f64,
}

impl BoltzmannWeights {
    /// Population-structure diagnostics for this weighting.
    pub fn diagnostics(&self) -> EnsembleDiagnostics {
        let kt = self.kt_hartree;
        let count_within = |mult: f64| -> usize {
            self.relative_energies.iter().filter(|&&de| de <= mult * kt).count()
        };
        let (max_index, &max_weight) = self
            .weights
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .expect("weights are non-empty by construction");
        // Effective number of contributing conformers (inverse participation
        // ratio / Kish effective sample size): 1 for a single dominant
        // conformer, N for N equally-weighted ones.
        let sum_sq: f64 = self.weights.iter().map(|w| w * w).sum();
        let effective_n = if sum_sq > 0.0 { 1.0 / sum_sq } else { 0.0 };
        EnsembleDiagnostics {
            n_conformers: self.weights.len(),
            n_within_kt: count_within(1.0),
            n_within_2kt: count_within(2.0),
            n_within_5kt: count_within(5.0),
            max_weight,
            max_weight_index: max_index,
            effective_n_conformers: effective_n,
            temperature_k: self.temperature_k,
        }
    }
}

/// Compute Boltzmann weights from raw energies (Hartree) at `temperature_k`.
///
/// Free function form for callers who hold energies without a `ConformerEnsemble`.
pub fn boltzmann_weights(
    energies: &[f64],
    temperature_k: f64,
) -> Result<BoltzmannWeights, ConformerError> {
    if energies.is_empty() {
        return Err(ConformerError::Empty);
    }
    if !temperature_k.is_finite() || temperature_k <= 0.0 {
        return Err(ConformerError::BadTemperature(temperature_k));
    }
    for (index, &e) in energies.iter().enumerate() {
        if !e.is_finite() {
            return Err(ConformerError::NonFiniteEnergy { index });
        }
    }
    let kt = BOLTZMANN_HARTREE_PER_K * temperature_k;

    // Locate the minimum. `partial_cmp` is safe here: finiteness was checked above.
    let (min_index, &e_min) = energies
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .expect("non-empty");

    let relative_energies: Vec<f64> = energies.iter().map(|&e| e - e_min).collect();

    // Shifted exponentials: the largest is exp(0) = 1, so no overflow is
    // possible and Z >= 1. Deep-lying conformers underflow to 0.0 gracefully
    // (exp(-746) -> 0), which is the correct physical answer, not an error.
    let boltz: Vec<f64> = relative_energies.iter().map(|&de| (-de / kt).exp()).collect();
    let z: f64 = boltz.iter().sum();

    // Z >= 1 by construction (the minimum contributes exactly 1), so this
    // division is always well-defined; the guard documents the invariant.
    debug_assert!(z >= 1.0, "partition function must be >= 1 after the E_min shift, got {z}");

    let weights: Vec<f64> = boltz.iter().map(|&b| b / z).collect();

    Ok(BoltzmannWeights {
        weights,
        relative_energies,
        temperature_k,
        kt_hartree: kt,
        min_index,
        partition_function: z,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Diagnostics
// ─────────────────────────────────────────────────────────────────────────────

/// Plain-language readout of an ensemble's population structure.
///
/// The two failure modes this is designed to expose:
///
/// * **One conformer carries ~all the weight** (`max_weight` near 1,
///   `effective_n_conformers` near 1) — the ensemble was unnecessary; a
///   single-point calculation on the minimum would have given the same answer.
/// * **No conformer dominates** (`max_weight` small, many conformers within
///   `kT`) — single-conformer results are wrong, and the reported spread on any
///   averaged property is the honest uncertainty.
#[derive(Debug, Clone, PartialEq)]
pub struct EnsembleDiagnostics {
    pub n_conformers: usize,
    /// Conformers with `E_i - E_min <= kT`. Always >= 1 (the minimum itself).
    pub n_within_kt: usize,
    /// Conformers with `E_i - E_min <= 2kT`.
    pub n_within_2kt: usize,
    /// Conformers with `E_i - E_min <= 5kT`.
    pub n_within_5kt: usize,
    /// Largest single population.
    pub max_weight: f64,
    /// Index of the conformer carrying `max_weight`.
    pub max_weight_index: usize,
    /// Inverse participation ratio `1 / Σ w_i²`: the effective number of
    /// conformers actually contributing. 1.0 for a single dominant conformer,
    /// `N` for `N` degenerate ones.
    pub effective_n_conformers: f64,
    pub temperature_k: f64,
}

impl EnsembleDiagnostics {
    /// `true` when one conformer carries at least `threshold` of the population
    /// (default sense: pass 0.95). The ensemble added nothing in that case.
    pub fn is_single_conformer_dominated(&self, threshold: f64) -> bool {
        self.max_weight >= threshold
    }

    /// A plain, non-numeric verdict suitable for printing next to any
    /// ensemble-averaged property.
    pub fn verdict(&self) -> &'static str {
        if self.n_conformers == 1 {
            "single conformer supplied: this is a single-point result, not an ensemble average"
        } else if self.max_weight >= 0.95 {
            "one conformer carries >=95% of the population: the ensemble was unnecessary, \
             the single-conformer answer would have been the same"
        } else if self.max_weight >= 0.5 {
            "one conformer dominates (>=50%) but others contribute: single-conformer results \
             are biased; use the ensemble average and quote its spread"
        } else {
            "no conformer dominates (<50%): single-conformer results are wrong for this molecule; \
             the ensemble average and its spread are the meaningful numbers"
        }
    }
}

impl std::fmt::Display for EnsembleDiagnostics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Conformer ensemble diagnostics (T = {:.2} K):", self.temperature_k)?;
        writeln!(f, "  conformers:                {}", self.n_conformers)?;
        writeln!(f, "  within  kT of minimum:     {}", self.n_within_kt)?;
        writeln!(f, "  within 2kT of minimum:     {}", self.n_within_2kt)?;
        writeln!(f, "  within 5kT of minimum:     {}", self.n_within_5kt)?;
        writeln!(
            f,
            "  max weight:                {:.6} (conformer {})",
            self.max_weight, self.max_weight_index
        )?;
        writeln!(f, "  effective # conformers:    {:.3}", self.effective_n_conformers)?;
        write!(f, "  verdict: {}", self.verdict())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Weighted averaging
// ─────────────────────────────────────────────────────────────────────────────

/// A weighted mean together with its weighted standard deviation.
///
/// Never returned without the spread: an ensemble-averaged property whose
/// conformer-to-conformer scatter is unknown cannot be interpreted.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WeightedStats {
    /// `Σ w_i x_i`.
    pub mean: f64,
    /// `sqrt(Σ w_i (x_i - mean)²)` — the population (not sample) weighted
    /// standard deviation. Exactly 0 for a one-conformer ensemble.
    pub std_dev: f64,
    /// Smallest value in the ensemble (unweighted, for range context).
    pub min: f64,
    /// Largest value in the ensemble (unweighted, for range context).
    pub max: f64,
}

/// Weighted mean and standard deviation of a scalar property.
///
/// `values[i]` is the property computed on conformer `i`; `weights[i]` its
/// Boltzmann population. Weights are used as given (they are already normalized
/// by [`boltzmann_weights`]), so this is exact for a normalized input.
///
/// The variance uses `Σ w_i (x_i - mean)²`, computed in the shifted form rather
/// than as `E[x²] - E[x]²`: the latter is a catastrophic-cancellation trap when
/// the values are large and their spread is small (exactly the regime of
/// absolute electronic energies), and can go negative from rounding alone.
pub fn weighted_stats(values: &[f64], weights: &[f64]) -> Result<WeightedStats, ConformerError> {
    if values.len() != weights.len() {
        return Err(ConformerError::PropertyShapeMismatch {
            expected: weights.len(),
            found: values.len(),
        });
    }
    if values.is_empty() {
        return Err(ConformerError::Empty);
    }
    for (index, &v) in values.iter().enumerate() {
        if !v.is_finite() {
            return Err(ConformerError::NonFiniteEnergy { index });
        }
    }

    let mean: f64 = values.iter().zip(weights).map(|(&v, &w)| w * v).sum();

    // Shifted-form variance. Clamped at 0 so an exactly-degenerate ensemble
    // cannot produce a -1e-19 variance and a NaN square root.
    let variance: f64 = values
        .iter()
        .zip(weights)
        .map(|(&v, &w)| {
            let d = v - mean;
            w * d * d
        })
        .sum::<f64>()
        .max(0.0);

    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    Ok(WeightedStats { mean, std_dev: variance.sqrt(), min, max })
}

/// Weighted mean and standard deviation of a **vector-valued** property
/// (dipole, per-atom charges, a flattened tensor), applied component-wise.
///
/// `values[i]` is conformer `i`'s property vector; all must have the same
/// length. The returned `Vec` has one [`WeightedStats`] per component.
///
/// Component-wise is the right default: a dipole's ensemble average is the
/// average of its Cartesian components (which can cancel — a symmetric ensemble
/// can average to a near-zero vector with large per-conformer magnitudes), and
/// the per-component spread is what exposes that. For the average *magnitude*,
/// pass the per-conformer magnitudes to [`weighted_stats`] instead; the two are
/// different physical questions and this module refuses to conflate them.
pub fn weighted_stats_vector(
    values: &[Vec<f64>],
    weights: &[f64],
) -> Result<Vec<WeightedStats>, ConformerError> {
    if values.len() != weights.len() {
        return Err(ConformerError::PropertyShapeMismatch {
            expected: weights.len(),
            found: values.len(),
        });
    }
    if values.is_empty() {
        return Err(ConformerError::Empty);
    }
    let dim = values[0].len();
    if let Some(bad) = values.iter().find(|v| v.len() != dim) {
        return Err(ConformerError::PropertyShapeMismatch { expected: dim, found: bad.len() });
    }

    let mut out = Vec::with_capacity(dim);
    let mut column = vec![0.0; values.len()];
    for k in 0..dim {
        for (i, v) in values.iter().enumerate() {
            column[i] = v[k];
        }
        out.push(weighted_stats(&column, weights)?);
    }
    Ok(out)
}

/// Weighted mean and standard deviation of a rank-2 tensor property
/// (polarizability `[3][3]`, quadrupole), applied element-wise.
///
/// All conformers must supply the same `(nrow, ncol)`. Returns a `nrow x ncol`
/// grid of [`WeightedStats`].
pub fn weighted_stats_tensor(
    values: &[Vec<Vec<f64>>],
    weights: &[f64],
) -> Result<Vec<Vec<WeightedStats>>, ConformerError> {
    if values.len() != weights.len() {
        return Err(ConformerError::PropertyShapeMismatch {
            expected: weights.len(),
            found: values.len(),
        });
    }
    if values.is_empty() {
        return Err(ConformerError::Empty);
    }
    let nrow = values[0].len();
    let ncol = if nrow > 0 { values[0][0].len() } else { 0 };
    for v in values {
        if v.len() != nrow {
            return Err(ConformerError::PropertyShapeMismatch { expected: nrow, found: v.len() });
        }
        for row in v {
            if row.len() != ncol {
                return Err(ConformerError::PropertyShapeMismatch {
                    expected: ncol,
                    found: row.len(),
                });
            }
        }
    }

    let mut out = Vec::with_capacity(nrow);
    let mut column = vec![0.0; values.len()];
    for r in 0..nrow {
        let mut row_stats = Vec::with_capacity(ncol);
        for c in 0..ncol {
            for (i, v) in values.iter().enumerate() {
                column[i] = v[r][c];
            }
            row_stats.push(weighted_stats(&column, weights)?);
        }
        out.push(row_stats);
    }
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// Multi-frame XYZ
// ─────────────────────────────────────────────────────────────────────────────

/// Parse a **multi-frame** XYZ file: concatenated standard XYZ blocks, the
/// format RDKit/OpenBabel emit for a conformer set.
///
/// ferric's [`Molecule::parse_xyz`] reads exactly one frame and silently ignores
/// everything after it, so feeding it a 50-conformer file yields conformer 0
/// with no warning. This reader consumes every frame.
///
/// Each frame is `<natoms>\n<comment>\n<natoms lines of "Sym x y z">`.
/// Coordinates are Angstroms, converted to Bohr, exactly as the single-frame
/// parser does. `charge` and `multiplicity` apply to all frames (a conformer set
/// is one species in one electronic state, by definition).
///
/// # Energies in the comment line
///
/// The comment line is *not* parsed for energies. Formats vary wildly
/// (`E = -155.4`, bare floats, RDKit's `MMFF94 -12.3`) and guessing would
/// silently mis-assign weights. Compute energies with ferric and pass them to
/// [`ConformerEnsemble::from_molecules_and_energies`], or read them yourself
/// via [`multi_xyz_comments`].
pub fn parse_multi_xyz(
    text: &str,
    charge: i32,
    multiplicity: usize,
) -> Result<Vec<Molecule>, FerricError> {
    let frames = split_multi_xyz(text)?;
    frames
        .into_iter()
        .enumerate()
        .map(|(i, frame)| {
            Molecule::parse_xyz(&frame, charge, multiplicity).map_err(|e| {
                FerricError::XyzParse(format!("frame {i} of multi-frame XYZ: {e}"))
            })
        })
        .collect()
}

/// Load a multi-frame XYZ file from disk. See [`parse_multi_xyz`].
pub fn load_multi_xyz(
    path: &str,
    charge: i32,
    multiplicity: usize,
) -> Result<Vec<Molecule>, FerricError> {
    let text = std::fs::read_to_string(path).map_err(|e| {
        FerricError::General(format!("cannot read multi-frame xyz file {path:?}: {e}"))
    })?;
    parse_multi_xyz(&text, charge, multiplicity)
}

/// The comment (second) line of each frame in a multi-frame XYZ, in file order.
///
/// Useful for callers who store energies or labels there and want to parse them
/// with their own, explicit convention.
pub fn multi_xyz_comments(text: &str) -> Result<Vec<String>, FerricError> {
    let frames = split_multi_xyz(text)?;
    Ok(frames
        .iter()
        .map(|f| f.lines().nth(1).unwrap_or("").trim().to_string())
        .collect())
}

/// Build an ensemble directly from a multi-frame XYZ file and matching energies.
///
/// Convenience wrapper: parse frames, then validate the shared-composition
/// invariant. The frame count must equal the energy count.
pub fn ensemble_from_multi_xyz(
    text: &str,
    charge: i32,
    multiplicity: usize,
    energies: &[f64],
) -> Result<ConformerEnsemble, FerricError> {
    let molecules = parse_multi_xyz(text, charge, multiplicity)?;
    Ok(ConformerEnsemble::from_molecules_and_energies(molecules, energies)?)
}

/// Split a multi-frame XYZ into per-frame text blocks.
///
/// Blank lines *between* frames are tolerated (many writers emit them); blank
/// lines inside a frame's atom block are not, since that would mean a missing
/// atom record and is exactly the corruption worth erroring on.
fn split_multi_xyz(text: &str) -> Result<Vec<String>, FerricError> {
    let lines: Vec<&str> = text.lines().collect();
    let mut frames = Vec::new();
    let mut i = 0usize;
    while i < lines.len() {
        // Skip inter-frame blank padding.
        if lines[i].trim().is_empty() {
            i += 1;
            continue;
        }
        let natoms: usize = lines[i].trim().parse().map_err(|e| {
            FerricError::XyzParse(format!(
                "multi-frame XYZ line {}: expected an atom count, got {:?} ({e})",
                i + 1,
                lines[i].trim()
            ))
        })?;
        // Header (count) + comment + natoms coordinate lines.
        let end = i + 2 + natoms;
        if end > lines.len() {
            return Err(FerricError::XyzParse(format!(
                "multi-frame XYZ: frame {} starting at line {} declares {natoms} atoms but the \
                 file ends after {} more lines",
                frames.len(),
                i + 1,
                lines.len().saturating_sub(i + 2)
            )));
        }
        let mut frame = String::new();
        for line in &lines[i..end] {
            frame.push_str(line);
            frame.push('\n');
        }
        frames.push(frame);
        i = end;
    }
    if frames.is_empty() {
        return Err(FerricError::XyzParse(
            "multi-frame XYZ contained no frames".into(),
        ));
    }
    Ok(frames)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const WATER: &str = "3\nwater\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n";

    fn water() -> Molecule {
        Molecule::parse_xyz(WATER, 0, 1).unwrap()
    }

    /// Water with the hydrogens listed before the oxygen: same composition,
    /// different ORDERING. This is the corruption case the invariant exists for.
    fn water_reordered() -> Molecule {
        let xyz = "3\nwater reordered\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\nO 0.000000 0.000000 0.117790\n";
        Molecule::parse_xyz(xyz, 0, 1).unwrap()
    }

    // ── Construction invariants ─────────────────────────────────────────────

    #[test]
    fn empty_ensemble_is_rejected() {
        assert_eq!(ConformerEnsemble::new(vec![]).unwrap_err(), ConformerError::Empty);
    }

    #[test]
    fn reordered_atoms_are_rejected() {
        let err = ConformerEnsemble::from_molecules(vec![water(), water_reordered()]).unwrap_err();
        match err {
            ConformerError::CompositionMismatch { index, atom, .. } => {
                assert_eq!(index, 1);
                assert_eq!(atom, 0, "first mismatching atom is index 0 (O vs H)");
            }
            other => panic!("expected CompositionMismatch, got {other:?}"),
        }
    }

    #[test]
    fn different_atom_count_is_rejected() {
        let h2 = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
        let err = ConformerEnsemble::from_molecules(vec![water(), h2]).unwrap_err();
        assert_eq!(
            err,
            ConformerError::AtomCountMismatch { index: 1, expected: 3, found: 2 }
        );
    }

    #[test]
    fn different_charge_or_multiplicity_is_rejected() {
        let cation = Molecule::parse_xyz(WATER, 1, 2).unwrap();
        let err = ConformerEnsemble::from_molecules(vec![water(), cation]).unwrap_err();
        match err {
            ConformerError::StateMismatch { index, expected, found } => {
                assert_eq!(index, 1);
                assert_eq!(expected, (0, 1));
                assert_eq!(found, (1, 2));
            }
            other => panic!("expected StateMismatch, got {other:?}"),
        }
    }

    #[test]
    fn ghost_vs_real_atom_is_rejected() {
        // Same symbol and Z, but a ghost contributes no electrons: not a conformer.
        let real = Molecule::parse_xyz("2\nHe2\nHe 0 0 0\nHe 0 0 3.0\n", 0, 1).unwrap();
        let ghosted = Molecule::parse_xyz("2\nHe+ghost\nHe 0 0 0\n@He 0 0 3.0\n", 0, 1).unwrap();
        let err = ConformerEnsemble::from_molecules(vec![real, ghosted]).unwrap_err();
        matches!(err, ConformerError::CompositionMismatch { atom: 1, .. });
    }

    #[test]
    fn push_rechecks_the_invariant() {
        let mut ens = ConformerEnsemble::from_molecules(vec![water()]).unwrap();
        assert!(ens.push(Conformer::new(water())).is_ok());
        assert_eq!(ens.len(), 2);
        assert!(ens.push(Conformer::new(water_reordered())).is_err());
        assert_eq!(ens.len(), 2, "a rejected push must not mutate the ensemble");
    }

    #[test]
    fn geometry_may_differ_freely() {
        // Same atoms in the same order, different coordinates: the whole point.
        let bent = Molecule::parse_xyz(
            "3\nbent water\nO 0.0 0.0 0.2\nH 0.0 0.9 -0.5\nH 0.0 -0.9 -0.5\n",
            0,
            1,
        )
        .unwrap();
        let ens = ConformerEnsemble::from_molecules(vec![water(), bent]).unwrap();
        assert_eq!(ens.len(), 2);
        assert_eq!(ens.n_atoms(), 3);
    }

    #[test]
    fn missing_energy_errors_clearly() {
        let ens = ConformerEnsemble::from_molecules(vec![water(), water()]).unwrap();
        assert_eq!(
            ens.energies().unwrap_err(),
            ConformerError::NonFiniteEnergy { index: 0 }
        );
    }

    // ── Boltzmann weights: hand-verified numbers ────────────────────────────

    #[test]
    fn single_conformer_has_weight_exactly_one() {
        let ens = ConformerEnsemble::from_molecules_and_energies(vec![water()], &[-76.02]).unwrap();
        let w = ens.boltzmann_weights_default().unwrap();
        assert_eq!(w.weights.len(), 1);
        assert_eq!(w.weights[0], 1.0, "single conformer must have weight EXACTLY 1.0");
        assert_eq!(w.partition_function, 1.0);
        assert_eq!(w.relative_energies[0], 0.0);
    }

    #[test]
    fn single_conformer_reproduces_the_single_point_answer_exactly() {
        // The property of a one-conformer ensemble must be the single-point
        // value bit-for-bit, with a standard deviation of exactly zero.
        let ens = ConformerEnsemble::from_molecules_and_energies(vec![water()], &[-76.026_760_1]).unwrap();
        let w = ens.boltzmann_weights_default().unwrap();
        let single_point_dipole = 0.812_345_678_901_23_f64;
        let stats = weighted_stats(&[single_point_dipole], &w.weights).unwrap();
        assert_eq!(
            stats.mean, single_point_dipole,
            "single-conformer mean must be bit-identical to the single-point value"
        );
        assert_eq!(stats.std_dev, 0.0, "single-conformer std dev must be EXACTLY 0");
        assert_eq!(stats.min, single_point_dipole);
        assert_eq!(stats.max, single_point_dipole);
    }

    #[test]
    fn two_degenerate_conformers_give_half_and_half() {
        let ens =
            ConformerEnsemble::from_molecules_and_energies(vec![water(), water()], &[-76.02, -76.02])
                .unwrap();
        let w = ens.boltzmann_weights_default().unwrap();
        assert_eq!(w.weights[0], 0.5, "degenerate pair must be exactly 0.5");
        assert_eq!(w.weights[1], 0.5);
        assert_eq!(w.partition_function, 2.0);
    }

    #[test]
    fn n_degenerate_conformers_give_uniform_weights() {
        let mols: Vec<Molecule> = (0..7).map(|_| water()).collect();
        let ens = ConformerEnsemble::from_molecules_and_energies(mols, &[-76.02; 7]).unwrap();
        let w = ens.boltzmann_weights_default().unwrap();
        for &wi in &w.weights {
            assert!((wi - 1.0 / 7.0).abs() < 1e-15, "expected 1/7, got {wi}");
        }
        let d = w.diagnostics();
        assert!(
            (d.effective_n_conformers - 7.0).abs() < 1e-12,
            "7 degenerate conformers must have effective N = 7, got {}",
            d.effective_n_conformers
        );
    }

    /// A conformer 10 kT above the minimum, checked against the closed-form
    /// hand computation, not merely asserted "small".
    ///
    /// Hand computation: with two conformers at ΔE = 0 and ΔE = 10 kT,
    ///   Z = exp(0) + exp(-10) = 1 + 4.5399929762484854e-05
    ///   w_hi = exp(-10)/Z = 4.5399929762484854e-05 / 1.0000453999297625
    ///        = 4.5397868702434395e-05
    ///   w_lo = 1 - w_hi   = 0.9999546021312976
    #[test]
    fn ten_kt_above_minimum_matches_hand_computation() {
        let kt = BOLTZMANN_HARTREE_PER_K * DEFAULT_TEMPERATURE_K;
        // kT at 298.15 K, hand-checked: 3.166811563455608e-6 * 298.15
        assert!(
            (kt - 9.441_848_676_442_895e-4).abs() < 1e-18,
            "kT at 298.15 K should be 9.441848676442895e-4 Ha, got {kt}"
        );

        let e_min = -76.0;
        let energies = [e_min, e_min + 10.0 * kt];
        let w = boltzmann_weights(&energies, DEFAULT_TEMPERATURE_K).unwrap();

        // Independently recomputed closed form (see doc comment above).
        let expected_hi = 4.539_786_870_243_439_5e-5;
        let expected_lo = 0.999_954_602_131_297_6;

        assert!(
            (w.weights[1] - expected_hi).abs() < 1e-15,
            "10 kT conformer weight should be {expected_hi:.17e}, got {:.17e}",
            w.weights[1]
        );
        assert!(
            (w.weights[0] - expected_lo).abs() < 1e-15,
            "minimum weight should be {expected_lo:.17e}, got {:.17e}",
            w.weights[0]
        );
        // "Negligible": under 0.005% of the population.
        assert!(w.weights[1] < 5e-5);
        // And the partition function is the hand value 1 + exp(-10).
        assert!((w.partition_function - (1.0 + (-10.0f64).exp())).abs() < 1e-15);
    }

    /// 1 kT and 2 kT spacings against exp(-1), exp(-2) by hand.
    #[test]
    fn one_and_two_kt_spacings_match_hand_computation() {
        let kt = BOLTZMANN_HARTREE_PER_K * DEFAULT_TEMPERATURE_K;
        let energies = [0.0, kt, 2.0 * kt];
        let w = boltzmann_weights(&energies, DEFAULT_TEMPERATURE_K).unwrap();
        let e1 = (-1.0f64).exp(); // 0.36787944117144233
        let e2 = (-2.0f64).exp(); // 0.1353352832366127
        let z = 1.0 + e1 + e2; // 1.503214724408055
        assert!((w.partition_function - z).abs() < 1e-14);
        assert!((w.weights[0] - 1.0 / z).abs() < 1e-14, "w0 should be 1/Z = {:.15}", 1.0 / z);
        assert!((w.weights[1] - e1 / z).abs() < 1e-14);
        assert!((w.weights[2] - e2 / z).abs() < 1e-14);
        // Literal expected values, so a change in the constant is caught.
        assert!((w.weights[0] - 0.665_240_955_774_042_2).abs() < 1e-12, "got {}", w.weights[0]);
        assert!((w.weights[1] - 0.244_728_471_054_635_5).abs() < 1e-12, "got {}", w.weights[1]);
        assert!((w.weights[2] - 0.090_030_573_171_322_3).abs() < 1e-12, "got {}", w.weights[2]);
    }

    #[test]
    fn weights_sum_to_one() {
        // Spread over several kT, deliberately unordered, minimum not first.
        let kt = BOLTZMANN_HARTREE_PER_K * DEFAULT_TEMPERATURE_K;
        let energies: Vec<f64> = vec![
            -155.0 + 3.0 * kt,
            -155.0 + 0.5 * kt,
            -155.0,
            -155.0 + 12.0 * kt,
            -155.0 + 1.7 * kt,
        ];
        let w = boltzmann_weights(&energies, DEFAULT_TEMPERATURE_K).unwrap();
        let sum: f64 = w.weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12, "weights must sum to 1 to 1e-12, got {sum:.18}");
        assert_eq!(w.min_index, 2, "minimum is at index 2, not index 0");
        assert!(w.relative_energies.iter().all(|&d| d >= 0.0));
        assert_eq!(w.relative_energies[2], 0.0);
    }

    #[test]
    fn emin_shift_survives_absolute_electronic_energies() {
        // Raw exp(-E/kT) on these overflows instantly: E/kT ~ 1.6e5.
        // The E_min-shifted form must be finite and correct.
        let energies = [-155.402_101, -155.400_100, -155.398_000];
        let w = boltzmann_weights(&energies, DEFAULT_TEMPERATURE_K).unwrap();
        assert!(w.weights.iter().all(|x| x.is_finite() && *x >= 0.0));
        let sum: f64 = w.weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12);
        // Lowest energy carries the largest weight.
        assert_eq!(w.min_index, 0);
        assert!(w.weights[0] > w.weights[1] && w.weights[1] > w.weights[2]);
        // Naive check that the unshifted form really would have overflowed.
        assert!((-energies[0] / w.kt_hartree).exp().is_infinite());
    }

    #[test]
    fn deeply_unfavourable_conformer_underflows_to_zero_not_nan() {
        let kt = BOLTZMANN_HARTREE_PER_K * DEFAULT_TEMPERATURE_K;
        let energies = [0.0, 5000.0 * kt];
        let w = boltzmann_weights(&energies, DEFAULT_TEMPERATURE_K).unwrap();
        assert_eq!(w.weights[1], 0.0, "exp(-5000) underflows to exactly 0, not NaN");
        assert_eq!(w.weights[0], 1.0);
        assert!(w.weights.iter().sum::<f64>() == 1.0);
    }

    #[test]
    fn temperature_must_be_positive_and_finite() {
        assert_eq!(
            boltzmann_weights(&[0.0], 0.0).unwrap_err(),
            ConformerError::BadTemperature(0.0)
        );
        assert!(matches!(
            boltzmann_weights(&[0.0], -5.0).unwrap_err(),
            ConformerError::BadTemperature(_)
        ));
        assert!(matches!(
            boltzmann_weights(&[0.0], f64::NAN).unwrap_err(),
            ConformerError::BadTemperature(_)
        ));
    }

    #[test]
    fn non_finite_energy_is_rejected() {
        assert_eq!(
            boltzmann_weights(&[0.0, f64::NAN], 298.15).unwrap_err(),
            ConformerError::NonFiniteEnergy { index: 1 }
        );
        assert_eq!(
            boltzmann_weights(&[0.0, f64::INFINITY], 298.15).unwrap_err(),
            ConformerError::NonFiniteEnergy { index: 1 }
        );
    }

    #[test]
    fn higher_temperature_flattens_the_populations() {
        let kt298 = BOLTZMANN_HARTREE_PER_K * DEFAULT_TEMPERATURE_K;
        let energies = [0.0, 3.0 * kt298];
        let cold = boltzmann_weights(&energies, 100.0).unwrap();
        let hot = boltzmann_weights(&energies, 1000.0).unwrap();
        assert!(
            hot.weights[1] > cold.weights[1],
            "the excited conformer must be more populated at higher T"
        );
        assert!(hot.diagnostics().max_weight < cold.diagnostics().max_weight);
    }

    // ── Weighted statistics ─────────────────────────────────────────────────

    #[test]
    fn weighted_mean_and_std_of_a_degenerate_pair() {
        // Two conformers at 0.5/0.5 with property values 1.0 and 3.0:
        //   mean = 2.0
        //   var  = 0.5*(1-2)^2 + 0.5*(3-2)^2 = 1.0  ->  std = 1.0
        let stats = weighted_stats(&[1.0, 3.0], &[0.5, 0.5]).unwrap();
        assert_eq!(stats.mean, 2.0);
        assert_eq!(stats.std_dev, 1.0);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 3.0);
    }

    #[test]
    fn weighted_std_is_zero_for_identical_values() {
        let stats = weighted_stats(&[2.5, 2.5, 2.5], &[0.2, 0.3, 0.5]).unwrap();
        assert_eq!(stats.mean, 2.5);
        assert_eq!(stats.std_dev, 0.0, "identical values must give EXACTLY zero spread");
    }

    #[test]
    fn weighted_std_handles_large_values_with_small_spread() {
        // The catastrophic-cancellation case E[x^2] - E[x]^2 gets wrong:
        // values ~ -155 Ha with a 1e-6 spread. Shifted form must be accurate.
        let values = [-155.400_001, -155.400_002, -155.400_003];
        let weights = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
        let stats = weighted_stats(&values, &weights).unwrap();

        // Uniform weights over {-d, 0, +d} about the centre give
        // var = 2d²/3, std = sqrt(2/3)·d. NOTE d is *not* exactly 1e-6: near
        // -155.4 the f64 ulp is 2.8e-14, so the literal spacing is actually
        // 9.99999997475242708e-7. Comparing against the ideal 1e-6 would fail
        // at the 2.5e-9 relative level for representation reasons alone, which
        // says nothing about the estimator. Use the spacing that really exists.
        let d = values[0] - values[1];
        let expected_std = (2.0f64 / 3.0).sqrt() * d;
        assert!(
            (stats.std_dev - expected_std).abs() < 1e-21,
            "expected std {expected_std:e}, got {:e}",
            stats.std_dev
        );
        // Cross-checked against an independent numpy computation of the same
        // three literals: 8.16496578866270349e-07.
        assert!(
            (stats.std_dev - 8.164_965_788_662_703e-7).abs() < 1e-21,
            "got {:e}",
            stats.std_dev
        );
        assert!(stats.std_dev > 0.0, "must not collapse to zero from cancellation");

        // The naive E[x²] - E[x]² form on these same values is the trap this
        // guards against: it loses every significant digit of the variance.
        let naive_var = values.iter().zip(&weights).map(|(&v, &w)| w * v * v).sum::<f64>()
            - stats.mean * stats.mean;
        assert!(
            (naive_var.max(0.0).sqrt() - expected_std).abs() > 1e-8,
            "the naive form was expected to be badly wrong here; if it is now accurate, \
             this test no longer demonstrates anything"
        );
    }

    #[test]
    fn weighted_stats_length_mismatch_errors() {
        assert!(matches!(
            weighted_stats(&[1.0, 2.0], &[1.0]).unwrap_err(),
            ConformerError::PropertyShapeMismatch { .. }
        ));
    }

    #[test]
    fn vector_property_is_averaged_component_wise() {
        // Two conformers with opposite dipoles at 0.5/0.5: the vector average
        // is zero but the per-component spread is large. This is exactly the
        // case a mean-without-spread would hide.
        let dipoles = vec![vec![1.0, 0.0, 0.5], vec![-1.0, 0.0, 0.5]];
        let stats = weighted_stats_vector(&dipoles, &[0.5, 0.5]).unwrap();
        assert_eq!(stats.len(), 3);
        assert_eq!(stats[0].mean, 0.0, "x components cancel");
        assert_eq!(stats[0].std_dev, 1.0, "but the spread is 1.0 -- the ensemble is NOT non-polar");
        assert_eq!(stats[2].mean, 0.5);
        assert_eq!(stats[2].std_dev, 0.0);
    }

    #[test]
    fn vector_property_shape_mismatch_errors() {
        let bad = vec![vec![1.0, 2.0], vec![1.0]];
        assert!(matches!(
            weighted_stats_vector(&bad, &[0.5, 0.5]).unwrap_err(),
            ConformerError::PropertyShapeMismatch { .. }
        ));
    }

    #[test]
    fn tensor_property_is_averaged_element_wise() {
        let a = vec![vec![1.0, 0.0, 0.0], vec![0.0, 2.0, 0.0], vec![0.0, 0.0, 3.0]];
        let b = vec![vec![3.0, 0.0, 0.0], vec![0.0, 2.0, 0.0], vec![0.0, 0.0, 1.0]];
        let stats = weighted_stats_tensor(&[a, b], &[0.5, 0.5]).unwrap();
        assert_eq!(stats.len(), 3);
        assert_eq!(stats[0][0].mean, 2.0);
        assert_eq!(stats[0][0].std_dev, 1.0);
        assert_eq!(stats[1][1].mean, 2.0);
        assert_eq!(stats[1][1].std_dev, 0.0);
        assert_eq!(stats[2][2].mean, 2.0);
    }

    // ── Diagnostics ─────────────────────────────────────────────────────────

    #[test]
    fn diagnostics_flag_a_dominated_ensemble() {
        let kt = BOLTZMANN_HARTREE_PER_K * DEFAULT_TEMPERATURE_K;
        // One minimum, everything else 10 kT up: totally dominated.
        let energies = [0.0, 10.0 * kt, 11.0 * kt, 12.0 * kt];
        let d = boltzmann_weights(&energies, DEFAULT_TEMPERATURE_K).unwrap().diagnostics();
        assert_eq!(d.n_conformers, 4);
        assert_eq!(d.n_within_kt, 1, "only the minimum itself is within kT");
        assert_eq!(d.n_within_2kt, 1);
        assert_eq!(d.n_within_5kt, 1);
        assert_eq!(d.max_weight_index, 0);
        assert!(d.max_weight > 0.999, "max weight {}", d.max_weight);
        assert!(d.is_single_conformer_dominated(0.95));
        assert!(d.effective_n_conformers < 1.001);
        assert!(d.verdict().contains("unnecessary"));
    }

    #[test]
    fn diagnostics_flag_a_flat_ensemble() {
        // Twenty conformers all within 0.5 kT: no single conformer is right.
        let kt = BOLTZMANN_HARTREE_PER_K * DEFAULT_TEMPERATURE_K;
        let energies: Vec<f64> = (0..20).map(|i| i as f64 * 0.025 * kt).collect();
        let d = boltzmann_weights(&energies, DEFAULT_TEMPERATURE_K).unwrap().diagnostics();
        assert_eq!(d.n_within_kt, 20);
        assert!(d.max_weight < 0.1, "max weight {}", d.max_weight);
        assert!(!d.is_single_conformer_dominated(0.95));
        assert!(d.effective_n_conformers > 19.0);
        assert!(d.verdict().contains("no conformer dominates"));
    }

    #[test]
    fn diagnostics_count_shells_correctly() {
        let kt = BOLTZMANN_HARTREE_PER_K * DEFAULT_TEMPERATURE_K;
        // ΔE / kT = 0, 0.5, 1.5, 3.0, 7.0
        let energies = [0.0, 0.5 * kt, 1.5 * kt, 3.0 * kt, 7.0 * kt];
        let d = boltzmann_weights(&energies, DEFAULT_TEMPERATURE_K).unwrap().diagnostics();
        assert_eq!(d.n_within_kt, 2, "0 and 0.5 kT");
        assert_eq!(d.n_within_2kt, 3, "+ 1.5 kT");
        assert_eq!(d.n_within_5kt, 4, "+ 3.0 kT; 7 kT is outside");
    }

    #[test]
    fn single_conformer_diagnostics_say_so() {
        let d = boltzmann_weights(&[-76.0], DEFAULT_TEMPERATURE_K).unwrap().diagnostics();
        assert_eq!(d.n_conformers, 1);
        assert_eq!(d.max_weight, 1.0);
        assert_eq!(d.effective_n_conformers, 1.0);
        assert!(d.verdict().contains("single-point"));
    }

    #[test]
    fn diagnostics_display_is_readable() {
        let d = boltzmann_weights(&[0.0, 1e-3], DEFAULT_TEMPERATURE_K).unwrap().diagnostics();
        let s = d.to_string();
        assert!(s.contains("max weight"));
        assert!(s.contains("verdict"));
        assert!(s.contains("298.15"));
    }

    // ── Multi-frame XYZ ─────────────────────────────────────────────────────

    const TWO_FRAME: &str = "\
3
frame 0 anti
O 0.000000 0.000000 0.117790
H 0.000000 0.755453 -0.471161
H 0.000000 -0.755453 -0.471161
3
frame 1 gauche
O 0.000000 0.000000 0.200000
H 0.000000 0.900000 -0.500000
H 0.000000 -0.900000 -0.500000
";

    #[test]
    fn single_frame_parser_silently_drops_extra_frames() {
        // Documents WHY this reader is needed: Molecule::parse_xyz on a
        // multi-frame file returns frame 0 and ignores the rest, with no error.
        let mol = Molecule::parse_xyz(TWO_FRAME, 0, 1).unwrap();
        assert_eq!(mol.atoms.len(), 3, "the single-frame parser sees only frame 0");
    }

    #[test]
    fn multi_xyz_reads_every_frame() {
        let mols = parse_multi_xyz(TWO_FRAME, 0, 1).unwrap();
        assert_eq!(mols.len(), 2);
        assert_eq!(mols[0].atoms.len(), 3);
        assert_eq!(mols[1].atoms.len(), 3);
        // Frame 1's oxygen is at a different z (0.2 Å -> Bohr).
        assert!((mols[0].atoms[0].zpos - 0.117_790 / 0.529_177_210_92).abs() < 1e-10);
        assert!((mols[1].atoms[0].zpos - 0.200_000 / 0.529_177_210_92).abs() < 1e-10);
    }

    #[test]
    fn multi_xyz_tolerates_blank_lines_between_frames() {
        let padded = TWO_FRAME.replace("3\nframe 1", "\n\n3\nframe 1");
        let mols = parse_multi_xyz(&padded, 0, 1).unwrap();
        assert_eq!(mols.len(), 2);
    }

    #[test]
    fn multi_xyz_comments_are_recoverable() {
        let comments = multi_xyz_comments(TWO_FRAME).unwrap();
        assert_eq!(comments, vec!["frame 0 anti", "frame 1 gauche"]);
    }

    #[test]
    fn truncated_multi_xyz_errors() {
        let truncated = "3\ntruncated\nO 0 0 0\nH 0 0 1\n";
        let err = parse_multi_xyz(truncated, 0, 1).unwrap_err();
        assert!(err.to_string().contains("declares 3 atoms"), "got: {err}");
    }

    #[test]
    fn garbage_count_line_errors() {
        let bad = "not-a-number\nc\nO 0 0 0\n";
        assert!(parse_multi_xyz(bad, 0, 1).is_err());
    }

    #[test]
    fn empty_multi_xyz_errors() {
        assert!(parse_multi_xyz("   \n\n", 0, 1).is_err());
    }

    #[test]
    fn ensemble_from_multi_xyz_end_to_end() {
        let kt = BOLTZMANN_HARTREE_PER_K * DEFAULT_TEMPERATURE_K;
        let ens = ensemble_from_multi_xyz(TWO_FRAME, 0, 1, &[-76.0, -76.0 + kt]).unwrap();
        assert_eq!(ens.len(), 2);
        let w = ens.boltzmann_weights_default().unwrap();
        let e1 = (-1.0f64).exp();

        // NOTE the tolerance: (-76.0 + kt) - (-76.0) does NOT round-trip to kt.
        // The ulp near -76 is 1.4e-14 against a kt of 9.4e-4, so the recovered
        // gap is 1.0000000000048 kT, giving a ~9.4e-13 deviation in the weight.
        // That is cancellation in this test's *energy construction*, not in the
        // weighting -- the exact-gap check below pins the estimator itself.
        assert!(
            (w.weights[1] - e1 / (1.0 + e1)).abs() < 1e-11,
            "got {:e}",
            w.weights[1]
        );
        assert!((w.weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);

        // Same physics with energies that ARE exactly one kT apart in f64
        // (0 and kt, no large offset to cancel against): here the closed form
        // must be reproduced to near machine precision.
        let exact = boltzmann_weights(&[0.0, kt], DEFAULT_TEMPERATURE_K).unwrap();
        assert!(
            (exact.weights[1] - e1 / (1.0 + e1)).abs() < 1e-15,
            "got {:e}",
            exact.weights[1]
        );
        assert!((exact.weights[0] - 1.0 / (1.0 + e1)).abs() < 1e-15);
    }

    #[test]
    fn ensemble_from_multi_xyz_energy_count_mismatch_errors() {
        assert!(ensemble_from_multi_xyz(TWO_FRAME, 0, 1, &[-76.0]).is_err());
    }

    #[test]
    fn set_energy_fills_in_later() {
        let mut ens = ConformerEnsemble::from_molecules(parse_multi_xyz(TWO_FRAME, 0, 1).unwrap())
            .unwrap();
        assert!(ens.energies().is_err());
        ens.set_energy(0, -76.0).unwrap();
        ens.set_energy(1, -75.999).unwrap();
        assert_eq!(ens.energies().unwrap(), vec![-76.0, -75.999]);
        assert!(ens.set_energy(2, -76.0).is_err(), "out-of-range index must error");
        assert!(ens.set_energy(0, f64::NAN).is_err(), "non-finite energy must error");
    }

    #[test]
    fn conformer_error_converts_to_ferric_error() {
        let e: FerricError = ConformerError::Empty.into();
        assert!(e.to_string().contains("empty"));
    }
}
