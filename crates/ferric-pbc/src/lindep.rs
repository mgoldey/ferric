//! Linear-dependence diagnostics for the periodic SCF drivers, and the
//! opt-in `exp_to_discard` basis filter (FINDINGS "Iteration 15 (Python,
//! linear dependence)", `reference/pbc/pbc_lindep.py`).
//!
//! # What the drivers do (unchanged by this module)
//!
//! The k-point drivers ([`crate::kscf`], [`crate::kuscf`]) cut, PER k, the
//! eigenvectors of the RAW (unnormalised) Hermitian `S(k)` whose eigenvalue
//! is below `lindep` (absolute, default `1e-6`). Kept counts MAY differ
//! between k-points and must: `eig S_supercell(Γ) = ∪_k eig S(k)` on a
//! Gamma-centred mesh, so an absolute per-k cut keeps exactly the space the
//! supercell's cut keeps (the k-mesh ≡ supercell anchor survives any
//! threshold). A diagonal-normalised cut or a per-k pivoted Cholesky breaks
//! that anchor (measured 9.4e-10 / 4e-7..2.6e-6 Ha per cell in the
//! prototype) and is deliberately NOT offered.
//!
//! The Gamma drivers go through `ferric_scf::rhf::canonical_orthogonalizer`,
//! which ALSO cuts on raw eigenvalues of the unnormalised real `S` (default
//! `LINDEP_THRESH = 1e-6`, overridable by `FERRIC_LINDEP_THRESH`), padding
//! the dropped directions back as zero MO columns with sentinel energy 1e6.
//! `ScfResult` carries no dropped count; [`LindepReport::from_real_overlap`]
//! reproduces the diagnostic from `S` for a caller who knows the threshold.
//! Wiring it into the Gamma result structs is a FOLLOW-UP (the effective
//! threshold is private to ferric-scf).
//!
//! # Noise floor
//!
//! `S(k) = Σ_L e^{ik·L} S_L` is formed by cancellation of O(1) real-space
//! terms, so its eigenvalue error is ABSOLUTE, `~ ε Σ_L |S_L|` (measured
//! 1e-15..1e-12). The report takes `floor = 1e3 · ε · Σ` and flags
//! ([`LindepReport::near_noise_floor`]) any threshold within `1e2` of it:
//! such a cut is a coin toss on roundoff and the kept space (and the
//! k-mesh ≡ supercell anchor) is no longer reproducible.
//!
//! `Σ` is ESTIMATED from the matrices the driver already has (no per-L
//! lattice data survives `periodic_hcore_kpts`): `Σ ≈ max_k max_μ Σ_ν
//! |S(k)_μν|`, the largest row ℓ1 norm over the mesh. By the triangle
//! inequality this is a LOWER bound on the true `max_μ Σ_L Σ_ν |S_L,μν|`,
//! EXACT at Γ when every overlap is non-negative (s-type / diffuse sets —
//! the functions that cause the dependence). With signed overlaps
//! (p/d functions) it can under-estimate, so the flag errs towards NOT
//! firing; a raised flag is always meaningful.
//!
//! # `exp_to_discard`
//!
//! PySCF `pbc.gto.Cell.exp_to_discard` semantics, OPT-IN only
//! ([`prepare_cell_basis`] with `Some(emin)`; `None` is byte-identical to
//! `PreparedBasis::new`): every primitive with exponent `< emin` is removed;
//! a contracted shell that keeps some primitives is renormalised to unit
//! self-overlap (ferric's convention, as at basis load); a shell that keeps
//! none is removed and listed. An element present in the cell that is left
//! with NO shell is a typed error ([`ExpToDiscardError::EmptiedElement`]) —
//! the atom would carry no basis function. It changes the basis (a
//! translation-invariant change, so the supercell anchor stays exact) and
//! costs the removed functions' whole contribution (+4.4 mHa per cell for
//! the prototype's diffuse s, ~1700x the canonical cut's cost), which is
//! why it is never applied automatically.

use crate::kscf::eigh_herm;
use crate::lattice::Cell;
use ferric_core::basis::{BasisSet, Shell};
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ndarray::Array2;
use ndarray_linalg::{Eigh, UPLO};
use num_complex::Complex64;
use std::fmt;

/// `floor = NOISE_FLOOR_FACTOR · ε · Σ_L |S_L|` (module doc).
pub const NOISE_FLOOR_FACTOR: f64 = 1e3;
/// A threshold below `NOISE_FLOOR_MARGIN · floor` is flagged.
pub const NOISE_FLOOR_MARGIN: f64 = 1e2;

/// The canonical cut at one k-point.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct KLindep {
    /// AO count (`S(k)` dimension).
    pub nao: usize,
    /// Eigenvectors kept (`λ >= threshold`).
    pub kept: usize,
    /// Smallest eigenvalue of `S(k)` (may be slightly negative: noise).
    pub min_eig: f64,
    /// Largest DROPPED eigenvalue, `None` if nothing was dropped.
    pub max_dropped: Option<f64>,
}

impl KLindep {
    /// Stats of the cut `λ >= threshold` on the eigenvalues `w`.
    pub fn from_eigenvalues(w: &[f64], threshold: f64) -> Self {
        let kept = w.iter().filter(|&&v| v >= threshold).count();
        let min_eig = w.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_dropped = w
            .iter()
            .cloned()
            .filter(|&v| !(v >= threshold))
            .fold(None, |m: Option<f64>, v| Some(m.map_or(v, |x| x.max(v))));
        Self {
            nao: w.len(),
            kept,
            min_eig,
            max_dropped,
        }
    }

    /// `nao − kept`.
    pub fn dropped(&self) -> usize {
        self.nao - self.kept
    }
}

/// Per-k canonical-orthogonaliser diagnostics of one SCF (module doc).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LindepReport {
    /// The absolute eigenvalue threshold applied to the raw `S(k)`.
    pub threshold: f64,
    /// One entry per mesh point, mesh order (time-reversal partners carry
    /// their representative's numbers: `S(−k) = S(k)*` has the same
    /// spectrum).
    pub per_k: Vec<KLindep>,
    /// `min_k kept(k)`.
    pub min_kept: usize,
    /// `max_k kept(k)`.
    pub max_kept: usize,
    /// `Σ_k kept(k)` — on a Gamma-centred mesh, the kept count of the
    /// equivalent supercell's Gamma cut.
    pub total_kept: usize,
    /// Estimate of `Σ_L |S_L|` (largest row ℓ1 norm of `S(k)` over the
    /// mesh; module doc).
    pub overlap_abs_sum: f64,
    /// `NOISE_FLOOR_FACTOR · ε · overlap_abs_sum`.
    pub noise_floor: f64,
    /// `threshold < NOISE_FLOOR_MARGIN · noise_floor`.
    pub near_noise_floor: bool,
}

/// Largest row ℓ1 norm `max_μ Σ_ν |S_μν|` of a complex matrix.
pub fn overlap_abs_row_sum(s: &Array2<Complex64>) -> f64 {
    s.rows()
        .into_iter()
        .map(|r| r.iter().map(|z| z.norm()).sum::<f64>())
        .fold(0.0_f64, f64::max)
}

impl LindepReport {
    /// Assemble the report from per-k stats and the `Σ_L |S_L|` estimate.
    pub fn from_parts(threshold: f64, per_k: Vec<KLindep>, overlap_abs_sum: f64) -> Self {
        let min_kept = per_k.iter().map(|k| k.kept).min().unwrap_or(0);
        let max_kept = per_k.iter().map(|k| k.kept).max().unwrap_or(0);
        let total_kept = per_k.iter().map(|k| k.kept).sum();
        let noise_floor = NOISE_FLOOR_FACTOR * f64::EPSILON * overlap_abs_sum;
        Self {
            threshold,
            per_k,
            min_kept,
            max_kept,
            total_kept,
            overlap_abs_sum,
            noise_floor,
            near_noise_floor: threshold < NOISE_FLOOR_MARGIN * noise_floor,
        }
    }

    /// Diagnose the cut `threshold` on every `S(k)` directly (one
    /// eigendecomposition per matrix; e.g. an explicit supercell's Γ `S`
    /// as a one-point list).
    pub fn from_overlaps(s: &[Array2<Complex64>], threshold: f64) -> Result<Self, FerricError> {
        let mut per_k = Vec::with_capacity(s.len());
        let mut sum = 0.0_f64;
        for (k, sk) in s.iter().enumerate() {
            let (w, _) = eigh_herm(&crate::hcore::kpoint::hermitize(sk))
                .map_err(|e| FerricError::Lapack(format!("S(k={k}) diag: {e}")))?;
            per_k.push(KLindep::from_eigenvalues(&w.to_vec(), threshold));
            sum = sum.max(overlap_abs_row_sum(sk));
        }
        Ok(Self::from_parts(threshold, per_k, sum))
    }

    /// Gamma (real) equivalent, e.g. the `periodic_hcore` `S` fed to
    /// `solve_rhf_injected` with the threshold that path uses
    /// (`1e-6` unless `FERRIC_LINDEP_THRESH` overrides it).
    pub fn from_real_overlap(s: &Array2<f64>, threshold: f64) -> Result<Self, FerricError> {
        let sym = (s + &s.t()) * 0.5;
        let (w, _) = sym
            .eigh(UPLO::Upper)
            .map_err(|e| FerricError::Lapack(format!("S diag: {e}")))?;
        let sum = s
            .rows()
            .into_iter()
            .map(|r| r.iter().map(|v| v.abs()).sum::<f64>())
            .fold(0.0_f64, f64::max);
        Ok(Self::from_parts(
            threshold,
            vec![KLindep::from_eigenvalues(&w.to_vec(), threshold)],
            sum,
        ))
    }

    /// `Σ_k (nao − kept(k))`.
    pub fn total_dropped(&self) -> usize {
        self.per_k.iter().map(KLindep::dropped).sum()
    }

    /// Print the noise-floor warning to stderr when flagged (the flag, not
    /// the text, is the tested contract).
    pub fn warn_if_near_noise_floor(&self, who: &str) {
        if self.near_noise_floor {
            eprintln!(
                "{who} WARNING: lindep threshold {:.1e} is within {NOISE_FLOOR_MARGIN:.0e}x of the \
                 S(k) noise floor {:.1e} (= {NOISE_FLOOR_FACTOR:.0e} eps x sum|S_L| ~ {:.3e}); \
                 eigenvalues near the cut are roundoff and the kept space is not reproducible \
                 (FINDINGS Iteration 15). Raise the threshold.",
                self.threshold, self.noise_floor, self.overlap_abs_sum
            );
        }
    }
}

// ---------------------------------------------------------------- exp_to_discard

/// One shell touched by [`exp_to_discard`].
#[derive(Debug, Clone, PartialEq)]
pub struct DiscardedShell {
    /// Element (atomic number key of the `BasisSet`).
    pub z: i32,
    /// Index of the shell in the ORIGINAL element shell list.
    pub shell_index: usize,
    /// Angular momentum.
    pub l: i32,
    /// Exponents removed from it.
    pub exponents_removed: Vec<f64>,
    /// `true` if the whole shell was removed; `false` if a contracted shell
    /// lost some primitives and was renormalised.
    pub whole_shell: bool,
}

/// What [`exp_to_discard`] removed.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ExpToDiscardReport {
    /// The exponent cut `emin`.
    pub exp_to_discard: f64,
    /// Touched shells, ordered by (z, shell_index).
    pub shells: Vec<DiscardedShell>,
}

impl ExpToDiscardReport {
    /// Whole shells removed.
    pub fn n_shells_removed(&self) -> usize {
        self.shells.iter().filter(|s| s.whole_shell).count()
    }

    /// Primitives removed (whole and partial shells).
    pub fn n_primitives_removed(&self) -> usize {
        self.shells.iter().map(|s| s.exponents_removed.len()).sum()
    }
}

/// Typed failure of [`exp_to_discard`].
#[derive(Debug, Clone, PartialEq)]
pub enum ExpToDiscardError {
    /// `emin` not finite and positive.
    InvalidThreshold(f64),
    /// An element of the cell is absent from the basis.
    MissingElement {
        /// Atomic number.
        z: i32,
    },
    /// Every shell of an element of the cell fell below `emin`.
    EmptiedElement {
        /// Atomic number.
        z: i32,
        /// Its shell count before the cut.
        n_shells: usize,
        /// The cut.
        emin: f64,
    },
}

impl fmt::Display for ExpToDiscardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidThreshold(e) => {
                write!(f, "exp_to_discard must be finite and > 0, got {e}")
            }
            Self::MissingElement { z } => {
                write!(
                    f,
                    "exp_to_discard: element Z = {z} of the cell is not in the basis"
                )
            }
            Self::EmptiedElement { z, n_shells, emin } => write!(
                f,
                "exp_to_discard {emin:e} removes all {n_shells} shells of element Z = {z}: the \
                 atom would carry no basis function"
            ),
        }
    }
}

impl std::error::Error for ExpToDiscardError {}

impl From<ExpToDiscardError> for FerricError {
    fn from(e: ExpToDiscardError) -> Self {
        FerricError::Basis(e.to_string())
    }
}

/// Unit self-overlap for coefficients over unit-normalised primitives
/// (same rule as `ferric_core::basis`'s private `renormalize_contraction`).
fn renormalize(l: i32, exps: &[f64], coefs: &mut [f64]) {
    let lf = l as f64;
    let mut s = 0.0_f64;
    for (a, ca) in exps.iter().zip(coefs.iter()) {
        for (b, cb) in exps.iter().zip(coefs.iter()) {
            s += ca * cb * (2.0 * (a * b).sqrt() / (a + b)).powf(lf + 1.5);
        }
    }
    if s > 0.0 {
        let scale = 1.0 / s.sqrt();
        for c in coefs.iter_mut() {
            *c *= scale;
        }
    }
}

/// Remove every primitive with exponent `< emin` from the shells of the
/// elements `zs` (module doc). Elements not in `zs` are copied unchanged;
/// ECP definitions are untouched. `emin` below every exponent returns a
/// basis EQUAL to `bs` and an empty report.
pub fn exp_to_discard(
    bs: &BasisSet,
    emin: f64,
    zs: &[i32],
) -> Result<(BasisSet, ExpToDiscardReport), ExpToDiscardError> {
    if !(emin.is_finite() && emin > 0.0) {
        return Err(ExpToDiscardError::InvalidThreshold(emin));
    }
    let mut zs: Vec<i32> = zs.to_vec();
    zs.sort_unstable();
    zs.dedup();
    let mut out = bs.clone();
    let mut report = ExpToDiscardReport {
        exp_to_discard: emin,
        shells: Vec::new(),
    };
    for &z in &zs {
        let shells = bs
            .for_element(z)
            .ok_or(ExpToDiscardError::MissingElement { z })?;
        let mut kept_shells: Vec<Shell> = Vec::with_capacity(shells.len());
        for (i, sh) in shells.iter().enumerate() {
            let keep: Vec<usize> = (0..sh.exponents.len())
                .filter(|&p| sh.exponents[p] >= emin)
                .collect();
            if keep.len() == sh.exponents.len() {
                kept_shells.push(sh.clone());
                continue;
            }
            let removed: Vec<f64> = (0..sh.exponents.len())
                .filter(|p| !keep.contains(p))
                .map(|p| sh.exponents[p])
                .collect();
            let whole = keep.is_empty();
            report.shells.push(DiscardedShell {
                z,
                shell_index: i,
                l: sh.l,
                exponents_removed: removed,
                whole_shell: whole,
            });
            if !whole {
                let exps: Vec<f64> = keep.iter().map(|&p| sh.exponents[p]).collect();
                let mut coefs: Vec<f64> = keep.iter().map(|&p| sh.coefficients[p]).collect();
                renormalize(sh.l, &exps, &mut coefs);
                kept_shells.push(Shell {
                    l: sh.l,
                    pure: sh.pure,
                    exponents: exps,
                    coefficients: coefs,
                });
            }
        }
        if kept_shells.is_empty() {
            return Err(ExpToDiscardError::EmptiedElement {
                z,
                n_shells: shells.len(),
                emin,
            });
        }
        out.shells.insert(z, kept_shells);
    }
    Ok((out, report))
}

/// Prepare the cell's AO basis, applying the OPT-IN `exp_to_discard`
/// (`None`: exactly `PreparedBasis::new(cell.mol(), bs)`, no report).
/// Every element of the cell (ghost atoms included — they carry basis
/// functions) is filtered.
pub fn prepare_cell_basis(
    cell: &Cell,
    bs: &BasisSet,
    exp_to_discard_min: Option<f64>,
) -> Result<(PreparedBasis, Option<ExpToDiscardReport>), FerricError> {
    match exp_to_discard_min {
        None => Ok((PreparedBasis::new(cell.mol(), bs)?, None)),
        Some(emin) => {
            let zs: Vec<i32> = cell.mol().atoms.iter().map(|a| a.z).collect();
            let (filtered, report) = exp_to_discard(bs, emin, &zs)?;
            Ok((PreparedBasis::new(cell.mol(), &filtered)?, Some(report)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn klindep_stats_and_noise_flag() {
        let k = KLindep::from_eigenvalues(&[-1e-15, 2e-7, 1e-3, 1.0], 1e-6);
        assert_eq!((k.nao, k.kept, k.dropped()), (4, 2, 2));
        assert_eq!(k.min_eig, -1e-15);
        assert_eq!(k.max_dropped, Some(2e-7));
        assert_eq!(
            KLindep::from_eigenvalues(&[1.0, 2.0], 1e-6).max_dropped,
            None
        );
        let r = LindepReport::from_parts(1e-12, vec![k.clone(), k], 50.0);
        assert_eq!((r.min_kept, r.max_kept, r.total_kept), (2, 2, 4));
        assert!(r.near_noise_floor, "floor {:e}", r.noise_floor);
        assert!(!LindepReport::from_parts(1e-6, vec![], 50.0).near_noise_floor);
    }
}
