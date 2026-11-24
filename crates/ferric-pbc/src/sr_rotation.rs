//! Column rotation of generally contracted orbital shells for the Gamma
//! short-range integral walks (design: `reference/pbc/sr-general-contraction-design.md`
//! §3 option (d) and §4 step 2).
//!
//! # What is rotated
//!
//! Correlation-consistent bases store the contraction COLUMNS of an atom's
//! s (and p) set as separate shells over one shared exponent list; the most
//! diffuse primitive is also a column of its own (cc-pVDZ C: s 9 primitives
//! in columns with 9, 9, 1 non-zero coefficients; p 4 primitives in columns
//! with 4, 1). A GROUP is the shells of one element template with the same
//! `l`, the same `pure` flag and bitwise the same exponent list. Inside a
//! group, every column `k` with more than one non-zero coefficient whose
//! support contains the primitive `q` of a single-primitive column `s` of
//! the group is replaced by
//!
//! ```text
//! k' = k − (c_kq / c_sq) s        (raw coefficients: c_kq set to exactly 0)
//! ```
//!
//! with its zero primitives DROPPED (the single-primitive columns used are
//! zero-dropped too). The span of the group is unchanged, and every rotated
//! column is then either wholly compact or wholly smooth at the default
//! range split (ω = 1, λ = 1), so the split walk needs one kept call per
//! pair instead of two, and each call has fewer primitives.
//!
//! # The back-transform (exact)
//!
//! libint renormalises every contraction it is given (`N = s(c)^{-1/2}`,
//! `s` = self-overlap over unit primitives, the range split's convention).
//! With `χ_k = N_k Σ_p c_kp g_p` for the parent and `χ'_k = N'_k Σ_{p∉Q} c_kp g_p`
//! for the rotated column (`Q` the removed primitives),
//!
//! ```text
//! χ_k = T_kk χ'_k + Σ_{s} T_ks χ_s,
//! T_kk = √(s(c'_k) / s(c_k)),     T_ks = (c_kq / c_sq) √(s(c_s) / s(c_k)),
//! ```
//!
//! a unit-lower-triangular-like map (diagonal `T_kk ≈ 1`) inside each group,
//! identity elsewhere. Every SR block is bilinear in the two orbitals and the
//! range split's compact/smooth partition is linear in the coefficient
//! vector over a fixed primitive set, so the SR sums transform EXACTLY:
//! `J3[μν, P] = Σ_{m,n} T_μm T_νn J3'[mn, P]`, `V_SR = T V'_SR Tᵀ` (function
//! by function: rotated shell `m` has the same `l` and AO layout as parent
//! shell `m`). The screen runs on the rotated shells' own bounds
//! (coefficients `c'` times `prim_norm`, i.e. the size of `T_kk χ'_k` in
//! parent units), so the kept triplet set differs from the unrotated walk's:
//! the result agrees with the unrotated build to the screening precision,
//! not bitwise.
//!
//! # Where it applies
//!
//! Only inside the SR 3-centre walk of [`crate::rsgdf::RsGdf::build`]
//! (unsplit and range split) and the hcore SR attraction of
//! [`crate::hcore::periodic_hcore`]: the rotated blocks are back-transformed
//! into the parent AO basis right after the walk. The SCF, S/T, the LR and
//! G = 0 parts, the metric and every exposed matrix stay in the parent
//! basis.
//!
//! # At k points
//!
//! The k-point ENERGY builds ([`crate::rsgdf::kpoint::KRsGdf::build`],
//! [`crate::hcore::kpoint::periodic_hcore_kpts`]) rotate the same way. `T` is
//! real and k-independent and acts on the AO indices only, so it commutes
//! with the Bloch phase sums: the SR residue bins `J3[r_L, r_T][μν]` and the
//! SR attraction `V_SR(k)` are back-transformed (a general pass,
//! `RotatedBasis::back_transform_rows`: the two orientations of a pair sit
//! in different bins, so the input is not symmetric under `μ ↔ ν`) before
//! the phase contraction, the LR and the G = 0 terms. The k-point force and
//! stress builds differentiate the rotated walk (module section "Forces and
//! stress": `Tᵀ W[r] T` per residue bin, `rsgdf::kpoint::kderiv`).
//!
//! # Forces and stress
//!
//! The Gamma force and stress walks differentiate the walk the energy ran.
//! `T` depends on the contraction coefficients only (no geometry), a rotated
//! column shares its centre and exponent list with its parent, so at every
//! geometry `χ_k = Σ_m T_km χ'_m` and `∂χ_k = Σ_m T_km ∂χ'_m` (a nuclear
//! derivative, and the strain derivative through the centres, act on the
//! centre only). Every SR block is bilinear in the two orbitals, hence for
//! any weight `W` over AO pairs (the fitted `Y[P, μν]` of the RS-GDF SR
//! 3-centre sum, the density of the hcore SR attraction)
//!
//! ```text
//! Σ_{μν} W_μν ∂J[μν] = Σ_{mn} W'_mn ∂J'[mn],     W' = Tᵀ W T,
//! ```
//!
//! so the derivative blocks are evaluated on the ROTATED shells and
//! contracted with the forward-transformed weight (`AoTranspose`); no
//! derivative block is back-transformed, and the screen is the energy's own
//! rotated screen. The aux basis, the metric, the LR and G = 0 terms stay in
//! the parent basis. Forces/stress with the rotation on equal those with it
//! off to the screening precision (`tests/pbc_sr_rotation.rs`).
//!
//! It is ON BY DEFAULT ([`SrColumnRotation::Auto`]) in the Gamma builds
//! (energy, force and stress) and the k-point energy, force and stress builds
//! (at k the derivative walks contract the rotated shells with `Tᵀ W[r] T`
//! per-residue weights, `rsgdf::kpoint::kderiv`), and silently off in
//! every build that does not implement it (the frozen
//! s1 oracles); an explicit
//! [`SrColumnRotation::On`] is refused by name there. Measured (FINDINGS, "Full
//! timing series on libint 2.13.1", Γ RHF cc-pVDZ / cc-pvdz-ri, range
//! split, gdf ω = 1, 6 threads): dry ice 25.9 → 22.4 s, diamond 7.84 →
//! 3.08 s; energies move ≤ 1.5e-11 Ha/cell at ω ≥ 0.7; bitwise across
//! thread counts.
//!
//! # Numerics
//!
//! * The Gamma back-transform is a fixed serial loop over (group, group)
//!   pairs on an exactly symmetric input; it writes each unordered function
//!   pair once and COPIES it into both `(μ, ν)` and `(ν, μ)`, so the output
//!   is exactly symmetric and bitwise identical across thread counts (the
//!   walks already are). The k-point back-transform (`back_transform_rows`)
//!   takes a non-symmetric input and forms every ordered element from its
//!   own two rows with the same fixed addend order; it is serial per tensor
//!   (the SR bins are processed in parallel, one bin per task), so it is
//!   bitwise across thread counts too, and its output is symmetric (or
//!   Hermitian) to round-off only.
//! * Identity: when nothing in the basis rotates (segmented bases, STO-3G,
//!   Pople sp shells — no group has a single-primitive column), detection
//!   returns `None` and the build runs today's code path bit for bit.
//! * Guard: a column whose `T_kk < `[`COLUMN_ROTATION_MIN_DIAG`] or with an
//!   `|T_ks| > `[`COLUMN_ROTATION_MAX_COEF`] is left unrotated (its row of
//!   `T` stays the identity); cc-pVDZ C has `T_ks` 0.0032 and 0.58.
//!
//! # Where the code lives
//!
//! The rotation itself (detection, `T`, the rotated shells and the
//! back-transforms) is `ferric_integrals::contraction_rotation`, shared with
//! the direct SCF J/K builder; the periodic walks use its stage 1 only
//! (`RotationLevel::Rot1`). This module keeps the periodic request types and
//! the Cell-level entry points.

use crate::lattice::Cell;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::contraction_rotation::RotationLevel;
pub(crate) use ferric_integrals::contraction_rotation::{
    AoTranspose, DerivRotMutant, RotatedBasis,
};
pub use ferric_integrals::contraction_rotation::{
    ColumnRotationMutant, COLUMN_ROTATION_MAX_COEF, COLUMN_ROTATION_MIN_DIAG,
};
use num_complex::Complex64;

/// The column rotation a Gamma SR walk runs
/// ([`SrColumnRotation::On`]; production or a mutation-test variant).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColumnRotation {
    /// [`ColumnRotationMutant::Production`] except in mutation tests.
    pub mutant: ColumnRotationMutant,
}

impl ColumnRotation {
    /// The production rotation.
    pub const fn new() -> Self {
        Self {
            mutant: ColumnRotationMutant::Production,
        }
    }
}

/// The column-rotation request of a build
/// ([`crate::rsgdf::RsGdfConfig::sr_column_rotation`],
/// [`crate::hcore::PeriodicHcoreConfig::sr_column_rotation`]).
///
/// * [`SrColumnRotation::Auto`] (the default): ON in the builds that
///   implement it — the k-point energy, force and stress builds and the Gamma
///   builds [`crate::rsgdf::RsGdf::build`]
///   (and `build_with_fit_parts`, [`crate::rsgdf::sr_walk_counts`],
///   [`crate::rsgdf::RsGdf::build_for_gradient`]) and
///   [`crate::hcore::periodic_hcore`] — and silently OFF in every build that
///   does not: the frozen s1 oracles. A basis with
///   nothing to rotate runs the unrotated walk bit for bit either way.
/// * [`SrColumnRotation::Off`]: the unrotated walk everywhere, bit for bit.
/// * [`SrColumnRotation::On`]: an explicit request; the builds that cannot
///   honour it refuse it by name instead of ignoring it.
///
/// The Gamma and k-point force and stress builders differentiate the walk the
/// energy ran (module doc, "Forces and stress"): the hcore's
/// [`crate::hcore::PeriodicHcore::sr_rotated_columns`] /
/// [`crate::hcore::kpoint::PeriodicHcoreK::sr_rotated_columns`] and the
/// [`crate::rsgdf::RsGdf`]'s / [`crate::rsgdf::kpoint::KRsGdf`]'s rotation say
/// which; the config must agree with the build (typed error otherwise).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SrColumnRotation {
    /// On where implemented, off elsewhere (see the type doc).
    #[default]
    Auto,
    /// The unrotated walk everywhere.
    Off,
    /// Explicit request (refused where it cannot be honoured).
    On(ColumnRotation),
}

impl SrColumnRotation {
    /// Explicit production rotation.
    pub const fn on() -> Self {
        Self::On(ColumnRotation::new())
    }

    /// Whether this is an explicit [`SrColumnRotation::On`].
    pub fn is_explicit_on(self) -> bool {
        matches!(self, Self::On(_))
    }

    /// The rotation a build that IMPLEMENTS it runs: `Auto` → the production
    /// rotation, `On(r)` → `r`, `Off` → `None`.
    pub fn resolve_supported(self) -> Option<ColumnRotation> {
        match self {
            Self::Auto => Some(ColumnRotation::new()),
            Self::Off => None,
            Self::On(r) => Some(r),
        }
    }

    /// For a build that does NOT implement it: `Auto` and `Off` resolve to
    /// off (`Ok(())`); an explicit `On` is a typed refusal naming `who` and
    /// `why`.
    pub fn refuse_explicit(self, who: &str, why: &str) -> Result<(), FerricError> {
        match self {
            Self::Auto | Self::Off => Ok(()),
            Self::On(r) => Err(FerricError::General(format!(
                "{who}: sr_column_rotation {:?} was requested explicitly, but {why} \
                 (use SrColumnRotation::Auto or Off)",
                r.mutant
            ))),
        }
    }
}

impl Default for ColumnRotation {
    /// Same as [`ColumnRotation::new`].
    fn default() -> Self {
        Self::new()
    }
}

/// The stage-1 rotation of `prep` (built from `cell.mol()` and its own
/// `basis_set()`), or `None` when nothing rotates (the identity: the caller
/// then runs the unrotated walk, bit for bit). `who` names the caller in
/// errors.
pub(crate) fn detect(
    cell: &Cell,
    prep: &PreparedBasis,
    rot: ColumnRotation,
    who: &str,
) -> Result<Option<RotatedBasis>, FerricError> {
    RotatedBasis::detect(cell.mol(), prep, rot.mutant, RotationLevel::Rot1, who)
}

/// [`RotatedBasis::back_transform_rows`] of a COMPLEX `(n², w)` row-major
/// tensor (`w` complex columns; an `(n, n)` matrix is `w = 1`): real and
/// imaginary parts transform alike because `T` is real.
pub(crate) fn back_transform_complex_rows(
    rb: &RotatedBasis,
    data: &mut [Complex64],
    w: usize,
) -> Result<(), FerricError> {
    let mut flat: Vec<f64> = data.iter().flat_map(|z| [z.re, z.im]).collect();
    rb.back_transform_rows(&mut flat, 2 * w)?;
    for (z, i) in data.iter_mut().zip((0..).step_by(2)) {
        *z = Complex64::new(flat[i], flat[i + 1]);
    }
    Ok(())
}
