//! Internal column rotation of generally contracted shells for the direct
//! four-centre J/K builder ([`super::DirectJK`]).
//!
//! # Why
//!
//! libint takes one segmented shell per contraction column, so a
//! correlation-consistent basis (cc-pVDZ C: s columns of 9, 9 and 1
//! primitives over one exponent list) recomputes the shared primitives per
//! column, and the diffuse primitive inside the long columns keeps every
//! tight × diffuse primitive pair alive at any distance. Rotating the
//! columns (`ferric_integrals::contraction_rotation`: the single-primitive
//! columns projected out of the long ones, then one optimized-general-
//! contraction elimination) leaves cc-pVDZ C with s 7, 7, 1 and p 3, 1
//! primitives and the same span. On hexane and octane / cc-pVDZ the
//! per-build direct J/K wall and libint thread time halve (measurements in
//! the book, section "Direct and incremental J/K builds").
//!
//! # What runs
//!
//! With `χ = T χ'` (parent AOs in terms of the rotated ones; `T` is block
//! diagonal per atom, `l` and exponent list, the identity elsewhere), for
//! any density
//!
//! ```text
//! D' = Tᵀ D T,     J(D) = T J'(D') Tᵀ,     K(D) = T K'(D') Tᵀ,
//! ```
//!
//! where `J'`, `K'` are the direct builds on the rotated shells. The
//! Schwarz table (and the CSB `M` or CSAM `X` table when the parent bounds
//! carry one), the density-max screen, the engine pool and its shell-pair
//! cache all belong to the rotated shells. Both transforms are serial and
//! `O(n² · g²)` with `g` the number of rows of `T` touching a function (at
//! most 3 for cc-pVDZ / cc-pVTZ), so the result is bitwise identical across
//! thread counts. The incremental (`ΔD`) build is the same formula on `ΔD`.
//!
//! The rotated basis, its Schwarz bounds and engines are built once per
//! builder, at its first direct build.
//!
//! # The switch
//!
//! `FERRIC_GC_ROTATE` ([`GcRotate::from_env`]): unset or `auto` = both
//! stages, `rot1` = stage 1 only, `off` / `0` / `false` = the unrotated
//! builder, bit for bit. A basis with nothing to rotate (def2, Pople,
//! STO-3G, pcseg) runs the unrotated builder bit for bit under every value.
//! The variable is in the checkpoint identity (`Route` class).

use crate::screening::{SchwarzBounds, ScreeningKind};
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::contraction_rotation::{
    AoTranspose, DerivRotMutant, RotatedBasis, RotationLevel,
};
use ferric_integrals::engine_pool::EnginePool;
use ndarray::Array2;

/// Environment variable selecting the rotation ([`GcRotate::from_env`]).
pub const GC_ROTATE_ENV: &str = "FERRIC_GC_ROTATE";

/// Which contraction rotation the direct four-centre J/K builder runs
/// (module doc).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GcRotate {
    /// Both stages: diffuse projection, then the optimized-general-
    /// contraction elimination. The default.
    #[default]
    Auto,
    /// Stage 1 only (the single-primitive columns projected out).
    Rot1,
    /// No rotation: the unrotated builder, bit for bit.
    Off,
}

impl GcRotate {
    /// `FERRIC_GC_ROTATE`: `off`, `0` or `false` (any case) → [`GcRotate::Off`];
    /// `rot1` → [`GcRotate::Rot1`]; unset or any other value (`auto`, `on`)
    /// → [`GcRotate::Auto`]. Same convention as `FERRIC_SCF_INCREMENTAL`.
    pub fn from_env() -> Self {
        Self::parse(std::env::var(GC_ROTATE_ENV).ok().as_deref())
    }

    /// The value of [`GcRotate::from_env`] for a variable value.
    pub fn parse(value: Option<&str>) -> Self {
        match value.map(str::to_ascii_lowercase).as_deref() {
            Some("off") | Some("0") | Some("false") => Self::Off,
            Some("rot1") => Self::Rot1,
            _ => Self::Auto,
        }
    }

    fn level(self) -> Option<RotationLevel> {
        match self {
            Self::Auto => Some(RotationLevel::Ogc),
            Self::Rot1 => Some(RotationLevel::Rot1),
            Self::Off => None,
        }
    }
}

/// The rotated side of a [`super::DirectJK`]: the rotated basis and `T`, its
/// screening bounds and engines, and the forward map.
pub(super) struct RotatedJk {
    pub(super) basis: RotatedBasis,
    pub(super) bounds: SchwarzBounds,
    pub(super) pool: Option<EnginePool>,
    forward: AoTranspose,
    /// Wall seconds of the detection and the rotated bounds.
    pub(super) setup_seconds: f64,
}

impl RotatedJk {
    /// The rotation of `prep` under `mode`, with bounds of the same kind as
    /// `bounds` on the rotated shells; `None` when `mode` is off or nothing
    /// rotates.
    pub(super) fn new(
        prep: &PreparedBasis,
        bounds: &SchwarzBounds,
        mode: GcRotate,
    ) -> Result<Option<Self>, FerricError> {
        let Some(level) = mode.level() else {
            return Ok(None);
        };
        let t0 = std::time::Instant::now();
        let Some(basis) = RotatedBasis::detect_prepared(prep, level)? else {
            return Ok(None);
        };
        let kind = if bounds.csb_m.is_some() {
            ScreeningKind::Csb
        } else if bounds.csam_x.is_some() {
            ScreeningKind::Csam
        } else {
            ScreeningKind::Schwarz
        };
        let rbounds = SchwarzBounds::compute_for_screening(bounds.op, &basis.prep, kind)?;
        let forward = basis.ao_transpose(DerivRotMutant::None);
        Ok(Some(Self {
            basis,
            bounds: rbounds,
            pool: None,
            forward,
            setup_seconds: t0.elapsed().as_secs_f64(),
        }))
    }

    /// `Tᵀ D T`.
    pub(super) fn forward(&self, d: &Array2<f64>) -> Array2<f64> {
        self.forward.matrix(d)
    }

    /// `out += T X' Tᵀ`.
    pub(super) fn add_back(
        &self,
        x_rot: &Array2<f64>,
        out: &mut Array2<f64>,
    ) -> Result<(), FerricError> {
        let x = self.basis.back_transform_general(x_rot)?;
        *out += &x;
        Ok(())
    }
}

/// Rotation state of a builder: requested (resolved at the first direct
/// build) or resolved.
pub(super) enum GcState {
    Pending(GcRotate),
    Resolved(Option<Box<RotatedJk>>),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_follows_the_switch_convention() {
        assert_eq!(GcRotate::parse(None), GcRotate::Auto);
        assert_eq!(GcRotate::parse(Some("auto")), GcRotate::Auto);
        assert_eq!(GcRotate::parse(Some("on")), GcRotate::Auto);
        assert_eq!(GcRotate::parse(Some("rot1")), GcRotate::Rot1);
        for off in ["off", "OFF", "0", "false", "False"] {
            assert_eq!(GcRotate::parse(Some(off)), GcRotate::Off, "{off}");
        }
    }
}
