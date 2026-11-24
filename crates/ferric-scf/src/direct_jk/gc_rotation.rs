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
//! [`RhfConfig::gc_rotation`](crate::rhf::RhfConfig::gc_rotation) (typed,
//! per job; `None` = not set), then `FERRIC_GC_ROTATE`, then the default
//! `auto`: the repository's config > env > default order
//! ([`GcRotate::resolve`]). Values: `auto` = both stages, `rot1` = stage 1
//! only, `off` = the unrotated builder, bit for bit. Any other value of the
//! variable is an error naming it. A basis with nothing to rotate (def2,
//! Pople, STO-3G, pcseg) runs the unrotated builder bit for bit under every
//! value. Both the config field and the parsed variable are in the
//! checkpoint identity (`Route` class).
//!
//! The design called the switch `column_rotation`; it is `gc_rotation` here
//! because the periodic code already has a `sr_column_rotation` with
//! different semantics (stage 1 only, explicit `On` refused where not
//! implemented).

use crate::screening::{SchwarzBounds, ScreeningKind};
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::contraction_rotation::{
    AoTranspose, DerivRotMutant, RotatedBasis, RotationLevel,
};
use ferric_integrals::engine_pool::EnginePool;
use ndarray::Array2;

/// Environment variable selecting the rotation when the config does not
/// ([`GcRotate::resolve`]).
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

/// `FERRIC_GC_ROTATE`: `auto`, `rot1` or `off`, case-insensitive, blanks
/// ignored; anything else is refused.
static GC_ROTATE: ferric_core::config::ConfigVar<GcRotate> = ferric_core::config::ConfigVar {
    env_name: GC_ROTATE_ENV,
    default: GcRotate::Auto,
    parse: GcRotate::parse_str,
    validate: ferric_core::config::accept_any,
};

impl GcRotate {
    /// The canonical spelling (`auto` / `rot1` / `off`).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Rot1 => "rot1",
            Self::Off => "off",
        }
    }

    /// Parse a spelling: `auto`, `rot1` or `off` (case-insensitive,
    /// surrounding blanks ignored). Anything else is an error.
    pub fn parse_str(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "rot1" => Ok(Self::Rot1),
            "off" => Ok(Self::Off),
            _ => Err(format!(
                "invalid value {s:?}: expected \"auto\", \"rot1\" or \"off\" (unset means \"auto\")"
            )),
        }
    }

    /// The rotation a builder runs: `explicit` (the config) if set, else
    /// `FERRIC_GC_ROTATE`, else [`GcRotate::Auto`]. A malformed variable is
    /// an error naming it.
    pub fn resolve(explicit: Option<GcRotate>) -> Result<Self, FerricError> {
        GC_ROTATE
            .resolve(explicit, ferric_core::config::env_lookup)
            .map(|r| r.value)
            .map_err(|e| FerricError::General(format!("{GC_ROTATE_ENV}: {e}")))
    }

    /// The parsed `FERRIC_GC_ROTATE` of this process (unset = `auto`).
    pub fn env_setting() -> Result<Self, FerricError> {
        Self::resolve(None)
    }

    fn level(self) -> Option<RotationLevel> {
        match self {
            Self::Auto => Some(RotationLevel::Ogc),
            Self::Rot1 => Some(RotationLevel::Rot1),
            Self::Off => None,
        }
    }
}

impl std::fmt::Display for GcRotate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
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
}

/// Rotation state of a builder: requested (resolved at the first direct
/// build) or resolved.
pub(super) enum GcState {
    /// The config's request (`None`: not set, the variable or the default
    /// decides).
    Pending(Option<GcRotate>),
    Resolved(Option<Box<RotatedJk>>),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_is_strict_and_canonical() {
        assert_eq!(GcRotate::parse_str("auto"), Ok(GcRotate::Auto));
        assert_eq!(GcRotate::parse_str(" AUTO "), Ok(GcRotate::Auto));
        assert_eq!(GcRotate::parse_str("Rot1"), Ok(GcRotate::Rot1));
        assert_eq!(GcRotate::parse_str("off"), Ok(GcRotate::Off));
        for bad in ["", "on", "no", "none", "0", "false", "of"] {
            assert!(GcRotate::parse_str(bad).is_err(), "{bad:?}");
        }
        for m in [GcRotate::Auto, GcRotate::Rot1, GcRotate::Off] {
            assert_eq!(GcRotate::parse_str(m.as_str()), Ok(m));
        }
        // An explicit (config) value wins over the variable.
        assert_eq!(
            GcRotate::resolve(Some(GcRotate::Rot1)).unwrap(),
            GcRotate::Rot1
        );
    }
}
