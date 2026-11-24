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
//! # Who rotates, and the cache
//!
//! The rotated basis and its bounds live in the [`SchwarzBounds`] of the
//! geometry ([`GcSlot`], the `gc` field), built at the first rotated build
//! and shared by every later one on those bounds, so a geometry pays the
//! detection and the rotated Schwarz table once, not once per builder. The
//! consumers:
//!
//! * the SCF builders: [`super::DirectJK`] (closed and open shell, full and
//!   incremental), `DirectJ` and `DirectK` (the J or K half when the other
//!   comes from RI, LinK or COSX), and the in-core ERI store, which then
//!   holds the rotated integrals (`crate::incore_jk`);
//! * the pooled builds that follow an SCF on the same bounds:
//!   [`crate::rhf::build_jk`], [`crate::rhf::build_jk_with_pool`] and its
//!   homogeneous, spin and multi-density siblings, i.e. the RHF/UHF/ROHF
//!   Newton and TRAH steps, the stability analyses, the CPHF/CPKS of the
//!   Hessian and of properties, and the Z-vector and OO-MP2 J/K;
//! * the analytic gradient's four-centre derivative walk
//!   (`crate::gradient::gradient_rotation`): `T` is geometry-independent and
//!   keeps each shell on its atom, so the derivative integrals of the
//!   rotated shells contracted with the rotated densities give the same
//!   gradient.
//!
//! The SCF drivers record their config's request on the bounds
//! ([`GcSlot::set_request`]), so the builds that follow run the same
//! rotation as the SCF. LinK, CFMM, the Hessian's second-derivative
//! integrals and the periodic code run the unrotated basis.
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
use std::sync::Arc;

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

/// The rotation of one basis at one level: the rotated basis and `T`, its
/// screening bounds, the forward map and (lazily) an engine pool on the
/// rotated shells. Built once per [`SchwarzBounds`] (so once per geometry)
/// and shared, through [`GcSlot`], by every direct J/K build on those
/// bounds: the SCF's [`super::DirectJK`], then the Newton, stability and
/// response builds ([`crate::rhf::build_jk_with_pool`] and its siblings).
pub struct RotatedJk {
    pub(crate) basis: RotatedBasis,
    pub(crate) bounds: SchwarzBounds,
    forward: AoTranspose,
    /// Parent `(nbasis, nshells, op)`: the slot serves only that basis.
    parent: (usize, usize, ferric_integrals::operator::Operator),
    /// Engines on the rotated shells for the pooled builders, keyed by
    /// `(operator, precision, rayon thread count)` of the pool the caller
    /// handed in.
    pool: std::sync::Mutex<Option<(PoolKey, Arc<EnginePool>)>>,
    /// Wall seconds of the detection and the rotated bounds.
    pub(crate) setup_seconds: f64,
}

type PoolKey = (ferric_integrals::operator::Operator, u64, usize);

impl std::fmt::Debug for RotatedJk {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RotatedJk")
            .field("basis", &self.basis)
            .field("setup_seconds", &self.setup_seconds)
            .finish_non_exhaustive()
    }
}

impl RotatedJk {
    /// The rotation of `prep` at `level`, with bounds of the same kind as
    /// `bounds` on the rotated shells; `None` when nothing rotates.
    fn new(
        prep: &PreparedBasis,
        bounds: &SchwarzBounds,
        level: RotationLevel,
    ) -> Result<Option<Self>, FerricError> {
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
        let mut rbounds = SchwarzBounds::compute_for_screening(bounds.op, &basis.prep, kind)?;
        // The rotated shells are never rotated again (a stage-1 basis still
        // has long columns sharing an exponent list).
        rbounds.gc = GcSlot::inert();
        let forward = basis.ao_transpose(DerivRotMutant::None);
        Ok(Some(Self {
            basis,
            bounds: rbounds,
            forward,
            parent: (prep.nbasis(), prep.nshells(), bounds.op),
            pool: std::sync::Mutex::new(None),
            setup_seconds: t0.elapsed().as_secs_f64(),
        }))
    }

    /// `Tᵀ D T`.
    pub(crate) fn forward(&self, d: &Array2<f64>) -> Array2<f64> {
        self.forward.matrix(d)
    }

    /// `out += T X' Tᵀ`.
    pub(crate) fn back_add(
        &self,
        x: &Array2<f64>,
        out: &mut Array2<f64>,
    ) -> Result<(), FerricError> {
        self.basis.back_transform_add(x, out)
    }

    /// `T X' Tᵀ` as a new matrix.
    pub(crate) fn back(&self, x: &Array2<f64>) -> Result<Array2<f64>, FerricError> {
        let n = self.basis.nbasis();
        let mut out = Array2::<f64>::zeros((n, n));
        self.back_add(x, &mut out)?;
        Ok(out)
    }

    /// An engine pool on the rotated shells for `op` at `precision`, built
    /// once per key and shared (the engines are per-slot mutexed, so two
    /// builds may run on it at once).
    pub(crate) fn pool(
        &self,
        op: ferric_integrals::operator::Operator,
        precision: f64,
    ) -> Result<Arc<EnginePool>, FerricError> {
        let key = (op, precision.to_bits(), rayon::current_num_threads());
        let mut g = self
            .pool
            .lock()
            .map_err(|_| FerricError::General("gc rotation: engine pool lock poisoned".into()))?;
        if let Some((k, p)) = g.as_ref() {
            if *k == key {
                return Ok(p.clone());
            }
        }
        let p = Arc::new(EnginePool::new(op, &self.basis.prep, precision)?);
        *g = Some((key, p.clone()));
        Ok(p)
    }

    /// The `D'` and `X'` buffers of a rotated build (`count · n²` doubles),
    /// charged to the memory pool while the guard lives.
    pub(crate) fn charge(
        &self,
        count: usize,
    ) -> Result<ferric_core::memory::pool::Reservation, FerricError> {
        let n = self.basis.nbasis();
        ferric_core::memory::pool::reserve_global(
            &format!("direct J/K contraction-rotation buffers (n={n})"),
            count * n * n * std::mem::size_of::<f64>(),
        )
    }
}

/// The contraction rotation cache of one [`SchwarzBounds`] (its `gc`
/// field): the rotation a solver asked for, and the rotated basis, bounds
/// and engines of each level, built at the first rotated build and shared
/// by every later one on these bounds (clones of the bounds share it, as
/// they share the in-core store). The SCF drivers record their
/// [`RhfConfig::gc_rotation`](crate::rhf::RhfConfig::gc_rotation) here
/// ([`GcSlot::set_request`]), so the Newton, stability and response builds
/// that follow on the same bounds run the same rotation as the SCF.
#[derive(Clone, Default)]
pub struct GcSlot(Arc<GcSlotInner>);

#[derive(Default)]
struct GcSlotInner {
    /// Bounds of a rotated basis: never rotate.
    inert: bool,
    request: std::sync::RwLock<Option<GcRotate>>,
    /// Per level (`[Ogc, Rot1]`): `None` = not detected yet, `Some(None)` =
    /// nothing rotates.
    levels: std::sync::Mutex<[Option<Option<Arc<RotatedJk>>>; 2]>,
}

impl std::fmt::Debug for GcSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GcSlot")
            .field("inert", &self.0.inert)
            .field("request", &self.request())
            .finish_non_exhaustive()
    }
}

impl GcSlot {
    fn inert() -> Self {
        GcSlot(Arc::new(GcSlotInner {
            inert: true,
            ..Default::default()
        }))
    }

    /// Whether these are the bounds of a rotated basis (never rotated).
    pub(crate) fn is_inert(&self) -> bool {
        self.0.inert
    }

    /// The rotation requested for builds on these bounds (`None`: not set,
    /// `FERRIC_GC_ROTATE` then `auto` decide).
    pub fn request(&self) -> Option<GcRotate> {
        self.0.request.read().ok().and_then(|g| *g)
    }

    /// Record the rotation a solver's config asks for (`None`: leave it to
    /// `FERRIC_GC_ROTATE` and the default).
    pub fn set_request(&self, mode: Option<GcRotate>) {
        if let Ok(mut g) = self.0.request.write() {
            *g = mode;
        }
    }

    /// The rotation of `prep` (the basis of `bounds`, whose slot this is)
    /// at `mode`, built on first use; `None` when `mode` is off, the slot is
    /// inert, or nothing rotates.
    pub(crate) fn rotation(
        &self,
        prep: &PreparedBasis,
        bounds: &SchwarzBounds,
        mode: GcRotate,
    ) -> Result<Option<Arc<RotatedJk>>, FerricError> {
        if self.0.inert {
            return Ok(None);
        }
        let Some(level) = mode.level() else {
            return Ok(None);
        };
        let idx = match level {
            RotationLevel::Ogc => 0,
            RotationLevel::Rot1 => 1,
        };
        let mut g = self
            .0
            .levels
            .lock()
            .map_err(|_| FerricError::General("gc rotation: cache lock poisoned".into()))?;
        let r = match &g[idx] {
            Some(r) => r.clone(),
            None => {
                let r = RotatedJk::new(prep, bounds, level)?.map(Arc::new);
                if let Some(r) = r.as_ref().filter(|_| ferric_core::timing::enabled()) {
                    eprintln!(
                        "[timing] direct-jk gc rotation ({mode:?}): {} columns rotated; setup \
                         (detection + rotated Schwarz bounds) {:.3} s, once per geometry",
                        r.basis.n_rotated_columns, r.setup_seconds
                    );
                }
                g[idx] = Some(r.clone());
                r
            }
        };
        if let Some(r) = &r {
            if r.parent != (prep.nbasis(), prep.nshells(), bounds.op) {
                return Err(FerricError::General(
                    "gc rotation: the bounds carry a rotation built for another basis or \
                     operator (rebuild the bounds for this basis)"
                        .into(),
                ));
            }
        }
        Ok(r)
    }

    /// The rotation the pooled direct J/K builds on `bounds` run (the
    /// recorded request, then `FERRIC_GC_ROTATE`, then `auto`), with an
    /// engine pool on the rotated shells matching `pool` (same operator and
    /// precision); `None` when the builds run unrotated, including for a
    /// pool of unknown kind ([`EnginePool::eri_key`]).
    pub(crate) fn for_pool(
        bounds: &SchwarzBounds,
        prep: &PreparedBasis,
        pool: &EnginePool,
    ) -> Result<Option<(Arc<RotatedJk>, Arc<EnginePool>)>, FerricError> {
        let Some((op, precision)) = pool.eri_key() else {
            return Ok(None);
        };
        Self::for_engines(bounds, prep, op, precision)
    }

    /// [`GcSlot::for_pool`] for engines of `op` at `precision`.
    pub(crate) fn for_engines(
        bounds: &SchwarzBounds,
        prep: &PreparedBasis,
        op: ferric_integrals::operator::Operator,
        precision: f64,
    ) -> Result<Option<(Arc<RotatedJk>, Arc<EnginePool>)>, FerricError> {
        if bounds.gc.0.inert {
            return Ok(None);
        }
        let mode = GcRotate::resolve(bounds.gc.request())?;
        let Some(r) = bounds.gc.rotation(prep, bounds, mode)? else {
            return Ok(None);
        };
        let pool = r.pool(op, precision)?;
        Ok(Some((r, pool)))
    }
}

/// The rotation of a single-matrix direct builder (`DirectJ`, `DirectK`):
/// the bounds' recorded request (then `FERRIC_GC_ROTATE`, then `auto`),
/// resolved at the builder's first build and kept for its lifetime.
#[derive(Default)]
pub(crate) struct BuilderGc(Option<Option<Arc<RotatedJk>>>);

impl BuilderGc {
    /// The rotation this builder runs (`None`: unrotated).
    pub(crate) fn resolve(
        &mut self,
        prep: &PreparedBasis,
        bounds: &SchwarzBounds,
    ) -> Result<Option<Arc<RotatedJk>>, FerricError> {
        if let Some(r) = &self.0 {
            return Ok(r.clone());
        }
        let r = if bounds.gc.is_inert() {
            None
        } else {
            let mode = GcRotate::resolve(bounds.gc.request())?;
            bounds.gc.rotation(prep, bounds, mode)?
        };
        self.0 = Some(r.clone());
        Ok(r)
    }
}

/// Rotation state of a builder: requested (resolved at the first direct
/// build) or resolved.
pub(super) enum GcState {
    /// The config's request (`None`: not set, the bounds' recorded request,
    /// then the variable or the default decides).
    Pending(Option<GcRotate>),
    Resolved(Option<Arc<RotatedJk>>),
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The rotation is built once per bounds: a second builder, a pooled
    /// build and a clone of the bounds all get the same cached basis, and
    /// the rotated bounds never rotate again.
    #[test]
    fn rotation_is_cached_per_bounds() {
        use ferric_core::basis;
        use ferric_core::mol::Molecule;
        use ferric_integrals::operator::Operator;
        let mol = Molecule::parse_xyz(
            "3\nwater\nO 0 0 0.1\nH 0 0.76 -0.47\nH 0 -0.76 -0.47\n",
            0,
            1,
        )
        .unwrap();
        let prep = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
        let a = bounds
            .gc
            .rotation(&prep, &bounds, GcRotate::Auto)
            .unwrap()
            .unwrap();
        let b = bounds
            .gc
            .rotation(&prep, &bounds, GcRotate::Auto)
            .unwrap()
            .unwrap();
        assert!(Arc::ptr_eq(&a, &b));
        let clone = bounds.clone();
        let c = clone
            .gc
            .rotation(&prep, &clone, GcRotate::Auto)
            .unwrap()
            .unwrap();
        assert!(Arc::ptr_eq(&a, &c));
        // Stage 1 is its own entry.
        let r1 = bounds
            .gc
            .rotation(&prep, &bounds, GcRotate::Rot1)
            .unwrap()
            .unwrap();
        assert!(!Arc::ptr_eq(&a, &r1));
        assert!(bounds
            .gc
            .rotation(&prep, &bounds, GcRotate::Off)
            .unwrap()
            .is_none());
        // The rotated bounds are inert, at every level.
        assert!(a.bounds.gc.is_inert());
        for m in [GcRotate::Auto, GcRotate::Rot1] {
            assert!(a
                .bounds
                .gc
                .rotation(&a.basis.prep, &a.bounds, m)
                .unwrap()
                .is_none());
        }
        // The pooled engines are cached per (operator, precision, threads).
        let p1 = a.pool(Operator::coulomb(), 1e-14).unwrap();
        let p2 = a.pool(Operator::coulomb(), 1e-14).unwrap();
        assert!(Arc::ptr_eq(&p1, &p2));
        let p3 = a.pool(Operator::coulomb(), 1e-16).unwrap();
        assert!(!Arc::ptr_eq(&p1, &p3));
        // Another basis on the same bounds is refused, not served.
        let other = PreparedBasis::new(&mol, &basis::bundled("cc-pvtz").unwrap()).unwrap();
        assert!(bounds.gc.rotation(&other, &bounds, GcRotate::Auto).is_err());
    }

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
