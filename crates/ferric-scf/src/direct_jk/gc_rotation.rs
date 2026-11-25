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
//! LinK, CFMM, the Hessian's second-derivative integrals and the periodic
//! code run the unrotated basis.
//!
//! # The switch, and who follows it
//!
//! The rule: **a build rotates only when a config asked for it.** A config
//! is an [`RhfConfig`](crate::rhf::RhfConfig) (its `gc_rotation` field) or a
//! CASSCF config's `gc_rotation` field (same type and meaning);
//! its value resolves as `Some(mode)` = `mode`, `None` = `FERRIC_GC_ROTATE`,
//! then the default `auto` (the repository's config > env > default order,
//! [`GcRotate::resolve`]). Values: `auto` = both stages, `rot1` = stage 1
//! only, `off` = the unrotated builder, bit for bit.
//!
//! * The SCF drivers record their config's value on the bounds
//!   ([`GcSlot::set_request`]); every build on those bounds (or a clone of
//!   them) then follows it: the SCF's own builders, the pooled Newton,
//!   TRAH, stability and response builds, the gradient's derivative walk.
//! * Bounds no SCF recorded on ([`SchwarzBounds::compute`] in a consumer
//!   that builds its own, the Python `jk` / `grad_jk` / `FockResponse`
//!   blocks) never rotate, whatever `FERRIC_GC_ROTATE` says: they are the
//!   unrotated builds bit for bit. A consumer that builds its own bounds
//!   for an SCF's geometry and wants the SCF's choice copies it
//!   ([`GcSlot::copy_request_from`], or [`GcSlot::set_request`] with the
//!   config's value): the KS Z-vector response (`zvector_ks`), the CASSCF
//!   direct backend and the GW exact exchange do.
//! * `FERRIC_GC_ROTATE` is read only where a config left the value unset,
//!   never on unrecorded bounds.
//!
//! The recorded value is shared by every clone of the bounds, and the last
//! writer wins (a second SCF on the same bounds re-records) for the
//! resolutions that follow it: a builder that has already resolved its
//! rotation (a `DirectJK`, `DirectJ` or `DirectK` past its first build)
//! keeps it, and a rotation already cached at a level stays cached.
//!
//! A change from phase 1: a standalone [`super::DirectJK`] on fresh bounds
//! (no SCF recorded on them, no request of its own) used to read
//! `FERRIC_GC_ROTATE` and rotate under `auto`; it now runs unrotated. Give
//! it `.with_gc_request(None)` to select by the variable (then `auto`), or
//! `.with_gc_rotation(mode)` for a fixed choice. A pooled build
//! rotates only when its engine pool says what it computes
//! ([`EnginePool::eri_key`]); a pool from `EnginePool::from_fn` has no key
//! and runs unrotated. A basis with nothing to rotate (def2, Pople, STO-3G,
//! pcseg) runs the unrotated builder bit for bit under every value. Both
//! the config field and the parsed variable are in the checkpoint identity
//! (`Route` class). The config spelling is strict (exactly `auto`, `rot1`
//! or `off`); the variable, like the other `FERRIC_*` switches, ignores
//! case and surrounding blanks.
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

/// Environment variable selecting the rotation where a config left it unset
/// ([`GcRotate::resolve`]); never read for unrecorded bounds (module doc).
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
/// ignored (the convention of the `FERRIC_*` switches); anything else is
/// refused.
static GC_ROTATE: ferric_core::config::ConfigVar<GcRotate> = ferric_core::config::ConfigVar {
    env_name: GC_ROTATE_ENV,
    default: GcRotate::Auto,
    parse: |s| GcRotate::parse_str(&s.trim().to_ascii_lowercase()),
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

    /// Strict parse of a config value (`[scf] gc_rotation`, the Python
    /// `gc_rotation=` kwarg): exactly `auto`, `rot1` or `off`. Anything
    /// else, other cases and padded values included, is an error.
    pub fn parse_str(s: &str) -> Result<Self, String> {
        match s {
            "auto" => Ok(Self::Auto),
            "rot1" => Ok(Self::Rot1),
            "off" => Ok(Self::Off),
            _ => Err(format!(
                "invalid value {s:?}: expected exactly \"auto\", \"rot1\" or \"off\""
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
        // The rotated walk runs the parent's J/K digest: share its record.
        rbounds.digest = bounds.digest.clone();
        let forward = basis.ao_transpose(DerivRotMutant::None);
        Ok(Some(Self {
            basis,
            bounds: rbounds,
            forward,
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
        let lock = || {
            self.pool
                .lock()
                .map_err(|_| FerricError::General("gc rotation: engine pool lock poisoned".into()))
        };
        if let Some((k, p)) = lock()?.as_ref() {
            if *k == key {
                return Ok(p.clone());
            }
        }
        // Built with the lock released (no lock is held across work that
        // may run on rayon), then inserted unless another caller won.
        let built = Arc::new(EnginePool::new(op, &self.basis.prep, precision)?);
        let mut g = lock()?;
        if let Some((k, p)) = g.as_ref() {
            if *k == key {
                return Ok(p.clone());
            }
        }
        *g = Some((key, built.clone()));
        Ok(built)
    }

    /// [`RotatedJk::charge`] agreed across the MPI ranks of `ctx`: the
    /// rotated walk reduces its partial matrices across the ranks, so a rank
    /// whose reservation fails must not leave the others waiting in that
    /// reduction. Every rank reserves, then all vote (a min all-reduce, as
    /// `df_k`'s Cholesky vote); unless every rank succeeded, every rank
    /// returns an error (its own, or one naming another rank) before any
    /// collective of the build. A no-op vote at one rank or without `mpi`.
    pub(crate) fn charge_collective(
        &self,
        ctx: &ferric_core::parallel::ParallelContext,
        count: usize,
    ) -> Result<ferric_core::memory::pool::Reservation, FerricError> {
        let local = self.charge(count);
        if all_ranks_agree(ctx, local.is_ok()) {
            return local;
        }
        match local {
            Err(e) => Err(e),
            Ok(_) => Err(FerricError::General(
                "direct J/K contraction-rotation buffers: the reservation failed on another MPI \
                 rank, so no rank runs the rotated build"
                    .into(),
            )),
        }
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

/// Whether `local` holds on every MPI rank of `ctx` (a min all-reduce;
/// `local` itself at one rank or without the `mpi` feature).
fn all_ranks_agree(ctx: &ferric_core::parallel::ParallelContext, local: bool) -> bool {
    #[cfg(feature = "mpi")]
    if let Some(world) = ctx.world() {
        use mpi::traits::CommunicatorCollectives;
        let mine = [i32::from(local)];
        let mut all = [0i32];
        world.all_reduce_into(
            &mine[..],
            &mut all[..],
            mpi::collective::SystemOperation::min(),
        );
        return all[0] == 1;
    }
    #[cfg(not(feature = "mpi"))]
    let _ = ctx;
    local
}

/// The contraction rotation cache of one [`SchwarzBounds`] (its `gc`
/// field): the rotation a config asked for, and the rotated basis, bounds
/// and engines of each level, built at the first rotated build and shared
/// by every later one on these bounds (clones of the bounds share it, as
/// they share the in-core store).
///
/// The request has three states: never recorded (fresh bounds: no build on
/// them rotates, whatever `FERRIC_GC_ROTATE` says), recorded unset (a
/// config left `gc_rotation` unset: `FERRIC_GC_ROTATE`, then `auto`) and
/// recorded value. The SCF drivers record their
/// [`RhfConfig::gc_rotation`](crate::rhf::RhfConfig::gc_rotation) here
/// ([`GcSlot::set_request`]), so the Newton, stability, response and
/// gradient builds that follow on the same bounds run the same rotation as
/// the SCF. The record is shared by every clone of the bounds; the last
/// writer wins.
#[derive(Clone, Default)]
pub struct GcSlot(Arc<GcSlotInner>);

/// What a config recorded on a [`GcSlot`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum Request {
    /// No config recorded anything: the builds run unrotated.
    #[default]
    Unrecorded,
    /// A config's `gc_rotation` (`None`: unset, the variable and the default
    /// decide).
    Recorded(Option<GcRotate>),
}

/// One cached level: the parent's `geometry_key` and the rotation (`None`:
/// nothing rotates).
type LevelEntry = (u64, Option<Arc<RotatedJk>>);

#[derive(Default)]
struct GcSlotInner {
    /// Bounds of a rotated basis: never rotate.
    inert: bool,
    /// `geometry_key` of the basis and operator the bounds were computed
    /// for ([`SchwarzBounds::compute`]); a rotation is served only for it.
    owner: Option<u64>,
    request: std::sync::RwLock<Request>,
    /// Per level (`[Ogc, Rot1]`): `None` = not detected yet.
    levels: std::sync::Mutex<[Option<LevelEntry>; 2]>,
}

impl std::fmt::Debug for GcSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GcSlot")
            .field("inert", &self.0.inert)
            .field("request", &self.request_state())
            .finish_non_exhaustive()
    }
}

impl GcSlot {
    /// A slot for the same geometry with nothing recorded and nothing
    /// cached: builds on bounds carrying it run unrotated
    /// ([`SchwarzBounds::unrotated`]).
    pub(crate) fn detached(&self) -> GcSlot {
        GcSlot(Arc::new(GcSlotInner {
            inert: self.0.inert,
            owner: self.0.owner,
            ..Default::default()
        }))
    }

    /// The slot of bounds computed for the geometry `key`
    /// (`incore_jk::geometry_key`).
    pub(crate) fn owned_by(key: u64) -> Self {
        GcSlot(Arc::new(GcSlotInner {
            owner: Some(key),
            ..Default::default()
        }))
    }

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

    fn request_state(&self) -> Request {
        self.0.request.read().map(|g| *g).unwrap_or_default()
    }

    /// What a config recorded: `None` when nothing was (the builds on these
    /// bounds run unrotated), `Some(value)` with the config's
    /// `gc_rotation` otherwise.
    pub fn recorded(&self) -> Option<Option<GcRotate>> {
        match self.request_state() {
            Request::Unrecorded => None,
            Request::Recorded(v) => Some(v),
        }
    }

    /// Record the value of a config's `gc_rotation` (`None`: unset, so
    /// `FERRIC_GC_ROTATE` and then `auto` decide). Shared by every clone of
    /// the bounds; the last writer wins.
    pub fn set_request(&self, mode: Option<GcRotate>) {
        if let Ok(mut g) = self.0.request.write() {
            *g = Request::Recorded(mode);
        }
    }

    /// Copy `other`'s record (or its absence) onto these bounds: for a
    /// consumer that builds its own bounds (another operator, say) for the
    /// geometry of an SCF's bounds and should follow the SCF's choice.
    pub fn copy_request_from(&self, other: &GcSlot) {
        let r = other.request_state();
        if let Ok(mut g) = self.0.request.write() {
            *g = r;
        }
    }

    /// The rotation the builds on these bounds run: `Off` for inert or
    /// unrecorded bounds, else the recorded value resolved
    /// ([`GcRotate::resolve`]). A malformed `FERRIC_GC_ROTATE` is an error
    /// only where it is read.
    pub fn mode(&self) -> Result<GcRotate, FerricError> {
        if self.0.inert {
            return Ok(GcRotate::Off);
        }
        match self.request_state() {
            Request::Unrecorded => Ok(GcRotate::Off),
            Request::Recorded(v) => GcRotate::resolve(v),
        }
    }

    /// The rotation of `prep` (the basis of `bounds`, whose slot this is)
    /// at `mode`, built on first use; `None` when `mode` is off, the slot is
    /// inert, or nothing rotates. An error when the slot holds a rotation of
    /// another geometry, basis or operator (bounds reused after the atoms
    /// moved): a stale rotation never answers.
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
        let key = crate::incore_jk::geometry_key(prep, bounds.op);
        let refuse = || {
            Err(FerricError::General(
                "gc rotation: the bounds were computed for another geometry, basis or \
                 operator; a rotation is never built or reused for another one (rebuild the \
                 bounds for this geometry)"
                    .into(),
            ))
        };
        // The owner first: a cold level must not be built (and cached) for a
        // geometry the bounds do not describe.
        if self.0.owner.is_some_and(|o| o != key) {
            return refuse();
        }
        let lock = || {
            self.0
                .levels
                .lock()
                .map_err(|_| FerricError::General("gc rotation: cache lock poisoned".into()))
        };
        let check = |(k, r): LevelEntry| {
            if k != key {
                return Err(FerricError::General(
                    "gc rotation: the bounds carry a rotation built for another geometry, \
                     basis or operator; it is never reused (rebuild the bounds for this \
                     geometry)"
                        .into(),
                ));
            }
            Ok(r)
        };
        if let Some(e) = lock()?[idx].clone() {
            return check(e);
        }
        // Built with the lock released: the rotated Schwarz table runs on
        // rayon, and a worker that steals another first use of this slot
        // must not block on a lock its own thread holds. Two first uses may
        // both build; the first to insert wins and both return its entry,
        // so every caller shares one `Arc`.
        let built = RotatedJk::new(prep, bounds, level)?.map(Arc::new);
        let mut g = lock()?;
        let entry = match &g[idx] {
            Some(winner) => winner.clone(),
            None => {
                if let Some(r) = built.as_ref().filter(|_| ferric_core::timing::enabled()) {
                    eprintln!(
                        "[timing] direct-jk gc rotation ({mode:?}): {} columns rotated; setup \
                         (detection + rotated Schwarz bounds) {:.3} s, once per geometry",
                        r.basis.n_rotated_columns, r.setup_seconds
                    );
                }
                g[idx] = Some((key, built));
                g[idx].clone().expect("just inserted")
            }
        };
        drop(g);
        check(entry)
    }

    /// The rotation the pooled direct J/K builds on `bounds` run
    /// ([`GcSlot::mode`]), with an engine pool on the rotated shells
    /// matching `pool` (same operator and precision); `None` when the
    /// builds run unrotated, including for a pool of unknown kind
    /// ([`EnginePool::eri_key`] `None`, a `from_fn` pool).
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
        let mode = bounds.gc.mode()?;
        let Some(r) = bounds.gc.rotation(prep, bounds, mode)? else {
            return Ok(None);
        };
        let pool = r.pool(op, precision)?;
        Ok(Some((r, pool)))
    }
}

/// The rotation of a single-matrix direct builder (`DirectJ`, `DirectK`):
/// the bounds' [`GcSlot::mode`], resolved at the builder's first build and
/// kept for its lifetime.
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
        let r = bounds.gc.rotation(prep, bounds, bounds.gc.mode()?)?;
        self.0 = Some(r.clone());
        Ok(r)
    }
}

/// Rotation state of a builder: requested (resolved at the first direct
/// build) or resolved.
pub(super) enum GcState {
    /// `None`: no request of its own, the bounds' [`GcSlot::mode`] decides;
    /// `Some(v)`: a config's `gc_rotation` value `v`
    /// ([`GcRotate::resolve`]).
    Pending(Option<Option<GcRotate>>),
    Resolved(Option<Arc<RotatedJk>>),
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::basis;
    use ferric_core::mol::Molecule;
    use ferric_integrals::operator::Operator;

    /// The rotation is built once per bounds: a second builder, a pooled
    /// build and a clone of the bounds all get the same cached basis, and
    /// the rotated bounds never rotate again.
    #[test]
    fn rotation_is_cached_per_bounds() {
        let (mol, prep) = water(0.1);
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

    fn water(z: f64) -> (Molecule, PreparedBasis) {
        let mol = Molecule::parse_xyz(
            &format!("3\nwater\nO 0 0 {z}\nH 0 0.76 -0.47\nH 0 -0.76 -0.47\n"),
            0,
            1,
        )
        .unwrap();
        let prep = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        (mol, prep)
    }

    /// Bounds no config recorded on run unrotated (whatever the variable
    /// says); a recorded value is what they run; a copy carries the record
    /// or its absence; clones share it, last writer wins.
    #[test]
    fn request_has_three_states() {
        let (_, prep) = water(0.1);
        let b = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
        assert_eq!(b.gc.recorded(), None);
        assert_eq!(b.gc.mode().unwrap(), GcRotate::Off);
        let fresh = SchwarzBounds::compute(Operator::erf(0.3), &prep).unwrap();
        fresh.gc.copy_request_from(&b.gc);
        assert_eq!(fresh.gc.recorded(), None);
        let clone = b.clone();
        b.gc.set_request(Some(GcRotate::Rot1));
        assert_eq!(clone.gc.recorded(), Some(Some(GcRotate::Rot1)));
        assert_eq!(clone.gc.mode().unwrap(), GcRotate::Rot1);
        clone.gc.set_request(Some(GcRotate::Off));
        assert_eq!(b.gc.mode().unwrap(), GcRotate::Off);
        b.gc.set_request(None);
        assert_eq!(b.gc.recorded(), Some(None));
        assert_eq!(b.gc.mode().unwrap(), GcRotate::env_setting().unwrap());
        fresh.gc.copy_request_from(&b.gc);
        assert_eq!(fresh.gc.recorded(), Some(None));
        // The rotated bounds are Off whatever is recorded on them.
        b.gc.set_request(Some(GcRotate::Auto));
        let r = b.gc.rotation(&prep, &b, GcRotate::Auto).unwrap().unwrap();
        r.bounds.gc.set_request(Some(GcRotate::Auto));
        assert_eq!(r.bounds.gc.mode().unwrap(), GcRotate::Off);
    }

    /// Bounds reused after the atoms moved refuse their cached rotation
    /// (the same basis and shell count, another geometry).
    #[test]
    fn rotation_of_another_geometry_is_refused() {
        let (_, prep) = water(0.1);
        let (_, moved) = water(0.15);
        assert_eq!(prep.nbasis(), moved.nbasis());
        let b = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
        assert!(b.gc.rotation(&prep, &b, GcRotate::Auto).unwrap().is_some());
        let e = b.gc.rotation(&moved, &b, GcRotate::Auto).unwrap_err();
        assert!(e.to_string().contains("another geometry"), "{e}");
        // A level not built yet is refused for the other geometry too (the
        // bounds' owner is checked before any build), and the owner's own
        // request at that level still builds afterwards.
        let e = b.gc.rotation(&moved, &b, GcRotate::Rot1).unwrap_err();
        assert!(e.to_string().contains("another geometry"), "{e}");
        assert!(b.gc.rotation(&prep, &b, GcRotate::Rot1).unwrap().is_some());
        // Fresh bounds, cold at every level: refused before any build.
        let c = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
        for m in [GcRotate::Auto, GcRotate::Rot1] {
            assert!(c.gc.rotation(&moved, &c, m).is_err(), "{m:?}");
        }
        assert!(c.gc.rotation(&prep, &c, GcRotate::Auto).unwrap().is_some());
    }

    /// Two bases that differ only in a contraction coefficient (same
    /// geometry, shells and exponents) are different geometries for the
    /// cache: the other one is refused, cold or warm.
    #[test]
    fn rotation_of_other_coefficients_is_refused() {
        let (mol, prep) = water(0.1);
        let mut bs = basis::bundled("cc-pvdz").unwrap();
        let sh = bs
            .shells
            .get_mut(&8)
            .unwrap()
            .iter_mut()
            .find(|sh| sh.coefficients.len() > 1)
            .unwrap();
        sh.coefficients[0] *= 1.0 + 1e-9;
        let other = PreparedBasis::new(&mol, &bs).unwrap();
        assert_eq!(prep.nbasis(), other.nbasis());
        assert_ne!(
            crate::incore_jk::geometry_key(&prep, Operator::coulomb()),
            crate::incore_jk::geometry_key(&other, Operator::coulomb())
        );
        let b = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
        assert!(b.gc.rotation(&other, &b, GcRotate::Auto).is_err());
        assert!(b.gc.rotation(&prep, &b, GcRotate::Auto).unwrap().is_some());
        assert!(b.gc.rotation(&other, &b, GcRotate::Auto).is_err());
    }

    /// Concurrent first uses from rayon workers (the rotated Schwarz table
    /// is itself rayon work) neither deadlock nor build two cached
    /// entries: every caller gets the same `Arc`, and so do concurrent
    /// first engine-pool requests.
    #[test]
    fn concurrent_first_use_shares_one_entry() {
        use rayon::prelude::*;
        let (_, prep) = water(0.1);
        let b = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let rs: Vec<Arc<RotatedJk>> = pool.install(|| {
            (0..8)
                .into_par_iter()
                .map(|_| b.gc.rotation(&prep, &b, GcRotate::Auto).unwrap().unwrap())
                .collect()
        });
        assert!(rs.iter().all(|r| Arc::ptr_eq(r, &rs[0])));
        let ps: Vec<Arc<EnginePool>> = pool.install(|| {
            (0..8)
                .into_par_iter()
                .map(|_| rs[0].pool(Operator::coulomb(), 1e-14).unwrap())
                .collect()
        });
        assert!(ps.iter().all(|p| Arc::ptr_eq(p, &ps[0])));
    }

    #[test]
    fn parse_is_strict_and_canonical() {
        assert_eq!(GcRotate::parse_str("auto"), Ok(GcRotate::Auto));
        assert_eq!(GcRotate::parse_str("rot1"), Ok(GcRotate::Rot1));
        assert_eq!(GcRotate::parse_str("off"), Ok(GcRotate::Off));
        for bad in [
            "", "on", "no", "none", "0", "false", "of", " auto ", "AUTO", "Rot1", "Off",
        ] {
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
