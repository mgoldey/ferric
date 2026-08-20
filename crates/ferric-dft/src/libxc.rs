//! Safe wrapper around libxc for ferric.
//!
//! Uses hand-crafted FFI declarations (opaque pointer strategy) to avoid
//! requiring `libclang` / `bindgen` at build time. The linker finds libxc
//! via `cargo:rustc-link-lib=xc` emitted from `build.rs`.
//!
//! `XcFunctional` is an RAII handle that owns a libxc `xc_func_type` allocated
//! via `xc_func_alloc`. `XcDef` aggregates one or more functionals (e.g.
//! PBE = GGA_X_PBE + GGA_C_PBE, wB97X-V = HYB_GGA_XC_WB97X_V + VV10) plus
//! optional CAM (range-separation) and VV10 parameters.

use std::ffi::CString;
use std::os::raw::c_int;

use rayon::prelude::*;
use thiserror::Error;

// ---------------------------------------------------------------------------
// Raw FFI declarations (opaque pointer to xc_func_type).
// We use *mut c_void so we never need to know the struct layout.
// ---------------------------------------------------------------------------

mod ffi {
    use std::os::raw::{c_char, c_int};

    // xc_func_type is a C struct whose layout involves deeply nested macros.
    // We treat it as fully opaque and always work through pointers returned
    // by xc_func_alloc / xc_func_free, which is the supported libxc pattern.
    #[repr(C)]
    pub struct XcFuncOpaque {
        _private: [u8; 0],
    }

    /// Opaque `xc_func_info_type`. Borrowed from a live `xc_func_type`; libxc
    /// owns it statically for the process lifetime, so it is never freed here.
    #[repr(C)]
    pub struct XcFuncInfoOpaque {
        _private: [u8; 0],
    }

    extern "C" {
        /// Allocate an uninitialized libxc functional handle.
        pub fn xc_func_alloc() -> *mut XcFuncOpaque;
        /// Initialize a functional by numeric ID and spin count (1=unpolarized, 2=polarized).
        pub fn xc_func_init(p: *mut XcFuncOpaque, functional: c_int, nspin: c_int) -> c_int;
        /// Release internal data of an initialized functional (does not free the handle).
        pub fn xc_func_end(p: *mut XcFuncOpaque);
        /// Free the handle allocated by [`xc_func_alloc`].
        pub fn xc_func_free(p: *mut XcFuncOpaque);

        /// Look up a functional's numeric ID by its string name (e.g. `"gga_x_pbe"`).
        pub fn xc_functional_get_number(name: *const c_char) -> c_int;

        /// Override a functional's built-in external parameters. Signature from
        /// libxc 5.2.3 `src/xc.h`:
        /// ```c
        /// void xc_func_set_ext_params(xc_func_type *p, const double *ext_params);
        /// ```
        /// `ext_params` must point to EXACTLY `xc_func_info_get_n_ext_params(info)`
        /// doubles — libxc reads that many with no length argument and no bounds
        /// check, so a short buffer is an out-of-bounds read. Callers must verify
        /// the count first (see [`XcFunctional::set_ext_params`]).
        pub fn xc_func_set_ext_params(p: *mut XcFuncOpaque, ext_params: *const f64);

        /// Opaque handle to a functional's static info record.
        pub fn xc_func_get_info(p: *const XcFuncOpaque) -> *const XcFuncInfoOpaque;

        /// Number of external parameters this functional accepts.
        pub fn xc_func_info_get_n_ext_params(info: *const XcFuncInfoOpaque) -> c_int;

        /// Name of external parameter `number` (e.g. `"_cx0"`). Borrowed, static.
        pub fn xc_func_info_get_ext_params_name(
            info: *const XcFuncInfoOpaque,
            number: c_int,
        ) -> *const c_char;

        /// libxc's built-in default for external parameter `number`.
        pub fn xc_func_info_get_ext_params_default_value(
            info: *const XcFuncInfoOpaque,
            number: c_int,
        ) -> f64;

        // `np` is `size_t` in the libxc 5.x C header; `usize` has identical
        // size and alignment on all supported targets (LP64 Linux x86_64 / aarch64).
        /// LDA energy density and first derivative (exc, vrho) at `np` grid points.
        pub fn xc_lda_exc_vxc(
            p: *const XcFuncOpaque,
            np: usize,
            rho: *const f64,
            zk: *mut f64,
            vrho: *mut f64,
        );

        /// GGA energy density and first derivatives (exc, vrho, vsigma) at `np` grid points.
        pub fn xc_gga_exc_vxc(
            p: *const XcFuncOpaque,
            np: usize,
            rho: *const f64,
            sigma: *const f64,
            zk: *mut f64,
            vrho: *mut f64,
            vsigma: *mut f64,
        );

        /// Meta-GGA exc + first derivatives. Signature verified against libxc
        /// 7.0.0 `src/xc.h`:
        /// ```c
        /// void xc_mgga_exc_vxc(const xc_func_type *p, size_t np,
        ///      const double *rho, const double *sigma,
        ///      const double *lapl, const double *tau,
        ///      double *zk, double *vrho, double *vsigma,
        ///      double *vlapl, double *vtau);
        /// ```
        /// `lapl` (density Laplacian) is passed but NEVER read for the
        /// functionals ferric supports (SCAN / r2SCAN / TPSS do not set
        /// `XC_FLAGS_NEEDS_LAPLACIAN`); the caller passes a zero buffer and
        /// discards `vlapl`.
        pub fn xc_mgga_exc_vxc(
            p: *const XcFuncOpaque,
            np: usize,
            rho: *const f64,
            sigma: *const f64,
            lapl: *const f64,
            tau: *const f64,
            zk: *mut f64,
            vrho: *mut f64,
            vsigma: *mut f64,
            vlapl: *mut f64,
            vtau: *mut f64,
        );

        /// Second functional derivative of LDA. For unpolarized (nspin=1)
        /// the layout is [v2rho2 per point]. For polarized (nspin=2) the
        /// layout is [v2rho2_αα, v2rho2_αβ, v2rho2_ββ] per point.
        pub fn xc_lda_fxc(
            p: *const XcFuncOpaque,
            np: usize,
            rho: *const f64,
            v2rho2: *mut f64,
        );

        /// Second functional derivatives of a GGA. For polarized (nspin=2) the
        /// output layouts are (per grid point):
        ///   v2rho2      → 3 components: [ρ_αρ_α, ρ_αρ_β, ρ_βρ_β]
        ///   v2rhosigma  → 6 components: [ρ_α·σ_αα, ρ_α·σ_αβ, ρ_α·σ_ββ,
        ///                                ρ_β·σ_αα, ρ_β·σ_αβ, ρ_β·σ_ββ]
        ///   v2sigma2    → 6 components: [σ_αα·σ_αα, σ_αα·σ_αβ, σ_αα·σ_ββ,
        ///                                σ_αβ·σ_αβ, σ_αβ·σ_ββ, σ_ββ·σ_ββ]
        /// (This is the standard libxc symmetric-pack ordering; note
        /// v2rhosigma is a full 2×3 block — 6, NOT 9 — because libxc packs the
        /// two ρ spins against the three σ channels with no symmetry to fold.)
        pub fn xc_gga_fxc(
            p: *const XcFuncOpaque,
            np: usize,
            rho: *const f64,
            sigma: *const f64,
            v2rho2: *mut f64,
            v2rhosigma: *mut f64,
            v2sigma2: *mut f64,
        );

        /// Retrieve CAM (range-separated) parameters: ω, α (long-range), β (short-range).
        pub fn xc_hyb_cam_coef(
            p: *const XcFuncOpaque,
            omega: *mut f64,
            alpha: *mut f64,
            beta: *mut f64,
        );

        /// Fraction of exact (HF) exchange for global hybrids.
        pub fn xc_hyb_exx_coef(p: *const XcFuncOpaque) -> f64;

        /// Returns the (b, C) VV10 nonlocal correlation parameters for a
        /// functional that carries them. For functionals without VV10, the
        /// values written are zero (per libxc convention) and the caller
        /// should treat (0, 0) as "no VV10".
        pub fn xc_nlc_coef(
            p: *const XcFuncOpaque,
            nlc_b: *mut f64,
            nlc_c: *mut f64,
        );
    }
}

// ---------------------------------------------------------------------------
// Raw-pointer Send wrapper for the chunked-parallel eval path
// ---------------------------------------------------------------------------

/// Wraps a `*mut f64` so it can be captured by a `Sync` closure passed to
/// `map_init`/`par_iter`. Each rayon chunk offsets this pointer into its own
/// disjoint `[g0, g1)` sub-range before touching it (see `eval_chunked`
/// callers), so distinct chunks never alias — the `Send` impl only asserts
/// that moving the raw pointer value across threads is fine, not that
/// concurrent access to the same address is (there never is any).
#[derive(Clone, Copy)]
struct SendPtr(*mut f64);
// SAFETY: see doc comment above — every use offsets into a disjoint,
// caller-verified sub-range per chunk, so no two chunks ever write (or read)
// the same address concurrently.
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

impl SendPtr {
    /// Offset accessor. Written as a method (not direct `.0` field access at
    /// the call site) so Rust 2021's disjoint-closure-capture analysis
    /// captures the whole `SendPtr` (which is `Sync`) rather than reaching
    /// straight through to the bare `*mut f64` field (which is not).
    #[inline(always)]
    unsafe fn add(self, offset: usize) -> *mut f64 {
        self.0.add(offset)
    }
}

// ---------------------------------------------------------------------------
// Public error type
// ---------------------------------------------------------------------------

#[derive(Error, Debug)]
#[non_exhaustive]
pub enum LibxcError {
    #[error("unknown libxc functional name: {0}")]
    UnknownName(String),
    #[error("xc_func_init failed for {name}: rc={rc}")]
    InitFailed { name: String, rc: c_int },
    #[error("CString conversion failed: {0}")]
    BadName(String),
    #[error("xc_func_alloc returned null")]
    AllocFailed,
    #[error("unsupported functional: {0}")]
    Unsupported(String),
    #[error("this functional exposes no external parameters")]
    NoExtParams,
    #[error("external parameter count mismatch: functional expects {expected}, got {got}")]
    ExtParamCount { expected: usize, got: usize },
    #[error("external parameter {position} is named {actual:?}, expected {expected:?}")]
    ExtParamName { position: usize, actual: String, expected: String },
}

impl From<LibxcError> for ferric_core::error::FerricError {
    fn from(e: LibxcError) -> Self { Self::General(e.to_string()) }
}

// ---------------------------------------------------------------------------
// Functional family
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FunctionalFamily {
    Lda,
    Gga,
    HybridGga,
    RangeSepGga,
    /// Semilocal meta-GGA (needs the kinetic-energy density τ). ferric supports
    /// SCAN / r2SCAN / TPSS — none of which need the density Laplacian.
    MetaGga,
}

// ---------------------------------------------------------------------------
// CAM and VV10 parameter structs
// ---------------------------------------------------------------------------

/// Range-separation (CAM) parameters.
///
/// libxc convention (from `xc.h`):
/// > "at short range the fraction of exact exchange is `cam_alpha + cam_beta`,
/// >  while at long range it is `cam_alpha`."
///
/// Ferric mapping:
/// - `c_sr = cam_alpha + cam_beta` (libxc `α + β` — short-range HF fraction)
/// - `c_lr = cam_alpha`            (libxc `α` — long-range HF fraction)
///
/// For wB97X-V: libxc returns `alpha=1.0, beta=-0.833`
/// → `c_sr = 0.167`, `c_lr = 1.0` (long-range-corrected: 100% HF at long range).
#[derive(Debug, Clone, Copy)]
pub struct CamCoeffs {
    pub omega: f64,
    /// Short-range exact-exchange coefficient (`cam_alpha + cam_beta` in libxc).
    pub c_sr: f64,
    /// Long-range exact-exchange coefficient (`cam_alpha` in libxc).
    pub c_lr: f64,
}

/// Non-local VV10 correlation parameters.
#[derive(Debug, Clone, Copy)]
pub struct Vv10Params {
    pub c: f64,
    pub b: f64,
}

// ---------------------------------------------------------------------------
// RAII handle
// ---------------------------------------------------------------------------

/// RAII handle around a heap-allocated libxc `xc_func_type`.
pub struct XcFunctional {
    ptr: *mut ffi::XcFuncOpaque,
    family: FunctionalFamily,
    pub nspin: u32,
    /// libxc's internal functional ID (from `xc_functional_get_number`).
    /// Kept so the chunked-parallel evaluation path (`par_chunks`, below) can
    /// cheaply build one fresh handle per rayon worker via [`Self::from_id`]
    /// without re-parsing / re-looking-up the name string each time.
    libxc_id: c_int,
    /// External parameters applied via [`Self::set_ext_params`], retained so
    /// [`Self::from_id`] can re-apply them to every per-worker clone.
    ///
    /// This is load-bearing, not bookkeeping: `par_chunks` builds a fresh handle
    /// per rayon worker, and a fresh handle carries libxc's *built-in* defaults.
    /// Without replaying the overrides here, a parallel run would silently
    /// evaluate stock coefficients while a serial run used the custom ones —
    /// producing a wrong energy with no error anywhere.
    ext_params: Option<Vec<f64>>,
}

impl std::fmt::Debug for XcFunctional {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("XcFunctional")
            .field("family", &self.family)
            .field("nspin", &self.nspin)
            .field("libxc_id", &self.libxc_id)
            .finish_non_exhaustive()
    }
}

/// Global mutex serializing libxc's non-thread-safe init/destroy operations.
/// libxc 5.x is documented as thread-safe for *evaluation* (xc_*_exc_vxc) but
/// NOT for initialization — xc_func_alloc and xc_func_init mutate shared
/// internal tables. Without serialization, concurrent `XcFunctional::new`
/// calls SIGSEGV.
static LIBXC_INIT_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

impl XcFunctional {
    /// Initialise a functional by name (e.g. `"LDA_X"`, `"GGA_X_PBE"`).
    pub fn new(name: &str, nspin: u32) -> Result<Self, LibxcError> {
        let c_name = CString::new(name).map_err(|_| LibxcError::BadName(name.to_string()))?;

        // Serialize all libxc state-mutating calls (lookup + alloc + init) so
        // concurrent test threads cannot trample each other. Evaluation calls
        // (xc_lda_exc_vxc / xc_gga_exc_vxc) are not under this lock — libxc 5.x
        // documents those as thread-safe on initialized handles.
        //
        // NOTE: everything from the lookup through `alloc_and_init_unlocked`
        // runs under this ONE guard — `std::sync::Mutex` is not reentrant, so
        // `alloc_and_init_unlocked` must NEVER itself try to (re-)acquire
        // `LIBXC_INIT_LOCK` (that shape self-deadlocked here once already).
        let _guard = LIBXC_INIT_LOCK.lock().unwrap_or_else(|p| p.into_inner());

        // SAFETY: c_name lives until the end of this statement; xc_functional_get_number
        // reads the C string and returns a (possibly negative) integer ID without retaining
        // the pointer.
        let id = unsafe { ffi::xc_functional_get_number(c_name.as_ptr()) };
        if id < 0 {
            return Err(LibxcError::UnknownName(name.to_string()));
        }

        let ptr = Self::alloc_and_init_unlocked(id, nspin, name)?;
        let family = infer_family_from_name(name);
        Ok(Self { ptr, family, nspin, libxc_id: id, ext_params: None })
    }

    /// Number of external parameters this functional exposes.
    pub fn n_ext_params(&self) -> usize {
        // SAFETY: self.ptr is a live, initialized handle; xc_func_get_info returns a
        // borrowed pointer into libxc's static tables (never freed, never null for an
        // initialized functional).
        unsafe {
            let info = ffi::xc_func_get_info(self.ptr);
            if info.is_null() {
                return 0;
            }
            ffi::xc_func_info_get_n_ext_params(info).max(0) as usize
        }
    }

    /// Names of this functional's external parameters, in libxc's own order.
    ///
    /// The order is the contract for [`Self::set_ext_params`], so callers should
    /// verify against these names rather than hardcoding positions.
    pub fn ext_param_names(&self) -> Vec<String> {
        // SAFETY: as in n_ext_params; each returned name is a static NUL-terminated
        // C string owned by libxc.
        unsafe {
            let info = ffi::xc_func_get_info(self.ptr);
            if info.is_null() {
                return Vec::new();
            }
            let n = ffi::xc_func_info_get_n_ext_params(info).max(0);
            (0..n)
                .map(|i| {
                    let p = ffi::xc_func_info_get_ext_params_name(info, i);
                    if p.is_null() {
                        String::new()
                    } else {
                        std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
                    }
                })
                .collect()
        }
    }

    /// libxc's built-in default values for the external parameters.
    pub fn ext_param_defaults(&self) -> Vec<f64> {
        // SAFETY: as in n_ext_params.
        unsafe {
            let info = ffi::xc_func_get_info(self.ptr);
            if info.is_null() {
                return Vec::new();
            }
            let n = ffi::xc_func_info_get_n_ext_params(info).max(0);
            (0..n).map(|i| ffi::xc_func_info_get_ext_params_default_value(info, i)).collect()
        }
    }

    /// Override this functional's external parameters.
    ///
    /// `params` must have exactly [`Self::n_ext_params`] entries, in libxc's order
    /// (see [`Self::ext_param_names`]). The length is checked here because
    /// `xc_func_set_ext_params` takes no length and performs no bounds check — a
    /// short slice would be an out-of-bounds read inside libxc.
    ///
    /// The values are retained so per-worker clones made by the parallel
    /// evaluation path reproduce them; see the `ext_params` field.
    pub fn set_ext_params(&mut self, params: &[f64]) -> Result<(), LibxcError> {
        let expected = self.n_ext_params();
        if expected == 0 {
            return Err(LibxcError::NoExtParams);
        }
        if params.len() != expected {
            return Err(LibxcError::ExtParamCount { expected, got: params.len() });
        }
        // SAFETY: params.len() == expected == the count libxc reports for this handle,
        // so libxc reads exactly within bounds. Serialized against concurrent
        // init/destroy because it mutates handle state.
        {
            let _guard = LIBXC_INIT_LOCK.lock().unwrap_or_else(|p| p.into_inner());
            // SAFETY: self.ptr is a valid xc_func_type handle (created in new,
            // guarded by LIBXC_INIT_LOCK). params.len() == n_ext_params (checked above).
            unsafe { ffi::xc_func_set_ext_params(self.ptr, params.as_ptr()) };
        }
        self.ext_params = Some(params.to_vec());
        Ok(())
    }

    /// The external parameters applied via [`Self::set_ext_params`], if any.
    pub fn applied_ext_params(&self) -> Option<&[f64]> {
        self.ext_params.as_deref()
    }

    /// Allocate + init a raw handle for an already-resolved libxc `id`. Does
    /// NOT itself take `LIBXC_INIT_LOCK` — every caller must hold that guard
    /// for the duration of this call (`new` already does; `from_id` takes its
    /// own guard around the call site below). Shared so the alloc/init/error
    /// logic exists exactly once.
    fn alloc_and_init_unlocked(
        id: c_int,
        nspin: u32,
        name_for_err: &str,
    ) -> Result<*mut ffi::XcFuncOpaque, LibxcError> {
        // SAFETY: xc_func_alloc allocates and zero-initialises a xc_func_type on the heap;
        // it returns null on allocation failure (checked immediately below).
        let ptr = unsafe { ffi::xc_func_alloc() };
        if ptr.is_null() {
            return Err(LibxcError::AllocFailed);
        }
        // SAFETY: ptr is non-null and freshly allocated; xc_func_init populates the struct.
        // On failure we call xc_func_free to release the allocation before returning.
        let rc = unsafe { ffi::xc_func_init(ptr, id, nspin as c_int) };
        if rc != 0 {
            unsafe { ffi::xc_func_free(ptr) };
            return Err(LibxcError::InitFailed { name: name_for_err.to_string(), rc });
        }
        Ok(ptr)
    }

    /// Build a fresh handle from an already-known libxc functional id, reusing
    /// this handle's `family`/`nspin`. Used exclusively to give each rayon
    /// worker its own non-shared `xc_func_type` in the chunked evaluation path
    /// — libxc handles must never be evaluated from more than one thread
    /// concurrently even though the docs call bare evaluation thread-safe on a
    /// *single* handle (we simply avoid the question by never sharing one).
    /// Build a fresh handle for an already-resolved id, replaying any external
    /// parameter overrides.
    ///
    /// `ext_params` MUST be threaded through: a fresh libxc handle carries the
    /// functional's built-in defaults, so a worker clone that skipped this would
    /// silently evaluate stock coefficients while the parent used custom ones.
    fn from_id(
        id: c_int,
        nspin: u32,
        family: FunctionalFamily,
        ext_params: Option<&[f64]>,
    ) -> Self {
        // Take LIBXC_INIT_LOCK ourselves for the alloc+init (mirrors `new`'s
        // guard scope; `alloc_and_init_unlocked` takes no lock itself — see
        // its doc comment).
        let ptr = {
            let _guard = LIBXC_INIT_LOCK.lock().unwrap_or_else(|p| p.into_inner());
            // Construction failure here would mean the id (harvested from an
            // already-successfully-constructed handle) suddenly stopped being
            // valid, which cannot happen — libxc's functional table is static
            // for the process lifetime. expect() documents that invariant.
            let ptr = Self::alloc_and_init_unlocked(id, nspin, "<worker clone>")
                .expect("re-initializing a previously-valid libxc functional id must succeed");
            // Replay overrides under the SAME guard as the init that created the
            // handle, so no other thread can observe it in a defaults-only state.
            if let Some(p) = ext_params {
                // SAFETY: the parent handle validated this slice's length against
                // n_ext_params for this same libxc id, and the id determines the
                // count, so it is still exact here.
                unsafe { ffi::xc_func_set_ext_params(ptr, p.as_ptr()) };
            }
            ptr
        };
        Self { ptr, family, nspin, libxc_id: id, ext_params: ext_params.map(|p| p.to_vec()) }
    }

    /// The functional family (LDA, GGA, hybrid, meta-GGA, etc.).
    pub fn family(&self) -> FunctionalFamily {
        self.family
    }

    pub(crate) fn set_family(&mut self, f: FunctionalFamily) {
        self.family = f;
    }

    // -----------------------------------------------------------------------
    // Chunked-parallel evaluation
    // -----------------------------------------------------------------------
    //
    // libxc evaluation is point-local (no point reads/writes another point's
    // data), so the point range can be split into contiguous chunks and each
    // chunk evaluated independently. Below `PAR_MIN_PTS` a single serial FFI
    // call is used — the same guard shape as `ao_grid.rs`'s
    // `PAR_WORK_THRESHOLD` (rayon spawn/join/steal overhead would dwarf the
    // work on small molecules / free-atom SCF grids). Above it, the range is
    // divided into chunks and each chunk gets its OWN freshly-constructed
    // `XcFunctional` handle (one per rayon worker via `map_init`) — libxc
    // handles must never be evaluated concurrently from multiple threads on
    // the SAME handle, so per-worker construction sidesteps that question
    // entirely rather than relying on the "evaluation is thread-safe on an
    // initialized handle" documentation.
    //
    // Each chunk writes into a *disjoint* slice of the caller's output
    // buffers (chunk `c` owns points `[c*chunk_len, (c+1)*chunk_len)`, scaled
    // by the per-point stride of each buffer), so the chunked result is
    // bit-identical to the single serial call by construction: every point's
    // output depends only on that point's own inputs, and every point is
    // written by exactly one chunk, exactly once, with the same libxc
    // parameters as the serial path (same functional id/nspin). There is no
    // cross-chunk reduction here (unlike `deterministic_point_sum` in
    // `vxc.rs`, which sums `exc·ρ`) — chunking a plain elementwise buffer
    // fill can never introduce a reduction-order difference.

    /// Below this many grid points, evaluate with a single serial FFI call —
    /// rayon spawn/join/steal + per-chunk libxc handle construction overhead
    /// would dwarf the work on small molecules / free-atom SCF grids. Mirrors
    /// the `PAR_WORK_THRESHOLD` guard shape in `ao_grid.rs` (pure function of
    /// point count only, never of thread count).
    const PAR_MIN_PTS: usize = 50_000;

    /// Target number of chunks to fan out across rayon workers. Chunk
    /// boundaries are a pure function of `npts` (never of thread count), so
    /// results stay thread-count independent — this only controls how many
    /// pieces `npts` is divided into, not which rayon worker each lands on.
    const CHUNKS: usize = 32;

    /// Split `[0, npts)` into `Self::CHUNKS` contiguous, equal-ish, disjoint
    /// ranges (a pure function of `npts`).
    fn chunk_ranges(npts: usize) -> Vec<(usize, usize)> {
        let chunk_len = npts.div_ceil(Self::CHUNKS).max(1);
        let n_chunks = npts.div_ceil(chunk_len);
        (0..n_chunks)
            .map(|c| {
                let g0 = c * chunk_len;
                let g1 = (g0 + chunk_len).min(npts);
                (g0, g1)
            })
            .collect()
    }

    /// Run `body` once per chunk of `npts` points, in parallel, each chunk
    /// getting its own freshly-constructed `XcFunctional` handle (via
    /// `map_init`, so construction is amortized per rayon *worker* rather
    /// than per chunk). Falls back to a single call of `body` with the whole
    /// range and `self` directly when `npts < PAR_MIN_PTS`.
    ///
    /// `body(functional, g0, g1)` must read/write only the point-range
    /// `[g0, g1)` of every buffer it closes over.
    fn eval_chunked<Body>(&self, npts: usize, body: Body)
    where
        Body: Fn(&XcFunctional, usize, usize) + Sync,
    {
        if npts < Self::PAR_MIN_PTS {
            body(self, 0, npts);
            return;
        }
        let ranges = Self::chunk_ranges(npts);
        ranges.into_par_iter().for_each_init(
            || {
                XcFunctional::from_id(
                    self.libxc_id,
                    self.nspin,
                    self.family,
                    self.ext_params.as_deref(),
                )
            },
            |worker, (g0, g1)| body(worker, g0, g1),
        );
    }

    /// LDA: feed ρ, get ε_xc and v_ρ.
    pub fn eval_lda_unpolarized(&self, rho: &[f64], exc: &mut [f64], vrho: &mut [f64]) {
        let n = rho.len();
        assert_eq!(exc.len(), n, "exc buffer length must match rho.len()");
        assert_eq!(vrho.len(), n, "vrho buffer length must match rho.len()");
        // SAFETY (per chunk): each call operates on a disjoint sub-slice
        // `[g0, g1)` of rho/exc/vrho (stride 1), using its own worker-local
        // handle; buffer lengths are verified above.
        let exc_ptr = SendPtr(exc.as_mut_ptr());
        let vrho_ptr = SendPtr(vrho.as_mut_ptr());
        self.eval_chunked(n, |func, g0, g1| unsafe {
            let len = g1 - g0;
            ffi::xc_lda_exc_vxc(
                func.ptr as *const _,
                len,
                rho.as_ptr().add(g0),
                exc_ptr.add(g0),
                vrho_ptr.add(g0),
            );
        });
    }

    /// LDA, polarized (nspin=2). Interleaved layout:
    ///   `rho[2g+0] = ρ_α(r_g)`, `rho[2g+1] = ρ_β(r_g)`
    ///   `vrho[2g+0] = ∂E_xc/∂ρ_α(r_g)`, `vrho[2g+1] = ∂E_xc/∂ρ_β(r_g)`
    /// `exc[g]` is the per-particle energy density (not interleaved).
    pub fn eval_lda_polarized(&self, rho: &[f64], exc: &mut [f64], vrho: &mut [f64]) {
        debug_assert!(self.nspin == 2, "eval_lda_polarized requires nspin=2");
        let n = exc.len();
        assert_eq!(rho.len(), 2 * n, "polarized rho buffer must be 2 * npts");
        assert_eq!(vrho.len(), 2 * n, "polarized vrho buffer must be 2 * npts");
        // SAFETY (per chunk): disjoint `[g0, g1)` sub-ranges (rho/vrho stride
        // 2, exc stride 1); own worker-local handle; buffer lengths verified above.
        let exc_ptr = SendPtr(exc.as_mut_ptr());
        let vrho_ptr = SendPtr(vrho.as_mut_ptr());
        self.eval_chunked(n, |func, g0, g1| unsafe {
            let len = g1 - g0;
            ffi::xc_lda_exc_vxc(
                func.ptr as *const _,
                len,
                rho.as_ptr().add(2 * g0),
                exc_ptr.add(g0),
                vrho_ptr.add(2 * g0),
            );
        });
    }

    /// LDA second derivative, polarized (nspin=2). Layout:
    ///   `rho[2g+0] = ρ_α`, `rho[2g+1] = ρ_β`
    ///   `v2rho2[3g+0] = ∂²E_xc/∂ρ_α²`,
    ///   `v2rho2[3g+1] = ∂²E_xc/∂ρ_α∂ρ_β`,
    ///   `v2rho2[3g+2] = ∂²E_xc/∂ρ_β²`
    pub fn eval_lda_fxc_polarized(&self, rho: &[f64], v2rho2: &mut [f64]) {
        debug_assert!(self.nspin == 2, "eval_lda_fxc_polarized requires nspin=2");
        let n = v2rho2.len() / 3;
        assert_eq!(rho.len(), 2 * n, "polarized rho buffer must be 2 * npts");
        assert_eq!(v2rho2.len(), 3 * n, "polarized v2rho2 buffer must be 3 * npts");
        // SAFETY: handle is non-null + fully initialised; lengths verified.
        unsafe {
            ffi::xc_lda_fxc(
                self.ptr as *const _,
                n,
                rho.as_ptr(),
                v2rho2.as_mut_ptr(),
            );
        }
    }

    /// GGA second derivatives, polarized (nspin=2). Inputs interleaved as in
    /// `eval_gga_polarized`:
    ///   `rho[2g+0]   = ρ_α`,  `rho[2g+1]   = ρ_β`
    ///   `sigma[3g+0] = σ_αα`, `sigma[3g+1] = σ_αβ`, `sigma[3g+2] = σ_ββ`
    ///
    /// Outputs (libxc symmetric-pack ordering; verified against libxc 7.0.0
    /// `util.c::internal_counters_set_gga` — 3 / 6 / 6 components per point —
    /// and PySCF `dft/libxc.py` layout notes):
    ///   `v2rho2[3g+0]      = ∂²E/∂ρ_α∂ρ_α`
    ///   `v2rho2[3g+1]      = ∂²E/∂ρ_α∂ρ_β`
    ///   `v2rho2[3g+2]      = ∂²E/∂ρ_β∂ρ_β`
    ///
    ///   `v2rhosigma[6g+0]  = ∂²E/∂ρ_α∂σ_αα`
    ///   `v2rhosigma[6g+1]  = ∂²E/∂ρ_α∂σ_αβ`
    ///   `v2rhosigma[6g+2]  = ∂²E/∂ρ_α∂σ_ββ`
    ///   `v2rhosigma[6g+3]  = ∂²E/∂ρ_β∂σ_αα`
    ///   `v2rhosigma[6g+4]  = ∂²E/∂ρ_β∂σ_αβ`
    ///   `v2rhosigma[6g+5]  = ∂²E/∂ρ_β∂σ_ββ`
    ///
    ///   `v2sigma2[6g+0]    = ∂²E/∂σ_αα∂σ_αα`
    ///   `v2sigma2[6g+1]    = ∂²E/∂σ_αα∂σ_αβ`
    ///   `v2sigma2[6g+2]    = ∂²E/∂σ_αα∂σ_ββ`
    ///   `v2sigma2[6g+3]    = ∂²E/∂σ_αβ∂σ_αβ`
    ///   `v2sigma2[6g+4]    = ∂²E/∂σ_αβ∂σ_ββ`
    ///   `v2sigma2[6g+5]    = ∂²E/∂σ_ββ∂σ_ββ`
    pub fn eval_gga_fxc_polarized(
        &self,
        rho: &[f64],
        sigma: &[f64],
        v2rho2: &mut [f64],
        v2rhosigma: &mut [f64],
        v2sigma2: &mut [f64],
    ) {
        debug_assert!(self.nspin == 2, "eval_gga_fxc_polarized requires nspin=2");
        let n = v2rho2.len() / 3;
        assert_eq!(rho.len(), 2 * n, "polarized rho buffer must be 2 * npts");
        assert_eq!(sigma.len(), 3 * n, "polarized sigma buffer must be 3 * npts");
        assert_eq!(v2rho2.len(), 3 * n, "polarized v2rho2 buffer must be 3 * npts");
        assert_eq!(v2rhosigma.len(), 6 * n, "polarized v2rhosigma buffer must be 6 * npts");
        assert_eq!(v2sigma2.len(), 6 * n, "polarized v2sigma2 buffer must be 6 * npts");
        // SAFETY: handle is non-null + fully initialised; lengths verified.
        unsafe {
            ffi::xc_gga_fxc(
                self.ptr as *const _,
                n,
                rho.as_ptr(),
                sigma.as_ptr(),
                v2rho2.as_mut_ptr(),
                v2rhosigma.as_mut_ptr(),
                v2sigma2.as_mut_ptr(),
            );
        }
    }

    /// GGA, polarized (nspin=2). Interleaved layouts:
    ///   `rho[2g+0]   = ρ_α`,  `rho[2g+1]   = ρ_β`
    ///   `sigma[3g+0] = σ_αα`, `sigma[3g+1] = σ_αβ`, `sigma[3g+2] = σ_ββ`
    ///   `vrho[2g+0]  = v_α`,  `vrho[2g+1]  = v_β`
    ///   `vsigma[3g+0]= v_σαα`,`vsigma[3g+1]= v_σαβ`, `vsigma[3g+2]= v_σββ`
    pub fn eval_gga_polarized(
        &self,
        rho: &[f64],
        sigma: &[f64],
        exc: &mut [f64],
        vrho: &mut [f64],
        vsigma: &mut [f64],
    ) {
        debug_assert!(self.nspin == 2, "eval_gga_polarized requires nspin=2");
        let n = exc.len();
        assert_eq!(rho.len(), 2 * n, "polarized rho buffer must be 2 * npts");
        assert_eq!(sigma.len(), 3 * n, "polarized sigma buffer must be 3 * npts");
        assert_eq!(vrho.len(), 2 * n, "polarized vrho buffer must be 2 * npts");
        assert_eq!(vsigma.len(), 3 * n, "polarized vsigma buffer must be 3 * npts");
        // SAFETY (per chunk): disjoint `[g0, g1)` sub-ranges (rho/vrho stride
        // 2, sigma/vsigma stride 3, exc stride 1); own worker-local handle.
        let exc_ptr = SendPtr(exc.as_mut_ptr());
        let vrho_ptr = SendPtr(vrho.as_mut_ptr());
        let vsigma_ptr = SendPtr(vsigma.as_mut_ptr());
        self.eval_chunked(n, |func, g0, g1| unsafe {
            let len = g1 - g0;
            ffi::xc_gga_exc_vxc(
                func.ptr as *const _,
                len,
                rho.as_ptr().add(2 * g0),
                sigma.as_ptr().add(3 * g0),
                exc_ptr.add(g0),
                vrho_ptr.add(2 * g0),
                vsigma_ptr.add(3 * g0),
            );
        });
    }

    /// GGA (including hybrid-GGA and RSH-GGA): feed ρ and σ = |∇ρ|².
    pub fn eval_gga_unpolarized(
        &self,
        rho: &[f64],
        sigma: &[f64],
        exc: &mut [f64],
        vrho: &mut [f64],
        vsigma: &mut [f64],
    ) {
        let n = rho.len();
        assert_eq!(sigma.len(), n, "sigma buffer length must match rho.len()");
        assert_eq!(exc.len(), n, "exc buffer length must match rho.len()");
        assert_eq!(vrho.len(), n, "vrho buffer length must match rho.len()");
        assert_eq!(vsigma.len(), n, "vsigma buffer length must match rho.len()");
        // SAFETY (per chunk): each call operates on a disjoint sub-slice
        // `[g0, g1)` of every buffer (stride 1 throughout, unpolarized), using
        // its own worker-local handle; buffer lengths are verified above.
        let exc_ptr = SendPtr(exc.as_mut_ptr());
        let vrho_ptr = SendPtr(vrho.as_mut_ptr());
        let vsigma_ptr = SendPtr(vsigma.as_mut_ptr());
        self.eval_chunked(n, |func, g0, g1| unsafe {
            let len = g1 - g0;
            ffi::xc_gga_exc_vxc(
                func.ptr as *const _,
                len,
                rho.as_ptr().add(g0),
                sigma.as_ptr().add(g0),
                exc_ptr.add(g0),
                vrho_ptr.add(g0),
                vsigma_ptr.add(g0),
            );
        });
    }

    /// Meta-GGA (unpolarized, nspin=1): feed ρ, σ = |∇ρ|², and τ = ½Σ_i|∇φ_i|².
    ///
    /// One value per grid point for every buffer:
    ///   `rho[g]`, `sigma[g]`, `tau[g]` (inputs),
    ///   `exc[g]`  = ε_xc(r_g)                 (per-particle energy density),
    ///   `vrho[g]` = ∂E_xc/∂ρ(r_g),
    ///   `vsigma[g]` = ∂E_xc/∂σ(r_g),
    ///   `vtau[g]` = ∂E_xc/∂τ(r_g).
    ///
    /// The density Laplacian is not needed by any supported meta-GGA (SCAN /
    /// r2SCAN / TPSS): a zero `lapl` buffer is passed and `vlapl` discarded.
    #[allow(clippy::too_many_arguments)]
    pub fn eval_mgga_unpolarized(
        &self,
        rho: &[f64],
        sigma: &[f64],
        tau: &[f64],
        exc: &mut [f64],
        vrho: &mut [f64],
        vsigma: &mut [f64],
        vtau: &mut [f64],
    ) {
        let n = rho.len();
        assert_eq!(sigma.len(), n, "sigma buffer length must match rho.len()");
        assert_eq!(tau.len(), n, "tau buffer length must match rho.len()");
        assert_eq!(exc.len(), n, "exc buffer length must match rho.len()");
        assert_eq!(vrho.len(), n, "vrho buffer length must match rho.len()");
        assert_eq!(vsigma.len(), n, "vsigma buffer length must match rho.len()");
        assert_eq!(vtau.len(), n, "vtau buffer length must match rho.len()");
        // SAFETY (per chunk): each call operates on a disjoint sub-slice
        // `[g0, g1)` of every buffer (stride 1, unpolarized), using its own
        // worker-local handle. `lapl`/`vlapl` are unused by every supported
        // meta-GGA but the C ABI still dereferences them, so each chunk hands
        // a real zeroed buffer sized to its own `len` (not `n`).
        let exc_ptr = SendPtr(exc.as_mut_ptr());
        let vrho_ptr = SendPtr(vrho.as_mut_ptr());
        let vsigma_ptr = SendPtr(vsigma.as_mut_ptr());
        let vtau_ptr = SendPtr(vtau.as_mut_ptr());
        self.eval_chunked(n, |func, g0, g1| {
            let len = g1 - g0;
            let lapl = vec![0.0_f64; len];
            let mut vlapl = vec![0.0_f64; len];
            unsafe {
                ffi::xc_mgga_exc_vxc(
                    func.ptr as *const _,
                    len,
                    rho.as_ptr().add(g0),
                    sigma.as_ptr().add(g0),
                    lapl.as_ptr(),
                    tau.as_ptr().add(g0),
                    exc_ptr.add(g0),
                    vrho_ptr.add(g0),
                    vsigma_ptr.add(g0),
                    vlapl.as_mut_ptr(),
                    vtau_ptr.add(g0),
                );
            }
        });
    }

    /// Meta-GGA, polarized (nspin=2). Interleaved layouts (mirroring the GGA
    /// polarized wrapper, with τ added):
    ///   `rho[2g+0]   = ρ_α`,   `rho[2g+1]   = ρ_β`
    ///   `sigma[3g+0] = σ_αα`,  `sigma[3g+1] = σ_αβ`, `sigma[3g+2] = σ_ββ`
    ///   `tau[2g+0]   = τ_α`,   `tau[2g+1]   = τ_β`
    ///   `vrho[2g+0]  = v_ρα`,  `vrho[2g+1]  = v_ρβ`
    ///   `vsigma[3g+0]= v_σαα`, `vsigma[3g+1]= v_σαβ`, `vsigma[3g+2]= v_σββ`
    ///   `vtau[2g+0]  = v_τα`,  `vtau[2g+1]  = v_τβ`
    /// `exc[g]` is the per-particle energy density (one per point).
    ///
    /// As in the unpolarized case, `lapl`/`vlapl` are zeroed / discarded (no
    /// supported meta-GGA needs the Laplacian). The polarized Laplacian buffers
    /// carry 2 components per point (`lapl[2g+σ]`).
    #[allow(clippy::too_many_arguments)]
    pub fn eval_mgga_polarized(
        &self,
        rho: &[f64],
        sigma: &[f64],
        tau: &[f64],
        exc: &mut [f64],
        vrho: &mut [f64],
        vsigma: &mut [f64],
        vtau: &mut [f64],
    ) {
        debug_assert!(self.nspin == 2, "eval_mgga_polarized requires nspin=2");
        let n = exc.len();
        assert_eq!(rho.len(), 2 * n, "polarized rho buffer must be 2 * npts");
        assert_eq!(sigma.len(), 3 * n, "polarized sigma buffer must be 3 * npts");
        assert_eq!(tau.len(), 2 * n, "polarized tau buffer must be 2 * npts");
        assert_eq!(vrho.len(), 2 * n, "polarized vrho buffer must be 2 * npts");
        assert_eq!(vsigma.len(), 3 * n, "polarized vsigma buffer must be 3 * npts");
        assert_eq!(vtau.len(), 2 * n, "polarized vtau buffer must be 2 * npts");
        // SAFETY (per chunk): disjoint `[g0, g1)` sub-ranges (rho/vrho/tau/vtau
        // stride 2, sigma/vsigma stride 3, exc stride 1); own worker-local
        // handle. `lapl`/`vlapl` sized per-chunk (2*len), same reasoning as
        // the unpolarized variant.
        let exc_ptr = SendPtr(exc.as_mut_ptr());
        let vrho_ptr = SendPtr(vrho.as_mut_ptr());
        let vsigma_ptr = SendPtr(vsigma.as_mut_ptr());
        let vtau_ptr = SendPtr(vtau.as_mut_ptr());
        self.eval_chunked(n, |func, g0, g1| {
            let len = g1 - g0;
            let lapl = vec![0.0_f64; 2 * len];
            let mut vlapl = vec![0.0_f64; 2 * len];
            unsafe {
                ffi::xc_mgga_exc_vxc(
                    func.ptr as *const _,
                    len,
                    rho.as_ptr().add(2 * g0),
                    sigma.as_ptr().add(3 * g0),
                    lapl.as_ptr(),
                    tau.as_ptr().add(2 * g0),
                    exc_ptr.add(g0),
                    vrho_ptr.add(2 * g0),
                    vsigma_ptr.add(3 * g0),
                    vlapl.as_mut_ptr(),
                    vtau_ptr.add(2 * g0),
                );
            }
        });
    }

    /// Returns CAM (range-separation) coefficients if this is an RSH functional.
    ///
    /// `omega` is the range-separation parameter; a zero value means no
    /// range-separation.
    pub fn cam_coefficients(&self) -> Option<CamCoeffs> {
        let mut omega = 0.0_f64;
        let mut alpha = 0.0_f64;
        let mut beta = 0.0_f64;
        // SAFETY: handle is non-null and fully initialised; xc_hyb_cam_coef writes
        // the three CAM parameters through valid stack pointers and does not retain them.
        unsafe {
            ffi::xc_hyb_cam_coef(
                self.ptr as *const _,
                &mut omega,
                &mut alpha,
                &mut beta,
            );
        }
        if omega == 0.0 {
            None
        } else {
            // libxc convention (from xc.h):
            //   "at short range the fraction of exact exchange is alpha+beta,
            //    while at long range it is alpha."
            // For wB97X-V: alpha=1.0, beta=-0.833 → c_sr=0.167, c_lr=1.0
            // (i.e., long-range-corrected: 100% HF at long range, 16.7% at short.)
            Some(CamCoeffs {
                omega,
                c_sr: alpha + beta,
                c_lr: alpha,
            })
        }
    }

    /// Exact-exchange mixing fraction for plain hybrids (ω = 0).
    pub fn exact_exchange_mix(&self) -> f64 {
        // SAFETY: handle is non-null and fully initialised; xc_hyb_exx_coef reads
        // a scalar from the struct and returns it by value.
        unsafe { ffi::xc_hyb_exx_coef(self.ptr as *const _) }
    }

    /// VV10 nonlocal-correlation parameters, if this functional carries them.
    /// Returns `None` for functionals where libxc reports b=0 (the convention
    /// for "no VV10").
    pub fn vv10_coeffs(&self) -> Option<Vv10Params> {
        let mut b = 0.0_f64;
        let mut c = 0.0_f64;
        // SAFETY: handle is non-null and fully initialised; xc_nlc_coef writes
        // two scalars and does not retain the pointers.
        unsafe {
            ffi::xc_nlc_coef(self.ptr as *const _, &mut b, &mut c);
        }
        if b > 0.0 && c > 0.0 {
            Some(Vv10Params { c, b })
        } else {
            None
        }
    }
}

impl Drop for XcFunctional {
    fn drop(&mut self) {
        // Serialize with init to avoid races on shared libxc internals.
        let _guard = LIBXC_INIT_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        // SAFETY: handle is non-null (allocated in new and never reassigned);
        // xc_func_end releases internal libxc resources (auxiliary functionals,
        // parameter arrays); xc_func_free deallocates the xc_func_type struct
        // itself. This pair is called exactly once because Drop runs once per value.
        unsafe {
            ffi::xc_func_end(self.ptr);
            ffi::xc_func_free(self.ptr);
        }
    }
}

// SAFETY: each XcFunctional owns a distinct heap-allocated xc_func_type.
// Sending it to another thread transfers sole ownership, which is safe.
unsafe impl Send for XcFunctional {}
// SAFETY: libxc's evaluation functions (xc_lda_exc_vxc, xc_gga_exc_vxc, etc.)
// only read the fields set at initialisation time (params, func_aux, mix_coef).
// No mutable global state is touched during evaluation. Concurrent shared
// references from multiple threads are therefore safe, provided no thread
// calls xc_func_init or xc_func_end on the same handle simultaneously —
// which Rust's ownership model prevents by design (&self vs &mut self).
unsafe impl Sync for XcFunctional {}

// ---------------------------------------------------------------------------
// Helper: infer family from name
// ---------------------------------------------------------------------------

fn infer_family_from_name(name: &str) -> FunctionalFamily {
    let n = name.to_uppercase();
    if n.contains("HYB_") {
        FunctionalFamily::HybridGga
    } else if n.starts_with("LDA") {
        FunctionalFamily::Lda
    } else {
        FunctionalFamily::Gga
    }
}

// ---------------------------------------------------------------------------
// High-level XcDef
// ---------------------------------------------------------------------------

/// High-level functional definition: aggregates one or more `XcFunctional`
/// handles, optional CAM coefficients, optional VV10 parameters, and an
/// optional plain-hybrid exact-exchange mixing fraction.
#[derive(Debug)]
pub struct XcDef {
    pub funcs: Vec<XcFunctional>,
    pub cam: Option<CamCoeffs>,
    pub vv10: Option<Vv10Params>,
    /// For plain hybrids (ω=0), the exact-exchange fraction.
    pub b3lyp_mix: Option<f64>,
}

/// Map a friendly functional name to its libxc component identifier(s).
///
/// Returns the canonical libxc names ferric should instantiate. Composite
/// functionals (pure LDA/GGA) map to an exchange + correlation pair; combined
/// `*_XC_*` functionals (hybrids, RSH) map to a single identifier. Returns
/// `None` if the name isn't a recognized friendly alias, in which case the
/// caller passes the name to libxc verbatim (for power users who already know
/// the canonical identifier).
///
/// Matching is case-insensitive and ignores common separators (`-`, `_`).
fn friendly_to_libxc(name: &str) -> Option<&'static [&'static str]> {
    // Normalize: uppercase, strip '-' and '_' so "PBE0", "pbe0", "wB97X-V",
    // "WB97X_V" all collapse to a canonical key.
    let key: String = name
        .to_uppercase()
        .chars()
        .filter(|c| *c != '-' && *c != '_')
        .collect();
    let pair: &'static [&'static str] = match key.as_str() {
        // --- LDA ---
        "LDA" | "SVWN" => &["LDA_X", "LDA_C_VWN"],
        // --- pure GGA (exchange + correlation pair) ---
        "PBE" => &["GGA_X_PBE", "GGA_C_PBE"],
        "PBESOL" => &["GGA_X_PBE_SOL", "GGA_C_PBE_SOL"],
        "BLYP" => &["GGA_X_B88", "GGA_C_LYP"],
        "BP86" => &["GGA_X_B88", "GGA_C_P86"],
        // --- pure meta-GGA (exchange + correlation pair; need τ) ---
        "SCAN" => &["MGGA_X_SCAN", "MGGA_C_SCAN"],
        "R2SCAN" => &["MGGA_X_R2SCAN", "MGGA_C_R2SCAN"],
        // --- hybrid / RSH (single combined identifier) ---
        // PBE0 is libxc's HYB_GGA_XC_PBEH ("PBE hybrid") — HYB_GGA_XC_PBE0
        // does not exist in libxc.
        "PBE0" | "PBEH" => &["HYB_GGA_XC_PBEH"],
        "B3LYP" => &["HYB_GGA_XC_B3LYP"],
        "WB97XV" => &["HYB_GGA_XC_WB97X_V"],
        _ => return None,
    };
    Some(pair)
}

/// Resolve a human-friendly functional name to an `XcDef`. nspin=1 for
/// closed-shell, nspin=2 for spin-polarized (UKS / ROKS).
///
/// Recognized friendly names: `LDA`/`SVWN`, `PBE`, `PBEsol`, `BLYP`, `BP86`,
/// `PBE0`, `B3LYP`, `wB97X-V` (case/separator-insensitive). Any other name is
/// passed to libxc verbatim, so canonical identifiers (e.g. `HYB_GGA_XC_PBEH`)
/// also work. Meta-GGA functionals are rejected — ferric has no mGGA kernel.
pub fn xc_def_from_name(name: &str) -> Result<XcDef, LibxcError> {
    xc_def_from_name_nspin(name, 1)
}


/// [`xc_def_from_name_nspin`] with the range-separation parameter ω
/// OVERRIDDEN (Bohr⁻¹): every component functional carrying an `_omega`
/// external parameter gets it replaced (all other parameters keep their
/// libxc defaults), and the definition's CAM ω follows in lockstep so the
/// SR/LR exchange operators ferric builds agree with the semilocal kernel.
///
/// Errors if NO component has an `_omega` parameter — silently ignoring an
/// ω override on a non-range-separated functional is exactly the
/// config-honesty violation this hard error prevents. α/β (the CAM mixing
/// coefficients) are deliberately NOT touched: standard optimal tuning
/// varies ω at fixed mixing.
pub fn xc_def_from_name_nspin_omega(
    name: &str,
    nspin: u32,
    omega: f64,
) -> Result<XcDef, LibxcError> {
    if !(omega.is_finite() && omega > 0.0) {
        return Err(LibxcError::ExtParamName {
            position: 0,
            actual: format!("omega = {omega}"),
            expected: "a finite, positive range-separation parameter".to_string(),
        });
    }
    let mut def = xc_def_from_name_nspin(name, nspin)?;
    let mut applied = false;
    for f in &mut def.funcs {
        let names = f.ext_param_names();
        if let Some(pos) = names.iter().position(|n| n == "_omega") {
            let mut values = f.ext_param_defaults();
            if values.len() != names.len() {
                return Err(LibxcError::ExtParamCount { expected: names.len(), got: values.len() });
            }
            values[pos] = omega;
            f.set_ext_params(&values)?;
            applied = true;
        }
    }
    if !applied {
        return Err(LibxcError::ExtParamName {
            position: 0,
            actual: name.to_string(),
            expected: "a range-separated functional with an _omega parameter".to_string(),
        });
    }
    match &mut def.cam {
        Some(cam) => cam.omega = omega,
        None => {
            return Err(LibxcError::ExtParamName {
                position: 0,
                actual: name.to_string(),
                expected: "a functional with CAM coefficients (RSH)".to_string(),
            })
        }
    }
    Ok(def)
}

/// nspin-aware variant. Use 1 for closed-shell and 2 for UKS / ROKS.
/// ωB97X-L-V external parameters, in libxc's `HYB_GGA_XC_WB97X_V` order.
///
/// Source: Ransford & Carter-Fenk, *Phys. Chem. Chem. Phys.* **2026**, 28, 14428,
/// Table 2 ("Final" column). `papers/wb97xlv.pdf`.
///
/// The functional is stock ωB97X-V's *form* with re-fitted coefficients, which is
/// exactly what libxc's external-parameter interface exposes — so no hand-written
/// B97 kernel is needed and libxc supplies the analytic vrho/vsigma derivatives.
///
/// Three of the fifteen linear coefficients are constrained rather than fitted, to
/// satisfy the uniform-electron-gas limit at adiabatic-connection parameter λ = 0.6:
///   * `_cx0  = 1 − λ  = 0.4`
///   * `_css0 = _cos0 = 1 − λ² = 0.64`
///
/// Exchange mixing: eqn (27) takes full long-range HF exchange plus λ-scaled
/// short-range HF exchange. Under libxc's CAM convention (`c_sr = α + β`,
/// `c_lr = α`) that is `α = 1.0`, `β = −0.4` ⇒ `c_lr = 1.0`, `c_sr = 0.6 = λ`,
/// with the complementary DFT exchange carrying `_cx0 = 1 − λ = 0.4`.
///
/// NOTE: `_omega = 0.1 a₀⁻¹` here, NOT ωB97X-V's 0.3 — the paper's scan found
/// ω = 0.1 optimal, which it notes is "quite satisfying, as we employed the
/// approximation that ω → 0" in deriving eqn (26).
pub const WB97X_L_V_EXT_PARAMS: [(&str, f64); 18] = [
    ("_cx0", 0.4),
    ("_cx1", 0.154),
    ("_cx2", -3.884),
    ("_cx3", 11.300),
    ("_cx4", -8.425),
    ("_css0", 0.64),
    ("_css1", -1.417),
    ("_css2", 4.716),
    ("_css3", -2.956),
    ("_css4", 0.861),
    ("_cos0", 0.64),
    ("_cos1", -1.194),
    ("_cos2", 1.348),
    ("_cos3", -11.869),
    ("_cos4", 9.571),
    ("_alpha", 1.0),
    ("_beta", -0.4),
    ("_omega", 0.1),
];

/// The ωB97X-L-V adiabatic-connection parameter λ (paper Table 2).
///
/// Scales the wave-function correlation contribution in eqn (27). Also fixes the
/// constrained coefficients above via `1 − λ` and `1 − λ²`.
pub const WB97X_L_V_LAMBDA: f64 = 0.6;

/// VV10 parameters for ωB97X-L-V (paper Table 2): b = 10.0, C = 0.01.
///
/// These differ from stock ωB97X-V's, so they must override whatever
/// `xc_nlc_coef` reports for the underlying libxc functional.
pub const WB97X_L_V_VV10: Vv10Params = Vv10Params { b: 10.0, c: 0.01 };

/// Apply named external parameters, verifying each name against libxc's own
/// ordering.
///
/// Positional application alone would silently scramble the coefficients if libxc
/// ever reordered its parameter list; checking the names makes that a hard error
/// instead of a wrong number.
fn apply_named_ext_params(
    f: &mut XcFunctional,
    spec: &[(&str, f64)],
) -> Result<(), LibxcError> {
    let names = f.ext_param_names();
    if names.len() != spec.len() {
        return Err(LibxcError::ExtParamCount { expected: names.len(), got: spec.len() });
    }
    for (i, (want, _)) in spec.iter().enumerate() {
        if names[i] != *want {
            return Err(LibxcError::ExtParamName {
                position: i,
                actual: names[i].clone(),
                expected: (*want).to_string(),
            });
        }
    }
    let values: Vec<f64> = spec.iter().map(|(_, v)| *v).collect();
    f.set_ext_params(&values)
}

/// Build the ωB97X-L-V exchange–correlation definition.
///
/// This is the DFT half of the double hybrid only. The wave-function half —
/// `λ · E_c,LinLCCD(hh)` evaluated with an erfc(ω)-attenuated operator on the
/// converged Kohn–Sham orbitals — is added post-SCF by the double-hybrid driver.
/// Using this `XcDef` on its own yields a range-separated hybrid, NOT ωB97X-L-V.
pub fn wb97x_l_v_def(nspin: u32) -> Result<XcDef, LibxcError> {
    let mut f = XcFunctional::new("HYB_GGA_XC_WB97X_V", nspin)?;
    apply_named_ext_params(&mut f, &WB97X_L_V_EXT_PARAMS)?;
    f.set_family(FunctionalFamily::RangeSepGga);

    // Take CAM coefficients from our overrides rather than from libxc's cached
    // hybrid struct: xc_func_set_ext_params updates the functional's internal
    // parameters, but ferric reads CAM separately to build the SR/LR exchange
    // operators, and those MUST agree with the ω/α/β we just set.
    let cam = CamCoeffs { omega: 0.1, c_sr: 1.0 + (-0.4), c_lr: 1.0 };

    Ok(XcDef {
        funcs: vec![f],
        cam: Some(cam),
        vv10: Some(WB97X_L_V_VV10),
        b3lyp_mix: None,
    })
}

/// Build an XC definition from a functional name and spin count, using default ω.
pub fn xc_def_from_name_nspin(name: &str, nspin: u32) -> Result<XcDef, LibxcError> {
    let upper = name.to_uppercase();

    // ωB97X-L-V: stock ωB97X-V form with the paper's re-fitted coefficients.
    // Checked before the friendly-name layer so it cannot collide with WB97XV.
    {
        let key: String =
            upper.chars().filter(|c| *c != '-' && *c != '_').collect();
        if key == "WB97XLV" {
            return wb97x_l_v_def(nspin);
        }
    }

    // Friendly-name alias layer: map common chemistry names to canonical libxc
    // component identifiers. Composite (LDA/GGA/mGGA) → X+C pair; hybrid/RSH → one.
    if let Some(components) = friendly_to_libxc(name) {
        // Single combined identifier ⇒ hybrid or range-separated functional.
        if components.len() == 1 {
            let mut f = XcFunctional::new(components[0], nspin)?;
            let cam = f.cam_coefficients();
            let vv10 = f.vv10_coeffs();
            if cam.is_some() {
                f.set_family(FunctionalFamily::RangeSepGga);
                return Ok(XcDef { funcs: vec![f], cam, vv10, b3lyp_mix: None });
            }
            let mix = f.exact_exchange_mix();
            f.set_family(FunctionalFamily::HybridGga);
            return Ok(XcDef {
                funcs: vec![f],
                cam: None,
                vv10,
                b3lyp_mix: if mix != 0.0 { Some(mix) } else { None },
            });
        }
        // Exchange + correlation pair ⇒ pure LDA, GGA, or meta-GGA. Detect the
        // family from the component identifier prefix (LDA_ / MGGA_ / else GGA).
        let fam = family_from_component_id(components[0]);
        let mut funcs = Vec::with_capacity(components.len());
        for id in components {
            let mut f = XcFunctional::new(id, nspin)?;
            f.set_family(fam);
            funcs.push(f);
        }
        return Ok(XcDef { funcs, cam: None, vv10: None, b3lyp_mix: None });
    }

    // Raw (non-friendly) meta-GGA identifiers: only the semilocal SCAN / r2SCAN
    // / TPSS families ferric can evaluate (τ-only, no Laplacian) are allowed
    // through as a single-component functional. Any other raw `MGGA_*` name is
    // rejected up front rather than mis-evaluated — including hybrid/deorbitalized
    // variants (HYB_MGGA_*, *SCANL, *_VV10, *_RVV10) which ferric has no path for.
    if upper.contains("MGGA") {
        let is_supported_raw_mgga = matches!(
            upper.as_str(),
            "MGGA_X_SCAN" | "MGGA_C_SCAN"
                | "MGGA_X_R2SCAN" | "MGGA_C_R2SCAN"
                | "MGGA_X_TPSS" | "MGGA_C_TPSS"
        );
        if !is_supported_raw_mgga {
            return Err(LibxcError::Unsupported(format!(
                "{name}: only SCAN / r2SCAN / TPSS meta-GGAs are supported \
                 (τ-only kernels; no Laplacian, hybrid-mGGA, or deorbitalized \
                 variants)"
            )));
        }
        let mut f = XcFunctional::new(name, nspin)?;
        f.set_family(FunctionalFamily::MetaGga);
        return Ok(XcDef { funcs: vec![f], cam: None, vv10: None, b3lyp_mix: None });
    }

    // Unrecognized friendly name: treat as a raw libxc identifier.
    {
        let mut f = XcFunctional::new(name, nspin)?;
        let cam = f.cam_coefficients();
        let vv10 = f.vv10_coeffs();
        if cam.is_some() {
            f.set_family(FunctionalFamily::RangeSepGga);
            Ok(XcDef { funcs: vec![f], cam, vv10, b3lyp_mix: None })
        } else {
            let mix = f.exact_exchange_mix();
            if mix != 0.0 {
                f.set_family(FunctionalFamily::HybridGga);
                Ok(XcDef {
                    funcs: vec![f],
                    cam: None,
                    vv10,
                    b3lyp_mix: Some(mix),
                })
            } else if name.to_uppercase().contains("LDA") {
                f.set_family(FunctionalFamily::Lda);
                Ok(XcDef { funcs: vec![f], cam: None, vv10, b3lyp_mix: None })
            } else {
                f.set_family(FunctionalFamily::Gga);
                Ok(XcDef { funcs: vec![f], cam: None, vv10, b3lyp_mix: None })
            }
        }
    }
}

/// Family of a canonical libxc component identifier, from its prefix.
/// `LDA_*` → Lda, `MGGA_*` → MetaGga, everything else → Gga (the pure-pair
/// path never sees hybrid/RSH single identifiers).
fn family_from_component_id(id: &str) -> FunctionalFamily {
    if id.starts_with("LDA") {
        FunctionalFamily::Lda
    } else if id.starts_with("MGGA") {
        FunctionalFamily::MetaGga
    } else {
        FunctionalFamily::Gga
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// PBE0 is libxc HYB_GGA_XC_PBEH, not HYB_GGA_XC_PBE0 (which does not
    /// exist). The friendly name "PBE0" must resolve and be a plain hybrid
    /// with ~25% exact exchange.
    #[test]
    fn pbe0_resolves_as_hybrid() {
        let def = xc_def_from_name("PBE0").expect("PBE0 should resolve");
        let mix = def.b3lyp_mix.expect("PBE0 is a hybrid; mix should be set");
        assert!(
            (mix - 0.25).abs() < 1e-6,
            "PBE0 exact-exchange fraction should be 0.25, got {mix}"
        );
        assert!(def.cam.is_none(), "PBE0 is not range-separated");
    }

    /// Common friendly names should all resolve to evaluable (LDA/GGA/hybrid)
    /// functionals.
    #[test]
    fn common_friendly_names_resolve() {
        for name in ["LDA", "PBE", "PBE0", "B3LYP", "BLYP", "PBEsol"] {
            assert!(
                xc_def_from_name(name).is_ok(),
                "friendly name {name} should resolve"
            );
        }
    }

    /// SCAN / r2SCAN friendly names resolve to a two-component meta-GGA pair,
    /// both tagged `FunctionalFamily::MetaGga`, with no exact exchange.
    #[test]
    fn scan_and_r2scan_resolve_as_metagga() {
        for name in ["SCAN", "r2SCAN", "R2SCAN"] {
            let def = xc_def_from_name(name).unwrap_or_else(|e| {
                panic!("{name} should resolve as meta-GGA, got {e:?}")
            });
            assert_eq!(def.funcs.len(), 2, "{name}: X + C pair");
            for f in &def.funcs {
                assert_eq!(
                    f.family(),
                    FunctionalFamily::MetaGga,
                    "{name}: component must be MetaGga family"
                );
            }
            assert!(def.b3lyp_mix.is_none(), "{name}: pure meta-GGA, no HF mix");
            assert!(def.cam.is_none(), "{name}: not range-separated");
        }
    }

    /// Supported raw meta-GGA identifiers resolve; UNsupported meta-GGA variants
    /// (hybrid, deorbitalized, +VV10) must fail with a clear `Unsupported` error
    /// rather than be silently mis-evaluated (ferric has no path for them).
    #[test]
    fn unsupported_metagga_variants_rejected_clearly() {
        // Supported τ-only kernels resolve.
        assert!(xc_def_from_name("MGGA_X_SCAN").is_ok());
        assert!(xc_def_from_name("MGGA_X_TPSS").is_ok());
        // Unsupported variants: deorbitalized (SCANL, needs Laplacian), +VV10,
        // and hybrid-mGGA all reject.
        for bad in ["MGGA_X_SCANL", "MGGA_C_SCAN_VV10", "HYB_MGGA_XC_R2SCAN"] {
            let ok = matches!(xc_def_from_name(bad), Err(LibxcError::Unsupported(_)));
            assert!(ok, "{bad} should give LibxcError::Unsupported");
        }
    }
}
