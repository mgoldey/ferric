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

    extern "C" {
        pub fn xc_func_alloc() -> *mut XcFuncOpaque;
        pub fn xc_func_init(p: *mut XcFuncOpaque, functional: c_int, nspin: c_int) -> c_int;
        pub fn xc_func_end(p: *mut XcFuncOpaque);
        pub fn xc_func_free(p: *mut XcFuncOpaque);

        pub fn xc_functional_get_number(name: *const c_char) -> c_int;

        // `np` is `size_t` in the libxc 5.x C header; `usize` has identical
        // size and alignment on all supported targets (LP64 Linux x86_64 / aarch64).
        pub fn xc_lda_exc_vxc(
            p: *const XcFuncOpaque,
            np: usize,
            rho: *const f64,
            zk: *mut f64,
            vrho: *mut f64,
        );

        pub fn xc_gga_exc_vxc(
            p: *const XcFuncOpaque,
            np: usize,
            rho: *const f64,
            sigma: *const f64,
            zk: *mut f64,
            vrho: *mut f64,
            vsigma: *mut f64,
        );

        pub fn xc_hyb_cam_coef(
            p: *const XcFuncOpaque,
            omega: *mut f64,
            alpha: *mut f64,
            beta: *mut f64,
        );

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
// Public error type
// ---------------------------------------------------------------------------

#[derive(Error, Debug)]
pub enum LibxcError {
    #[error("unknown libxc functional name: {0}")]
    UnknownName(String),
    #[error("xc_func_init failed for {name}: rc={rc}")]
    InitFailed { name: String, rc: c_int },
    #[error("CString conversion failed: {0}")]
    BadName(String),
    #[error("xc_func_alloc returned null")]
    AllocFailed,
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
        let _guard = LIBXC_INIT_LOCK.lock().unwrap_or_else(|p| p.into_inner());

        // SAFETY: c_name lives until the end of this statement; xc_functional_get_number
        // reads the C string and returns a (possibly negative) integer ID without retaining
        // the pointer.
        let id = unsafe { ffi::xc_functional_get_number(c_name.as_ptr()) };
        if id < 0 {
            return Err(LibxcError::UnknownName(name.to_string()));
        }

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
            return Err(LibxcError::InitFailed { name: name.to_string(), rc });
        }

        let family = infer_family_from_name(name);
        Ok(Self { ptr, family, nspin })
    }

    pub fn family(&self) -> FunctionalFamily {
        self.family
    }

    pub(crate) fn set_family(&mut self, f: FunctionalFamily) {
        self.family = f;
    }

    /// LDA: feed ρ, get ε_xc and v_ρ.
    pub fn eval_lda_unpolarized(&self, rho: &[f64], exc: &mut [f64], vrho: &mut [f64]) {
        let n = rho.len();
        assert_eq!(exc.len(), n, "exc buffer length must match rho.len()");
        assert_eq!(vrho.len(), n, "vrho buffer length must match rho.len()");
        // SAFETY: handle is non-null and fully initialised (constructed by new);
        // buffer lengths are verified above; libxc reads rho and writes exc/vrho
        // for exactly n grid points without retaining any pointers.
        unsafe {
            ffi::xc_lda_exc_vxc(
                self.ptr as *const _,
                n,
                rho.as_ptr(),
                exc.as_mut_ptr(),
                vrho.as_mut_ptr(),
            );
        }
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
        unsafe {
            ffi::xc_lda_exc_vxc(
                self.ptr as *const _,
                n,
                rho.as_ptr(),
                exc.as_mut_ptr(),
                vrho.as_mut_ptr(),
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
        unsafe {
            ffi::xc_gga_exc_vxc(
                self.ptr as *const _,
                n,
                rho.as_ptr(),
                sigma.as_ptr(),
                exc.as_mut_ptr(),
                vrho.as_mut_ptr(),
                vsigma.as_mut_ptr(),
            );
        }
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
        // SAFETY: handle is non-null and fully initialised (constructed by new);
        // buffer lengths are verified above; libxc reads rho/sigma and writes
        // exc/vrho/vsigma for exactly n grid points without retaining any pointers.
        unsafe {
            ffi::xc_gga_exc_vxc(
                self.ptr as *const _,
                n,
                rho.as_ptr(),
                sigma.as_ptr(),
                exc.as_mut_ptr(),
                vrho.as_mut_ptr(),
                vsigma.as_mut_ptr(),
            );
        }
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
pub struct XcDef {
    pub funcs: Vec<XcFunctional>,
    pub cam: Option<CamCoeffs>,
    pub vv10: Option<Vv10Params>,
    /// For plain hybrids (ω=0), the exact-exchange fraction.
    pub b3lyp_mix: Option<f64>,
}

/// Resolve a human-friendly functional name to an `XcDef`. nspin=1 for
/// closed-shell, nspin=2 for spin-polarized (UKS / ROKS).
///
/// Supported short names: `"LDA"`, `"PBE"`, `"B3LYP"`, `"wB97X-V"`.
/// Anything else is passed directly to libxc as a raw functional name.
pub fn xc_def_from_name(name: &str) -> Result<XcDef, LibxcError> {
    xc_def_from_name_nspin(name, 1)
}

/// nspin-aware variant. Use 1 for closed-shell and 2 for UKS / ROKS.
pub fn xc_def_from_name_nspin(name: &str, nspin: u32) -> Result<XcDef, LibxcError> {
    match name {
        "LDA" => {
            let mut x = XcFunctional::new("LDA_X", nspin)?;
            x.set_family(FunctionalFamily::Lda);
            let mut c = XcFunctional::new("LDA_C_VWN", nspin)?;
            c.set_family(FunctionalFamily::Lda);
            Ok(XcDef { funcs: vec![x, c], cam: None, vv10: None, b3lyp_mix: None })
        }
        "PBE" => {
            let mut x = XcFunctional::new("GGA_X_PBE", nspin)?;
            x.set_family(FunctionalFamily::Gga);
            let mut c = XcFunctional::new("GGA_C_PBE", nspin)?;
            c.set_family(FunctionalFamily::Gga);
            Ok(XcDef { funcs: vec![x, c], cam: None, vv10: None, b3lyp_mix: None })
        }
        "B3LYP" => {
            let mut f = XcFunctional::new("HYB_GGA_XC_B3LYP", nspin)?;
            f.set_family(FunctionalFamily::HybridGga);
            let mix = f.exact_exchange_mix();
            Ok(XcDef {
                funcs: vec![f],
                cam: None,
                vv10: None,
                b3lyp_mix: if mix != 0.0 { Some(mix) } else { None },
            })
        }
        "wB97X-V" | "WB97X-V" | "wb97x-v" => {
            let mut f = XcFunctional::new("HYB_GGA_XC_WB97X_V", nspin)?;
            f.set_family(FunctionalFamily::RangeSepGga);
            let cam = f.cam_coefficients();
            let vv10 = f.vv10_coeffs();
            Ok(XcDef { funcs: vec![f], cam, vv10, b3lyp_mix: None })
        }
        other => {
            let mut f = XcFunctional::new(other, nspin)?;
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
                } else if other.to_uppercase().contains("LDA") {
                    f.set_family(FunctionalFamily::Lda);
                    Ok(XcDef { funcs: vec![f], cam: None, vv10, b3lyp_mix: None })
                } else {
                    f.set_family(FunctionalFamily::Gga);
                    Ok(XcDef { funcs: vec![f], cam: None, vv10, b3lyp_mix: None })
                }
            }
        }
    }
}
