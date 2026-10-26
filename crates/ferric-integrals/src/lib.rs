//! Electron integral evaluation via libint2 FFI.
//!
//! This crate wraps libint2 through a C++ shim (`shim/shim.cc`) to compute:
//!
//! - **One-electron integrals**: overlap, kinetic, nuclear attraction
//! - **Two-electron repulsion integrals** (ERIs): 4-center shell quartets
//! - **Three-center integrals** (P|mu nu) for density fitting / RI methods
//! - **Two-center integrals** (P|Q) for Coulomb metric
//! - **Derivative integrals**: first derivatives of 1e and 2e integrals
//! - **Schwarz screening**: upper bounds on shell-pair integrals
//!
//! The [`basis_bridge::PreparedBasis`] struct owns the libint2 basis handle and
//! manages the Rust-to-C++ lifetime boundary.

/// GTO basis-function evaluation on real-space grids (Cartesian, Becke-Lebedev, etc.).
pub mod ao_grid;
/// Third AO derivatives on points (fixed-grid GGA Hessian).
pub mod ao_grid_third;
/// [`PreparedBasis`](basis_bridge::PreparedBasis): Rust↔libint2 basis handle and shell metadata.
pub mod basis_bridge;
/// BLAS thread-count guard (re-exported from ferric-core for convenience).
pub mod blas_threads;
/// Complementary auxiliary basis sets (CABS) for F12 methods.
pub mod cabs;
/// Per-grid-point three-center-one-electron AO blocks `A^g_{mu,nu}` for
/// seminumerical exchange (COSX / sn-LinK).
pub mod cosx_a;
/// Shell-pair screening bound shared by `cosx_a` and `md3c1e` (Hölder bound on
/// the primitive expansion; never underestimates, decays as `K_AB / R`).
pub mod cosx_screen;
/// CSAM (combined Schwarz approximation) integral screening: the
/// NON-RIGOROUS multiplicative estimate of Eqs. (9)/(11)/(12) of the same
/// paper, ported from Psi4's `shell_significant_csam()`. Tighter than plain
/// Schwarz, but it can UNDERESTIMATE the true integral — an opt-in
/// accuracy-vs-threshold tradeoff, not a free speedup. Contrast [`csb`] below,
/// which is the rigorous member of the family.
pub mod csam;
/// CSB (combined Schwarz bound) integral screening: the RIGOROUS
/// `min{Q_uv Q_ls, M_ul M_vs, M_us M_vl}` bound of Thompson & Ochsenfeld,
/// JCP 147, 144101 (2017), Eq. (8). Never looser than plain Schwarz.
pub mod csb;
/// Effective core potential integral evaluation.
pub mod ecp;
/// Raw C-ABI bindings to the ECP shim (`shim/ecp_shim.cc`).
pub mod ecp_ffi;
/// ferric-owned ECP quadrature (pure Rust): the `Quadrature` backend of
/// [`ecp`] — analytic angular projection, windowed radial quadrature.
pub mod ecp_quad;
/// Low-level integral engine: 2e/3c/2c shell-block compute calls.
pub mod engine;
/// Thread-safe pool of integral engines for rayon-parallel Fock builds.
pub mod engine_pool;
/// Raw C-ABI bindings to the libint2 shim (`shim/shim.cc`).
pub mod ffi;
/// From-scratch McMurchie-Davidson 3-center-1-electron kernel producing the
/// same `A^g` blocks as `cosx_a` for a BATCH of grid points, grid axis innermost.
pub mod md3c1e;
/// One-electron integrals: overlap, kinetic, nuclear attraction, multipole moments.
pub mod oneelectron;
/// Integral operator kinds: Coulomb, erf/erfc-attenuated, Yukawa, Slater geminal.
pub mod operator;
/// QQR distance-dependent integral screening (Ochsenfeld-style).
pub mod qqr3;
/// Schwarz upper bounds for integral screening.
pub mod schwarz;
/// [`SiteBasis`](site_basis::SiteBasis): one Gaussian per MM site as a fake auxiliary
/// `PreparedBasis`, used for Gaussian-smeared QM/MM charge potentials and their gradients.
pub mod site_basis;
/// [`ThreeIndexSource`](three_index_source::ThreeIndexSource) trait: abstract 3-index tensor provider.
pub mod three_index_source;
/// Three-center density-fitting integrals (P|mu nu) with optional batching.
pub mod threeindex;

/// Version of the libint2 headers the integral shim was compiled against
/// (e.g. `"2.7.2"`), or `"unknown"` if those headers define no
/// `LIBINT_VERSION`. Reported by `ferric --version` and `ferric.build_info()`.
pub fn libint_version() -> &'static str {
    // SAFETY: the shim returns a pointer to a string literal (static storage,
    // NUL-terminated, never NULL) and touches no libint state.
    let s = unsafe { std::ffi::CStr::from_ptr(ffi::scf_libint_version()) };
    s.to_str().unwrap_or("unknown")
}

pub use engine::libint_max_deriv_order;

/// Integral classes reported by [`libint_max_am`], in the shim's `cls` order
/// (the order `ferric --version` prints them in).
pub const LIBINT_AM_CLASSES: [&str; 4] = ["eri", "eri3", "eri2", "onebody"];

/// Maximum angular momentum the linked libint2 was generated for, per
/// integral class, indexed by derivative order.
///
/// Keys are `"eri"` (4-centre, `LIBINT2_MAX_AM_eri*`), `"eri3"`
/// (`LIBINT2_MAX_AM_3eri*`), `"eri2"` (`LIBINT2_MAX_AM_2eri*`) and
/// `"onebody"` (`LIBINT2_MAX_AM_overlap*`). Entry `d` is the limit for
/// derivative order `d`; orders the build lacks are dropped from the end, so
/// a first-derivative-only build gives two entries. This is how the PyPI
/// wheel (small libint) and a conda-forge 2.13.1 build are told apart, since
/// both carry the same version and commit.
pub fn libint_max_am() -> std::collections::BTreeMap<&'static str, Vec<i32>> {
    LIBINT_AM_CLASSES
        .iter()
        .enumerate()
        .map(|(cls, &name)| {
            // SAFETY: returns a compile-time constant; any argument is valid.
            let mut ams: Vec<i32> = (0..3)
                .map(|deriv| unsafe { ffi::scf_libint_max_am(cls as i32, deriv) })
                .collect();
            while ams.last() == Some(&-1) {
                ams.pop();
            }
            (name, ams)
        })
        .collect()
}

/// `LIBINT2_MAX_AM_R12kG12` of the linked libint2 (geminal/F12 integrals),
/// or `None` if the build lacks `LIBINT2_SUPPORT_G12` or that limit.
pub fn libint_g12_max_am() -> Option<i32> {
    // SAFETY: returns a compile-time constant, no arguments.
    let am = unsafe { ffi::scf_libint_g12_max_am() };
    (am >= 0).then_some(am)
}

/// How the integral shim links libint2: `"static"` (a prefix with
/// `libint2.a`) or `"shared"` (`libint2.so`, e.g. conda-forge 2.13.1).
pub fn libint_linkage() -> &'static str {
    env!("FERRIC_LIBINT_LINKAGE")
}
