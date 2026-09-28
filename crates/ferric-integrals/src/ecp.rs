//! ECP integrals: the libecpint shim ([`crate::ecp_ffi`]) or ferric's own
//! quadrature ([`crate::ecp_quad`]), selected by [`EcpBackend`].
//!
//! Computes the dense **spherical** ECP matrix `V_ECP` for a molecule with ECP
//! centers, matching libint's spherical AO basis (and PySCF's `ECPscalar`).
//!
//! Both backends compute integrals over **bare Cartesian** Gaussians and apply
//! no internal normalization. To match the production (spherical) convention
//! this wrapper:
//!   1. requires the caller to supply contraction coefficients with the
//!      primitive normalization `gto_norm(l, α)` folded in (the bare-Cartesian
//!      convention libcint uses with `cart=True`);
//!   2. gets the Cartesian `V_ECP` from the selected backend (identical
//!      layouts: CCA order, row-major);
//!   3. applies the per-shell Cartesian→spherical transform `Cᵀ V Cᵀ` using the
//!      libcint `cart2sph` matrices.
//!
//! Verified: `c2sᵀ · (gto_norm-folded libecpint Cartesian) · c2s` reproduces
//! PySCF's spherical `ECPscalar` to ~1e-9 (see `tests/ecp_matrix.rs`); the
//! quadrature backend to ~1e-13 (`tests/ecp_quadrature.rs`).
//!
//! Backend selection: every public function reads `FERRIC_ECP_BACKEND`
//! (`quadrature` — the default — or `libecpint`; anything else is an error,
//! never a silent default); the `*_with_backend` variants take it explicitly.

use crate::ecp_ffi::{
    ferric_ecp_block, ferric_ecp_block_deriv, ferric_ecp_matrix, ferric_ecp_matrix_deriv,
    ferric_ecp_natoms, CEcpCenter, CEcpGShell, FERRIC_ECP_OK,
};
use crate::ecp_quad::{self, QuadKnobs};
use ferric_core::config::{accept_any, ConfigVar, Resolved};
use ferric_core::FerricError;
use std::os::raw::c_int;

/// Which engine evaluates the ECP integrals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EcpBackend {
    /// The libecpint C++ library via `shim/ecp_shim.cc`, kept as a
    /// cross-check backend: its projector values carry ~1e-7..3.5e-5
    /// non-smooth error (FINDINGS "ECP derivative clean-band discrepancy —
    /// 2026-09-27"; tests/ecp_quadrature.rs parity).
    Libecpint,
    /// ferric's own analytic-angular / windowed-radial quadrature
    /// ([`crate::ecp_quad`], FINDINGS "Iteration 25"). The default.
    Quadrature,
}

impl EcpBackend {
    /// Strict parser: exactly `libecpint` or `quadrature` (lower case);
    /// anything else is an error.
    pub fn parse_config_str(s: &str) -> Result<Self, String> {
        match s {
            "libecpint" => Ok(Self::Libecpint),
            "quadrature" => Ok(Self::Quadrature),
            other => Err(format!(
                "unknown ECP backend {other:?} (expected \"libecpint\" or \"quadrature\")"
            )),
        }
    }

    /// The config spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Libecpint => "libecpint",
            Self::Quadrature => "quadrature",
        }
    }
}

impl std::fmt::Display for EcpBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// `FERRIC_ECP_BACKEND`: result-affecting, so a malformed value is an error.
static ECP_BACKEND: ConfigVar<EcpBackend> = ConfigVar {
    env_name: "FERRIC_ECP_BACKEND",
    default: EcpBackend::Quadrature,
    parse: EcpBackend::parse_config_str,
    validate: accept_any,
};

/// Resolve the ECP backend with precedence explicit > env > default
/// (`get` is the env lookup — inject a closure in tests).
pub fn resolve_ecp_backend(
    explicit: Option<EcpBackend>,
    get: impl Fn(&str) -> Option<String>,
) -> Result<Resolved<EcpBackend>, FerricError> {
    ECP_BACKEND
        .resolve(explicit, get)
        .map_err(|e| FerricError::General(format!("FERRIC_ECP_BACKEND: {e}")))
}

/// The process-wide ECP backend (`FERRIC_ECP_BACKEND`, default quadrature).
pub fn ecp_backend() -> Result<EcpBackend, FerricError> {
    resolve_ecp_backend(None, ferric_core::config::env_lookup).map(|r| r.value)
}

/// A Cartesian Gaussian basis shell for ECP evaluation.
///
/// `coefficients` must already include the primitive normalization
/// `gto_norm(l, α)` (i.e. be in the bare-Cartesian convention libecpint expects).
#[derive(Debug, Clone)]
pub struct EcpGaussianShell {
    pub l: i32,
    pub center: [f64; 3],
    pub exponents: Vec<f64>,
    pub coefficients: Vec<f64>,
}

/// One ECP center: a flat list of semilocal primitives, each tagged with its
/// angular momentum, r-power, exponent, and coefficient. The maximum-`am`
/// channel is the local term (libecpint determines this).
#[derive(Debug, Clone)]
pub struct EcpCenter {
    pub center: [f64; 3],
    pub ams: Vec<i32>,
    pub ns: Vec<i32>,
    pub exponents: Vec<f64>,
    pub coefficients: Vec<f64>,
}

/// libcint primitive Gaussian normalization `gto_norm(l, α)`:
/// `sqrt( 2^(2l+3) (l+1)! (2α)^(l+3/2) / ( (2l+2)! √π ) )`.
///
/// Fold this into a stored (non-primitive-normalized) contraction coefficient to
/// obtain the bare-Cartesian coefficient libecpint expects.
pub fn gto_norm(l: i32, alpha: f64) -> f64 {
    fn factorial(n: u64) -> f64 {
        (1..=n).map(|k| k as f64).product()
    }
    let lf = l as f64;
    let num = 2f64.powf(2.0 * lf + 3.0) * factorial((l + 1) as u64) * (2.0 * alpha).powf(lf + 1.5);
    let den = factorial((2 * l + 2) as u64) * std::f64::consts::PI.sqrt();
    (num / den).sqrt()
}

#[inline]
fn ncart(l: i32) -> usize {
    (((l + 1) * (l + 2)) / 2) as usize
}

#[inline]
fn nsph(l: i32) -> usize {
    (2 * l + 1) as usize
}

/// Per-shell Cartesian→spherical transform matrices `C` (ncart × nsph), in
/// libcint convention, row-major. `V_sph = Cᵀ V_cart C`.
/// Supported up to l = 4 (g) — covers def2 / cc-pVnZ-PP orbital bases.
///
/// Convention note (load-bearing for other consumers): the Cartesian rows are
/// in CCA order (`lx` descending, then `ly` descending) and the spherical
/// columns are `m = -l..=+l` — both identical to libint2's STANDARD orderings.
/// The INPUT Cartesians, however, are libcint's radially-normalized ones (the
/// `(l,0,0)` component has self-overlap `4π/(2l+1)`), which is why `C2S0` is
/// `1/√(4π)` rather than 1. ferric/libint2's Cartesian convention normalizes
/// the `(l,0,0)` component to unity, so a consumer in that convention must
/// scale this matrix by `√(4π/(2l+1))` (see `md3c1e::ferric_cart2sph`); the
/// two conventions are related by exactly that shell-wide constant and by
/// nothing else, because both leave the other Cartesian components
/// un-normalized relative to `(l,0,0)`.
pub(crate) fn cart2sph(l: i32) -> &'static [f64] {
    match l {
        0 => &C2S0,
        1 => &C2S1,
        2 => &C2S2,
        3 => &C2S3,
        4 => &C2S4,
        _ => &[],
    }
}

// l=0: (1,1)
static C2S0: [f64; 1] = [0.282_094_791_773_878_14];
// l=1: (3,3)
static C2S1: [f64; 9] = [
    0.488_602_511_902_919_9,
    0.0,
    0.0,
    0.0,
    0.488_602_511_902_919_9,
    0.0,
    0.0,
    0.0,
    0.488_602_511_902_919_9,
];
// l=2: (6,5)
static C2S2: [f64; 30] = [
    0.0,
    0.0,
    -0.315_391_565_252_52,
    0.0,
    0.546_274_215_296_039_6,
    1.092_548_430_592_079_2,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.092_548_430_592_079_2,
    0.0,
    0.0,
    0.0,
    -0.315_391_565_252_52,
    0.0,
    -0.546_274_215_296_039_6,
    0.0,
    1.092_548_430_592_079_2,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.630_783_130_505_04,
    0.0,
    0.0,
];
// l=3: (10,7)
static C2S3: [f64; 70] = [
    0.0,
    0.0,
    0.0,
    0.0,
    -0.457_045_799_464_465_7,
    0.0,
    0.590_043_589_926_643_5,
    1.770_130_769_779_930_4,
    0.0,
    -0.457_045_799_464_465_7,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    -1.119_528_997_770_346_2,
    0.0,
    1.445_305_721_320_277_1,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    -0.457_045_799_464_465_7,
    0.0,
    -1.770_130_769_779_930_4,
    0.0,
    2.890_611_442_640_554_3,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.828_183_197_857_862_9,
    0.0,
    0.0,
    -0.590_043_589_926_643_5,
    0.0,
    -0.457_045_799_464_465_7,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    -1.119_528_997_770_346_2,
    0.0,
    -1.445_305_721_320_277_1,
    0.0,
    0.0,
    0.0,
    1.828_183_197_857_862_9,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.746_352_665_180_230_8,
    0.0,
    0.0,
    0.0,
];
// l=4: (15,9)
static C2S4: [f64; 135] = [
    0.0,
    0.0,
    0.0,
    0.0,
    0.317_356_640_745_612_93,
    0.0,
    -0.473_087_347_878_78,
    0.0,
    0.625_835_735_449_176_1,
    2.503_342_941_796_704_6,
    0.0,
    -0.946_174_695_757_56,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    -2.007_139_630_671_867_6,
    0.0,
    1.770_130_769_779_930_4,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.634_713_281_491_225_9,
    0.0,
    0.0,
    0.0,
    -3.755_014_412_695_057,
    0.0,
    5.310_392_309_339_791,
    0.0,
    -2.007_139_630_671_867_6,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    -2.538_853_125_964_903_4,
    0.0,
    2.838_524_087_272_680_2,
    0.0,
    0.0,
    -2.503_342_941_796_704_6,
    0.0,
    -0.946_174_695_757_56,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    -2.007_139_630_671_867_6,
    0.0,
    -5.310_392_309_339_791,
    0.0,
    0.0,
    0.0,
    5.677_048_174_545_360_5,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    2.676_186_174_229_157,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.317_356_640_745_612_93,
    0.0,
    0.473_087_347_878_78,
    0.0,
    0.625_835_735_449_176_1,
    0.0,
    -1.770_130_769_779_930_4,
    0.0,
    -2.007_139_630_671_867_6,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    -2.538_853_125_964_903_4,
    0.0,
    -2.838_524_087_272_680_2,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    2.676_186_174_229_157,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.846_284_375_321_634_5,
    0.0,
    0.0,
    0.0,
    0.0,
];

/// Compute the dense spherical ECP matrix `V_ECP` for the given Cartesian shells
/// and ECP centers. Returns an `nsph × nsph` matrix (row-major), where `nsph`
/// is the total number of spherical basis functions (`Σ_s (2 l_s + 1)`).
///
/// `shells` coefficients must be in the bare-Cartesian convention (primitive
/// `gto_norm` folded in). The shell order defines the output AO order, matching
/// libint's spherical ordering per shell. Backend: [`ecp_backend`].
pub fn ecp_matrix_spherical(
    shells: &[EcpGaussianShell],
    ecps: &[EcpCenter],
) -> Result<Vec<f64>, FerricError> {
    ecp_matrix_spherical_with_backend(ecp_backend()?, shells, ecps)
}

/// [`ecp_matrix_spherical`] with an explicit backend.
pub fn ecp_matrix_spherical_with_backend(
    backend: EcpBackend,
    shells: &[EcpGaussianShell],
    ecps: &[EcpCenter],
) -> Result<Vec<f64>, FerricError> {
    if shells.is_empty() || ecps.is_empty() {
        return Err(FerricError::Libint(
            "ecp_matrix_spherical: empty input".into(),
        ));
    }
    for sh in shells {
        if sh.l > 4 {
            return Err(FerricError::Libint(format!(
                "ECP: angular momentum l={} > 4 not supported by cart2sph table",
                sh.l
            )));
        }
    }
    let v_cart = match backend {
        EcpBackend::Libecpint => matrix_cart_libecpint(shells, ecps)?,
        EcpBackend::Quadrature => {
            ecp_quad::validate(&shells.iter().collect::<Vec<_>>(), ecps)?;
            ecp_quad::matrix_cart(shells, ecps, &QuadKnobs::default())
        }
    };
    Ok(cart_to_sph(shells, &v_cart))
}

/// libecpint's Cartesian molecular matrix (`ferric_ecp_matrix`).
fn matrix_cart_libecpint(
    shells: &[EcpGaussianShell],
    ecps: &[EcpCenter],
) -> Result<Vec<f64>, FerricError> {
    let (c_shells, c_ecps, _keep) = build_c_arrays(shells, ecps);

    let ncart_total: usize = shells.iter().map(|s| ncart(s.l)).sum();
    let mut v_cart = vec![0.0f64; ncart_total * ncart_total];
    // SAFETY: c_shells/c_ecps are valid C-repr arrays built by build_c_arrays
    // (backed by _keep). v_cart is pre-sized to ncart_total². Status checked.
    let status = unsafe {
        ferric_ecp_matrix(
            c_shells.as_ptr(),
            c_shells.len() as c_int,
            c_ecps.as_ptr(),
            c_ecps.len() as c_int,
            v_cart.as_mut_ptr(),
        )
    };
    if status != FERRIC_ECP_OK {
        return Err(FerricError::Libint(format!(
            "ferric_ecp_matrix failed: {status}"
        )));
    }
    Ok(v_cart)
}

/// Rectangular spherical ECP block between two INDEPENDENT shell lists at
/// arbitrary centres: `V[p, q] = Σ_{(a, b, u) enabled} ⟨bra_a|U_u|ket_b⟩`,
/// `nsph(bra) × nsph(ket)`, row-major, per-shell `Cᵀ V_cart C` exactly as
/// [`ecp_matrix_spherical`] (so a square call with `bra == ket` and every
/// triple enabled reproduces its matrix, up to libecpint's own bra-side
/// screen that `ecp_matrix_spherical` applies and this does not).
///
/// This is the periodic-ECP kernel: the ket shells may be lattice images of
/// the bra shells and the ECP centres lattice images of the atoms. `mask`
/// (`None` = every triple) has length `bra.len() * ket.len() * ecps.len()`,
/// index `(a * ket.len() + b) * ecps.len() + u`; the caller owns the
/// screening, so the truncation is the caller's to report. Coefficients are
/// in the bare-Cartesian convention (`gto_norm` folded in).
pub fn ecp_block_spherical(
    bra: &[EcpGaussianShell],
    ket: &[EcpGaussianShell],
    ecps: &[EcpCenter],
    mask: Option<&[u8]>,
) -> Result<Vec<f64>, FerricError> {
    ecp_block_spherical_with_backend(ecp_backend()?, bra, ket, ecps, mask)
}

/// [`ecp_block_spherical`] with an explicit backend.
pub fn ecp_block_spherical_with_backend(
    backend: EcpBackend,
    bra: &[EcpGaussianShell],
    ket: &[EcpGaussianShell],
    ecps: &[EcpCenter],
    mask: Option<&[u8]>,
) -> Result<Vec<f64>, FerricError> {
    match backend {
        EcpBackend::Libecpint => {
            validate_block_inputs("ecp_block_spherical", bra, ket, ecps, mask)?;
            let v_cart = block_cart_libecpint(bra, ket, ecps, mask)?;
            Ok(cart_to_sph_rect(bra, ket, &v_cart))
        }
        EcpBackend::Quadrature => {
            ecp_block_spherical_quadrature_knobs(bra, ket, ecps, mask, &QuadKnobs::default())
        }
    }
}

/// The quadrature backend of [`ecp_block_spherical`] with explicit accuracy
/// knobs — for the screening and negative-control tests only
/// ([`QuadKnobs::default`] is production).
#[doc(hidden)]
pub fn ecp_block_spherical_quadrature_knobs(
    bra: &[EcpGaussianShell],
    ket: &[EcpGaussianShell],
    ecps: &[EcpCenter],
    mask: Option<&[u8]>,
    knobs: &QuadKnobs,
) -> Result<Vec<f64>, FerricError> {
    validate_block_inputs("ecp_block_spherical", bra, ket, ecps, mask)?;
    ecp_quad::validate(&bra.iter().chain(ket).collect::<Vec<_>>(), ecps)?;
    let v_cart = ecp_quad::block_cart(bra, ket, ecps, mask, knobs);
    Ok(cart_to_sph_rect(bra, ket, &v_cart))
}

/// libecpint's Cartesian rectangular block (`ferric_ecp_block`).
fn block_cart_libecpint(
    bra: &[EcpGaussianShell],
    ket: &[EcpGaussianShell],
    ecps: &[EcpCenter],
    mask: Option<&[u8]>,
) -> Result<Vec<f64>, FerricError> {
    let (c_bra, c_ecps, _keep) = build_c_arrays(bra, ecps);
    let (c_ket, _, _keep_ket) = build_c_arrays(ket, &[]);
    let nc_bra: usize = bra.iter().map(|s| ncart(s.l)).sum();
    let nc_ket: usize = ket.iter().map(|s| ncart(s.l)).sum();
    let mut v_cart = vec![0.0f64; nc_bra * nc_ket];
    // SAFETY: c_bra/c_ket/c_ecps are valid C-repr arrays whose pointers alias
    // `bra`/`ket`/`ecps` and `_keep` (all alive across the call); `mask` is
    // null or exactly nbra*nket*necp bytes (checked by the caller); v_cart
    // holds nc_bra*nc_ket doubles and that length is passed for the shim's
    // cross-check. Status checked below.
    let status = unsafe {
        ferric_ecp_block(
            c_bra.as_ptr(),
            c_bra.len() as c_int,
            c_ket.as_ptr(),
            c_ket.len() as c_int,
            c_ecps.as_ptr(),
            c_ecps.len() as c_int,
            mask.map_or(std::ptr::null(), |m| m.as_ptr()),
            v_cart.as_mut_ptr(),
            v_cart.len() as i64,
        )
    };
    if status != FERRIC_ECP_OK {
        return Err(FerricError::Libint(format!(
            "ferric_ecp_block failed: {status}"
        )));
    }
    Ok(v_cart)
}

/// Shared Rust-side checks of the rectangular block kernels (the shim
/// re-validates everything it dereferences): non-empty lists, `l ∈ 0..=4`
/// (the cart2sph table), matching exponent/coefficient lengths, non-ragged
/// ECP term lists and the mask length.
fn validate_block_inputs(
    who: &str,
    bra: &[EcpGaussianShell],
    ket: &[EcpGaussianShell],
    ecps: &[EcpCenter],
    mask: Option<&[u8]>,
) -> Result<(), FerricError> {
    if bra.is_empty() || ket.is_empty() || ecps.is_empty() {
        return Err(FerricError::Libint(format!("{who}: empty input")));
    }
    for sh in bra.iter().chain(ket) {
        if sh.l < 0 || sh.l > 4 {
            return Err(FerricError::Libint(format!(
                "{who}: angular momentum l={} outside 0..=4 (cart2sph table)",
                sh.l
            )));
        }
        if sh.exponents.is_empty() || sh.exponents.len() != sh.coefficients.len() {
            return Err(FerricError::Libint(format!(
                "{who}: shell has {} exponents and {} coefficients",
                sh.exponents.len(),
                sh.coefficients.len()
            )));
        }
    }
    for e in ecps {
        let n = e.ams.len();
        if n == 0 || e.ns.len() != n || e.exponents.len() != n || e.coefficients.len() != n {
            return Err(FerricError::Libint(format!(
                "{who}: ragged or empty ECP term lists"
            )));
        }
    }
    if let Some(m) = mask {
        if m.len() != bra.len() * ket.len() * ecps.len() {
            return Err(FerricError::Libint(format!(
                "{who}: mask has {} entries, expected {} x {} x {}",
                m.len(),
                bra.len(),
                ket.len(),
                ecps.len()
            )));
        }
    }
    Ok(())
}

/// First derivatives of [`ecp_block_spherical`]'s block, split by which
/// centre moves (all `nrow × ncol`, row-major, spherical, same AO order as
/// [`ecp_block_spherical`]).
#[derive(Debug, Clone)]
pub struct EcpBlockDeriv {
    /// Spherical rows (`Σ_bra 2l+1`).
    pub nrow: usize,
    /// Spherical columns (`Σ_ket 2l+1`).
    pub ncol: usize,
    /// `bra[x] = Σ_{(a,b,u) enabled} ∂⟨a|U_u|b⟩/∂A_x` — the bra shell's
    /// centre moves.
    pub bra: [Vec<f64>; 3],
    /// `ket[x] = Σ ∂⟨a|U_u|b⟩/∂B_x` — the ket shell's centre moves.
    pub ket: [Vec<f64>; 3],
    /// `centre[g][x] = Σ_{u: group(u) = g} ∂⟨a|U_u|b⟩/∂C_x
    /// = −(bra + ket)` restricted to that group's triples.
    pub centre: Vec<[Vec<f64>; 3]>,
}

/// First derivatives of the rectangular spherical ECP block
/// ([`ecp_block_spherical`]) with respect to the three moving centres of
/// every enabled triple (bra shell at `A`, ket shell at `B`, ECP centre at
/// `C`) — the periodic-ECP force kernel (`ferric_ecp_block_deriv`).
///
/// `centre_group[u] < ngroup` names the group (e.g. the cell atom of an ECP
/// image) the centre derivative of ECP `u` is accumulated under; the caller
/// folds `bra` rows by the bra shells' atoms and `ket` columns by the ket
/// shells' atoms. No libecpint atom inference is involved.
///
/// The three slots are the TRUE partial derivatives for every triple,
/// including a shell sitting ON its ECP centre: libecpint's
/// `compute_shell_pair_derivative` reports `A = −B, C = 0` there (right only
/// for per-atom totals); the shim instead evaluates both shell derivatives
/// with `left_shell_derivative` unconditionally and sets the centre to
/// `−(bra + ket)` (translation invariance per triple, so `bra + ket +
/// Σ_g centre[g] = 0` element-wise to roundoff). Off-centre this is bitwise
/// libecpint's own derivative.
///
/// Coefficients are bare-Cartesian (`gto_norm` folded in), as for
/// [`ecp_block_spherical`]; `l ≤ 4` (and `l + 1 ≤ LIBECPINT_MAX_L`).
pub fn ecp_block_deriv_spherical(
    bra: &[EcpGaussianShell],
    ket: &[EcpGaussianShell],
    ecps: &[EcpCenter],
    mask: Option<&[u8]>,
    centre_group: &[usize],
    ngroup: usize,
) -> Result<EcpBlockDeriv, FerricError> {
    ecp_block_deriv_spherical_with_backend(
        ecp_backend()?,
        bra,
        ket,
        ecps,
        mask,
        centre_group,
        ngroup,
    )
}

/// [`ecp_block_deriv_spherical`] with an explicit backend. The quadrature
/// backend has no on-centre special case at all (its projections are exact
/// on-centre); its centre slot is `−(bra + ket)` per triple as well.
pub fn ecp_block_deriv_spherical_with_backend(
    backend: EcpBackend,
    bra: &[EcpGaussianShell],
    ket: &[EcpGaussianShell],
    ecps: &[EcpCenter],
    mask: Option<&[u8]>,
    centre_group: &[usize],
    ngroup: usize,
) -> Result<EcpBlockDeriv, FerricError> {
    validate_block_inputs("ecp_block_deriv_spherical", bra, ket, ecps, mask)?;
    validate_centre_groups(ecps.len(), centre_group, ngroup)?;
    let (d_bra, d_ket, d_cen) = match backend {
        EcpBackend::Libecpint => {
            block_deriv_cart_libecpint(bra, ket, ecps, mask, centre_group, ngroup)?
        }
        EcpBackend::Quadrature => {
            ecp_quad::validate(&bra.iter().chain(ket).collect::<Vec<_>>(), ecps)?;
            ecp_quad::block_deriv_cart(
                bra,
                ket,
                ecps,
                mask,
                centre_group,
                ngroup,
                &QuadKnobs::default(),
            )
        }
    };
    let nc_bra: usize = bra.iter().map(|s| ncart(s.l)).sum();
    let nc_ket: usize = ket.iter().map(|s| ncart(s.l)).sum();
    let blk = nc_bra * nc_ket;
    let sph = |c: &[f64]| cart_to_sph_rect(bra, ket, c);
    let three = |v: &[f64]| -> [Vec<f64>; 3] {
        [
            sph(&v[..blk]),
            sph(&v[blk..2 * blk]),
            sph(&v[2 * blk..3 * blk]),
        ]
    };
    let nrow: usize = bra.iter().map(|s| nsph(s.l)).sum();
    let ncol: usize = ket.iter().map(|s| nsph(s.l)).sum();
    Ok(EcpBlockDeriv {
        nrow,
        ncol,
        bra: three(&d_bra),
        ket: three(&d_ket),
        centre: (0..ngroup)
            .map(|g| three(&d_cen[g * 3 * blk..(g + 1) * 3 * blk]))
            .collect(),
    })
}

/// `ngroup ∈ 1..=c_int::MAX`, one group per ECP centre, every group in range.
fn validate_centre_groups(
    necp: usize,
    centre_group: &[usize],
    ngroup: usize,
) -> Result<(), FerricError> {
    if ngroup == 0 || ngroup > c_int::MAX as usize {
        return Err(FerricError::Libint(format!(
            "ecp_block_deriv_spherical: ngroup = {ngroup} must lie in 1..=c_int::MAX"
        )));
    }
    if centre_group.len() != necp {
        return Err(FerricError::Libint(format!(
            "ecp_block_deriv_spherical: {} centre groups for {} ECP centres",
            centre_group.len(),
            necp
        )));
    }
    if let Some((u, &g)) = centre_group.iter().enumerate().find(|&(_, &g)| g >= ngroup) {
        return Err(FerricError::Libint(format!(
            "ecp_block_deriv_spherical: centre_group[{u}] = {g} outside 0..{ngroup}"
        )));
    }
    Ok(())
}

/// libecpint's Cartesian block derivative (`ferric_ecp_block_deriv`):
/// `(bra, ket, centre)` in the layout documented at
/// [`crate::ecp_quad`]'s `block_deriv_cart`.
fn block_deriv_cart_libecpint(
    bra: &[EcpGaussianShell],
    ket: &[EcpGaussianShell],
    ecps: &[EcpCenter],
    mask: Option<&[u8]>,
    centre_group: &[usize],
    ngroup: usize,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), FerricError> {
    let groups: Vec<c_int> = centre_group.iter().map(|&g| g as c_int).collect();
    let (c_bra, c_ecps, _keep) = build_c_arrays(bra, ecps);
    let (c_ket, _, _keep_ket) = build_c_arrays(ket, &[]);
    let nc_bra: usize = bra.iter().map(|s| ncart(s.l)).sum();
    let nc_ket: usize = ket.iter().map(|s| ncart(s.l)).sum();
    let blk = nc_bra * nc_ket;
    let mut d_bra = vec![0.0f64; 3 * blk];
    let mut d_ket = vec![0.0f64; 3 * blk];
    let mut d_cen = vec![0.0f64; ngroup * 3 * blk];
    // SAFETY: c_bra/c_ket/c_ecps alias `bra`/`ket`/`ecps` and `_keep*`, all
    // alive across the call; `mask` is null or nbra*nket*necp bytes and
    // `groups` holds necp ints in 0..ngroup (both checked by the caller);
    // d_bra and d_ket hold 3*blk doubles, d_cen ngroup*3*blk, and blk is
    // passed for the shim's cross-check. Status checked below.
    let status = unsafe {
        ferric_ecp_block_deriv(
            c_bra.as_ptr(),
            c_bra.len() as c_int,
            c_ket.as_ptr(),
            c_ket.len() as c_int,
            c_ecps.as_ptr(),
            c_ecps.len() as c_int,
            mask.map_or(std::ptr::null(), |m| m.as_ptr()),
            groups.as_ptr(),
            ngroup as c_int,
            d_bra.as_mut_ptr(),
            d_ket.as_mut_ptr(),
            d_cen.as_mut_ptr(),
            blk as i64,
        )
    };
    if status != FERRIC_ECP_OK {
        return Err(FerricError::Libint(format!(
            "ferric_ecp_block_deriv failed: {status}"
        )));
    }
    Ok((d_bra, d_ket, d_cen))
}

/// Owns the `c_int` conversions and per-ECP vectors that the `CEcpCenter`
/// pointers alias. Must outlive any FFI call using those pointers.
struct CEcpBacking {
    _ams: Vec<Vec<c_int>>,
    _ns: Vec<Vec<c_int>>,
}

/// Build the C-ABI shell + ECP arrays. The returned [`CEcpBacking`] owns the
/// storage the `CEcpCenter` pointers alias and must be kept alive across the
/// FFI call (the `shells`/`ecps` slices themselves back the other pointers).
fn build_c_arrays(
    shells: &[EcpGaussianShell],
    ecps: &[EcpCenter],
) -> (Vec<CEcpGShell>, Vec<CEcpCenter>, CEcpBacking) {
    let c_shells: Vec<CEcpGShell> = shells
        .iter()
        .map(|s| CEcpGShell {
            l: s.l as c_int,
            nprim: s.exponents.len() as c_int,
            x: s.center[0],
            y: s.center[1],
            z: s.center[2],
            exponents: s.exponents.as_ptr(),
            coefficients: s.coefficients.as_ptr(),
        })
        .collect();

    // ECP int arrays need stable storage of the c_int conversions.
    let ams_store: Vec<Vec<c_int>> = ecps
        .iter()
        .map(|e| e.ams.iter().map(|&a| a as c_int).collect())
        .collect();
    let ns_store: Vec<Vec<c_int>> = ecps
        .iter()
        .map(|e| e.ns.iter().map(|&n| n as c_int).collect())
        .collect();
    let c_ecps: Vec<CEcpCenter> = ecps
        .iter()
        .enumerate()
        .map(|(i, e)| CEcpCenter {
            x: e.center[0],
            y: e.center[1],
            z: e.center[2],
            nterm: e.ams.len() as c_int,
            ams: ams_store[i].as_ptr(),
            ns: ns_store[i].as_ptr(),
            exponents: e.exponents.as_ptr(),
            coefficients: e.coefficients.as_ptr(),
        })
        .collect();

    (
        c_shells,
        c_ecps,
        CEcpBacking {
            _ams: ams_store,
            _ns: ns_store,
        },
    )
}

/// Cartesian -> spherical for one dense `ncart x ncart` matrix:
/// `V_sph = Cᵀ V_cart C`, applied block-diagonally per shell pair.
/// Returns an `nsph x nsph` row-major matrix.
fn cart_to_sph(shells: &[EcpGaussianShell], v_cart: &[f64]) -> Vec<f64> {
    let ncart_total: usize = shells.iter().map(|s| ncart(s.l)).sum();
    let nsph_total: usize = shells.iter().map(|s| nsph(s.l)).sum();
    // Cartesian and spherical offsets per shell.
    let mut cart_off = Vec::with_capacity(shells.len());
    let mut sph_off = Vec::with_capacity(shells.len());
    {
        let mut c = 0;
        let mut s = 0;
        for sh in shells {
            cart_off.push(c);
            sph_off.push(s);
            c += ncart(sh.l);
            s += nsph(sh.l);
        }
    }

    let mut v_sph = vec![0.0f64; nsph_total * nsph_total];
    // For each shell pair (A, B): V_sph[A,B] = C_Aᵀ V_cart[A,B] C_B.
    for (a, sha) in shells.iter().enumerate() {
        let nca = ncart(sha.l);
        let nsa = nsph(sha.l);
        let ca = cart2sph(sha.l); // (nca x nsa)
        for (b, shb) in shells.iter().enumerate() {
            let ncb = ncart(shb.l);
            let nsb = nsph(shb.l);
            let cb = cart2sph(shb.l); // (ncb x nsb)

            // tmp = V_cart[A,B] (nca x ncb) · C_B (ncb x nsb)  ->  (nca x nsb)
            let mut tmp = vec![0.0f64; nca * nsb];
            for i in 0..nca {
                for q in 0..nsb {
                    let mut acc = 0.0;
                    for k in 0..ncb {
                        let vc = v_cart[(cart_off[a] + i) * ncart_total + (cart_off[b] + k)];
                        acc += vc * cb[k * nsb + q];
                    }
                    tmp[i * nsb + q] = acc;
                }
            }
            // V_sph[A,B] = C_Aᵀ (nsa x nca) · tmp (nca x nsb) -> (nsa x nsb)
            for p in 0..nsa {
                for q in 0..nsb {
                    let mut acc = 0.0;
                    for i in 0..nca {
                        acc += ca[i * nsa + p] * tmp[i * nsb + q];
                    }
                    v_sph[(sph_off[a] + p) * nsph_total + (sph_off[b] + q)] = acc;
                }
            }
        }
    }

    v_sph
}

/// Rectangular Cartesian -> spherical: `V_sph[A,B] = C_Aᵀ V_cart[A,B] C_B`
/// for every (bra shell A, ket shell B). `v_cart` is
/// `ncart(bra) × ncart(ket)` row-major; returns `nsph(bra) × nsph(ket)`.
fn cart_to_sph_rect(
    bra: &[EcpGaussianShell],
    ket: &[EcpGaussianShell],
    v_cart: &[f64],
) -> Vec<f64> {
    let offsets = |sh: &[EcpGaussianShell]| {
        let (mut c, mut s) = (0usize, 0usize);
        let mut out = Vec::with_capacity(sh.len());
        for x in sh {
            out.push((c, s));
            c += ncart(x.l);
            s += nsph(x.l);
        }
        (out, c, s)
    };
    let (off_a, _nca_tot, nsa_tot) = offsets(bra);
    let (off_b, ncb_tot, nsb_tot) = offsets(ket);
    let mut v_sph = vec![0.0f64; nsa_tot * nsb_tot];
    for (a, sha) in bra.iter().enumerate() {
        let (ca0, sa0) = off_a[a];
        let nca = ncart(sha.l);
        let nsa = nsph(sha.l);
        let ca = cart2sph(sha.l);
        for (b, shb) in ket.iter().enumerate() {
            let (cb0, sb0) = off_b[b];
            let ncb = ncart(shb.l);
            let nsb = nsph(shb.l);
            let cb = cart2sph(shb.l);
            let mut tmp = vec![0.0f64; nca * nsb];
            for i in 0..nca {
                for q in 0..nsb {
                    let mut acc = 0.0;
                    for k in 0..ncb {
                        acc += v_cart[(ca0 + i) * ncb_tot + (cb0 + k)] * cb[k * nsb + q];
                    }
                    tmp[i * nsb + q] = acc;
                }
            }
            for p in 0..nsa {
                for q in 0..nsb {
                    let mut acc = 0.0;
                    for i in 0..nca {
                        acc += ca[i * nsa + p] * tmp[i * nsb + q];
                    }
                    v_sph[(sa0 + p) * nsb_tot + (sb0 + q)] = acc;
                }
            }
        }
    }
    v_sph
}

/// First derivatives of the spherical ECP matrix with respect to every atomic
/// coordinate: `dV_ECP/dR`.
///
/// Returns `(derivs, natoms)` where `derivs` has length `3 * natoms`, each entry
/// an `nsph × nsph` row-major matrix, ordered `{A_x, A_y, A_z, B_x, ...}`.
///
/// **The atom ordering is libecpint's, not the caller's.** libecpint takes no
/// atom list: it infers atom ids by deduplicating shell and ECP centers (1e-4
/// Bohr tolerance, shells first then ECP centers, in order of first appearance —
/// see `ECPIntegrator::init`). For a molecule where every atom carries at least
/// one basis shell and shells are emitted atom-by-atom, this coincides with the
/// caller's atom order; it does **not** in general. Callers must map ids back
/// via [`ecp_deriv_atom_ids`] rather than assuming 1:1 — see
/// [`crate::oneelectron::ecp_potential_deriv`], which does exactly that. The
/// quadrature backend infers the SAME ids (it shares that dedup), so the
/// contract does not depend on the backend.
///
/// The A/B/C (bra, ket, ECP center) contributions are already summed per atom,
/// so each matrix is the total derivative w.r.t. that one coordinate (for the
/// quadrature backend: the per-atom fold of [`ecp_block_deriv_spherical`]'s
/// three slots). Backend: [`ecp_backend`].
pub fn ecp_matrix_deriv_spherical(
    shells: &[EcpGaussianShell],
    ecps: &[EcpCenter],
) -> Result<(Vec<Vec<f64>>, usize), FerricError> {
    ecp_matrix_deriv_spherical_with_backend(ecp_backend()?, shells, ecps)
}

/// [`ecp_matrix_deriv_spherical`] with an explicit backend.
pub fn ecp_matrix_deriv_spherical_with_backend(
    backend: EcpBackend,
    shells: &[EcpGaussianShell],
    ecps: &[EcpCenter],
) -> Result<(Vec<Vec<f64>>, usize), FerricError> {
    if shells.is_empty() || ecps.is_empty() {
        return Err(FerricError::Libint(
            "ecp_matrix_deriv_spherical: empty input".into(),
        ));
    }
    for sh in shells {
        if sh.l > 4 {
            return Err(FerricError::Libint(format!(
                "ECP gradient: angular momentum l={} > 4 not supported by cart2sph table",
                sh.l
            )));
        }
    }
    let (d_cart, natoms) = match backend {
        EcpBackend::Libecpint => matrix_deriv_cart_libecpint(shells, ecps)?,
        EcpBackend::Quadrature => {
            ecp_quad::validate(&shells.iter().collect::<Vec<_>>(), ecps)?;
            let centres = inferred_atom_centres(shells, ecps);
            let id = |c: &[f64; 3]| {
                centres
                    .iter()
                    .position(|e| same_centre(e, c))
                    .expect("every centre was interned")
            };
            let atom_of_shell: Vec<usize> = shells.iter().map(|s| id(&s.center)).collect();
            let atom_of_ecp: Vec<usize> = ecps.iter().map(|e| id(&e.center)).collect();
            let d = ecp_quad::matrix_deriv_cart(
                shells,
                ecps,
                &atom_of_shell,
                &atom_of_ecp,
                centres.len(),
                &QuadKnobs::default(),
            );
            (d, centres.len())
        }
    };
    let ncart_total: usize = shells.iter().map(|s| ncart(s.l)).sum();
    let block = ncart_total * ncart_total;
    let derivs = (0..3 * natoms)
        .map(|c| cart_to_sph(shells, &d_cart[c * block..(c + 1) * block]))
        .collect();
    Ok((derivs, natoms))
}

/// libecpint's Cartesian molecular derivative (`ferric_ecp_matrix_deriv`) and
/// its inferred atom count.
fn matrix_deriv_cart_libecpint(
    shells: &[EcpGaussianShell],
    ecps: &[EcpCenter],
) -> Result<(Vec<f64>, usize), FerricError> {
    let (c_shells, c_ecps, _keep) = build_c_arrays(shells, ecps);

    // SAFETY: c_shells/c_ecps are valid C-repr arrays (backed by _keep).
    // Returns the number of atoms or a negative error code.
    let natoms = unsafe {
        ferric_ecp_natoms(
            c_shells.as_ptr(),
            c_shells.len() as c_int,
            c_ecps.as_ptr(),
            c_ecps.len() as c_int,
        )
    };
    if natoms <= 0 {
        return Err(FerricError::Libint(format!(
            "ferric_ecp_natoms failed: {natoms}"
        )));
    }
    let natoms = natoms as usize;

    let ncart_total: usize = shells.iter().map(|s| ncart(s.l)).sum();
    let mut d_cart = vec![0.0f64; 3 * natoms * ncart_total * ncart_total];
    let mut got_natoms: c_int = 0;
    // SAFETY: same arrays as above; d_cart sized for 3*natoms*ncart² doubles.
    let status = unsafe {
        ferric_ecp_matrix_deriv(
            c_shells.as_ptr(),
            c_shells.len() as c_int,
            c_ecps.as_ptr(),
            c_ecps.len() as c_int,
            d_cart.as_mut_ptr(),
            &mut got_natoms,
        )
    };
    if status != FERRIC_ECP_OK {
        return Err(FerricError::Libint(format!(
            "ferric_ecp_matrix_deriv failed: {status}"
        )));
    }
    if got_natoms as usize != natoms {
        return Err(FerricError::Libint(format!(
            "ECP gradient: natoms disagreement ({natoms} predicted, {got_natoms} computed)"
        )));
    }
    Ok((d_cart, natoms))
}

/// libecpint's centre-dedup tolerance (L1, Bohr).
const ATOM_TOL: f64 = 1e-4;

fn same_centre(a: &[f64; 3], b: &[f64; 3]) -> bool {
    (a[0] - b[0]).abs() + (a[1] - b[1]).abs() + (a[2] - b[2]).abs() < ATOM_TOL
}

/// `ECPIntegrator::init`'s inferred atom centres: shell centres first in the
/// order given, then any ECP centre not already seen (L1 tolerance 1e-4 Bohr).
fn inferred_atom_centres(shells: &[EcpGaussianShell], ecps: &[EcpCenter]) -> Vec<[f64; 3]> {
    let mut inferred: Vec<[f64; 3]> = Vec::new();
    for c in shells
        .iter()
        .map(|s| s.center)
        .chain(ecps.iter().map(|e| e.center))
    {
        if !inferred.iter().any(|e| same_centre(e, &c)) {
            inferred.push(c);
        }
    }
    inferred
}

/// Map each of libecpint's inferred atom ids back to an index into `centers`
/// (the caller's own atom list, in Bohr).
///
/// Replicates `ECPIntegrator::init`'s dedup order — shell centers first in the
/// order given, then any ECP center not already seen — using the same 1e-4 Bohr
/// L1 tolerance. Returns a vector of length `natoms` whose `i`-th entry is the
/// index into `centers` that libecpint atom `i` corresponds to.
///
/// Errors if any inferred center cannot be matched to a caller atom, which would
/// mean the derivative rows could not be attributed and the gradient would be
/// silently misassigned.
pub fn ecp_deriv_atom_ids(
    shells: &[EcpGaussianShell],
    ecps: &[EcpCenter],
    centers: &[[f64; 3]],
) -> Result<Vec<usize>, FerricError> {
    inferred_atom_centres(shells, ecps)
        .iter()
        .map(|c| {
            centers
                .iter()
                .position(|a| same_centre(a, c))
                .ok_or_else(|| {
                    FerricError::Libint(format!(
                        "ECP gradient: libecpint center [{:.6}, {:.6}, {:.6}] matches no atom; \
                     derivative rows cannot be attributed",
                        c[0], c[1], c[2]
                    ))
                })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Strict backend knob: exact spellings only, unknown -> error (never a
    /// silent default), absent -> quadrature, explicit beats env.
    #[test]
    fn ecp_backend_knob_is_strict() {
        let env =
            |v: &'static str| move |k: &str| (k == "FERRIC_ECP_BACKEND").then(|| v.to_string());
        let none = |_: &str| None::<String>;
        assert_eq!(
            resolve_ecp_backend(None, none).unwrap().value,
            EcpBackend::Quadrature
        );
        assert_eq!(
            resolve_ecp_backend(None, env("quadrature")).unwrap().value,
            EcpBackend::Quadrature
        );
        assert_eq!(
            resolve_ecp_backend(None, env(" libecpint ")).unwrap().value,
            EcpBackend::Libecpint
        );
        for bad in ["", "Quadrature", "quad", "libecp", "pyscf", "1"] {
            assert!(
                resolve_ecp_backend(None, env(bad)).is_err(),
                "{bad:?} accepted"
            );
        }
        assert_eq!(
            resolve_ecp_backend(Some(EcpBackend::Libecpint), env("quadrature"))
                .unwrap()
                .value,
            EcpBackend::Libecpint
        );
        for b in [EcpBackend::Libecpint, EcpBackend::Quadrature] {
            assert_eq!(EcpBackend::parse_config_str(&b.to_string()), Ok(b));
        }
    }

    #[test]
    fn test_gto_norm_matches_libcint() {
        // Cross-checked against pyscf.gto.gto_norm.
        assert!((gto_norm(0, 1.0) - 2.526_475_110_984_259).abs() < 1e-12);
        assert!((gto_norm(1, 1.0) - 2.917_322_170_855_303).abs() < 1e-12);
        assert!((gto_norm(2, 1.0) - 2.609_332_274_519_885).abs() < 1e-12);
        assert!((gto_norm(3, 2.5) - 15.501_559_129_019_617).abs() < 1e-10);
    }
}
