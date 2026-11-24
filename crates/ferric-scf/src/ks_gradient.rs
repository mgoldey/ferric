//! Analytical nuclear gradient for closed-shell Kohn-Sham DFT.
//!
//! Composition:
//!
//! ```text
//!     ∇E_KS = ∇E_nn + ∇E_1e + ∇E_2e_scaled + ∇E_xc
//! ```
//!
//! Where:
//! - `∇E_nn + ∇E_1e` are identical to HF (nuclear repulsion + dT/dR, dV/dR,
//!   −W·dS/dR Pulay term). Reuses `oneelectron_gradient` from `gradient.rs`.
//! - `∇E_2e_scaled` uses the hybrid two-particle density
//!   `Γ_KS = ½·D·D − (c_K/4)·D·D` where `c_K = k_mix.sr` for plain hybrids
//!   (ω=0) and `c_K = 0` for pure DFT. Range-separated hybrids (ω>0, e.g.
//!   wB97X) split into c_SR·K[erfc(ω)] + c_LR·K[erf(ω)], each contributed via
//!   `twoelectron_k_gradient` with the corresponding operator.
//!   Those four-centre forms apply only when the SCF used exact J/K. When it
//!   density-fitted them (RI-J is the default for every functional; RSH
//!   exchange is always fitted), `ScfResult::df_jk` records the builders and
//!   the 2e term differentiates the FITTED energy instead
//!   (`crate::df_gradient::routed_two_electron_gradient`).
//! - `∇E_xc` is the XC derivative on the moving grid: the AO term plus the
//!   grid response (points ride their home atom, Becke weights move), PySCF's
//!   `grid_response=True` convention. The VV10 nonlocal term adds the pair
//!   distance term of its double sum, so the gradient is the derivative of the
//!   energy the SCF computes.
//!
//! Supports LDA, GGA, plain hybrid-GGA (B3LYP), range-separated hybrid-GGA
//! (wB97X-family), and meta-GGA (SCAN / r2SCAN) — semilocal piece + scaled
//! exact-exchange piece + (when the functional carries VV10, e.g. wB97X-V) the
//! VV10 nonlocal-correlation gradient via
//! `ferric_dft::gradient::vv10_gradient_from_density`.
//!
//! GGA / hybrid-GGA require AO Hessians for the v_σ-coupled term in ∇E_xc.
//! Meta-GGA additionally needs them for the τ term
//! (`ferric_dft::gradient::mgga_tau_partials`), since ∂τ/∂R hits one of τ's two
//! ∂χ factors and turns it into ∂²χ.
//!
//! Meta-GGA status (2026-07-27): closed shell is validated against both finite
//! difference of ferric's own energy and PySCF (≤1.6e-5 / ≤6.6e-6 Ha/Bohr at
//! STO-3G and 6-31G; see `tests/dft_gradient_mgga.rs`). The open-shell path is
//! implemented and wired, but a **pre-existing spin-polarized SCAN SCF energy
//! defect** (~2e-4 Ha on OH/STO-3G, while OH/PBE matches PySCF to 3e-8 and
//! closed-shell SCAN matches to ~1e-8) limits the open-shell gradient to ~5e-4
//! agreement with PySCF. That defect is in the energy path, not here.
//!
//! Structure: every public `ks_gradient_{closed,uks,roks}*` entry point is a
//! thin shim over one spin-generic core — `KsGradInputs::scf_gradient` (the
//! SCF's own densities, optional polarizable term) and
//! `KsGradInputs::gradient_at` (the gradient expression at given densities),
//! with the reference selected by [`Spin`] and the densities carried as a
//! [`TwoElectronDensity`](crate::df_gradient::TwoElectronDensity). The closed and open paths keep their own kernels
//! and summation order, so each entry point's result is bit-identical to the
//! former per-reference implementations (`tests/ks_gradient_bit_identity.rs`).

use crate::df_gradient::TwoElectronDensity;
use crate::gradient::{
    build_energy_weighted_density, build_energy_weighted_density_uhf, oneelectron_gradient,
};
use crate::result::{ScfResult, Spin};
use crate::screening::SchwarzBounds;
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_dft::gradient::{
    xc_gradient_closed_gga_from_density, xc_gradient_closed_lda_from_density,
    xc_gradient_closed_mgga_from_density, xc_gradient_uks_from_density,
    xc_gradient_uks_mgga_from_density,
};
use ferric_dft::grid::AtomicGridConfig;
use ferric_dft::libxc::{xc_def_from_name, FunctionalFamily};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ndarray::Array2;
use std::borrow::Cow;

/// The XC grids a KS gradient is differentiated on.
///
/// The gradient is the derivative of the energy the SCF converged, which is a
/// functional of the SCF's `dft_grid` (and, for VV10 functionals, its
/// `nlc_grid`), including any angular pruning. `Default` is the SCF's own
/// default pair (`AtomicGridConfig::production` main: 75x110 pruned with
/// `nwchem-radial` and a 1e-14 weight cut; flat 50x50 nonlocal), so a caller that
/// ran its SCF with no grid override gets identical bits from either entry
/// point. A caller that set `dft_grid` / `nlc_grid` must pass
/// [`KsGradGrids::from_config`]; the plain `ks_gradient_*` entry points always
/// differentiate the default grids.
#[derive(Debug, Clone)]
pub struct KsGradGrids {
    /// Main XC grid (`RhfConfig::dft_grid`).
    pub main: AtomicGridConfig,
    /// VV10 nonlocal grid (`RhfConfig::nlc_grid`).
    pub nlc: AtomicGridConfig,
}

impl Default for KsGradGrids {
    fn default() -> Self {
        Self {
            main: AtomicGridConfig::production(),
            nlc: AtomicGridConfig::default_nlc(),
        }
    }
}

impl KsGradGrids {
    /// The grids `config`'s SCF used (same defaulting as `solve_rhf` /
    /// `solve_uhf` / `solve_rohf`).
    #[must_use]
    pub fn from_config(config: &crate::rhf::RhfConfig) -> Self {
        let d = Self::default();
        Self {
            main: config.dft_grid.clone().unwrap_or(d.main),
            nlc: config.nlc_grid.clone().unwrap_or(d.nlc),
        }
    }
}

/// Compute the closed-shell KS nuclear gradient.
///
/// Returns a `(natoms, 3)` array of dE/dR in atomic units.
///
/// Families: LDA, GGA, plain hybrid-GGA, range-separated hybrid-GGA, and
/// meta-GGA (SCAN / r2SCAN) all dispatch to the matching
/// `ferric_dft::gradient` driver; the exact-exchange fraction is contributed
/// separately via the `twoelectron_*` helpers below.
///
/// Known error sources (all measured against PySCF, see
/// `tests/dft_gradient_vs_pyscf.rs` and `tests/dft_gradient_mgga.rs`):
/// - Grid integration: ~1e-5 Ha/Bohr at the default (75, 110) grid for s/p
///   bases; ~6e-5 once d shells are present (the same residual for GGA and
///   meta-GGA — it is an AO-Hessian/grid limit, not functional-specific).
/// - VV10 nonlocal correlation IS included when the functional carries it.
#[allow(clippy::too_many_arguments)]
pub fn ks_gradient_closed(
    mol: &Molecule,
    prep: &PreparedBasis,
    bs: &ferric_core::basis::BasisSet,
    op: Operator,
    bounds: &SchwarzBounds,
    xc_name: &str,
    result: &ScfResult,
    ext: Option<&ferric_core::external_potential::ExternalPotential>,
) -> Result<Array2<f64>, FerricError> {
    ks_gradient_closed_with_exchange(mol, prep, bs, op, bounds, xc_name, result, ext, None)
}

/// [`ks_gradient_closed`] with the exact-exchange term taken from the builder
/// the SCF actually used.
///
/// `cosx = None` is exactly [`ks_gradient_closed`] (same code path, so an
/// exact-K gradient is unchanged bit for bit). `cosx = Some(cfg)` is for an SCF
/// run with `k_builder = "cosx"`: the four-centre integrals then supply only
/// Coulomb (`Γ_J = ½·D·D`) and the `−(c_x/4)·tr[D K(D)]` exchange term is
/// differentiated on the COSX grid by
/// [`crate::cosx_gradient::cosx_exchange_gradient`] — the derivative of the
/// energy the SCF computed, not of exact exchange. Refused for range-separated
/// functionals (COSX supports the Coulomb operator only) and for
/// `overlap_fit = true` (the fitted energy's Z-vector term needs the XC Fock
/// derivative; see `cosx_gradient`'s module doc).
#[allow(clippy::too_many_arguments)]
pub fn ks_gradient_closed_with_exchange(
    mol: &Molecule,
    prep: &PreparedBasis,
    bs: &ferric_core::basis::BasisSet,
    op: Operator,
    bounds: &SchwarzBounds,
    xc_name: &str,
    result: &ScfResult,
    ext: Option<&ferric_core::external_potential::ExternalPotential>,
    cosx: Option<&crate::cosx_k::CosxConfig>,
) -> Result<Array2<f64>, FerricError> {
    let grids = KsGradGrids::default();
    KsGradInputs::new(mol, prep, bs, op, bounds, xc_name, result, ext)
        .scf_gradient(Spin::Restricted, &KsGradOpts::exchange(cosx, &grids))
}

/// [`ks_gradient_closed_with_exchange`] on the XC grids the SCF ran on: `grids` is
/// normally [`KsGradGrids::from_config`] of the SCF's config. The plain entry
/// point differentiates the default grids.
#[allow(clippy::too_many_arguments)]
pub fn ks_gradient_closed_with_exchange_on_grids(
    mol: &Molecule,
    prep: &PreparedBasis,
    bs: &ferric_core::basis::BasisSet,
    op: Operator,
    bounds: &SchwarzBounds,
    xc_name: &str,
    result: &ScfResult,
    ext: Option<&ferric_core::external_potential::ExternalPotential>,
    cosx: Option<&crate::cosx_k::CosxConfig>,
    grids: &KsGradGrids,
) -> Result<Array2<f64>, FerricError> {
    KsGradInputs::new(mol, prep, bs, op, bounds, xc_name, result, ext)
        .scf_gradient(Spin::Restricted, &KsGradOpts::exchange(cosx, grids))
}

/// The closed-shell KS gradient EXPRESSION evaluated at an arbitrary density
/// `d` and energy-weighted density `w` (the Pulay term is −Σ w ∂S/∂R):
/// one-electron + ECP + two-electron (by the route `result` recorded: RI-J /
/// fitted K / COSX / exact) + XC (with grid response) + VV10, each a function
/// of `d` alone. With `d = result.density_r()` and `w` from
/// `build_energy_weighted_density` this IS [`ks_gradient_closed_with_exchange`].
///
/// `result` supplies only the two-electron route and the spin check. Linear
/// in `w`; the one-electron, ECP and two-electron parts are linear or
/// bilinear in `d`. The Z-vector relaxation term
/// ([`crate::zvector_ks`]) takes its directional derivative along a
/// density perturbation.
#[allow(clippy::too_many_arguments)]
pub fn ks_gradient_closed_for_density(
    mol: &Molecule,
    prep: &PreparedBasis,
    bs: &ferric_core::basis::BasisSet,
    op: Operator,
    bounds: &SchwarzBounds,
    xc_name: &str,
    result: &ScfResult,
    ext: Option<&ferric_core::external_potential::ExternalPotential>,
    cosx: Option<&crate::cosx_k::CosxConfig>,
    d: &Array2<f64>,
    w: &Array2<f64>,
    grids: &KsGradGrids,
) -> Result<Array2<f64>, FerricError> {
    let density = TwoElectronDensity::Closed(d);
    KsGradInputs::new(mol, prep, bs, op, bounds, xc_name, result, ext).gradient_at(
        Spin::Restricted,
        density,
        w,
        cosx,
        grids,
    )
}

/// [`ks_gradient_closed`] plus the QM-atom-centre Fock-term contribution of
/// Thole-damped polarizable embedding — the closed-shell-KS sibling of
/// [`crate::gradient::rhf_gradient_with_polarizable`]. `sites`/`dipoles`
/// are typically `config.polarizable.as_ref()` and
/// `result.induced_dipoles.as_ref()`; when either is `None`, or
/// `sites.sites` is empty, this is EXACTLY `ks_gradient_closed`
/// (`polarizable_gradient_term` returns an all-zero array in that case,
/// added unconditionally — same pattern as the UHF wrapper). Uses
/// `result.density_total()`, which equals `density_r()` for a
/// `Spin::Restricted` result, matching
/// [`crate::polarizable::qm_gradient_contribution`]'s documented
/// total-density-only dependence (the term is a fixed-mu Hellmann-Feynman
/// contraction of the SAME `V_ind` Fock term added in the closed-shell RHF
/// case, orthogonal to the XC/VV10 pieces this function adds on top of the
/// HF-shaped part of the gradient).
#[allow(clippy::too_many_arguments)]
pub fn ks_gradient_closed_with_polarizable(
    mol: &Molecule,
    prep: &PreparedBasis,
    bs: &ferric_core::basis::BasisSet,
    op: Operator,
    bounds: &SchwarzBounds,
    xc_name: &str,
    result: &ScfResult,
    ext: Option<&ferric_core::external_potential::ExternalPotential>,
    sites: Option<&crate::polarizable::PolarizableSites>,
    dipoles: Option<&Array2<f64>>,
) -> Result<Array2<f64>, FerricError> {
    let grids = KsGradGrids::default();
    ks_gradient_closed_with_polarizable_on_grids(
        mol, prep, bs, op, bounds, xc_name, result, ext, sites, dipoles, &grids,
    )
}

/// [`ks_gradient_closed_with_polarizable`] on the XC grids the SCF ran on.
#[allow(clippy::too_many_arguments)]
pub fn ks_gradient_closed_with_polarizable_on_grids(
    mol: &Molecule,
    prep: &PreparedBasis,
    bs: &ferric_core::basis::BasisSet,
    op: Operator,
    bounds: &SchwarzBounds,
    xc_name: &str,
    result: &ScfResult,
    ext: Option<&ferric_core::external_potential::ExternalPotential>,
    sites: Option<&crate::polarizable::PolarizableSites>,
    dipoles: Option<&Array2<f64>>,
    grids: &KsGradGrids,
) -> Result<Array2<f64>, FerricError> {
    let opts = KsGradOpts {
        polarizable: sites.zip(dipoles),
        ..KsGradOpts::exchange(None, grids)
    };
    KsGradInputs::new(mol, prep, bs, op, bounds, xc_name, result, ext)
        .scf_gradient(Spin::Restricted, &opts)
}

/// Spin-polarized (UKS) Kohn-Sham nuclear gradient.
///
/// Composition mirrors `ks_gradient_closed`:
///
/// ```text
///   ∇E_UKS = ∇E_nn + ∇E_1e + ∇E_2e_scaled_k + ∇E_xc[α,β] + ∇E_vv10[total]
/// ```
///
/// Range-separated hybrids (ω > 0) take their exchange gradient as
/// c_SR·K[erfc(ω)] + c_LR·K[erf(ω)] per spin, the open-shell form of
/// `ks_gradient_closed`'s RSH path.
#[allow(clippy::too_many_arguments)]
pub fn ks_gradient_uks(
    mol: &Molecule,
    prep: &PreparedBasis,
    bs: &ferric_core::basis::BasisSet,
    op: Operator,
    bounds: &SchwarzBounds,
    xc_name: &str,
    result: &ScfResult,
    ext: Option<&ferric_core::external_potential::ExternalPotential>,
) -> Result<Array2<f64>, FerricError> {
    ks_gradient_uks_with_exchange(mol, prep, bs, op, bounds, xc_name, result, ext, None)
}

/// [`ks_gradient_uks`] with the exact-exchange term taken from the builder the
/// SCF actually used: the open-shell sibling of
/// [`ks_gradient_closed_with_exchange`].
///
/// `cosx = None` is exactly [`ks_gradient_uks`] (same code path, so an exact-K
/// gradient is unchanged bit for bit). `cosx = Some(cfg)` is for a UKS SCF run
/// with `k_builder = "cosx"`: the Coulomb term comes from the builder the SCF
/// recorded (four-centre, or RI-J for RIJCOSX) and the exchange
/// `−½·c_x·Σ_σ tr[D_σ K_COSX(D_σ)]` is differentiated on the COSX grid by
/// [`crate::cosx_gradient::cosx_exchange_gradient`], per spin, with the grid
/// riding its atoms and the Becke weight response. Refused for range-separated
/// functionals and for `overlap_fit = true` (see
/// [`crate::cosx_gradient::check_fitted_ks_supported`]).
#[allow(clippy::too_many_arguments)]
pub fn ks_gradient_uks_with_exchange(
    mol: &Molecule,
    prep: &PreparedBasis,
    bs: &ferric_core::basis::BasisSet,
    op: Operator,
    bounds: &SchwarzBounds,
    xc_name: &str,
    result: &ScfResult,
    ext: Option<&ferric_core::external_potential::ExternalPotential>,
    cosx: Option<&crate::cosx_k::CosxConfig>,
) -> Result<Array2<f64>, FerricError> {
    let grids = KsGradGrids::default();
    KsGradInputs::new(mol, prep, bs, op, bounds, xc_name, result, ext)
        .scf_gradient(Spin::Unrestricted, &KsGradOpts::exchange(cosx, &grids))
}

/// [`ks_gradient_uks_with_exchange`] on the XC grids the SCF ran on: `grids` is
/// normally [`KsGradGrids::from_config`] of the SCF's config. The plain entry
/// point differentiates the default grids.
#[allow(clippy::too_many_arguments)]
pub fn ks_gradient_uks_with_exchange_on_grids(
    mol: &Molecule,
    prep: &PreparedBasis,
    bs: &ferric_core::basis::BasisSet,
    op: Operator,
    bounds: &SchwarzBounds,
    xc_name: &str,
    result: &ScfResult,
    ext: Option<&ferric_core::external_potential::ExternalPotential>,
    cosx: Option<&crate::cosx_k::CosxConfig>,
    grids: &KsGradGrids,
) -> Result<Array2<f64>, FerricError> {
    KsGradInputs::new(mol, prep, bs, op, bounds, xc_name, result, ext)
        .scf_gradient(Spin::Unrestricted, &KsGradOpts::exchange(cosx, grids))
}

/// The UKS gradient EXPRESSION evaluated at arbitrary spin densities `d_a`,
/// `d_b` and energy-weighted density `w` (the Pulay term is −Σ w ∂S/∂R): the
/// open-shell sibling of [`ks_gradient_closed_for_density`]. With the SCF's
/// spin densities and `w` from `build_energy_weighted_density_uhf` this IS
/// [`ks_gradient_uks`].
///
/// `result` supplies only the two-electron route and the spin check; `cosx`
/// selects the COSX exchange derivative exactly as in
/// [`ks_gradient_uks_with_exchange`]. Linear in `w`; the one-electron, ECP and
/// two-electron parts (COSX exchange included: `tr[D_σ K_COSX(D_σ)]` is a
/// quadratic form) are linear or bilinear in (`d_a`, `d_b`). The unrestricted
/// Z-vector relaxation term
/// ([`crate::zvector_ks::relaxation_gradient_unrestricted`]) takes its
/// directional derivative along a spin-density perturbation.
#[allow(clippy::too_many_arguments)]
pub fn ks_gradient_uks_for_density(
    mol: &Molecule,
    prep: &PreparedBasis,
    bs: &ferric_core::basis::BasisSet,
    op: Operator,
    bounds: &SchwarzBounds,
    xc_name: &str,
    result: &ScfResult,
    ext: Option<&ferric_core::external_potential::ExternalPotential>,
    cosx: Option<&crate::cosx_k::CosxConfig>,
    d_a: &Array2<f64>,
    d_b: &Array2<f64>,
    w: &Array2<f64>,
    grids: &KsGradGrids,
) -> Result<Array2<f64>, FerricError> {
    let density = TwoElectronDensity::Open {
        alpha: d_a,
        beta: d_b,
    };
    KsGradInputs::new(mol, prep, bs, op, bounds, xc_name, result, ext).gradient_at(
        Spin::Unrestricted,
        density,
        w,
        cosx,
        grids,
    )
}

/// [`ks_gradient_uks`] plus the QM-atom-centre Fock-term contribution of
/// Thole-damped polarizable embedding — the UKS sibling of
/// [`ks_gradient_closed_with_polarizable`] /
/// [`crate::gradient::uhf_gradient_with_polarizable`]. Same None/empty ⇒
/// bit-identical-to-plain-`ks_gradient_uks` contract, via
/// `result.density_total()` (`= density_alpha + density_beta` for UKS).
#[allow(clippy::too_many_arguments)]
pub fn ks_gradient_uks_with_polarizable(
    mol: &Molecule,
    prep: &PreparedBasis,
    bs: &ferric_core::basis::BasisSet,
    op: Operator,
    bounds: &SchwarzBounds,
    xc_name: &str,
    result: &ScfResult,
    ext: Option<&ferric_core::external_potential::ExternalPotential>,
    sites: Option<&crate::polarizable::PolarizableSites>,
    dipoles: Option<&Array2<f64>>,
) -> Result<Array2<f64>, FerricError> {
    let grids = KsGradGrids::default();
    ks_gradient_uks_with_polarizable_on_grids(
        mol, prep, bs, op, bounds, xc_name, result, ext, sites, dipoles, &grids,
    )
}

/// [`ks_gradient_uks_with_polarizable`] on the XC grids the SCF ran on.
#[allow(clippy::too_many_arguments)]
pub fn ks_gradient_uks_with_polarizable_on_grids(
    mol: &Molecule,
    prep: &PreparedBasis,
    bs: &ferric_core::basis::BasisSet,
    op: Operator,
    bounds: &SchwarzBounds,
    xc_name: &str,
    result: &ScfResult,
    ext: Option<&ferric_core::external_potential::ExternalPotential>,
    sites: Option<&crate::polarizable::PolarizableSites>,
    dipoles: Option<&Array2<f64>>,
    grids: &KsGradGrids,
) -> Result<Array2<f64>, FerricError> {
    let opts = KsGradOpts {
        polarizable: sites.zip(dipoles),
        ..KsGradOpts::exchange(None, grids)
    };
    KsGradInputs::new(mol, prep, bs, op, bounds, xc_name, result, ext)
        .scf_gradient(Spin::Unrestricted, &opts)
}

/// Restricted-open-shell KS (ROKS) nuclear gradient.
///
/// Structure mirrors `ks_gradient_uks` but uses the ROHF energy-weighted
/// density (`crate::gradient::rohf_energy_weighted_density`, built from the
/// spin Focks, which include `V_xc^σ`) and the same UKS XC gradient via
/// `xc_gradient_uks_from_density`.
/// The per-spin densities from a ROKS `ScfResult` already satisfy the
/// projector structure so the UKS XC path applies verbatim.
#[allow(clippy::too_many_arguments)]
pub fn ks_gradient_roks(
    mol: &Molecule,
    prep: &PreparedBasis,
    bs: &ferric_core::basis::BasisSet,
    op: Operator,
    bounds: &SchwarzBounds,
    xc_name: &str,
    result: &ScfResult,
    ext: Option<&ferric_core::external_potential::ExternalPotential>,
) -> Result<Array2<f64>, FerricError> {
    ks_gradient_roks_with_exchange(mol, prep, bs, op, bounds, xc_name, result, ext, None)
}

/// [`ks_gradient_roks`] with the exact-exchange term taken from the builder
/// the SCF actually used. `cosx = None` is exactly [`ks_gradient_roks`];
/// `cosx = Some(cfg)` (an SCF with `k_builder = "cosx"`) differentiates
/// `−½·c_x·Σ_σ tr[D_σ K_COSX(D_σ)]` on the COSX grid, as
/// [`ks_gradient_uks_with_exchange`] does. Nothing else changes: the ROKS
/// energy is the UKS functional at the ROKS spin densities, the fit-off COSX
/// exchange is a quadratic form whose `D_σ`-derivative is the `K_COSX(D_σ)` the
/// SCF put in its spin Focks, and the energy-weighted density is built from
/// those Focks (`ScfResult::rohf_spin_focks`). Refused for range-separated
/// functionals and for `overlap_fit = true`.
#[allow(clippy::too_many_arguments)]
pub fn ks_gradient_roks_with_exchange(
    mol: &Molecule,
    prep: &PreparedBasis,
    bs: &ferric_core::basis::BasisSet,
    op: Operator,
    bounds: &SchwarzBounds,
    xc_name: &str,
    result: &ScfResult,
    ext: Option<&ferric_core::external_potential::ExternalPotential>,
    cosx: Option<&crate::cosx_k::CosxConfig>,
) -> Result<Array2<f64>, FerricError> {
    let grids = KsGradGrids::default();
    KsGradInputs::new(mol, prep, bs, op, bounds, xc_name, result, ext)
        .scf_gradient(Spin::RestrictedOpen, &KsGradOpts::exchange(cosx, &grids))
}

/// [`ks_gradient_roks_with_exchange`] on the XC grids the SCF ran on: `grids` is
/// normally [`KsGradGrids::from_config`] of the SCF's config. The plain entry
/// point differentiates the default grids.
#[allow(clippy::too_many_arguments)]
pub fn ks_gradient_roks_with_exchange_on_grids(
    mol: &Molecule,
    prep: &PreparedBasis,
    bs: &ferric_core::basis::BasisSet,
    op: Operator,
    bounds: &SchwarzBounds,
    xc_name: &str,
    result: &ScfResult,
    ext: Option<&ferric_core::external_potential::ExternalPotential>,
    cosx: Option<&crate::cosx_k::CosxConfig>,
    grids: &KsGradGrids,
) -> Result<Array2<f64>, FerricError> {
    KsGradInputs::new(mol, prep, bs, op, bounds, xc_name, result, ext)
        .scf_gradient(Spin::RestrictedOpen, &KsGradOpts::exchange(cosx, grids))
}

/// The ROKS gradient EXPRESSION evaluated at arbitrary spin densities `d_a`,
/// `d_b` and energy-weighted density `w` (the Pulay term is −Σ w ∂S/∂R): the
/// restricted-open-shell sibling of [`ks_gradient_uks_for_density`]. With the
/// SCF's spin densities and `w` from the ROHF energy-weighted density this IS
/// [`ks_gradient_roks`].
///
/// `result` supplies only the two-electron route and the spin check; `cosx`
/// selects the COSX exchange derivative as in
/// [`ks_gradient_roks_with_exchange`]. Linear in `w`; the one-electron, ECP
/// and two-electron parts are linear or bilinear in (`d_a`, `d_b`). The ROKS
/// Z-vector relaxation term ([`crate::zvector_ks::relaxation_gradient_roks`])
/// takes its directional derivative along a shared-orbital rotation.
#[allow(clippy::too_many_arguments)]
pub fn ks_gradient_roks_for_density(
    mol: &Molecule,
    prep: &PreparedBasis,
    bs: &ferric_core::basis::BasisSet,
    op: Operator,
    bounds: &SchwarzBounds,
    xc_name: &str,
    result: &ScfResult,
    ext: Option<&ferric_core::external_potential::ExternalPotential>,
    cosx: Option<&crate::cosx_k::CosxConfig>,
    d_a: &Array2<f64>,
    d_b: &Array2<f64>,
    w: &Array2<f64>,
    grids: &KsGradGrids,
) -> Result<Array2<f64>, FerricError> {
    let density = TwoElectronDensity::Open {
        alpha: d_a,
        beta: d_b,
    };
    KsGradInputs::new(mol, prep, bs, op, bounds, xc_name, result, ext).gradient_at(
        Spin::RestrictedOpen,
        density,
        w,
        cosx,
        grids,
    )
}

/// 2e K-only gradient with a chosen Coulomb-like operator (Coulomb, erfc, erf).
///
/// Used by the RSH gradient path: builds `Σ Γ_K · d(μν|λσ)/dR` with
/// `Γ_K = -(c_K/4) · D·D` (anti-symmetrized exchange piece). Caller is expected
/// to add this on top of a separate J-only gradient (from
/// `twoelectron_gradient_scaled_k` with `c_K=0`, Coulomb operator).
pub fn twoelectron_k_gradient(
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    d: &Array2<f64>,
    c_k: f64,
) -> Result<Array2<f64>, FerricError> {
    if let Some(r) = crate::gradient::gradient_rotation(prep, op, bounds)? {
        return twoelectron_k_gradient(&r.basis.prep, op, &r.bounds, &r.forward(d), c_k);
    }
    let max_d = d.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    crate::gradient::par_twoelectron_gradient(prep, op, bounds, max_d, |mu, nu, la, sg| {
        -0.25 * c_k * d[(mu, la)] * d[(nu, sg)]
    })
}

/// 2e gradient with hybrid K scaling. Mirrors `twoelectron_gradient` from
/// `gradient.rs` but uses `Γ_KS = 0.5·D·D − (c_K/4)·D·D` instead of HF's
/// `0.5·D·D − 0.25·D·D`.
pub fn twoelectron_gradient_scaled_k(
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    d: &Array2<f64>,
    c_k: f64,
) -> Result<Array2<f64>, FerricError> {
    if let Some(r) = crate::gradient::gradient_rotation(prep, op, bounds)? {
        return twoelectron_gradient_scaled_k(&r.basis.prep, op, &r.bounds, &r.forward(d), c_k);
    }
    let max_d = d.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    crate::gradient::par_twoelectron_gradient(prep, op, bounds, max_d, |mu, nu, la, sg| {
        0.5 * d[(mu, nu)] * d[(la, sg)] - 0.25 * c_k * d[(mu, la)] * d[(nu, sg)]
    })
}

/// Scaled-K UHF 2e gradient: same as `twoelectron_gradient_uhf` but the
/// exchange piece is multiplied by `c_K`. Used by UKS where hybrids include
/// a fraction of exact exchange.
///   Γ_UKS = 0.5·D·D − 0.5·c_K·(D_α·D_α + D_β·D_β)
pub fn twoelectron_gradient_uhf_scaled_k(
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    d_total: &Array2<f64>,
    d_alpha: &Array2<f64>,
    d_beta: &Array2<f64>,
    c_k: f64,
) -> Result<Array2<f64>, FerricError> {
    if let Some(r) = crate::gradient::gradient_rotation(prep, op, bounds)? {
        return twoelectron_gradient_uhf_scaled_k(
            &r.basis.prep,
            op,
            &r.bounds,
            &r.forward(d_total),
            &r.forward(d_alpha),
            &r.forward(d_beta),
            c_k,
        );
    }
    let max_d = d_total.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    crate::gradient::par_twoelectron_gradient(prep, op, bounds, max_d, |mu, nu, la, sg| {
        0.5 * d_total[(mu, nu)] * d_total[(la, sg)]
            - 0.5
                * c_k
                * (d_alpha[(mu, la)] * d_alpha[(nu, sg)] + d_beta[(mu, la)] * d_beta[(nu, sg)])
    })
}

/// UHF K-only gradient with a chosen operator: Σ Γ_K · d(μν|λσ)/dR with
/// `Γ_K = -0.5 · c_K · (D_α·D_α + D_β·D_β)`. Used by RSH UKS gradients to
/// build the c_SR·K[erfc(ω)] and c_LR·K[erf(ω)] pieces separately.
pub fn twoelectron_k_gradient_uhf(
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    d_alpha: &Array2<f64>,
    d_beta: &Array2<f64>,
    c_k: f64,
) -> Result<Array2<f64>, FerricError> {
    if let Some(r) = crate::gradient::gradient_rotation(prep, op, bounds)? {
        return twoelectron_k_gradient_uhf(
            &r.basis.prep,
            op,
            &r.bounds,
            &r.forward(d_alpha),
            &r.forward(d_beta),
            c_k,
        );
    }
    let d_total: Array2<f64> = d_alpha + d_beta;
    let max_d = d_total.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    crate::gradient::par_twoelectron_gradient(prep, op, bounds, max_d, |mu, nu, la, sg| {
        -0.5 * c_k * (d_alpha[(mu, la)] * d_alpha[(nu, sg)] + d_beta[(mu, la)] * d_beta[(nu, sg)])
    })
}

/// The arguments every KS gradient entry point shares: the system, its
/// integrals context, the functional and the converged SCF.
#[derive(Clone, Copy)]
struct KsGradInputs<'a> {
    mol: &'a Molecule,
    prep: &'a PreparedBasis,
    bs: &'a ferric_core::basis::BasisSet,
    op: Operator,
    bounds: &'a SchwarzBounds,
    xc_name: &'a str,
    result: &'a ScfResult,
    ext: Option<&'a ferric_core::external_potential::ExternalPotential>,
}

/// The optional pieces of a KS gradient taken from the SCF's own setup: the
/// COSX exchange builder, the XC grids, and the polarizable-embedding sites
/// with their induced dipoles.
struct KsGradOpts<'a> {
    cosx: Option<&'a crate::cosx_k::CosxConfig>,
    grids: &'a KsGradGrids,
    polarizable: Option<(&'a crate::polarizable::PolarizableSites, &'a Array2<f64>)>,
}

impl<'a> KsGradOpts<'a> {
    fn exchange(cosx: Option<&'a crate::cosx_k::CosxConfig>, grids: &'a KsGradGrids) -> Self {
        Self {
            cosx,
            grids,
            polarizable: None,
        }
    }
}

/// The public entry point named in panics for a `spin` reference.
fn entry_name(spin: Spin) -> &'static str {
    match spin {
        Spin::Restricted => "ks_gradient_closed",
        Spin::Unrestricted => "ks_gradient_uks",
        Spin::RestrictedOpen => "ks_gradient_roks",
    }
}

fn assert_spin(result: &ScfResult, spin: Spin) {
    let name = entry_name(spin);
    assert!(
        result.spin == spin,
        "{name}: ScfResult.spin must be {spin:?}"
    );
}

/// The total density `D` (`D_α + D_β` for an open shell).
fn total_density(density: TwoElectronDensity<'_>) -> Cow<'_, Array2<f64>> {
    match density {
        TwoElectronDensity::Closed(d) => Cow::Borrowed(d),
        TwoElectronDensity::Open { alpha, beta } => Cow::Owned(alpha + beta),
    }
}

impl<'a> KsGradInputs<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        mol: &'a Molecule,
        prep: &'a PreparedBasis,
        bs: &'a ferric_core::basis::BasisSet,
        op: Operator,
        bounds: &'a SchwarzBounds,
        xc_name: &'a str,
        result: &'a ScfResult,
        ext: Option<&'a ferric_core::external_potential::ExternalPotential>,
    ) -> Self {
        Self {
            mol,
            prep,
            bs,
            op,
            bounds,
            xc_name,
            result,
            ext,
        }
    }

    /// The `spin`-reference KS gradient at the SCF's own densities and
    /// energy-weighted density, plus the polarizable-embedding term when
    /// `opts` carries one.
    fn scf_gradient(&self, spin: Spin, opts: &KsGradOpts) -> Result<Array2<f64>, FerricError> {
        let (mol, prep, result) = (self.mol, self.prep, self.result);
        crate::x2c::refuse_relativistic(result.relativistic, "analytic nuclear gradient")?;
        let (density, w) = if spin == Spin::Restricted {
            let w = build_energy_weighted_density(result, (mol.nelec() / 2) as usize);
            (TwoElectronDensity::Closed(result.density_r()), w)
        } else {
            assert_spin(result, spin);
            let alpha = &result.density_alpha;
            let beta = result.density_beta.as_ref();
            let beta = beta.unwrap_or_else(|| panic!("{}: missing density_beta", entry_name(spin)));
            let w = if spin == Spin::Unrestricted {
                let nelec = mol.nelec() as i64;
                let two_s = mol.multiplicity as i64 - 1;
                let nocc_a = ((nelec + two_s) / 2) as usize;
                let nocc_b = ((nelec - two_s) / 2) as usize;
                build_energy_weighted_density_uhf(result, prep, nocc_a, nocc_b)?
            } else {
                crate::gradient::rohf_energy_weighted_density(result)?
            };
            (TwoElectronDensity::Open { alpha, beta }, w)
        };
        let mut grad = self.gradient_at(spin, density, &w, opts.cosx, opts.grids)?;
        if let Some((sites, dipoles)) = opts.polarizable {
            let d_total = result.density_total();
            grad +=
                &crate::polarizable::polarizable_gradient_term(mol, prep, sites, dipoles, d_total)?;
        }
        Ok(grad)
    }

    /// The KS gradient EXPRESSION of the `spin` reference at `density` and
    /// energy-weighted density `w`: one-electron + ECP + two-electron (by the
    /// route `result` recorded) + XC + VV10.
    fn gradient_at(
        &self,
        spin: Spin,
        density: TwoElectronDensity,
        w: &Array2<f64>,
        cosx: Option<&crate::cosx_k::CosxConfig>,
        grids: &KsGradGrids,
    ) -> Result<Array2<f64>, FerricError> {
        let Self {
            mol,
            prep,
            bs,
            xc_name,
            result,
            ext,
            ..
        } = *self;
        crate::x2c::refuse_relativistic(result.relativistic, "analytic nuclear gradient")?;
        if spin == Spin::Restricted {
            if let Some(cfg) = cosx {
                crate::cosx_gradient::check_gradient_supported(cfg)?;
                // Fitted COSX + KS needs the XC Fock derivative in its
                // Z-vector term, which ferric does not have: refused, never an
                // inexact gradient.
                crate::cosx_gradient::check_fitted_ks_supported(cfg, Some(xc_name))?;
            }
            refuse_ghosts(mol, "ks_gradient_closed")?;
        }
        assert_spin(result, spin);

        let xc =
            xc_def_from_name(xc_name).map_err(|e| FerricError::General(format!("libxc: {e:?}")))?;
        let k_mix = ferric_dft::libxc::k_mix_from_xc_def(&xc);
        let d_total = total_density(density);

        // 1e + nuclear repulsion gradient — identical to HF — and the ECP
        // term Σ D dV_ECP/dR (zero for an all-electron basis): V_ECP is in the
        // KS hcore (driver::prepare), so its derivative belongs here too.
        let t = ferric_core::timing::Stage::start("ks-grad:1e+ecp");
        let mut grad = oneelectron_gradient(mol, prep, &d_total, w, ext)?;
        grad += &crate::gradient::ecp_gradient(mol, prep, &d_total)?;
        t.end();

        let t = ferric_core::timing::Stage::start("ks-grad:2e");
        self.add_two_electron_gradient(&mut grad, density, &d_total, &k_mix, cosx)?;
        t.end();

        let t = ferric_core::timing::Stage::start("ks-grad:xc");
        add_xc_gradient(&mut grad, mol, prep, bs, xc_name, density, grids)?;
        t.end();

        Ok(grad)
    }

    /// `grad += ∇E_2e`. When the SCF density-fitted J and/or K (RI-J is the
    /// default for every functional; RSH exchange is ALWAYS fitted), the
    /// recorded route differentiates that fitted energy — see
    /// `crate::df_gradient`. With COSX exchange, J comes from the SCF's
    /// Coulomb builder and K from the COSX grid. Otherwise (all-exact SCF):
    ///   * ω = 0: Γ = ½·D·D minus the exchange scaled by c_K = k_mix.sr
    ///     (closed `(c_K/4)·D·D`, open `½·c_K·(D_α·D_α + D_β·D_β)`)
    ///   * ω > 0 (RSH): J (Coulomb, c_K = 0) + c_SR·K[erfc(ω)] + c_LR·K[erf(ω)]
    fn add_two_electron_gradient(
        &self,
        grad: &mut Array2<f64>,
        density: TwoElectronDensity,
        d_total: &Array2<f64>,
        k_mix: &ferric_dft::xc_trait::KMix,
        cosx: Option<&crate::cosx_k::CosxConfig>,
    ) -> Result<(), FerricError> {
        let Self {
            mol,
            prep,
            op,
            bounds,
            result,
            ..
        } = *self;
        if let Some(cfg) = cosx {
            return match density {
                TwoElectronDensity::Closed(d) => {
                    self.add_closed_cosx_two_electron_gradient(grad, d, k_mix, cfg)
                }
                TwoElectronDensity::Open { alpha, beta } => {
                    *grad +=
                        &self.open_shell_cosx_two_electron_gradient(k_mix, cfg, alpha, beta)?;
                    Ok(())
                }
            };
        }
        if let Some(route) = crate::gradient::active_df_route(result) {
            let mix = crate::df_gradient::ExchangeMix::from_k_mix(k_mix);
            *grad += &crate::df_gradient::routed_two_electron_gradient(
                mol, prep, op, bounds, density, route, mix,
            )?;
        } else if k_mix.omega > 0.0 {
            let (erfc, erf) = (Operator::erfc(k_mix.omega), Operator::erf(k_mix.omega));
            *grad += &scaled_k_gradient(prep, op, bounds, density, d_total, 0.0)?;
            *grad += &k_only_gradient(prep, erfc, bounds, density, k_mix.sr)?;
            *grad += &k_only_gradient(prep, erf, bounds, density, k_mix.lr)?;
        } else {
            *grad += &scaled_k_gradient(prep, op, bounds, density, d_total, k_mix.sr)?;
        }
        Ok(())
    }

    /// Closed-shell `grad += ∇E_2e` of an SCF whose exchange came from COSX:
    /// J from the Coulomb builder the SCF used — RI-J when the route recorded
    /// one (the RIJCOSX composite), else the four-centre integrals
    /// (Γ_J = ½·D·D, c_K = 0), never the routed J+K path, which would
    /// differentiate EXACT exchange next to it — then
    /// E_x = −(c_x/4)·tr[D K_COSX(D)] differentiated on the COSX grid.
    fn add_closed_cosx_two_electron_gradient(
        &self,
        grad: &mut Array2<f64>,
        d: &Array2<f64>,
        k_mix: &ferric_dft::xc_trait::KMix,
        cfg: &crate::cosx_k::CosxConfig,
    ) -> Result<(), FerricError> {
        let (mol, prep) = (self.mol, self.prep);
        refuse_cosx_range_separated(self.xc_name, k_mix)?;
        match crate::gradient::active_df_route(self.result) {
            Some(route) => {
                if route.k_aux.is_some() || route.rsh_k.is_some() {
                    return Err(FerricError::General(
                        "COSX gradient: the SCF recorded a density-fitted exchange route next to \
                         COSX exchange; RIJCOSX fits Coulomb only"
                            .into(),
                    ));
                }
                *grad += &crate::df_gradient::df_j_gradient(mol, prep, route, d)?;
            }
            None => *grad += &twoelectron_gradient_scaled_k(prep, self.op, self.bounds, d, 0.0)?,
        }
        if k_mix.sr != 0.0 {
            let terms = [(d, -0.25 * k_mix.sr)];
            *grad += &crate::cosx_gradient::cosx_exchange_gradient(mol, prep, cfg, &terms)?;
        }
        Ok(())
    }

    /// Open-shell (UKS / ROKS) two-electron gradient of an SCF whose exchange
    /// came from COSX: the Coulomb term `½ J'(D, D)` (`D = D_α + D_β`) from the
    /// builder the SCF used (four-centre, or the RI-J route of RIJCOSX) and the
    /// exchange `−½·c_x·Σ_σ tr[D_σ K_COSX(D_σ)]` differentiated on the COSX grid
    /// (one density term per spin, `[(D_α, −c_x/2), (D_β, −c_x/2)]`).
    ///
    /// Refused, never approximated: a configuration the COSX gradient cannot
    /// differentiate (`check_gradient_supported`), the overlap fit with a
    /// functional (its Z-vector needs the XC Fock nuclear derivative), and a
    /// range-separated functional (COSX has no erf/erfc form).
    fn open_shell_cosx_two_electron_gradient(
        &self,
        k_mix: &ferric_dft::xc_trait::KMix,
        cfg: &crate::cosx_k::CosxConfig,
        d_a: &Array2<f64>,
        d_b: &Array2<f64>,
    ) -> Result<Array2<f64>, FerricError> {
        let (mol, prep, xc_name) = (self.mol, self.prep, self.xc_name);
        crate::cosx_gradient::check_gradient_supported(cfg)?;
        crate::cosx_gradient::check_fitted_ks_supported(cfg, Some(xc_name))?;
        refuse_cosx_range_separated(xc_name, k_mix)?;
        let d_total = d_a + d_b;
        let (op, bounds, result) = (self.op, self.bounds, self.result);
        let mut grad = crate::gradient::cosx_run_coulomb_gradient(
            mol, prep, op, bounds, result, &d_total, None,
        )?;
        if k_mix.sr != 0.0 {
            let c = -0.5 * k_mix.sr;
            grad += &crate::cosx_gradient::cosx_exchange_gradient(
                mol,
                prep,
                cfg,
                &[(d_a, c), (d_b, c)],
            )?;
        }
        Ok(grad)
    }
}

/// COSX supports the Coulomb operator only: no erf/erfc exchange form.
fn refuse_cosx_range_separated(
    xc_name: &str,
    k_mix: &ferric_dft::xc_trait::KMix,
) -> Result<(), FerricError> {
    if k_mix.omega > 0.0 {
        return Err(FerricError::General(format!(
            "COSX exchange gradient: range-separated functional '{xc_name}' (omega = {}) has no \
             COSX form (k_builder = \"cosx\" supports the Coulomb operator only)",
            k_mix.omega
        )));
    }
    Ok(())
}

/// The J + scaled-K four-centre gradient of `density`; `d_total` is its
/// total density.
fn scaled_k_gradient(
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    density: TwoElectronDensity,
    d_total: &Array2<f64>,
    c_k: f64,
) -> Result<Array2<f64>, FerricError> {
    match density {
        TwoElectronDensity::Closed(d) => twoelectron_gradient_scaled_k(prep, op, bounds, d, c_k),
        TwoElectronDensity::Open { alpha, beta } => {
            twoelectron_gradient_uhf_scaled_k(prep, op, bounds, d_total, alpha, beta, c_k)
        }
    }
}

/// The K-only four-centre gradient of `density` with operator `op`.
fn k_only_gradient(
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    density: TwoElectronDensity,
    c_k: f64,
) -> Result<Array2<f64>, FerricError> {
    match density {
        TwoElectronDensity::Closed(d) => twoelectron_k_gradient(prep, op, bounds, d, c_k),
        TwoElectronDensity::Open { alpha, beta } => {
            twoelectron_k_gradient_uhf(prep, op, bounds, alpha, beta, c_k)
        }
    }
}

/// Family flags of an XC definition: `(needs_gga, needs_mgga)`.
fn xc_families(xc: &ferric_dft::libxc::XcDef) -> (bool, bool) {
    let mut needs_gga = false;
    let mut needs_mgga = false;
    for f in &xc.funcs {
        match f.family() {
            FunctionalFamily::Lda => {}
            FunctionalFamily::Gga | FunctionalFamily::HybridGga | FunctionalFamily::RangeSepGga => {
                needs_gga = true
            }
            // Meta-GGA: the τ-dependent AO-derivative + grid-response terms are
            // handled by the `*_mgga_from_density` drivers.
            FunctionalFamily::MetaGga => {
                needs_gga = true;
                needs_mgga = true;
            }
        }
    }
    (needs_gga, needs_mgga)
}

/// The XC part of the closed-shell KS gradient at fixed total density `d`:
/// the semilocal term (with grid response) plus VV10, exactly as
/// [`ks_gradient_closed_for_density`] adds it. Exact exchange is not
/// included (it is a two-electron term). Refuses ghost atoms, as the full
/// closed-shell gradient does.
pub fn xc_gradient_closed(
    mol: &Molecule,
    prep: &PreparedBasis,
    bs: &ferric_core::basis::BasisSet,
    xc_name: &str,
    d: &Array2<f64>,
    grids: &KsGradGrids,
) -> Result<Array2<f64>, FerricError> {
    xc_gradient(mol, prep, bs, xc_name, TwoElectronDensity::Closed(d), grids)
}

/// Open-shell twin of [`xc_gradient_closed`] at fixed `(d_a, d_b)`, as the
/// UKS and ROKS gradients add it (VV10 from the total density).
pub fn xc_gradient_open(
    mol: &Molecule,
    prep: &PreparedBasis,
    bs: &ferric_core::basis::BasisSet,
    xc_name: &str,
    (d_a, d_b): (&Array2<f64>, &Array2<f64>),
    grids: &KsGradGrids,
) -> Result<Array2<f64>, FerricError> {
    let density = TwoElectronDensity::Open {
        alpha: d_a,
        beta: d_b,
    };
    xc_gradient(mol, prep, bs, xc_name, density, grids)
}

fn xc_gradient(
    mol: &Molecule,
    prep: &PreparedBasis,
    bs: &ferric_core::basis::BasisSet,
    xc_name: &str,
    density: TwoElectronDensity,
    grids: &KsGradGrids,
) -> Result<Array2<f64>, FerricError> {
    refuse_ghosts(mol, "the XC gradient")?;
    let mut grad = Array2::zeros((mol.atoms.len(), 3));
    add_xc_gradient(&mut grad, mol, prep, bs, xc_name, density, grids)?;
    Ok(grad)
}

fn refuse_ghosts(mol: &Molecule, what: &str) -> Result<(), FerricError> {
    if mol.atoms.iter().any(|a| a.ghost) {
        return Err(FerricError::General(format!(
            "{what} is not implemented for molecules containing ghost atoms"
        )));
    }
    Ok(())
}

/// `grad += XC; grad += VV10`, the two separate additions the full KS
/// gradient always made, so its result is bit-identical. Closed shell
/// dispatches on the functional family (LDA fast path, GGA needs Hessians);
/// open shell (UKS and ROKS) uses the polarized drivers. VV10 is a function
/// of the total density, identical in both cases.
fn add_xc_gradient(
    grad: &mut Array2<f64>,
    mol: &Molecule,
    prep: &PreparedBasis,
    bs: &ferric_core::basis::BasisSet,
    xc_name: &str,
    density: TwoElectronDensity,
    grids: &KsGradGrids,
) -> Result<(), FerricError> {
    let xc =
        xc_def_from_name(xc_name).map_err(|e| FerricError::General(format!("libxc: {e:?}")))?;
    let (needs_gga, needs_mgga) = xc_families(&xc);
    let (grid_cfg, s2a, offs, dims) = (
        &grids.main,
        prep.shell_to_atom(),
        prep.shell_offsets(),
        prep.shell_dims(),
    );
    let xc_grad = match density {
        TwoElectronDensity::Closed(d) => if needs_mgga {
            xc_gradient_closed_mgga_from_density(mol, bs, d, xc_name, grid_cfg, s2a, offs, dims)
        } else if needs_gga {
            xc_gradient_closed_gga_from_density(mol, bs, d, xc_name, grid_cfg, s2a, offs, dims)
        } else {
            xc_gradient_closed_lda_from_density(mol, bs, d, xc_name, grid_cfg, s2a, offs, dims)
        }
        .map_err(|e| FerricError::General(format!("xc gradient: {e:?}")))?,
        TwoElectronDensity::Open { alpha, beta } => if needs_mgga {
            xc_gradient_uks_mgga_from_density(
                mol, bs, alpha, beta, xc_name, grid_cfg, s2a, offs, dims,
            )
        } else {
            xc_gradient_uks_from_density(mol, bs, alpha, beta, xc_name, grid_cfg, s2a, offs, dims)
        }
        .map_err(|e| FerricError::General(format!("uks xc gradient: {e:?}")))?,
    };
    *grad += &xc_grad;

    // VV10 nonlocal-correlation gradient (only if the functional advertises it).
    if let Some(vv10_params) = xc.vv10 {
        let vv10_grad = ferric_dft::gradient::vv10_gradient_from_density(
            mol,
            bs,
            &total_density(density),
            &vv10_params,
            &grids.nlc,
            s2a,
            offs,
            dims,
        )
        .map_err(|e| FerricError::General(format!("vv10 gradient: {e:?}")))?;
        *grad += &vv10_grad;
    }

    Ok(())
}
