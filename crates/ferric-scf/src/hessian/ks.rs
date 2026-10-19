//! Opt-in analytic Hessian of closed-shell Kohn-Sham LDA/GGA, pure or global
//! hybrid, with exact four-centre J/K or RI-J (density-fitted Coulomb, exact K)
//! on a grid FIXED IN SPACE (issue #407).
//!
//! The energy differentiated is
//!
//! ```text
//!   E = V_nn + Tr[D h] + ½ Tr[D J(D)] − (a/4) Tr[D K(D)] + Σ_g w_g f(ρ(r_g), σ(r_g)),
//!   D = 2 C_occ C_occᵀ,
//! ```
//!
//! with `a` the exact-exchange fraction (0 for a pure functional) and the grid
//! points and weights frozen at the reference geometry (PySCF
//! `grid_response=False`; see [`crate::ks_fixed_grid`]). It is NOT the
//! derivative of the energy the production KS SCF minimizes on an atom-following
//! grid: that gradient carries the Becke weight response, which this Hessian
//! omits, so the two differ by the grid's translation error and weight
//! derivatives (measured in `tests/ks_fixed_grid_hessian.rs`). That is why the
//! entry points here are explicit and [`HessianMethod::Auto`] and
//! [`HessianMethod::Analytic`] never reach them.
//!
//! The Hessian is the RHF one with `G(D) = J(D) − ½K(D)` replaced by
//! `G(D) = J(D) − (a/2)K(D) + V1_xc[D]`:
//!
//! * skeleton: the RHF terms with `Γ = ½ D D − (a/4) D D`, plus the XC second
//!   derivative at fixed `D` ([`FixedGridKs::skeleton_hessian`]);
//! * `F^x` gains `∂V_xc/∂x|_D` ([`FixedGridKs::first_order_fock`]) and the
//!   exchange of `F^x` is scaled by `a`;
//! * the CPHF `G` (right-hand side, matrix-vector product and the first-order
//!   orbital energies) gains the XC kernel response `V1_xc[D1]`
//!   ([`crate::ks_fixed_grid::XcResponse`]; for a GGA it includes the `σ`
//!   couplings) and carries the exchange scaled by `a`.
//!
//! # RI-J
//!
//! When the SCF's Coulomb matrix is the fitted one (`df_j_aux` named, or a
//! recorded RI-J route), the energy differentiated has `½ Tr[D J_DF(D)]` in
//! place of the exact Coulomb term and everything else is unchanged: the
//! four-centre `Γ_J` skeleton is replaced by the RI-J skeleton of
//! [`crate::hessian_df::rij_terms`] (3c, 2c-metric and fit-response pieces),
//! `J^x` of `F^x` by the fitted one (including the auxiliary-fit part), and the
//! CPHF `G` takes the SCF's own `DfJ` for its Coulomb part. The XC, exact-K
//! and kernel pieces are the exact-J ones, untouched.
//!
//! # RI-K (global hybrids)
//!
//! When the SCF's exchange matrix is the fitted one (`df_k_aux` named or
//! auto-defaulted for a hybrid, or a recorded RI-K route), the exchange energy is
//! `−(a/4) Tr[D K_DF(D)]`, i.e. the one closed-shell channel of
//! [`crate::hessian_df_k`] with weight `α = −a/4` (the Hartree-Fock `−¼`
//! scaled by `a`) and `X = √2 C_occ`. Every RI-K piece is linear in `α` at fixed
//! `X`, so the HF RI-K terms are reused unchanged with that weight: the
//! four-centre `Γ_K` skeleton is replaced by the RI-K skeleton (3c, 2c-metric
//! and fit-response pieces), the exchange of `F^x` by `2α K^x_DF = −(a/2)K^x_DF`,
//! and the CPHF `G` takes the SCF's own `DfK` (scaled by `a/2`) for its exchange
//! part. A pure functional (`a = 0`) never builds K, so a named `df_k_aux` is
//! inert there, as in the SCF. RI-K composes with exact or RI-J Coulomb.
//!
//! Refused with a typed error naming the next unit: meta-GGA, VV10,
//! range-separated hybrids (and so range-separated RI-K), open-shell
//! references, ECPs and external potentials.
//!
//! [`HessianMethod::Auto`]: crate::frequencies::HessianMethod::Auto
//! [`HessianMethod::Analytic`]: crate::frequencies::HessianMethod::Analytic

use super::{
    check_supported, cpks_response_with, first_order_ao, has_ecp, max_abs, nuclear_term,
    parts_from_terms, skeleton_hess_1e, skeleton_hess_2e, skeleton_hess_overlap, FockSpec,
    Reference, RhfHessianParts, TwoElectronBuilder,
};
use crate::gradient::build_energy_weighted_density;
use crate::hessian_df::RijMutants;
use crate::hessian_df_k::RikMutants;
use crate::ks_fixed_grid::{FixedGridKs, XcGeometry, XcMutants};
use crate::result::{ScfResult, Spin};
use crate::rhf::RhfConfig;
use crate::screening::SchwarzBounds;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_core::FerricError;
use ferric_dft::libxc::{k_mix_from_xc_def, xc_def_from_name, FunctionalFamily};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ndarray::Array2;

/// Deliberately broken constructions for the mutation tests; `Default` is
/// production.
#[derive(Debug, Clone, Copy)]
pub struct KsFixedGridMutants {
    /// Scale factors on the four XC pieces of skeleton and first-order Fock.
    pub xc: XcMutants,
    /// Scale factor on the kernel response inside the CPHF (0 drops it).
    pub response_fxc: f64,
    /// Scale factor on the exact-exchange fraction `a` in the four-centre
    /// skeleton `Γ_K` (0 treats the hybrid as pure, 1 is production).
    pub skeleton_exx: f64,
    /// The same for the exact-exchange part of the fixed-density first-order
    /// Fock matrix.
    pub fock_exx: f64,
    /// The same for the exact-exchange part of the CPHF `G`.
    pub response_exx: f64,
    /// Scale on the gradient-dependent kernel derivatives (`f_σ`, `f_ρσ`,
    /// `f_σσ`) inside the CPHF `G` (0 treats a GGA kernel as an LDA one).
    pub response_gradient: f64,
    /// RI-J constructions broken one at a time (see [`RijMutants`]).
    pub rij: RijMutants,
    /// RI-J: keep the exact four-centre `Γ_J` skeleton instead of the fitted one.
    pub exact_j_skeleton: bool,
    /// RI-J: build `J^x` of the first-order Fock matrix from exact integrals.
    pub exact_j_first_order: bool,
    /// RI-J: CPHF Coulomb from the exact four-centre `J` instead of the SCF's `DfJ`.
    pub exact_j_response: bool,
    /// RI-K constructions broken one at a time (see [`RikMutants`]; its
    /// `cphf_*` fields act only on the Hartree-Fock path).
    pub rik: RikMutants,
    /// RI-K: keep the exact four-centre `Γ_K` skeleton instead of the fitted one.
    pub exact_k_skeleton: bool,
    /// RI-K: build the exchange of the first-order Fock matrix from exact integrals.
    pub exact_k_first_order: bool,
    /// RI-K: CPHF exchange from the exact four-centre `K` instead of the SCF's `DfK`.
    pub exact_k_response: bool,
    /// RI-K: the fitted skeleton and `K^x` with the Hartree-Fock weight `−¼`
    /// instead of `−a/4` (the exact-exchange fraction forgotten).
    pub rik_alpha_ignores_exx: bool,
}

impl Default for KsFixedGridMutants {
    fn default() -> Self {
        KsFixedGridMutants {
            xc: XcMutants::default(),
            response_fxc: 1.0,
            skeleton_exx: 1.0,
            fock_exx: 1.0,
            response_exx: 1.0,
            response_gradient: 1.0,
            rij: RijMutants::default(),
            exact_j_skeleton: false,
            exact_j_first_order: false,
            exact_j_response: false,
            rik: RikMutants::default(),
            exact_k_skeleton: false,
            exact_k_first_order: false,
            exact_k_response: false,
            rik_alpha_ignores_exx: false,
        }
    }
}

fn unsupported(what: &str, next: &str) -> FerricError {
    FerricError::General(format!(
        "the fixed-grid analytic Kohn-Sham Hessian does not support {what} (next unit of \
         issue #407: {next}); use the finite-difference Hessian"
    ))
}

/// Typed refusals for everything outside closed-shell LDA/GGA (pure or global hybrid)
/// with exact or RI-J Coulomb and exact or RI-K exchange.
fn ks_refusal(
    mol: &Molecule,
    prep: &PreparedBasis,
    rks: &ScfResult,
    config: &RhfConfig,
) -> Result<(), FerricError> {
    let name = config.xc.as_deref().ok_or_else(|| {
        FerricError::General(
            "the fixed-grid analytic Kohn-Sham Hessian needs a Kohn-Sham configuration \
             (RhfConfig::xc is unset); use rhf_hessian for Hartree-Fock"
                .into(),
        )
    })?;
    let xc = xc_def_from_name(name)
        .map_err(|e| FerricError::General(format!("fixed-grid KS Hessian: libxc: {e:?}")))?;
    functional_refusal(name, &xc)?;
    if rks.spin != Spin::Restricted {
        return Err(unsupported(
            "an open-shell (UKS/ROKS) reference",
            "spin-polarized XC kernel",
        ));
    }
    let route = crate::gradient::active_df_route(rks);
    if route.is_some_and(|r| r.rsh_k.is_some()) {
        return Err(unsupported(
            "range-separated density-fitted exchange",
            "range-separated exchange (erfc/erf pair) in the CPHF",
        ));
    }
    if has_ecp(mol, prep.basis_set()) || config.external_potential.is_some() {
        return Err(unsupported(
            "an ECP or an external potential",
            "ECP and embedding terms with the XC kernel",
        ));
    }
    Ok(())
}

fn functional_refusal(name: &str, xc: &ferric_dft::libxc::XcDef) -> Result<(), FerricError> {
    let mix = k_mix_from_xc_def(xc);
    if xc.vv10.is_some() {
        return Err(unsupported(
            &format!("VV10 nonlocal correlation ('{name}')"),
            "VV10 response kernel",
        ));
    }
    if xc.is_double_hybrid() {
        return Err(unsupported(
            &format!("a double hybrid ('{name}'): its MP2 term is not differentiated"),
            "double-hybrid MP2 Hessian",
        ));
    }
    if mix.omega > 0.0 {
        return Err(unsupported(
            &format!("a range-separated hybrid ('{name}')"),
            "range-separated exchange (erfc/erf pair) in the CPHF",
        ));
    }
    if xc
        .funcs
        .iter()
        .any(|f| f.family() == FunctionalFamily::MetaGga)
    {
        return Err(unsupported(
            &format!("a meta-GGA ('{name}')"),
            "tau-dependent XC kernel",
        ));
    }
    if mix.sr != mix.lr {
        return Err(unsupported(
            &format!("a functional with unequal short- and long-range exchange ('{name}')"),
            "range-separated exchange (erfc/erf pair) in the CPHF",
        ));
    }
    Ok(())
}

/// `config` as the closed-shell checklist sees it: no functional, and an unset
/// `df_j_aux` resolved the way `solve_rhf` resolves it (a Kohn-Sham SCF fits J
/// over the default basis), so the SAME `RhfConfig` serves the SCF and the
/// Hessian. A stated basis that differs from the recorded route is refused here
/// with a Kohn-Sham message.
fn checklist_config(
    rks: &ScfResult,
    config: &RhfConfig,
    exx: f64,
) -> Result<RhfConfig, FerricError> {
    let mut hf = RhfConfig {
        xc: None,
        ..config.clone()
    };
    // A pure functional never builds K (the SCF's consumption gate), so a named
    // `df_k_aux` is inert for it and the route records no exchange fit.
    if exx == 0.0 {
        hf.df_k_aux = Some(String::new());
    }
    let Some(route) = crate::gradient::active_df_route(rks) else {
        return Ok(hf);
    };
    let default = Some(crate::fock_assembly::DEFAULT_JK_AUX);
    if hf.df_j_aux.is_none() && route.j_aux.as_deref() == default {
        hf.df_j_aux = route.j_aux.clone();
    }
    if hf.df_k_aux.is_none() && route.k_aux.as_deref() == default {
        hf.df_k_aux = route.k_aux.clone();
    }
    for (what, stated, recorded, named) in [
        ("J", &hf.df_j_aux, &route.j_aux, &config.df_j_aux),
        ("K", &hf.df_k_aux, &route.k_aux, &config.df_k_aux),
    ] {
        if stated.clone().filter(|v| !v.is_empty()) != *recorded {
            let key = if what == "J" { "df_j_aux" } else { "df_k_aux" };
            return Err(FerricError::General(format!(
                "the fixed-grid Kohn-Sham Hessian: the SCF fitted {what} over {recorded:?} but the \
                 configuration states {key} = {named:?}. Pass the RhfConfig the SCF ran with \
                 ({key} = None means the default {:?} where the SCF fits {what}, as in \
                 solve_rhf; Some(\"\") means exact {what}). Do not fall back to \
                 harmonic_frequencies for a KS result: it differentiates the moving-grid surface \
                 (with the Becke weight response), a different one.",
                crate::fock_assembly::DEFAULT_JK_AUX
            )));
        }
    }
    Ok(hf)
}

/// One density fit (RI-J or RI-K) of the SCF being differentiated: a factory
/// for its auxiliary basis (called once per consumer; the basis is cheap to
/// prepare) and the three-index budget. NOT public API beyond the exactness
/// anchors.
#[doc(hidden)]
pub struct KsFit<'a> {
    /// Prepares the auxiliary basis at the reference geometry.
    pub aux: &'a dyn Fn() -> Result<PreparedBasis, FerricError>,
    /// Resident three-index budget handed to the fitters, bytes.
    pub budget_bytes: usize,
}

/// The RI-J fit (the same type as the RI-K one).
#[doc(hidden)]
pub type KsJFit<'a> = KsFit<'a>;

/// The fits of the SCF: Coulomb (`None` = exact J) and exchange (`None` =
/// exact K, or a pure functional, which has no K).
#[doc(hidden)]
#[derive(Clone, Copy, Default)]
pub struct KsFits<'a> {
    /// RI-J.
    pub j: Option<&'a KsFit<'a>>,
    /// RI-K.
    pub k: Option<&'a KsFit<'a>>,
}

/// The RI-J auxiliary basis name and budget of the SCF, or `None` for exact J:
/// the recorded route when the SCF ran the production path, else the name the
/// configuration states (the injected fixed-grid SCF records no route).
fn resolve_j_fit(rks: &ScfResult, config: &RhfConfig) -> Option<(String, usize)> {
    if let Some(r) = crate::gradient::active_df_route(rks) {
        return r.j_aux.clone().map(|n| (n, r.budget_bytes));
    }
    let name = config.df_j_aux.clone().filter(|v| !v.is_empty())?;
    Some((
        name,
        crate::rhf::resolve_three_index_budget(config.three_index_budget_bytes),
    ))
}

/// [`resolve_j_fit`] for the exchange: `None` for exact K and for a pure
/// functional (`exx = 0`, whose SCF never builds K).
fn resolve_k_fit(rks: &ScfResult, config: &RhfConfig, exx: f64) -> Option<(String, usize)> {
    if exx == 0.0 {
        return None;
    }
    if let Some(r) = crate::gradient::active_df_route(rks) {
        return r.k_aux.clone().map(|n| (n, r.budget_bytes));
    }
    let name = config.df_k_aux.clone().filter(|v| !v.is_empty())?;
    Some((
        name,
        crate::rhf::resolve_three_index_budget(config.three_index_budget_bytes),
    ))
}

/// Largest `|F_rebuilt − F_scf|` (Hartree) for which `rks` is accepted as a
/// converged SCF of the energy this Hessian differentiates. The SCF stores the
/// Fock matrix built from its FINAL density, so for the right energy the
/// rebuilt matrix agrees to rounding whatever the density convergence:
/// measured 0 to 3.6e-15 Ha (injected SCF at density_conv 1e-11, production
/// SCF at 1e-11 and at the library default 1e-6). The bar is 1e-8, seven
/// orders above that; the wrong-reference scenarios sit at 0.14 (a point charge
/// of 0.5 the configuration lacks), 0.17 (a PBE result) and 2.0 (an HF result).
const FOCK_CONSISTENCY_TOL: f64 = 1e-8;

/// The Fock matrix rebuilt from `rks.density_r()` with the (exact or RI-J) J,
/// (exact or RI-K) K and fixed-grid KS energy this Hessian differentiates must
/// reproduce `rks.fock_r()`. The configuration only names the functional, so an
/// HF result, a result of another functional, or one converged with an external
/// potential or field that `config` does not carry would otherwise return a
/// silently wrong Hessian.
fn fock_consistency(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    rks: &ScfResult,
    grid: &FixedGridKs,
    fits: KsFits<'_>,
) -> Result<(), FerricError> {
    let diff = rks_fixed_grid_fock_residual(ctx, mol, prep, bounds, rks, grid, fits)?;
    if diff > FOCK_CONSISTENCY_TOL {
        return Err(FerricError::General(format!(
            "the SCF result is not a converged closed-shell Kohn-Sham solution (with the Coulomb \
             and exchange matrices the fits state) of the molecule, basis and functional passed \
             (the Fock matrix rebuilt at its density differs from the stored one by {diff:.2e} \
             Ha > {FOCK_CONSISTENCY_TOL:.0e}): a Hartree-Fock or other-functional result, a \
             missing external potential, or an unconverged SCF"
        )));
    }
    Ok(())
}

/// `max |F_rebuilt(D) − F_scf|` of [`fock_consistency`], Hartree. With
/// `fits.j` the Coulomb matrix is the fitted one, with `fits.k` the exchange.
#[doc(hidden)]
pub fn rks_fixed_grid_fock_residual(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    rks: &ScfResult,
    grid: &FixedGridKs,
    fits: KsFits<'_>,
) -> Result<f64, FerricError> {
    use crate::fock::{JBuilder, KBuilder};
    let d = rks.density_r();
    let mut j = Array2::<f64>::zeros(d.dim());
    let mut k = Array2::<f64>::zeros(d.dim());
    crate::rhf::build_jk(ctx, prep, bounds, 1e-14, d, &mut j, &mut k)?;
    if let Some(fit) = fits.j {
        j.fill(0.0);
        let dfbs = (fit.aux)()?;
        crate::df_j::DfJ::new(bounds.op, prep, &dfbs, fit.budget_bytes)?.build(d, &mut j)?;
    }
    if let Some(fit) = fits.k {
        k.fill(0.0);
        let dfbs = (fit.aux)()?;
        crate::df_k::DfK::new(bounds.op, prep, &dfbs, fit.budget_bytes)?.build(d, &mut k)?;
    }
    let h =
        ferric_integrals::oneelectron::hcore_ecp_with_external(prep, mol, prep.basis_set(), None)?;
    let (_exc, v) = grid.energy_vxc(&XcGeometry::new(mol, prep)?, d)?;
    let f = h + j + v - &(0.5 * grid.exact_exchange() * &k);
    Ok(max_abs(&(f - rks.fock_r())))
}

/// The analytic Hessian of the fixed-grid closed-shell KS energy, (3N, 3N).
/// See the module doc for what it is and is not the Hessian of.
pub fn rks_fixed_grid_hessian(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    rks: &ScfResult,
    config: &RhfConfig,
) -> Result<Array2<f64>, FerricError> {
    Ok(rks_fixed_grid_hessian_parts(ctx, mol, prep, op, bounds, rks, config)?.total())
}

/// [`rks_fixed_grid_hessian`] with the per-term breakdown. The XC second
/// derivative is reported in `two_electron`.
pub fn rks_fixed_grid_hessian_parts(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    rks: &ScfResult,
    config: &RhfConfig,
) -> Result<RhfHessianParts, FerricError> {
    ks_refusal(mol, prep, rks, config)?;
    let name = config.xc.as_deref().unwrap_or_default();
    let cfg = config.dft_grid.clone().unwrap_or_default();
    let grid = FixedGridKs::new(mol, name, &cfg)?;
    rks_fixed_grid_hessian_parts_with(
        ctx,
        mol,
        prep,
        op,
        bounds,
        rks,
        config,
        &grid,
        &KsFixedGridMutants::default(),
    )
}

/// [`rks_fixed_grid_hessian_parts`] on a caller-built grid and with
/// deliberately broken constructions (mutation tests).
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn rks_fixed_grid_hessian_parts_with(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    rks: &ScfResult,
    config: &RhfConfig,
    grid: &FixedGridKs,
    mutants: &KsFixedGridMutants,
) -> Result<RhfHessianParts, FerricError> {
    ks_refusal(mol, prep, rks, config)?;
    // Everything but the functional is the closed-shell checklist; a stated or
    // recorded RI-J / RI-K basis goes through the same engine and metric probes
    // as RHF.
    let exx = grid.exact_exchange();
    let j_fit = resolve_j_fit(rks, config);
    let k_fit = resolve_k_fit(rks, config, exx);
    let hf_config = checklist_config(rks, config, exx)?;
    check_supported(Reference::Rhf, mol, prep, op, bounds, rks, &hf_config)?;
    let aux = |name: String| move || crate::df_gradient::aux_basis(mol, &name);
    let j_factory = j_fit.clone().map(|(n, _)| aux(n));
    let k_factory = k_fit.clone().map(|(n, _)| aux(n));
    let jf = j_factory
        .as_ref()
        .zip(j_fit)
        .map(|(f, (_, budget_bytes))| KsFit {
            aux: f,
            budget_bytes,
        });
    let kf = k_factory
        .as_ref()
        .zip(k_fit)
        .map(|(f, (_, budget_bytes))| KsFit {
            aux: f,
            budget_bytes,
        });
    let fits = KsFits {
        j: jf.as_ref(),
        k: kf.as_ref(),
    };
    rks_fixed_grid_hessian_core(ctx, mol, prep, op, bounds, rks, config, grid, fits, mutants)
}

/// [`rks_fixed_grid_hessian_parts_with`] for an SCF whose Coulomb and/or
/// exchange matrices are fitted over the caller's auxiliary bases (exactness
/// anchors: a basis that spans the density's products, a richer basis, a
/// mutated fit). The caller owns the checks `check_supported` would run for a
/// named basis.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn rks_fixed_grid_hessian_parts_with_aux(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    rks: &ScfResult,
    config: &RhfConfig,
    grid: &FixedGridKs,
    fits: KsFits<'_>,
    mutants: &KsFixedGridMutants,
) -> Result<RhfHessianParts, FerricError> {
    ks_refusal(mol, prep, rks, config)?;
    let hf_config = RhfConfig {
        xc: None,
        df_j_aux: None,
        df_k_aux: None,
        ..config.clone()
    };
    check_supported(Reference::Rhf, mol, prep, op, bounds, rks, &hf_config)?;
    rks_fixed_grid_hessian_core(ctx, mol, prep, op, bounds, rks, config, grid, fits, mutants)
}

/// The fitted Coulomb and exchange pieces of the skeleton and of `F^x`.
struct FittedTerms {
    /// Sum of the RI-J and RI-K skeletons (zero when nothing is fitted).
    skeleton: Array2<f64>,
    /// Fitted `J^x` and `−(a/2) K^x_DF` added to the fixed-density `F^x`.
    first_order: Vec<Array2<f64>>,
    /// The global-pool charge of the first-order Fock / CPHF phase that
    /// follows the RI-K terms (held until the Hessian returns).
    _cphf_charge: Option<ferric_core::memory::pool::Reservation>,
}

/// RI-J (`rij_terms_with_aux`) and RI-K (`rik_channel_terms_with_aux`, the one
/// closed-shell channel `X = √2 C_occ` with weight `α = −a/4`) pieces. The RI-K
/// pieces are linear in `α` at fixed `X`, so the exact-exchange mutants scale
/// the skeleton and `K^x` separately exactly as a scaled `α` would.
#[allow(clippy::too_many_arguments)]
fn fitted_terms(
    natoms: usize,
    prep: &PreparedBasis,
    op: Operator,
    rks: &ScfResult,
    nocc: usize,
    exx: f64,
    fits: KsFits<'_>,
    mutants: &KsFixedGridMutants,
) -> Result<FittedTerms, FerricError> {
    let d = rks.density_r();
    let n3 = 3 * natoms;
    let mut skeleton = Array2::<f64>::zeros((n3, n3));
    let mut first_order = vec![Array2::<f64>::zeros(d.dim()); n3];
    if let Some(fit) = fits.j {
        let rij = crate::hessian_df::rij_terms_with_aux(
            natoms,
            prep,
            (fit.aux)()?,
            op,
            fit.budget_bytes,
            d,
            &mutants.rij,
        )?;
        if !mutants.exact_j_skeleton {
            skeleton += &rij.skeleton;
        }
        if !mutants.exact_j_first_order {
            for (f, j) in first_order.iter_mut().zip(&rij.first_order_j) {
                *f += j;
            }
        }
    }
    let mut cphf_charge = None;
    if let Some(fit) = fits.k {
        let x = super::rik_occupied_factor(rks, nocc, d)?;
        let channel = crate::hessian_df_k::RikChannel {
            x: &x,
            alpha: crate::hessian_df_k::RIK_ALPHA
                * if mutants.rik_alpha_ignores_exx {
                    1.0
                } else {
                    exx
                },
        };
        let dfbs = (fit.aux)()?;
        let naux = dfbs.nbasis();
        let mut rik = crate::hessian_df_k::rik_channel_terms_with_aux(
            natoms,
            prep,
            dfbs,
            op,
            fit.budget_bytes,
            &[channel],
            &mutants.rik,
        )?;
        cphf_charge = Some(crate::hessian_df_k::reserve_cphf_phase(
            natoms,
            prep.nbasis(),
            &[nocc],
            naux,
        )?);
        if !mutants.exact_k_skeleton {
            skeleton.scaled_add(mutants.skeleton_exx, &rik.skeleton);
        }
        if !mutants.exact_k_first_order {
            let kx = rik.channels.remove(0).first_order_fock;
            for (f, k) in first_order.iter_mut().zip(&kx) {
                f.scaled_add(mutants.fock_exx, k);
            }
        }
    }
    Ok(FittedTerms {
        skeleton,
        first_order,
        _cphf_charge: cphf_charge,
    })
}

/// The fitters the CPHF `G` uses: the SCF's own `DfJ` / `DfK` (the mutants
/// swap in the exact four-centre matrices).
fn cphf_fitters(
    prep: &PreparedBasis,
    op: Operator,
    fits: KsFits<'_>,
    mutants: &KsFixedGridMutants,
) -> Result<
    (
        Option<crate::df_j::DfJ<'static>>,
        Option<crate::df_k::DfK<'static>>,
    ),
    FerricError,
> {
    let df_j = match fits.j {
        Some(fit) if !mutants.exact_j_response => Some(crate::df_j::DfJ::new(
            op,
            prep,
            &(fit.aux)()?,
            fit.budget_bytes,
        )?),
        _ => None,
    };
    let df_k = match fits.k {
        Some(fit) if !mutants.exact_k_response => Some(crate::df_k::DfK::new(
            op,
            prep,
            &(fit.aux)()?,
            fit.budget_bytes,
        )?),
        _ => None,
    };
    Ok((df_j, df_k))
}

#[allow(clippy::too_many_arguments)]
fn rks_fixed_grid_hessian_core(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    rks: &ScfResult,
    config: &RhfConfig,
    grid: &FixedGridKs,
    fits: KsFits<'_>,
    mutants: &KsFixedGridMutants,
) -> Result<RhfHessianParts, FerricError> {
    fock_consistency(ctx, mol, prep, bounds, rks, grid, fits)?;

    let natoms = mol.atoms.len();
    let nocc = (mol.nelec() / 2) as usize;
    let d = rks.density_r();
    let w = build_energy_weighted_density(rks, nocc);
    let geo = XcGeometry::new(mol, prep)?;
    let exx = grid.exact_exchange();

    let nuclear = nuclear_term(mol, config);
    let one_electron = skeleton_hess_1e(prep, natoms, &[], d)?;
    let overlap = skeleton_hess_overlap(prep, natoms, &w)?;
    let fitted = fitted_terms(natoms, prep, op, rks, nocc, exx, fits, mutants)?;
    // A fitted Coulomb (exchange) term replaces the four-centre Gamma_J
    // (Gamma_K); the mutants keep the exact one.
    let keep_j = fits.j.is_none() || mutants.exact_j_skeleton;
    let keep_k = fits.k.is_none() || mutants.exact_k_skeleton;
    let c_j = if keep_j { 0.5 } else { 0.0 };
    let c_k = if keep_k {
        -0.25 * exx * mutants.skeleton_exx
    } else {
        0.0
    };
    let mut two_electron = if c_j == 0.0 && c_k == 0.0 {
        Array2::<f64>::zeros((3 * natoms, 3 * natoms))
    } else {
        let gam = |mu: usize, nu: usize, la: usize, sg: usize| {
            c_j * d[(mu, nu)] * d[(la, sg)] + c_k * d[(mu, la)] * d[(nu, sg)]
        };
        skeleton_hess_2e(prep, op, bounds, natoms, max_abs(d), &gam)?
    };
    two_electron += &grid.skeleton_hessian(&geo, d, &mutants.xc)?;
    two_electron += &fitted.skeleton;

    let no_density = Array2::<f64>::zeros(d.dim());
    let fitted_j_first = fits.j.is_some() && !mutants.exact_j_first_order;
    let fitted_k_first = fits.k.is_some() && !mutants.exact_k_first_order;
    let spec = FockSpec {
        coulomb: if fitted_j_first { &no_density } else { d },
        exchange: if fitted_k_first { &no_density } else { d },
        k_scale: 0.5 * exx * mutants.fock_exx,
    };
    let mut first = first_order_ao(prep, op, bounds, natoms, &[], &[spec])?;
    for (f, g) in first.f1[0].iter_mut().zip(&fitted.first_order) {
        *f += g;
    }
    for (f, v) in first.f1[0]
        .iter_mut()
        .zip(grid.first_order_fock(&geo, d, &mutants.xc)?)
    {
        *f += &v;
    }
    let kernel = grid.response(&geo, d, mutants.response_fxc, mutants.response_gradient)?;
    let kernel_fn = |d1: &Array2<f64>| kernel.apply(d1);
    let (df_j, df_k) = cphf_fitters(prep, op, fits, mutants)?;
    let g = TwoElectronBuilder::new(ctx, prep, bounds)?
        .fitted(df_j, df_k, false)
        .with_xc_response(Some(&kernel_fn), exx * mutants.response_exx);
    let resp = cpks_response_with(&g, rks, nocc, &first)?;
    Ok(parts_from_terms(
        [nuclear, one_electron, overlap, two_electron],
        resp,
    ))
}
