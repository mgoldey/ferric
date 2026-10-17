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
//! and kernel pieces are the exact-J ones, untouched. RI-K is refused.
//!
//! Refused with a typed error naming the next unit: meta-GGA, VV10,
//! range-separated hybrids, RI-K, open-shell references, ECPs and
//! external potentials.
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
        }
    }
}

fn unsupported(what: &str, next: &str) -> FerricError {
    FerricError::General(format!(
        "the fixed-grid analytic Kohn-Sham Hessian does not support {what} (next unit of \
         issue #407: {next}); use the finite-difference Hessian"
    ))
}

/// Typed refusals for everything outside closed-shell exact-J/K LDA/GGA (pure or global hybrid).
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
    let named = |s: &Option<String>| s.as_deref().is_some_and(|v| !v.is_empty());
    let route = crate::gradient::active_df_route(rks);
    if named(&config.df_k_aux)
        || route.is_some_and(|r| r.k_aux.is_some())
        || route.is_some_and(|r| r.rsh_k.is_some())
    {
        return Err(unsupported(
            "density-fitted exchange (RI-K); converge the SCF with exact K \
             (df_k_aux = Some(\"\"), a hybrid's K is exact by default)",
            "RI-K skeleton and CPHF with the XC kernel",
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

/// The RI-J fit of the SCF being differentiated: a factory for its auxiliary
/// basis (called once per consumer; the basis is cheap to prepare) and the
/// three-index budget. NOT public API beyond the exactness anchors.
#[doc(hidden)]
pub struct KsJFit<'a> {
    /// Prepares the auxiliary basis at the reference geometry.
    pub aux: &'a dyn Fn() -> Result<PreparedBasis, FerricError>,
    /// Resident three-index budget handed to the fitters, bytes.
    pub budget_bytes: usize,
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

/// Largest `|F_rebuilt − F_scf|` (Hartree) for which `rks` is accepted as a
/// converged SCF of the energy this Hessian differentiates. The SCF stores the
/// Fock matrix built from its FINAL density, so for the right energy the
/// rebuilt matrix agrees to rounding whatever the density convergence:
/// measured 0 to 3.6e-15 Ha (injected SCF at density_conv 1e-11, production
/// SCF at 1e-11 and at the library default 1e-6). The bar is 1e-8, seven
/// orders above that; the wrong-reference scenarios sit at 0.14 (a point charge
/// of 0.5 the configuration lacks), 0.17 (a PBE result) and 2.0 (an HF result).
const FOCK_CONSISTENCY_TOL: f64 = 1e-8;

/// The Fock matrix rebuilt from `rks.density_r()` with the (exact or RI-J) J, exact K, fixed-grid
/// KS energy this Hessian differentiates must reproduce `rks.fock_r()`. The
/// configuration only names the functional, so an HF result, a result of
/// another functional, or one converged with an external potential or field
/// that `config` does not carry would otherwise return a silently wrong Hessian.
fn fock_consistency(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    rks: &ScfResult,
    grid: &FixedGridKs,
    fit: Option<&KsJFit<'_>>,
) -> Result<(), FerricError> {
    let diff = rks_fixed_grid_fock_residual(ctx, mol, prep, bounds, rks, grid, fit)?;
    if diff > FOCK_CONSISTENCY_TOL {
        return Err(FerricError::General(format!(
            "the SCF result is not a converged closed-shell Kohn-Sham solution (with the Coulomb matrix the fit states) \
             of the molecule, basis and functional passed (the Fock matrix rebuilt at its \
             density differs from the stored one by {diff:.2e} Ha > {FOCK_CONSISTENCY_TOL:.0e}): \
             a Hartree-Fock or other-functional result, a missing external potential, or an \
             unconverged SCF"
        )));
    }
    Ok(())
}

/// `max |F_rebuilt(D) − F_scf|` of [`fock_consistency`], Hartree. With `fit`
/// the Coulomb matrix is the fitted one.
#[doc(hidden)]
pub fn rks_fixed_grid_fock_residual(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    rks: &ScfResult,
    grid: &FixedGridKs,
    fit: Option<&KsJFit<'_>>,
) -> Result<f64, FerricError> {
    use crate::fock::JBuilder;
    let d = rks.density_r();
    let mut j = Array2::<f64>::zeros(d.dim());
    let mut k = Array2::<f64>::zeros(d.dim());
    crate::rhf::build_jk(ctx, prep, bounds, 1e-14, d, &mut j, &mut k)?;
    if let Some(fit) = fit {
        j.fill(0.0);
        let dfbs = (fit.aux)()?;
        crate::df_j::DfJ::new(bounds.op, prep, &dfbs, fit.budget_bytes)?.build(d, &mut j)?;
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
    // recorded RI-J basis goes through the same engine and metric probes as RHF.
    let j_fit = resolve_j_fit(rks, config);
    let hf_config = RhfConfig {
        xc: None,
        ..config.clone()
    };
    check_supported(Reference::Rhf, mol, prep, op, bounds, rks, &hf_config)?;
    let aux = |name: String| move || crate::df_gradient::aux_basis(mol, &name);
    match j_fit {
        None => rks_fixed_grid_hessian_core(
            ctx, mol, prep, op, bounds, rks, config, grid, None, mutants,
        ),
        Some((name, budget_bytes)) => {
            let factory = aux(name);
            let fit = KsJFit {
                aux: &factory,
                budget_bytes,
            };
            rks_fixed_grid_hessian_core(
                ctx,
                mol,
                prep,
                op,
                bounds,
                rks,
                config,
                grid,
                Some(&fit),
                mutants,
            )
        }
    }
}

/// [`rks_fixed_grid_hessian_parts_with`] for an SCF whose Coulomb matrix is the
/// RI-J one over the caller's auxiliary basis (exactness anchors: a basis that
/// spans the density's products, a richer basis, a mutated fit). The caller
/// owns the checks `check_supported` would run for a named basis.
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
    fit: &KsJFit<'_>,
    mutants: &KsFixedGridMutants,
) -> Result<RhfHessianParts, FerricError> {
    ks_refusal(mol, prep, rks, config)?;
    let hf_config = RhfConfig {
        xc: None,
        df_j_aux: None,
        ..config.clone()
    };
    check_supported(Reference::Rhf, mol, prep, op, bounds, rks, &hf_config)?;
    rks_fixed_grid_hessian_core(
        ctx,
        mol,
        prep,
        op,
        bounds,
        rks,
        config,
        grid,
        Some(fit),
        mutants,
    )
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
    fit: Option<&KsJFit<'_>>,
    mutants: &KsFixedGridMutants,
) -> Result<RhfHessianParts, FerricError> {
    fock_consistency(ctx, mol, prep, bounds, rks, grid, fit)?;

    let natoms = mol.atoms.len();
    let nocc = (mol.nelec() / 2) as usize;
    let d = rks.density_r();
    let w = build_energy_weighted_density(rks, nocc);
    let geo = XcGeometry::new(mol, prep)?;

    let nuclear = nuclear_term(mol, config);
    let one_electron = skeleton_hess_1e(prep, natoms, &[], d)?;
    let overlap = skeleton_hess_overlap(prep, natoms, &w)?;
    let a_skel = grid.exact_exchange() * mutants.skeleton_exx;
    // A fitted Coulomb term replaces the four-centre Gamma_J (the mutant keeps it).
    let keep_gamma_j = if fit.is_some() && !mutants.exact_j_skeleton {
        0.0
    } else {
        1.0
    };
    let gam_j = |mu: usize, nu: usize, la: usize, sg: usize| {
        keep_gamma_j * 0.5 * d[(mu, nu)] * d[(la, sg)] - 0.25 * a_skel * d[(mu, la)] * d[(nu, sg)]
    };
    let mut two_electron = skeleton_hess_2e(prep, op, bounds, natoms, max_abs(d), &gam_j)?;
    two_electron += &grid.skeleton_hessian(&geo, d, &mutants.xc)?;
    let rij = match fit {
        Some(fit) => Some(crate::hessian_df::rij_terms_with_aux(
            natoms,
            prep,
            (fit.aux)()?,
            op,
            fit.budget_bytes,
            d,
            &mutants.rij,
        )?),
        None => None,
    };
    if let Some(rij) = &rij {
        two_electron += &rij.skeleton;
    }

    let no_density = Array2::<f64>::zeros(d.dim());
    let fitted_first = rij.is_some() && !mutants.exact_j_first_order;
    let spec = FockSpec {
        coulomb: if fitted_first { &no_density } else { d },
        exchange: d,
        k_scale: 0.5 * grid.exact_exchange() * mutants.fock_exx,
    };
    let mut first = first_order_ao(prep, op, bounds, natoms, &[], &[spec])?;
    if let (Some(rij), true) = (&rij, fitted_first) {
        for (f, j) in first.f1[0].iter_mut().zip(&rij.first_order_j) {
            *f += j;
        }
    }
    for (f, v) in first.f1[0]
        .iter_mut()
        .zip(grid.first_order_fock(&geo, d, &mutants.xc)?)
    {
        *f += &v;
    }
    let kernel = grid.response(&geo, d, mutants.response_fxc, mutants.response_gradient)?;
    let kernel_fn = |d1: &Array2<f64>| kernel.apply(d1);
    let df_j = match fit {
        Some(fit) if !mutants.exact_j_response => Some(crate::df_j::DfJ::new(
            op,
            prep,
            &(fit.aux)()?,
            fit.budget_bytes,
        )?),
        _ => None,
    };
    let g = TwoElectronBuilder::new(ctx, prep, bounds)?
        .fitted(df_j, None, false)
        .with_xc_response(
            Some(&kernel_fn),
            grid.exact_exchange() * mutants.response_exx,
        );
    let resp = cpks_response_with(&g, rks, nocc, &first)?;
    Ok(parts_from_terms(
        [nuclear, one_electron, overlap, two_electron],
        resp,
    ))
}
