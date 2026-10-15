//! Opt-in analytic Hessian of closed-shell Kohn-Sham LDA with exact
//! four-centre J on a grid FIXED IN SPACE (unit 1 of issue #407).
//!
//! The energy differentiated is
//!
//! ```text
//!   E = V_nn + Tr[D h] + ½ Tr[D J(D)] + Σ_g w_g f(ρ(r_g)),   D = 2 C_occ C_occᵀ,
//! ```
//!
//! with the grid points and weights frozen at the reference geometry (PySCF
//! `grid_response=False`; see [`crate::ks_fixed_grid`]). It is NOT the
//! derivative of the energy the production KS SCF minimizes on an atom-following
//! grid: that gradient carries the Becke weight response, which this Hessian
//! omits, so the two differ by the grid's translation error and weight
//! derivatives (measured in `tests/ks_fixed_grid_hessian.rs`). That is why the
//! entry points here are explicit and [`HessianMethod::Auto`] and
//! [`HessianMethod::Analytic`] never reach them.
//!
//! The Hessian is the RHF one with `G(D) = J(D) − ½K(D)` replaced by
//! `G(D) = J(D) + V1_xc[D]`:
//!
//! * skeleton: the RHF terms with `Γ_J = ½ D D` only, plus the XC second
//!   derivative at fixed `D` ([`FixedGridLda::skeleton_hessian`]);
//! * `F^x` gains `∂V_xc/∂x|_D` ([`FixedGridLda::first_order_fock`]);
//! * the CPHF `G` (right-hand side, matrix-vector product and the first-order
//!   orbital energies) gains the kernel response `V1_xc[D1] = Σ_g w f'' ρ1 χ χ`.
//!
//! Refused with a typed error naming the next unit: GGA, meta-GGA, VV10,
//! global and range-separated hybrids, density fitting, open-shell references,
//! ECPs and external potentials.
//!
//! [`HessianMethod::Auto`]: crate::frequencies::HessianMethod::Auto
//! [`HessianMethod::Analytic`]: crate::frequencies::HessianMethod::Analytic

use super::{
    check_supported, cpks_response, first_order_ao, has_ecp, max_abs, nuclear_term,
    parts_from_terms, skeleton_hess_1e, skeleton_hess_2e, skeleton_hess_overlap, FockSpec,
    Reference, RhfHessianParts,
};
use crate::gradient::build_energy_weighted_density;
use crate::ks_fixed_grid::{FixedGridLda, XcGeometry, XcMutants};
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
}

impl Default for KsFixedGridMutants {
    fn default() -> Self {
        KsFixedGridMutants {
            xc: XcMutants::default(),
            response_fxc: 1.0,
        }
    }
}

fn unsupported(what: &str, next: &str) -> FerricError {
    FerricError::General(format!(
        "the fixed-grid analytic Kohn-Sham Hessian does not support {what} (next unit of \
         issue #407: {next}); use the finite-difference Hessian"
    ))
}

/// Typed refusals for everything outside closed-shell pure-LDA exact-J.
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
    if named(&config.df_j_aux)
        || named(&config.df_k_aux)
        || crate::gradient::active_df_route(rks).is_some()
    {
        return Err(unsupported(
            "density-fitted J/K (RI); converge the SCF with exact J \
             (df_j_aux = Some(\"\"), the KS default is RI-J)",
            "RI-J/RI-K skeleton and CPHF with the XC kernel",
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
    if xc.funcs.iter().any(|f| f.family() != FunctionalFamily::Lda) {
        return Err(unsupported(
            &format!("a GGA or hybrid-GGA XC kernel ('{name}')"),
            "GGA XC kernel (needs third AO derivatives) and hybrid exchange scaling",
        ));
    }
    if mix.sr != 0.0 || mix.lr != 0.0 {
        return Err(unsupported(
            &format!("a global hybrid ('{name}')"),
            "hybrid exchange scaling a in the skeleton and CPHF",
        ));
    }
    Ok(())
}

/// The analytic Hessian of the fixed-grid closed-shell LDA energy, (3N, 3N).
/// See the module doc for what it is and is not the Hessian of.
pub fn rks_lda_fixed_grid_hessian(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    rks: &ScfResult,
    config: &RhfConfig,
) -> Result<Array2<f64>, FerricError> {
    Ok(rks_lda_fixed_grid_hessian_parts(ctx, mol, prep, op, bounds, rks, config)?.total())
}

/// [`rks_lda_fixed_grid_hessian`] with the per-term breakdown. The XC second
/// derivative is reported in `two_electron`.
pub fn rks_lda_fixed_grid_hessian_parts(
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
    let grid = FixedGridLda::new(mol, name, &cfg)?;
    rks_lda_fixed_grid_hessian_parts_with(
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

/// [`rks_lda_fixed_grid_hessian_parts`] on a caller-built grid and with
/// deliberately broken constructions (mutation tests).
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn rks_lda_fixed_grid_hessian_parts_with(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    rks: &ScfResult,
    config: &RhfConfig,
    grid: &FixedGridLda,
    mutants: &KsFixedGridMutants,
) -> Result<RhfHessianParts, FerricError> {
    ks_refusal(mol, prep, rks, config)?;
    // Everything but the functional is the closed-shell exact-J/K checklist.
    let hf_config = RhfConfig {
        xc: None,
        ..config.clone()
    };
    check_supported(Reference::Rhf, mol, prep, op, bounds, rks, &hf_config)?;

    let natoms = mol.atoms.len();
    let nocc = (mol.nelec() / 2) as usize;
    let d = rks.density_r();
    let w = build_energy_weighted_density(rks, nocc);
    let geo = XcGeometry::new(mol, prep)?;

    let nuclear = nuclear_term(mol, config);
    let one_electron = skeleton_hess_1e(prep, natoms, &[], d)?;
    let overlap = skeleton_hess_overlap(prep, natoms, &w)?;
    let gam_j = |mu: usize, nu: usize, la: usize, sg: usize| 0.5 * d[(mu, nu)] * d[(la, sg)];
    let mut two_electron = skeleton_hess_2e(prep, op, bounds, natoms, max_abs(d), &gam_j)?;
    two_electron += &grid.skeleton_hessian(&geo, d, &mutants.xc)?;

    let spec = FockSpec {
        coulomb: d,
        exchange: d,
        k_scale: 0.0,
    };
    let mut first = first_order_ao(prep, op, bounds, natoms, &[], &[spec])?;
    for (f, v) in first.f1[0]
        .iter_mut()
        .zip(grid.first_order_fock(&geo, d, &mutants.xc)?)
    {
        *f += &v;
    }
    let kernel = grid.response(&geo, d, mutants.response_fxc)?;
    let kernel_fn = |d1: &Array2<f64>| kernel.apply(d1);
    let resp = cpks_response(
        ctx,
        prep,
        bounds,
        rks,
        nocc,
        &first,
        (None, None),
        false,
        Some(&kernel_fn),
    )?;
    Ok(parts_from_terms(
        [nuclear, one_electron, overlap, two_electron],
        resp,
    ))
}
