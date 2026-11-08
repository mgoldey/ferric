//! Analytic Hessian of spin-unrestricted Kohn-Sham (UKS) LDA/GGA, pure or
//! global hybrid, with exact or RI-J Coulomb and exact or RI-K exchange, on the
//! PRODUCTION (moving) grid: the surface `ks_gradient_uks` differentiates,
//! Becke weight response included.
//!
//! The energy differentiated is the UHF one with exchange scaled by the
//! global-hybrid fraction `a` plus the spin-polarized XC energy,
//!
//! ```text
//!   E = V_nn + Tr[D h] + ½ Tr[D J(D)] − (a/2) Σ_s Tr[D^s K(D^s)] + E_xc[D^α, D^β],
//! ```
//!
//! so the Hessian is the UHF one ([`super::uhf_hessian`]) with
//!
//! * skeleton: `Γ = ½ D D − (a/2) Σ_s D^s D^s` (RI-J on the total density and
//!   one RI-K channel per spin with weight `−a/2` where the SCF fitted them,
//!   `super::uhf_df::open_shell_two_electron`), plus the polarized XC second
//!   derivative at fixed `(D^α, D^β)` on the moving grid
//!   ([`crate::ks_fixed_grid::PolarizedKs::skeleton_hessian_moving`]);
//! * `F^{s,x}` = the UHF one with `K^x` scaled by `a`, plus `∂V^s_xc/∂x|_D`
//!   ([`crate::ks_fixed_grid::PolarizedKs::first_order_fock_moving`]);
//! * the coupled α/β CPKS `v_s[D1] = J[D1^α + D1^β] − a K[D1^s] +
//!   V1^s_xc[D1^α, D1^β]`, with the spin-coupled kernel
//!   ([`crate::ks_fixed_grid::PolarizedXcResponse`]: the αα, αβ and ββ blocks
//!   of `f_xc`, `σ` couplings included for a GGA).
//!
//! Equations: `crates/ferric-scf/theory.md` ("Open shell"); the spin algebra
//! is proved in `scripts/uks_xc_hessian_sympy.py`. Refused with a typed error:
//! meta-GGA, VV10, range-separated and double hybrids, ROKS, ECPs, external
//! potentials and everything the UHF Hessian refuses.

use super::ks::{
    checklist_config, ecp_or_embedding_refusal, functional_refusal, unsupported,
    FOCK_CONSISTENCY_TOL,
};
use super::uhf_df::{open_shell_two_electron, OpenShellDensities};
use super::*;
use crate::ks_fixed_grid::{FixedGridKs, GridMode, PolarizedKs, XcGeometry, XcMutants};
use ferric_dft::libxc::xc_def_from_name;

/// Typed refusals for everything outside UKS LDA/GGA (pure or global hybrid)
/// with exact or RI-J Coulomb and exact or RI-K exchange, that need the SCF.
fn uks_refusal(
    mol: &Molecule,
    prep: &PreparedBasis,
    uks: &ScfResult,
    config: &RhfConfig,
) -> Result<(), FerricError> {
    let name = config.xc.as_deref().ok_or_else(|| {
        FerricError::General(
            "the analytic UKS Hessian needs a Kohn-Sham configuration (RhfConfig::xc is unset); \
             use uhf_hessian for Hartree-Fock"
                .into(),
        )
    })?;
    let xc = xc_def_from_name(name)
        .map_err(|e| FerricError::General(format!("UKS Hessian: libxc: {e:?}")))?;
    functional_refusal(name, &xc)?;
    match uks.spin {
        Spin::Unrestricted if uks.rohf_spin_focks.is_some() => {
            return Err(unsupported(
                "a semicanonicalized ROKS result",
                "the ROKS analytic Hessian (Roothaan-coupled response)",
            ))
        }
        Spin::Unrestricted => {}
        Spin::Restricted => {
            return Err(FerricError::General(
                "the analytic UKS Hessian needs an unrestricted (UKS) result; a restricted \
                 closed-shell KS result is differentiated by rks_moving_grid_hessian"
                    .into(),
            ))
        }
        Spin::RestrictedOpen => {
            return Err(unsupported(
                "an ROKS reference",
                "the ROKS analytic Hessian (Roothaan-coupled response)",
            ))
        }
    }
    if crate::gradient::active_df_route(uks).is_some_and(|r| r.rsh_k.is_some()) {
        return Err(unsupported(
            "range-separated density-fitted exchange",
            "range-separated exchange (erfc/erf pair) in the CPHF",
        ));
    }
    ecp_or_embedding_refusal(mol, prep.basis_set(), config)
}

/// The refusals of [`uks_moving_grid_hessian`] that need no SCF (the UKS
/// counterpart of [`super::rks_moving_grid_hessian_preflight`]): no functional,
/// meta-GGA, VV10, range-separated or double hybrids, an ECP, an external
/// potential, and everything the UHF Hessian refuses before its SCF.
pub fn uks_moving_grid_hessian_preflight(
    mol: &Molecule,
    bs: &ferric_core::basis::BasisSet,
    op: Operator,
    config: &RhfConfig,
) -> Result<(), FerricError> {
    let name = config.xc.as_deref().ok_or_else(|| {
        FerricError::General(
            "the analytic moving-grid UKS Hessian needs a Kohn-Sham functional (RhfConfig::xc \
             is unset): Hartree-Fock has no XC grid; use the analytic UHF Hessian (hessian = \
             \"analytic\" or \"auto\")"
                .into(),
        )
    })?;
    let xc = xc_def_from_name(name)
        .map_err(|e| FerricError::General(format!("moving-grid UKS Hessian: libxc: {e:?}")))?;
    functional_refusal(name, &xc)?;
    ecp_or_embedding_refusal(mol, bs, config)?;
    let hf = RhfConfig {
        xc: None,
        ..config.clone()
    };
    if let Some(what) = super::preflight_problem(Reference::Uhf, op, &hf)? {
        return Err(unsupported(
            &what,
            "support for it in the analytic UHF Hessian",
        ));
    }
    Ok(())
}

/// THE pre-SCF support predicate of [`uks_moving_grid_hessian`] (the UKS
/// counterpart of [`super::rks_moving_grid_hessian_available`]):
/// [`uks_moving_grid_hessian_preflight`] plus the UHF Hessian's molecule- and
/// basis-level checks on the fitting bases the SCF will use (`solve_uhf` fits
/// only a named basis) and the size of
/// the response's resident AO table. `HessianMethod::AnalyticMovingGrid` (which
/// returns the error) and `HessianMethod::Auto` for a UKS reference (which runs
/// finite differences instead) both consult it.
pub fn uks_moving_grid_hessian_available(
    mol: &Molecule,
    prep: &PreparedBasis,
    op: Operator,
    config: &RhfConfig,
) -> Result<(), FerricError> {
    uks_moving_grid_hessian_preflight(mol, prep.basis_set(), op, config)?;
    let name = config.xc.as_deref().unwrap_or_default();
    let grid = FixedGridKs::new(
        mol,
        name,
        &config
            .dft_grid
            .clone()
            .unwrap_or_else(ferric_dft::grid::AtomicGridConfig::production),
    )?;
    grid.check_response_table(prep.nbasis())?;
    // `solve_uhf` fits J (K) only over a NAMED auxiliary basis: unset means
    // exact (unlike the closed-shell KS SCF, which defaults to RI-J).
    let resolve = |requested: &Option<String>| match requested.as_deref() {
        None | Some("") => Some(String::new()),
        Some(n) => Some(n.to_string()),
    };
    let exx = grid.exact_exchange();
    let hf = RhfConfig {
        xc: None,
        df_j_aux: resolve(&config.df_j_aux),
        df_k_aux: if exx == 0.0 {
            Some(String::new())
        } else {
            resolve(&config.df_k_aux)
        },
        ..config.clone()
    };
    super::available(Reference::Uhf, mol, prep, op, &hf).map_err(|e| {
        FerricError::General(format!(
            "the analytic UKS Hessian does not support this system (the UHF analytic Hessian's \
             check: {e}); use the finite-difference Hessian"
        ))
    })
}

/// `max_s max |F^s_rebuilt − F^s_scf|` (Hartree): the UKS Fock matrices rebuilt
/// at the SCF's spin densities with the J, K and polarized XC this Hessian
/// differentiates. A Hartree-Fock result, another functional's result, or one
/// converged with an external potential `config` lacks would otherwise give a
/// silently wrong Hessian.
#[doc(hidden)]
pub fn uks_fock_residual(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    uks: &ScfResult,
    grid: &FixedGridKs,
) -> Result<f64, FerricError> {
    use crate::fock::{JBuilder, KBuilder};
    let (da, db) = uhf_spin_densities(uks)?;
    let fb = uks
        .fock_beta
        .as_ref()
        .ok_or_else(|| refusal(Reference::Uhf, "a UKS result without a β Fock matrix"))?;
    let exx = grid.exact_exchange();
    let route = crate::gradient::active_df_route(uks);
    let n = da.nrows();
    let build = |d: &Array2<f64>| -> Result<(Array2<f64>, Array2<f64>), FerricError> {
        let mut j = Array2::<f64>::zeros((n, n));
        let mut k = Array2::<f64>::zeros((n, n));
        crate::rhf::build_jk(ctx, prep, bounds, 1e-14, d, &mut j, &mut k)?;
        Ok((j, k))
    };
    let total = da + db;
    let mut j = build(&total)?.0;
    if let Some(r) = route.filter(|r| r.j_aux.is_some()) {
        j.fill(0.0);
        df_j_builder(mol, prep, r)?.build(&total, &mut j)?;
    }
    let mut ks = [Array2::<f64>::zeros((n, n)), Array2::<f64>::zeros((n, n))];
    if exx != 0.0 {
        let fitted_k = route.filter(|r| r.k_aux.is_some());
        let mut dfk = fitted_k.map(|r| df_k_builder(mol, prep, r)).transpose()?;
        for (k, d) in ks.iter_mut().zip([da, db]) {
            match dfk.as_mut() {
                Some(df) => {
                    df.build(d, k)?;
                }
                None => *k = build(d)?.1,
            }
        }
    }
    let h =
        ferric_integrals::oneelectron::hcore_ecp_with_external(prep, mol, prep.basis_set(), None)?;
    let (_exc, v) = grid
        .polarized()?
        .energy_vxc(&XcGeometry::new(mol, prep)?, [da, db])?;
    let mut worst = 0.0f64;
    for (s, f_scf) in [&uks.fock_alpha, fb].into_iter().enumerate() {
        let f = &h + &j + &v[s] - &(exx * &ks[s]);
        worst = worst.max(max_abs(&(f - f_scf)));
    }
    Ok(worst)
}

/// The analytic UKS Hessian on the production (moving) grid, (3N, 3N),
/// Hartree/Bohr². `uks` is a converged [`crate::uhf::solve_uhf`] result with
/// `config.xc` set (`config` the configuration it ran with). See the module
/// doc for the energy, scope and refusals.
pub fn uks_moving_grid_hessian(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    uks: &ScfResult,
    config: &RhfConfig,
) -> Result<Array2<f64>, FerricError> {
    Ok(uks_moving_grid_hessian_parts(ctx, mol, prep, op, bounds, uks, config)?.total())
}

/// [`uks_moving_grid_hessian`] with the per-term breakdown (the XC second
/// derivative is reported in `two_electron`).
pub fn uks_moving_grid_hessian_parts(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    uks: &ScfResult,
    config: &RhfConfig,
) -> Result<UhfHessianParts, FerricError> {
    uks_refusal(mol, prep, uks, config)?;
    let name = config.xc.as_deref().unwrap_or_default();
    let cfg = config
        .dft_grid
        .clone()
        .unwrap_or_else(ferric_dft::grid::AtomicGridConfig::production);
    let grid = FixedGridKs::new(mol, name, &cfg)?;
    uks_hessian_parts_with(
        ctx,
        mol,
        prep,
        op,
        bounds,
        uks,
        config,
        &grid,
        &KsFixedGridMutants::default(),
        GridMode::Moving,
    )
}

/// [`uks_moving_grid_hessian_parts`] on a caller-built grid (built by
/// [`FixedGridKs::new`] at this geometry), in either grid mode, with
/// deliberately broken constructions (mutation tests). `GridMode::Fixed` is
/// the Hessian of the energy on a grid frozen in space (no Becke weight
/// response; library anchors only).
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn uks_hessian_parts_with(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    uks: &ScfResult,
    config: &RhfConfig,
    grid: &FixedGridKs,
    mutants: &KsFixedGridMutants,
    mode: GridMode,
) -> Result<UhfHessianParts, FerricError> {
    uks_refusal(mol, prep, uks, config)?;
    // Opt-in stage timing (`FERRIC_TIMING=1`), whole calls only.
    let mut phases = ferric_core::timing::PhaseAccumulator::new().report_on_drop("UKS Hessian");
    let exx = grid.exact_exchange();
    let hf = checklist_config(uks, config, exx)?;
    check_supported_except_functional(Reference::Uhf, mol, prep, op, bounds, uks, &hf)?;
    let t = phases.begin("Fock consistency check");
    let diff = uks_fock_residual(ctx, mol, prep, bounds, uks, grid)?;
    phases.stop(t);
    if diff > FOCK_CONSISTENCY_TOL {
        return Err(FerricError::General(format!(
            "the SCF result is not a converged UKS solution (with the Coulomb and exchange \
             matrices its route states) of the molecule, basis and functional passed (the Fock \
             matrices rebuilt at its spin densities differ from the stored ones by {diff:.2e} \
             Ha > {FOCK_CONSISTENCY_TOL:.0e}): a Hartree-Fock or other-functional result, a \
             missing external potential, or an unconverged SCF"
        )));
    }

    let natoms = mol.atoms.len();
    let nocc = uhf_occupations(mol);
    let (da, db) = uhf_spin_densities(uks)?;
    let d = da + db;
    let w = build_energy_weighted_density_uhf(uks, prep, nocc[0], nocc[1])?;
    let nuclear = nuclear_term(mol, &hf);
    let t = phases.begin("1e + overlap skeleton");
    let one_electron = skeleton_hess_1e(prep, natoms, &[], &d)?;
    let overlap = skeleton_hess_overlap(prep, natoms, &w)?;
    phases.stop(t);
    let t = phases.begin("2e skeleton + J/K first-order");
    let mut fitted = open_shell_two_electron(
        mol,
        prep,
        op,
        bounds,
        uks,
        OpenShellDensities {
            alpha: da,
            beta: db,
            total: &d,
            nocc,
            k_skeleton: exx * mutants.skeleton_exx,
            k_fock: exx * mutants.fock_exx,
        },
        &hf,
        || uks_occupied_factors(uks, nocc, [da, db]),
        &mutants.rij,
        &mutants.rik,
        &UhfDfMutants::default(),
    )?;
    phases.stop(t);
    let geo = XcGeometry::new(mol, prep)?;
    let pol: PolarizedKs<'_> = grid.polarized()?;
    let t = phases.begin("XC skeleton (2nd derivative)");
    fitted.two_electron += &pol.skeleton_hessian_for(mode, &geo, [da, db], &mutants.xc)?;
    phases.stop(t);
    let t = phases.begin("XC first-order Fock");
    let vx = pol.first_order_fock_for(mode, &geo, [da, db], &mutants.xc)?;
    phases.stop(t);
    for (f1, v1) in fitted.first.f1.iter_mut().zip(&vx) {
        for (f, v) in f1.iter_mut().zip(v1) {
            *f += v;
        }
    }
    let kernel_mutants = XcMutants {
        gradient_terms: mutants.response_gradient,
        spin: mutants.xc.spin,
        ..XcMutants::default()
    };
    let t = phases.begin("XC response kernel build");
    let kernel = pol.response(&geo, [da, db], mutants.response_fxc, &kernel_mutants)?;
    phases.stop(t);
    let kernel_fn = |d1: [&Array2<f64>; 2]| kernel.apply(d1);
    let g = TwoElectronBuilder::new(ctx, prep, bounds)?
        .fitted(fitted.df_j.take(), fitted.df_k.take(), false)
        .with_uks_xc_response(&kernel_fn, exx * mutants.response_exx);
    let t = phases.begin("coupled UKS CPHF");
    let resp = ucphf_response_with(&g, uks, nocc, &fitted.first)?;
    phases.stop(t);
    g.report_phases("UKS CPHF G builds");
    report_cphf_iterations("UKS", &resp);
    Ok(parts_from_terms(
        [nuclear, one_electron, overlap, fitted.two_electron],
        resp,
    ))
}

/// The occupied factors `X_σ = C_σ,occ` of a UKS result (the RI-K channels).
fn uks_occupied_factors(
    uks: &ScfResult,
    nocc: [usize; 2],
    dens: [&Array2<f64>; 2],
) -> Result<[Array2<f64>; 2], FerricError> {
    let cs = [uks.mos_a(), uks.mos_b()];
    let mut out = [Array2::<f64>::zeros((0, 0)), Array2::<f64>::zeros((0, 0))];
    for sp in 0..2 {
        let x = cs[sp].slice(s![.., ..nocc[sp]]).to_owned();
        if max_abs(&(x.dot(&x.t()) - dens[sp])) > OCC_CONSISTENCY_TOL {
            return Err(refusal(
                Reference::Uhf,
                "a UKS result whose spin density is not built from its first nocc MOs (RI-K \
                 needs the occupied factor)",
            ));
        }
        out[sp] = x;
    }
    Ok(out)
}

/// The UKS CPKS two-electron-plus-XC operator `v_s[D1^α, D1^β]` the Hessian
/// uses (`J[D1^α + D1^β] − a K[D1^s] + V1^s_xc`, the SCF's fitted J/K where it
/// fitted them), for the matvec anchors. NOT public API.
#[doc(hidden)]
pub fn uks_response_operator(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    uks: &ScfResult,
    grid: &FixedGridKs,
    mutants: &XcMutants,
    d1: [&Array2<f64>; 2],
) -> Result<[Array2<f64>; 2], FerricError> {
    let (da, db) = uhf_spin_densities(uks)?;
    let geo = XcGeometry::new(mol, prep)?;
    let pol = grid.polarized()?;
    let kernel = pol.response(&geo, [da, db], 1.0, mutants)?;
    let kernel_fn = |d: [&Array2<f64>; 2]| kernel.apply(d);
    let route = crate::gradient::active_df_route(uks);
    let df_j = route
        .filter(|r| r.j_aux.is_some())
        .map(|r| df_j_builder(mol, prep, r))
        .transpose()?;
    let df_k = route
        .filter(|r| r.k_aux.is_some())
        .map(|r| df_k_builder(mol, prep, r))
        .transpose()?;
    let g = TwoElectronBuilder::new(ctx, prep, bounds)?
        .fitted(df_j, df_k, false)
        .with_uks_xc_response(&kernel_fn, grid.exact_exchange());
    g.g_uhf(&[d1[0].clone(), d1[1].clone()])
}

/// [`analytic_frequencies`] from [`uks_moving_grid_hessian`] at a converged
/// UKS SCF; `hessian_source` is
/// [`crate::frequencies::HessianSource::AnalyticMovingGrid`].
pub fn analytic_frequencies_uks_moving_grid(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    uks: &ScfResult,
    config: &RhfConfig,
) -> Result<crate::frequencies::FrequencyResult, FerricError> {
    let masses = crate::frequencies::atom_masses(mol)?;
    let parts = uks_moving_grid_hessian_parts(ctx, mol, prep, op, bounds, uks, config)?;
    let mut result = frequencies_from_parts(mol, &parts, &masses, uks.energy)?;
    result.hessian_source = crate::frequencies::HessianSource::AnalyticMovingGrid;
    Ok(result)
}
