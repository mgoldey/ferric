//! RI-MP2-driven geometry optimization.
//!
//! The search loop is the SCF crate's
//! ([`ferric_scf::optimize::optimize_geometry_with_energy_gradient`]), in
//! Cartesian or redundant internal coordinates per
//! `OptimizeConfig::coord_system`; this module supplies the surfaces:
//! `total_rimp2_gradient_on` (the analytic RI-MP2 gradient),
//! `total_u_rimp2_gradient_on` (its unrestricted sibling on a UHF reference),
//! `total_scs_mp2_gradient_on`, `total_att_rimp2_gradient_on` (attenuated
//! RI-MP2) and its unrestricted sibling `total_u_att_rimp2_gradient_on`, the
//! terfc RI-MP2 surfaces (`total_terfc_rimp2_gradient_on`,
//! `total_u_terfc_rimp2_gradient_on`), SCS-MP2(2terfc)
//! (`scs_2terfc_gradient`), and the OO-RI-MP2 zero-response gradient.

use crate::gradient::{exact_scf_jk, total_rimp2_gradient_on, total_scs_mp2_gradient_on};
use crate::rimp2::RiMp2Config;
use ferric_core::basis::BasisSet;
use ferric_core::external_potential::ExternalPotential;
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::operator::Operator;
use ferric_scf::optimize::{optimize_geometry_with_energy_gradient, OptimizeConfig};
use ferric_scf::scf_jk::ScfJk;
use ferric_scf::warm_start::ScfWarmStart;
use ndarray::Array2;

/// Result of an RI-MP2 geometry optimization.
#[derive(Debug, Clone)]
#[must_use]
pub struct RiMp2OptimizeResult {
    /// Final geometry (coordinates in Bohr).
    pub mol: Molecule,
    /// RHF + RI-MP2 total energy at the final geometry (Hartree).
    pub energy: f64,
    /// BFGS steps taken.
    pub steps: usize,
    /// Whether the energy-change, max-gradient and RMS-gradient criteria of
    /// the `OptimizeConfig` were all met within `max_steps`.
    pub converged: bool,
    /// Per-evaluation SCF warm-start records, in order (see
    /// `ferric_scf::warm_start`; every record is cold `Disabled` when
    /// `OptimizeConfig::scf_warm_start` is off).
    pub scf_warm_start: Vec<ferric_scf::warm_start::WarmStartRecord>,
    /// Energy at every geometry the optimizer evaluated, in order (the
    /// starting geometry first).
    pub energy_trace: Vec<f64>,
    /// The search that ran (see `ferric_scf::optimize::SearchSettings`).
    pub search: ferric_scf::optimize::SearchSettings,
}

/// Optimize a molecular geometry on the RI-MP2 PES using
/// `total_rimp2_gradient` (analytical Z-vector/relaxed-density RI-MP2
/// gradient, RHF + correlation together).
pub fn optimize_geometry_rimp2(
    mol: &Molecule,
    obs_basis: &BasisSet,
    aux_basis: &BasisSet,
    op: Operator,
    mp2_config: &RiMp2Config,
    opt_config: &OptimizeConfig,
    ext: Option<&ExternalPotential>,
) -> Result<RiMp2OptimizeResult, FerricError> {
    optimize_geometry_rimp2_on(
        mol,
        obs_basis,
        aux_basis,
        op,
        mp2_config,
        opt_config,
        ext,
        &exact_scf_jk(),
    )
}

/// [`optimize_geometry_rimp2`] with the reference SCF on the J/K route
/// `scf_jk` (`"exact"` or `"rijk"`; see
/// [`crate::gradient::total_rimp2_gradient_on`]).
#[allow(clippy::too_many_arguments)]
pub fn optimize_geometry_rimp2_on(
    mol: &Molecule,
    obs_basis: &BasisSet,
    aux_basis: &BasisSet,
    op: Operator,
    mp2_config: &RiMp2Config,
    opt_config: &OptimizeConfig,
    ext: Option<&ExternalPotential>,
    scf_jk: &ScfJk,
) -> Result<RiMp2OptimizeResult, FerricError> {
    bfgs_optimize(mol, opt_config, |m, warm| {
        total_rimp2_gradient_on(m, obs_basis, aux_basis, op, mp2_config, ext, scf_jk, warm)
    })
}

/// Optimize an open-shell geometry on the unrestricted RI-MP2 surface
/// `E_UHF + E_UMP2`, with the analytic gradient
/// [`crate::u_rimp2_gradient::total_u_rimp2_gradient_on`] (UHF reference of
/// the molecule's charge and multiplicity on `scf_jk`, `"exact"` or
/// `"rijk"`). The optimizer is the one of [`optimize_geometry_rimp2_on`];
/// [`RiMp2OptimizeResult::energy`] is the UHF + U-MP2 total.
#[allow(clippy::too_many_arguments)]
pub fn optimize_geometry_u_rimp2_on(
    mol: &Molecule,
    obs_basis: &BasisSet,
    aux_basis: &BasisSet,
    op: Operator,
    mp2_config: &RiMp2Config,
    opt_config: &OptimizeConfig,
    ext: Option<&ExternalPotential>,
    scf_jk: &ScfJk,
) -> Result<RiMp2OptimizeResult, FerricError> {
    bfgs_optimize(mol, opt_config, |m, warm| {
        crate::u_rimp2_gradient::total_u_rimp2_gradient_on(
            m, obs_basis, aux_basis, op, mp2_config, ext, scf_jk, warm,
        )
    })
}

/// Optimize a closed-shell geometry on the SCS-MP2 surface
/// `E_HF + c_os E_os + c_ss E_ss` (`scs_config`), with the analytic
/// gradient [`crate::gradient::total_scs_mp2_gradient_on`] and the reference
/// SCF on `scf_jk` (`"exact"` or `"rijk"`). The optimizer is the one of
/// [`optimize_geometry_rimp2_on`].
pub fn optimize_geometry_scs_mp2_on(
    mol: &Molecule,
    obs_basis: &BasisSet,
    aux_basis: &BasisSet,
    scs_config: &crate::scs::ScsMp2Config,
    opt_config: &OptimizeConfig,
    ext: Option<&ExternalPotential>,
    scf_jk: &ScfJk,
) -> Result<RiMp2OptimizeResult, FerricError> {
    bfgs_optimize(mol, opt_config, |m, warm| {
        total_scs_mp2_gradient_on(m, obs_basis, aux_basis, scs_config, ext, scf_jk, warm)
    })
}

/// Optimize a closed-shell geometry on the attenuated RI-MP2 surface
/// `E_HF + s · E_c[erfc(ωr)/r]` (`att_config`, Coulomb RHF reference on
/// `scf_jk`, `"exact"` or `"rijk"`), with the analytic gradient
/// [`crate::att_rimp2_gradient::total_att_rimp2_gradient_on`]. The optimizer
/// is the one of [`optimize_geometry_rimp2_on`].
pub fn optimize_geometry_att_rimp2_on(
    mol: &Molecule,
    obs_basis: &BasisSet,
    aux_basis: &BasisSet,
    att_config: &crate::attenuated::AttenuatedMp2Config,
    opt_config: &OptimizeConfig,
    ext: Option<&ExternalPotential>,
    scf_jk: &ScfJk,
) -> Result<RiMp2OptimizeResult, FerricError> {
    bfgs_optimize(mol, opt_config, |m, warm| {
        crate::att_rimp2_gradient::total_att_rimp2_gradient_on(
            m, obs_basis, aux_basis, att_config, ext, scf_jk, warm,
        )
    })
}

/// Optimize a closed-shell geometry on the terfc RI-MP2 surface
/// `E_HF + E_c[terfc(r; r0, ω)/r]` (`op` is [`Operator::terfc`] or
/// [`Operator::terfc_with_omega`], `ri` the frozen core and memory budget;
/// Coulomb RHF reference on `scf_jk`, `"exact"` or `"rijk"`), with the
/// analytic gradient
/// [`crate::att_rimp2_gradient::total_terfc_rimp2_gradient_on`]. The
/// optimizer is the one of [`optimize_geometry_rimp2_on`].
#[allow(clippy::too_many_arguments)]
pub fn optimize_geometry_terfc_rimp2_on(
    mol: &Molecule,
    obs_basis: &BasisSet,
    aux_basis: &BasisSet,
    op: Operator,
    ri: &RiMp2Config,
    opt_config: &OptimizeConfig,
    ext: Option<&ExternalPotential>,
    scf_jk: &ScfJk,
) -> Result<RiMp2OptimizeResult, FerricError> {
    bfgs_optimize(mol, opt_config, |m, warm| {
        crate::att_rimp2_gradient::total_terfc_rimp2_gradient_on(
            m, obs_basis, aux_basis, op, ri, ext, scf_jk, warm,
        )
    })
}

/// Optimize an open-shell geometry on the attenuated RI-MP2 surface
/// `E_UHF + s · E_c[erfc(ωr)/r]` (`att_config`, Coulomb UHF of the molecule's
/// charge and multiplicity on `scf_jk`, `"exact"` or `"rijk"`), with the
/// analytic gradient
/// [`crate::att_rimp2_gradient::total_u_att_rimp2_gradient_on`]. The
/// optimizer is the one of [`optimize_geometry_rimp2_on`].
pub fn optimize_geometry_u_att_rimp2_on(
    mol: &Molecule,
    obs_basis: &BasisSet,
    aux_basis: &BasisSet,
    att_config: &crate::attenuated::AttenuatedMp2Config,
    opt_config: &OptimizeConfig,
    ext: Option<&ExternalPotential>,
    scf_jk: &ScfJk,
) -> Result<RiMp2OptimizeResult, FerricError> {
    bfgs_optimize(mol, opt_config, |m, warm| {
        crate::att_rimp2_gradient::total_u_att_rimp2_gradient_on(
            m, obs_basis, aux_basis, att_config, ext, scf_jk, warm,
        )
    })
}

/// Optimize an open-shell geometry on the terfc RI-MP2 surface
/// `E_UHF + E_c[terfc(r; r0, ω)/r]` (`op`, `ri` as in
/// [`optimize_geometry_terfc_rimp2_on`]; Coulomb UHF of the molecule's charge
/// and multiplicity on `scf_jk`), with the analytic gradient
/// [`crate::att_rimp2_gradient::total_u_terfc_rimp2_gradient_on`].
#[allow(clippy::too_many_arguments)]
pub fn optimize_geometry_u_terfc_rimp2_on(
    mol: &Molecule,
    obs_basis: &BasisSet,
    aux_basis: &BasisSet,
    op: Operator,
    ri: &RiMp2Config,
    opt_config: &OptimizeConfig,
    ext: Option<&ExternalPotential>,
    scf_jk: &ScfJk,
) -> Result<RiMp2OptimizeResult, FerricError> {
    bfgs_optimize(mol, opt_config, |m, warm| {
        crate::att_rimp2_gradient::total_u_terfc_rimp2_gradient_on(
            m, obs_basis, aux_basis, op, ri, ext, scf_jk, warm,
        )
    })
}

/// Optimize a geometry on the SCS-MP2(2terfc) surface of `config`
/// ([`crate::scs::scs_mp2_2terfc`]) with its analytic gradient: a singlet
/// on a Coulomb RHF
/// ([`crate::scs_2terfc_gradient::total_scs_mp2_2terfc_gradient_on`]), a
/// higher multiplicity on a Coulomb UHF
/// ([`crate::scs_2terfc_gradient::total_u_scs_mp2_2terfc_gradient_on`]),
/// either on `scf_jk` (`"exact"` or `"rijk"`).
pub fn optimize_geometry_scs_mp2_2terfc_on(
    mol: &Molecule,
    obs_basis: &BasisSet,
    aux_basis: &BasisSet,
    config: &crate::scs::ScsMp2TerfcConfig,
    opt_config: &OptimizeConfig,
    ext: Option<&ExternalPotential>,
    scf_jk: &ScfJk,
) -> Result<RiMp2OptimizeResult, FerricError> {
    let surface = if mol.multiplicity > 1 {
        crate::scs_2terfc_gradient::total_u_scs_mp2_2terfc_gradient_on
    } else {
        crate::scs_2terfc_gradient::total_scs_mp2_2terfc_gradient_on
    };
    bfgs_optimize(mol, opt_config, |m, warm| {
        surface(m, obs_basis, aux_basis, config, ext, scf_jk, warm)
    })
}

/// The shared optimizer loop
/// ([`ferric_scf::optimize::optimize_geometry_with_energy_gradient`]) on the
/// energy and gradient `eval_on` returns, each geometry's RHF warm-started
/// from the previous one's through the carrier `eval_on` receives
/// (`opt_config.scf_warm_start`, see `ferric_scf::warm_start`).
fn bfgs_optimize(
    mol: &Molecule,
    opt_config: &OptimizeConfig,
    mut eval_on: impl FnMut(&Molecule, &mut ScfWarmStart) -> Result<(f64, Array2<f64>), FerricError>,
) -> Result<RiMp2OptimizeResult, FerricError> {
    let mut warm = ScfWarmStart::new(opt_config.scf_warm_start);
    let out = optimize_geometry_with_energy_gradient(mol, opt_config, |m| eval_on(m, &mut warm))?;
    if let Some(line) = warm.summary() {
        println!("{line}");
    }
    Ok(RiMp2OptimizeResult {
        mol: out.mol,
        energy: out.energy,
        steps: out.steps,
        converged: out.converged,
        scf_warm_start: warm.records().to_vec(),
        energy_trace: out.energy_trace,
        search: out.search,
    })
}

/// The certificate's `gradient` line of an RI-MP2 optimization whose
/// reference SCF is on `scf_jk` (four-centre, or RI-JK on its aux).
pub fn rimp2_gradient_label(scf_jk: &ScfJk) -> String {
    match scf_jk.aux.as_deref() {
        Some(aux) => format!("analytic (Z-vector), RI-JK reference ({aux})"),
        None => "analytic (Z-vector)".to_string(),
    }
}

/// The certificate's `gradient` line of an OO-RI-MP2 optimization on the
/// reference J/K route of `rhf_config`: four-centre, or RI-JK on one
/// JK-fitting basis.
pub fn oo_rimp2_gradient_label(rhf_config: &ferric_scf::rhf::RhfConfig) -> String {
    let base = "analytic (zero-response: orbital-optimized, no Z-vector)";
    let route = ferric_scf::scf_jk::effective_jk(rhf_config);
    match (route.mode, route.j_aux) {
        ("rijk", Some(aux)) => format!("{base}, RI-JK reference ({aux})"),
        _ => base.to_string(),
    }
}

/// Optimize a closed-shell geometry on the OO-RI-MP2 surface.
///
/// At every geometry: the RHF reference on `rhf_config` (warm-started per
/// `opt_config.scf_warm_start`), the orbital optimisation
/// ([`crate::oo_rimp2::oo_ri_mp2`] with `oo_config`) from it, and the
/// zero-response analytic gradient
/// ([`crate::oo_rimp2_gradient::oo_ri_mp2_gradient`]) at the converged
/// orbitals; the external potential of `rhf_config` enters the energy and the
/// gradient. The BFGS driver is
/// [`ferric_scf::optimize::optimize_geometry_with_energy_gradient`], the one
/// the SCF optimizers use.
///
/// The reference J/K route of `rhf_config` is the route of the whole surface:
/// four-centre, or RI-JK (`df_j_aux == df_k_aux`, the `scf_jk = "rijk"`
/// spelling), whose fitted J/K the orbital steps and the gradient reuse. The
/// RI-J tensor is stored in f64 on both (a geometry task,
/// [`ferric_scf::rhf::RhfConfig::for_geometry_task`]).
///
/// Refused before any integral: `oo_config.frozen_core > 0` (the frozen
/// core-active and core-virtual rotations are not optimised, so the
/// zero-response gradient is not the derivative of that energy), any other
/// fitted reference (RI-J with four-centre or COSX exchange, or RI-J and
/// RI-K on different bases: OO-MP2 has no route for them), and a PCM or
/// COSMO reaction field (OO-MP2 does not include one). An orbital
/// optimisation that does not converge at some geometry is an error: the
/// zero-response gradient holds only at a stationary point.
#[allow(clippy::too_many_arguments)]
pub fn optimize_geometry_oo_rimp2(
    mol: &Molecule,
    obs_basis: &BasisSet,
    aux_basis: &BasisSet,
    op: Operator,
    rhf_config: &ferric_scf::rhf::RhfConfig,
    oo_config: &crate::oo_rimp2::OoRiMp2Config,
    opt_config: &OptimizeConfig,
) -> Result<ferric_scf::optimize::OptimizeResult, FerricError> {
    refuse_oo_rimp2_optimize(rhf_config, oo_config)?;
    ferric_scf::optimize::refuse_internal_coordinates_in_fixed_frame(
        opt_config.coord_system,
        rhf_config,
    )?;
    let rhf_config = rhf_config.for_geometry_task();
    let mut warm = ScfWarmStart::new(opt_config.scf_warm_start);
    let out = ferric_scf::optimize::optimize_geometry_with_energy_gradient(mol, opt_config, |m| {
        oo_rimp2_energy_gradient(
            m,
            obs_basis,
            aux_basis,
            op,
            &rhf_config,
            oo_config,
            &mut warm,
        )
    });
    if let Some(line) = warm.summary() {
        println!("{line}");
    }
    out
}

/// `(E_OO-RI-MP2, dE/dR)` at `m`: the RHF reference on `rhf_config`
/// (already a geometry task, [`ferric_scf::rhf::RhfConfig::for_geometry_task`]),
/// the orbital optimisation, and the zero-response analytic gradient. The
/// surface of [`optimize_geometry_oo_rimp2`] and of the CLI frequency run;
/// call [`check_oo_rimp2_surface`] first.
pub fn oo_rimp2_energy_gradient(
    m: &Molecule,
    obs_basis: &BasisSet,
    aux_basis: &BasisSet,
    op: Operator,
    rhf_config: &ferric_scf::rhf::RhfConfig,
    oo_config: &crate::oo_rimp2::OoRiMp2Config,
    warm: &mut ScfWarmStart,
) -> Result<(f64, Array2<f64>), FerricError> {
    let ext = rhf_config.external_potential.as_ref();
    let ctx = ferric_core::parallel::ParallelContext::default();
    let obs = ferric_integrals::basis_bridge::PreparedBasis::new(m, obs_basis)?;
    let dfbs = ferric_integrals::basis_bridge::PreparedBasis::new(m, aux_basis)?;
    let route = ferric_scf::scf_jk::effective_jk(rhf_config);
    if let ("rijk", Some(aux)) = (route.mode, route.j_aux.as_deref()) {
        // The orbital steps and the gradient keep the RI-J/RI-K tensors
        // of the reference: check the peak before any integral.
        let shape = crate::gradient_memory::FittedGradientShape::new(
            m,
            &obs,
            &dfbs,
            aux,
            oo_config.frozen_core,
        )?;
        crate::gradient_memory::preflight_fitted_gradient(
            shape,
            ferric_core::memory::resolve_budget_bytes(oo_config.memory_budget_bytes),
        )?;
    }
    let bounds = ferric_scf::screening::SchwarzBounds::compute_for_screening(
        op,
        &obs,
        rhf_config.screening,
    )?;
    let rhf = ferric_scf::warm_start::solve_rhf_warm(&ctx, m, &obs, op, &bounds, rhf_config, warm)?;
    if !rhf.converged {
        return Err(FerricError::ScfConvergence {
            iterations: rhf.iterations,
            last_energy: rhf.energy,
        });
    }
    let oo = crate::oo_rimp2::oo_ri_mp2(m, &obs, &dfbs, op, &bounds, &rhf, oo_config, ext)?;
    if !oo.converged {
        return Err(FerricError::General(format!(
            "OO-RI-MP2 optimize: the orbital optimisation did not converge at this \
             geometry ({} iterations, orbital gradient norm {:.2e}); the zero-response \
             gradient holds only at a stationary point",
            oo.iterations, oo.grad_norm
        )));
    }
    let g = crate::oo_rimp2_gradient::oo_ri_mp2_gradient(m, &obs, &dfbs, op, &bounds, &oo, 0, ext)?;
    Ok((oo.total_energy, g))
}

/// What [`optimize_geometry_oo_rimp2`] refuses before any integral.
pub fn check_oo_rimp2_surface(
    rhf_config: &ferric_scf::rhf::RhfConfig,
    oo_config: &crate::oo_rimp2::OoRiMp2Config,
) -> Result<(), FerricError> {
    refuse_oo_rimp2_optimize(rhf_config, oo_config)
}

fn refuse_oo_rimp2_optimize(
    rhf_config: &ferric_scf::rhf::RhfConfig,
    oo_config: &crate::oo_rimp2::OoRiMp2Config,
) -> Result<(), FerricError> {
    let refuse = |why: String| Err(FerricError::General(format!("OO-RI-MP2 optimize: {why}")));
    if oo_config.frozen_core > 0 {
        return refuse(format!(
            "frozen_core = {} is not supported: the core-active and core-virtual rotations \
             are not optimised, so the zero-response gradient is not the derivative of the \
             frozen-core energy; use frozen_core = 0",
            oo_config.frozen_core
        ));
    }
    let route = ferric_scf::scf_jk::effective_jk(rhf_config);
    if !matches!(route.mode, "exact" | "rijk") {
        return refuse(format!(
            "the reference SCF J/K route is \"{}\"; {}",
            route.mode,
            ferric_scf::scf_jk::OO_BUILDS_EXACT_OR_RIJK
        ));
    }
    if rhf_config.pcm.is_some() || rhf_config.cosmo.is_some() {
        return refuse(
            "a PCM/COSMO reaction field is not supported: OO-RI-MP2 has no solvation term"
                .to_string(),
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::basis;

    /// `optimize_geometry_oo_rimp2` refuses, before any integral, the setups
    /// its zero-response gradient does not differentiate; the default
    /// (four-centre) and the RI-JK reference pass the same check.
    #[test]
    fn oo_rimp2_optimize_refuses_what_it_does_not_differentiate() {
        use crate::oo_rimp2::OoRiMp2Config;
        use ferric_scf::rhf::RhfConfig;
        let refused = |rhf: &RhfConfig, oo: &OoRiMp2Config, needle: &str| {
            let r = refuse_oo_rimp2_optimize(rhf, oo);
            assert!(
                matches!(&r, Err(e) if e.to_string().contains(needle)),
                "{needle}: {r:?}"
            );
        };
        let (rhf, oo) = (RhfConfig::default(), OoRiMp2Config::default());
        assert!(refuse_oo_rimp2_optimize(&rhf, &oo).is_ok());
        let frozen = OoRiMp2Config {
            frozen_core: 1,
            ..OoRiMp2Config::default()
        };
        refused(&rhf, &frozen, "frozen_core = 1");
        let rijk = RhfConfig {
            df_j_aux: Some("def2-universal-jkfit".into()),
            df_k_aux: Some("def2-universal-jkfit".into()),
            ..RhfConfig::default()
        };
        assert!(refuse_oo_rimp2_optimize(&rijk, &oo).is_ok());
        assert!(oo_rimp2_gradient_label(&rijk).ends_with("RI-JK reference (def2-universal-jkfit)"));
        let rijcosx = RhfConfig {
            df_j_aux: Some("def2-universal-jkfit".into()),
            k_builder: Some(ferric_scf::options::KBuilderKind::Cosx),
            ..RhfConfig::default()
        };
        refused(&rijcosx, &oo, "\"rijcosx\"");
        let split = RhfConfig {
            df_j_aux: Some("def2-universal-jkfit".into()),
            df_k_aux: Some("cc-pvtz-jkfit".into()),
            ..RhfConfig::default()
        };
        refused(&split, &oo, "\"mixed\"");
        let cosmo = RhfConfig {
            cosmo: Some(ferric_scf::cosmo::CosmoConfig::default()),
            ..RhfConfig::default()
        };
        refused(&cosmo, &oo, "PCM/COSMO");
    }

    #[test]
    fn test_optimize_h2_sto3g_rimp2() {
        // Start from a stretched bond: 1.0 Angstrom.
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 1.0\n", 0, 1).unwrap();
        let op = Operator::coulomb();
        let obs_basis = basis::bundled("sto-3g").unwrap();
        let aux_basis = basis::bundled("cc-pvdz-ri").unwrap();
        let mp2_config = RiMp2Config {
            frozen_core: 0,
            memory_budget_bytes: None,
            ..Default::default()
        };
        let opt_config = OptimizeConfig {
            trust_radius: 0.1,
            ..Default::default()
        };

        let result = optimize_geometry_rimp2(
            &mol,
            &obs_basis,
            &aux_basis,
            op,
            &mp2_config,
            &opt_config,
            None,
        )
        .unwrap();

        assert!(result.converged);
        let dist = (result.mol.atoms[0].zpos - result.mol.atoms[1].zpos).abs();
        eprintln!("H2/STO-3G RI-MP2 optimized distance: {:.6} Bohr", dist);
        // RHF/STO-3G optimum is ~1.346 Bohr; RI-MP2 correlation should shift it
        // modestly (H2/STO-3G has only one virtual so MP2 correlation is small,
        // but the geometry must still be physically reasonable, not NaN/wild).
        assert!(dist.is_finite());
        assert!(
            (0.8..2.5).contains(&dist),
            "dist = {dist}, expected a reasonable H2 bond length"
        );
    }

    /// Warm vs cold RI-MP2 optimization: the same optimization within the
    /// optimizer's own tolerances (same step count, final energies within
    /// `e_conv`, final geometries within the optimizer's step resolution),
    /// warm start used after the first evaluation, and fewer SCF iterations.
    #[test]
    fn warm_start_rimp2_optimization_matches_cold() {
        let mol = Molecule::parse_xyz(
            "3\ndistorted water\nO 0 0 0\nH 0 0.9 0\nH 0 -0.3 0.85\n",
            0,
            1,
        )
        .unwrap();
        let obs = basis::bundled("sto-3g").unwrap();
        let aux = basis::bundled("cc-pvdz-ri").unwrap();
        let mp2 = RiMp2Config::default();
        let run = |warm: bool| {
            let opt = OptimizeConfig {
                scf_warm_start: warm,
                ..Default::default()
            };
            optimize_geometry_rimp2(&mol, &obs, &aux, Operator::coulomb(), &mp2, &opt, None)
                .unwrap()
        };
        let (w, c) = (run(true), run(false));
        let it = |r: &RiMp2OptimizeResult| {
            r.scf_warm_start
                .iter()
                .map(|x| x.iterations)
                .collect::<Vec<_>>()
        };
        let dx = w
            .mol
            .atoms
            .iter()
            .zip(&c.mol.atoms)
            .map(|(a, b)| {
                (a.x - b.x)
                    .abs()
                    .max((a.y - b.y).abs())
                    .max((a.zpos - b.zpos).abs())
            })
            .fold(0.0f64, f64::max);
        let de = (w.energy - c.energy).abs();
        eprintln!(
            "RI-MP2 optimize: steps warm {} cold {}; |dE| {de:.2e}; max|dx| {dx:.2e} Bohr; \
             SCF iterations warm {:?} cold {:?}",
            w.steps,
            c.steps,
            it(&w),
            it(&c)
        );
        let opt = OptimizeConfig::default();
        assert!(w.converged && c.converged);
        assert_eq!(w.steps, c.steps);
        assert!(de < opt.e_conv, "|dE| {de:e}");
        // Converged geometries agree to well inside one step of the size the
        // gradient threshold allows (|dx| ~ g_max / k, k ~ 0.5 Ha/Bohr^2).
        assert!(dx < opt.g_max_thresh, "max|dx| {dx:e}");
        assert_eq!(
            w.scf_warm_start[0].cold,
            Some(ferric_scf::warm_start::ColdReason::FirstGeometry)
        );
        assert!(w.scf_warm_start[1..].iter().all(|r| r.cold.is_none()));
        assert!(it(&w).iter().sum::<usize>() < it(&c).iter().sum::<usize>());
    }
}
