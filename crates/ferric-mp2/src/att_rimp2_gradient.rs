//! Analytic nuclear gradient of attenuated RI-MP2 (erfc), the energy of
//! [`crate::attenuated::attenuated_ri_mp2`]:
//!
//! ```text
//!   E = E_HF[1/r] + s · E_c[erfc(ωr)/r]
//! ```
//!
//! The reference is Coulomb RHF. Only the correlation uses `erfc(ωr)/r`: the
//! `(P|ia)` integrals AND the RI metric `(P|Q)` (the energy builds both with
//! the same operator, see `ri_mp2_spin_components`). So the gradient is the
//! relaxed RI-MP2 gradient with the operators split
//! (`book/src/15-correlated-gradients/02-rimp2-gradient.md#rimp2-gradient-attenuated-erfc-ri-mp2`):
//!
//! * erfc: the amplitudes and unrelaxed densities, `b_full` and the
//!   Lagrangian `Imat` of the Z-vector right-hand side, and the 3c/2c
//!   integral response (`d(P|erfc|μν)`, `d(P|erfc|Q)`);
//! * Coulomb: the Z-vector's orbital Hessian, the `veff` of the correlation
//!   density, and the four-centre (or RI-JK) derivative term, all of which
//!   come from the HF stationarity condition and the Coulomb Fock matrix.
//!
//! The scaling `s` enters as the spin scaling `c_os = c_ss = s`
//! ([`SpinScaling`](crate::rimp2::SpinScaling)). The QQR-3 screened energy (`screen_thresh`) is not
//! differentiated and is refused.
//!
//! The terfc-attenuated energy `E_HF[1/r] + E_c[terfc(r; r0, ω)/r]` (no
//! scaling; the `att_operator = "terfc"` energy of the CLI) has the same
//! gradient with terfc in place of erfc
//! (`book/src/15-correlated-gradients/02-rimp2-gradient.md#rimp2-gradient-attenuated-erfc-ri-mp2`):
//! [`terfc_rimp2_gradient_analytical`](crate::att_rimp2_gradient::terfc_rimp2_gradient_analytical),
//! [`total_terfc_rimp2_gradient_on`](crate::att_rimp2_gradient::total_terfc_rimp2_gradient_on).
//! The 3c/2c derivative integrals come from the terf table engine
//! (`book/src/04-operators/02-terf-table-engine.md#terf-table-engine-derivative-integrals`). On a
//! UHF reference the terfc energy `E_UHF + E_c[terfc]`
//! ([`crate::u_rimp2::u_ri_mp2`] on terfc) has the unrestricted gradient
//! with the same split
//! (`book/src/15-correlated-gradients/07-u-rimp2-gradient.md#u-rimp2-gradient-attenuated-erfc-variant`):
//! [`u_terfc_rimp2_gradient_analytical`](crate::att_rimp2_gradient::u_terfc_rimp2_gradient_analytical),
//! [`total_u_terfc_rimp2_gradient_on`](crate::att_rimp2_gradient::total_u_terfc_rimp2_gradient_on).
//!
//! On a UHF reference ([`u_att_rimp2_gradient_analytical`](crate::att_rimp2_gradient::u_att_rimp2_gradient_analytical),
//! [`total_u_att_rimp2_gradient_on`](crate::att_rimp2_gradient::total_u_att_rimp2_gradient_on)) the energy is
//! `E_UHF[1/r] + s · (E_αα + E_ββ + E_αβ)[erfc]`
//! (`book/src/10-mp2/04-attenuated-mp2.md#attenuated-mp2`) and the
//! gradient is the unrestricted RI-MP2 gradient of
//! [`crate::u_rimp2_gradient`] with the same split
//! (`book/src/15-correlated-gradients/07-u-rimp2-gradient.md#u-rimp2-gradient-attenuated-erfc-variant`):
//! erfc in
//! the per-spin amplitudes, `x^σ`, `Imat^σ` and the 3c/2c response; Coulomb
//! in the coupled UHF Z-vector, the `veff^σ` of the correlation densities and
//! the two-electron derivative. A ROHF reference is refused
//! ([`crate::u_rimp2_gradient::ROHF_REFUSAL`]).

use crate::attenuated::AttenuatedMp2Config;
use crate::gradient::{rimp2_gradient_analytical_timed, total_mp2_gradient_scaled_on};
use crate::rimp2::{RiMp2Config, SpinScaling};
use crate::u_rimp2_gradient::{solve_reference, u_correlated_gradient_timed, UCorrelation};
use crate::zvector::ZvectorControls;
use ferric_core::basis::BasisSet;
use ferric_core::external_potential::ExternalPotential;
use ferric_core::mol::Molecule;
use ferric_core::timing::PhaseAccumulator;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::{Operator, OperatorKind};
use ferric_scf::rhf::RhfConfig;
use ferric_scf::scf_jk::ScfJk;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::warm_start::ScfWarmStart;
use ferric_scf::ScfResult;
use ndarray::Array2;

/// The pieces of `config` the gradient reads, checked: the erfc operator,
/// the RI-MP2 configuration of the energy and the spin scaling `s, s`.
fn att_parts(
    config: &AttenuatedMp2Config,
) -> Result<(Operator, RiMp2Config, SpinScaling), FerricError> {
    if !(config.omega.is_finite() && config.omega > 0.0) {
        return Err(FerricError::General(format!(
            "attenuated RI-MP2 gradient: omega must be finite and > 0 Bohr⁻¹ (got {})",
            config.omega
        )));
    }
    if !config.scaling.is_finite() {
        return Err(FerricError::General(format!(
            "attenuated RI-MP2 gradient: scaling must be finite (got {})",
            config.scaling
        )));
    }
    if config.screen_thresh.is_some() {
        return Err(FerricError::General(
            "attenuated RI-MP2 gradient: screen_thresh is not supported (the QQR-3 screened \
             energy is not differentiated); leave it unset"
                .into(),
        ));
    }
    let ri = RiMp2Config {
        frozen_core: config.frozen_core,
        memory_budget_bytes: config.memory_budget_bytes,
        ..Default::default()
    };
    let scaling = SpinScaling {
        c_os: config.scaling,
        c_ss: config.scaling,
    };
    Ok((Operator::erfc(config.omega), ri, scaling))
}

/// An error unless `op` is the plain Coulomb operator: the attenuated
/// RI-MP2 energy is defined on a Coulomb HF reference.
fn require_coulomb_reference(op: Operator) -> Result<(), FerricError> {
    if op.is_composite() || op.kind != OperatorKind::Coulomb {
        return Err(FerricError::General(format!(
            "attenuated RI-MP2 gradient: the reference SCF must be Coulomb HF (its Schwarz \
             bounds are for {:?})",
            op.kind
        )));
    }
    Ok(())
}

/// `dE/dR`, `(natoms, 3)`, of `rhf.energy + s · E_c[erfc]` (the total
/// [`crate::attenuated::attenuated_ri_mp2`] reports for `config`) on the
/// converged Coulomb RHF `rhf`, whose J/K route `bounds` describes
/// (four-centre, or the RI-JK route `rhf` carries). `ext` adds the external
/// potential's Hellmann-Feynman and classical terms, as in
/// [`crate::gradient::rimp2_gradient_analytical`].
pub fn att_rimp2_gradient_analytical(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    bounds: &SchwarzBounds,
    rhf: &ScfResult,
    config: &AttenuatedMp2Config,
    ext: Option<&ExternalPotential>,
) -> Result<Array2<f64>, FerricError> {
    ferric_scf::x2c::refuse_relativistic(rhf.relativistic, "attenuated RI-MP2 analytic gradient")?;
    require_coulomb_reference(bounds.op)?;
    let (op, ri, scaling) = att_parts(config)?;
    let mut acc = PhaseAccumulator::new().report_on_drop("att-RI-MP2 gradient");
    rimp2_gradient_analytical_timed(
        mol,
        obs,
        dfbs,
        op,
        bounds,
        rhf,
        &ri,
        scaling,
        ext,
        ZvectorControls::DEFAULT,
        &mut acc,
    )
}

/// The total attenuated RI-MP2 energy (`E_HF + s · E_c[erfc]`, Coulomb RHF
/// solved here at `density_conv = 1e-9`) and its analytic gradient at one
/// geometry, the reference on `scf_jk` (`"exact"` or `"rijk"`, as in
/// [`crate::gradient::total_rimp2_gradient_on`]) and warm-started through
/// `warm`.
pub fn total_att_rimp2_gradient_on(
    mol: &Molecule,
    obs_basis: &BasisSet,
    aux_basis: &BasisSet,
    config: &AttenuatedMp2Config,
    ext: Option<&ExternalPotential>,
    scf_jk: &ScfJk,
    warm: &mut ScfWarmStart,
) -> Result<(f64, Array2<f64>), FerricError> {
    let (op, ri, scaling) = att_parts(config)?;
    total_mp2_gradient_scaled_on(
        mol,
        obs_basis,
        aux_basis,
        Operator::coulomb(),
        op,
        &ri,
        scaling,
        ext,
        scf_jk,
        warm,
    )
}

/// An error unless `op` is a terfc operator ([`Operator::terfc`] or
/// [`Operator::terfc_with_omega`]) with finite, positive `r0` and `ω`, and
/// `ri` is plain RI-MP2 (no decoupled metric, no κ regularization).
fn require_terfc(op: Operator, ri: &RiMp2Config) -> Result<(), FerricError> {
    if op.is_composite() || op.kind != OperatorKind::Terfc {
        return Err(FerricError::General(format!(
            "terfc RI-MP2 gradient: the correlation operator must be terfc (got {:?})",
            op.kind
        )));
    }
    if !(op.distance.is_finite() && op.distance > 0.0 && op.omega.is_finite() && op.omega > 0.0) {
        return Err(FerricError::General(format!(
            "terfc RI-MP2 gradient: r0 and omega must be finite and > 0 (got r0 = {} Bohr, \
             omega = {} Bohr⁻¹)",
            op.distance, op.omega
        )));
    }
    if ri.metric_op.is_some() || ri.kappa.is_some() {
        return Err(FerricError::General(
            "terfc RI-MP2 gradient: a decoupled metric_op or kappa regularization is not \
             differentiated; leave both unset"
                .into(),
        ));
    }
    Ok(())
}

/// `dE/dR`, `(natoms, 3)`, of `rhf.energy + E_c[terfc]` (the total the
/// `att_operator = "terfc"` energy reports: RI-MP2 with `op` in both the
/// `(P|ia)` integrals and the metric) on the converged Coulomb RHF `rhf`,
/// whose J/K route `bounds` describes. `op` is [`Operator::terfc`] or
/// [`Operator::terfc_with_omega`]; `ri` carries the frozen core and memory
/// budget. `ext` as in [`att_rimp2_gradient_analytical`].
///
/// # Errors
///
/// A non-Coulomb reference, a non-terfc `op`, a `metric_op` or `kappa` in
/// `ri`, and the errors of the RI-MP2 gradient (unconverged reference or
/// Z-vector, X2C).
#[allow(clippy::too_many_arguments)]
pub fn terfc_rimp2_gradient_analytical(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    bounds: &SchwarzBounds,
    rhf: &ScfResult,
    op: Operator,
    ri: &RiMp2Config,
    ext: Option<&ExternalPotential>,
) -> Result<Array2<f64>, FerricError> {
    ferric_scf::x2c::refuse_relativistic(rhf.relativistic, "terfc RI-MP2 analytic gradient")?;
    require_coulomb_reference(bounds.op)?;
    require_terfc(op, ri)?;
    let mut acc = PhaseAccumulator::new().report_on_drop("terfc-RI-MP2 gradient");
    rimp2_gradient_analytical_timed(
        mol,
        obs,
        dfbs,
        op,
        bounds,
        rhf,
        ri,
        SpinScaling::MP2,
        ext,
        ZvectorControls::DEFAULT,
        &mut acc,
    )
}

/// The total terfc RI-MP2 energy (`E_HF + E_c[terfc]`, Coulomb RHF solved
/// here at `density_conv = 1e-9`) and its analytic gradient at one geometry,
/// the reference on `scf_jk` (`"exact"` or `"rijk"`) and warm-started
/// through `warm`. `op` and `ri` as in [`terfc_rimp2_gradient_analytical`].
#[allow(clippy::too_many_arguments)]
pub fn total_terfc_rimp2_gradient_on(
    mol: &Molecule,
    obs_basis: &BasisSet,
    aux_basis: &BasisSet,
    op: Operator,
    ri: &RiMp2Config,
    ext: Option<&ExternalPotential>,
    scf_jk: &ScfJk,
    warm: &mut ScfWarmStart,
) -> Result<(f64, Array2<f64>), FerricError> {
    require_terfc(op, ri)?;
    total_mp2_gradient_scaled_on(
        mol,
        obs_basis,
        aux_basis,
        Operator::coulomb(),
        op,
        ri,
        SpinScaling::MP2,
        ext,
        scf_jk,
        warm,
    )
}

/// `dE/dR`, `(natoms, 3)`, of `uhf.energy + E_c[terfc]` (`E_c = E_αα +
/// E_ββ + E_αβ`, the energy [`crate::u_rimp2::u_ri_mp2`] reports on `op`) on
/// the converged, aufbau Coulomb UHF `uhf`, whose J/K route `bounds`
/// describes. `op`, `ri` and `ext` as in [`terfc_rimp2_gradient_analytical`].
///
/// # Errors
///
/// The errors of [`u_att_rimp2_gradient_analytical`] (a ROHF, restricted,
/// unconverged, non-aufbau or X2C reference; no beta electron; a non-Coulomb
/// reference) and the `op` / `ri` refusals of
/// [`terfc_rimp2_gradient_analytical`].
#[allow(clippy::too_many_arguments)]
pub fn u_terfc_rimp2_gradient_analytical(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    bounds: &SchwarzBounds,
    uhf: &ScfResult,
    op: Operator,
    ri: &RiMp2Config,
    ext: Option<&ExternalPotential>,
) -> Result<Array2<f64>, FerricError> {
    require_coulomb_reference(bounds.op)?;
    require_terfc(op, ri)?;
    let mut acc = PhaseAccumulator::new().report_on_drop("U-terfc-RI-MP2 gradient");
    u_correlated_gradient_timed(
        mol,
        obs,
        dfbs,
        UCorrelation::uniform(op, 1.0),
        bounds,
        uhf,
        ri,
        ext,
        ZvectorControls::DEFAULT,
        &mut acc,
    )
}

/// The total open-shell terfc RI-MP2 energy (`E_UHF + E_c[terfc]`, the
/// Coulomb UHF of the molecule's charge and multiplicity solved here at
/// `density_conv = 1e-9`) and its analytic gradient at one geometry, the
/// reference on `scf_jk` (`"exact"` or `"rijk"`) and warm-started through
/// `warm`. `op` and `ri` as in [`terfc_rimp2_gradient_analytical`].
#[allow(clippy::too_many_arguments)]
pub fn total_u_terfc_rimp2_gradient_on(
    mol: &Molecule,
    obs_basis: &BasisSet,
    aux_basis: &BasisSet,
    op: Operator,
    ri: &RiMp2Config,
    ext: Option<&ExternalPotential>,
    scf_jk: &ScfJk,
    warm: &mut ScfWarmStart,
) -> Result<(f64, Array2<f64>), FerricError> {
    let scf_base = RhfConfig {
        density_conv: 1e-9,
        external_potential: ext.cloned(),
        ..Default::default()
    };
    total_u_terfc_rimp2_gradient_with(
        mol, obs_basis, aux_basis, op, ri, ext, scf_jk, warm, scf_base,
    )
}

/// [`total_u_terfc_rimp2_gradient_on`] with the UHF solved on `scf_base`
/// (before the geometry-task and `scf_jk` adjustments), for the
/// finite-difference tests.
#[allow(clippy::too_many_arguments)]
#[doc(hidden)]
pub fn total_u_terfc_rimp2_gradient_with(
    mol: &Molecule,
    obs_basis: &BasisSet,
    aux_basis: &BasisSet,
    op: Operator,
    ri: &RiMp2Config,
    ext: Option<&ExternalPotential>,
    scf_jk: &ScfJk,
    warm: &mut ScfWarmStart,
    scf_base: RhfConfig,
) -> Result<(f64, Array2<f64>), FerricError> {
    require_terfc(op, ri)?;
    let (uhf, obs, dfbs, bounds) = solve_reference(
        mol,
        obs_basis,
        aux_basis,
        Operator::coulomb(),
        ri,
        scf_jk,
        warm,
        scf_base,
    )?;
    let mut acc = PhaseAccumulator::new().report_on_drop("U-terfc-RI-MP2 energy + gradient");
    let energy = acc.time("terfc U-RI-MP2 energy", || {
        crate::u_rimp2::u_ri_mp2(mol, &obs, &dfbs, op, &uhf, ri).map(|r| r.total_energy)
    })?;
    let grad = u_correlated_gradient_timed(
        mol,
        &obs,
        &dfbs,
        UCorrelation::uniform(op, 1.0),
        &bounds,
        &uhf,
        ri,
        ext,
        ZvectorControls::DEFAULT,
        &mut acc,
    )?;
    Ok((energy, grad))
}

/// The total open-shell terfc RI-MP2 energy on the reference of
/// [`total_u_terfc_rimp2_gradient_with`] (same SCF settings and route),
/// without the gradient: the finite-difference reference of the tests.
/// Returns the UHF too, so a test can check its convergence.
#[doc(hidden)]
pub fn total_u_terfc_rimp2_energy_with(
    mol: &Molecule,
    obs_basis: &BasisSet,
    aux_basis: &BasisSet,
    op: Operator,
    ri: &RiMp2Config,
    scf_jk: &ScfJk,
    scf_base: RhfConfig,
) -> Result<(f64, ScfResult), FerricError> {
    require_terfc(op, ri)?;
    let (uhf, obs, dfbs, _) = solve_reference(
        mol,
        obs_basis,
        aux_basis,
        Operator::coulomb(),
        ri,
        scf_jk,
        &mut ScfWarmStart::new(false),
        scf_base,
    )?;
    let e = crate::u_rimp2::u_ri_mp2(mol, &obs, &dfbs, op, &uhf, ri)?.total_energy;
    Ok((e, uhf))
}

/// `dE/dR`, `(natoms, 3)`, of `uhf.energy + s · E_c[erfc]` (the total
/// [`crate::attenuated::attenuated_ri_mp2`] reports for `config` on a UHF
/// reference: `E_c = E_αα + E_ββ + E_αβ`) on the converged, aufbau Coulomb
/// UHF `uhf`, whose J/K route `bounds` describes (four-centre, or the RI-JK
/// route `uhf` carries). `ext` as in
/// [`crate::u_rimp2_gradient::u_rimp2_gradient_analytical`].
///
/// # Errors
///
/// The errors of [`crate::u_rimp2_gradient::u_rimp2_gradient_analytical`]
/// (a ROHF, restricted, unconverged, non-aufbau or X2C reference; no beta
/// electron; an unconverged Z-vector), a non-Coulomb reference, and the
/// `config` refusals of [`att_rimp2_gradient_analytical`].
pub fn u_att_rimp2_gradient_analytical(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    bounds: &SchwarzBounds,
    uhf: &ScfResult,
    config: &AttenuatedMp2Config,
    ext: Option<&ExternalPotential>,
) -> Result<Array2<f64>, FerricError> {
    require_coulomb_reference(bounds.op)?;
    let (op, ri, scaling) = att_parts(config)?;
    let mut acc = PhaseAccumulator::new().report_on_drop("U-att-RI-MP2 gradient");
    u_correlated_gradient_timed(
        mol,
        obs,
        dfbs,
        UCorrelation::uniform(op, scaling.c_os),
        bounds,
        uhf,
        &ri,
        ext,
        ZvectorControls::DEFAULT,
        &mut acc,
    )
}

/// The total open-shell attenuated RI-MP2 energy (`E_UHF + s · E_c[erfc]`,
/// the Coulomb UHF of the molecule's charge and multiplicity solved here at
/// `density_conv = 1e-9`) and its analytic gradient at one geometry, the
/// reference on `scf_jk` (`"exact"` or `"rijk"`, as in
/// [`crate::u_rimp2_gradient::total_u_rimp2_gradient_on`]) and
/// warm-started through `warm`.
pub fn total_u_att_rimp2_gradient_on(
    mol: &Molecule,
    obs_basis: &BasisSet,
    aux_basis: &BasisSet,
    config: &AttenuatedMp2Config,
    ext: Option<&ExternalPotential>,
    scf_jk: &ScfJk,
    warm: &mut ScfWarmStart,
) -> Result<(f64, Array2<f64>), FerricError> {
    let scf_base = RhfConfig {
        density_conv: 1e-9,
        external_potential: ext.cloned(),
        ..Default::default()
    };
    total_u_att_rimp2_gradient_with(
        mol, obs_basis, aux_basis, config, ext, scf_jk, warm, scf_base,
    )
}

/// [`total_u_att_rimp2_gradient_on`](crate::att_rimp2_gradient::total_u_att_rimp2_gradient_on) with the UHF solved on `scf_base`
/// (before the geometry-task and `scf_jk` adjustments): the
/// finite-difference tests converge every displaced reference far below the
/// default.
#[allow(clippy::too_many_arguments)]
#[doc(hidden)]
pub fn total_u_att_rimp2_gradient_with(
    mol: &Molecule,
    obs_basis: &BasisSet,
    aux_basis: &BasisSet,
    config: &AttenuatedMp2Config,
    ext: Option<&ExternalPotential>,
    scf_jk: &ScfJk,
    warm: &mut ScfWarmStart,
    scf_base: RhfConfig,
) -> Result<(f64, Array2<f64>), FerricError> {
    let (op, ri, scaling) = att_parts(config)?;
    let (uhf, obs, dfbs, bounds) = solve_reference(
        mol,
        obs_basis,
        aux_basis,
        Operator::coulomb(),
        &ri,
        scf_jk,
        warm,
        scf_base,
    )?;
    let mut acc = PhaseAccumulator::new().report_on_drop("U-att-RI-MP2 energy + gradient");
    let energy = acc.time("attenuated U-RI-MP2 energy", || {
        crate::attenuated::attenuated_ri_mp2(mol, &obs, &dfbs, &uhf, config).map(|r| r.total_energy)
    })?;
    let grad = u_correlated_gradient_timed(
        mol,
        &obs,
        &dfbs,
        UCorrelation::uniform(op, scaling.c_os),
        &bounds,
        &uhf,
        &ri,
        ext,
        ZvectorControls::DEFAULT,
        &mut acc,
    )?;
    Ok((energy, grad))
}

/// The total open-shell attenuated RI-MP2 energy on the reference of
/// [`total_u_att_rimp2_gradient_with`] (same SCF settings and route), without
/// the gradient: the finite-difference reference of the tests. Returns the
/// UHF too, so a test can check its convergence.
#[doc(hidden)]
pub fn total_u_att_rimp2_energy_with(
    mol: &Molecule,
    obs_basis: &BasisSet,
    aux_basis: &BasisSet,
    config: &AttenuatedMp2Config,
    scf_jk: &ScfJk,
    scf_base: RhfConfig,
) -> Result<(f64, ScfResult), FerricError> {
    let (_, ri, _) = att_parts(config)?;
    let (uhf, obs, dfbs, _) = solve_reference(
        mol,
        obs_basis,
        aux_basis,
        Operator::coulomb(),
        &ri,
        scf_jk,
        &mut ScfWarmStart::new(false),
        scf_base,
    )?;
    let e = crate::attenuated::attenuated_ri_mp2(mol, &obs, &dfbs, &uhf, config)?.total_energy;
    Ok((e, uhf))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gradient::rimp2_gradient_analytical_timed;
    use ferric_core::basis;
    use ferric_scf::rhf::solve_rhf;
    use ferric_scf::scf_jk::ScfJkMode;
    use ferric_scf::Spin;

    const WATER: &str = "3\nwater, off C2v\nO 0.02 -0.01 0.117790\nH 0.0 0.785453 -0.451161\nH 0.03 -0.735453 -0.491161\n";
    const OH: &str = "2\nOH\nO 0 0 0\nH 0 0 0.97\n";

    /// Both anchors compare two solves, so each Z-vector must sit far below
    /// the comparison bar.
    const TIGHT: ZvectorControls = ZvectorControls {
        max_iter: 200,
        tol: 1e-12,
    };

    /// The bar of both anchors, derived in `u_rimp2_gradient.rs`
    /// (`closed_limit`): two Z-vector residuals `< 1e-12` move the gradient
    /// by `≲ 10 · 2 · 1e-12 = 2e-11`, operation-order rounding is `≈ 1e-14`.
    /// For the ω → 0 anchor the operator itself differs by
    /// `erfc(ωr)/r − 1/r = −2ω/√π + O(ω³r²)`: at `ω = 1e-7` Bohr⁻¹ the shift
    /// of the fitted `(ia|jb)` is `(2ω/√π) q_ia q_jb ≲ 1.1e-7 · 1e-4`, and of
    /// the gradient the same order, `≲ 1e-11`
    /// (`book/src/15-correlated-gradients/02-rimp2-gradient.md#rimp2-gradient-attenuated-erfc-ri-mp2`).
    const ANCHOR_BAR: f64 = 1e-10;

    fn att(omega_bohr_inv: f64, scaling: f64, frozen_core: usize) -> AttenuatedMp2Config {
        AttenuatedMp2Config {
            omega: omega_bohr_inv,
            scaling,
            frozen_core,
            screen_thresh: None,
            memory_budget_bytes: None,
        }
    }

    fn max_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        (a - b).iter().fold(0.0f64, |m, v| m.max(v.abs()))
    }

    /// The RHF `rhf` written as the UHF with identical α and β orbitals,
    /// densities and Fock matrices: a UHF stationary point.
    fn lift_to_unrestricted(rhf: &ScfResult) -> ScfResult {
        let mut u = rhf.clone();
        u.spin = Spin::Unrestricted;
        u.density_beta = Some(rhf.density_alpha.clone());
        u.mos_beta = Some(rhf.mos_alpha.clone());
        u.eps_beta = Some(rhf.eps_alpha.clone());
        u.fock_beta = Some(rhf.fock_alpha.clone());
        u
    }

    fn scf_cfg(fit: bool) -> RhfConfig {
        let mut cfg = RhfConfig {
            density_conv: 1e-11,
            max_iter: 300,
            ..Default::default()
        }
        .for_geometry_task();
        if fit {
            ScfJk {
                mode: ScfJkMode::RiJk,
                aux: Some("def2-universal-jkfit".to_string()),
                k_builder: None,
            }
            .apply(&mut cfg);
        }
        cfg
    }

    /// EXACTNESS ANCHOR (closed-shell limit). A closed-shell Coulomb RHF
    /// written as a UHF (α ≡ β) runs the open-shell attenuated gradient; it
    /// must equal the closed-shell attenuated gradient
    /// ([`rimp2_gradient_analytical_timed`] with erfc and `(s, s)`) on the
    /// same orbitals. ω = 1.0 Å⁻¹ and `s = 1.2` so the scaling and the
    /// operator are both away from their trivial values.
    fn closed_limit(fit: bool, frozen_core: usize) -> f64 {
        let mol = Molecule::parse_xyz(WATER, 0, 1).unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("6-31g").unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        let coul = Operator::coulomb();
        let bounds = SchwarzBounds::compute(coul, &obs).unwrap();
        let ctx = ferric_core::parallel::ParallelContext::default();
        let rhf = solve_rhf(&ctx, &mol, &obs, coul, &bounds, &scf_cfg(fit)).unwrap();
        assert!(rhf.converged);
        assert_eq!(rhf.df_jk.is_some(), fit, "route not recorded");
        let uhf = lift_to_unrestricted(&rhf);
        let config = att(
            1.0 * crate::attenuated::BOHR_INV_PER_ANG_INV,
            1.2,
            frozen_core,
        );
        let (op, ri, scaling) = att_parts(&config).unwrap();
        let g_r = rimp2_gradient_analytical_timed(
            &mol,
            &obs,
            &dfbs,
            op,
            &bounds,
            &rhf,
            &ri,
            scaling,
            None,
            TIGHT,
            &mut PhaseAccumulator::new(),
        )
        .unwrap();
        let g_u = u_correlated_gradient_timed(
            &mol,
            &obs,
            &dfbs,
            UCorrelation::uniform(op, scaling.c_os),
            &bounds,
            &uhf,
            &ri,
            None,
            TIGHT,
            &mut PhaseAccumulator::new(),
        )
        .unwrap();
        let d = max_abs_diff(&g_r, &g_u);
        eprintln!(
            "att closed limit fit={fit} fc={frozen_core}: max|g_U - g_R| = {d:.3e}, max|g| = {:.3e}",
            g_r.iter().fold(0.0f64, |m, v| m.max(v.abs()))
        );
        d
    }

    #[test]
    fn open_shell_att_gradient_matches_closed_in_the_trivial_limit() {
        let d = closed_limit(false, 0);
        assert!(d < ANCHOR_BAR, "max|g_U - g_R| = {d:.3e}");
    }

    #[test]
    fn open_shell_att_gradient_matches_closed_in_the_trivial_limit_frozen_core() {
        let d = closed_limit(false, 1);
        assert!(d < ANCHOR_BAR, "max|g_U - g_R| = {d:.3e}");
    }

    #[test]
    fn open_shell_att_gradient_matches_closed_in_the_trivial_limit_rijk() {
        let d = closed_limit(true, 0);
        assert!(d < ANCHOR_BAR, "max|g_U - g_R| = {d:.3e}");
    }

    /// EXACTNESS ANCHOR (operator limit). On a genuine open shell (the OH
    /// doublet, UHF), the attenuated gradient at ω = 1e-7 Bohr⁻¹ equals the
    /// U-RI-MP2 (Coulomb) gradient on the same UHF. Non-vacuity: at the
    /// production ω = 0.420 Å⁻¹ the two differ by more than 1e-5 Ha/Bohr.
    #[test]
    fn open_shell_att_gradient_is_u_rimp2_as_omega_goes_to_zero() {
        let mol = Molecule::parse_xyz(OH, 0, 2).unwrap();
        let obs_bs = basis::bundled("6-31g").unwrap();
        let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
        let coul = Operator::coulomb();
        let bounds = SchwarzBounds::compute(coul, &obs).unwrap();
        let ctx = ferric_core::parallel::ParallelContext::default();
        let uhf = ferric_scf::uhf::solve_uhf(&ctx, &mol, &obs, &bounds, &scf_cfg(false)).unwrap();
        assert!(uhf.converged && uhf.spin == Spin::Unrestricted);
        let run = |omega: f64| {
            let (op, ri, scaling) = att_parts(&att(omega, 1.0, 0)).unwrap();
            u_correlated_gradient_timed(
                &mol,
                &obs,
                &dfbs,
                UCorrelation::uniform(op, scaling.c_os),
                &bounds,
                &uhf,
                &ri,
                None,
                TIGHT,
                &mut PhaseAccumulator::new(),
            )
            .unwrap()
        };
        let g_mp2 = u_correlated_gradient_timed(
            &mol,
            &obs,
            &dfbs,
            UCorrelation::uniform(coul, 1.0),
            &bounds,
            &uhf,
            &RiMp2Config::default(),
            None,
            TIGHT,
            &mut PhaseAccumulator::new(),
        )
        .unwrap();
        let d = max_abs_diff(&run(1e-7), &g_mp2);
        let gap = max_abs_diff(
            &run(0.420 * crate::attenuated::BOHR_INV_PER_ANG_INV),
            &g_mp2,
        );
        eprintln!("OH: max|g_att(1e-7) - g_UMP2| = {d:.3e}; at 0.420 A^-1: {gap:.3e}");
        assert!(d < ANCHOR_BAR, "omega -> 0: {d:.3e}");
        assert!(gap > 1e-5, "non-vacuity: {gap:.3e}");
    }

    /// The public entry point refuses a ROHF reference (with the reason), a
    /// non-Coulomb reference, and the config the closed-shell gradient
    /// refuses.
    #[test]
    fn open_shell_att_gradient_refuses_what_it_does_not_differentiate() {
        let mol = Molecule::parse_xyz(OH, 0, 2).unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("sto-3g").unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        let coul = Operator::coulomb();
        let bounds = SchwarzBounds::compute(coul, &obs).unwrap();
        let ctx = ferric_core::parallel::ParallelContext::default();
        let good = att(0.2222, 1.0, 0);
        let rohf =
            ferric_scf::rohf::solve_rohf(&ctx, &mol, &obs, coul, &bounds, &RhfConfig::default())
                .unwrap();
        let err = u_att_rimp2_gradient_analytical(&mol, &obs, &dfbs, &bounds, &rohf, &good, None)
            .unwrap_err();
        assert!(format!("{err}").contains("needs a UHF reference"), "{err}");
        let uhf =
            ferric_scf::uhf::solve_uhf(&ctx, &mol, &obs, &bounds, &RhfConfig::default()).unwrap();
        assert!(
            u_att_rimp2_gradient_analytical(&mol, &obs, &dfbs, &bounds, &uhf, &good, None).is_ok()
        );
        let erfc_bounds = SchwarzBounds::compute(Operator::erfc(0.2222), &obs).unwrap();
        assert!(u_att_rimp2_gradient_analytical(
            &mol,
            &obs,
            &dfbs,
            &erfc_bounds,
            &uhf,
            &good,
            None
        )
        .is_err());
        for bad in [
            att(0.0, 1.0, 0),
            att(f64::NAN, 1.0, 0),
            att(0.2222, f64::INFINITY, 0),
            AttenuatedMp2Config {
                screen_thresh: Some(1e-10),
                ..good.clone()
            },
        ] {
            assert!(
                u_att_rimp2_gradient_analytical(&mol, &obs, &dfbs, &bounds, &uhf, &bad, None)
                    .is_err()
            );
        }
    }
}
