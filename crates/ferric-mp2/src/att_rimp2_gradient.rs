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
//! (`crates/ferric-mp2/theory.md`, section "Attenuated RI-MP2 gradient"):
//!
//! * erfc: the amplitudes and unrelaxed densities, `b_full` and the
//!   Lagrangian `Imat` of the Z-vector right-hand side, and the 3c/2c
//!   integral response (`d(P|erfc|μν)`, `d(P|erfc|Q)`);
//! * Coulomb: the Z-vector's orbital Hessian, the `veff` of the correlation
//!   density, and the four-centre (or RI-JK) derivative term, all of which
//!   come from the HF stationarity condition and the Coulomb Fock matrix.
//!
//! The scaling `s` enters as the spin scaling `c_os = c_ss = s`
//! ([`SpinScaling`]). The QQR-3 screened energy (`screen_thresh`) is not
//! differentiated and is refused, as is the terfc operator (it has no
//! derivative engine).

use crate::attenuated::AttenuatedMp2Config;
use crate::gradient::{rimp2_gradient_analytical_timed, total_mp2_gradient_scaled_on};
use crate::rimp2::{RiMp2Config, SpinScaling};
use crate::zvector::ZvectorControls;
use ferric_core::basis::BasisSet;
use ferric_core::external_potential::ExternalPotential;
use ferric_core::mol::Molecule;
use ferric_core::timing::PhaseAccumulator;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::{Operator, OperatorKind};
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
