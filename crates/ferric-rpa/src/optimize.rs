//! RPA-driven geometry optimization.
//!
//! The search loop is the SCF crate's
//! ([`ferric_scf::optimize::optimize_geometry_with_energy_gradient`]); this
//! module supplies the surface: `total_rpa_gradient_by` (a finite difference
//! of E_RHF + E_c^RPA, or the analytic gradient, per
//! [`RpaGradientMethod`](crate::gradient::RpaGradientMethod)).

use crate::gradient::{total_rpa_gradient_by, RpaGradientMethod};
use crate::PdepRpaConfig;
use ferric_core::basis::BasisSet;
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::operator::Operator;
use ferric_scf::optimize::{optimize_geometry_with_energy_gradient, OptimizeConfig};
use ferric_scf::warm_start::{ScfWarmStart, WarmStartRecord};

/// Result of an RPA geometry optimization.
#[derive(Debug, Clone)]
#[must_use]
pub struct RpaOptimizeResult {
    /// Geometry after the last BFGS step (coordinates in Bohr).
    pub mol: Molecule,
    /// Total energy E_RHF + E_c^RPA at `mol`, Hartree.
    pub energy: f64,
    /// Number of BFGS steps taken.
    pub steps: usize,
    /// Whether |ΔE| < `e_conv`, max |g| < `g_max_thresh` and RMS g <
    /// `g_rms_thresh` were all met before `max_steps`.
    pub converged: bool,
    /// Per-SCF warm-start records in evaluation order: each step's reference
    /// RHF followed, on the finite-difference route, by its `6N` displaced RHFs (see `ferric_scf::warm_start`;
    /// all cold `Disabled` when `OptimizeConfig::scf_warm_start` is off).
    pub scf_warm_start: Vec<WarmStartRecord>,
    /// Energy at every geometry the optimizer evaluated, in order (the
    /// starting geometry first).
    pub energy_trace: Vec<f64>,
    /// The search that ran (see `ferric_scf::optimize::SearchSettings`).
    pub search: ferric_scf::optimize::SearchSettings,
}

/// Optimize a molecular geometry on the RPA PES, the gradient from
/// `gradient` (a central finite difference of the total RPA energy with a
/// fresh RHF + PDEP-RPA at every displaced point, or the analytic gradient).
///
/// The search is [`ferric_scf::optimize::optimize_geometry_with_energy_gradient`]:
/// the energy and gradient at a geometry are the closure below, and
/// `opt_config.coord_system` selects Cartesian or redundant internal
/// coordinates as for the SCF optimizers.
pub fn optimize_geometry_rpa(
    mol: &Molecule,
    obs_basis: &BasisSet,
    aux_basis: &BasisSet,
    op: Operator,
    rpa_config: &PdepRpaConfig,
    opt_config: &OptimizeConfig,
    gradient: RpaGradientMethod,
) -> Result<RpaOptimizeResult, FerricError> {
    // Each step's reference RHF starts from the previous step's, and the
    // displaced RHFs of its finite-difference gradient from that reference
    // (`opt_config.scf_warm_start`, see `ferric_scf::warm_start`).
    let mut warm = ScfWarmStart::new(opt_config.scf_warm_start);
    let out = optimize_geometry_with_energy_gradient(mol, opt_config, |m| {
        total_rpa_gradient_by(m, obs_basis, aux_basis, op, rpa_config, gradient, &mut warm)
    })?;
    if let Some(line) = warm.summary() {
        println!("{line}");
    }
    Ok(RpaOptimizeResult {
        mol: out.mol,
        energy: out.energy,
        steps: out.steps,
        converged: out.converged,
        scf_warm_start: warm.records().to_vec(),
        energy_trace: out.energy_trace,
        search: out.search,
    })
}
