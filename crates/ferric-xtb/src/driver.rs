//! Geometry optimization and harmonic frequencies on a GFN-xTB surface.
//!
//! Both drivers reuse the existing `ferric-scf` machinery through its
//! closure-based entry points
//! ([`ferric_scf::optimize::optimize_geometry_with_energy_gradient`],
//! [`ferric_scf::frequencies::harmonic_frequencies_from_gradient`]); only the
//! energy/gradient evaluation differs. No second optimizer or Hessian code
//! lives here.
//!
//! Every geometry gets a fresh [`XtbCalculator`] (the xtb molecule handle is
//! built from the coordinates), dropped before the next one is created, which
//! keeps libxtb driven from one thread at a time.
//!
//! An SCC that does not converge makes libxtb queue an error, which surfaces as
//! `Err` from [`xtb_energy_gradient`] and therefore aborts the optimization or
//! the Hessian instead of letting the search walk on a non-stationary density.

use crate::calculator::{XtbCalculator, XtbConfig};
use ferric_core::error::FerricError;
use ferric_core::mol::Molecule;
use ferric_scf::frequencies::{harmonic_frequencies_from_gradient, FrequencyResult};
use ferric_scf::optimize::{
    optimize_geometry_with_energy_gradient, OptimizeConfig, OptimizeResult,
};
use ndarray::Array2;

/// Central-difference displacement (Bohr) for the xTB Hessian. Matches the
/// SCF frequency default (`ferric_scf::frequencies::DEFAULT_DELTA`).
pub use ferric_scf::frequencies::DEFAULT_DELTA as DEFAULT_HESSIAN_DELTA;

/// Energy (Hartree) and gradient (Hartree/Bohr, `[natoms, 3]`) at `mol`.
pub fn xtb_energy_gradient(
    mol: &Molecule,
    config: XtbConfig,
) -> Result<(f64, Array2<f64>), FerricError> {
    let out = XtbCalculator::new(mol, config)?.singlepoint()?;
    Ok((out.energy, out.gradient))
}

/// Relax `mol` on the GFN-xTB surface with ferric's BFGS optimizer.
///
/// `opt_config.max_steps` bounds the geometry steps; `config.scc_max_iter`
/// bounds the SCC iterations at each geometry.
pub fn optimize_geometry_xtb(
    mol: &Molecule,
    config: XtbConfig,
    opt_config: &OptimizeConfig,
) -> Result<OptimizeResult, FerricError> {
    optimize_geometry_with_energy_gradient(mol, opt_config, |m| xtb_energy_gradient(m, config))
}

/// Harmonic frequencies on the GFN-xTB surface: central differences (`delta`
/// Bohr) of the analytic xTB gradient, then the shared mass-weighting and
/// translation/rotation projection. `mol` should be an xTB stationary point.
pub fn harmonic_frequencies_xtb(
    mol: &Molecule,
    config: XtbConfig,
    delta: f64,
) -> Result<FrequencyResult, FerricError> {
    let (energy, _) = xtb_energy_gradient(mol, config)?;
    harmonic_frequencies_from_gradient(mol, delta, energy, |m| {
        xtb_energy_gradient(m, config).map(|(_, g)| g)
    })
}
