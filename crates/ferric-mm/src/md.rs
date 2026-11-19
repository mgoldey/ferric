//! Force-field molecular dynamics front end (`run_ffmd`); see `ferric-md`.
//!
//! [`MmForceProvider`] hands the AMBER-form energy and analytic gradient of
//! [`crate::gradient_with`] to the `ferric-md` integrators. An
//! [`MmTopology`] carries no elements or masses, so the caller supplies them
//! in the [`MdSystem`].

use crate::cutoff::{gradient_with, NonbondedMethod};
use crate::energy::MmEnergy;
use crate::topology::MmTopology;
use ferric_core::FerricError;
use ferric_md::{run_md, ForceProvider, MdConfig, MdObserver, MdResult, MdSystem};
use ndarray::Array2;

/// A [`ForceProvider`] backed by the MM force field.
pub struct MmForceProvider<'a> {
    /// The force-field topology.
    pub top: &'a MmTopology,
    /// How the nonbonded sum is evaluated.
    pub method: NonbondedMethod,
    /// Energy components of the most recent evaluation.
    pub last: MmEnergy,
}

impl<'a> MmForceProvider<'a> {
    /// A provider for `top` under `method`, with `last` zeroed.
    pub fn new(top: &'a MmTopology, method: NonbondedMethod) -> Self {
        Self {
            top,
            method,
            last: MmEnergy::default(),
        }
    }
}

impl ForceProvider for MmForceProvider<'_> {
    fn n_atoms(&self) -> usize {
        self.top.n_atoms()
    }

    fn energy_gradient(&mut self, coords: &Array2<f64>) -> Result<(f64, Array2<f64>), FerricError> {
        let (e, g) = gradient_with(self.top, coords, &self.method)?;
        self.last = e;
        Ok((e.total, g))
    }

    fn label(&self) -> String {
        match self.method {
            NonbondedMethod::NoCutoff => "mm".to_string(),
            NonbondedMethod::CutoffNonPeriodic(_) => "mm:cutoff".to_string(),
        }
    }
}

/// Molecular dynamics on the MM surface of `top`, with the symbols, masses
/// and starting coordinates (Bohr) of `system`.
pub fn run_ffmd(
    top: &MmTopology,
    system: &MdSystem,
    method: &NonbondedMethod,
    config: &MdConfig,
    observers: &mut [&mut dyn MdObserver],
) -> Result<MdResult, FerricError> {
    if system.coords.nrows() != top.n_atoms() {
        return Err(FerricError::General(format!(
            "FFMD: the system has {} atoms but the topology has {}",
            system.coords.nrows(),
            top.n_atoms()
        )));
    }
    let mut provider = MmForceProvider::new(top, *method);
    run_md(system, &mut provider, config, observers)
}
