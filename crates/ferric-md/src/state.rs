//! Trajectory frames, state, observers and the run result.

use crate::config::MdConfig;
use crate::system::MdSystem;
use ferric_core::error::FerricError;
use ndarray::Array2;

/// Scalar observables at one recorded step (energies in Hartree).
#[derive(Debug, Clone)]
pub struct MdFrame {
    /// Step index.
    pub step: usize,
    /// Simulation time in fs.
    pub time_fs: f64,
    /// Potential energy.
    pub potential: f64,
    /// Kinetic energy.
    pub kinetic: f64,
    /// Potential plus kinetic.
    pub total: f64,
    /// Total minus accumulated thermostat-injected kinetic energy.
    pub conserved: f64,
    /// Instantaneous temperature in K.
    pub temperature_k: f64,
}

/// Full dynamical state at one step (atomic units).
#[derive(Debug, Clone)]
pub struct MdState {
    /// Scalar observables.
    pub frame: MdFrame,
    /// Positions, Bohr.
    pub coords: Array2<f64>,
    /// Velocities, Bohr / a.u. time.
    pub velocities: Array2<f64>,
    /// Gradient, Hartree/Bohr.
    pub gradient: Array2<f64>,
}

/// Callback invoked on each recorded frame.
pub trait MdObserver {
    /// Called for step 0 and every `stride`-th step.
    fn on_frame(&mut self, system: &MdSystem, state: &MdState) -> Result<(), FerricError>;
}

/// Result of an MD run.
#[derive(Debug, Clone)]
#[must_use]
pub struct MdResult {
    /// Step 0 plus every `stride`-th step.
    pub frames: Vec<MdFrame>,
    /// Positions in Bohr; empty if `!keep_trajectory`.
    pub positions: Vec<Array2<f64>>,
    /// Velocities per recorded frame; empty if `!keep_trajectory`.
    pub velocities: Vec<Array2<f64>>,
    /// State after the last step.
    pub final_state: MdState,
    /// Degrees of freedom used for temperatures.
    pub n_dof: usize,
    /// Number of force evaluations (`n_steps + 1`).
    pub n_force_evaluations: usize,
    /// Provider label.
    pub label: String,
    /// The configuration that produced this result.
    pub config: MdConfig,
}

impl MdResult {
    /// Largest `|conserved - conserved_0|` over the frames (Hartree).
    pub fn max_conserved_drift(&self) -> f64 {
        todo!()
    }

    /// Mean temperature over frames after skipping `skip_frames` (K).
    pub fn mean_temperature(&self, skip_frames: usize) -> f64 {
        let _ = skip_frames;
        todo!()
    }
}
