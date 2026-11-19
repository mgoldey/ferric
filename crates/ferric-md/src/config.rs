//! Run configuration: thermostat choice and integration parameters.

use ferric_core::error::FerricError;
use ndarray::Array2;

/// Thermostat / ensemble selection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Thermostat {
    /// Microcanonical velocity Verlet.
    Nve,
    /// Berendsen weak coupling (equilibration only; not canonical).
    Berendsen {
        /// Coupling time constant in fs.
        tau_fs: f64,
    },
    /// Langevin dynamics integrated with BAOAB.
    Langevin {
        /// Friction coefficient in 1/ps.
        friction_per_ps: f64,
    },
}

impl Thermostat {
    /// Parse `"nve" | "berendsen" | "langevin"` plus the matching parameter.
    pub fn parse_config_str(
        kind: &str,
        tau_fs: Option<f64>,
        friction_per_ps: Option<f64>,
    ) -> Result<Self, String> {
        let _ = (kind, tau_fs, friction_per_ps);
        todo!()
    }

    /// Lowercase config name of the variant.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Nve => "nve",
            Self::Berendsen { .. } => "berendsen",
            Self::Langevin { .. } => "langevin",
        }
    }
}

/// Molecular dynamics run configuration (user units: fs, K).
#[derive(Debug, Clone)]
pub struct MdConfig {
    /// Number of time steps.
    pub n_steps: usize,
    /// Time step in fs (default 0.5).
    pub dt_fs: f64,
    /// Thermostat (default NVE).
    pub thermostat: Thermostat,
    /// Target temperature in K; required for Berendsen and Langevin.
    pub temperature_k: Option<f64>,
    /// Initial-velocity temperature; `None` falls back to `temperature_k`,
    /// else zero velocities.
    pub init_temperature_k: Option<f64>,
    /// Explicit initial velocities (Bohr / a.u. time); overrides the draw.
    pub initial_velocities: Option<Array2<f64>>,
    /// RNG seed (default 0).
    pub seed: u64,
    /// Remove centre-of-mass motion (default true).
    pub remove_com_motion: bool,
    /// Record every `stride`-th step (default 1).
    pub stride: usize,
    /// Keep positions and velocities of recorded frames (default true).
    pub keep_trajectory: bool,
}

impl Default for MdConfig {
    fn default() -> Self {
        Self {
            n_steps: 0,
            dt_fs: 0.5,
            thermostat: Thermostat::Nve,
            temperature_k: None,
            init_temperature_k: None,
            initial_velocities: None,
            seed: 0,
            remove_com_motion: true,
            stride: 1,
            keep_trajectory: true,
        }
    }
}

impl MdConfig {
    /// Check parameters for consistency with a system of `n_atoms` atoms.
    pub fn validate(&self, n_atoms: usize) -> Result<(), FerricError> {
        let _ = n_atoms;
        todo!()
    }
}
