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
        let kind = kind.trim().to_ascii_lowercase();
        let thermostat = match kind.as_str() {
            "nve" => Self::Nve,
            "berendsen" => Self::Berendsen {
                tau_fs: tau_fs.ok_or("thermostat \"berendsen\" requires tau_fs")?,
            },
            "langevin" => Self::Langevin {
                friction_per_ps: friction_per_ps
                    .ok_or("thermostat \"langevin\" requires friction_per_ps")?,
            },
            other => {
                return Err(format!(
                    "unknown thermostat \"{other}\"; expected one of nve, berendsen, langevin"
                ))
            }
        };
        if tau_fs.is_some() && !matches!(thermostat, Self::Berendsen { .. }) {
            return Err(format!("tau_fs applies only to berendsen, not {kind}"));
        }
        if friction_per_ps.is_some() && !matches!(thermostat, Self::Langevin { .. }) {
            return Err(format!(
                "friction_per_ps applies only to langevin, not {kind}"
            ));
        }
        thermostat.check_parameters()?;
        Ok(thermostat)
    }

    /// Reject non-physical thermostat parameters. `tau_fs` must be positive
    /// (`+inf` is allowed and reproduces NVE); `friction_per_ps` must be
    /// finite and non-negative (zero reproduces NVE up to round-off).
    fn check_parameters(self) -> Result<(), String> {
        match self {
            Self::Nve => Ok(()),
            Self::Berendsen { tau_fs } if tau_fs > 0.0 => Ok(()),
            Self::Berendsen { tau_fs } => {
                Err(format!("berendsen tau_fs must be positive, got {tau_fs}"))
            }
            Self::Langevin { friction_per_ps }
                if friction_per_ps.is_finite() && friction_per_ps >= 0.0 =>
            {
                Ok(())
            }
            Self::Langevin { friction_per_ps } => Err(format!(
                "langevin friction_per_ps must be finite and non-negative, got {friction_per_ps}"
            )),
        }
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
    ///
    /// Requires `n_atoms > 0`, a finite positive `dt_fs`, `stride >= 1`,
    /// valid thermostat parameters, a target temperature for Berendsen and
    /// Langevin, finite non-negative temperatures, and finite `(n_atoms, 3)`
    /// initial velocities when given.
    pub fn validate(&self, n_atoms: usize) -> Result<(), FerricError> {
        let err = |m: String| FerricError::General(format!("invalid MD configuration: {m}"));
        if n_atoms == 0 {
            return Err(err("the system has no atoms".into()));
        }
        if !(self.dt_fs.is_finite() && self.dt_fs > 0.0) {
            return Err(err(format!(
                "dt_fs must be finite and positive, got {}",
                self.dt_fs
            )));
        }
        if self.stride == 0 {
            return Err(err("stride must be at least 1".into()));
        }
        self.thermostat.check_parameters().map_err(err)?;
        self.validate_temperatures().map_err(err)?;
        self.validate_initial_velocities(n_atoms).map_err(err)
    }

    fn validate_temperatures(&self) -> Result<(), String> {
        if self.thermostat != Thermostat::Nve && self.temperature_k.is_none() {
            return Err(format!(
                "thermostat {} requires temperature_k",
                self.thermostat.as_str()
            ));
        }
        for (name, t) in [
            ("temperature_k", self.temperature_k),
            ("init_temperature_k", self.init_temperature_k),
        ] {
            if let Some(t) = t.filter(|t| !(t.is_finite() && *t >= 0.0)) {
                return Err(format!("{name} must be finite and non-negative, got {t}"));
            }
        }
        Ok(())
    }

    fn validate_initial_velocities(&self, n_atoms: usize) -> Result<(), String> {
        let Some(v) = &self.initial_velocities else {
            return Ok(());
        };
        if v.dim() != (n_atoms, 3) {
            return Err(format!(
                "initial_velocities has shape {:?}, expected ({n_atoms}, 3)",
                v.dim()
            ));
        }
        if !v.iter().all(|x| x.is_finite()) {
            return Err("initial_velocities contains non-finite values".into());
        }
        Ok(())
    }
}
