//! The MD driver: velocity Verlet, Berendsen and Langevin BAOAB.
//!
//! All propagation is in atomic units: positions in Bohr, velocities in
//! Bohr per a.u. time, masses in electron masses, energies in Hartree. One
//! force evaluation is made per step plus one at step 0.
//!
//! * Velocity Verlet: `v += -(dt/2) g/m; x += dt v; g = F(x); v += -(dt/2) g/m`.
//! * Berendsen: a velocity-Verlet step, then
//!   `v *= sqrt(1 + (dt/tau)(T0/T - 1))`. With `tau = inf` the factor is
//!   exactly 1 and the trajectory is bitwise NVE.
//! * Langevin BAOAB: `B(dt/2) A(dt/2) O(dt) A(dt/2) B(dt/2)` with the
//!   Ornstein-Uhlenbeck step `v = c v + sqrt((1 - c^2) kT/m) xi`,
//!   `c = exp(-gamma dt)`.
//!
//! The conserved quantity is `E_pot + E_kin - Q`, where `Q` accumulates the
//! kinetic energy the thermostat injected (Berendsen rescale, Langevin O
//! step); it is constant up to integration error for every thermostat.

use crate::config::{MdConfig, Thermostat};
use crate::provider::ForceProvider;
use crate::rng::Xoshiro256pp;
use crate::state::{MdFrame, MdObserver, MdResult, MdState};
use crate::system::MdSystem;
use crate::units::{AMU_TO_ELECTRON_MASS, AU_TIME_FS, BOLTZMANN_HARTREE_PER_K};
use crate::velocities::{kinetic_energy, maxwell_boltzmann, remove_com_momentum, temperature};
use ferric_core::error::FerricError;
use ndarray::Array2;

/// Femtoseconds per picosecond.
const FS_PER_PS: f64 = 1000.0;

/// Propagate `system` with forces from `provider` according to `config`,
/// calling each observer on every recorded frame.
///
/// Initial velocities are `config.initial_velocities` verbatim when given;
/// otherwise Maxwell-Boltzmann at `init_temperature_k` (falling back to
/// `temperature_k`), with the centre-of-mass velocity removed when
/// `remove_com_motion` is set and the system has more than one atom, then
/// rescaled so the instantaneous temperature is exactly the requested one.
/// With neither temperature set the velocities start at zero.
///
/// Degrees of freedom: `3N - 3` for NVE and Berendsen when
/// `remove_com_motion` is set and `N > 1`, else `3N`. Langevin always uses
/// `3N`: its noise does not conserve momentum, so COM motion is removed
/// only at initialisation.
///
/// Errors: invalid configuration, a provider/system atom-count mismatch, a
/// provider error (wrapped with the step index), or a non-finite or
/// misshapen energy/gradient, and any observer error.
pub fn run_md(
    system: &MdSystem,
    provider: &mut dyn ForceProvider,
    config: &MdConfig,
    observers: &mut [&mut dyn MdObserver],
) -> Result<MdResult, FerricError> {
    config.validate(system.n_atoms())?;
    check_system(system, provider)?;
    let mut dynamics = Dynamics::start(system, provider, config)?;
    let mut recorder = Recorder::new(config);
    recorder.record(system, dynamics.state(0, config.dt_fs), observers)?;
    for step in 1..=config.n_steps {
        dynamics.step(provider, config, step)?;
        if step % config.stride == 0 {
            recorder.record(system, dynamics.state(step, config.dt_fs), observers)?;
        }
    }
    let final_state = dynamics.state(config.n_steps, config.dt_fs);
    Ok(recorder.finish(final_state, dynamics.n_dof, provider.label(), config))
}

/// The provider and system must agree on the number of atoms.
fn check_system(system: &MdSystem, provider: &dyn ForceProvider) -> Result<(), FerricError> {
    let n = system.n_atoms();
    if provider.n_atoms() != n || system.coords.dim() != (n, 3) || system.masses_amu.len() != n {
        return Err(FerricError::General(format!(
            "MD system has {n} atoms (coordinates {:?}, {} masses) but the force provider \
             expects {}",
            system.coords.dim(),
            system.masses_amu.len(),
            provider.n_atoms()
        )));
    }
    Ok(())
}

/// Degrees of freedom used for temperatures and the initial rescale.
fn degrees_of_freedom(config: &MdConfig, n_atoms: usize) -> usize {
    let langevin = matches!(config.thermostat, Thermostat::Langevin { .. });
    if config.remove_com_motion && n_atoms > 1 && !langevin {
        3 * n_atoms - 3
    } else {
        3 * n_atoms
    }
}

/// Initial velocities per the rules documented on [`run_md`].
fn initial_velocities(
    config: &MdConfig,
    masses: &[f64],
    n_dof: usize,
    rng: &mut Xoshiro256pp,
) -> Array2<f64> {
    if let Some(v) = &config.initial_velocities {
        return v.clone();
    }
    let n = masses.len();
    let Some(t0) = config.init_temperature_k.or(config.temperature_k) else {
        return Array2::zeros((n, 3));
    };
    let mut v = maxwell_boltzmann(masses, t0, rng);
    if config.remove_com_motion && n > 1 {
        remove_com_momentum(masses, &mut v);
    }
    let t = temperature(kinetic_energy(masses, &v), n_dof);
    if t > 0.0 {
        v *= (t0 / t).sqrt();
    }
    v
}

/// One force evaluation: wraps provider errors with the step index and
/// rejects non-finite or misshapen results.
fn evaluate(
    provider: &mut dyn ForceProvider,
    x: &Array2<f64>,
    step: usize,
) -> Result<(f64, Array2<f64>), FerricError> {
    let (e, g) = provider.energy_gradient(x).map_err(|err| {
        FerricError::General(format!("MD step {step}: force evaluation failed: {err}"))
    })?;
    if g.dim() != x.dim() {
        return Err(FerricError::General(format!(
            "MD step {step}: gradient has shape {:?}, expected {:?}",
            g.dim(),
            x.dim()
        )));
    }
    if !e.is_finite() || !g.iter().all(|v| v.is_finite()) {
        return Err(FerricError::General(format!(
            "MD step {step}: non-finite energy or gradient (E = {e})"
        )));
    }
    Ok((e, g))
}

/// Phase-space state plus integrator bookkeeping.
struct Dynamics {
    x: Array2<f64>,
    v: Array2<f64>,
    g: Array2<f64>,
    e: f64,
    /// Accumulated thermostat-injected kinetic energy.
    q: f64,
    masses: Vec<f64>,
    /// Time step in a.u. time.
    dt: f64,
    n_dof: usize,
    rng: Xoshiro256pp,
}

impl Dynamics {
    /// Initial velocities plus the step-0 force evaluation.
    fn start(
        system: &MdSystem,
        provider: &mut dyn ForceProvider,
        config: &MdConfig,
    ) -> Result<Self, FerricError> {
        let n_dof = degrees_of_freedom(config, system.n_atoms());
        let mut rng = Xoshiro256pp::seed_from_u64(config.seed);
        let masses: Vec<f64> = system
            .masses_amu
            .iter()
            .map(|m| m * AMU_TO_ELECTRON_MASS)
            .collect();
        let v = initial_velocities(config, &masses, n_dof, &mut rng);
        let (e, g) = evaluate(provider, &system.coords, 0)?;
        Ok(Self {
            x: system.coords.clone(),
            v,
            g,
            e,
            q: 0.0,
            masses,
            dt: config.dt_fs / AU_TIME_FS,
            n_dof,
            rng,
        })
    }

    fn step(
        &mut self,
        provider: &mut dyn ForceProvider,
        config: &MdConfig,
        step: usize,
    ) -> Result<(), FerricError> {
        let t0 = config.temperature_k.unwrap_or(0.0);
        match config.thermostat {
            Thermostat::Nve => self.verlet(provider, step),
            Thermostat::Berendsen { tau_fs } => {
                self.verlet(provider, step)?;
                self.berendsen(tau_fs / AU_TIME_FS, t0);
                Ok(())
            }
            Thermostat::Langevin { friction_per_ps } => {
                let gamma = friction_per_ps * AU_TIME_FS / FS_PER_PS;
                self.baoab(provider, step, gamma, t0)
            }
        }
    }

    fn verlet(&mut self, provider: &mut dyn ForceProvider, step: usize) -> Result<(), FerricError> {
        self.kick(0.5 * self.dt);
        self.drift(self.dt);
        self.force(provider, step)?;
        self.kick(0.5 * self.dt);
        Ok(())
    }

    fn baoab(
        &mut self,
        provider: &mut dyn ForceProvider,
        step: usize,
        gamma: f64,
        t0: f64,
    ) -> Result<(), FerricError> {
        self.kick(0.5 * self.dt);
        self.drift(0.5 * self.dt);
        self.ornstein_uhlenbeck(gamma, t0);
        self.drift(0.5 * self.dt);
        self.force(provider, step)?;
        self.kick(0.5 * self.dt);
        Ok(())
    }

    /// `v -= h g / m`.
    fn kick(&mut self, h: f64) {
        let rows = self.g.rows().into_iter().zip(&self.masses);
        for (mut v, (g, &m)) in self.v.rows_mut().into_iter().zip(rows) {
            v.scaled_add(-h / m, &g);
        }
    }

    /// `x += h v`.
    fn drift(&mut self, h: f64) {
        self.x.scaled_add(h, &self.v);
    }

    fn force(&mut self, provider: &mut dyn ForceProvider, step: usize) -> Result<(), FerricError> {
        (self.e, self.g) = evaluate(provider, &self.x, step)?;
        Ok(())
    }

    /// Berendsen velocity rescale towards `t0`, `tau` in a.u. time.
    fn berendsen(&mut self, tau: f64, t0: f64) {
        let ke = kinetic_energy(&self.masses, &self.v);
        let t = temperature(ke, self.n_dof);
        if t <= 0.0 {
            return;
        }
        let lambda = (1.0 + (self.dt / tau) * (t0 / t - 1.0)).max(0.0).sqrt();
        self.v *= lambda;
        self.q += kinetic_energy(&self.masses, &self.v) - ke;
    }

    /// Exact Ornstein-Uhlenbeck step over a full `dt`.
    fn ornstein_uhlenbeck(&mut self, gamma: f64, t0: f64) {
        let ke = kinetic_energy(&self.masses, &self.v);
        let c = (-gamma * self.dt).exp();
        let kt = BOLTZMANN_HARTREE_PER_K * t0;
        for (mut v, &m) in self.v.rows_mut().into_iter().zip(&self.masses) {
            let sigma = ((1.0 - c * c) * kt / m).sqrt();
            for vk in v.iter_mut() {
                *vk = c * *vk + sigma * self.rng.next_gaussian();
            }
        }
        self.q += kinetic_energy(&self.masses, &self.v) - ke;
    }

    fn state(&self, step: usize, dt_fs: f64) -> MdState {
        let kinetic = kinetic_energy(&self.masses, &self.v);
        let total = self.e + kinetic;
        MdState {
            frame: MdFrame {
                step,
                time_fs: step as f64 * dt_fs,
                potential: self.e,
                kinetic,
                total,
                conserved: total - self.q,
                temperature_k: temperature(kinetic, self.n_dof),
            },
            coords: self.x.clone(),
            velocities: self.v.clone(),
            gradient: self.g.clone(),
        }
    }
}

/// Collects recorded frames and drives the observers.
struct Recorder {
    keep_trajectory: bool,
    frames: Vec<MdFrame>,
    positions: Vec<Array2<f64>>,
    velocities: Vec<Array2<f64>>,
}

impl Recorder {
    fn new(config: &MdConfig) -> Self {
        Self {
            keep_trajectory: config.keep_trajectory,
            frames: Vec::new(),
            positions: Vec::new(),
            velocities: Vec::new(),
        }
    }

    fn record(
        &mut self,
        system: &MdSystem,
        state: MdState,
        observers: &mut [&mut dyn MdObserver],
    ) -> Result<(), FerricError> {
        for obs in observers.iter_mut() {
            obs.on_frame(system, &state)?;
        }
        self.frames.push(state.frame);
        if self.keep_trajectory {
            self.positions.push(state.coords);
            self.velocities.push(state.velocities);
        }
        Ok(())
    }

    fn finish(
        self,
        final_state: MdState,
        n_dof: usize,
        label: String,
        config: &MdConfig,
    ) -> MdResult {
        MdResult {
            frames: self.frames,
            positions: self.positions,
            velocities: self.velocities,
            final_state,
            n_dof,
            n_force_evaluations: config.n_steps + 1,
            label,
            config: config.clone(),
        }
    }
}
