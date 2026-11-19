//! Molecular dynamics core: integrators, thermostats and trajectory output.
//!
//! The crate is method-agnostic. Any energy/gradient source implementing
//! [`ForceProvider`] (ab initio, force field, or a closure via
//! [`FnProvider`]) can be propagated by [`run_md`] with velocity Verlet (NVE),
//! Berendsen (equilibration only) or Langevin BAOAB dynamics. Internally
//! everything is in atomic units (Bohr, Hartree, electron masses, a.u. time);
//! [`MdConfig`] takes user units (fs, K, ps^-1).

pub mod config;
pub mod driver;
pub mod io;
pub mod provider;
pub mod rng;
pub mod state;
pub mod system;
pub mod units;
pub mod velocities;

pub use config::{MdConfig, Thermostat};
pub use driver::run_md;
pub use io::{EnergyCsvWriter, XyzTrajectoryWriter};
pub use provider::{FnProvider, ForceProvider};
pub use rng::Xoshiro256pp;
pub use state::{MdFrame, MdObserver, MdResult, MdState};
pub use system::MdSystem;
pub use units::{
    AMU_TO_ELECTRON_MASS, ANGSTROM_TO_BOHR, AU_TIME_FS, BOHR_TO_ANGSTROM, BOLTZMANN_HARTREE_PER_K,
};
pub use velocities::{kinetic_energy, maxwell_boltzmann, remove_com_momentum, temperature};
