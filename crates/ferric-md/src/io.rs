//! Trajectory and energy writers implementing [`MdObserver`].

use crate::state::{MdObserver, MdState};
use crate::system::MdSystem;
use ferric_core::error::FerricError;
use std::io::BufWriter;
use std::path::Path;

/// Multi-frame XYZ writer (Angstrom); comment line
/// `step=.. time_fs=.. e_pot=.. e_tot=..`.
pub struct XyzTrajectoryWriter {
    out: BufWriter<std::fs::File>,
}

impl XyzTrajectoryWriter {
    /// Create (truncate) the file at `path`.
    pub fn create(path: &Path) -> Result<Self, FerricError> {
        let _ = path;
        todo!()
    }
}

impl MdObserver for XyzTrajectoryWriter {
    fn on_frame(&mut self, system: &MdSystem, state: &MdState) -> Result<(), FerricError> {
        let _ = (&self.out, system, state);
        todo!()
    }
}

/// CSV writer with header
/// `step,time_fs,e_pot,e_kin,e_tot,conserved,temperature_k`.
pub struct EnergyCsvWriter {
    out: BufWriter<std::fs::File>,
}

impl EnergyCsvWriter {
    /// Create (truncate) the file at `path` and write the header.
    pub fn create(path: &Path) -> Result<Self, FerricError> {
        let _ = path;
        todo!()
    }
}

impl MdObserver for EnergyCsvWriter {
    fn on_frame(&mut self, system: &MdSystem, state: &MdState) -> Result<(), FerricError> {
        let _ = (&self.out, system, state);
        todo!()
    }
}
