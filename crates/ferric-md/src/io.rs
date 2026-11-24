//! Trajectory and energy writers implementing [`MdObserver`].

use crate::state::{MdObserver, MdState};
use crate::system::MdSystem;
use crate::units::BOHR_TO_ANGSTROM;
use ferric_core::error::FerricError;
use std::io::{BufWriter, Write};
use std::path::Path;

fn create_file(path: &Path) -> Result<BufWriter<std::fs::File>, FerricError> {
    let file = std::fs::File::create(path).map_err(|e| {
        FerricError::General(format!("cannot create MD output {}: {e}", path.display()))
    })?;
    Ok(BufWriter::new(file))
}

/// Multi-frame XYZ writer (Angstrom); comment line
/// `step=.. time_fs=.. e_pot=.. e_tot=..`.
///
/// Each frame is flushed as it is written, so the file is readable while
/// the run is in progress and write errors surface at the frame that hit
/// them.
pub struct XyzTrajectoryWriter {
    out: BufWriter<std::fs::File>,
}

impl XyzTrajectoryWriter {
    /// Create (truncate) the file at `path`.
    pub fn create(path: &Path) -> Result<Self, FerricError> {
        Ok(Self {
            out: create_file(path)?,
        })
    }
}

impl MdObserver for XyzTrajectoryWriter {
    fn on_frame(&mut self, system: &MdSystem, state: &MdState) -> Result<(), FerricError> {
        let f = &state.frame;
        writeln!(self.out, "{}", system.symbols.len())?;
        writeln!(
            self.out,
            "step={} time_fs={:.6} e_pot={:.12} e_tot={:.12}",
            f.step, f.time_fs, f.potential, f.total
        )?;
        for (sym, r) in system.symbols.iter().zip(state.coords.rows()) {
            let [x, y, z] = [r[0], r[1], r[2]].map(|c| c * BOHR_TO_ANGSTROM);
            writeln!(self.out, "{sym:<3} {x:18.12} {y:18.12} {z:18.12}")?;
        }
        self.out.flush()?;
        Ok(())
    }
}

/// CSV writer with header
/// `step,time_fs,e_pot,e_kin,e_tot,conserved,temperature_k`.
///
/// Rows are flushed as they are written.
pub struct EnergyCsvWriter {
    out: BufWriter<std::fs::File>,
}

impl EnergyCsvWriter {
    /// Create (truncate) the file at `path` and write the header.
    pub fn create(path: &Path) -> Result<Self, FerricError> {
        let mut out = create_file(path)?;
        writeln!(
            out,
            "step,time_fs,e_pot,e_kin,e_tot,conserved,temperature_k"
        )?;
        out.flush()?;
        Ok(Self { out })
    }
}

impl MdObserver for EnergyCsvWriter {
    fn on_frame(&mut self, _system: &MdSystem, state: &MdState) -> Result<(), FerricError> {
        let f = &state.frame;
        writeln!(
            self.out,
            "{},{:.6},{:.12},{:.12},{:.12},{:.12},{:.6}",
            f.step, f.time_fs, f.potential, f.kinetic, f.total, f.conserved, f.temperature_k
        )?;
        self.out.flush()?;
        Ok(())
    }
}
