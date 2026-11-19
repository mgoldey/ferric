//! Unit constants for molecular dynamics (atomic units internally).

pub use ferric_core::conformers::BOLTZMANN_HARTREE_PER_K;
pub use ferric_core::units::{ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM};

/// Atomic mass unit expressed in electron masses (1 / CODATA m_e in u).
pub const AMU_TO_ELECTRON_MASS: f64 = 1.0 / 5.485_799_090_65e-4;

/// One atomic unit of time in femtoseconds.
pub const AU_TIME_FS: f64 = 2.418_884_326_585_7e-2;
