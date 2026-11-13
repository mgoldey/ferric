//! Linear-response TDDFT: Tamm-Dancoff approximation (TDA/CIS) and
//! full Casida equations for closed-shell references, by a dense or a
//! matrix-free Davidson eigensolver.

pub mod certificate;
pub mod davidson;
mod matvec;
pub mod tddft;

pub use tddft::{run_tddft, TddftConfig, TddftMethod, TddftResult, TddftSolver};
