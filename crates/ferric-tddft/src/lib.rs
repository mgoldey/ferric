//! Linear-response TDDFT: Tamm-Dancoff approximation (TDA/CIS) and
//! full Casida equations for closed-shell references, singlets or triplets,
//! including range-separated hybrids, by a dense or a
//! matrix-free Davidson eigensolver.

pub mod certificate;
pub mod davidson;
mod matvec;
pub mod tddft;

pub use tddft::{run_tddft, ResponseSpin, TddftConfig, TddftMethod, TddftResult, TddftSolver};
