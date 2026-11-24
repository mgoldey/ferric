//! Linear-response TDDFT: Tamm-Dancoff approximation (TDA/CIS) and
//! full Casida equations for closed-shell references (singlets or triplets)
//! and spin-conserving excitations of UHF/UKS references, including
//! range-separated hybrids, by a dense or a matrix-free Davidson eigensolver.

pub mod certificate;
pub mod davidson;
mod matvec;
pub mod tddft;
mod unrestricted;

pub use tddft::{
    run_tddft, run_tddft_on, ResponseSpin, TddftConfig, TddftMethod, TddftResult, TddftSolver,
};
