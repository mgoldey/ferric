//! Linear-response TDDFT: Tamm-Dancoff approximation (TDA/CIS) and
//! full Casida equations for closed-shell references.

pub mod tddft;

pub use tddft::{run_tddft, TddftConfig, TddftMethod, TddftResult};
