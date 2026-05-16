//! Self-consistent field (SCF) solvers for Hartree-Fock theory.
//!
//! This crate implements:
//!
//! - **Closed-shell RHF** ([`rhf::solve_rhf`]) with DIIS convergence acceleration
//!   and Schwarz integral screening
//! - **Analytical RHF nuclear gradients** ([`gradient::rhf_gradient`]) including
//!   one-electron, two-electron, and nuclear repulsion contributions
//! - **DIIS extrapolation** ([`diis::Diis`]) for accelerating SCF convergence
//! - **Schwarz screening** ([`screening::SchwarzBounds`]) for integral prescreening
//! - **Core Hamiltonian initial guess** ([`guess::hcore_guess`])

pub mod screening;
pub mod qqr;
pub mod pairs;
pub mod fock;
pub mod direct_j;
pub mod direct_k;
pub mod direct_jk;
pub mod df_j;
pub mod df_k;
pub mod link_k;
pub mod diis;
pub mod guess;
pub mod rhf;
pub mod gradient;
pub mod optimize;
pub mod cfmm;
