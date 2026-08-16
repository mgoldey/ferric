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

// needless_range_loop: DIIS coefficient and MO-index loops read clearer with
// explicit indices than with iterator/enumerate chains.
#![allow(clippy::needless_range_loop)]

pub mod result;
pub use result::{ScfResult, Spin};

pub mod properties;

pub mod screening;
pub mod omega_tuning;
pub mod semicanonical;
pub mod qqr;
pub mod pairs;
pub mod fock;
pub(crate) mod fock_assembly;
pub(crate) mod driver;
pub mod cosmo;
pub use cosmo::{CosmoCavity, CosmoConfig, CosmoResult};
pub mod reduce;
pub mod quartet_scatter;
pub mod direct_j;
pub mod direct_k;
pub mod direct_jk;
/// Re-export: [`EnginePool`](ferric_integrals::engine_pool::EnginePool) moved
/// down to `ferric-integrals` so integral-level code (schwarz, 3-index) can use
/// it too. Kept here so existing `ferric_scf::engine_pool::…` paths still work.
pub use ferric_integrals::engine_pool;
pub mod df_j;
pub mod df_k;
pub mod link_k;
pub mod diis;
pub mod guess;
pub mod smearing;
pub mod rhf;
pub mod rhf_newton;
pub mod uhf;
pub use uhf::{solve_uhf, solve_uhf_fockmod, UhfConfig};
pub mod cdft_driver;
pub use cdft_driver::{solve_cdft_uhf, CdftResult};
pub mod cdft_coupling;
pub use cdft_coupling::{coupling_hab, DiabaticState, HabResult};
pub mod rohf;
pub use rohf::{solve_rohf, RohfConfig};
pub mod mom;
pub mod rohf_newton;
pub mod uhf_newton;
pub mod davidson_local;
pub mod rohf_ah;
pub mod gradient;
pub use gradient::{rhf_gradient, rohf_gradient, uhf_gradient};
pub mod ks_gradient;
pub use ks_gradient::ks_gradient_closed;
pub mod optimize;
pub mod frequencies;
pub use frequencies::{
    harmonic_frequencies, FrequencyConfig, FrequencyReference, FrequencyResult,
};
pub mod cfmm;
pub mod qmmm;
pub use qmmm::{QmSelection, QmmmAtom, QmmmSystem};
pub mod ladder;
pub use ladder::{solve_rhf_ladder, default_ladder, ksdft_ladder, Rung, LadderResult, RungOutcome};
