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

// Compile-time guard against the one MPI feature combination that is silently
// WRONG rather than merely unsupported: `ferric-core/mpi` ON while this crate's
// own `mpi` is OFF.
//
// In that state `ParallelContext` reports a real multi-rank world, so `DfJ`/`DfK`
// stripe the aux band per rank (`new_banded(Some(ctx))`) — but the cross-rank
// Allreduce that sums the per-rank partial J/K back into the full matrix is
// gated on THIS crate's `mpi` feature and never compiles in. Each rank then
// converges "successfully" to a different, wrong energy built from only its own
// band (measured: -np 2 water RHF gave ~+108111 and ~+108176 Ha on two ranks).
//
// No crate's own `mpi` feature can reach this state — ferric-scf/mp2/rpa/gw all
// chain `ferric-scf/mpi` — so this only fires for a hand-rolled
// `--features ferric-core/mpi`. Failing at compile time turns a wrong number
// into a build error.
const _: () = assert!(
    !(ferric_core::parallel::MPI_ENABLED && !cfg!(feature = "mpi")),
    "ferric-core/mpi is enabled but ferric-scf/mpi is not. This combination \
     silently produces WRONG energies: ParallelContext reports a multi-rank \
     world so DF-J/DF-K stripe the aux band, but ferric-scf's cross-rank \
     Allreduce is cfg'd out, leaving each rank with only its own band. \
     Enable ferric-scf/mpi (or a downstream crate's mpi feature, which chains \
     it) instead of ferric-core/mpi alone."
);

/// SCF convergence result types and spin-channel descriptor.
pub mod result;
pub use result::{ScfResult, Spin};

/// Post-SCF one-electron property evaluation (dipole, charges, ESP).
pub mod properties;

/// Schwarz-based integral screening re-exports.
pub mod screening;
/// Range-separated hybrid ω-tuning via IP/EA condition.
pub mod omega_tuning;
/// Semi-canonical orbital transformation for frozen-core post-HF.
pub mod semicanonical;
/// QQR distance-dependent screening adapter for the SCF layer.
pub mod qqr;
/// Significant and density-screened shell-pair lists.
pub mod pairs;
/// J/K builder traits and composite Fock matrix assembly.
pub mod fock;
pub(crate) mod fock_assembly;
pub(crate) mod driver;
/// COSMO conductor-like implicit solvation model.
pub mod cosmo;
pub use cosmo::{CosmoCavity, CosmoConfig, CosmoResult};
/// MPI reduction helpers for distributed Fock matrix assembly.
pub mod reduce;
/// Shell-quartet work distribution for integral-direct builds.
pub mod quartet_scatter;
/// Integral-direct Coulomb (J) matrix builder.
pub mod direct_j;
/// Integral-direct exchange (K) matrix builder.
pub mod direct_k;
/// Combined integral-direct J+K Fock builder.
pub mod direct_jk;
/// Re-export: [`EnginePool`](ferric_integrals::engine_pool::EnginePool) moved
/// down to `ferric-integrals` so integral-level code (schwarz, 3-index) can use
/// it too. Kept here so existing `ferric_scf::engine_pool::…` paths still work.
pub use ferric_integrals::engine_pool;
/// Density-fitted Coulomb (J) matrix builder (RI-J).
pub mod df_j;
/// Density-fitted exchange (K) matrix builder (RI-K).
pub mod df_k;
/// LinK exchange builder: linear-scaling K via Schwarz-screened column lists.
pub mod link_k;
/// DIIS convergence accelerator for SCF iterations.
pub mod diis;
/// Initial guess generators: core Hamiltonian, SAD, read-in.
pub mod guess;
/// Fermi-smearing (fractional occupation) for metallic/near-degenerate systems.
pub mod smearing;
/// Closed-shell restricted Hartree-Fock solver.
pub mod rhf;
/// Newton-step RHF solver with exact orbital Hessian.
pub mod rhf_newton;
/// Unrestricted Hartree-Fock solver (α/β spin channels).
pub mod uhf;
pub use uhf::{solve_uhf, solve_uhf_fockmod, UhfConfig};
/// Constrained DFT solver: charge/spin constraints via Becke-weight operator.
pub mod cdft_driver;
pub use cdft_driver::{solve_cdft_uhf, CdftResult};
/// cDFT electronic coupling (H_ab) via the Wu–Van Voorhis scheme.
pub mod cdft_coupling;
pub use cdft_coupling::{coupling_hab, DiabaticState, HabResult};
/// Restricted open-shell Hartree-Fock solver (ROHF).
pub mod rohf;
pub use rohf::{solve_rohf, RohfConfig};
/// Maximum Overlap Method for tracking orbital character across SCF iterations.
pub mod mom;
/// Newton-step ROHF solver with f_xc kernel acceleration.
pub mod rohf_newton;
/// Newton-step UHF/UKS solver with coupled α/β orbital rotations.
pub mod uhf_newton;
/// Davidson eigensolver adapted for local SCF orbital optimization.
pub mod davidson_local;
/// Augmented Hessian (AH) ROHF solver for difficult convergence cases.
pub mod rohf_ah;
/// Analytical nuclear gradients for RHF, UHF, and ROHF.
pub mod gradient;
pub use gradient::{rhf_gradient, rohf_gradient, uhf_gradient};
/// Analytical nuclear gradients for Kohn-Sham DFT (XC + grid response).
pub mod ks_gradient;
pub use ks_gradient::ks_gradient_closed;
/// Geometry optimization via L-BFGS with energy/gradient convergence.
pub mod optimize;
/// Harmonic vibrational frequencies from finite-difference Hessian.
pub mod frequencies;
/// Analytic RHF second derivatives (Hessian) — nuclear term implemented,
/// electronic terms stubbed pending LIBINT2_MAX_DERIV_ORDER >= 2.
pub mod hessian;
pub use frequencies::{
    harmonic_frequencies, FrequencyConfig, FrequencyReference, FrequencyResult,
};
/// Continuous Fast Multipole Method (CFMM) for long-range Coulomb.
///
/// CORRECT BUT NOT PRODUCTION — still gated behind the off-by-default
/// `cfmm-incomplete` feature.
///
/// The integral kernels that used to be unimplemented stubs (returning an
/// identically-zero J) are now implemented and cross-checked against
/// independent constructions: in the trivial all-near-field limit `CfmmJ`
/// reproduces the direct/dense J to ~1e-14, and with the far field engaged
/// on alkane_10/STO-3G it agrees to 8.2e-5 against max|J| = 24.5.
///
/// The gate REMAINS because correctness is not the only bar for a public
/// `JBuilder`:
///
/// * it is currently far SLOWER than [`direct_j::DirectJ`] (the near field is
///   an unscreened, unsymmetrized ordered-pair loop) — there is no measured
///   scaling benefit, so nothing should select it for performance;
/// * the well-separatedness test, while now extent-aware, has only been
///   validated on compact bases (STO-3G / cc-pVDZ);
/// * the far field is flat (leaf-to-leaf M2L), with no hierarchical
///   M2M/L2L pass, so it does not yet have FMM's asymptotic behaviour.
///
/// See the `cfmm` module docs for the full status list.
#[cfg(feature = "cfmm-incomplete")]
pub mod cfmm;
/// QM/MM system setup: atom selection, link atoms, embedding charges.
pub mod qmmm;
pub use qmmm::{QmSelection, QmmmAtom, QmmmSystem};
/// Jacob's ladder solver: run a sequence of methods (HF→DFT→MP2→…) reusing orbitals.
pub mod ladder;
pub use ladder::{solve_rhf_ladder, default_ladder, ksdft_ladder, Rung, LadderResult, RungOutcome};
