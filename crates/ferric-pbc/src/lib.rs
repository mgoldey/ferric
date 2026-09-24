//! Periodic-boundary-condition primitives for ferric — Stages 0-1 (library only).
//!
//! * [`lattice`] — [`Cell`]: a [`Molecule`](ferric_core::mol::Molecule)
//!   plus three lattice vectors, with volume, reciprocal vectors, real-space
//!   image translations and reciprocal-space G vectors.
//! * [`ewald`] — Ewald nuclear repulsion (neutralising-background convention,
//!   matching PySCF `Cell.energy_nuc()`) and the Gamma-point Madelung constant
//!   (matching `pyscf.pbc.tools.madelung`).
//! * [`mod@pair_ft`] — analytic Fourier transform of lattice-summed Gaussian pair
//!   densities, `P_mn(G) = Σ_L ∫ φ_m(r) φ_n(r−L) e^{−iG·r} dr`, in ferric's AO
//!   basis (the `PreparedBasis` order and normalization libint2 uses).
//! * [`hcore`] — Stage 1: Gamma-point lattice-summed `S`, `T`, Ewald-split
//!   nuclear attraction (Gaussian nuclei through the shifted 3-centre engine
//!   + analytic pair FT) and `E_nn` ([`PeriodicHcore`]).
//! * [`budget`] — memory gating: every large buffer is reserved against
//!   ferric's unified budget before allocation (Stage 1 step 10).
//! * [`dense_aft`] — TEST/ORACLE ONLY: dense pure-AFT `nao⁴` ERI tensor with
//!   `JBuilder`/`KBuilder` impls (Madelung `exxdiv`) for
//!   `ferric_scf::rhf::solve_rhf_injected`; hard size cap.
//! * [`rsgdf`] — Stage 1 step 9: Gamma-point range-separated Gaussian density
//!   fitting (PySCF-RSGDF G = 0 convention, eig-with-lindep metric solve) and
//!   `JBuilder`/`KBuilder` impls on the fitted B tensor (Madelung `exxdiv`).
//! * [`mp2`] — Stage 6: Gamma-point closed-shell MP2 on the RS-GDF B (the
//!   SCF's own B → `B[k,ia]` → ferric-mp2's `spin_components_from_b_ov`) or
//!   the dense-AFT oracle, with explicit Madelung-shifted denominators.
//!
//! Units: Bohr and Hartree throughout; G vectors in Bohr⁻¹.
//!
//! Not wired into the CLI or the Python bindings yet — see
//! `reference/pbc/stage1-design.md` for the staging plan.

pub mod budget;
pub mod dense_aft;
pub mod ewald;
pub mod hcore;
pub mod lattice;
pub mod mp2;
pub mod pair_ft;
pub mod rsgdf;

pub use dense_aft::{DenseAftEri, ExxDiv};
pub use ewald::{ewald_nuclear_repulsion, madelung_constant};
pub use hcore::{periodic_hcore, PeriodicHcore, PeriodicHcoreConfig};
pub use lattice::Cell;
pub use mp2::{gamma_mp2, GammaMp2Config, GammaMp2Integrals, GammaMp2Result, Mp2Denominators};
pub use pair_ft::pair_ft;
pub use rsgdf::{RsGdf, RsGdfConfig};
