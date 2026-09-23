//! Periodic-boundary-condition primitives for ferric — Stage 0 (library only).
//!
//! * [`lattice`] — [`Cell`](lattice::Cell): a [`Molecule`](ferric_core::mol::Molecule)
//!   plus three lattice vectors, with volume, reciprocal vectors, real-space
//!   image translations and reciprocal-space G vectors.
//! * [`ewald`] — Ewald nuclear repulsion (neutralising-background convention,
//!   matching PySCF `Cell.energy_nuc()`) and the Gamma-point Madelung constant
//!   (matching `pyscf.pbc.tools.madelung`).
//! * [`pair_ft`] — analytic Fourier transform of lattice-summed Gaussian pair
//!   densities, `P_mn(G) = Σ_L ∫ φ_m(r) φ_n(r−L) e^{−iG·r} dr`, in ferric's AO
//!   basis (the `PreparedBasis` order and normalization libint2 uses).
//!
//! Units: Bohr and Hartree throughout; G vectors in Bohr⁻¹.
//!
//! Nothing here is wired into the SCF, the CLI or the Python bindings yet —
//! see `reference/pbc/FINDINGS.md` for the staging plan.

pub mod ewald;
pub mod lattice;
pub mod pair_ft;

pub use ewald::{ewald_nuclear_repulsion, madelung_constant};
pub use lattice::Cell;
pub use pair_ft::pair_ft;
