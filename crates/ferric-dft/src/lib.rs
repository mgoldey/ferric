//! Kohn–Sham DFT: exchange–correlation on numerical grids.
//!
//! This crate supplies the DFT half of the engine:
//!
//! - **libxc bridge** ([`libxc`]) — LDA/GGA/hybrid/range-separated functionals.
//! - **Integration grids** ([`grid`], [`lebedev`], [`radial`]) — Becke-partitioned
//!   atomic grids with Treutler–Ahlrichs radial and Lebedev angular quadrature;
//!   [`becke`] holds the fuzzy-cell partition (and its nuclear gradients).
//! - **Density on the grid** ([`density_on_grid`], [`ao_grid`]) — AO values χ and
//!   gradients ∇χ at grid points, and ρ/∇ρ/σ from a density matrix.
//! - **Potentials** ([`vxc`]) — the V_xc Fock contribution (closed/open-shell);
//!   [`vv10`] nonlocal correlation; [`fxc`] the XC kernel.
//! - **Gradients** ([`gradient`]) — analytical XC nuclear gradients with grid response.
//! - **High-level KS glue** ([`ks`], [`xc_trait`]) — caches grid + χ, adds V_xc in place.
//! - **Constrained DFT weights** ([`cdft`]) — the grid-Becke weight operator W and
//!   fragment populations used by the cDFT solver in `ferric-scf`.

// Lint policy for numerical grid/XC code:
// - excessive_precision: Lebedev node/weight tables transcribed at full source
//   precision on purpose.
// - needless_range_loop: grid-point and spin/Cartesian-component loops read
//   clearer with explicit indices.
// - identity_op: `stride * g + 0` keeps the component index aligned with the
//   `+ 1` / `+ 2` rows below it; the `+ 0` is intentional visual structure.
// - op_ref: `a + &b + &b.t()` references `b` because it is reused on the same
//   line; dropping the `&` would move `b` and break the second use.
#![allow(clippy::excessive_precision)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::identity_op)]
#![allow(clippy::op_ref)]

/// AO basis function values and derivatives on grid points.
pub mod ao_grid;
/// Grid-Becke weight operator for constrained DFT fragment populations.
pub mod cdft;
/// libxc bindings: functional evaluation (exc, vxc, fxc) for LDA/GGA/mGGA/hybrid.
pub mod libxc;
/// Becke fuzzy-cell atomic partitioning and its nuclear gradient.
pub mod becke;
/// Electron density ρ, gradient ∇ρ, and kinetic-energy density τ on grid points.
pub mod density_on_grid;
/// Analytical XC nuclear gradients with grid response terms.
pub mod gradient;
/// Grid point storage and atom-to-grid-point mapping.
pub mod grid;
/// High-level Kohn-Sham driver: grid+χ cache, V_xc accumulation into Fock matrix.
pub mod ks;
/// Lebedev angular quadrature nodes and weights (up to order 131).
pub mod lebedev;
/// Grid pruning strategies for reducing angular quadrature near nuclei.
pub mod prune;
/// Radial quadrature: Treutler-Ahlrichs M4 and Mura-Knowles log grids.
pub mod radial;
/// VV10 nonlocal correlation functional.
pub mod vv10;
/// V_xc Fock-matrix contribution from the XC potential on the grid.
pub mod vxc;
/// XC kernel (f_xc) for Newton-step SCF solvers (LDA and GGA).
pub mod fxc;
/// [`XcFunctional`](xc_trait::XcFunctional) trait: abstract interface to XC evaluators.
pub mod xc_trait;

/// Crate-wide serialization for tests that SET or transitively READ the
/// process-global `FERRIC_MEM_BUDGET_GB` env var. `ks` and `ao_grid` each
/// kept a private lock, which cannot stop a cross-module race against a
/// test that resolves the budget mid-flight (fxc kernel construction runs
/// the checked AO-grid budget resolve). One shared lock closes the race;
/// poisoning is tolerated via `into_inner` at use sites.
#[cfg(test)]
pub(crate) static TEST_BUDGET_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
