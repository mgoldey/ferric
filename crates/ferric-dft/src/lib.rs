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

pub mod ao_grid;
pub mod cdft;
pub mod libxc;
pub mod becke;
pub mod density_on_grid;
pub mod gradient;
pub mod grid;
pub mod ks;
pub mod lebedev;
pub mod radial;
pub mod vv10;
pub mod vxc;
pub mod fxc;
pub mod xc_trait;
