//! Facade crate re-exporting the ferric quantum chemistry engine's public API.
//!
//! Instead of importing from individual workspace crates, use:
//!
//! ```no_run
//! use ferric::prelude::*;
//!
//! let mol: Molecule = "3\nwater\nO 0 0 0.117790\nH 0 0.755453 -0.471161\nH 0 -0.755453 -0.471161\n".parse().unwrap();
//! let bs = basis::bundled("sto-3g").unwrap();
//! let prep = PreparedBasis::new(&mol, &bs).unwrap();
//! let op = Operator::coulomb();
//! let bounds = SchwarzBounds::compute(op, &prep).unwrap();
//! let ctx = ParallelContext::default();
//! let result = solve_rhf(&ctx, &mol, &prep, op, &bounds, &RhfConfig::default()).unwrap();
//! println!("{result}");
//! ```

pub use ferric_core as core;
pub use ferric_integrals as integrals;
pub use ferric_scf as scf;
pub use ferric_dft as dft;
pub use ferric_mp2 as mp2;
pub use ferric_rpa as rpa;
pub use ferric_gw as gw;
pub use ferric_cc as cc;
pub use ferric_ci as ci;
pub use ferric_pcm as pcm;
pub use ferric_export as export;
pub use ferric_tensors as tensors;
pub use ferric_quadrature as quadrature;

/// Common imports for a typical ferric calculation.
pub mod prelude {
    pub use ferric_core::basis::{self, BasisSet, Shell};
    pub use ferric_core::error::FerricError;
    pub use ferric_core::mol::{Atom, Molecule};
    pub use ferric_core::parallel::ParallelContext;

    pub use ferric_integrals::basis_bridge::PreparedBasis;
    pub use ferric_integrals::operator::Operator;

    pub use ferric_scf::rhf::{solve_rhf, RhfConfig};
    pub use ferric_scf::uhf::{solve_uhf, UhfConfig};
    pub use ferric_scf::screening::SchwarzBounds;
    pub use ferric_scf::result::{ScfResult, ScfExit};
}
