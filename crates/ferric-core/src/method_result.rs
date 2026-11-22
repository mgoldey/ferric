//! [`MethodResult`](crate::method_result::MethodResult): what every method result with a total energy and a
//! validation certificate offers, whatever the method.
//!
//! The trait is deliberately small (ARCHITECTURE.md, "Shared structure"): it
//! is separate from the orbital-reference view (spin, orbitals, eigenvalues,
//! occupations), since an MP2 result is a method result but not a reference.
//!
//! Implementers are the types that hold both numbers together. No kernel
//! result does today: a certificate is built by a free function from the
//! kernel result plus inputs the result does not keep (the molecule, the
//! config, the reference's certificate, the auxiliary basis name), e.g.
//! `ferric_scf::certificate::scf_certificate`. The result types that do hold
//! both are the `ferric-python` result classes, which implement this trait
//! and generate their shared Python getters from it.
//!
//! Not in the trait:
//!
//! * timings: only periodic results carry them, as a
//!   `ferric_pbc::timing::PbcTimings` this crate cannot name;
//! * `converged` / `iterations`: a correlated or task result has no single
//!   iteration count of the same meaning.

use std::borrow::Cow;

use crate::certificate::Certificate;

/// A method result: one total energy and the certificate that grades it.
pub trait MethodResult {
    /// The total energy in Hartree: electronic plus nuclear repulsion, plus
    /// whatever the method adds on top of its reference (the correlation
    /// energy, a dispersion correction). Per cell for a periodic result.
    fn total_energy(&self) -> f64;

    /// The validation certificate of this result. Borrowed when the result
    /// stores it, owned when it is built on demand from stored inputs.
    fn certificate(&self) -> Cow<'_, Certificate>;
}
