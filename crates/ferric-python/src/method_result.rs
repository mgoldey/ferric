//! [`MethodResult`] and the helpers behind the shared Python result getters.
//!
//! The trait lives here, not in ferric-core, because every implementer is a
//! ferric-python result class and nothing consumes it generically yet: its
//! only callers are the getter helpers below. It moves to ferric-core when a
//! kernel or CLI result implements it, or when generic code needs it
//! (ARCHITECTURE.md, roadmap item 9b).
//!
//! * [`MethodResult`]: one total energy and the certificate that grades it.
//! * `stored_method_result!` implements it for a class that stores both.
//! * `periodic_method_result!` implements it for a periodic class, whose
//!   certificate is built on demand by `pbc::pbc_certificate_stamped`.
//! * Each result class declares its shared getters (`certificate`, the
//!   `energy`/`total_energy` alias) as one-line `#[getter]`s in its own
//!   `#[pymethods]` block, calling [`certificate_getter`] /
//!   [`total_energy_getter`], with the alias docstring from `alias_doc!`.
//!   pyo3 0.23 rejects a macro item inside `#[pymethods]`, so the getters
//!   are written out rather than generated.

use std::borrow::Cow;

use ferric_core::certificate::Certificate;

/// A method result: one total energy and the certificate that grades it.
///
/// Not in the trait: timings (only periodic results carry them, as a
/// ferric-pbc `PbcTimings`) and `converged`/`iterations` (a correlated or
/// task result has no single iteration count of the same meaning).
pub(crate) trait MethodResult {
    /// The total energy in Hartree: electronic plus nuclear repulsion, plus
    /// whatever the method adds on top of its reference (the correlation
    /// energy, a dispersion correction). Per cell for a periodic result. For
    /// CDFT it is the UHF/UKS energy at the constrained density without the
    /// constraint term; for CASCI/CASSCF/NEVPT2 it is the total including the
    /// active-space correlation (CASSCF: and the orbital optimisation;
    /// NEVPT2: and the perturbative correction) on top of the RHF reference.
    fn total_energy(&self) -> f64;

    /// The validation certificate of this result. Borrowed when the result
    /// stores it, owned when it is built on demand from stored inputs.
    fn certificate(&self) -> Cow<'_, Certificate>;
}

/// `MethodResult` for a result class that stores its certificate:
/// `stored_method_result!(PyX, total_energy: <f64 field>, certificate: <Certificate field>);`.
macro_rules! stored_method_result {
    ($ty:ident, total_energy: $e:ident, certificate: $c:ident) => {
        impl $crate::method_result::MethodResult for $ty {
            fn total_energy(&self) -> f64 {
                self.$e
            }
            fn certificate(
                &self,
            ) -> ::std::borrow::Cow<'_, ::ferric_core::certificate::Certificate> {
                ::std::borrow::Cow::Borrowed(&self.$c)
            }
        }
    };
}

/// `MethodResult` for a periodic result class: `total_energy` is its
/// `energy` field and the certificate is `pbc::pbc_certificate_stamped` of
/// its `jk`, `auxbasis`, `exxdiv`, `converged` and `iterations` fields plus
/// the given capability and k-point mesh (`None` at Gamma), both
/// expressions in the result bound as `$s`:
/// `periodic_method_result!(PyX, |r| capability: <&str-like>, mesh: <Option<..>>);`.
macro_rules! periodic_method_result {
    ($ty:ident, |$s:ident| capability: $cap:expr, mesh: $mesh:expr) => {
        impl $crate::method_result::MethodResult for $ty {
            fn total_energy(&self) -> f64 {
                self.energy
            }
            fn certificate(
                &self,
            ) -> ::std::borrow::Cow<'_, ::ferric_core::certificate::Certificate> {
                let $s = self;
                ::std::borrow::Cow::Owned($crate::pbc::pbc_certificate_stamped(
                    &$cap,
                    &$s.jk,
                    $s.auxbasis.as_deref(),
                    &$s.exxdiv,
                    $mesh,
                    $s.converged,
                    $s.iterations,
                ))
            }
        }
    };
}

/// The fixed docstring of the `energy`/`total_energy` alias getters (#676):
/// `#[doc = alias_doc!(energy)]` on `total_energy`, `#[doc =
/// alias_doc!(total_energy)]` on `energy`.
macro_rules! alias_doc {
    (energy) => {
        "Alias of `energy` (the same value), so callers can read either spelling."
    };
    (total_energy) => {
        "Alias of `total_energy` (the same value), so callers can read either spelling."
    };
}

/// Body of a `total_energy` (or `energy`) alias getter of a `MethodResult`.
pub(crate) fn total_energy_getter<R: MethodResult>(r: &R) -> f64 {
    r.total_energy()
}

/// Body of the `certificate` getter of a `MethodResult`: the dict form of its
/// certificate (`crate::certificate_dict`).
pub(crate) fn certificate_getter<R: MethodResult>(
    py: pyo3::Python<'_>,
    r: &R,
) -> pyo3::PyResult<pyo3::PyObject> {
    crate::certificate_dict(py, &r.certificate())
}
