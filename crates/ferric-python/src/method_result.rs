//! [`MethodResult`] and the macros that build the shared Python result
//! getters from it.
//!
//! The trait lives here, not in ferric-core, because every implementer is a
//! ferric-python result class and nothing consumes it generically yet: its
//! only caller is [`result_getters!`]. It moves to ferric-core when a kernel
//! or CLI result implements it, or when generic code needs it (ARCHITECTURE.md,
//! roadmap item 9b).
//!
//! * [`MethodResult`]: one total energy and the certificate that grades it.
//! * `stored_method_result!` implements it for a class that stores both.
//! * `periodic_method_result!` implements it for a periodic class, whose
//!   certificate is built on demand by `pbc::pbc_certificate_stamped`.
//! * `result_getters!` declares a class's shared Python getters
//!   (`certificate`, `timings`, the `energy`/`total_energy` alias) once.

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

/// The Python getters the result classes share, declared once per class as
/// a second `#[pymethods]` block (pyo3's `multiple-pymethods`: a macro item
/// inside a `#[pymethods]` impl is a hard error). Entries, in any order,
/// each ending in `;`, the getter's doc comments above it:
///
/// * `total_energy: via trait;` / `energy: via trait;`: the alias getter
///   (#676) with its fixed docstring ("Alias of `energy` ..." /
///   "Alias of `total_energy` ..."), returning
///   `MethodResult::total_energy`;
/// * `total_energy: from field <name>;`: the same `total_energy` alias, read
///   from a field, for a class that is not a `MethodResult`;
/// * `certificate: via trait;`: the dict form of
///   `MethodResult::certificate`;
/// * `certificate: from field <name>;`: the same from a stored
///   `Certificate` field, for a class that is not a `MethodResult`;
/// * `timings: from field <name>;`: `pbc::timings_dict` of a `PbcTimings`
///   field.
///
/// Anything else is a compile error naming these forms. Each getter keeps
/// the name, return type, value, docstring and errors of the hand-written
/// one it replaced.
macro_rules! result_getters {
    ($ty:ident { $($body:tt)* }) => {
        result_getters!(@munch $ty [] $($body)*);
    };
    (@munch $ty:ident [$($acc:tt)*]) => {
        #[::pyo3::pymethods]
        impl $ty {
            $($acc)*
        }
    };
    (@munch $ty:ident [$($acc:tt)*] total_energy: via trait; $($rest:tt)*) => {
        result_getters!(@munch $ty [$($acc)*
            /// Alias of `energy` (the same value), so callers can read either spelling.
            #[getter]
            fn total_energy(&self) -> f64 {
                $crate::method_result::MethodResult::total_energy(self)
            }
        ] $($rest)*);
    };
    (@munch $ty:ident [$($acc:tt)*] total_energy: from field $f:ident; $($rest:tt)*) => {
        result_getters!(@munch $ty [$($acc)*
            /// Alias of `energy` (the same value), so callers can read either spelling.
            #[getter]
            fn total_energy(&self) -> f64 {
                self.$f
            }
        ] $($rest)*);
    };
    (@munch $ty:ident [$($acc:tt)*] energy: via trait; $($rest:tt)*) => {
        result_getters!(@munch $ty [$($acc)*
            /// Alias of `total_energy` (the same value), so callers can read either spelling.
            #[getter]
            fn energy(&self) -> f64 {
                $crate::method_result::MethodResult::total_energy(self)
            }
        ] $($rest)*);
    };
    (@munch $ty:ident [$($acc:tt)*] $(#[$($doc:tt)*])* certificate: via trait; $($rest:tt)*) => {
        result_getters!(@munch $ty [$($acc)*
            $(#[$($doc)*])*
            #[getter]
            fn certificate(&self, py: ::pyo3::Python<'_>) -> ::pyo3::PyResult<::pyo3::PyObject> {
                $crate::certificate_dict(
                    py,
                    &$crate::method_result::MethodResult::certificate(self),
                )
            }
        ] $($rest)*);
    };
    (@munch $ty:ident [$($acc:tt)*] $(#[$($doc:tt)*])* certificate: from field $f:ident; $($rest:tt)*) => {
        result_getters!(@munch $ty [$($acc)*
            $(#[$($doc)*])*
            #[getter]
            fn certificate(&self, py: ::pyo3::Python<'_>) -> ::pyo3::PyResult<::pyo3::PyObject> {
                $crate::certificate_dict(py, &self.$f)
            }
        ] $($rest)*);
    };
    (@munch $ty:ident [$($acc:tt)*] $(#[$($doc:tt)*])* timings: from field $f:ident; $($rest:tt)*) => {
        result_getters!(@munch $ty [$($acc)*
            $(#[$($doc)*])*
            #[getter]
            fn timings(
                &self,
                py: ::pyo3::Python<'_>,
            ) -> ::pyo3::PyResult<::pyo3::Py<::pyo3::types::PyDict>> {
                $crate::pbc::timings_dict(py, &self.$f)
            }
        ] $($rest)*);
    };
    (@munch $ty:ident [$($acc:tt)*] $($bad:tt)+) => {
        ::core::compile_error!(concat!(
            "result_getters!(", stringify!($ty), "): unrecognised entry `",
            stringify!($($bad)+),
            "`; the entries are `total_energy: via trait;`, `energy: via trait;`, ",
            "`total_energy: from field <name>;`, `certificate: via trait;`, ",
            "`certificate: from field <name>;` and `timings: from field <name>;`, ",
            "each optionally preceded by doc comments"
        ));
    };
}
