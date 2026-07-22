//! Re-export of [`ferric_core::blas_threads`].
//!
//! The implementation lives in `ferric-core` (dependency-free, reachable from
//! any crate in the workspace without a new dependency edge — see that
//! module's doc for the full rationale/hazard model). This module re-exports
//! everything so existing `ferric_integrals::blas_threads::*` call sites keep
//! working unchanged.

pub use ferric_core::blas_threads::*;
