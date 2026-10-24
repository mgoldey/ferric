//! Opt-in stage timing for the GW/RPA pipeline. Zero cost unless
//! `FERRIC_TIMING` is set in the environment; then each `Stage` prints
//! `[timing] <label> <ms>` to stderr at `.end()`. Used to MEASURE (not
//! estimate) where the GW/RPA pipeline spends wall-time. See
//! crates/ferric-gw/examples/gw_profile.rs.
//!
//! Re-export shim: the implementation (and the one `FERRIC_TIMING`
//! `ConfigVar` descriptor) now lives in [`ferric_core::timing`], so ferric-scf
//! and ferric-mp2, which depend on ferric-core but not on ferric-rpa, can use
//! the same timers under the same flag. `crate::timing::Stage` call sites keep
//! working verbatim.

pub use ferric_core::timing::{enabled, PhaseAccumulator, Stage};
