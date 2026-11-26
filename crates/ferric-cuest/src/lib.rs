//! Optional NVIDIA cuEST backend for ferric.
//!
//! Without the `cuest` feature (the default) this crate is CPU-only. It holds
//! the parts that do not need a GPU: basis conversion ([`basis`]), the cuEST
//! compatibility audit ([`audit`]), the AO-ordering layer ([`layout`]) and the
//! comparison against stored libint references ([`mod@reference`]).
//!
//! With `cuest` on it links `libcuest` (cuEST 0.3.x) and the CUDA runtime, and
//! `oe::compute_one_electron` builds S, T and V on the GPU. Nothing in ferric's
//! SCF calls this yet: the first milestone is matching libint's one-electron
//! matrices (crates/ferric-cuest/tests/oe_vs_libint.rs).

#![deny(unsafe_op_in_unsafe_fn)]

use thiserror::Error;

/// Errors produced by the cuEST backend.
#[derive(Debug, Error)]
pub enum CuestError {
    #[error("cuEST feature not enabled at build time")]
    FeatureDisabled,
    /// No CUDA device, or a device cuEST does not support (it needs sm_86+).
    #[error("cuEST cannot run on this device: {0}")]
    DeviceUnavailable(String),
    /// A cuEST call returned a non-success status.
    #[error("{call} returned CUEST_STATUS_{name} ({status})")]
    Call {
        call: &'static str,
        status: i32,
        name: &'static str,
    },
    /// A CUDA runtime call failed.
    #[error("{call} failed: {msg} (cudaError {code})")]
    Cuda {
        call: &'static str,
        code: i32,
        msg: String,
    },
    /// The molecule/basis cannot be expressed in cuEST.
    #[error("unsupported by the cuEST backend: {0}")]
    Unsupported(String),
}

pub type Result<T> = std::result::Result<T, CuestError>;

pub mod audit;
pub mod basis;
pub mod ecp;
pub mod error;
pub mod grid;
pub mod layout;
pub mod pairlist;
pub mod reference;
pub mod workspace;

#[cfg(feature = "cuest")]
pub mod cuda;
#[cfg(feature = "cuest")]
pub mod ffi;
#[cfg(feature = "cuest")]
pub mod handle;
#[cfg(feature = "cuest")]
pub mod oe;

/// Whether this build links cuEST (it says nothing about the device).
pub fn cuest_linked() -> bool {
    cfg!(feature = "cuest")
}
