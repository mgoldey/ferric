//! Raw FFI declarations for the libecpint ECP shim (`shim/ecp_shim.cc`).
//!
//! Independent of the libint2 shim FFI ([`crate::ffi`]). Computes the dense
//! Cartesian ECP matrix `V_ECP` for a molecule with ECP centers.
//!
//! libecpint applies no internal normalization to the Gaussian contraction
//! coefficients, so the caller must supply fully-normalized coefficients
//! (primitive normalization folded in, contraction normalized).

use std::os::raw::{c_double, c_int};

/// One Gaussian basis shell (Cartesian), flattened for the ECP shim.
#[repr(C)]
pub struct CEcpGShell {
    pub l: c_int,
    pub nprim: c_int,
    pub x: c_double,
    pub y: c_double,
    pub z: c_double,
    pub exponents: *const c_double,
    pub coefficients: *const c_double,
}

/// One ECP center: a flat list of `nterm` semilocal primitives.
#[repr(C)]
pub struct CEcpCenter {
    pub x: c_double,
    pub y: c_double,
    pub z: c_double,
    pub nterm: c_int,
    pub ams: *const c_int,
    pub ns: *const c_int,
    pub exponents: *const c_double,
    pub coefficients: *const c_double,
}

extern "C" {
    /// Compute the dense Cartesian ECP matrix into `out_vecp` (ncart*ncart, row-major).
    /// Returns 0 on success, negative on error.
    pub fn ferric_ecp_matrix(
        shells: *const CEcpGShell,
        nshell: c_int,
        ecps: *const CEcpCenter,
        necp: c_int,
        out_vecp: *mut c_double,
    ) -> c_int;

    /// Total number of Cartesian functions for the given shells.
    pub fn ferric_ecp_ncart(shells: *const CEcpGShell, nshell: c_int) -> c_int;
}

pub const FERRIC_ECP_OK: c_int = 0;
