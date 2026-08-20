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

impl std::fmt::Debug for CEcpGShell {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CEcpGShell")
            .field("l", &self.l)
            .field("nprim", &self.nprim)
            .field("x", &self.x)
            .field("y", &self.y)
            .field("z", &self.z)
            .finish_non_exhaustive()
    }
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

impl std::fmt::Debug for CEcpCenter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CEcpCenter")
            .field("x", &self.x)
            .field("y", &self.y)
            .field("z", &self.z)
            .field("nterm", &self.nterm)
            .finish_non_exhaustive()
    }
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

    /// Number of distinct atomic centers libecpint infers from these shells +
    /// ECPs (dedup by 1e-4 Bohr, shells first then ECP centers). Needed to size
    /// the derivative buffer and to map libecpint atom ids back onto the
    /// caller's own atom list. Negative on error.
    pub fn ferric_ecp_natoms(
        shells: *const CEcpGShell,
        nshell: c_int,
        ecps: *const CEcpCenter,
        necp: c_int,
    ) -> c_int;

    /// Compute first derivatives of the Cartesian ECP matrix w.r.t. every atomic
    /// coordinate into `out_derivs` (3*natoms*ncart*ncart, row-major, ordered
    /// `{A_x, A_y, A_z, B_x, ...}` over libecpint's inferred atom ids).
    /// `out_natoms` receives the inferred natoms. Returns 0 on success.
    pub fn ferric_ecp_matrix_deriv(
        shells: *const CEcpGShell,
        nshell: c_int,
        ecps: *const CEcpCenter,
        necp: c_int,
        out_derivs: *mut c_double,
        out_natoms: *mut c_int,
    ) -> c_int;
}

/// Success status code from the ECP C shim.
pub const FERRIC_ECP_OK: c_int = 0;
