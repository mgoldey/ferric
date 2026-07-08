//! Raw FFI declarations for the libint2 C++ shim (`shim/shim.cc`).
//!
//! These are unsafe C-linkage functions. Prefer the safe wrappers in
//! [`crate::engine`] and [`crate::basis_bridge`].

use std::os::raw::{c_char, c_double, c_int, c_void};

/// C-compatible shell descriptor passed to the libint2 shim.
#[repr(C)]
pub struct CShell {
    pub l: c_int,
    pub nprim: c_int,
    pub atom_index: c_int,
    pub pure: c_int,
    pub exponents: *const c_double,
    pub coefficients: *const c_double,
}

/// C-compatible atom descriptor (atomic number + Cartesian position in Bohr).
#[repr(C)]
pub struct CAtom {
    pub atomic_number: c_int,
    pub x: c_double,
    pub y: c_double,
    pub z: c_double,
}

extern "C" {
    pub fn scf_libint_init();
    pub fn scf_libint_finalize();
    pub fn scf_basis_create(shells: *const CShell, nshells: c_int, atoms: *const CAtom, natoms: c_int) -> *mut c_void;
    pub fn scf_basis_destroy(bs: *mut c_void);
    pub fn scf_basis_nbasis(bs: *const c_void) -> c_int;
    pub fn scf_basis_nshells(bs: *const c_void) -> c_int;
    pub fn scf_basis_shell_dims(bs: *const c_void, out: *mut c_int);
    pub fn scf_basis_max_dims(bs: *const c_void, max_nprim: *mut c_int, max_l: *mut c_int);
    pub fn scf_engine_create(op_kind: c_int, omega: c_double, max_nprim: c_int, max_l: c_int, precision: c_double) -> *mut c_void;
    pub fn scf_engine_destroy(eng: *mut c_void);
    pub fn scf_engine_set_point_charges(eng: *mut c_void, atoms: *const CAtom, natoms: c_int) -> c_int;
    pub fn scf_compute_1e_block(eng: *mut c_void, bs: *const c_void, sh1: c_int, sh2: c_int, out: *mut c_double) -> c_int;
    pub fn scf_compute_eri_quartet(eng: *mut c_void, bs: *const c_void, sh1: c_int, sh2: c_int, sh3: c_int, sh4: c_int, out: *mut c_double) -> c_int;
    pub fn scf_compute_schwarz(eng: *mut c_void, bs: *const c_void, qmat: *mut c_double) -> c_int;
    pub fn scf_engine_create_deriv(op_kind: c_int, omega: c_double, max_nprim: c_int, max_l: c_int, precision: c_double) -> *mut c_void;
    pub fn scf_engine_create_geminal(op_kind: c_int, ngauss: c_int, exps: *const c_double, coefs: *const c_double, max_nprim: c_int, max_l: c_int, precision: c_double) -> *mut c_void;
    pub fn scf_compute_1e_deriv_block(eng: *mut c_void, bs: *const c_void, sh1: c_int, sh2: c_int, out: *mut c_double) -> c_int;
    pub fn scf_compute_eri_deriv_quartet(eng: *mut c_void, bs: *const c_void, sh1: c_int, sh2: c_int, sh3: c_int, sh4: c_int, out: *mut c_double) -> c_int;
    pub fn scf_engine_create_3center(op_kind: c_int, omega: c_double, max_nprim: c_int, max_l: c_int, precision: c_double) -> *mut c_void;
    pub fn scf_engine_create_2center(op_kind: c_int, omega: c_double, max_nprim: c_int, max_l: c_int, precision: c_double) -> *mut c_void;
    pub fn scf_compute_eri3(eng: *mut c_void, obs: *const c_void, dfbs: *const c_void, shP: c_int, sh1: c_int, sh2: c_int, out: *mut c_double) -> c_int;
    pub fn scf_compute_eri2(eng: *mut c_void, dfbs: *const c_void, shP: c_int, shQ: c_int, out: *mut c_double) -> c_int;
    pub fn scf_engine_create_3center_deriv(op_kind: c_int, omega: c_double, max_nprim: c_int, max_l: c_int, precision: c_double) -> *mut c_void;
    pub fn scf_engine_create_2center_deriv(op_kind: c_int, omega: c_double, max_nprim: c_int, max_l: c_int, precision: c_double) -> *mut c_void;
    pub fn scf_compute_eri3_deriv(eng: *mut c_void, obs: *const c_void, dfbs: *const c_void, shP: c_int, sh1: c_int, sh2: c_int, out: *mut c_double) -> c_int;
    pub fn scf_compute_eri2_deriv(eng: *mut c_void, dfbs: *const c_void, shP: c_int, shQ: c_int, out: *mut c_double) -> c_int;
    // Exact terfc(r,r0)/r via 2D interpolation tables (Dutoi/Goldey). table_dir may be
    // null (falls back to FERRIC_TERF_TABLE_DIR). See shim.h / terf-tables/terf_plan.md.
    pub fn scf_engine_create_terfc_3center(r0: c_double, omega: c_double, max_nprim: c_int, max_l: c_int, precision: c_double, table_dir: *const c_char) -> *mut c_void;
    pub fn scf_engine_create_terfc_2center(r0: c_double, omega: c_double, max_nprim: c_int, max_l: c_int, precision: c_double, table_dir: *const c_char) -> *mut c_void;
    pub fn scf_compute_terfc_eri3(eng: *mut c_void, obs: *const c_void, dfbs: *const c_void, shP: c_int, sh1: c_int, sh2: c_int, out: *mut c_double) -> c_int;
    pub fn scf_compute_terfc_eri2(eng: *mut c_void, dfbs: *const c_void, shP: c_int, shQ: c_int, out: *mut c_double) -> c_int;
    // terf(r,r0)/r = tempered LONG-RANGE complement of terfc (terf + terfc = coulomb),
    // same tables/curvature constraint. See shim.h.
    pub fn scf_engine_create_terf_3center(r0: c_double, omega: c_double, max_nprim: c_int, max_l: c_int, precision: c_double, table_dir: *const c_char) -> *mut c_void;
    pub fn scf_engine_create_terf_2center(r0: c_double, omega: c_double, max_nprim: c_int, max_l: c_int, precision: c_double, table_dir: *const c_char) -> *mut c_void;
    pub fn scf_compute_terf_eri3(eng: *mut c_void, obs: *const c_void, dfbs: *const c_void, shP: c_int, sh1: c_int, sh2: c_int, out: *mut c_double) -> c_int;
    pub fn scf_compute_terf_eri2(eng: *mut c_void, dfbs: *const c_void, shP: c_int, shQ: c_int, out: *mut c_double) -> c_int;
    pub fn scf_compute_dipole(
        bs: *const c_void,
        origin: *const c_double,
        nbas: c_int,
        out: *mut c_double,
    ) -> c_int;
}

pub const OP_COULOMB: c_int = 0;
pub const OP_ERF_COULOMB: c_int = 1;
pub const OP_ERFC_COULOMB: c_int = 2;
pub const OP_OVERLAP: c_int = 100;
pub const OP_KINETIC: c_int = 101;
pub const OP_NUCLEAR: c_int = 102;
pub const OP_EMULTIPOLE1: c_int = 103;
// Geminal (F12) two-electron operators — see scf_engine_create_geminal.
pub const OP_CGTG: c_int = 200;
pub const OP_CGTG_X_COULOMB: c_int = 201;
pub const OP_DELCGTG2: c_int = 202;
