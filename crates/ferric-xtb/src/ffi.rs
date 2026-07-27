//! Raw declarations for the subset of xtb's C API (`xtb.h`) that ferric uses.
//!
//! Compiled only under the `xtb` feature. Mirrors xtb 6.7.1's `include/xtb.h`.
//!
//! # Error-reporting contract
//!
//! Unlike ferric's libint2/libecpint shims (which return status codes directly),
//! xtb's C API returns `void` from nearly every entry point and instead records
//! failures on the *environment* object. The caller MUST invoke
//! [`xtb_checkEnvironment`] after every call that takes an environment; a
//! non-zero return means an error is queued, retrievable with [`xtb_getError`].
//! Ignoring the check silently produces garbage results rather than an error.
//!
//! xtb is Fortran with a C-ABI veneer. A Fortran runtime error inside the
//! library terminates the process rather than unwinding, so there is no foreign
//! exception to catch across the boundary (contrast the C++ shims, which must
//! `try`/`catch`); the environment status codes are the whole error channel.
//!
//! # Units
//!
//! Coordinates in **Bohr**, energy in **Hartree**, gradient in **Hartree/Bohr**.

use std::os::raw::{c_char, c_double, c_int};

/// Opaque handle types. xtb defines these as pointers to incomplete structs.
pub type XtbEnvironment = *mut std::ffi::c_void;
pub type XtbMolecule = *mut std::ffi::c_void;
pub type XtbCalculator = *mut std::ffi::c_void;
pub type XtbResults = *mut std::ffi::c_void;

extern "C" {
    // --- environment ---
    pub fn xtb_newEnvironment() -> XtbEnvironment;
    pub fn xtb_delEnvironment(env: *mut XtbEnvironment);
    /// Returns the number of queued errors; 0 means OK.
    pub fn xtb_checkEnvironment(env: XtbEnvironment) -> c_int;
    /// Copies the queued error message into `buffer` and empties the stack.
    pub fn xtb_getError(env: XtbEnvironment, buffer: *mut c_char, buffersize: *const c_int);
    /// Verbosity: 0 = muted, 1 = minimal, 2 = full.
    pub fn xtb_setVerbosity(env: XtbEnvironment, verbosity: c_int);

    // --- molecule (positions in Bohr) ---
    pub fn xtb_newMolecule(
        env: XtbEnvironment,
        natoms: *const c_int,
        numbers: *const c_int,
        positions: *const c_double,
        charge: *const c_double,
        uhf: *const c_int,
        lattice: *const c_double,
        periodic: *const bool,
    ) -> XtbMolecule;
    pub fn xtb_delMolecule(mol: *mut XtbMolecule);

    // --- calculator ---
    pub fn xtb_newCalculator() -> XtbCalculator;
    pub fn xtb_delCalculator(calc: *mut XtbCalculator);
    pub fn xtb_loadGFN1xTB(
        env: XtbEnvironment,
        mol: XtbMolecule,
        calc: XtbCalculator,
        filename: *mut c_char,
    );
    pub fn xtb_loadGFN2xTB(
        env: XtbEnvironment,
        mol: XtbMolecule,
        calc: XtbCalculator,
        filename: *mut c_char,
    );
    pub fn xtb_loadGFNFF(
        env: XtbEnvironment,
        mol: XtbMolecule,
        calc: XtbCalculator,
        filename: *mut c_char,
    );
    pub fn xtb_setAccuracy(env: XtbEnvironment, calc: XtbCalculator, accuracy: c_double);
    pub fn xtb_setMaxIter(env: XtbEnvironment, calc: XtbCalculator, iterations: c_int);
    pub fn xtb_setElectronicTemp(env: XtbEnvironment, calc: XtbCalculator, temperature: c_double);

    // --- single point ---
    pub fn xtb_singlepoint(
        env: XtbEnvironment,
        mol: XtbMolecule,
        calc: XtbCalculator,
        res: XtbResults,
    );

    // --- results ---
    pub fn xtb_newResults() -> XtbResults;
    pub fn xtb_delResults(res: *mut XtbResults);
    /// Energy in Hartree.
    pub fn xtb_getEnergy(env: XtbEnvironment, res: XtbResults, energy: *mut c_double);
    /// Gradient in Hartree/Bohr, `[natoms][3]` row-major.
    pub fn xtb_getGradient(env: XtbEnvironment, res: XtbResults, gradient: *mut c_double);
    /// Mulliken-style partial charges in e, `[natoms]`.
    pub fn xtb_getCharges(env: XtbEnvironment, res: XtbResults, charges: *mut c_double);
}
