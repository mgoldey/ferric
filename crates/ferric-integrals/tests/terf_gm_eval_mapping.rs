//! STEP 1 GATE for the libint2 core-eval port.
//!
//! libint2 hands a core evaluator `(Gm, rho, T, mmax, <oper params>)`, while
//! `terf_aux` wants `(S, s, phi_over_theta)`. `terf_gm_eval_impl` recovers the
//! latter from the former:
//!
//!   phi2           = rho * omega^2 / (rho + omega^2)
//!   S              = T * (phi2 / rho)        [since T == rho * PQ2]
//!   s              = phi2 * r0^2
//!   phi_over_theta = sqrt(phi2 / rho)
//!
//! Nothing about the numerics changes -- same tables, same interpolation, only
//! the argument recovery is new. So the two paths must agree BIT-FOR-BIT, not
//! merely closely. A tolerance here would hide exactly the algebra error this
//! test exists to catch.
//!
//! If this fails, the port stops: every later step assumes this mapping.

use ferric_integrals::ffi;
use std::ffi::CString;
use std::os::raw::{c_char, c_double, c_int};

#[test]
fn terf_gm_eval_reproduces_terf_aux_bit_for_bit() {
    let Ok(dir) = std::env::var("FERRIC_TERF_TABLE_DIR") else {
        eprintln!("SKIP: FERRIC_TERF_TABLE_DIR unset");
        return;
    };
    let cdir = CString::new(dir).unwrap();
    let mut mismatches: c_int = -1;
    let mut worst: c_double = -1.0;
    // SAFETY: valid NUL-terminated dir, two live out-params. The shim wraps its
    // body in try/catch and returns a negative status on failure.
    let compared = unsafe {
        ffi::scf_terf_gm_eval_matches_terf_aux(
            cdir.as_ptr() as *const c_char,
            &mut mismatches,
            &mut worst,
        )
    };
    assert!(
        compared > 0,
        "gate returned status {compared} (negative = error)"
    );
    // Anti-inertness: the sweep must actually exercise the kernel. A mapping bug
    // that made every call fail would otherwise pass with compared == 0.
    assert!(
        compared > 500,
        "only {compared} samples compared -- the sweep is not exercising terf_aux"
    );
    eprintln!("terf_gm_eval vs terf_aux: {compared} samples, {mismatches} mismatches, worst abs diff {worst:.3e}");
    assert_eq!(
        mismatches, 0,
        "libint2 argument mapping is NOT bit-identical to terf_aux \
         ({mismatches} of {compared} samples differ, worst abs {worst:.3e}) -- \
         the (rho,T,omega,r0) -> (S,s,phi_over_theta) algebra is wrong"
    );
}
