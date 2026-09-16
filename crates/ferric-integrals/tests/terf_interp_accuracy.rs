//! How much accuracy does lowering the interpolation order K actually cost?
//!
//! Measured against `terf_G_series` -- the EXACT Poisson series the tables were
//! generated from -- not against the K=10 table, which is itself an
//! approximation. Comparing approximations to each other would understate error.
use ferric_integrals::ffi;
use std::ffi::CString;
use std::os::raw::{c_char, c_double};

#[test]
#[ignore = "diagnostic sweep; run with --ignored --nocapture"]
fn interp_error_vs_exact_series() {
    let Ok(dir) = std::env::var("FERRIC_TERF_TABLE_DIR") else {
        eprintln!("SKIP: FERRIC_TERF_TABLE_DIR unset"); return;
    };
    let cdir = CString::new(dir).unwrap();
    let mut worst: c_double = -1.0;
    // SAFETY: valid dir, live out-param; shim wraps in try/catch.
    let n = unsafe {
        ffi::scf_terf_interp_accuracy(cdir.as_ptr() as *const c_char, 8, &mut worst)
    };
    assert!(n > 0, "probe status {n}");
    eprintln!("K={}  samples={n}  worst rel err vs exact series = {worst:.3e}",
              std::env::var("FERRIC_TERF_K").unwrap_or("10".into()));
}
