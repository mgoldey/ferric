//! Weekly half of `ecp_hessian_combinations.rs` (issue #406): the analytic
//! RHF Hessian of a system with TWO different ECP elements (synthetic H-Br-I
//! cation, def2-SVP with a scaled second ECP on Br) against central FD of
//! ferric's analytic ECP gradient, with the skeleton-removed negative
//! control. It took ~163 s on CI (runs 40957025137 / 40953725119, mean),
//! above the 120 s per-commit threshold. Per-commit canary:
//! `two_ecp_elements_masked_stencil_matches_unmasked_and_is_translation_invariant`
//! (same system, same per-ECP-centre fold in the skeleton; it catches the
//! fold mutant). Weekly only: the two-ECP CPHF response term (`dV_ECP/dx` of
//! both centres in `F^x`), which only this FD comparison sees. Shared
//! harness: `ecp_hessian_combinations_support/mod.rs`.
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo test -p ferric-scf \
//!   --test validation_ecp_hessian_combinations -- --ignored
//! ```

#[path = "ecp_hessian_combinations_support/mod.rs"]
mod support;
use support::*;

#[test]
#[ignore = "validation: two-ECP-element analytic Hessian vs FD of the analytic gradient (H-Br-I+, def2-SVP); CI ~163 s"]
fn two_ecp_elements_hessian_matches_fd_of_analytic_gradient() {
    check_two_ecp_fd(false);
}
