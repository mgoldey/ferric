//! Re-exports from `ferric_integrals::ao_grid` (canonical home of GTO grid evaluation).
//!
//! All production items now live in `ferric-integrals`; this shim preserves backward
//! compatibility for existing `ferric_dft::ao_grid::*` import paths.

pub use ferric_integrals::ao_grid::*;

#[cfg(test)]
mod budget_guard_tests {
    use super::*;

    use crate::TEST_BUDGET_ENV_LOCK as ENV_LOCK;
    const VAR: &str = ferric_core::memory::ENV_UNIFIED;

    fn clear() {
        std::env::remove_var(VAR);
    }

    #[test]
    fn planes_formula_matches_documented_counts() {
        assert_eq!(AoGridKind::ValueOnly.planes(), 1);
        assert_eq!(AoGridKind::ValueAndGrad.planes(), 4);
        assert_eq!(AoGridKind::ValueGradHess.planes(), 13);
    }

    #[test]
    fn large_scale_hess_allocation_rejected_under_tiny_budget() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear();
        std::env::set_var(VAR, "1");

        let nbf = 1500usize;
        let npts = 412_500usize;
        let err = check_ao_grid_budget(AoGridKind::ValueGradHess, nbf, npts).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("nbf=1500"), "message should cite nbf: {msg}");
        assert!(msg.contains("npts=412500"), "message should cite npts: {msg}");
        assert!(
            msg.contains("GB"),
            "message should state the estimated/budgeted sizes in GB: {msg}"
        );
        assert!(
            msg.contains("FERRIC_MEM_BUDGET_GB"),
            "message should point at the remediation knob: {msg}"
        );

        clear();
    }

    #[test]
    fn large_scale_grad_allocation_rejected_under_tiny_budget() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear();
        std::env::set_var(VAR, "1");

        let nbf = 1500usize;
        let npts = 412_500usize;
        let err = check_ao_grid_budget(AoGridKind::ValueAndGrad, nbf, npts).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("nbf=1500"));
        assert!(msg.contains("npts=412500"));
        assert!(msg.contains("GB"));

        clear();
    }

    #[test]
    fn small_scale_allocation_not_rejected_under_realistic_budget() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear();
        std::env::set_var(VAR, "2");

        let nbf = 25usize;
        let npts = 35_000usize;

        assert!(
            check_ao_grid_budget(AoGridKind::ValueGradHess, nbf, npts).is_ok(),
            "water/cc-pVDZ-scale Hessian AO grid must fit a 2 GiB budget"
        );
        assert!(
            check_ao_grid_budget(AoGridKind::ValueAndGrad, nbf, npts).is_ok(),
            "water/cc-pVDZ-scale grad-only AO grid must fit a 2 GiB budget"
        );

        clear();
    }

    #[test]
    fn moderate_scale_allocation_not_rejected_under_default_budget() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear();

        let nbf = 150usize;
        let npts = 150_000usize;

        assert!(
            check_ao_grid_budget(AoGridKind::ValueAndGrad, nbf, npts).is_ok(),
            "moderate-scale grad-only AO grid must fit the default budget"
        );

        clear();
    }
}
