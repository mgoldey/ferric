//! Pins the GW default settings that `validation_gw_defaults.rs` calibrated
//! (2026-11-06). Changing one of these without re-running that calibration
//! fails here first. Cheap, so it runs in every per-commit shard; the
//! calibration itself is validation tier.

use ferric_gw::{GwConfig, DEFAULT_N_QUAD};
use ferric_rpa::config::{PdepRpaConfig, QuadratureScheme};

#[test]
fn gw_defaults_are_the_calibrated_values() {
    assert_eq!(DEFAULT_N_QUAD, 40);
    let p = PdepRpaConfig::default();
    assert_eq!(p.trunc_thresh, 1e-4);
    assert_eq!(p.quadrature.scheme, QuadratureScheme::GaussLegendre);
    assert_eq!(p.quadrature.u0, 0.5);
    let g = GwConfig::default();
    assert_eq!(g.ev_conv_thresh, 1e-4);
    assert_eq!(g.max_ev_iter, 20);
    assert_eq!(g.pade_npts, 0);
}
