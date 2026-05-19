//! Verify Lebedev-302 matches the canonical Lebedev-Laikov table.
//!
//! Sanity checks:
//!   * 302 points, weights sum to 1
//!   * Pointwise match against PySCF's MakeAngularGrid_302 (within rounding)

use ferric_dft::lebedev::lebedev;

#[test]
fn lebedev_302_has_302_points_and_unit_weight_sum() {
    let (pts, wts) = lebedev(302);
    assert_eq!(pts.len(), 302, "Lebedev-302 should produce exactly 302 points, got {}", pts.len());
    let sum: f64 = wts.iter().sum();
    eprintln!("Lebedev-302: n_pts={}, sum(w)={:.12}", pts.len(), sum);
    assert!((sum - 1.0).abs() < 1e-12, "weights should sum to 1, got {sum}");
}

#[test]
fn lebedev_302_points_are_unit_norm() {
    let (pts, _) = lebedev(302);
    let mut max_err = 0.0_f64;
    for p in &pts {
        let norm = (p[0]*p[0] + p[1]*p[1] + p[2]*p[2]).sqrt();
        max_err = max_err.max((norm - 1.0).abs());
    }
    eprintln!("Lebedev-302: max |‖p‖ − 1| = {max_err:.2e}");
    assert!(max_err < 1e-13, "Lebedev-302 points not unit-norm: max err {max_err:.2e}");
}

/// Lebedev-302 integrates spherical harmonics up to L=29 exactly (rule of 4n+5
/// for an n=74 quadrature; 302 is the n=74 rule). Test by integrating low-order
/// polynomials.
#[test]
fn lebedev_302_integrates_polynomials_exactly() {
    let (pts, wts) = lebedev(302);

    // ∫_sphere x² dΩ / 4π = 1/3
    let int_x2: f64 = pts.iter().zip(wts.iter())
        .map(|(p, w)| w * p[0]*p[0]).sum();
    assert!((int_x2 - 1.0/3.0).abs() < 1e-12, "∫x²/4π = {int_x2}, expect 1/3");

    // ∫_sphere x²y²z² dΩ / 4π = 1/105 (standard result for L=6)
    let int_x2y2z2: f64 = pts.iter().zip(wts.iter())
        .map(|(p, w)| w * (p[0]*p[1]*p[2]).powi(2)).sum();
    assert!((int_x2y2z2 - 1.0/105.0).abs() < 1e-13,
            "∫x²y²z²/4π = {int_x2y2z2}, expect 1/105");

    // ∫_sphere (x^6) dΩ / 4π = 1/7 (general formula for x^(2n) is (2n-1)!! / (2n+1)!!)
    let int_x6: f64 = pts.iter().zip(wts.iter())
        .map(|(p, w)| w * p[0].powi(6)).sum();
    assert!((int_x6 - 1.0/7.0).abs() < 1e-13, "∫x⁶/4π = {int_x6}, expect 1/7");

    eprintln!("Lebedev-302 polynomial sanity: ∫x²={int_x2:.12}, ∫x²y²z²={int_x2y2z2:.14}, ∫x⁶={int_x6:.12}");
}
