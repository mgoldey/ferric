//! Weekly half of `laplace_energy_is_thread_independent.rs`: CONTRACT 2, the
//! Laplace-MP2 correlation energy bit-identical across rayon worker counts at
//! the higher quadrature order n_quad = 7 (benzene/cc-pVDZ, 0.5 GiB budget).
//! It took ~134 s on CI (runs 40957025137 / 40953725119, mean), above the
//! 120 s per-commit threshold. The per-commit canary is CONTRACT 1
//! (`the_laplace_correlation_energy_is_bit_identical_across_worker_counts`,
//! n_quad = 5), which the restored ambient-divisor defect fails; the
//! mutation record is in the per-commit file.

#[path = "laplace_thread_support/mod.rs"]
mod support;
use support::*;

/// CONTRACT 2: the same holds at a different quadrature order.
///
/// `n_quad` sets how many terms the point-sum folds, so it is the axis the
/// `.reduce()` tree-fold defect is most sensitive to: with n_quad = 1 there is
/// nothing to reassociate and any tree shape agrees, which would make a
/// single-order test pass while the defect was live. Use an order with enough
/// terms to have distinguishable tree shapes.
#[test]
#[ignore = "validation: Laplace-MP2 energy bit-identical across worker counts at n_quad = 7; CI ~134 s"]
fn bit_identity_holds_at_a_higher_quadrature_order() {
    let f = fixture();
    let n_quad = 7;
    let reference = energy_at(&f, n_quad, 1);
    for n in THREAD_COUNTS {
        assert_eq!(
            energy_at(&f, n_quad, n).to_bits(),
            reference.to_bits(),
            "n_quad={n_quad}: the energy is not bit-identical between 1 and {n} workers"
        );
    }
}
