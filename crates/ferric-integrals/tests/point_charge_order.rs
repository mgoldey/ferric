//! The ORDER of an external point-charge list changes the one-electron potential `V` only at the
//! summation-roundoff level (#444).
//!
//! Hypotheses. Physics/design: `V = -sum_i q_i <mu|1/|r-R_i||nu>` is a sum over the charges, so
//! permuting the list can only reorder a floating-point sum: a change of a few ulps of the largest
//! partial sum, and the SAME order must be bit-identical run to run. Artifact: if the charge loop
//! carried state between charges (a cached boys table, a screening cut keyed on the first charge,
//! an early exit), a permutation would move `V` by far more than roundoff; dropping a single charge
//! is the opposite control (it must be visible).
//!
//! Measured on the #444 reproducer (74 atoms STO-3G, 6458 AMBER charges, max|V| = 105): file order vs
//! nearest-first 3.2e-13, vs two random shuffles 1.2e-12 and 6.0e-13, i.e. 3e-15 .. 1e-14 relative.

use ferric_core::basis;
use ferric_core::external_potential::{ExternalPotential, PointCharge};
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron::nuclear_with_external;
use ndarray::Array2;

/// Deterministic pseudo-random field: `n` charges of both signs 6..40 Bohr from the origin.
fn field(n: usize) -> Vec<PointCharge> {
    let mut s = 0x9E37_79B9_7F4A_7C15_u64;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    (0..n)
        .map(|_| {
            let (u, v, w) = (next() * 2.0 - 1.0, next() * 2.0 - 1.0, next() * 2.0 - 1.0);
            let norm = (u * u + v * v + w * w).sqrt().max(1e-3);
            let r = 6.0 + 34.0 * next();
            PointCharge {
                q: next() * 2.0 - 1.0,
                x: r * u / norm,
                y: r * v / norm,
                z: r * w / norm,
            }
        })
        .collect()
}

fn v_of(prep: &PreparedBasis, charges: Vec<PointCharge>) -> Array2<f64> {
    let ext = ExternalPotential {
        point_charges: charges,
        smeared_charges: Vec::new(),
        field: None,
    };
    nuclear_with_external(prep, &ext)
}

fn max_abs(a: &Array2<f64>) -> f64 {
    a.iter().fold(0.0_f64, |m, x| m.max(x.abs()))
}

#[test]
fn charge_order_moves_v_by_roundoff_only_and_a_dropped_charge_is_visible() {
    let mol = Molecule::parse_xyz(
        "3\nwater\nO 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692\n",
        0,
        1,
    )
    .unwrap();
    let prep = PreparedBasis::new(&mol, &basis::bundled("sto-3g").unwrap()).unwrap();
    let base_charges = field(2000);
    let base = v_of(&prep, base_charges.clone());
    let scale = max_abs(&base);

    // Same order twice: bit-identical.
    assert_eq!(v_of(&prep, base_charges.clone()), base);

    // Permutations: reversed, rotated, and a deterministic shuffle.
    let mut rev = base_charges.clone();
    rev.reverse();
    let mut rot = base_charges.clone();
    rot.rotate_left(777);
    let mut shuf = base_charges.clone();
    let mut s = 12345_u64;
    for i in (1..shuf.len()).rev() {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        shuf.swap(i, (s >> 33) as usize % (i + 1));
    }
    let mut worst = 0.0_f64;
    for (name, p) in [("reversed", rev), ("rotated", rot), ("shuffled", shuf)] {
        let d = max_abs(&(&v_of(&prep, p) - &base));
        eprintln!(
            "{name}: max|dV| {d:.3e} (max|V| {scale:.3e}, relative {:.2e})",
            d / scale
        );
        worst = worst.max(d / scale);
    }
    assert!(
        worst < 1e-13,
        "order moved V by {worst:.2e} relative; summation roundoff is ~1e-14"
    );

    // Opposite control: dropping ONE charge must move V far above that bar.
    let mut fewer = base_charges;
    fewer.pop();
    let dropped = max_abs(&(&v_of(&prep, fewer) - &base)) / scale;
    eprintln!("one charge dropped: relative {dropped:.2e}");
    assert!(
        dropped > 1e3 * worst.max(1e-16),
        "a dropped charge ({dropped:.2e}) is not separated from roundoff"
    );
}
