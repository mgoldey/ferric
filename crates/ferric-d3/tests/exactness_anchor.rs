//! Exactness anchors for D3(BJ), written BEFORE any measurement.
//!
//! Per the repo's experimental protocol: "every approximation has a trivial
//! limit where it does nothing". A dispersion correction has three:
//!
//!   1. A SINGLE ATOM has no pair to disperse with     => E_disp == 0 exactly.
//!   2. An INFINITELY SEPARATED pair has no overlap    => E_disp -> 0.
//!   3. ZERO scaling coefficients (s6 = s8 = 0)        => E_disp == 0 exactly.
//!
//! (1) and (3) are EXACT zeros (empty sum / multiplication by zero), so they
//! are asserted at bit level, not at a tolerance. (2) is a limit, so it is
//! asserted as a decay: the R^-6 leading term means E(2R) must be at least
//! ~50x smaller than E(R) far outside the damping radius, and the magnitude at
//! 1000 Bohr must be below any energy anyone would report.
//!
//! These must pass before ANY number is measured.

use ferric_d3::{d3bj_energy, d3bj_gradient, D3Params};

/// Damping parameters for PBE (Grimme JCC 32, 1456 (2011) Table 1).
/// Used only as a concrete non-trivial parameter set for the anchors; the
/// anchors are about the STRUCTURE of the sum, not about these values.
fn pbe() -> D3Params {
    D3Params {
        s6: 1.0,
        s8: 0.7875,
        a1: 0.4289,
        a2: 4.4407,
    }
}

/// Anchor 1: a single atom has no pairs, so the dispersion sum is EMPTY.
///
/// This is an exact zero (a sum over zero terms), so it is asserted exactly.
/// A `!= 0.0` here means the implementation fabricated a one-body term.
#[test]
fn single_atom_has_exactly_zero_dispersion() {
    for z in [1u8, 6, 7, 8, 9, 17, 35] {
        let e = d3bj_energy(&[z], &[[0.0, 0.0, 0.0]], &pbe()).unwrap();
        assert_eq!(
            e, 0.0,
            "a single atom (Z={z}) must have EXACTLY zero dispersion energy, got {e:e}"
        );
    }
}

/// Anchor 1b: the empty system is also exactly zero, and must not panic.
#[test]
fn empty_system_has_exactly_zero_dispersion() {
    let e = d3bj_energy(&[], &[], &pbe()).unwrap();
    assert_eq!(e, 0.0, "the empty system must have exactly zero dispersion");
}

/// Anchor 2: an infinitely separated pair has no dispersion.
///
/// A limit, not an exact zero, so this asserts BOTH that the magnitude is
/// negligible at large R AND that it decays at the R^-6 rate the leading
/// term demands. Asserting only "small at 1000 Bohr" would also pass for an
/// implementation that returns a constant tiny number, so the decay ratio is
/// the load-bearing half of this test.
#[test]
fn infinitely_separated_pair_tends_to_zero() {
    let p = pbe();
    let e_at = |r: f64| d3bj_energy(&[6, 6], &[[0.0, 0.0, 0.0], [0.0, 0.0, r]], &p).unwrap();

    let e_1000 = e_at(1000.0);
    assert!(
        e_1000.abs() < 1e-14,
        "a pair 1000 Bohr apart must have negligible dispersion, got {e_1000:e}"
    );

    // R^-6 decay: doubling R must cut the energy by >= ~50x (64x ideal, with
    // slack for the R^-8 term and for the damping denominator's constant a2).
    let e_20 = e_at(20.0);
    let e_40 = e_at(40.0);
    assert!(
        e_20.abs() > 0.0,
        "the 20 Bohr pair must have a NONZERO energy, else the decay ratio below \
         is vacuous (0/0). Got {e_20:e}"
    );
    let ratio = e_20.abs() / e_40.abs();
    assert!(
        ratio > 50.0,
        "doubling the separation must cut E_disp by >=50x (R^-6 leading term); \
         E(20)={e_20:e}, E(40)={e_40:e}, ratio={ratio:.2}"
    );
}

/// Anchor 3: zeroed scaling coefficients switch the correction off EXACTLY.
///
/// This is the "vacuous mask" limit. It is distinct from anchors 1 and 2: those
/// zero the GEOMETRY, this zeroes the PARAMETERS, so it catches a term that was
/// added outside the s6/s8 scaling (e.g. a stray constant or an unscaled ATM
/// contribution).
#[test]
fn zero_scaling_coefficients_give_exactly_zero() {
    let off = D3Params {
        s6: 0.0,
        s8: 0.0,
        a1: 0.4289,
        a2: 4.4407,
    };
    // A real, compact molecule -- the geometry where the correction is LARGEST,
    // so a leak is most visible.
    let z = [8u8, 1, 1];
    let xyz = [[0.0, 0.0, 0.0], [0.0, 1.43, 1.11], [0.0, -1.43, 1.11]];
    let e = d3bj_energy(&z, &xyz, &off).unwrap();
    assert_eq!(
        e, 0.0,
        "s6 = s8 = 0 must switch the correction off EXACTLY, got {e:e}"
    );

    // ... and the same geometry with real parameters must be NONZERO, else the
    // assertion above is vacuous (it would pass for a function that always
    // returns 0.0).
    let e_on = d3bj_energy(&z, &xyz, &pbe()).unwrap();
    assert!(
        e_on.abs() > 1e-9,
        "the same geometry with real parameters must be NONZERO, else the \
         zero-parameter anchor is vacuous. Got {e_on:e}"
    );
}

/// Anchor 4: dispersion is ATTRACTIVE. A positive total on a bound geometry
/// is a sign error, which no magnitude-only test would catch.
#[test]
fn dispersion_is_attractive() {
    let e = d3bj_energy(&[6, 6], &[[0.0, 0.0, 0.0], [0.0, 0.0, 7.0]], &pbe()).unwrap();
    assert!(
        e < 0.0,
        "D3 dispersion must be ATTRACTIVE (negative), got {e:e}"
    );
}

/// Anchor 5: translation and rotation invariance. The energy depends only on
/// interatomic distances, so rigid motions must not change it.
#[test]
fn energy_is_invariant_under_rigid_motion() {
    let p = pbe();
    let z = [8u8, 1, 1];
    let xyz = [[0.0, 0.0, 0.0], [0.0, 1.43, 1.11], [0.0, -1.43, 1.11]];
    let e0 = d3bj_energy(&z, &xyz, &p).unwrap();

    // Translation by an arbitrary vector.
    let shifted: Vec<[f64; 3]> = xyz
        .iter()
        .map(|c| [c[0] + 3.7, c[1] - 1.2, c[2] + 9.9])
        .collect();
    let e_t = d3bj_energy(&z, &shifted, &p).unwrap();
    assert!(
        (e_t - e0).abs() < 1e-14,
        "translation must not change E_disp: {e0:e} vs {e_t:e}"
    );

    // Rotation by 90 degrees about z: (x, y, z) -> (-y, x, z).
    let rotated: Vec<[f64; 3]> = xyz.iter().map(|c| [-c[1], c[0], c[2]]).collect();
    let e_r = d3bj_energy(&z, &rotated, &p).unwrap();
    assert!(
        (e_r - e0).abs() < 1e-14,
        "rotation must not change E_disp: {e0:e} vs {e_r:e}"
    );
}

/// Anchor 6: an unsupported element must ERROR, never silently contribute zero.
///
/// Repo convention (`tools/campaign/hierarchy.py` rule 5): a tier that cannot
/// answer returns None/UNEVALUATED, never a neutral-looking number. Silently
/// treating an out-of-table element as having no dispersion would be exactly
/// that fabricated neutral value.
#[test]
fn unsupported_element_errors_rather_than_contributing_zero() {
    // Z = 104 (rutherfordium) is beyond any D3 parameterisation.
    let r = d3bj_energy(&[104, 1], &[[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]], &pbe());
    assert!(
        r.is_err(),
        "an element outside the D3 parameter table must ERROR, not silently \
         contribute zero dispersion; got {r:?}"
    );
}

/// Non-finite coordinates must be REFUSED, not propagated.
///
/// MEASURED before the guard:
///
/// ```text
///   NaN coordinate -> Ok(NaN)
///   inf coordinate -> Ok(0.0)      <- the dangerous one
/// ```
///
/// The zero is what makes this worth a hard error. A NaN propagates visibly
/// into whatever consumes it and someone eventually notices; `0.0` is
/// indistinguishable from "computed, and the atoms are simply far apart", so a
/// broken geometry reports no dispersion and the run looks fine.
///
/// Reachable from an optimizer that took a bad step, or a coordinate that came
/// through a failed unit conversion.
#[test]
fn non_finite_coordinates_are_refused_by_both_energy_and_gradient() {
    let p = D3Params {
        s6: 1.0,
        s8: 1.2177,
        a1: 0.4145,
        a2: 4.8593,
    };
    for (tag, bad) in [("NaN", f64::NAN), ("inf", f64::INFINITY)] {
        let coords = vec![[0.0, 0.0, 0.0], [0.0, 0.0, bad]];
        let e = d3bj_energy(&[18, 18], &coords, &p);
        assert!(
            e.is_err(),
            "{tag} coordinate must be refused by the energy, got {e:?}"
        );
        assert!(
            e.unwrap_err().to_string().contains("non-finite"),
            "{tag}: the error must name the cause"
        );
        assert!(
            d3bj_gradient(&[18, 18], &coords, &p).is_err(),
            "{tag} coordinate must be refused by the gradient too"
        );
    }

    // THE ANCHOR: a finite geometry still works, so this is about the VALUES
    // and not about the guard rejecting everything.
    let good = vec![[0.0, 0.0, 0.0], [0.0, 0.0, 7.1]];
    assert!(d3bj_energy(&[18, 18], &good, &p).is_ok());
    assert!(d3bj_gradient(&[18, 18], &good, &p).is_ok());
}
