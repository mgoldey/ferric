//! Stage 1 exactness anchor for the COSX per-grid-point A-matrix primitive.
//!
//! # Why this test exists
//!
//! `ferric_integrals::cosx_a::a_matrix_at_point` retains the per-grid-point
//! block `A^g_{mu,nu}` as a matrix. Nothing about "it returned a matrix of
//! plausible numbers" proves those numbers are the right integrals. The
//! repo's CONSISTENCY-IS-NOT-CORROBORATION rule says agreement between two
//! runs of the *same* construction proves nothing about systematic error; you
//! need an **independent construction**.
//!
//! The independent construction available here is
//! [`ferric_scf::properties::esp_at_points`], which was written for a
//! completely different purpose (electrostatic potentials for CHELPG/RESP),
//! is validated against PySCF, and consumes the same libint2 primitive by
//! contracting it against the density on the fly instead of retaining it.
//! Contracting the retained blocks against `D` and adding the nuclear term
//! must therefore reproduce `esp_at_points` exactly -- not to quadrature
//! accuracy, but to floating-point round-off, because it is literally the
//! same sum performed in a different order.
//!
//! # Sign bookkeeping (deliberate asymmetry)
//!
//! `cosx_a` returns the **repulsive** `+1/|r-r_g|` kernel that COSX needs,
//! whereas libint2 (and hence `esp_at_points`) works with the **attractive**
//! `-1/|r-r_g|`. `esp_at_points` computes `V_elec = + sum_munu D_munu <mu|-1/r|nu>`.
//! So reconstructing it from `cosx_a`'s blocks requires flipping the sign back:
//!
//! ```text
//!     V_elec = - sum_munu D_munu A^g_{mu,nu}
//! ```
//!
//! Getting this backwards is exactly the bug that Stage 0's python prototype
//! hit, so it is asserted here rather than assumed.

use ferric_core::basis::bundled;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::cosx_a::{a_matrix_at_point, CosxScreen, PairBounds};
use ferric_scf::properties::esp_at_points;
use ndarray::Array2;

/// Water, a few probe points deliberately placed off-atom and off-symmetry so
/// that no accidental cancellation can hide an index-transposition error.
fn setup() -> (Molecule, PreparedBasis, Array2<f64>, Vec<[f64; 3]>) {
    let mol = Molecule::parse_xyz(
        "3
water
O  0.0000  0.0000  0.1173
H  0.0000  0.7572 -0.4692
H  0.0000 -0.7572 -0.4692
",
        0,
        1,
    )
    .expect("parse water");

    let bs = bundled("sto-3g").expect("sto-3g");
    let prep = PreparedBasis::new(&mol, &bs).expect("prepared basis");
    let nbf = prep.nbasis();

    // A deliberately NON-symmetric, non-diagonal density-like matrix. Using a
    // real SCF density would work too, but a synthetic asymmetric-then-
    // symmetrized matrix makes an accidental transpose visible: a symmetric
    // test matrix cannot detect a mu<->nu swap.
    let mut d = Array2::<f64>::zeros((nbf, nbf));
    for i in 0..nbf {
        for j in 0..nbf {
            d[(i, j)] = 0.1 * ((i * 7 + j * 13) % 11) as f64 + 0.01 * (i as f64) - 0.003 * (j as f64);
        }
    }
    // esp_at_points assumes a symmetric density (it sweeps the lower triangle
    // and doubles the off-diagonal), so symmetrize -- but the *values* remain
    // irregular, which is what defeats accidental cancellation.
    let dsym = 0.5 * (&d + &d.t());

    let points = vec![
        [0.3, 0.2, 0.7],
        [-1.1, 0.45, -0.2],
        [2.0, -1.3, 0.9],
        [0.0, 0.0, 3.5],
    ];

    (mol, prep, dsym, points)
}

/// Reconstruct `esp_at_points`' scalar from the retained COSX blocks.
///
/// `sign` is the factor applied to the contraction; the physical value is
/// `-1.0`. The parameter exists so the mutation test can flip it and show the
/// assertion actually fires.
fn esp_from_retained_blocks(
    mol: &Molecule,
    prep: &PreparedBasis,
    d: &Array2<f64>,
    points: &[[f64; 3]],
    sign: f64,
) -> Vec<f64> {
    points
        .iter()
        .map(|r| {
            let pt = a_matrix_at_point(prep, r, None, CosxScreen::none()).expect("A matrix");
            // V_elec = sign * sum_munu D_munu A_munu  (physical sign = -1)
            let v_elec: f64 = sign * (d * &pt.a).sum();

            let mut v_nuc = 0.0_f64;
            for atom in &mol.atoms {
                let dx = r[0] - atom.x;
                let dy = r[1] - atom.y;
                let dz = r[2] - atom.zpos;
                v_nuc += atom.z as f64 / (dx * dx + dy * dy + dz * dz).sqrt();
            }
            v_nuc + v_elec
        })
        .collect()
}

#[test]
fn cosx_a_blocks_reproduce_esp_at_points() {
    let (mol, prep, d, points) = setup();

    let reference = esp_at_points(&mol, &prep, &d, &points).expect("esp_at_points");
    let reconstructed = esp_from_retained_blocks(&mol, &prep, &d, &points, -1.0);

    assert_eq!(reference.len(), reconstructed.len());
    let mut max_dev = 0.0_f64;
    for (i, (a, b)) in reference.iter().zip(reconstructed.iter()).enumerate() {
        let dev = (a - b).abs();
        max_dev = max_dev.max(dev);
        assert!(
            dev < 1e-12,
            "point {i}: esp_at_points={a:.16e} vs retained-block reconstruction={b:.16e} \
             (dev {dev:.3e} exceeds 1e-12)"
        );
    }
    // Report so a human reading CI output sees the actual agreement, not just
    // a green check.
    println!("cosx_a anchor: max deviation vs esp_at_points = {max_dev:.3e}");

    // Guard against the degenerate pass where every value is ~0 (which would
    // satisfy the tolerance vacuously).
    let max_abs = reference.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    assert!(
        max_abs > 0.1,
        "reference ESP values are all ~0 ({max_abs:.3e}); the anchor would pass vacuously"
    );
}

/// Mutation test: flipping the probe/contraction sign MUST break the anchor.
///
/// A test that has never been seen to fail is an assumption. This runs the
/// same reconstruction with the wrong sign and asserts the agreement is
/// destroyed -- proving the 1e-12 bar above is actually load-bearing and not
/// satisfied by construction.
#[test]
fn cosx_a_anchor_fails_under_sign_mutation() {
    let (mol, prep, d, points) = setup();

    let reference = esp_at_points(&mol, &prep, &d, &points).expect("esp_at_points");
    let mutated = esp_from_retained_blocks(&mol, &prep, &d, &points, 1.0);

    let max_dev = reference
        .iter()
        .zip(mutated.iter())
        .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));

    assert!(
        max_dev > 1e-6,
        "sign mutation did NOT break the anchor (max dev {max_dev:.3e}); the anchor is vacuous"
    );
    println!("cosx_a mutation proof: wrong sign gives max deviation {max_dev:.3e} (anchor is live)");
}

/// Stage 2 anchor (i), written BEFORE the sweep: the screened A-matrix must
/// reproduce the unscreened one in the zero-threshold limit.
#[test]
fn cosx_a_zero_threshold_matches_unscreened() {
    let (_mol, prep, _d, points) = setup();
    let bounds = PairBounds::build(&prep).expect("pair bounds");

    for (i, r) in points.iter().enumerate() {
        let unscreened = a_matrix_at_point(&prep, r, None, CosxScreen::none()).expect("unscreened");
        let screened =
            a_matrix_at_point(&prep, r, Some(&bounds), CosxScreen::at(0.0)).expect("screened");

        assert_eq!(
            screened.pairs_kept, screened.pairs_total,
            "point {i}: zero threshold dropped pairs ({} of {})",
            screened.pairs_kept, screened.pairs_total
        );
        let dev = (&unscreened.a - &screened.a).mapv(f64::abs).fold(0.0_f64, |m, &v| m.max(v));
        assert!(
            dev < 1e-14,
            "point {i}: zero-threshold screened A differs from unscreened by {dev:.3e}"
        );
    }
}

/// Stage 2 anchor (ii): reachability. A screen that never drops anything at
/// the production threshold makes every screened measurement vacuous, so
/// prove it drops pairs — on a case where a SOUND bound can: two waters
/// 28 Bohr apart, whose 25 intermolecular shell pairs have negligible
/// overlap (K_AB ~ exp(-p R^2)). Paired with the soundness half: whatever
/// was dropped must contribute below the threshold, and no intramolecular
/// pair may go.
///
/// The original version probed a single water from 40 Bohr away and expected
/// drops. Only the unsound signed-overlap bound could satisfy that — a valid
/// 1/R bound on a compact pair at 40 Bohr is ~1e-2, far above 1e-6 — so that
/// expectation was itself evidence of the bug (#46).
#[test]
fn cosx_screen_actually_drops_pairs() {
    let mol = Molecule::parse_xyz(
        "6\nwater dimer, 15 A apart\n\
         O 0 0 0\nH 0 0.7572 0.5868\nH 0 -0.7572 0.5868\n\
         O 15 0 0\nH 15 0.7572 0.5868\nH 15 -0.7572 0.5868\n",
        0,
        1,
    )
    .expect("water dimer");
    let bs = bundled("sto-3g").expect("sto-3g");
    let prep = PreparedBasis::new(&mol, &bs).expect("prepared basis");
    let bounds = PairBounds::build(&prep).expect("pair bounds");
    let t = 1e-7;
    // Near the first water (Bohr), off every nucleus.
    let probe = [0.5, 0.2, 0.3];

    let unscreened = a_matrix_at_point(&prep, &probe, None, CosxScreen::none()).expect("unscreened");
    let screened = a_matrix_at_point(&prep, &probe, Some(&bounds), CosxScreen::at(t)).expect("screened");

    // Pin the pair-count convention so the intramolecular arithmetic below is
    // checked, not assumed: upper triangle including the diagonal.
    let nsh = bounds.nshells();
    assert_eq!(screened.pairs_total, nsh * (nsh + 1) / 2, "pair-count convention changed");
    let nsh_per_water = nsh / 2;
    let intramolecular = 2 * (nsh_per_water * (nsh_per_water + 1) / 2);

    assert!(
        screened.pairs_kept < screened.pairs_total,
        "screen at {t:e} kept ALL {} pairs on a 28-Bohr water dimer; the screen is vacuous",
        screened.pairs_total
    );
    assert!(
        screened.pairs_kept >= intramolecular,
        "screen dropped an intramolecular pair: kept {} < {intramolecular}",
        screened.pairs_kept
    );
    let dev = (&unscreened.a - &screened.a).mapv(f64::abs).fold(0.0_f64, |m, &v| m.max(v));
    assert!(
        dev < t,
        "screen at {t:e} dropped pairs that mattered: max|A_unscr - A_scr| = {dev:.3e}"
    );
    println!(
        "cosx screen reachability (water dimer, 28 Bohr): {} of {} pairs kept (>= {intramolecular} intramolecular), max dev {dev:.3e}",
        screened.pairs_kept, screened.pairs_total
    );
}
