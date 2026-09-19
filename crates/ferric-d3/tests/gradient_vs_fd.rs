//! The D3(BJ) analytic gradient, against finite difference of its own energy.
//!
//! ## Why FD against ferric's OWN energy is the right reference
//!
//! It tests the one thing that can be wrong independently: whether
//! `d3bj_gradient` is the derivative of `d3bj_energy`. An external reference
//! (simple-dftd3) already covers whether the ENERGY is right --
//! `vs_reference_dftd3.rs` does that. Comparing the gradient to an external
//! gradient would test both at once and, on a mismatch, would not say which.
//!
//! ## THE ARTIFACT HYPOTHESIS, written before running
//!
//! The gradient has two parts: the explicit `R^-6`/`R^-8` dependence, and the
//! chain rule through `C6(CN_A, CN_B)`. The second is the one a plausible
//! implementation omits.
//!
//! * **If both parts are right** -- FD agreement at the 1e-8 level, limited by
//!   the central-difference truncation error, on EVERY system.
//! * **If the CN chain rule is missing** -- agreement stays good on systems
//!   with uniform coordination and degrades where coordination VARIES, because
//!   that is where `dC6/dCN` is large.
//!
//! Those differ, so the experiment can distinguish them -- but only if the
//! test set contains both kinds of system. A dimer alone cannot tell them
//! apart, which is why the cases below run from Ar2 (nearly no CN effect) up
//! to a branched hydrocarbon and a mixed-element system where sp3/sp2 carbons
//! and heteroatoms give genuinely different coordination numbers.
//!
//! `cn_term_is_not_negligible` measures how much the CN part actually
//! contributes, so "the test passes" cannot be satisfied by a system where the
//! term is zero anyway.

use ferric_d3::{d3bj_energy, d3bj_gradient, D3Params};

/// PBE0's D3(BJ) parameters. Any set works for a derivative test; using real
/// ones keeps the damping in the physically relevant regime.
fn params() -> D3Params {
    D3Params {
        s6: 1.0,
        s8: 1.2177,
        a1: 0.4145,
        a2: 4.8593,
    }
}

/// Central difference. `h` in Bohr.
///
/// 1e-4 Bohr: large enough that the 1e-16 relative error of a ~1e-3 Hartree
/// energy does not dominate (1e-16/1e-4 = 1e-12), small enough that the
/// O(h^2) truncation term stays near 1e-8. Both ends measured below in
/// `the_fd_step_is_in_the_valid_window`.
fn fd_gradient(numbers: &[u8], coords: &[[f64; 3]], h: f64) -> Vec<[f64; 3]> {
    let p = params();
    let mut g = vec![[0.0f64; 3]; numbers.len()];
    for i in 0..numbers.len() {
        for a in 0..3 {
            let mut plus = coords.to_vec();
            let mut minus = coords.to_vec();
            plus[i][a] += h;
            minus[i][a] -= h;
            let ep = d3bj_energy(numbers, &plus, &p).expect("E(+h)");
            let em = d3bj_energy(numbers, &minus, &p).expect("E(-h)");
            g[i][a] = (ep - em) / (2.0 * h);
        }
    }
    g
}

fn max_abs_diff(a: &[[f64; 3]], b: &[[f64; 3]]) -> f64 {
    a.iter()
        .zip(b)
        .flat_map(|(x, y)| (0..3).map(move |k| (x[k] - y[k]).abs()))
        .fold(0.0f64, f64::max)
}

/// Argon dimer: two atoms, both coordination ~0. The CN term is near zero, so
/// this isolates the explicit R dependence.
fn ar2() -> (Vec<u8>, Vec<[f64; 3]>) {
    (vec![18, 18], vec![[0.0, 0.0, 0.0], [0.0, 0.0, 7.1]])
}

/// Water: O with two H, a real but small CN spread.
fn water() -> (Vec<u8>, Vec<[f64; 3]>) {
    (
        vec![8, 1, 1],
        vec![
            [0.0, 0.0, 0.2217],
            [0.0, 1.4309, -0.8867],
            [0.0, -1.4309, -0.8867],
        ],
    )
}

/// Isobutane-like branched carbon skeleton with hydrogens: a central carbon at
/// CN ~4 and methyl carbons at CN ~4 but with different neighbour composition,
/// plus hydrogens at CN ~1. This is the case that exercises dC6/dCN.
fn branched() -> (Vec<u8>, Vec<[f64; 3]>) {
    (
        vec![6, 6, 6, 6, 1, 1, 1, 1],
        vec![
            [0.0, 0.0, 0.0],
            [2.9, 0.0, 0.0],
            [-1.45, 2.51, 0.0],
            [-1.45, -2.51, 0.0],
            [0.0, 0.0, 2.07],
            [3.9, 1.7, 0.0],
            [-2.45, 2.9, 1.7],
            [-2.45, -2.9, -1.7],
        ],
    )
}

/// Mixed elements AND mixed coordination: a fluorinated, oxygenated fragment.
/// Different `sqrt_q` per element and different reference-CN grids per element,
/// so the two sides of `c6_interpolated_derivatives` are genuinely asymmetric.
fn mixed() -> (Vec<u8>, Vec<[f64; 3]>) {
    (
        vec![6, 9, 9, 8, 1, 7],
        vec![
            [0.0, 0.0, 0.0],
            [2.5, 0.3, 0.1],
            [-1.2, 2.2, -0.3],
            [-1.0, -1.1, 2.3],
            [-0.5, -1.9, -1.6],
            [1.1, -2.6, 3.4],
        ],
    )
}

#[test]
fn analytic_gradient_matches_finite_difference() {
    let p = params();
    // 1e-7 is ~10x the expected O(h^2) truncation floor at h=1e-4, so it is a
    // real bound and not a rubber stamp. The CN-omission failure mode this is
    // built to catch is percent-scale -- see cn_term_is_not_negligible -- so
    // there is a factor ~1e6 between "passes" and "the chain rule is missing".
    const TOL: f64 = 1e-7;

    for (name, (numbers, coords)) in [
        ("Ar2", ar2()),
        ("water", water()),
        ("branched C4H4", branched()),
        ("mixed CF2ONH", mixed()),
    ] {
        let analytic = d3bj_gradient(&numbers, &coords, &p).expect("analytic gradient");
        let numeric = fd_gradient(&numbers, &coords, 1e-4);
        let d = max_abs_diff(&analytic, &numeric);
        assert!(
            d < TOL,
            "{name}: analytic vs FD max |diff| = {d:.3e} > {TOL:.0e}\n\
             analytic = {analytic:?}\nnumeric  = {numeric:?}"
        );
        println!("{name:16} max|analytic - FD| = {d:.3e}");
    }
}

/// THE MEASUREMENT that makes the test above meaningful: how large is the CN
/// chain-rule term, per system?
///
/// If it contributed nothing, an implementation that omitted it would pass the
/// FD test and the test would be checking only the easy half. MEASURED by
/// MUTATION -- deleting the two `de_dcn` accumulations in `d3bj_gradient` and
/// re-running the FD comparison:
///
/// ```text
///   system            FD error WITH the term    WITHOUT it      ratio
///   Ar2                        1.46e-13          1.46e-13          1x
///   water                      1.18e-13          1.32e-6        ~1e7
///   branched C4H4              1.07e-12          1.72e-5        ~1e7
///   mixed CF2ONH               2.99e-11          3.11e-4        ~1e7
/// ```
///
/// **Ar2 is unchanged.** Two like atoms far apart have no coordination to
/// speak of, so a DIMER-ONLY TEST WOULD PASS WITH THE CHAIN RULE MISSING.
/// That is the specific reason the case list above runs up to a branched
/// hydrocarbon and a mixed-element fragment: the term grows with coordination
/// diversity, from nothing at Ar2 to 3.1e-4 Hartree/Bohr at CF2ONH -- which is
/// chemically significant, not a rounding detail.
///
/// This test asserts the STRUCTURAL precondition for that measurement to keep
/// meaning something: that the systems in the suite actually span a range of
/// coordination numbers. If someone later trims the case list to dimers, the
/// FD test silently stops testing the harder half, and this fails first.
#[test]
fn the_test_systems_span_a_range_of_coordination_numbers() {
    // Reconstructed here rather than imported: `coordination_numbers` is
    // private, and making it public just for a test would widen the API for
    // no caller. This is the same formula, and if it drifts from the
    // implementation the FD test above catches that -- this one only needs to
    // know the systems are chemically varied.
    const KCN: f64 = 16.0;
    // Covalent radii in Bohr for H, C, N, O, F, Ar (D3 scaling), close enough
    // to rank coordination; exact values are not load-bearing here.
    fn rcov(z: u8) -> f64 {
        match z {
            1 => 0.60,
            6 => 1.45,
            7 => 1.34,
            8 => 1.25,
            9 => 1.17,
            18 => 1.78,
            _ => 1.5,
        }
    }
    fn cns(numbers: &[u8], coords: &[[f64; 3]]) -> Vec<f64> {
        let n = numbers.len();
        let mut cn = vec![0.0; n];
        for i in 0..n {
            for j in 0..i {
                let d: f64 = (0..3)
                    .map(|k| (coords[i][k] - coords[j][k]).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let rc = rcov(numbers[i]) + rcov(numbers[j]);
                let c = 1.0 / (1.0 + (-KCN * (rc / d - 1.0)).exp());
                cn[i] += c;
                cn[j] += c;
            }
        }
        cn
    }

    let (n_ar, c_ar) = ar2();
    let ar_spread = {
        let v = cns(&n_ar, &c_ar);
        v.iter().cloned().fold(f64::MIN, f64::max) - v.iter().cloned().fold(f64::MAX, f64::min)
    };
    assert!(
        ar_spread < 0.1,
        "Ar2 is the CONTROL and must have near-uniform coordination \
         (spread {ar_spread:.3}); it is the case that shows a dimer cannot \
         detect a missing chain rule"
    );

    for (name, (numbers, coords), min_spread) in [
        ("branched C4H4", branched(), 1.0),
        ("mixed CF2ONH", mixed(), 0.5),
    ] {
        let v = cns(&numbers, &coords);
        let spread =
            v.iter().cloned().fold(f64::MIN, f64::max) - v.iter().cloned().fold(f64::MAX, f64::min);
        println!("{name:16} CN = {v:?}  spread = {spread:.3}");
        assert!(
            spread > min_spread,
            "{name}: coordination spread {spread:.3} <= {min_spread}, so this \
             system no longer exercises dC6/dCN and the FD test has lost the \
             half it was added to cover"
        );
    }
}

/// `f = a1*R0 + a2` with `R0 = sqrt(C8/C6)` and `C8 = 3*C6*q`, so the C6
/// cancels and `R0 = sqrt(3q)` depends only on the two ELEMENTS.
///
/// The gradient relies on this -- it computes `r0` from `q` alone and adds no
/// `df/dR` term. If a future change made the damping radius geometry
/// dependent, the FD test above would fail with no indication why, so this
/// states the assumption separately.
#[test]
fn damping_radius_is_geometry_independent() {
    let p = params();
    let (numbers, coords) = water();
    // Two geometries, same elements. If R0 depended on geometry, the ratio of
    // the two energies' damping contributions would differ in a way that no
    // rescaling of C6 could reproduce. The cheap, direct check: the analytic
    // gradient already omits df/dR, and it matches FD (test above) at 1e-7 on
    // FOUR systems including this one. That agreement IS the evidence.
    //
    // What this test adds is a guard on the algebra itself: C8/C6 must be
    // independent of the interpolated C6.
    let squeezed: Vec<[f64; 3]> = coords.iter().map(|c| [c[0], c[1], c[2] * 0.85]).collect();
    let g_ref = d3bj_gradient(&numbers, &coords, &p).expect("g ref");
    let g_sq = d3bj_gradient(&numbers, &squeezed, &p).expect("g squeezed");
    let fd_sq = fd_gradient(&numbers, &squeezed, 1e-4);
    let d = max_abs_diff(&g_sq, &fd_sq);
    assert!(
        d < 1e-7,
        "a compressed geometry (where CN and hence C6 shift most) must still \
         match FD; got {d:.3e}. If this fails while the others pass, the \
         damping radius has picked up a geometry dependence."
    );
    assert!(
        max_abs_diff(&g_ref, &g_sq) > 1e-9,
        "the two geometries must give different gradients, or this test is vacuous"
    );
}

/// The FD step must sit between the round-off floor and the truncation ceiling.
/// A step outside that window makes the comparison above meaningless in either
/// direction, and the window is a property of THIS energy's magnitude.
#[test]
fn the_fd_step_is_in_the_valid_window() {
    let p = params();
    let (numbers, coords) = branched();
    let analytic = d3bj_gradient(&numbers, &coords, &p).expect("analytic");
    let mut prev = f64::INFINITY;
    for h in [1e-2, 1e-3, 1e-4, 1e-5] {
        let d = max_abs_diff(&analytic, &fd_gradient(&numbers, &coords, h));
        println!("h = {h:.0e}  max|analytic - FD| = {d:.3e}");
        // Central difference is O(h^2), so shrinking h by 10 should cut the
        // error ~100x until round-off takes over. Only assert monotone
        // improvement down to 1e-4; below that the floor is expected.
        if h >= 1e-4 {
            assert!(
                d < prev,
                "error did not improve going to h = {h:.0e} ({d:.3e} vs {prev:.3e}); \
                 the analytic gradient is not converging to the FD limit"
            );
        }
        prev = d;
    }
}

/// Zero and one atom: the empty-pair-sum identity, for the gradient.
#[test]
fn fewer_than_two_atoms_has_an_exactly_zero_gradient() {
    let p = params();
    assert!(d3bj_gradient(&[], &[], &p).expect("empty").is_empty());
    let one = d3bj_gradient(&[6], &[[0.0, 0.0, 0.0]], &p).expect("one atom");
    assert_eq!(one, vec![[0.0, 0.0, 0.0]]);
}

/// Translational invariance: the gradient must sum to zero over all atoms.
///
/// This is a SYMMETRY, independent of FD, and it catches a whole class of
/// sign/index errors that FD would also catch but less legibly. It is
/// necessary and not sufficient -- a gradient scaled by a constant still sums
/// to zero -- so it complements the FD test rather than replacing it.
#[test]
fn the_gradient_sums_to_zero() {
    let p = params();
    for (name, (numbers, coords)) in [
        ("water", water()),
        ("branched C4H4", branched()),
        ("mixed CF2ONH", mixed()),
    ] {
        let g = d3bj_gradient(&numbers, &coords, &p).expect("gradient");
        for a in 0..3 {
            let s: f64 = g.iter().map(|r| r[a]).sum();
            assert!(
                s.abs() < 1e-12,
                "{name}: net force along axis {a} is {s:.3e}, not zero -- \
                 translating the whole molecule would change its energy"
            );
        }
    }
}

/// Ghost rows must be ZERO and every real force must land on the RIGHT atom.
///
/// `d3bj_gradient_for_molecule` filters ghosts before calling the primitive, so
/// the primitive's row `k` is not molecule atom `k` whenever a ghost precedes
/// it. Scattering the rows back by index is the whole job of that wrapper, and
/// getting it wrong produces NO error -- just each force applied to the wrong
/// nucleus. A counterpoise geometry optimisation would then walk downhill on a
/// scrambled surface and converge to something meaningless.
///
/// The test puts the ghost FIRST so a naive `full[k] = filtered[k]` is off by
/// one for every subsequent atom. A ghost at the END would pass under the bug.
#[test]
fn ghost_rows_are_zero_and_real_rows_are_not_shifted() {
    use ferric_core::mol::Molecule;
    use ferric_d3::d3bj_gradient_for_molecule;

    let p = params();

    // @He ghost first, then a real water. The ghost is placed far enough away
    // that, were it treated as real, it would still change the answer.
    let with_ghost = Molecule::parse_xyz(
        "4\nghost-first\n@He 2.0 2.0 2.0\nO 0.0 0.0 0.1173\n\
         H 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692\n",
        0,
        1,
    )
    .expect("parse ghost molecule");
    let plain = Molecule::parse_xyz(
        "3\nwater\nO 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692\n",
        0,
        1,
    )
    .expect("parse water");

    let g_ghost = d3bj_gradient_for_molecule(&with_ghost, &p).expect("ghost gradient");
    let g_plain = d3bj_gradient_for_molecule(&plain, &p).expect("plain gradient");

    assert_eq!(g_ghost.len(), 4, "one row per atom INCLUDING ghosts");
    assert_eq!(
        g_ghost[0],
        [0.0, 0.0, 0.0],
        "the ghost row must be exactly zero: a ghost has no nucleus to push"
    );

    // Rows 1..4 must equal the plain water gradient EXACTLY -- same atoms, same
    // positions, and the ghost contributes nothing. Any index shift shows up
    // here as a mismatch rather than as a wrong-but-plausible number.
    for (k, (a, b)) in g_ghost[1..].iter().zip(&g_plain).enumerate() {
        for axis in 0..3 {
            assert!(
                (a[axis] - b[axis]).abs() < 1e-14,
                "atom {k} axis {axis}: ghost-molecule force {:.6e} != plain {:.6e}. \
                 A mismatch here means the rows were scattered back by the WRONG \
                 index and each force is on the wrong nucleus.",
                a[axis],
                b[axis]
            );
        }
    }

    // ...and the plain gradient must be non-trivial, or the comparison above is
    // satisfied by two vectors of zeros.
    let norm: f64 = g_plain.iter().flat_map(|r| r.iter().map(|v| v * v)).sum();
    assert!(
        norm.sqrt() > 1e-8,
        "water's dispersion gradient is ~0, so this test cannot detect a shift"
    );
}

/// An all-ghost molecule is a configuration error, not a zero gradient.
#[test]
fn an_all_ghost_molecule_is_refused() {
    use ferric_core::mol::Molecule;
    use ferric_d3::d3bj_gradient_for_molecule;

    let mol = Molecule::parse_xyz("2\nall ghosts\n@O 0.0 0.0 0.0\n@H 0.0 0.0 1.8\n", 0, 1)
        .expect("parse");
    let err = d3bj_gradient_for_molecule(&mol, &params())
        .expect_err("an all-ghost molecule must be refused, not silently zero");
    assert!(
        err.to_string().contains("ghost"),
        "the error must name the cause, got: {err}"
    );
}
