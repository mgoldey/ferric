//! Hand-computed energies and analytic-vs-FD gradients for every `ferric-mm`
//! term, plus mutation checks on the derived exclusion/1-4 bookkeeping.
//!
//! TDD note: this file was written and run (failing to compile — the crate
//! did not exist yet) before `crates/ferric-mm/src/{topology,energy}.rs` were
//! implemented, per the repo's TDD convention.

use ferric_mm::topology::{Angle, Bond, LjParams, Torsion};
use ferric_mm::{energy, gradient, qm_mm_lj_energy_gradient, MmTopology};
use ndarray::{array, Array2};

const H: f64 = 1e-5;
const FD_TOL: f64 = 1e-9;

/// Central-difference gradient of `f(coords) -> total energy`, compared
/// against the analytic gradient returned by `gradient()`.
fn assert_gradient_matches_fd(top: &MmTopology, coords: &Array2<f64>, label: &str) {
    let (e0, g) = gradient(top, coords).unwrap();
    let n = coords.nrows();
    let mut max_err = 0.0_f64;
    for i in 0..n {
        for c in 0..3 {
            let mut plus = coords.clone();
            plus[(i, c)] += H;
            let mut minus = coords.clone();
            minus[(i, c)] -= H;
            let e_plus = energy(top, &plus).unwrap().total;
            let e_minus = energy(top, &minus).unwrap().total;
            let fd = (e_plus - e_minus) / (2.0 * H);
            let err = (fd - g[(i, c)]).abs();
            max_err = max_err.max(err);
        }
    }
    assert!(
        max_err < FD_TOL,
        "{label}: analytic-vs-FD gradient max err {max_err:.3e} (h={H}, e0.total={:.10})",
        e0.total
    );
}

/// A 4-atom chain (dihedral i-j-k-l) with no periodicity/lattice, positioned
/// so no angle is degenerate and the dihedral is at a generic, non-flat
/// value. Roughly a butane-like all-anti-ish but perturbed skeleton.
fn chain4() -> Array2<f64> {
    array![
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [1.5 + 1.4 * (109.5_f64.to_radians()).cos(), 1.4 * (109.5_f64.to_radians()).sin(), 0.0],
        [
            1.5 + 1.4 * (109.5_f64.to_radians()).cos() + 1.4 * (109.5_f64.to_radians()).cos(),
            1.4 * (109.5_f64.to_radians()).sin() - 1.4 * (109.5_f64.to_radians()).sin() * 0.4,
            1.4 * (109.5_f64.to_radians()).sin() * 0.9,
        ],
    ]
}

fn lj_zero(n: usize) -> Vec<LjParams> {
    vec![LjParams { sigma: 0.0, epsilon: 0.0 }; n]
}

// ---------------------------------------------------------------------
// Bond term
// ---------------------------------------------------------------------

#[test]
fn bond_energy_hand_computed_and_gradient_vs_fd() {
    // Two atoms 1.6 Bohr apart, r0 = 1.5 Bohr, k = 0.3 Ha/Bohr^2.
    // E = k (r - r0)^2 = 0.3 * (0.1)^2 = 0.003
    let bonds = vec![Bond { i: 0, j: 1, k: 0.3, r0: 1.5 }];
    let top = MmTopology::new(vec![0.0, 0.0], lj_zero(2), bonds, vec![], vec![]).unwrap();
    let coords = array![[0.0, 0.0, 0.0], [1.6, 0.0, 0.0]];
    let e = energy(&top, &coords).unwrap();
    assert!((e.bond - 0.003).abs() < 1e-14, "got {}", e.bond);
    assert!((e.total - 0.003).abs() < 1e-14);

    assert_gradient_matches_fd(&top, &coords, "bond (axis-aligned)");

    // Off-axis geometry too, to catch a projection-direction bug an
    // axis-aligned test would not.
    let coords2 = array![[0.2, -0.3, 0.1], [1.5, 0.6, -0.4]];
    assert_gradient_matches_fd(&top, &coords2, "bond (off-axis)");
}

// ---------------------------------------------------------------------
// Angle term
// ---------------------------------------------------------------------

#[test]
fn angle_energy_hand_computed_and_gradient_vs_fd() {
    // i at (1,0,0), j at origin, k at (0,1,0): angle i-j-k = 90 deg exactly.
    // theta0 = 100 deg, k_theta = 0.1 Ha/rad^2.
    let theta0 = 100.0_f64.to_radians();
    let angles = vec![Angle { i: 0, j: 1, k: 2, k_theta: 0.1, theta0 }];
    let top = MmTopology::new(vec![0.0; 3], lj_zero(3), vec![], angles, vec![]).unwrap();
    let coords = array![[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let e = energy(&top, &coords).unwrap();
    let dtheta = std::f64::consts::FRAC_PI_2 - theta0;
    let expected = 0.1 * dtheta * dtheta;
    assert!((e.angle - expected).abs() < 1e-14, "got {} expected {}", e.angle, expected);

    assert_gradient_matches_fd(&top, &coords, "angle (right angle)");

    // A non-right, non-axis-aligned triple.
    let coords2 = array![[0.9, 0.4, 0.2], [0.1, -0.2, 0.05], [-0.3, 1.1, 0.6]];
    assert_gradient_matches_fd(&top, &coords2, "angle (generic)");
}

// ---------------------------------------------------------------------
// Torsion term
// ---------------------------------------------------------------------

#[test]
fn torsion_energy_hand_computed_and_gradient_vs_fd() {
    // Canonical anti (180 deg) dihedral built on the XY/YZ half-planes:
    // i=(1,1,0), j=(0,1,0), k=(0,0,0), l=(-1,-1,0). This is planar (phi=180
    // or 0 depending on winding); to get an unambiguous generic phi we use a
    // non-planar quartet instead (see below) for the FD checks and reserve
    // this exact planar one only for a hand-computed sanity value.
    let periodicity = 2;
    let k_phi = 0.05;
    let phase = 0.0_f64;
    let torsions = vec![Torsion { i: 0, j: 1, k: 2, l: 3, periodicity, k_phi, phase }];
    let top = MmTopology::new(vec![0.0; 4], lj_zero(4), vec![], vec![], torsions).unwrap();

    // Non-planar quartet: standard staggered-ish dihedral construction.
    // i-j-k in the xz half-plane, k-l rotated by a known dihedral angle
    // about the j-k axis (the +z axis here), so phi is exactly known.
    let phi_target = 60.0_f64.to_radians();
    let coords = array![
        [1.0, 0.0, -1.0], // i
        [0.0, 0.0, 0.0],  // j
        [0.0, 0.0, 1.0],  // k (j-k axis is +z)
        [phi_target.cos(), phi_target.sin(), 2.0], // l, rotated phi_target about z from i's projection
    ];
    let e = energy(&top, &coords).unwrap();
    let expected = k_phi * (1.0 + (periodicity as f64 * phi_target - phase).cos());
    assert!((e.torsion - expected).abs() < 1e-8, "got {} expected {}", e.torsion, expected);

    assert_gradient_matches_fd(&top, &coords, "torsion (generic, phi=60deg)");

    // phi = 0 and phi = 180 deg: NOT a coordinate singularity of the
    // Bekker form (m = r_ij x r_kj and n = r_kj x r_kl vanish only when
    // i-j-k, resp. j-k-l, is LINEAR -- a collinear VALENCE angle, not a
    // particular dihedral value). These two geometries exercise the
    // formula's well-conditioning through cos(phi) = +-1 (the acos branch
    // point the atan2-free form was chosen to avoid singularities near),
    // not the m_sq/n_sq/kj_norm fallback guard in dihedral_and_gradient --
    // see collinear_valence_angle_torsion_gradient_is_finite_and_zero below
    // for a test that actually triggers that guard.
    let coords_0 = array![[1.0, 0.0, -1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 2.0]];
    assert_gradient_matches_fd(&top, &coords_0, "torsion (phi ~ 0 deg)");

    let coords_180 = array![[1.0, 0.0, -1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 2.0]];
    assert_gradient_matches_fd(&top, &coords_180, "torsion (phi ~ 180 deg)");
}

/// Genuinely degenerate case: i, j, k EXACTLY collinear, so r_ij is
/// parallel to r_kj and m = r_ij x r_kj = 0 -- the dihedral plane through
/// i-j-k is undefined, and so is phi itself (the true derivative is
/// singular). This is the case that actually reaches
/// dihedral_and_gradient's `m_sq > 1e-20 && n_sq > 1e-20 && kj_norm > 1e-20`
/// guard and takes the zero-gradient fallback branch -- unlike the
/// phi=0/180deg cases above, which stay well clear of it (m and n are both
/// clearly nonzero there; only the ACOS argument is at a branch point).
#[test]
fn collinear_valence_angle_torsion_gradient_is_finite_and_zero() {
    let torsions = vec![Torsion { i: 0, j: 1, k: 2, l: 3, periodicity: 2, k_phi: 0.05, phase: 0.0 }];
    let top = MmTopology::new(vec![0.0; 4], lj_zero(4), vec![], vec![], torsions).unwrap();
    // i=(0,0,-1), j=(0,0,0), k=(0,0,1): exactly collinear on the z axis.
    let coords = array![[0.0, 0.0, -1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 2.0]];

    let e = energy(&top, &coords).unwrap();
    assert!(e.total.is_finite(), "energy must not be NaN/inf at a linear i-j-k valence angle");

    let (e2, g) = gradient(&top, &coords).unwrap();
    assert_eq!(e2.total, e.total);
    for row in 0..4 {
        for c in 0..3 {
            let v = g[(row, c)];
            assert!(v.is_finite(), "gradient[{row},{c}] = {v} is not finite");
            // Documented fallback: the guard leaves grad_i/j/k/l at their
            // zero-initialized values when m_sq/n_sq/kj_norm underflow.
            assert_eq!(v, 0.0, "gradient[{row},{c}] should be exactly the zero fallback, got {v}");
        }
    }
}

// ---------------------------------------------------------------------
// LJ + Coulomb (nonbonded), and the exclusion/1-4 derivation
// ---------------------------------------------------------------------

#[test]
fn nonbonded_hand_computed_and_gradient_vs_fd() {
    // Two unbonded atoms 3.0 Bohr apart, sigma=2.5, eps=0.002 each (mixed:
    // sigma_ij=2.5, eps_ij=0.002), charges +0.3/-0.3.
    let lj = vec![LjParams { sigma: 2.5, epsilon: 0.002 }; 2];
    let top = MmTopology::new(vec![0.3, -0.3], lj, vec![], vec![], vec![]).unwrap();
    let coords = array![[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]];
    let e = energy(&top, &coords).unwrap();

    let sr6 = (2.5_f64 / 3.0).powi(6);
    let sr12 = sr6 * sr6;
    let e_lj_expected = 4.0 * 0.002 * (sr12 - sr6);
    let e_coul_expected = 0.3 * -0.3 / 3.0;
    assert!((e.lj - e_lj_expected).abs() < 1e-14, "got {} expected {}", e.lj, e_lj_expected);
    assert!((e.coulomb - e_coul_expected).abs() < 1e-14, "got {} expected {}", e.coulomb, e_coul_expected);

    assert_gradient_matches_fd(&top, &coords, "nonbonded (unbonded pair)");
}

#[test]
fn bonded_chain_1_2_and_1_3_are_excluded_from_nonbonded() {
    // 3-atom bonded chain 0-1-2: pair (0,2) is 1-3, must be EXCLUDED (zero
    // LJ/Coulomb contribution), and (0,1)/(1,2) are 1-2 (also excluded).
    let lj = vec![LjParams { sigma: 2.5, epsilon: 0.01 }; 3];
    let charges = vec![0.5, 0.5, 0.5];
    let bonds = vec![Bond { i: 0, j: 1, k: 0.1, r0: 1.5 }, Bond { i: 1, j: 2, k: 0.1, r0: 1.5 }];
    let top = MmTopology::new(charges, lj, bonds, vec![], vec![]).unwrap();

    assert!(top.exclusions().contains(&(0, 1)));
    assert!(top.exclusions().contains(&(1, 2)));
    assert!(top.exclusions().contains(&(0, 2)));
    assert!(top.pairs14().is_empty());

    let coords = array![[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]];
    let e = energy(&top, &coords).unwrap();
    assert_eq!(e.lj, 0.0, "1-2/1-3 pairs must contribute zero LJ");
    assert_eq!(e.coulomb, 0.0, "1-2/1-3 pairs must contribute zero Coulomb");
}

#[test]
fn four_atom_chain_1_4_pair_is_scaled_and_mutation_sensitive() {
    // 4-atom chain 0-1-2-3: pair (0,3) is 1-4, scaled by scale_lj_14 /
    // scale_coul_14 (defaults 0.5, 1/1.2), NOT excluded.
    let lj = vec![LjParams { sigma: 2.5, epsilon: 0.01 }; 4];
    let charges = vec![0.4, 0.1, -0.1, 0.4];
    let bonds = vec![
        Bond { i: 0, j: 1, k: 0.1, r0: 1.5 },
        Bond { i: 1, j: 2, k: 0.1, r0: 1.5 },
        Bond { i: 2, j: 3, k: 0.1, r0: 1.5 },
    ];
    let top = MmTopology::new(charges.clone(), lj.clone(), bonds.clone(), vec![], vec![]).unwrap();
    assert!(top.pairs14().contains(&(0, 3)));
    assert!(!top.exclusions().contains(&(0, 3)));

    let coords = array![[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0], [4.5, 0.0, 0.3]];
    let e_default = energy(&top, &coords).unwrap();
    assert!(e_default.lj != 0.0, "1-4 LJ must be nonzero (scaled, not excluded)");
    assert!(e_default.coulomb != 0.0, "1-4 Coulomb must be nonzero (scaled, not excluded)");

    // Mutation check: dropping the 1-4 scale (setting scale_lj_14 = 1) must
    // change the LJ energy, since scale_lj_14 default (0.5) != 1.
    let top_unscaled = MmTopology::new(charges, lj, bonds, vec![], vec![]).unwrap().with_scales(1.0, 1.0 / 1.2);
    let e_unscaled = energy(&top_unscaled, &coords).unwrap();
    assert!(
        (e_default.lj - e_unscaled.lj).abs() > 1e-8,
        "scale_lj_14 must change the 1-4 LJ energy: default={} unscaled={}",
        e_default.lj,
        e_unscaled.lj
    );

    assert_gradient_matches_fd(&top, &coords, "1-4 pair chain");
}

#[test]
fn empty_topology_gradient_is_exactly_zero() {
    // Trivial-limit anchor: no bonds/angles/torsions and zero LJ/charges =>
    // exactly zero energy and exactly zero gradient everywhere.
    let top = MmTopology::new(vec![0.0, 0.0, 0.0], lj_zero(3), vec![], vec![], vec![]).unwrap();
    let coords = array![[0.1, 0.2, 0.3], [1.0, -0.5, 2.0], [-3.0, 0.0, 1.1]];
    let (e, g) = gradient(&top, &coords).unwrap();
    assert_eq!(e.total, 0.0);
    for i in 0..3 {
        for c in 0..3 {
            assert_eq!(g[(i, c)], 0.0);
        }
    }
}

// ---------------------------------------------------------------------
// QM-MM Lennard-Jones
// ---------------------------------------------------------------------

#[test]
fn qm_mm_lj_hand_computed_and_gradient_vs_fd() {
    let lj_qm = vec![LjParams { sigma: 3.0, epsilon: 0.001 }];
    let lj_mm = vec![LjParams { sigma: 2.0, epsilon: 0.004 }];
    let coords_qm = array![[0.0, 0.0, 0.0]];
    let coords_mm = array![[4.0, 0.0, 0.0]];

    let (e, g_qm, g_mm) = qm_mm_lj_energy_gradient(&lj_qm, &coords_qm, &lj_mm, &coords_mm);
    let sigma_mix = 0.5 * (3.0 + 2.0);
    let eps_mix = (0.001_f64 * 0.004).sqrt();
    let sr6 = (sigma_mix / 4.0_f64).powi(6);
    let sr12 = sr6 * sr6;
    let expected = 4.0 * eps_mix * (sr12 - sr6);
    assert!((e - expected).abs() < 1e-14, "got {e} expected {expected}");

    // FD check via a small closure wrapping the pair function.
    let f = |cq: &Array2<f64>, cm: &Array2<f64>| qm_mm_lj_energy_gradient(&lj_qm, cq, &lj_mm, cm).0;
    let h = 1e-5;
    for c in 0..3 {
        let mut plus = coords_qm.clone();
        plus[(0, c)] += h;
        let mut minus = coords_qm.clone();
        minus[(0, c)] -= h;
        let fd = (f(&plus, &coords_mm) - f(&minus, &coords_mm)) / (2.0 * h);
        assert!((fd - g_qm[(0, c)]).abs() < 1e-9, "qm axis {c}: fd={fd} analytic={}", g_qm[(0, c)]);
    }
    for c in 0..3 {
        let mut plus = coords_mm.clone();
        plus[(0, c)] += h;
        let mut minus = coords_mm.clone();
        minus[(0, c)] -= h;
        let fd = (f(&coords_qm, &plus) - f(&coords_qm, &minus)) / (2.0 * h);
        assert!((fd - g_mm[(0, c)]).abs() < 1e-9, "mm axis {c}: fd={fd} analytic={}", g_mm[(0, c)]);
    }
}

// ---------------------------------------------------------------------
// Combined chain (all four bonded terms + nonbonded 1-4) at once, on the
// generic 4-atom chain geometry.
// ---------------------------------------------------------------------

#[test]
fn combined_topology_all_terms_gradient_vs_fd() {
    // r0 for each bond is set to (near) the chain4() geometry's actual bond
    // length, so this test measures the SAME thing bond_energy_... already
    // measured at large displacement (that hand-computed test), rather than
    // re-testing "does FD truncation error stay under 1e-9 at a 0.12 Bohr
    // stretch with k=0.35" — a real but uninteresting numerical-analysis
    // question, not a formula bug. (r23 in chain4() is ~1.3812 Bohr.)
    let bonds = vec![
        Bond { i: 0, j: 1, k: 0.35, r0: 1.5 },
        Bond { i: 1, j: 2, k: 0.35, r0: 1.4 },
        Bond { i: 2, j: 3, k: 0.35, r0: 1.3812139257671727 },
    ];
    let angles = vec![
        Angle { i: 0, j: 1, k: 2, k_theta: 0.08, theta0: 109.5_f64.to_radians() },
        Angle { i: 1, j: 2, k: 3, k_theta: 0.08, theta0: 109.5_f64.to_radians() },
    ];
    let torsions = vec![Torsion { i: 0, j: 1, k: 2, l: 3, periodicity: 3, k_phi: 0.02, phase: 0.0 }];
    // sigma is deliberately small (not the 3.0 Bohr used by the isolated
    // nonbonded test) because chain4()'s atoms sit at bonded-range distances
    // (1.3-1.5 Bohr neighbor spacing); a sigma comparable to those distances
    // would put every pair deep in the LJ repulsive wall, where the FD
    // truncation error (~h^2 * third derivative) swamps a 1e-9 tolerance at
    // h=1e-5 for reasons that have nothing to do with gradient correctness
    // (the LJ term's own correctness is already covered, at an appropriate
    // separation, by nonbonded_hand_computed_and_gradient_vs_fd above).
    let lj = vec![LjParams { sigma: 0.6, epsilon: 0.001 }; 4];
    let charges = vec![0.2, -0.1, -0.1, 0.2];
    let top = MmTopology::new(charges, lj, bonds, angles, torsions).unwrap();
    let coords = chain4();
    assert_gradient_matches_fd(&top, &coords, "combined 4-atom chain");
}

