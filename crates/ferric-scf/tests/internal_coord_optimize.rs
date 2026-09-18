//! Redundant internal-coordinate geometry optimization: the exactness anchor,
//! same-stationary-point cross-checks, and iteration-count comparisons.
//!
//! # The exactness anchor (written FIRST, per CLAUDE.md)
//!
//! Every approximation has a trivial limit where it does nothing. Here that
//! limit is
//! [`CoordSystem::Cartesian`](ferric_scf::optimize::CoordSystem::Cartesian) —
//! the pre-existing BFGS path, and still the default. With the
//! internal-coordinate machinery merged but *not selected*, the Cartesian
//! optimizer must produce **bit-identical** results: same final energy bits,
//! same step count, same final coordinate bits. Anything else means the change
//! altered the existing algorithm rather than extending it.
//!
//! The constants at the bottom of this file were captured from `origin/main`
//! (b58556c0) with `ferric_core::internal_coords` absent entirely; see
//! [`cartesian_path_is_bit_identical`] for the re-capture recipe.

use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::operator::Operator;
use ferric_scf::optimize::{optimize_geometry, CoordSystem, OptimizeConfig, OptimizeResult};
use ferric_scf::rhf::RhfConfig;

fn rhf_cfg() -> RhfConfig {
    RhfConfig {
        energy_conv: 1e-10,
        ..Default::default()
    }
}

/// H2 started from a stretched 1.0 Angstrom bond — the same system as the
/// in-crate F2-1 anchor, re-driven here so the integration suite pins it too.
fn h2_stretched() -> Molecule {
    Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 1.0\n", 0, 1).unwrap()
}

/// Water, distorted away from equilibrium in both bond lengths and the angle.
fn water_distorted() -> Molecule {
    Molecule::parse_xyz("3\nH2O\nO 0 0 0\nH 0 0.9 0\nH 0 -0.3 0.85\n", 0, 1).unwrap()
}

/// Hydrogen peroxide, distorted. The HOOH torsion is the whole point of this
/// system: it is the softest coordinate in the molecule and the one Cartesian
/// BFGS handles worst.
fn h2o2_distorted() -> Molecule {
    Molecule::parse_xyz(
        "4\nH2O2\nO  0.000  0.734  -0.055\nO  0.000 -0.734  -0.055\n\
         H  0.839  0.885   0.435\nH -0.839 -0.885   0.435\n",
        0,
        1,
    )
    .unwrap()
}

/// Ethane started near-eclipsed and stretched: six soft C-H stretches, twelve
/// angles, nine torsions about one rotatable bond.
fn ethane_distorted() -> Molecule {
    Molecule::parse_xyz(
        "8\nethane\n\
         C  0.000  0.000  0.800\nC  0.000  0.000 -0.800\n\
         H  1.050  0.000  1.180\nH -0.525  0.909  1.180\nH -0.525 -0.909  1.180\n\
         H  1.000  0.150 -1.180\nH -0.575  0.840 -1.180\nH -0.425 -0.990 -1.180\n",
        0,
        1,
    )
    .unwrap()
}

fn run(mol: &Molecule, cfg: &OptimizeConfig) -> OptimizeResult {
    let ctx = ParallelContext::default();
    optimize_geometry(&ctx, mol, "sto-3g", Operator::coulomb(), &rhf_cfg(), cfg).unwrap()
}

fn cartesian_cfg() -> OptimizeConfig {
    OptimizeConfig {
        trust_radius: 0.1,
        ..Default::default()
    }
}

fn internal_cfg() -> OptimizeConfig {
    OptimizeConfig {
        trust_radius: 0.1,
        coord_system: CoordSystem::RedundantInternal,
        ..Default::default()
    }
}

/// Print the full bit-level fingerprint of a run, so the anchor constants can
/// be re-captured mechanically if the reference platform ever changes.
fn fingerprint(tag: &str, r: &OptimizeResult) {
    eprintln!(
        "[anchor {tag}] steps = {}, converged = {}, energy bits = {:#018x} ({:.17})",
        r.steps,
        r.converged,
        r.energy.to_bits(),
        r.energy
    );
    for (i, a) in r.mol.atoms.iter().enumerate() {
        eprintln!(
            "[anchor {tag}]   atom {i} {:>2} [{:#018x}, {:#018x}, {:#018x}],",
            a.symbol,
            a.x.to_bits(),
            a.y.to_bits(),
            a.zpos.to_bits()
        );
    }
}

// ---------------------------------------------------------------------------
// EXACTNESS ANCHOR: the Cartesian path is untouched
// ---------------------------------------------------------------------------

/// **THE exactness anchor.** With the internal-coordinate machinery present but
/// `CoordSystem::Cartesian` selected (the default), the optimizer must
/// reproduce the pre-change trajectory bit-for-bit on all three systems.
///
/// # Re-capture recipe
///
/// ```text
/// OPENBLAS_NUM_THREADS=1 cargo test -p ferric-scf \
///   --test internal_coord_optimize cartesian_path_is_bit_identical -- --nocapture
/// ```
/// and read the `[anchor ...]` lines.
///
/// # Why `to_bits()` and not a tolerance
///
/// A tolerance cannot distinguish "the change is a no-op" from "the change
/// altered the arithmetic by less than the tolerance". Only bit equality pins
/// the trivial limit. Cross-machine BLAS reduction-order noise is a known
/// caveat (see the F2-1 anchor note in `optimize.rs`): a few-ulp difference on
/// a *different* CPU/OpenBLAS kernel set is GEMM noise, not a regression. On
/// the capture machine it must be exact, which is what makes it useful here.
#[test]
fn cartesian_path_is_bit_identical() {
    let cfg = cartesian_cfg();

    let h2 = run(&h2_stretched(), &cfg);
    fingerprint("h2", &h2);
    assert!(h2.converged);
    assert_bits("h2", &h2, H2_STEPS, H2_ENERGY_BITS, H2_COORD_BITS);

    let w = run(&water_distorted(), &cfg);
    fingerprint("h2o", &w);
    assert!(w.converged);
    assert_bits("h2o", &w, H2O_STEPS, H2O_ENERGY_BITS, H2O_COORD_BITS);

    let p = run(&h2o2_distorted(), &cfg);
    fingerprint("h2o2", &p);
    assert!(p.converged);
    assert_bits("h2o2", &p, H2O2_STEPS, H2O2_ENERGY_BITS, H2O2_COORD_BITS);
}

fn assert_bits(tag: &str, r: &OptimizeResult, steps: usize, energy_bits: u64, coords: &[[u64; 3]]) {
    assert_eq!(r.steps, steps, "{tag}: step count drifted");
    assert_eq!(
        r.energy.to_bits(),
        energy_bits,
        "{tag}: final energy is not bit-identical: got {:#018x} ({:.17}), want {:#018x} ({:.17})",
        r.energy.to_bits(),
        r.energy,
        energy_bits,
        f64::from_bits(energy_bits),
    );
    assert_eq!(r.mol.atoms.len(), coords.len(), "{tag}: atom count changed");
    for (i, (a, w)) in r.mol.atoms.iter().zip(coords).enumerate() {
        for (c, (got, expect)) in [a.x, a.y, a.zpos].iter().zip(w).enumerate() {
            assert_eq!(
                got.to_bits(),
                *expect,
                "{tag}: atom {i} coord {c} not bit-identical: {:#018x} ({:.17}) vs {:#018x} ({:.17})",
                got.to_bits(),
                got,
                expect,
                f64::from_bits(*expect),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// A converged geometry is a converged geometry
// ---------------------------------------------------------------------------

/// A tightly converged Cartesian run, used as the reference minimum.
///
/// Gradient thresholds three orders below the production defaults, so the
/// result is the genuine stationary point rather than "wherever the default
/// thresholds happened to stop".
fn tight_reference(mol: &Molecule) -> OptimizeResult {
    run(
        mol,
        &OptimizeConfig {
            trust_radius: 0.1,
            max_steps: 400,
            g_max_thresh: 1e-6,
            g_rms_thresh: 1e-6,
            e_conv: 1e-12,
            coord_system: CoordSystem::Cartesian,
        },
    )
}

/// Internals and Cartesians must reach the **same stationary point**. This is
/// the physics invariant: the coordinate system is a parameterization of the
/// search, not of the potential-energy surface, so a genuinely different
/// answer is a bug, not a new result.
///
/// # Why this compares against a tight reference and not the two runs directly
///
/// Comparing the two loosely converged runs to each other measures the wrong
/// thing. Both stop when `|g|_max < 4.5e-4`, which on a soft coordinate leaves
/// real energy on the table, and the two paths leave *different* amounts. On
/// H2O2 the measured spread is: Cartesian -148.764984030, internal
/// -148.764996648, tight reference **-148.764996657** — the two disagree by
/// 1.3e-5 Ha, but internals are within 9e-9 Ha of the true minimum and it is
/// Cartesians that stopped short. Asserting `|E_cart - E_int| < 1e-6` would
/// therefore have flagged the *better* result as the bug.
///
/// The invariant that actually holds, and the one worth testing, is: **both
/// runs sit in the same basin as the tight reference, neither below it, and
/// neither further above it than its own convergence slack allows.**
///
/// The geometry comparison uses rotation/translation-invariant quantities (the
/// sorted interatomic-distance list), because the two paths take different
/// routes and land in different — equally valid — orientations.
fn assert_same_stationary_point(
    tag: &str,
    reference: &OptimizeResult,
    cart: &OptimizeResult,
    int: &OptimizeResult,
) {
    assert!(cart.converged, "{tag}: Cartesian run did not converge");
    assert!(int.converged, "{tag}: internal run did not converge");
    assert!(
        reference.converged,
        "{tag}: the tight reference run did not converge — the test has no baseline"
    );

    let dr_cart = cart.energy - reference.energy;
    let dr_int = int.energy - reference.energy;
    eprintln!(
        "[same-point {tag}] E_ref = {:.12} | cart {} steps, E - E_ref = {dr_cart:+.3e} \
         | int {} steps, E - E_ref = {dr_int:+.3e}",
        reference.energy, cart.steps, int.steps
    );

    // Neither may be BELOW the tight reference by more than SCF noise: that
    // would mean the reference is not the minimum and the whole comparison is
    // meaningless.
    for (label, d) in [("cartesian", dr_cart), ("internal", dr_int)] {
        assert!(
            d > -1e-9,
            "{tag}: the {label} run landed {d:.3e} Ha BELOW the tight reference — \
             the reference is not the stationary point"
        );
        // 1e-4 Ha is the energy slack a |g|_max of 4.5e-4 Ha/Bohr permits on a
        // soft mode. Above that, a run is in a different basin, not merely
        // loosely converged.
        assert!(
            d < 1e-4,
            "{tag}: the {label} run is {d:.3e} Ha above the tight reference — \
             that is a different stationary point, not convergence slack"
        );
    }

    // Geometry: both must match the reference's shape.
    //
    // The bar is DERIVED, not guessed. Stopping at |g|_max = 4.5e-4 Ha/Bohr on
    // a torsion whose force constant is ~0.005 Ha/rad^2 (the Schlegel estimate,
    // and within a factor of a few of the truth) leaves up to
    //   Δφ ≈ |g| / k ≈ 4.5e-4 / 0.005 ≈ 0.09 rad
    // of angular slack. On H2O2 the H···H distance that the torsion controls
    // has a lever arm of ~1.4 Bohr per hydrogen, so that is
    //   Δr ≈ 2 × 1.4 × 0.09 ≈ 0.25 Bohr
    // of permitted displacement in the softest interatomic distance. 0.3 Bohr
    // is that estimate rounded up. Anything beyond it is a different basin —
    // a rotamer or a broken bond — not convergence slack.
    //
    // MEASURED, and the asymmetry is the point: on H2O2 the CARTESIAN run sits
    // 5.4e-2 Bohr from the reference while the internal run sits at 6e-4. Both
    // pass, but the run that needs the loose bar is the Cartesian one, which is
    // exactly the weakness internals were added to address.
    const GEOMETRY_SLACK_BOHR: f64 = 0.3;
    let dref = sorted_distances(&reference.mol);
    for (label, r) in [("cartesian", cart), ("internal", int)] {
        let d = sorted_distances(&r.mol);
        assert_eq!(d.len(), dref.len());
        let max_dr = d
            .iter()
            .zip(&dref)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        eprintln!("[same-point {tag}]   {label}: max |Δr| vs reference = {max_dr:.3e} Bohr");
        assert!(
            max_dr < GEOMETRY_SLACK_BOHR,
            "{tag}: the {label} geometry differs from the reference by {max_dr:.3e} Bohr, \
             beyond the {GEOMETRY_SLACK_BOHR} Bohr the gradient threshold can explain \
             — different stationary points"
        );
    }
}

fn sorted_distances(mol: &Molecule) -> Vec<f64> {
    let mut d = Vec::new();
    for i in 0..mol.atoms.len() {
        for j in (i + 1)..mol.atoms.len() {
            let (a, b) = (&mol.atoms[i], &mol.atoms[j]);
            d.push(((a.x - b.x).powi(2) + (a.y - b.y).powi(2) + (a.zpos - b.zpos).powi(2)).sqrt());
        }
    }
    d.sort_by(|a, b| a.partial_cmp(b).unwrap());
    d
}

#[test]
fn water_same_stationary_point() {
    let m = water_distorted();
    assert_same_stationary_point(
        "h2o",
        &tight_reference(&m),
        &run(&m, &cartesian_cfg()),
        &run(&m, &internal_cfg()),
    );
}

#[test]
fn h2o2_same_stationary_point() {
    let m = h2o2_distorted();
    assert_same_stationary_point(
        "h2o2",
        &tight_reference(&m),
        &run(&m, &cartesian_cfg()),
        &run(&m, &internal_cfg()),
    );
}

#[test]
fn h2_same_stationary_point() {
    // A diatomic has exactly one internal coordinate. This is the degenerate
    // case for the B-matrix (3N = 6 Cartesians, 1 internal, 5 zero modes) and
    // the one most likely to break a naive generalized inverse.
    let m = h2_stretched();
    assert_same_stationary_point(
        "h2",
        &tight_reference(&m),
        &run(&m, &cartesian_cfg()),
        &run(&m, &internal_cfg()),
    );
}

#[test]
fn ethane_same_stationary_point() {
    let m = ethane_distorted();
    assert_same_stationary_point(
        "ethane",
        &tight_reference(&m),
        &run(&m, &cartesian_cfg()),
        &run(&m, &internal_cfg()),
    );
}

/// At the SAME convergence threshold, internals must land CLOSER to the true
/// minimum than Cartesians do on a torsion-dominated system.
///
/// This is the load-bearing claim, asserted rather than merely printed. The
/// mechanism: `|g|_max < 4.5e-4` is a Cartesian test, and in Cartesians a
/// residual torsional gradient is spread thinly across many components, so the
/// test passes while the torsion is still far from relaxed. In internals the
/// torsion is one coordinate with its own (small) force constant, so the same
/// threshold pins it much harder.
///
/// # Artifact hypothesis
///
/// If internals were merely *lucky* on this system — a different path that
/// happened to stop nearer — the advantage would be a small factor and would
/// not track the softness of the coordinate. If the mechanism above is real,
/// the advantage should be large (orders of magnitude) and should appear on
/// the torsional system (H2O2, ethane) and not on the rigid one (water). That
/// is what is measured, and `water_same_stationary_point`'s printed numbers
/// are the control: there the two are within a factor of 2 of each other.
#[test]
fn internals_land_closer_to_the_true_minimum_on_torsional_systems() {
    for (name, m) in [("h2o2", h2o2_distorted()), ("ethane", ethane_distorted())] {
        let reference = tight_reference(&m);
        let cart = run(&m, &cartesian_cfg());
        let int = run(&m, &internal_cfg());
        assert!(reference.converged && cart.converged && int.converged);

        let e_cart = cart.energy - reference.energy;
        let e_int = int.energy - reference.energy;
        eprintln!(
            "[accuracy {name}] cartesian is {e_cart:.3e} Ha above the minimum, \
             internal is {e_int:.3e} Ha — internals {:.0}x closer",
            e_cart / e_int.max(f64::MIN_POSITIVE)
        );
        assert!(
            e_int < e_cart,
            "{name}: internals ({e_int:.3e} Ha above the minimum) should beat \
             Cartesians ({e_cart:.3e} Ha) at the same threshold"
        );
        // A factor of 10 is well inside the measured 37x (ethane) and 1350x
        // (H2O2), while leaving room for the SCF noise floor.
        assert!(
            e_int * 10.0 < e_cart,
            "{name}: internals are only {:.1}x closer to the minimum than \
             Cartesians — the torsion advantage has largely disappeared",
            e_cart / e_int.max(f64::MIN_POSITIVE)
        );
    }
}

/// A small floppy CATION — the system class the external assessment flagged
/// ("the optimizer is weak on floppy cations") and the reason this work exists.
///
/// Protonated methanol, CH3OH2+ (7 atoms, +1, closed shell): the C-O torsion is
/// soft, the O-H bonds are polarized by the charge, and the whole thing is the
/// small end of the terpinyl-cation failure mode. Run with UHF-free RHF since
/// it is a closed-shell cation.
fn methanol_cation_distorted() -> Molecule {
    Molecule::parse_xyz(
        "7\nCH3OH2+\n\
         C  0.000  0.000  0.000\nO  1.480  0.000  0.000\n\
         H -0.380  1.020  0.050\nH -0.400 -0.500  0.900\nH -0.390 -0.520 -0.880\n\
         H  1.820  0.700  0.600\nH  1.830  0.150 -0.900\n",
        1,
        1,
    )
    .unwrap()
}

#[test]
fn floppy_cation_same_stationary_point() {
    let m = methanol_cation_distorted();
    assert_same_stationary_point(
        "ch3oh2+",
        &tight_reference(&m),
        &run(&m, &cartesian_cfg()),
        &run(&m, &internal_cfg()),
    );
}

// ---------------------------------------------------------------------------
// Iteration counts (the actual claim being made)
// ---------------------------------------------------------------------------

/// Reports Cartesian-vs-internal **iteration counts** on every test system.
///
/// Iterations, not wall time: an iteration is one energy+gradient evaluation,
/// which dominates the cost by orders of magnitude, and unlike wall time it is
/// reproducible on a loaded box. The assertion is deliberately weak (internals
/// must not be catastrophically worse); the numbers themselves are the output.
#[test]
fn iteration_count_ledger() {
    let systems: &[(&str, Molecule)] = &[
        ("h2", h2_stretched()),
        ("h2o", water_distorted()),
        ("h2o2", h2o2_distorted()),
        ("ethane", ethane_distorted()),
        ("ch3oh2+", methanol_cation_distorted()),
    ];
    eprintln!("system   | cart steps | int steps | E_cart          | E_int");
    eprintln!("---------+------------+-----------+-----------------+----------------");
    for (name, m) in systems {
        let c = run(m, &cartesian_cfg());
        let i = run(m, &internal_cfg());
        eprintln!(
            "{name:8} | {:10} | {:9} | {:15.9} | {:15.9}",
            c.steps, i.steps, c.energy, i.energy
        );
        assert!(c.converged, "{name}: Cartesian did not converge");
        assert!(i.converged, "{name}: internal did not converge");
        assert!(
            i.steps <= c.steps * 2 + 4,
            "{name}: internals took {} steps vs Cartesian {} — not a wash, a regression",
            i.steps,
            c.steps
        );
    }
}

// ---------------------------------------------------------------------------
// Larger measurement — run deliberately, on a quiet box
// ---------------------------------------------------------------------------

/// `#[ignore]`d: the iteration-count comparison on genuinely floppy systems,
/// where the internal-coordinate advantage should be largest.
///
/// Not run by default because it is many SCF+gradient evaluations on 14-atom
/// systems and would be noise on a loaded box. Run it deliberately:
///
/// ```text
/// OPENBLAS_NUM_THREADS=1 cargo test -p ferric-scf --release \
///   --test internal_coord_optimize floppy_system_iteration_ledger \
///   -- --ignored --nocapture
/// ```
///
/// n-butane has three rotatable torsions (one C-C-C-C plus two methyl
/// rotations) and is the smallest system where the torsional subspace really
/// dominates. The all-anti input is perturbed into a twisted, stretched start
/// so both optimizers have real work to do.
///
/// What to look for: `int steps` should be a clear fraction of `cart steps`,
/// and `E_int` should be at or below `E_cart`. If internals are SLOWER here,
/// the empirical Hessian or the step clip is mistuned for larger systems —
/// that is the measurement this test exists to expose, so a bad result is a
/// finding, not a failure to hide.
#[test]
#[ignore = "many SCF+gradient evaluations on 14-atom systems; run on a quiet box"]
fn floppy_system_iteration_ledger() {
    let butane_twisted = Molecule::load_xyz("../../testdata/molecules/alkane_4.xyz")
        .map(|mut m| {
            // Twist the backbone out of all-anti and stretch it, so the run
            // starts far from the minimum in exactly the soft coordinates.
            for (i, a) in m.atoms.iter_mut().enumerate() {
                if i >= 2 {
                    let t: f64 = 0.25;
                    let (x, y) = (a.x, a.y);
                    a.x = x * t.cos() - y * t.sin();
                    a.y = x * t.sin() + y * t.cos();
                    a.zpos *= 1.04;
                }
            }
            m
        })
        .expect("testdata/molecules/alkane_4.xyz must exist");

    let systems: &[(&str, Molecule)] = &[
        ("butane", butane_twisted),
        ("ethane", ethane_distorted()),
        ("ch3oh2+", methanol_cation_distorted()),
    ];

    eprintln!(
        "system   | cart steps | int steps | E_cart          | E_int           | dE (int-cart)"
    );
    eprintln!(
        "---------+------------+-----------+-----------------+-----------------+--------------"
    );
    for (name, m) in systems {
        let c = run(m, &cartesian_cfg());
        let i = run(m, &internal_cfg());
        eprintln!(
            "{name:8} | {:10} | {:9} | {:15.9} | {:15.9} | {:+.3e}",
            c.steps,
            i.steps,
            c.energy,
            i.energy,
            i.energy - c.energy
        );
        eprintln!(
            "         |  converged: cart {} / int {}",
            c.converged, i.converged
        );
    }
}

// ---------------------------------------------------------------------------
// Anchor constants
// ---------------------------------------------------------------------------

// Captured 2026-09-18 on the development box from origin/main b58556c0
// ("fix(gw): a bit-identity anchor cannot be pinned to one machine's bits"),
// with `ferric_core::internal_coords` absent entirely and `OptimizeConfig`
// carrying no `coord_system` field.
const H2_STEPS: usize = 7;
const H2_ENERGY_BITS: u64 = 0xbff1_e14d_d9dd_63b7; // -1.11750588515699945
const H2_COORD_BITS: &[[u64; 3]] = &[
    [
        0x0000_0000_0000_0000,
        0x0000_0000_0000_0000,
        0x3fd1_66df_ad4e_b9aa,
    ],
    [
        0x0000_0000_0000_0000,
        0x0000_0000_0000_0000,
        0x3ff9_e299_8aa2_c763,
    ],
];

const H2O_STEPS: usize = 7;
const H2O_ENERGY_BITS: u64 = 0xc052_bdd1_5356_3e20; // -74.96590121671806628
const H2O_COORD_BITS: &[[u64; 3]] = &[
    [
        0x3ca4_10bf_071d_9e39,
        0xbfb5_bb98_c4dc_da47,
        0xbfbe_4c65_1a33_aad9,
    ],
    [
        0x3cc0_ac04_650c_7ed1,
        0x3ffc_750a_db71_510b,
        0x3fa1_b626_026c_0320,
    ],
    [
        0xbcc5_b034_26d3_ea10,
        0xbfe1_ea40_dd85_aa24,
        0x3ffb_0a5a_5f6e_2484,
    ],
];

const H2O2_STEPS: usize = 6;
const H2O2_ENERGY_BITS: u64 = 0xc062_987a_bfc9_b867; // -148.76498402975559543
const H2O2_COORD_BITS: &[[u64; 3]] = &[
    [
        0x3f42_e2f4_8ea7_53a9,
        0x3ff5_1819_ed37_931d,
        0xbfbc_324c_617f_688e,
    ],
    [
        0xbf42_e2f4_8eb1_6eee,
        0xbff5_1819_ed37_8820,
        0xbfbc_324c_617f_77f2,
    ],
    [
        0x3ff9_a0fc_e208_da4e,
        0x3ffb_07e3_88e3_c1f8,
        0x3fea_80ed_db99_54bb,
    ],
    [
        0xbff9_a0fc_e208_d924,
        0xbffb_07e3_88e3_bf62,
        0x3fea_80ed_db99_540d,
    ],
];
