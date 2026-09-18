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
//! optimizer must produce the pre-change trajectory: same step count, same
//! final energy, same final coordinates. Anything else means the change
//! altered the existing algorithm rather than extending it.
//!
//! # Why the bit-level leg is LIVE-vs-LIVE and the frozen constants are a band
//!
//! A hardcoded `u64` can only ever be right on the box that produced it. The
//! constants at the bottom of this file were captured on the development box;
//! CI's H2 final energy is `0xbff1e14dd9dd63b2` against this box's
//! `0xbff1e14dd9dd63b7` — **5 ulp, 1.1e-15 Ha**, bit-stable *within* each
//! machine and different *between* them. That is the same per-machine BLAS
//! dispatch that moved this repo's GW QP energies and cDFT iteration counts;
//! see `ferric-gw/tests/mwe_gw_pool_is_inert_without_a_pool.rs`, which carries
//! the worked example and the measured dev-box-vs-CI bit patterns.
//!
//! Re-recording would only move the failure to the next machine, so the bits
//! are asserted where they are meaningful — between two runs in the SAME
//! process on the SAME CPU:
//!
//! * [`cartesian_path_is_reproducible_and_matches_the_recorded_path`] runs the
//!   Cartesian optimizer **twice** and requires `to_bits()` equality. This is
//!   the determinism leg: it fails if the optimizer picked up any run-to-run
//!   nondeterminism (thread-order reduction, uninitialised scratch, iteration
//!   over a hashed container) that a frozen literal captured once would hide.
//! * [`selecting_internals_does_not_perturb_the_cartesian_path`] is the leg
//!   that actually tests *this change*: Cartesian-selected and
//!   internals-selected runs in one process, asserting the Cartesian leg is
//!   bit-identical to a Cartesian run made while the internal machinery has
//!   also been exercised. This is STRICTER than the frozen literal was, because
//!   the only difference between the two legs is the thing under test.
//! * The frozen constants are kept as a LOOSE SANITY BAND
//!   ([`ENERGY_SANITY_BAND`] / [`COORD_SANITY_BAND`]), orders above the
//!   measured ~1e-15 cross-machine spread. They catch a gross regression
//!   (a different stationary point, a different step count) without pinning
//!   one box's last hex digit, and they stop the live-vs-live legs from passing
//!   by both being equally wrong.

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

/// **THE exactness anchor, determinism leg.** The Cartesian path must be
/// reproducible bit-for-bit within one process, and must land inside the
/// recorded sanity band.
///
/// # What each leg buys
///
/// The `to_bits()` comparison is LIVE-vs-LIVE — two runs of the *same*
/// configuration, back to back on the same CPU with the same BLAS dispatch.
/// The only thing it can catch is genuine run-to-run nondeterminism, and that
/// is precisely what it is for: a thread-order-dependent reduction or an
/// iteration over a hashed container would make the optimizer's trajectory
/// irreproducible, and a literal captured in a single run cannot see it.
///
/// The band comparison catches a real change in what the optimizer computes.
/// It is deliberately loose relative to the ~1e-15 Ha cross-machine spread
/// (see the module doc), and the **step count is still asserted exactly** —
/// that is an integer, unaffected by BLAS dispatch, and a change in it is a
/// change in the algorithm.
///
/// # Re-recording the band constants
///
/// ```text
/// OPENBLAS_NUM_THREADS=1 cargo test -p ferric-scf \
///   --test internal_coord_optimize \
///   cartesian_path_is_reproducible_and_matches_the_recorded_path -- --nocapture
/// ```
/// and read the `[anchor ...]` lines. Re-record only when a change to the
/// numerics is deliberate — the band is wide enough that BLAS noise never
/// requires it.
#[test]
fn cartesian_path_is_reproducible_and_matches_the_recorded_path() {
    let cfg = cartesian_cfg();

    for (tag, mol, steps, energy_bits, coords) in [
        (
            "h2",
            h2_stretched(),
            H2_STEPS,
            H2_ENERGY_BITS,
            H2_COORD_BITS,
        ),
        (
            "h2o",
            water_distorted(),
            H2O_STEPS,
            H2O_ENERGY_BITS,
            H2O_COORD_BITS,
        ),
        (
            "h2o2",
            h2o2_distorted(),
            H2O2_STEPS,
            H2O2_ENERGY_BITS,
            H2O2_COORD_BITS,
        ),
    ] {
        let first = run(&mol, &cfg);
        fingerprint(tag, &first);
        assert!(first.converged, "{tag}: the Cartesian run must converge");

        // LIVE vs LIVE: same config, same process, same CPU. Bits are
        // meaningful here because nothing differs but the run itself.
        let again = run(&mol, &cfg);
        assert_same_path_bits(tag, &first, &again);

        // Recorded values, as a band.
        assert_within_band(tag, &first, steps, energy_bits, coords);
    }
}

/// Strict `to_bits()` equality between two runs made in the SAME process.
///
/// This is the only place in this file where bit equality is asserted, and it
/// is sound because both sides share a CPU, a BLAS dispatch and a thread pool.
fn assert_same_path_bits(tag: &str, a: &OptimizeResult, b: &OptimizeResult) {
    assert_eq!(
        a.steps, b.steps,
        "{tag}: the Cartesian optimizer is not reproducible — two identical runs \
         in one process took {} and {} steps",
        a.steps, b.steps
    );
    assert_eq!(
        a.energy.to_bits(),
        b.energy.to_bits(),
        "{tag}: the Cartesian optimizer is not reproducible — two identical runs \
         in one process gave {:#018x} ({:.17}) and {:#018x} ({:.17}). \
         Same CPU, same BLAS, so this is real nondeterminism, not dispatch.",
        a.energy.to_bits(),
        a.energy,
        b.energy.to_bits(),
        b.energy,
    );
    assert_eq!(a.mol.atoms.len(), b.mol.atoms.len(), "{tag}: atom count");
    for (i, (x, y)) in a.mol.atoms.iter().zip(&b.mol.atoms).enumerate() {
        for (c, (p, q)) in [x.x, x.y, x.zpos]
            .iter()
            .zip([y.x, y.y, y.zpos].iter())
            .enumerate()
        {
            assert_eq!(
                p.to_bits(),
                q.to_bits(),
                "{tag}: atom {i} coord {c} is not reproducible across two runs \
                 in one process: {:#018x} vs {:#018x}",
                p.to_bits(),
                q.to_bits(),
            );
        }
    }
}

/// Absolute band on the final energy. The measured cross-machine spread on the
/// H2 anchor is 5 ulp ≈ 1.1e-15 Ha; 1e-9 Ha is six orders above that and still
/// far below the 1e-4 Ha that separates distinct stationary points.
const ENERGY_SANITY_BAND: f64 = 1e-9;

/// Absolute band on each final coordinate, in Bohr. Same reasoning: many orders
/// above BLAS noise, far below the 0.3 Bohr that distinguishes basins.
const COORD_SANITY_BAND: f64 = 1e-6;

/// Compare against the RECORDED path as a band, with the step count exact.
fn assert_within_band(
    tag: &str,
    r: &OptimizeResult,
    steps: usize,
    energy_bits: u64,
    coords: &[[u64; 3]],
) {
    // An integer is not subject to BLAS dispatch: a drift here is algorithmic.
    assert_eq!(r.steps, steps, "{tag}: step count drifted");

    let want_e = f64::from_bits(energy_bits);
    let de = (r.energy - want_e).abs();
    assert!(
        de < ENERGY_SANITY_BAND,
        "{tag}: final energy left the {ENERGY_SANITY_BAND:e} Ha band around the \
         recorded value: got {:#018x} ({:.17}), recorded {:#018x} ({:.17}), \
         delta {de:.3e}. That is far beyond the ~1e-15 cross-machine spread, so it \
         is a real change in the Cartesian path, not BLAS dispatch.",
        r.energy.to_bits(),
        r.energy,
        energy_bits,
        want_e,
    );

    assert_eq!(r.mol.atoms.len(), coords.len(), "{tag}: atom count changed");
    for (i, (a, w)) in r.mol.atoms.iter().zip(coords).enumerate() {
        for (c, (got, expect)) in [a.x, a.y, a.zpos].iter().zip(w).enumerate() {
            let want = f64::from_bits(*expect);
            let dr = (got - want).abs();
            assert!(
                dr < COORD_SANITY_BAND,
                "{tag}: atom {i} coord {c} left the {COORD_SANITY_BAND:e} Bohr band: \
                 got {:.17} ({:#018x}), recorded {want:.17} ({:#018x}), delta {dr:.3e}",
                got,
                got.to_bits(),
                expect,
            );
        }
    }
}

/// **THE exactness anchor, no-op leg.** Selecting internals must not perturb
/// the Cartesian path.
///
/// This is the assertion that actually tests *this change*, and it is the one
/// the frozen constants were reaching for. Both legs run in the SAME process:
/// a Cartesian run made before the internal-coordinate machinery is touched,
/// and a Cartesian run made after an internals run has driven it. If selecting
/// internals left any state behind — a cached B-matrix, a mutated config, a
/// perturbed global — the second Cartesian run would differ, and `to_bits()`
/// sees it. Sharing a CPU makes bit equality the right instrument here.
#[test]
fn selecting_internals_does_not_perturb_the_cartesian_path() {
    let cart = cartesian_cfg();
    let int = internal_cfg();

    for (tag, mol) in [
        ("h2", h2_stretched()),
        ("h2o", water_distorted()),
        ("h2o2", h2o2_distorted()),
    ] {
        let before = run(&mol, &cart);
        assert!(before.converged, "{tag}: Cartesian must converge");

        // Exercise the internal-coordinate path in between.
        let via_internals = run(&mol, &int);
        assert!(via_internals.converged, "{tag}: internal must converge");

        let after = run(&mol, &cart);
        eprintln!(
            "[no-op {tag}] cartesian {} steps E = {:.17} | internals {} steps | \
             cartesian again {} steps E = {:.17}",
            before.steps, before.energy, via_internals.steps, after.steps, after.energy
        );
        assert_same_path_bits(tag, &before, &after);
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
// Anchor constants — a SANITY BAND, not a bit pin
// ---------------------------------------------------------------------------

// Captured 2026-09-18 on the development box from origin/main b58556c0
// ("fix(gw): a bit-identity anchor cannot be pinned to one machine's bits"),
// with `ferric_core::internal_coords` absent entirely and `OptimizeConfig`
// carrying no `coord_system` field.
//
// These are stored as `u64` so the recorded value is exact and re-recording is
// mechanical, but they are COMPARED as floats inside ENERGY_SANITY_BAND /
// COORD_SANITY_BAND. The measured cross-machine spread on H2 is 5 ulp: this box
// gives 0xbff1e14dd9dd63b7, CI gives 0xbff1e14dd9dd63b2. Bit-pinning them made
// CI red for a difference of 1.1e-15 Ha. The step counts below ARE asserted
// exactly — they are integers and do not move with BLAS dispatch.
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
