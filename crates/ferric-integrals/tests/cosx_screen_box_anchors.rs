//! Anchors for the axis-aligned-box region bound
//! (`PairBounds::coarse_estimate_box`), written BEFORE it compiled per the
//! repo rule (EXACTNESS ANCHOR FIRST).
//!
//! # What is being added, and why
//!
//! `coarse_estimate_sphere(s1, s2, centre, radius)` bounds `estimate` over a
//! BALL. COSX's md3c1e screen calls it over the bounding sphere of a
//! 256-point grid sub-batch. The Becke grid is emitted atom-major,
//! radial-major, angular-minor, so 256 consecutive points are ~2.3 whole
//! Lebedev spheres of ONE atom: the bounding sphere of that set has the
//! radius of the outer Lebedev shell, which for outer radial points is tens
//! of Bohr. `R_c = |M - centre| - radius - |AB|/2` then clamps to 0 and the
//! bound degenerates to its distance-free `total`. Measured in the Python
//! reference study (`scripts/queue/out/snlink_python_results.md` §4.1): the
//! sphere query returns its `R = 0` value for 68-80% of (shell-pair, batch)
//! decisions on water/methane/ethane.
//!
//! An axis-aligned box over the SAME points is never looser, because the
//! distance from `M` to the box is the EXACT `min_{r in box} |M - r|`, while
//! the sphere's `|M - centre| - radius` is a lower bound on the same minimum
//! over a strictly larger region (the ball circumscribing the box's points).
//! So the box bound is the natural tightening and it costs the same arithmetic
//! (three clamped differences, one sqrt).
//!
//! # Anchors
//!
//! * `box_bound_never_underestimates_inside_the_box` — the required property.
//!   For every shell pair and many boxes, `coarse_estimate_box >= estimate(r)`
//!   for every probe `r` INSIDE the box. This is the only anchor whose failure
//!   is a correctness bug rather than a lost opportunity.
//! * `box_bound_is_never_looser_than_the_enclosing_sphere` — the reason the
//!   change exists. Over the same point set, box <= sphere. Reported as a
//!   distribution, asserted as an inequality.
//! * `box_bound_degenerates_less_often_than_the_sphere` — NON-INERTNESS. The
//!   fraction of (pair, region) queries that return the distance-free `total`
//!   must be strictly LOWER for the box on a realistic Lebedev-arc region. A
//!   bound that is valid but degenerates just as often buys nothing, and this
//!   anchor is what fails if the box tightening is inert.
//! * `box_over_a_single_point_matches_the_point_bound` — trivial limit: a
//!   degenerate box (lo == hi == r) reproduces `coarse_estimate(r)` bitwise.
//!
//! Mutation proofs are recorded in the commit message; each anchor was run
//! against a deliberately broken bound (see `MUTATIONS` below).

use ferric_core::basis::bundled;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::cosx_a::PairBounds;
use ferric_quadrature::lebedev::lebedev;

fn testdata(rel: &str) -> String {
    format!("{}/../../{}", env!("CARGO_MANIFEST_DIR"), rel)
}

struct Lcg(u64);
impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

/// Mutations these anchors were checked against (applied by hand to
/// `coarse_estimate_box`, recorded in the commit message):
///
/// * `box_is_the_sphere` — return `coarse_estimate_sphere` over the box's
///   circumscribing sphere. Valid, so the underestimation anchor stays green;
///   `box_bound_degenerates_less_often_than_the_sphere` goes RED. This is the
///   inertness mutation and it is the one that matters.
/// * `drop_the_clamp` — use `M_d - lo_d` instead of
///   `max(0, lo_d - M_d, M_d - hi_d)`. Makes the distance too large when `M`
///   is inside the box, so the bound underestimates:
///   `box_bound_never_underestimates_inside_the_box` goes RED.
/// * `forget_half` — drop the `|AB|/2` term. Underestimates for pairs on
///   distinct centres: same anchor goes RED.
#[allow(dead_code)]
const MUTATIONS: () = ();

/// Axis-aligned bounding box of a point set as `(lo, hi)`.
fn aabb(pts: &[[f64; 3]]) -> ([f64; 3], [f64; 3]) {
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for p in pts {
        for d in 0..3 {
            lo[d] = lo[d].min(p[d]);
            hi[d] = hi[d].max(p[d]);
        }
    }
    (lo, hi)
}

/// Centroid + enclosing radius (the same construction `cosx_k::bounding_sphere`
/// uses), so the comparison is against what the screen does today.
fn bounding_sphere(pts: &[[f64; 3]]) -> ([f64; 3], f64) {
    let n = pts.len().max(1) as f64;
    let mut c = [0.0_f64; 3];
    for p in pts {
        for d in 0..3 {
            c[d] += p[d];
        }
    }
    for v in &mut c {
        *v /= n;
    }
    let r2 = pts
        .iter()
        .map(|p| (0..3).map(|d| (p[d] - c[d]) * (p[d] - c[d])).sum::<f64>())
        .fold(0.0_f64, f64::max);
    (c, r2.sqrt())
}

struct Case {
    label: &'static str,
    prep: PreparedBasis,
    mol: Molecule,
}

fn cases() -> Vec<Case> {
    let water = Molecule::parse_xyz(
        "3\nwater\nO 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692\n",
        0,
        1,
    )
    .expect("water");
    let butane = Molecule::load_xyz(&testdata("testdata/molecules/alkane_4.xyz")).expect("alkane_4");
    let bw = bundled("cc-pvdz").expect("cc-pvdz");
    let bb = bundled("def2-svp").expect("def2-svp");
    vec![
        Case { label: "water/cc-pVDZ", prep: PreparedBasis::new(&water, &bw).expect("prep"), mol: water },
        Case { label: "butane/def2-SVP", prep: PreparedBasis::new(&butane, &bb).expect("prep"), mol: butane },
    ]
}

/// A set of realistic COSX regions: contiguous arcs of a Lebedev sphere
/// centred on each atom, at several radii and several arc lengths. These are
/// exactly the point sets the sub-batched screen groups over.
fn lebedev_regions(mol: &Molecule, arc: usize) -> Vec<Vec<[f64; 3]>> {
    let (pts, _w) = lebedev(110).expect("lebedev 110");
    let mut out = Vec::new();
    for atom in &mol.atoms {
        for &r in &[0.3_f64, 1.0, 3.0, 8.0, 20.0] {
            for chunk in pts.chunks(arc) {
                out.push(
                    chunk
                        .iter()
                        .map(|p| [atom.x + r * p[0], atom.y + r * p[1], atom.zpos + r * p[2]])
                        .collect(),
                );
            }
        }
    }
    out
}

/// Probe points strictly inside a box: the corners, the centre, and random
/// interior points. The corners matter — the bound must hold at the extremes,
/// which is where a wrong clamp shows up.
fn probes_in_box(lo: &[f64; 3], hi: &[f64; 3], rng: &mut Lcg) -> Vec<[f64; 3]> {
    let mut v = Vec::new();
    for c in 0..8 {
        v.push([
            if c & 1 == 0 { lo[0] } else { hi[0] },
            if c & 2 == 0 { lo[1] } else { hi[1] },
            if c & 4 == 0 { lo[2] } else { hi[2] },
        ]);
    }
    v.push([0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), 0.5 * (lo[2] + hi[2])]);
    for _ in 0..24 {
        v.push([
            lo[0] + rng.next_f64() * (hi[0] - lo[0]),
            lo[1] + rng.next_f64() * (hi[1] - lo[1]),
            lo[2] + rng.next_f64() * (hi[2] - lo[2]),
        ]);
    }
    v
}

/// THE required property: over its box, the region bound never falls below
/// the pointwise bound it is standing in for.
#[test]
fn box_bound_never_underestimates_inside_the_box() {
    for c in cases() {
        let pb = PairBounds::build(&c.prep).expect("bounds");
        let nsh = pb.nshells();
        let mut rng = Lcg(0x5eed_1234);
        let mut checked = 0usize;
        let mut worst = 0.0_f64;
        for region in lebedev_regions(&c.mol, 16) {
            let (lo, hi) = aabb(&region);
            let probes = probes_in_box(&lo, &hi, &mut rng);
            for s1 in 0..nsh {
                for s2 in 0..=s1 {
                    let b = pb.coarse_estimate_box(s1, s2, &lo, &hi);
                    for r in &probes {
                        let e = pb.estimate(s1, s2, r);
                        assert!(
                            b >= e,
                            "{}: box bound {b:.6e} < pointwise estimate {e:.6e} for pair ({s1},{s2}) at {r:?} \
                             in box {lo:?}..{hi:?}",
                            c.label
                        );
                        worst = worst.max(if b > 0.0 { e / b } else { 0.0 });
                        checked += 1;
                    }
                }
            }
        }
        println!("{}: {checked} (pair, probe-in-box) checks, tightest true/bound = {worst:.4}", c.label);
        assert!(checked > 100_000, "{}: only {checked} checks — the anchor is too thin", c.label);
    }
}

/// The reason the box exists: over the SAME points it is never looser than
/// the enclosing sphere the screen uses today.
#[test]
fn box_bound_is_never_looser_than_the_enclosing_sphere() {
    for c in cases() {
        let pb = PairBounds::build(&c.prep).expect("bounds");
        let nsh = pb.nshells();
        let mut tighter = 0usize;
        let mut total = 0usize;
        let mut best_ratio = 1.0_f64;
        for region in lebedev_regions(&c.mol, 16) {
            let (lo, hi) = aabb(&region);
            let (cen, rad) = bounding_sphere(&region);
            for s1 in 0..nsh {
                for s2 in 0..=s1 {
                    let bx = pb.coarse_estimate_box(s1, s2, &lo, &hi);
                    let sp = pb.coarse_estimate_sphere(s1, s2, &cen, rad);
                    assert!(
                        bx <= sp * (1.0 + 1e-12),
                        "{}: box bound {bx:.6e} EXCEEDS sphere bound {sp:.6e} for pair ({s1},{s2})",
                        c.label
                    );
                    if bx < sp * (1.0 - 1e-12) {
                        tighter += 1;
                        best_ratio = best_ratio.min(bx / sp);
                    }
                    total += 1;
                }
            }
        }
        println!(
            "{}: box strictly tighter on {tighter}/{total} ({:.1}%) queries; best box/sphere = {best_ratio:.4e}",
            c.label,
            100.0 * tighter as f64 / total as f64
        );
        assert!(tighter > 0, "{}: the box is never tighter than the sphere — the change is inert", c.label);
    }
}

/// NON-INERTNESS. The whole point is that the box query degenerates to the
/// distance-free `total` LESS often than the sphere query does. If this does
/// not hold, tightening the region buys nothing and the lane should stop.
#[test]
fn box_bound_degenerates_less_often_than_the_sphere() {
    for c in cases() {
        let pb = PairBounds::build(&c.prep).expect("bounds");
        let nsh = pb.nshells();
        let mut deg_box = 0usize;
        let mut deg_sph = 0usize;
        let mut total = 0usize;
        for region in lebedev_regions(&c.mol, 16) {
            let (lo, hi) = aabb(&region);
            let (cen, rad) = bounding_sphere(&region);
            for s1 in 0..nsh {
                for s2 in 0..=s1 {
                    let m = pb.max_estimate(s1, s2);
                    if pb.coarse_estimate_box(s1, s2, &lo, &hi) >= m {
                        deg_box += 1;
                    }
                    if pb.coarse_estimate_sphere(s1, s2, &cen, rad) >= m {
                        deg_sph += 1;
                    }
                    total += 1;
                }
            }
        }
        let (fb, fs) = (deg_box as f64 / total as f64, deg_sph as f64 / total as f64);
        println!("{}: degenerate fraction  sphere {fs:.4}  box {fb:.4}  ({total} queries)", c.label);
        assert!(fb < fs, "{}: box degenerates as often as the sphere ({fb:.4} vs {fs:.4}) — inert", c.label);
    }
}

/// Trivial limit: a box collapsed onto one point IS that point, so the region
/// bound must reproduce the point bound bitwise.
#[test]
fn box_over_a_single_point_matches_the_point_bound() {
    for c in cases() {
        let pb = PairBounds::build(&c.prep).expect("bounds");
        let nsh = pb.nshells();
        let mut rng = Lcg(0xabcd_0001);
        for _ in 0..64 {
            let r = [
                8.0 * (rng.next_f64() - 0.5),
                8.0 * (rng.next_f64() - 0.5),
                8.0 * (rng.next_f64() - 0.5),
            ];
            for s1 in 0..nsh {
                for s2 in 0..=s1 {
                    let bx = pb.coarse_estimate_box(s1, s2, &r, &r);
                    let pt = pb.coarse_estimate(s1, s2, &r);
                    assert_eq!(
                        bx.to_bits(),
                        pt.to_bits(),
                        "{}: degenerate box {bx:.17e} != point bound {pt:.17e} for pair ({s1},{s2})",
                        c.label
                    );
                }
            }
        }
    }
}
