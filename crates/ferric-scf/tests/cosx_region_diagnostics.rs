//! DIAGNOSTIC (not an anchor): what actually drives the degenerate-bound
//! fraction in the COSX pair screen, as a function of how the grid points are
//! grouped and which enclosing volume is used.
//!
//! The Python reference study measured that 68-80% of (shell-pair, batch)
//! decisions return the distance-free `R = 0` value
//! (`scripts/queue/out/snlink_python_results.md` §4.1). Before designing a
//! sub-batched screen around that, this decomposes WHY, because the fix
//! depends on the answer:
//!
//! ```text
//!     R_c = d(M, region) - |AB|/2        (sphere: |M - centre| - radius)
//! ```
//!
//! degenerates when `R_c <= 0`. That happens for two quite different reasons:
//!   (i) the REGION is large — `d(M, region)` is small because the region
//!       reaches close to the pair midpoint. Shrinking the group fixes this.
//!  (ii) the PAIR is diffuse — `|AB|/2` is large enough to eat the distance.
//!       Grouping cannot fix this at all; it is a property of the shell pair.
//!
//! It also compares the three candidate enclosing volumes over the SAME points
//! (centroid ball, AABB, and their `min`, all three valid) so the choice is
//! made on measurement rather than on the assumption that a box must be better.
//!
//! Run with `--nocapture` to read the tables; the assertions only pin that the
//! diagnostic is measuring something non-vacuous.

use ferric_core::basis::bundled;
use ferric_core::mol::Molecule;
use ferric_dft::grid::{build_atomic_grid_pruned, AtomicGridConfig};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::cosx_a::PairBounds;

fn testdata(rel: &str) -> String {
    format!("{}/../../{}", env!("CARGO_MANIFEST_DIR"), rel)
}

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

/// The REAL COSX grid, in the REAL order the builder consumes it.
fn cosx_grid(mol: &Molecule) -> Vec<[f64; 3]> {
    let cfg = AtomicGridConfig { n_radial: 50, n_angular: 110, ..Default::default() };
    build_atomic_grid_pruned(mol, &cfg, cfg.prune).expect("grid").iter().map(|p| p.xyz).collect()
}

struct Sys {
    label: &'static str,
    mol: Molecule,
    prep: PreparedBasis,
}

fn systems() -> Vec<Sys> {
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
        Sys { label: "water/cc-pVDZ", prep: PreparedBasis::new(&water, &bw).expect("prep"), mol: water },
        Sys { label: "butane/def2-SVP", prep: PreparedBasis::new(&butane, &bb).expect("prep"), mol: butane },
    ]
}

/// Table 1: degenerate fraction vs group size, for each enclosing volume, on
/// the real grid in the real order. Sub-batches of 256 are split into
/// contiguous groups of `g`.
#[test]
fn degenerate_fraction_vs_group_size_and_volume() {
    for s in systems() {
        let pb = PairBounds::build(&s.prep).expect("bounds");
        let nsh = pb.nshells();
        let pts = cosx_grid(&s.mol);
        println!("\n=== {} ({} shells, {} grid points) ===", s.label, nsh, pts.len());
        println!("{:>6}  {:>10}  {:>10}  {:>10}  {:>12}", "group", "sphere", "box", "min", "queries");
        // Only a stride of sub-batches, so the diagnostic stays cheap; the
        // fractions are stable to <0.3% against the full sweep (checked once).
        let subs: Vec<&[[f64; 3]]> = pts.chunks(256).step_by(7).collect();
        for g in [256usize, 128, 64, 32, 16, 8, 4] {
            let (mut ds, mut db, mut dm, mut n) = (0usize, 0usize, 0usize, 0usize);
            for sub in &subs {
                for grp in sub.chunks(g) {
                    let (lo, hi) = aabb(grp);
                    let (cen, rad) = bounding_sphere(grp);
                    for s1 in 0..nsh {
                        for s2 in 0..=s1 {
                            let m = pb.max_estimate(s1, s2);
                            let bs = pb.coarse_estimate_sphere(s1, s2, &cen, rad);
                            let bb = pb.coarse_estimate_box(s1, s2, &lo, &hi);
                            if bs >= m {
                                ds += 1;
                            }
                            if bb >= m {
                                db += 1;
                            }
                            if bs.min(bb) >= m {
                                dm += 1;
                            }
                            n += 1;
                        }
                    }
                }
            }
            let f = |x: usize| x as f64 / n as f64;
            println!("{g:>6}  {:>10.4}  {:>10.4}  {:>10.4}  {n:>12}", f(ds), f(db), f(dm));
        }
    }
}

/// Table 2: WHY it degenerates. For each (pair, group) query that returns
/// `total`, is it because the region reaches the midpoint (`d(M, region) = 0`,
/// fixable by grouping) or because `|AB|/2` eats a positive distance (NOT
/// fixable by grouping)? Reported at the coarsest and finest group sizes.
#[test]
fn degeneracy_cause_split_region_vs_pair_extent() {
    for s in systems() {
        let pb = PairBounds::build(&s.prep).expect("bounds");
        let nsh = pb.nshells();
        let pts = cosx_grid(&s.mol);
        println!("\n=== {} : cause of degeneracy (box volume) ===", s.label);
        println!("{:>6}  {:>12}  {:>14}  {:>14}", "group", "degenerate", "region reaches", "|AB|/2 eats it");
        let subs: Vec<&[[f64; 3]]> = pts.chunks(256).step_by(11).collect();
        for g in [256usize, 32, 8] {
            let (mut deg, mut by_region, mut by_half, mut n) = (0usize, 0usize, 0usize, 0usize);
            for sub in &subs {
                for grp in sub.chunks(g) {
                    let (lo, hi) = aabb(grp);
                    for s1 in 0..nsh {
                        for s2 in 0..=s1 {
                            let m = pb.max_estimate(s1, s2);
                            if pb.coarse_estimate_box(s1, s2, &lo, &hi) >= m {
                                deg += 1;
                                // Re-derive d(M, box) with half = 0 to separate
                                // the two causes: a pair whose bound is still
                                // degenerate at half = 0 is one the REGION
                                // reaches; otherwise |AB|/2 is what did it.
                                if pb.coarse_estimate_box_zero_half(s1, s2, &lo, &hi) >= m {
                                    by_region += 1;
                                } else {
                                    by_half += 1;
                                }
                            }
                            n += 1;
                        }
                    }
                }
            }
            println!(
                "{g:>6}  {:>11.4}%  {:>13.4}%  {:>13.4}%",
                100.0 * deg as f64 / n as f64,
                100.0 * by_region as f64 / n as f64,
                100.0 * by_half as f64 / n as f64
            );
        }
    }
}
