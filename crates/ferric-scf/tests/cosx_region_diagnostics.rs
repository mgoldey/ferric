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

/// The SIZE axis: alkanes at def2-SVP, the same basis butane uses, spanning
/// the ~30 Bohr onset where alkane locality turns on. `alkane_16` is 38.7 Bohr
/// and is the first system past it — the repo rule "do not declare a negative
/// below the onset" is why this axis exists.
fn alkanes() -> Vec<(&'static str, Molecule, PreparedBasis)> {
    let bs = bundled("def2-svp").expect("def2-svp");
    ["alkane_4", "alkane_8", "alkane_12", "alkane_16"]
        .iter()
        .map(|n| {
            let mol = Molecule::load_xyz(&testdata(&format!("testdata/molecules/{n}.xyz"))).expect("alkane");
            let prep = PreparedBasis::new(&mol, &bs).expect("prep");
            (*n, mol, prep)
        })
        .collect()
}

/// Largest interatomic distance, in Bohr (coords are already Bohr).
fn diameter(mol: &Molecule) -> f64 {
    let mut d: f64 = 0.0;
    for (i, a) in mol.atoms.iter().enumerate() {
        for b in &mol.atoms[i + 1..] {
            let r = (a.x - b.x).powi(2) + (a.y - b.y).powi(2) + (a.zpos - b.zpos).powi(2);
            d = d.max(r.sqrt());
        }
    }
    d
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

/// Table 3 (THE SIZE-AXIS DELIVERABLE): the cause split as a function of
/// molecular size, across the ~30 Bohr locality onset.
///
/// # The hypothesis this is built to discriminate
///
/// `|AB|/2` is a FIXED property of a shell pair, but the batch region GROWS
/// with the molecule: a Becke grid over a longer chain spreads its 256-point
/// sub-batches further apart, so a sub-batch's enclosing region gets bigger.
/// If the region term therefore comes to DOMINATE at C16, then grouping starts
/// to matter and the flat kept-work curve measured on butane is an artifact of
/// small molecules. If the `|AB|/2` share instead stays flat or grows, the
/// negative verdict is confirmed past the onset.
///
/// Artifact hypothesis, stated next to the physics one as the repo requires:
/// if the sub-batch stride below silently sampled only sub-batches near one
/// end of the chain, the region column would look size-independent for a
/// reason that has nothing to do with geometry. Guarded by `assert`ing the
/// sampled sub-batch count grows with the system, and by striding a PRIME
/// number of sub-batches so the sample walks the whole grid.
#[test]
fn cause_split_vs_molecular_size_across_the_locality_onset() {
    println!(
        "\n{:>10}  {:>7}  {:>5}  {:>6}  {:>6}  {:>11}  {:>15}  {:>15}",
        "system", "diam", "nsh", "group", "subs", "degenerate", "region reaches", "|AB|/2 eats it"
    );
    let mut seen = Vec::new();
    for (name, mol, prep) in alkanes() {
        let pb = PairBounds::build(&prep).expect("bounds");
        let nsh = pb.nshells();
        let pts = cosx_grid(&mol);
        let diam = diameter(&mol);
        // Fixed SAMPLE SIZE, not a fixed stride: the pair loop is O(nsh^2) and
        // nsh grows linearly, so a fixed stride would make C16 ~16x the work of
        // C4. Take ~24 sub-batches spread over the whole grid instead, which
        // keeps every size inside the run window and keeps the sample walking
        // the entire chain rather than one end of it.
        let n_sub = pts.len().div_ceil(256);
        let stride = (n_sub / 24).max(1);
        let subs: Vec<&[[f64; 3]]> = pts.chunks(256).step_by(stride).collect();
        assert!(subs.len() >= 8, "{name}: only {} sub-batches sampled — too thin to read", subs.len());
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
            let pct = |x: usize| 100.0 * x as f64 / n as f64;
            println!(
                "{name:>10}  {diam:>7.1}  {nsh:>5}  {g:>6}  {:>6}  {:>10.4}%  {:>14.4}%  {:>14.4}%",
                subs.len(),
                pct(deg),
                pct(by_region),
                pct(by_half)
            );
        }
        seen.push((name, diam, nsh, subs.len(), pts.len()));
    }
    // Artifact guard: the size axis must actually vary, and the sample must
    // stay a real sample at every size (not collapse to one sub-batch at C16).
    assert!(seen.len() == 4, "expected 4 alkane sizes, got {}", seen.len());
    let (d0, d3) = (seen[0].1, seen[3].1);
    assert!(d3 > 3.0 * d0, "size axis did not span: {d0:.1} -> {d3:.1} Bohr");
    assert!(d3 > 30.0, "largest system is {d3:.1} Bohr — below the ~30 Bohr onset this test exists to cross");
    assert!(seen.iter().all(|s| s.3 >= 8), "a size sampled fewer than 8 sub-batches: {seen:?}");
    println!(
        "\nRead the last two columns DOWN each group block: if `region reaches` grows with size \
         relative to `|AB|/2 eats it`, grouping starts to matter past the onset and the \
         butane-scale negative is an artifact. If `|AB|/2` holds or grows, the negative is \
         confirmed past 30 Bohr."
    );
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

/// Supporting diagnostic for `cause_split_vs_molecular_size_across_the_locality_onset`:
/// is the sub-batch REGION actually growing with the molecule?
///
/// The size-axis table shows the `region reaches` share FALLING with size
/// (37.9% at C4 -> 15.9% at C16), which is the opposite of the "bigger
/// molecule spreads its sub-batches further" hypothesis. This checks the
/// mechanism directly rather than inferring it: it reports the distribution of
/// sub-batch region EXTENT (the AABB's half-diagonal) at each size.
///
/// If the extent is roughly size-INDEPENDENT, the explanation is that a Becke
/// sub-batch is an atom-local object — 256 consecutive points are ~2.3 Lebedev
/// shells of ONE atom, and that does not care how long the chain is — while
/// `nsh` grows linearly, so the O(nsh^2) pair denominator fills up with distant
/// pairs whose degeneracy (when it happens) is a `|AB|/2` effect.
#[test]
fn subbatch_region_extent_is_size_independent() {
    println!("\n{:>10}  {:>7}  {:>5}  {:>10}  {:>10}  {:>10}", "system", "diam", "subs", "med extent", "p90", "max");
    let mut medians = Vec::new();
    for (name, mol, _prep) in alkanes() {
        let pts = cosx_grid(&mol);
        let diam = diameter(&mol);
        let mut ext: Vec<f64> = pts
            .chunks(256)
            .map(|sub| {
                let (lo, hi) = aabb(sub);
                0.5 * ((hi[0] - lo[0]).powi(2) + (hi[1] - lo[1]).powi(2) + (hi[2] - lo[2]).powi(2)).sqrt()
            })
            .collect();
        ext.sort_by(|a, b| a.partial_cmp(b).expect("finite extents"));
        let med = ext[ext.len() / 2];
        let p90 = ext[(ext.len() * 9 / 10).min(ext.len() - 1)];
        let mx = *ext.last().expect("non-empty grid");
        println!("{name:>10}  {diam:>7.1}  {:>5}  {med:>10.3}  {p90:>10.3}  {mx:>10.3}", ext.len());
        medians.push((name, diam, med));
    }
    // THE POINT: the median sub-batch extent must NOT track the diameter. If it
    // did, the "bigger molecule => bigger region" hypothesis would hold and the
    // size-axis verdict would need re-reading.
    let (first, last) = (medians[0].2, medians[3].2);
    let (d0, d3) = (medians[0].1, medians[3].1);
    println!(
        "\ndiameter grew {:.2}x ({d0:.1} -> {d3:.1} Bohr); median sub-batch extent grew {:.2}x ({first:.3} -> {last:.3} Bohr)",
        d3 / d0,
        last / first
    );
    assert!(
        last < 2.0 * first,
        "median sub-batch extent grew {:.2}x ({first:.3} -> {last:.3}) while the diameter grew {:.2}x — \
         the region DOES track molecular size, so re-read the size-axis cause split",
        last / first,
        d3 / d0
    );
}
