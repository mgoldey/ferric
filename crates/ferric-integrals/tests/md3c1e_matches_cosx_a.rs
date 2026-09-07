//! Exactness anchor for the McMurchie-Davidson 3c1e kernel (`md3c1e`)
//! against the libint2-backed `cosx_a` path — an INDEPENDENT construction of
//! the same integrals (libint2's Obara-Saika/HGP-class engine with its own
//! Boys function and its own cart→sph transform, vs ferric's from-scratch MD
//! recursion). Agreement here is corroboration, not consistency.
//!
//! Coverage that is required, not incidental:
//! * every `(l_a, l_b)` combination up to `(4, 4)` — butane/def2-QZVP has g
//!   on carbon; a reachability assert fails the test if any combination was
//!   never visited, so a g-free basis cannot pass vacuously;
//! * degenerate probes: ON every nucleus (`T = 0` for same-centre pairs),
//!   `1e-4` and `1e-8` Bohr off a nucleus, and 50/60/200 Bohr away;
//! * batch padding: batches of 13 points (not a multiple of `TILE`) plus
//!   the single-point drop-in path;
//! * screening semantics of the drop-in vs `cosx_a` at the same threshold.
//!
//! Bar: `max|diff| <= 1e-12 * max(1, max|A_block|)` per shell-pair block.

use ferric_core::basis::bundled;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::cosx_a::{self, CosxScreen, PairBounds};
use ferric_integrals::engine::Engine;
use ferric_integrals::ffi;
use ferric_integrals::md3c1e::{self, Md3c1e};
use ndarray::Array2;

const BAR: f64 = 1e-12;
const N_RANDOM: usize = 50;
const BATCH: usize = 13;

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

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Class {
    Random,
    OnNucleus,
    Near1em4,
    Near1em8,
    Far,
}

fn probes(mol: &Molecule) -> Vec<(Class, [f64; 3])> {
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for a in &mol.atoms {
        for (d, c) in [a.x, a.y, a.zpos].iter().enumerate() {
            lo[d] = lo[d].min(*c);
            hi[d] = hi[d].max(*c);
        }
    }
    let mut rng = Lcg(20260907);
    let mut out = Vec::new();
    for _ in 0..N_RANDOM {
        let mut r = [0.0; 3];
        for d in 0..3 {
            r[d] = lo[d] - 3.0 + (hi[d] - lo[d] + 6.0) * rng.next_f64();
        }
        out.push((Class::Random, r));
    }
    for a in &mol.atoms {
        out.push((Class::OnNucleus, [a.x, a.y, a.zpos]));
    }
    for a in [&mol.atoms[0], &mol.atoms[mol.atoms.len() - 1]] {
        out.push((Class::Near1em4, [a.x + 1e-4, a.y, a.zpos]));
        out.push((Class::Near1em8, [a.x, a.y + 1e-8, a.zpos]));
    }
    out.push((Class::Far, [50.0, 0.0, 0.0]));
    out.push((Class::Far, [0.0, 0.0, 60.0]));
    out.push((Class::Far, [200.0, 30.0, 0.0]));
    out
}

struct Stats {
    /// max scaled |diff| per (l_a, l_b) with l_a <= l_b
    by_l: [[f64; 5]; 5],
    visited: [[usize; 5]; 5],
    by_class: Vec<(Class, f64)>,
}

impl Stats {
    fn new() -> Self {
        Self { by_l: [[0.0; 5]; 5], visited: [[0; 5]; 5], by_class: Vec::new() }
    }
    fn record_class(&mut self, c: Class, v: f64) {
        if let Some(e) = self.by_class.iter_mut().find(|e| e.0 == c) {
            e.1 = e.1.max(v);
        } else {
            self.by_class.push((c, v));
        }
    }
}

/// Compare one full `(nbf, nbf)` matrix pair block-by-block; returns the
/// worst scaled diff over the matrix.
fn compare_blocks(kern: &Md3c1e, got: &Array2<f64>, want: &Array2<f64>, stats: &mut Stats, label: &str) -> f64 {
    let nsh = kern.nshells();
    let mut worst = 0.0_f64;
    for s1 in 0..nsh {
        for s2 in 0..=s1 {
            let (o1, n1) = (kern.shell_offset(s1), kern.shell_dim(s1));
            let (o2, n2) = (kern.shell_offset(s2), kern.shell_dim(s2));
            let mut diff = 0.0_f64;
            let mut mag = 0.0_f64;
            for i in 0..n1 {
                for j in 0..n2 {
                    let w = want[(o1 + i, o2 + j)];
                    let g = got[(o1 + i, o2 + j)];
                    diff = diff.max((g - w).abs());
                    mag = mag.max(w.abs());
                    // symmetric fill
                    let gt = got[(o2 + j, o1 + i)];
                    assert_eq!(g, gt, "{label}: md3c1e output not symmetric at ({},{})", o1 + i, o2 + j);
                }
            }
            let scaled = diff / mag.max(1.0);
            let (la, lb) = (kern.shell_l(s1).min(kern.shell_l(s2)), kern.shell_l(s1).max(kern.shell_l(s2)));
            stats.by_l[la][lb] = stats.by_l[la][lb].max(scaled);
            stats.visited[la][lb] += 1;
            worst = worst.max(scaled);
            assert!(
                scaled <= BAR,
                "{label}: shell pair ({s1},{s2}) l=({la},{lb}) max|diff| {diff:.3e} (max|A| {mag:.3e}, scaled {scaled:.3e}) > {BAR:e}"
            );
        }
    }
    worst
}

fn run_case(name: &str, xyz: &str, basis: &str, stats: &mut Stats) {
    let mol = Molecule::load_xyz(&testdata(xyz)).expect("xyz");
    let bs = bundled(basis).expect("basis");
    let prep = PreparedBasis::new(&mol, &bs).expect("prep");
    let kern = Md3c1e::new(&prep).expect("md3c1e");
    let mut scr = kern.scratch();
    let mut eng = Engine::new_1e(ffi::OP_NUCLEAR, &prep, 1e-14).expect("nuclear engine");
    let pts = probes(&mol);
    let nbf = prep.nbasis();
    println!(
        "[{name}] {basis}: natoms={} nsh={} nbf={nbf} L_max={} probes={} fma={}",
        mol.atoms.len(),
        prep.nshells(),
        prep.max_l(),
        pts.len(),
        kern.uses_fma()
    );

    // Reference, one point at a time.
    let want: Vec<Array2<f64>> = pts
        .iter()
        .map(|(_, r)| cosx_a::a_matrix_at_point_with(&mut eng, &prep, r, None, CosxScreen::none()).expect("cosx_a").a)
        .collect();

    // Batched MD path, batches of BATCH (exercises tile padding).
    let mut worst = 0.0_f64;
    for (b0, chunk) in pts.chunks(BATCH).enumerate() {
        let coords: Vec<[f64; 3]> = chunk.iter().map(|(_, r)| *r).collect();
        let batch = kern.a_matrices(&coords, None, CosxScreen::none(), &mut scr).expect("md3c1e batch");
        assert_eq!(batch.pairs_kept, batch.pairs_total);
        for (k, (class, _)) in chunk.iter().enumerate() {
            let got = batch.a.index_axis(ndarray::Axis(0), k).to_owned();
            let w = compare_blocks(&kern, &got, &want[b0 * BATCH + k], stats, &format!("{name} batch pt {}", b0 * BATCH + k));
            stats.record_class(*class, w);
            worst = worst.max(w);
        }
    }

    // Single-point drop-in path on the degenerate probes.
    for (k, (class, r)) in pts.iter().enumerate() {
        if *class == Class::Random {
            continue;
        }
        let got = md3c1e::a_matrix_at_point_with(&kern, &mut scr, r, None, CosxScreen::none()).expect("drop-in").a;
        let w = compare_blocks(&kern, &got, &want[k], stats, &format!("{name} drop-in pt {k} {class:?}"));
        worst = worst.max(w);
    }
    println!("[{name}] worst scaled |diff| over all probes and paths: {worst:.3e}");
}

#[test]
fn md3c1e_matches_cosx_a_on_every_l_pair_including_degenerate_probes() {
    let mut stats = Stats::new();
    run_case("water", "testdata/molecules/water.xyz", "cc-pvdz", &mut stats);
    run_case("butane", "testdata/molecules/alkane_4.xyz", "def2-svp", &mut stats);
    run_case("butane", "testdata/molecules/alkane_4.xyz", "def2-qzvp", &mut stats);

    println!("\nmax scaled |A_md - A_cosx_a| per (l_a, l_b):");
    println!("{:>6} {:>6} {:>12} {:>8}", "l_a", "l_b", "max|diff|", "blocks");
    for la in 0..=4 {
        for lb in la..=4 {
            println!("{la:>6} {lb:>6} {:>12.3e} {:>8}", stats.by_l[la][lb], stats.visited[la][lb]);
            assert!(
                stats.visited[la][lb] > 0,
                "reachability: (l_a, l_b) = ({la}, {lb}) was never visited — the anchor would pass vacuously for that combination"
            );
        }
    }
    println!("\nmax scaled |diff| per probe class:");
    for (c, v) in &stats.by_class {
        println!("  {c:<12?} {v:.3e}");
    }
    for c in [Class::Random, Class::OnNucleus, Class::Near1em4, Class::Near1em8, Class::Far] {
        assert!(stats.by_class.iter().any(|e| e.0 == c), "probe class {c:?} never exercised");
    }
}

/// The drop-in must reproduce cosx_a's screening decisions exactly at a
/// single point (same `PairBounds`, same threshold, same kept count).
#[test]
fn md3c1e_drop_in_screens_identically_to_cosx_a() {
    let mol = Molecule::load_xyz(&testdata("testdata/molecules/alkane_4.xyz")).expect("xyz");
    let bs = bundled("def2-svp").expect("basis");
    let prep = PreparedBasis::new(&mol, &bs).expect("prep");
    let bounds = PairBounds::build(&prep).expect("bounds");
    let screen = CosxScreen::at(1e-7);
    let pts = [[7.5, -3.0, 2.0], [0.4, 0.2, 0.1], [30.0, 0.0, 0.0]];
    let mut any_dropped = false;
    for r in &pts {
        let want = cosx_a::a_matrix_at_point(&prep, r, Some(&bounds), screen).expect("cosx_a");
        let got = md3c1e::a_matrix_at_point(&prep, r, Some(&bounds), screen).expect("md3c1e");
        assert_eq!(got.pairs_total, want.pairs_total);
        assert_eq!(got.pairs_kept, want.pairs_kept, "kept count differs at {r:?}");
        any_dropped |= want.pairs_kept < want.pairs_total;
        let diff = (&got.a - &want.a).mapv(f64::abs).fold(0.0_f64, |m, &v| m.max(v));
        assert!(diff <= BAR, "screened A differs by {diff:.3e} at {r:?}");
    }
    assert!(any_dropped, "screen dropped nothing at any probe: the test did not exercise screening");
}

/// The FMA (AVX2) and portable code paths are two compilations of one
/// algorithm; they must agree to rounding. Skips (passes) if FMA is absent.
#[test]
fn md3c1e_fma_and_portable_paths_agree() {
    let mol = Molecule::load_xyz(&testdata("testdata/molecules/water.xyz")).expect("xyz");
    let bs = bundled("cc-pvdz").expect("basis");
    let prep = PreparedBasis::new(&mol, &bs).expect("prep");
    let mut kern = Md3c1e::new(&prep).expect("md3c1e");
    if !kern.uses_fma() {
        println!("no AVX2/FMA on this host; nothing to compare");
        return;
    }
    let pts: Vec<[f64; 3]> = (0..20).map(|i| [0.3 * i as f64 - 2.0, 0.17 * i as f64, 1.0 - 0.1 * i as f64]).collect();
    let mut scr = kern.scratch();
    let a = kern.a_matrices(&pts, None, CosxScreen::none(), &mut scr).expect("fma").a;
    kern.set_use_fma(false);
    let b = kern.a_matrices(&pts, None, CosxScreen::none(), &mut scr).expect("portable").a;
    let diff = (&a - &b).mapv(f64::abs).fold(0.0_f64, |m, &v| m.max(v));
    println!("fma vs portable max|diff| = {diff:.3e}");
    assert!(diff <= 1e-13, "fma vs portable differ by {diff:.3e}");
}
