//! Anchors for the COSX shell-pair screen (`cosx_a::PairBounds`), shared by
//! the libint2 path (`cosx_a`) and the MD kernel (`md3c1e`).
//!
//! History (2026-09-07): the original bound `sqrt(max|S_block|)/d` was built
//! from the SIGNED overlap block, which vanishes by symmetry for same-centre
//! s-p / s-d / p-d pairs while their `A^g` block is O(1). Measured on
//! water/cc-pVDZ (50,110) at `t = 1e-7`: `max|K_scr - K_unscr| = 0.71`
//! against a grid error of `4.8e-5`. A bound must never underestimate; these
//! anchors pin that.
//!
//! * `screen_bound_never_underestimates_any_shell_pair` — every shell pair ×
//!   ≥ 200 probes (random, ON every nucleus, 1e-4 Bohr off, 50/60/200 Bohr
//!   away) on water/cc-pVDZ, butane/def2-SVP and butane/def2-QZVP (has g):
//!   `max_block |A^g| <= bound`. Also asserts the same-centre mixed-l bounds
//!   are NOT ~0 (the exact failure mode) and prints the tightness
//!   distribution `true / bound`.
//! * `legacy_signed_overlap_bound_underestimates` — the OLD formula, rebuilt
//!   here, run through the SAME checker, must be caught violating. This is
//!   the permanent mutation test: if the checker ever stops seeing that
//!   failure, the checker is broken.
//! * `screen_zero_threshold_matches_unscreened` — trivial limit, exercising
//!   the live bound object.
//! * `screen_drops_pairs_on_octane_def2svp_at_1e7` — reachability: the
//!   screen must not be vacuous (reported, not tuned).
//! * `screened_k_matches_unscreened_k_below_grid_error_water_ccpvdz` — the
//!   number that was 0.71: full weighted (50,110) COSX-K contraction against
//!   a fixed hcore-guess density, screened vs unscreened at 1e-7 and 1e-8,
//!   asserted `< 1e-6` (grid error vs exact ERIs printed alongside).

use ferric_core::basis::{bundled, BasisSet};
use ferric_core::mol::Molecule;
use ferric_integrals::ao_grid::eval_basis_on_points;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::cosx_a::{CosxScreen, PairBounds};
use ferric_integrals::engine::Engine;
use ferric_integrals::md3c1e::Md3c1e;
use ferric_integrals::oneelectron;
use ferric_integrals::operator::Operator;
use ferric_quadrature::lebedev::lebedev;
use ndarray::{Array2, Axis};

fn testdata(rel: &str) -> String {
    format!("{}/../../{}", env!("CARGO_MANIFEST_DIR"), rel)
}

struct Lcg(u64);
impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 11
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Class {
    Random,
    OnNucleus,
    Near1em4,
    Far,
}

/// ≥ 200 probes: 160 random in the molecular box (+3 Bohr margin), one ON
/// every nucleus, 1e-4 Bohr off every nucleus, and 50/60/200 Bohr away.
fn probes(mol: &Molecule, n_random: usize) -> Vec<(Class, [f64; 3])> {
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
    for _ in 0..n_random {
        let mut r = [0.0; 3];
        for d in 0..3 {
            r[d] = lo[d] - 3.0 + (hi[d] - lo[d] + 6.0) * rng.next_f64();
        }
        out.push((Class::Random, r));
    }
    for a in &mol.atoms {
        out.push((Class::OnNucleus, [a.x, a.y, a.zpos]));
        out.push((Class::Near1em4, [a.x + 1e-4, a.y - 0.5e-4, a.zpos]));
    }
    out.push((Class::Far, [50.0, 0.0, 0.0]));
    out.push((Class::Far, [0.0, 0.0, 60.0]));
    out.push((Class::Far, [200.0, 30.0, 0.0]));
    out
}

/// A screening estimator under test: `(s1, s2, r) -> upper bound on max_block|A^g|`.
trait Estimator {
    fn estimate(&self, s1: usize, s2: usize, r: &[f64; 3]) -> f64;
}

impl Estimator for PairBounds {
    fn estimate(&self, s1: usize, s2: usize, r: &[f64; 3]) -> f64 {
        PairBounds::estimate(self, s1, s2, r)
    }
}

/// The ORIGINAL (unsound) bound, reconstructed verbatim: `sqrt(max|S_block|)`
/// over the SIGNED overlap block, divided by the distance to the shell-centre
/// midpoint (floored at 1e-8).
struct LegacySignedOverlap {
    nsh: usize,
    mag: Vec<f64>,
    centre: Vec<[f64; 3]>,
}

fn tri(s1: usize, s2: usize) -> usize {
    s1 * (s1 + 1) / 2 + s2
}

impl LegacySignedOverlap {
    fn build(prep: &PreparedBasis) -> Self {
        let nsh = prep.nshells();
        let dims = prep.shell_dims();
        let offs = prep.shell_offsets();
        let centres = prep.shell_centers();
        let s = oneelectron::overlap(prep);
        let npair = nsh * (nsh + 1) / 2;
        let mut mag = vec![0.0; npair];
        let mut centre = vec![[0.0; 3]; npair];
        for s1 in 0..nsh {
            for s2 in 0..=s1 {
                let mut m = 0.0_f64;
                for i in 0..dims[s1] {
                    for j in 0..dims[s2] {
                        m = m.max(s[(offs[s1] + i, offs[s2] + j)].abs());
                    }
                }
                let idx = tri(s1, s2);
                mag[idx] = m.sqrt();
                let (c1, c2) = (centres[s1], centres[s2]);
                centre[idx] = [0.5 * (c1[0] + c2[0]), 0.5 * (c1[1] + c2[1]), 0.5 * (c1[2] + c2[2])];
            }
        }
        Self { nsh, mag, centre }
    }
}

impl Estimator for LegacySignedOverlap {
    fn estimate(&self, s1: usize, s2: usize, r: &[f64; 3]) -> f64 {
        debug_assert!(s1 < self.nsh && s2 <= s1);
        let idx = tri(s1, s2);
        let c = self.centre[idx];
        let d2 = (r[0] - c[0]).powi(2) + (r[1] - c[1]).powi(2) + (r[2] - c[2]).powi(2);
        self.mag[idx] / d2.sqrt().max(1e-8)
    }
}

/// One violation record: `(s1, s2, l1, l2, class, true, bound)`.
type Violation = (usize, usize, usize, usize, Class, f64, f64);

struct CheckReport {
    violations: Vec<Violation>,
    /// `true / bound` for every (pair, probe) with `true > 1e-14`.
    ratios: Vec<f64>,
    /// per (l_lo, l_hi): (max ratio, count)
    by_l: [[(f64, usize); 5]; 5],
    /// Same-centre pairs with `l1 != l2`, at the probe where the true block
    /// max is largest: `(s1, s2, l1, l2, true, bound)`.
    same_centre_mixed: Vec<(usize, usize, usize, usize, f64, f64)>,
}

/// Run every shell pair × every probe through `est` against the exact
/// blocks from `md3c1e`. Never asserts — the callers decide what a
/// violation means (the legacy test WANTS them).
fn check_estimator(name: &str, xyz: &str, basis: &str, est: &dyn Estimator, n_random: usize) -> CheckReport {
    let mol = Molecule::load_xyz(&testdata(xyz)).expect("xyz");
    let bs = bundled(basis).expect("basis");
    let prep = PreparedBasis::new(&mol, &bs).expect("prep");
    let kern = Md3c1e::new(&prep).expect("md3c1e");
    let mut scr = kern.scratch();
    let pts = probes(&mol, n_random);
    assert!(pts.len() >= 200, "{name}: only {} probes", pts.len());
    let nsh = prep.nshells();
    let atom_of = prep.shell_to_atom().to_vec();
    println!(
        "[{name}] {basis}: natoms={} nsh={} nbf={} L_max={} probes={}",
        mol.atoms.len(),
        nsh,
        prep.nbasis(),
        prep.max_l(),
        pts.len()
    );

    let mut rep = CheckReport {
        violations: Vec::new(),
        ratios: Vec::new(),
        by_l: [[(0.0, 0); 5]; 5],
        same_centre_mixed: Vec::new(),
    };
    const BATCH: usize = 16;
    for chunk in pts.chunks(BATCH) {
        let coords: Vec<[f64; 3]> = chunk.iter().map(|(_, r)| *r).collect();
        let batch = kern.a_matrices(&coords, None, CosxScreen::none(), &mut scr).expect("md3c1e batch");
        for (k, (class, r)) in chunk.iter().enumerate() {
            let a = batch.a.index_axis(Axis(0), k);
            for s1 in 0..nsh {
                for s2 in 0..=s1 {
                    let (o1, n1) = (kern.shell_offset(s1), kern.shell_dim(s1));
                    let (o2, n2) = (kern.shell_offset(s2), kern.shell_dim(s2));
                    let mut t = 0.0_f64;
                    for i in 0..n1 {
                        for j in 0..n2 {
                            t = t.max(a[(o1 + i, o2 + j)].abs());
                        }
                    }
                    let b = est.estimate(s1, s2, r);
                    let (l1, l2) = (kern.shell_l(s1), kern.shell_l(s2));
                    let (la, lb) = (l1.min(l2), l1.max(l2));
                    if t > 1e-14 {
                        let ratio = t / b;
                        rep.ratios.push(ratio);
                        let e = &mut rep.by_l[la][lb];
                        e.0 = e.0.max(ratio);
                        e.1 += 1;
                    }
                    if t > b {
                        rep.violations.push((s1, s2, l1, l2, *class, t, b));
                    }
                    // Same-centre mixed-l pairs: track the probe where the
                    // TRUE value is largest (ON the nucleus it vanishes by
                    // parity/angular orthogonality — the bug bit off-centre).
                    if s1 != s2 && l1 != l2 && atom_of[s1] == atom_of[s2] {
                        match rep.same_centre_mixed.iter_mut().find(|e| e.0 == s1 && e.1 == s2) {
                            Some(e) if e.4 >= t => {}
                            Some(e) => {
                                e.4 = t;
                                e.5 = b;
                            }
                            None => rep.same_centre_mixed.push((s1, s2, l1, l2, t, b)),
                        }
                    }
                }
            }
        }
    }
    rep
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let i = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[i.min(sorted.len() - 1)]
}

fn print_tightness(name: &str, rep: &CheckReport) {
    let mut r = rep.ratios.clone();
    r.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "[{name}] tightness true/bound over {} (pair,probe) samples: min {:.3e}  p10 {:.3e}  p50 {:.3e}  p90 {:.3e}  p99 {:.3e}  max {:.3e}",
        r.len(),
        percentile(&r, 0.0),
        percentile(&r, 0.10),
        percentile(&r, 0.50),
        percentile(&r, 0.90),
        percentile(&r, 0.99),
        percentile(&r, 1.0)
    );
    println!("[{name}] max true/bound per (l_a, l_b):");
    for la in 0..=4 {
        for lb in la..=4 {
            let (m, n) = rep.by_l[la][lb];
            if n > 0 {
                println!("    ({la},{lb}) max {m:.3e}  n={n}");
            }
        }
    }
}

const CASES: [(&str, &str, &str, usize); 3] = [
    ("water", "testdata/molecules/water.xyz", "cc-pvdz", 195),
    ("butane", "testdata/molecules/alkane_4.xyz", "def2-svp", 170),
    ("butane", "testdata/molecules/alkane_4.xyz", "def2-qzvp", 170),
];

#[test]
fn screen_bound_never_underestimates_any_shell_pair() {
    let mut visited = [[false; 5]; 5];
    for (name, xyz, basis, n_random) in CASES {
        let mol = Molecule::load_xyz(&testdata(xyz)).expect("xyz");
        let bs = bundled(basis).expect("basis");
        let prep = PreparedBasis::new(&mol, &bs).expect("prep");
        let bounds = PairBounds::build(&prep).expect("pair bounds");
        let rep = check_estimator(name, xyz, basis, &bounds, n_random);
        print_tightness(&format!("{name}/{basis}"), &rep);
        for la in 0..=4 {
            for lb in la..=4 {
                visited[la][lb] |= rep.by_l[la][lb].1 > 0;
            }
        }

        // The exact failure mode: same-centre pairs with l1 != l2, probed on
        // their own nucleus. Their bound must be O(true), never ~0.
        assert!(!rep.same_centre_mixed.is_empty(), "{name}/{basis}: no same-centre mixed-l pair found");
        println!("[{name}/{basis}] same-centre mixed-l pairs at their worst probe (s1,s2,l1,l2, true, bound) — first 30 of {}:", rep.same_centre_mixed.len());
        let mut worst_ratio = 0.0_f64;
        for (n, (s1, s2, l1, l2, t, b)) in rep.same_centre_mixed.iter().enumerate() {
            if n < 30 {
                println!("    ({s1},{s2}) l=({l1},{l2})  true {t:.4e}  bound {b:.4e}");
            }
            worst_ratio = worst_ratio.max(t / b);
            assert!(*b >= 1e-3, "{name}/{basis}: same-centre pair ({s1},{s2}) l=({l1},{l2}) bound {b:.3e} is ~0 while |A| = {t:.3e}");
            assert!(*b >= *t, "{name}/{basis}: same-centre pair ({s1},{s2}) l=({l1},{l2}) bound {b:.3e} < true {t:.3e}");
        }
        let n_big = rep.same_centre_mixed.iter().filter(|e| e.4 > 1e-2).count();
        println!("[{name}/{basis}] same-centre mixed-l worst true/bound = {worst_ratio:.3e}; pairs with true > 1e-2 at some probe: {n_big}/{}", rep.same_centre_mixed.len());
        assert!(n_big > 0, "{name}/{basis}: no same-centre mixed-l pair reached |A| > 1e-2 at any probe — the failure mode was not exercised");

        if !rep.violations.is_empty() {
            for (s1, s2, l1, l2, c, t, b) in rep.violations.iter().take(20) {
                println!("  VIOLATION ({s1},{s2}) l=({l1},{l2}) {c:?}: true {t:.4e} > bound {b:.4e}");
            }
        }
        assert!(
            rep.violations.is_empty(),
            "{name}/{basis}: {} (pair, probe) combinations where max|A| > bound (first: {:?})",
            rep.violations.len(),
            rep.violations[0]
        );
    }
    for la in 0..=4 {
        for lb in la..=4 {
            assert!(visited[la][lb], "reachability: (l_a,l_b)=({la},{lb}) never visited");
        }
    }
}

/// Permanent mutation test: the legacy signed-overlap bound, run through the
/// identical checker, MUST be caught underestimating same-centre mixed-l
/// pairs by O(1). If this test ever fails, the checker has lost its teeth.
#[test]
fn legacy_signed_overlap_bound_underestimates() {
    let (name, xyz, basis, n_random) = CASES[0];
    let mol = Molecule::load_xyz(&testdata(xyz)).expect("xyz");
    let bs = bundled(basis).expect("basis");
    let prep = PreparedBasis::new(&mol, &bs).expect("prep");
    let legacy = LegacySignedOverlap::build(&prep);
    let rep = check_estimator(&format!("LEGACY {name}"), xyz, basis, &legacy, n_random);
    let worst = rep.violations.iter().map(|v| v.5 - v.6).fold(0.0_f64, f64::max);
    let n_same_centre_zero = rep.same_centre_mixed.iter().filter(|e| e.5 < 1e-6 && e.4 > 1e-2).count();
    println!(
        "[LEGACY] violations={} worst underestimate (true - bound)={worst:.3e} same-centre mixed-l pairs with bound<1e-6 while true>1e-2: {n_same_centre_zero}/{}",
        rep.violations.len(),
        rep.same_centre_mixed.len()
    );
    assert!(!rep.violations.is_empty(), "the legacy bound was not caught violating: checker is broken");
    assert!(worst > 0.1, "legacy worst underestimate {worst:.3e} should be O(0.1-1)");
    assert!(n_same_centre_zero > 0, "legacy same-centre mixed-l bounds should be ~0; checker did not see it");
}

/// Trivial limit: a threshold of 0 (vacuous) and a threshold far below any
/// bound must both reproduce the unscreened matrices bit-for-bit and keep
/// every pair — through the live `PairBounds`.
#[test]
fn screen_zero_threshold_matches_unscreened() {
    let mol = Molecule::load_xyz(&testdata("testdata/molecules/alkane_4.xyz")).expect("xyz");
    let bs = bundled("def2-svp").expect("basis");
    let prep = PreparedBasis::new(&mol, &bs).expect("prep");
    let kern = Md3c1e::new(&prep).expect("md3c1e");
    let bounds = PairBounds::build(&prep).expect("bounds");
    let mut scr = kern.scratch();
    let pts: Vec<[f64; 3]> = probes(&mol, 40).iter().map(|(_, r)| *r).collect();
    let want = kern.a_matrices(&pts, None, CosxScreen::none(), &mut scr).expect("unscreened");
    for t in [0.0, -1.0, 1e-300] {
        let got = kern.a_matrices(&pts, Some(&bounds), CosxScreen::at(t), &mut scr).expect("screened");
        assert_eq!(got.pairs_kept, got.pairs_total, "t={t}: dropped pairs in the trivial limit");
        assert_eq!(got.a, want.a, "t={t}: screened matrices differ from unscreened");
    }
}

/// Treutler-Ahlrichs ξ (JCP 102, 346 (1995) Table I) for Z = 1..10.
fn ta_xi(z: i32) -> f64 {
    const XI: [f64; 11] = [1.0, 0.8, 0.9, 1.8, 1.4, 1.3, 1.1, 0.9, 0.9, 0.9, 0.9];
    if (z as usize) < XI.len() {
        XI[z as usize]
    } else {
        1.5
    }
}

/// TA-M4 radial nodes and weights (small→large r), the same formula as
/// `ferric_dft::radial::treutler_ahlrichs_m4`.
fn ta_m4(z: i32, n: usize) -> (Vec<f64>, Vec<f64>) {
    let alpha = 0.6_f64;
    let ln2 = ta_xi(z) / 2.0_f64.ln();
    let pi = std::f64::consts::PI;
    let np1 = (n + 1) as f64;
    let mut rs = Vec::with_capacity(n);
    let mut ws = Vec::with_capacity(n);
    for k in 1..=n {
        let theta = pi * k as f64 / np1;
        let x = theta.cos();
        let log_term = ((1.0 - x) / 2.0).ln();
        let r = -ln2 * (1.0 + x).powf(alpha) * log_term;
        let dr_dx = ln2 * (1.0 + x).powf(alpha) * (-alpha / (1.0 + x) * log_term + 1.0 / (1.0 - x));
        let w_cheb = (pi / np1) * theta.sin();
        rs.push(r);
        ws.push(w_cheb * dr_dx * 4.0 * pi * r * r);
    }
    rs.reverse();
    ws.reverse();
    (rs, ws)
}

fn bragg_slater_bohr(z: i32) -> f64 {
    let r_a: f64 = match z {
        1 => 0.35,
        2 => 0.35,
        3 => 1.45,
        4 => 1.05,
        5 => 0.85,
        6 => 0.70,
        7 => 0.65,
        8 => 0.60,
        9 => 0.50,
        10 => 0.45,
        _ => 1.00,
    };
    r_a * 1.8897259886
}

/// Becke fuzzy weights at `r` for every atom (Becke 1988, k=3 smoothing,
/// Bragg-Slater size adjustment) — the formula of `ferric_dft::becke::becke_weights_all`.
fn becke_weights_all(mol: &Molecule, r: [f64; 3]) -> Vec<f64> {
    let natoms = mol.atoms.len();
    let dists: Vec<f64> = mol
        .atoms
        .iter()
        .map(|a| ((r[0] - a.x).powi(2) + (r[1] - a.y).powi(2) + (r[2] - a.zpos).powi(2)).sqrt())
        .collect();
    let mut p = vec![1.0_f64; natoms];
    for a in 0..natoms {
        let ra = bragg_slater_bohr(mol.atoms[a].z);
        for b in 0..natoms {
            if a == b {
                continue;
            }
            let rb = bragg_slater_bohr(mol.atoms[b].z);
            let (aa, ab) = (&mol.atoms[a], &mol.atoms[b]);
            let r_ab = ((aa.x - ab.x).powi(2) + (aa.y - ab.y).powi(2) + (aa.zpos - ab.zpos).powi(2)).sqrt();
            let mu = (dists[a] - dists[b]) / r_ab;
            let chi = ra / rb;
            let u = (chi - 1.0) / (chi + 1.0);
            let corr = (u / (u * u - 1.0)).clamp(-0.5, 0.5);
            let mut x = mu + corr * (1.0 - mu * mu);
            for _ in 0..3 {
                x = 0.5 * x * (3.0 - x * x);
            }
            p[a] *= 0.5 * (1.0 - x);
        }
    }
    let total: f64 = p.iter().sum();
    p.iter().map(|v| v / total).collect()
}

/// The flat (n_rad, n_ang) Becke grid: positions and weights.
fn becke_grid(mol: &Molecule, n_rad: usize, n_ang: usize) -> (Vec<[f64; 3]>, Vec<f64>) {
    let (leb, leb_w) = lebedev(n_ang);
    let mut pts = Vec::new();
    let mut wts = Vec::new();
    for (ai, atom) in mol.atoms.iter().enumerate() {
        let (rs, ws) = ta_m4(atom.z, n_rad);
        for (r, wr) in rs.iter().zip(&ws) {
            for (p, wa) in leb.iter().zip(&leb_w) {
                let x = [atom.x + r * p[0], atom.y + r * p[1], atom.zpos + r * p[2]];
                let wb = becke_weights_all(mol, x)[ai];
                pts.push(x);
                wts.push(wr * wa * wb);
            }
        }
    }
    (pts, wts)
}

/// Grid positions only (no weights) — for the sparsity-count harness.
fn grid_positions(mol: &Molecule, n_rad: usize, n_ang: usize) -> Vec<[f64; 3]> {
    let (leb, _) = lebedev(n_ang);
    let mut pts = Vec::new();
    for atom in &mol.atoms {
        for r in ta_m4(atom.z, n_rad).0 {
            for p in &leb {
                pts.push([atom.x + r * p[0], atom.y + r * p[1], atom.zpos + r * p[2]]);
            }
        }
    }
    pts
}

fn sample_indices(n: usize, k: usize, seed: u64) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..n).collect();
    let mut rng = Lcg(seed);
    let k = k.min(n);
    for i in 0..k {
        let j = i + (rng.next_u64() as usize) % (n - i);
        idx.swap(i, j);
    }
    idx.truncate(k);
    idx
}

/// Reachability: at `t = 1e-7` on octane/def2-SVP the screen must drop a
/// non-trivial fraction of shell pairs over a fixed 500-point sample of the
/// (50,110) grid. Reported, not tuned: a valid bound that drops nothing
/// would be a finding, and this test would then fail loudly.
#[test]
fn screen_drops_pairs_on_octane_def2svp_at_1e7() {
    let mol = Molecule::load_xyz(&testdata("testdata/molecules/alkane_8.xyz")).expect("xyz");
    let bs = bundled("def2-svp").expect("basis");
    let prep = PreparedBasis::new(&mol, &bs).expect("prep");
    let bounds = PairBounds::build(&prep).expect("bounds");
    let grid = grid_positions(&mol, 50, 110);
    let idx = sample_indices(grid.len(), 500, 20260907);
    let nsh = prep.nshells();
    let total = nsh * (nsh + 1) / 2;
    let t = 1e-7;
    let mut kept_sum = 0usize;
    let mut kept_min = usize::MAX;
    let mut kept_max = 0usize;
    for &g in &idx {
        let r = grid[g];
        let mut kept = 0usize;
        for s1 in 0..nsh {
            for s2 in 0..=s1 {
                if bounds.estimate(s1, s2, &r) >= t {
                    kept += 1;
                }
            }
        }
        kept_sum += kept;
        kept_min = kept_min.min(kept);
        kept_max = kept_max.max(kept);
    }
    let frac = kept_sum as f64 / (idx.len() * total) as f64;
    println!(
        "[octane/def2-svp] t={t:e}: nsh={nsh} pairs={total} mean kept fraction {frac:.4} (per-point kept min {kept_min} max {kept_max}) over {} points",
        idx.len()
    );
    assert!(frac < 0.95, "screen is vacuous at t={t:e}: kept fraction {frac:.4}");
}

/// Symmetric Jacobi eigensolver (small matrices only). Returns (eigenvalues, eigenvectors as columns).
fn jacobi_eigh(a: &Array2<f64>) -> (Vec<f64>, Array2<f64>) {
    let n = a.nrows();
    let mut m = a.clone();
    let mut v = Array2::<f64>::eye(n);
    for _sweep in 0..100 {
        let mut off = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                off += m[(i, j)] * m[(i, j)];
            }
        }
        if off < 1e-24 {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                if m[(p, q)].abs() < 1e-300 {
                    continue;
                }
                let theta = (m[(q, q)] - m[(p, p)]) / (2.0 * m[(p, q)]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (mkp, mkq) = (m[(k, p)], m[(k, q)]);
                    m[(k, p)] = c * mkp - s * mkq;
                    m[(k, q)] = s * mkp + c * mkq;
                }
                for k in 0..n {
                    let (mpk, mqk) = (m[(p, k)], m[(q, k)]);
                    m[(p, k)] = c * mpk - s * mqk;
                    m[(q, k)] = s * mpk + c * mqk;
                }
                for k in 0..n {
                    let (vkp, vkq) = (v[(k, p)], v[(k, q)]);
                    v[(k, p)] = c * vkp - s * vkq;
                    v[(k, q)] = s * vkp + c * vkq;
                }
            }
        }
    }
    ((0..n).map(|i| m[(i, i)]).collect(), v)
}

/// Closed-shell hcore-guess density `D = 2 C_occ C_occ^T` via Löwdin orthogonalization.
fn hcore_guess_density(prep: &PreparedBasis, nocc: usize) -> Array2<f64> {
    let s = oneelectron::overlap(prep);
    let h = oneelectron::hcore(prep);
    let (sv, su) = jacobi_eigh(&s);
    let n = s.nrows();
    let mut x = Array2::<f64>::zeros((n, n));
    for k in 0..n {
        let f = 1.0 / sv[k].sqrt();
        for i in 0..n {
            for j in 0..n {
                x[(i, j)] += su[(i, k)] * f * su[(j, k)];
            }
        }
    }
    let hp = x.dot(&h).dot(&x);
    let (ev, vecs) = jacobi_eigh(&hp);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| ev[a].partial_cmp(&ev[b]).unwrap());
    let mut c = Array2::<f64>::zeros((n, nocc));
    for (col, &k) in order.iter().take(nocc).enumerate() {
        let ck = x.dot(&vecs.column(k).to_owned());
        c.column_mut(col).assign(&ck);
    }
    2.0 * c.dot(&c.t())
}

/// COSX exchange `K_{mu,nu} = Σ_g w_g χ_mu(g) [A^g (D χ(g))]_nu`, symmetrized,
/// through `Md3c1e::for_each_pair` with the given screen. Returns `(K, kept, total)`.
fn cosx_k(kern: &Md3c1e, mol: &Molecule, bs: &BasisSet, pts: &[[f64; 3]], wts: &[f64], d: &Array2<f64>, bounds: Option<&PairBounds>, screen: CosxScreen) -> (Array2<f64>, usize, usize) {
    let nbf = kern.nbasis();
    let nsh = kern.nshells();
    let mut k = Array2::<f64>::zeros((nbf, nbf));
    let mut scr = kern.scratch();
    let mut kept = 0usize;
    let mut total = 0usize;
    const B: usize = 256;
    for (c0, chunk) in pts.chunks(B).enumerate() {
        let npts = chunk.len();
        let chi = eval_basis_on_points(mol, bs, chunk).expect("ao values");
        assert_eq!(chi.nrows(), nbf);
        let f = d.dot(&chi); // (nbf, npts)
        let mut y = Array2::<f64>::zeros((nbf, npts));
        let (kp, tot) = kern
            .for_each_pair(chunk, bounds, screen, &mut scr, |s1, s2, blk| {
                let (o1, n1) = (kern.shell_offset(s1), kern.shell_dim(s1));
                let (o2, n2) = (kern.shell_offset(s2), kern.shell_dim(s2));
                for i in 0..n1 {
                    for j in 0..n2 {
                        let row = &blk[(i * n2 + j) * npts..(i * n2 + j + 1) * npts];
                        for g in 0..npts {
                            y[(o1 + i, g)] += row[g] * f[(o2 + j, g)];
                        }
                        if s1 != s2 {
                            for g in 0..npts {
                                y[(o2 + j, g)] += row[g] * f[(o1 + i, g)];
                            }
                        }
                    }
                }
            })
            .expect("for_each_pair");
        kept += kp;
        total += tot;
        let _ = nsh;
        for g in 0..npts {
            let w = wts[c0 * B + g];
            for mu in 0..nbf {
                let xw = w * chi[(mu, g)];
                if xw == 0.0 {
                    continue;
                }
                for nu in 0..nbf {
                    k[(mu, nu)] += xw * y[(nu, g)];
                }
            }
        }
    }
    let ks = 0.5 * (&k + &k.t());
    (ks, kept, total)
}

/// Exact `K_{mu,nu} = Σ_{lam,sig} (mu lam | nu sig) D_{lam,sig}` from libint2 ERIs.
fn exact_k(prep: &PreparedBasis, d: &Array2<f64>) -> Array2<f64> {
    let nbf = prep.nbasis();
    let nsh = prep.nshells();
    let dims = prep.shell_dims();
    let offs = prep.shell_offsets();
    let mut eng = Engine::new_2e(Operator::coulomb(), prep, 1e-14).expect("eri engine");
    let mut k = Array2::<f64>::zeros((nbf, nbf));
    for s1 in 0..nsh {
        for s2 in 0..nsh {
            for s3 in 0..nsh {
                for s4 in 0..nsh {
                    let Some(q) = eng.compute_quartet(prep, s1, s2, s3, s4) else { continue };
                    let (n1, n2, n3, n4) = (dims[s1], dims[s2], dims[s3], dims[s4]);
                    for i in 0..n1 {
                        for j in 0..n2 {
                            for kk in 0..n3 {
                                for l in 0..n4 {
                                    let v = q[((i * n2 + j) * n3 + kk) * n4 + l];
                                    // (mu lam | nu sig): mu=i, lam=j, nu=kk, sig=l
                                    k[(offs[s1] + i, offs[s3] + kk)] += v * d[(offs[s2] + j, offs[s4] + l)];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    k
}

fn max_abs(a: &Array2<f64>) -> f64 {
    a.iter().fold(0.0_f64, |m, v| m.max(v.abs()))
}

/// The number that was 0.71.
#[test]
fn screened_k_matches_unscreened_k_below_grid_error_water_ccpvdz() {
    let mol = Molecule::load_xyz(&testdata("testdata/molecules/water.xyz")).expect("xyz");
    let bs = bundled("cc-pvdz").expect("basis");
    let prep = PreparedBasis::new(&mol, &bs).expect("prep");
    let kern = Md3c1e::new(&prep).expect("md3c1e");
    let bounds = PairBounds::build(&prep).expect("bounds");
    let d = hcore_guess_density(&prep, 5);
    let (pts, wts) = becke_grid(&mol, 50, 110);
    assert_eq!(pts.len(), 3 * 50 * 110);
    let wsum: f64 = wts.iter().sum();
    println!("[water/cc-pvdz] (50,110) grid: {} points, Σw = {wsum:.6}", pts.len());

    let (k_ref, kept0, total0) = cosx_k(&kern, &mol, &bs, &pts, &wts, &d, None, CosxScreen::none());
    assert_eq!(kept0, total0);
    let k_exact = exact_k(&prep, &d);
    let grid_err = max_abs(&(&k_ref - &k_exact));
    println!("[water/cc-pvdz] max|K| = {:.4e}; grid error max|K_cosx - K_exact| = {grid_err:.3e}", max_abs(&k_exact));
    assert!(grid_err < 1e-3, "COSX grid error {grid_err:.3e} is implausibly large — harness bug, not a screen question");

    for t in [1e-7, 1e-8] {
        let (k_scr, kept, total) = cosx_k(&kern, &mol, &bs, &pts, &wts, &d, Some(&bounds), CosxScreen::at(t));
        let err = max_abs(&(&k_scr - &k_ref));
        println!(
            "[water/cc-pvdz] t={t:e}: max|K_scr - K_unscr| = {err:.3e}  (batch-level kept pairs {kept}/{total} = {:.3})",
            kept as f64 / total as f64
        );
        assert!(err < 1e-6, "t={t:e}: screened K differs from unscreened by {err:.3e} (grid error is {grid_err:.3e}); the screen is dropping real pairs");
    }
}
