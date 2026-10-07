//! Task D2: the f32 block path of the md3c1e kernel (`pair_block_f32`) against
//! the f64 block (`pair_block`), per angular-momentum class, water/def2-QZVP
//! (O carries up to g, H up to f: every class `(l_hi, l_lo)` with `l <= 4`
//! that the basis has is visited and the test fails if a class with `l_hi = 4`
//! is never reached).
//!
//! Error metric per class `(l_hi, l_lo)`: `max|dA| / max|A|` over every shell
//! pair of the class and every probe point, in units of `u32 = 2^-24`.
//!
//! # Bounds
//!
//! 1. DERIVED (rigorous, worst case): `Md3c1e::pair_block_f32_bound` (module
//!    doc of `md3c1e/f32_block.rs`: Higham chain depth `D = nnz + 16 l_tot +
//!    49`, `S_pp` from the absolute-value shadow recursion, sum term
//!    `gamma_{n_pp}(u64) sum|A_pp|` for `F64`, `2 u32 sum|A_pp|` for
//!    `CompensatedF32`, `gamma_{n_pp-1}(u32) sum|A_pp|` for `F32`). Asserted
//!    ELEMENT-WISE (`|dA| <= bound`) for both clean variants.
//! 2. DISCRIMINATING (measured, two-sided): the derived bound is a worst case
//!    that does not see the three defects (they sit inside it), so the bar
//!    that separates them is the geometric midpoint of the two measured
//!    sides of each class; the table below is that measurement.
//!
//! MEASURED SIDES (water/def2-QZVP, 48 probes, the `measure_table` printer;
//! `0.6 u32` mean truncation shift is `0.5 ulp` relative):
//!
//! ```text
//!  class blocks | max|dA|/max|A| F64=Comp | max|dA|/derived bound | kappa max |
//!               |  accumulation: max|B_plainF32 - B_F64sum|/max|A| | truncE: slope of
//!               |  (B_truncE - B_nearest) on A         (all in units of u32 = 2^-24)
//!  (0,0)  120 |  1.862 | 6.3e-2 | 1.0e0 | 1.380 | -0.573
//!  (1,0)  150 |  1.927 | 9.4e-2 | 7.6e2 | 1.188 | -0.601
//!  (1,1)   55 |  1.296 | 9.5e-2 | 3.0e2 | 0.750 | -0.777
//!  (2,0)  105 |  1.926 | 9.2e-2 | 9.1e3 | 1.260 | -0.798
//!  (2,1)   70 |  2.813 | 9.8e-2 | 1.7e3 | 0.381 | -0.713
//!  (2,2)   28 |  2.947 | 6.4e-2 | 5.5e3 | 0     | -0.630
//!  (3,0)   60 |  2.329 | 1.25e-1| 5.5e3 | 0.593 | -0.765
//!  (3,1)   40 |  8.215 | 7.1e-2 | 8.2e3 | 2.470 | -0.851
//!  (3,2)   28 |  6.448 | 4.9e-2 | 6.2e3 | 0     | -0.620
//!  (3,3)   10 |  5.009 | 4.5e-2 | 5.9e3 | 0     | -0.584
//!  (4,0)   15 | 17.05  | 2.9e-2 | 4.1e4 | 1.831 | -0.786
//!  (4,1)   10 | 12.77  | 3.4e-2 | 1.7e4 | 1.303 | -0.537
//!  (4,2)    7 | 12.36  | 3.1e-2 | 4.1e4 | 0     | -1.061
//!  (4,3)    4 |  8.419 | 2.4e-2 | 5.4e3 | 0     | -0.132
//!  (4,4)    1 | 10.14  | 1.7e-2 | 1.1e4 | 0     | -1.014
//! ```
//!
//! `CompensatedF32` is bit-equal to `F64` on the error column and differs
//! from it by at most 8.9e-8 u32 (accumulation column, `Comp`); the accumulation
//! column is 0 exactly in classes whose shell pairs have a single primitive
//! pair (nothing to accumulate), where the plain-f32 mutant is not a defect.
//! The spike's per-class table (0.7-18 u32 for (0,0)..(4,4)) is reproduced:
//! 1.3-17.1 u32 here.
//!
//! The three mutants are NOT all outside the DERIVED bound: it is a worst
//! case (D = 50..180 roundings) and the measured errors sit 8-60x inside it
//! (max ratio 0.125), so plain-f32 accumulation and truncation, whose errors
//! are `O(u32)` like the clean path's, are inside it; only the dropped
//! primitive pair (error ~ 1.3e7 u32 = O(1) relative) is outside. They are
//! separated by three bars, each the geometric midpoint of its two measured
//! sides (clean side = the `CompensatedF32` path, which differs from the
//! `F64` sum only by the accumulation):
//!
//! | bar | quantity | clean side (max) | defect side (min) | bar | margins |
//! |---|---|---|---|---|---|
//! | `BAR_ACC` | `max|B_v - B_F64sum|/max|A|` (u32) | 8.906e-8 (Comp) | 0.3807 (plain F32, classes with > 1 primitive pair) | 1.84e-4 | 2.1e3 each |
//! | `BAR_SLOPE` | `|slope of (B - B_F64sum) on A|` (u32) | 7.396e-10 (Comp) | 0.1321 (truncE) | 3.13e-5 | 4.2e4 / 4.2e3 |
//! | `BAR_GROSS` | `max|dA|/max|A|` (u32) | 17.05 (F64/Comp) | 1.323e7 (dropLast) | 1.50e4 | 8.8e2 each |
//!
//! (round-to-nearest coefficient noise has no mean; truncation shrinks every
//! coefficient by `0.5 u32` on average, so the slope of its effect on `A` is
//! the isolating statistic. The slope of the clean error itself on `A` is NOT
//! usable: it reaches +2.2 u32 on single-block classes.)
//!
//! Mutants (each is a real defect injected into the production path, not a
//! test-side re-implementation): plain f32 accumulation across primitive pairs
//! (`PrimPairSum::F32`, the spike's unsafe case); E-coefficients truncated
//! (round toward zero) instead of rounded to nearest (`F32Fault::TruncateE`);
//! the last primitive pair dropped (`F32Fault::DropLastPrimPair`, gross).

use ferric_core::basis::bundled;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::md3c1e::{BlockPrecision, F32Fault, Md3c1e, PairRoute, PrimPairSum, U32};

const BAR_ACC: f64 = 1.84e-4;
const BAR_SLOPE: f64 = 3.13e-5;
const BAR_GROSS: f64 = 1.50e4;

const WATER: &str = "3\nwater\nO 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692\n";

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

/// 48 probes: 16 within 1 bohr of an atom (the Becke-grid bulk), 16 within 4
/// bohr, 16 in an 8-bohr box, fixed seed.
fn probes(mol: &Molecule) -> Vec<[f64; 3]> {
    let atoms: Vec<[f64; 3]> = mol.atoms.iter().map(|a| [a.x, a.y, a.zpos]).collect();
    let mut rng = Lcg(0x05ee_df32);
    let mut pts = Vec::new();
    for (i, radius) in [1.0, 4.0, 8.0].into_iter().enumerate() {
        for k in 0..16 {
            let c = atoms[(i * 16 + k) % atoms.len()];
            let mut p = c;
            for d in p.iter_mut() {
                *d += radius * (2.0 * rng.next() - 1.0);
            }
            pts.push(p);
        }
    }
    pts
}

struct Setup {
    kern: Md3c1e,
    pts: Vec<[f64; 3]>,
}

fn setup() -> Setup {
    let mol = Molecule::parse_xyz(WATER, 0, 1).unwrap();
    let bs = bundled("def2-qzvp").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let kern = Md3c1e::new(&prep).unwrap();
    Setup {
        kern,
        pts: probes(&mol),
    }
}

/// Per-class results of one (variant, fault) configuration.
#[derive(Clone, Copy, Default)]
struct ClassStat {
    max_abs_err: f64,
    max_abs_a: f64,
    /// max over elements of |dA| / bound.
    max_ratio_to_bound: f64,
    /// max |B - B_F64variant| (the f32 path with the f64 primitive-pair sum
    /// and no fault): isolates what the accumulation / the fault changes.
    max_abs_diff_vs_f64sum: f64,
    /// sum (B - B_F64sum) A: slope numerator of the isolated change.
    dot_ca: f64,
    /// sum A^2 over the class.
    sum_a2: f64,
    blocks: usize,
}

impl ClassStat {
    /// Relative error in units of u32.
    fn rel_u32(&self) -> f64 {
        self.max_abs_err / self.max_abs_a / U32
    }

    /// `max|B - B_F64sum| / max|A|` in u32.
    fn diff_u32(&self) -> f64 {
        self.max_abs_diff_vs_f64sum / self.max_abs_a / U32
    }

    /// Regression slope of `(B - A)` on `A`, in u32: a systematic scale error
    /// (round-to-nearest noise averages out, truncation does not).
    fn diff_bias_u32(&self) -> f64 {
        self.dot_ca / self.sum_a2 / U32
    }
}

type Table = [[ClassStat; 5]; 5];

fn measure(s: &Setup, sum: PrimPairSum, fault: F32Fault) -> Table {
    let mut t: Table = [[ClassStat::default(); 5]; 5];
    let mut scr = s.kern.scratch();
    let n = s.pts.len();
    let nsh = s.kern.nshells();
    for s1 in 0..nsh {
        for s2 in 0..=s1 {
            let need = s.kern.shell_dim(s1) * s.kern.shell_dim(s2) * n;
            let (mut a, mut b, mut c, mut bound, mut sabs) = (
                vec![0.0; need],
                vec![0.0; need],
                vec![0.0; need],
                vec![0.0; need],
                vec![0.0; need],
            );
            s.kern.pair_block(s1, s2, &s.pts, &mut scr, &mut a).unwrap();
            s.kern
                .pair_block_f32_with(s1, s2, &s.pts, &mut scr, sum, fault, &mut b)
                .unwrap();
            s.kern
                .pair_block_f32(s1, s2, &s.pts, &mut scr, PrimPairSum::F64, &mut c)
                .unwrap();
            s.kern
                .pair_block_f32_bound(s1, s2, &s.pts, &mut scr, sum, &mut bound, &mut sabs)
                .unwrap();
            let (l1, l2) = (s.kern.shell_l(s1), s.kern.shell_l(s2));
            let st = &mut t[l1.max(l2)][l1.min(l2)];
            st.blocks += 1;
            for i in 0..need {
                let e = (a[i] - b[i]).abs();
                st.max_abs_err = st.max_abs_err.max(e);
                st.max_abs_a = st.max_abs_a.max(a[i].abs());
                st.max_abs_diff_vs_f64sum = st.max_abs_diff_vs_f64sum.max((b[i] - c[i]).abs());
                st.dot_ca += (b[i] - c[i]) * a[i];
                st.sum_a2 += a[i] * a[i];
                if bound[i] > 0.0 {
                    st.max_ratio_to_bound = st.max_ratio_to_bound.max(e / bound[i]);
                }
            }
        }
    }
    t
}

fn kappas(s: &Setup) -> [[f64; 5]; 5] {
    let mut k = [[0.0_f64; 5]; 5];
    let mut scr = s.kern.scratch();
    let n = s.pts.len();
    for s1 in 0..s.kern.nshells() {
        for s2 in 0..=s1 {
            let need = s.kern.shell_dim(s1) * s.kern.shell_dim(s2) * n;
            let (mut a, mut bound, mut sabs) = (vec![0.0; need], vec![0.0; need], vec![0.0; need]);
            s.kern.pair_block(s1, s2, &s.pts, &mut scr, &mut a).unwrap();
            s.kern
                .pair_block_f32_bound(
                    s1,
                    s2,
                    &s.pts,
                    &mut scr,
                    PrimPairSum::F64,
                    &mut bound,
                    &mut sabs,
                )
                .unwrap();
            let amax = a.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            let (l1, l2) = (s.kern.shell_l(s1), s.kern.shell_l(s2));
            let c = &mut k[l1.max(l2)][l1.min(l2)];
            for i in 0..need {
                if a[i].abs() > 1e-3 * amax {
                    *c = c.max(sabs[i] / a[i].abs());
                }
            }
        }
    }
    k
}

#[test]
#[ignore = "measurement printer; run with --nocapture to read the table"]
fn measure_table() {
    let s = setup();
    let cfgs: [(&str, PrimPairSum, F32Fault); 5] = [
        ("F64", PrimPairSum::F64, F32Fault::None),
        ("Comp", PrimPairSum::CompensatedF32, F32Fault::None),
        ("plainF32", PrimPairSum::F32, F32Fault::None),
        ("truncE", PrimPairSum::F64, F32Fault::TruncateE),
        ("dropLast", PrimPairSum::F64, F32Fault::DropLastPrimPair),
    ];
    let tabs: Vec<Table> = cfgs.iter().map(|c| measure(&s, c.1, c.2)).collect();
    let kap = kappas(&s);
    eprintln!("class blocks | max|dA|/max|A| in u32: F64 Comp plainF32 truncE dropLast | max|B-B_F64sum|/max|A| in u32: Comp plainF32 truncE | slope of (B-B_F64sum) on A in u32: Comp plainF32 truncE | max|dA|/bound: F64 Comp plain | kappa");
    for hi in 0..5 {
        for lo in 0..=hi {
            let b = tabs[0][hi][lo].blocks;
            if b == 0 {
                continue;
            }
            let f = |g: fn(&ClassStat) -> f64, ids: &[usize]| -> String {
                ids.iter()
                    .map(|&k| format!("{:9.3e}", g(&tabs[k][hi][lo])))
                    .collect::<Vec<_>>()
                    .join(" ")
            };
            eprintln!(
                "({hi},{lo}) {b:4} | {} | {} | {} | {} | {:.1}",
                f(ClassStat::rel_u32, &[0, 1, 2, 3, 4]),
                f(ClassStat::diff_u32, &[1, 2, 3]),
                f(ClassStat::diff_bias_u32, &[1, 2, 3]),
                f(|c| c.max_ratio_to_bound, &[0, 1, 2]),
                kap[hi][lo]
            );
        }
    }
}

fn each_class(t: &Table, mut f: impl FnMut(usize, usize, &ClassStat)) {
    for hi in 0..5 {
        for lo in 0..=hi {
            if t[hi][lo].blocks > 0 {
                f(hi, lo, &t[hi][lo]);
            }
        }
    }
}

#[test]
fn every_class_to_g_is_reached() {
    let s = setup();
    let t = measure(&s, PrimPairSum::F64, F32Fault::None);
    for hi in 0..5 {
        for lo in 0..=hi {
            assert!(t[hi][lo].blocks > 0, "class ({hi},{lo}) never visited");
        }
    }
}

/// Both clean variants inside the DERIVED bound element-wise, and inside
/// `BAR_GROSS`; no f32-range fallback on this basis.
#[test]
fn clean_variants_are_inside_the_derived_bound() {
    let s = setup();
    for sum in [PrimPairSum::F64, PrimPairSum::CompensatedF32] {
        let t = measure(&s, sum, F32Fault::None);
        each_class(&t, |hi, lo, c| {
            assert!(
                c.max_ratio_to_bound <= 1.0,
                "{} ({hi},{lo}): |dA| / bound = {}",
                sum.as_str(),
                c.max_ratio_to_bound
            );
            assert!(c.rel_u32() < BAR_GROSS, "{} ({hi},{lo})", sum.as_str());
        });
    }
    let mut scr = s.kern.scratch();
    let n = s.pts.len();
    for s1 in 0..s.kern.nshells() {
        for s2 in 0..=s1 {
            let mut out = vec![0.0; s.kern.shell_dim(s1) * s.kern.shell_dim(s2) * n];
            let p = s
                .kern
                .pair_block_f32(s1, s2, &s.pts, &mut scr, PrimPairSum::F64, &mut out)
                .unwrap();
            assert_eq!(p, BlockPrecision::F32, "pair ({s1},{s2}) fell back");
        }
    }
}

/// Mutant 1: plain f32 accumulation across primitive pairs. Per class its
/// isolated accumulation error is exactly 0 (single primitive pair: nothing
/// to accumulate) or above `BAR_ACC`; the compensated variant is below it in
/// every class; and at least one class is a real defect.
#[test]
fn mutant_plain_f32_accumulation_is_outside_and_compensated_inside() {
    let s = setup();
    let plain = measure(&s, PrimPairSum::F32, F32Fault::None);
    let comp = measure(&s, PrimPairSum::CompensatedF32, F32Fault::None);
    let mut defective = 0;
    each_class(&plain, |hi, lo, c| {
        let d = c.diff_u32();
        assert!(d == 0.0 || d > BAR_ACC, "plain ({hi},{lo}): {d}");
        defective += (d > BAR_ACC) as usize;
    });
    assert!(defective >= 8, "only {defective} classes accumulate");
    each_class(&comp, |hi, lo, c| {
        assert!(c.diff_u32() < BAR_ACC, "Comp ({hi},{lo}): {}", c.diff_u32());
    });
}

/// Mutant 2: truncated E coefficients, in every class; clean variants below.
#[test]
fn mutant_truncated_coefficients_are_outside_in_every_class() {
    let s = setup();
    let trunc = measure(&s, PrimPairSum::F64, F32Fault::TruncateE);
    let comp = measure(&s, PrimPairSum::CompensatedF32, F32Fault::None);
    each_class(&trunc, |hi, lo, c| {
        let b = c.diff_bias_u32();
        assert!(b < -BAR_SLOPE, "trunc ({hi},{lo}): slope {b}");
    });
    each_class(&comp, |hi, lo, c| {
        assert!(c.diff_bias_u32().abs() < BAR_SLOPE, "Comp ({hi},{lo})");
    });
}

/// Mutant 3: the last primitive pair dropped; outside the DERIVED bound and
/// `BAR_GROSS` in every class.
#[test]
fn mutant_dropped_primitive_pair_is_outside_in_every_class() {
    let s = setup();
    let t = measure(&s, PrimPairSum::F64, F32Fault::DropLastPrimPair);
    each_class(&t, |hi, lo, c| {
        assert!(c.rel_u32() > BAR_GROSS, "drop ({hi},{lo}): {}", c.rel_u32());
        assert!(c.max_ratio_to_bound > 1.0, "drop ({hi},{lo})");
    });
}

/// `for_each_pair_routed`: F64-routed blocks are bit-equal to `pair_block`,
/// F32-routed to `pair_block_f32`, Drop skips, and the counts add up.
#[test]
fn routed_sweep_matches_the_per_block_calls() {
    let s = setup();
    let n = s.pts.len();
    let mut scr = s.kern.scratch();
    let route = |s1: usize, s2: usize| match (s1 + 2 * s2) % 3 {
        0 => PairRoute::Drop,
        1 => PairRoute::F64,
        _ => PairRoute::F32,
    };
    let mut seen = Vec::new();
    let counts = s
        .kern
        .for_each_pair_routed(
            &s.pts,
            route,
            PrimPairSum::CompensatedF32,
            &mut scr,
            |a, b, blk| {
                seen.push((a, b, blk.to_vec()));
            },
        )
        .unwrap();
    let mut scr2 = s.kern.scratch();
    assert_eq!(counts.kept, seen.len());
    assert_eq!(counts.f32_fallbacks, 0);
    let nsh = s.kern.nshells();
    assert_eq!(counts.total, nsh * (nsh + 1) / 2);
    let mut n32 = 0;
    for (a, b, blk) in &seen {
        let mut r = vec![0.0; blk.len()];
        match route(*a, *b) {
            PairRoute::F64 => s
                .kern
                .pair_block(*a, *b, &s.pts, &mut scr2, &mut r)
                .unwrap(),
            _ => {
                n32 += 1;
                let p = s
                    .kern
                    .pair_block_f32(
                        *a,
                        *b,
                        &s.pts,
                        &mut scr2,
                        PrimPairSum::CompensatedF32,
                        &mut r,
                    )
                    .unwrap();
                assert_eq!(p, BlockPrecision::F32);
            }
        }
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&r), bits(blk), "pair ({a},{b})");
        assert_eq!(blk.len(), s.kern.shell_dim(*a) * s.kern.shell_dim(*b) * n);
    }
    assert_eq!(counts.f32_blocks, n32);
}
