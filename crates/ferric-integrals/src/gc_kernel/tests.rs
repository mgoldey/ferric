//! The shared-primitive kernel against libint, integral by integral.

use super::*;
use crate::contraction_rotation::RotationLevel;
use crate::engine::Engine;
use crate::engine_pool::ERI_PRECISION;
use crate::operator::Operator;
use ferric_core::basis::{self, BasisSet, Shell};
use ferric_core::mol::Molecule;
use std::collections::HashMap;

/// Largest |kernel − libint| over every covered column quartet of `kb`, and
/// the largest |libint| integral, per class (`0`: (ss|ss), `1`: one p).
/// Every shell order is visited, so the p sits in each of the four
/// positions.
///
/// `arbitrated`: the shell quartets (any permutation) where libint is known
/// to be the less accurate side ([`arbitrate`]); empty for every fixture but
/// HCl cc-pVTZ.
fn compare(
    prep: &PreparedBasis,
    kb: &GcKernelBasis,
    arbitrated: &[[usize; 4]],
) -> [(f64, f64, usize); 2] {
    let mut eng = Engine::new_2e(Operator::coulomb(), prep, ERI_PRECISION).unwrap();
    let nsh = prep.nshells();
    let mut scratch = KernelScratch::default();
    let mut block = Vec::new();
    let mut col = [0.0; 3];
    let mut out = [(0.0f64, 0.0f64, 0usize); 2];
    let sp: Vec<usize> = (0..nsh).filter(|&s| kb.shell_l(s) <= 1).collect();
    for &s1 in &sp {
        for &s2 in &sp {
            for &s3 in &sp {
                for &s4 in &sp {
                    let l = [s1, s2, s3, s4].map(|s| kb.shell_l(s)).iter().sum::<i32>();
                    if l > 1 {
                        continue;
                    }
                    let (PairKind::Data(b), PairKind::Data(k)) =
                        (kb.pair_of(s1, s2), kb.pair_of(s3, s4))
                    else {
                        // An empty pair: libint must agree it is zero.
                        let lib = eng.compute_quartet(prep, s1, s2, s3, s4);
                        let mx = lib.map_or(0.0, |q| q.iter().fold(0.0f64, |m, x| m.max(x.abs())));
                        assert!(mx < 1e-14, "empty pair but libint {mx:e}");
                        continue;
                    };
                    block.resize(kb.block_len(b, k), 0.0);
                    kb.eval(b, k, &mut block, &mut scratch);
                    let n = kb.gather(&block, b, k, [s1, s2, s3, s4], &mut col);
                    assert!(
                        col[..n].iter().all(|x| x.is_finite()),
                        "({s1} {s2}|{s3} {s4}): non-finite kernel value {:?}",
                        &col[..n]
                    );
                    let lib = eng.compute_quartet(prep, s1, s2, s3, s4);
                    let c = &mut out[l as usize];
                    c.2 += 1;
                    match lib {
                        Some(q) => {
                            assert_eq!(q.len(), n);
                            assert!(
                                q.iter().all(|y| y.is_finite()),
                                "({s1} {s2}|{s3} {s4}): non-finite libint value {q:?}"
                            );
                            for (x, y) in col[..n].iter().zip(q) {
                                let err = (x - y).abs();
                                c.0 = c.0.max(err);
                                c.1 = c.1.max(y.abs());
                                // Per integral: relative where it is not
                                // tiny, absolute where it is.
                                if err > 1e-13 * y.abs() + 1e-15 {
                                    let mut key = [s1, s2, s3, s4];
                                    key.sort_unstable();
                                    assert!(
                                        arbitrated.contains(&key),
                                        "({s1} {s2}|{s3} {s4}): kernel {x:e} vs libint {y:e} \
                                         beyond 1e-13·|I| + 1e-15"
                                    );
                                    arbitrate(prep, [s1, s2, s3, s4], *x, *y);
                                }
                            }
                        }
                        None => {
                            // libint screened the quartet to zero: the
                            // kernel's values must be zero to the same
                            // per-integral bar.
                            for x in &col[..n] {
                                c.0 = c.0.max(x.abs());
                                assert!(
                                    x.abs() <= 1e-15,
                                    "({s1} {s2}|{s3} {s4}): libint screened it, kernel {x:e}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

/// Where the kernel and libint differ by more than the per-integral bar
/// `1e-13·|I| + 1e-15`, decide who is right against an independent
/// reference. Only the quartets the caller lists for its fixture reach
/// here, and each must be a one-centre `(ss|ss)` quartet: there the
/// integral has the closed form `Σ c c c c · 2π^{5/2} / (p q √(p+q))`,
/// summed here with Neumaier compensation (relative error ~1e-15), and the
/// kernel must match it to 5e-14 (these columns mix signs, so the sum
/// cancels) and be closer than libint by at least 2×. Measured: two such
/// quartets of Cl cc-pVTZ s columns, where libint is 1.5e-13 and 5e-13
/// relative off the reference and the kernel 1.1e-15 and 2.0e-14 (an
/// mpmath evaluation at 50 digits puts the second at 2.4e-14 for the
/// kernel and 5.0e-13 for libint). Anything else fails.
fn arbitrate(prep: &PreparedBasis, s: [usize; 4], kernel: f64, libint: f64) {
    let ls = prep.located_shells();
    let one_centre_s = s
        .iter()
        .all(|&x| ls[x].l == 0 && ls[x].center == ls[s[0]].center);
    assert!(
        one_centre_s,
        "{s:?}: kernel {kernel:e} vs libint {libint:e} beyond 1e-13·|I| + 1e-15, and no \
         reference to arbitrate (not a one-centre (ss|ss))"
    );
    let prims: Vec<Vec<(f64, f64)>> = s
        .iter()
        .map(|&x| {
            let (e, c): (Vec<f64>, Vec<f64>) = ls[x]
                .exponents
                .iter()
                .zip(ls[x].coefficients)
                .filter(|(_, c)| **c != 0.0)
                .map(|(e, c)| (*e, *c))
                .unzip();
            let cn = libint_coefficients(0, &e, &c);
            e.into_iter().zip(cn).collect()
        })
        .collect();
    let pre = 2.0 * std::f64::consts::PI.powf(2.5);
    let (mut sum, mut comp) = (0.0f64, 0.0f64);
    for &(a, ca) in &prims[0] {
        for &(b, cb) in &prims[1] {
            let p = a + b;
            for &(c, cc) in &prims[2] {
                for &(d, cd) in &prims[3] {
                    let q = c + d;
                    let t = ca * cb * cc * cd * pre / (p * q * (p + q).sqrt());
                    let u = sum + t;
                    comp += if sum.abs() >= t.abs() {
                        (sum - u) + t
                    } else {
                        (t - u) + sum
                    };
                    sum = u;
                }
            }
        }
    }
    let reference = sum + comp;
    let rel = |x: f64| (x - reference).abs() / reference.abs();
    eprintln!(
        "{s:?}: one-centre (ss|ss) reference {reference:e}: kernel rel {:.1e}, libint rel {:.1e}",
        rel(kernel),
        rel(libint)
    );
    assert!(
        rel(kernel) <= 5e-14 && rel(libint) > 2.0 * rel(kernel),
        "{s:?}: the kernel ({kernel:e}) is not the one matching the reference {reference:e} \
         (libint {libint:e})"
    );
}

/// Partition of the shells into (atom, l, pure) sets (the kernel splits
/// them further into exponent-sharing runs).
fn atom_l_parts(prep: &PreparedBasis) -> Vec<Vec<usize>> {
    let ls = prep.located_shells();
    let s2a = prep.shell_to_atom();
    let mut map: Vec<((usize, i32, bool), Vec<usize>)> = Vec::new();
    for s in 0..ls.len() {
        let k = (s2a[s], ls[s].l, ls[s].pure);
        match map.iter_mut().find(|(x, _)| *x == k) {
            Some((_, v)) => v.push(s),
            None => map.push((k, vec![s])),
        }
    }
    map.into_iter().map(|(_, v)| v).collect()
}

fn check(prep: &PreparedBasis, kb: &GcKernelBasis, what: &str) {
    check_arbitrated(prep, kb, what, &[]);
}

/// [`check`] with the shell quartets (sorted shell indices) where libint is
/// known to miss the per-integral bar ([`arbitrate`]).
fn check_arbitrated(
    prep: &PreparedBasis,
    kb: &GcKernelBasis,
    what: &str,
    arbitrated: &[[usize; 4]],
) {
    let r = compare(prep, kb, arbitrated);
    for (class, (err, scale, n)) in r.iter().enumerate() {
        assert!(*n > 0, "{what}: class {class} has no quartets");
        assert!(
            *err <= 1e-14 * scale,
            "{what}: class {class}: max|err| {err:e} vs max|I| {scale:e} over {n} quartets"
        );
    }
}

fn load(name: &str) -> Molecule {
    Molecule::load_xyz(&format!("../../testdata/molecules/{name}.xyz")).unwrap()
}

/// Rotated cc-pVDZ, aug-cc-pVDZ and cc-pVTZ shells (the groups the direct
/// builder uses), and the unrotated ones grouped by atom and l (3-column
/// groups: cc-pVDZ C s is three columns over nine primitives).
#[test]
fn kernel_matches_libint_on_correlation_consistent_shells() {
    for (mol, basis) in [
        ("ch3oh", "cc-pvdz"),
        ("water", "aug-cc-pvdz"),
        ("water", "cc-pvtz"),
        // A heavier all-electron atom: Cl cc-pVTZ.
        ("hcl", "cc-pvtz"),
    ] {
        let m = load(mol);
        let parent = PreparedBasis::new(&m, &basis::bundled(basis).unwrap()).unwrap();
        let rot = RotatedBasis::detect_prepared(&parent, RotationLevel::Ogc)
            .unwrap()
            .expect("cc basis rotates");
        let kb = GcKernelBasis::for_rotated(&rot, ERI_PRECISION).unwrap();
        assert!(kb.groups().iter().any(|g| g.ncols() > 1));
        // The two Cl cc-pVTZ s-column quartets where libint misses the
        // per-integral bar ([`arbitrate`]); none elsewhere.
        let hcl = mol == "hcl" && basis == "cc-pvtz";
        let (rot_known, unrot_known): (&[[usize; 4]], &[[usize; 4]]) = if hcl {
            (&[[0, 1, 1, 2]], &[[0, 1, 1, 3]])
        } else {
            (&[], &[])
        };
        check_arbitrated(&rot.prep, &kb, &format!("{mol} {basis} rotated"), rot_known);
        let kb =
            GcKernelBasis::from_groups(&parent, &atom_l_parts(&parent), ERI_PRECISION).unwrap();
        assert!(kb.groups().iter().any(|g| g.ncols() >= 3));
        check_arbitrated(
            &parent,
            &kb,
            &format!("{mol} {basis} unrotated"),
            unrot_known,
        );
    }
}

/// Cartesian p (libint order x, y, z), which is what the basis loaders
/// make, and pure p (libint order y, z, x), which no loader makes today:
/// cc-pVDZ with every p shell forced to each, rotated and unrotated groups.
#[test]
fn kernel_matches_libint_with_cartesian_and_pure_p() {
    let m = load("water");
    let loaded = basis::bundled("cc-pvdz").unwrap();
    assert!(
        loaded
            .shells
            .values()
            .flatten()
            .all(|sh| sh.l != 1 || !sh.pure),
        "the loader now makes pure p: drop this note in gc_kernel.rs"
    );
    for pure in [false, true] {
        let mut bs = loaded.clone();
        for shells in bs.shells.values_mut() {
            for sh in shells.iter_mut() {
                if sh.l == 1 {
                    sh.pure = pure;
                }
            }
        }
        let prep = PreparedBasis::new(&m, &bs).unwrap();
        let kb = GcKernelBasis::from_groups(&prep, &atom_l_parts(&prep), ERI_PRECISION).unwrap();
        assert!(kb.groups().iter().any(|g| g.l == 1 && g.pure == pure));
        check(&prep, &kb, &format!("water cc-pvdz p pure={pure}"));
        let rot = RotatedBasis::detect_prepared(&prep, RotationLevel::Ogc)
            .unwrap()
            .expect("cc basis rotates");
        let kb = GcKernelBasis::for_rotated(&rot, ERI_PRECISION).unwrap();
        check(
            &rot.prep,
            &kb,
            &format!("water cc-pvdz p pure={pure} rotated"),
        );
    }
}

/// A column listing one exponent twice (two primitives with the same
/// exponent) contributes both coefficients to that union primitive, as
/// libint sums them.
#[test]
fn duplicated_exponent_in_a_column_is_summed() {
    let mut bs = BasisSet {
        name: "gc-kernel-dup".into(),
        shells: HashMap::new(),
        ecps: HashMap::new(),
    };
    let s = |e: Vec<f64>, c: Vec<f64>| Shell {
        l: 0,
        pure: true,
        exponents: e,
        coefficients: c,
    };
    // H: s with 1.2 twice; a second column sharing 1.2 (one group of two);
    // and a p.
    bs.shells.insert(
        1,
        vec![
            s(vec![5.0, 1.2, 1.2, 0.3], vec![0.2, 0.3, 0.25, 0.5]),
            s(vec![1.2, 0.3], vec![-0.4, 1.0]),
            Shell {
                l: 1,
                pure: false,
                exponents: vec![0.8, 0.8],
                coefficients: vec![0.6, 0.5],
            },
        ],
    );
    let m = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.9\n", 0, 1).unwrap();
    let prep = PreparedBasis::new(&m, &bs).unwrap();
    let kb = GcKernelBasis::from_groups(&prep, &atom_l_parts(&prep), ERI_PRECISION).unwrap();
    assert!(kb.groups().iter().any(|g| g.ncols() == 2));
    check(&prep, &kb, "duplicated exponent");
}

/// One-primitive shells placed so the Boys argument `T = α R²` (`α = 1`,
/// one s primitive per centre pair) hits 0, every edge kind of the table
/// (grid points, midpoints where the nearest point switches) and both sides
/// of the asymptotic switch at 40; tight × diffuse pairs on top.
#[test]
fn kernel_matches_libint_at_boys_edges_and_extreme_exponents() {
    let shell = |l: i32, e: f64| Shell {
        l,
        pure: true,
        exponents: vec![e],
        coefficients: vec![1.0],
    };
    let mut bs = BasisSet {
        name: "gc-kernel-test".into(),
        shells: HashMap::new(),
        ecps: HashMap::new(),
    };
    // H: s(1.0), p(1.0); He: s(1.0), s(1e4)+s(0.01) contracted, p(0.03).
    bs.shells.insert(1, vec![shell(0, 1.0), shell(1, 1.0)]);
    bs.shells.insert(
        2,
        vec![
            shell(0, 1.0),
            Shell {
                l: 0,
                pure: true,
                exponents: vec![1e4, 0.01],
                coefficients: vec![0.3, 0.8],
            },
            shell(1, 0.03),
        ],
    );
    for t in [
        0.0, 1e-9, 0.05, 0.05000001, 0.1, 0.15, 3.25, 17.35, 39.9, 39.95, 39.99999, 40.0, 40.00001,
        41.0, 60.0,
    ] {
        let r = f64::sqrt(t);
        // `Molecule` coordinates are Angstrom; T = α R² in Bohr.
        let r_ang = r / ferric_core::units::ANGSTROM_TO_BOHR;
        let xyz = format!("2\nedge\nH 0 0 0\nHe 0 0 {r_ang:.17}\n");
        let m = Molecule::parse_xyz(&xyz, 0, 2).unwrap();
        let prep = PreparedBasis::new(&m, &bs).unwrap();
        let parts: Vec<Vec<usize>> = (0..prep.nshells()).map(|s| vec![s]).collect();
        let kb = GcKernelBasis::from_groups(&prep, &parts, ERI_PRECISION).unwrap();
        check(&prep, &kb, &format!("edge T = {t}"));
    }
}

/// The kernel's Boys table against libint's `FmEval_Chebyshev7`, on and
/// between the grid points, at the nearest-point switch, around the
/// asymptotic switch and far out.
#[test]
fn boys_table_matches_libint() {
    let mut ts: Vec<f64> = vec![
        0.0,
        1e-12,
        39.95,
        39.99999999,
        40.0,
        40.0000001,
        55.0,
        117.0,
    ];
    for k in 0..400 {
        let t0 = k as f64 * 0.1;
        ts.extend([
            t0,
            t0 + 0.0499999999,
            t0 + 0.05,
            t0 + 0.0500000001,
            t0 + 0.0731,
        ]);
    }
    for &t in &ts {
        let lib = crate::engine::boys_fm_chebyshev7(t, 1).unwrap();
        let ours = boys_for_tests(t, 1);
        for m in 0..2 {
            let rel = (ours[m] - lib[m]).abs() / lib[m].abs();
            assert!(
                rel < 5e-15,
                "F_{m}({t}): {} vs libint {} (rel {rel:e})",
                ours[m],
                lib[m]
            );
        }
    }
}

/// A partition that mixes atoms or l, misses a shell or repeats one is
/// refused.
#[test]
fn bad_partitions_are_refused() {
    let m = load("water");
    let prep = PreparedBasis::new(&m, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let mut parts = atom_l_parts(&prep);
    assert!(GcKernelBasis::from_groups(&prep, &parts, ERI_PRECISION).is_ok());
    let last = parts.pop().unwrap();
    assert!(GcKernelBasis::from_groups(&prep, &parts, ERI_PRECISION).is_err());
    parts.push(last.clone());
    parts.push(last);
    assert!(GcKernelBasis::from_groups(&prep, &parts, ERI_PRECISION).is_err());
    let all: Vec<Vec<usize>> = vec![(0..prep.nshells()).collect()];
    assert!(GcKernelBasis::from_groups(&prep, &all, ERI_PRECISION).is_err());
}

/// A one-centre group far from the origin (about 1e8 Bohr on each axis):
/// every position is kept relative to the pair's own centre, so `P − Q`
/// and `P − A` are exactly zero and the parity-zero `(ss|sp)` integrals are
/// exactly 0, and the `(ss|ss)` integrals equal those of the same atom at
/// the origin to rounding.
#[test]
fn one_centre_quartets_far_from_the_origin_keep_parity() {
    let mut bs = BasisSet {
        name: "gc-kernel-far".into(),
        shells: HashMap::new(),
        ecps: HashMap::new(),
    };
    bs.shells.insert(
        1,
        vec![
            Shell {
                l: 0,
                pure: true,
                exponents: vec![3.0, 0.9, 0.2],
                coefficients: vec![0.2, 0.5, 0.6],
            },
            Shell {
                l: 0,
                pure: true,
                exponents: vec![0.9, 0.2],
                coefficients: vec![-0.3, 1.0],
            },
            Shell {
                l: 1,
                pure: false,
                exponents: vec![0.7],
                coefficients: vec![1.0],
            },
        ],
    );
    let blocks = |x: f64| {
        let m = Molecule::parse_xyz(&format!("1\nH\nH {x} {x} {x}\n"), 0, 2).unwrap();
        let prep = PreparedBasis::new(&m, &bs).unwrap();
        let kb = GcKernelBasis::from_groups(&prep, &atom_l_parts(&prep), ERI_PRECISION).unwrap();
        let mut scratch = KernelScratch::default();
        let (mut out, mut col) = (Vec::new(), [0.0; 3]);
        let mut vals: Vec<(i32, f64)> = Vec::new();
        for s in [
            [0, 0, 1, 1],
            [0, 1, 0, 1],
            [1, 0, 2, 0],
            [2, 0, 1, 1],
            [0, 0, 0, 2],
        ] {
            let (PairKind::Data(b), PairKind::Data(k)) =
                (kb.pair_of(s[0], s[1]), kb.pair_of(s[2], s[3]))
            else {
                panic!("{s:?} has no pair data");
            };
            out.resize(kb.block_len(b, k), 0.0);
            kb.eval(b, k, &mut out, &mut scratch);
            let n = kb.gather(&out, b, k, s, &mut col);
            let l = s.iter().map(|&x| kb.shell_l(x)).sum::<i32>();
            vals.extend(col[..n].iter().map(|&v| (l, v)));
        }
        vals
    };
    // 5.3e7 Angstrom is about 1e8 Bohr.
    let (near, far) = (blocks(0.0), blocks(5.3e7));
    for ((l, a), (_, b)) in near.iter().zip(&far) {
        if *l == 1 {
            assert_eq!(*b, 0.0, "a parity-zero one-centre integral far out: {b:e}");
            assert_eq!(
                *a, 0.0,
                "a parity-zero one-centre integral at the origin: {a:e}"
            );
        } else {
            assert!(
                *a != 0.0 && (a - b).abs() <= 1e-15 * a.abs(),
                "{a:e} vs {b:e}"
            );
        }
    }
}
