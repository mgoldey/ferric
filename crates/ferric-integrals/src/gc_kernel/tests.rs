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
fn compare(prep: &PreparedBasis, kb: &GcKernelBasis) -> [(f64, f64, usize); 2] {
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
                    let lib = eng.compute_quartet(prep, s1, s2, s3, s4);
                    let c = &mut out[l as usize];
                    c.2 += 1;
                    match lib {
                        Some(q) => {
                            assert_eq!(q.len(), n);
                            for (x, y) in col[..n].iter().zip(q) {
                                c.0 = c.0.max((x - y).abs());
                                c.1 = c.1.max(y.abs());
                            }
                        }
                        None => {
                            for x in &col[..n] {
                                c.0 = c.0.max(x.abs());
                            }
                        }
                    }
                }
            }
        }
    }
    out
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
    let r = compare(prep, kb);
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
    ] {
        let m = load(mol);
        let parent = PreparedBasis::new(&m, &basis::bundled(basis).unwrap()).unwrap();
        let rot = RotatedBasis::detect_prepared(&parent, RotationLevel::Ogc)
            .unwrap()
            .expect("cc basis rotates");
        let kb = GcKernelBasis::for_rotated(&rot, ERI_PRECISION).unwrap();
        assert!(kb.groups().iter().any(|g| g.ncols() > 1));
        check(&rot.prep, &kb, &format!("{mol} {basis} rotated"));
        let kb =
            GcKernelBasis::from_groups(&parent, &atom_l_parts(&parent), ERI_PRECISION).unwrap();
        assert!(kb.groups().iter().any(|g| g.ncols() >= 3));
        check(&parent, &kb, &format!("{mol} {basis} unrotated"));
    }
}

/// Cartesian p (libint order x, y, z) as well as pure (y, z, x).
#[test]
fn kernel_matches_libint_with_cartesian_p() {
    let m = load("water");
    let mut bs = basis::bundled("cc-pvdz").unwrap();
    for shells in bs.shells.values_mut() {
        for sh in shells.iter_mut() {
            if sh.l == 1 {
                sh.pure = false;
            }
        }
    }
    let prep = PreparedBasis::new(&m, &bs).unwrap();
    let kb = GcKernelBasis::from_groups(&prep, &atom_l_parts(&prep), ERI_PRECISION).unwrap();
    assert!(kb.groups().iter().any(|g| g.l == 1 && !g.pure));
    check(&prep, &kb, "water cc-pvdz cartesian p");
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
