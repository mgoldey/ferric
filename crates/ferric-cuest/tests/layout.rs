//! CPU tests of the AO-layout / ordering-inference layer, on the stored
//! libint references. These run on any box; they make sure the tool that
//! will interpret the GPU run is itself right (and fails when it should).

use ferric_cuest::layout::{classify, infer_map, max_abs_diff, AoLayout, AoMap};
use ferric_cuest::reference::{all_references, OneElectronReference};
use ndarray::Array2;

const TOL: f64 = 1e-9;

fn refs() -> Vec<OneElectronReference> {
    let r: Vec<_> = all_references().into_iter().map(|(_, r)| r).collect();
    assert_eq!(
        r.len(),
        4,
        "expected 4 stored references in testdata/reference/cuest"
    );
    r
}

fn layout_of(r: &OneElectronReference) -> AoLayout {
    AoLayout::new(&r.molecule(), &r.basis_set().unwrap()).unwrap()
}

/// The layout's shell order and AO count must be the ones libint used.
#[test]
fn layout_matches_the_reference_shell_order() {
    for r in refs() {
        let lay = layout_of(&r);
        let ls: Vec<i32> = lay.shells.iter().map(|s| s.l).collect();
        assert_eq!(ls, r.shell_l, "{}: shell order", r.basis);
        assert_eq!(lay.nao, r.nao, "{}: nao", r.basis);
    }
}

#[test]
fn pure_flag_choice_per_basis() {
    let want = [
        ("6-31g*", false),
        ("cc-pvdz", true),
        ("cc-pvtz", true),
        ("sto-3g", false),
    ];
    for r in refs() {
        let w = want.iter().find(|(b, _)| *b == r.basis).unwrap().1;
        assert_eq!(layout_of(&r).cuest_pure, w, "{}", r.basis);
    }
}

#[test]
fn map_round_trips() {
    for r in refs() {
        let lay = layout_of(&r);
        let map = AoMap::hypothesis(&lay);
        let s = r.s_matrix();
        assert_eq!(map.to_ferric(&map.to_cuest(&s)), s, "{}", r.basis);
    }
}

fn infer_with(
    r: &OneElectronReference,
    lay: &AoLayout,
    cuest: [&Array2<f64>; 3],
) -> Vec<ferric_cuest::layout::ClassVerdict> {
    let (s, t, v) = (r.s_matrix(), r.t_matrix(), r.v_matrix());
    let inferred = infer_map(lay, &cuest, &[&s, &t, &v]);
    classify(lay, &inferred, TOL)
}

/// Trivial limit: feed "cuEST" matrices that ARE the hypothesis applied to
/// libint's. Every class must come back clean and equal to the hypothesis.
#[test]
fn inference_recovers_the_hypothesis_exactly() {
    for r in refs() {
        let lay = layout_of(&r);
        let map = AoMap::hypothesis(&lay);
        let (s, t, v) = (
            map.to_cuest(&r.s_matrix()),
            map.to_cuest(&r.t_matrix()),
            map.to_cuest(&r.v_matrix()),
        );
        for c in infer_with(&r, &lay, [&s, &t, &v]) {
            assert!(
                c.problem.is_none(),
                "{} l={}: {:?}",
                r.basis,
                c.l,
                c.problem
            );
            assert!(
                c.matches_hypothesis,
                "{} l={}: {:?} vs {:?}",
                r.basis, c.l, c.components, c.hypothesis
            );
        }
    }
}

/// A different, known order with sign flips (Psi4-style m = 0, +1, -1, +2, -2
/// with alternating phases) must be recovered exactly and reported as NOT
/// matching the hypothesis. This is what a wrong guess looks like on the GPU.
#[test]
fn inference_recovers_a_foreign_order_and_flags_it() {
    for r in refs().into_iter().filter(|r| r.basis.starts_with("cc-")) {
        let lay = layout_of(&r);
        let foreign = |l: i32, n: usize| -> Vec<(usize, f64)> {
            if l == 0 {
                return vec![(0, 1.0)];
            }
            // cuEST component k <- ferric component perm[k], sign alternates.
            let mut perm: Vec<usize> = vec![l as usize];
            for j in 1..=l as usize {
                perm.push(l as usize + j);
                perm.push(l as usize - j);
            }
            if n == 3 && l == 1 {
                perm = vec![2, 0, 1];
            }
            perm.into_iter()
                .enumerate()
                .map(|(k, j)| (j, if k % 2 == 1 { -1.0 } else { 1.0 }))
                .collect()
        };
        let map = AoMap::from_components(&lay, |sh| foreign(sh.l, sh.nfunc()));
        let (s, t, v) = (
            map.to_cuest(&r.s_matrix()),
            map.to_cuest(&r.t_matrix()),
            map.to_cuest(&r.v_matrix()),
        );
        let verdicts = infer_with(&r, &lay, [&s, &t, &v]);
        for c in &verdicts {
            assert!(
                c.problem.is_none(),
                "{} l={}: {:?}",
                r.basis,
                c.l,
                c.problem
            );
            let n = lay.shells.iter().find(|s| s.l == c.l).unwrap().nfunc();
            assert_eq!(
                c.components.as_ref().unwrap(),
                &foreign(c.l, n),
                "{} l={}",
                r.basis,
                c.l
            );
        }
        assert!(
            verdicts.iter().any(|c| !c.matches_hypothesis),
            "{}: foreign order not flagged",
            r.basis
        );
        // And reordering with the recovered map reproduces libint.
        let rec = AoMap::from_components(&lay, |sh| {
            verdicts
                .iter()
                .find(|c| {
                    (c.l, c.cuest_pure, c.ferric_pure) == (sh.l, sh.cuest_pure, sh.ferric_pure)
                })
                .unwrap()
                .components
                .clone()
                .unwrap()
        });
        assert!(max_abs_diff(&rec.to_ferric(&s), &r.s_matrix()) < 1e-15);
    }
}

/// Mutation: a normalization error on one shell (factor 1.05) must be
/// reported as a problem, with the factor visible, not silently absorbed.
#[test]
fn inference_flags_a_normalization_error() {
    let r = refs().into_iter().find(|r| r.basis == "cc-pvdz").unwrap();
    let lay = layout_of(&r);
    let map = AoMap::hypothesis(&lay);
    let d_shell = lay.shells.iter().position(|s| s.l == 2).unwrap();
    let (o, n) = (lay.shells[d_shell].offset, lay.shells[d_shell].nfunc());
    let scale = |m: Array2<f64>| {
        let mut m = map.to_cuest(&m);
        for i in o..o + n {
            m.row_mut(i).mapv_inplace(|x| x * 1.05);
            m.column_mut(i).mapv_inplace(|x| x * 1.05);
        }
        m
    };
    let (s, t, v) = (
        scale(r.s_matrix()),
        scale(r.t_matrix()),
        scale(r.v_matrix()),
    );
    let verdicts = infer_with(&r, &lay, [&s, &t, &v]);
    let d = verdicts.iter().find(|c| c.l == 2).unwrap();
    let p = d.problem.as_deref().expect("scaled d shell not flagged");
    assert!(p.contains("1.05"), "factor not reported: {p}");
}

/// The stored references are what libint gives: S is symmetric with a unit
/// diagonal for s, p and pure shells (libint normalizes a Cartesian l>=2
/// shell so only its x^l-type components have unit norm: S[xy,xy] = 1/3);
/// V is negative on the diagonal.
#[test]
fn references_are_sane() {
    for r in refs() {
        let (s, v) = (r.s_matrix(), r.v_matrix());
        for sh in layout_of(&r)
            .shells
            .iter()
            .filter(|sh| sh.l < 2 || sh.ferric_pure)
        {
            for i in sh.offset..sh.offset + sh.nfunc() {
                assert!(
                    (s[[i, i]] - 1.0).abs() < 1e-10,
                    "{} S[{i},{i}] = {}",
                    r.basis,
                    s[[i, i]]
                );
            }
        }
        for i in 0..r.nao {
            assert!(v[[i, i]] < 0.0, "{} V[{i},{i}] = {}", r.basis, v[[i, i]]);
        }
        assert!(max_abs_diff(&s, &s.t().to_owned()) < 1e-14, "{}", r.basis);
    }
}
