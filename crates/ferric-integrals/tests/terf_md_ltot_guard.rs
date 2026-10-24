//! The hand-written (MD) terf/terfc 3- and 2-center paths keep their Boys /
//! terf-aux vector in a TERFC_DIMM = 24 deep stack buffer. A block whose
//! total angular momentum reaches 24 must be REFUSED with a status that the
//! Rust side reports (Err from the checked call, a panic from the unchecked
//! one), never computed with a skipped primitive or returned as a zero
//! block. eri3 reaches 24 at three l = 8 shells, eri2 at two l = 12 shells.

use ferric_core::basis::{BasisSet, Shell};
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::ffi;
use ferric_integrals::operator::Operator;
use std::collections::HashMap;
use std::panic::{catch_unwind, AssertUnwindSafe};

fn ensure_tables() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        if std::env::var("FERRIC_TERF_TABLE_DIR").is_err() {
            std::env::set_var(
                "FERRIC_TERF_TABLE_DIR",
                concat!(env!("CARGO_MANIFEST_DIR"), "/../../terf-tables"),
            );
        }
    });
}

/// One single-primitive pure shell per listed l on each of two H atoms 1 A
/// apart (two centres: a one-centre block of odd total l vanishes by parity).
/// Shell index = atom * ls.len() + position in `ls`.
fn basis_with(ls: &[i32], pure: bool) -> PreparedBasis {
    let mol = Molecule::parse_xyz("2\n\nH 0 0 0\nH 0 0 1.0\n", 0, 1).unwrap();
    let shells = ls
        .iter()
        .map(|&l| Shell {
            l,
            pure,
            exponents: vec![1.1],
            coefficients: vec![1.0],
        })
        .collect();
    let mut map = HashMap::new();
    map.insert(1, shells);
    PreparedBasis::new(
        &mol,
        &BasisSet {
            name: "high-l".into(),
            shells: map,
            ecps: HashMap::new(),
        },
    )
    .unwrap()
}

fn ops() -> [Operator; 2] {
    [Operator::terf(2.0), Operator::terfc(2.0)]
}

#[test]
fn eri3_refuses_total_l_at_the_table_depth() {
    ensure_tables();
    let obs = basis_with(&[7, 8], true); // atom 0: shells 0 (l=7), 1 (l=8); atom 1: 2, 3
    for op in ops() {
        let mut eng = Engine::new_3center(op, &obs, &obs, 0.0).unwrap();
        // 8 + 8 + 7 = 23: the deepest block the tables serve.
        let ok = eng
            .compute_eri3_checked(&obs, &obs, 1, 1, 2)
            .unwrap()
            .map(<[f64]>::to_vec);
        let ok = ok.expect("l = 8+8+7 block must be computed, not screened");
        assert!(
            ok.iter().all(|v| v.is_finite()) && ok.iter().any(|v| *v != 0.0),
            "{op:?}"
        );
        // 8 + 8 + 8 = 24: refused through the checked path ...
        assert_eq!(
            eng.compute_eri3_checked(&obs, &obs, 1, 1, 1)
                .map(|b| b.map(<[f64]>::to_vec)),
            Err(ffi::SCF_EINVAL),
            "{op:?}"
        );
        // ... and a panic, not None, through the unchecked one.
        let r = catch_unwind(AssertUnwindSafe(|| {
            eng.compute_eri3(&obs, &obs, 1, 1, 1).map(<[f64]>::to_vec)
        }));
        assert!(
            r.is_err(),
            "{op:?}: unchecked eri3 returned {:?} instead of panicking",
            r.unwrap()
        );
    }
}

/// Cartesian shells: libint2 2.7.2's solid-harmonic tables stop at l = 10
/// (it asserts), and the guard is about total l, not the pure transform.
#[test]
fn eri2_refuses_total_l_at_the_table_depth() {
    ensure_tables();
    let dfbs = basis_with(&[11, 12], false); // atom 0: shells 0 (l=11), 1 (l=12); atom 1: 2, 3
    for op in ops() {
        let mut eng = Engine::new_2center(op, &dfbs, 0.0).unwrap();
        let ok = eng.compute_eri2_checked(&dfbs, 1, 2).unwrap().to_vec(); // 12 + 11 = 23
        assert!(
            ok.iter().all(|v| v.is_finite()) && ok.iter().any(|v| *v != 0.0),
            "{op:?}"
        );
        assert_eq!(
            eng.compute_eri2_checked(&dfbs, 1, 1).map(<[f64]>::to_vec),
            Err(ffi::SCF_EINVAL),
            "{op:?}"
        );
        let r = catch_unwind(AssertUnwindSafe(|| eng.compute_eri2(&dfbs, 1, 1).to_vec()));
        assert!(r.is_err(), "{op:?}: unchecked eri2 did not panic");
    }
}
