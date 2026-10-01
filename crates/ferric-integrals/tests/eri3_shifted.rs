//! `Engine::compute_eri3_shifted` / `scf_compute_eri3_shifted` (Stage-1 PBC
//! step 4 prerequisite; the design's step-7 shim, landed early because the
//! Gaussian-nucleus SR attraction needs it).
//!
//! * Exactness anchor: all-zero shifts are BITWISE equal to `compute_eri3`.
//! * Direction anchors (one per shifted slot): shifting the aux shell by `M`
//!   or the second orbital shell by `L` equals an ordinary, UNSHIFTED
//!   integral in a basis built on a displaced molecule (an independent
//!   construction — libint2 shells built from different coordinates). A
//!   lattice sum over ±L is blind to the shift sign; these are not.
//! * Translation invariance: shifting all three shells by the same vector
//!   leaves the integral unchanged (roundoff only).
//! * Honesty: non-finite shift / out-of-range shell → `Err`.
//!
//! Mutations these catch: sign flip `O − shift` in the shim (direction
//! anchors), shift applied to the wrong shell slot (direction anchors: each
//! slot is moved alone), copying the wrong shell index (zero-shift anchor).

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::operator::Operator;

const WATER: &str = "3
water
O  0.0000  0.0000  0.1173
H  0.0000  0.7572 -0.4692
H  0.0000 -0.7572 -0.4692
";

fn water() -> Molecule {
    Molecule::parse_xyz(WATER, 0, 1).expect("water")
}

fn translated(mol: &Molecule, d: [f64; 3]) -> Molecule {
    let mut m = mol.clone();
    for a in &mut m.atoms {
        a.x += d[0];
        a.y += d[1];
        a.zpos += d[2];
    }
    m
}

/// `mol` followed by a ghost copy displaced by `d` (shells nsh..2nsh).
fn with_ghost_copy(mol: &Molecule, d: [f64; 3]) -> Molecule {
    let mut m = mol.clone();
    for a in &translated(mol, d).atoms {
        let mut g = a.clone();
        g.ghost = true;
        m.atoms.push(g);
    }
    m
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()))
}

fn ops() -> [(&'static str, Operator); 2] {
    [
        ("coulomb", Operator::coulomb()),
        ("erfc(0.7)", Operator::erfc(0.7)),
    ]
}

#[test]
fn zero_shift_is_bitwise_identical_to_compute_eri3() {
    let mol = water();
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let aux = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    for (name, op) in ops() {
        let mut plain = Engine::new_3center(op, &obs, &aux, 1e-14).unwrap();
        let mut shifted = Engine::new_3center(op, &obs, &aux, 1e-14).unwrap();
        let mut nonzero = 0usize;
        for p in 0..aux.nshells() {
            for s1 in 0..obs.nshells() {
                for s2 in 0..obs.nshells() {
                    let a = plain
                        .compute_eri3(&obs, &aux, p, s1, s2)
                        .map(|b| b.to_vec());
                    let b = shifted
                        .compute_eri3_shifted(&obs, &aux, p, s1, s2, [[0.0; 3]; 3])
                        .expect("shifted eri3")
                        .map(|b| b.to_vec());
                    match (a, b) {
                        (None, None) => {}
                        (Some(a), Some(b)) => {
                            assert_eq!(a.len(), b.len(), "{name} ({p}|{s1},{s2}) length");
                            for (i, (x, y)) in a.iter().zip(&b).enumerate() {
                                assert_eq!(
                                    x.to_bits(),
                                    y.to_bits(),
                                    "{name} ({p}|{s1},{s2})[{i}]: {x} vs {y}"
                                );
                            }
                            nonzero += a.iter().filter(|v| **v != 0.0).count();
                        }
                        (a, b) => panic!(
                            "{name} ({p}|{s1},{s2}): screened mismatch {:?} vs {:?}",
                            a.is_some(),
                            b.is_some()
                        ),
                    }
                }
            }
        }
        assert!(nonzero > 0, "{name}: vacuous");
    }
}

#[test]
fn orbital_shift_equals_displaced_ghost_copy() {
    let mol = water();
    let l = [1.3, -0.4, 2.1];
    let obs_bs = basis::bundled("cc-pvdz").unwrap();
    let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let aux = PreparedBasis::new(&mol, &aux_bs).unwrap();
    let obs2 = PreparedBasis::new(&with_ghost_copy(&mol, l), &obs_bs).unwrap();
    let nsh = obs.nshells();
    assert_eq!(obs2.nshells(), 2 * nsh);
    for (name, op) in ops() {
        let mut eng = Engine::new_3center(op, &obs, &aux, 1e-14).unwrap();
        let mut eng2 = Engine::new_3center(op, &obs2, &aux, 1e-14).unwrap();
        let mut worst = 0.0_f64;
        let mut biggest = 0.0_f64;
        for p in 0..aux.nshells() {
            for s1 in 0..nsh {
                for s2 in 0..nsh {
                    let a = eng
                        .compute_eri3_shifted(&obs, &aux, p, s1, s2, [[0.0; 3], [0.0; 3], l])
                        .unwrap()
                        .map(|b| b.to_vec());
                    let b = eng2
                        .compute_eri3(&obs2, &aux, p, s1, nsh + s2)
                        .map(|b| b.to_vec());
                    if let (Some(a), Some(b)) = (&a, &b) {
                        worst = worst.max(max_abs_diff(a, b));
                        biggest = biggest.max(b.iter().fold(0.0_f64, |m, v| m.max(v.abs())));
                    } else if a.is_some() != b.is_some() {
                        // One side screened: the other must be negligible.
                        let v = a.or(b).unwrap();
                        worst = worst.max(v.iter().fold(0.0_f64, |m, x| m.max(x.abs())));
                    }
                }
            }
        }
        eprintln!(
            "{name}: orbital-shift vs ghost copy max|d| = {worst:.2e} (max |I| {biggest:.2e})"
        );
        assert!(biggest > 1e-3, "{name}: vacuous comparison");
        assert!(worst < 1e-12, "{name}: {worst:.3e}");
    }
}

#[test]
fn aux_shift_equals_displaced_aux_basis() {
    let mol = water();
    let m = [-2.2, 0.9, 1.7];
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
    let aux = PreparedBasis::new(&mol, &aux_bs).unwrap();
    let aux_m = PreparedBasis::new(&translated(&mol, m), &aux_bs).unwrap();
    for (name, op) in ops() {
        let mut eng = Engine::new_3center(op, &obs, &aux, 1e-14).unwrap();
        let mut eng_m = Engine::new_3center(op, &obs, &aux_m, 1e-14).unwrap();
        let mut worst = 0.0_f64;
        let mut biggest = 0.0_f64;
        for p in 0..aux.nshells() {
            for s1 in 0..obs.nshells() {
                for s2 in 0..obs.nshells() {
                    let a = eng
                        .compute_eri3_shifted(&obs, &aux, p, s1, s2, [m, [0.0; 3], [0.0; 3]])
                        .unwrap()
                        .map(|b| b.to_vec());
                    let b = eng_m
                        .compute_eri3(&obs, &aux_m, p, s1, s2)
                        .map(|b| b.to_vec());
                    if let (Some(a), Some(b)) = (&a, &b) {
                        worst = worst.max(max_abs_diff(a, b));
                        biggest = biggest.max(b.iter().fold(0.0_f64, |mx, v| mx.max(v.abs())));
                    } else if a.is_some() != b.is_some() {
                        let v = a.or(b).unwrap();
                        worst = worst.max(v.iter().fold(0.0_f64, |mx, x| mx.max(x.abs())));
                    }
                }
            }
        }
        eprintln!(
            "{name}: aux-shift vs displaced aux max|d| = {worst:.2e} (max |I| {biggest:.2e})"
        );
        assert!(biggest > 1e-3, "{name}: vacuous comparison");
        assert!(worst < 1e-12, "{name}: {worst:.3e}");
    }
}

#[test]
fn common_shift_is_translation_invariant() {
    let mol = water();
    let obs = PreparedBasis::new(&mol, &basis::bundled("sto-3g").unwrap()).unwrap();
    let aux = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let t = [7.5, -3.25, 11.0];
    let mut eng = Engine::new_3center(Operator::erfc(1.1), &obs, &aux, 1e-14).unwrap();
    let mut worst = 0.0_f64;
    for p in 0..aux.nshells() {
        for s1 in 0..obs.nshells() {
            for s2 in 0..obs.nshells() {
                let a = eng
                    .compute_eri3_shifted(&obs, &aux, p, s1, s2, [[0.0; 3]; 3])
                    .unwrap()
                    .map(|b| b.to_vec());
                let b = eng
                    .compute_eri3_shifted(&obs, &aux, p, s1, s2, [t, t, t])
                    .unwrap()
                    .map(|b| b.to_vec());
                if let (Some(a), Some(b)) = (a, b) {
                    worst = worst.max(max_abs_diff(&a, &b));
                }
            }
        }
    }
    assert!(worst < 1e-12, "translation invariance {worst:.3e}");
}

#[test]
fn non_finite_shift_and_bad_shell_are_errors() {
    let mol = water();
    let obs = PreparedBasis::new(&mol, &basis::bundled("sto-3g").unwrap()).unwrap();
    let aux = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let mut eng = Engine::new_3center(Operator::coulomb(), &obs, &aux, 1e-14).unwrap();
    let bad = [[0.0; 3], [f64::NAN, 0.0, 0.0], [0.0; 3]];
    assert!(eng.compute_eri3_shifted(&obs, &aux, 0, 0, 0, bad).is_err());
    let inf = [[0.0; 3], [0.0; 3], [0.0, f64::INFINITY, 0.0]];
    assert!(eng.compute_eri3_shifted(&obs, &aux, 0, 0, 0, inf).is_err());
    assert!(eng
        .compute_eri3_shifted(&obs, &aux, aux.nshells(), 0, 0, [[0.0; 3]; 3])
        .is_err());
    assert!(eng
        .compute_eri3_shifted(&obs, &aux, 0, 0, obs.nshells(), [[0.0; 3]; 3])
        .is_err());
}
