//! `Engine::compute_eri2_shifted` / `scf_compute_eri2_shifted` (Stage-1 PBC
//! step 7/9: the RS-GDF aux-metric lattice sum `Σ_T (P_0 | Q_T)_erfc`).
//!
//! * Exactness anchor: a zero shift is BITWISE equal to `compute_eri2`.
//! * Direction anchor: shifting the ket shell by `T` equals an ordinary,
//!   UNSHIFTED `(P | Q')` in a basis built on the molecule plus a ghost copy
//!   displaced by `T` (an independent construction — libint2 shells built
//!   from different coordinates). A lattice sum over ±T is blind to the
//!   shift sign; this is not.
//! * Bra/ket consistency: `(P_0 | Q_T) = (Q_0 | P_{−T})ᵀ` (translation
//!   invariance + the symmetry of the kernel).
//! * Honesty: non-finite shift / out-of-range shell → `Err`.
//!
//! Mutations these catch: sign flip `O − shift` in the shim (direction and
//! bra/ket anchors), moving the bra instead of the ket (direction anchor),
//! copying the wrong shell index (zero-shift anchor).

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

/// `mol` followed by a ghost copy displaced by `d` (shells nsh..2nsh).
fn with_ghost_copy(mol: &Molecule, d: [f64; 3]) -> Molecule {
    let mut m = mol.clone();
    for a in &mol.atoms {
        let mut g = a.clone();
        g.x += d[0];
        g.y += d[1];
        g.zpos += d[2];
        g.ghost = true;
        m.atoms.push(g);
    }
    m
}

fn ops() -> [(&'static str, Operator); 2] {
    [
        ("coulomb", Operator::coulomb()),
        ("erfc(0.7)", Operator::erfc(0.7)),
    ]
}

#[test]
fn zero_shift_is_bitwise_identical_to_compute_eri2() {
    let mol = water();
    let aux = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    for (name, op) in ops() {
        let mut plain = Engine::new_2center(op, &aux, 1e-14).unwrap();
        let mut shifted = Engine::new_2center(op, &aux, 1e-14).unwrap();
        let mut nonzero = 0usize;
        for p in 0..aux.nshells() {
            for q in 0..aux.nshells() {
                let a = plain.compute_eri2(&aux, p, q).to_vec();
                let b = shifted
                    .compute_eri2_shifted(&aux, p, q, [0.0; 3])
                    .expect("shifted eri2")
                    .to_vec();
                assert_eq!(a.len(), b.len(), "{name} ({p}|{q}) length");
                for (i, (x, y)) in a.iter().zip(&b).enumerate() {
                    assert_eq!(
                        x.to_bits(),
                        y.to_bits(),
                        "{name} ({p}|{q})[{i}]: {x} vs {y}"
                    );
                }
                nonzero += a.iter().filter(|v| **v != 0.0).count();
            }
        }
        assert!(nonzero > 0, "{name}: vacuous");
    }
}

#[test]
fn ket_shift_equals_displaced_ghost_copy() {
    let mol = water();
    let t = [1.3, -0.4, 2.1];
    let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
    let aux = PreparedBasis::new(&mol, &aux_bs).unwrap();
    let aux2 = PreparedBasis::new(&with_ghost_copy(&mol, t), &aux_bs).unwrap();
    let nsh = aux.nshells();
    assert_eq!(aux2.nshells(), 2 * nsh);
    for (name, op) in ops() {
        let mut eng = Engine::new_2center(op, &aux, 1e-14).unwrap();
        let mut eng2 = Engine::new_2center(op, &aux2, 1e-14).unwrap();
        let mut worst = 0.0_f64;
        let mut biggest = 0.0_f64;
        for p in 0..nsh {
            for q in 0..nsh {
                let a = eng.compute_eri2_shifted(&aux, p, q, t).unwrap().to_vec();
                let b = eng2.compute_eri2(&aux2, p, nsh + q).to_vec();
                assert_eq!(a.len(), b.len());
                for (x, y) in a.iter().zip(&b) {
                    worst = worst.max((x - y).abs());
                    biggest = biggest.max(y.abs());
                }
            }
        }
        eprintln!("{name}: ket-shift vs ghost copy max|d| = {worst:.2e} (max |I| {biggest:.2e})");
        assert!(biggest > 1e-3, "{name}: vacuous comparison");
        assert!(worst < 1e-12, "{name}: {worst:.3e}");
    }
}

#[test]
fn bra_ket_swap_with_negated_shift_is_the_transpose() {
    let mol = water();
    let aux = PreparedBasis::new(&mol, &basis::bundled("def2-universal-jkfit").unwrap()).unwrap();
    let dims = aux.shell_dims().to_vec();
    let t = [-2.2, 0.9, 1.7];
    let mt = [2.2, -0.9, -1.7];
    let mut eng = Engine::new_2center(Operator::erfc(1.1), &aux, 1e-14).unwrap();
    let mut worst = 0.0_f64;
    let mut biggest = 0.0_f64;
    for p in 0..aux.nshells() {
        for q in 0..aux.nshells() {
            let a = eng.compute_eri2_shifted(&aux, p, q, t).unwrap().to_vec();
            let b = eng.compute_eri2_shifted(&aux, q, p, mt).unwrap().to_vec();
            let (np, nq) = (dims[p], dims[q]);
            for i in 0..np {
                for j in 0..nq {
                    let x = a[i * nq + j];
                    let y = b[j * np + i];
                    worst = worst.max((x - y).abs());
                    biggest = biggest.max(x.abs());
                }
            }
        }
    }
    assert!(biggest > 1e-3, "vacuous");
    assert!(worst < 1e-12, "(P_0|Q_T) vs (Q_0|P_-T)^T: {worst:.3e}");
}

#[test]
fn non_finite_shift_and_bad_shell_are_errors() {
    let mol = water();
    let aux = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let mut eng = Engine::new_2center(Operator::coulomb(), &aux, 1e-14).unwrap();
    assert!(eng
        .compute_eri2_shifted(&aux, 0, 0, [f64::NAN, 0.0, 0.0])
        .is_err());
    assert!(eng
        .compute_eri2_shifted(&aux, 0, 0, [0.0, f64::INFINITY, 0.0])
        .is_err());
    assert!(eng
        .compute_eri2_shifted(&aux, aux.nshells(), 0, [0.0; 3])
        .is_err());
    assert!(eng
        .compute_eri2_shifted(&aux, 0, aux.nshells(), [0.0; 3])
        .is_err());
}
