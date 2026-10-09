//! Shifted 2- and 3-centre DERIVATIVE integrals — the SR primitives of the
//! Gamma RS-GDF forces (`ferric_pbc::grad`, FINDINGS "Iteration 18").
//!
//! `Engine::compute_eri2_deriv_shifted` / `scf_compute_eri2_deriv_shifted`
//! (new):
//! * Exactness anchor: a zero shift is BITWISE equal to `compute_eri2_deriv`.
//! * Direction anchor: the ket shifted by `T` equals an ordinary, UNSHIFTED
//!   derivative in a basis with a ghost copy displaced by `T` (independent
//!   construction; a ±T lattice sum is blind to the shift sign, this is not).
//! * Finite difference of `compute_eri2_shifted` over the ket shift: the
//!   `d/dQ` block is the derivative of the SHIFTED energy integral, and
//!   `d/dP = −d/dQ` (translation invariance of a 2-centre kernel).
//! * Honesty: non-finite shift / out-of-range shell → `Err`.
//!
//! `Engine::compute_eri3_deriv_shifted` (existing, used by the SR attraction
//! gradient on s/p ORBITAL shells only): finite difference of
//! `compute_eri3_shifted` over each of the three shifts, with p, d and f AUX
//! shells under `erfc(ω)`. Iteration 16 found libint2's translation-
//! invariance block losing precision for a 1e16 Gaussian nucleus; with
//! ordinary aux exponents this is measured here, not assumed, before the
//! RS-GDF force uses the aux (d/dP) and orbital (d/dsh1, d/dsh2) blocks.
//!
//! FD: central differences at `h` and `h/2`, Richardson-combined
//! (`O(h⁴)` truncation), `h = 1e-3` Bohr: truncation ~1e-12 and roundoff
//! `~ε|I|/h ~ 1e-13 |I|`. Bar `FD_BAR · max(1, max|dI|)`, three decades
//! above that floor and far below a lost-precision block (Iteration 16:
//! O(1e-2..1e-6)). NOT YET MEASURED in Rust — the printed worst values are
//! the numbers to record.

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

const FD_H: f64 = 1e-3;
const FD_BAR: f64 = 1e-8;

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

fn add(v: [f64; 3], x: usize, h: f64) -> [f64; 3] {
    let mut o = v;
    o[x] += h;
    o
}

fn max_abs(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |m, x| m.max(x.abs()))
}

/// Richardson-combined central difference of `f` (a block-valued function
/// of a scalar displacement) at `h`: `(4 D(h/2) − D(h))/3`.
fn richardson<F>(mut f: F, h: f64) -> Vec<f64>
where
    F: FnMut(f64) -> Vec<f64>,
{
    let mut central = |hh: f64| -> Vec<f64> {
        let p = f(hh);
        let m = f(-hh);
        p.iter()
            .zip(&m)
            .map(|(a, b)| (a - b) / (2.0 * hh))
            .collect()
    };
    let d1 = central(h);
    let d2 = central(0.5 * h);
    d1.iter()
        .zip(&d2)
        .map(|(a, b)| (4.0 * b - a) / 3.0)
        .collect()
}

// ------------------------------------------------------------------ 2-centre

#[test]
fn eri2_deriv_zero_shift_is_bitwise_identical_to_compute_eri2_deriv() {
    let mol = water();
    let aux = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    for op in [Operator::coulomb(), Operator::erfc(0.7)] {
        let mut plain = Engine::new_2center_deriv(op, &aux, 1e-14).unwrap();
        let mut shifted = Engine::new_2center_deriv(op, &aux, 1e-14).unwrap();
        let mut nonzero = 0usize;
        for p in 0..aux.nshells() {
            for q in 0..aux.nshells() {
                let a = plain.compute_eri2_deriv(&aux, p, q).map(|b| b.to_vec());
                let b = shifted
                    .compute_eri2_deriv_shifted(&aux, p, q, [0.0; 3])
                    .expect("shifted eri2 deriv")
                    .map(|b| b.to_vec());
                match (a, b) {
                    (None, None) => {}
                    (Some(a), Some(b)) => {
                        assert_eq!(a.len(), b.len(), "({p}|{q}) length");
                        for (i, (x, y)) in a.iter().zip(&b).enumerate() {
                            assert_eq!(x.to_bits(), y.to_bits(), "({p}|{q})[{i}]: {x} vs {y}");
                        }
                        nonzero += a.iter().filter(|v| **v != 0.0).count();
                    }
                    (a, b) => panic!(
                        "({p}|{q}): screened mismatch {:?} vs {:?}",
                        a.is_some(),
                        b.is_some()
                    ),
                }
            }
        }
        assert!(nonzero > 0, "vacuous");
    }
}

#[test]
fn eri2_deriv_ket_shift_equals_displaced_ghost_copy() {
    let mol = water();
    let t = [1.3, -0.4, 2.1];
    let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
    let aux = PreparedBasis::new(&mol, &aux_bs).unwrap();
    let aux2 = PreparedBasis::new(&with_ghost_copy(&mol, t), &aux_bs).unwrap();
    let nsh = aux.nshells();
    assert_eq!(aux2.nshells(), 2 * nsh);
    let op = Operator::erfc(1.0);
    let mut eng = Engine::new_2center_deriv(op, &aux, 1e-14).unwrap();
    let mut eng2 = Engine::new_2center_deriv(op, &aux2, 1e-14).unwrap();
    let mut worst = 0.0_f64;
    let mut biggest = 0.0_f64;
    for p in 0..nsh {
        for q in 0..nsh {
            let a = eng
                .compute_eri2_deriv_shifted(&aux, p, q, t)
                .unwrap()
                .map(|b| b.to_vec());
            let b = eng2
                .compute_eri2_deriv(&aux2, p, nsh + q)
                .map(|b| b.to_vec());
            match (a, b) {
                (Some(a), Some(b)) => {
                    assert_eq!(a.len(), b.len());
                    for (x, y) in a.iter().zip(&b) {
                        worst = worst.max((x - y).abs());
                    }
                    biggest = biggest.max(max_abs(&b));
                }
                (Some(v), None) | (None, Some(v)) => worst = worst.max(max_abs(&v)),
                (None, None) => {}
            }
        }
    }
    eprintln!("eri2 deriv ket-shift vs ghost copy: max|d| = {worst:.2e} (max |dI| {biggest:.2e})");
    assert!(biggest > 1e-3, "vacuous comparison");
    assert!(worst < 1e-12, "{worst:.3e}");
}

#[test]
fn eri2_deriv_shifted_matches_fd_of_eri2_shifted() {
    let mol = water();
    let aux = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let dims = aux.shell_dims().to_vec();
    let op = Operator::erfc(1.0);
    let t0 = [0.9, -1.4, 0.6];
    let mut eng_d = Engine::new_2center_deriv(op, &aux, 1e-20).unwrap();
    let mut eng = Engine::new_2center(op, &aux, 1e-20).unwrap();
    let (mut worst_q, mut worst_p, mut worst_ti, mut biggest) =
        (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    for p in 0..aux.nshells() {
        for q in 0..aux.nshells() {
            let n = dims[p] * dims[q];
            let blk = eng_d
                .compute_eri2_deriv_shifted(&aux, p, q, t0)
                .unwrap()
                .map(|b| b.to_vec())
                .unwrap_or_else(|| vec![0.0; 6 * n]);
            assert_eq!(blk.len(), 6 * n);
            biggest = biggest.max(max_abs(&blk));
            for x in 0..3 {
                let fd = richardson(
                    |h| {
                        eng.compute_eri2_shifted(&aux, p, q, add(t0, x, h))
                            .unwrap()
                            .to_vec()
                    },
                    FD_H,
                );
                for i in 0..n {
                    let dp = blk[x * n + i];
                    let dq = blk[(3 + x) * n + i];
                    worst_q = worst_q.max((dq - fd[i]).abs());
                    worst_p = worst_p.max((dp + fd[i]).abs());
                    worst_ti = worst_ti.max((dp + dq).abs());
                }
            }
        }
    }
    eprintln!(
        "eri2 deriv shifted vs FD (erfc 1.0, aux cc-pvdz-ri s/p/d/f): d/dQ {worst_q:.2e}, \
         d/dP (= −FD) {worst_p:.2e}, |d/dP + d/dQ| {worst_ti:.2e}, max|dI| {biggest:.2e}"
    );
    let bar = FD_BAR * biggest.max(1.0);
    assert!(biggest > 1e-3, "vacuous");
    assert!(worst_q < bar, "d/dQ vs FD {worst_q:e}");
    assert!(worst_p < bar, "d/dP vs −FD {worst_p:e}");
    assert!(worst_ti < bar, "translation invariance {worst_ti:e}");
}

#[test]
fn eri2_deriv_shifted_rejects_bad_input() {
    let mol = water();
    let aux = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let mut eng = Engine::new_2center_deriv(Operator::coulomb(), &aux, 1e-14).unwrap();
    assert!(eng
        .compute_eri2_deriv_shifted(&aux, 0, 0, [f64::NAN, 0.0, 0.0])
        .is_err());
    assert!(eng
        .compute_eri2_deriv_shifted(&aux, 0, 0, [0.0, f64::INFINITY, 0.0])
        .is_err());
    assert!(eng
        .compute_eri2_deriv_shifted(&aux, aux.nshells(), 0, [0.0; 3])
        .is_err());
    assert!(eng
        .compute_eri2_deriv_shifted(&aux, 0, aux.nshells(), [0.0; 3])
        .is_err());
}

// ------------------------------------------------------------------ 3-centre

/// FD of every block of `compute_eri3_deriv_shifted` (aux `d/dP`, bra
/// `d/dsh1`, ket `d/dsh2`) over the corresponding shift, erfc(1.0), orbital
/// cc-pVDZ (s/p/d) and aux cc-pVDZ-RI (s/p/d on H, up to f on O), at
/// non-zero base shifts so every shift path is exercised. Worst error is
/// reported per block and per aux angular momentum.
#[test]
fn eri3_deriv_shifted_blocks_match_fd_with_p_d_aux_under_erfc() {
    let mol = water();
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let aux = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let (od, ad) = (obs.shell_dims().to_vec(), aux.shell_dims().to_vec());
    let aux_l: Vec<i32> = aux.located_shells().iter().map(|s| s.l).collect();
    let op = Operator::erfc(1.0);
    let base = [[-0.7, 0.5, 0.9], [0.0; 3], [1.3, -0.4, 2.1]];
    let mut eng_d = Engine::new_3center_deriv(op, &obs, &aux, 1e-20).unwrap();
    let mut eng = Engine::new_3center(op, &obs, &aux, 1e-20).unwrap();
    // worst[slot][l_aux], slot 0 = aux, 1 = sh1, 2 = sh2
    let mut worst = [[0.0_f64; 5]; 3];
    let mut biggest = [[0.0_f64; 5]; 3];
    for p in 0..aux.nshells() {
        let lp = aux_l[p].clamp(0, 4) as usize;
        for s1 in 0..obs.nshells() {
            for s2 in 0..obs.nshells() {
                let n = ad[p] * od[s1] * od[s2];
                let blk = eng_d
                    .compute_eri3_deriv_shifted(&obs, &aux, p, s1, s2, base)
                    .unwrap()
                    .map(|b| b.to_vec())
                    .unwrap_or_else(|| vec![0.0; 9 * n]);
                assert_eq!(blk.len(), 9 * n);
                for slot in 0..3 {
                    for x in 0..3 {
                        let fd = richardson(
                            |h| {
                                let mut sh = base;
                                sh[slot][x] += h;
                                eng.compute_eri3_shifted(&obs, &aux, p, s1, s2, sh)
                                    .unwrap()
                                    .map(|b| b.to_vec())
                                    .unwrap_or_else(|| vec![0.0; n])
                            },
                            FD_H,
                        );
                        let an = &blk[(3 * slot + x) * n..(3 * slot + x + 1) * n];
                        for i in 0..n {
                            worst[slot][lp] = worst[slot][lp].max((an[i] - fd[i]).abs());
                            biggest[slot][lp] = biggest[slot][lp].max(an[i].abs());
                        }
                    }
                }
            }
        }
    }
    let names = ["aux d/dP", "bra d/dsh1", "ket d/dsh2"];
    let mut seen_l = [false; 5];
    for &l in &aux_l {
        seen_l[l.clamp(0, 4) as usize] = true;
    }
    assert!(
        seen_l[1] && seen_l[2],
        "the aux basis must contain p and d shells"
    );
    for slot in 0..3 {
        for l in 0..5 {
            if !seen_l[l] {
                continue;
            }
            eprintln!(
                "eri3 deriv shifted vs FD, {} aux l = {l}: max|analytic − FD| = {:.2e} \
                 (max |dI| {:.2e})",
                names[slot], worst[slot][l], biggest[slot][l]
            );
            assert!(biggest[slot][l] > 1e-4, "{} l = {l}: vacuous", names[slot]);
            let bar = FD_BAR * biggest[slot][l].max(1.0);
            assert!(
                worst[slot][l] < bar,
                "{} aux l = {l}: {:e}",
                names[slot],
                worst[slot][l]
            );
        }
    }
}
