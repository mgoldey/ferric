//! First derivatives of the erfc 3- and 2-centre integrals, `d(P|erfc|μν)/dR_A`
//! and `d(P|erfc|Q)/dR_A`, from `Engine::new_3center_deriv` /
//! `Engine::new_2center_deriv` (libint2 through the shim, op_kind 2), against
//! finite differences of the integrals themselves at displaced geometries.
//!
//! These blocks are the erfc integral response of the attenuated RI-MP2
//! gradient (`ferric_mp2::gradient::att_rimp2_gradient_analytical`).
//!
//! The analytic per-atom derivative sums the centre blocks (aux `d/dP`, bra
//! `d/dsh1`, ket `d/dsh2`) of every centre that sits on atom `A`. The FD moves
//! atom `A` and rebuilds the `PreparedBasis`, an independent construction.
//!
//! FD: central differences at `h` and `h/2`, Richardson-combined (`O(h⁴)`
//! truncation). At `h = 1e-3` Bohr the truncation is ~`h⁴ |I⁽⁵⁾| ~ 1e-12` and
//! the roundoff `~ε|I|/h ~ 1e-13 |I|`; the bar `FD_BAR · max(1, max|dI|)`
//! is three decades above that floor.
//!
//! Non-vacuity: the same derivative under Coulomb must differ from the erfc
//! one by far more than the bar, so a deriv engine that silently ran
//! Coulomb would fail.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::operator::Operator;
use ferric_integrals::threeindex;
use ndarray::{Array2, Array3};

const WATER: &str = "3
water
O  0.0000  0.0000  0.1173
H  0.0000  0.7572 -0.4692
H  0.0000 -0.7572 -0.4692
";

const FD_H: f64 = 1e-3;
const FD_BAR: f64 = 1e-8;
/// 0.420 Å⁻¹, the att-rimp2 default, in Bohr⁻¹.
const OMEGA: f64 = 0.420 * 0.529_177_210_903;

fn displaced(mol: &Molecule, atom: usize, x: usize, h: f64) -> Molecule {
    let mut m = mol.clone();
    match x {
        0 => m.atoms[atom].x += h,
        1 => m.atoms[atom].y += h,
        _ => m.atoms[atom].zpos += h,
    }
    m
}

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

/// `d(P|op|μν)/dR_{atom,x}`, `(naux, nbf, nbf)`, from the deriv engine.
fn eri3_deriv_atom(
    op: Operator,
    obs: &PreparedBasis,
    aux: &PreparedBasis,
    atom: usize,
    x: usize,
) -> Array3<f64> {
    let mut eng = Engine::new_3center_deriv(op, obs, aux, 1e-14).unwrap();
    let (dob, oob, at_ob) = (obs.shell_dims(), obs.shell_offsets(), obs.shell_to_atom());
    let (dax, oax, at_ax) = (aux.shell_dims(), aux.shell_offsets(), aux.shell_to_atom());
    let mut out = Array3::<f64>::zeros((aux.nbasis(), obs.nbasis(), obs.nbasis()));
    for sp in 0..aux.nshells() {
        for s1 in 0..obs.nshells() {
            for s2 in 0..obs.nshells() {
                let Some(d) = eng.compute_eri3_deriv(obs, aux, sp, s1, s2) else {
                    continue;
                };
                let (np, n1, n2) = (dax[sp], dob[s1], dob[s2]);
                let bs = np * n1 * n2;
                for p in 0..np {
                    for i in 0..n1 {
                        for j in 0..n2 {
                            let idx = (p * n1 + i) * n2 + j;
                            let mut v = 0.0;
                            if at_ax[sp] == atom {
                                v += d[x * bs + idx];
                            }
                            if at_ob[s1] == atom {
                                v += d[(3 + x) * bs + idx];
                            }
                            if at_ob[s2] == atom {
                                v += d[(6 + x) * bs + idx];
                            }
                            out[(oax[sp] + p, oob[s1] + i, oob[s2] + j)] = v;
                        }
                    }
                }
            }
        }
    }
    out
}

/// `d(P|op|Q)/dR_{atom,x}`, `(naux, naux)`, from the deriv engine.
fn eri2_deriv_atom(op: Operator, aux: &PreparedBasis, atom: usize, x: usize) -> Array2<f64> {
    let mut eng = Engine::new_2center_deriv(op, aux, 1e-14).unwrap();
    let (dax, oax, at_ax) = (aux.shell_dims(), aux.shell_offsets(), aux.shell_to_atom());
    let mut out = Array2::<f64>::zeros((aux.nbasis(), aux.nbasis()));
    for sp in 0..aux.nshells() {
        for sq in 0..aux.nshells() {
            let Some(d) = eng.compute_eri2_deriv(aux, sp, sq) else {
                continue;
            };
            let (np, nq) = (dax[sp], dax[sq]);
            let bs = np * nq;
            for p in 0..np {
                for q in 0..nq {
                    let idx = p * nq + q;
                    let mut v = 0.0;
                    if at_ax[sp] == atom {
                        v += d[x * bs + idx];
                    }
                    if at_ax[sq] == atom {
                        v += d[(3 + x) * bs + idx];
                    }
                    out[(oax[sp] + p, oax[sq] + q)] = v;
                }
            }
        }
    }
    out
}

fn max_abs<'a>(v: impl Iterator<Item = &'a f64>) -> f64 {
    v.fold(0.0_f64, |m, x| m.max(x.abs()))
}

#[test]
fn erfc_eri3_and_eri2_derivatives_match_finite_differences() {
    let mol = Molecule::parse_xyz(WATER, 0, 1).unwrap();
    let obs_bs = basis::bundled("cc-pvdz").unwrap();
    let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let aux = PreparedBasis::new(&mol, &aux_bs).unwrap();
    let erfc = Operator::erfc(OMEGA);
    let (mut worst3, mut worst2, mut gap3, mut gap2) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for atom in 0..mol.atoms.len() {
        for x in 0..3 {
            let a3 = eri3_deriv_atom(erfc, &obs, &aux, atom, x);
            let fd3 = richardson(
                |h| {
                    let m = displaced(&mol, atom, x, h);
                    let o = PreparedBasis::new(&m, &obs_bs).unwrap();
                    let a = PreparedBasis::new(&m, &aux_bs).unwrap();
                    threeindex::eri3_tensor(erfc, &o, &a)
                        .unwrap()
                        .iter()
                        .copied()
                        .collect()
                },
                FD_H,
            );
            let scale3 = max_abs(a3.iter()).max(1.0);
            let err3 = max_abs(
                a3.iter()
                    .zip(&fd3)
                    .map(|(a, f)| a - f)
                    .collect::<Vec<_>>()
                    .iter(),
            );
            worst3 = worst3.max(err3 / scale3);
            let c3 = eri3_deriv_atom(Operator::coulomb(), &obs, &aux, atom, x);
            gap3 = gap3.max(max_abs((&c3 - &a3).iter()));

            let a2 = eri2_deriv_atom(erfc, &aux, atom, x);
            let fd2 = richardson(
                |h| {
                    let m = displaced(&mol, atom, x, h);
                    let a = PreparedBasis::new(&m, &aux_bs).unwrap();
                    threeindex::coulomb_metric_2c(erfc, &a)
                        .unwrap()
                        .iter()
                        .copied()
                        .collect()
                },
                FD_H,
            );
            let scale2 = max_abs(a2.iter()).max(1.0);
            let err2 = max_abs(
                a2.iter()
                    .zip(&fd2)
                    .map(|(a, f)| a - f)
                    .collect::<Vec<_>>()
                    .iter(),
            );
            worst2 = worst2.max(err2 / scale2);
            let c2 = eri2_deriv_atom(Operator::coulomb(), &aux, atom, x);
            gap2 = gap2.max(max_abs((&c2 - &a2).iter()));
        }
    }
    eprintln!(
        "erfc(ω={OMEGA:.6}) deriv vs FD: 3c {worst3:.2e}, 2c {worst2:.2e} (bar {FD_BAR:.0e}); \
         max|Coulomb − erfc|: 3c {gap3:.2e}, 2c {gap2:.2e}"
    );
    assert!(worst3 < FD_BAR, "erfc 3c derivative vs FD: {worst3:.3e}");
    assert!(worst2 < FD_BAR, "erfc 2c derivative vs FD: {worst2:.3e}");
    assert!(
        gap3 > 1e3 * FD_BAR && gap2 > 1e3 * FD_BAR,
        "erfc and Coulomb derivatives are indistinguishable: 3c {gap3:.3e}, 2c {gap2:.3e}"
    );
}
