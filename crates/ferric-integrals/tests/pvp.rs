//! Tests of `ferric_integrals::pvp`: the pVp integral
//! `W_mn = sum_a <d_a m | V | d_a n>`.
//!
//! * closed forms for same-centre s/s and p/p primitives
//!   (`scripts/pvp_ss_sympy.py`);
//! * an independent construction: the mixed central difference of ordinary
//!   nuclear-attraction integrals with the bra and ket basis centres
//!   displaced separately, `d^2 V_mn / dA_a dB_a`, on coincident and on
//!   separate centres, with s, p and d shells;
//! * symmetry, and linearity in the charges across the chunk boundary.

use ferric_core::basis::{BasisSet, Shell};
use ferric_core::external_potential::PointCharge;
use ferric_core::mol::{Atom, Molecule};
use ferric_core::special::erf;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::{libint_max_deriv_order, Engine};
use ferric_integrals::ffi;
use ferric_integrals::pvp::{pvp, CHARGE_CHUNK};
use ndarray::Array2;
use std::collections::HashMap;
use std::f64::consts::PI;

/// libint2's engine precision is 1e-14 per primitive integral; W sums three
/// derivative blocks, each a short sum of such integrals, so a relative
/// bar of 10 x 3 x 1e-14 bounds the rounding of one element.
const ROUNDING_REL: f64 = 3e-13;

fn skip() -> bool {
    if libint_max_deriv_order() < 2 {
        eprintln!("SKIP: libint2 generated without second derivatives");
        return true;
    }
    false
}

fn shell(l: i32, exps: &[f64]) -> Shell {
    Shell {
        l,
        pure: true,
        exponents: exps.to_vec(),
        coefficients: vec![1.0; exps.len()],
    }
}

/// Ghost centres (no charge) at `pos`, centre `k` carrying the shells
/// `shells[k]` (keyed by a distinct fake Z = k + 1).
fn ghost_prep(pos: &[[f64; 3]], shells: &[Vec<Shell>]) -> PreparedBasis {
    let atoms = pos
        .iter()
        .enumerate()
        .map(|(k, p)| Atom {
            symbol: "X".into(),
            z: k as i32 + 1,
            x: p[0],
            y: p[1],
            zpos: p[2],
            ghost: true,
            n_core_ecp: 0,
        })
        .collect();
    let mol = Molecule {
        atoms,
        charge: 0,
        multiplicity: 1,
    };
    let mut map = HashMap::new();
    for (k, s) in shells.iter().enumerate() {
        map.insert(k as i32 + 1, s.clone());
    }
    let bs = BasisSet {
        name: "pvp-test".into(),
        shells: map,
        ecps: HashMap::new(),
    };
    PreparedBasis::new(&mol, &bs).expect("prep")
}

fn pc(q: f64, x: f64, y: f64, z: f64) -> PointCharge {
    PointCharge { q, x, y, z }
}

/// Ordinary nuclear-attraction matrix over exactly `charges`.
fn nuclear_over(prep: &PreparedBasis, charges: &[PointCharge]) -> Array2<f64> {
    let n = prep.nbasis();
    let dims = prep.shell_dims().to_vec();
    let offs = prep.shell_offsets().to_vec();
    let mut eng = Engine::new_1e(ffi::OP_NUCLEAR, prep, 1e-14).unwrap();
    eng.set_point_charges_only(charges).unwrap();
    let mut out = Array2::zeros((n, n));
    for s1 in 0..dims.len() {
        for s2 in 0..dims.len() {
            let blk = eng.compute_1e_block(prep, s1, s2).to_vec();
            for i in 0..dims[s1] {
                for j in 0..dims[s2] {
                    out[(offs[s1] + i, offs[s2] + j)] = blk[i * dims[s2] + j];
                }
            }
        }
    }
    out
}

/// `I2(p, D) = int r_A^2 exp(-p r_A^2) / |r - C| d^3r` (pvp_ss_sympy.py, 1).
fn i2(p: f64, d: f64) -> f64 {
    if d == 0.0 {
        2.0 * PI / (p * p)
    } else {
        3.0 * PI.powf(1.5) * erf(p.sqrt() * d) / (2.0 * d * p.powf(2.5))
            - PI * (-p * d * d).exp() / (p * p)
    }
}

fn norm_s(a: f64) -> f64 {
    (2.0 * a / PI).powf(0.75)
}

fn norm_p(a: f64) -> f64 {
    norm_s(a) * 2.0 * a.sqrt()
}

#[test]
fn same_centre_s_and_p_match_the_closed_forms() {
    if skip() {
        return;
    }
    let (a, b) = (0.7, 1.3);
    let centre = [0.3, -0.2, 0.5];
    let prep = ghost_prep(
        &[centre],
        &[vec![
            shell(0, &[a]),
            shell(0, &[b]),
            shell(1, &[a]),
            shell(1, &[b]),
        ]],
    );
    // A charge on the centre, then one off it (D = 0.9 and 1.7).
    let cases: [(f64, [f64; 3]); 3] = [
        (1.0, [0.0, 0.0, 0.0]),
        (1.0, [0.0, 0.9, 0.0]),
        (2.0, [1.0, 1.2, -0.6]),
    ];
    for (q, off) in cases {
        let d = (off[0] * off[0] + off[1] * off[1] + off[2] * off[2]).sqrt();
        let c = pc(
            q,
            centre[0] + off[0],
            centre[1] + off[1],
            centre[2] + off[2],
        );
        let w = pvp(&prep, &[c]).unwrap();
        for (i, j, ea, eb) in [(0, 0, a, a), (0, 1, a, b), (1, 1, b, b)] {
            let want = -4.0 * ea * eb * q * norm_s(ea) * norm_s(eb) * i2(ea + eb, d);
            let got = w[(i, j)];
            eprintln!(
                "s/s ({i},{j}) D={d:.3}: rel dev {:.2e}",
                (got - want).abs() / want.abs()
            );
            assert!(
                (got - want).abs() <= ROUNDING_REL * want.abs(),
                "s/s ({i},{j}) D={d}: pvp {got:.16e} closed form {want:.16e}"
            );
        }
        if d == 0.0 {
            // p/p trace over the three components of the (a, b) shell pair.
            let p = a + b;
            let want = -q
                * norm_p(a)
                * norm_p(b)
                * 4.0
                * PI
                * (1.5 / p - 1.0 / p + 4.0 * a * b / p.powi(3));
            let got: f64 = (0..3).map(|k| w[(2 + k, 5 + k)]).sum();
            eprintln!("p/p trace: rel dev {:.2e}", (got - want).abs() / want.abs());
            assert!(
                (got - want).abs() <= ROUNDING_REL * want.abs(),
                "p/p trace: pvp {got:.16e} closed form {want:.16e}"
            );
        }
    }
}

/// Mixed central difference `sum_a d^2 V_mn / dA_a dB_a` with bra centre 0
/// and ket centre 1 displaced separately, at step `h`.
fn fd_cross_block(
    pos: &[[f64; 3]; 2],
    shells: &[Vec<Shell>],
    charges: &[PointCharge],
    h: f64,
) -> Array2<f64> {
    let nb0 = ghost_prep(&[pos[0]], &shells[..1]).nbasis();
    let mut acc: Option<Array2<f64>> = None;
    for a in 0..3 {
        for (sa, sb, sign) in [
            (1.0, 1.0, 1.0),
            (1.0, -1.0, -1.0),
            (-1.0, 1.0, -1.0),
            (-1.0, -1.0, 1.0),
        ] {
            let mut p = *pos;
            p[0][a] += sa * h;
            p[1][a] += sb * h;
            let v = nuclear_over(&ghost_prep(&p, shells), charges);
            let blk = v.slice(ndarray::s![..nb0, nb0..]).to_owned() * (sign / (4.0 * h * h));
            acc = Some(match acc {
                None => blk,
                Some(x) => x + blk,
            });
        }
    }
    acc.unwrap()
}

#[test]
fn pvp_matches_mixed_fd_of_nuclear_attraction() {
    if skip() {
        return;
    }
    let shells = vec![
        vec![shell(0, &[1.1]), shell(1, &[0.8]), shell(2, &[0.9])],
        vec![shell(0, &[0.6]), shell(1, &[1.4]), shell(2, &[0.7])],
    ];
    let charges = [
        pc(3.0, 0.0, 0.0, 0.0),
        pc(1.0, 1.3, -0.4, 0.8),
        pc(0.5, -0.9, 1.1, -0.3),
    ];
    let a = [0.2, 0.1, -0.3];
    let separate = [0.9, -0.5, 0.6];
    for (label, pos) in [("coincident", [a, a]), ("separate", [a, separate])] {
        let prep = ghost_prep(&pos, &shells);
        let nb0 = ghost_prep(&[pos[0]], &shells[..1]).nbasis();
        let w = pvp(&prep, &charges).unwrap();
        let wb = w.slice(ndarray::s![..nb0, nb0..]).to_owned();
        let h = 2e-3;
        let fd_h = fd_cross_block(&pos, &shells, &charges, h);
        let fd_2h = fd_cross_block(&pos, &shells, &charges, 2.0 * h);
        // Central differences: error(2h) = 4 error(h) + O(h^4), so
        // |fd(2h) - fd(h)| = 3 |error(h)|; allow that plus the round-off
        // of a second difference, ~eps |V| / h^2 with |V| <= 10.
        let roundoff = 10.0 * f64::EPSILON * 10.0 / (h * h);
        let mut worst = 0.0f64;
        for ((&x, &f1), &f2) in wb.iter().zip(fd_h.iter()).zip(fd_2h.iter()) {
            let bar = (f2 - f1).abs() + roundoff;
            worst = worst.max((x - f1).abs() / bar);
        }
        let scale = wb.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        eprintln!("{label}: max |W| {scale:.3e}, worst |W - FD(h)| / bar = {worst:.3e}");
        assert!(scale > 0.1, "{label}: block is trivially small ({scale})");
        assert!(
            worst <= 1.0,
            "{label}: pvp vs FD exceeds the derived bar by {worst:.2}x"
        );
    }
}

#[test]
fn pvp_is_symmetric_and_linear_in_the_charges() {
    if skip() {
        return;
    }
    let shells = vec![
        vec![shell(0, &[3.0, 0.5]), shell(1, &[0.8]), shell(2, &[0.9])],
        vec![shell(0, &[0.6]), shell(1, &[1.4, 0.3])],
    ];
    let prep = ghost_prep(&[[0.0, 0.0, 0.0], [0.4, 1.1, -0.7]], &shells);
    // More charges than one chunk, so the sum crosses a chunk boundary.
    let charges: Vec<PointCharge> = (0..CHARGE_CHUNK + 3)
        .map(|k| {
            let t = k as f64;
            pc(
                0.5 + 0.25 * t,
                (0.7 * t).sin(),
                (1.3 * t).cos(),
                0.2 * t - 1.0,
            )
        })
        .collect();
    let w = pvp(&prep, &charges).unwrap();
    let scale = w.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let asym = (&w - &w.t()).iter().fold(0.0f64, |m, v| m.max(v.abs()));
    assert!(
        asym <= ROUNDING_REL * scale,
        "asymmetry {asym:.3e} at scale {scale:.3e}"
    );
    let mut sum = Array2::<f64>::zeros(w.dim());
    for c in &charges {
        sum = sum + pvp(&prep, std::slice::from_ref(c)).unwrap();
    }
    let dev = (&w - &sum).iter().fold(0.0f64, |m, v| m.max(v.abs()));
    eprintln!("scale {scale:.3e}, asymmetry {asym:.2e}, chunked-vs-single {dev:.2e}");
    assert!(
        dev <= ROUNDING_REL * scale * charges.len() as f64,
        "chunked sum differs from the per-charge sum by {dev:.3e}"
    );
}
