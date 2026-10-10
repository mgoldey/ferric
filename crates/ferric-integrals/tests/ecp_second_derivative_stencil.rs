//! Sizing probe for an analytic ECP Hessian (benchmarks/hessian-sizing/REPORT.md):
//! how well does a fourth-order central stencil of the analytic FIRST-derivative
//! ECP matrices (`ecp_potential_deriv`) reproduce the second derivatives of
//! `V_ECP`, on the active ECP backend (`FERRIC_ECP_BACKEND`, default quadrature)?
//!
//! Three numbers are printed per backend, all on HI/def2-SVP at a tilted
//! geometry so every Cartesian component is nonzero:
//!
//! * `step`  = max |S(h) - S(2h)|: stencil self-consistency (truncation + the
//!   backend's integral jitter divided by h);
//! * `val`   = max |S(h) - D2(V)|: stencil of first derivatives against an
//!   INDEPENDENT construction, the fourth-order second difference of the VALUE
//!   matrices (atom diagonal blocks `d²V/dR_a,x dR_a,y` and the cross pair);
//! * the largest second-derivative element, for scale.
//!
//! Run: `OPENBLAS_NUM_THREADS=1 cargo test -p ferric-integrals --test
//! ecp_second_derivative_stencil -- --ignored --nocapture` and again with
//! `FERRIC_ECP_BACKEND=libecpint`.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::oneelectron::{ecp_potential, ecp_potential_deriv};
use ndarray::Array2;

// Three distinct atoms, so (H shell, H shell | I ECP) and (H shell, I shell | I ECP)
// triples with the ECP centre off both shells are exercised.
const XYZ: &str = "3\n\nH 0.30 -0.40 0.20\nI 0.00 0.00 1.61\nH -1.10 0.70 2.90\n";

fn displaced(mol: &Molecule, atom: usize, axis: usize, d: f64) -> Molecule {
    let mut m = mol.clone();
    match axis {
        0 => m.atoms[atom].x += d,
        1 => m.atoms[atom].y += d,
        _ => m.atoms[atom].zpos += d,
    }
    m
}

fn max_abs(a: &Array2<f64>) -> f64 {
    a.iter().fold(0.0f64, |m, v| m.max(v.abs()))
}

/// Stencil of `dV/dR_{a,x}` with respect to coordinate (b, y).
fn stencil_first(
    mol: &Molecule,
    bs: &basis::BasisSet,
    (a, x): (usize, usize),
    (b, y): (usize, usize),
    h: f64,
) -> Array2<f64> {
    let w = [(-2.0, 1.0), (-1.0, -8.0), (1.0, 8.0), (2.0, -1.0)];
    let mut out: Option<Array2<f64>> = None;
    for (off, wt) in w {
        let m = displaced(mol, b, y, off * h);
        let d = ecp_potential_deriv(&m, bs).unwrap().unwrap();
        let term = &d[a][x] * (wt / (12.0 * h));
        out = Some(match out {
            Some(o) => o + term,
            None => term,
        });
    }
    out.unwrap()
}

/// Fourth-order second difference of the value matrix, same coordinate twice
/// (`(a,x)` = `(b,y)`).
fn second_difference_diag(
    mol: &Molecule,
    bs: &basis::BasisSet,
    (b, y): (usize, usize),
    h: f64,
) -> Array2<f64> {
    let c = [
        (-2.0, -1.0),
        (-1.0, 16.0),
        (0.0, -30.0),
        (1.0, 16.0),
        (2.0, -1.0),
    ];
    let mut out: Option<Array2<f64>> = None;
    for (off, wt) in c {
        let v = ecp_potential(&displaced(mol, b, y, off * h), bs).expect("V_ECP");
        let term = v * (wt / (12.0 * h * h));
        out = Some(match out {
            Some(o) => o + term,
            None => term,
        });
    }
    out.unwrap()
}

/// Fourth-order (Richardson) mixed second difference of the value matrices for
/// coordinates `(a, x)` and `(b, y)`.
fn second_difference_mixed(
    mol: &Molecule,
    bs: &basis::BasisSet,
    (a, x): (usize, usize),
    (b, y): (usize, usize),
    h: f64,
) -> Array2<f64> {
    let m = |h: f64| -> Array2<f64> {
        let mut out: Option<Array2<f64>> = None;
        for (sa, sb) in [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)] {
            let g = displaced(&displaced(mol, a, x, sa * h), b, y, sb * h);
            let v = ecp_potential(&g, bs).expect("V_ECP") * (sa * sb / (4.0 * h * h));
            out = Some(match out {
                Some(o) => o + v,
                None => v,
            });
        }
        out.unwrap()
    };
    (4.0 * m(h) - m(2.0 * h)) / 3.0
}

#[test]
#[ignore = "sizing probe (~25 s quadrature, ~150 s libecpint): see benchmarks/hessian-sizing/REPORT.md"]
fn ecp_first_derivative_stencil_gives_second_derivatives() {
    let bs = basis::bundled("def2-svp").unwrap();
    let mut mol = Molecule::parse_xyz(XYZ, 1, 1).unwrap();
    mol.apply_ecp(&bs);
    assert!(mol.atoms[1].n_core_ecp > 0, "I must carry an ECP");
    let (h1, h2) = (2e-3, 4e-3);
    let (mut worst_step, mut worst_val, mut scale) = (0.0f64, 0.0f64, 0.0f64);
    let n = mol.atoms.len();
    for (a, x, b, y) in (0..n).flat_map(|a| {
        (0..3).flat_map(move |x| (a..n).flat_map(move |b| (0..3).map(move |y| (a, x, b, y))))
    }) {
        if (a, x) > (b, y) {
            continue;
        }
        let s1 = stencil_first(&mol, &bs, (a, x), (b, y), h1);
        let s2 = stencil_first(&mol, &bs, (a, x), (b, y), h2);
        let d2 = if (a, x) == (b, y) {
            second_difference_diag(&mol, &bs, (a, x), 1e-2)
        } else {
            second_difference_mixed(&mol, &bs, (a, x), (b, y), 1e-2)
        };
        // S(h) is only symmetric up to the backend's jitter; the value D2 is exactly.
        let (s1s, s2s) = ((&s1 + &s1.t()) * 0.5, (&s2 + &s2.t()) * 0.5);
        worst_step = worst_step.max(max_abs(&(&s1s - &s2s)));
        worst_val = worst_val.max(max_abs(&(&s1s - &d2)));
        scale = scale.max(max_abs(&s1s));
    }
    eprintln!(
        "ECP d2V/dR dR (HIH/def2-SVP, all atom pairs, h={h1:.0e}): max |d2V| {scale:.3e}; \
         |S(h)-S(2h)| {worst_step:.3e}; |S(h)-D2(values,1e-2)| {worst_val:.3e}"
    );
    assert!(scale > 1e-3, "vacuous probe");
    assert!(
        worst_val < 1e-5,
        "stencil vs value second difference: {worst_val:.3e}"
    );
}
