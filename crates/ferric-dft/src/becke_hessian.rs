//! First and second nuclear derivatives of a Becke home-cell weight with the
//! point held FIXED in the lab frame.
//!
//! The moving-grid analytic Kohn-Sham Hessian (issue #490) needs, for every
//! grid point `r` of home atom `h`, the derivatives of
//! `w^h(r) = P_h(r) / Σ_C P_C(r)` with respect to the nuclei other than `h`
//! at fixed `r` (the point rides `h`, so moving another nucleus leaves it in
//! place; the `h` rows follow from translation invariance in the caller). The
//! partition is exactly [`crate::becke::becke_weights_all`]'s: Becke 1988, three
//! smoothing iterations, Bragg radii, size adjustment clamped to `|a| ≤ ½`.
//! Derivatives are returned for EVERY nucleus, the home one included (also at
//! fixed `r`), so the first derivatives are those of
//! [`crate::becke::becke_weights_and_grad`] for the home row.
//!
//! Construction. With `μ_AB = (r_A − r_B)/R_AB` (`r_X = |r − R_X|`),
//! `ν = μ + a_AB(1 − μ²)`, `s_AB = ½(1 − f(f(f(ν))))`, each pair `s_AB` is a
//! function of the six coordinates `(R_A, R_B)` only; its local gradient and
//! 6×6 Hessian follow from
//!
//! ```text
//! μ_x  = N_x/R − μ R_x/R,                        N = r_A − r_B, R = R_AB
//! μ_xy = N_xy/R − (N_x R_y + N_y R_x)/R² − μ R_xy/R + 2μ R_x R_y/R²
//! s_x  = s_μ μ_x,   s_xy = s_μμ μ_x μ_y + s_μ μ_xy
//! ```
//!
//! The cell `P_A = Π_B s_AB` is differentiated in log form,
//! `P' = P L'`, `P'' = P (L'' + L' L'ᵀ)` with `L = Σ_B ln s_AB`, and
//! `w = P_h/T`, `T = Σ_C P_C`:
//!
//! ```text
//! w'  = P_h'/T − P_h T'/T²
//! w'' = P_h''/T − (P_h' T'ᵀ + T' P_h'ᵀ)/T² − P_h T''/T² + 2 P_h T' T'ᵀ/T³
//! ```
//!
//! A cell with `P_A ≤ CELL_SCREEN · T` contributes its value to `T` but no
//! derivatives: every factor `s` of Becke's step vanishes as `(1 − ν)^8`, so such
//! a cell's first and second derivatives are below `~1e-16 T` (and the log form
//! never divides by a vanishing `s`). Cost `O(n³)` per point for `n` atoms.

use crate::becke::bragg_slater_bohr;
use ferric_core::mol::Molecule;

/// Relative cell screen: cells with `P_A ≤ CELL_SCREEN · Σ_C P_C` carry no
/// derivatives (see the module doc).
pub const CELL_SCREEN: f64 = 1e-24;

/// The home-cell weight of one point and its nuclear derivatives at fixed `r`.
#[derive(Debug, Clone)]
pub struct HomeWeightDerivs {
    /// `w^h(r)` (equal to `becke_weights_all(mol, r)[h]` up to rounding).
    pub weight: f64,
    /// `∂w^h/∂R_{c,k}` at index `3c + k`.
    pub grad: Vec<f64>,
    /// `∂²w^h/∂R_{b,i}∂R_{c,j}`, row-major `(3n, 3n)`.
    pub hess: Vec<f64>,
}

/// One cell factor `s_AB` with its gradient and Hessian in the local
/// coordinates `(R_A,x, R_A,y, R_A,z, R_B,x, R_B,y, R_B,z)`.
struct PairTerm {
    s: f64,
    d: [f64; 6],
    dd: [[f64; 6]; 6],
}

fn atom_pos(mol: &Molecule, a: usize) -> [f64; 3] {
    let at = &mol.atoms[a];
    [at.x, at.y, at.zpos]
}

/// Becke's size adjustment `a_AB` (the expression of `becke_weights_all`).
fn size_adjustment(za: i32, zb: i32) -> f64 {
    let chi = bragg_slater_bohr(za) / bragg_slater_bohr(zb);
    let u = (chi - 1.0) / (chi + 1.0);
    (u / (u * u - 1.0)).clamp(-0.5, 0.5)
}

/// `f(f(f(ν)))` with its first and second derivatives in `ν`.
fn smooth3(nu: f64) -> (f64, f64, f64) {
    let (mut f, mut d1, mut d2) = (nu, 1.0, 0.0);
    for _ in 0..3 {
        let p1 = 1.5 * (1.0 - f * f);
        let p2 = -3.0 * f;
        d2 = p2 * d1 * d1 + p1 * d2;
        d1 *= p1;
        f = 0.5 * f * (3.0 - f * f);
    }
    (f, d1, d2)
}

/// Distance from `c` to `r` and the unit vector `(r − c)/|r − c|` (zero at
/// coincidence) with `(I − u uᵀ)/|r − c|` (zero at coincidence).
fn radial(r: [f64; 3], c: [f64; 3]) -> (f64, [f64; 3], [[f64; 3]; 3]) {
    let v = [r[0] - c[0], r[1] - c[1], r[2] - c[2]];
    let d = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let inv = if d > 1e-30 { 1.0 / d } else { 0.0 };
    let u = [v[0] * inv, v[1] * inv, v[2] * inv];
    let mut p = [[0.0; 3]; 3];
    for (i, row) in p.iter_mut().enumerate() {
        for (j, x) in row.iter_mut().enumerate() {
            *x = (f64::from(u8::from(i == j)) - u[i] * u[j]) * inv;
        }
    }
    (d, u, p)
}

/// `μ_AB` with its local gradient and Hessian.
fn mu_derivs(ra: [f64; 3], rb: [f64; 3], r: [f64; 3]) -> Option<(f64, [f64; 6], [[f64; 6]; 6])> {
    let (da, ua, pa) = radial(r, ra);
    let (db, ub, pb) = radial(r, rb);
    // R_AB and its derivatives: R_x = (e, −e), R_xx = [[K, −K], [−K, K]].
    let (rr, e_neg, k) = radial(rb, ra); // e_neg = (R_B − R_A)/R = −e
    if rr < 1e-12 {
        return None;
    }
    let mu = (da - db) / rr;
    let mut nx = [0.0; 6];
    let mut rx = [0.0; 6];
    for c in 0..3 {
        nx[c] = -ua[c];
        nx[3 + c] = ub[c];
        rx[c] = -e_neg[c];
        rx[3 + c] = e_neg[c];
    }
    let mut mux = [0.0; 6];
    for i in 0..6 {
        mux[i] = nx[i] / rr - mu * rx[i] / rr;
    }
    let mut muxx = [[0.0; 6]; 6];
    for i in 0..6 {
        for j in 0..6 {
            let (bi, bj, ci, cj) = (i / 3, j / 3, i % 3, j % 3);
            let nxx = match (bi, bj) {
                (0, 0) => pa[ci][cj],
                (1, 1) => -pb[ci][cj],
                _ => 0.0,
            };
            let rxx = if bi == bj { k[ci][cj] } else { -k[ci][cj] };
            muxx[i][j] = nxx / rr - (nx[i] * rx[j] + nx[j] * rx[i]) / (rr * rr) - mu * rxx / rr
                + 2.0 * mu * rx[i] * rx[j] / (rr * rr);
        }
    }
    Some((mu, mux, muxx))
}

/// `s_AB` with its local gradient and Hessian, or `None` for coincident nuclei
/// (skipped, as in `becke_weights_all`).
fn pair_term(ra: [f64; 3], rb: [f64; 3], r: [f64; 3], a_corr: f64) -> Option<PairTerm> {
    let (mu, mux, muxx) = mu_derivs(ra, rb, r)?;
    let nu = mu + a_corr * (1.0 - mu * mu);
    let (f, f1, f2) = smooth3(nu);
    let n1 = 1.0 - 2.0 * a_corr * mu;
    let s_mu = -0.5 * f1 * n1;
    let s_mumu = -0.5 * f2 * n1 * n1 + 0.5 * f1 * 2.0 * a_corr;
    let mut d = [0.0; 6];
    let mut dd = [[0.0; 6]; 6];
    for i in 0..6 {
        d[i] = s_mu * mux[i];
        for j in 0..6 {
            dd[i][j] = s_mumu * mux[i] * mux[j] + s_mu * muxx[i][j];
        }
    }
    Some(PairTerm {
        s: 0.5 * (1.0 - f),
        d,
        dd,
    })
}

/// Gradient and Hessian of `ln P_A` (`L'`, `L''`) over the `3n` coordinates.
fn log_cell_derivs(pairs: &[Option<PairTerm>], a: usize, n: usize) -> (Vec<f64>, Vec<f64>) {
    let m = 3 * n;
    let mut lg = vec![0.0; m];
    let mut lh = vec![0.0; m * m];
    for (b, pt) in pairs[a * n..(a + 1) * n].iter().enumerate() {
        let Some(pt) = pt else { continue };
        let inv = 1.0 / pt.s;
        let idx = [3 * a, 3 * a + 1, 3 * a + 2, 3 * b, 3 * b + 1, 3 * b + 2];
        for i in 0..6 {
            lg[idx[i]] += pt.d[i] * inv;
            for j in 0..6 {
                lh[idx[i] * m + idx[j]] += pt.dd[i][j] * inv - pt.d[i] * pt.d[j] * inv * inv;
            }
        }
    }
    (lg, lh)
}

/// The home-cell Becke weight at `r` and its first and second derivatives with
/// respect to every nucleus, the point held fixed. See the module doc.
pub fn becke_home_weight_hess(mol: &Molecule, r: [f64; 3], home: usize) -> HomeWeightDerivs {
    becke_home_weight_hess_screened(mol, r, home, CELL_SCREEN)
}

/// [`becke_home_weight_hess`] with an explicit relative cell screen (0 keeps
/// every cell; the screen's exactness test).
#[doc(hidden)]
pub fn becke_home_weight_hess_screened(
    mol: &Molecule,
    r: [f64; 3],
    home: usize,
    screen: f64,
) -> HomeWeightDerivs {
    let n = mol.atoms.len();
    let m = 3 * n;
    let zero = |w: f64| HomeWeightDerivs {
        weight: w,
        grad: vec![0.0; m],
        hess: vec![0.0; m * m],
    };
    if n <= 1 {
        return zero(if n == 1 { 1.0 } else { 0.0 });
    }
    let mut pairs: Vec<Option<PairTerm>> = Vec::with_capacity(n * n);
    for a in 0..n {
        for b in 0..n {
            pairs.push(if a == b {
                None
            } else {
                let a_corr = size_adjustment(mol.atoms[a].z, mol.atoms[b].z);
                pair_term(atom_pos(mol, a), atom_pos(mol, b), r, a_corr)
            });
        }
    }
    let cells: Vec<f64> = (0..n)
        .map(|a| {
            pairs[a * n..(a + 1) * n]
                .iter()
                .flatten()
                .map(|p| p.s)
                .product()
        })
        .collect();
    let t: f64 = cells.iter().sum();
    if t < 1e-30 {
        return zero(0.0);
    }
    let mut tg = vec![0.0; m];
    let mut th = vec![0.0; m * m];
    let mut pg = vec![0.0; m];
    let mut ph = vec![0.0; m * m];
    for (a, &p) in cells.iter().enumerate() {
        if p <= screen * t || p == 0.0 {
            continue;
        }
        let (lg, lh) = log_cell_derivs(&pairs, a, n);
        let (g_out, h_out) = if a == home {
            (&mut pg, &mut ph)
        } else {
            (&mut tg, &mut th)
        };
        for i in 0..m {
            g_out[i] += p * lg[i];
            for j in 0..m {
                h_out[i * m + j] += p * (lh[i * m + j] + lg[i] * lg[j]);
            }
        }
    }
    // T' and T'' include the home cell.
    for i in 0..m {
        tg[i] += pg[i];
        for j in 0..m {
            th[i * m + j] += ph[i * m + j];
        }
    }
    let ph0 = cells[home];
    let (it, it2) = (1.0 / t, 1.0 / (t * t));
    let grad: Vec<f64> = (0..m).map(|i| pg[i] * it - ph0 * tg[i] * it2).collect();
    let mut hess = vec![0.0; m * m];
    for i in 0..m {
        for j in 0..m {
            hess[i * m + j] = ph[i * m + j] * it
                - (pg[i] * tg[j] + tg[i] * pg[j]) * it2
                - ph0 * th[i * m + j] * it2
                + 2.0 * ph0 * tg[i] * tg[j] * it2 * it;
        }
    }
    HomeWeightDerivs {
        weight: ph0 * it,
        grad,
        hess,
    }
}

#[cfg(test)]
mod tests {
    //! Mutation ledger: see `hessian_matches_difference_of_the_analytic_gradient`.
    use super::*;
    use crate::becke::{becke_weights_all, becke_weights_and_grad};
    use ferric_core::mol::Molecule;

    /// Distorted H2O plus a distant N (four atoms, three elements, no symmetry).
    fn mol4() -> Molecule {
        Molecule::parse_xyz(
            "4\n\nO 0.01 -0.02 0.12\nH 0.03 0.77 -0.46\nH -0.02 -0.74 -0.49\nN 1.9 0.3 0.8\n",
            0,
            2,
        )
        .unwrap()
    }

    /// Points spread around the molecule: near nuclei, in bonds, in cell
    /// boundaries, and outside.
    fn points() -> Vec<[f64; 3]> {
        let mut pts = Vec::new();
        for i in 0..7 {
            for j in 0..5 {
                let t = i as f64 * 0.9 - 2.4;
                let u = j as f64 * 0.8 - 1.5;
                pts.push([t, u + 0.13 * t, 0.37 * u - 0.21 * t + 0.4]);
            }
        }
        pts
    }

    fn displaced(mol: &Molecule, c: usize, h: f64) -> Molecule {
        let mut m = mol.clone();
        match c % 3 {
            0 => m.atoms[c / 3].x += h,
            1 => m.atoms[c / 3].y += h,
            _ => m.atoms[c / 3].zpos += h,
        }
        m
    }

    #[test]
    fn single_atom_has_zero_derivatives() {
        let mol = Molecule::parse_xyz("1\n\nO 0.1 0.2 0.3\n", 0, 1).unwrap();
        let d = becke_home_weight_hess(&mol, [0.5, -0.4, 1.1], 0);
        assert_eq!(d.weight, 1.0);
        assert!(d.grad.iter().chain(&d.hess).all(|&x| x == 0.0));
    }

    /// The value and first derivative reproduce the existing Becke routines.
    #[test]
    fn value_and_gradient_match_becke_weights_and_grad() {
        let mol = mol4();
        let mut worst = (0.0f64, 0.0f64);
        for r in points() {
            let all = becke_weights_all(&mol, r);
            let (_, dw) = becke_weights_and_grad(&mol, r);
            for h in 0..4 {
                let d = becke_home_weight_hess(&mol, r, h);
                worst.0 = worst.0.max((d.weight - all[h]).abs());
                for c in 0..4 {
                    for k in 0..3 {
                        worst.1 = worst.1.max((d.grad[3 * c + k] - dw[h][c][k]).abs());
                    }
                }
            }
        }
        eprintln!("value {:.2e}, gradient {:.2e}", worst.0, worst.1);
        assert!(worst.0 < 1e-14 && worst.1 < 1e-12, "{worst:?}");
    }

    /// The Hessian against a Richardson central difference (h = 1e-3, 5e-4)
    /// of the INDEPENDENT first-derivative routine `becke_weights_and_grad`,
    /// point fixed. Measured max 3.9e-12 over 35 points × 4 homes (max |w''|
    /// 2.3); bar 1e-10. Mutants (made by hand, each run against this test):
    /// `s_μ μ_xy` dropped 1.1; the sign of `2 P_h T'T'ᵀ/T³` flipped 0.48; the
    /// `−s's'ᵀ/s²` log term dropped 3.4; the `(N_x R_y + N_y R_x)/R²` term of
    /// `μ_xy` dropped 0.83.
    #[test]
    fn hessian_matches_difference_of_the_analytic_gradient() {
        let mol = mol4();
        let m = 12;
        let grad_at = |mol: &Molecule, r: [f64; 3], h: usize| -> Vec<f64> {
            let (_, dw) = becke_weights_and_grad(mol, r);
            (0..m).map(|i| dw[h][i / 3][i % 3]).collect()
        };
        let mut worst = 0.0f64;
        let mut largest = 0.0f64;
        for r in points() {
            for h in 0..4 {
                let an = becke_home_weight_hess(&mol, r, h);
                for c in 0..m {
                    let cd = |s: f64| -> Vec<f64> {
                        let p = grad_at(&displaced(&mol, c, s), r, h);
                        let q = grad_at(&displaced(&mol, c, -s), r, h);
                        p.iter().zip(&q).map(|(a, b)| (a - b) / (2.0 * s)).collect()
                    };
                    let (c1, c2) = (cd(1e-3), cd(5e-4));
                    for i in 0..m {
                        let fd = (4.0 * c2[i] - c1[i]) / 3.0;
                        worst = worst.max((an.hess[i * m + c] - fd).abs());
                        largest = largest.max(an.hess[i * m + c].abs());
                    }
                }
            }
        }
        eprintln!(
            "Becke weight Hessian vs FD of the gradient: {worst:.2e} (max |w''| {largest:.2e})"
        );
        assert!(worst < 1e-10, "{worst:.3e}");
        assert!(
            largest > 1e-2,
            "the points must exercise the weight Hessian"
        );
    }

    /// The Hessian is symmetric.
    #[test]
    fn hessian_is_symmetric() {
        let mol = mol4();
        for r in points() {
            let d = becke_home_weight_hess(&mol, r, 2);
            for i in 0..12 {
                for j in 0..12 {
                    assert!((d.hess[i * 12 + j] - d.hess[j * 12 + i]).abs() < 1e-12);
                }
            }
        }
    }

    /// The cell screen changes nothing measurable (it drops only cells below
    /// `1e-24 T`): screened vs unscreened over the test points.
    #[test]
    fn cell_screen_is_inert() {
        let mol = mol4();
        let mut worst = 0.0f64;
        for r in points() {
            for h in 0..4 {
                let a = becke_home_weight_hess(&mol, r, h);
                let b = becke_home_weight_hess_screened(&mol, r, h, 0.0);
                for (x, y) in a.hess.iter().zip(&b.hess) {
                    worst = worst.max((x - y).abs());
                }
            }
        }
        assert!(worst < 1e-14, "{worst:.3e}");
    }
}
