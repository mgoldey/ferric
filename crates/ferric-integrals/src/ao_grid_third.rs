//! Third spatial derivatives of the AOs on arbitrary points, for the
//! fixed-grid GGA Kohn-Sham Hessian (issue #407).
//!
//! Every AO is `Σ_terms coef · x^a y^b z^c · Σ_p c_p N_p exp(-α_p r²)`, a
//! polynomial times a sum of Gaussians, and `exp(-α r²)` factorizes over the
//! axes. The derivative `∂x^i ∂y^j ∂z^k` of one primitive is therefore the
//! product of three 1-D derivatives `∂^n [t^a exp(-α t²)]`, which obey
//! `∂^{n+1}[t^a e] = a ∂^n[t^{a-1} e] - 2α ∂^n[t^{a+1} e]`. That gives every
//! derivative order for every `l <= 4` from one short recursion instead of a
//! hand-unrolled formula per shell; the pure-shell polynomials are the same
//! real solid harmonics `ao_grid::eval_shell` uses (sympy-expanded, so a unit
//! test pins this module against `eval_shell_grad_hess` to rounding).
//!
//! Only the third-order planes are exposed: the value, gradient and Hessian
//! come from [`crate::ao_grid`].

use crate::ao_grid::{collect_shells, AoGridKind, GtoEvalError, LocatedShell};
use ferric_core::basis::{num_functions, BasisSet};
use ferric_core::mol::Molecule;
use ndarray::Array3;

/// The ten unique third derivatives `(nx, ny, nz)` in PySCF/libxc order:
/// xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz.
pub const THIRD_ORDER: [(usize, usize, usize); 10] = [
    (3, 0, 0),
    (2, 1, 0),
    (2, 0, 1),
    (1, 2, 0),
    (1, 1, 1),
    (1, 0, 2),
    (0, 3, 0),
    (0, 2, 1),
    (0, 1, 2),
    (0, 0, 3),
];

/// Largest `l` this module supports (the same bound as `ao_grid`).
const MAX_L: i32 = 4;

/// Highest power of a coordinate in the base table: `l` plus the 3 derivatives.
const MAX_POW: usize = MAX_L as usize + 3;

type Term = (f64, u8, u8, u8);

// pure l=2
const PURE2: [&[Term]; 5] = [
    &[(1.7320508075688772, 1, 1, 0)],
    &[(1.7320508075688772, 0, 1, 1)],
    &[(1.0, 0, 0, 2), (-0.5, 0, 2, 0), (-0.5, 2, 0, 0)],
    &[(1.7320508075688772, 1, 0, 1)],
    &[
        (-0.8660254037844386, 0, 2, 0),
        (0.8660254037844386, 2, 0, 0),
    ],
];
// pure l=3
const PURE3: [&[Term]; 7] = [
    &[
        (-0.7905694150420949, 0, 3, 0),
        (2.3717082451262845, 2, 1, 0),
    ],
    &[(3.872983346207417, 1, 1, 1)],
    &[
        (2.449489742783178, 0, 1, 2),
        (-0.6123724356957945, 0, 3, 0),
        (-0.6123724356957945, 2, 1, 0),
    ],
    &[(1.0, 0, 0, 3), (-1.5, 0, 2, 1), (-1.5, 2, 0, 1)],
    &[
        (2.449489742783178, 1, 0, 2),
        (-0.6123724356957945, 1, 2, 0),
        (-0.6123724356957945, 3, 0, 0),
    ],
    &[
        (-1.9364916731037085, 0, 2, 1),
        (1.9364916731037085, 2, 0, 1),
    ],
    &[
        (-2.3717082451262845, 1, 2, 0),
        (0.7905694150420949, 3, 0, 0),
    ],
];
// pure l=4
const PURE4: [&[Term]; 9] = [
    &[(-2.958039891549808, 1, 3, 0), (2.958039891549808, 3, 1, 0)],
    &[(-2.091650066335189, 0, 3, 1), (6.274950199005566, 2, 1, 1)],
    &[
        (6.708203932499369, 1, 1, 2),
        (-1.118033988749895, 1, 3, 0),
        (-1.118033988749895, 3, 1, 0),
    ],
    &[
        (3.1622776601683795, 0, 1, 3),
        (-2.3717082451262845, 0, 3, 1),
        (-2.3717082451262845, 2, 1, 1),
    ],
    &[
        (1.0, 0, 0, 4),
        (-3.0, 0, 2, 2),
        (0.375, 0, 4, 0),
        (-3.0, 2, 0, 2),
        (0.75, 2, 2, 0),
        (0.375, 4, 0, 0),
    ],
    &[
        (3.1622776601683795, 1, 0, 3),
        (-2.3717082451262845, 1, 2, 1),
        (-2.3717082451262845, 3, 0, 1),
    ],
    &[
        (-3.3541019662496847, 0, 2, 2),
        (0.5590169943749475, 0, 4, 0),
        (3.3541019662496847, 2, 0, 2),
        (-0.5590169943749475, 4, 0, 0),
    ],
    &[(-6.274950199005566, 1, 2, 1), (2.091650066335189, 3, 0, 1)],
    &[
        (0.739509972887452, 0, 4, 0),
        (-4.437059837324712, 2, 2, 0),
        (0.739509972887452, 4, 0, 0),
    ],
];

/// Monomial exponents `(a, b, c)` of Cartesian function `k` of a shell of
/// angular momentum `l`, in libint2 order (x power descending, then y).
fn cartesian_monomial(l: usize, k: usize) -> (u8, u8, u8) {
    let mut idx = 0;
    for a in (0..=l).rev() {
        for b in (0..=l - a).rev() {
            if idx == k {
                return (a as u8, b as u8, (l - a - b) as u8);
            }
            idx += 1;
        }
    }
    unreachable!("function index {k} out of range for l = {l}")
}

/// `N(α, l) = (2α/π)^(3/4) (4α)^(l/2) / √((2l-1)!!)`: the primitive
/// normalization `ao_grid::radial` applies.
fn primitive_norm(alpha: f64, l: i32) -> f64 {
    let dbl_fact: f64 = match l {
        2 => 3.0,
        3 => 15.0,
        4 => 105.0,
        _ => 1.0,
    };
    (2.0 * alpha / std::f64::consts::PI).powf(0.75) * (4.0 * alpha).powi(l).sqrt() / dbl_fact.sqrt()
}

/// `table[n][a] = ∂^n [t^a exp(-α t²)]` at `t`, `n <= 3`, `a <= MAX_L`.
fn one_dim(alpha: f64, t: f64) -> [[f64; MAX_L as usize + 1]; 4] {
    let e = (-alpha * t * t).exp();
    // Base row n = 0 for a = 0..=MAX_POW, then lower the power window by one
    // per derivative order: row n needs a <= MAX_POW - n.
    let mut cur = [0.0f64; MAX_POW + 1];
    let mut p = 1.0;
    for slot in cur.iter_mut() {
        *slot = p * e;
        p *= t;
    }
    let mut out = [[0.0; MAX_L as usize + 1]; 4];
    for n in 0..4 {
        out[n].copy_from_slice(&cur[..=MAX_L as usize]);
        let mut next = [0.0f64; MAX_POW + 1];
        for a in 0..MAX_POW - n {
            let down = if a > 0 { a as f64 * cur[a - 1] } else { 0.0 };
            next[a] = down - 2.0 * alpha * cur[a + 1];
        }
        cur = next;
    }
    out
}

/// Accumulate, for every derivative `(nx, ny, nz)` in `combos` and function
/// `i` of `sh`, the derivative at the offset `(dx, dy, dz)` through `put`.
fn accumulate(
    sh: &LocatedShell,
    (dx, dy, dz): (f64, f64, f64),
    combos: &[(usize, usize, usize)],
    mut put: impl FnMut(usize, usize, f64),
) {
    let l = sh.l;
    let nfun = num_functions(l, sh.pure);
    let poly: Option<&[&[Term]]> = match (l, sh.pure) {
        (2, true) => Some(&PURE2),
        (3, true) => Some(&PURE3),
        (4, true) => Some(&PURE4),
        _ => None,
    };
    for (alpha, coef) in sh.exponents.iter().zip(sh.coefficients) {
        let w = coef * primitive_norm(*alpha, l);
        let (fx, fy, fz) = (
            one_dim(*alpha, dx),
            one_dim(*alpha, dy),
            one_dim(*alpha, dz),
        );
        for (d, &(nx, ny, nz)) in combos.iter().enumerate() {
            for i in 0..nfun {
                let cart;
                let terms: &[Term] = match poly {
                    Some(p) => p[i],
                    None => {
                        let (a, b, c) = cartesian_monomial(l as usize, i);
                        cart = [(1.0, a, b, c)];
                        &cart
                    }
                };
                let acc: f64 = terms
                    .iter()
                    .map(|&(c, a, b, cc)| {
                        c * fx[nx][a as usize] * fy[ny][b as usize] * fz[nz][cc as usize]
                    })
                    .sum();
                put(d, i, w * acc);
            }
        }
    }
}

/// The ten unique third derivatives of the functions of `sh` at the offset
/// `(dx, dy, dz)` from its centre: `out[d][i]` is `THIRD_ORDER[d]` of function
/// `i`. The caller zeroes `out`.
pub fn eval_shell_third(sh: &LocatedShell, dx: f64, dy: f64, dz: f64, out: &mut [[f64; 15]; 10]) {
    accumulate(sh, (dx, dy, dz), &THIRD_ORDER, |d, i, v| out[d][i] += v);
}

/// Derivatives of total order `order` (any `0..=3`) of one shell, `[combo][function]`
/// with the combos in x-power-descending order (order 1: x y z; order 2:
/// xx xy xz yy yz zz). Oracle support for pinning this module to `ao_grid`'s
/// analytic orders 0 to 2.
#[doc(hidden)]
pub fn eval_shell_orders(
    sh: &LocatedShell,
    dx: f64,
    dy: f64,
    dz: f64,
    order: usize,
) -> Vec<Vec<f64>> {
    let mut combos = Vec::new();
    for nx in (0..=order).rev() {
        for ny in (0..=order - nx).rev() {
            combos.push((nx, ny, order - nx - ny));
        }
    }
    let mut out = vec![vec![0.0; num_functions(sh.l, sh.pure)]; combos.len()];
    accumulate(sh, (dx, dy, dz), &combos, |d, i, v| out[d][i] += v);
    out
}

/// The ten unique third derivatives of every AO at `points`: shape
/// `(10, nbasis, npts)`, plane order [`THIRD_ORDER`]. Same function order as
/// [`crate::ao_grid::eval_basis_on_points`].
pub fn eval_basis_third_on_points(
    mol: &Molecule,
    bs: &BasisSet,
    points: &[[f64; 3]],
) -> Result<Array3<f64>, GtoEvalError> {
    use rayon::prelude::*;
    let shells = collect_shells(mol, bs)?;
    if let Some(sh) = shells.iter().find(|s| s.l > MAX_L) {
        return Err(GtoEvalError::UnsupportedL { l: sh.l });
    }
    let nbf: usize = shells.iter().map(|s| num_functions(s.l, s.pure)).sum();
    crate::ao_grid::check_ao_grid_budget(AoGridKind::Third, nbf, points.len())?;
    // Evaluate per point (rows of the transposed table), then transpose.
    let cols: Vec<Vec<f64>> = points
        .par_iter()
        .map(|p| {
            let mut col = vec![0.0; 10 * nbf];
            let mut row0 = 0;
            for sh in &shells {
                let mut buf = [[0.0f64; 15]; 10];
                eval_shell_third(
                    sh,
                    p[0] - sh.center[0],
                    p[1] - sh.center[1],
                    p[2] - sh.center[2],
                    &mut buf,
                );
                let n = num_functions(sh.l, sh.pure);
                for d in 0..10 {
                    col[d * nbf + row0..d * nbf + row0 + n].copy_from_slice(&buf[d][..n]);
                }
                row0 += n;
            }
            col
        })
        .collect();
    let mut out = Array3::<f64>::zeros((10, nbf, points.len()));
    for (g, col) in cols.iter().enumerate() {
        for d in 0..10 {
            for mu in 0..nbf {
                out[(d, mu, g)] = col[d * nbf + mu];
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ao_grid::eval_shell_grad_hess;

    const EXPS: [f64; 3] = [3.1, 0.9, 0.27];
    const COEFS: [f64; 3] = [0.35, 0.55, 0.4];
    const POINT: [f64; 3] = [0.43, -0.71, 0.58];

    fn shell(l: i32, pure: bool) -> LocatedShell<'static> {
        LocatedShell {
            l,
            pure,
            exponents: &EXPS,
            coefficients: &COEFS,
            center: [0.0; 3],
        }
    }

    /// Every supported shell kind (ferric never builds a pure `l < 2` shell).
    fn kinds() -> Vec<(i32, bool)> {
        (0..=4)
            .flat_map(|l| [(l, false), (l, true)])
            .filter(|&(l, pure)| !(pure && l < 2))
            .collect()
    }

    fn hessian_at(sh: &LocatedShell, p: [f64; 3]) -> [[f64; 15]; 9] {
        let n = num_functions(sh.l, sh.pure);
        let mut val = [0.0; 15];
        let mut grad = [[0.0; 15]; 3];
        let mut hess = [[0.0; 15]; 9];
        eval_shell_grad_hess(sh, p[0], p[1], p[2], &mut val[..n], &mut grad, &mut hess).unwrap();
        hess
    }

    /// Exactness anchor: orders 0 to 2 of this module are `ao_grid`'s analytic
    /// value, gradient and Hessian for every supported shell.
    #[test]
    fn orders_zero_to_two_match_the_analytic_ao_grid() {
        let [dx, dy, dz] = POINT;
        let pairs = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)];
        for (l, pure) in kinds() {
            let sh = shell(l, pure);
            let n = num_functions(l, pure);
            let mut val = [0.0; 15];
            let mut grad = [[0.0; 15]; 3];
            let mut hess = [[0.0; 15]; 9];
            eval_shell_grad_hess(&sh, dx, dy, dz, &mut val[..n], &mut grad, &mut hess).unwrap();
            let o0 = eval_shell_orders(&sh, dx, dy, dz, 0);
            let o1 = eval_shell_orders(&sh, dx, dy, dz, 1);
            let o2 = eval_shell_orders(&sh, dx, dy, dz, 2);
            for i in 0..n {
                assert!((o0[0][i] - val[i]).abs() < 1e-13, "l={l} pure={pure} value");
                for a in 0..3 {
                    assert!((o1[a][i] - grad[a][i]).abs() < 1e-13, "l={l} gradient");
                }
                for (k, &(a, b)) in pairs.iter().enumerate() {
                    let d = (o2[k][i] - hess[3 * a + b][i]).abs();
                    assert!(d < 1e-12, "l={l} pure={pure} hess ({a},{b}) fn {i}: {d:e}");
                }
            }
        }
    }

    /// The third derivatives are the (Richardson) central difference of the
    /// analytic Hessian along one axis of the multiset.
    #[test]
    fn third_derivatives_match_a_difference_of_the_analytic_hessian() {
        let h = 2e-3;
        for (l, pure) in kinds() {
            let sh = shell(l, pure);
            let n = num_functions(l, pure);
            let mut buf = [[0.0f64; 15]; 10];
            eval_shell_third(&sh, POINT[0], POINT[1], POINT[2], &mut buf);
            let mut worst = 0.0f64;
            for (d, &(nx, ny, nz)) in THIRD_ORDER.iter().enumerate() {
                // The multiset {g, a, b} of axes: differentiate H[a][b] along g.
                let mut axes = Vec::new();
                for (ax, cnt) in [(0, nx), (1, ny), (2, nz)] {
                    axes.extend(std::iter::repeat_with(|| ax).take(cnt));
                }
                let (g, a, b) = (axes[0], axes[1], axes[2]);
                let diff = |step: f64| -> Vec<f64> {
                    let (mut pp, mut pm) = (POINT, POINT);
                    pp[g] += step;
                    pm[g] -= step;
                    let (hp, hm) = (hessian_at(&sh, pp), hessian_at(&sh, pm));
                    (0..n)
                        .map(|i| (hp[3 * a + b][i] - hm[3 * a + b][i]) / (2.0 * step))
                        .collect()
                };
                let (coarse, fine) = (diff(h), diff(0.5 * h));
                for i in 0..n {
                    let fd = (4.0 * fine[i] - coarse[i]) / 3.0;
                    worst = worst.max((fd - buf[d][i]).abs() / (1.0 + fd.abs()));
                }
            }
            assert!(worst < 1e-8, "l={l} pure={pure}: {worst:e}");
        }
    }
}
