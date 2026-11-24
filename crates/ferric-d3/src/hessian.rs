//! Analytic nuclear Hessian of the two-body D3(BJ) energy.
//!
//! The energy is `E = Σ_{P=(i,j)} -C6_P(CN_i, CN_j) G_P(r_P)` with
//! `G = s6/(r^6+f^6) + 3 q s8/(r^8+f^8)` (`C8 = 3 C6 q` is folded into `G`),
//! and every `CN_a = Σ_b c_ab(r_ab)`. Its second derivative has four parts
//! (`book/src/25-dispersion/01-d3bj.md#d3bj-hessian`):
//!
//! ```text
//! H = Σ_P  pair block of φ_P = -C6_P G_P at fixed C6       (direct)
//!   + Σ_a (u_a ⊗ J_a + J_a ⊗ u_a)                          (mixed: r and CN)
//!   + Σ_ab M_ab J_a ⊗ J_b                                  (CN-CN, second order in C6)
//!   + Σ_P (e_i + e_j) pair block of c_P                    (second derivative of CN)
//! ```
//!
//! with `J_a = ∂CN_a/∂x` (a 3N row), `e_a = ∂E/∂CN_a` (the gradient's
//! `de_dcn`), `u_a = Σ_{P∋a} -G'_P (∂C6_P/∂CN_a) ∂r_P/∂x` and
//! `M_ab = Σ_{P∋a,b} -G_P ∂²C6_P/∂CN_a∂CN_b`. The pair block of a radial
//! function `φ(r)` is `φ'' n nᵀ + (φ'/r)(1 - n nᵀ)` on `(i,i)` and `(j,j)`
//! and its negative on `(i,j)` and `(j,i)`.
//!
//! The three-body (ATM) term has no Hessian here: `s9 != 0` is refused.

use crate::{
    c6_interpolated, checked_numbers, coordination_numbers, distance, rcov, reference_c6,
    reference_weights, sqrt_q, tables, D3Params, KCN, WF,
};
use ferric_core::FerricError;
use ndarray::Array2;

/// `C6` and its first and second derivatives in `(CN_i, CN_j)`:
/// `[C6, ∂_i, ∂_j, ∂_ii, ∂_ij, ∂_jj]`.
///
/// `C6 = N/D`, so with subscripts for derivatives
/// `C6_a = (N_a - C6 D_a)/D` and
/// `C6_ab = (N_ab - C6_a D_b - C6_b D_a - C6 D_ab)/D`.
/// `∂W/∂CN = α W` and `∂²W/∂CN² = (α² - 2 wf) W` with `α = -2 wf (CN - CNref)`.
/// In the underflow branch of [`c6_interpolated`] (nearest reference,
/// piecewise constant) every derivative is zero, as in the gradient.
pub(crate) fn c6_second_derivatives(zi: usize, zj: usize, cni: f64, cnj: f64) -> [f64; 6] {
    let wi = reference_weights(zi, cni);
    let wj = reference_weights(zj, cnj);
    let alpha = |z: usize, cn: f64, iref: usize| {
        -2.0 * WF * (cn - tables::REFERENCE_CN[iref + tables::MAX_REF * (z - 1)])
    };
    // [value, d_i, d_j, d_ii, d_ij, d_jj] of N and of D.
    let mut n = [0.0f64; 6];
    let mut d = [0.0f64; 6];
    for (iref, &a) in wi.iter().enumerate() {
        let ai = alpha(zi, cni, iref);
        let (da, dda) = (ai * a, (ai * ai - 2.0 * WF) * a);
        for (jref, &b) in wj.iter().enumerate() {
            let aj = alpha(zj, cnj, jref);
            let (db, ddb) = (aj * b, (aj * aj - 2.0 * WF) * b);
            let c = reference_c6(iref, jref, zi, zj);
            let terms = [a * b, da * b, a * db, dda * b, da * db, a * ddb];
            for k in 0..6 {
                n[k] += c * terms[k];
                d[k] += terms[k];
            }
        }
    }
    if d[0] <= 1e-300 {
        return [c6_interpolated(zi, zj, cni, cnj), 0.0, 0.0, 0.0, 0.0, 0.0];
    }
    let c6 = n[0] / d[0];
    let ci = (n[1] - c6 * d[1]) / d[0];
    let cj = (n[2] - c6 * d[2]) / d[0];
    let cii = (n[3] - 2.0 * ci * d[1] - c6 * d[3]) / d[0];
    let cij = (n[4] - ci * d[2] - cj * d[1] - c6 * d[4]) / d[0];
    let cjj = (n[5] - 2.0 * cj * d[2] - c6 * d[5]) / d[0];
    [c6, ci, cj, cii, cij, cjj]
}

/// `G(r) = s6/d6 + 3 q s8/d8` and its first two derivatives, with
/// `d6 = r^6 + f^6`, `d8 = r^8 + f^8`.
fn radial_g(params: &D3Params, qq: f64, r: f64) -> [f64; 3] {
    let f = params.a1 * (3.0 * qq).sqrt() + params.a2;
    let (f2, r2) = (f * f, r * r);
    let (f6, r4) = (f2 * f2 * f2, r2 * r2);
    let (r6, f8) = (r4 * r2, f2 * f2 * f2 * f2);
    let r8 = r6 * r2;
    let (d6, d8) = (r6 + f6, r8 + f8);
    let (s6, s8) = (params.s6, 3.0 * qq * params.s8);
    let g = s6 / d6 + s8 / d8;
    let g1 = -6.0 * s6 * r4 * r / (d6 * d6) - 8.0 * s8 * r6 * r / (d8 * d8);
    let g2 = s6 * (-30.0 * r4 / (d6 * d6) + 72.0 * r8 * r2 / (d6 * d6 * d6))
        + s8 * (-56.0 * r6 / (d8 * d8) + 128.0 * r8 * r6 / (d8 * d8 * d8));
    [g, g1, g2]
}

/// One atom pair: indices, unit vector `n = (R_i - R_j)/r` and distance.
struct Pair {
    i: usize,
    j: usize,
    n: [f64; 3],
    r: f64,
}

/// Add the Hessian of a radial pair function with `φ'(r) = d1`,
/// `φ''(r) = d2` to `h`.
fn add_pair_block(h: &mut Array2<f64>, p: &Pair, d1: f64, d2: f64) {
    for a in 0..3 {
        for b in 0..3 {
            let nn = p.n[a] * p.n[b];
            let delta = if a == b { 1.0 } else { 0.0 };
            let k = d2 * nn + d1 / p.r * (delta - nn);
            h[(3 * p.i + a, 3 * p.i + b)] += k;
            h[(3 * p.j + a, 3 * p.j + b)] += k;
            h[(3 * p.i + a, 3 * p.j + b)] -= k;
            h[(3 * p.j + a, 3 * p.i + b)] -= k;
        }
    }
}

/// `row[3i + a] += s n_a`, `row[3j + a] -= s n_a`: `s ∂r_P/∂x` into a 3N row.
fn add_radial_row(m: &mut Array2<f64>, row: usize, p: &Pair, s: f64) {
    for a in 0..3 {
        m[(row, 3 * p.i + a)] += s * p.n[a];
        m[(row, 3 * p.j + a)] -= s * p.n[a];
    }
}

/// Counting function `c(r)` of the pair and its first two derivatives.
fn counting(zi: usize, zj: usize, r: f64) -> [f64; 3] {
    let rc = rcov(zi) + rcov(zj);
    let c = 1.0 / (1.0 + (-KCN * (rc / r - 1.0)).exp());
    let a = -KCN * rc / (r * r);
    let c1 = a * c * (1.0 - c);
    // d/dr [a c(1-c)] = (-2a/r) c(1-c) + a (1-2c) c'
    let c2 = -2.0 / r * c1 + a * (1.0 - 2.0 * c) * c1;
    [c, c1, c2]
}

fn checked_input(
    numbers: &[u8],
    coords: &[[f64; 3]],
    params: &D3Params,
) -> Result<Vec<usize>, FerricError> {
    if numbers.len() != coords.len() {
        return Err(FerricError::General(format!(
            "D3 Hessian: {} atomic numbers but {} coordinates",
            numbers.len(),
            coords.len()
        )));
    }
    if let Some((i, c)) = coords
        .iter()
        .enumerate()
        .find(|(_, c)| !c.iter().all(|v| v.is_finite()))
    {
        return Err(FerricError::General(format!(
            "D3 Hessian: atom {i} has a non-finite coordinate {c:?}"
        )));
    }
    if params.s9 != 0.0 {
        return Err(FerricError::General(format!(
            "D3 Hessian: the three-body (ATM, s9 = {}) term has no second derivative \
             (not implemented; next unit: the ATM Hessian with its CN chain rule). \
             Named-functional D3(BJ) parameters use s9 = 0.",
            params.s9
        )));
    }
    checked_numbers(numbers)
}

/// Analytic nuclear Hessian `∂²E/∂x∂y` of the two-body [`crate::d3bj_energy`],
/// Hartree/Bohr², `3N × 3N` with index `3·atom + xyz`, including the
/// coordination-number chain rule to second order (module docs).
///
/// Errors: mismatched lengths, a non-finite coordinate, an element outside
/// the D3 tables, coincident atoms, and `params.s9 != 0` (no three-body
/// Hessian). Fewer than two atoms gives an exact zero matrix.
pub fn d3bj_hessian(
    numbers: &[u8],
    coords: &[[f64; 3]],
    params: &D3Params,
) -> Result<Array2<f64>, FerricError> {
    let z = checked_input(numbers, coords, params)?;
    let nat = z.len();
    let n3 = 3 * nat;
    let mut h = Array2::<f64>::zeros((n3, n3));
    if nat < 2 {
        return Ok(h);
    }
    let cn = coordination_numbers(&z, coords);
    let mut pairs = Vec::with_capacity(nat * (nat - 1) / 2);
    for i in 0..nat {
        for j in 0..i {
            let r = distance(&coords[i], &coords[j]);
            if r <= f64::EPSILON {
                return Err(FerricError::General(format!(
                    "D3 Hessian: atoms {i} and {j} are at the same position; \
                     the dispersion energy diverges."
                )));
            }
            let n = [0, 1, 2].map(|a| (coords[i][a] - coords[j][a]) / r);
            pairs.push(Pair { i, j, n, r });
        }
    }
    // J_a = dCN_a/dx; each pair's c_ij enters CN_i and CN_j.
    let mut jac = Array2::<f64>::zeros((nat, n3));
    let counts: Vec<[f64; 3]> = pairs
        .iter()
        .map(|p| counting(z[p.i], z[p.j], p.r))
        .collect();
    for (p, c) in pairs.iter().zip(&counts) {
        add_radial_row(&mut jac, p.i, p, c[1]);
        add_radial_row(&mut jac, p.j, p, c[1]);
    }
    let mut e_cn = vec![0.0f64; nat];
    let mut u = Array2::<f64>::zeros((nat, n3));
    let mut m = Array2::<f64>::zeros((nat, nat));
    for p in &pairs {
        let (i, j) = (p.i, p.j);
        let [c6, ci, cj, cii, cij, cjj] = c6_second_derivatives(z[i], z[j], cn[i], cn[j]);
        let [g, g1, g2] = radial_g(params, sqrt_q(z[i]) * sqrt_q(z[j]), p.r);
        add_pair_block(&mut h, p, -c6 * g1, -c6 * g2);
        e_cn[i] -= g * ci;
        e_cn[j] -= g * cj;
        add_radial_row(&mut u, i, p, -g1 * ci);
        add_radial_row(&mut u, j, p, -g1 * cj);
        m[(i, i)] -= g * cii;
        m[(j, j)] -= g * cjj;
        m[(i, j)] -= g * cij;
        m[(j, i)] -= g * cij;
    }
    let mixed = u.t().dot(&jac);
    h += &mixed;
    h += &mixed.t();
    h += &jac.t().dot(&m.dot(&jac));
    for (p, c) in pairs.iter().zip(&counts) {
        let w = e_cn[p.i] + e_cn[p.j];
        add_pair_block(&mut h, p, w * c[1], w * c[2]);
    }
    Ok(h)
}

/// [`d3bj_hessian`] for a [`ferric_core::mol::Molecule`], `3N × 3N` over the
/// molecule's OWN atoms in their own order. Ghost centres are excluded from
/// the dispersion sum (as in [`crate::d3bj_energy_for_molecule`]) and their
/// rows and columns are exact zeros; a molecule of only ghosts is refused.
pub fn d3bj_hessian_for_molecule(
    mol: &ferric_core::mol::Molecule,
    params: &D3Params,
) -> Result<Array2<f64>, FerricError> {
    let real: Vec<usize> = (0..mol.atoms.len())
        .filter(|&i| !mol.atoms[i].ghost)
        .collect();
    if real.is_empty() && !mol.atoms.is_empty() {
        return Err(FerricError::General(format!(
            "D3 Hessian: all {} centres in this molecule are ghosts, so there are no \
             real atoms to correct.",
            mol.atoms.len()
        )));
    }
    let numbers: Vec<u8> = real.iter().map(|&i| mol.atoms[i].z as u8).collect();
    let coords: Vec<[f64; 3]> = real
        .iter()
        .map(|&i| [mol.atoms[i].x, mol.atoms[i].y, mol.atoms[i].zpos])
        .collect();
    let filtered = d3bj_hessian(&numbers, &coords, params)?;
    let n3 = 3 * mol.atoms.len();
    let mut full = Array2::<f64>::zeros((n3, n3));
    for (ka, &ia) in real.iter().enumerate() {
        for (kb, &ib) in real.iter().enumerate() {
            for a in 0..3 {
                for b in 0..3 {
                    full[(3 * ia + a, 3 * ib + b)] = filtered[(3 * ka + a, 3 * kb + b)];
                }
            }
        }
    }
    Ok(full)
}
