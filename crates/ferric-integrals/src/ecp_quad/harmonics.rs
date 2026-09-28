//! Real spherical harmonics as monomial expansions, exact sphere integrals of
//! monomials, and the two angular tables of the ECP quadrature (built lazily,
//! once, immutable afterwards — safe to share across rayon workers).
//!
//! * `Y_lm` (m = −l..l): Helgaker/Jørgensen/Olsen Eq. 6.4.47–6.4.50 solid
//!   harmonics (no Condon–Shortley phase, `S_11 = x`, `S_1−1 = y`) scaled by
//!   `√((2l+1)/4π)` — unit-sphere orthonormal. libcint's cart2sph (ecp.rs's
//!   C2S tables) is built from exactly these (asserted in the tests).
//! * Type-2 table `Q(l, λ, pqs)[m][μ] = ∫ Y_lm Y_λμ Ω_x^p Ω_y^q Ω_z^s dΩ`.
//! * Type-1 table `M(λ, PQS)[μ] = ∫ Ω_x^P Ω_y^Q Ω_z^S Y_λμ dΩ`.

use std::f64::consts::PI;
use std::sync::OnceLock;

/// Largest `l` of the harmonic tables.
pub const LMAX_Y: usize = 16;
/// Largest ECP semi-local channel `l` of the type-2 table.
pub const Q_LMAX: usize = 5;
/// Largest shell `l` (after the derivative's +1 raise) of the type-2 table.
pub const Q_SHELL_LMAX: usize = 5;
/// Largest `λ` of the type-2 table.
pub const Q_LAM_MAX: usize = Q_LMAX + Q_SHELL_LMAX;
/// Largest total monomial degree / `λ` of the type-1 table (`l_a + l_b`, one
/// of them raised by the derivative).
pub const M_LMAX: usize = 10;

/// One monomial `coef · x^a y^b z^c`.
#[derive(Debug, Clone, Copy)]
pub struct Mono {
    pub a: u8,
    pub b: u8,
    pub c: u8,
    pub coef: f64,
}

#[inline]
pub fn ncart(l: usize) -> usize {
    (l + 1) * (l + 2) / 2
}

/// Index of `(i, j, k)` (`i + j + k = l`) in the CCA Cartesian order
/// (`lx` descending, then `ly` descending) of its shell.
#[inline]
pub fn cart_index(j: usize, k: usize) -> usize {
    let pp = j + k;
    pp * (pp + 1) / 2 + k
}

/// CCA Cartesian exponents of a shell of angular momentum `l`.
pub fn cart_list(l: usize) -> Vec<[usize; 3]> {
    let mut v = Vec::with_capacity(ncart(l));
    for lx in (0..=l).rev() {
        for ly in (0..=l - lx).rev() {
            v.push([lx, ly, l - lx - ly]);
        }
    }
    v
}

/// Flat index of `(p, q, s)` over ALL degrees `p + q + s = 0, 1, 2, …`
/// (degree blocks in order, CCA order inside a block).
#[inline]
pub fn pqs_index(p: usize, q: usize, s: usize) -> usize {
    let n = p + q + s;
    n * (n + 1) * (n + 2) / 6 + cart_index(q, s)
}

/// Number of `(p, q, s)` with `p + q + s ≤ nmax`.
#[inline]
pub fn npqs(nmax: usize) -> usize {
    (nmax + 1) * (nmax + 2) * (nmax + 3) / 6
}

fn factorial(n: usize) -> f64 {
    (1..=n).map(|k| k as f64).product()
}

pub(crate) fn binom(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    let k = k.min(n - k);
    let mut r = 1.0;
    for i in 0..k {
        r = r * (n - i) as f64 / (i + 1) as f64;
    }
    r.round()
}

/// `(n)!!` for odd `n ≥ −1` (`(−1)!! = 1`).
fn dfact_odd(n: i64) -> f64 {
    let mut r = 1.0;
    let mut k = n;
    while k > 1 {
        r *= k as f64;
        k -= 2;
    }
    r
}

/// `∫ x^a y^b z^c dΩ` over the unit sphere (zero unless all exponents even).
pub fn sphere_monomial(a: usize, b: usize, c: usize) -> f64 {
    if a % 2 == 1 || b % 2 == 1 || c % 2 == 1 {
        return 0.0;
    }
    4.0 * PI * dfact_odd(a as i64 - 1) * dfact_odd(b as i64 - 1) * dfact_odd(c as i64 - 1)
        / dfact_odd((a + b + c + 1) as i64)
}

fn build_ylm(l: usize) -> Vec<Vec<Mono>> {
    let li = l as i64;
    let mut out = Vec::with_capacity(2 * l + 1);
    for m in -li..=li {
        let am = m.unsigned_abs() as usize;
        let nlm = (2.0 * factorial(l + am) * factorial(l - am) / if m == 0 { 2.0 } else { 1.0 })
            .sqrt()
            / (2f64.powi(am as i32) * factorial(l));
        let v0 = usize::from(m < 0);
        let mut dense = vec![0.0; ncart(l)];
        for t in 0..=(l - am) / 2 {
            for u in 0..=t {
                let mut w = v0;
                while w <= am {
                    let sgn = if (t + (w - v0) / 2) % 2 == 1 {
                        -1.0
                    } else {
                        1.0
                    };
                    let c = sgn
                        * 0.25f64.powi(t as i32)
                        * binom(l, t)
                        * binom(l - t, am + t)
                        * binom(t, u)
                        * binom(am, w);
                    // key (2t + am − 2u − w, 2u + w, l − 2t − am)
                    let (jy, kz) = (2 * u + w, l - 2 * t - am);
                    dense[cart_index(jy, kz)] += c * nlm;
                    w += 2;
                }
            }
        }
        let scale = ((2 * l + 1) as f64 / (4.0 * PI)).sqrt();
        let terms = cart_list(l)
            .into_iter()
            .zip(dense)
            .filter(|&(_, c)| c != 0.0)
            .map(|([a, b, c], coef)| Mono {
                a: a as u8,
                b: b as u8,
                c: c as u8,
                coef: coef * scale,
            })
            .collect();
        out.push(terms);
    }
    out
}

/// `Y_lm` (m = −l..l) as monomial lists, `l ≤ LMAX_Y`.
pub fn ylm(l: usize) -> &'static [Vec<Mono>] {
    static Y: OnceLock<Vec<Vec<Vec<Mono>>>> = OnceLock::new();
    assert!(l <= LMAX_Y, "ylm: l = {l} > {LMAX_Y}");
    &Y.get_or_init(|| (0..=LMAX_Y).map(build_ylm).collect())[l]
}

/// `out[l² + l + m] = Y_lm(u)` for `l = 0..=lmax`, unit vector `u`.
pub fn y_values(lmax: usize, u: [f64; 3], out: &mut Vec<f64>) {
    out.clear();
    out.resize((lmax + 1) * (lmax + 1), 0.0);
    let mut px = [1.0; LMAX_Y + 1];
    let mut py = [1.0; LMAX_Y + 1];
    let mut pz = [1.0; LMAX_Y + 1];
    for k in 1..=lmax {
        px[k] = px[k - 1] * u[0];
        py[k] = py[k - 1] * u[1];
        pz[k] = pz[k - 1] * u[2];
    }
    for l in 0..=lmax {
        for (mi, terms) in ylm(l).iter().enumerate() {
            out[l * l + mi] = terms
                .iter()
                .map(|t| t.coef * px[t.a as usize] * py[t.b as usize] * pz[t.c as usize])
                .sum();
        }
    }
}

/// Neumaier-compensated sum (the table entries are signed sums of O(1..10)
/// terms that cancel to exact zeros by symmetry).
fn fsum(it: impl Iterator<Item = f64>) -> f64 {
    let (mut s, mut c) = (0.0_f64, 0.0_f64);
    for x in it {
        let t = s + x;
        if s.abs() >= x.abs() {
            c += (s - t) + x;
        } else {
            c += (x - t) + s;
        }
        s = t;
    }
    s + c
}

fn build_q(l: usize, lam: usize, p: usize, q: usize, s: usize) -> Vec<f64> {
    let n = p + q + s;
    if (l + lam + n) % 2 == 1 || l.abs_diff(lam) > n {
        return Vec::new();
    }
    let (ya, yb) = (ylm(l), ylm(lam));
    let mut out = vec![0.0; (2 * l + 1) * (2 * lam + 1)];
    for (i, da) in ya.iter().enumerate() {
        for (j, db) in yb.iter().enumerate() {
            out[i * (2 * lam + 1) + j] = fsum(da.iter().flat_map(|ta| {
                db.iter().map(move |tb| {
                    ta.coef
                        * tb.coef
                        * sphere_monomial(
                            (ta.a + tb.a) as usize + p,
                            (ta.b + tb.b) as usize + q,
                            (ta.c + tb.c) as usize + s,
                        )
                })
            }));
        }
    }
    let mx = out.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let cut = 1e-15 * mx.max(1.0);
    for v in &mut out {
        if v.abs() < cut {
            *v = 0.0;
        }
    }
    out
}

/// Type-2 angular table `Q[m][μ]` (`(2l+1) × (2λ+1)`, row-major) for
/// `l ≤ Q_LMAX`, `λ ≤ Q_LAM_MAX`, `p + q + s ≤ Q_SHELL_LMAX`; EMPTY when it
/// vanishes identically (parity or `|l − λ| > p + q + s`). Built lazily per
/// entry (thread-safe; immutable once built).
pub fn q_table(l: usize, lam: usize, p: usize, q: usize, s: usize) -> &'static [f64] {
    const NP: usize = (Q_SHELL_LMAX + 1) * (Q_SHELL_LMAX + 2) * (Q_SHELL_LMAX + 3) / 6;
    const N: usize = (Q_LMAX + 1) * (Q_LAM_MAX + 1) * NP;
    static TAB: OnceLock<Vec<OnceLock<Vec<f64>>>> = OnceLock::new();
    assert!(
        l <= Q_LMAX && lam <= Q_LAM_MAX && p + q + s <= Q_SHELL_LMAX,
        "q_table({l}, {lam}, {p}{q}{s}) outside the table"
    );
    let tab = TAB.get_or_init(|| (0..N).map(|_| OnceLock::new()).collect());
    let idx = (l * (Q_LAM_MAX + 1) + lam) * NP + pqs_index(p, q, s);
    tab[idx].get_or_init(|| build_q(l, lam, p, q, s))
}

/// Type-1 angular table: `m_table()[λ][pqs_index(P,Q,S)]` is
/// `∫ Ω^{PQS} Y_λμ dΩ` (μ = −λ..λ), `λ, P+Q+S ≤ M_LMAX`.
pub fn m_table() -> &'static [Vec<Vec<f64>>] {
    static TAB: OnceLock<Vec<Vec<Vec<f64>>>> = OnceLock::new();
    TAB.get_or_init(|| {
        (0..=M_LMAX)
            .map(|lam| {
                let y = ylm(lam);
                let mut per = vec![Vec::new(); npqs(M_LMAX)];
                for n in 0..=M_LMAX {
                    for [p, q, s] in cart_list(n) {
                        per[pqs_index(p, q, s)] = y
                            .iter()
                            .map(|terms| {
                                fsum(terms.iter().map(|t| {
                                    t.coef
                                        * sphere_monomial(
                                            t.a as usize + p,
                                            t.b as usize + q,
                                            t.c as usize + s,
                                        )
                                }))
                            })
                            .collect();
                    }
                }
                per
            })
            .collect()
    })
}

/// libcint cart2sph (ncart × nsph, row-major) built from [`ylm`]: rows CCA
/// Cartesians, columns p → (x, y, z), l ≥ 2 → m = −l..l.
pub fn c2s(l: usize) -> Vec<f64> {
    let y = ylm(l);
    let nsph = 2 * l + 1;
    let order: Vec<usize> = if l == 1 {
        vec![2, 0, 1]
    } else {
        (0..nsph).collect()
    };
    let mut m = vec![0.0; ncart(l) * nsph];
    for (j, &mi) in order.iter().enumerate() {
        for t in &y[mi] {
            m[cart_index(t.b as usize, t.c as usize) * nsph + j] = t.coef;
        }
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unit-sphere orthonormality of the real harmonics, l ≤ 9 (bar 1e-13).
    #[test]
    fn real_harmonics_orthonormal() {
        let mut worst = 0.0_f64;
        for l in 0..10 {
            for l2 in 0..=l {
                for (i, a) in ylm(l).iter().enumerate() {
                    for (j, b) in ylm(l2).iter().enumerate() {
                        let v: f64 = a
                            .iter()
                            .flat_map(|x| {
                                b.iter().map(move |y| {
                                    x.coef
                                        * y.coef
                                        * sphere_monomial(
                                            (x.a + y.a) as usize,
                                            (x.b + y.b) as usize,
                                            (x.c + y.c) as usize,
                                        )
                                })
                            })
                            .sum();
                        let want = f64::from(u8::from(l == l2 && i == j));
                        worst = worst.max((v - want).abs());
                    }
                }
            }
        }
        assert!(worst < 1e-13, "{worst:e}");
    }

    /// The generated cart2sph IS ecp.rs's C2S0..C2S4 (so the quadrature's
    /// Cartesian output and the libecpint shim's share one spherical map).
    #[test]
    fn generated_cart2sph_equals_ecp_rs_tables() {
        for l in 0..=4usize {
            let ours = c2s(l);
            let theirs = crate::ecp::cart2sph(l as i32);
            assert_eq!(ours.len(), theirs.len(), "l = {l}");
            let d = ours
                .iter()
                .zip(theirs)
                .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
            assert!(d < 1e-14, "l = {l}: {d:e}");
        }
    }

    #[test]
    fn index_helpers_match_cca_order() {
        for l in 0..=6 {
            for (i, [_, j, k]) in cart_list(l).into_iter().enumerate() {
                assert_eq!(cart_index(j, k), i);
            }
        }
        let mut seen = vec![false; npqs(5)];
        for n in 0..=5 {
            for [p, q, s] in cart_list(n) {
                seen[pqs_index(p, q, s)] = true;
            }
        }
        assert!(seen.iter().all(|&b| b));
    }
}
