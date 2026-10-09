//! Local (type-1) channel `⟨a|U_L|b⟩`.
//!
//! About the ECP centre C, `χ_a χ_b = poly(r) · e^{−(α+β)r² + 2k·r − α|a|² − β|b|²}`
//! with `k = α a + β b`; `e^{2k·r} = 4π Σ_λ i_λ(2|k|r) Σ_μ Y_λμ(k̂) Y_λμ(Ω)`, so
//! the angular integral of each monomial `x^P y^Q z^S = r^N Ω^{PQS}` is
//! `4π Σ_λ k̃_λ(2|k|r) Σ_μ Y_λμ(k̂) ∫Ω^{PQS} Y_λμ` against the envelope
//! `e^{−p(r − |k|/p)² − K₁}` (`p = α+β+ζ`, `K₁ = α|a|² + β|b|² − |k|²/p ≥ 0`).
//! Per primitive pair: `R1[N][λ]` accumulated over terms × nodes, then one
//! contraction with the (pair-specific, `k̂`-dependent) angular factors.

use super::bessel::ktil;
use super::harmonics::{binom, cart_list, m_table, ncart, npqs, pqs_index, y_values};
use super::radial::{nodes, MAX_NODES, RAD_T};
use super::{Mutation, QShell, QuadKnobs, Term, FOURPI, SCREEN};

/// `(x − a)^i` as coefficients of `x^p`, `p = 0..=i`.
fn poly1d(i: usize, a: f64) -> Vec<f64> {
    (0..=i)
        .map(|p| binom(i, p) * (-a).powi((i - p) as i32))
        .collect()
}

fn convolve(x: &[f64], y: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; x.len() + y.len() - 1];
    for (i, a) in x.iter().enumerate() {
        for (j, b) in y.iter().enumerate() {
            out[i + j] += a * b;
        }
    }
    out
}

/// Adds `⟨sa|U_local|sb⟩` (Cartesian, `ncart(la) × ncart(lb)` row-major) to `v`.
/// `a = A − C`, `b = B − C`.
pub(crate) fn type1_cart(
    sa: &QShell,
    sb: &QShell,
    a: [f64; 3],
    b: [f64; 3],
    local: &[Term],
    knobs: &QuadKnobs,
    v: &mut [f64],
) {
    let (la, lb) = (sa.l, sb.l);
    let lt = la + lb;
    let nl = lt + 1;
    let a2 = a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
    let b2 = b[0] * b[0] + b[1] * b[1] + b[2] * b[2];
    let unscaled = knobs.mutation == Mutation::UnscaledBessel;
    let (ca_list, cb_list) = (cart_list(la), cart_list(lb));
    let ncb = ncart(lb);
    // Product polynomial about C per Cartesian pair, per axis.
    let polys: Vec<[Vec<f64>; 3]> = ca_list
        .iter()
        .flat_map(|ia| {
            cb_list.iter().map(move |ib| {
                [0, 1, 2].map(|x| convolve(&poly1d(ia[x], a[x]), &poly1d(ib[x], b[x])))
            })
        })
        .collect();
    let mt = m_table();
    let mut r1 = vec![0.0; nl * nl];
    let mut xq = vec![0.0; npqs(lt)];
    let mut yv = Vec::new();
    let (mut rr, mut ww) = ([0.0; MAX_NODES], [0.0; MAX_NODES]);
    let mut kt = [0.0; 32];
    let rmax_geom = a2.max(b2).sqrt();
    for (&al, &cfa) in sa.exps.iter().zip(&sa.coefs) {
        for (&be, &cfb) in sb.exps.iter().zip(&sb.coefs) {
            let kv = [
                al * a[0] + be * b[0],
                al * a[1] + be * b[1],
                al * a[2] + be * b[2],
            ];
            let kn = (kv[0] * kv[0] + kv[1] * kv[1] + kv[2] * kv[2]).sqrt();
            r1.fill(0.0);
            let mut any = false;
            for t in local {
                let p = al + be + t.zeta;
                let r0 = kn / p;
                let kk = al * a2 + be * b2 - kn * kn / p;
                let ccd = cfa * cfb * t.d;
                if knobs.screen
                    && 1e3
                        * ccd.abs()
                        * FOURPI
                        * (-kk).exp()
                        * (std::f64::consts::PI / p).sqrt()
                        * (rmax_geom + r0 + RAD_T / p.sqrt() + 1.0).powi(lt as i32 + t.n)
                        < SCREEN
                {
                    continue;
                }
                any = true;
                let nn = nodes(p, r0, t.n + lt as i32, &knobs.rule, &mut rr, &mut ww);
                for i in 0..nn {
                    let r = rr[i];
                    let z = 2.0 * kn * r;
                    ktil(lt, z, &mut kt);
                    let mut env = if unscaled {
                        let e = z.exp();
                        kt[..nl].iter_mut().for_each(|x| *x *= e);
                        FOURPI * (-p * r * r - al * a2 - be * b2).exp() * t.d * r.powi(t.n) * ww[i]
                    } else {
                        FOURPI * (-p * (r - r0) * (r - r0) - kk).exp() * t.d * r.powi(t.n) * ww[i]
                    };
                    for n in 0..nl {
                        let row = &mut r1[n * nl..(n + 1) * nl];
                        for (dst, k) in row.iter_mut().zip(&kt[..nl]) {
                            *dst += env * k;
                        }
                        env *= r;
                    }
                }
            }
            if !any {
                continue;
            }
            let u = if kn > 0.0 {
                [kv[0] / kn, kv[1] / kn, kv[2] / kn]
            } else {
                [0.0, 0.0, 1.0]
            };
            y_values(lt, u, &mut yv);
            // X[PQS] = Σ_{λ ≤ N, λ ≡ N (2)} (Σ_μ M[λ][PQS][μ] Y_λμ(k̂)) R1[N][λ]
            for n in 0..=lt {
                for [pp, qq, ss] in cart_list(n) {
                    let idx = pqs_index(pp, qq, ss);
                    let mut s = 0.0;
                    for lam in (n % 2..=n).step_by(2) {
                        let m = &mt[lam][idx];
                        let y = &yv[lam * lam..lam * lam + 2 * lam + 1];
                        let ang: f64 = m.iter().zip(y).map(|(x, y)| x * y).sum();
                        s += ang * r1[n * nl + lam];
                    }
                    xq[idx] = s;
                }
            }
            let cc = cfa * cfb;
            for (pair, [px, py, pz]) in polys.iter().enumerate() {
                let mut s = 0.0;
                for (pp, cx) in px.iter().enumerate() {
                    if *cx == 0.0 {
                        continue;
                    }
                    for (qq, cy) in py.iter().enumerate() {
                        if *cy == 0.0 {
                            continue;
                        }
                        for (ss, cz) in pz.iter().enumerate() {
                            if *cz == 0.0 {
                                continue;
                            }
                            s += cx * cy * cz * xq[pqs_index(pp, qq, ss)];
                        }
                    }
                }
                let (ia, ib) = (pair / ncb, pair % ncb);
                v[ia * ncb + ib] += cc * s;
            }
        }
    }
}
