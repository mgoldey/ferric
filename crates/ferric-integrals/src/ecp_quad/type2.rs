//! Semi-local (type-2) channels: `Σ_m ⟨a| l m⟩ U_l ⟨l m |b⟩`.
//!
//! `F_lm(r) = ∫ Y_lm χ(C + rΩ) dΩ = 4π e^{−α(r−|a|)²} Σ_{N,λ} T[c,m,N,λ] r^N k̃_λ(2α|a|r)`
//! with the r-independent angular tensor
//! `T[c,m,N,λ] = Σ_{p+q+s=N} C(i,p)C(j,q)C(k,s)(−a_x)^{i−p}(−a_y)^{j−q}(−a_z)^{k−s}
//!               Σ_μ Y_λμ(â) ∫ Y_lm Y_λμ Ω_x^p Ω_y^q Ω_z^s dΩ`.
//! All primitive/term/node work goes into ONE radial tensor per (shell pair,
//! centre, l),
//! `R[N+N', λ, λ'] = Σ_{αβ} c_α c_β Σ_terms d ∫ (4π)² r^{n+N+N'} e^{−p(r−r0)²−K} k̃_λ(2α|a|r) k̃_λ'(2β|b|r) dr`
//! (`p = α+β+ζ`, `r0 = (α|a|+β|b|)/p`, `K = α|a|²+β|b|²−p r0² ≥ 0`: never
//! overflows), then one angular contraction `V = Σ T^A T^B R`.

use super::bessel::ktil;
use super::harmonics::{binom, cart_list, ncart, q_table, y_values};
use super::radial::{nodes, MAX_NODES, RAD_T};
use super::{Mutation, QShell, QuadKnobs, Term, FOURPI, SCREEN};

/// Dimensions of a projection tensor `T[c][m][N][λ]`.
#[inline]
fn dims(la: usize, l: usize) -> (usize, usize, usize, usize) {
    (ncart(la), 2 * l + 1, la + 1, l + la + 1)
}

/// `T[c][m][N][λ]` (row-major) for a shell of angular momentum `la`
/// displaced by `a = A − C` from the ECP centre, channel `l`. On-centre
/// (`a = 0`) only `N = l_a`, `λ = l` survive (`k̃_λ(0) = δ_λ0` does the rest).
/// `truncate` (negative control only) drops the `λ > l` terms.
pub(crate) fn proj_tensor(la: usize, l: usize, a: [f64; 3], truncate: bool) -> Vec<f64> {
    let an = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    let u = if an > 0.0 {
        [a[0] / an, a[1] / an, a[2] / an]
    } else {
        [0.0, 0.0, 1.0]
    };
    let (nc, nm, nn, nlam) = dims(la, l);
    let mut yv = Vec::new();
    y_values(l + la, u, &mut yv);
    let mut t = vec![0.0; nc * nm * nn * nlam];
    for (ic, [i, j, k]) in cart_list(la).into_iter().enumerate() {
        for p in 0..=i {
            for q in 0..=j {
                for s in 0..=k {
                    let pref = binom(i, p)
                        * binom(j, q)
                        * binom(k, s)
                        * (-a[0]).powi((i - p) as i32)
                        * (-a[1]).powi((j - q) as i32)
                        * (-a[2]).powi((k - s) as i32);
                    if pref == 0.0 {
                        continue;
                    }
                    let n = p + q + s;
                    let lam_lo = if n < l { l - n } else { (l + n) % 2 };
                    for lam in (lam_lo..=l + n).step_by(2) {
                        if truncate && lam > l {
                            continue;
                        }
                        let qt = q_table(l, lam, p, q, s);
                        if qt.is_empty() {
                            continue;
                        }
                        let nmu = 2 * lam + 1;
                        let y = &yv[lam * lam..lam * lam + nmu];
                        for m in 0..nm {
                            let row = &qt[m * nmu..(m + 1) * nmu];
                            let acc: f64 = row.iter().zip(y).map(|(x, y)| x * y).sum();
                            t[((ic * nm + m) * nn + n) * nlam + lam] += pref * acc;
                        }
                    }
                }
            }
        }
    }
    t
}

/// Loose upper-bound estimate of `|∫ r² U F^A F^B|` for one (primitive pair,
/// term): the envelope integral `e^{−K} √(π/p)` × `(4π)² |c c d|` × the
/// polynomial factor at the window edge, × 1e3 safety (prototype
/// `_prim_screen_bound`; screened == unscreened to ≤ 1e-20 on the fixtures).
#[inline]
fn screen_bound(
    la: usize,
    lb: usize,
    ccd: f64,
    an: f64,
    bn: f64,
    p: f64,
    r0: f64,
    kk: f64,
    n: i32,
) -> f64 {
    let rmax = r0 + RAD_T / p.sqrt();
    let poly = (an + rmax).powi(la as i32)
        * (bn + rmax).powi(lb as i32)
        * rmax.max(1.0).powi(n)
        * 2f64.powi((la + lb) as i32);
    1e3 * ccd.abs() * FOURPI * FOURPI * (-kk).exp() * (std::f64::consts::PI / p).sqrt() * poly
}

/// The radial tensor `R[K][λ][λ']` (`K ≤ l_a + l_b`, `λ ≤ l + l_a`,
/// `λ' ≤ l + l_b`) of one channel, or `None` when every window screened out.
pub(crate) fn radial_tensor(
    sa: &QShell,
    sb: &QShell,
    an: f64,
    bn: f64,
    l: usize,
    terms: &[Term],
    knobs: &QuadKnobs,
) -> Option<Vec<f64>> {
    let (la, lb) = (sa.l, sb.l);
    let (na, nb, nk) = (l + la + 1, l + lb + 1, la + lb + 1);
    let unscaled = knobs.mutation == Mutation::UnscaledBessel;
    let mut rt = vec![0.0; nk * na * nb];
    let mut any = false;
    let (mut rr, mut ww) = ([0.0; MAX_NODES], [0.0; MAX_NODES]);
    let (mut ka, mut kb) = ([0.0; 32], [0.0; 32]);
    for (&al, &ca) in sa.exps.iter().zip(&sa.coefs) {
        for (&be, &cb) in sb.exps.iter().zip(&sb.coefs) {
            for t in terms {
                let p = al + be + t.zeta;
                let r0 = (al * an + be * bn) / p;
                let kk = al * an * an + be * bn * bn - p * r0 * r0;
                let ccd = ca * cb * t.d;
                if knobs.screen && screen_bound(la, lb, ccd, an, bn, p, r0, kk, t.n) < SCREEN {
                    continue;
                }
                any = true;
                let nn = nodes(p, r0, t.n + (la + lb) as i32, &knobs.rule, &mut rr, &mut ww);
                let pref = ccd * FOURPI * FOURPI;
                for i in 0..nn {
                    let r = rr[i];
                    let (za, zb) = (2.0 * al * an * r, 2.0 * be * bn * r);
                    ktil(na - 1, za, &mut ka);
                    ktil(nb - 1, zb, &mut kb);
                    let mut g = if unscaled {
                        // NEGATIVE CONTROL: unscaled i_n and the unfactored
                        // Gaussians (overflow × underflow at long range).
                        let (ea, eb) = (za.exp(), zb.exp());
                        ka[..na].iter_mut().for_each(|v| *v *= ea);
                        kb[..nb].iter_mut().for_each(|v| *v *= eb);
                        pref * ww[i]
                            * r.powi(t.n)
                            * (-al * (r * r + an * an) - be * (r * r + bn * bn) - t.zeta * r * r)
                                .exp()
                    } else {
                        pref * ww[i] * r.powi(t.n) * (-p * (r - r0) * (r - r0) - kk).exp()
                    };
                    for k in 0..nk {
                        for lam in 0..na {
                            let tv = g * ka[lam];
                            let row = &mut rt[(k * na + lam) * nb..(k * na + lam + 1) * nb];
                            for (dst, kbv) in row.iter_mut().zip(&kb[..nb]) {
                                *dst += tv * kbv;
                            }
                        }
                        g *= r;
                    }
                }
            }
        }
    }
    any.then_some(rt)
}

/// `V[c][c'] += Σ_{m,N,λ,N',λ'} T^A[c,m,N,λ] T^B[c',m,N',λ'] R[N+N',λ,λ']`.
pub(crate) fn contract(
    la: usize,
    lb: usize,
    l: usize,
    ta: &[f64],
    tb: &[f64],
    rt: &[f64],
    v: &mut [f64],
) {
    let (nca, nm, nna, na) = dims(la, l);
    let (ncb, _, nnb, nb) = dims(lb, l);
    let blk = nm * nna * na; // per-Cartesian stride of T^A and of G
                             // G[c'][m][N][λ] = Σ_{N',λ'} T^B[c',m,N',λ'] R[N+N',λ,λ']
    let mut g = vec![0.0; ncb * blk];
    for cb in 0..ncb {
        for m in 0..nm {
            for np in 0..nnb {
                let tbrow =
                    &tb[((cb * nm + m) * nnb + np) * nb..((cb * nm + m) * nnb + np + 1) * nb];
                if tbrow.iter().all(|&x| x == 0.0) {
                    continue;
                }
                for n in 0..nna {
                    for lam in 0..na {
                        let rrow = &rt[((n + np) * na + lam) * nb..((n + np) * na + lam + 1) * nb];
                        let s: f64 = tbrow.iter().zip(rrow).map(|(x, y)| x * y).sum();
                        g[cb * blk + (m * nna + n) * na + lam] += s;
                    }
                }
            }
        }
    }
    for ca in 0..nca {
        let tarow = &ta[ca * blk..(ca + 1) * blk];
        for cb in 0..ncb {
            let grow = &g[cb * blk..(cb + 1) * blk];
            v[ca * ncb + cb] += tarow.iter().zip(grow).map(|(x, y)| x * y).sum::<f64>();
        }
    }
}

/// Cross-check construction (debug/test only): `F^A_lm(r)`, `F^B_lm(r)`
/// formed explicitly at every node, then `Σ_m F^A F^B` — no radial tensor.
/// Uses the same nodes and `T`, an independent contraction order.
pub(crate) fn direct(
    sa: &QShell,
    sb: &QShell,
    a: [f64; 3],
    b: [f64; 3],
    l: usize,
    terms: &[Term],
    knobs: &QuadKnobs,
    v: &mut [f64],
) {
    let (la, lb) = (sa.l, sb.l);
    let an = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    let bn = (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).sqrt();
    let ta = proj_tensor(la, l, a, false);
    let tb = proj_tensor(lb, l, b, false);
    let (nca, nm, nna, na) = dims(la, l);
    let (ncb, _, nnb, nb) = dims(lb, l);
    let (mut rr, mut ww) = ([0.0; MAX_NODES], [0.0; MAX_NODES]);
    let (mut ka, mut kb) = ([0.0; 32], [0.0; 32]);
    let f = |t: &[f64], nc: usize, nn: usize, nl: usize, alpha: f64, dn: f64, r: f64, k: &[f64]| {
        let env = FOURPI * (-alpha * (r - dn) * (r - dn)).exp();
        let mut out = vec![0.0; nc * nm];
        for c in 0..nc {
            for m in 0..nm {
                let mut s = 0.0;
                let mut rn = 1.0;
                for n in 0..nn {
                    for lam in 0..nl {
                        s += t[((c * nm + m) * nn + n) * nl + lam] * rn * k[lam];
                    }
                    rn *= r;
                }
                out[c * nm + m] = s * env;
            }
        }
        out
    };
    for (&al, &ca) in sa.exps.iter().zip(&sa.coefs) {
        for (&be, &cb) in sb.exps.iter().zip(&sb.coefs) {
            for t in terms {
                let p = al + be + t.zeta;
                let k = nodes(
                    p,
                    (al * an + be * bn) / p,
                    t.n + (la + lb) as i32,
                    &knobs.rule,
                    &mut rr,
                    &mut ww,
                );
                for i in 0..k {
                    let r = rr[i];
                    ktil(na - 1, 2.0 * al * an * r, &mut ka);
                    ktil(nb - 1, 2.0 * be * bn * r, &mut kb);
                    let fa = f(&ta[..], nca, nna, na, al, an, r, &ka[..]);
                    let fb = f(&tb[..], ncb, nnb, nb, be, bn, r, &kb[..]);
                    let wgt = ca * cb * t.d * ww[i] * r.powi(t.n) * (-t.zeta * r * r).exp();
                    for x in 0..nca {
                        for y in 0..ncb {
                            let s: f64 = (0..nm).map(|m| fa[x * nm + m] * fb[y * nm + m]).sum();
                            v[x * ncb + y] += wgt * s;
                        }
                    }
                }
            }
        }
    }
}
