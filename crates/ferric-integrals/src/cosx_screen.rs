//! Shell-pair screening for the per-grid-point three-centre-one-electron
//! blocks `A^g_{mu,nu} = ∫ chi_mu chi_nu / |r - r_g|` (COSX / sn-LinK), shared
//! by [`crate::cosx_a`] (libint2) and [`crate::md3c1e`] (MD kernel).
//!
//! # Why this module exists
//!
//! The first screen (2026-09-06) bounded a pair by `sqrt(max|S_block|) / d`
//! from the **signed** overlap block. That vanishes by symmetry for
//! same-centre s-p, s-d and p-d pairs while their `A^g` block is O(1)
//! (water/cc-pVDZ: 15/21 same-centre pairs dropped with `|A^g|` up to 0.233;
//! `max|K_scr - K_unscr| = 0.71` against a grid error of 5e-5). Same class of
//! bug as the Schwarz fix (#34): **a bound must never underestimate**.
//!
//! # Why NOT Cauchy-Schwarz on the diagonals
//!
//! `1/|r-C| > 0`, so `<f,g>_C = ∫ f g / |r-C|` is an inner product and
//! `|A^g_{mu,nu}| <= sqrt(A^g_{mu,mu} A^g_{nu,nu})` is valid with never-vanishing
//! diagonals. But `A^g_{mu,mu}(C) ~ 1/|A-C|` carries NO factor of the pair
//! overlap `K_AB = exp(-ab/p |A-B|^2)`: for `mu` on atom A and `nu` on a distant
//! atom B the bound is `~1/sqrt(|A-C||B-C|)` while the truth is `~K_AB/|P-C|`.
//! At any grid extent (`R <~ 30` Bohr) that bound exceeds every threshold in
//! use (`1e-7`), so it retains every pair — valid and vacuous. Screening needs
//! the overlap decay, which lives at the primitive-pair level.
//!
//! # The bound (Hölder on the primitive expansion)
//!
//! Every AO is a contraction over primitives with the shell normalization
//! folded in, `chi_mu = Σ_i c_i P_mu(r-A) e^{-a_i r_A^2}` where `P_mu` is one
//! Cartesian monomial `x^{lx} y^{ly} z^{lz}` of total degree `l_a`. For a
//! primitive pair `(a at A, b at B)` the Gaussian product theorem gives
//!
//! ```text
//!     e^{-a r_A^2} e^{-b r_B^2} = K_AB e^{-p rho^2},   p = a+b,  P = (aA+bB)/p,  rho = |r-P|.
//! ```
//!
//! With `|(r-A)_d| <= rho + d_A` (`d_A = |P-A|`) for every component `d`,
//!
//! ```text
//!     |P_mu(r-A) P_nu(r-B)| <= (rho + d_A)^{l_a} (rho + d_B)^{l_b} = Σ_k q_k rho^k,   q_k >= 0.
//! ```
//!
//! For `k = 0`: `∫ e^{-p rho^2} / |r-C| = (2π/p) F_0(p R^2)` exactly (`R = |P-C|`).
//! For `k >= 1`: `rho^k e^{-p rho^2} <= M_k e^{-p rho^2 / 2}` with
//! `M_k = sup_rho rho^k e^{-p rho^2/2} = (k/(p e))^{k/2}`, so
//! `∫ rho^k e^{-p rho^2}/|r-C| <= M_k (4π/p) F_0(p R^2 / 2)`.
//! Finally `F_0(T) <= min(1, ½ sqrt(π/T))` (`F_0 = ½ sqrt(π/T) erf(sqrt T)`).
//! Summing `|c_i c_j|` over primitive pairs (triangle inequality) bounds the
//! max over the Cartesian block; a pure (spherical) side multiplies by the
//! max absolute column sum of its cart→sph matrix (`|Σ_m C_mi x_m| <= Σ_m |C_mi| max|x|`).
//!
//! Every step is an inequality that holds pointwise, so the bound cannot
//! underestimate the exact value; a relative `ROUNDING_SLACK` covers the
//! only case where it is tight to rounding (single-primitive same-centre s-s
//! ON its nucleus, where `F_0(0) = 1` makes the bound equal the integral).
//! Pinned by `tests/cosx_screen_anchors.rs`.
//!
//! # Structure that survives
//!
//! Per primitive pair the bound is `|c_a c_b| K_AB × [β0 F_0^+(pR^2) + β1 F_0^+(pR^2/2)]`
//! — it decays as `K_AB` in the pair separation and as `1/R` in the distance
//! from the pair centre to the grid point, which is what gives screening its
//! asymptotic bite. What is lost relative to the exact block: (i) the
//! polynomial bound (a factor growing with `l_a + l_b` and with `d_A, d_B`);
//! (ii) the halved exponent for `k >= 1` (≤ 2√2 in the far field);
//! (iii) same-centre mixed-`l` pairs have zero monopole and truly decay as
//! `1/R^2`, but are bounded here as `1/R`. Tightness is REPORTED by the anchor
//! (`true/bound` distribution), not assumed.
//!
//! # Cost
//!
//! [`PairBounds::estimate`](crate::cosx_screen::PairBounds::estimate) costs one `sqrt`, one division and ~6 flops per
//! primitive pair of the shell pair — the same loop shape as the kernel's
//! Boys evaluation but without the `exp`/table lookup and without the
//! R-tensor/contraction work behind it (hundreds of flops per primitive pair
//! per point). [`PairBounds::any_exceeds`](crate::cosx_screen::PairBounds::any_exceeds) adds an O(1) early-out on the
//! pair's total weight `Σ β` so negligible-overlap pairs never enter the
//! per-point loop.

use ferric_core::error::FerricError;
use std::f64::consts::PI;

use crate::basis_bridge::PreparedBasis;
use crate::md3c1e::{ferric_cart2sph, prim_norm, MAX_L};

/// Screening policy for the per-grid-point shell-pair sweep.
///
/// A shell pair `(s1, s2)` is retained at grid point `r_g` when
/// [`PairBounds::estimate`]`(s1, s2, r_g) >= threshold`. `threshold <= 0.0`
/// disables screening entirely, which is the exactness anchor's trivial
/// limit (`screen_zero_threshold_matches_unscreened`).
#[derive(Debug, Clone, Copy)]
pub struct CosxScreen {
    /// Screening threshold; `<= 0.0` means "keep every pair".
    pub threshold: f64,
}

impl CosxScreen {
    /// The trivial limit: no pair is ever dropped.
    pub fn none() -> Self {
        Self { threshold: 0.0 }
    }

    /// A screen at the given threshold.
    pub fn at(threshold: f64) -> Self {
        Self { threshold }
    }

    /// True when this screen drops nothing by construction.
    pub fn is_vacuous(&self) -> bool {
        self.threshold <= 0.0
    }
}

/// Relative slack absorbing floating-point rounding in the one tight case
/// (see the module doc). `1e-10` is ~1e6 ulps: far above any accumulated
/// rounding in a few hundred flops, far below any screening threshold.
const ROUNDING_SLACK: f64 = 1.0 + 1e-10;
/// Extra relative slack on the coarse gate so `coarse >= fine` holds despite
/// the two being summed in different orders (they tie to the ulp when
/// `R_c <= 0`); the coarse bound only gates the fine evaluation, so this
/// affects cost, never validity.
const COARSE_SLACK: f64 = 1.0 + 1e-9;

/// One primitive pair's contribution to a shell pair's bound.
#[derive(Clone, Copy)]
struct PrimTerm {
    /// Gaussian product centre `P`.
    cen: [f64; 3],
    /// Weight of the `k = 0` term, already multiplied by its `R -> 0` value
    /// (`F_0^+ = 1`): `|c_a c_b| K_AB s_a s_b (2π/p) q_0`.
    beta0: f64,
    /// Weight of the `k >= 1` terms at `R -> 0`: `|c_a c_b| K_AB s_a s_b (4π/p) Σ_k q_k M_k`.
    beta1: f64,
    /// `½ sqrt(π/p)`: `F_0^+(p R^2) = min(1, g0 / R)`.
    g0: f64,
    /// `½ sqrt(2π/p)`: `F_0^+(p R^2 / 2) = min(1, g1 / R)`.
    g1: f64,
}

/// Per-shell-pair screening data, precomputed once per geometry.
///
/// Holds, for each shell pair `(s1, s2)` with `s1 >= s2`, the list of
/// primitive-pair terms of the Hölder bound described in the module doc,
/// plus their total weight (the bound's value at `R = 0`, an upper bound for
/// every point).
pub struct PairBounds {
    nsh: usize,
    /// `term_start[tri(s1,s2)] .. term_start[tri(s1,s2)+1]` indexes `terms`.
    term_start: Vec<u32>,
    terms: Vec<PrimTerm>,
    /// Per pair, the single-sqrt coarse bound's data (see [`PairBounds::coarse_estimate`]).
    coarse: Vec<Coarse>,
}

/// Per-shell-pair data for the coarse (one-sqrt) bound: every product centre
/// `P_ij` lies on the segment `AB`, so `|P_ij - r| >= |M - r| - |AB|/2 =: R_c`
/// with `M` the midpoint, and every term is non-increasing in its `R`.
#[derive(Clone, Copy)]
struct Coarse {
    mid: [f64; 3],
    /// `|A - B| / 2`.
    half: f64,
    /// `Σ (beta0 + beta1)`: the bound at `R = 0` (an upper bound everywhere).
    total: f64,
    /// `Σ (beta0 g0 + beta1 g1)`: the far-field numerator, `estimate <= sum_bg / R_c`.
    sum_bg: f64,
}

#[inline]
fn tri(s1: usize, s2: usize) -> usize {
    // s1 >= s2 assumed
    s1 * (s1 + 1) / 2 + s2
}

/// Max absolute column sum of the cart→sph matrix for shell `l` (1 for
/// Cartesian or `l < 2`, where ferric never builds pure shells).
fn pure_factor(l: usize, pure: bool) -> f64 {
    if !pure || l < 2 {
        return 1.0;
    }
    let c = ferric_cart2sph(l);
    let nf = 2 * l + 1;
    let ncart = (l + 1) * (l + 2) / 2;
    (0..nf).map(|j| (0..ncart).map(|n| c[n * nf + j].abs()).sum::<f64>()).fold(0.0, f64::max)
}

/// Binomial coefficient for the small `l` in play.
fn binom(n: usize, k: usize) -> f64 {
    let mut v = 1.0_f64;
    for i in 0..k {
        v = v * (n - i) as f64 / (i + 1) as f64;
    }
    v
}

/// `sup_rho rho^k e^{-p rho^2 / 2} = (k / (p e))^{k/2}`; `1` for `k = 0`.
fn m_k(k: usize, p: f64) -> f64 {
    if k == 0 {
        1.0
    } else {
        (k as f64 / (p * std::f64::consts::E)).powf(0.5 * k as f64)
    }
}

impl PairBounds {
    /// Build the pair bounds for `prep`. Errors on `l > 4` (the shared
    /// normalization table stops there, as does the MD kernel).
    pub fn build(prep: &PreparedBasis) -> Result<Self, FerricError> {
        let nsh = prep.nshells();
        let shells = prep.located_shells();
        let npair = nsh * (nsh + 1) / 2;
        let mut term_start = Vec::with_capacity(npair + 1);
        let mut terms = Vec::new();
        let mut coarse = Vec::with_capacity(npair);

        struct Sh {
            l: usize,
            center: [f64; 3],
            exps: Vec<f64>,
            coefs: Vec<f64>,
            sfac: f64,
        }
        let mut sh = Vec::with_capacity(nsh);
        for (s, ls) in shells.iter().enumerate() {
            if ls.l < 0 || ls.l as usize > MAX_L {
                return Err(FerricError::Basis(format!("cosx_screen: shell {s} has l={} (supported 0..={MAX_L})", ls.l)));
            }
            if ls.exponents.len() != ls.coefficients.len() {
                return Err(FerricError::Basis(format!("cosx_screen: shell {s} exponent/coefficient length mismatch")));
            }
            let l = ls.l as usize;
            sh.push(Sh {
                l,
                center: ls.center,
                exps: ls.exponents.to_vec(),
                coefs: ls.exponents.iter().zip(ls.coefficients).map(|(&a, &c)| (c * prim_norm(a, l)).abs()).collect(),
                sfac: pure_factor(l, ls.pure),
            });
        }

        for s1 in 0..nsh {
            for s2 in 0..=s1 {
                term_start.push(terms.len() as u32);
                let (sa, sb) = (&sh[s1], &sh[s2]);
                let (la, lb) = (sa.l, sb.l);
                let q = [sa.center[0] - sb.center[0], sa.center[1] - sb.center[1], sa.center[2] - sb.center[2]];
                let ab2 = q[0] * q[0] + q[1] * q[1] + q[2] * q[2];
                let ab = ab2.sqrt();
                let sfac = sa.sfac * sb.sfac;
                let mut tot = 0.0_f64;
                let mut sum_bg = 0.0_f64;
                for (&a, &ca) in sa.exps.iter().zip(&sa.coefs) {
                    for (&b, &cb) in sb.exps.iter().zip(&sb.coefs) {
                        let p = a + b;
                        let kab = (-(a * b / p) * ab2).exp();
                        let w = ca * cb * kab * sfac;
                        if w == 0.0 {
                            continue;
                        }
                        let cen = [
                            (a * sa.center[0] + b * sb.center[0]) / p,
                            (a * sa.center[1] + b * sb.center[1]) / p,
                            (a * sa.center[2] + b * sb.center[2]) / p,
                        ];
                        // d_A = |P-A| = b/p |A-B|, d_B = a/p |A-B|.
                        let da = b / p * ab;
                        let db = a / p * ab;
                        // q_k: coefficients of rho^k in (rho+d_A)^{la} (rho+d_B)^{lb}.
                        let mut qk = [0.0_f64; 2 * MAX_L + 1];
                        for m in 0..=la {
                            let ca_m = binom(la, m) * da.powi((la - m) as i32);
                            for n in 0..=lb {
                                qk[m + n] += ca_m * binom(lb, n) * db.powi((lb - n) as i32);
                            }
                        }
                        let beta0 = w * (2.0 * PI / p) * qk[0];
                        let mut s1k = 0.0_f64;
                        for (k, &qv) in qk.iter().enumerate().skip(1) {
                            if qv != 0.0 {
                                s1k += qv * m_k(k, p);
                            }
                        }
                        let beta1 = w * (4.0 * PI / p) * s1k;
                        let g0 = 0.5 * (PI / p).sqrt();
                        let g1 = 0.5 * (2.0 * PI / p).sqrt();
                        tot += beta0 + beta1;
                        sum_bg += beta0 * g0 + beta1 * g1;
                        terms.push(PrimTerm { cen, beta0: beta0 * ROUNDING_SLACK, beta1: beta1 * ROUNDING_SLACK, g0, g1 });
                    }
                }
                // Heaviest terms first so the early-exit in `exceeds` usually
                // needs one term for a kept pair.
                let lo = *term_start.last().expect("pushed above") as usize;
                terms[lo..].sort_by(|x, y| (y.beta0 + y.beta1).partial_cmp(&(x.beta0 + x.beta1)).expect("finite weights"));
                coarse.push(Coarse {
                    mid: [
                        0.5 * (sa.center[0] + sb.center[0]),
                        0.5 * (sa.center[1] + sb.center[1]),
                        0.5 * (sa.center[2] + sb.center[2]),
                    ],
                    half: 0.5 * ab,
                    total: tot * ROUNDING_SLACK * COARSE_SLACK,
                    sum_bg: sum_bg * ROUNDING_SLACK * COARSE_SLACK,
                });
            }
        }
        term_start.push(terms.len() as u32);
        Ok(Self { nsh, term_start, terms, coarse })
    }

    /// Number of shells this was built for.
    pub fn nshells(&self) -> usize {
        self.nsh
    }

    /// Number of primitive-pair terms across all shell pairs (cost proxy).
    pub fn nterms(&self) -> usize {
        self.terms.len()
    }

    /// The bound's value at `R = 0` for pair `(s1, s2)` — an upper bound for
    /// [`PairBounds::estimate`] at every point.
    #[inline]
    pub fn max_estimate(&self, s1: usize, s2: usize) -> f64 {
        let (s1, s2) = if s1 >= s2 { (s1, s2) } else { (s2, s1) };
        self.coarse[tri(s1, s2)].total
    }

    /// One-sqrt upper bound on [`PairBounds::estimate`]`(s1, s2, r)`:
    /// `min(total, sum_bg / R_c)` with `R_c = max(0, |M - r| - |AB|/2)`.
    /// Valid because every product centre lies on the segment `AB` (so its
    /// `R >= R_c`) and each term is non-increasing in `R` with
    /// `beta F_0^+(...) <= beta g / R`. Anchored as `coarse >= fine` on the
    /// same probe set as the underestimation anchor.
    #[inline]
    pub fn coarse_estimate(&self, s1: usize, s2: usize, r: &[f64; 3]) -> f64 {
        let (s1, s2) = if s1 >= s2 { (s1, s2) } else { (s2, s1) };
        let c = &self.coarse[tri(s1, s2)];
        let dx = r[0] - c.mid[0];
        let dy = r[1] - c.mid[1];
        let dz = r[2] - c.mid[2];
        let rc = (dx * dx + dy * dy + dz * dz).sqrt() - c.half;
        if rc <= 0.0 {
            c.total
        } else {
            c.total.min(c.sum_bg / rc)
        }
    }

    /// Upper bound on [`PairBounds::estimate`]`(s1, s2, r)` for EVERY `r`
    /// within `radius` of `centre` — the coarse bound at
    /// `R_c = max(0, |M - centre| - radius - |AB|/2)`. This is the O(1)-per-pair
    /// test a K builder should run once per spatially local batch; only pairs
    /// that pass it need any per-point work.
    #[inline]
    pub fn coarse_estimate_sphere(&self, s1: usize, s2: usize, centre: &[f64; 3], radius: f64) -> f64 {
        let (s1, s2) = if s1 >= s2 { (s1, s2) } else { (s2, s1) };
        let c = &self.coarse[tri(s1, s2)];
        let dx = centre[0] - c.mid[0];
        let dy = centre[1] - c.mid[1];
        let dz = centre[2] - c.mid[2];
        let rc = (dx * dx + dy * dy + dz * dz).sqrt() - radius - c.half;
        if rc <= 0.0 {
            c.total
        } else {
            c.total.min(c.sum_bg / rc)
        }
    }

    /// `estimate(s1, s2, r) >= threshold`, evaluated cheaply: an O(1)
    /// early-out on the `R = 0` value, then the one-sqrt coarse bound, then
    /// the per-term sum (heaviest first) stopping as soon as the partial sum
    /// reaches the threshold — valid because every term is non-negative.
    /// Identical decisions to comparing [`PairBounds::estimate`] directly.
    #[inline]
    pub fn exceeds(&self, s1: usize, s2: usize, r: &[f64; 3], threshold: f64) -> bool {
        let (s1, s2) = if s1 >= s2 { (s1, s2) } else { (s2, s1) };
        let idx = tri(s1, s2);
        if self.coarse[idx].total < threshold || self.coarse_estimate(s1, s2, r) < threshold {
            return false;
        }
        let (lo, hi) = (self.term_start[idx] as usize, self.term_start[idx + 1] as usize);
        let mut acc = 0.0_f64;
        for t in &self.terms[lo..hi] {
            let dx = r[0] - t.cen[0];
            let dy = r[1] - t.cen[1];
            let dz = r[2] - t.cen[2];
            let d = (dx * dx + dy * dy + dz * dz).sqrt();
            acc += if d <= t.g0 {
                t.beta0 + t.beta1
            } else {
                let inv = 1.0 / d;
                t.beta0 * (t.g0 * inv) + t.beta1 * (t.g1 * inv).min(1.0)
            };
            if acc >= threshold {
                return true;
            }
        }
        false
    }

    /// Upper bound on `max_{mu in s1, nu in s2} |A^g_{mu,nu}|` at point `r`.
    #[inline]
    pub fn estimate(&self, s1: usize, s2: usize, r: &[f64; 3]) -> f64 {
        let (s1, s2) = if s1 >= s2 { (s1, s2) } else { (s2, s1) };
        let idx = tri(s1, s2);
        let (lo, hi) = (self.term_start[idx] as usize, self.term_start[idx + 1] as usize);
        let mut acc = 0.0_f64;
        for t in &self.terms[lo..hi] {
            let dx = r[0] - t.cen[0];
            let dy = r[1] - t.cen[1];
            let dz = r[2] - t.cen[2];
            let d = (dx * dx + dy * dy + dz * dz).sqrt();
            if d <= t.g0 {
                // Both F_0^+ factors saturate at 1 (g1 > g0).
                acc += t.beta0 + t.beta1;
            } else {
                let inv = 1.0 / d;
                acc += t.beta0 * (t.g0 * inv) + t.beta1 * (t.g1 * inv).min(1.0);
            }
        }
        acc
    }

    /// Whether `estimate(s1, s2, r) >= threshold` for ANY `r` in `pts`, with
    /// an O(1) early-out when even the `R = 0` value is below threshold.
    /// This is the batch-level keep rule of `md3c1e`.
    pub fn any_exceeds(&self, s1: usize, s2: usize, pts: &[[f64; 3]], threshold: f64) -> bool {
        if self.max_estimate(s1, s2) < threshold {
            return false;
        }
        pts.iter().any(|r| self.exceeds(s1, s2, r, threshold))
    }
}
