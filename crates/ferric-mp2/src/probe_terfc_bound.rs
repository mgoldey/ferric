//! THROWAWAY PROBE (branch research/terfc-screening-bound). NOT production code.
//!
//! Distance-aware screening bounds for attenuated-Coulomb kernels K(r)=att(r)/r
//! (terfc / erfc / Coulomb), built ONLY from analytic shell data (no integrals):
//!
//!  * absolute-value Gaussian majorant of every shell function:
//!    |phi_f(r)| <= M(r) = sum_i d_i exp(-beta_i |r-A|^2)   (all components f of
//!    the shell; uses |Y_lm|^2 <= (2l+1)/4pi and r^l e^{-a r^2} <= m_l e^{-f a r^2})
//!  * shell-pair majorant M_a M_b = sum_ij w_ij exp(-p_ij |r-P_ij|^2)
//!  * Bound 1 (B1): N_ab N_cd K(R_gap), N = L1 mass of the majorant, R_gap = centre
//!    distance minus the two mass-quantile extents (delta-controlled, NOT rigorous
//!    beyond delta).
//!  * Bound 2 (B2): rigorous (no extents): att(r) <= C_eps exp(-kappa_eps r^2) then
//!    1/r = (2/sqrt(pi)) int_0^inf exp(-t^2 r^2) dt, so the bound is a sum of
//!    analytic ssss Gaussian integrals followed by a 1-D t quadrature.
#![allow(clippy::too_many_arguments)]

use ferric_integrals::ao_grid::LocatedShell;
use std::f64::consts::PI;

/// erfc to ~1e-15 relative (Taylor for |x|<2.5, continued fraction beyond).
pub fn erfc(x: f64) -> f64 {
    if x < 0.0 {
        return 2.0 - erfc(-x);
    }
    if x < 2.5 {
        let mut sum = 0.0;
        let mut term = x;
        let x2 = x * x;
        let mut n = 0.0;
        for _ in 0..200 {
            sum += term / (2.0 * n + 1.0);
            n += 1.0;
            term *= -x2 / n;
            if term.abs() < 1e-18 {
                break;
            }
        }
        1.0 - 2.0 / PI.sqrt() * sum
    } else {
        // erfc(x) = exp(-x^2)/sqrt(pi) * 1/(x+ (1/2)/(x+ 1/(x+ (3/2)/(x+ ...))))
        let mut f = x;
        for k in (1..=80).rev() {
            f = x + (k as f64 * 0.5) / f;
        }
        (-x * x).exp() / PI.sqrt() / f
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum KKind {
    Coul,
    Erfc,
    Terfc,
}

/// Radial kernel K(r) = att(r)/r.  r0, w in Bohr, Bohr^-1.
#[derive(Clone, Copy, Debug)]
pub struct Kern {
    pub kind: KKind,
    pub r0: f64,
    pub w: f64,
}

impl Kern {
    pub fn att(&self, r: f64) -> f64 {
        match self.kind {
            KKind::Coul => 1.0,
            KKind::Erfc => erfc(self.w * r),
            KKind::Terfc => 0.5 * (erfc(self.w * (r - self.r0)) + erfc(self.w * (r + self.r0))),
        }
    }
    pub fn k(&self, r: f64) -> f64 {
        self.att(r) / r
    }
    /// att(r) <= c * exp(-kappa r^2) for all r >= 0 (eps in (0,1]; eps=1 -> (1,0) trivial).
    pub fn gmaj(&self, eps: f64) -> (f64, f64) {
        match self.kind {
            KKind::Coul => (1.0, 0.0),
            KKind::Erfc => (1.0, self.w * self.w),
            KKind::Terfc => {
                let w2 = self.w * self.w;
                ((w2 * self.r0 * self.r0 * (1.0 / eps - 1.0)).exp(), w2 * (1.0 - eps))
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct ShellMaj {
    pub center: [f64; 3],
    pub d: Vec<f64>,
    pub beta: Vec<f64>,
    pub nprim: usize,
    pub nfn: usize,
}

fn dfact_odd(n: i32) -> f64 {
    // (2l-1)!! style: n = 2l-1
    let mut r = 1.0;
    let mut k = n;
    while k > 1 {
        r *= k as f64;
        k -= 2;
    }
    r
}

/// Gamma(l + 3/2)
fn gamma_l32(l: i32) -> f64 {
    // Gamma(l+3/2) = (2l+1)!! / 2^(l+1) * sqrt(pi)
    dfact_odd(2 * l + 1) / 2f64.powi(l + 1) * PI.sqrt()
}

/// Majorant of a (contracted) shell.  `f` in (0,1]: r^l e^{-a r^2} <= m e^{-f a r^2}.
pub fn shell_maj(sh: &LocatedShell, f: f64) -> ShellMaj {
    let l = sh.l;
    assert!(sh.pure || l <= 1, "probe: Cartesian l>=2 unsupported");
    let ff = if l == 0 { 1.0 } else { f };
    // coefficients of un-normalized r^l e^{-a r^2}: c * N(a,l)  (ao_grid::radial convention)
    let cn: Vec<f64> = sh
        .exponents
        .iter()
        .zip(sh.coefficients)
        .map(|(&a, &c)| {
            let n = (2.0 * a / PI).powf(0.75) * (4.0 * a).powi(l).sqrt() / dfact_odd(2 * l - 1).sqrt();
            c * n
        })
        .collect();
    // radial norm: int R^2 r^2 dr = sum_ij c_i c_j Gamma(l+3/2) / (2 (a_i+a_j)^(l+3/2))
    let mut nn = 0.0;
    for i in 0..cn.len() {
        for j in 0..cn.len() {
            let s = sh.exponents[i] + sh.exponents[j];
            nn += cn[i] * cn[j] * gamma_l32(l) / (2.0 * s.powf(l as f64 + 1.5));
        }
    }
    let nrad = 1.0 / nn.sqrt();
    let ang = ((2 * l + 1) as f64 / (4.0 * PI)).sqrt();
    let mut d = vec![];
    let mut beta = vec![];
    for (i, &a) in sh.exponents.iter().enumerate() {
        let m = if l == 0 {
            1.0
        } else {
            (l as f64 / (2.0 * std::f64::consts::E * (1.0 - ff) * a)).powf(l as f64 / 2.0)
        };
        d.push(ang * nrad * cn[i].abs() * m);
        beta.push(ff * a);
    }
    let nfn = if sh.pure { 2 * l as usize + 1 } else { ((l + 1) * (l + 2) / 2) as usize };
    ShellMaj { center: sh.center, d, beta, nprim: sh.exponents.len(), nfn }
}

#[derive(Clone, Debug)]
pub struct Prim {
    pub w: f64,
    pub p: f64,
    pub c: [f64; 3],
}

#[derive(Clone, Debug)]
pub struct PairMaj {
    pub prims: Vec<Prim>,
    pub mass: f64,
    pub cen: [f64; 3],
    pub nprim: usize, // contracted primitive-pair count (cost weight)
    pub nfn: usize,
}

pub fn dist2(a: [f64; 3], b: [f64; 3]) -> f64 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)
}

fn finish(prims: Vec<Prim>, nprim: usize, nfn: usize) -> PairMaj {
    let mut mass = 0.0;
    let mut cen = [0.0; 3];
    for q in &prims {
        let m = q.w * (PI / q.p).powf(1.5);
        mass += m;
        for k in 0..3 {
            cen[k] += m * q.c[k];
        }
    }
    for c in cen.iter_mut() {
        *c /= mass;
    }
    // prune negligible primitive pairs (relative mass < 1e-14)
    let prims: Vec<Prim> = prims
        .into_iter()
        .filter(|q| q.w * (PI / q.p).powf(1.5) >= 1e-14 * mass)
        .collect();
    PairMaj { prims, mass, cen, nprim, nfn }
}

pub fn pair_maj(a: &ShellMaj, b: &ShellMaj) -> PairMaj {
    let r2 = dist2(a.center, b.center);
    let mut prims = vec![];
    for (i, &bi) in a.beta.iter().enumerate() {
        for (j, &bj) in b.beta.iter().enumerate() {
            let p = bi + bj;
            let mu = bi * bj / p;
            let w = a.d[i] * b.d[j] * (-mu * r2).exp();
            let mut c = [0.0; 3];
            for k in 0..3 {
                c[k] = (bi * a.center[k] + bj * b.center[k]) / p;
            }
            prims.push(Prim { w, p, c });
        }
    }
    finish(prims, a.nprim * b.nprim, a.nfn * b.nfn)
}

pub fn single_maj(a: &ShellMaj) -> PairMaj {
    let prims = a
        .beta
        .iter()
        .zip(&a.d)
        .map(|(&p, &w)| Prim { w, p, c: a.center })
        .collect();
    finish(prims, a.nprim, a.nfn)
}

/// fraction of an isotropic 3-D Gaussian exp(-p r^2) mass outside radius rho
fn frac_out(p: f64, rho: f64) -> f64 {
    if rho <= 0.0 {
        return 1.0;
    }
    let x = p.sqrt() * rho;
    erfc(x) + 2.0 / PI.sqrt() * x * (-x * x).exp()
}

impl PairMaj {
    /// radius e around `cen` outside which at most `delta` of the L1 mass lies
    pub fn extent(&self, delta: f64) -> f64 {
        let ok = |e: f64| -> bool {
            let mut out = 0.0;
            for q in &self.prims {
                let m = q.w * (PI / q.p).powf(1.5);
                out += m * frac_out(q.p, e - dist2(q.c, self.cen).sqrt());
            }
            out <= delta * self.mass
        };
        let (mut lo, mut hi) = (0.0f64, 200.0f64);
        for _ in 0..60 {
            let mid = 0.5 * (lo + hi);
            if ok(mid) {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        hi
    }
}

/// B1: N1 N2 K(gap); +inf (vacuous) when the extents overlap.
pub fn b1(p1: &PairMaj, e1: f64, p2: &PairMaj, e2: f64, k: &Kern) -> f64 {
    let gap = dist2(p1.cen, p2.cen).sqrt() - e1 - e2;
    if gap <= 1e-6 {
        return f64::INFINITY;
    }
    p1.mass * p2.mass * k.k(gap)
}

const GLX: [f64; 3] = [0.2386191860831969, 0.6612093864662645, 0.9324695142031521];
const GLW: [f64; 3] = [0.4679139345726910, 0.3607615730481386, 0.1713244923791704];

/// (2/sqrt(pi)) int_0^inf dt  I(kappa+t^2),
/// I(tau) = int int e^{-p|r1-P|^2} e^{-tau r12^2} e^{-q|r2-Q|^2}.
pub fn gd_ssss(p: f64, q: f64, r2: f64, kappa: f64) -> f64 {
    let a = p * q / (p + q);
    let t0 = (a + kappa).sqrt();
    let integrand = |t: f64| -> f64 {
        let tau = kappa + t * t;
        let den = p * q + (p + q) * tau;
        PI.powi(3) / den.powf(1.5) * (-p * q * tau * r2 / den).exp()
    };
    let width = if r2 > 0.0 { (a + kappa).sqrt() / (a * r2.sqrt()) } else { t0 };
    let tmin = 1e-3 * t0.min(width.max(1e-12));
    let tmax = 40.0 * t0;
    let mut total = 0.0;
    let mut lo = 0.0;
    let mut hi = tmin;
    loop {
        let mid = 0.5 * (lo + hi);
        let hw = 0.5 * (hi - lo);
        let mut s = 0.0;
        for i in 0..3 {
            s += GLW[i] * (integrand(mid + hw * GLX[i]) + integrand(mid - hw * GLX[i]));
        }
        total += hw * s;
        if hi >= tmax {
            break;
        }
        lo = hi;
        hi *= 2.0;
    }
    // rigorous remainder: I <= pi^3 ((p+q) t^2)^(-3/2)  (drops a,kappa,exp<=1)
    total += PI.powi(3) * (p + q).powf(-1.5) / (2.0 * hi * hi);
    2.0 / PI.sqrt() * total
}

pub const EPS_GRID: [f64; 12] = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];

/// B2: rigorous Gaussian-majorant bound, minimised over the eps grid.
pub fn b2(p1: &PairMaj, p2: &PairMaj, k: &Kern) -> f64 {
    let mut best = f64::INFINITY;
    let grid: &[f64] = if k.kind == KKind::Terfc { &EPS_GRID } else { &[1.0] };
    for &eps in grid {
        let (c, kappa) = k.gmaj(eps);
        if !c.is_finite() || c > 1e300 {
            continue;
        }
        let mut s = 0.0;
        for u in &p1.prims {
            for v in &p2.prims {
                s += u.w * v.w * gd_ssss(u.p, v.p, dist2(u.c, v.c), kappa);
            }
        }
        best = best.min(c * s);
    }
    best
}

// ---------------------------------------------------------------------------
// lmp2_direct hook (throwaway): count / apply distance-aware skips on the
// triples that the production maps (aux_radius, ao_tail, virt_radius) leave.
// ---------------------------------------------------------------------------
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;

pub struct ProbeSkip {
    pub tau: f64,
    pub apply: bool,
    pub rule: u8, // 0 = schwarz, 1 = B1, 2 = min(S,B1)
    pub kern: Kern,
    pub ext_p: Vec<f64>,
    pub ext_k: Vec<f64>,
    pub pm: Vec<PairMaj>,
    pub pp: Vec<PairMaj>,
    pub qa: Vec<f64>,
    pub qp: Vec<f64>,
    /// evaluate the (expensive) rigorous B2 on every b2_stride-th seen triple (0 = never)
    pub b2_stride: u64,
    /// COUNT-ONLY: skip EVERY triple after counting (energy is garbage; only CNT is meaningful)
    pub no_eval: bool,
    /// bitsets over (sp * npairs + k): unique map-surviving triples / their S and S|B1 skippability
    pub seen: Vec<AtomicU64>,
    pub sk_s: Vec<AtomicU64>,
    pub sk_cmb: Vec<AtomicU64>,
}
pub static PROBE: RwLock<Option<ProbeSkip>> = RwLock::new(None);
/// [seen, seen_weight(x1e0), skippable_S, skippable_B1, skippable_cmb, w_S, w_B1, w_cmb]
pub static CNT: [AtomicU64; 12] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];

/// true => the caller must skip this (sp, ua, ub) triple.
pub fn probe_skip(sp: usize, ua: usize, ub: usize) -> bool {
    let g = PROBE.read().unwrap();
    let Some(p) = g.as_ref() else { return false };
    let (a, b) = if ua >= ub { (ua, ub) } else { (ub, ua) };
    let k = a * (a + 1) / 2 + b;
    let s = p.qa[sp] * p.qp[k] < p.tau;
    let bb = b1(&p.pp[sp], p.ext_p[sp], &p.pm[k], p.ext_k[k], &p.kern) < p.tau;
    let w = (p.pp[sp].nprim * p.pm[k].nprim * p.pp[sp].nfn * p.pm[k].nfn) as u64;
    {
        let bit = sp * p.pm.len() + k;
        let (word, m) = (bit / 64, 1u64 << (bit % 64));
        p.seen[word].fetch_or(m, Ordering::Relaxed);
        if s {
            p.sk_s[word].fetch_or(m, Ordering::Relaxed);
        }
        if s || bb {
            p.sk_cmb[word].fetch_or(m, Ordering::Relaxed);
        }
    }
    let seen = CNT[0].fetch_add(1, Ordering::Relaxed);
    if p.b2_stride > 0 && seen % p.b2_stride == 0 {
        let v = b2(&p.pp[sp], &p.pm[k], &p.kern) < p.tau;
        CNT[8].fetch_add(1, Ordering::Relaxed);
        if s {
            CNT[9].fetch_add(1, Ordering::Relaxed);
        }
        if v {
            CNT[10].fetch_add(1, Ordering::Relaxed);
        }
        if s || v {
            CNT[11].fetch_add(1, Ordering::Relaxed);
        }
    }
    CNT[1].fetch_add(w, Ordering::Relaxed);
    if s {
        CNT[2].fetch_add(1, Ordering::Relaxed);
        CNT[5].fetch_add(w, Ordering::Relaxed);
    }
    if bb {
        CNT[3].fetch_add(1, Ordering::Relaxed);
        CNT[6].fetch_add(w, Ordering::Relaxed);
    }
    if s || bb {
        CNT[4].fetch_add(1, Ordering::Relaxed);
        CNT[7].fetch_add(w, Ordering::Relaxed);
    }
    p.no_eval
        || p.apply
            && match p.rule {
            0 => s,
            1 => bb,
            _ => s || bb,
        }
}

/// (ss|ss)-type Coulomb integral of two unnormalised Gaussians e^{-p|r-P|^2}, e^{-q|r-Q|^2}.
pub fn coul_ssss(p: f64, q: f64, r2: f64) -> f64 {
    let a = p * q / (p + q);
    let x = a * r2;
    let f0 = if x < 1e-10 { 1.0 - x / 3.0 } else { 0.5 * PI.sqrt() / x.sqrt() * (1.0 - erfc(x.sqrt())) };
    2.0 * PI.powf(2.5) / (p * q * (p + q).sqrt()) * f0
}

/// Primitive-pair-level bounds (u = bra primitive pair, v = ket primitive pair/aux primitive).
pub fn prim_b_coul(u: &Prim, v: &Prim) -> f64 {
    u.w * v.w * coul_ssss(u.p, v.p, dist2(u.c, v.c))
}
pub fn prim_b_kern(u: &Prim, v: &Prim, k: &Kern, grid: &[f64]) -> f64 {
    let mut best = f64::INFINITY;
    for &eps in grid {
        let (c, kappa) = k.gmaj(eps);
        if !c.is_finite() || c > 1e300 {
            continue;
        }
        best = best.min(c * u.w * v.w * gd_ssss(u.p, v.p, dist2(u.c, v.c), kappa));
    }
    best
}

pub fn new_bitset(nbits: usize) -> Vec<AtomicU64> {
    (0..nbits / 64 + 1).map(|_| AtomicU64::new(0)).collect()
}
/// (unique seen, unique S-skippable, unique S|B1-skippable)
pub fn uniq_counts() -> (u64, u64, u64) {
    let g = PROBE.read().unwrap();
    let Some(p) = g.as_ref() else { return (0, 0, 0) };
    let pc = |v: &Vec<AtomicU64>| v.iter().map(|x| x.load(Ordering::Relaxed).count_ones() as u64).sum();
    (pc(&p.seen), pc(&p.sk_s), pc(&p.sk_cmb))
}
