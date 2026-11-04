//! Split-RI-J: a direct (recompute-every-build) RI-J Coulomb builder that never
//! forms the three-index tensor `(P|μν)`.
//!
//! ```text
//!   d_P   = Σ_μν (P|μν) D_μν          c = V⁻¹ d          J_μν = Σ_P (P|μν) c_P
//! ```
//!
//! # Source and what "split" means here
//!
//! Neese, J. Comput. Chem. 24, 1740 (2003), "An improvement of the resolution of
//! the identity approximation for the formation of the Coulomb matrix" (ORCA's
//! default RI-J, `Split-RI-J`). The paper itself was not reachable when this
//! was written (Wiley paywall); what is confirmed from the abstract and the ORCA
//! 6.1 manual (§2.8.2) is: same Coulomb energy as standard RI-J, largest gains
//! for bases with many high-l functions (about 2x over standard RI-J), slightly
//! more memory, SCF and gradients only. The construction below is the
//! McMurchie-Davidson reading of the idea, and is exact (no new approximation
//! on top of RI-J): the three-centre integral is never completed. It is split
//! at the Hermite level, and the density or the fit coefficients are contracted
//! in between.
//!
//! With the bra pair expanded in Hermite Gaussians on the product centre P
//! (exponent p) and the aux function on its centre Q (exponent q),
//!
//! ```text
//!   (μν|K) = Σ_prims c_a c_b c_k · 2π^{5/2}/(p q √(p+q))
//!            · Σ_{tuv} E^{μν}_{tuv} Σ_{τνφ} (−1)^{τ+ν+φ} E^K_{τνφ} R_{t+τ,u+ν,v+φ}(α, P−Q),
//! ```
//!
//! * **d pass.** The density is pushed through the bra side FIRST, once per
//!   primitive pair: `H_tuv = c_a c_b Σ_{ij ∈ cart} D^{cart}_{ij} E^{ij}_{tuv}`
//!   (`D^{cart} = C_A D C_Bᵀ`, the spherical transform applied to the density,
//!   not to the integrals). Then, for every aux primitive,
//!   `W_{τνφ} += pref · Σ_tuv H_tuv R_{t+τ,u+ν,v+φ}` is summed over ALL bra
//!   primitive pairs before the aux Hermite→Cartesian→spherical transform is
//!   applied, once per aux primitive.
//! * **J pass.** Mirror image: the fit coefficients are pushed through the aux
//!   side first (`G_{τνφ} = c_k (−1)^{τ+ν+φ} Σ_c C^{cart}_c E^K_{c,τνφ}`, once
//!   per aux primitive), every bra primitive pair accumulates the Hermite
//!   potential `W_tuv = Σ_K pref Σ_{τνφ} G_{τνφ} R_{t+τ,…}` over all aux, and
//!   only then is it transformed to the Cartesian and spherical pair
//!   `J_μν += c_a c_b Σ_tuv E^{μν}_{tuv} W_tuv`.
//!
//! The inner work per (bra primitive pair, aux primitive) is therefore
//! the R tensor plus an `N_herm(l_a+l_b) × N_herm(l_K)` contraction. The
//! `n_cart(a)·n_cart(b)·n_cart(K)` integral block, its spherical transform and
//! its contraction with the density are never formed. Those are the terms that
//! grow fastest with the bra angular momenta.
//!
//! # Integral layer
//!
//! ferric's (P|μν) integrals come from libint2 as finished spherical shell
//! blocks, and the split cannot be applied to those. This module is therefore
//! its own McMurchie-Davidson Coulomb 3-centre kernel. It reuses the public,
//! tested pieces of [`ferric_integrals::md3c1e`] (Cartesian order, the
//! cart→sph tables for l ≤ 4, the Boys reference evaluator) and adds a
//! Boys table up to order [`MAX_LTOT`], an R recursion of arbitrary order,
//! and a solid-harmonic generator for aux shells with l = 5, 6.
//!
//! # Screening (consistent with RI-J's `integral_thresh`)
//!
//! Shell pairs are kept with the rule [`PairScreen`](ferric_integrals::three_index_source::PairScreen)
//! uses for stored RI-J: `Q_AB · max_P Q_P > thresh`. Within a kept pair, an
//! aux shell K is skipped when `Q_AB · Q_K <= thresh`, `Q_K = max_{P∈K} √(P|P)`.
//! By Schwarz, every element dropped either way satisfies `|(P|μν)| <= thresh`,
//! which is the same per-element guarantee the stored tier gives. Both passes
//! use the SAME kept set, so J is exactly `B̃ᵀ V⁻¹ B̃ D` for one effective
//! tensor `B̃`. The screen is density-independent: E_J stays a smooth quadratic
//! form in D. `thresh <= 0` keeps everything except primitive pairs whose
//! `|c_a c_b| e^{-μR²}` underflows [`PRIM_SCREEN`].
//!
//! # Far field (large T)
//!
//! When `T = α|P−Q|²` is large, `F_n(T)` is its asymptotic form
//! `Γ(n+½)/(2T^{n+½})` to within `e^{-T} T^{n+½}/((T−n)Γ(n+½))` relative
//! (the neglected tail of `∫_0^∞`). Then `R(α, X) = S(X)/√α`, where `S` is
//! the derivative tensor of `√π/(2|X|)`, and the term is
//! `2π^{5/2} (pq)^{-3/2} Σ H S`: a point-multipole interaction, with no
//! exponent left but the two charges. A term is far when `T >=
//! T_far[l_ab + l_K]`, the smallest `T` at which that bound is below
//! [`FAR_EPS`] = 1e-14 for every order (`far_t_table`). Every aux function
//! sits on one atom, so per (primitive pair, aux atom) one tensor `S(P−Q)` and
//! one contraction serve all of the atom's far aux primitives:
//!
//! * d pass: the field `M = p^{-3/2} Σ_j H_j S_{j+k}`, then each far aux
//!   primitive adds `2π^{5/2} q^{-3/2} M`;
//! * J pass: the far aux primitives' `q^{-3/2} G` are summed, then contracted
//!   once against `S`.
//!
//! This is the far field of CFMM (White & Head-Gordon 1994; the gated
//! `cfmm` module) reduced to RI-J: no tree and no translations, because the
//! aux side is single-centre and each primitive pair's Hermite expansion is
//! its own multipole. Both passes use the same far/near rule, so `J` is still
//! `B̃ᵀ V⁻¹ B̃ D` for one tensor `B̃`. The error per term is at most
//! `FAR_EPS` times the term's `|.|` sum (the R recursion is linear, and its
//! `|.|` recursion bounds the error), the scale the tests use. On
//! n-C20H42 / def2-SVP, 71% of terms are far and need 13x fewer tensor
//! evaluations; the d and J passes are 1.7x and 1.6x faster, with
//! `max|Δd_P| = 3e-13` on `|d_P| <= 1.1e3`. The `_abs` scale passes and
//! [`SplitRiJ::set_far_field`]`(false)` use the exact kernel for every term.
//!
//! # Determinism
//!
//! There are no cross-task reductions. The d pass parallelises over fixed
//! blocks of aux shells: each owns its `d_P` rows and loops over all pairs in
//! the stored order. The J pass parallelises over kept shell pairs: each owns
//! its J block. Hermite densities are built per pair and concatenated in
//! order. Every sum therefore has a fixed order, whatever the thread count.
//!
//! # Memory
//!
//! `O(Σ_primpairs N_herm(l_a+l_b))` for the Hermite densities (built per
//! build, freed after the d pass) plus `O(Σ_auxprims N_herm(l_K))`, plus the
//! naux² Cholesky factor DfJ also holds. There is no naux·nao² term.

use crate::fock::JBuilder;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::md3c1e;
use ferric_integrals::operator::{Operator, OperatorKind};
use ferric_integrals::threeindex::coulomb_metric_2c;
use ndarray::{Array1, Array2, OwnedRepr};
use ndarray_linalg::cholesky::{CholeskyFactorized, FactorizeC, SolveC};
use ndarray_linalg::UPLO;
use rayon::prelude::*;
use std::sync::OnceLock;

/// Highest orbital-basis angular momentum supported (g).
pub const MAX_L_OBS: usize = 4;
/// Highest auxiliary-basis angular momentum supported (i; def2-universal-jkfit's maximum).
pub const MAX_L_AUX: usize = 6;
const MAX_LAB: usize = 2 * MAX_L_OBS;
/// Highest total Hermite order `l_a + l_b + l_K`.
pub const MAX_LTOT: usize = MAX_LAB + MAX_L_AUX;
/// Extent of each axis of the Hermite index table.
const HX: usize = MAX_LTOT + 1;
const NONE: u32 = u32::MAX;
/// Primitive pairs with `|c_a c_b| e^{-ab/p |A-B|²}` below this are dropped:
/// an underflow guard, ~18 orders below any tolerance in the tree.
pub const PRIM_SCREEN: f64 = 1e-30;
/// Aux shells per d-pass task. A fixed partition: it decides only the
/// grouping of tasks, never a summation order (each aux primitive's sum runs
/// over all pairs in the stored order in every task layout).
const AUX_BLOCK: usize = 8;

/// Relative error allowed per far-field term (see the module doc, "Far field").
pub const FAR_EPS: f64 = 1e-14;

/// Split-RI-J builders constructed in this process (test observable).
static CONSTRUCTED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// How many Split-RI-J builders this process has constructed. Test hook: it
/// shows which RI-J builder an SCF took, which its energy bits cannot (the
/// free-atom guess SCFs take their own route).
#[doc(hidden)]
pub fn constructed_count() -> usize {
    CONSTRUCTED.load(std::sync::atomic::Ordering::Relaxed)
}

/// The memory pool label of a Split-RI-J builder's footprint.
const POOL_LABEL: &str = "Split-RI-J (metric factor, primitive pairs, per-build Hermite buffers)";

/// `2 π^{5/2}`.
const TWO_PI_52: f64 = 34.986_836_655_249_725;

#[inline]
fn n_herm(s: usize) -> usize {
    (s + 1) * (s + 2) * (s + 3) / 6
}

// ---------------------------------------------------------------------------
// Boys function, orders 0..=MAX_LTOT
// ---------------------------------------------------------------------------

const BOYS_T_SWITCH: f64 = 35.0;
const BOYS_TAYLOR: usize = 10;
const BOYS_NTAB: usize = MAX_LTOT + BOYS_TAYLOR;
const BOYS_STRIDE: usize = BOYS_NTAB + 1;
/// Nodes `k/8`, `k = 0..=281`: covers `round(8T) <= 280` for `T < 35`.
const BOYS_NODES: usize = 282;
const INV_FACT: [f64; BOYS_TAYLOR] = [
    1.0,
    1.0,
    1.0 / 2.0,
    1.0 / 6.0,
    1.0 / 24.0,
    1.0 / 120.0,
    1.0 / 720.0,
    1.0 / 5040.0,
    1.0 / 40320.0,
    1.0 / 362880.0,
];

fn boys_table() -> &'static [f64] {
    static TAB: OnceLock<Vec<f64>> = OnceLock::new();
    TAB.get_or_init(|| {
        let mut vals = vec![0.0_f64; BOYS_NODES * BOYS_STRIDE];
        for (k, row) in vals.chunks_mut(BOYS_STRIDE).enumerate() {
            md3c1e::boys_direct(BOYS_NTAB, k as f64 * 0.125, row);
        }
        vals
    })
}

/// `F_0(T)..F_nmax(T)` into `out[0..=nmax]`, `nmax <= MAX_LTOT`.
///
/// The structure of `md3c1e`'s evaluator: below `T = 35` a 10-term Taylor
/// expansion of the top order about the nearest `1/8` node (first omitted
/// term `< (1/16)^10/10! ≈ 2.5e-19` relative), then the stable downward
/// recursion; at and above 35 the asymptotic `F_0` (neglected tail `< 1e-16`
/// relative there) and upward recursion, stable while `2n+1 < 2T`, i.e. for
/// every `n <= MAX_LTOT = 14` at `T >= 35`.
pub fn boys(nmax: usize, t: f64, out: &mut [f64]) {
    debug_assert!(nmax <= MAX_LTOT && out.len() > nmax);
    let expt = exp_neg(t);
    if t < BOYS_T_SWITCH {
        let tab = boys_table();
        let k = (t * 8.0 + 0.5) as usize;
        let nd = k as f64 * 0.125 - t;
        let node = &tab[k * BOYS_STRIDE + nmax..k * BOYS_STRIDE + nmax + BOYS_TAYLOR];
        let mut f = node[BOYS_TAYLOR - 1] * INV_FACT[BOYS_TAYLOR - 1];
        for j in (0..BOYS_TAYLOR - 1).rev() {
            f = f * nd + node[j] * INV_FACT[j];
        }
        out[nmax] = f;
        let two_t = 2.0 * t;
        for n in (0..nmax).rev() {
            f = (two_t * f + expt) / (2 * n + 1) as f64;
            out[n] = f;
        }
    } else {
        let mut f = 0.5 * (std::f64::consts::PI / t).sqrt();
        out[0] = f;
        let inv_2t = 0.5 / t;
        for n in 0..nmax {
            f = ((2 * n + 1) as f64 * f - expt) * inv_2t;
            out[n + 1] = f;
        }
    }
}

/// `e^{-t}`, skipping the call where it is exactly zero: `exp` underflows
/// to `+0.0` for every argument below about `-745.13`, so `t >= 746` gives
/// the same bits without the call (16% of the Boys calls on n-C20H42).
#[inline]
fn exp_neg(t: f64) -> f64 {
    if t >= 746.0 {
        0.0
    } else {
        (-t).exp()
    }
}

/// [`boys`] for `LANES` arguments, into `out[n][lane]`. When every lane takes
/// the same branch the lanes run in lockstep (their division chains overlap);
/// otherwise each lane calls [`boys`]. Either way every lane performs
/// [`boys`]'s operations in its order, so the bits are the same.
#[inline]
fn boys_lanes(nmax: usize, t: &Lane, out: &mut [Lane]) {
    let small = t.map(|x| x < BOYS_T_SWITCH);
    let expt = t.map(exp_neg);
    if small.iter().all(|&b| b) {
        let tab = boys_table();
        let mut nd = [0.0; LANES];
        let mut base = [0usize; LANES];
        for i in 0..LANES {
            let k = (t[i] * 8.0 + 0.5) as usize;
            nd[i] = k as f64 * 0.125 - t[i];
            base[i] = k * BOYS_STRIDE + nmax;
        }
        let mut f = [0.0; LANES];
        for i in 0..LANES {
            f[i] = tab[base[i] + BOYS_TAYLOR - 1] * INV_FACT[BOYS_TAYLOR - 1];
        }
        for j in (0..BOYS_TAYLOR - 1).rev() {
            for i in 0..LANES {
                f[i] = f[i] * nd[i] + tab[base[i] + j] * INV_FACT[j];
            }
        }
        out[nmax] = f;
        let two_t = t.map(|x| 2.0 * x);
        for n in (0..nmax).rev() {
            let d = (2 * n + 1) as f64;
            for i in 0..LANES {
                f[i] = (two_t[i] * f[i] + expt[i]) / d;
            }
            out[n] = f;
        }
    } else if small.iter().all(|&b| !b) {
        let mut f = t.map(|x| 0.5 * (std::f64::consts::PI / x).sqrt());
        out[0] = f;
        let inv_2t = t.map(|x| 0.5 / x);
        for n in 0..nmax {
            let c = (2 * n + 1) as f64;
            for i in 0..LANES {
                f[i] = (c * f[i] - expt[i]) * inv_2t[i];
            }
            out[n + 1] = f;
        }
    } else {
        let mut fb = [0.0; MAX_LTOT + 1];
        for i in 0..LANES {
            boys(nmax, t[i], &mut fb);
            for n in 0..=nmax {
                out[n][i] = fb[n];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Hermite indexing, sum tables, R recursion
// ---------------------------------------------------------------------------

struct Hermite {
    /// `(t,u,v) -> canonical index`, flat `[t][u][v]`, extent `HX`.
    idx: Vec<u16>,
    /// Canonical order: increasing total order, then `t` desc, then `u` desc,
    /// so `{t+u+v <= s}` is the prefix of length `n_herm(s)`.
    list: Vec<[u8; 3]>,
}

impl Hermite {
    #[inline]
    fn index(&self, t: usize, u: usize, v: usize) -> usize {
        self.idx[(t * HX + u) * HX + v] as usize
    }
}

fn hermite() -> &'static Hermite {
    static H: OnceLock<Hermite> = OnceLock::new();
    H.get_or_init(|| {
        let mut idx = vec![u16::MAX; HX * HX * HX];
        let mut list = Vec::with_capacity(n_herm(MAX_LTOT));
        for s in 0..=MAX_LTOT {
            for t in (0..=s).rev() {
                for u in (0..=s - t).rev() {
                    let v = s - t - u;
                    idx[(t * HX + u) * HX + v] = list.len() as u16;
                    list.push([t as u8, u as u8, v as u8]);
                }
            }
        }
        Hermite { idx, list }
    })
}

/// `tab[k * n_herm(lab) + j] = index(t_j + τ_k, u_j + ν_k, v_j + φ_k)`: the
/// R^0 entry pairing bra Hermite `j` (order `<= lab`) with aux Hermite `k`
/// (order `<= lk`).
fn sum_table(lab: usize, lk: usize) -> &'static [u16] {
    static TABS: OnceLock<Vec<Vec<u16>>> = OnceLock::new();
    let tabs = TABS.get_or_init(|| {
        let h = hermite();
        let mut out = Vec::with_capacity((MAX_LAB + 1) * (MAX_L_AUX + 1));
        for lab in 0..=MAX_LAB {
            for lk in 0..=MAX_L_AUX {
                let (nj, nk) = (n_herm(lab), n_herm(lk));
                let mut t = Vec::with_capacity(nj * nk);
                for &[a, b, c] in &h.list[..nk] {
                    for &[x, y, z] in &h.list[..nj] {
                        t.push(
                            h.index((a + x) as usize, (b + y) as usize, (c + z) as usize) as u16,
                        );
                    }
                }
                out.push(t);
            }
        }
        out
    });
    &tabs[lab * (MAX_L_AUX + 1) + lk]
}

/// `R[dst] = X[dir] · R[src] + coef · R[src2]` (no second term when `src2 == NONE`).
struct RStep {
    dst: u32,
    src: u32,
    src2: u32,
    coef: f64,
    dir: u8,
}

/// The `R^n_{tuv}` recursion for one total order `L` over `[n][tuv]` storage
/// (block `n` holds `t+u+v <= L-n`; block 0 first, so the first `n_herm(L)`
/// entries are `R^0_{tuv}` in canonical order).
struct RProgram {
    steps: Vec<RStep>,
    nvec: usize,
    base: [usize; MAX_LTOT + 2],
}

fn r_programs() -> &'static [RProgram] {
    static P: OnceLock<Vec<RProgram>> = OnceLock::new();
    P.get_or_init(|| (0..=MAX_LTOT).map(build_r_program).collect())
}

fn build_r_program(l: usize) -> RProgram {
    let h = hermite();
    let mut base = [0usize; MAX_LTOT + 2];
    for n in 0..=l {
        base[n + 1] = base[n] + n_herm(l - n);
    }
    let mut steps = Vec::new();
    for s in 1..=l {
        for n in 0..=(l - s) {
            for &[t, u, v] in &h.list[n_herm(s - 1)..n_herm(s)] {
                let (t, u, v) = (t as usize, u as usize, v as usize);
                let dst = (base[n] + h.index(t, u, v)) as u32;
                let up = base[n + 1];
                // Helgaker eq. 9.9.18-20:
                // R^n_{t+1,u,v} = t R^{n+1}_{t-1,u,v} + X_PQ R^{n+1}_{t,u,v}.
                let (src, src2, coef, dir) = if t > 0 {
                    let s2 = if t >= 2 {
                        (up + h.index(t - 2, u, v)) as u32
                    } else {
                        NONE
                    };
                    ((up + h.index(t - 1, u, v)) as u32, s2, (t - 1) as f64, 0u8)
                } else if u > 0 {
                    let s2 = if u >= 2 {
                        (up + h.index(t, u - 2, v)) as u32
                    } else {
                        NONE
                    };
                    ((up + h.index(t, u - 1, v)) as u32, s2, (u - 1) as f64, 1u8)
                } else {
                    let s2 = if v >= 2 {
                        (up + h.index(t, u, v - 2)) as u32
                    } else {
                        NONE
                    };
                    ((up + h.index(t, u, v - 1)) as u32, s2, (v - 1) as f64, 2u8)
                };
                steps.push(RStep {
                    dst,
                    src,
                    src2,
                    coef,
                    dir,
                });
            }
        }
    }
    RProgram {
        steps,
        nvec: base[l + 1],
        base,
    }
}

/// `R^0_{tuv}(α, X)` for `t+u+v <= l` into `r[..n_herm(l)]` (canonical order).
/// `ABS` evaluates the same recursion on `|X|` and `|R^n_000|`: the sum of
/// the absolute values of every term, the running-error scale.
#[inline]
fn r_eval<const ABS: bool>(l: usize, alpha: f64, x: [f64; 3], fb: &mut [f64], r: &mut [f64]) {
    let prog = &r_programs()[l];
    let t = alpha * (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
    boys(l, t, fb);
    let m = if ABS { 2.0 * alpha } else { -2.0 * alpha };
    let mut pw = 1.0;
    for n in 0..=l {
        r[prog.base[n]] = pw * fb[n];
        pw *= m;
    }
    let xs = if ABS {
        [x[0].abs(), x[1].abs(), x[2].abs()]
    } else {
        x
    };
    for st in &prog.steps {
        let mut v = xs[st.dir as usize] * r[st.src as usize];
        if st.src2 != NONE {
            v += st.coef * r[st.src2 as usize];
        }
        r[st.dst as usize] = v;
    }
}

/// Primitive pairs evaluated together by [`r_eval_lanes`] and the batched
/// contractions: one `RStep` decode and one gather index serve every lane.
const LANES: usize = 4;

type Lane = [f64; LANES];

/// [`r_eval`] for `LANES` independent `(α, X)` at once, into `r[idx][lane]`.
/// Each lane runs exactly [`r_eval`]'s operations in the same order, so lane
/// `i` is bit-identical to a scalar call with `(alpha[i], x[.][i])`.
#[inline]
fn r_eval_lanes<const ABS: bool>(
    l: usize,
    alpha: &Lane,
    x: &[Lane; 3],
    fb: &mut [Lane],
    r: &mut [Lane],
) {
    let prog = &r_programs()[l];
    let mut t = [0.0; LANES];
    for i in 0..LANES {
        t[i] = alpha[i] * (x[0][i] * x[0][i] + x[1][i] * x[1][i] + x[2][i] * x[2][i]);
    }
    boys_lanes(l, &t, fb);
    for i in 0..LANES {
        let m = if ABS { 2.0 * alpha[i] } else { -2.0 * alpha[i] };
        let mut pw = 1.0;
        for n in 0..=l {
            r[prog.base[n]][i] = pw * fb[n][i];
            pw *= m;
        }
    }
    let xs = if ABS { x.map(|d| d.map(f64::abs)) } else { *x };
    for st in &prog.steps {
        let xd = &xs[st.dir as usize];
        let src = r[st.src as usize];
        let mut v = [0.0; LANES];
        for i in 0..LANES {
            v[i] = xd[i] * src[i];
        }
        if st.src2 != NONE {
            let s2 = r[st.src2 as usize];
            for i in 0..LANES {
                v[i] += st.coef * s2[i];
            }
        }
        r[st.dst as usize] = v;
    }
}

/// Bound on the relative error of the large-T form of the Boys function,
/// `F_n(T) ≈ Γ(n+½)/(2 T^{n+½})`: the neglected tail is
/// `∫_1^∞ t^{2n} e^{-Tt²} dt <= e^{-T} / (2(T−n))` (use `ln t² <= t²−1` and
/// `t²−1 >= 2(t−1)`), so the relative error is at most
/// `e^{-T} T^{n+½} / ((T−n) Γ(n+½))`. Returned in log form for `T > n`.
fn ln_far_rel_err(n: usize, t: f64) -> f64 {
    let nf = n as f64;
    // ln Γ(n+½) = ln((2n−1)!!) − n ln 2 + ½ ln π.
    let mut ln_gamma = 0.5 * std::f64::consts::PI.ln();
    for k in 1..=n {
        ln_gamma += ((2 * k - 1) as f64 / 2.0).ln();
    }
    -t + (nf + 0.5) * t.ln() - (t - nf).ln() - ln_gamma
}

/// `T_far[L]`: the smallest `T` (on a 1/4 grid) from which every order
/// `n <= L` of the large-T Boys form is within [`FAR_EPS`] relative. Each
/// bound decreases with `T` beyond `n + 2`, where the scan starts.
fn far_t_table() -> &'static [f64; MAX_LTOT + 1] {
    static T: OnceLock<[f64; MAX_LTOT + 1]> = OnceLock::new();
    T.get_or_init(|| {
        let ln_eps = FAR_EPS.ln();
        let mut out = [0.0; MAX_LTOT + 1];
        for (l, slot) in out.iter_mut().enumerate() {
            let mut t = l as f64 + 2.0;
            while (0..=l).any(|n| ln_far_rel_err(n, t) > ln_eps) {
                t += 0.25;
            }
            *slot = t;
        }
        out
    })
}

/// `R^0_{tuv}` for `t+u+v <= l` with the large-T Boys form and `α = 1`: the
/// derivative tensor of `√π/(2|X|)`. The exact `R(α, X)` at large `T` is this
/// divided by `√α` (each `R^n_000` scales as `α^{-1/2}`, and the recursion has
/// no `α`), so `pref · R = 2π^{5/2} (pq)^{-3/2} · S`: exponent-free apart
/// from the two charges.
fn r_eval_far(l: usize, x: [f64; 3], fb: &mut [f64], r: &mut [f64]) {
    let prog = &r_programs()[l];
    let t = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
    let mut f = 0.5 * (std::f64::consts::PI / t).sqrt();
    let inv_2t = 0.5 / t;
    let mut pw = 1.0;
    for n in 0..=l {
        fb[n] = f;
        r[prog.base[n]] = pw * f;
        f *= (2 * n + 1) as f64 * inv_2t;
        pw *= -2.0;
    }
    for st in &prog.steps {
        let mut v = x[st.dir as usize] * r[st.src as usize];
        if st.src2 != NONE {
            v += st.coef * r[st.src2 as usize];
        }
        r[st.dst as usize] = v;
    }
}

/// `P − Q` per lane for the primitive pairs of `chunk` (lanes past
/// `chunk.len()` repeat the first, and their results are discarded).
#[inline]
fn lane_offsets(chunk: &[&PrimPair], q: [f64; 3]) -> [Lane; 3] {
    let mut x = [[0.0; LANES]; 3];
    for i in 0..LANES {
        let pp = chunk.get(i).unwrap_or(&chunk[0]);
        for d in 0..3 {
            x[d][i] = pp.centre[d] - q[d];
        }
    }
    x
}

/// The reduced exponent `α = pq/(p+q)` and `2π^{5/2}/(pq√(p+q))` per lane,
/// written exactly as the scalar kernel writes them.
#[inline]
fn lane_alpha_pref(chunk: &[&PrimPair], qc: f64) -> (Lane, Lane) {
    let mut alpha = [0.0; LANES];
    let mut pref = [0.0; LANES];
    for i in 0..LANES {
        let p = chunk.get(i).unwrap_or(&chunk[0]).p;
        let pq = p + qc;
        alpha[i] = p * qc / pq;
        pref[i] = TWO_PI_52 / (p * qc * pq.sqrt());
    }
    (alpha, pref)
}

// ---------------------------------------------------------------------------
// Hermite E coefficients
// ---------------------------------------------------------------------------

/// `md3c1e::e_table`'s recursion and layout (`(i*(lb+1)+j)*(la+lb+1)+t`,
/// `q = A_x - B_x`); `ABS` uses `|X_PA|`, `|X_PB|` (every term non-negative).
fn e_table<const ABS: bool>(la: usize, lb: usize, a: f64, b: f64, q: f64, out: &mut [f64]) {
    let p = a + b;
    let mu = a * b / p;
    let inv2p = 0.5 / p;
    let (mut xpa, mut xpb) = (-(b / p) * q, (a / p) * q);
    if ABS {
        xpa = xpa.abs();
        xpb = xpb.abs();
    }
    let sj = la + lb + 1;
    let si = (lb + 1) * sj;
    out[..(la + 1) * si].iter_mut().for_each(|v| *v = 0.0);
    out[0] = (-mu * q * q).exp();
    for i in 0..la {
        for t in 0..=i + 1 {
            out[(i + 1) * si + t] = e_step(&out[i * si..i * si + sj], i, t, inv2p, xpa);
        }
    }
    for j in 0..lb {
        for i in 0..=la {
            let row = i * si + j * sj;
            for t in 0..=i + j + 1 {
                out[i * si + (j + 1) * sj + t] = e_step(&out[row..row + sj], i + j, t, inv2p, xpb);
            }
        }
    }
}

#[inline]
fn e_step(row: &[f64], ij_sum: usize, t: usize, inv2p: f64, xp: f64) -> f64 {
    let e_m1 = if t >= 1 { row[t - 1] } else { 0.0 };
    let e_0 = if t <= ij_sum { row[t] } else { 0.0 };
    let e_p1 = if t < ij_sum { row[t + 1] } else { 0.0 };
    inv2p * e_m1 + xp * e_0 + (t + 1) as f64 * e_p1
}

// ---------------------------------------------------------------------------
// Normalisation and cart -> sph
// ---------------------------------------------------------------------------

fn dfact(n: i32) -> f64 {
    let mut r = 1.0;
    let mut k = n;
    while k > 1 {
        r *= k as f64;
        k -= 2;
    }
    r
}

/// `N(a,l) = (2a/π)^{3/4} (4a)^{l/2} / √((2l-1)!!)`: `md3c1e::prim_norm`
/// for any `l` (unit-normalised `(l,0,0)` component).
fn prim_norm(a: f64, l: usize) -> f64 {
    (2.0 * a / std::f64::consts::PI).powf(0.75) * (4.0 * a).powi(l as i32).sqrt()
        / dfact(2 * l as i32 - 1).sqrt()
}

fn fac(n: i32) -> f64 {
    (1..=n.max(0)).fold(1.0, |a, k| a * k as f64)
}

fn binom(n: i32, k: i32) -> f64 {
    if k < 0 || k > n {
        return 0.0;
    }
    fac(n) / (fac(k) * fac(n - k))
}

fn parity(i: i32) -> i32 {
    if i % 2 != 0 {
        -1
    } else {
        1
    }
}

/// Real-solid-harmonic coefficient after libint2's `solidharmonics.h`
/// (`SolidHarmonicsCoefficients::coeff`), reproduced from memory: pure
/// function `m` is `Σ_c coeff(l,m,c) · cart_c`, `m = -l..=l` (libint2 STANDARD
/// order). Only the SHAPE and SIGN of each `m` column are relied on;
/// [`solid_harmonic_c2s`] fixes the scale.
fn solid_harmonic_coeff(l: i32, m: i32, lx: i32, ly: i32, lz: i32) -> f64 {
    let abs_m = m.abs();
    if (lx + ly - abs_m) % 2 != 0 {
        return 0.0;
    }
    let j = (lx + ly - abs_m) / 2;
    if j < 0 {
        return 0.0;
    }
    let comp = if m >= 0 { 1 } else { -1 };
    let i = abs_m - lx;
    if comp != parity(i.abs()) {
        return 0.0;
    }
    let mut pfac = ((fac(2 * lx) * fac(2 * ly) * fac(2 * lz)) / fac(2 * l)
        * (fac(l - abs_m) / fac(l))
        * (1.0 / fac(l + abs_m))
        * (fac(l) / (fac(lx) * fac(ly) * fac(lz))))
    .sqrt();
    pfac /= (1i64 << l) as f64;
    let sign = if m < 0 {
        parity((i - 1) / 2)
    } else {
        parity(i / 2)
    };
    pfac *= sign as f64;
    let i_min = j;
    let i_max = (l - abs_m) / 2;
    let mut sum = 0.0;
    for ii in i_min..=i_max {
        let mut pfac1 = binom(l, ii) * binom(ii, j);
        pfac1 *= parity(ii) as f64 * fac(2 * (l - ii)) / fac(l - abs_m - 2 * ii);
        let mut sum1 = 0.0;
        let k_min = ((lx - abs_m) / 2).max(0);
        let k_max = j.min(lx / 2);
        for k in k_min..=k_max {
            if lx - 2 * k <= abs_m {
                sum1 += binom(j, k) * binom(abs_m, lx - 2 * k) * parity(k) as f64;
            }
        }
        sum += pfac1 * sum1;
    }
    sum *= (dfact(2 * l - 1) / (dfact(2 * lx - 1) * dfact(2 * ly - 1) * dfact(2 * lz - 1))).sqrt();
    if m == 0 {
        pfac * sum
    } else {
        std::f64::consts::SQRT_2 * pfac * sum
    }
}

/// Angular overlap of two Cartesian components of one shell in ferric's
/// convention (one shell-wide normalisation, `(l,0,0)` unit-normalised):
/// `Π_x (a_x+b_x-1)!! / (2l-1)!!`, zero unless every `a_x+b_x` is even.
fn cart_metric(a: &[u8; 3], b: &[u8; 3], l: usize) -> f64 {
    let mut v = 1.0;
    for x in 0..3 {
        let s = (a[x] + b[x]) as i32;
        if s % 2 != 0 {
            return 0.0;
        }
        v *= dfact(s - 1);
    }
    v / dfact(2 * l as i32 - 1)
}

/// `(ncart × 2l+1)` row-major cart→sph matrix: the shape and sign of each
/// column from [`solid_harmonic_coeff`], and the scale fixed explicitly so
/// each pure function has unit norm under [`cart_metric`]. A hand check of
/// `l = 2, m = 0` showed the recalled prefactor is off by an `m`-independent
/// constant (√2 too large: `√2 zz − (xx+yy)/√2` instead of `zz − (xx+yy)/2`),
/// while the ratios are right. Normalising here removes any such constant;
/// the shape and sign are checked against md3c1e's libint2 tables for l ≤ 4
/// (`solid_harmonic_generator_matches_the_libint2_tables`).
#[doc(hidden)]
pub fn solid_harmonic_c2s(l: usize) -> Vec<f64> {
    let comps = md3c1e::cart_components(l);
    let (nc, nsph) = (comps.len(), 2 * l + 1);
    let mut out = vec![0.0; nc * nsph];
    for (ci, c) in comps.iter().enumerate() {
        for (mi, m) in (-(l as i32)..=l as i32).enumerate() {
            out[ci * nsph + mi] =
                solid_harmonic_coeff(l as i32, m, c[0] as i32, c[1] as i32, c[2] as i32);
        }
    }
    for mi in 0..nsph {
        let mut n2 = 0.0;
        for i in 0..nc {
            for k in 0..nc {
                n2 +=
                    out[i * nsph + mi] * cart_metric(&comps[i], &comps[k], l) * out[k * nsph + mi];
            }
        }
        let inv = 1.0 / n2.sqrt();
        for i in 0..nc {
            out[i * nsph + mi] *= inv;
        }
    }
    out
}

/// The cart→sph matrix used for a pure shell of `l`: `md3c1e`'s
/// libint2-validated table for `l <= 4`, the generator above for `l = 5, 6`.
fn c2s_matrix(l: usize) -> Vec<f64> {
    if l <= md3c1e::MAX_L {
        md3c1e::ferric_cart2sph(l)
    } else {
        solid_harmonic_c2s(l)
    }
}

// ---------------------------------------------------------------------------
// Shell data
// ---------------------------------------------------------------------------

struct Shell {
    l: usize,
    pure: bool,
    center: [f64; 3],
    exps: Vec<f64>,
    /// Contraction coefficients with `N(a, l)` folded in.
    coefs: Vec<f64>,
    comps: Vec<[u8; 3]>,
    ncart: usize,
    nfun: usize,
    off: usize,
}

fn load_shells(prep: &PreparedBasis, max_l: usize, what: &str) -> Result<Vec<Shell>, FerricError> {
    let dims = prep.shell_dims();
    let offs = prep.shell_offsets();
    let mut out = Vec::with_capacity(prep.nshells());
    for (s, sh) in prep.located_shells().iter().enumerate() {
        if sh.l < 0 || sh.l as usize > max_l {
            return Err(FerricError::Basis(format!(
                "split-RI-J: {what} shell {s} has l={} (supported 0..={max_l})",
                sh.l
            )));
        }
        let l = sh.l as usize;
        if sh.pure && l == 1 {
            return Err(FerricError::Basis(format!(
                "split-RI-J: {what} shell {s} is a pure p shell (libint2 orders it y,z,x; ferric never builds one)"
            )));
        }
        let pure = sh.pure && l >= 2;
        let ncart = (l + 1) * (l + 2) / 2;
        let nfun = if pure { 2 * l + 1 } else { ncart };
        if nfun != dims[s] || sh.exponents.len() != sh.coefficients.len() {
            return Err(FerricError::Basis(format!(
                "split-RI-J: {what} shell {s} (l={l}, pure={pure}) has {nfun} functions / {} exps / {} coefs, libint2 reports {} functions",
                sh.exponents.len(),
                sh.coefficients.len(),
                dims[s]
            )));
        }
        out.push(Shell {
            l,
            pure,
            center: sh.center,
            exps: sh.exponents.to_vec(),
            coefs: sh
                .exponents
                .iter()
                .zip(sh.coefficients)
                .map(|(&a, &c)| c * prim_norm(a, l))
                .collect(),
            comps: md3c1e::cart_components(l),
            ncart,
            nfun,
            off: offs[s],
        });
    }
    Ok(out)
}

#[derive(Clone, Copy, Debug)]
struct ShellPair {
    a: u32,
    b: u32,
    /// Schwarz bound `Q_AB`.
    q: f64,
    prim0: u32,
    prim1: u32,
}

#[derive(Clone, Copy, Debug)]
struct PrimPair {
    ea: f64,
    eb: f64,
    p: f64,
    centre: [f64; 3],
    cab: f64,
    h_off: usize,
}

#[derive(Default)]
struct Scratch {
    ex: Vec<f64>,
    ey: Vec<f64>,
    ez: Vec<f64>,
    r: Vec<f64>,
    fb: Vec<f64>,
    w: Vec<f64>,
    cart: Vec<f64>,
    tmp: Vec<f64>,
    /// Lane-major R tensor, Hermite densities and Hermite potentials.
    rl: Vec<Lane>,
    fbl: Vec<Lane>,
    hl: Vec<Lane>,
    wl: Vec<Lane>,
}

impl Scratch {
    fn new() -> Self {
        let ne = (MAX_L_OBS + 1) * (MAX_L_OBS + 1) * (MAX_LAB + 1);
        Scratch {
            ex: vec![0.0; ne],
            ey: vec![0.0; ne],
            ez: vec![0.0; ne],
            r: vec![0.0; r_programs()[MAX_LTOT].nvec],
            fb: vec![0.0; MAX_LTOT + 1],
            w: Vec::new(),
            cart: Vec::new(),
            tmp: Vec::new(),
            rl: vec![[0.0; LANES]; r_programs()[MAX_LTOT].nvec],
            fbl: vec![[0.0; LANES]; MAX_LTOT + 1],
            hl: Vec::new(),
            wl: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// The builder
// ---------------------------------------------------------------------------

/// Direct Split-RI-J Coulomb builder (see the module doc).
pub struct SplitRiJ {
    obs: Vec<Shell>,
    aux: Vec<Shell>,
    nao: usize,
    naux: usize,
    pairs: Vec<ShellPair>,
    prims: Vec<PrimPair>,
    h_len: usize,
    /// `Q_K = max_{P∈K} √(P|P)` per aux shell.
    aux_q: Vec<f64>,
    /// Offset of aux shell K's primitive Hermite vectors in the G store.
    g_off: Vec<usize>,
    g_len: usize,
    aux_blocks: Vec<(usize, usize)>,
    /// Aux shells grouped by centre: `(k0, k1)` runs of one atom.
    aux_atoms: Vec<(usize, usize)>,
    /// Far-field terms by the large-T form (see the module doc); off for the
    /// exact-kernel reference.
    far: bool,
    /// cart→sph (ncart × nsph) by `l`, and its elementwise absolute value.
    c2s: Vec<Vec<f64>>,
    c2s_abs: Vec<Vec<f64>>,
    thresh: f64,
    total_shell_pairs: usize,
    kept_blocks: usize,
    total_blocks: usize,
    v_chol: CholeskyFactorized<OwnedRepr<f64>>,
    /// The shared memory pool's record of [`Self::memory_bytes`], held for the
    /// builder's lifetime (soft: inert when no pool is installed or it is full,
    /// as for the stored tiers' last-resort planes).
    _charge: ferric_core::memory::pool::Reservation,
}

impl std::fmt::Debug for SplitRiJ {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SplitRiJ")
            .field("nao", &self.nao)
            .field("naux", &self.naux)
            .field("kept_shell_pairs", &self.pairs.len())
            .field("prim_pairs", &self.prims.len())
            .field("thresh", &self.thresh)
            .finish_non_exhaustive()
    }
}

impl SplitRiJ {
    /// Build for `obs`/`aux`, computing the aux Coulomb metric. `integral_thresh`
    /// is `[scf] integral_thresh` (`<= 0` keeps every shell pair and aux shell).
    pub fn new(
        op: Operator,
        obs: &PreparedBasis,
        aux: &PreparedBasis,
        integral_thresh: f64,
    ) -> Result<Self, FerricError> {
        let v = coulomb_metric_2c(op, aux)?;
        Self::with_metric(op, obs, aux, v, integral_thresh)
    }

    /// Whether [`Self::new`] accepts this operator and these bases: the plain
    /// Coulomb operator, orbital shells up to `l = MAX_L_OBS` and aux shells
    /// up to `l = MAX_L_AUX`. `jk_storage = "auto"` checks this before it
    /// picks Split-RI-J.
    pub fn supports(op: Operator, obs: &PreparedBasis, aux: &PreparedBasis) -> bool {
        op.kind == OperatorKind::Coulomb
            && !op.is_composite()
            && load_shells(obs, MAX_L_OBS, "orbital").is_ok()
            && load_shells(aux, MAX_L_AUX, "auxiliary").is_ok()
    }

    /// [`Self::new`] with the `(naux, naux)` metric `v` already built (it is
    /// consumed by the Cholesky factorisation; DfJ's `v_chol` convention).
    pub fn with_metric(
        op: Operator,
        obs: &PreparedBasis,
        aux: &PreparedBasis,
        v: Array2<f64>,
        integral_thresh: f64,
    ) -> Result<Self, FerricError> {
        if op.kind != OperatorKind::Coulomb || op.is_composite() {
            return Err(FerricError::General(format!(
                "split-RI-J supports the plain Coulomb operator only, got {:?}",
                op.kind
            )));
        }
        let obs_sh = load_shells(obs, MAX_L_OBS, "orbital")?;
        let aux_sh = load_shells(aux, MAX_L_AUX, "auxiliary")?;
        let (nao, naux) = (obs.nbasis(), aux.nbasis());
        if v.dim() != (naux, naux) {
            return Err(FerricError::General(format!(
                "split-RI-J: metric is {:?}, expected ({naux}, {naux})",
                v.dim()
            )));
        }
        let aux_q: Vec<f64> = aux_sh
            .iter()
            .map(|s| {
                (0..s.nfun)
                    .map(|i| v[(s.off + i, s.off + i)].abs().sqrt())
                    .fold(0.0_f64, f64::max)
            })
            .collect();
        let q_aux_max = aux_q.iter().copied().fold(0.0_f64, f64::max);
        let q = ferric_integrals::schwarz::schwarz(op, obs)?;
        let thresh = integral_thresh;

        let mut pairs = Vec::new();
        let mut prims = Vec::new();
        let mut h_len = 0usize;
        let (mut kept_blocks, mut total_blocks) = (0usize, 0usize);
        let nsh = obs_sh.len();
        for s1 in 0..nsh {
            for s2 in 0..=s1 {
                let qab = q[(s1, s2)];
                total_blocks += aux_sh.len();
                // PairScreen::from_shell_bounds's rule, verbatim.
                if thresh > 0.0 && qab * q_aux_max <= thresh {
                    continue;
                }
                let (sa, sb) = (&obs_sh[s1], &obs_sh[s2]);
                let prim0 = prims.len() as u32;
                let ab = [
                    sa.center[0] - sb.center[0],
                    sa.center[1] - sb.center[1],
                    sa.center[2] - sb.center[2],
                ];
                let r2 = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
                let nh = n_herm(sa.l + sb.l);
                for (ia, &ea) in sa.exps.iter().enumerate() {
                    for (ib, &eb) in sb.exps.iter().enumerate() {
                        let p = ea + eb;
                        let cab = sa.coefs[ia] * sb.coefs[ib];
                        if (cab * (-ea * eb / p * r2).exp()).abs() < PRIM_SCREEN {
                            continue;
                        }
                        let mut centre = [0.0; 3];
                        for x in 0..3 {
                            centre[x] = (ea * sa.center[x] + eb * sb.center[x]) / p;
                        }
                        prims.push(PrimPair {
                            ea,
                            eb,
                            p,
                            centre,
                            cab,
                            h_off: h_len,
                        });
                        h_len += nh;
                    }
                }
                if prims.len() as u32 == prim0 {
                    continue;
                }
                kept_blocks += aux_q
                    .iter()
                    .filter(|&&qk| !(thresh > 0.0 && qab * qk <= thresh))
                    .count();
                pairs.push(ShellPair {
                    a: s1 as u32,
                    b: s2 as u32,
                    q: qab,
                    prim0,
                    prim1: prims.len() as u32,
                });
            }
        }
        let mut g_off = Vec::with_capacity(aux_sh.len());
        let mut g_len = 0usize;
        for s in &aux_sh {
            g_off.push(g_len);
            g_len += s.exps.len() * n_herm(s.l);
        }
        let mut aux_atoms: Vec<(usize, usize)> = Vec::new();
        for (k, sk) in aux_sh.iter().enumerate() {
            match aux_atoms.last_mut() {
                Some(run) if aux_sh[run.0].center == sk.center => run.1 = k + 1,
                _ => aux_atoms.push((k, k + 1)),
            }
        }
        let _ = AUX_BLOCK;
        let aux_blocks = aux_atoms.clone();
        let c2s: Vec<Vec<f64>> = (0..=MAX_L_AUX.max(MAX_L_OBS)).map(c2s_matrix).collect();
        let c2s_abs: Vec<Vec<f64>> = c2s
            .iter()
            .map(|m| m.iter().map(|x| x.abs()).collect::<Vec<f64>>())
            .collect();
        let v_chol = v.factorizec(UPLO::Lower).map_err(|e| {
            FerricError::Lapack(format!(
                "Cholesky of the split-RI-J Coulomb metric failed (V is not numerically \
                 positive definite: the auxiliary basis is linearly dependent): {e}"
            ))
        })?;
        let mut me = SplitRiJ {
            obs: obs_sh,
            aux: aux_sh,
            nao,
            naux,
            pairs,
            prims,
            h_len,
            aux_q,
            g_off,
            g_len,
            aux_blocks,
            aux_atoms,
            far: true,
            c2s,
            c2s_abs,
            thresh,
            total_shell_pairs: nsh * (nsh + 1) / 2,
            kept_blocks,
            total_blocks,
            v_chol,
            _charge: ferric_core::memory::pool::Reservation::inert(POOL_LABEL),
        };
        CONSTRUCTED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        me._charge = ferric_core::memory::pool::try_reserve_global(POOL_LABEL, me.memory_bytes())
            .unwrap_or_else(|| ferric_core::memory::pool::Reservation::inert(POOL_LABEL));
        Ok(me)
    }

    /// Switch the far-field terms off (`false`: every term on the exact
    /// kernel) or on (the default). Test and benchmark hook.
    #[doc(hidden)]
    pub fn set_far_field(&mut self, on: bool) {
        self.far = on;
    }

    /// Orbital and auxiliary basis sizes.
    pub fn dims(&self) -> (usize, usize) {
        (self.nao, self.naux)
    }

    /// Kept and total orbital shell pairs.
    pub fn shell_pair_counts(&self) -> (usize, usize) {
        (self.pairs.len(), self.total_shell_pairs)
    }

    /// Kept and total (shell pair, aux shell) blocks.
    pub fn block_counts(&self) -> (usize, usize) {
        (self.kept_blocks, self.total_blocks)
    }

    /// Resident bytes outside a build (pair lists, Cholesky factor) plus the
    /// per-build transient peak (Hermite densities, aux Hermite vectors, J
    /// blocks). Used for memory planning.
    pub fn memory_bytes(&self) -> usize {
        let resident = self.prims.len() * std::mem::size_of::<PrimPair>()
            + self.pairs.len() * std::mem::size_of::<ShellPair>()
            + self.naux * self.naux * 8;
        let transient = (self.h_len + self.g_len + self.naux) * 8
            + self
                .pairs
                .iter()
                .map(|sp| self.obs[sp.a as usize].nfun * self.obs[sp.b as usize].nfun * 8)
                .sum::<usize>();
        resident + transient
    }

    /// `d_P = Σ_μν (P|μν) D_μν` over the kept blocks (symmetrised density,
    /// as DfJ's packed tier: `D_μν + D_νμ` off the diagonal shell pair).
    pub fn fit_rhs(&self, d: &Array2<f64>) -> Result<Array1<f64>, FerricError> {
        self.check_d(d)?;
        Ok(self.d_pass::<false, true>(d))
    }

    /// `c = V⁻¹ d` by the backward-stable Cholesky solve DfJ uses.
    pub fn solve_metric(&self, d_p: &Array1<f64>) -> Result<Array1<f64>, FerricError> {
        self.v_chol
            .solvec(d_p)
            .map_err(|e| FerricError::Lapack(format!("split-RI-J metric solve failed: {e}")))
    }

    /// `J_μν = Σ_P (P|μν) c_P` over the kept blocks; `j` is overwritten
    /// (dropped pairs are zero) and exactly symmetric.
    pub fn coulomb_from_coefficients(
        &self,
        c: &Array1<f64>,
        j: &mut Array2<f64>,
    ) -> Result<(), FerricError> {
        self.check_cj(c, j)?;
        self.j_pass::<false, true>(c, j);
        Ok(())
    }

    /// Running-error scale of [`Self::fit_rhs`]: the same pass with every
    /// factor replaced by its absolute value (`|D|`, `|E|`, `|R|` of the
    /// absolute recursions, `|C|`), i.e. `Σ |terms|` of the computed sum.
    /// The floating-point error of `fit_rhs` is at most
    /// `γ_k · fit_rhs_abs`, `k = rounding_depth()`. Test hook.
    #[doc(hidden)]
    pub fn fit_rhs_abs(&self, d: &Array2<f64>) -> Result<Array1<f64>, FerricError> {
        self.check_d(d)?;
        Ok(self.d_pass::<true, true>(d))
    }

    /// Running-error scale of [`Self::coulomb_from_coefficients`] (see
    /// [`Self::fit_rhs_abs`]). Test hook.
    #[doc(hidden)]
    pub fn coulomb_abs(&self, c: &Array1<f64>, j: &mut Array2<f64>) -> Result<(), FerricError> {
        self.check_cj(c, j)?;
        self.j_pass::<true, true>(c, j);
        Ok(())
    }

    /// [`Self::fit_rhs`] on the one-primitive-at-a-time kernel the lane
    /// kernel must reproduce bit for bit. Test hook.
    #[doc(hidden)]
    pub fn fit_rhs_scalar(&self, d: &Array2<f64>) -> Result<Array1<f64>, FerricError> {
        self.check_d(d)?;
        Ok(self.d_pass::<false, false>(d))
    }

    /// [`Self::coulomb_from_coefficients`] on the scalar kernel. Test hook.
    #[doc(hidden)]
    pub fn coulomb_scalar(&self, c: &Array1<f64>, j: &mut Array2<f64>) -> Result<(), FerricError> {
        self.check_cj(c, j)?;
        self.j_pass::<false, false>(c, j);
        Ok(())
    }

    /// An upper bound on the number of rounding steps in the longest
    /// dependency chain of either pass, the `k` of Higham's `γ_k = k u/(1-k u)`:
    /// E recursions (3 per order per direction, plus 2 for the product),
    /// the Hermite density's Cartesian sum and both cart→sph transforms, the
    /// Boys evaluation (≤ 10 Taylor + 14 recursion + exp), the R recursion
    /// (2 per order), the Hermite contraction, the sum over every bra
    /// primitive pair (d pass) or every aux primitive Hermite term (J pass),
    /// and the aux transforms.
    pub fn rounding_depth(&self) -> usize {
        let lab = self
            .pairs
            .iter()
            .map(|sp| self.obs[sp.a as usize].l + self.obs[sp.b as usize].l)
            .max()
            .unwrap_or(0);
        let lk = self.aux.iter().map(|s| s.l).max().unwrap_or(0);
        let nc_obs = self.obs.iter().map(|s| s.ncart).max().unwrap_or(1);
        let nf_obs = self.obs.iter().map(|s| s.nfun).max().unwrap_or(1);
        let nc_aux = self.aux.iter().map(|s| s.ncart).max().unwrap_or(1);
        let nf_aux = self.aux.iter().map(|s| s.nfun).max().unwrap_or(1);
        let naux_prims: usize = self.aux.iter().map(|s| s.exps.len()).sum();
        let e = 3 * lab + 4;
        let dens = 2 * nf_obs + nc_obs * nc_obs + 2;
        let boys = 32;
        let r = 2 * (lab + lk) + 2;
        let contract = n_herm(lab) + n_herm(lk) + 2;
        let outer = self.prims.len().max(naux_prims * n_herm(lk));
        let aux = nc_aux + nf_aux + n_herm(lk) + 4;
        e + dens + boys + r + contract + outer + aux
    }

    fn check_d(&self, d: &Array2<f64>) -> Result<(), FerricError> {
        if d.dim() != (self.nao, self.nao) {
            return Err(FerricError::General(format!(
                "split-RI-J: density is {:?}, expected ({n}, {n})",
                d.dim(),
                n = self.nao
            )));
        }
        Ok(())
    }

    fn check_cj(&self, c: &Array1<f64>, j: &Array2<f64>) -> Result<(), FerricError> {
        if c.len() != self.naux || j.dim() != (self.nao, self.nao) {
            return Err(FerricError::General(format!(
                "split-RI-J: c has {} entries, J is {:?}; expected {} and ({n}, {n})",
                c.len(),
                j.dim(),
                self.naux,
                n = self.nao
            )));
        }
        Ok(())
    }

    #[inline]
    fn c2s_for<const ABS: bool>(&self, l: usize) -> &[f64] {
        if ABS {
            &self.c2s_abs[l]
        } else {
            &self.c2s[l]
        }
    }

    #[inline]
    fn keep_block(&self, sp: &ShellPair, k: usize) -> bool {
        !(self.thresh > 0.0 && sp.q * self.aux_q[k] <= self.thresh)
    }

    fn e_tables<const ABS: bool>(&self, sp: &ShellPair, pp: &PrimPair, scr: &mut Scratch) {
        let (sa, sb) = (&self.obs[sp.a as usize], &self.obs[sp.b as usize]);
        let (la, lb) = (sa.l, sb.l);
        e_table::<ABS>(
            la,
            lb,
            pp.ea,
            pp.eb,
            sa.center[0] - sb.center[0],
            &mut scr.ex,
        );
        e_table::<ABS>(
            la,
            lb,
            pp.ea,
            pp.eb,
            sa.center[1] - sb.center[1],
            &mut scr.ey,
        );
        e_table::<ABS>(
            la,
            lb,
            pp.ea,
            pp.eb,
            sa.center[2] - sb.center[2],
            &mut scr.ez,
        );
    }

    // ----------------------------------------------------------------- d pass

    fn d_pass<const ABS: bool, const BATCH: bool>(&self, d: &Array2<f64>) -> Array1<f64> {
        // Hermite densities, one slice per primitive pair, in stored order,
        // each shell pair writing its own range of `hbuf` (no second copy).
        let mut hbuf = vec![0.0; self.h_len];
        let mut ranges: Vec<&mut [f64]> = Vec::with_capacity(self.pairs.len());
        let mut rest = hbuf.as_mut_slice();
        for sp in &self.pairs {
            let (mine, tail) = std::mem::take(&mut rest).split_at_mut(self.pair_h_len(sp));
            ranges.push(mine);
            rest = tail;
        }
        debug_assert!(rest.is_empty());
        self.pairs
            .par_iter()
            .zip(ranges)
            .for_each_init(Scratch::new, |scr, (sp, out)| {
                self.pair_hermite_density::<ABS>(sp, d, scr, out);
            });

        let blocks: Vec<Vec<f64>> = self
            .aux_blocks
            .par_iter()
            .map_init(Scratch::new, |scr, &(k0, k1)| {
                if !ABS && self.far {
                    self.d_block_far(k0, k1, &hbuf, scr)
                } else if BATCH {
                    self.d_block_lanes::<ABS>(k0, k1, &hbuf, scr)
                } else {
                    self.d_block::<ABS>(k0, k1, &hbuf, scr)
                }
            })
            .collect();
        let mut d_p = Array1::<f64>::zeros(self.naux);
        for (&(k0, _), vals) in self.aux_blocks.iter().zip(&blocks) {
            let p0 = self.aux[k0].off;
            for (i, &v) in vals.iter().enumerate() {
                d_p[p0 + i] = v;
            }
        }
        d_p
    }

    /// Length of shell pair `sp`'s Hermite densities in the d-pass buffer.
    fn pair_h_len(&self, sp: &ShellPair) -> usize {
        let lab = self.obs[sp.a as usize].l + self.obs[sp.b as usize].l;
        (sp.prim1 - sp.prim0) as usize * n_herm(lab)
    }

    /// `H_tuv` for every primitive pair of `sp`, concatenated, into `out`
    /// (zeroed, `pair_h_len(sp)` long).
    fn pair_hermite_density<const ABS: bool>(
        &self,
        sp: &ShellPair,
        d: &Array2<f64>,
        scr: &mut Scratch,
        out: &mut [f64],
    ) {
        let (sa, sb) = (&self.obs[sp.a as usize], &self.obs[sp.b as usize]);
        let diag = sp.a == sp.b;
        let (nfa, nfb, nca, ncb) = (sa.nfun, sb.nfun, sa.ncart, sb.ncart);
        // Spherical weights W (nfa × nfb).
        scr.w.clear();
        for i in 0..nfa {
            for jj in 0..nfb {
                let (mu, nu) = (sa.off + i, sb.off + jj);
                let v = if diag {
                    let x = d[(mu, nu)];
                    if ABS {
                        x.abs()
                    } else {
                        x
                    }
                } else if ABS {
                    d[(mu, nu)].abs() + d[(nu, mu)].abs()
                } else {
                    d[(mu, nu)] + d[(nu, mu)]
                };
                scr.w.push(v);
            }
        }
        // D^cart = C_A W C_Bᵀ: tmp (nca × nfb) then cart (nca × ncb).
        scr.tmp.clear();
        if sa.pure {
            let ca = self.c2s_for::<ABS>(sa.l);
            scr.tmp.resize(nca * nfb, 0.0);
            for m in 0..nca {
                for i in 0..nfa {
                    let c = ca[m * nfa + i];
                    if c != 0.0 {
                        for jj in 0..nfb {
                            scr.tmp[m * nfb + jj] += c * scr.w[i * nfb + jj];
                        }
                    }
                }
            }
        } else {
            scr.tmp.extend_from_slice(&scr.w);
        }
        scr.cart.clear();
        if sb.pure {
            let cb = self.c2s_for::<ABS>(sb.l);
            scr.cart.resize(nca * ncb, 0.0);
            for m in 0..nca {
                for n in 0..ncb {
                    let mut s = 0.0;
                    for jj in 0..nfb {
                        s += scr.tmp[m * nfb + jj] * cb[n * nfb + jj];
                    }
                    scr.cart[m * ncb + n] = s;
                }
            }
        } else {
            scr.cart.extend_from_slice(&scr.tmp);
        }

        let h = hermite();
        let nh = n_herm(sa.l + sb.l);
        let (p0, p1) = (sp.prim0 as usize, sp.prim1 as usize);
        debug_assert_eq!(out.len(), (p1 - p0) * nh);
        let sl = sa.l + sb.l + 1;
        let si = (sb.l + 1) * sl;
        for (k, pp) in self.prims[p0..p1].iter().enumerate() {
            self.e_tables::<ABS>(sp, pp, scr);
            let hk = &mut out[k * nh..(k + 1) * nh];
            let cab = if ABS { pp.cab.abs() } else { pp.cab };
            for (ia, ca) in sa.comps.iter().enumerate() {
                for (ib, cb) in sb.comps.iter().enumerate() {
                    let dv = scr.cart[ia * ncb + ib];
                    if dv == 0.0 {
                        continue;
                    }
                    let wv = cab * dv;
                    let (ix, iy, iz) = (ca[0] as usize, ca[1] as usize, ca[2] as usize);
                    let (jx, jy, jz) = (cb[0] as usize, cb[1] as usize, cb[2] as usize);
                    let rx = &scr.ex[ix * si + jx * sl..ix * si + jx * sl + ix + jx + 1];
                    let ry = &scr.ey[iy * si + jy * sl..iy * si + jy * sl + iy + jy + 1];
                    let rz = &scr.ez[iz * si + jz * sl..iz * si + jz * sl + iz + jz + 1];
                    for (t, &etx) in rx.iter().enumerate() {
                        let a1 = wv * etx;
                        if a1 == 0.0 {
                            continue;
                        }
                        for (u, &ety) in ry.iter().enumerate() {
                            let a2 = a1 * ety;
                            if a2 == 0.0 {
                                continue;
                            }
                            for (v, &etz) in rz.iter().enumerate() {
                                hk[h.index(t, u, v)] += a2 * etz;
                            }
                        }
                    }
                }
            }
        }
    }

    /// `d_P` for the aux functions of shells `k0..k1`.
    fn d_block<const ABS: bool>(
        &self,
        k0: usize,
        k1: usize,
        hbuf: &[f64],
        scr: &mut Scratch,
    ) -> Vec<f64> {
        // Per-aux-primitive Hermite accumulators W_{τνφ}.
        let mut acc_off = Vec::with_capacity(k1 - k0 + 1);
        let mut n_acc = 0usize;
        for s in &self.aux[k0..k1] {
            acc_off.push(n_acc);
            n_acc += s.exps.len() * n_herm(s.l);
        }
        let mut acc = vec![0.0_f64; n_acc];

        for sp in &self.pairs {
            let lab = self.obs[sp.a as usize].l + self.obs[sp.b as usize].l;
            let nj = n_herm(lab);
            for pp in &self.prims[sp.prim0 as usize..sp.prim1 as usize] {
                let hv = &hbuf[pp.h_off..pp.h_off + nj];
                if hv.iter().all(|&x| x == 0.0) {
                    continue;
                }
                for (kk, k) in (k0..k1).enumerate() {
                    if !self.keep_block(sp, k) {
                        continue;
                    }
                    let sk = &self.aux[k];
                    let (lk, nk) = (sk.l, n_herm(sk.l));
                    let ltot = lab + lk;
                    let tab = sum_table(lab, lk);
                    let x = [
                        pp.centre[0] - sk.center[0],
                        pp.centre[1] - sk.center[1],
                        pp.centre[2] - sk.center[2],
                    ];
                    for (ic, &qc) in sk.exps.iter().enumerate() {
                        let pq = pp.p + qc;
                        let alpha = pp.p * qc / pq;
                        let pref = TWO_PI_52 / (pp.p * qc * pq.sqrt());
                        r_eval::<ABS>(ltot, alpha, x, &mut scr.fb, &mut scr.r);
                        let w = &mut acc[acc_off[kk] + ic * nk..acc_off[kk] + (ic + 1) * nk];
                        for (kh, wk) in w.iter_mut().enumerate() {
                            let row = &tab[kh * nj..(kh + 1) * nj];
                            let mut s = 0.0;
                            for (jh, &hj) in hv.iter().enumerate() {
                                s += hj * scr.r[row[jh] as usize];
                            }
                            *wk += pref * s;
                        }
                    }
                }
            }
        }

        self.d_block_finish::<ABS>(k0, k1, &acc, &acc_off)
    }

    /// Aux Hermite -> Cartesian -> spherical for [`Self::d_block`]'s
    /// accumulators, once per aux primitive.
    fn d_block_finish<const ABS: bool>(
        &self,
        k0: usize,
        k1: usize,
        acc: &[f64],
        acc_off: &[usize],
    ) -> Vec<f64> {
        let h = hermite();
        let nout: usize = self.aux[k0..k1].iter().map(|s| s.nfun).sum();
        let mut out = Vec::with_capacity(nout);
        let mut ek = vec![0.0; (MAX_L_AUX + 1) * (MAX_L_AUX + 1)];
        for (kk, sk) in self.aux[k0..k1].iter().enumerate() {
            let (lk, nk) = (sk.l, n_herm(sk.l));
            let mut dcart = vec![0.0; sk.ncart];
            for (ic, &qc) in sk.exps.iter().enumerate() {
                e_table::<ABS>(lk, 0, qc, 0.0, 0.0, &mut ek);
                let w = &acc[acc_off[kk] + ic * nk..acc_off[kk] + (ic + 1) * nk];
                let cc = if ABS {
                    sk.coefs[ic].abs()
                } else {
                    sk.coefs[ic]
                };
                for (ci, c) in sk.comps.iter().enumerate() {
                    let (cx, cy, cz) = (c[0] as usize, c[1] as usize, c[2] as usize);
                    let mut s = 0.0;
                    for t in 0..=cx {
                        let e1 = ek[cx * (lk + 1) + t];
                        if e1 == 0.0 {
                            continue;
                        }
                        for u in 0..=cy {
                            let e2 = e1 * ek[cy * (lk + 1) + u];
                            if e2 == 0.0 {
                                continue;
                            }
                            for v in 0..=cz {
                                let e3 = e2 * ek[cz * (lk + 1) + v];
                                let sign = if !ABS && (t + u + v) % 2 == 1 {
                                    -1.0
                                } else {
                                    1.0
                                };
                                s += sign * e3 * w[h.index(t, u, v)];
                            }
                        }
                    }
                    dcart[ci] += cc * s;
                }
            }
            if sk.pure {
                let cmat = self.c2s_for::<ABS>(lk);
                for m in 0..sk.nfun {
                    let mut s = 0.0;
                    for (ci, &dc) in dcart.iter().enumerate() {
                        s += cmat[ci * sk.nfun + m] * dc;
                    }
                    out.push(s);
                }
            } else {
                out.extend_from_slice(&dcart);
            }
        }
        out
    }

    /// [`Self::d_block`] for one aux atom `k0..k1`, with every term whose
    /// `T = α|P−Q|²` is at least [`far_t_table`]'s bound for its order taken
    /// in the large-T form: one derivative tensor and one contraction per
    /// (primitive pair, atom) give the field `M`, and each far aux primitive
    /// adds `2π^{5/2} q^{-3/2} M`. Near terms use the exact kernel.
    fn d_block_far(&self, k0: usize, k1: usize, hbuf: &[f64], scr: &mut Scratch) -> Vec<f64> {
        let mut acc_off = Vec::with_capacity(k1 - k0 + 1);
        let mut n_acc = 0usize;
        for s in &self.aux[k0..k1] {
            acc_off.push(n_acc);
            n_acc += s.exps.len() * n_herm(s.l);
        }
        let mut acc = vec![0.0_f64; n_acc];
        let qc = self.aux[k0].center;
        let tf = far_t_table();
        let mut far_terms: Vec<(usize, usize)> = Vec::new();
        let mut field = vec![0.0; n_herm(MAX_L_AUX)];
        for sp in &self.pairs {
            let lab = self.obs[sp.a as usize].l + self.obs[sp.b as usize].l;
            let nj = n_herm(lab);
            for pp in &self.prims[sp.prim0 as usize..sp.prim1 as usize] {
                let hv = &hbuf[pp.h_off..pp.h_off + nj];
                if hv.iter().all(|&x| x == 0.0) {
                    continue;
                }
                let x = [
                    pp.centre[0] - qc[0],
                    pp.centre[1] - qc[1],
                    pp.centre[2] - qc[2],
                ];
                let r2 = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
                far_terms.clear();
                let mut l_far = 0;
                for (kk, k) in (k0..k1).enumerate() {
                    if !self.keep_block(sp, k) {
                        continue;
                    }
                    let sk = &self.aux[k];
                    let (lk, nk) = (sk.l, n_herm(sk.l));
                    let ltot = lab + lk;
                    let tab = sum_table(lab, lk);
                    for (ic, &q) in sk.exps.iter().enumerate() {
                        let pq = pp.p + q;
                        // α r² >= T_far without the division.
                        if pp.p * q * r2 >= tf[ltot] * pq {
                            far_terms.push((kk, ic));
                            l_far = l_far.max(lk);
                            continue;
                        }
                        let alpha = pp.p * q / pq;
                        let pref = TWO_PI_52 / (pp.p * q * pq.sqrt());
                        r_eval::<false>(ltot, alpha, x, &mut scr.fb, &mut scr.r);
                        let w = &mut acc[acc_off[kk] + ic * nk..acc_off[kk] + (ic + 1) * nk];
                        for (kh, wk) in w.iter_mut().enumerate() {
                            let row = &tab[kh * nj..(kh + 1) * nj];
                            let mut s = 0.0;
                            for (jh, &hj) in hv.iter().enumerate() {
                                s += hj * scr.r[row[jh] as usize];
                            }
                            *wk += pref * s;
                        }
                    }
                }
                if far_terms.is_empty() {
                    continue;
                }
                r_eval_far(lab + l_far, x, &mut scr.fb, &mut scr.r);
                let tab = sum_table(lab, l_far);
                let pm = 1.0 / (pp.p * pp.p.sqrt());
                for (kh, m) in field[..n_herm(l_far)].iter_mut().enumerate() {
                    let row = &tab[kh * nj..(kh + 1) * nj];
                    let mut s = 0.0;
                    for (jh, &hj) in hv.iter().enumerate() {
                        s += hj * scr.r[row[jh] as usize];
                    }
                    *m = pm * s;
                }
                for &(kk, ic) in &far_terms {
                    let sk = &self.aux[k0 + kk];
                    let nk = n_herm(sk.l);
                    let q = sk.exps[ic];
                    let f = TWO_PI_52 / (q * q.sqrt());
                    let w = &mut acc[acc_off[kk] + ic * nk..acc_off[kk] + (ic + 1) * nk];
                    for (wk, &m) in w.iter_mut().zip(&field) {
                        *wk += f * m;
                    }
                }
            }
        }
        self.d_block_finish::<false>(k0, k1, &acc, &acc_off)
    }

    /// [`Self::j_pair`] with [`Self::d_block_far`]'s far/near split: per
    /// (primitive pair, aux atom), the far aux primitives' Hermite
    /// coefficients are summed with weights `q^{-3/2}` and contracted once
    /// against the derivative tensor.
    fn j_pair_far(&self, sp: &ShellPair, g: &[f64], scr: &mut Scratch) -> Vec<f64> {
        let (sa, sb) = (&self.obs[sp.a as usize], &self.obs[sp.b as usize]);
        let lab = sa.l + sb.l;
        let nj = n_herm(lab);
        let tf = far_t_table();
        let mut jcart = vec![0.0_f64; sa.ncart * sb.ncart];
        let mut wpair = vec![0.0_f64; nj];
        let mut gsum = vec![0.0_f64; n_herm(MAX_L_AUX)];
        for pp in &self.prims[sp.prim0 as usize..sp.prim1 as usize] {
            wpair.iter_mut().for_each(|x| *x = 0.0);
            for &(k0, k1) in &self.aux_atoms {
                let qc = self.aux[k0].center;
                let x = [
                    pp.centre[0] - qc[0],
                    pp.centre[1] - qc[1],
                    pp.centre[2] - qc[2],
                ];
                let r2 = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
                let mut l_far = None;
                for k in k0..k1 {
                    if !self.keep_block(sp, k) {
                        continue;
                    }
                    let sk = &self.aux[k];
                    let (lk, nk) = (sk.l, n_herm(sk.l));
                    let ltot = lab + lk;
                    let tab = sum_table(lab, lk);
                    for (ic, &q) in sk.exps.iter().enumerate() {
                        let gk = &g[self.g_off[k] + ic * nk..self.g_off[k] + (ic + 1) * nk];
                        if gk.iter().all(|&v| v == 0.0) {
                            continue;
                        }
                        let pq = pp.p + q;
                        if pp.p * q * r2 >= tf[ltot] * pq {
                            let f = 1.0 / (q * q.sqrt());
                            for (gs, &gv) in gsum.iter_mut().zip(gk) {
                                *gs += f * gv;
                            }
                            l_far = Some(l_far.map_or(lk, |l: usize| l.max(lk)));
                            continue;
                        }
                        let alpha = pp.p * q / pq;
                        let pref = TWO_PI_52 / (pp.p * q * pq.sqrt());
                        r_eval::<false>(ltot, alpha, x, &mut scr.fb, &mut scr.r);
                        for (kh, &gv) in gk.iter().enumerate() {
                            if gv == 0.0 {
                                continue;
                            }
                            let gp = pref * gv;
                            let row = &tab[kh * nj..(kh + 1) * nj];
                            for (jh, wv) in wpair.iter_mut().enumerate() {
                                *wv += gp * scr.r[row[jh] as usize];
                            }
                        }
                    }
                }
                let Some(l_far) = l_far else {
                    continue;
                };
                r_eval_far(lab + l_far, x, &mut scr.fb, &mut scr.r);
                let tab = sum_table(lab, l_far);
                let pm = TWO_PI_52 / (pp.p * pp.p.sqrt());
                for (kh, gs) in gsum[..n_herm(l_far)].iter_mut().enumerate() {
                    if *gs != 0.0 {
                        let gp = pm * *gs;
                        let row = &tab[kh * nj..(kh + 1) * nj];
                        for (jh, wv) in wpair.iter_mut().enumerate() {
                            *wv += gp * scr.r[row[jh] as usize];
                        }
                    }
                    *gs = 0.0;
                }
            }
            self.add_cartesian_pair::<false>(sp, pp, &wpair, &mut jcart, scr);
        }
        self.j_spherical::<false>(sp, jcart)
    }

    /// [`Self::d_block`] with `LANES` bra primitive pairs of one shell pair
    /// per R evaluation. Every `W` element still receives its terms in
    /// primitive-pair order, and each term is computed exactly as
    /// [`Self::d_block`] does, so the result is bit-identical.
    fn d_block_lanes<const ABS: bool>(
        &self,
        k0: usize,
        k1: usize,
        hbuf: &[f64],
        scr: &mut Scratch,
    ) -> Vec<f64> {
        let mut acc_off = Vec::with_capacity(k1 - k0 + 1);
        let mut n_acc = 0usize;
        for s in &self.aux[k0..k1] {
            acc_off.push(n_acc);
            n_acc += s.exps.len() * n_herm(s.l);
        }
        let mut acc = vec![0.0_f64; n_acc];
        let mut live: Vec<&PrimPair> = Vec::new();
        for sp in &self.pairs {
            let lab = self.obs[sp.a as usize].l + self.obs[sp.b as usize].l;
            let nj = n_herm(lab);
            live.clear();
            live.extend(
                self.prims[sp.prim0 as usize..sp.prim1 as usize]
                    .iter()
                    .filter(|pp| hbuf[pp.h_off..pp.h_off + nj].iter().any(|&x| x != 0.0)),
            );
            for chunk in live.chunks(LANES) {
                let n = chunk.len();
                scr.hl.clear();
                scr.hl.resize(nj, [0.0; LANES]);
                for (i, pp) in chunk.iter().enumerate() {
                    for (jh, &hv) in hbuf[pp.h_off..pp.h_off + nj].iter().enumerate() {
                        scr.hl[jh][i] = hv;
                    }
                }
                for (kk, k) in (k0..k1).enumerate() {
                    if !self.keep_block(sp, k) {
                        continue;
                    }
                    let sk = &self.aux[k];
                    let (lk, nk) = (sk.l, n_herm(sk.l));
                    let tab = sum_table(lab, lk);
                    let x = lane_offsets(chunk, sk.center);
                    for (ic, &qc) in sk.exps.iter().enumerate() {
                        let (alpha, pref) = lane_alpha_pref(chunk, qc);
                        r_eval_lanes::<ABS>(lab + lk, &alpha, &x, &mut scr.fbl, &mut scr.rl);
                        let w = &mut acc[acc_off[kk] + ic * nk..acc_off[kk] + (ic + 1) * nk];
                        for (kh, wk) in w.iter_mut().enumerate() {
                            let row = &tab[kh * nj..(kh + 1) * nj];
                            let mut sv = [0.0; LANES];
                            for (hj, &ri) in scr.hl.iter().zip(row) {
                                let rr = &scr.rl[ri as usize];
                                for i in 0..LANES {
                                    sv[i] += hj[i] * rr[i];
                                }
                            }
                            for i in 0..n {
                                *wk += pref[i] * sv[i];
                            }
                        }
                    }
                }
            }
        }
        self.d_block_finish::<ABS>(k0, k1, &acc, &acc_off)
    }

    // ----------------------------------------------------------------- J pass

    fn j_pass<const ABS: bool, const BATCH: bool>(&self, c: &Array1<f64>, j: &mut Array2<f64>) {
        let g = self.aux_hermite_coefficients::<ABS>(c);
        let blocks: Vec<Vec<f64>> = self
            .pairs
            .par_iter()
            .map_init(Scratch::new, |scr, sp| {
                if !ABS && self.far {
                    self.j_pair_far(sp, &g, scr)
                } else if BATCH {
                    self.j_pair_lanes::<ABS>(sp, &g, scr)
                } else {
                    self.j_pair::<ABS>(sp, &g, scr)
                }
            })
            .collect();
        j.fill(0.0);
        for (sp, blk) in self.pairs.iter().zip(&blocks) {
            let (sa, sb) = (&self.obs[sp.a as usize], &self.obs[sp.b as usize]);
            let nfb = sb.nfun;
            for i in 0..sa.nfun {
                for jj in 0..nfb {
                    let (mu, nu) = (sa.off + i, sb.off + jj);
                    if sp.a == sp.b && jj > i {
                        continue; // diagonal pair: the lower triangle, mirrored
                    }
                    let v = blk[i * nfb + jj];
                    j[(mu, nu)] = v;
                    j[(nu, mu)] = v;
                }
            }
        }
    }

    /// `G_{τνφ} = c_k (−1)^{τ+ν+φ} Σ_c C^cart_c E^K_{c,τνφ}` per aux primitive.
    fn aux_hermite_coefficients<const ABS: bool>(&self, c: &Array1<f64>) -> Vec<f64> {
        let h = hermite();
        let mut g = vec![0.0; self.g_len];
        let mut ek = vec![0.0; (MAX_L_AUX + 1) * (MAX_L_AUX + 1)];
        for (k, sk) in self.aux.iter().enumerate() {
            let (lk, nk) = (sk.l, n_herm(sk.l));
            let csph: Vec<f64> = (0..sk.nfun)
                .map(|m| {
                    let x = c[sk.off + m];
                    if ABS {
                        x.abs()
                    } else {
                        x
                    }
                })
                .collect();
            let ccart: Vec<f64> = if sk.pure {
                let cmat = self.c2s_for::<ABS>(lk);
                (0..sk.ncart)
                    .map(|ci| {
                        (0..sk.nfun)
                            .map(|m| cmat[ci * sk.nfun + m] * csph[m])
                            .sum::<f64>()
                    })
                    .collect()
            } else {
                csph
            };
            for (ic, &qc) in sk.exps.iter().enumerate() {
                e_table::<ABS>(lk, 0, qc, 0.0, 0.0, &mut ek);
                let gk = &mut g[self.g_off[k] + ic * nk..self.g_off[k] + (ic + 1) * nk];
                for (ci, cmp) in sk.comps.iter().enumerate() {
                    let cv = ccart[ci];
                    if cv == 0.0 {
                        continue;
                    }
                    let (cx, cy, cz) = (cmp[0] as usize, cmp[1] as usize, cmp[2] as usize);
                    for t in 0..=cx {
                        let e1 = cv * ek[cx * (lk + 1) + t];
                        if e1 == 0.0 {
                            continue;
                        }
                        for u in 0..=cy {
                            let e2 = e1 * ek[cy * (lk + 1) + u];
                            if e2 == 0.0 {
                                continue;
                            }
                            for v in 0..=cz {
                                gk[h.index(t, u, v)] += e2 * ek[cz * (lk + 1) + v];
                            }
                        }
                    }
                }
                let cc = if ABS {
                    sk.coefs[ic].abs()
                } else {
                    sk.coefs[ic]
                };
                for (kh, gv) in gk.iter_mut().enumerate() {
                    let [t, u, v] = h.list[kh];
                    let odd = (t + u + v) % 2 == 1;
                    *gv *= if !ABS && odd { -cc } else { cc };
                }
            }
        }
        g
    }

    /// The `(nfa × nfb)` spherical J block of shell pair `sp`.
    fn j_pair<const ABS: bool>(&self, sp: &ShellPair, g: &[f64], scr: &mut Scratch) -> Vec<f64> {
        let (sa, sb) = (&self.obs[sp.a as usize], &self.obs[sp.b as usize]);
        let lab = sa.l + sb.l;
        let nj = n_herm(lab);
        let mut jcart = vec![0.0_f64; sa.ncart * sb.ncart];
        let mut wpair = vec![0.0_f64; nj];
        for pp in &self.prims[sp.prim0 as usize..sp.prim1 as usize] {
            wpair.iter_mut().for_each(|x| *x = 0.0);
            for (k, sk) in self.aux.iter().enumerate() {
                if !self.keep_block(sp, k) {
                    continue;
                }
                let (lk, nk) = (sk.l, n_herm(sk.l));
                let ltot = lab + lk;
                let tab = sum_table(lab, lk);
                let x = [
                    pp.centre[0] - sk.center[0],
                    pp.centre[1] - sk.center[1],
                    pp.centre[2] - sk.center[2],
                ];
                for (ic, &qc) in sk.exps.iter().enumerate() {
                    let gk = &g[self.g_off[k] + ic * nk..self.g_off[k] + (ic + 1) * nk];
                    if gk.iter().all(|&v| v == 0.0) {
                        continue;
                    }
                    let pq = pp.p + qc;
                    let alpha = pp.p * qc / pq;
                    let pref = TWO_PI_52 / (pp.p * qc * pq.sqrt());
                    r_eval::<ABS>(ltot, alpha, x, &mut scr.fb, &mut scr.r);
                    for (kh, &gv) in gk.iter().enumerate() {
                        if gv == 0.0 {
                            continue;
                        }
                        let gp = pref * gv;
                        let row = &tab[kh * nj..(kh + 1) * nj];
                        for (jh, wv) in wpair.iter_mut().enumerate() {
                            *wv += gp * scr.r[row[jh] as usize];
                        }
                    }
                }
            }
            self.add_cartesian_pair::<ABS>(sp, pp, &wpair, &mut jcart, scr);
        }
        self.j_spherical::<ABS>(sp, jcart)
    }

    /// [`Self::j_pair`] with `LANES` bra primitive pairs per R evaluation.
    /// Each lane's Hermite potential receives its terms in the scalar order,
    /// and the lanes are transformed into `jcart` in primitive-pair order, so
    /// the block is bit-identical.
    fn j_pair_lanes<const ABS: bool>(
        &self,
        sp: &ShellPair,
        g: &[f64],
        scr: &mut Scratch,
    ) -> Vec<f64> {
        let (sa, sb) = (&self.obs[sp.a as usize], &self.obs[sp.b as usize]);
        let lab = sa.l + sb.l;
        let nj = n_herm(lab);
        let mut jcart = vec![0.0_f64; sa.ncart * sb.ncart];
        let mut wpair = vec![0.0_f64; nj];
        let prims: Vec<&PrimPair> = self.prims[sp.prim0 as usize..sp.prim1 as usize]
            .iter()
            .collect();
        for chunk in prims.chunks(LANES) {
            scr.wl.clear();
            scr.wl.resize(nj, [0.0; LANES]);
            for (k, sk) in self.aux.iter().enumerate() {
                if !self.keep_block(sp, k) {
                    continue;
                }
                let (lk, nk) = (sk.l, n_herm(sk.l));
                let tab = sum_table(lab, lk);
                let x = lane_offsets(chunk, sk.center);
                for (ic, &qc) in sk.exps.iter().enumerate() {
                    let gk = &g[self.g_off[k] + ic * nk..self.g_off[k] + (ic + 1) * nk];
                    if gk.iter().all(|&v| v == 0.0) {
                        continue;
                    }
                    let (alpha, pref) = lane_alpha_pref(chunk, qc);
                    r_eval_lanes::<ABS>(lab + lk, &alpha, &x, &mut scr.fbl, &mut scr.rl);
                    for (kh, &gv) in gk.iter().enumerate() {
                        if gv == 0.0 {
                            continue;
                        }
                        let gp = pref.map(|p| p * gv);
                        let row = &tab[kh * nj..(kh + 1) * nj];
                        for (wv, &ri) in scr.wl.iter_mut().zip(row) {
                            let rr = &scr.rl[ri as usize];
                            for i in 0..LANES {
                                wv[i] += gp[i] * rr[i];
                            }
                        }
                    }
                }
            }
            for (i, pp) in chunk.iter().enumerate() {
                for (w, wl) in wpair.iter_mut().zip(&scr.wl) {
                    *w = wl[i];
                }
                self.add_cartesian_pair::<ABS>(sp, pp, &wpair, &mut jcart, scr);
            }
        }
        self.j_spherical::<ABS>(sp, jcart)
    }

    /// `jcart += c_a c_b Σ_tuv E^{μν}_{tuv} W_tuv` for one primitive pair.
    fn add_cartesian_pair<const ABS: bool>(
        &self,
        sp: &ShellPair,
        pp: &PrimPair,
        wpair: &[f64],
        jcart: &mut [f64],
        scr: &mut Scratch,
    ) {
        let (sa, sb) = (&self.obs[sp.a as usize], &self.obs[sp.b as usize]);
        let ncb = sb.ncart;
        let h = hermite();
        let sl = sa.l + sb.l + 1;
        let si = (sb.l + 1) * sl;
        self.e_tables::<ABS>(sp, pp, scr);
        let cab = if ABS { pp.cab.abs() } else { pp.cab };
        for (ia, ca) in sa.comps.iter().enumerate() {
            for (ib, cb) in sb.comps.iter().enumerate() {
                let (ix, iy, iz) = (ca[0] as usize, ca[1] as usize, ca[2] as usize);
                let (jx, jy, jz) = (cb[0] as usize, cb[1] as usize, cb[2] as usize);
                let rx = &scr.ex[ix * si + jx * sl..ix * si + jx * sl + ix + jx + 1];
                let ry = &scr.ey[iy * si + jy * sl..iy * si + jy * sl + iy + jy + 1];
                let rz = &scr.ez[iz * si + jz * sl..iz * si + jz * sl + iz + jz + 1];
                let mut s = 0.0;
                for (t, &etx) in rx.iter().enumerate() {
                    if etx == 0.0 {
                        continue;
                    }
                    for (u, &ety) in ry.iter().enumerate() {
                        let a2 = etx * ety;
                        if a2 == 0.0 {
                            continue;
                        }
                        for (v, &etz) in rz.iter().enumerate() {
                            s += a2 * etz * wpair[h.index(t, u, v)];
                        }
                    }
                }
                jcart[ia * ncb + ib] += cab * s;
            }
        }
    }

    /// The spherical block `J_sph = C_Aᵀ J_cart C_B` of shell pair `sp`.
    fn j_spherical<const ABS: bool>(&self, sp: &ShellPair, jcart: Vec<f64>) -> Vec<f64> {
        let (sa, sb) = (&self.obs[sp.a as usize], &self.obs[sp.b as usize]);
        let (nca, ncb, nfa, nfb) = (sa.ncart, sb.ncart, sa.nfun, sb.nfun);
        let right: Vec<f64> = if sb.pure {
            let cb = self.c2s_for::<ABS>(sb.l);
            let mut t = vec![0.0; nca * nfb];
            for m in 0..nca {
                for jj in 0..nfb {
                    let mut s = 0.0;
                    for n in 0..ncb {
                        s += jcart[m * ncb + n] * cb[n * nfb + jj];
                    }
                    t[m * nfb + jj] = s;
                }
            }
            t
        } else {
            jcart
        };
        if sa.pure {
            let ca = self.c2s_for::<ABS>(sa.l);
            let mut out = vec![0.0; nfa * nfb];
            for i in 0..nfa {
                for jj in 0..nfb {
                    let mut s = 0.0;
                    for m in 0..nca {
                        s += ca[m * nfa + i] * right[m * nfb + jj];
                    }
                    out[i * nfb + jj] = s;
                }
            }
            out
        } else {
            right
        }
    }
}

impl JBuilder for SplitRiJ {
    fn build(&mut self, d: &Array2<f64>, j: &mut Array2<f64>) -> Result<usize, FerricError> {
        use ferric_core::timing::Stage;
        let t = Stage::start("split-rij: d pass");
        let d_p = self.fit_rhs(d)?;
        t.end();
        let c_p = self.solve_metric(&d_p)?;
        let t = Stage::start("split-rij: J pass");
        self.coulomb_from_coefficients(&c_p, j)?;
        t.end();
        Ok(0)
    }

    fn reset(&mut self) {}
}

/// The RI-J builder an SCF holds: the stored tensor ([`DfJ`](crate::df_j::DfJ))
/// or the direct Split-RI-J. Both are [`JBuilder`]s, so call sites that only
/// `build` (rhf/uhf/rohf) compile unchanged when `build_df_jk` returns
/// `Option<RiJ>` instead of `Option<DfJ>`.
// One per SCF, never in a collection, so the Stored/Split size gap costs nothing
// and boxing the common (stored) case would only add an indirection.
#[allow(clippy::large_enum_variant)]
pub enum RiJ<'a> {
    Stored(crate::df_j::DfJ<'a>),
    Split(Box<SplitRiJ>),
}

impl<'a> RiJ<'a> {
    /// The stored builder, for DfJ-only hooks (e.g. `reserve_device_for_k`).
    pub fn as_stored_mut(&mut self) -> Option<&mut crate::df_j::DfJ<'a>> {
        match self {
            RiJ::Stored(s) => Some(s),
            RiJ::Split(_) => None,
        }
    }
}

impl std::fmt::Debug for RiJ<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RiJ::Stored(s) => s.fmt(f),
            RiJ::Split(s) => s.fmt(f),
        }
    }
}

impl JBuilder for RiJ<'_> {
    fn build(&mut self, d: &Array2<f64>, j: &mut Array2<f64>) -> Result<usize, FerricError> {
        match self {
            RiJ::Stored(s) => s.build(d, j),
            RiJ::Split(s) => s.build(d, j),
        }
    }

    fn reset(&mut self) {
        match self {
            RiJ::Stored(s) => s.reset(),
            RiJ::Split(s) => s.reset(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The generator reproduces md3c1e's libint2-validated tables for l = 2..4.
    #[test]
    fn solid_harmonic_generator_matches_the_libint2_tables() {
        for l in 2..=md3c1e::MAX_L {
            let gen = solid_harmonic_c2s(l);
            let tab = md3c1e::ferric_cart2sph(l);
            assert_eq!(gen.len(), tab.len());
            for (i, (a, b)) in gen.iter().zip(&tab).enumerate() {
                assert!(
                    (a - b).abs() <= 1e-14 * b.abs().max(1.0),
                    "l={l} entry {i}: {a} vs {b}"
                );
            }
        }
    }

    /// Pure functions are orthonormal under [`cart_metric`]. The diagonal is
    /// unit by construction (the generator normalises), so the force of this
    /// test is the OFF-diagonal: distinct `m` are orthogonal only if each
    /// column has the right shape. That is the only check l = 5, 6 get, since
    /// md3c1e has no table there. md3c1e's own l ≤ 4 tables are included,
    /// which validates the metric convention itself. Bar 1e-12: exact zero in
    /// exact arithmetic, ≤ nc² ≈ 800 terms of O(1) → ~1e-13 rounding; a wrong
    /// shape is O(0.1).
    #[test]
    fn generated_pure_functions_are_orthonormal() {
        let mut cases: Vec<(usize, Vec<f64>)> = (2..=MAX_L_AUX)
            .map(|l| (l, solid_harmonic_c2s(l)))
            .collect();
        cases.extend((2..=md3c1e::MAX_L).map(|l| (l, md3c1e::ferric_cart2sph(l))));
        for (l, c) in cases {
            let comps = md3c1e::cart_components(l);
            let (nc, ns) = (comps.len(), 2 * l + 1);
            for m1 in 0..ns {
                for m2 in 0..ns {
                    let mut s = 0.0;
                    for i in 0..nc {
                        for k in 0..nc {
                            s += c[i * ns + m1]
                                * cart_metric(&comps[i], &comps[k], l)
                                * c[k * ns + m2];
                        }
                    }
                    let want = if m1 == m2 { 1.0 } else { 0.0 };
                    assert!((s - want).abs() < 1e-12, "l={l} m1={m1} m2={m2}: {s}");
                }
            }
        }
    }

    /// The table path against the direct series/asymptotic reference. Bar:
    /// each side is ≤ ~20 roundings deep (10-term Horner or the series, then
    /// ≤ 14 downward steps, each with a contraction factor ≤ 1), so their
    /// difference is ≤ 2·20·u ≈ 4.4e-15 relative; 2e-14 leaves 4.5x headroom.
    /// A wrong node, a wrong Taylor sign or an off-by-one order is O(1e-3+).
    #[test]
    fn boys_table_matches_direct_evaluation() {
        let mut a = [0.0; MAX_LTOT + 1];
        let mut b = [0.0; MAX_LTOT + 1];
        for i in 0..2000 {
            let t = i as f64 * 0.0371;
            boys(MAX_LTOT, t, &mut a);
            md3c1e::boys_direct(MAX_LTOT, t, &mut b);
            for n in 0..=MAX_LTOT {
                assert!(
                    (a[n] - b[n]).abs() <= 2e-14 * b[n],
                    "n={n} T={t}: {} vs {}",
                    a[n],
                    b[n]
                );
            }
        }
    }

    /// `(s s | s)` primitive in closed form: `2π^{5/2}/(pq√(p+q)) e^{-μR²} F_0(αR²)`.
    #[test]
    fn r0_and_e0_give_the_ssss_closed_form() {
        let (a, b, q) = (1.3, 0.4, 0.7);
        let (ax, bx) = (0.0, 1.1);
        let mut e = [0.0; 1];
        e_table::<false>(0, 0, a, b, ax - bx, &mut e);
        let p = a + b;
        let px = (a * ax + b * bx) / p;
        let qx = -0.5;
        let alpha = p * q / (p + q);
        let mut fb = [0.0; MAX_LTOT + 1];
        let mut r = vec![0.0; r_programs()[0].nvec];
        r_eval::<false>(0, alpha, [px - qx, 0.0, 0.0], &mut fb, &mut r);
        let got = TWO_PI_52 / (p * q * (p + q).sqrt()) * e[0] * r[0];
        let mut f0 = [0.0; 1];
        md3c1e::boys_direct(0, alpha * (px - qx) * (px - qx), &mut f0);
        let want = TWO_PI_52 / (p * q * (p + q).sqrt()) * (-a * b / p * 1.21).exp() * f0[0];
        // Table vs direct Boys (bar above) dominates; the rest is a few ulp.
        assert!((got - want).abs() <= 2e-14 * want, "{got} vs {want}");
    }
}
