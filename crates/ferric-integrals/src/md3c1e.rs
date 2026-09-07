//! From-scratch McMurchie-Davidson three-center-one-electron kernel for
//! seminumerical exchange (COSX / sn-LinK).
//!
//! Builds, for a **batch** of grid points `r_g`, the per-point AO blocks
//!
//! ```text
//!     A^g_{mu,nu} = \int chi_mu(r) chi_nu(r) / |r - r_g| dr
//! ```
//!
//! with the grid axis as the innermost, contiguous, vectorized dimension. This
//! is the same quantity [`crate::cosx_a`] obtains from libint2's nuclear-attraction
//! engine one point at a time; `cosx_a` is this module's exactness oracle
//! (`tests/md3c1e_matches_cosx_a.rs`) and its functions are untouched.
//!
//! # Sign convention (load-bearing)
//!
//! This module produces the **repulsive** kernel `+1/|r - r_g|` directly —
//! the same sign `cosx_a::a_matrix_at_point` returns (and PySCF's
//! `int1e_grids`), and the *opposite* of raw libint2, which `cosx_a` negates.
//! A caller replacing the libint2 path must therefore drop that negation; a
//! caller replacing `cosx_a` changes nothing.
//!
//! # Method and why it fits this shape
//!
//! McMurchie-Davidson (Helgaker/Jørgensen/Olsen ch. 9): the primitive pair
//! density is expanded in Hermite Gaussians on the product centre `P`,
//! `chi_a chi_b = Σ_tuv E_t^{ij} E_u^{kl} E_v^{mn} Λ_tuv`, and the Coulomb
//! kernel is applied analytically, `A = (2π/p) Σ_tuv E_t E_u E_v R^0_tuv(p, P-r_g)`.
//! The E-coefficients depend only on `(l_a, l_b, a, b, A, B)` and are
//! **grid-independent**; they are built once per primitive pair, outside the
//! grid loop. Only the R-tensor sees the grid point, through `T = p|P-r_g|²`
//! (Boys function) and the components of `P - r_g`, and its recursion is a
//! pure multiply-add chain that vectorizes over a tile of [`TILE`](crate::md3c1e::TILE) points.
//! That split is exactly what libint2 cannot exploit (its per-primitive work
//! runs inside the per-charge loop; see `scripts/3c1e_spec.md` §2).
//!
//! # Conventions reproduced (all verified against the tree, see the spec §5)
//!
//! * Shell order/offsets: `PreparedBasis` order (atom-major, basis order within
//!   the atom). Obtained via [`PreparedBasis::located_shells`](crate::basis_bridge::PreparedBasis::located_shells).
//! * Cartesian components: CCA order (`lx` descending, then `ly` descending),
//!   i.e. `xx, xy, xz, yy, yz, zz`. Cartesian shells carry ONE shell-wide
//!   normalization (the `(l,0,0)` component is unit-normalized; `xy` is not) —
//!   the textbook per-component double factorial is deliberately NOT applied.
//! * Primitive normalization `N(a,l) = (2a/π)^{3/4} (4a)^{l/2} / √((2l-1)!!)`
//!   is folded into the contraction coefficients here; the stored `BasisSet`
//!   coefficients do not contain it (same as `ao_grid::radial`). The
//!   contraction-level renormalization is already done at basis load time.
//! * Pure (spherical) shells only for `l >= 2`, ordered `m = -l..=+l`
//!   (libint2 STANDARD). The transform matrix is the crate's existing
//!   `ecp::cart2sph` table (libcint convention) rescaled by `√(4π/(2l+1))`
//!   into ferric's Cartesian convention — see [`ferric_cart2sph`](crate::md3c1e::ferric_cart2sph). A pure
//!   `l = 1` shell is rejected (ferric never constructs one; libint2 would
//!   order it `y, z, x`).
//! * General contractions are already split into separate segmented shells by
//!   the basis loader, so nothing special is needed; E-tables are not shared
//!   across them (an optional optimization the spec notes).
//!
//! # Boys function
//!
//! `F_n(T)` for `n <= 8` (`l_a + l_b <= 8`, i.e. up to g-g). For `T < 35`: a
//! 10-term Taylor expansion about the nearest node of a table with spacing
//! `1/8` (node values from the spec's Taylor-series/downward-recursion
//! evaluator, [`boys_direct`](crate::md3c1e::boys_direct)), then downward recursion; for `T >= 35`: the
//! asymptotic `F_0 = ½√(π/T)` and upward recursion. The switch is at 35, not
//! the common 25, because the asymptotic form's neglected `erfc` tail is
//! `1.5e-12` relative at `T = 25` and `< 1e-16` only from `T ≈ 35` (measured
//! against mpmath, see the unit tests). `T = 0` (probe on a nucleus) is
//! exact by construction: there is no division by `T` anywhere.
//!
//! # Layout
//!
//! Per shell pair and batch: the Cartesian accumulator is `[ncart_a][ncart_b][G]`,
//! the R-tensor `[(n,t,u,v)][TILE]`, and the output block `[nf_a][nf_b][G]`
//! (grid innermost). A COSX K builder consumes that layout directly:
//! `Y[o1+i][g] += blk[i][j][g] * F[o2+j][g]` is a vector operation over `g`.
//! Use [`Md3c1e::for_each_pair`](crate::md3c1e::Md3c1e::for_each_pair) for that; [`Md3c1e::a_matrices`](crate::md3c1e::Md3c1e::a_matrices) assembles
//! dense `(G, nbf, nbf)` matrices for tests and small G.

use ferric_core::error::FerricError;
use ndarray::{Array2, Array3};
use std::sync::OnceLock;

use crate::basis_bridge::PreparedBasis;
use crate::cosx_a::{CosxPoint, CosxScreen, PairBounds};
use crate::ecp;

/// Highest per-shell angular momentum supported (g).
pub const MAX_L: usize = 4;
/// Highest total Hermite order `l_a + l_b` supported (g-g).
pub const MAX_LTOT: usize = 2 * MAX_L;
/// Grid points per SIMD tile — the innermost loop length of every
/// grid-dependent operation. Batches are padded up to a multiple of this.
pub const TILE: usize = 8;

/// Number of Hermite indices `(t,u,v)` with `t+u+v <= MAX_LTOT`: C(11,3).
const N_TUV: usize = 165;
const NONE: u32 = u32::MAX;
/// Primitive pairs whose `|pref| K_AB` falls below this are skipped. It is an
/// underflow guard (the integrals are bounded by this scale), not an accuracy
/// knob: `1e-30` is ~18 orders below any tolerance in the tree.
const PRIM_SCREEN: f64 = 1e-30;

// ---------------------------------------------------------------------------
// Boys function
// ---------------------------------------------------------------------------

/// Taylor/downward below this `T`, asymptotic/upward at or above it.
pub const BOYS_T_SWITCH: f64 = 35.0;
const BOYS_H: f64 = 0.125;
const BOYS_INV_H: f64 = 8.0;
/// Taylor terms about the nearest node: `|δ| <= 1/16`, so the first omitted
/// term is `< (1/16)^10 / 10! ≈ 2.5e-19` relative.
const BOYS_TAYLOR_TERMS: usize = 10;
const BOYS_NMAX_TAB: usize = MAX_LTOT + BOYS_TAYLOR_TERMS;
const BOYS_STRIDE: usize = BOYS_NMAX_TAB + 1;
/// Nodes `T_k = k/8`, `k = 0..=281`, covering `T < 35` with `round(T*8) <= 280`.
const BOYS_NODES: usize = 282;

const INV_FACT: [f64; BOYS_TAYLOR_TERMS] = [
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

/// `1/(2n+1)` for `n = 0..=MAX_LTOT`.
const INV_ODD: [f64; MAX_LTOT + 1] = {
    let mut a = [0.0_f64; MAX_LTOT + 1];
    let mut n = 0;
    while n <= MAX_LTOT {
        a[n] = 1.0 / (2 * n + 1) as f64;
        n += 1;
    }
    a
};

/// Direct evaluation of `F_0(T) .. F_nmax(T)` at one `T` — the spec's
/// algorithm (`scripts/3c1e_reference.py::boys_vec`), scalar.
///
/// `T < 35`: Taylor series for the TOP order `F_nmax(T) = e^{-T} Σ_k (2T)^k /
/// (2 nmax + 2k + 1)!!`, then the stable downward recursion
/// `F_n = (2T F_{n+1} + e^{-T}) / (2n+1)`. `T >= 35`: `F_0 = ½√(π/T)` and the
/// upward recursion `F_{n+1} = ((2n+1) F_n - e^{-T}) / (2T)`, stable there.
///
/// Exact at `T = 0` (no `0/0`: the series collapses to its first term). This
/// is the construction the hot path's table is built from, and the
/// independent reference the table path is tested against.
pub fn boys_direct(nmax: usize, t: f64, out: &mut [f64]) {
    debug_assert!(out.len() > nmax);
    if t < BOYS_T_SWITCH {
        boys_series(nmax, t, out);
    } else {
        let expt = (-t).exp();
        out[0] = 0.5 * (std::f64::consts::PI / t).sqrt();
        let inv_2t = 0.5 / t;
        for n in 0..nmax {
            out[n + 1] = ((2 * n + 1) as f64 * out[n] - expt) * inv_2t;
        }
    }
}

/// The series branch of [`boys_direct`] with no `T` gate: Taylor series for
/// `F_nmax(T)` to `1e-18` relative, then downward recursion. Converges for
/// any finite `T` (slowly for large `T`); used as the branch-independent
/// reference in the breakpoint guard test.
pub fn boys_series(nmax: usize, t: f64, out: &mut [f64]) {
    debug_assert!(out.len() > nmax);
    let expt = (-t).exp();
    let two_t = 2.0 * t;
    let mut term = 1.0 / (2 * nmax + 1) as f64;
    let mut total = term;
    let mut k = 0usize;
    loop {
        term *= two_t / (2 * nmax + 2 * k + 3) as f64;
        total += term;
        k += 1;
        if term < 1e-18 * total || k > 2000 {
            break;
        }
    }
    out[nmax] = expt * total;
    for n in (0..nmax).rev() {
        out[n] = (two_t * out[n + 1] + expt) / (2 * n + 1) as f64;
    }
}

struct BoysTable {
    /// `[node][n]`, `n = 0..=BOYS_NMAX_TAB`.
    vals: Vec<f64>,
}

static BOYS_TABLE: OnceLock<BoysTable> = OnceLock::new();

fn boys_table() -> &'static BoysTable {
    BOYS_TABLE.get_or_init(|| {
        let mut vals = vec![0.0_f64; BOYS_NODES * BOYS_STRIDE];
        for (k, row) in vals.as_chunks_mut::<BOYS_STRIDE>().0.iter_mut().enumerate() {
            boys_direct(BOYS_NMAX_TAB, k as f64 * BOYS_H, row);
        }
        BoysTable { vals }
    })
}

/// Table-accelerated `F_0..F_nmax` at one `T` (the hot-path evaluator).
///
/// Same branch structure as [`boys_direct`]; below the switch the top order
/// comes from a [`BOYS_TAYLOR_TERMS`]-term Taylor expansion about the nearest
/// table node instead of the full series.
#[inline]
fn boys_tabulated(tab: &BoysTable, nmax: usize, t: f64, out: &mut [f64], stride: usize) {
    let expt = (-t).exp();
    if t < BOYS_T_SWITCH {
        let k = (t * BOYS_INV_H + 0.5) as usize;
        let neg_delta = k as f64 * BOYS_H - t;
        let node = &tab.vals[k * BOYS_STRIDE + nmax..k * BOYS_STRIDE + nmax + BOYS_TAYLOR_TERMS];
        let mut f = node[BOYS_TAYLOR_TERMS - 1] * INV_FACT[BOYS_TAYLOR_TERMS - 1];
        for j in (0..BOYS_TAYLOR_TERMS - 1).rev() {
            f = f * neg_delta + node[j] * INV_FACT[j];
        }
        out[nmax * stride] = f;
        let two_t = 2.0 * t;
        for n in (0..nmax).rev() {
            f = (two_t * f + expt) * INV_ODD[n];
            out[n * stride] = f;
        }
    } else {
        let mut f = 0.5 * (std::f64::consts::PI / t).sqrt();
        out[0] = f;
        let inv_2t = 0.5 / t;
        for n in 0..nmax {
            f = ((2 * n + 1) as f64 * f - expt) * inv_2t;
            out[(n + 1) * stride] = f;
        }
    }
}

/// Public scalar entry point for the table path (tests, diagnostics).
pub fn boys(nmax: usize, t: f64, out: &mut [f64]) {
    assert!(nmax <= MAX_LTOT, "boys: nmax {nmax} > MAX_LTOT {MAX_LTOT}");
    assert!(out.len() > nmax);
    boys_tabulated(boys_table(), nmax, t, out, 1);
}

// ---------------------------------------------------------------------------
// Hermite indexing and the R-tensor recursion "program"
// ---------------------------------------------------------------------------

struct Hermite {
    /// `(t,u,v) -> canonical index`, flat `[t][u][v]` with extent 9 each.
    idx: Vec<u16>,
    /// Canonical order: increasing total order, then `t` desc, then `u` desc.
    list: Vec<[u8; 3]>,
    /// Number of `(t,u,v)` with `t+u+v <= s`, `s = 0..=MAX_LTOT`.
    count_le: [usize; MAX_LTOT + 1],
}

impl Hermite {
    #[inline]
    fn index(&self, t: usize, u: usize, v: usize) -> usize {
        self.idx[(t * 9 + u) * 9 + v] as usize
    }
}

static HERMITE: OnceLock<Hermite> = OnceLock::new();

fn hermite() -> &'static Hermite {
    HERMITE.get_or_init(|| {
        let mut idx = vec![u16::MAX; 9 * 9 * 9];
        let mut list = Vec::with_capacity(N_TUV);
        let mut count_le = [0usize; MAX_LTOT + 1];
        for s in 0..=MAX_LTOT {
            for t in (0..=s).rev() {
                for u in (0..=s - t).rev() {
                    let v = s - t - u;
                    idx[(t * 9 + u) * 9 + v] = list.len() as u16;
                    list.push([t as u8, u as u8, v as u8]);
                }
            }
            count_le[s] = list.len();
        }
        debug_assert_eq!(list.len(), N_TUV);
        Hermite { idx, list, count_le }
    })
}

/// One recursion step `R[dst] = coef * R[src2] + PC[dir] * R[src]` (or
/// `PC[dir] * R[src]` when `src2 == NONE`). Indices are tile numbers.
struct RStep {
    dst: u32,
    src: u32,
    src2: u32,
    coef: f64,
    dir: u8,
}

/// The full `R^n_{tuv}` recursion for one total order `L`, as a dependency-
/// ordered list of steps over `[n][tuv]` storage. Grid-independent.
struct RProgram {
    steps: Vec<RStep>,
    /// Total number of `(n, t, u, v)` tiles: `Σ_n C(L-n+3, 3)`.
    nvec: usize,
    /// Tile offset of the `n`-block, `n = 0..=L` (`base[0] == 0`, so the
    /// first `C(L+3,3)` tiles are `R^0_{tuv}` in canonical order).
    base: [usize; MAX_LTOT + 2],
}

static R_PROGRAMS: OnceLock<Vec<RProgram>> = OnceLock::new();

fn r_programs() -> &'static [RProgram] {
    R_PROGRAMS.get_or_init(|| (0..=MAX_LTOT).map(|l| build_r_program(l, hermite())).collect())
}

fn build_r_program(l: usize, h: &Hermite) -> RProgram {
    let mut base = [0usize; MAX_LTOT + 2];
    for n in 0..=l {
        base[n + 1] = base[n] + h.count_le[l - n];
    }
    let nvec = base[l + 1];
    let mut steps = Vec::new();
    for s in 1..=l {
        let lo = h.count_le[s - 1];
        let hi = h.count_le[s];
        for n in 0..=(l - s) {
            for &[t, u, v] in &h.list[lo..hi] {
                let (t, u, v) = (t as usize, u as usize, v as usize);
                let dst = (base[n] + h.index(t, u, v)) as u32;
                let up = base[n + 1];
                let (src, src2, coef, dir) = if t > 0 {
                    let s2 = if t >= 2 { (up + h.index(t - 2, u, v)) as u32 } else { NONE };
                    ((up + h.index(t - 1, u, v)) as u32, s2, (t - 1) as f64, 0u8)
                } else if u > 0 {
                    let s2 = if u >= 2 { (up + h.index(t, u - 2, v)) as u32 } else { NONE };
                    ((up + h.index(t, u - 1, v)) as u32, s2, (u - 1) as f64, 1u8)
                } else {
                    let s2 = if v >= 2 { (up + h.index(t, u, v - 2)) as u32 } else { NONE };
                    ((up + h.index(t, u, v - 1)) as u32, s2, (v - 1) as f64, 2u8)
                };
                steps.push(RStep { dst, src, src2, coef, dir });
            }
        }
    }
    RProgram { steps, nvec, base }
}

// ---------------------------------------------------------------------------
// Hermite E-coefficients (grid-independent)
// ---------------------------------------------------------------------------

/// Fill the E-table for one Cartesian direction: `E_t^{ij}` at
/// `(i*(lb+1) + j)*(la+lb+1) + t`, for `i <= la`, `j <= lb`, `t <= i+j`
/// (zero elsewhere). `q = A_x - B_x`.
///
/// Recursion (Helgaker eq. 9.5.6-7, with `X_PA = -(b/p) q`, `X_PB = (a/p) q`):
/// `E_t^{i+1,j} = E_{t-1}^{ij}/(2p) + X_PA E_t^{ij} + (t+1) E_{t+1}^{ij}` and
/// the same with `X_PB` for `j+1`. `E_0^{00} = exp(-a b q² / p)`.
fn e_table(la: usize, lb: usize, a: f64, b: f64, q: f64, out: &mut [f64]) {
    let p = a + b;
    let mu = a * b / p;
    let inv2p = 0.5 / p;
    let xpa = -(b / p) * q;
    let xpb = (a / p) * q;
    let sj = la + lb + 1;
    let si = (lb + 1) * sj;
    for v in out[..(la + 1) * si].iter_mut() {
        *v = 0.0;
    }
    out[0] = (-mu * q * q).exp();
    // Raise i at j = 0.
    for i in 0..la {
        let row = i * si;
        for t in 0..=i + 1 {
            out[(i + 1) * si + t] = e_step(&out[row..row + sj], i, t, inv2p, xpa);
        }
    }
    // Raise j for every i.
    for j in 0..lb {
        for i in 0..=la {
            let row = i * si + j * sj;
            for t in 0..=i + j + 1 {
                out[i * si + (j + 1) * sj + t] = e_step(&out[row..row + sj], i + j, t, inv2p, xpb);
            }
        }
    }
}

/// One E-recursion step from the row `E_*^{(ij)}` (valid `t <= ij_sum`).
#[inline]
fn e_step(row: &[f64], ij_sum: usize, t: usize, inv2p: f64, xp: f64) -> f64 {
    let e_m1 = if t >= 1 { row[t - 1] } else { 0.0 };
    let e_0 = if t <= ij_sum { row[t] } else { 0.0 };
    let e_p1 = if t < ij_sum { row[t + 1] } else { 0.0 };
    inv2p * e_m1 + xp * e_0 + (t + 1) as f64 * e_p1
}

// ---------------------------------------------------------------------------
// Cartesian components, normalization, cart->sph
// ---------------------------------------------------------------------------

/// Cartesian `(lx, ly, lz)` components of shell `l` in CCA order.
pub fn cart_components(l: usize) -> Vec<[u8; 3]> {
    let mut out = Vec::with_capacity((l + 1) * (l + 2) / 2);
    for lx in (0..=l).rev() {
        for ly in (0..=l - lx).rev() {
            out.push([lx as u8, ly as u8, (l - lx - ly) as u8]);
        }
    }
    out
}

/// `N(a, l) = (2a/π)^{3/4} (4a)^{l/2} / √((2l-1)!!)` — the primitive
/// normalization that makes the `(l,0,0)` Cartesian component unit-normalized
/// (identical to `ao_grid::radial`'s factor and to libint2's).
fn prim_norm(a: f64, l: usize) -> f64 {
    const DFACT: [f64; MAX_L + 1] = [1.0, 1.0, 3.0, 15.0, 105.0];
    (2.0 * a / std::f64::consts::PI).powf(0.75) * (4.0 * a).powi(l as i32).sqrt() / DFACT[l].sqrt()
}

/// Cartesian→spherical matrix `(ncart × 2l+1)`, row-major, for ferric's
/// Cartesian convention (unit `(l,0,0)` component): the crate's `ecp::cart2sph`
/// table (libcint convention) scaled by `√(4π/(2l+1))`. `l = 0, 1` return
/// identity-equivalent matrices (ferric never builds pure shells there).
pub fn ferric_cart2sph(l: usize) -> Vec<f64> {
    let scale = (4.0 * std::f64::consts::PI / (2 * l + 1) as f64).sqrt();
    ecp::cart2sph(l as i32).iter().map(|c| c * scale).collect()
}

// ---------------------------------------------------------------------------
// The kernel
// ---------------------------------------------------------------------------

struct ShellData {
    l: usize,
    pure: bool,
    center: [f64; 3],
    exps: Vec<f64>,
    /// Contraction coefficients WITH `N(a, l)` folded in.
    coefs: Vec<f64>,
    ncart: usize,
    nfun: usize,
    off: usize,
}

/// Grid-independent kernel state for one molecule/basis: shell data with
/// normalization folded in, Cartesian component lists and cart→sph matrices.
/// `O(nshells)` memory; the per-primitive-pair tables are built per shell
/// pair inside each batch call (hoisted out of the grid loop, not persisted).
///
/// `Send + Sync`; per-thread work buffers live in [`Md3c1eScratch`].
pub struct Md3c1e {
    shells: Vec<ShellData>,
    nbf: usize,
    comps: Vec<Vec<[u8; 3]>>,
    c2s: Vec<Vec<f64>>,
    use_fma: bool,
}

/// Per-thread work buffers for [`Md3c1e`]. Obtain via [`Md3c1e::scratch`];
/// reuse across calls (every buffer grows to its high-water mark once).
#[derive(Default)]
pub struct Md3c1eScratch {
    r: Vec<f64>,
    boys: Vec<f64>,
    e: Vec<f64>,
    coef_vals: Vec<f64>,
    coef_idx: Vec<u32>,
    coef_start: Vec<u32>,
    cart: Vec<f64>,
    tmp: Vec<f64>,
    block: Vec<f64>,
}

/// Operation count per grid point for a full unscreened sweep, in the spec's
/// convention (`scripts/3c1e_spec.md` §7): R-tensor recursion FLOPs and
/// E×E×E-against-R contraction FLOPs; Boys evaluation and cart→sph excluded.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Md3c1eFlops {
    /// Seeding `(-2p)^n F_n` plus one multiply-add (3 flops) or one multiply
    /// (1 flop) per recursion element, per primitive pair.
    pub r_tensor: f64,
    /// Two flops per nonzero `E_t E_u E_v` coefficient per primitive pair.
    pub contraction: f64,
    /// Surviving primitive pairs in the sweep — each costs one Boys
    /// evaluation (with one `exp`) per grid point, which the FLOP columns
    /// exclude; `prim_pairs * t_boys` is the Boys share of the per-point time.
    pub prim_pairs: f64,
}

impl Md3c1eFlops {
    /// `r_tensor + contraction`.
    pub fn total(&self) -> f64 {
        self.r_tensor + self.contraction
    }
}

/// Result of a batched dense assembly ([`Md3c1e::a_matrices`]).
pub struct Md3c1eBatch {
    /// `(npts, nbf, nbf)`, `+1/|r-r_g|` sign, symmetric in the last two axes.
    pub a: Array3<f64>,
    /// Shell pairs evaluated (after batch-level screening).
    pub pairs_kept: usize,
    /// Shell pairs considered (`nsh (nsh+1) / 2`).
    pub pairs_total: usize,
}

#[inline(always)]
fn fma<const FMA: bool>(a: f64, b: f64, c: f64) -> f64 {
    if FMA {
        a.mul_add(b, c)
    } else {
        a * b + c
    }
}

/// `cart[mn][tile] += Σ_k coef_k * R^0[idx_k][tile]` for every Cartesian
/// component pair `mn`. `idx` is pre-multiplied by `TILE`.
#[inline(always)]
fn contract_tile<const FMA: bool>(
    r0: &[f64],
    vals: &[f64],
    idx: &[u32],
    start: &[u32],
    cart: &mut [f64],
    tile_off: usize,
    gpad: usize,
) {
    for mn in 0..start.len() - 1 {
        let mut acc = [0.0_f64; TILE];
        for k in start[mn] as usize..start[mn + 1] as usize {
            let c = vals[k];
            let src = &r0[idx[k] as usize..idx[k] as usize + TILE];
            for g in 0..TILE {
                acc[g] = fma::<FMA>(c, src[g], acc[g]);
            }
        }
        let dst = &mut cart[mn * gpad + tile_off..mn * gpad + tile_off + TILE];
        for g in 0..TILE {
            dst[g] += acc[g];
        }
    }
}

/// Everything grid-dependent for one primitive pair on one tile of points.
#[inline(always)]
fn tile_kernel<const FMA: bool>(
    prog: &RProgram,
    l: usize,
    p: f64,
    pc: &[[f64; TILE]; 3],
    boys: &[f64],
    r: &mut [f64],
    vals: &[f64],
    idx: &[u32],
    start: &[u32],
    cart: &mut [f64],
    tile_off: usize,
    gpad: usize,
) {
    build_r_tile::<FMA>(prog, l, p, pc, boys, r);
    contract_tile::<FMA>(r, vals, idx, start, cart, tile_off, gpad);
}

/// Seed `R^n_{000} = (-2p)^n F_n(T)` and run the recursion program for one
/// tile. `boys` is `[(n)][TILE]`, `r` is `[(n,t,u,v) tile][TILE]`. Source
/// tiles are copied to locals before the destination is borrowed, which is
/// both borrow-clean and frees the vectorizer from alias proofs.
#[inline(always)]
fn build_r_tile<const FMA: bool>(prog: &RProgram, l: usize, p: f64, pc: &[[f64; TILE]; 3], boys: &[f64], r: &mut [f64]) {
    let mut fac = 1.0_f64;
    for n in 0..=l {
        let dst = &mut r[prog.base[n] * TILE..prog.base[n] * TILE + TILE];
        let src = &boys[n * TILE..n * TILE + TILE];
        for g in 0..TILE {
            dst[g] = fac * src[g];
        }
        fac *= -2.0 * p;
    }
    for st in &prog.steps {
        let s = st.src as usize * TILE;
        let a: [f64; TILE] = r[s..s + TILE].try_into().expect("tile");
        let pcd = &pc[st.dir as usize];
        let d = st.dst as usize * TILE;
        if st.src2 == NONE {
            let dst = &mut r[d..d + TILE];
            for g in 0..TILE {
                dst[g] = pcd[g] * a[g];
            }
        } else {
            let s2 = st.src2 as usize * TILE;
            let a2: [f64; TILE] = r[s2..s2 + TILE].try_into().expect("tile");
            let c = st.coef;
            let dst = &mut r[d..d + TILE];
            for g in 0..TILE {
                dst[g] = fma::<FMA>(c, a2[g], pcd[g] * a[g]);
            }
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn tile_kernel_avx2(
    prog: &RProgram,
    l: usize,
    p: f64,
    pc: &[[f64; TILE]; 3],
    boys: &[f64],
    r: &mut [f64],
    vals: &[f64],
    idx: &[u32],
    start: &[u32],
    cart: &mut [f64],
    tile_off: usize,
    gpad: usize,
) {
    tile_kernel::<true>(prog, l, p, pc, boys, r, vals, idx, start, cart, tile_off, gpad)
}

fn detect_fma() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        false
    }
}

impl Md3c1e {
    /// Build the kernel state for `prep`. Errors on `l > 4` and on a pure
    /// `l = 1` shell; every shell dimension is cross-checked against
    /// `prep.shell_dims()` so the AO layout provably matches libint2's.
    pub fn new(prep: &PreparedBasis) -> Result<Self, FerricError> {
        let dims = prep.shell_dims();
        let offs = prep.shell_offsets();
        let mut shells = Vec::with_capacity(prep.nshells());
        for (s, sh) in prep.located_shells().iter().enumerate() {
            if sh.l < 0 || sh.l as usize > MAX_L {
                return Err(FerricError::Basis(format!(
                    "md3c1e: shell {s} has l={} (supported 0..={MAX_L})",
                    sh.l
                )));
            }
            let l = sh.l as usize;
            if sh.pure && l == 1 {
                return Err(FerricError::Basis(format!(
                    "md3c1e: shell {s} is a pure p shell; unsupported (libint2 orders pure p as y,z,x and ferric never constructs one)"
                )));
            }
            let pure = sh.pure && l >= 2;
            let ncart = (l + 1) * (l + 2) / 2;
            let nfun = if pure { 2 * l + 1 } else { ncart };
            if nfun != dims[s] {
                return Err(FerricError::Basis(format!(
                    "md3c1e: shell {s} (l={l}, pure={pure}) has {nfun} functions but libint2 reports {}",
                    dims[s]
                )));
            }
            if sh.exponents.len() != sh.coefficients.len() {
                return Err(FerricError::Basis(format!("md3c1e: shell {s} exponent/coefficient length mismatch")));
            }
            let coefs = sh.exponents.iter().zip(sh.coefficients).map(|(&a, &c)| c * prim_norm(a, l)).collect();
            shells.push(ShellData {
                l,
                pure,
                center: sh.center,
                exps: sh.exponents.to_vec(),
                coefs,
                ncart,
                nfun,
                off: offs[s],
            });
        }
        let comps = (0..=MAX_L).map(cart_components).collect();
        let c2s = (0..=MAX_L).map(ferric_cart2sph).collect();
        Ok(Self { shells, nbf: prep.nbasis(), comps, c2s, use_fma: detect_fma() })
    }

    /// Fresh work buffers for this kernel.
    pub fn scratch(&self) -> Md3c1eScratch {
        Md3c1eScratch::default()
    }

    /// Number of basis functions.
    pub fn nbasis(&self) -> usize {
        self.nbf
    }
    /// Number of shells.
    pub fn nshells(&self) -> usize {
        self.shells.len()
    }
    /// Functions in shell `s`.
    pub fn shell_dim(&self, s: usize) -> usize {
        self.shells[s].nfun
    }
    /// AO offset of shell `s`.
    pub fn shell_offset(&self, s: usize) -> usize {
        self.shells[s].off
    }
    /// Angular momentum of shell `s`.
    pub fn shell_l(&self, s: usize) -> usize {
        self.shells[s].l
    }
    /// Whether the AVX2+FMA code path is in use (runtime-detected).
    pub fn uses_fma(&self) -> bool {
        self.use_fma
    }
    /// Force the portable (non-FMA) path; for cross-checking the two paths.
    pub fn set_use_fma(&mut self, on: bool) {
        self.use_fma = on && detect_fma();
    }

    /// Grid-independent primitive-pair setup: E-tables for x, y, z and the
    /// per-component-pair list of nonzero `pref * E_t E_u E_v` coefficients
    /// (with their `R^0_{tuv}` tile index). Returns `(p, P)`, or `None` when
    /// the pair prefactor underflows [`PRIM_SCREEN`].
    #[allow(clippy::too_many_arguments)]
    fn prim_pair_setup(
        &self,
        sa: &ShellData,
        sb: &ShellData,
        a: f64,
        ca: f64,
        b: f64,
        cb: f64,
        scr: &mut Md3c1eScratch,
    ) -> Option<(f64, [f64; 3])> {
        let p = a + b;
        let pref = 2.0 * std::f64::consts::PI / p * ca * cb;
        let q = [sa.center[0] - sb.center[0], sa.center[1] - sb.center[1], sa.center[2] - sb.center[2]];
        let kab = (-(a * b / p) * (q[0] * q[0] + q[1] * q[1] + q[2] * q[2])).exp();
        if (pref * kab).abs() < PRIM_SCREEN {
            return None;
        }
        let pcen = [
            (a * sa.center[0] + b * sb.center[0]) / p,
            (a * sa.center[1] + b * sb.center[1]) / p,
            (a * sa.center[2] + b * sb.center[2]) / p,
        ];
        let (la, lb) = (sa.l, sb.l);
        let lt = la + lb;
        let sj = lt + 1;
        let si = (lb + 1) * sj;
        let e_len = (la + 1) * si;
        scr.e.resize(3 * e_len, 0.0);
        for d in 0..3 {
            e_table(la, lb, a, b, q[d], &mut scr.e[d * e_len..(d + 1) * e_len]);
        }
        let (ex, rest) = scr.e.split_at(e_len);
        let (ey, ez) = rest.split_at(e_len);
        let h = hermite();
        scr.coef_vals.clear();
        scr.coef_idx.clear();
        scr.coef_start.clear();
        for &[ax, ay, az] in &self.comps[la] {
            for &[bx, by, bz] in &self.comps[lb] {
                scr.coef_start.push(scr.coef_vals.len() as u32);
                let (ax, ay, az) = (ax as usize, ay as usize, az as usize);
                let (bx, by, bz) = (bx as usize, by as usize, bz as usize);
                let rx = &ex[(ax * (lb + 1) + bx) * sj..];
                let ry = &ey[(ay * (lb + 1) + by) * sj..];
                let rz = &ez[(az * (lb + 1) + bz) * sj..];
                for t in 0..=ax + bx {
                    let etx = pref * rx[t];
                    if etx == 0.0 {
                        continue;
                    }
                    for u in 0..=ay + by {
                        let ety = etx * ry[u];
                        if ety == 0.0 {
                            continue;
                        }
                        for v in 0..=az + bz {
                            let c = ety * rz[v];
                            if c != 0.0 {
                                scr.coef_vals.push(c);
                                scr.coef_idx.push((h.index(t, u, v) * TILE) as u32);
                            }
                        }
                    }
                }
            }
        }
        scr.coef_start.push(scr.coef_vals.len() as u32);
        Some((p, pcen))
    }

    /// The Cartesian block of shell pair `(s1, s2)` over `pts`, accumulated
    /// into `scr.cart` as `[ncart_1][ncart_2][gpad]`. Returns `gpad`.
    fn cart_block(&self, s1: usize, s2: usize, pts: &[[f64; 3]], scr: &mut Md3c1eScratch) -> usize {
        let (sa, sb) = (&self.shells[s1], &self.shells[s2]);
        let npts = pts.len();
        let ntile = npts.div_ceil(TILE);
        let gpad = ntile * TILE;
        let lt = sa.l + sb.l;
        let prog = &r_programs()[lt];
        let tab = boys_table();
        scr.cart.clear();
        scr.cart.resize(sa.ncart * sb.ncart * gpad, 0.0);
        scr.r.resize(prog.nvec * TILE, 0.0);
        scr.boys.resize((lt + 1) * TILE, 0.0);
        for (&a, &ca) in sa.exps.iter().zip(&sa.coefs) {
            for (&b, &cb) in sb.exps.iter().zip(&sb.coefs) {
                let Some((p, pcen)) = self.prim_pair_setup(sa, sb, a, ca, b, cb, scr) else {
                    continue;
                };
                let Md3c1eScratch { r, boys, coef_vals, coef_idx, coef_start, cart, .. } = scr;
                for tile in 0..ntile {
                    let mut pc = [[0.0_f64; TILE]; 3];
                    let mut t = [0.0_f64; TILE];
                    for g in 0..TILE {
                        // Padded lanes repeat the last point; their results are
                        // discarded by the caller (finite, never NaN).
                        let rg = pts[(tile * TILE + g).min(npts - 1)];
                        pc[0][g] = pcen[0] - rg[0];
                        pc[1][g] = pcen[1] - rg[1];
                        pc[2][g] = pcen[2] - rg[2];
                        t[g] = p * (pc[0][g] * pc[0][g] + pc[1][g] * pc[1][g] + pc[2][g] * pc[2][g]);
                    }
                    for g in 0..TILE {
                        boys_tabulated(tab, lt, t[g], &mut boys[g..], TILE);
                    }
                    let off = tile * TILE;
                    if self.use_fma {
                        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                        // SAFETY: `use_fma` is only ever true when
                        // `detect_fma()` confirmed AVX2 and FMA at runtime.
                        unsafe {
                            tile_kernel_avx2(prog, lt, p, &pc, boys, r, coef_vals, coef_idx, coef_start, cart, off, gpad);
                        }
                    } else {
                        tile_kernel::<false>(prog, lt, p, &pc, boys, r, coef_vals, coef_idx, coef_start, cart, off, gpad);
                    }
                }
            }
        }
        gpad
    }

    /// Cartesian `[ncart_1][ncart_2][gpad]` → output `[nf_1][nf_2][npts]`,
    /// applying the cart→sph transform on whichever sides are pure.
    fn cart_to_out(&self, s1: usize, s2: usize, npts: usize, gpad: usize, scr: &mut Md3c1eScratch, out: &mut [f64]) {
        let (sa, sb) = (&self.shells[s1], &self.shells[s2]);
        let (nca, ncb, nfa, nfb) = (sa.ncart, sb.ncart, sa.nfun, sb.nfun);
        // Right side: tmp[m][j][gpad] = Σ_n C_b[n][j] cart[m][n][gpad].
        let right: &[f64] = if sb.pure {
            let cb = &self.c2s[sb.l];
            scr.tmp.clear();
            scr.tmp.resize(nca * nfb * gpad, 0.0);
            for m in 0..nca {
                for j in 0..nfb {
                    let dst = &mut scr.tmp[(m * nfb + j) * gpad..(m * nfb + j + 1) * gpad];
                    for n in 0..ncb {
                        let c = cb[n * nfb + j];
                        if c != 0.0 {
                            let src = &scr.cart[(m * ncb + n) * gpad..(m * ncb + n + 1) * gpad];
                            for (d, s) in dst.iter_mut().zip(src) {
                                *d += c * s;
                            }
                        }
                    }
                }
            }
            &scr.tmp
        } else {
            &scr.cart
        };
        // Left side: out[i][j][g] = Σ_m C_a[m][i] right[m][j][g].
        for i in 0..nfa {
            for j in 0..nfb {
                let dst = &mut out[(i * nfb + j) * npts..(i * nfb + j + 1) * npts];
                if sa.pure {
                    let ca = &self.c2s[sa.l];
                    dst.iter_mut().for_each(|d| *d = 0.0);
                    for m in 0..nca {
                        let c = ca[m * nfa + i];
                        if c != 0.0 {
                            let src = &right[(m * nfb + j) * gpad..(m * nfb + j) * gpad + npts];
                            for (d, s) in dst.iter_mut().zip(src) {
                                *d += c * s;
                            }
                        }
                    }
                } else {
                    dst.copy_from_slice(&right[(i * nfb + j) * gpad..(i * nfb + j) * gpad + npts]);
                }
            }
        }
    }

    /// The `A^g` block of shell pair `(s1, s2)` — any order, `s1 == s2`
    /// allowed — over the batch `pts`, written to `out` as
    /// `out[(i * nf_2 + j) * npts + g]` (grid innermost), `+1/|r-r_g|` sign.
    /// `out.len()` must be at least `nf_1 * nf_2 * npts`.
    ///
    /// This is the fundamental batched entry point; everything else is a
    /// wrapper. Cost per call is one primitive-pair setup plus
    /// `ceil(npts / TILE)` tiles, so amortization needs `npts >> TILE`.
    pub fn pair_block(&self, s1: usize, s2: usize, pts: &[[f64; 3]], scr: &mut Md3c1eScratch, out: &mut [f64]) -> Result<(), FerricError> {
        let nsh = self.shells.len();
        if s1 >= nsh || s2 >= nsh {
            return Err(FerricError::General(format!("md3c1e::pair_block: shell index out of range ({s1}, {s2}) for {nsh} shells")));
        }
        if pts.is_empty() {
            return Ok(());
        }
        let need = self.shells[s1].nfun * self.shells[s2].nfun * pts.len();
        if out.len() < need {
            return Err(FerricError::General(format!("md3c1e::pair_block: out has {} elements, need {need}", out.len())));
        }
        let gpad = self.cart_block(s1, s2, pts, scr);
        self.cart_to_out(s1, s2, pts.len(), gpad, scr, out);
        Ok(())
    }

    /// Whether pair `(s1, s2)` survives the screen for this batch: kept when
    /// ANY point's estimate reaches the threshold (conservative), and always
    /// kept for a vacuous screen or absent bounds — the same rule as
    /// `cosx_a::a_matrix_at_point_with` when `pts.len() == 1`.
    fn keep_pair(s1: usize, s2: usize, pts: &[[f64; 3]], bounds: Option<&PairBounds>, screen: CosxScreen) -> bool {
        if screen.is_vacuous() {
            return true;
        }
        match bounds {
            None => true,
            Some(b) => pts.iter().any(|r| b.estimate(s1, s2, r) >= screen.threshold),
        }
    }

    /// Sweep every shell pair `s1 >= s2` for the batch `pts`, calling
    /// `f(s1, s2, block)` with the block in [`Md3c1e::pair_block`] layout
    /// (`[nf_1][nf_2][npts]`). Returns `(pairs_kept, pairs_total)`.
    ///
    /// This is the entry point a COSX K builder should call: for each block,
    /// `Y[o1+i][g] += blk[i][j][g] * F[o2+j][g]` and (for `s1 != s2`)
    /// `Y[o2+j][g] += blk[i][j][g] * F[o1+i][g]`, both vector ops over `g`.
    pub fn for_each_pair<F>(
        &self,
        pts: &[[f64; 3]],
        bounds: Option<&PairBounds>,
        screen: CosxScreen,
        scr: &mut Md3c1eScratch,
        mut f: F,
    ) -> Result<(usize, usize), FerricError>
    where
        F: FnMut(usize, usize, &[f64]),
    {
        let nsh = self.shells.len();
        let mut kept = 0usize;
        let mut total = 0usize;
        let mut block = std::mem::take(&mut scr.block);
        let result = (|| {
            for s1 in 0..nsh {
                for s2 in 0..=s1 {
                    total += 1;
                    if !Self::keep_pair(s1, s2, pts, bounds, screen) {
                        continue;
                    }
                    kept += 1;
                    let need = self.shells[s1].nfun * self.shells[s2].nfun * pts.len();
                    block.resize(need, 0.0);
                    self.pair_block(s1, s2, pts, scr, &mut block[..need])?;
                    f(s1, s2, &block[..need]);
                }
            }
            Ok(())
        })();
        scr.block = block;
        result.map(|()| (kept, total))
    }

    /// Dense `(npts, nbf, nbf)` matrices for the batch, symmetric fill.
    /// Memory is `npts * nbf² * 8` bytes — for large bases keep `npts` small
    /// or use [`Md3c1e::for_each_pair`].
    pub fn a_matrices(&self, pts: &[[f64; 3]], bounds: Option<&PairBounds>, screen: CosxScreen, scr: &mut Md3c1eScratch) -> Result<Md3c1eBatch, FerricError> {
        let npts = pts.len();
        let nbf = self.nbf;
        let mut a = Array3::<f64>::zeros((npts, nbf, nbf));
        let shells = &self.shells;
        let (pairs_kept, pairs_total) = self.for_each_pair(pts, bounds, screen, scr, |s1, s2, blk| {
            let (sa, sb) = (&shells[s1], &shells[s2]);
            for i in 0..sa.nfun {
                for j in 0..sb.nfun {
                    let row = &blk[(i * sb.nfun + j) * npts..(i * sb.nfun + j + 1) * npts];
                    for (g, &v) in row.iter().enumerate() {
                        a[(g, sa.off + i, sb.off + j)] = v;
                        a[(g, sb.off + j, sa.off + i)] = v;
                    }
                }
            }
        })?;
        Ok(Md3c1eBatch { a, pairs_kept, pairs_total })
    }

    /// Operation count per grid point for the full unscreened sweep (see
    /// [`Md3c1eFlops`]). Walks the same primitive-pair setup as a real batch,
    /// so it reflects the actual nonzero-coefficient structure.
    pub fn flops_per_point(&self) -> Md3c1eFlops {
        let mut scr = self.scratch();
        let mut r_tensor = 0.0_f64;
        let mut contraction = 0.0_f64;
        let mut prim_pairs = 0.0_f64;
        let progs = r_programs();
        for s1 in 0..self.shells.len() {
            for s2 in 0..=s1 {
                let (sa, sb) = (&self.shells[s1], &self.shells[s2]);
                let prog = &progs[sa.l + sb.l];
                let r_flops: f64 = (sa.l as f64)
                    + prog.steps.iter().map(|st| if st.src2 == NONE { 1.0 } else { 3.0 }).sum::<f64>();
                for (&a, &ca) in sa.exps.iter().zip(&sa.coefs) {
                    for (&b, &cb) in sb.exps.iter().zip(&sb.coefs) {
                        if self.prim_pair_setup(sa, sb, a, ca, b, cb, &mut scr).is_none() {
                            continue;
                        }
                        r_tensor += r_flops;
                        contraction += 2.0 * scr.coef_vals.len() as f64;
                        prim_pairs += 1.0;
                    }
                }
            }
        }
        Md3c1eFlops { r_tensor, contraction, prim_pairs }
    }
}

/// Build the A-matrix at one grid point with a pre-built kernel — the
/// analogue of `cosx_a::a_matrix_at_point_with` (engine → kernel + scratch).
/// Identical output and screening semantics to `cosx_a` at that point; this
/// path pays a full primitive-pair setup for a single (padded) tile, so it is
/// NOT where the speed is — use [`Md3c1e::for_each_pair`] for sweeps.
pub fn a_matrix_at_point_with(
    kern: &Md3c1e,
    scr: &mut Md3c1eScratch,
    r: &[f64; 3],
    bounds: Option<&PairBounds>,
    screen: CosxScreen,
) -> Result<CosxPoint, FerricError> {
    let batch = kern.a_matrices(std::slice::from_ref(r), bounds, screen, scr)?;
    let nbf = kern.nbf;
    let a: Array2<f64> = batch.a.into_shape_with_order((nbf, nbf)).map_err(|e| FerricError::General(format!("md3c1e: reshape: {e}")))?;
    Ok(CosxPoint { a, pairs_kept: batch.pairs_kept, pairs_total: batch.pairs_total })
}

/// Drop-in for `cosx_a::a_matrix_at_point`: same signature, same output
/// (`+1/|r-r_g|`, `(nbf, nbf)`), same screening rule. Builds a throwaway
/// [`Md3c1e`]; sweeps must build one kernel and use the batched API.
pub fn a_matrix_at_point(prep: &PreparedBasis, r: &[f64; 3], bounds: Option<&PairBounds>, screen: CosxScreen) -> Result<CosxPoint, FerricError> {
    let kern = Md3c1e::new(prep)?;
    let mut scr = kern.scratch();
    a_matrix_at_point_with(&kern, &mut scr, r, bounds, screen)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `F_n(T)`, `n = 0..=8`, from mpmath at 50 decimal digits via the closed
    /// form `F_n(T) = γ(n+½, T) / (2 T^{n+½})` (cross-checked against 50-dps
    /// quadrature to 20 digits at `n=7, T=80`). Generated 2026-09-07 with
    /// mpmath 1.3.0; scipy `quad` is NOT an adequate oracle here (spec §4).
    const BOYS_ORACLE: &[(f64, [f64; 9])] = &[
        (0.0, [1.0, 0.33333333333333333, 0.2, 0.14285714285714286, 0.11111111111111111, 0.090909090909090909, 0.076923076923076923, 0.066666666666666667, 0.058823529411764706]),
        (1e-14, [0.99999999999999667, 0.33333333333333133, 0.19999999999999857, 0.14285714285714175, 0.1111111111111102, 0.09090909090909014, 0.076923076923076256, 0.066666666666666078, 0.05882352941176418]),
        (1e-06, [0.99999966666676667, 0.33333313333340476, 0.1999998571429127, 0.1428570317460772, 0.11111102020205866, 0.090909013986047319, 0.076923010256439668, 0.066666607843163571, 0.058823476780209568]),
        (0.001, [0.99966676664286177, 0.33313340474339003, 0.1998571983975501, 0.14274607718775947, 0.11102024047063182, 0.090832201155699437, 0.076856439659405016, 0.066607869445109679, 0.058770921635096437]),
        (0.1, [0.96764331263559183, 0.31402947299816129, 0.18625500479262147, 0.13218802963573895, 0.10239394707106532, 0.083540528018141482, 0.070541950817983683, 0.061039712989141552, 0.053791384005818538]),
        (0.7, [0.80849580691258348, 0.22279321651512426, 0.1227102469671166, 0.083547093602981055, 0.063031679592469896, 0.050499866100585381, 0.042080873796449753, 0.036047182544598041, 0.031516024555400769]),
        (2.5, [0.54629197178514799, 0.092841394632249839, 0.039287837054570145, 0.022870837329790386, 0.015602172536926781, 0.011666910841688446, 0.0092502041269348226, 0.0076335310052507798, 0.0064835932909725804]),
        (7.0, [0.33490105817655928, 0.023856369729357483, 0.0050469448016084238, 0.0017373458601776859, 0.00080353850397780609, 0.00045142604073183847, 0.00028955746303540764, 0.00020374036099327022, 0.00015315881781032408]),
        (12.0, [0.25583143052938306, 0.010659386929876239, 0.0013321673573864745, 0.00027727885727412685, 8.0616991190231656e-5, 2.9975362848281529e-5, 1.3482699124073692e-5, 7.0471198441512411e-6, 4.1484410545391836e-6]),
        (20.0, [0.19816636482997365, 0.0049541590692205008, 0.000371561878662697, 4.6445183303996564e-5, 8.1278555493588377e-6, 1.8287159697651775e-6, 5.0284536284486285e-7, 1.6337321408401946e-7, 6.1213426440946335e-8]),
        (29.9, [0.16207250569912541, 0.0027102425702177592, 0.00013596534633026522, 1.136833999243994e-5, 1.3307421378538772e-6, 2.0027891533891585e-7, 3.6840601426594427e-8, 8.008824667692306e-9, 2.0089007792250321e-9]),
        (34.99, [0.14982109588297675, 0.0021409130592022878, 9.1779639577110979e-5, 6.5575621303932269e-6, 6.5594362549465186e-7, 8.4359711757859646e-8, 1.3260314785647437e-8, 2.4633336892907723e-9, 5.2800806948432636e-10]),
        (35.0, [0.14979969134027405, 0.0021399955905753346, 9.1714096738933905e-5, 6.5510069099148431e-6, 6.55100690982477e-7, 8.4227231688739733e-8, 1.3235707827794648e-8, 2.4580600161545536e-9, 5.2672713731152326e-10]),
        (35.01, [0.14977829596898158, 0.0021390787770491426, 9.164861941083696e-5, 6.5444601121616765e-6, 6.5425908004152385e-7, 8.4094997425728035e-8, 1.3211153542684525e-8, 2.4527991349708777e-9, 5.2544967723971991e-10]),
        (50.0, [0.12533141373155003, 0.0012533141373155003, 3.7599424119465008e-5, 1.8799712059732504e-6, 1.3159798441812752e-7, 1.1843818597631475e-8, 1.3028200457394603e-9, 1.6936660594612792e-10, 2.5404990891917259e-11]),
        (80.0, [0.099083182440150275, 0.00061926989025093922, 1.161131044220511e-5, 3.628534513189097e-7, 1.5874838495202299e-8, 8.9295966535512934e-10, 6.1390976993165142e-11, 4.9880168806946678e-12, 4.6762658256512511e-13]),
        (200.0, [0.062665706865775013, 0.00015666426716443753, 1.1749820037332815e-6, 1.4687275046666019e-8, 2.5702731331665532e-10, 5.7831145496247448e-12, 1.5903565011468048e-13, 5.1686586287271157e-15, 1.9382469857726684e-16]),
        (10000.0, [0.0088622692545275801, 4.4311346272637901e-7, 6.6467019408956851e-11, 1.6616754852239213e-14, 5.8158641982837245e-18, 2.617138889227676e-21, 1.4394263890752218e-24, 9.3562715289889417e-28, 7.0172036467417063e-31]),
    ];

    fn max_rel(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| ((x - y) / y).abs()).fold(0.0, f64::max)
    }

    #[test]
    fn boys_table_path_matches_mpmath_oracle() {
        let mut worst = 0.0_f64;
        for &(t, ref want) in BOYS_ORACLE {
            let mut got = [0.0; 9];
            boys(MAX_LTOT, t, &mut got);
            let r = max_rel(&got, want);
            worst = worst.max(r);
            assert!(r < 1e-15, "table Boys vs mpmath at T={t}: rel {r:.3e}");
        }
        println!("table Boys vs mpmath (n<=8): worst rel {worst:.3e}");
    }

    #[test]
    fn boys_direct_matches_mpmath_oracle() {
        for &(t, ref want) in BOYS_ORACLE {
            let mut got = [0.0; 9];
            boys_direct(MAX_LTOT, t, &mut got);
            let r = max_rel(&got, want);
            assert!(r < 1e-15, "direct Boys vs mpmath at T={t}: rel {r:.3e}");
        }
    }

    /// The table path against the spec's series evaluator on a dense sweep
    /// (two constructions of the same function), all orders, including
    /// tiny T and the region just below the switch.
    #[test]
    fn boys_table_matches_series_on_dense_sweep() {
        let mut worst = 0.0_f64;
        let mut ts: Vec<f64> = (0..2600).map(|i| i as f64 * 0.013457).collect();
        ts.extend([0.0, 1e-300, 1e-18, 1e-12, 1e-9, 3e-4, 34.999999, 34.9375, 35.0 - 1e-13]);
        for &t in &ts {
            let mut a = [0.0; 9];
            let mut b = [0.0; 9];
            boys(MAX_LTOT, t, &mut a);
            boys_direct(MAX_LTOT, t, &mut b);
            let r = max_rel(&a, &b);
            worst = worst.max(r);
            // Both sides are double precision; the SERIES side sums ~100
            // terms near T=30 and carries a few e-15 of rounding (measured
            // 2.6e-15 at T=25.9), so this cross-construction bar is 1e-14.
            // The absolute anchoring at 1e-15 is the mpmath oracle test.
            assert!(r < 1e-14, "table vs series at T={t}: rel {r:.3e}");
        }
        println!("table vs series sweep: worst rel {worst:.3e}");
    }

    /// Breakpoint guard (spec §4): the asymptotic `F_0` must be exact to
    /// double precision AT the switch. Uses the gate-free series as the
    /// reference so this catches a switch moved down to 25 (where the
    /// neglected erfc tail is 1.5e-12), which the integral anchor's 1e-12
    /// bar demonstrably cannot.
    #[test]
    fn boys_asymptotic_breakpoint_is_exact() {
        let mut ex = [0.0; 1];
        boys_series(0, BOYS_T_SWITCH, &mut ex);
        let asym = 0.5 * (std::f64::consts::PI / BOYS_T_SWITCH).sqrt();
        let rel = ((asym - ex[0]) / ex[0]).abs();
        assert!(rel < 1e-15, "asymptotic F_0 at the switch T={BOYS_T_SWITCH}: rel err {rel:.3e}");
        // Reachability: the same guard MUST fail at the common choice of 25,
        // otherwise it is arithmetic, not a measurement.
        boys_series(0, 25.0, &mut ex);
        let rel25 = ((0.5 * (std::f64::consts::PI / 25.0).sqrt() - ex[0]) / ex[0]).abs();
        assert!(rel25 > 1e-13, "guard is not discriminating: rel err at T=25 is only {rel25:.3e}");
    }

    #[test]
    fn hermite_canonical_index_is_a_prefix_ordering() {
        let h = hermite();
        assert_eq!(h.list.len(), N_TUV);
        for (k, &[t, u, v]) in h.list.iter().enumerate() {
            assert_eq!(h.index(t as usize, u as usize, v as usize), k);
            let s = (t + u + v) as usize;
            assert!(k < h.count_le[s]);
            assert!(s == 0 || k >= h.count_le[s - 1]);
        }
        assert_eq!(h.count_le[MAX_LTOT], N_TUV);
    }

    #[test]
    fn r_program_dependencies_are_ordered() {
        for (l, prog) in r_programs().iter().enumerate() {
            let mut ready = vec![false; prog.nvec];
            for n in 0..=l {
                ready[prog.base[n]] = true; // seeds
            }
            for st in &prog.steps {
                assert!(ready[st.src as usize], "L={l}: step reads unready tile");
                if st.src2 != NONE {
                    assert!(ready[st.src2 as usize], "L={l}: step reads unready tile");
                }
                assert!(!ready[st.dst as usize], "L={l}: tile written twice");
                ready[st.dst as usize] = true;
            }
            assert!(ready.iter().all(|&r| r), "L={l}: some tile never computed");
        }
    }

    #[test]
    fn e_table_recovers_gaussian_product_overlap() {
        // Σ_t-independent check: E_0^{ij} times sqrt(pi/p) is the 1-D overlap
        // <x^i_A | x^j_B>; test i=j=1 against the closed form.
        let (a, b, q) = (0.7_f64, 1.3_f64, 0.45_f64);
        let p = a + b;
        let mut e = vec![0.0; 2 * 2 * 3];
        e_table(1, 1, a, b, q, &mut e);
        let xpa = -(b / p) * q;
        let xpb = (a / p) * q;
        let kab = (-(a * b / p) * q * q).exp();
        // E_0^{11} = K (X_PA X_PB + 1/(2p))
        let want = kab * (xpa * xpb + 0.5 / p);
        let got = e[(2 + 1) * 3]; // (i*(lb+1) + j) * (la+lb+1), i=j=1
        assert!((got - want).abs() < 1e-15, "E_0^{{11}} {got} vs {want}");
    }

    #[test]
    fn ferric_cart2sph_is_unit_on_l00_component_scale() {
        // l=0: exactly 1; l=2, m=0 column: coefficient of xx is -1/2 (matches
        // ao_grid's S(2,0) = (2z² - x² - y²)/2).
        assert!((ferric_cart2sph(0)[0] - 1.0).abs() < 1e-15);
        let c2 = ferric_cart2sph(2);
        assert!((c2[2] + 0.5).abs() < 1e-14, "xx -> m=0 coefficient {}", c2[2]);
        assert!((c2[5 * 5 + 2] - 1.0).abs() < 1e-14, "zz -> m=0 coefficient {}", c2[5 * 5 + 2]);
    }
}
