//! The f32 block path of [`Md3c1e`](super::Md3c1e): a transcription of
//! `cart_block` with `f32` locals for the grid-dependent stages (Boys table
//! lookup, R-tensor recursion, E x R contraction), used for the Hölder-routed
//! COSX units whose K contribution is far below the fp64 threshold.
//!
//! # What is f32 and what is not
//!
//! * grid-independent primitive-pair setup (E-tables, `pref E E E`, `p`, `P`):
//!   f64 (shared with the f64 path), the coefficients rounded once to f32;
//! * `P - r_g` formed in f64 and rounded (a device kernel gets the same from
//!   coordinates relative to a local origin; the spike measured the naive
//!   f32 subtraction as a separate, worse case and it is not used);
//! * `T`, Boys (node table stored as f32 at construction, 10-term Taylor,
//!   downward recursion, `exp`), R recursion, contraction: f32;
//! * the sum ACROSS primitive pairs: [`PrimPairSum`];
//! * cart->sph and everything after: f64 (`cart_to_out`, the fold).
//!
//! If any accumulated Cartesian element is not finite (f32 range: the seed
//! `(2p)^n F_n` overflows for very tight primitives at high `l`) the whole
//! block is recomputed by the f64 path and reported as
//! [`BlockPrecision::F64Fallback`]; underflow is benign (absolute error far
//! below any bound here).
//!
//! # The a-priori error bound (Higham, ASNA 2nd ed. ch. 3-4)
//!
//! `pair_block_f32_bound` returns, per output element, a FIRST-ORDER
//! WORST-CASE bound `|dA| <= sum_pp B_pp + V` that assumes no underflow or
//! subnormals in the f32 chain (see "Assumptions"). With `u32 = 2^-24`,
//! `u64 = 2^-53`, `gamma_n(u) = n u / (1 - n u)` and, for one primitive pair,
//! `S_pp = sum_k |c_k| Rbar_k` (the same recursion run in f64 with absolute
//! values: `|pc|`, positive seeds `(2p)^n F_n`, `|c|`; because every R
//! recursion term is added with either sign, the error of `R_k` is bounded
//! relative to `Rbar_k`, not to `|R_k|`):
//!
//! ```text
//!   B_pp = e^{1/8} gamma_D(u32) S_pp + 2 gamma_D(u64) S_pp
//!   D    = nnz + 18 l_tot + 47          (nnz = terms of this Cartesian element)
//! ```
//!
//! `D` counts the rounding operations on the longest dependency chain, with
//! the non-FMA count (an FMA only removes roundings):
//!
//! * contraction `nnz + 1`: operand rounding of `c`, its product, `nnz - 1` adds;
//! * seed `fac_n = (-2 p)^n` times Boys: the ROUNDED `p` enters `n` times
//!   (`n` roundings' worth of relative error) plus `n - 1` products, plus the
//!   product with `F_n`: `2 l_tot + 1`;
//! * recursion: each level multiplies by the rounded `pc` (reused at every
//!   level, so its operand rounding is paid per level), the coefficient
//!   product and the add: `4 l_tot`;
//! * the argument `T = p |pc|^2` carries `7u` (pc rounded, squared, 2 adds, `p`
//!   rounded, times `p`) and `F_n(T)` amplifies a relative `T` error by
//!   `T F_{n+1} / F_n <= n + 1/2`: `7 l_tot + 4`;
//! * Boys `38 + 5 l_tot`: node rounded, 9 Horner steps of 4 roundings, one
//!   more for the factorial constant; 5 per downward or upward step including
//!   the f32 `exp`, assumed within 1 ulp = `2 u32`.
//!
//! Total `nnz + (2 + 4 + 7 + 5) l_tot + (1 + 1 + 4 + 38 + 3) = nnz + 18 l_tot
//! + 47` (the constant keeps 3 spare roundings for the `p` and `c` operands).
//! The Taylor terms sum to at most `e^{1/16} F` and the node value is within
//! `e^{1/16}` of `F(T)` (`F_{n+1} <= F_n`, `|delta| <= 1/16`), hence the
//! factor `e^{1/8}`. The `2 gamma_D(u64)` term covers the f64 reference and
//! the f64 stages. The sum across primitive pairs adds `V`, with `n_pp`
//! surviving primitive pairs and `A_pp` the per-pair block taken from the
//! f64 recursion (the kernel's own `A_pp` differ from it by the same first
//! order terms, so using the exact `|A_pp|` in `V` is a second-order
//! substitution):
//!
//! * `F64`: `gamma_{n_pp}(u64) sum |A_pp|` (f64 accumulator);
//! * `CompensatedF32`: `2 u32 sum |A_pp|` (the plan's Higham ASNA Thm 4.8
//!   form `(2u + O(n u^2)) sum |x|`; the `O(n_pp u^2)` term is DROPPED here
//!   (the realised second-order term is `n_pp (n_pp - 1) u32^2` because
//!   `hi + lo` is returned to f64 unrounded), so this is first order too);
//! * `F32`: `gamma_{n_pp - 1}(u32) sum |A_pp|` (recursive summation).
//!
//! The bound is then pushed through the cart->sph transform with `|C|`.
//!
//! # Assumptions
//!
//! First order in `u` (second-order terms are dropped throughout, `gamma`
//! keeps its denominator but `S_pp` and `A_pp` are taken from the exact f64
//! recursion rather than the perturbed one). No underflow or subnormals in
//! any f32 intermediate: an operation that underflows adds an absolute error
//! `<= 2^-150` that no relative bound covers; the guard test
//! (`tests/md3c1e_f32_error.rs::the_no_underflow_assumption_holds_on_the_probe_set`)
//! asserts every nonzero `S` element on the probe set is far above that
//! scale, and an overflowing block is not bounded at all but detected and
//! recomputed in f64 ([`BlockPrecision::F64Fallback`]). The bound does not
//! claim to be tight (see the measured margins in
//! `tests/md3c1e_f32_error.rs`).

use super::*;

/// How the sum across primitive pairs of one shell pair is carried.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PrimPairSum {
    /// f32 within a primitive pair, one f64 add per element per primitive
    /// pair (the spike's measured-safe variant).
    F64,
    /// f32 pairs `(hi, lo)` with Knuth TwoSum across primitive pairs; the
    /// result is `hi + lo` in f64 (the variant a device kernel wants).
    CompensatedF32,
    /// Plain f32 accumulator across primitive pairs, one f64 conversion per
    /// element per shell pair (Laqua/Kussmann/Ochsenfeld 2021 practice; depth
    /// = all primitive pairs). Not compensated.
    F32,
}

impl PrimPairSum {
    /// Every variant, in declaration order.
    pub const ALL: [PrimPairSum; 3] = [Self::F64, Self::CompensatedF32, Self::F32];

    /// Stable name for tables and messages.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::F64 => "F64",
            Self::CompensatedF32 => "CompensatedF32",
            Self::F32 => "F32",
        }
    }
}

/// Precision route of one (shell pair, sub-batch) unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairRoute {
    /// Not evaluated (the screen drops it).
    Drop,
    /// Kept and computed in f64.
    F64,
    /// Kept; computed by the f32 block ([`Md3c1e::pair_block_f32`]).
    F32,
}

/// What [`Md3c1e::pair_block_f32`] actually computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockPrecision {
    /// The f32 path ran and every accumulated element was finite.
    F32,
    /// A non-finite element appeared (f32 range); the block was recomputed in f64.
    F64Fallback,
}

/// Counts returned by [`Md3c1e::for_each_pair_routed`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RoutedCounts {
    /// Pairs evaluated (any route but `Drop`).
    pub kept: usize,
    /// Pairs considered.
    pub total: usize,
    /// Blocks computed in f32 (fallbacks excluded).
    pub f32_blocks: usize,
    /// `F32`-routed blocks recomputed in f64 after a non-finite element.
    pub f32_fallbacks: usize,
}

/// Defects injectable into the f32 path, for the mutation checks of the
/// error tests only.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum F32Fault {
    /// No defect (what [`Md3c1e::pair_block_f32`] runs).
    None,
    /// The E-coefficients are rounded toward zero instead of to nearest.
    TruncateE,
    /// The last primitive pair of the shell pair is skipped.
    DropLastPrimPair,
}

/// Per-thread f32 work buffers (a field of [`Md3c1eScratch`]).
#[derive(Default)]
pub(super) struct F32Scratch {
    r: Vec<f32>,
    boys: Vec<f32>,
    vals: Vec<f32>,
    hi: Vec<f32>,
    lo: Vec<f32>,
    wide: Vec<f64>,
}

impl F32Scratch {
    fn reset(&mut self, sum: PrimPairSum, ncomp: usize, nvec: usize, lt: usize) {
        self.r.resize(nvec * TILE, 0.0);
        self.boys.resize((lt + 1) * TILE, 0.0);
        self.hi.clear();
        self.lo.clear();
        self.wide.clear();
        match sum {
            PrimPairSum::F64 => self.wide.resize(ncomp, 0.0),
            PrimPairSum::F32 => self.hi.resize(ncomp, 0.0),
            PrimPairSum::CompensatedF32 => {
                self.hi.resize(ncomp, 0.0);
                self.lo.resize(ncomp, 0.0);
            }
        }
    }

    /// The f64 Cartesian block the accumulators represent.
    fn finish(&self, sum: PrimPairSum, cart: &mut Vec<f64>) {
        cart.clear();
        match sum {
            PrimPairSum::F64 => cart.extend_from_slice(&self.wide),
            PrimPairSum::F32 => cart.extend(self.hi.iter().map(|&v| v as f64)),
            PrimPairSum::CompensatedF32 => cart.extend(
                self.hi
                    .iter()
                    .zip(&self.lo)
                    .map(|(&h, &l)| h as f64 + l as f64),
            ),
        }
    }
}

/// `x` rounded toward zero to f32.
#[doc(hidden)]
pub fn trunc_f32(x: f64) -> f32 {
    let y = x as f32;
    if y.is_finite() && (y as f64).abs() > x.abs() {
        f32::from_bits(y.to_bits() - 1)
    } else {
        y
    }
}

#[inline(always)]
fn fma32<const FMA: bool>(a: f32, b: f32, c: f32) -> f32 {
    if FMA {
        a.mul_add(b, c)
    } else {
        a * b + c
    }
}

/// f32 transcription of `boys_tabulated` (node table pre-rounded to f32).
#[inline(always)]
fn boys32(tab: &[f32], nmax: usize, t: f32, out: &mut [f32], stride: usize) {
    let expt = (-t).exp();
    if t < BOYS_T_SWITCH as f32 {
        let k = (t * BOYS_INV_H as f32 + 0.5) as usize;
        let neg_delta = k as f32 * BOYS_H as f32 - t;
        let node = &tab[k * BOYS_STRIDE + nmax..k * BOYS_STRIDE + nmax + BOYS_TAYLOR_TERMS];
        let mut f = node[BOYS_TAYLOR_TERMS - 1] * INV_FACT[BOYS_TAYLOR_TERMS - 1] as f32;
        for j in (0..BOYS_TAYLOR_TERMS - 1).rev() {
            f = f * neg_delta + node[j] * INV_FACT[j] as f32;
        }
        out[nmax * stride] = f;
        let two_t = 2.0 * t;
        for n in (0..nmax).rev() {
            f = (two_t * f + expt) * INV_ODD[n] as f32;
            out[n * stride] = f;
        }
    } else {
        let mut f = 0.5 * (std::f32::consts::PI / t).sqrt();
        out[0] = f;
        let inv_2t = 0.5 / t;
        for n in 0..nmax {
            f = ((2 * n + 1) as f32 * f - expt) * inv_2t;
            out[(n + 1) * stride] = f;
        }
    }
}

/// One primitive pair's grid-independent f32 operands.
struct Pp32<'a> {
    prog: &'a RProgram,
    l: usize,
    p: f32,
    vals: &'a [f32],
    idx: &'a [u32],
    start: &'a [u32],
}

/// The accumulators a tile folds into.
struct Acc32<'a> {
    sum: PrimPairSum,
    hi: &'a mut [f32],
    lo: &'a mut [f32],
    wide: &'a mut [f64],
}

impl Acc32<'_> {
    #[inline(always)]
    fn add(&mut self, i: usize, v: f32) {
        match self.sum {
            PrimPairSum::F64 => self.wide[i] += v as f64,
            PrimPairSum::F32 => self.hi[i] += v,
            PrimPairSum::CompensatedF32 => {
                // Knuth TwoSum: s + e = hi + v exactly.
                let h = self.hi[i];
                let s = h + v;
                let bb = s - h;
                let e = (h - (s - bb)) + (v - bb);
                self.hi[i] = s;
                self.lo[i] += e;
            }
        }
    }
}

/// R recursion + contraction for one tile of one primitive pair, in f32.
#[inline(always)]
fn tile_f32<const FMA: bool>(
    pp: &Pp32<'_>,
    pc: &[[f32; TILE]; 3],
    boys: &[f32],
    r: &mut [f32],
    acc: &mut Acc32<'_>,
    tile_off: usize,
    gpad: usize,
) {
    let prog = pp.prog;
    let mut fac = 1.0_f32;
    for n in 0..=pp.l {
        for g in 0..TILE {
            r[prog.base[n] * TILE + g] = fac * boys[n * TILE + g];
        }
        fac *= -2.0 * pp.p;
    }
    for st in &prog.steps {
        let s = st.src as usize * TILE;
        let a: [f32; TILE] = r[s..s + TILE].try_into().expect("tile");
        let pcd = &pc[st.dir as usize];
        let d = st.dst as usize * TILE;
        if st.src2 == NONE {
            for g in 0..TILE {
                r[d + g] = pcd[g] * a[g];
            }
        } else {
            let s2 = st.src2 as usize * TILE;
            let a2: [f32; TILE] = r[s2..s2 + TILE].try_into().expect("tile");
            let c = st.coef as f32;
            for g in 0..TILE {
                r[d + g] = fma32::<FMA>(c, a2[g], pcd[g] * a[g]);
            }
        }
    }
    for mn in 0..pp.start.len() - 1 {
        let mut a = [0.0_f32; TILE];
        for k in pp.start[mn] as usize..pp.start[mn + 1] as usize {
            let c = pp.vals[k];
            let src = &r[pp.idx[k] as usize..pp.idx[k] as usize + TILE];
            for g in 0..TILE {
                a[g] = fma32::<FMA>(c, src[g], a[g]);
            }
        }
        for g in 0..TILE {
            acc.add(mn * gpad + tile_off + g, a[g]);
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn tile_f32_avx2(
    pp: &Pp32<'_>,
    pc: &[[f32; TILE]; 3],
    boys: &[f32],
    r: &mut [f32],
    acc: &mut Acc32<'_>,
    tile_off: usize,
    gpad: usize,
) {
    tile_f32::<true>(pp, pc, boys, r, acc, tile_off, gpad)
}

/// `gamma_n(u) = n u / (1 - n u)`.
fn gamma(n: usize, u: f64) -> f64 {
    let nu = n as f64 * u;
    nu / (1.0 - nu)
}

/// Unit roundoff of f32 (round to nearest): `2^-24`.
pub const U32: f64 = 1.0 / 16_777_216.0;
/// Unit roundoff of f64 (round to nearest): `2^-53`.
pub const U64: f64 = 1.0 / 9_007_199_254_740_992.0;

/// Rounding operations on the longest chain of one element (see the module
/// doc): `nnz + 18 l_tot + 47`.
#[doc(hidden)]
pub fn f32_chain_depth(l_tot: usize, nnz: usize) -> usize {
    nnz + 18 * l_tot + 47
}

impl Md3c1e {
    /// Validate a block call; `Ok(false)` for an empty batch.
    fn check_block(
        &self,
        what: &str,
        s1: usize,
        s2: usize,
        pts: &[[f64; 3]],
        out_len: usize,
    ) -> Result<bool, FerricError> {
        let nsh = self.shells.len();
        if s1 >= nsh || s2 >= nsh {
            return Err(FerricError::General(format!(
                "md3c1e::{what}: shell index out of range ({s1}, {s2}) for {nsh} shells"
            )));
        }
        if pts.is_empty() {
            return Ok(false);
        }
        let need = self.shells[s1].nfun * self.shells[s2].nfun * pts.len();
        if out_len < need {
            return Err(FerricError::General(format!(
                "md3c1e::{what}: out has {out_len} elements, need {need}"
            )));
        }
        Ok(true)
    }

    /// [`Md3c1e::pair_block`] with the grid-dependent stages in f32 and the
    /// primitive-pair sum carried per `sum`; same output layout and sign, `out`
    /// is f64. A non-finite accumulated element recomputes the block in f64
    /// (returned as [`BlockPrecision::F64Fallback`]).
    pub fn pair_block_f32(
        &self,
        s1: usize,
        s2: usize,
        pts: &[[f64; 3]],
        scr: &mut Md3c1eScratch,
        sum: PrimPairSum,
        out: &mut [f64],
    ) -> Result<BlockPrecision, FerricError> {
        self.pair_block_f32_with(s1, s2, pts, scr, sum, F32Fault::None, out)
    }

    /// [`Md3c1e::pair_block_f32`] with an injectable defect (mutation tests).
    #[doc(hidden)]
    pub fn pair_block_f32_with(
        &self,
        s1: usize,
        s2: usize,
        pts: &[[f64; 3]],
        scr: &mut Md3c1eScratch,
        sum: PrimPairSum,
        fault: F32Fault,
        out: &mut [f64],
    ) -> Result<BlockPrecision, FerricError> {
        if !self.check_block("pair_block_f32", s1, s2, pts, out.len())? {
            return Ok(BlockPrecision::F32);
        }
        let gpad = self.cart_block_f32(s1, s2, pts, scr, sum, fault);
        if scr.cart.iter().any(|v| !v.is_finite()) {
            self.pair_block(s1, s2, pts, scr, out)?;
            return Ok(BlockPrecision::F64Fallback);
        }
        self.cart_to_out(s1, s2, pts.len(), gpad, scr, out);
        Ok(BlockPrecision::F32)
    }

    /// `cart_block` in f32 (accumulators per `sum`), result in `scr.cart`.
    fn cart_block_f32(
        &self,
        s1: usize,
        s2: usize,
        pts: &[[f64; 3]],
        scr: &mut Md3c1eScratch,
        sum: PrimPairSum,
        fault: F32Fault,
    ) -> usize {
        let (sa, sb) = (&self.shells[s1], &self.shells[s2]);
        let ntile = pts.len().div_ceil(TILE);
        let gpad = ntile * TILE;
        let lt = sa.l + sb.l;
        let prog = &r_programs()[lt];
        scr.f32s
            .reset(sum, sa.ncart * sb.ncart * gpad, prog.nvec, lt);
        let last = (sa.exps.len() - 1, sb.exps.len() - 1);
        for (ia, (&a, &ca)) in sa.exps.iter().zip(&sa.coefs).enumerate() {
            for (ib, (&b, &cb)) in sb.exps.iter().zip(&sb.coefs).enumerate() {
                if fault == F32Fault::DropLastPrimPair && (ia, ib) == last {
                    continue;
                }
                let Some((p, pcen)) = self.prim_pair_setup(sa, sb, a, ca, b, cb, scr) else {
                    continue;
                };
                self.prim_pair_f32(prog, lt, p, pcen, pts, gpad, sum, fault, scr);
            }
        }
        scr.f32s.finish(sum, &mut scr.cart);
        gpad
    }

    /// One surviving primitive pair, every tile, folded into the accumulators.
    fn prim_pair_f32(
        &self,
        prog: &RProgram,
        lt: usize,
        p: f64,
        pcen: [f64; 3],
        pts: &[[f64; 3]],
        gpad: usize,
        sum: PrimPairSum,
        fault: F32Fault,
        scr: &mut Md3c1eScratch,
    ) {
        let Md3c1eScratch {
            coef_vals,
            coef_idx,
            coef_start,
            f32s,
            ..
        } = scr;
        let F32Scratch {
            r,
            boys,
            vals,
            hi,
            lo,
            wide,
        } = f32s;
        vals.clear();
        vals.extend(coef_vals.iter().map(|&c| {
            if fault == F32Fault::TruncateE {
                trunc_f32(c)
            } else {
                c as f32
            }
        }));
        let pp = Pp32 {
            prog,
            l: lt,
            p: p as f32,
            vals,
            idx: coef_idx,
            start: coef_start,
        };
        let mut acc = Acc32 { sum, hi, lo, wide };
        let npts = pts.len();
        for tile in 0..gpad / TILE {
            let mut pc = [[0.0_f32; TILE]; 3];
            let mut t = [0.0_f32; TILE];
            for g in 0..TILE {
                let rg = pts[(tile * TILE + g).min(npts - 1)];
                for d in 0..3 {
                    pc[d][g] = (pcen[d] - rg[d]) as f32;
                }
                t[g] = pp.p * (pc[0][g] * pc[0][g] + pc[1][g] * pc[1][g] + pc[2][g] * pc[2][g]);
            }
            for g in 0..TILE {
                boys32(&self.boys32, lt, t[g], &mut boys[g..], TILE);
            }
            let off = tile * TILE;
            if self.use_fma {
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                // SAFETY: `use_fma` is only true when `detect_fma()` confirmed
                // AVX2 and FMA at runtime.
                unsafe {
                    tile_f32_avx2(&pp, &pc, boys, r, &mut acc, off, gpad);
                }
            } else {
                tile_f32::<false>(&pp, &pc, boys, r, &mut acc, off, gpad);
            }
        }
    }

    /// Sweep like [`Md3c1e::for_each_pair_where`], with a per-pair precision
    /// route: `Drop` skips, `F64` runs `pair_block`, `F32` runs
    /// [`Md3c1e::pair_block_f32`] with `sum`. The sweep order is the fixed
    /// `s1` outer / `s2` inner order, so an all-`F64` route reproduces the
    /// unrouted sweep bit-for-bit.
    pub fn for_each_pair_routed<K, F>(
        &self,
        pts: &[[f64; 3]],
        mut route: K,
        sum: PrimPairSum,
        scr: &mut Md3c1eScratch,
        mut f: F,
    ) -> Result<RoutedCounts, FerricError>
    where
        K: FnMut(usize, usize) -> PairRoute,
        F: FnMut(usize, usize, &[f64]),
    {
        let nsh = self.shells.len();
        let mut counts = RoutedCounts::default();
        let mut block = std::mem::take(&mut scr.block);
        let result = (|| {
            for s1 in 0..nsh {
                for s2 in 0..=s1 {
                    counts.total += 1;
                    let r = route(s1, s2);
                    if r == PairRoute::Drop {
                        continue;
                    }
                    counts.kept += 1;
                    let need = self.shells[s1].nfun * self.shells[s2].nfun * pts.len();
                    block.resize(need, 0.0);
                    if r == PairRoute::F32 {
                        match self.pair_block_f32(s1, s2, pts, scr, sum, &mut block[..need])? {
                            BlockPrecision::F32 => counts.f32_blocks += 1,
                            BlockPrecision::F64Fallback => counts.f32_fallbacks += 1,
                        }
                    } else {
                        self.pair_block(s1, s2, pts, scr, &mut block[..need])?;
                    }
                    f(s1, s2, &block[..need]);
                }
            }
            Ok(())
        })();
        scr.block = block;
        result.map(|()| counts)
    }

    /// The a-priori bound of the module doc, per output element of the block
    /// (`bound`, [`Md3c1e::pair_block`] layout) and the block-level
    /// `sum_pp S_pp` pushed through `|C|` (`sabs`, same layout), so
    /// `kappa = sabs / |A|`. `sum` selects the primitive-pair term `V`.
    #[doc(hidden)]
    pub fn pair_block_f32_bound(
        &self,
        s1: usize,
        s2: usize,
        pts: &[[f64; 3]],
        scr: &mut Md3c1eScratch,
        sum: PrimPairSum,
        bound: &mut [f64],
        sabs: &mut [f64],
    ) -> Result<(), FerricError> {
        if !self.check_block(
            "pair_block_f32_bound",
            s1,
            s2,
            pts,
            bound.len().min(sabs.len()),
        )? {
            return Ok(());
        }
        let (sa, sb) = (&self.shells[s1], &self.shells[s2]);
        let gpad = pts.len().div_ceil(TILE) * TILE;
        let lt = sa.l + sb.l;
        let ncomp = sa.ncart * sb.ncart * gpad;
        let mut tot = BoundAcc::new(ncomp);
        for (&a, &ca) in sa.exps.iter().zip(&sa.coefs) {
            for (&b, &cb) in sb.exps.iter().zip(&sb.coefs) {
                let Some((p, pcen)) = self.prim_pair_setup(sa, sb, a, ca, b, cb, scr) else {
                    continue;
                };
                self.bound_prim_pair(&mut tot, lt, p, pcen, pts, gpad, scr);
            }
        }
        let vterm = tot.finish(sum);
        self.abs_to_out(s1, s2, pts.len(), gpad, &vterm, bound);
        self.abs_to_out(s1, s2, pts.len(), gpad, &tot.sabs, sabs);
        Ok(())
    }

    /// Add one primitive pair's `B_pp`, `S_pp` and `|A_pp|` to `tot`.
    fn bound_prim_pair(
        &self,
        tot: &mut BoundAcc,
        lt: usize,
        p: f64,
        pcen: [f64; 3],
        pts: &[[f64; 3]],
        gpad: usize,
        scr: &mut Md3c1eScratch,
    ) {
        let prog = &r_programs()[lt];
        let tab = boys_table();
        let ncomp = tot.sabs.len();
        scr.r.resize(prog.nvec * TILE, 0.0);
        scr.boys.resize((lt + 1) * TILE, 0.0);
        let mut alt = vec![0.0_f64; (lt + 1) * TILE];
        let vals_abs: Vec<f64> = scr.coef_vals.iter().map(|v| v.abs()).collect();
        let mut signed = vec![0.0_f64; ncomp];
        let mut absval = vec![0.0_f64; ncomp];
        let npts = pts.len();
        for tile in 0..gpad / TILE {
            let mut pc = [[0.0_f64; TILE]; 3];
            let mut pca = [[0.0_f64; TILE]; 3];
            let mut t = [0.0_f64; TILE];
            for g in 0..TILE {
                let rg = pts[(tile * TILE + g).min(npts - 1)];
                for d in 0..3 {
                    pc[d][g] = pcen[d] - rg[d];
                    pca[d][g] = pc[d][g].abs();
                }
                t[g] = p * (pc[0][g] * pc[0][g] + pc[1][g] * pc[1][g] + pc[2][g] * pc[2][g]);
            }
            for g in 0..TILE {
                boys_tabulated(tab, lt, t[g], &mut scr.boys[g..], TILE);
            }
            for n in 0..=lt {
                let sgn = if n % 2 == 0 { 1.0 } else { -1.0 };
                for g in 0..TILE {
                    alt[n * TILE + g] = sgn * scr.boys[n * TILE + g];
                }
            }
            let off = tile * TILE;
            let Md3c1eScratch {
                r,
                boys,
                coef_vals,
                coef_idx,
                coef_start,
                ..
            } = scr;
            tile_kernel::<false>(
                prog,
                lt,
                p,
                &pc,
                boys,
                r,
                coef_vals,
                coef_idx,
                coef_start,
                &mut signed,
                off,
                gpad,
            );
            tile_kernel::<false>(
                prog,
                lt,
                p,
                &pca,
                &alt,
                r,
                &vals_abs,
                coef_idx,
                coef_start,
                &mut absval,
                off,
                gpad,
            );
        }
        let e18 = (1.0_f64 / 8.0).exp();
        for mn in 0..scr.coef_start.len() - 1 {
            let nnz = (scr.coef_start[mn + 1] - scr.coef_start[mn]) as usize;
            let d = f32_chain_depth(lt, nnz);
            let gam = e18 * gamma(d, U32) + 2.0 * gamma(d, U64);
            for g in 0..gpad {
                let i = mn * gpad + g;
                tot.b[i] += gam * absval[i];
                tot.sabs[i] += absval[i];
                tot.asum[i] += signed[i].abs();
            }
        }
        tot.npp += 1;
    }

    /// `out[(i nf2 + j) npts + g] = sum |C_a| |C_b| src[(m ncb + n) gpad + g]`.
    fn abs_to_out(
        &self,
        s1: usize,
        s2: usize,
        npts: usize,
        gpad: usize,
        src: &[f64],
        out: &mut [f64],
    ) {
        let (sa, sb) = (&self.shells[s1], &self.shells[s2]);
        let coef = |s: &ShellData, m: usize, i: usize| -> f64 {
            if s.pure {
                self.c2s[s.l][m * s.nfun + i].abs()
            } else {
                (m == i) as usize as f64
            }
        };
        for i in 0..sa.nfun {
            for j in 0..sb.nfun {
                let dst = &mut out[(i * sb.nfun + j) * npts..(i * sb.nfun + j + 1) * npts];
                dst.iter_mut().for_each(|d| *d = 0.0);
                for m in 0..sa.ncart {
                    let ca = coef(sa, m, i);
                    if ca == 0.0 {
                        continue;
                    }
                    for n in 0..sb.ncart {
                        let c = ca * coef(sb, n, j);
                        if c == 0.0 {
                            continue;
                        }
                        let s = &src[(m * sb.ncart + n) * gpad..(m * sb.ncart + n) * gpad + npts];
                        for (d, v) in dst.iter_mut().zip(s) {
                            *d += c * v;
                        }
                    }
                }
            }
        }
    }
}

/// Cartesian-level accumulators of the bound oracle.
struct BoundAcc {
    /// `sum_pp B_pp`.
    b: Vec<f64>,
    /// `sum_pp S_pp`.
    sabs: Vec<f64>,
    /// `sum_pp |A_pp|`.
    asum: Vec<f64>,
    npp: usize,
}

impl BoundAcc {
    fn new(n: usize) -> Self {
        Self {
            b: vec![0.0; n],
            sabs: vec![0.0; n],
            asum: vec![0.0; n],
            npp: 0,
        }
    }

    /// `b + V(sum)` per element.
    fn finish(&self, sum: PrimPairSum) -> Vec<f64> {
        let n = self.npp;
        let v = match sum {
            PrimPairSum::F64 => gamma(n, U64),
            PrimPairSum::CompensatedF32 => 2.0 * U32,
            PrimPairSum::F32 => gamma(n.saturating_sub(1), U32),
        };
        self.b
            .iter()
            .zip(&self.asum)
            .map(|(b, a)| b + v * a)
            .collect()
    }
}
