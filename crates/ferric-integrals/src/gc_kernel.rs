//! Shared-primitive Coulomb kernel for the low-`l` classes of generally
//! contracted groups (`[scf] gc_kernel`).
//!
//! # What it computes
//!
//! A *group* is a set of shells on one atom with one `l` whose nonzero
//! exponent sets intersect (joined transitively): after the contraction
//! rotation ([`crate::contraction_rotation`]) cc-pVDZ carbon's core and
//! valence s columns are one group of 2 columns over 8 union primitives, its
//! diffuse s column another. libint computes every column (shell) quartet
//! from its own primitives, so the primitives the columns of a group share
//! are recomputed once per column. This kernel evaluates a whole *group
//! quartet* from one pass over the union primitives and contracts it into
//! every column quartet at once.
//!
//! It covers the classes with `l1 + l2 + l3 + l4 <= 1` only: `(ss|ss)` and
//! `(ss|sp)` with the p in any position. Those are where libint is per-call
//! bound and sharing pays (measured 3.2× and 2.3× against libint per group
//! quartet on octane cc-pVDZ); for higher `l` the hand-written recursion is
//! slower than libint, so every other class stays on libint.
//!
//! * **Operator:** the Coulomb operator only. A caller must not use it for
//!   any other operator (range-separated, Yukawa, geminal, ...).
//! * **Normalisation:** each column's coefficients are normalised exactly as
//!   libint normalises a shell (the primitive norm folded in, the
//!   contraction rescaled to unit self-overlap), then scattered into the
//!   group's `n_union × m_columns` matrix with zeros where a column lacks a
//!   primitive.
//! * **Primitive pairs:** per ordered group pair, libint's `Original`
//!   primitive-pair screen (`ln max|c_i| + ln max|c_j| − μ r² < ln ε`, the
//!   maxima over the group's columns) at the caller's engine precision; a
//!   same-group s pair folds `(j, i)` into `(i, j)`.
//! * **`(ss|ss)`:** one `F_0` per primitive quartet, contracted into every
//!   ket column pair, then every bra column pair.
//! * **`(ss|sp)` classes:** the Obara–Saika VRR `[e0|f0]` per primitive
//!   quartet (`l_ab + l_cd = 1`), the same contraction, then the
//!   Head-Gordon–Pople HRR once per contracted column quartet.
//! * **Boys function:** a 7th-order Taylor table on a 0.1 grid (truncation
//!   ≤ 6e-17), the asymptotic form from `T = 40`.
//!
//! Accuracy against libint (rotated cc-pVDZ / aug-cc-pVDZ / cc-pVTZ shells):
//! ≤ 1e-14 of the class's largest integral; the relative error of an
//! individual tiny integral can be larger because libint drops primitive
//! quartets below its own `ln_scr` cut, which this kernel keeps.
//!
//! # Layout
//!
//! [`eval`](crate::gc_kernel::GcKernelBasis::eval) writes a group quartet
//! as `out[(ab · nf_ab + f_ab) · (m_cd · nf_cd) + cd · nf_cd + f_cd]`, with
//! `ab = c_a · m_b + c_b` the bra column pair, `f_ab = f_a · n_b + f_b` its
//! Cartesian function pair (p as x, y, z), and the same on the ket.
//! [`gather`](crate::gc_kernel::GcKernelBasis::gather) copies one column
//! quartet out of it in libint's layout and function order (pure p as y, z,
//! x).
//!
//! The basis loaders make every p shell Cartesian (`pure = false`, libint
//! order x, y, z); the pure-p order is supported for hand-built bases and
//! tested with cc-pVDZ's p shells forced pure.

use crate::basis_bridge::PreparedBasis;
use crate::contraction_rotation::RotatedBasis;
use ferric_core::FerricError;
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// Boys function
// ---------------------------------------------------------------------------

/// Grid step of the Boys table.
const BOYS_H: f64 = 0.1;
/// From here on `F_m` is its asymptotic form (`e^{-T}` below 5e-18).
const BOYS_TMAX: f64 = 40.0;
/// Highest order the kernels need (`l1 + l2 + l3 + l4 <= 1`).
const BOYS_MAX_M: usize = 1;
/// Stored orders per grid point: `F_0 .. F_{MAX_M + 7}` (the Taylor series of
/// `F_m` to 7th order reads `F_m .. F_{m+7}`).
const BOYS_NTAB: usize = BOYS_MAX_M + 8;

/// `F_m(T)` per grid point `T_k = k · BOYS_H`.
struct BoysTable {
    tab: Vec<[f64; BOYS_NTAB]>,
}

impl BoysTable {
    fn new() -> Self {
        let npt = (BOYS_TMAX / BOYS_H) as usize + 2;
        let tab = (0..npt)
            .map(|k| {
                let t = k as f64 * BOYS_H;
                let mtop = BOYS_NTAB - 1;
                // The series of F_mtop, then the (stable) downward recursion.
                let mut term = 1.0 / (2 * mtop + 1) as f64;
                let mut sum = term;
                let mut i = 1;
                while term > 1e-18 * sum || i < 5 {
                    term *= 2.0 * t / (2 * mtop + 2 * i + 1) as f64;
                    sum += term;
                    i += 1;
                    if i > 4000 {
                        break;
                    }
                }
                let et = (-t).exp();
                let mut f = [0.0; BOYS_NTAB];
                f[mtop] = et * sum;
                for m in (0..mtop).rev() {
                    f[m] = (2.0 * t * f[m + 1] + et) / (2 * m + 1) as f64;
                }
                f
            })
            .collect();
        BoysTable { tab }
    }

    /// The process-wide table (built once, ≈ 29 kB).
    fn get() -> &'static BoysTable {
        static TABLE: OnceLock<BoysTable> = OnceLock::new();
        TABLE.get_or_init(BoysTable::new)
    }

    /// `F_0(t)`.
    #[inline(always)]
    fn f0(&self, t: f64) -> f64 {
        if t >= BOYS_TMAX {
            return 0.5 * (std::f64::consts::PI / t).sqrt();
        }
        let k = (t * (1.0 / BOYS_H) + 0.5) as usize;
        let d = t - k as f64 * BOYS_H;
        taylor7(&self.tab[k][..8], -d)
    }

    /// `F_0(t)` and `F_1(t)`.
    #[inline(always)]
    fn f01(&self, t: f64) -> [f64; 2] {
        if t >= BOYS_TMAX {
            let f0 = 0.5 * (std::f64::consts::PI / t).sqrt();
            return [f0, f0 * (0.5 / t)];
        }
        let k = (t * (1.0 / BOYS_H) + 0.5) as usize;
        let d = t - k as f64 * BOYS_H;
        let f = &self.tab[k];
        [taylor7(&f[..8], -d), taylor7(&f[1..9], -d)]
    }
}

/// `Σ_{n=0}^{7} g[n] md^n / n!` (Horner).
#[inline(always)]
fn taylor7(g: &[f64], md: f64) -> f64 {
    g[0] + md
        * (g[1]
            + md * (g[2] * 0.5
                + md * (g[3] * (1.0 / 6.0)
                    + md * (g[4] * (1.0 / 24.0)
                        + md * (g[5] * (1.0 / 120.0)
                            + md * (g[6] * (1.0 / 720.0) + md * g[7] * (1.0 / 5040.0)))))))
}

/// `F_0 .. F_mmax` at `t` from the kernel's table (`mmax <= 1`), for tests
/// against libint's own Boys evaluator.
#[doc(hidden)]
pub fn boys_for_tests(t: f64, mmax: usize) -> Vec<f64> {
    assert!(
        mmax <= BOYS_MAX_M,
        "gc_kernel Boys table: m <= {BOYS_MAX_M}"
    );
    let b = BoysTable::get();
    if mmax == 0 {
        vec![b.f0(t)]
    } else {
        b.f01(t).to_vec()
    }
}

// ---------------------------------------------------------------------------
// Groups and pair data
// ---------------------------------------------------------------------------

/// One evaluation group: its union primitives and the libint-normalised
/// coefficients of its columns.
#[derive(Debug, Clone)]
pub struct KernelGroup {
    /// Angular momentum of every column.
    pub l: i32,
    /// Spherical (`true`) or Cartesian columns (all alike).
    pub pure: bool,
    /// Centre (Bohr).
    pub center: [f64; 3],
    /// Union exponents (first appearance order over the columns).
    pub exponents: Vec<f64>,
    /// `exponents.len() × shells.len()` row-major: the libint-normalised
    /// coefficient of union primitive `i` in column `c`, zero where the
    /// column lacks it.
    pub coefficients: Vec<f64>,
    /// The columns: shell indices of the basis, ascending.
    pub shells: Vec<usize>,
}

impl KernelGroup {
    /// Number of columns.
    pub fn ncols(&self) -> usize {
        self.shells.len()
    }
}

fn double_factorial(n: i32) -> f64 {
    let mut r = 1.0;
    let mut k = n;
    while k > 1 {
        r *= k as f64;
        k -= 2;
    }
    r
}

/// libint's coefficient handling of one shell: fold the primitive norm in,
/// then rescale the contraction to unit self-overlap.
fn libint_coefficients(l: i32, exps: &[f64], c: &[f64]) -> Vec<f64> {
    use std::f64::consts::PI;
    let lf = l as f64;
    let df = double_factorial(2 * l - 1);
    let mut v: Vec<f64> = exps
        .iter()
        .zip(c)
        .map(|(&a, &c)| {
            let two_a = 2.0 * a;
            let n = (2f64.powf(lf) * two_a.powf(lf + 1.5) / (PI.powf(1.5) * df)).sqrt();
            c * n
        })
        .collect();
    let mut s = 0.0;
    for (i, &a) in exps.iter().enumerate() {
        for (j, &b) in exps.iter().enumerate() {
            s += v[i] * v[j] * df * PI.powf(1.5) / (2f64.powf(lf) * (a + b).powf(lf + 1.5));
        }
    }
    let f = 1.0 / s.sqrt();
    for x in &mut v {
        *x *= f;
    }
    v
}

/// Number of Cartesian functions with total `l <= L`.
const NCUM: [usize; 3] = [1, 4, 10];
/// `INC[e][i]`: the index of `e + 1_i` (`e` of `l <= 1`).
const INC: [[usize; 3]; 4] = [[1, 2, 3], [4, 5, 6], [5, 7, 8], [6, 8, 9]];

/// One HRR output function: up to two `(e index, coefficient)` terms.
type HrrTerm = [(usize, f64); 2];

/// HRR terms of one side (`(a|b+1_i) = (a+1_i|b) + AB_i (a|b)`): per output
/// function (`a`-major over the two shells' Cartesian functions), up to two
/// terms. `l_a + l_b <= 1`.
fn hrr_terms(la: i32, lb: i32, ab: [f64; 3]) -> Vec<HrrTerm> {
    let fa: &[usize] = if la == 0 { &[0] } else { &[1, 2, 3] };
    let mut t = Vec::new();
    for &a in fa {
        if lb == 0 {
            t.push([(a, 1.0), (0, 0.0)]);
        } else {
            for (i, &up) in INC[a].iter().enumerate() {
                t.push([(up, 1.0), (a, ab[i])]);
            }
        }
    }
    t
}

/// Primitive-pair data of one ordered group pair.
#[derive(Debug)]
struct PairData {
    la: i32,
    lb: i32,
    /// Columns of the two groups.
    ma: usize,
    mb: usize,
    /// Surviving primitive pairs.
    n: usize,
    p: Vec<f64>,
    pc: Vec<[f64; 3]>,
    /// `P − A`.
    pa: Vec<[f64; 3]>,
    /// `√2 π^{5/4} exp(−μ r²) / p`.
    u: Vec<f64>,
    /// `n × (ma · mb)` coefficient products (a same-group s pair has
    /// `(j, i)` folded into `(i, j)`).
    cw: Vec<f64>,
    hrr: Vec<HrrTerm>,
}

impl PairData {
    fn new(ga: &KernelGroup, gb: &KernelGroup, same: bool, ln_eps: f64) -> Self {
        let (ma, mb) = (ga.ncols(), gb.ncols());
        let (a, b) = (ga.center, gb.center);
        let ab = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
        let r2 = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
        // The (j, i) fold needs both indices to be s.
        let fold = same && ga.l == 0;
        let cmax = |g: &KernelGroup, i: usize| {
            let m = g.ncols();
            (0..m)
                .map(|c| g.coefficients[i * m + c].abs())
                .fold(0.0, f64::max)
        };
        let ca: Vec<f64> = (0..ga.exponents.len()).map(|i| cmax(ga, i)).collect();
        let cb: Vec<f64> = (0..gb.exponents.len()).map(|i| cmax(gb, i)).collect();
        let mut pr = PairData {
            la: ga.l,
            lb: gb.l,
            ma,
            mb,
            n: 0,
            p: vec![],
            pc: vec![],
            pa: vec![],
            u: vec![],
            cw: vec![],
            hrr: hrr_terms(ga.l, gb.l, ab),
        };
        let pref = 2f64.sqrt() * std::f64::consts::PI.powf(1.25);
        for i in 0..ga.exponents.len() {
            let j0 = if fold { i } else { 0 };
            for j in j0..gb.exponents.len() {
                let (ea, eb) = (ga.exponents[i], gb.exponents[j]);
                let p = ea + eb;
                let mu = ea * eb / p;
                if ca[i].ln() + cb[j].ln() - mu * r2 < ln_eps {
                    continue;
                }
                let pc = [
                    (ea * a[0] + eb * b[0]) / p,
                    (ea * a[1] + eb * b[1]) / p,
                    (ea * a[2] + eb * b[2]) / p,
                ];
                pr.p.push(p);
                pr.pc.push(pc);
                pr.pa.push([pc[0] - a[0], pc[1] - a[1], pc[2] - a[2]]);
                pr.u.push(pref * (-mu * r2).exp() / p);
                for c1 in 0..ma {
                    for c2 in 0..mb {
                        let mut w = ga.coefficients[i * ma + c1] * gb.coefficients[j * mb + c2];
                        if fold && i != j {
                            w += ga.coefficients[j * ma + c1] * gb.coefficients[i * mb + c2];
                        }
                        pr.cw.push(w);
                    }
                }
                pr.n += 1;
            }
        }
        pr
    }

    /// Column pairs.
    fn mpair(&self) -> usize {
        self.ma * self.mb
    }
}

/// An ordered group pair the kernel has data for (`l_a + l_b <= 1`, at
/// least one primitive pair surviving the screen).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PairId(u32);

impl PairId {
    /// A dense index (`0..GcKernelBasis::n_pairs()`), e.g. a cache key.
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// What the kernel knows about an ordered group pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairKind {
    /// `l_a + l_b > 1`: not covered, use libint.
    NotCovered,
    /// Covered, and every primitive pair is screened out: every integral of
    /// it is (numerically) zero.
    Empty,
    /// Covered.
    Data(PairId),
}

/// Per shell: its group and column.
#[derive(Debug, Clone, Copy)]
struct Slot {
    group: u32,
    col: u32,
}

/// libint's pure p order (y, z, x) as Cartesian indices, and the Cartesian
/// order (x, y, z).
const P_PURE: u8 = 0;
const P_CART: u8 = 1;
const P_MAP: [[usize; 3]; 2] = [[1, 2, 0], [0, 1, 2]];

const NOT_COVERED: u32 = u32::MAX;
const EMPTY: u32 = u32::MAX - 1;

/// The shared-primitive groups of a basis and the pair data of every
/// ordered group pair with `l_a + l_b <= 1`, built once per geometry.
#[derive(Debug)]
pub struct GcKernelBasis {
    groups: Vec<KernelGroup>,
    slot: Vec<Slot>,
    /// Per shell: its `l`.
    shell_l: Vec<i32>,
    /// Per shell: its group's index among the `l <= 1` groups (`u32::MAX`
    /// for higher `l`), its column, and its libint function order
    /// ([`P_PURE`] / [`P_CART`] for p, unused for s).
    shell_sp: Vec<u32>,
    shell_col: Vec<u32>,
    shell_pmap: Vec<u8>,
    /// Number of groups with `l <= 1`.
    nsp: usize,
    /// `nsp × nsp`: index into `pairs`, or [`NOT_COVERED`] / [`EMPTY`].
    pidx: Vec<u32>,
    pairs: Vec<PairData>,
    nbasis: usize,
    precision: f64,
}

/// Scratch of [`GcKernelBasis::eval`] (one per worker).
#[derive(Debug, Default)]
pub struct KernelScratch {
    a: Vec<f64>,
    b: Vec<f64>,
}

impl GcKernelBasis {
    /// The groups of a contraction-rotated basis (its rotation groups, split
    /// into runs of columns whose nonzero exponents intersect), on
    /// `rot.prep`'s shells, with primitive pairs screened at `precision`
    /// (the engine precision of the walk that uses it).
    pub fn for_rotated(rot: &RotatedBasis, precision: f64) -> Result<Self, FerricError> {
        Self::from_groups(&rot.prep, rot.groups(), precision)
    }

    /// The groups of `prep` from a partition `parts` of its shells (each part
    /// on one atom with one `l` and one `pure`), each part split into runs
    /// of shells whose nonzero exponent sets intersect. An error when `parts`
    /// is not a partition or a part mixes atoms, `l` or `pure`.
    pub fn from_groups(
        prep: &PreparedBasis,
        parts: &[Vec<usize>],
        precision: f64,
    ) -> Result<Self, FerricError> {
        let ls = prep.located_shells();
        let s2a = prep.shell_to_atom();
        let nsh = ls.len();
        let err = |m: String| Err(FerricError::General(format!("gc kernel: {m}")));
        let mut seen = vec![false; nsh];
        let nonzero_bits = |s: usize| -> Vec<u64> {
            ls[s]
                .exponents
                .iter()
                .zip(ls[s].coefficients)
                .filter(|(_, c)| **c != 0.0)
                .map(|(e, _)| e.to_bits())
                .collect()
        };
        let mut groups: Vec<KernelGroup> = Vec::new();
        for part in parts {
            let Some(&first) = part.first() else {
                continue;
            };
            for &s in part {
                if s >= nsh || seen[s] {
                    return err(format!("shell {s} is out of range or in two groups"));
                }
                seen[s] = true;
                if s2a[s] != s2a[first] || ls[s].l != ls[first].l || ls[s].pure != ls[first].pure {
                    return err(format!(
                        "shells {first} and {s} differ in atom, l or pure but share a group"
                    ));
                }
            }
            // Split into runs joined by a shared nonzero exponent
            // (union-find, smallest member as root).
            let nz: Vec<Vec<u64>> = part.iter().map(|&s| nonzero_bits(s)).collect();
            let mut root: Vec<usize> = (0..part.len()).collect();
            fn find(r: &mut [usize], x: usize) -> usize {
                let mut y = x;
                while r[y] != y {
                    y = r[y];
                }
                r[x] = y;
                y
            }
            for a in 0..part.len() {
                for b in 0..a {
                    if nz[a].iter().any(|x| nz[b].contains(x)) {
                        let (ra, rb) = (find(&mut root, a), find(&mut root, b));
                        root[ra.max(rb)] = ra.min(rb);
                    }
                }
            }
            let mut runs: Vec<(usize, Vec<usize>)> = Vec::new();
            for a in 0..part.len() {
                let r = find(&mut root, a);
                match runs.iter_mut().find(|(k, _)| *k == r) {
                    Some((_, v)) => v.push(part[a]),
                    None => runs.push((r, vec![part[a]])),
                }
            }
            for (_, mut shells) in runs {
                shells.sort_unstable();
                groups.push(Self::make_group(&ls, shells));
            }
        }
        if let Some(s) = seen.iter().position(|x| !x) {
            return err(format!("shell {s} is in no group"));
        }
        Ok(Self::assemble(prep, groups, precision))
    }

    fn make_group(ls: &[crate::ao_grid::LocatedShell<'_>], shells: Vec<usize>) -> KernelGroup {
        let mut exps: Vec<f64> = Vec::new();
        for &s in &shells {
            for (e, c) in ls[s].exponents.iter().zip(ls[s].coefficients) {
                if *c != 0.0 && !exps.iter().any(|x| x.to_bits() == e.to_bits()) {
                    exps.push(*e);
                }
            }
        }
        let (n, m) = (exps.len(), shells.len());
        let mut coef = vec![0.0; n * m];
        for (col, &s) in shells.iter().enumerate() {
            let (e, c): (Vec<f64>, Vec<f64>) = ls[s]
                .exponents
                .iter()
                .zip(ls[s].coefficients)
                .filter(|(_, c)| **c != 0.0)
                .map(|(e, c)| (*e, *c))
                .unzip();
            let cn = libint_coefficients(ls[s].l, &e, &c);
            for (ei, ci) in e.iter().zip(cn) {
                let i = exps
                    .iter()
                    .position(|x| x.to_bits() == ei.to_bits())
                    .expect("every nonzero exponent is in the union");
                // `+=`: a column may list one exponent twice; libint sums
                // the two primitives.
                coef[i * m + col] += ci;
            }
        }
        KernelGroup {
            l: ls[shells[0]].l,
            pure: ls[shells[0]].pure,
            center: ls[shells[0]].center,
            exponents: exps,
            coefficients: coef,
            shells,
        }
    }

    fn assemble(prep: &PreparedBasis, groups: Vec<KernelGroup>, precision: f64) -> Self {
        let nsh = prep.nshells();
        let mut slot = vec![Slot { group: 0, col: 0 }; nsh];
        for (g, gr) in groups.iter().enumerate() {
            for (c, &s) in gr.shells.iter().enumerate() {
                slot[s] = Slot {
                    group: g as u32,
                    col: c as u32,
                };
            }
        }
        let mut sp = vec![u32::MAX; groups.len()];
        let mut nsp = 0usize;
        for (g, gr) in groups.iter().enumerate() {
            if gr.l <= 1 {
                sp[g] = nsp as u32;
                nsp += 1;
            }
        }
        let sp_groups: Vec<usize> = (0..groups.len()).filter(|&g| groups[g].l <= 1).collect();
        let ln_eps = precision.ln();
        let mut pidx = vec![NOT_COVERED; nsp * nsp];
        let mut pairs = Vec::new();
        for (x, &gx) in sp_groups.iter().enumerate() {
            for (y, &gy) in sp_groups.iter().enumerate() {
                if groups[gx].l + groups[gy].l > 1 {
                    continue;
                }
                let pd = PairData::new(&groups[gx], &groups[gy], gx == gy, ln_eps);
                pidx[x * nsp + y] = if pd.n == 0 {
                    EMPTY
                } else {
                    pairs.push(pd);
                    (pairs.len() - 1) as u32
                };
            }
        }
        let shell_l = slot.iter().map(|x| groups[x.group as usize].l).collect();
        let shell_sp = slot.iter().map(|x| sp[x.group as usize]).collect();
        let shell_col = slot.iter().map(|x| x.col).collect();
        let shell_pmap = slot
            .iter()
            .map(|x| {
                if groups[x.group as usize].pure {
                    P_PURE
                } else {
                    P_CART
                }
            })
            .collect();
        GcKernelBasis {
            groups,
            slot,
            shell_l,
            shell_sp,
            shell_col,
            shell_pmap,
            nsp,
            pidx,
            pairs,
            nbasis: prep.nbasis(),
            precision,
        }
    }

    /// The groups.
    pub fn groups(&self) -> &[KernelGroup] {
        &self.groups
    }

    /// Group and column of shell `s`.
    pub fn slot(&self, s: usize) -> (usize, usize) {
        let x = self.slot[s];
        (x.group as usize, x.col as usize)
    }

    /// Angular momentum of shell `s`.
    #[inline(always)]
    pub fn shell_l(&self, s: usize) -> i32 {
        self.shell_l[s]
    }

    /// Number of pairs with data ([`PairId::index`] range).
    pub fn n_pairs(&self) -> usize {
        self.pairs.len()
    }

    /// Number of AOs of the basis it was built for.
    pub fn nbasis(&self) -> usize {
        self.nbasis
    }

    /// The primitive-pair screening precision it was built with.
    pub fn precision(&self) -> f64 {
        self.precision
    }

    /// The ordered group pair of shells `(s1, s2)` (shell order: the group
    /// of `s1` first).
    #[inline]
    pub fn pair_of(&self, s1: usize, s2: usize) -> PairKind {
        let (a, b) = (self.shell_sp[s1], self.shell_sp[s2]);
        if a == u32::MAX || b == u32::MAX {
            return PairKind::NotCovered;
        }
        match self.pidx[a as usize * self.nsp + b as usize] {
            NOT_COVERED => PairKind::NotCovered,
            EMPTY => PairKind::Empty,
            i => PairKind::Data(PairId(i)),
        }
    }

    /// Column quartets of the group quartet `(bra|ket)`.
    pub fn columns(&self, bra: PairId, ket: PairId) -> usize {
        self.pairs[bra.index()].mpair() * self.pairs[ket.index()].mpair()
    }

    /// Number of doubles [`eval`](Self::eval) writes for `(bra|ket)`.
    pub fn block_len(&self, bra: PairId, ket: PairId) -> usize {
        let (pb, pk) = (&self.pairs[bra.index()], &self.pairs[ket.index()]);
        pb.mpair() * pk.mpair() * pb.hrr.len() * pk.hrr.len()
    }

    /// Every column quartet of the group quartet `(bra|ket)` into
    /// `out[..block_len]` (module doc, "Layout"). The bra and ket must
    /// together hold at most one p (`l_a + l_b + l_c + l_d <= 1`).
    pub fn eval(&self, bra: PairId, ket: PairId, out: &mut [f64], scratch: &mut KernelScratch) {
        let boys = BoysTable::get();
        let (pb, pk) = (&self.pairs[bra.index()], &self.pairs[ket.index()]);
        let lab = (pb.la + pb.lb) as usize;
        let lcd = (pk.la + pk.lb) as usize;
        let len = self.block_len(bra, ket);
        assert!(out.len() >= len, "gc kernel: output block too short");
        match (lab, lcd) {
            (0, 0) => kernel_ssss(boys, pb, pk, &mut out[..len], scratch),
            (1, 0) => kernel_l1::<1, 0>(boys, pb, pk, &mut out[..len], scratch),
            (0, 1) => kernel_l1::<0, 1>(boys, pb, pk, &mut out[..len], scratch),
            _ => panic!("gc kernel: class (l_ab, l_cd) = ({lab}, {lcd}) is not covered"),
        }
    }

    /// The column quartet `(s1 s2|s3 s4)` of a block [`eval`](Self::eval)
    /// wrote for `(bra|ket)` = the group pairs of `(s1, s2)` and `(s3, s4)`,
    /// in libint's layout (`((a · n2 + b) · n3 + c) · n4 + d`) and function
    /// order (pure p as y, z, x). Returns the number of integrals.
    pub fn gather(
        &self,
        block: &[f64],
        bra: PairId,
        ket: PairId,
        s: [usize; 4],
        dst: &mut [f64],
    ) -> usize {
        let (pb, pk) = (&self.pairs[bra.index()], &self.pairs[ket.index()]);
        let col = |x: usize| self.shell_col[s[x]] as usize;
        let ab = col(0) * pb.mb + col(1);
        let cd = col(2) * pk.mb + col(3);
        let (nfab, nfcd) = (pb.hrr.len(), pk.hrr.len());
        let mcd = pk.mpair();
        let base = ab * nfab * mcd * nfcd + cd * nfcd;
        if nfab * nfcd == 1 {
            // (ss|ss): one integral.
            dst[0] = block[base];
            return 1;
        }
        // One p, at position `x`: its three functions in libint's order.
        let x = (0..4)
            .find(|&x| self.shell_l[s[x]] == 1)
            .expect("a covered quartet that is not (ss|ss) holds one p");
        let map = &P_MAP[self.shell_pmap[s[x]] as usize];
        // The p's stride in the block: on the bra, `f_ab` (its Cartesian
        // index, the other bra index being s) times the ket width; on the
        // ket, `f_cd` itself.
        let stride = if x < 2 { mcd * nfcd } else { 1 };
        for (d, &f) in dst[..3].iter_mut().zip(map) {
            *d = block[base + f * stride];
        }
        3
    }
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

/// `(ss|ss)` for every column quartet of a group quartet:
/// `out[ab · m_cd + cd]`.
#[inline(never)]
fn kernel_ssss(
    boys: &BoysTable,
    bra: &PairData,
    ket: &PairData,
    out: &mut [f64],
    scratch: &mut KernelScratch,
) {
    let mab = bra.mpair();
    let mcd = ket.mpair();
    out.fill(0.0);
    if scratch.a.len() < mcd {
        scratch.a.resize(mcd, 0.0);
    }
    let t = &mut scratch.a[..mcd];
    for ij in 0..bra.n {
        let p = bra.p[ij];
        let pc = bra.pc[ij];
        t.fill(0.0);
        for kl in 0..ket.n {
            let q = ket.p[kl];
            let qc = ket.pc[kl];
            let pq = p + q;
            let rho = p * q / pq;
            let dx = pc[0] - qc[0];
            let dy = pc[1] - qc[1];
            let dz = pc[2] - qc[2];
            let tt = rho * (dx * dx + dy * dy + dz * dz);
            let v = ket.u[kl] / pq.sqrt() * boys.f0(tt);
            let cw = &ket.cw[kl * mcd..(kl + 1) * mcd];
            for (x, w) in t.iter_mut().zip(cw) {
                *x += v * w;
            }
        }
        let ub = bra.u[ij];
        let cw = &bra.cw[ij * mab..(ij + 1) * mab];
        for (ab, w) in cw.iter().enumerate() {
            let w = ub * w;
            for (o, x) in out[ab * mcd..(ab + 1) * mcd].iter_mut().zip(t.iter()) {
                *o += w * x;
            }
        }
    }
}

/// Obara–Saika VRR of one primitive quartet with `l_ab + l_cd = 1`: the
/// `[e0|f0]` for `|e| <= LAB`, `|f| <= LCD` into `out[e · NF + f]`, from
/// `F_0`, `F_1` already scaled by the primitive prefactor.
#[inline(always)]
fn vrr1<const LAB: usize, const LCD: usize>(
    f: [f64; 2],
    pa: [f64; 3],
    wp: [f64; 3],
    qc: [f64; 3],
    wq: [f64; 3],
    out: &mut [f64; 4],
) {
    out[0] = f[0];
    if LAB == 1 {
        // [1_i 0|0 0] = PA_i F0 + WP_i F1.
        for i in 0..3 {
            out[1 + i] = pa[i] * f[0] + wp[i] * f[1];
        }
    } else if LCD == 1 {
        // [0 0|1_i 0] = QC_i F0 + WQ_i F1.
        for i in 0..3 {
            out[1 + i] = qc[i] * f[0] + wq[i] * f[1];
        }
    }
}

/// The `(ss|ss)`-plus-one-p kernel (`l_ab + l_cd = 1`): VRR per primitive
/// quartet, contraction into every column pair, HRR once per contracted
/// column quartet. Cartesian p in x, y, z order; layout of the module doc.
#[inline(never)]
fn kernel_l1<const LAB: usize, const LCD: usize>(
    boys: &BoysTable,
    bra: &PairData,
    ket: &PairData,
    out: &mut [f64],
    scratch: &mut KernelScratch,
) {
    let ne = NCUM[LAB];
    let nf = NCUM[LCD];
    let nef = ne * nf;
    let mab = bra.mpair();
    let mcd = ket.mpair();
    if scratch.a.len() < mcd * nef {
        scratch.a.resize(mcd * nef, 0.0);
    }
    if scratch.b.len() < mab * mcd * nef {
        scratch.b.resize(mab * mcd * nef, 0.0);
    }
    let h = &mut scratch.a[..mcd * nef];
    let acc = &mut scratch.b[..mab * mcd * nef];
    acc.fill(0.0);
    let mut v = [0.0f64; 4];
    for ij in 0..bra.n {
        let p = bra.p[ij];
        let pc = bra.pc[ij];
        let pa = bra.pa[ij];
        h.fill(0.0);
        for kl in 0..ket.n {
            let q = ket.p[kl];
            let qc = ket.pc[kl];
            let pq = p + q;
            let rho = p * q / pq;
            let d = [pc[0] - qc[0], pc[1] - qc[1], pc[2] - qc[2]];
            let tt = rho * (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
            let pre = ket.u[kl] / pq.sqrt();
            let f = boys.f01(tt);
            let f = [f[0] * pre, f[1] * pre];
            let a = q / pq;
            let b = p / pq;
            let wp = [-a * d[0], -a * d[1], -a * d[2]];
            let wq = [b * d[0], b * d[1], b * d[2]];
            vrr1::<LAB, LCD>(f, pa, wp, ket.pa[kl], wq, &mut v);
            let cw = &ket.cw[kl * mcd..(kl + 1) * mcd];
            for (cd, w) in cw.iter().enumerate() {
                for (o, x) in h[cd * nef..(cd + 1) * nef].iter_mut().zip(&v[..nef]) {
                    *o += w * x;
                }
            }
        }
        let ub = bra.u[ij];
        let cw = &bra.cw[ij * mab..(ij + 1) * mab];
        for (ab, w) in cw.iter().enumerate() {
            let w = ub * w;
            for (o, x) in acc[ab * mcd * nef..(ab + 1) * mcd * nef]
                .iter_mut()
                .zip(h.iter())
            {
                *o += w * x;
            }
        }
    }
    // HRR on the contracted [e0|f0], once per column quartet.
    let (hb, hk) = (&bra.hrr, &ket.hrr);
    let (nfab, nfcd) = (hb.len(), hk.len());
    for ab in 0..mab {
        for cd in 0..mcd {
            let src = &acc[(ab * mcd + cd) * nef..(ab * mcd + cd + 1) * nef];
            for (fab, tb) in hb.iter().enumerate() {
                let dst0 = (ab * nfab + fab) * (mcd * nfcd) + cd * nfcd;
                for (fcd, tk) in hk.iter().enumerate() {
                    let mut s = 0.0;
                    for &(e, cb) in tb {
                        if cb == 0.0 {
                            continue;
                        }
                        for &(fi, ck) in tk {
                            if ck == 0.0 {
                                continue;
                            }
                            s += cb * ck * src[e * nf + fi];
                        }
                    }
                    out[dst0 + fcd] = s;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests;
