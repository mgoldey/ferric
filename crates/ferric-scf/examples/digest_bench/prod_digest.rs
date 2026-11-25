//! Baseline ("A") digest: a VERBATIM copy of the closed-shell digest code of
//! `crates/ferric-scf/src/quartet_scatter.rs` as of origin/main 58df6850, so
//! it can be fed stored integrals instead of libint output.
//! Lines 38-240 (`JkMode`, `add_j` / `add_k`) and 819-1168 (`DigestScratch`,
//! the blocked kernels) are copied unchanged; `digest_bra_pair` is the body of
//! `scatter_bra_pair_probed`'s per-quartet block (lines 520-526 setup and
//! 697-812) with the libint call replaced by the stored block and the probe
//! timers removed; its one edit is the const parameter `MIN` in place of
//! `BLOCKED_MIN_INTEGRALS` (`A16` = production). The bench checks `A16`
//! bitwise against the real `scatter_bra_pair` (included from the source
//! file) before timing.
#![allow(dead_code, clippy::all)]

use ndarray::{Array2, Array3};

/// What a single quartet-scatter pass accumulates. Each variant carries a
/// zeroed `nbf×nbf` local partial for exactly the matrices this mode needs —
/// no wasted allocation for J-only/K-only callers.
///
/// ## The `Uhf` variant and the `d` argument
///
/// `JOnly`/`KOnly`/`Both` are the CLOSED-SHELL modes: one density `d` drives
/// both the J contraction and the K contraction, and it arrives as
/// `scatter_bra_pair`'s `d` parameter.
///
/// `Uhf` is the OPEN-SHELL mode: J is contracted against the TOTAL density
/// while K is contracted against each SPIN density separately
/// (`F_σ = H + J[D_α+D_β] − c_K·K[D_σ]`). Its α/β densities therefore cannot
/// come from the single `d` parameter and are carried in the variant itself;
/// `d` supplies the J density (`D_total`) exactly as for `Both`. Every `add_k`
/// call site in the scatter fires TWICE in this mode — once per spin — reusing
/// the SAME integral `v`, which is the entire point: one quartet pass replaces
/// the previous `DirectJ(D_total) + DirectK(D_α) + DirectK(D_β)` three-pass
/// build.
#[derive(Debug)]
pub(crate) enum JkMode<'d> {
    JOnly(Array2<f64>),
    KOnly(Array2<f64>),
    Both(Array2<f64>, Array2<f64>),
    /// `(J, K_α, K_β, D_α, D_β)` — see the variant note above.
    Uhf(
        Array2<f64>,
        Array2<f64>,
        Array2<f64>,
        &'d Array2<f64>,
        &'d Array2<f64>,
    ),
    /// MANY densities in one pass (the block CPHF/CPKS): J of every `jd`
    /// density and K of every `kd` density, each stored interleaved as
    /// `(nbf, nbf, m)` so the per-integral loop over densities is contiguous.
    /// The `d` argument of the scatter is ignored.
    Multi(Box<MultiJk<'d>>),
}

/// The buffers of [`JkMode::Multi`].
#[derive(Debug)]
pub(crate) struct MultiJk<'d> {
    /// `J[μ, ν, i] = Σ (μν|λσ) jd[λ, σ, i]`.
    pub(crate) j: Array3<f64>,
    /// `K[μ, λ, i] = Σ (μν|λσ) kd[ν, σ, i]`.
    pub(crate) k: Array3<f64>,
    pub(crate) jd: &'d Array3<f64>,
    pub(crate) kd: &'d Array3<f64>,
}

/// `dst[i] += src[i] * v` over the `m` densities of one `(row, col)` lane.
///
/// # Safety
/// `dst` and `src` must be standard-layout `(nbf, nbf, m)` arrays with the same
/// `m`, and `row, col, la, sg < nbf`.
#[inline(always)]
unsafe fn add_lane(
    dst: &mut Array3<f64>,
    row: usize,
    col: usize,
    src: &Array3<f64>,
    la: usize,
    sg: usize,
    v: f64,
) {
    let (n, m) = (dst.dim().1, dst.dim().2);
    if m == 0 {
        return;
    }
    let d = dst.as_mut_ptr().add((row * n + col) * m);
    let s = src.as_ptr().add((la * n + sg) * m);
    for i in 0..m {
        *d.add(i) += *s.add(i) * v;
    }
}

impl<'d> JkMode<'d> {
    pub(crate) fn new_j(nbf: usize) -> Self {
        JkMode::JOnly(Array2::zeros((nbf, nbf)))
    }
    pub(crate) fn new_k(nbf: usize) -> Self {
        JkMode::KOnly(Array2::zeros((nbf, nbf)))
    }
    pub(crate) fn new_both(nbf: usize) -> Self {
        JkMode::Both(Array2::zeros((nbf, nbf)), Array2::zeros((nbf, nbf)))
    }
    pub(crate) fn new_uhf(nbf: usize, d_a: &'d Array2<f64>, d_b: &'d Array2<f64>) -> Self {
        JkMode::Uhf(
            Array2::zeros((nbf, nbf)),
            Array2::zeros((nbf, nbf)),
            Array2::zeros((nbf, nbf)),
            d_a,
            d_b,
        )
    }

    /// A [`JkMode::Multi`] for the interleaved `(nbf, nbf, m)` density stacks
    /// `jd` (Coulomb) and `kd` (exchange; `m = 0` builds none).
    pub(crate) fn new_multi(jd: &'d Array3<f64>, kd: &'d Array3<f64>) -> Self {
        JkMode::Multi(Box::new(MultiJk {
            j: Array3::zeros(jd.dim()),
            k: Array3::zeros(kd.dim()),
            jd,
            kd,
        }))
    }

    /// Whether this mode accumulates a Coulomb matrix.
    ///
    /// Single source of truth for the `FourPairK` validity invariant (see the
    /// `debug_assert!` at the top of [`scatter_bra_pair`]): a density screen
    /// that omits the `d12`/`d34` pairings is only sound where this is false.
    /// Deliberately mirrors the variant set matched by [`JkMode::add_j`] — if a
    /// new J-bearing variant is added there it must be added here too, and the
    /// exhaustive `match` (no wildcard arm) is what forces that.
    #[inline(always)]
    pub(crate) fn accumulates_j(&self) -> bool {
        match self {
            JkMode::JOnly(_) | JkMode::Both(..) | JkMode::Uhf(..) => true,
            JkMode::Multi(m) => m.jd.dim().2 > 0,
            JkMode::KOnly(_) => false,
        }
    }

    /// Short variant name for diagnostics — avoids `{:?}` on a mode, which
    /// would print entire `nbf×nbf` matrices into a panic message.
    #[inline(always)]
    pub(crate) fn label(&self) -> &'static str {
        match self {
            JkMode::JOnly(_) => "JOnly",
            JkMode::KOnly(_) => "KOnly",
            JkMode::Both(..) => "Both",
            JkMode::Uhf(..) => "Uhf",
            JkMode::Multi(..) => "Multi",
        }
    }

    /// Digest one computed block with the blocked kernels ([`digest_j`],
    /// [`digest_k`]); `false` for [`JkMode::Multi`], which keeps the
    /// per-element scatter.
    #[inline]
    fn digest_blocked(
        &mut self,
        q: &[f64],
        b: &Block,
        d: &Array2<f64>,
        s: &mut DigestScratch,
    ) -> bool {
        match self {
            JkMode::JOnly(j) => digest_j(q, b, d, j, s),
            JkMode::KOnly(k) => digest_k(q, b, d, 0, k, s),
            JkMode::Both(j, k) => {
                digest_j(q, b, d, j, s);
                digest_k(q, b, d, 0, k, s);
            }
            JkMode::Uhf(j, k_a, k_b, d_a, d_b) => {
                digest_j(q, b, d, j, s);
                digest_k(q, b, d_a, 0, k_a, s);
                digest_k(q, b, d_b, 1, k_b, s);
            }
            JkMode::Multi(_) => return false,
        }
        true
    }

    /// `local_j[(row,col)] += d[(la,sg)] * v` — no-op when this mode has no J.
    #[inline(always)]
    fn add_j(&mut self, row: usize, col: usize, d: &Array2<f64>, la: usize, sg: usize, v: f64) {
        match self {
            // SAFETY: indices are in [0, nbf) — guaranteed by the
            // shell offset/dim loop in scatter_quartet.
            JkMode::JOnly(j) | JkMode::Both(j, _) | JkMode::Uhf(j, _, _, _, _) => unsafe {
                *j.uget_mut((row, col)) += d.uget((la, sg)) * v;
            },
            // SAFETY: as above; both stacks are standard-layout (nbf, nbf, m).
            JkMode::Multi(m) => unsafe { add_lane(&mut m.j, row, col, m.jd, la, sg, v) },
            JkMode::KOnly(_) => {}
        }
    }

    /// `local_k[(row,col)] += d[(la,sg)] * v` — no-op when this mode has no K.
    ///
    /// In `Uhf` mode the caller-supplied `d` (the J/total density) is IGNORED
    /// and the two spin densities carried by the variant are used instead, one
    /// per spin buffer.
    #[inline(always)]
    fn add_k(&mut self, row: usize, col: usize, d: &Array2<f64>, la: usize, sg: usize, v: f64) {
        match self {
            // SAFETY: indices are in [0, nbf) — guaranteed by the
            // shell offset/dim loop in scatter_quartet.
            JkMode::KOnly(k) | JkMode::Both(_, k) => unsafe {
                *k.uget_mut((row, col)) += d.uget((la, sg)) * v;
            },
            JkMode::Uhf(_, k_a, k_b, d_a, d_b) => unsafe {
                *k_a.uget_mut((row, col)) += d_a.uget((la, sg)) * v;
                *k_b.uget_mut((row, col)) += d_b.uget((la, sg)) * v;
            },
            // SAFETY: as above; both stacks are standard-layout (nbf, nbf, m).
            JkMode::Multi(m) => unsafe { add_lane(&mut m.k, row, col, m.kd, la, sg, v) },
            JkMode::JOnly(_) => {}
        }
    }
}

/// Smallest quartet block (`n1·n2·n3·n4` integrals) that takes the blocked
/// digestion; smaller blocks keep the per-element 8-fold scatter, whose
/// per-quartet setup is cheaper than gathering density tiles. Measured on
/// octane at 12 threads (digest thread-seconds, per-element / 16 / 256):
/// cc-pVDZ 10.8 / 10.1 / 11.0, cc-pVTZ 234 / 162 / 208.
const BLOCKED_MIN_INTEGRALS: usize = 16;

/// Shell-block scratch for [`digest_j`] / [`digest_k`]: density tiles gathered
/// once per quartet and the output tiles the block contracts into, so each
/// integral is multiplied into small L1-resident buffers and each `nbf × nbf`
/// output element is written once per quartet, not once per integral.
#[derive(Debug, Default)]
pub(crate) struct DigestScratch {
    /// 8 density tiles (4 gathers + their transposed gathers).
    dt: [Vec<f64>; 8],
    /// 8 exchange output tiles (4 direct + 4 transposed images).
    kt: [Vec<f64>; 8],
    /// Coulomb: `E34`, `E12` symmetrized densities and the `S34` output tile.
    e34: Vec<f64>,
    e12: Vec<f64>,
    s34: Vec<f64>,
    /// `sym[k][i][s]`: exchange density `k` has block `(s_i, s)` equal bit for
    /// bit to the transpose of block `(s, s_i)`, for the current bra pair
    /// `(s_0, s_1) = (s1, s2)`. Filled by [`DigestScratch::prepare`].
    sym: [[Vec<bool>; 2]; 2],
}

/// Whether block `(sa, sb)` of `d` is bit for bit the transpose of `(sb, sa)`.
fn block_is_symmetric(
    d: &Array2<f64>,
    offs: &[usize],
    dims: &[usize],
    sa: usize,
    sb: usize,
) -> bool {
    let (oa, na, ob, nb) = (offs[sa], dims[sa], offs[sb], dims[sb]);
    (0..na).all(|i| {
        (0..nb).all(|j| {
            // SAFETY: in range by shell offset/dim construction.
            unsafe { d.uget((oa + i, ob + j)).to_bits() == d.uget((ob + j, oa + i)).to_bits() }
        })
    })
}

impl DigestScratch {
    /// Record, for the bra pair `(s1, s2)`, which density blocks on rows `s1`
    /// and `s2` are transposes of their mirror blocks, for each exchange
    /// density of `mode` (`d` itself for the closed-shell modes).
    fn prepare(
        &mut self,
        mode: &JkMode<'_>,
        d: &Array2<f64>,
        offs: &[usize],
        dims: &[usize],
        s1: usize,
        s2: usize,
    ) {
        let dens: [Option<&Array2<f64>>; 2] = match mode {
            JkMode::KOnly(_) | JkMode::Both(..) => [Some(d), None],
            JkMode::Uhf(_, _, _, d_a, d_b) => [Some(*d_a), Some(*d_b)],
            JkMode::JOnly(_) | JkMode::Multi(_) => [None, None],
        };
        let nsh = dims.len();
        for (k, dk) in dens.iter().enumerate() {
            let Some(dk) = dk else { continue };
            for (i, si) in [s1, s2].into_iter().enumerate() {
                let row = &mut self.sym[k][i];
                row.clear();
                row.extend((0..nsh).map(|s| block_is_symmetric(dk, offs, dims, si, s)));
            }
        }
    }

    fn ensure(&mut self, m: usize) {
        let m2 = m * m;
        for v in self.dt.iter_mut().chain(self.kt.iter_mut()) {
            if v.len() < m2 {
                v.resize(m2, 0.0);
            }
        }
        for v in [&mut self.e34, &mut self.e12, &mut self.s34] {
            if v.len() < m2 {
                v.resize(m2, 0.0);
            }
        }
    }
}

/// Geometry of one computed quartet block `(s1 s2 | s3 s4)`.
#[derive(Debug, Clone, Copy)]
struct Block {
    s: [usize; 4],
    n: [usize; 4],
    o: [usize; 4],
    sym12: bool,
    sym34: bool,
    sym1234: bool,
}

/// `tile[r * cols + c] = m[(r0 + r, c0 + c)]`.
#[inline(always)]
fn gather(m: &Array2<f64>, r0: usize, rows: usize, c0: usize, cols: usize, tile: &mut [f64]) {
    for r in 0..rows {
        for c in 0..cols {
            // SAFETY: r0 + r, c0 + c < nbf by shell offset/dim construction.
            tile[r * cols + c] = unsafe { *m.uget((r0 + r, c0 + c)) };
        }
    }
}

/// `tile[r * cols + c] = m[(c0 + c, r0 + r)]` (the transposed block).
#[inline(always)]
fn gather_t(m: &Array2<f64>, r0: usize, rows: usize, c0: usize, cols: usize, tile: &mut [f64]) {
    for r in 0..rows {
        for c in 0..cols {
            // SAFETY: as in `gather`.
            tile[r * cols + c] = unsafe { *m.uget((c0 + c, r0 + r)) };
        }
    }
}

/// `m[(r0 + r, c0 + c)] += tile[r * cols + c]`.
#[inline(always)]
fn scatter(m: &mut Array2<f64>, r0: usize, rows: usize, c0: usize, cols: usize, tile: &[f64]) {
    for r in 0..rows {
        for c in 0..cols {
            // SAFETY: as in `gather`.
            unsafe { *m.uget_mut((r0 + r, c0 + c)) += tile[r * cols + c] };
        }
    }
}

/// `m[(c0 + c, r0 + r)] += tile[r * cols + c]` (the transposed image).
#[inline(always)]
fn scatter_t(m: &mut Array2<f64>, r0: usize, rows: usize, c0: usize, cols: usize, tile: &[f64]) {
    for r in 0..rows {
        for c in 0..cols {
            // SAFETY: as in `gather`.
            unsafe { *m.uget_mut((c0 + c, r0 + r)) += tile[r * cols + c] };
        }
    }
}

#[inline(always)]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut s = 0.0;
    for (x, y) in a.iter().zip(b) {
        s += x * y;
    }
    s
}

#[inline(always)]
fn axpy(y: &mut [f64], a: f64, x: &[f64]) {
    for (yi, xi) in y.iter_mut().zip(x) {
        *yi += a * xi;
    }
}

/// Coulomb digestion of one block: every `J` image the 8-fold loop writes,
/// `J[μν] (+ J[νμ]) += Σ_λσ v (D[λσ] (+ D[σλ]))` and, for distinct bra and ket
/// pairs, `J[λσ] (+ J[σλ]) += Σ_μν v (D[μν] (+ D[νμ]))`, the bracketed terms
/// present exactly when the matching permutation is a distinct image.
#[inline]
fn digest_j(q: &[f64], b: &Block, d: &Array2<f64>, j: &mut Array2<f64>, s: &mut DigestScratch) {
    let [n1, n2, n3, n4] = b.n;
    let [o1, o2, o3, o4] = b.o;
    let (n12, n34) = (n1 * n2, n3 * n4);
    let e34 = &mut s.e34[..n34];
    gather(d, o3, n3, o4, n4, e34);
    if b.sym34 {
        for c in 0..n3 {
            for dd in 0..n4 {
                // SAFETY: in range by shell construction.
                e34[c * n4 + dd] += unsafe { *d.uget((o4 + dd, o3 + c)) };
            }
        }
    }
    let e12 = &mut s.e12[..n12];
    let s34 = &mut s.s34[..n34];
    if b.sym1234 {
        gather(d, o1, n1, o2, n2, e12);
        if b.sym12 {
            for a in 0..n1 {
                for bb in 0..n2 {
                    // SAFETY: in range by shell construction.
                    e12[a * n2 + bb] += unsafe { *d.uget((o2 + bb, o1 + a)) };
                }
            }
        }
        s34.fill(0.0);
    }
    for ab in 0..n12 {
        let row = &q[ab * n34..(ab + 1) * n34];
        let v = dot(row, e34);
        let (a, bb) = (ab / n2, ab % n2);
        // SAFETY: in range by shell construction.
        unsafe {
            *j.uget_mut((o1 + a, o2 + bb)) += v;
            if b.sym12 {
                *j.uget_mut((o2 + bb, o1 + a)) += v;
            }
        }
        if b.sym1234 {
            axpy(s34, e12[ab], row);
        }
    }
    if b.sym1234 {
        scatter(j, o3, n3, o4, n4, s34);
        if b.sym34 {
            scatter_t(j, o3, n3, o4, n4, s34);
        }
    }
}

/// Exchange digestion of one block into `k` against density `dk`: the four
/// direct images `K[μλ], K[νλ], K[μσ], K[νσ]` and, for distinct bra and ket
/// pairs, their four transposed images `K[λμ], K[λν], K[σμ], K[σν]`, each
/// present exactly when the 8-fold loop writes it.
///
/// A transposed image is the transpose of its direct tile whenever the
/// density tile it reads equals the transpose of the direct one bit for bit
/// (any symmetric density): the products and their order are then the same,
/// so it is added transposed instead of recomputed. Otherwise it is
/// contracted from its own gathered tile.
#[inline]
#[allow(clippy::too_many_arguments)]
fn digest_k(
    q: &[f64],
    b: &Block,
    dk: &Array2<f64>,
    slot: usize,
    k: &mut Array2<f64>,
    s: &mut DigestScratch,
) {
    let [n1, n2, n3, n4] = b.n;
    let [o1, o2, o3, o4] = b.o;
    let [_, _, sh3, sh4] = b.s;
    let (s24, s14, s23, s13) = (n2 * n4, n1 * n4, n2 * n3, n1 * n3);
    let present = [true, b.sym12, b.sym34, b.sym12 && b.sym34];
    // Direct-image densities D[ν σ], D[μ σ], D[ν λ], D[μ λ] in dt[0..4]; the
    // transposed-image ones D[σ ν], D[σ μ], D[λ ν], D[λ μ] in dt[4..8], stored
    // in the same (bra, ket) index order as their direct partners. `own[i]`:
    // transposed image `i` needs its own contraction, because its density
    // block is not the bitwise transpose of the direct one.
    let geo = [
        (o2, n2, o4, n4),
        (o1, n1, o4, n4),
        (o2, n2, o3, n3),
        (o1, n1, o3, n3),
    ];
    let sym = &s.sym[slot];
    let mirror = [sym[1][sh4], sym[0][sh4], sym[1][sh3], sym[0][sh3]];
    let mut own = [false; 4];
    for i in 0..4 {
        if present[i] {
            let (r0, rows, c0, cols) = geo[i];
            gather(dk, r0, rows, c0, cols, &mut s.dt[i][..rows * cols]);
            if b.sym1234 && !mirror[i] {
                own[i] = true;
                gather_t(dk, r0, rows, c0, cols, &mut s.dt[4 + i][..rows * cols]);
            }
        }
    }
    let dt = &s.dt;
    let (d24, d14, d23, d13) = (&dt[0][..s24], &dt[1][..s14], &dt[2][..s23], &dt[3][..s13]);
    let (t42, t41, t32, t31) = (&dt[4][..s24], &dt[5][..s14], &dt[6][..s23], &dt[7][..s13]);
    let [k13, k23, k14, k24, u13, u23, u14, u24] = &mut s.kt;
    let (k13, k23, k14, k24) = (
        &mut k13[..s13],
        &mut k23[..s23],
        &mut k14[..s14],
        &mut k24[..s24],
    );
    let (u13, u23, u14, u24) = (
        &mut u13[..s13],
        &mut u23[..s23],
        &mut u14[..s14],
        &mut u24[..s24],
    );
    for (i, t) in [&mut *k13, &mut *k23, &mut *k14, &mut *k24]
        .into_iter()
        .enumerate()
    {
        if present[i] {
            t.fill(0.0);
        }
    }
    for (i, t) in [&mut *u13, &mut *u23, &mut *u14, &mut *u24]
        .into_iter()
        .enumerate()
    {
        if own[i] {
            t.fill(0.0);
        }
    }
    for a in 0..n1 {
        for bb in 0..n2 {
            for c in 0..n3 {
                let row = &q[((a * n2 + bb) * n3 + c) * n4..][..n4];
                k13[a * n3 + c] += dot(row, &d24[bb * n4..][..n4]);
                if own[0] {
                    u13[a * n3 + c] += dot(row, &t42[bb * n4..][..n4]);
                }
                if b.sym12 {
                    k23[bb * n3 + c] += dot(row, &d14[a * n4..][..n4]);
                    if own[1] {
                        u23[bb * n3 + c] += dot(row, &t41[a * n4..][..n4]);
                    }
                }
                if b.sym34 {
                    axpy(&mut k14[a * n4..][..n4], d23[bb * n3 + c], row);
                    if own[2] {
                        axpy(&mut u14[a * n4..][..n4], t32[bb * n3 + c], row);
                    }
                }
                if b.sym12 && b.sym34 {
                    axpy(&mut k24[bb * n4..][..n4], d13[a * n3 + c], row);
                    if own[3] {
                        axpy(&mut u24[bb * n4..][..n4], t31[a * n3 + c], row);
                    }
                }
            }
        }
    }
    scatter(k, o1, n1, o3, n3, k13);
    if b.sym12 {
        scatter(k, o2, n2, o3, n3, k23);
    }
    if b.sym34 {
        scatter(k, o1, n1, o4, n4, k14);
    }
    if b.sym12 && b.sym34 {
        scatter(k, o2, n2, o4, n4, k24);
    }
    if b.sym1234 {
        scatter_t(k, o1, n1, o3, n3, if own[0] { u13 } else { k13 });
        if b.sym12 {
            scatter_t(k, o2, n2, o3, n3, if own[1] { u23 } else { k23 });
        }
        if b.sym34 {
            scatter_t(k, o1, n1, o4, n4, if own[2] { u14 } else { k14 });
        }
        if b.sym12 && b.sym34 {
            scatter_t(k, o2, n2, o4, n4, if own[3] { u24 } else { k24 });
        }
    }
}

/// One bra pair's stored survivors, digested exactly as
/// `scatter_bra_pair_probed` digests libint's blocks (scratch set-up per bra
/// pair included, as production does it). `MIN` is
/// `BLOCKED_MIN_INTEGRALS` (16 in production; other values are variants).
#[allow(clippy::too_many_arguments)]
pub(crate) fn digest_bra_pair<const MIN: usize>(
    dims: &[usize],
    offs: &[usize],
    d: &Array2<f64>,
    s1: usize,
    s2: usize,
    kets: &[(u32, u32, u64)],
    ints: &[f64],
    mode: &mut JkMode<'_>,
) {
    let mut scratch = DigestScratch::default();
    scratch.ensure(dims.iter().copied().max().unwrap_or(1));
    scratch.prepare(mode, d, offs, dims, s1, s2);
    let (n1, n2) = (dims[s1], dims[s2]);
    let (o1, o2) = (offs[s1], offs[s2]);
    let sym12 = s1 != s2;
    for &(s3, s4, off) in kets {
        let (s3, s4) = (s3 as usize, s4 as usize);
        let q = &ints[off as usize..];
        {
            let (n3, n4) = (dims[s3], dims[s4]);
            let (o3, o4) = (offs[s3], offs[s4]);
            let sym34 = s3 != s4;
            let sym1234 = (s1, s2) != (s3, s4);

            // Fast path for STO-3G / small shells (n=1): every index is
            // fixed, so the scatter collapses to at most 8 scalar adds
            // with no inner loop. Ported from `build_jk_with_pool` to all
            // three callers.
            if n1 == 1 && n2 == 1 && n3 == 1 && n4 == 1 {
                // SAFETY: q has at least n1*n2*n3*n4 >= 1 element from the engine.
                let v = unsafe { *q.get_unchecked(0) };
                mode.add_j(o1, o2, d, o3, o4, v);
                mode.add_k(o1, o3, d, o2, o4, v);
                if sym12 {
                    mode.add_j(o2, o1, d, o3, o4, v);
                    mode.add_k(o2, o3, d, o1, o4, v);
                }
                if sym34 {
                    mode.add_j(o1, o2, d, o4, o3, v);
                    mode.add_k(o1, o4, d, o2, o3, v);
                }
                if sym12 && sym34 {
                    mode.add_j(o2, o1, d, o4, o3, v);
                    mode.add_k(o2, o4, d, o1, o3, v);
                }
                if sym1234 {
                    mode.add_j(o3, o4, d, o1, o2, v);
                    mode.add_k(o3, o1, d, o4, o2, v);
                    if sym12 {
                        mode.add_j(o3, o4, d, o2, o1, v);
                        mode.add_k(o3, o2, d, o4, o1, v);
                    }
                    if sym34 {
                        mode.add_j(o4, o3, d, o1, o2, v);
                        mode.add_k(o4, o1, d, o3, o2, v);
                    }
                    if sym12 && sym34 {
                        mode.add_j(o4, o3, d, o2, o1, v);
                        mode.add_k(o4, o2, d, o3, o1, v);
                    }
                }
                continue;
            }

            // Larger shells: the blocked digestion (every mode but Multi).
            let blk = Block {
                s: [s1, s2, s3, s4],
                n: [n1, n2, n3, n4],
                o: [o1, o2, o3, o4],
                sym12,
                sym34,
                sym1234,
            };
            if n1 * n2 * n3 * n4 >= MIN && mode.digest_blocked(q, &blk, d, &mut scratch) {
                continue;
            }
            // Small blocks and Multi: the per-element 8-fold scatter.
            for a in 0..n1 {
                for b in 0..n2 {
                    for c in 0..n3 {
                        for dd in 0..n4 {
                            // SAFETY: flat index < n1*n2*n3*n4 by loop bounds; q has that many elements from the engine.
                            let v = unsafe { *q.get_unchecked(((a * n2 + b) * n3 + c) * n4 + dd) };
                            let mu = o1 + a;
                            let nu = o2 + b;
                            let la = o3 + c;
                            let sg = o4 + dd;

                            mode.add_j(mu, nu, d, la, sg, v);
                            mode.add_k(mu, la, d, nu, sg, v);
                            if sym12 {
                                mode.add_j(nu, mu, d, la, sg, v);
                                mode.add_k(nu, la, d, mu, sg, v);
                            }
                            if sym34 {
                                mode.add_j(mu, nu, d, sg, la, v);
                                mode.add_k(mu, sg, d, nu, la, v);
                            }
                            if sym12 && sym34 {
                                mode.add_j(nu, mu, d, sg, la, v);
                                mode.add_k(nu, sg, d, mu, la, v);
                            }
                            if sym1234 {
                                mode.add_j(la, sg, d, mu, nu, v);
                                mode.add_k(la, mu, d, sg, nu, v);
                                if sym12 {
                                    mode.add_j(la, sg, d, nu, mu, v);
                                    mode.add_k(la, nu, d, sg, mu, v);
                                }
                                if sym34 {
                                    mode.add_j(sg, la, d, mu, nu, v);
                                    mode.add_k(sg, mu, d, la, nu, v);
                                }
                                if sym12 && sym34 {
                                    mode.add_j(sg, la, d, nu, mu, v);
                                    mode.add_k(sg, nu, d, la, mu, v);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
