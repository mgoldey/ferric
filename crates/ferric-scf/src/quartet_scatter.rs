//! Shared quartet-scatter kernel for the direct J/K/JK Fock builders.
//!
//! `rhf::build_jk_with_pool`, `DirectJK::build`, and `DirectJ::build` /
//! `DirectK::build` independently re-implemented the same shell-pair work
//! list, the same Häser-Ahlrichs density screen, the same MPI striping, the
//! same grouped-deterministic reduction, and the same 8-fold-symmetry
//! quartet scatter. This module factors that common kernel out ONCE so
//! future fixes (the `INTERRUPT` check, the `n=1` fast path) apply to all
//! three builders instead of accumulating as silent divergences.
//!
//! ## What varies across callers
//!
//! - **What gets accumulated**: J only, K only, or both from the same
//!   integral (see `JkMode`).
//! - **The density screen**: `build_jk`/`DirectJK` use the full six-pairwise
//!   `d_max_shell` table (`max(d12,d34,d13,d14,d23,d24)`); `DirectJ`/`DirectK`
//!   use a single global `max_d` scalar. Parameterized via `DensityScreen`
//!   — this is a difference in SCREENING TIGHTNESS only: a looser screen can
//!   only *admit* quartets a tighter one would also admit (never the
//!   reverse), so the semantics of each existing caller are preserved
//!   exactly, byte-for-byte, by passing the matching variant.
//!
//! ## What must NOT vary (the bit-identity invariant)
//!
//! The shell-pair list, its MPI-rank filter, the deterministic group
//! partition (`reduce::deterministic_group_size`), and the fold order inside
//! each group are a pure function of the (already-computed, caller-supplied)
//! work list — never of the thread count. This kernel does not change any of
//! that: it just runs the per-quartet scatter that used to be duplicated
//! three times. See `direct_builders_bit_identical_across_thread_counts`
//! (direct_jk.rs) and `build_jk_bit_identical_across_thread_counts` (rhf.rs)
//! for the regression gate.

use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ndarray::{Array2, Array3};
use std::sync::{Arc, RwLock};

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
    /// The bra-pair-local digest ([`JkDigest::Pair`]): half partials `Jh`,
    /// `Kb` (and `Kt`) of the densities a [`PairSpec`] prepared, turned into
    /// J and K by [`PairSpec::finish`] after the ordered group sum. The `d`
    /// argument of the scatter is ignored. Made by [`PairSpec::mode`].
    Pair(Box<PairJk<'d>>),
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

    /// The half partials of a [`JkMode::Pair`], in [`PairSpec::n_partials`]
    /// order. Panics on any other mode.
    pub(crate) fn into_pair_partials(self) -> Vec<Array2<f64>> {
        match self {
            JkMode::Pair(p) => {
                p.spec
                    .quartets
                    .fetch_add(p.quartets, std::sync::atomic::Ordering::Relaxed);
                p.mats
            }
            other => unreachable!("a pair-digest walk got JkMode::{}", other.label()),
        }
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
            JkMode::Pair(p) => p.spec.has_j(),
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
            JkMode::Pair(..) => "Pair",
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
            JkMode::Pair(_) => unreachable!("the pair digest has its own kernel"),
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
            JkMode::Pair(_) => unreachable!("the pair digest has its own kernel"),
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
            JkMode::Pair(_) => unreachable!("the pair digest has its own kernel"),
            JkMode::JOnly(_) => {}
        }
    }
}

/// The Häser-Ahlrichs density-weighted pair screen, parameterized so
/// `build_jk`/`DirectJK` (six-pairwise `d_max_shell` table) and
/// `DirectJ`/`DirectK` (single global `max_d` scalar) keep their existing,
/// distinct screening semantics exactly.
#[derive(Debug)]
pub(crate) enum DensityScreen<'a> {
    /// `dmax = max(d12, d34, d13, d14, d23, d24)` from the shell-blocked
    /// `d_max_shell` table (build_jk / DirectJK).
    SixPair(&'a Array2<f64>),
    /// `dmax = max(d13, d14, d23, d24)` from the same shell-blocked
    /// `d_max_shell` table — the EXCHANGE-ONLY pairings (LinK).
    ///
    /// **K-only builders; using this where J is also accumulated under-screens
    /// J.** The omitted `d12`/`d34` are exactly the pairings the J contraction
    /// reads, so a fused J+K sweep driven by this key would discard quartets
    /// carrying a significant `D[s1,s2]` or `D[s3,s4]` whenever the four
    /// exchange blocks happen to be small — silently, and with an error that
    /// grows with system size. Enforced by a `debug_assert!` in
    /// [`scatter_bra_pair`]; LinK's own loop does not route through that
    /// function, so its correctness rests on the same restriction being true
    /// there by construction (it accumulates K only).
    ///
    /// # Why a K-only builder needs its own variant
    ///
    /// A shell quartet `(s1 s2 | s3 s4)` contracts against D through different
    /// blocks depending on what is being built. The J contraction reads
    /// `D[s3,s4]` (and `D[s1,s2]` by 8-fold symmetry); the K contraction reads
    /// `D[s1,s3]`, `D[s1,s4]`, `D[s2,s3]`, `D[s2,s4]` and their transposes.
    /// [`SixPair`](DensityScreen::SixPair) takes the max over BOTH sets because
    /// its callers build J and K from the same integral in one sweep, so a
    /// single screen decision has to be conservative for both.
    ///
    /// A K-only builder never touches `d12` or `d34`. Including them would be
    /// VALID — a max over a superset is an upper bound on a max over a subset,
    /// so it can only ever admit more quartets — but strictly looser, and it
    /// would pin LinK's quartet count to exactly `build_jk`'s instead of below
    /// it. Since LinK exists to do LESS work than the default builder, the
    /// four-pairing max is the screen that lets it.
    ///
    /// Validity: every density element the LinK scatter multiplies lies in one
    /// of the four blocks, so this is an elementwise upper bound on all of
    /// them. The transposed reads (`D[s3,s1]` etc., reached under `sym1234`)
    /// are covered because `build_d_max_shell` is built from `|D|` and every
    /// density LinK serves is symmetric, making the table symmetric too:
    /// `t[(a,b)] == t[(b,a)]`.
    FourPairK(&'a Array2<f64>),
    /// A single global `max |D_μν|` scalar (DirectJ / DirectK).
    Global(f64),
}

impl<'a> DensityScreen<'a> {
    /// `dmax` for a given (s1,s2,s3,s4) shell quartet, matching each caller's
    /// existing formula exactly.
    ///
    /// `pub(crate)` because LinK drives its own quartet loop (pair-list rather
    /// than canonical) and so calls this directly instead of going through
    /// [`scatter_bra_pair`] — sharing the ONE implementation of each density
    /// key rather than growing a second copy in `link_k.rs`.
    #[inline(always)]
    pub(crate) fn dmax(&self, s1: usize, s2: usize, s3: usize, s4: usize) -> f64 {
        match self {
            DensityScreen::SixPair(t) => {
                let d12 = t[(s1, s2)];
                let d34 = t[(s3, s4)];
                let d13 = t[(s1, s3)];
                let d14 = t[(s1, s4)];
                let d23 = t[(s2, s3)];
                let d24 = t[(s2, s4)];
                d12.max(d34).max(d13).max(d14).max(d23).max(d24)
            }
            DensityScreen::FourPairK(t) => {
                // The exchange pairings only — see the variant doc for why
                // d12/d34 are deliberately absent.
                let d13 = t[(s1, s3)];
                let d14 = t[(s1, s4)];
                let d23 = t[(s2, s3)];
                let d24 = t[(s2, s4)];
                d13.max(d14).max(d23).max(d24)
            }
            DensityScreen::Global(m) => *m,
        }
    }
}

/// Build the `d_max_shell[(si,sj)] = max|D_μν|` table over shell blocks
/// (μ ∈ si, ν ∈ sj), shared by every caller of [`DensityScreen::SixPair`].
pub(crate) fn build_d_max_shell(prep: &PreparedBasis, d: &Array2<f64>) -> Array2<f64> {
    let nsh = prep.nshells();
    let dims = prep.shell_dims();
    let offs = prep.shell_offsets();
    let mut d_max_shell = Array2::<f64>::zeros((nsh, nsh));
    for si in 0..nsh {
        for sj in 0..nsh {
            let (oi, ni) = (offs[si], dims[si]);
            let (oj, nj) = (offs[sj], dims[sj]);
            let mut m = 0.0f64;
            for a in 0..ni {
                for b in 0..nj {
                    // SAFETY: oi+a < nbf and oj+b < nbf by shell offset/dim construction.
                    let v = unsafe { d.uget((oi + a, oj + b)).abs() };
                    if v > m {
                        m = v;
                    }
                }
            }
            d_max_shell[(si, sj)] = m;
        }
    }
    d_max_shell
}

/// Shell-blocked density-max table for the OPEN-SHELL combined J+K build:
/// `d_max_shell[(si,sj)] = max over (μ∈si, ν∈sj) of |D_α| + |D_β|`.
///
/// # Why the sum, and why this is the conservative choice
///
/// One quartet pass now feeds THREE contractions with a SINGLE screen
/// decision: `J[D_α+D_β]`, `K[D_α]`, and `K[D_β]`. A screen that drops a
/// quartet drops it from all three, so the table must be an elementwise upper
/// bound on every density any of them contracts. Elementwise:
///
/// * `|D_α| ≤ |D_α| + |D_β|` and `|D_β| ≤ |D_α| + |D_β|`  (trivially), and
/// * `|D_α + D_β| ≤ |D_α| + |D_β|`  (triangle inequality).
///
/// so `|D_α| + |D_β|` bounds all three WITHOUT assuming anything about the
/// sign or definiteness of either spin density — it holds for the SCF
/// densities, for the ΔD deltas of the incremental path (where the two
/// channels routinely move in OPPOSITE directions and `|ΔD_total|` can be far
/// smaller than either channel), and for any intermediate DIIS iterate.
///
/// Taking `max(|D_α|,|D_β|)` instead would be tighter by up to 2× but is NOT
/// a bound on `|D_α + D_β|` (equal same-sign channels give `|D_total| = 2·max`),
/// so it would under-screen J. The sum is the cheapest key that is correct for
/// all three, and it is still dramatically tighter than the single global
/// `max|D|` scalar the previous three-pass open-shell path screened on.
pub(crate) fn build_d_max_shell_spin_sum(
    prep: &PreparedBasis,
    d_a: &Array2<f64>,
    d_b: &Array2<f64>,
) -> Array2<f64> {
    let nsh = prep.nshells();
    let dims = prep.shell_dims();
    let offs = prep.shell_offsets();
    let mut d_max_shell = Array2::<f64>::zeros((nsh, nsh));
    for si in 0..nsh {
        for sj in 0..nsh {
            let (oi, ni) = (offs[si], dims[si]);
            let (oj, nj) = (offs[sj], dims[sj]);
            let mut m = 0.0f64;
            for a in 0..ni {
                for b in 0..nj {
                    // SAFETY: oi+a < nbf and oj+b < nbf by shell offset/dim construction.
                    let v = unsafe {
                        d_a.uget((oi + a, oj + b)).abs() + d_b.uget((oi + a, oj + b)).abs()
                    };
                    if v > m {
                        m = v;
                    }
                }
            }
            d_max_shell[(si, sj)] = m;
        }
    }
    d_max_shell
}

/// The canonical (s1,s2) bra-pair work list `{(s1,s2) : 0<=s2<=s1<nsh}`,
/// shared by every caller (before any caller-specific bra-thresh
/// pre-filter or MPI striping is applied).
pub(crate) fn canonical_bra_pairs(nsh: usize) -> Vec<(usize, usize)> {
    (0..nsh)
        .flat_map(|s1| (0..=s1).map(move |s2| (s1, s2)))
        .collect()
}

/// Scatter one screened (s1,s2) bra pair's contribution into `mode`,
/// iterating the canonical `s3<=s1, s4<=s4max` ket loop, applying the
/// Schwarz×density screen, and running the 8-fold-symmetry quartet scatter
/// (including the `n==1` shell fast path). `check_interrupt` mirrors
/// `build_jk_with_pool`'s per-(s1,s3-every-100) `INTERRUPT` polling — now run
/// by every caller, not just `build_jk`. Returns the number of quartets that
/// passed the screen and were computed (non-degenerate) by libint2, matching
/// each caller's existing `computed_quartets` bookkeeping.
///
/// `m_table`: optional CSB `M` table
/// ([`crate::screening::CsbBounds::m`]), built for the SAME operator as
/// `q_table`. `None` — every caller at the default `[scf] screening =
/// "schwarz"` — is byte-identical to before this parameter existed: the screen
/// below reduces to the original `b12 * b34 * dmax < thresh` with the `min`
/// refinement branch not taken at all (the `Option` is matched ONCE per BRA
/// PAIR, outside the `(s3,s4)` loop, never per candidate).
///
/// When `Some`, this applies Eq. (8) of Thompson & Ochsenfeld, JCP 147, 144101
/// (2017) — `min{Q_µν Q_λσ, M_µλ M_νσ, M_µσ M_νλ}` — as the bound, in place of
/// the plain Schwarz product. This is the hot-loop reimplementation of
/// [`crate::screening::CsbBounds::estimate`]'s math, kept deliberately in sync
/// with it (row slices instead of a `Bound` vtable call, so the extra two
/// multiplies and two `min`s are paid without a dynamic dispatch on top). The
/// `min` is seeded with the Schwarz product for the same structural reason it
/// is there: the refinement can only ever TIGHTEN, never loosen, whatever the
/// `M` table contains.
///
/// `x_table`: optional CSAM `X` table
/// ([`crate::screening::SchwarzBounds::csam_x`]), likewise built for the SAME
/// operator as `q_table`. `None` is again byte-identical to the plain Schwarz
/// screen. When `Some`, the plain Schwarz product is multiplied by
/// `sqrt(max(X[s1,s3]*X[s2,s4], X[s1,s4]*X[s2,s3])).min(1.0)` — the hot-loop
/// reimplementation of [`crate::screening::CsamBounds::estimate_nonrigorous`].
///
/// **NOTE: the CSAM refinement is a NON-RIGOROUS estimate.** It can drop a
/// quartet whose true contribution exceeds the threshold (see
/// `ferric_integrals::csam`'s module header). It is reached only via the
/// explicit `[scf] screening = "csam"` opt-in. The CSB refinement carries no
/// such caveat — it is a rigorous `min`.
///
/// `m_table` and `x_table` are MUTUALLY EXCLUSIVE per call: they come from
/// `SchwarzBounds::{csb_m, csam_x}`, which
/// [`crate::screening::SchwarzBounds::compute_for_screening`] populates from a
/// three-way `match` on ONE [`crate::screening::ScreeningKind`], so at most one
/// is ever `Some`. The loop below nevertheless checks CSB FIRST and CSAM only
/// in its `else`, so were both somehow attached (reachable only by
/// hand-mutating the struct) the RIGOROUS refinement would win — the safe
/// direction, and the same precedence
/// [`crate::screening::LinkBound`]'s `schwarz_ref_estimate` uses.
///
/// The two rows `M[s1, :]`/`M[s2, :]` (or `X[s1, :]`/`X[s2, :]`) are sliced out
/// ONCE here — before the `(s3, s4)` loop — so the inner loop pays only
/// `&[f64]` indexing rather than repeated strided `Array2` lookups. That hoist
/// is what keeps either refinement from adding an `Option` branch to the ~46M-
/// candidate inner loop the perf profile identified (`scatter_bra_pair` at
/// 28.96% of a benzene/aug-cc-pVDZ RHF wall clock).
#[allow(clippy::too_many_arguments)]
pub(crate) fn scatter_bra_pair(
    engine: &mut Engine,
    prep: &PreparedBasis,
    dims: &[usize],
    offs: &[usize],
    q_table: &Array2<f64>,
    m_table: Option<&Array2<f64>>,
    x_table: Option<&Array2<f64>>,
    screen: &DensityScreen,
    thresh: f64,
    d: &Array2<f64>,
    s1: usize,
    s2: usize,
    mode: &mut JkMode<'_>,
    check_interrupt: bool,
) -> usize {
    scatter_bra_pair_probed::<false>(
        engine,
        prep,
        dims,
        offs,
        q_table,
        m_table,
        x_table,
        screen,
        thresh,
        d,
        s1,
        s2,
        mode,
        check_interrupt,
        &mut ScatterTimes::default(),
    )
}

/// Wall time a bra-pair scatter spent in its two measurable stages, summed
/// over the quartets it computed (see [`scatter_bra_pair_probed`]).
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct ScatterTimes {
    /// Inside `Engine::compute_quartet` (libint2 plus the shim's copy-out).
    pub(crate) libint_ns: u64,
    /// Contracting the returned block into J/K (the 8-fold digestion).
    pub(crate) digest_ns: u64,
    /// Quartets that passed the screen (computed or zero-screened by libint2).
    pub(crate) survivors: u64,
    /// `(s3, s4)` candidates the screen looked at.
    pub(crate) candidates: u64,
}

impl ScatterTimes {
    /// Add `other`'s counters into `self`.
    pub(crate) fn add(&mut self, other: &ScatterTimes) {
        self.libint_ns += other.libint_ns;
        self.digest_ns += other.digest_ns;
        self.survivors += other.survivors;
        self.candidates += other.candidates;
    }
}

/// Nanoseconds since `t0`, saturating.
#[inline(always)]
fn ns_since(t0: std::time::Instant) -> u64 {
    u64::try_from(t0.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

/// [`scatter_bra_pair`] with an optional stage probe. With `PROBE = false`
/// (what [`scatter_bra_pair`] passes) no clock is read and `times` is not
/// touched, so the arithmetic and the generated hot loop are those of the
/// unprobed kernel. With `PROBE = true` two clock reads bracket each computed
/// quartet's libint2 call and its digestion, which is what the
/// `FERRIC_TIMING` table of the direct J/K builders reports.
#[allow(clippy::too_many_arguments)]
pub(crate) fn scatter_bra_pair_probed<const PROBE: bool>(
    engine: &mut Engine,
    prep: &PreparedBasis,
    dims: &[usize],
    offs: &[usize],
    q_table: &Array2<f64>,
    m_table: Option<&Array2<f64>>,
    x_table: Option<&Array2<f64>>,
    screen: &DensityScreen,
    thresh: f64,
    d: &Array2<f64>,
    s1: usize,
    s2: usize,
    mode: &mut JkMode<'_>,
    check_interrupt: bool,
    times: &mut ScatterTimes,
) -> usize {
    use std::sync::atomic::Ordering;

    // INVARIANT: the exchange-only density key is valid only where J is not
    // accumulated. `FourPairK` omits the d12/d34 pairings that bound the J
    // contraction, so pairing it with any J-accumulating mode under-screens J
    // — silently, and extensively (the error grows with system size, per
    // Hollman/Schaefer/Valeev on Schwarz truncation). Debug-only: this is a
    // static property of each call site, not data-dependent, so one debug run
    // of the suite is enough to prove every site correct, and release builds
    // pay nothing.
    debug_assert!(
        !(matches!(screen, DensityScreen::FourPairK(_)) && mode.accumulates_j()),
        "DensityScreen::FourPairK used with a J-accumulating JkMode ({}). FourPairK takes the max \
         over the four EXCHANGE pairings only and omits d12/d34, which are precisely the blocks \
         the J contraction reads — so this combination under-screens J. Use SixPair for any mode \
         that accumulates J.",
        mode.label()
    );

    let mut local_count = 0usize;
    // The legacy digests' tile scratch; the pair digest has its own
    // (`PairJk`) and never reads this one.
    let mut scratch = DigestScratch::default();
    if !matches!(mode, JkMode::Pair(_)) {
        scratch.ensure(dims.iter().copied().max().unwrap_or(1));
        scratch.prepare(mode, d, offs, dims, s1, s2);
    }
    let b12 = q_table[(s1, s2)];
    let (n1, n2) = (dims[s1], dims[s2]);
    let (o1, o2) = (offs[s1], offs[s2]);
    let sym12 = s1 != s2;

    // Hoisted CSB `M` rows for this bra pair (see the doc above): one row
    // slice each, taken ONCE per (s1,s2), not per (s3,s4) candidate.
    // `ArrayView::to_slice()` ties the slice's lifetime to the BACKING ARRAY
    // (`m_table`'s own borrow), not to the temporary `row(..)` view, so the
    // resulting `&[f64]` is valid for the rest of this function. It returns
    // `Some` for a standard-layout (C-contiguous) array's row, which every
    // `csb_m_table` output is (built via `Array2::zeros` plus index
    // assignment, never a view/transpose/slice of a larger array), so the
    // `expect` cannot fire in practice; it exists to fail loudly rather than
    // silently take a slower path if that ever changed.
    let m_rows: Option<(&[f64], &[f64])> = m_table.map(|m| {
        (
            m.row(s1).to_slice().expect("csb M row(s1) not contiguous"),
            m.row(s2).to_slice().expect("csb M row(s2) not contiguous"),
        )
    });

    // Hoisted CSAM `X` rows, on exactly the same terms as the CSB `M` rows
    // above (same once-per-bra-pair slicing, same contiguity reasoning — every
    // `csam_x_table` output is likewise built via `Array2::zeros` plus index
    // assignment, never a view/transpose/slice of a larger array).
    let x_rows: Option<(&[f64], &[f64])> = x_table.map(|x| {
        (
            x.row(s1).to_slice().expect("csam X row(s1) not contiguous"),
            x.row(s2).to_slice().expect("csam X row(s2) not contiguous"),
        )
    });

    // The pair digest's per-bra-pair start (its `end` follows the loop).
    if let JkMode::Pair(p) = mode {
        let t = PROBE.then(std::time::Instant::now);
        p.begin(dims, offs, s1, s2);
        if let Some(t) = t {
            times.digest_ns += ns_since(t);
        }
    }

    for s3 in 0..=s1 {
        if check_interrupt && s3 % 100 == 0 && ferric_core::INTERRUPT.load(Ordering::Relaxed) {
            break;
        }
        let s4max = if s3 == s1 { s2 } else { s3 };
        for s4 in 0..=s4max {
            if PROBE {
                times.candidates += 1;
            }
            let b34 = q_table[(s3, s4)];
            let dmax = screen.dmax(s1, s2, s3, s4);
            // Plain Schwarz product: the seed of the CSB `min` below, and the
            // whole bound when `m_rows` is `None` (the default path, which is
            // then byte-identical to the pre-CSB code).
            let mut bound = b12 * b34;
            // CSB, Eq. (8): min{ Q_µν Q_λσ, M_µλ M_νσ, M_µσ M_νλ }. Seeded
            // with the Schwarz product already in `bound`, so a defective `M`
            // table can only ever make this TIGHTER — never looser — which is
            // the structural half of the safety argument in
            // `crate::screening::CsbBounds`'s type docs. The other half (that
            // a too-SMALL `M` would win this `min` and break validity) is
            // defended by `csb_m_table`'s precision/floor discipline and
            // caught by `tests/csb_screening.rs::csb_is_a_valid_upper_bound`.
            //
            // Applied to the OPERATOR bound only, before the density factor:
            // `dmax` is a density magnitude, not an integral bound, and Eq. (8)
            // bounds the integral. Multiplying it in afterwards keeps the
            // Häser-Ahlrichs product structure `|(bound on integral)| · max|D|`
            // that every caller's threshold is calibrated against.
            if let Some((mr1, mr2)) = m_rows {
                // SAFETY-BY-CONSTRUCTION: s3, s4 < nsh and the rows have
                // length nsh, so these index in bounds.
                let eq6 = mr1[s3] * mr2[s4];
                let eq7 = mr1[s4] * mr2[s3];
                bound = bound.min(eq6).min(eq7);
            } else if let Some((xr1, xr2)) = x_rows {
                // CSAM, the NON-RIGOROUS alternative: multiply the Schwarz
                // product by sqrt(max(X[s1,s3]*X[s2,s4], X[s1,s4]*X[s2,s3])).
                // Same formula as `crate::screening::CsamBounds`'s
                // `refinement_factor` (kept in sync deliberately: this is the
                // hot-loop reimplementation of that method's math, not an
                // approximation of it, using row slices instead of a `Bound`
                // trait call so the sqrt/max cost is paid without a vtable
                // dispatch on top). `.min(1.0)` mirrors
                // `estimate_nonrigorous`'s clamp, so roundoff can never make
                // this LOOSER than plain Schwarz.
                //
                // In the `else` of the CSB branch, not beside it: the two
                // tables are mutually exclusive by construction (see this
                // function's doc), and checking CSB first means the RIGOROUS
                // refinement wins if that invariant is ever broken by hand.
                let term_a = xr1[s3] * xr2[s4];
                let term_b = xr1[s4] * xr2[s3];
                let factor = term_a.max(term_b).max(0.0).sqrt().min(1.0);
                bound *= factor;
            }
            // NOTE (merge, 2026-09-14): `dmax` multiplies the FINISHED operator
            // bound here rather than being folded in before the refinement.
            // For CSB that ordering is required (the `min` must compare
            // integral bounds against integral bounds, not density-scaled
            // ones). For CSAM it is merely a pure reassociation of a product —
            // algebraically identical to the CSAM branch's own
            // `(b12*b34*dmax)*factor`, but NOT bitwise identical to it. The
            // byte-identity anchor that matters is unaffected: with both
            // tables `None` this is still exactly `b12 * b34 * dmax < thresh`.
            if bound * dmax < thresh {
                continue;
            }

            if PROBE {
                times.survivors += 1;
            }
            let t_lib = PROBE.then(std::time::Instant::now);
            let block = engine.compute_quartet(prep, s1, s2, s3, s4);
            let t_dig = PROBE.then(std::time::Instant::now);
            if let (Some(a), Some(b)) = (t_lib, t_dig) {
                times.libint_ns += u64::try_from((b - a).as_nanos()).unwrap_or(u64::MAX);
            }
            if let Some(q) = block {
                local_count += 1;
                let (n3, n4) = (dims[s3], dims[s4]);
                let (o3, o4) = (offs[s3], offs[s4]);
                let sym34 = s3 != s4;
                let sym1234 = (s1, s2) != (s3, s4);

                // The bra-pair-local digest replaces all three branches below.
                if let JkMode::Pair(p) = mode {
                    p.quartet(q, dims, offs, s3, s4);
                    if let Some(t) = t_dig {
                        times.digest_ns += ns_since(t);
                    }
                    continue;
                }

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
                    if let Some(t) = t_dig {
                        times.digest_ns += ns_since(t);
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
                if n1 * n2 * n3 * n4 >= BLOCKED_MIN_INTEGRALS
                    && mode.digest_blocked(q, &blk, d, &mut scratch)
                {
                    if let Some(t) = t_dig {
                        times.digest_ns += ns_since(t);
                    }
                    continue;
                }
                // Small blocks and Multi: the per-element 8-fold scatter.
                for a in 0..n1 {
                    for b in 0..n2 {
                        for c in 0..n3 {
                            for dd in 0..n4 {
                                // SAFETY: flat index < n1*n2*n3*n4 by loop bounds; q has that many elements from the engine.
                                let v =
                                    unsafe { *q.get_unchecked(((a * n2 + b) * n3 + c) * n4 + dd) };
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
                if let Some(t) = t_dig {
                    times.digest_ns += ns_since(t);
                }
            }
        }
    }
    if let JkMode::Pair(p) = mode {
        let t = PROBE.then(std::time::Instant::now);
        p.end();
        if let Some(t) = t {
            times.digest_ns += ns_since(t);
        }
    }
    local_count
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
            JkMode::JOnly(_) | JkMode::Multi(_) | JkMode::Pair(_) => [None, None],
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

// ---------------------------------------------------------------------------
// The digest switch.
// ---------------------------------------------------------------------------

/// Environment variable selecting the J/K digest where a config left it
/// unset ([`JkDigest::resolve`]).
pub const JK_DIGEST_ENV: &str = "FERRIC_JK_DIGEST";

/// How the direct four-centre J/K builders contract each computed quartet
/// block into J and K.
///
/// * `Pair` (the default): the bra-pair-local digest. For each bra pair
///   `(s1 s2)` the bra J block is a running dot product written once per
///   bra pair, the ket J block is written once per quartet into a half
///   matrix `Jh`, and K is accumulated only into the rows of the two bra
///   shells (`Kb`; against `Dᵀ` into `Kt` for a non-symmetric density).
///   `J = Jh + Jhᵀ` and `K = Kb + Kbᵀ` (or `Kb + Ktᵀ`) once per build. Same
///   integrals, same screening, same group partition; only the summation
///   order differs, so results move at the 1e-14 relative level (energies by
///   about 1e-12 Ha).
/// * `Legacy`: the 8-fold image scatter that preceded it, bit for bit.
///
/// The closed-shell, open-shell, J-only and K-only direct walks
/// (`DirectJK`, `DirectJ`, `DirectK`, `rhf::build_jk_with_pool`) follow it.
/// The block-CPHF multi-density build, the open-shell response build, the
/// in-core store, LinK, CFMM, density fitting and the gradients keep their
/// own digests; the open-shell guess Fock always runs `Legacy`
/// ([`SchwarzBounds::for_guess`](crate::screening::SchwarzBounds)).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum JkDigest {
    /// The bra-pair-local digest. The default.
    #[default]
    Pair,
    /// The 8-fold image scatter of earlier releases, bit for bit.
    Legacy,
}

/// `FERRIC_JK_DIGEST`: `pair` or `legacy`, case-insensitive, surrounding
/// blanks ignored; anything else, an empty or all-blank value included, is
/// refused (the convention of the other enumerated `FERRIC_*` switches,
/// `FERRIC_GC_ROTATE` and `FERRIC_DFK_METRIC`). Unset means `pair`.
static JK_DIGEST: ferric_core::config::ConfigVar<JkDigest> = ferric_core::config::ConfigVar {
    env_name: JK_DIGEST_ENV,
    default: JkDigest::Pair,
    parse: |s| JkDigest::parse_str(&s.trim().to_ascii_lowercase()),
    validate: ferric_core::config::accept_any,
};

impl JkDigest {
    /// The canonical spelling (`pair` / `legacy`).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pair => "pair",
            Self::Legacy => "legacy",
        }
    }

    /// Strict parse of a config value (`[scf] jk_digest`): exactly `pair` or
    /// `legacy`. Anything else, other cases and padded values included, is
    /// an error.
    pub fn parse_str(s: &str) -> Result<Self, String> {
        match s {
            "pair" => Ok(Self::Pair),
            "legacy" => Ok(Self::Legacy),
            _ => Err(format!(
                "invalid value {s:?}: expected exactly \"pair\" or \"legacy\""
            )),
        }
    }

    /// The digest a build runs: `explicit` (a config's) if set, else
    /// `FERRIC_JK_DIGEST`, else [`JkDigest::Pair`]. A malformed variable is
    /// an error naming it.
    pub fn resolve(explicit: Option<JkDigest>) -> Result<Self, FerricError> {
        JK_DIGEST
            .resolve(explicit, ferric_core::config::env_lookup)
            .map(|r| r.value)
            .map_err(|e| FerricError::General(format!("{JK_DIGEST_ENV}: {e}")))
    }

    /// [`resolve`](Self::resolve) for a record (a run log, a checkpoint
    /// identity): the canonical spelling of the digest the builds run, or,
    /// for a malformed variable, `unresolved` with its raw text (the builds
    /// themselves refuse it).
    pub fn resolved_label(explicit: Option<JkDigest>) -> String {
        match Self::resolve(explicit) {
            Ok(d) => d.as_str().to_string(),
            Err(_) => format!("unresolved {:?}", std::env::var(JK_DIGEST_ENV).ok()),
        }
    }

    /// The parsed `FERRIC_JK_DIGEST` of this process (unset = `pair`).
    pub fn env_setting() -> Result<Self, FerricError> {
        Self::resolve(None)
    }
}

impl std::fmt::Display for JkDigest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The J/K digest a config recorded on one
/// [`SchwarzBounds`](crate::screening::SchwarzBounds) (its `digest` field).
///
/// The SCF drivers record their
/// [`RhfConfig::jk_digest`](crate::rhf::RhfConfig::jk_digest) here
/// ([`DigestSlot::set`]), so every direct build on those bounds, or on a
/// clone of them, or on the rotated bounds of their contraction rotation,
/// runs the SCF's digest: the SCF's own builders and the Newton, stability
/// and response builds that follow. Nothing recorded, or a config that left
/// the field unset, means `FERRIC_JK_DIGEST`, then `pair`
/// ([`JkDigest::resolve`]); so the variable reaches every build in the
/// process, bounds an SCF never saw included. Shared by every clone; the
/// last writer wins.
#[derive(Clone, Default)]
pub struct DigestSlot(Arc<RwLock<Option<JkDigest>>>);

impl std::fmt::Debug for DigestSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("DigestSlot").field(&self.recorded()).finish()
    }
}

impl DigestSlot {
    /// Record a config's `jk_digest` (`None`: unset, the variable and the
    /// default decide).
    pub fn set(&self, digest: Option<JkDigest>) {
        if let Ok(mut g) = self.0.write() {
            *g = digest;
        }
    }

    /// What a config recorded (`None`: nothing, or unset).
    pub fn recorded(&self) -> Option<JkDigest> {
        self.0.read().map(|g| *g).unwrap_or_default()
    }

    /// Copy `other`'s record onto these bounds: for a consumer that builds
    /// its own bounds for an SCF's geometry and should follow its digest.
    pub fn copy_from(&self, other: &DigestSlot) {
        self.set(other.recorded());
    }

    /// The digest the builds on these bounds run ([`JkDigest::resolve`] of
    /// the record).
    pub fn mode(&self) -> Result<JkDigest, FerricError> {
        JkDigest::resolve(self.recorded())
    }

    /// [`mode`](Self::mode) agreed across the MPI ranks of `ctx`: every
    /// rank runs the same digest, or every rank returns an error, before
    /// the build's first collective (the pair digest reserves its memory
    /// with one, the legacy digest does not, so ranks on different digests
    /// would issue different collectives). A rank whose own resolution
    /// failed returns that error; the others name the disagreement. Without
    /// MPI, or at one rank, this is `mode`.
    pub(crate) fn mode_collective(
        &self,
        ctx: &ferric_core::parallel::ParallelContext,
    ) -> Result<JkDigest, FerricError> {
        use crate::direct_jk::gc_rotation::all_ranks_agree;
        let local = self.mode();
        // Two collectives on every rank, whatever it resolved.
        let all_pair = all_ranks_agree(ctx, matches!(local, Ok(JkDigest::Pair)));
        let all_legacy = all_ranks_agree(ctx, matches!(local, Ok(JkDigest::Legacy)));
        match local {
            Ok(d) if all_pair || all_legacy => Ok(d),
            Ok(d) => Err(FerricError::General(format!(
                "J/K digest: this rank resolved {d}, another MPI rank a different digest \
                 or none ({JK_DIGEST_ENV} differs between ranks); set [scf] jk_digest, \
                 or the variable alike on every rank"
            ))),
            Err(e) => Err(e),
        }
    }
}

// ---------------------------------------------------------------------------
// The bra-pair-local digest (`JkDigest::Pair`).
// ---------------------------------------------------------------------------

/// Relative tolerance of [`is_symmetric_to_roundoff`].
pub(crate) const PAIR_SYMMETRY_TOL: f64 = 1e-14;

/// What a density's symmetry is judged against
/// ([`is_symmetric_to_roundoff`]).
#[derive(Debug, Clone, Copy)]
pub(crate) enum SymScale<'a> {
    /// A full density: relative to its own `max|D|`.
    Own,
    /// The density change `ΔD` of an incremental build: relative to
    /// `max(max|ΔD|, reference[c])`, where `reference[c]` is `max|D|` of
    /// exchange channel `c`'s full density at the builder's last full build
    /// ([`PairSpec::scales`]). A missing entry falls back to [`SymScale::Own`].
    Delta(&'a [f64]),
}

impl SymScale<'_> {
    fn reference(&self, channel: usize) -> Option<f64> {
        match self {
            SymScale::Own => None,
            SymScale::Delta(r) => r.get(channel).copied(),
        }
    }
}

/// Every matrix of a direct J/K build (`(name, shape)`) is `n × n`,
/// `n = prep.nbasis()`: the density screens (`build_d_max_shell`) and the
/// kernels index them without bounds checks, so a wrong shape is refused at
/// each builder's entry, before any of them runs.
pub(crate) fn check_nbf_shapes(
    who: &str,
    n: usize,
    mats: &[(&str, (usize, usize))],
) -> Result<(), FerricError> {
    for &(name, dim) in mats {
        if dim != (n, n) {
            return Err(FerricError::General(format!(
                "{who}: {name} has shape {dim:?}, expected ({n}, {n}) (prep.nbasis())"
            )));
        }
    }
    Ok(())
}

/// `(max|D − Dᵀ|, max|D|)` of a square matrix; `None` when `d` is not
/// square or holds a non-finite element.
fn symmetry_stats(d: &Array2<f64>) -> Option<(f64, f64)> {
    let n = d.nrows();
    if d.ncols() != n {
        return None;
    }
    let (mut asym, mut big) = (0.0f64, 0.0f64);
    for i in 0..n {
        for j in 0..=i {
            let (a, b) = (d[(i, j)], d[(j, i)]);
            if !a.is_finite() || !b.is_finite() {
                // `f64::max` would drop a NaN, and an infinite element
                // makes the test meaningless: not symmetric.
                return None;
            }
            asym = asym.max((a - b).abs());
            big = big.max(a.abs()).max(b.abs());
        }
    }
    Some((asym, big))
}

/// Whether `d` is square, finite and symmetric to roundoff, which decides
/// between the symmetric and the general kernel of the pair digest.
///
/// The rule: `max|D − Dᵀ| ≤ PAIR_SYMMETRY_TOL · S` with
///
/// * `S = max|D|` for a full density (`reference = None`): purely
///   relative, so an antisymmetric part is kept (general kernel) at any
///   absolute size once it exceeds 1e-14 of the density;
/// * `S = max(max|D|, reference)` for the `ΔD` of an incremental build,
///   with `reference = max|D_full|` of the builder's last full build: a
///   density difference inherits the absolute rounding asymmetry of the two
///   full densities it is the difference of (about 1e-16 of `max|D_full|`),
///   which is large relative to a small `ΔD`, and symmetrising it moves K
///   by that much of the full K, below the integral screen.
///
/// A non-square matrix or one with a NaN or infinite element is not
/// symmetric (the general kernel then carries it through as it is).
pub(crate) fn is_symmetric_to_roundoff(d: &Array2<f64>, reference: Option<f64>) -> bool {
    match symmetry_stats(d) {
        Some((asym, big)) => asym <= PAIR_SYMMETRY_TOL * reference.map_or(big, |r| big.max(r)),
        None => false,
    }
}

/// A density matrix of the pair digest in standard layout: the caller's
/// own when it already is, else a copy (a plain reference, so a
/// [`PairSpec`] stays covariant in its lifetime).
#[derive(Debug)]
enum PairMat<'a> {
    Borrowed(&'a Array2<f64>),
    Owned(Array2<f64>),
}

impl<'a> PairMat<'a> {
    fn standard(d: &'a Array2<f64>) -> Self {
        if d.is_standard_layout() {
            PairMat::Borrowed(d)
        } else {
            PairMat::Owned(d.as_standard_layout().into_owned())
        }
    }

    /// Doubles [`PairMat::standard`] copies for `d`.
    fn standard_len(d: &Array2<f64>) -> usize {
        if d.is_standard_layout() {
            0
        } else {
            d.len()
        }
    }

    fn get(&self) -> &Array2<f64> {
        match self {
            PairMat::Borrowed(d) => d,
            PairMat::Owned(d) => d,
        }
    }

    /// Doubles held beyond the caller's matrix.
    #[cfg(test)]
    fn owned_len(&self) -> usize {
        match self {
            PairMat::Borrowed(_) => 0,
            PairMat::Owned(d) => d.len(),
        }
    }
}

/// One exchange density of the pair digest.
///
/// Planned when its [`PairSpec`] is made (the symmetry check and `max|D|`,
/// no allocation), prepared ([`PairDensity::materialize`]) once the build
/// has reserved the memory of its copies ([`PairSpec::sum_groups`]).
#[derive(Debug)]
pub(crate) struct PairDensity<'a> {
    /// The caller's density.
    src: &'a Array2<f64>,
    /// `src` is not symmetric to roundoff: the general kernel.
    general: bool,
    /// The density the kernel reads, standard layout: `½(D + Dᵀ)` (exactly
    /// symmetric, owned) when `D` is symmetric to roundoff, else `D` itself
    /// (borrowed when it is already in standard layout). `src` until
    /// materialized.
    d: PairMat<'a>,
    /// `Dᵀ` in standard layout for a non-symmetric `D` once materialized;
    /// `None` otherwise (the general kernel then reads `d` for `Dᵀ`, which
    /// it equals).
    dt: Option<Array2<f64>>,
    /// `max|D|` (finite elements; the symmetry check's own scale).
    scale: f64,
}

impl<'a> PairDensity<'a> {
    /// `d` planned: symmetric to roundoff ([`is_symmetric_to_roundoff`]
    /// against `reference`) or not, and its scale. Allocates nothing.
    fn plan(d: &'a Array2<f64>, reference: Option<f64>) -> Self {
        PairDensity {
            src: d,
            general: !is_symmetric_to_roundoff(d, reference),
            d: PairMat::Borrowed(d),
            dt: None,
            scale: symmetry_stats(d).map_or(0.0, |s| s.1),
        }
    }

    /// The planned density prepared: symmetrised explicitly when it is
    /// symmetric to roundoff, so the result does not depend on which
    /// triangle the kernel reads and K comes out exactly symmetric; kept as
    /// is, with its transpose, otherwise (the general kernel).
    fn materialize(&mut self) {
        let d = self.src;
        let n = d.nrows();
        if self.general {
            let mut dt = Array2::<f64>::zeros((d.ncols(), n));
            dt.assign(&d.t());
            self.d = PairMat::standard(d);
            self.dt = Some(dt);
        } else {
            let mut s = Array2::<f64>::zeros((n, n));
            for i in 0..n {
                for j in 0..=i {
                    let v = midpoint(d[(i, j)], d[(j, i)]);
                    s[(i, j)] = v;
                    s[(j, i)] = v;
                }
            }
            self.d = PairMat::Owned(s);
        }
    }

    /// [`plan`](Self::plan) then [`materialize`](Self::materialize).
    #[cfg(test)]
    pub(crate) fn prepare(d: &'a Array2<f64>, reference: Option<f64>) -> Self {
        let mut p = Self::plan(d, reference);
        p.materialize();
        p
    }

    /// Whether this density is not symmetric to roundoff.
    pub(crate) fn is_general(&self) -> bool {
        self.general
    }

    /// Doubles this density holds beyond its input once materialized (the
    /// symmetrised copy, or a standard-layout copy and the transpose).
    fn owned_len(&self) -> usize {
        if self.general {
            PairMat::standard_len(self.src) + self.src.len()
        } else {
            self.src.len()
        }
    }
}

/// `½(a + b)` of two finite values, finite whenever the midpoint is:
/// `0.5 · (a + b)` (exact but for the one rounding of the sum), else, when
/// the sum overflows, `0.5·a + 0.5·b` (equal values stay as they are).
fn midpoint(a: f64, b: f64) -> f64 {
    let s = a + b;
    if s.is_finite() {
        0.5 * s
    } else {
        0.5 * a + 0.5 * b
    }
}

/// Where the Coulomb density of a [`PairSpec`] lives.
#[derive(Debug)]
enum PairJd<'a> {
    /// K only.
    None,
    /// Its own: the caller's `src`, read as `mat` (borrowed when `src` is
    /// in standard layout, else a copy made when the spec is materialized).
    Own {
        src: &'a Array2<f64>,
        mat: PairMat<'a>,
    },
    /// The first exchange density (the closed-shell job: J of the
    /// symmetrised density equals J of `D`, whose symmetric part is all J
    /// reads, and a non-symmetric `D` is read as it is).
    Channel0,
}

/// What one pair-digest build contracts: an optional Coulomb density and up
/// to two exchange densities, prepared once per build and shared by every
/// group. [`PairSpec::sum_groups`] runs a walk's groups on it and finishes
/// J and K.
#[derive(Debug)]
pub(crate) struct PairSpec<'a> {
    jd: PairJd<'a>,
    /// Exchange densities (0, 1 or 2).
    kd: Vec<PairDensity<'a>>,
    /// Every exchange channel runs the general kernel (some density was not
    /// symmetric to roundoff).
    general: bool,
    nbf: usize,
    /// Largest shell dimension (scratch size).
    max_dim: usize,
    /// Quartets the kernel digested (summed from the group accumulators).
    quartets: std::sync::atomic::AtomicUsize,
    /// The densities are prepared ([`PairSpec::materialize`]).
    prepared: bool,
    /// Time spent preparing the densities (`FERRIC_TIMING`).
    prep_ns: u64,
}

/// J and K of one pair-digest build ([`PairSpec::sum_groups`]), with the
/// stage times `FERRIC_TIMING` reports.
#[derive(Debug)]
pub(crate) struct PairOut {
    /// `J` (`None` for K only).
    pub(crate) j: Option<Array2<f64>>,
    /// `K` per exchange channel.
    pub(crate) k: Vec<Array2<f64>>,
    /// Density preparation (`PairSpec` construction), seconds.
    pub(crate) prep_s: f64,
    /// `finish` (`J = Jh + Jhᵀ`, `K = Kb + Kbᵀ` or `Kb + Ktᵀ`), seconds.
    pub(crate) finish_s: f64,
    /// The kernel per exchange channel ([`PairSpec::kernel_label`]).
    pub(crate) kernel: String,
}

impl<'a> PairSpec<'a> {
    fn new(
        nbf: usize,
        dims: &[usize],
        jd: Option<&'a Array2<f64>>,
        kd: &[&'a Array2<f64>],
        sym: SymScale<'_>,
    ) -> Self {
        let t0 = std::time::Instant::now();
        let kd: Vec<PairDensity<'a>> = kd
            .iter()
            .enumerate()
            .map(|(c, d)| PairDensity::plan(d, sym.reference(c)))
            .collect();
        let general = kd.iter().any(PairDensity::is_general);
        let jd = match jd {
            None => PairJd::None,
            Some(d) => PairJd::Own {
                src: d,
                mat: PairMat::Borrowed(d),
            },
        };
        PairSpec {
            jd,
            kd,
            general,
            nbf,
            max_dim: dims.iter().copied().max().unwrap_or(1),
            quartets: std::sync::atomic::AtomicUsize::new(0),
            prepared: false,
            prep_ns: ns_since(t0),
        }
    }

    /// Prepare the planned densities (the copies [`charged_bytes`]
    /// counts): after the build's reservation, so a refused one allocates
    /// none of them. Idempotent.
    ///
    /// [`charged_bytes`]: Self::charged_bytes
    fn materialize(&mut self) {
        if self.prepared {
            return;
        }
        let t0 = std::time::Instant::now();
        for p in &mut self.kd {
            p.materialize();
        }
        if let PairJd::Own { src, mat } = &mut self.jd {
            *mat = PairMat::standard(src);
        }
        self.prepared = true;
        self.prep_ns += ns_since(t0);
    }

    /// Every density this spec contracts is `nbf × nbf`; else the kernel,
    /// which indexes them with stride `nbf`, would read past the matrix.
    fn check_shapes(&self) -> Result<(), FerricError> {
        let want = (self.nbf, self.nbf);
        let jd = match &self.jd {
            PairJd::Own { src, .. } => Some(*src),
            _ => None,
        };
        for d in jd.into_iter().chain(self.kd.iter().map(|p| p.src)) {
            if d.dim() != want {
                return Err(FerricError::General(format!(
                    "direct J/K (pair digest): density shape {:?}, expected {want:?} \
                     (prep.nbasis())",
                    d.dim()
                )));
            }
        }
        Ok(())
    }

    /// Closed-shell J and K of `d` (`JkMode::Both`'s job). J reads the
    /// prepared exchange density (no copy of its own).
    pub(crate) fn both(nbf: usize, dims: &[usize], d: &'a Array2<f64>, sym: SymScale<'_>) -> Self {
        let mut s = Self::new(nbf, dims, None, &[d], sym);
        s.jd = PairJd::Channel0;
        s
    }

    /// Open-shell `J[d_total]`, `K[d_a]`, `K[d_b]` (`JkMode::Uhf`'s job). The
    /// symmetry check runs per spin; if either spin is not symmetric, both
    /// channels run the general kernel (the symmetric one still symmetrised).
    pub(crate) fn uhf(
        nbf: usize,
        dims: &[usize],
        d_total: &'a Array2<f64>,
        d_a: &'a Array2<f64>,
        d_b: &'a Array2<f64>,
        sym: SymScale<'_>,
    ) -> Self {
        Self::new(nbf, dims, Some(d_total), &[d_a, d_b], sym)
    }

    /// J of `d` only.
    pub(crate) fn j_only(nbf: usize, dims: &[usize], d: &'a Array2<f64>) -> Self {
        Self::new(nbf, dims, Some(d), &[], SymScale::Own)
    }

    /// K of `d` only.
    pub(crate) fn k_only(nbf: usize, dims: &[usize], d: &'a Array2<f64>) -> Self {
        Self::new(nbf, dims, None, &[d], SymScale::Own)
    }

    /// Whether the exchange channels run the general kernel.
    #[cfg(test)]
    pub(crate) fn is_general(&self) -> bool {
        self.general
    }

    /// `max|D|` per exchange channel: the reference a later incremental
    /// build on the same walk passes as [`SymScale::Delta`].
    pub(crate) fn scales(&self) -> Vec<f64> {
        self.kd.iter().map(|p| p.scale).collect()
    }

    /// The exchange kernel, for `FERRIC_TIMING`: `none` (J only),
    /// `symmetric`, or `general` with each channel's input (`sym` when that
    /// spin was symmetric to roundoff and runs the general form only
    /// because the other did not).
    pub(crate) fn kernel_label(&self) -> String {
        if self.kd.is_empty() {
            return "none (J only)".into();
        }
        if !self.general {
            return "symmetric".into();
        }
        let each: Vec<&str> = self
            .kd
            .iter()
            .map(|p| if p.is_general() { "nonsym" } else { "sym" })
            .collect();
        format!("general [{}]", each.join(", "))
    }

    fn has_j(&self) -> bool {
        !matches!(self.jd, PairJd::None)
    }

    /// The Coulomb density (standard layout), `None` for K only.
    fn jd(&self) -> Option<&Array2<f64>> {
        match &self.jd {
            PairJd::None => None,
            PairJd::Own { mat, .. } => Some(mat.get()),
            PairJd::Channel0 => Some(self.kd[0].d.get()),
        }
    }

    /// Number of half partials a group accumulates: `Jh`, one `Kb` per
    /// exchange channel and, for the general kernel, one `Kt` per channel.
    pub(crate) fn n_partials(&self) -> usize {
        let nk = self.kd.len();
        usize::from(self.has_j()) + nk + if self.general { nk } else { 0 }
    }

    /// Bytes this build holds beyond the legacy digest's: the prepared
    /// density copies and the half accumulators (from the plan, so known
    /// before any is allocated).
    fn charged_bytes(&self) -> usize {
        let own_j = match &self.jd {
            PairJd::Own { src, .. } => PairMat::standard_len(src),
            _ => 0,
        };
        let own: usize = own_j + self.kd.iter().map(PairDensity::owned_len).sum::<usize>();
        (own + self.n_partials() * self.nbf * self.nbf) * std::mem::size_of::<f64>()
    }

    /// `n_partials` zeroed `nbf × nbf` matrices (an accumulator for the
    /// ordered sum).
    pub(crate) fn zeros(&self) -> Vec<Array2<f64>> {
        (0..self.n_partials())
            .map(|_| Array2::zeros((self.nbf, self.nbf)))
            .collect()
    }

    /// A fresh group accumulator (on a materialized spec).
    pub(crate) fn mode(&self) -> JkMode<'_> {
        debug_assert!(self.prepared, "pair digest: mode() before materialize()");
        JkMode::Pair(Box::new(PairJk {
            spec: self,
            mats: self.zeros(),
            jb: vec![0.0; self.max_dim * self.max_dim],
            e12: vec![0.0; self.max_dim * self.max_dim],
            e34: vec![0.0; self.max_dim * self.max_dim],
            s34: vec![0.0; self.max_dim * self.max_dim],
            bra: PairBra::default(),
            quartets: 0,
        }))
    }

    /// One walk's groups on this spec: `run(g, mode)` scatters group `g`'s
    /// bra pairs into a fresh accumulator, the half partials are summed in
    /// ascending group order ([`crate::reduce::ordered_stream_sum_many`]:
    /// bit-identical across thread counts), then finished
    /// ([`PairSpec::finish`]; linear, so it commutes with the rank
    /// reduction the caller applies next). The density copies and
    /// accumulators are charged to the memory pool for the build (on every
    /// rank or none) before any of them is allocated, and held through the
    /// fold. A density that is not `nbf × nbf` is an error.
    pub(crate) fn sum_groups<F>(
        &mut self,
        ctx: &ferric_core::parallel::ParallelContext,
        n_groups: usize,
        band_bytes: usize,
        times: Option<&mut crate::reduce::StreamTimes>,
        run: F,
    ) -> Result<PairOut, FerricError>
    where
        F: Fn(usize, &mut JkMode<'_>) + Sync,
    {
        self.check_shapes()?;
        let _charge = crate::direct_jk::gc_rotation::reserve_collective(
            ctx,
            &format!(
                "direct J/K pair-digest densities and accumulators (n={})",
                self.nbf
            ),
            self.charged_bytes(),
        )?;
        self.materialize();
        let mut halves = self.zeros();
        crate::reduce::ordered_stream_sum_many(
            &mut halves,
            n_groups,
            self.nbf,
            band_bytes,
            |g| {
                let mut mode = self.mode();
                run(g, &mut mode);
                Ok(mode.into_pair_partials())
            },
            times,
        )?;
        let t0 = std::time::Instant::now();
        let (j, k) = self.finish(halves);
        let finish_s = ns_since(t0) as f64 * 1e-9;
        pair_probe::record(|| pair_probe::PairBuild {
            coulomb: self.has_j(),
            general: self.general,
            symmetric_inputs: self.kd.iter().map(|p| !p.is_general()).collect(),
            quartets: self.quartets.load(std::sync::atomic::Ordering::Relaxed),
        });
        Ok(PairOut {
            j,
            k,
            prep_s: self.prep_ns as f64 * 1e-9,
            finish_s,
            kernel: self.kernel_label(),
        })
    }

    /// The summed half partials as `(J, [K per channel])`: `J = Jh + Jhᵀ`
    /// (`None` for K only), `K = Kb + Kbᵀ` (symmetric kernel) or
    /// `K = Kb + Ktᵀ` (general), each formed in place in its `Jh` / `Kb`.
    pub(crate) fn finish(
        &self,
        mut mats: Vec<Array2<f64>>,
    ) -> (Option<Array2<f64>>, Vec<Array2<f64>>) {
        assert_eq!(mats.len(), self.n_partials(), "pair digest: partial count");
        let nj = usize::from(self.has_j());
        let nk = self.kd.len();
        let kt = if self.general {
            mats.split_off(nj + nk)
        } else {
            Vec::new()
        };
        let mut kb = mats.split_off(nj);
        let j = mats.pop().map(|mut jh| {
            add_own_transpose(&mut jh);
            jh
        });
        for (c, k) in kb.iter_mut().enumerate() {
            match kt.get(c) {
                Some(t) => {
                    let n = k.nrows();
                    for r in 0..n {
                        for col in 0..n {
                            k[(r, col)] += t[(col, r)];
                        }
                    }
                }
                None => add_own_transpose(k),
            }
        }
        (j, kb)
    }
}

/// `a ← a + aᵀ` in place (exactly symmetric).
fn add_own_transpose(a: &mut Array2<f64>) {
    let n = a.nrows();
    for r in 0..n {
        for c in 0..r {
            let s = a[(r, c)] + a[(c, r)];
            a[(r, c)] = s;
            a[(c, r)] = s;
        }
        a[(r, r)] += a[(r, r)];
    }
}

/// A test probe of the pair digest: which builds ran it and on which
/// kernel. Off unless [`start`](pair_probe::start)ed, per thread (a build
/// records on the thread that called it), so production builds pay one
/// thread-local read. Not part of the API.
#[doc(hidden)]
pub mod pair_probe {
    use std::cell::RefCell;

    /// One finished pair-digest build.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct PairBuild {
        /// It built a Coulomb matrix.
        pub coulomb: bool,
        /// Its exchange channels ran the general (non-symmetric) kernel.
        pub general: bool,
        /// Per exchange channel: the input was symmetric to roundoff.
        pub symmetric_inputs: Vec<bool>,
        /// Quartets the pair kernel digested.
        pub quartets: usize,
    }

    thread_local! {
        static LOG: RefCell<Option<Vec<PairBuild>>> = const { RefCell::new(None) };
    }

    /// Start recording this thread's pair-digest builds (clearing any record).
    pub fn start() {
        LOG.with(|l| *l.borrow_mut() = Some(Vec::new()));
    }

    /// The builds recorded since [`start`], and stop recording.
    pub fn take() -> Vec<PairBuild> {
        LOG.with(|l| l.borrow_mut().take().unwrap_or_default())
    }

    pub(crate) fn record(build: impl FnOnce() -> PairBuild) {
        LOG.with(|l| {
            if let Some(v) = l.borrow_mut().as_mut() {
                v.push(build());
            }
        });
    }
}

/// The current bra pair of a [`PairJk`].
#[derive(Debug, Default, Clone, Copy)]
struct PairBra {
    n: [usize; 2],
    o: [usize; 2],
    s: [usize; 2],
    sym12: bool,
}

/// A group accumulator of the pair digest ([`JkMode::Pair`]): the half
/// partials (layout of [`PairSpec::n_partials`]) and the per-bra-pair
/// scratch.
#[derive(Debug)]
pub(crate) struct PairJk<'d> {
    spec: &'d PairSpec<'d>,
    mats: Vec<Array2<f64>>,
    /// Bra J accumulator `J_bra[μν]`, written once per bra pair.
    jb: Vec<f64>,
    /// `E12 = D[μν] (+ D[νμ])` of the current bra pair.
    e12: Vec<f64>,
    /// `E34 = D[λσ] (+ D[σλ])` of the current quartet.
    e34: Vec<f64>,
    /// Ket J block of the current quartet.
    s34: Vec<f64>,
    bra: PairBra,
    /// Quartets digested (added to the spec's count when the group ends).
    quartets: usize,
}

impl PairJk<'_> {
    /// Start bra pair `(s1, s2)`: zero the bra J accumulator and form `E12`.
    fn begin(&mut self, dims: &[usize], offs: &[usize], s1: usize, s2: usize) {
        let (n1, n2) = (dims[s1], dims[s2]);
        let (o1, o2) = (offs[s1], offs[s2]);
        let sym12 = s1 != s2;
        self.bra = PairBra {
            n: [n1, n2],
            o: [o1, o2],
            s: [s1, s2],
            sym12,
        };
        if let Some(jd) = self.spec.jd() {
            self.jb[..n1 * n2].fill(0.0);
            for a in 0..n1 {
                for b in 0..n2 {
                    // SAFETY: o1 + a, o2 + b < nbf by shell offset/dim construction.
                    let mut v = unsafe { *jd.uget((o1 + a, o2 + b)) };
                    if sym12 {
                        v += unsafe { *jd.uget((o2 + b, o1 + a)) };
                    }
                    self.e12[a * n2 + b] = v;
                }
            }
        }
    }

    /// Flush the bra pair's J into `Jh` (0.5 for `s1 = s2`, whose transpose
    /// image `Jh + Jhᵀ` would otherwise double).
    fn end(&mut self) {
        if !self.spec.has_j() {
            return;
        }
        let [n1, n2] = self.bra.n;
        let [o1, o2] = self.bra.o;
        let w12 = if self.bra.sym12 { 1.0 } else { 0.5 };
        let jh = &mut self.mats[0];
        for a in 0..n1 {
            for b in 0..n2 {
                // SAFETY: in range by shell construction.
                unsafe { *jh.uget_mut((o1 + a, o2 + b)) += w12 * self.jb[a * n2 + b] };
            }
        }
    }

    /// Digest one computed block `(s1 s2|s3 s4)` of the current bra pair.
    fn quartet(&mut self, q: &[f64], dims: &[usize], offs: &[usize], s3: usize, s4: usize) {
        self.quartets += 1;
        let spec = self.spec;
        match (spec.has_j(), spec.kd.len(), spec.general) {
            (true, 1, false) => self.quartet_t::<true, 1, false>(q, dims, offs, s3, s4),
            (true, 1, true) => self.quartet_t::<true, 1, true>(q, dims, offs, s3, s4),
            (true, 2, false) => self.quartet_t::<true, 2, false>(q, dims, offs, s3, s4),
            (true, 2, true) => self.quartet_t::<true, 2, true>(q, dims, offs, s3, s4),
            (true, 0, _) => self.quartet_t::<true, 0, false>(q, dims, offs, s3, s4),
            (false, 1, false) => self.quartet_t::<false, 1, false>(q, dims, offs, s3, s4),
            (false, 1, true) => self.quartet_t::<false, 1, true>(q, dims, offs, s3, s4),
            (false, 2, false) => self.quartet_t::<false, 2, false>(q, dims, offs, s3, s4),
            (false, 2, true) => self.quartet_t::<false, 2, true>(q, dims, offs, s3, s4),
            (j, nk, g) => {
                unreachable!("pair digest: no kernel for J={j} channels={nk} general={g}")
            }
        }
    }

    #[inline(always)]
    fn quartet_t<const J: bool, const NK: usize, const GEN: bool>(
        &mut self,
        q: &[f64],
        dims: &[usize],
        offs: &[usize],
        s3: usize,
        s4: usize,
    ) {
        let spec = self.spec;
        let n = spec.nbf;
        let [n1, n2] = self.bra.n;
        let (n3, n4) = (dims[s3], dims[s4]);
        assert!(q.len() >= n1 * n2 * n3 * n4, "pair digest: short block");
        let g = PairGeom {
            n: [n1, n2, n3, n4],
            o: [self.bra.o[0], self.bra.o[1], offs[s3], offs[s4]],
            sym12: self.bra.sym12,
            sym34: s3 != s4,
            sym1234: (self.bra.s[0], self.bra.s[1]) != (s3, s4),
        };
        let nj = usize::from(J);
        let mut kb = [std::ptr::null_mut::<f64>(); 2];
        let mut kt = [std::ptr::null_mut::<f64>(); 2];
        let mut dk = [std::ptr::null::<f64>(); 2];
        let mut dkt = [std::ptr::null::<f64>(); 2];
        let mut jh = std::ptr::null_mut::<f64>();
        for (i, m) in self.mats.iter_mut().enumerate() {
            let p = m.as_mut_ptr();
            if J && i == 0 {
                jh = p;
            } else if i < nj + NK {
                kb[i - nj] = p;
            } else {
                kt[i - nj - NK] = p;
            }
        }
        for (c, kd) in spec.kd.iter().enumerate().take(NK) {
            dk[c] = kd.d.get().as_ptr();
            if GEN {
                // A channel symmetric to roundoff (symmetrised) is its own
                // transpose.
                dkt[c] = kd.dt.as_ref().map_or(dk[c], |t| t.as_ptr());
            }
        }
        let jd = spec.jd().map_or(std::ptr::null(), |d| d.as_ptr());
        let sc = PairScratch {
            jb: self.jb.as_mut_ptr(),
            e12: self.e12.as_ptr(),
            e34: self.e34.as_mut_ptr(),
            s34: self.s34.as_mut_ptr(),
        };
        let p = PairPtrs {
            n,
            jd,
            jh,
            dk,
            dkt,
            kb,
            kt,
        };
        // SAFETY: every index the kernel forms is inside its shell block of an
        // `nbf × nbf` standard-layout matrix (the half partials, the prepared
        // densities, `jd`), the scratch holds `max_dim²` entries, and `q` holds
        // the `n1·n2·n3·n4` integrals of the block (asserted above).
        unsafe {
            match n4 {
                1 => pair_quartet::<1, J, NK, GEN>(q.as_ptr(), &g, &p, &sc),
                3 => pair_quartet::<3, J, NK, GEN>(q.as_ptr(), &g, &p, &sc),
                5 => pair_quartet::<5, J, NK, GEN>(q.as_ptr(), &g, &p, &sc),
                _ => pair_quartet::<0, J, NK, GEN>(q.as_ptr(), &g, &p, &sc),
            }
        }
    }
}

/// Geometry of one quartet of the pair digest.
struct PairGeom {
    n: [usize; 4],
    o: [usize; 4],
    sym12: bool,
    sym34: bool,
    sym1234: bool,
}

/// Raw row-major pointers of one pair-digest quartet: the Coulomb density
/// and `Jh`, and per exchange channel `D`, `Dᵀ` (general kernel), `Kb` and
/// `Kt` (general kernel). All `n × n`.
struct PairPtrs {
    n: usize,
    jd: *const f64,
    jh: *mut f64,
    dk: [*const f64; 2],
    dkt: [*const f64; 2],
    kb: [*mut f64; 2],
    kt: [*mut f64; 2],
}

/// The per-bra-pair scratch of a [`PairJk`] as raw pointers.
struct PairScratch {
    jb: *mut f64,
    e12: *const f64,
    e34: *mut f64,
    s34: *mut f64,
}

/// `Σ a[i] b[i]` over `N` (or, for `N = 0`, `n`) entries.
#[inline(always)]
unsafe fn dotp<const N: usize>(a: *const f64, b: *const f64, n: usize) -> f64 {
    let n = if N > 0 { N } else { n };
    let mut s = 0.0;
    for i in 0..n {
        s += *a.add(i) * *b.add(i);
    }
    s
}

/// `y[i] += a x[i]` over `N` (or, for `N = 0`, `n`) entries.
#[inline(always)]
unsafe fn axpyp<const N: usize>(y: *mut f64, a: f64, x: *const f64, n: usize) {
    let n = if N > 0 { N } else { n };
    for i in 0..n {
        *y.add(i) += a * *x.add(i);
    }
}

/// One quartet `(μν|λσ)` of the pair digest. `N4 > 0` fixes `n4` at compile
/// time; `J` adds the Coulomb terms; `NK` exchange channels; `GEN` the
/// general-density form (the ket-row images in `Kt` against `Dᵀ`).
///
/// Coulomb: `J_bra[μν] += Σ_λσ v E34[λσ]` and, for distinct bra and ket
/// pairs, `Jh[λσ] += w34 Σ_μν v E12[μν]` (`w34 = ½` for `s3 = s4`, whose
/// transpose image `Jh + Jhᵀ` would otherwise double).
///
/// Exchange, rows μ and ν only: `Kb[μλ] += w D[νσ] v`, `Kb[νλ] += w D[μσ] v`,
/// `Kb[μσ] += w D[νλ] v`, `Kb[νσ] += w D[μλ] v` (each where the 8-fold loop
/// writes that image). The ket-row images `K[λμ] += D[σν] v` … are the
/// transposes of these for a symmetric `D` (`K = Kb + Kbᵀ`); the
/// self-paired quartet `(s3 s4) = (s1 s2)` has none, so it takes `w = ½`
/// there. The general form accumulates them, transposed, into `Kt` against
/// `Dᵀ` (`K = Kb + Ktᵀ`), with `w = 1`.
///
/// # Safety
/// See the call site in [`PairJk::quartet_t`].
#[inline(always)]
unsafe fn pair_quartet<const N4: usize, const J: bool, const NK: usize, const GEN: bool>(
    q: *const f64,
    g: &PairGeom,
    p: &PairPtrs,
    sc: &PairScratch,
) {
    let [n1, n2, n3, n4] = g.n;
    let [o1, o2, o3, o4] = g.o;
    let n4 = if N4 > 0 { N4 } else { n4 };
    let n = p.n;
    let (sym12, sym34, sym1234) = (g.sym12, g.sym34, g.sym1234);
    let n34 = n3 * n4;
    if J {
        for c in 0..n3 {
            for dd in 0..n4 {
                let mut v = *p.jd.add((o3 + c) * n + o4 + dd);
                if sym34 {
                    v += *p.jd.add((o4 + dd) * n + o3 + c);
                }
                *sc.e34.add(c * n4 + dd) = v;
            }
        }
        if sym1234 {
            for i in 0..n34 {
                *sc.s34.add(i) = 0.0;
            }
        }
    }
    let w = if GEN || sym1234 { 1.0 } else { 0.5 };
    for a in 0..n1 {
        let mu = o1 + a;
        for b in 0..n2 {
            let nu = o2 + b;
            let ab = a * n2 + b;
            let e12 = if J { *sc.e12.add(ab) } else { 0.0 };
            let mut jacc = 0.0;
            for c in 0..n3 {
                let la = o3 + c;
                let r = q.add((ab * n3 + c) * n4);
                if J {
                    jacc += dotp::<N4>(r, sc.e34.add(c * n4), n4);
                    if sym1234 {
                        axpyp::<N4>(sc.s34.add(c * n4), e12, r, n4);
                    }
                }
                for ch in 0..NK {
                    let (dmu, dnu) = (p.dk[ch].add(mu * n), p.dk[ch].add(nu * n));
                    let (kmu, knu) = (p.kb[ch].add(mu * n), p.kb[ch].add(nu * n));
                    *kmu.add(la) += w * dotp::<N4>(r, dnu.add(o4), n4);
                    if sym12 {
                        *knu.add(la) += w * dotp::<N4>(r, dmu.add(o4), n4);
                    }
                    if sym34 {
                        axpyp::<N4>(kmu.add(o4), w * *dnu.add(la), r, n4);
                        if sym12 {
                            axpyp::<N4>(knu.add(o4), w * *dmu.add(la), r, n4);
                        }
                    }
                    if GEN && sym1234 {
                        // Ket-row images K[λμ] += D[σν] v …, stored
                        // transposed in Kt's bra rows against Dᵀ's rows.
                        let (tmu, tnu) = (p.kt[ch].add(mu * n), p.kt[ch].add(nu * n));
                        let (dtmu, dtnu) = (p.dkt[ch].add(mu * n), p.dkt[ch].add(nu * n));
                        *tmu.add(la) += dotp::<N4>(r, dtnu.add(o4), n4);
                        if sym12 {
                            *tnu.add(la) += dotp::<N4>(r, dtmu.add(o4), n4);
                        }
                        if sym34 {
                            axpyp::<N4>(tmu.add(o4), *dtnu.add(la), r, n4);
                            if sym12 {
                                axpyp::<N4>(tnu.add(o4), *dtmu.add(la), r, n4);
                            }
                        }
                    }
                }
            }
            if J {
                *sc.jb.add(ab) += jacc;
            }
        }
    }
    if J && sym1234 {
        let w34 = if sym34 { 1.0 } else { 0.5 };
        for c in 0..n3 {
            for dd in 0..n4 {
                *p.jh.add((o3 + c) * n + o4 + dd) += w34 * *sc.s34.add(c * n4 + dd);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 4-shell density-max table with a deliberate asymmetry: the J pairings
    /// (d12, d34) are LARGE while the four exchange pairings are small. This is
    /// exactly the configuration on which `FourPairK` and `SixPair` must
    /// disagree, and on which using `FourPairK` for a J build would lose a
    /// significant Coulomb contribution.
    fn j_heavy_table() -> Array2<f64> {
        let mut t = Array2::<f64>::from_elem((4, 4), 1e-9);
        // d12 / d34 (and their symmetric partners): large.
        t[(0, 1)] = 1.0;
        t[(1, 0)] = 1.0;
        t[(2, 3)] = 1.0;
        t[(3, 2)] = 1.0;
        t
    }

    /// The substantive property: on a J-heavy table the exchange-only key is
    /// orders of magnitude smaller than the six-pairing key.
    ///
    /// This is what makes `FourPairK` a real tightening rather than a relabel —
    /// and simultaneously why it is unsound for J. If this ever returns equal
    /// values the variant has stopped doing anything.
    #[test]
    fn four_pair_k_is_strictly_tighter_than_six_pair_on_a_j_heavy_quartet() {
        let t = j_heavy_table();
        let six = DensityScreen::SixPair(&t).dmax(0, 1, 2, 3);
        let four = DensityScreen::FourPairK(&t).dmax(0, 1, 2, 3);
        assert_eq!(
            six, 1.0,
            "SixPair must pick up the large d12/d34 J pairings"
        );
        assert_eq!(
            four, 1e-9,
            "FourPairK must see only the small exchange pairings"
        );
        assert!(
            four < six,
            "FourPairK ({four:e}) must be tighter than SixPair ({six:e})"
        );
    }

    /// `FourPairK <= SixPair` for EVERY quartet of a random-ish table — the
    /// validity direction. A max over a subset can never exceed a max over the
    /// superset, so a violation here means the pairing indices are wrong.
    #[test]
    fn four_pair_k_never_exceeds_six_pair() {
        let n = 5usize;
        let mut t = Array2::<f64>::zeros((n, n));
        let mut state: u64 = 0x9E3779B97F4A7C15;
        for i in 0..n {
            for j in 0..=i {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let v = (state >> 33) as f64 / u32::MAX as f64;
                t[(i, j)] = v;
                t[(j, i)] = v;
            }
        }
        for s1 in 0..n {
            for s2 in 0..n {
                for s3 in 0..n {
                    for s4 in 0..n {
                        let six = DensityScreen::SixPair(&t).dmax(s1, s2, s3, s4);
                        let four = DensityScreen::FourPairK(&t).dmax(s1, s2, s3, s4);
                        assert!(
                            four <= six,
                            "FourPairK {four} > SixPair {six} at ({s1},{s2},{s3},{s4}) — the \
                             exchange pairings are not a subset of the six, so the indices are wrong"
                        );
                    }
                }
            }
        }
    }

    /// `accumulates_j` must agree with what `add_j` actually writes.
    ///
    /// The `FourPairK` guard is only as good as this predicate: if a J-bearing
    /// mode ever reports `false`, the `debug_assert!` in `scatter_bra_pair`
    /// goes quiet and the under-screening it exists to catch ships silently.
    /// So this checks the predicate against OBSERVED behaviour — it drives
    /// `add_j` on each mode and asserts J moved exactly when the flag says it
    /// should — rather than restating the same match arms a second time.
    #[test]
    fn accumulates_j_matches_what_add_j_actually_writes() {
        let d = Array2::<f64>::from_elem((1, 1), 1.0);
        let d_a = Array2::<f64>::zeros((1, 1));
        let d_b = Array2::<f64>::zeros((1, 1));
        let stack = ndarray::Array3::<f64>::from_elem((1, 1, 2), 1.0);
        let none = ndarray::Array3::<f64>::zeros((1, 1, 0));

        for mut mode in [
            JkMode::new_j(1),
            JkMode::new_k(1),
            JkMode::new_both(1),
            JkMode::new_uhf(1, &d_a, &d_b),
            JkMode::new_multi(&stack, &stack),
            JkMode::new_multi(&none, &stack),
        ] {
            let flag = mode.accumulates_j();
            let label = mode.label();
            mode.add_j(0, 0, &d, 0, 0, 1.0);
            let wrote_j = match &mode {
                JkMode::JOnly(j) | JkMode::Both(j, _) | JkMode::Uhf(j, ..) => j[(0, 0)] != 0.0,
                JkMode::Multi(m) => m.j.iter().any(|&x| x != 0.0),
                JkMode::KOnly(_) => false,
                JkMode::Pair(_) => unreachable!("not in this list"),
            };
            assert_eq!(
                flag, wrote_j,
                "{label}: accumulates_j() = {flag} but add_j actually wrote J = {wrote_j}. The \
                 FourPairK guard in scatter_bra_pair depends on this predicate being exact."
            );
        }
    }

    // ---------------------------------------------------------------- pair

    #[test]
    fn jk_digest_spelling_is_strict() {
        assert_eq!(JkDigest::parse_str("pair"), Ok(JkDigest::Pair));
        assert_eq!(JkDigest::parse_str("legacy"), Ok(JkDigest::Legacy));
        for bad in ["Pair", " pair", "LEGACY", "off", "", "8fold"] {
            assert!(JkDigest::parse_str(bad).is_err(), "{bad:?} accepted");
        }
        assert_eq!(JkDigest::default(), JkDigest::Pair);
        // The variable ignores case and blanks, like the other switches.
        let get = |v: &'static str| move |_: &str| Some(v.to_string());
        let r = |v| JK_DIGEST.resolve(None, get(v)).map(|r| r.value);
        assert_eq!(r(" LEGACY "), Ok(JkDigest::Legacy));
        assert_eq!(r("pair"), Ok(JkDigest::Pair));
        assert!(r("none").is_err());
        // An empty or all-blank value is refused, as `FERRIC_GC_ROTATE`
        // refuses it (not read as unset).
        assert!(r("").is_err());
        assert!(r("   ").is_err());
        // A config value wins over the variable.
        let cfg = JK_DIGEST.resolve(Some(JkDigest::Pair), get("legacy"));
        assert_eq!(cfg.map(|r| r.value), Ok(JkDigest::Pair));
    }

    #[test]
    fn digest_slot_is_shared_by_clones_and_copied() {
        let a = DigestSlot::default();
        assert_eq!(a.recorded(), None);
        let b = a.clone();
        a.set(Some(JkDigest::Legacy));
        assert_eq!(b.recorded(), Some(JkDigest::Legacy));
        assert_eq!(b.mode().unwrap(), JkDigest::Legacy);
        let c = DigestSlot::default();
        c.copy_from(&b);
        assert_eq!(c.recorded(), Some(JkDigest::Legacy));
        a.set(None);
        assert_eq!(b.recorded(), None);
        assert_eq!(c.recorded(), Some(JkDigest::Legacy), "a copy is not shared");
    }

    fn sym_matrix(n: usize, seed: u64) -> Array2<f64> {
        let mut s = seed;
        let mut d = Array2::from_shape_fn((n, n), |_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        });
        d = &d + &d.t();
        d
    }

    /// The density check, relative for a full density: roundoff skew
    /// passes, a real skew fails at any size, an antisymmetric part of
    /// absolute size 1e-15 included (no absolute floor), and a non-finite
    /// element on either side fails.
    #[test]
    fn symmetry_check_is_at_the_roundoff_level() {
        let d = sym_matrix(9, 1);
        assert!(is_symmetric_to_roundoff(&d, None));
        let mut e = d.clone();
        e[(2, 5)] += 4e-16; // roundoff of an O(1) density
        assert!(is_symmetric_to_roundoff(&e, None));
        // The tolerance itself: 0.5e-14 relative passes, 2e-14 fails (a
        // tolerance off by a decade either way flips one of these).
        let big = d.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        for (rel, want) in [(0.5e-14, true), (2e-14, false)] {
            let mut x = d.clone();
            x[(2, 5)] += rel * big;
            assert_eq!(is_symmetric_to_roundoff(&x, None), want, "skew {rel:e}");
        }
        // Genuine skews.
        for skew in [1e-13, 1e-10, 1e-6, 1e-2] {
            let mut x = d.clone();
            x[(3, 1)] += skew;
            assert!(!is_symmetric_to_roundoff(&x, None), "skew {skew:e} passed");
        }
        // A big density with a 1e-13 relative skew.
        let mut big = 1e3 * &d;
        big[(0, 4)] += 1e3 * 1e-13 * 2.0;
        assert!(!is_symmetric_to_roundoff(&big, None));
        // Purely antisymmetric, of absolute size 1e-15: kept (general).
        let a = Array2::from_shape_fn((9, 9), |(i, j)| 1e-15 * (i as f64 - j as f64).signum());
        assert!(!is_symmetric_to_roundoff(&a, None));
        assert!(PairDensity::prepare(&a, None).is_general());
        // Non-finite on one side only, or on both.
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut x = d.clone();
            x[(1, 4)] = bad;
            assert!(!is_symmetric_to_roundoff(&x, None), "{bad} on one side");
            x[(4, 1)] = bad;
            assert!(!is_symmetric_to_roundoff(&x, None), "{bad} on both sides");
            assert!(
                !is_symmetric_to_roundoff(&x, Some(1.0)),
                "{bad} with a reference"
            );
        }
        assert!(!is_symmetric_to_roundoff(&Array2::zeros((2, 3)), None));
        assert!(is_symmetric_to_roundoff(&Array2::zeros((3, 3)), None));
    }

    /// The incremental `ΔD`: against the full density's scale, a 1e-17
    /// skew on a 1e-7 `ΔD` stays symmetric; on its own scale (a full
    /// build) it does not; a skew of 1e-13 of the reference does not either.
    #[test]
    fn symmetry_check_of_a_delta_density_uses_the_reference() {
        let d = sym_matrix(9, 7);
        let reference = d.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let mut dd = 1e-7 * &d;
        dd[(2, 5)] += 1e-17;
        assert!(is_symmetric_to_roundoff(&dd, Some(reference)));
        assert!(!is_symmetric_to_roundoff(&dd, None));
        let refs = [reference];
        let spec = PairSpec::both(9, &[3, 3, 3], &dd, SymScale::Delta(&refs));
        assert!(!spec.is_general());
        assert!(PairSpec::both(9, &[3, 3, 3], &dd, SymScale::Own).is_general());
        // No reference recorded for the channel: its own scale.
        assert!(PairSpec::both(9, &[3, 3, 3], &dd, SymScale::Delta(&[])).is_general());
        dd[(2, 5)] += 1e-13 * reference;
        assert!(!is_symmetric_to_roundoff(&dd, Some(reference)));
        // The recorded scale is max|D| per channel.
        assert_eq!(
            PairSpec::both(9, &[3, 3, 3], &d, SymScale::Own).scales(),
            vec![reference]
        );
    }

    #[test]
    fn pair_density_symmetrises_or_keeps_the_transpose() {
        let mut d = sym_matrix(6, 2);
        d[(1, 4)] += 1e-17;
        let p = PairDensity::prepare(&d, None);
        assert!(!p.is_general());
        assert!(
            p.d.get() == p.d.get().t(),
            "symmetrised density must be exactly symmetric"
        );
        d[(1, 4)] += 1e-3;
        let p = PairDensity::prepare(&d, None);
        assert!(p.is_general());
        assert_eq!(p.d.get(), &d);
        assert_eq!(
            p.d.owned_len(),
            0,
            "a standard-layout density is not copied"
        );
        assert_eq!(p.dt.as_ref().unwrap(), &d.t());
        assert_eq!(p.owned_len(), 36);
        // A column-major density is copied into standard layout.
        let f = d.t().to_owned();
        assert!(!f.is_standard_layout());
        let p = PairDensity::prepare(&f, None);
        assert!(p.d.owned_len() == 36 && p.d.get().is_standard_layout());
        assert_eq!(p.d.get(), &d.t());
    }

    /// One non-symmetric spin sends both channels to the general kernel, and
    /// the partial layout follows (`Jh`, `Kb` per channel, `Kt` per channel).
    #[test]
    fn pair_spec_layout_and_kernel_choice() {
        let dims = [1usize, 3, 5];
        let own = SymScale::Own;
        let (s, a) = (sym_matrix(9, 3), sym_matrix(9, 4));
        let mut g = a.clone();
        g[(0, 8)] += 1e-4;
        let t = &s + &g;
        let sa = &s + &a;
        assert!(!PairSpec::both(9, &dims, &s, own).is_general());
        assert_eq!(PairSpec::both(9, &dims, &s, own).n_partials(), 2);
        assert_eq!(
            PairSpec::both(9, &dims, &s, own).kernel_label(),
            "symmetric"
        );
        let both_g = PairSpec::both(9, &dims, &g, own);
        assert!(both_g.is_general());
        assert_eq!(both_g.n_partials(), 3);
        assert_eq!(both_g.kernel_label(), "general [nonsym]");
        assert_eq!(PairSpec::uhf(9, &dims, &sa, &s, &a, own).n_partials(), 3);
        let uhf_g = PairSpec::uhf(9, &dims, &t, &s, &g, own);
        assert!(uhf_g.is_general());
        assert_eq!(uhf_g.n_partials(), 5);
        assert_eq!(uhf_g.kernel_label(), "general [sym, nonsym]");
        assert_eq!(PairSpec::j_only(9, &dims, &g).n_partials(), 1);
        assert!(!PairSpec::j_only(9, &dims, &g).is_general());
        assert_eq!(
            PairSpec::j_only(9, &dims, &g).kernel_label(),
            "none (J only)"
        );
        assert_eq!(PairSpec::k_only(9, &dims, &g).n_partials(), 2);
        let mut k_spec = PairSpec::k_only(9, &dims, &s);
        k_spec.materialize();
        let m = k_spec.mode();
        assert!(!m.accumulates_j());
        assert_eq!(m.label(), "Pair");
        let mut j_spec = PairSpec::j_only(9, &dims, &s);
        j_spec.materialize();
        assert!(j_spec.mode().accumulates_j());
        // The closed-shell J reads the prepared exchange density: no copy.
        let both = PairSpec::both(9, &dims, &g, own);
        assert!(std::ptr::eq(both.jd().unwrap().as_ptr(), g.as_ptr()));
        assert_eq!(
            both.charged_bytes(),
            (81 + 3 * 81) * 8,
            "Dᵀ and three partials"
        );
        let both = PairSpec::both(9, &dims, &s, own);
        assert_eq!(
            both.charged_bytes(),
            (81 + 2 * 81) * 8,
            "½(D+Dᵀ) and two partials"
        );
        // The open-shell J of a standard-layout total is borrowed.
        let mut uhf = PairSpec::uhf(9, &dims, &sa, &s, &a, own);
        uhf.materialize();
        assert!(std::ptr::eq(uhf.jd().unwrap().as_ptr(), sa.as_ptr()));
    }

    /// The copies [`PairSpec::charged_bytes`] counts are made only by
    /// `materialize` (after the build's reservation): a planned spec holds
    /// none, and the charge read before agrees with what is held after.
    #[test]
    fn pair_spec_allocates_its_copies_only_when_materialized() {
        let dims = [1usize, 3, 5];
        let own = SymScale::Own;
        let s = sym_matrix(9, 3);
        let mut g = s.clone();
        g[(0, 8)] += 1e-4;
        let f = s.t().to_owned();
        assert!(!f.is_standard_layout());
        let held = |spec: &PairSpec<'_>| {
            let j = match &spec.jd {
                PairJd::Own { mat, .. } => mat.owned_len(),
                _ => 0,
            };
            j + spec
                .kd
                .iter()
                .map(|p| p.d.owned_len() + p.dt.as_ref().map_or(0, |t| t.len()))
                .sum::<usize>()
        };
        for mut spec in [
            PairSpec::both(9, &dims, &s, own),
            PairSpec::both(9, &dims, &g, own),
            PairSpec::uhf(9, &dims, &f, &s, &g, own),
            PairSpec::j_only(9, &dims, &f),
            PairSpec::k_only(9, &dims, &f),
        ] {
            assert_eq!(held(&spec), 0, "a planned spec holds no copy");
            let charged = spec.charged_bytes();
            spec.materialize();
            assert_eq!(
                (held(&spec) + spec.n_partials() * 81) * 8,
                charged,
                "{}",
                spec.kernel_label()
            );
        }
    }

    /// Without MPI the rank agreement is the slot's own resolution.
    #[test]
    fn digest_mode_collective_is_mode_at_one_rank() {
        let ctx = ferric_core::parallel::ParallelContext::default();
        let slot = DigestSlot::default();
        for d in [JkDigest::Pair, JkDigest::Legacy] {
            slot.set(Some(d));
            assert_eq!(slot.mode_collective(&ctx).unwrap(), d);
        }
        slot.set(None);
        assert_eq!(
            slot.mode_collective(&ctx).ok(),
            JkDigest::env_setting().ok()
        );
    }

    /// A density that is not `nbf × nbf` is refused before the kernel
    /// (which indexes it with stride `nbf`) or any allocation.
    #[test]
    fn pair_spec_refuses_a_density_of_the_wrong_shape() {
        let ctx = ferric_core::parallel::ParallelContext::default();
        let dims = [1usize, 1, 1];
        let run = |_: usize, _: &mut JkMode<'_>| unreachable!("no group runs");
        let bad = Array2::<f64>::zeros((2, 3));
        let ok = sym_matrix(3, 1);
        let wide = Array2::<f64>::zeros((3, 4));
        for mut spec in [
            PairSpec::both(3, &dims, &bad, SymScale::Own),
            PairSpec::k_only(3, &dims, &wide),
            PairSpec::j_only(3, &dims, &bad),
            PairSpec::uhf(3, &dims, &bad, &ok, &ok, SymScale::Own),
            PairSpec::uhf(3, &dims, &ok, &ok, &wide, SymScale::Own),
            PairSpec::both(2, &[1, 1], &ok, SymScale::Own),
        ] {
            let e = spec.sum_groups(&ctx, 1, 1 << 20, None, run).unwrap_err();
            assert!(e.to_string().contains("density shape"), "{e}");
            assert!(!spec.prepared);
        }
    }

    /// The symmetrised density of a finite matrix stays finite where the
    /// plain `½(a + b)` would overflow, and is that expression bit for bit
    /// wherever the sum is finite.
    #[test]
    fn symmetrising_does_not_overflow() {
        let d = Array2::from_elem((1, 1), 1e308);
        let p = PairDensity::prepare(&d, None);
        assert_eq!(p.d.get()[(0, 0)], 1e308);
        let mut d = Array2::from_elem((2, 2), f64::MAX);
        d[(0, 1)] = f64::MAX * (1.0 - 1e-15);
        d[(1, 0)] = f64::MAX;
        let p = PairDensity::prepare(&d, None);
        assert!(!p.is_general());
        assert!(p.d.get().iter().all(|v| v.is_finite()));
        assert_eq!(p.d.get()[(0, 0)], f64::MAX);
        for (a, b) in [(0.1, 0.3), (1e-300, 3e-310), (-2.5, 7.0), (5e-324, 5e-324)] {
            assert_eq!(midpoint(a, b).to_bits(), (0.5 * (a + b)).to_bits());
        }
    }

    /// `finish`: `J = Jh + Jhᵀ`, `K = Kb + Kbᵀ` (symmetric) or `Kb + Ktᵀ`.
    #[test]
    fn pair_finish_forms_the_full_matrices() {
        let dims = [3usize];
        let s = sym_matrix(3, 5);
        let mut g = s.clone();
        g[(0, 2)] += 1.0;
        let h = Array2::from_shape_fn((3, 3), |(i, j)| (3 * i + j) as f64);
        let k2 = Array2::from_shape_fn((3, 3), |(i, j)| (i as f64) - 2.0 * j as f64);
        let spec = PairSpec::both(3, &dims, &s, SymScale::Own);
        let (j, k) = spec.finish(vec![h.clone(), k2.clone()]);
        assert_eq!(j.unwrap(), &h + &h.t());
        assert_eq!(k[0], &k2 + &k2.t());
        let spec = PairSpec::both(3, &dims, &g, SymScale::Own);
        let (_, k) = spec.finish(vec![h.clone(), k2.clone(), h.clone()]);
        assert_eq!(k[0], &k2 + &h.t());
    }

    /// The guard must actually FIRE on the invalid combination.
    ///
    /// Repo rule: a test you have never seen fail is an assumption. This is the
    /// mutation test for the `debug_assert!` itself, expressed as the predicate
    /// the assert evaluates — proving the condition is REACHABLE rather than
    /// vacuously true. (It checks the predicate rather than calling
    /// `scatter_bra_pair`, which would need a live libint2 `Engine`; the assert
    /// is that function's first statement, so the predicate is the whole of its
    /// logic.)
    #[test]
    fn four_pair_k_with_a_j_mode_is_detected_as_invalid() {
        let t = j_heavy_table();
        let screen = DensityScreen::FourPairK(&t);
        let d_a = Array2::<f64>::zeros((1, 1));
        let d_b = Array2::<f64>::zeros((1, 1));

        let invalid =
            |m: &JkMode<'_>| matches!(screen, DensityScreen::FourPairK(_)) && m.accumulates_j();

        assert!(
            invalid(&JkMode::new_j(1)),
            "FourPairK + JOnly must be flagged invalid"
        );
        assert!(
            invalid(&JkMode::new_both(1)),
            "FourPairK + Both must be flagged invalid"
        );
        assert!(
            invalid(&JkMode::new_uhf(1, &d_a, &d_b)),
            "FourPairK + Uhf must be flagged invalid"
        );
        // ...and must NOT fire on the one legitimate combination.
        assert!(
            !invalid(&JkMode::new_k(1)),
            "FourPairK + KOnly is the intended use and must not be flagged"
        );
    }
}
