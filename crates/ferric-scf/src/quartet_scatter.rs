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

use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ndarray::Array2;

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
    Uhf(Array2<f64>, Array2<f64>, Array2<f64>, &'d Array2<f64>, &'d Array2<f64>),
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
        }
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
#[allow(clippy::too_many_arguments)]
pub(crate) fn scatter_bra_pair(
    engine: &mut Engine,
    prep: &PreparedBasis,
    dims: &[usize],
    offs: &[usize],
    q_table: &Array2<f64>,
    screen: &DensityScreen,
    thresh: f64,
    d: &Array2<f64>,
    s1: usize,
    s2: usize,
    mode: &mut JkMode<'_>,
    check_interrupt: bool,
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
    let b12 = q_table[(s1, s2)];
    let (n1, n2) = (dims[s1], dims[s2]);
    let (o1, o2) = (offs[s1], offs[s2]);
    let sym12 = s1 != s2;

    for s3 in 0..=s1 {
        if check_interrupt
            && s3 % 100 == 0
            && ferric_core::INTERRUPT.load(Ordering::Relaxed)
        {
            return local_count;
        }
        let s4max = if s3 == s1 { s2 } else { s3 };
        for s4 in 0..=s4max {
            let b34 = q_table[(s3, s4)];
            let dmax = screen.dmax(s1, s2, s3, s4);
            if b12 * b34 * dmax < thresh {
                continue;
            }

            if let Some(q) = engine.compute_quartet(prep, s1, s2, s3, s4) {
                local_count += 1;
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

                // General path for larger shells.
                for a in 0..n1 {
                    for b in 0..n2 {
                        for c in 0..n3 {
                            for dd in 0..n4 {
                                // SAFETY: flat index < n1*n2*n3*n4 by loop bounds; q has that many elements from the engine.
                                let v = unsafe {
                                    *q.get_unchecked(((a * n2 + b) * n3 + c) * n4 + dd)
                                };
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
    local_count
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
        assert_eq!(six, 1.0, "SixPair must pick up the large d12/d34 J pairings");
        assert_eq!(four, 1e-9, "FourPairK must see only the small exchange pairings");
        assert!(four < six, "FourPairK ({four:e}) must be tighter than SixPair ({six:e})");
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
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
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

        for mut mode in [
            JkMode::new_j(1),
            JkMode::new_k(1),
            JkMode::new_both(1),
            JkMode::new_uhf(1, &d_a, &d_b),
        ] {
            let flag = mode.accumulates_j();
            let label = mode.label();
            mode.add_j(0, 0, &d, 0, 0, 1.0);
            let wrote_j = match &mode {
                JkMode::JOnly(j) | JkMode::Both(j, _) | JkMode::Uhf(j, ..) => j[(0, 0)] != 0.0,
                JkMode::KOnly(_) => false,
            };
            assert_eq!(
                flag, wrote_j,
                "{label}: accumulates_j() = {flag} but add_j actually wrote J = {wrote_j}. The \
                 FourPairK guard in scatter_bra_pair depends on this predicate being exact."
            );
        }
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

        let invalid = |m: &JkMode<'_>| matches!(screen, DensityScreen::FourPairK(_)) && m.accumulates_j();

        assert!(invalid(&JkMode::new_j(1)), "FourPairK + JOnly must be flagged invalid");
        assert!(invalid(&JkMode::new_both(1)), "FourPairK + Both must be flagged invalid");
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
