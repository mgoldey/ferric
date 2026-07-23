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
//!   integral (see [`Accumulate`] / [`JkMode`]).
//! - **The density screen**: `build_jk`/`DirectJK` use the full six-pairwise
//!   `d_max_shell` table (`max(d12,d34,d13,d14,d23,d24)`); `DirectJ`/`DirectK`
//!   use a single global `max_d` scalar. Parameterized via [`DensityScreen`]
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
pub enum JkMode {
    JOnly(Array2<f64>),
    KOnly(Array2<f64>),
    Both(Array2<f64>, Array2<f64>),
}

impl JkMode {
    pub fn new_j(nbf: usize) -> Self {
        JkMode::JOnly(Array2::zeros((nbf, nbf)))
    }
    pub fn new_k(nbf: usize) -> Self {
        JkMode::KOnly(Array2::zeros((nbf, nbf)))
    }
    pub fn new_both(nbf: usize) -> Self {
        JkMode::Both(Array2::zeros((nbf, nbf)), Array2::zeros((nbf, nbf)))
    }

    /// `local_j[(row,col)] += d[(la,sg)] * v` — no-op when this mode has no J.
    #[inline(always)]
    fn add_j(&mut self, row: usize, col: usize, d: &Array2<f64>, la: usize, sg: usize, v: f64) {
        match self {
            JkMode::JOnly(j) | JkMode::Both(j, _) => unsafe {
                *j.uget_mut((row, col)) += d.uget((la, sg)) * v;
            },
            JkMode::KOnly(_) => {}
        }
    }

    /// `local_k[(row,col)] += d[(la,sg)] * v` — no-op when this mode has no K.
    #[inline(always)]
    fn add_k(&mut self, row: usize, col: usize, d: &Array2<f64>, la: usize, sg: usize, v: f64) {
        match self {
            JkMode::KOnly(k) | JkMode::Both(_, k) => unsafe {
                *k.uget_mut((row, col)) += d.uget((la, sg)) * v;
            },
            JkMode::JOnly(_) => {}
        }
    }
}

/// The Häser-Ahlrichs density-weighted pair screen, parameterized so
/// `build_jk`/`DirectJK` (six-pairwise `d_max_shell` table) and
/// `DirectJ`/`DirectK` (single global `max_d` scalar) keep their existing,
/// distinct screening semantics exactly.
pub enum DensityScreen<'a> {
    /// `dmax = max(d12, d34, d13, d14, d23, d24)` from the shell-blocked
    /// `d_max_shell` table (build_jk / DirectJK).
    SixPair(&'a Array2<f64>),
    /// A single global `max |D_μν|` scalar (DirectJ / DirectK).
    Global(f64),
}

impl<'a> DensityScreen<'a> {
    /// `dmax` for a given (s1,s2,s3,s4) shell quartet, matching each caller's
    /// existing formula exactly.
    #[inline(always)]
    fn dmax(&self, s1: usize, s2: usize, s3: usize, s4: usize) -> f64 {
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
            DensityScreen::Global(m) => *m,
        }
    }
}

/// Build the `d_max_shell[(si,sj)] = max|D_μν|` table over shell blocks
/// (μ ∈ si, ν ∈ sj), shared by every caller of [`DensityScreen::SixPair`].
pub fn build_d_max_shell(prep: &PreparedBasis, d: &Array2<f64>) -> Array2<f64> {
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

/// The canonical (s1,s2) bra-pair work list `{(s1,s2) : 0<=s2<=s1<nsh}`,
/// shared by every caller (before any caller-specific bra-thresh
/// pre-filter or MPI striping is applied).
pub fn canonical_bra_pairs(nsh: usize) -> Vec<(usize, usize)> {
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
pub fn scatter_bra_pair(
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
    mode: &mut JkMode,
    check_interrupt: bool,
) -> usize {
    use std::sync::atomic::Ordering;

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
