//! Combined direct Coulomb + exchange (J+K) matrix construction from a single quartet pass.

use crate::screening::SchwarzBounds;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_core::parallel::ParallelContext;
use ndarray::Array2;

/// Is the open-shell INCREMENTAL (ΔD) Fock build enabled?
///
/// **Default OFF**, unlike the closed-shell `solve_rhf` path (which defaults
/// ON via `FERRIC_SCF_INCREMENTAL`). Opt in with `FERRIC_SCF_UHF_INCREMENTAL=1`.
///
/// # Why it is off, with the measurement
///
/// The implementation is correct — `build_uhf_incremental` reproduces a
/// from-scratch rebuild to ~1e-14, including the adversarial opposing-spin-delta
/// case (see the direct_jk.rs unit tests and `open_shell_combined_jk.rs`). It
/// simply does not PAY at any size measured here, because the
/// Häser-Ahlrichs screen only starts rejecting quartets once `max|ΔD|` drops
/// below ~1e-8, and an SCF run reaches that only in its last iteration or two:
///
/// ```text
/// (measured, CH3/cc-pVDZ, quartet counts; the DEFAULT density_conv is 1e-6)
///   density_conv 1e-6 : full 87120  incr 87120   -> 1.000x
///   density_conv 1e-8 : full 101640 incr 101636  -> 1.000x
///   density_conv 1e-10: full 123420 incr 122984  -> 1.004x
///   density_conv 1e-12: full 152460 incr 154530  -> 0.987x  (WORSE, +4 iters)
///
/// (measured, larger systems, default thresholds)
///   alkane_4 (+1,doublet)/cc-pVDZ: 21412773 -> 21201021 quartets = 1.010x
///   alkane_8 (+1,doublet)/6-31G:   90222585 -> 86214445 quartets = 1.046x
/// ```
///
/// Wall-clock showed no reliable win either: on this (shared, load-average-16)
/// box the alkane_8 timing moved between 1.006x and 1.740x across repeat runs
/// while the quartet counts were bit-stable, i.e. the timing spread was machine
/// contention, not signal. A 4.6% work reduction cannot produce a 74% wall
/// reduction; the large figures were artifacts and are not claimed.
///
/// The same measurement applies to the long-shipped CLOSED-shell incremental
/// path, which was found to give 1.000x on water/cc-pVDZ and methane/cc-pVDZ
/// and 1.016x on alkane_4/cc-pVDZ — so this is a property of the screen's
/// threshold scale, NOT of the open-shell port. That closed-shell default is
/// left exactly as it was (changing it is out of scope for this work), but it
/// is worth an independent look.
///
/// Keeping the code, tested and switchable, rather than deleting it: the
/// mechanism is sound and becomes profitable if `integral_thresh` is ever
/// loosened relative to `density_conv`, or on systems large enough for
/// locality to bite. Defaulting it ON today would buy a per-iteration density
/// clone and extra SCF-loop state for ~1% fewer quartets.
pub fn open_shell_incremental_enabled() -> bool {
    matches!(
        std::env::var("FERRIC_SCF_UHF_INCREMENTAL").ok().as_deref(),
        Some("1") | Some("on") | Some("true") | Some("ON") | Some("TRUE")
    )
}

/// Is the combined single-pass open-shell J+K build enabled?
///
/// Default ON. `FERRIC_SCF_COMBINED_JK=0` (or `off`/`false`) restores the
/// historical three-pass open-shell Fock build (`DirectJ(D_total)` +
/// `DirectK(D_α)` + `DirectK(D_β)`) in `solve_uhf`/`solve_rohf`, so the two can
/// be compared for correctness and timing in one binary. Same string
/// convention as `FERRIC_SCF_INCREMENTAL`.
pub fn combined_open_shell_jk_enabled() -> bool {
    !matches!(
        std::env::var("FERRIC_SCF_COMBINED_JK").ok().as_deref(),
        Some("0") | Some("off") | Some("false") | Some("OFF") | Some("FALSE")
    )
}

/// Combined Coulomb (J) and exchange (K) matrix builder.
///
/// Iterates all screened canonical (s1,s2,s3,s4) quartets **once** and accumulates
/// both J and K from the same computed integral, avoiding the 2× ERI evaluation cost
/// of calling DirectJ and DirectK separately.
///
/// The work list is the screened bra-pair list (O(nsh²) memory); the ket loop and
/// the Häser-Ahlrichs density screen run inside the parallel group workers, with
/// ~n_pairs/1024 pairs per group amortizing load imbalance across rayon's
/// dynamic scheduling (same structure as `rhf::build_jk`). The former flat
/// quartet pre-enumeration was a serial O(nsh⁴) loop with an unbounded
/// surviving-quartet Vec reallocated every SCF iteration.
pub struct DirectJK<'a> {
    ctx: &'a ParallelContext,
    prep: &'a PreparedBasis,
    bounds: &'a SchwarzBounds,
    thresh: f64,
    /// Fully-resolved unified memory budget (TOML > env > auto), passed in by
    /// the solver — caps the reduction band scratch via `resolve_band_bytes`.
    mem_budget: usize,
    // Lazily built on first build() and reused for the builder's lifetime:
    // libint2 engine construction is serialized behind a global ctor mutex,
    // so hoist the builder out of the SCF loop to pay it once, not per iteration.
    pool: Option<crate::engine_pool::EnginePool>,
}

impl<'a> std::fmt::Debug for DirectJK<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DirectJK")
            .field("thresh", &self.thresh)
            .field("mem_budget", &self.mem_budget)
            .finish_non_exhaustive()
    }
}

impl<'a> DirectJK<'a> {
    /// Create a combined screened J+K builder.
    pub fn new(
        ctx: &'a ParallelContext,
        prep: &'a PreparedBasis,
        bounds: &'a SchwarzBounds,
        thresh: f64,
        mem_budget: usize,
    ) -> Self {
        DirectJK { ctx, prep, bounds, thresh, mem_budget, pool: None }
    }

    /// Incremental Fock build: given the DENSITY CHANGE `delta_d = D_new - D_last`,
    /// ACCUMULATE `ΔJ = J(delta_d)` and `ΔK = K(delta_d)` onto the caller's
    /// existing `j`/`k` buffers (which must already hold `J(D_last)`/`K(D_last)`),
    /// yielding `J(D_new)`/`K(D_new)`. The caller must NOT zero `j`/`k` first.
    ///
    /// This is mathematically EXACT (not an approximation): J and K are linear in
    /// D, so `J(D_last) + J(ΔD) == J(D_last + ΔD) == J(D_new)` in infinite
    /// precision. In f64 it differs from a from-scratch `build(&D_new, ..)` only
    /// by the reassociation floor of summing many small increments vs one full
    /// contraction — kept small by a periodic full rebuild in the SCF loop (see
    /// `solve_rhf`'s `incremental_full_rebuild_every` guard). The screen inside
    /// `build_d_max_shell` is driven by `delta_d`, so as SCF converges (ΔD → 0)
    /// almost every quartet is Häser-Ahlrichs-screened out and late iterations
    /// become nearly free (the PySCF `pyscf/scf/hf.py` / Psi4 `CompositeJK.cc`
    /// incremental-Fock scheme).
    ///
    /// Bit-identity across `RAYON_NUM_THREADS` is preserved: this shares the exact
    /// same deterministic grouped reduction as `build` — only the input density
    /// (a delta vs the full D) and the caller's zero-vs-accumulate choice differ.
    pub fn build_incremental(
        &mut self,
        delta_d: &Array2<f64>,
        j: &mut Array2<f64>,
        k: &mut Array2<f64>,
    ) -> Result<usize, FerricError> {
        // `build` already accumulates `J(d)`/`K(d)` onto the caller's buffers via
        // `*j += &total_j; *k += &total_k`, and screens with `build_d_max_shell(d)`.
        // Passing `delta_d` (without the caller zeroing) is therefore exactly the
        // incremental update — no separate kernel needed.
        self.build(delta_d, j, k)
    }

    /// The screened canonical bra-pair work list for a given shell-blocked
    /// density-max table, plus the lazily-constructed engine pool.
    ///
    /// Factored out of [`build`](Self::build) / [`build_uhf`](Self::build_uhf)
    /// because both need the IDENTICAL list construction: the `bra_thresh`
    /// pre-filter, the MPI rank striping, and the one-engine-per-rayon-thread
    /// pool hoist. Keeping it in one place is what makes the open-shell path
    /// screen exactly as tightly as the closed-shell one instead of drifting.
    fn screened_bra_pairs(
        &mut self,
        d_max_shell: &ndarray::Array2<f64>,
    ) -> Result<Vec<(usize, usize)>, FerricError> {
        let nsh = self.prep.nshells();
        let max_d = d_max_shell.iter().cloned().fold(0.0f64, f64::max);
        let max_q: f64 = self.bounds.q.iter().cloned().fold(0.0f64, f64::max);
        let bra_thresh =
            if max_q > 0.0 { self.thresh / (max_q * max_d.max(1e-30)) } else { self.thresh };
        let q_table = &self.bounds.q;
        let mut shell_pairs: Vec<(usize, usize)> = Vec::new();
        for s1 in 0..nsh {
            for s2 in 0..=s1 {
                if q_table[(s1, s2)] > bra_thresh {
                    shell_pairs.push((s1, s2));
                }
            }
        }
        // MPI rank striping (see `ParallelContext::stripe` doc; size == 1
        // keeps the full list unchanged).
        let shell_pairs: Vec<(usize, usize)> = self.ctx.stripe(shell_pairs);

        // One engine per rayon thread (see engine_pool): constructing it in the
        // fold init below would fire once per work-chunk and storm the global
        // libint2 ctor mutex (catastrophic for heavy-element bases).
        if self.pool.is_none() {
            self.pool = Some(crate::engine_pool::EnginePool::new(self.bounds.op, self.prep, 1e-14)?);
        }
        Ok(shell_pairs)
    }

    /// Open-shell combined build: ONE quartet pass producing `J[D_α+D_β]`,
    /// `K[D_α]` and `K[D_β]`, ACCUMULATED onto the caller's three buffers.
    ///
    /// This replaces the previous open-shell three-pass Fock build
    /// (`DirectJ(D_total)` + `DirectK(D_α)` + `DirectK(D_β)`), which evaluated
    /// every surviving shell quartet THREE times and screened each pass on the
    /// loose single global `max|D|` scalar. Here each quartet's integral is
    /// computed once and scattered into all three matrices, under the tight
    /// six-pairwise Häser-Ahlrichs shell screen the closed-shell path already
    /// used.
    ///
    /// The screening key is `|D_α| + |D_β|` elementwise — the cheapest key that
    /// is simultaneously an upper bound on all three contracted densities; see
    /// `quartet_scatter::build_d_max_shell_spin_sum` (private — a code span,
    /// not a doc link) for why the sum rather than the (tighter but unsound
    /// for J) max.
    ///
    /// `d_total` is passed explicitly rather than formed here so the caller's
    /// existing `&d_a + &d_b` temporary is reused, and so the INCREMENTAL path
    /// can pass `ΔD_total` — which is *not* recoverable from the two spin
    /// deltas without an extra allocation.
    ///
    /// Bit-identity across `RAYON_NUM_THREADS` holds by the same argument as
    /// [`build`](Self::build): the pair list, group partition and fold order are
    /// pure functions of the work list, never of the thread count.
    pub fn build_uhf(
        &mut self,
        d_total: &Array2<f64>,
        d_a: &Array2<f64>,
        d_b: &Array2<f64>,
        j: &mut Array2<f64>,
        k_a: &mut Array2<f64>,
        k_b: &mut Array2<f64>,
    ) -> Result<usize, FerricError> {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use crate::quartet_scatter::{
            build_d_max_shell_spin_sum, scatter_bra_pair, DensityScreen, JkMode,
        };

        self.ctx.check_interrupted()?;

        let dims = self.prep.shell_dims();
        let offs = self.prep.shell_offsets();
        let thresh = self.thresh;
        let computed_quartets = AtomicUsize::new(0);

        let d_max_shell = build_d_max_shell_spin_sum(self.prep, d_a, d_b);
        let shell_pairs = self.screened_bra_pairs(&d_max_shell)?;

        let q_table = &self.bounds.q;
        let prep = self.prep;
        let nbf = prep.nbasis();
        let pool = self.pool.as_ref().expect("pool initialized by screened_bra_pairs");

        let n_pairs = shell_pairs.len();
        let group_size = crate::reduce::deterministic_group_size(n_pairs);
        let n_groups = n_pairs.div_ceil(group_size.max(1)).max(1);
        let screen = DensityScreen::SixPair(&d_max_shell);

        let mut total_j = Array2::<f64>::zeros((nbf, nbf));
        let mut total_k_a = Array2::<f64>::zeros((nbf, nbf));
        let mut total_k_b = Array2::<f64>::zeros((nbf, nbf));
        let band_bytes = crate::reduce::resolve_band_bytes(self.mem_budget);
        crate::reduce::grouped_deterministic_sum_triple(
            &mut total_j,
            &mut total_k_a,
            &mut total_k_b,
            n_groups,
            nbf,
            band_bytes,
            |g| {
                let lo = g * group_size;
                let hi = (lo + group_size).min(n_pairs);
                let mut mode = JkMode::new_uhf(nbf, d_a, d_b);
                let mut local_count = 0usize;
                for &(s1, s2) in &shell_pairs[lo..hi] {
                    if ferric_core::INTERRUPT.load(Ordering::Relaxed) {
                        continue;
                    }
                    pool.with(|engine| {
                        local_count += scatter_bra_pair(
                            engine, prep, dims, offs, q_table, &screen, thresh, d_total, s1, s2,
                            &mut mode, true,
                        );
                    });
                }
                let (local_j, local_k_a, local_k_b) = match mode {
                    JkMode::Uhf(j, ka, kb, _, _) => (j, ka, kb),
                    _ => unreachable!("DirectJK::build_uhf always uses JkMode::Uhf"),
                };
                computed_quartets.fetch_add(local_count, Ordering::Relaxed);
                Ok((local_j, local_k_a, local_k_b))
            },
        )?;

        *j += &total_j;
        *k_a += &total_k_a;
        *k_b += &total_k_b;

        #[cfg(feature = "mpi")]
        if let Some(world) = self.ctx.world() {
            use mpi::traits::CommunicatorCollectives;
            for m in [&mut *j, &mut *k_a, &mut *k_b] {
                let mut global = Array2::zeros(m.dim());
                world.all_reduce_into(
                    m.as_slice().unwrap(),
                    global.as_slice_mut().unwrap(),
                    mpi::collective::SystemOperation::sum(),
                );
                *m = global;
            }
        }

        Ok(computed_quartets.load(Ordering::SeqCst))
    }

    /// Incremental open-shell Fock build: given the per-spin density CHANGES
    /// (`Δ_total = ΔD_α + ΔD_β`), ACCUMULATE the corresponding `ΔJ`/`ΔK_α`/`ΔK_β`
    /// onto buffers that already hold the previous iteration's `J`/`K_α`/`K_β`.
    /// The caller must NOT zero them first.
    ///
    /// Exact by linearity, exactly as in the closed-shell
    /// [`build_incremental`](Self::build_incremental): J and K are linear in the
    /// density, so `K(D_last) + K(ΔD) == K(D_new)` in infinite precision, and in
    /// f64 the difference from a from-scratch rebuild is the reassociation floor
    /// — bounded by the caller's periodic full rebuild.
    ///
    /// ## Screening key (the open-shell-specific decision)
    ///
    /// The screen runs on `|ΔD_α| + |ΔD_β|`, NOT on `|ΔD_total|`. The two spin
    /// channels converge independently and routinely move in opposite
    /// directions — a spin-flip rearrangement can leave `ΔD_total ≈ 0` while
    /// `ΔD_α` and `ΔD_β` are both large. Screening on the total would then drop
    /// quartets that carry a large, real contribution to `ΔK_α` and `ΔK_β`
    /// individually, silently corrupting the spin Focks while the Coulomb term
    /// looked fine. The spin sum bounds all three deltas (triangle inequality),
    /// so it is safe for whichever channel is largest, and it costs nothing
    /// extra: as SCF converges BOTH channels go to zero together and the screen
    /// tightens just as hard as the closed-shell one.
    pub fn build_uhf_incremental(
        &mut self,
        delta_total: &Array2<f64>,
        delta_a: &Array2<f64>,
        delta_b: &Array2<f64>,
        j: &mut Array2<f64>,
        k_a: &mut Array2<f64>,
        k_b: &mut Array2<f64>,
    ) -> Result<usize, FerricError> {
        // `build_uhf` already ACCUMULATES onto the caller's buffers and screens
        // on the spin sum of whatever densities it is handed, so passing the
        // deltas (without the caller zeroing) IS the incremental update — the
        // same relationship `build_incremental` has to `build`.
        self.build_uhf(delta_total, delta_a, delta_b, j, k_a, k_b)
    }

    /// Build J and K matrices simultaneously from a single pass over shell quartets.
    /// Returns the number of unique quartets computed.
    ///
    /// ACCUMULATES onto `j`/`k` (`*j += J(d)`, `*k += K(d)`) — callers doing a
    /// full rebuild zero the buffers first; the incremental path
    /// ([`build_incremental`](Self::build_incremental)) passes `ΔD` and does not.
    pub fn build(
        &mut self,
        d: &Array2<f64>,
        j: &mut Array2<f64>,
        k: &mut Array2<f64>,
    ) -> Result<usize, FerricError> {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use crate::quartet_scatter::{build_d_max_shell, scatter_bra_pair, DensityScreen, JkMode};

        self.ctx.check_interrupted()?;

        let dims = self.prep.shell_dims();
        let offs = self.prep.shell_offsets();
        let thresh = self.thresh;
        let computed_quartets = AtomicUsize::new(0);

        // Shell-blocked density-max table d_max_shell[(si, sj)] = max |D_μν| over
        // (μ ∈ shell si, ν ∈ shell sj). The Häser-Ahlrichs density-weighted screen
        // uses the max of all six pair maxima (12,34,13,14,23,24) because J contracts
        // D against (λ,σ),(μ,ν) and K against (μ,λ),(μ,σ),(ν,λ),(ν,σ).
        let d_max_shell = build_d_max_shell(self.prep, d);

        // Work list = screened canonical bra pairs (s1,s2), O(nsh²) memory. The
        // ket (s3,s4) loop and the pair-wise density screen —
        //   sqrt(Q_{12}) * sqrt(Q_{34}) * D_max_pair, with
        //   D_max_pair = max(d12, d34, d13, d14, d23, d24)
        // — run inside the parallel group workers below. The previous flat
        // quartet pre-enumeration was a serial O(nsh⁴) loop rebuilt (and its
        // Vec reallocated) EVERY SCF iteration, with unbounded memory in the
        // surviving-quartet count (32 B/entry — GB-scale on large direct
        // jobs); the pair list is the same structure `rhf::build_jk` already
        // uses with this reduction, and moves the screening work onto the
        // rayon workers. Shared verbatim with `build_uhf` via
        // `screened_bra_pairs` (which also hoists the engine pool).
        let shell_pairs = self.screened_bra_pairs(&d_max_shell)?;

        let q_table = &self.bounds.q;
        let prep = self.prep;
        let nbf = prep.nbasis();
        let pool = self.pool.as_ref().expect("pool initialized by screened_bra_pairs");

        // Deterministic, memory-bounded reduction (see direct_k / reduce.rs). The
        // old `fold(..).reduce(..)` tree held one J and one K nbf² partial per
        // work-chunk (~2× the direct-K footprint, ~21 GB at 50-atom/aug-cc-pVTZ,
        // 32 threads) and combined them in a worker-count-dependent order. Group
        // the canonical quartet list, fold each group's (J,K) serially, and sum
        // group partials in strict group order — bit-identical across thread
        // counts, live set bounded to one byte-budgeted band.
        // Group partition is a pure function of the quartet list (never the
        // thread count): group boundaries set the floating-point association of
        // the per-group folds, so a thread-dependent partition would break
        // bit-identity across RAYON_NUM_THREADS.
        let n_pairs = shell_pairs.len();
        let group_size = crate::reduce::deterministic_group_size(n_pairs);
        let n_groups = n_pairs.div_ceil(group_size.max(1)).max(1);
        let screen = DensityScreen::SixPair(&d_max_shell);

        let mut total_j = Array2::<f64>::zeros((nbf, nbf));
        let mut total_k = Array2::<f64>::zeros((nbf, nbf));
        let band_bytes = crate::reduce::resolve_band_bytes(self.mem_budget);
        crate::reduce::grouped_deterministic_sum_pair(
            &mut total_j,
            &mut total_k,
            n_groups,
            nbf,
            band_bytes,
            |g| {
                let lo = g * group_size;
                let hi = (lo + group_size).min(n_pairs);
                let mut mode = JkMode::new_both(nbf);
                let mut local_count = 0usize;
                for &(s1, s2) in &shell_pairs[lo..hi] {
                    if ferric_core::INTERRUPT.load(Ordering::Relaxed) {
                        continue;
                    }
                    pool.with(|engine| {
                        local_count += scatter_bra_pair(
                            engine, prep, dims, offs, q_table, &screen, thresh, d, s1, s2,
                            &mut mode, true,
                        );
                    });
                }
                let (local_j, local_k) = match mode {
                    JkMode::Both(j, k) => (j, k),
                    _ => unreachable!("DirectJK::build always uses JkMode::Both"),
                };
                computed_quartets.fetch_add(local_count, Ordering::Relaxed);
                Ok((local_j, local_k))
            },
        )?;

        *j += &total_j;
        *k += &total_k;

        #[cfg(feature = "mpi")]
        if let Some(world) = self.ctx.world() {
            use mpi::traits::CommunicatorCollectives;
            let mut j_global = Array2::zeros(j.dim());
            let mut k_global = Array2::zeros(k.dim());
            world.all_reduce_into(
                j.as_slice().unwrap(),
                j_global.as_slice_mut().unwrap(),
                mpi::collective::SystemOperation::sum(),
            );
            world.all_reduce_into(
                k.as_slice().unwrap(),
                k_global.as_slice_mut().unwrap(),
                mpi::collective::SystemOperation::sum(),
            );
            *j = j_global;
            *k = k_global;
        }

        Ok(computed_quartets.load(Ordering::SeqCst))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::direct_j::DirectJ;
    use crate::direct_k::DirectK;
    use crate::fock::{JBuilder, KBuilder};
    use ferric_core::basis;
    use ferric_core::mol::Molecule;
    use ferric_integrals::operator::Operator;

    /// Regression guard for the grouped deterministic reduction: J and K from
    /// the direct builders (DirectJK combined, DirectJ, DirectK) must be
    /// bit-identical regardless of the rayon worker count. The old
    /// `fold(..).reduce(..)` trees combined per-chunk partials in a
    /// worker-count-dependent order, so J/K (and the SCF energy) drifted ~µHa
    /// with RAYON_NUM_THREADS. The group partition is a pure function of the
    /// work list and group partials are summed in ascending group order, so the
    /// result cannot depend on the thread count.
    /// Two open-shell test densities (α ≠ β, both symmetric) plus their sum,
    /// on H2O/cc-pVDZ. Shared by the `build_uhf` anchors below.
    fn uhf_test_densities(
        n: usize,
    ) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
        let mut d_a = Array2::<f64>::zeros((n, n));
        let mut d_b = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                d_a[(i, j)] = 0.01 * ((i * 7 + j * 3) % 11) as f64;
                // Deliberately DIFFERENT structure, so any code path that
                // silently used one spin density for both K matrices (or the
                // total for either) is caught.
                d_b[(i, j)] = 0.007 * ((i * 5 + j * 13) % 9) as f64 - 0.002;
            }
        }
        let d_a = 0.5 * (&d_a + &d_a.t());
        let d_b = 0.5 * (&d_b + &d_b.t());
        let d_total = &d_a + &d_b;
        (d_a, d_b, d_total)
    }

    /// EXACTNESS ANCHOR for the combined open-shell build.
    ///
    /// `build_uhf` must reproduce, to a tight numerical tolerance, exactly what
    /// the three-pass path it replaces produced: `J` from `DirectJ(D_total)`,
    /// `K_α` from `DirectK(D_α)`, `K_β` from `DirectK(D_β)`.
    ///
    /// The tolerance is not zero because this is deliberately NOT a
    /// bit-identity claim: the combined build screens on the six-pairwise
    /// spin-sum shell table while `DirectJ`/`DirectK` screen on a single global
    /// `max|D|` scalar, so the two admit different quartet SETS (the combined
    /// one is TIGHTER — it may legitimately drop quartets the loose screen
    /// kept). Screening is an approximation controlled by `thresh`, so the
    /// agreement bar is the screening threshold, not the f64 floor. Run at a
    /// very tight `thresh` (1e-14) both paths converge to the exact J/K.
    ///
    /// ARTIFACT HYPOTHESIS: if the α/β densities were crossed, or one spin
    /// density were used for both K buffers, K_α and K_β would come out equal
    /// (or swapped) — the asymmetric `d_a`/`d_b` above make that a loud
    /// failure rather than a silent pass.
    #[test]
    fn build_uhf_matches_three_pass_directj_plus_two_directk() {
        let mol =
            Molecule::parse_xyz("3\nH2O\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.26\n", 0, 1).unwrap();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let n = prep.nbasis();
        let (d_a, d_b, d_total) = uhf_test_densities(n);
        let ctx = ParallelContext::default();
        let thresh = 1e-14;

        // Reference: the three separate passes the open-shell solvers used to run.
        let mut j_ref = Array2::zeros((n, n));
        let mut ka_ref = Array2::zeros((n, n));
        let mut kb_ref = Array2::zeros((n, n));
        let mut dj = DirectJ::new(&ctx, &prep, &bounds, thresh, usize::MAX);
        <DirectJ as JBuilder>::build(&mut dj, &d_total, &mut j_ref).unwrap();
        let mut dk = DirectK::new(&ctx, &prep, &bounds, thresh, usize::MAX);
        <DirectK as KBuilder>::build(&mut dk, &d_a, &mut ka_ref).unwrap();
        <DirectK as KBuilder>::build(&mut dk, &d_b, &mut kb_ref).unwrap();

        // Combined: one pass.
        let mut j = Array2::zeros((n, n));
        let mut ka = Array2::zeros((n, n));
        let mut kb = Array2::zeros((n, n));
        let mut djk = DirectJK::new(&ctx, &prep, &bounds, thresh, usize::MAX);
        djk.build_uhf(&d_total, &d_a, &d_b, &mut j, &mut ka, &mut kb).unwrap();

        let max_abs_diff = |x: &Array2<f64>, y: &Array2<f64>| -> f64 {
            (x - y).iter().map(|v| v.abs()).fold(0.0f64, f64::max)
        };
        let tol = 1e-10;
        assert!(
            max_abs_diff(&j, &j_ref) < tol,
            "combined J differs from DirectJ(D_total) by {}",
            max_abs_diff(&j, &j_ref)
        );
        assert!(
            max_abs_diff(&ka, &ka_ref) < tol,
            "combined K_alpha differs from DirectK(D_alpha) by {}",
            max_abs_diff(&ka, &ka_ref)
        );
        assert!(
            max_abs_diff(&kb, &kb_ref) < tol,
            "combined K_beta differs from DirectK(D_beta) by {}",
            max_abs_diff(&kb, &kb_ref)
        );
        // The two spin exchange matrices must actually DIFFER — otherwise the
        // test above would pass just as well for a builder that ignored one of
        // the two densities.
        assert!(
            max_abs_diff(&ka, &kb) > 1e-3,
            "test densities failed to produce distinct K_alpha/K_beta — this \
             test cannot detect a crossed-density bug"
        );
    }

    /// EXACTNESS ANCHOR (trivial limit): in the CLOSED-SHELL limit `D_α = D_β`,
    /// the open-shell build must collapse onto the closed-shell one — both spin
    /// exchange matrices equal to `K(D_α)`, and `J` equal to `J(2·D_α)`.
    ///
    /// This is the "vacuous approximation" anchor the experimental protocol
    /// requires: it fixes the *relationship* between the new path and the
    /// long-trusted one, independent of any screening difference, and it is
    /// bit-exact because with `D_α = D_β` the spin-sum screen table equals
    /// `build_d_max_shell(2·D_α)`'s and the two builds traverse the same
    /// quartets in the same order.
    #[test]
    fn build_uhf_matches_closed_shell_build_when_spins_are_equal() {
        let mol =
            Molecule::parse_xyz("3\nH2O\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.26\n", 0, 1).unwrap();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let n = prep.nbasis();
        let (d_half, _, _) = uhf_test_densities(n);
        let d_total = &d_half + &d_half;
        let ctx = ParallelContext::default();
        let thresh = 1e-12;

        // Closed-shell reference: J(D_total) and K(D_half) from `build`.
        // (`build` uses ONE density for both, so call it twice with the density
        // each matrix actually needs and keep only the relevant output.)
        let mut j_ref = Array2::zeros((n, n));
        let mut k_scratch = Array2::zeros((n, n));
        let mut djk_ref = DirectJK::new(&ctx, &prep, &bounds, thresh, usize::MAX);
        djk_ref.build(&d_total, &mut j_ref, &mut k_scratch).unwrap();
        let mut j_scratch = Array2::zeros((n, n));
        let mut k_ref = Array2::zeros((n, n));
        djk_ref.build(&d_half, &mut j_scratch, &mut k_ref).unwrap();

        let mut j = Array2::zeros((n, n));
        let mut ka = Array2::zeros((n, n));
        let mut kb = Array2::zeros((n, n));
        let mut djk = DirectJK::new(&ctx, &prep, &bounds, thresh, usize::MAX);
        djk.build_uhf(&d_total, &d_half, &d_half, &mut j, &mut ka, &mut kb).unwrap();

        assert_eq!(ka, kb, "equal spin densities must give bit-identical K_alpha/K_beta");
        let max_abs_diff = |x: &Array2<f64>, y: &Array2<f64>| -> f64 {
            (x - y).iter().map(|v| v.abs()).fold(0.0f64, f64::max)
        };
        assert!(
            max_abs_diff(&j, &j_ref) < 1e-10,
            "closed-shell limit J mismatch: {}",
            max_abs_diff(&j, &j_ref)
        );
        assert!(
            max_abs_diff(&ka, &k_ref) < 1e-10,
            "closed-shell limit K mismatch: {}",
            max_abs_diff(&ka, &k_ref)
        );
    }

    /// The incremental open-shell update must reproduce a from-scratch build.
    ///
    /// `build_uhf(D1) ==  build_uhf(D0) then build_uhf_incremental(D1 - D0)`,
    /// to the f64 reassociation floor. The deltas here move the two spin
    /// channels in OPPOSITE directions (ΔD_α > 0, ΔD_β < 0) so that ΔD_total
    /// is much smaller than either channel — precisely the configuration in
    /// which screening on ΔD_total instead of |ΔD_α| + |ΔD_β| would drop real
    /// contributions to the spin exchange matrices.
    #[test]
    fn build_uhf_incremental_matches_full_rebuild_with_opposing_spin_deltas() {
        let mol =
            Molecule::parse_xyz("3\nH2O\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.26\n", 0, 1).unwrap();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let n = prep.nbasis();
        let (a0, b0, _) = uhf_test_densities(n);
        let ctx = ParallelContext::default();
        let thresh = 1e-12;

        // Opposing spin deltas: the total barely moves, each channel moves a lot.
        let mut bump = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                bump[(i, j)] = 0.004 * (((i * 11 + j * 17) % 7) as f64 - 3.0);
            }
        }
        let bump = 0.5 * (&bump + &bump.t());
        let a1 = &a0 + &bump;
        let b1 = &b0 - &bump; // ΔD_total == 0 exactly by construction
        let t0 = &a0 + &b0;
        let t1 = &a1 + &b1;

        // From-scratch build at the new density.
        let mut j_full = Array2::zeros((n, n));
        let mut ka_full = Array2::zeros((n, n));
        let mut kb_full = Array2::zeros((n, n));
        let mut djk_f = DirectJK::new(&ctx, &prep, &bounds, thresh, usize::MAX);
        djk_f.build_uhf(&t1, &a1, &b1, &mut j_full, &mut ka_full, &mut kb_full).unwrap();

        // Build at the old density, then accumulate the delta.
        let mut j_inc = Array2::zeros((n, n));
        let mut ka_inc = Array2::zeros((n, n));
        let mut kb_inc = Array2::zeros((n, n));
        let mut djk_i = DirectJK::new(&ctx, &prep, &bounds, thresh, usize::MAX);
        djk_i.build_uhf(&t0, &a0, &b0, &mut j_inc, &mut ka_inc, &mut kb_inc).unwrap();
        let da = &a1 - &a0;
        let db = &b1 - &b0;
        let dt = &t1 - &t0;
        djk_i
            .build_uhf_incremental(&dt, &da, &db, &mut j_inc, &mut ka_inc, &mut kb_inc)
            .unwrap();

        let max_abs_diff = |x: &Array2<f64>, y: &Array2<f64>| -> f64 {
            (x - y).iter().map(|v| v.abs()).fold(0.0f64, f64::max)
        };
        let tol = 1e-10;
        assert!(
            max_abs_diff(&j_inc, &j_full) < tol,
            "incremental J drift {}",
            max_abs_diff(&j_inc, &j_full)
        );
        assert!(
            max_abs_diff(&ka_inc, &ka_full) < tol,
            "incremental K_alpha drift {} — a ΔD_total screening key would fail \
             HERE, where ΔD_total is zero but each spin channel moved",
            max_abs_diff(&ka_inc, &ka_full)
        );
        assert!(
            max_abs_diff(&kb_inc, &kb_full) < tol,
            "incremental K_beta drift {}",
            max_abs_diff(&kb_inc, &kb_full)
        );
        // Confirm the test is actually in the regime it claims: the spin deltas
        // must be large while the total delta is negligible.
        let max_dt = dt.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        let max_da = da.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        assert!(
            max_da > 1e-3 && max_dt < 1e-12,
            "test is not in the opposing-delta regime: max|dD_a|={max_da}, max|dD_total|={max_dt}"
        );
    }

    /// Thread-count bit-identity for the new open-shell path, matching the
    /// existing guarantee for `build`/`DirectJ`/`DirectK`.
    #[test]
    fn build_uhf_bit_identical_across_thread_counts() {
        let mol =
            Molecule::parse_xyz("3\nH2O\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.26\n", 0, 1).unwrap();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let n = prep.nbasis();
        let (d_a, d_b, d_total) = uhf_test_densities(n);

        let run = |threads: usize| -> (Array2<f64>, Array2<f64>, Array2<f64>) {
            let pool =
                rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            pool.install(|| {
                let ctx = ParallelContext::default();
                let mut j = Array2::zeros((n, n));
                let mut ka = Array2::zeros((n, n));
                let mut kb = Array2::zeros((n, n));
                let mut djk = DirectJK::new(&ctx, &prep, &bounds, 1e-14, usize::MAX);
                djk.build_uhf(&d_total, &d_a, &d_b, &mut j, &mut ka, &mut kb).unwrap();
                (j, ka, kb)
            })
        };
        let r1 = run(1);
        let r4 = run(4);
        assert_eq!(r1.0, r4.0, "combined UHF J must be bit-identical across thread counts");
        assert_eq!(r1.1, r4.1, "combined UHF K_alpha must be bit-identical across thread counts");
        assert_eq!(r1.2, r4.2, "combined UHF K_beta must be bit-identical across thread counts");
    }

    #[test]
    fn direct_builders_bit_identical_across_thread_counts() {
        let mol = Molecule::parse_xyz(
            "3\nH2O\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.26\n",
            0,
            1,
        )
        .unwrap();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let n = prep.nbasis();

        // Dense symmetric density so every quartet contributes.
        let mut d = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                d[(i, j)] = 0.01 * ((i * 7 + j * 3) % 11) as f64;
            }
        }
        let d = 0.5 * (&d + &d.t());

        let build_all = |threads: usize| -> (Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>) {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                let ctx = ParallelContext::default();
                let mut j_jk = Array2::zeros((n, n));
                let mut k_jk = Array2::zeros((n, n));
                let mut djk = DirectJK::new(&ctx, &prep, &bounds, 1e-14, usize::MAX);
                djk.build(&d, &mut j_jk, &mut k_jk).unwrap();

                let mut j_only = Array2::zeros((n, n));
                let mut dj = DirectJ::new(&ctx, &prep, &bounds, 1e-14, usize::MAX);
                <DirectJ as JBuilder>::build(&mut dj, &d, &mut j_only).unwrap();

                let mut k_only = Array2::zeros((n, n));
                let mut dk = DirectK::new(&ctx, &prep, &bounds, 1e-14, usize::MAX);
                <DirectK as KBuilder>::build(&mut dk, &d, &mut k_only).unwrap();

                (j_jk, k_jk, j_only, k_only)
            })
        };

        let r1 = build_all(1);
        let r4 = build_all(4);
        assert_eq!(r1.0, r4.0, "DirectJK J must be bit-identical across thread counts");
        assert_eq!(r1.1, r4.1, "DirectJK K must be bit-identical across thread counts");
        assert_eq!(r1.2, r4.2, "DirectJ J must be bit-identical across thread counts");
        assert_eq!(r1.3, r4.3, "DirectK K must be bit-identical across thread counts");
    }
}
