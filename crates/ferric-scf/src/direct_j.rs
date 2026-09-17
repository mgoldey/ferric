//! Direct integral-driven Coulomb (J) matrix construction.

use crate::fock::JBuilder;
use crate::screening::SchwarzBounds;
use ferric_core::parallel::ParallelContext;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ndarray::Array2;

/// Screened Coulomb (J) matrix builder.
///
/// Parallelizes with Rayon over the screened bra-pair (s1,s2) work list
/// (O(nsh²) memory), running the ket loop + screens inside the group workers —
/// see the work-list comment in `build` and the matching DirectJK structure.
pub struct DirectJ<'a> {
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

impl<'a> std::fmt::Debug for DirectJ<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DirectJ")
            .field("thresh", &self.thresh)
            .field("mem_budget", &self.mem_budget)
            .finish_non_exhaustive()
    }
}

impl<'a> DirectJ<'a> {
    /// Create a screened direct Coulomb (J) builder.
    pub fn new(
        ctx: &'a ParallelContext,
        prep: &'a PreparedBasis,
        bounds: &'a SchwarzBounds,
        thresh: f64,
        mem_budget: usize,
    ) -> Self {
        DirectJ {
            ctx,
            prep,
            bounds,
            thresh,
            mem_budget,
            pool: None,
        }
    }
}

impl<'a> JBuilder for DirectJ<'a> {
    fn build(&mut self, d: &Array2<f64>, j: &mut Array2<f64>) -> Result<usize, FerricError> {
        use crate::quartet_scatter::{scatter_bra_pair, DensityScreen, JkMode};
        use std::sync::atomic::{AtomicUsize, Ordering};

        let nsh = self.prep.nshells();
        let dims = self.prep.shell_dims();
        let offs = self.prep.shell_offsets();
        let max_d = d.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        let thresh = self.thresh;
        let computed_quartets = AtomicUsize::new(0);

        let max_q: f64 = self.bounds.q.iter().cloned().fold(0.0f64, f64::max);
        let bra_thresh = if max_q > 0.0 { thresh / max_q } else { thresh };
        let q_table = &self.bounds.q;
        // CSB `M` table when `[scf] screening = "csb"` selected it; `None`
        // (the default) leaves the hot loop on plain Schwarz, byte-identical.
        let m_table = self.bounds.csb_m.as_ref();
        // CSAM `X` table when `[scf] screening = "csam"` selected it. Mutually
        // exclusive with `m_table` by construction (one `ScreeningKind` match
        // in `compute_for_screening` attaches at most one); `None` for both is
        // the byte-identical plain-Schwarz default.
        let x_table = self.bounds.csam_x.as_ref();
        let op = self.bounds.op;
        let prep = self.prep;

        // Work list = screened canonical bra pairs (s1,s2), O(nsh²) memory; the
        // ket (s3,s4) loop + Schwarz/density screen run inside the parallel
        // group workers below. The former flat quartet pre-enumeration was a
        // serial O(nsh⁴) loop (with per-bra nested Vec collects) rebuilt every
        // SCF iteration, holding the unbounded surviving-quartet list in
        // memory; grouping ~n_pairs/1024 pairs per task amortizes ket-loop
        // load imbalance across rayon's dynamic scheduling (same structure as
        // `rhf::build_jk` / DirectJK).
        let mut shell_pairs: Vec<(usize, usize)> = Vec::new();
        for s1 in 0..nsh {
            for s2 in 0..=s1 {
                if q_table[(s1, s2)] > bra_thresh {
                    shell_pairs.push((s1, s2));
                }
            }
        }
        // MPI rank striping (see `ParallelContext::stripe` doc; size == 1 is a no-op).
        let shell_pairs: Vec<(usize, usize)> = self.ctx.stripe(shell_pairs);

        // One engine per rayon thread (see engine_pool) — avoids the per-chunk
        // libint2-ctor-mutex storm that made heavy-element bases 10×+ slower.
        if self.pool.is_none() {
            self.pool = Some(crate::engine_pool::EnginePool::new(op, prep, 1e-14)?);
        }
        let pool = self.pool.as_ref().expect("pool initialized above");

        // Deterministic, memory-bounded reduction (see reduce.rs). The old rayon
        // `fold(..).reduce(..)` tree held one nbf² J partial per work-chunk and
        // combined them in a worker-count-dependent order (thread-count-dependent
        // rounding + unbounded partial memory). Group the quartet list, fold each
        // group serially into one partial (parallel across groups), and sum group
        // partials in strict group order — bit-identical across RAYON_NUM_THREADS,
        // live set bounded to one byte-budgeted band of partials.
        // Group partition is a pure function of the quartet list (never the
        // thread count): group boundaries set the floating-point association of
        // the per-group folds, so a thread-dependent partition would break
        // bit-identity across RAYON_NUM_THREADS.
        let n_pairs = shell_pairs.len();
        let group_size = crate::reduce::deterministic_group_size(n_pairs);
        let n_groups = n_pairs.div_ceil(group_size.max(1)).max(1);
        let nbf = prep.nbasis();

        let band_bytes = crate::reduce::resolve_band_bytes(self.mem_budget);
        let screen = DensityScreen::Global(max_d);
        // Accumulate into a rank-LOCAL partial rather than straight onto `j`, so
        // the MPI reduction below acts only on this rank's contribution. `j` is
        // an ACCUMULATE-onto buffer (`grouped_deterministic_sum` adds to prior
        // contents), so Allreducing `j` itself would multiply anything the caller
        // already had in it by the world size — the bug that WAS live in the
        // sibling `DirectJK` path. See `reduce::reduce_partial_across_ranks`.
        //
        // HONESTY NOTE: this change is DEFENSIVE, and it is UNTESTED. `DirectJ`
        // is reached only when DF-J/LinK routing selects it, and every such
        // caller in `rhf.rs`/`uhf.rs`/`rohf.rs` zeroes `j` first, so the
        // carry-over term is zero and `N*0 == 0`. Mutation-tested: reverting this
        // to the old accumulate-then-reduce order leaves the whole MPI suite
        // GREEN at -np 2. So there is no regression test standing behind this
        // ordering here — it is correct-by-construction only. Do not "simplify"
        // it back on the grounds that the tests still pass; they cannot see it.
        let mut total_j = Array2::<f64>::zeros((nbf, nbf));
        crate::reduce::grouped_deterministic_sum(&mut total_j, n_groups, nbf, band_bytes, |g| {
            let lo = g * group_size;
            let hi = (lo + group_size).min(n_pairs);
            let mut mode = JkMode::new_j(nbf);
            let mut local_count = 0usize;
            for &(s1, s2) in &shell_pairs[lo..hi] {
                if ferric_core::INTERRUPT.load(Ordering::Relaxed) {
                    continue;
                }
                pool.with(|engine| {
                    local_count += scatter_bra_pair(
                        engine, prep, dims, offs, q_table, m_table, x_table, &screen, thresh, d,
                        s1, s2, &mut mode, true,
                    );
                });
            }
            let local_j = match mode {
                JkMode::JOnly(j) => j,
                _ => unreachable!("DirectJ::build always uses JkMode::JOnly"),
            };
            computed_quartets.fetch_add(local_count, Ordering::Relaxed);
            Ok(local_j)
        })?;

        crate::reduce::reduce_partial_across_ranks(self.ctx, &mut total_j);
        *j += &total_j;

        Ok(computed_quartets.load(Ordering::SeqCst))
    }

    fn reset(&mut self) {}
}
