//! Direct integral-driven exchange (K) matrix construction.

use crate::fock::KBuilder;
use crate::screening::SchwarzBounds;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_core::parallel::ParallelContext;
use ndarray::Array2;

/// Direct exchange (K) matrix builder using O(N^4) shell quartets.
pub struct DirectK<'a> {
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

impl<'a> std::fmt::Debug for DirectK<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DirectK")
            .field("thresh", &self.thresh)
            .field("mem_budget", &self.mem_budget)
            .finish_non_exhaustive()
    }
}

impl<'a> DirectK<'a> {
    /// Create a screened direct exchange (K) builder.
    pub fn new(
        ctx: &'a ParallelContext,
        prep: &'a PreparedBasis,
        bounds: &'a SchwarzBounds,
        thresh: f64,
        mem_budget: usize,
    ) -> Self {
        DirectK { ctx, prep, bounds, thresh, mem_budget, pool: None }
    }
}

impl<'a> KBuilder for DirectK<'a> {
    fn build(&mut self, d: &Array2<f64>, k: &mut Array2<f64>) -> Result<usize, FerricError> {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use crate::quartet_scatter::{canonical_bra_pairs, scatter_bra_pair, DensityScreen, JkMode};

        let nsh = self.prep.nshells();
        let dims = self.prep.shell_dims();
        let offs = self.prep.shell_offsets();
        let max_d = d.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        let computed_quartets = AtomicUsize::new(0);

        let shell_pairs: Vec<_> = canonical_bra_pairs(nsh);

        // MPI rank striping (see `ParallelContext::stripe` doc).
        let pairs_for_this_rank: Vec<_> = self.ctx.stripe(shell_pairs);

        // One engine per rayon thread (see engine_pool) — avoids the per-chunk
        // libint2-ctor-mutex storm that made heavy-element bases 10×+ slower.
        if self.pool.is_none() {
            self.pool = Some(crate::engine_pool::EnginePool::new(self.bounds.op, self.prep, 1e-14)?);
        }
        let pool = self.pool.as_ref().expect("pool initialized above");

        // Deterministic, memory-bounded reduction. The old rayon
        // `fold(..).reduce(..)` tree combined one nbf² partial per work-chunk in
        // a worker-count-dependent order (both non-deterministic across thread
        // counts AND unbounded: ~10.8 GB at 50-atom/aug-cc-pVTZ, 32 threads).
        // Instead, partition the bra-pair list into fixed-size GROUPS, fold each
        // group's quartets serially into one partial (parallel across groups),
        // and sum the group partials in strict group order via
        // `grouped_deterministic_sum`. Group order is thread-count-independent,
        // so K is bit-identical across RAYON_NUM_THREADS; the live set is one
        // byte-budgeted band of partials, not one-per-chunk.
        let pairs = &pairs_for_this_rank;
        let n_pairs = pairs.len();
        // Group partition is a pure function of the pair list (never the thread
        // count): group boundaries set the floating-point association of the
        // per-group folds, so a thread-dependent partition would break
        // bit-identity across RAYON_NUM_THREADS. Memory is bounded separately by
        // the reduce helper's band width, not by the group count.
        let group_size = crate::reduce::deterministic_group_size(n_pairs);
        let n_groups = n_pairs.div_ceil(group_size);
        let nbf = self.prep.nbasis();

        let band_bytes = crate::reduce::resolve_band_bytes(self.mem_budget);
        let screen = DensityScreen::Global(max_d);
        crate::reduce::grouped_deterministic_sum(k, n_groups, nbf, band_bytes, |g| {
            let lo = g * group_size;
            let hi = (lo + group_size).min(n_pairs);
            let mut mode = JkMode::new_k(nbf);
            let mut local_count = 0usize;
            for &(s1, s2) in &pairs[lo..hi] {
                if ferric_core::INTERRUPT.load(Ordering::Relaxed) {
                    continue;
                }
                pool.with(|engine| {
                    local_count += scatter_bra_pair(
                        engine, self.prep, dims, offs, &self.bounds.q, &screen, self.thresh, d,
                        s1, s2, &mut mode, true,
                    );
                });
            }
            let local_k = match mode {
                JkMode::KOnly(k) => k,
                _ => unreachable!("DirectK::build always uses JkMode::KOnly"),
            };
            computed_quartets.fetch_add(local_count, Ordering::Relaxed);
            Ok(local_k)
        })?;

        #[cfg(feature = "mpi")]
        if let Some(world) = self.ctx.world() {
            use mpi::traits::CommunicatorCollectives;
            let mut k_global = Array2::zeros(k.dim());
            world.all_reduce_into(k.as_slice().unwrap(), k_global.as_slice_mut().unwrap(), mpi::collective::SystemOperation::sum());
            *k = k_global;
        }

        Ok(computed_quartets.load(Ordering::SeqCst))
    }

    fn update_density(&mut self, _d: &Array2<f64>) {}
    fn reset(&mut self) {}
}
