//! Direct integral-driven exchange (K) matrix construction.

use crate::fock::KBuilder;
use crate::screening::SchwarzBounds;
use ferric_core::parallel::ParallelContext;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
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
    /// The contraction rotation of generally contracted shells
    /// ([`crate::direct_jk::gc_rotation`]), resolved at the first build.
    gc: crate::direct_jk::gc_rotation::BuilderGc,
    /// Engines on the rotated shells, owned like `pool`.
    rot_pool: Option<crate::engine_pool::EnginePool>,
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
        DirectK {
            ctx,
            prep,
            bounds,
            thresh,
            mem_budget,
            pool: None,
            gc: Default::default(),
            rot_pool: None,
        }
    }

    /// The build on the rotated shells of `bounds.gc` when the basis has
    /// generally contracted shells and the rotation is on
    /// ([`crate::direct_jk::gc_rotation`]): `D' = Tᵀ D T`, this builder's
    /// walk on the rotated basis, then `x += T X' Tᵀ`. `None`: unrotated.
    fn build_rotated(
        &mut self,
        d: &Array2<f64>,
        x: &mut Array2<f64>,
    ) -> Result<Option<usize>, FerricError> {
        let Some(r) = self.gc.resolve(self.prep, self.bounds)? else {
            return Ok(None);
        };
        let _charge = r.charge_collective(self.ctx, 2)?;
        let dr = r.forward(d);
        let n = dr.nrows();
        let mut xr = Array2::<f64>::zeros((n, n));
        let mut inner = DirectK::new(
            self.ctx,
            &r.basis.prep,
            &r.bounds,
            self.thresh,
            self.mem_budget,
        );
        inner.pool = self.rot_pool.take();
        let computed = <DirectK as KBuilder>::build(&mut inner, &dr, &mut xr);
        self.rot_pool = inner.pool.take();
        let computed = computed?;
        r.back_add(&xr, x)?;
        Ok(Some(computed))
    }
}

impl<'a> KBuilder for DirectK<'a> {
    fn build(&mut self, d: &Array2<f64>, k: &mut Array2<f64>) -> Result<usize, FerricError> {
        crate::quartet_scatter::check_nbf_shapes(
            "DirectK::build",
            self.prep.nbasis(),
            &[("d", d.dim()), ("k", k.dim())],
        )?;
        if let Some(computed) = self.build_rotated(d, k)? {
            return Ok(computed);
        }
        use crate::quartet_scatter::{
            canonical_bra_pairs, scatter_bra_pair, DensityScreen, JkDigest, JkMode, PairSpec,
        };
        use std::sync::atomic::{AtomicUsize, Ordering};

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
            self.pool = Some(crate::engine_pool::EnginePool::new(
                self.bounds.op,
                self.prep,
                ferric_integrals::engine_pool::eri_precision(),
            )?);
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
        // Rank-LOCAL partial, reduced below before it reaches `k`. `k` is an
        // ACCUMULATE-onto buffer, so Allreducing `k` itself would multiply any
        // pre-existing caller contents by the world size — the bug that WAS live
        // in `DirectJK`. See `reduce::reduce_partial_across_ranks`.
        //
        // HONESTY NOTE: defensive and UNTESTED, exactly as in `direct_j.rs`.
        // Every in-tree caller zeroes `k` first, so the carry-over is zero;
        // mutation-testing confirmed that reverting this ordering leaves the MPI
        // suite GREEN at -np 2. Correct-by-construction only — no test covers it.
        // One group's bra pairs scattered into `mode`.
        let run = |g: usize, mode: &mut JkMode<'_>| {
            let lo = g * group_size;
            let hi = (lo + group_size).min(n_pairs);
            let mut local_count = 0usize;
            for &(s1, s2) in &pairs[lo..hi] {
                if ferric_core::INTERRUPT.load(Ordering::Relaxed) {
                    continue;
                }
                pool.with(|engine| {
                    local_count += scatter_bra_pair(
                        engine,
                        self.prep,
                        dims,
                        offs,
                        &self.bounds.q,
                        self.bounds.csb_m.as_ref(),
                        self.bounds.csam_x.as_ref(),
                        &screen,
                        self.thresh,
                        d,
                        s1,
                        s2,
                        mode,
                        true,
                    );
                });
            }
            computed_quartets.fetch_add(local_count, Ordering::Relaxed);
        };
        let mut total_k = match self.bounds.digest.mode_collective(self.ctx)? {
            JkDigest::Legacy => {
                let mut total_k = Array2::<f64>::zeros((nbf, nbf));
                crate::reduce::grouped_deterministic_sum(
                    &mut total_k,
                    n_groups,
                    nbf,
                    band_bytes,
                    |g| {
                        let mut mode = JkMode::new_k(nbf);
                        run(g, &mut mode);
                        match mode {
                            JkMode::KOnly(k) => Ok(k),
                            _ => unreachable!("DirectK::build's legacy walk uses JkMode::KOnly"),
                        }
                    },
                )?;
                total_k
            }
            JkDigest::Pair => PairSpec::k_only(nbf, dims, d)
                .sum_groups(self.ctx, n_groups, band_bytes, None, run)?
                .k
                .pop()
                .expect("the K-only pair digest builds K"),
        };

        crate::reduce::reduce_partial_across_ranks(self.ctx, &mut total_k);
        *k += &total_k;

        Ok(computed_quartets.load(Ordering::SeqCst))
    }

    fn update_density(&mut self, _d: &Array2<f64>) {}
    fn reset(&mut self) {}
}
