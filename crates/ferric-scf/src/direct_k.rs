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
    // Lazily built on first build() and reused for the builder's lifetime:
    // libint2 engine construction is serialized behind a global ctor mutex,
    // so hoist the builder out of the SCF loop to pay it once, not per iteration.
    pool: Option<crate::engine_pool::EnginePool>,
}

impl<'a> DirectK<'a> {
    pub fn new(ctx: &'a ParallelContext, prep: &'a PreparedBasis, bounds: &'a SchwarzBounds, thresh: f64) -> Self {
        DirectK { ctx, prep, bounds, thresh, pool: None }
    }
}

impl<'a> KBuilder for DirectK<'a> {
    fn build(&mut self, d: &Array2<f64>, k: &mut Array2<f64>) -> Result<usize, FerricError> {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let nsh = self.prep.nshells();
        let dims = self.prep.shell_dims();
        let offs = self.prep.shell_offsets();
        let max_d = d.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        let computed_quartets = AtomicUsize::new(0);

        let shell_pairs: Vec<_> = (0..nsh)
            .flat_map(|s1| (0..=s1).map(move |s2| (s1, s2)))
            .collect();

        // MPI logic: only process pairs that belong to this rank
        let pairs_for_this_rank: Vec<_> = shell_pairs.into_iter()
            .enumerate()
            .filter(|(idx, _)| idx % self.ctx.size == self.ctx.rank)
            .map(|(_, p)| p)
            .collect();

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

        crate::reduce::grouped_deterministic_sum(k, n_groups, nbf, |g| {
            let lo = g * group_size;
            let hi = (lo + group_size).min(n_pairs);
            let mut local_k = Array2::<f64>::zeros((nbf, nbf));
            let mut local_count = 0usize;
            for &(s1, s2) in &pairs[lo..hi] {
                let b12 = self.bounds.q[(s1, s2)];
                let (n1, n2) = (dims[s1], dims[s2]);
                let (o1, o2) = (offs[s1], offs[s2]);
                let sym12 = s1 != s2;

                for s3 in 0..=s1 {
                    let s4max = if s3 == s1 { s2 } else { s3 };
                    for s4 in 0..=s4max {
                        let b34 = self.bounds.q[(s3, s4)];
                        if b12 * b34 * max_d < self.thresh {
                            continue;
                        }
                        let computed = pool.with(|engine| {
                            engine.compute_quartet(self.prep, s1, s2, s3, s4).map(|q| {
                                let (n3, n4) = (dims[s3], dims[s4]);
                                let (o3, o4) = (offs[s3], offs[s4]);
                                let sym34 = s3 != s4;
                                let sym1234 = (s1, s2) != (s3, s4);
                                for a in 0..n1 {
                                    for b in 0..n2 {
                                        for c in 0..n3 {
                                            for dd in 0..n4 {
                                                let v = q[((a * n2 + b) * n3 + c) * n4 + dd];
                                                let mu = o1 + a;
                                                let nu = o2 + b;
                                                let la = o3 + c;
                                                let sg = o4 + dd;

                                                local_k[(mu, la)] += d[(nu, sg)] * v;
                                                if sym12 { local_k[(nu, la)] += d[(mu, sg)] * v; }
                                                if sym34 { local_k[(mu, sg)] += d[(nu, la)] * v; }
                                                if sym12 && sym34 { local_k[(nu, sg)] += d[(mu, la)] * v; }

                                                if sym1234 {
                                                    local_k[(la, mu)] += d[(sg, nu)] * v;
                                                    if sym34 { local_k[(sg, mu)] += d[(la, nu)] * v; }
                                                    if sym12 { local_k[(la, nu)] += d[(sg, mu)] * v; }
                                                    if sym12 && sym34 { local_k[(sg, nu)] += d[(la, mu)] * v; }
                                                }
                                            }
                                        }
                                    }
                                }
                            }).is_some()
                        });
                        if computed { local_count += 1; }
                    }
                }
            }
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
