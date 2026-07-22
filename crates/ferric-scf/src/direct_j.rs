use crate::fock::JBuilder;
use crate::screening::SchwarzBounds;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_core::parallel::ParallelContext;
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

impl<'a> DirectJ<'a> {
    pub fn new(
        ctx: &'a ParallelContext,
        prep: &'a PreparedBasis,
        bounds: &'a SchwarzBounds,
        thresh: f64,
        mem_budget: usize,
    ) -> Self {
        DirectJ { ctx, prep, bounds, thresh, mem_budget, pool: None }
    }
}

impl<'a> JBuilder for DirectJ<'a> {
    fn build(&mut self, d: &Array2<f64>, j: &mut Array2<f64>) -> Result<usize, FerricError> {
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
        let op = self.bounds.op;
        let prep = self.prep;
        let rank = self.ctx.rank;
        let size = self.ctx.size;

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
        // MPI rank striping (disjoint covering partition; size == 1 is a no-op).
        let shell_pairs: Vec<(usize, usize)> = shell_pairs
            .into_iter()
            .enumerate()
            .filter(|(idx, _)| idx % size == rank)
            .map(|(_, p)| p)
            .collect();

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
        crate::reduce::grouped_deterministic_sum(j, n_groups, nbf, band_bytes, |g| {
            let lo = g * group_size;
            let hi = (lo + group_size).min(n_pairs);
            let mut local_j = Array2::<f64>::zeros((nbf, nbf));
            let mut local_count = 0usize;
            for &(s1, s2) in &shell_pairs[lo..hi] {
                let b12 = q_table[(s1, s2)];
                let (n1, n2) = (dims[s1], dims[s2]);
                let (o1, o2) = (offs[s1], offs[s2]);
                let sym12 = s1 != s2;

                pool.with(|engine| {
                    for s3 in 0..=s1 {
                        let s4max = if s3 == s1 { s2 } else { s3 };
                        for s4 in 0..=s4max {
                            if b12 * q_table[(s3, s4)] * max_d < thresh {
                                continue;
                            }
                            if let Some(q) = engine.compute_quartet(prep, s1, s2, s3, s4) {
                                local_count += 1;
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

                                                local_j[(mu, nu)] += d[(la, sg)] * v;
                                                if sym12 { local_j[(nu, mu)] += d[(la, sg)] * v; }
                                                if sym34 { local_j[(mu, nu)] += d[(sg, la)] * v; }
                                                if sym12 && sym34 { local_j[(nu, mu)] += d[(sg, la)] * v; }

                                                if sym1234 {
                                                    local_j[(la, sg)] += d[(mu, nu)] * v;
                                                    if sym34 { local_j[(sg, la)] += d[(mu, nu)] * v; }
                                                    if sym12 { local_j[(la, sg)] += d[(nu, mu)] * v; }
                                                    if sym12 && sym34 { local_j[(sg, la)] += d[(nu, mu)] * v; }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                });
            }
            computed_quartets.fetch_add(local_count, Ordering::Relaxed);
            Ok(local_j)
        })?;

        #[cfg(feature = "mpi")]
        if let Some(world) = self.ctx.world() {
            use mpi::traits::CommunicatorCollectives;
            let mut j_global = Array2::zeros(j.dim());
            world.all_reduce_into(j.as_slice().unwrap(), j_global.as_slice_mut().unwrap(), mpi::collective::SystemOperation::sum());
            *j = j_global;
        }

        Ok(computed_quartets.load(Ordering::SeqCst))
    }

    fn reset(&mut self) {}
}
