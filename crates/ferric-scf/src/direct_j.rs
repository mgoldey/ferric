use crate::fock::JBuilder;
use crate::screening::SchwarzBounds;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_core::parallel::ParallelContext;
use ndarray::Array2;

/// Screened Coulomb (J) matrix builder.
///
/// Enumerates all screened canonical (s1,s2,s3,s4) quartets upfront and
/// parallelizes over them with Rayon, giving balanced fine-grained tasks.
pub struct DirectJ<'a> {
    ctx: &'a ParallelContext,
    prep: &'a PreparedBasis,
    bounds: &'a SchwarzBounds,
    thresh: f64,
    // Lazily built on first build() and reused for the builder's lifetime:
    // libint2 engine construction is serialized behind a global ctor mutex,
    // so hoist the builder out of the SCF loop to pay it once, not per iteration.
    pool: Option<crate::engine_pool::EnginePool>,
}

impl<'a> DirectJ<'a> {
    pub fn new(ctx: &'a ParallelContext, prep: &'a PreparedBasis, bounds: &'a SchwarzBounds, thresh: f64) -> Self {
        DirectJ { ctx, prep, bounds, thresh, pool: None }
    }
}

impl<'a> JBuilder for DirectJ<'a> {
    fn build(&mut self, d: &Array2<f64>, j: &mut Array2<f64>) -> Result<usize, FerricError> {
        use rayon::prelude::*;
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

        // Enumerate all screened canonical quartets upfront.
        // Parallelizing over bra pairs (s1,s2) leaves the O(N²) ket loop serial per task,
        // causing load imbalance. Flat quartet enumeration gives ~equal work per task.
        let quads: Vec<(usize, usize, usize, usize)> = (0..nsh)
            .flat_map(|s1| (0..=s1).map(move |s2| (s1, s2)))
            .filter(|&(s1, s2)| q_table[(s1, s2)] > bra_thresh)
            .flat_map(|(s1, s2)| {
                let b12 = q_table[(s1, s2)];
                (0..=s1).flat_map(move |s3| {
                    let s4max = if s3 == s1 { s2 } else { s3 };
                    (0..=s4max)
                        .filter(move |&s4| b12 * q_table[(s3, s4)] * max_d >= thresh)
                        .map(move |s4| (s1, s2, s3, s4))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
            })
            .enumerate()
            .filter(|(idx, _)| idx % size == rank)
            .map(|(_, q)| q)
            .collect();

        // One engine per rayon thread (see engine_pool) — avoids the per-chunk
        // libint2-ctor-mutex storm that made heavy-element bases 10×+ slower.
        if self.pool.is_none() {
            self.pool = Some(crate::engine_pool::EnginePool::new(op, prep, 1e-14)?);
        }
        let pool = self.pool.as_ref().expect("pool initialized above");
        let total_j = quads.into_par_iter().fold(
            || (Array2::zeros(j.raw_dim()), 0usize),
            |(mut local_j, mut local_count), (s1, s2, s3, s4)| {
                let (n1, n2) = (dims[s1], dims[s2]);
                let (o1, o2) = (offs[s1], offs[s2]);
                let sym12 = s1 != s2;

                let computed = pool.with(|engine| {
                    engine.compute_quartet(prep, s1, s2, s3, s4).map(|q| {
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
                    }).is_some()
                });
                if computed { local_count += 1; }
                (local_j, local_count)
            }
        ).map(|(local_j, count)| {
            computed_quartets.fetch_add(count, Ordering::Relaxed);
            local_j
        }).reduce(
            || Array2::zeros(j.raw_dim()),
            |mut acc, next| { acc += &next; acc }
        );

        *j += &total_j;

        #[cfg(feature = "mpi")]
        if let Some(world) = &self.ctx.world {
            let mut j_global = Array2::zeros(j.dim());
            world.all_reduce_into(j.as_slice().unwrap(), j_global.as_slice_mut().unwrap(), mpi::collective::SystemOperation::sum());
            *j = j_global;
        }

        Ok(computed_quartets.load(Ordering::SeqCst))
    }

    fn reset(&mut self) {}
}
