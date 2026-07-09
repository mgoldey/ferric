use crate::screening::SchwarzBounds;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_core::parallel::ParallelContext;
use ndarray::Array2;

/// Combined Coulomb (J) and exchange (K) matrix builder.
///
/// Enumerates all screened canonical (s1,s2,s3,s4) quartets **once** and accumulates
/// both J and K from the same computed integral, avoiding the 2× ERI evaluation cost
/// of calling DirectJ and DirectK separately.
///
/// The flat quartet pre-enumeration (same as DirectJ) gives balanced Rayon tasks
/// versus the bra-pair-then-serial-ket structure used by the old DirectK.
pub struct DirectJK<'a> {
    ctx: &'a ParallelContext,
    prep: &'a PreparedBasis,
    bounds: &'a SchwarzBounds,
    thresh: f64,
    // Lazily built on first build() and reused for the builder's lifetime:
    // libint2 engine construction is serialized behind a global ctor mutex,
    // so hoist the builder out of the SCF loop to pay it once, not per iteration.
    pool: Option<crate::engine_pool::EnginePool>,
}

impl<'a> DirectJK<'a> {
    pub fn new(
        ctx: &'a ParallelContext,
        prep: &'a PreparedBasis,
        bounds: &'a SchwarzBounds,
        thresh: f64,
    ) -> Self {
        DirectJK { ctx, prep, bounds, thresh, pool: None }
    }

    /// Build J and K matrices simultaneously from a single pass over shell quartets.
    /// Returns the number of unique quartets computed.
    pub fn build(
        &mut self,
        d: &Array2<f64>,
        j: &mut Array2<f64>,
        k: &mut Array2<f64>,
    ) -> Result<usize, FerricError> {
        use std::sync::atomic::{AtomicUsize, Ordering};

        self.ctx.check_interrupted()?;

        let nsh = self.prep.nshells();
        let dims = self.prep.shell_dims();
        let offs = self.prep.shell_offsets();
        let thresh = self.thresh;
        let computed_quartets = AtomicUsize::new(0);

        // Shell-blocked density-max table d_max_shell[(si, sj)] = max |D_μν| over
        // (μ ∈ shell si, ν ∈ shell sj). The Häser-Ahlrichs density-weighted screen
        // uses the max of all six pair maxima (12,34,13,14,23,24) because J contracts
        // D against (λ,σ),(μ,ν) and K against (μ,λ),(μ,σ),(ν,λ),(ν,σ).
        let mut d_max_shell = Array2::<f64>::zeros((nsh, nsh));
        for si in 0..nsh {
            for sj in 0..nsh {
                let (oi, ni) = (offs[si], dims[si]);
                let (oj, nj) = (offs[sj], dims[sj]);
                let mut m = 0.0f64;
                for a in 0..ni {
                    for b in 0..nj {
                        let v = unsafe { d.uget((oi + a, oj + b)).abs() };
                        if v > m { m = v; }
                    }
                }
                d_max_shell[(si, sj)] = m;
            }
        }
        let max_d = d_max_shell.iter().cloned().fold(0.0f64, f64::max);

        let max_q: f64 = self.bounds.q.iter().cloned().fold(0.0f64, f64::max);
        let bra_thresh = if max_q > 0.0 { thresh / (max_q * max_d.max(1e-30)) } else { thresh };
        let q_table = &self.bounds.q;
        let op = self.bounds.op;
        let prep = self.prep;
        let rank = self.ctx.rank;
        let size = self.ctx.size;
        let nbf = prep.nbasis();

        // Enumerate all screened canonical quartets upfront for balanced parallel tasks.
        // Pair-wise density screen: contribution upper bound is
        //   sqrt(Q_{12}) * sqrt(Q_{34}) * D_max_pair, where
        //   D_max_pair = max(d12, d34, d13, d14, d23, d24).
        let dms = &d_max_shell;
        let mut quads: Vec<(usize, usize, usize, usize)> = Vec::new();
        for s1 in 0..nsh {
            for s2 in 0..=s1 {
                if q_table[(s1, s2)] <= bra_thresh { continue; }
                let b12 = q_table[(s1, s2)];
                let d12 = dms[(s1, s2)];
                for s3 in 0..=s1 {
                    let s4max = if s3 == s1 { s2 } else { s3 };
                    let d13 = dms[(s1, s3)];
                    let d23 = dms[(s2, s3)];
                    for s4 in 0..=s4max {
                        let d34 = dms[(s3, s4)];
                        let d14 = dms[(s1, s4)];
                        let d24 = dms[(s2, s4)];
                        let dmax = d12.max(d34).max(d13).max(d14).max(d23).max(d24);
                        if b12 * q_table[(s3, s4)] * dmax >= thresh {
                            quads.push((s1, s2, s3, s4));
                        }
                    }
                }
            }
        }
        // MPI rank striping
        let quads: Vec<(usize, usize, usize, usize)> = quads
            .into_iter()
            .enumerate()
            .filter(|(idx, _)| idx % size == rank)
            .map(|(_, q)| q)
            .collect();

        // One engine per rayon thread (see engine_pool): constructing it in the
        // fold init below would fire once per work-chunk and storm the global
        // libint2 ctor mutex (catastrophic for heavy-element bases).
        if self.pool.is_none() {
            self.pool = Some(crate::engine_pool::EnginePool::new(op, prep, 1e-14)?);
        }
        let pool = self.pool.as_ref().expect("pool initialized above");

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
        let n_quads = quads.len();
        let group_size = crate::reduce::deterministic_group_size(n_quads);
        let n_groups = n_quads.div_ceil(group_size);

        let mut total_j = Array2::<f64>::zeros((nbf, nbf));
        let mut total_k = Array2::<f64>::zeros((nbf, nbf));
        crate::reduce::grouped_deterministic_sum_pair(
            &mut total_j,
            &mut total_k,
            n_groups,
            nbf,
            |g| {
                let lo = g * group_size;
                let hi = (lo + group_size).min(n_quads);
                let mut local_j = Array2::<f64>::zeros((nbf, nbf));
                let mut local_k = Array2::<f64>::zeros((nbf, nbf));
                let mut local_count = 0usize;
                for &(s1, s2, s3, s4) in &quads[lo..hi] {
                    let (n1, n2) = (dims[s1], dims[s2]);
                    let (o1, o2) = (offs[s1], offs[s2]);
                    let sym12 = s1 != s2;

                    pool.with(|engine| {
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
                                            let v = unsafe {
                                                *q.get_unchecked(((a * n2 + b) * n3 + c) * n4 + dd)
                                            };
                                            let mu = o1 + a;
                                            let nu = o2 + b;
                                            let la = o3 + c;
                                            let sg = o4 + dd;

                                            // J contributions
                                            unsafe {
                                                *local_j.uget_mut((mu, nu)) += d.uget((la, sg)) * v;
                                                *local_k.uget_mut((mu, la)) += d.uget((nu, sg)) * v;
                                                if sym12 {
                                                    *local_j.uget_mut((nu, mu)) += d.uget((la, sg)) * v;
                                                    *local_k.uget_mut((nu, la)) += d.uget((mu, sg)) * v;
                                                }
                                                if sym34 {
                                                    *local_j.uget_mut((mu, nu)) += d.uget((sg, la)) * v;
                                                    *local_k.uget_mut((mu, sg)) += d.uget((nu, la)) * v;
                                                }
                                                if sym12 && sym34 {
                                                    *local_j.uget_mut((nu, mu)) += d.uget((sg, la)) * v;
                                                    *local_k.uget_mut((nu, sg)) += d.uget((mu, la)) * v;
                                                }
                                                if sym1234 {
                                                    *local_j.uget_mut((la, sg)) += d.uget((mu, nu)) * v;
                                                    *local_k.uget_mut((la, mu)) += d.uget((sg, nu)) * v;
                                                    if sym12 {
                                                        *local_j.uget_mut((la, sg)) += d.uget((nu, mu)) * v;
                                                        *local_k.uget_mut((la, nu)) += d.uget((sg, mu)) * v;
                                                    }
                                                    if sym34 {
                                                        *local_j.uget_mut((sg, la)) += d.uget((mu, nu)) * v;
                                                        *local_k.uget_mut((sg, mu)) += d.uget((la, nu)) * v;
                                                    }
                                                    if sym12 && sym34 {
                                                        *local_j.uget_mut((sg, la)) += d.uget((nu, mu)) * v;
                                                        *local_k.uget_mut((sg, nu)) += d.uget((la, mu)) * v;
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
                Ok((local_j, local_k))
            },
        )?;

        *j += &total_j;
        *k += &total_k;

        #[cfg(feature = "mpi")]
        if let Some(world) = &self.ctx.world {
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
                let mut djk = DirectJK::new(&ctx, &prep, &bounds, 1e-14);
                djk.build(&d, &mut j_jk, &mut k_jk).unwrap();

                let mut j_only = Array2::zeros((n, n));
                let mut dj = DirectJ::new(&ctx, &prep, &bounds, 1e-14);
                <DirectJ as JBuilder>::build(&mut dj, &d, &mut j_only).unwrap();

                let mut k_only = Array2::zeros((n, n));
                let mut dk = DirectK::new(&ctx, &prep, &bounds, 1e-14);
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
