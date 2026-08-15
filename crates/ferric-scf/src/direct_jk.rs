use crate::screening::SchwarzBounds;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_core::parallel::ParallelContext;
use ndarray::Array2;

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

impl<'a> DirectJK<'a> {
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

        let nsh = self.prep.nshells();
        let dims = self.prep.shell_dims();
        let offs = self.prep.shell_offsets();
        let thresh = self.thresh;
        let computed_quartets = AtomicUsize::new(0);

        // Shell-blocked density-max table d_max_shell[(si, sj)] = max |D_μν| over
        // (μ ∈ shell si, ν ∈ shell sj). The Häser-Ahlrichs density-weighted screen
        // uses the max of all six pair maxima (12,34,13,14,23,24) because J contracts
        // D against (λ,σ),(μ,ν) and K against (μ,λ),(μ,σ),(ν,λ),(ν,σ).
        let d_max_shell = build_d_max_shell(self.prep, d);
        let max_d = d_max_shell.iter().cloned().fold(0.0f64, f64::max);

        let max_q: f64 = self.bounds.q.iter().cloned().fold(0.0f64, f64::max);
        let bra_thresh = if max_q > 0.0 { thresh / (max_q * max_d.max(1e-30)) } else { thresh };
        let q_table = &self.bounds.q;
        let op = self.bounds.op;
        let prep = self.prep;
        let nbf = prep.nbasis();

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
        // rayon workers.
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
