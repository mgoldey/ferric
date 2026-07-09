//! Density-fitted exchange matrix builder (DF-K / RI-K).
//!
//! Replaces the O(N^4) direct ERI exchange build with O(N^3 · naux) GEMMs using
//! the same 3-center auxiliary expansion as DF-J. For closed-shell RHF:
//!
//!   K_{μν} = Σ_{λσ} (μλ|νσ) D_{λσ}
//!          ≈ Σ_P Σ_{λσ} B^P_{μλ} B^P_{νσ} D_{λσ}        (RI with V^{-1/2}-dressed B)
//!
//! Computed as two passes over (P,μ,ν):
//!   Z[P,μ,σ] = Σ_λ B[P,μ,λ] · D[λ,σ]
//!   K[μ,ν]   = Σ_{P,σ} Z[P,μ,σ] · B[P,ν,σ]
//!
//! Each aux-block is contracted in rayon-parallel aux-chunks. Per chunk the
//! per-P (n,n)×(n,n) GEMM stack is collapsed into two wide GEMMs on repacked
//! operands (see `build`); BLAS stays single-threaded (OPENBLAS_NUM_THREADS=1
//! under rayon), so chunks are the parallel unit.
//!
//! The dressed 3-center tensor B[P,μ,ν] = Σ_Q V^{-1/2}_{PQ} (Q|μν) is built once
//! at construction (via ThreeIndexSource::build_dressed) and reused every SCF
//! iteration.  The dressed source is budget-bounded: in-core when
//! naux·nao²·8 ≤ budget_bytes, else aux-blocked disk-spill.
//!
//! Accuracy of DF-K depends critically on the auxiliary basis: use a JK-fit
//! basis (e.g. `def2-universal-jkfit`), not an RI/MP2-fit basis.

use crate::fock::KBuilder;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_integrals::three_index_source::ThreeIndexSource;
use ferric_integrals::threeindex::coulomb_metric_2c;
use ndarray::linalg::general_mat_mul;
use ndarray::{Array2, Array3};
use ndarray_linalg::{Eigh, UPLO};

/// DF-K exchange builder. Caches the V^{-1/2}-dressed 3-center source; the
/// per-chunk repack/GEMM scratch in `build` is allocated per rayon task
/// (O(chunk·n²), negligible against the O(chunk·n³) GEMM it feeds).
pub struct DfK {
    /// Budget-bounded dressed 3-center source B[P,μ,ν] = Σ_Q V^{-1/2}_{PQ} (Q|μν).
    dressed: ThreeIndexSource,
}

/// V^{-1/2} via symmetric eigendecomposition with canonical orthogonalization.
/// The 2-center metric `(P|w(r12)|Q)` is positive-definite analytically, but
/// for range-separated operators (erf, erfc) with JK-fit aux on heavy atoms,
/// some eigenvalues can be near-zero and turn slightly negative under
/// floating-point roundoff. Drop those modes — equivalent to PySCF's
/// `lindep` threshold in `df.aux_e2`.
fn v_inv_sqrt_lindep(v: &Array2<f64>) -> Result<Array2<f64>, FerricError> {
    let naux = v.nrows();
    let (evals, evecs) = v
        .eigh(UPLO::Upper)
        .map_err(|e| FerricError::Lapack(format!("V eigh in DfK: {e}")))?;
    const LINDEP_THRESH: f64 = 1e-10;
    let mut u_scaled = evecs.clone();
    let mut n_dropped: usize = 0;
    for k in 0..naux {
        if evals[k] < LINDEP_THRESH {
            // Zero out this column so its (column-vector outer product) contributes
            // nothing to V^{-1/2}.
            for r in 0..naux {
                u_scaled[(r, k)] = 0.0;
            }
            n_dropped += 1;
        } else {
            let s = 1.0 / evals[k].sqrt();
            for r in 0..naux {
                u_scaled[(r, k)] *= s;
            }
        }
    }
    // Silent on n_dropped: this is expected for range-separated operators
    // (erf, erfc) with JK-fit aux on heavy atoms and is benign.
    let _ = n_dropped;
    Ok(u_scaled.dot(&evecs.t())) // (naux, naux)
}

impl DfK {
    /// Build the DF-K cache from orbital and auxiliary bases.
    ///
    /// Computes V^{-1/2} = U · diag(λ^{-1/2}) · U^T from the symmetric eigendecomp
    /// of the (P|Q) Coulomb metric, then forms B[P,μ,ν] = Σ_Q V^{-1/2}_{PQ} (Q|μν)
    /// via ThreeIndexSource::build_dressed, honouring `budget_bytes`.
    pub fn new(
        op: Operator,
        obs: &PreparedBasis,
        dfbs: &PreparedBasis,
        budget_bytes: usize,
    ) -> Result<Self, FerricError> {
        let v = coulomb_metric_2c(op, dfbs)?;
        let v_inv_sqrt = v_inv_sqrt_lindep(&v)?;

        // Build raw source then dress it: B[P,μν] = Σ_Q V^{-1/2}_{PQ} (Q|μν).
        let mut raw = ThreeIndexSource::build(op, obs, dfbs, budget_bytes)?;
        let dressed = ThreeIndexSource::build_dressed(&mut raw, &v_inv_sqrt, budget_bytes)?;

        Ok(DfK { dressed })
    }
}

impl KBuilder for DfK {
    fn build(&mut self, d: &Array2<f64>, k: &mut Array2<f64>) -> Result<usize, FerricError> {
        let n = self.dressed.nao();
        k.fill(0.0);
        // Aux-chunk width: wide enough that the (n, c·n)×(c·n, n) contraction
        // below runs at BLAS3 efficiency even for small n, capped so per-task
        // scratch (3·c·n² doubles) stays modest for large n.
        let chunk = (4096 / n.max(1)).clamp(4, 64);
        self.dressed.for_each_block(|blk| {
            let b = blk.data.shape()[0];
            // Contract this block in rayon-parallel aux-chunks, each collapsing
            // its per-P GEMM stack into two wide GEMMs. Block streaming (and its
            // disk IO on the spill path) stays sequential in for_each_block; only
            // the in-memory contraction of the current block fans out.
            // Compute each aux-chunk's K contribution in rayon-parallel, then
            // fold into `k` in strict chunk order. A rayon `try_reduce` combines
            // partials in a tree whose shape depends on the worker count, so
            // floating-point non-associativity made the K matrix (and thus the
            // SCF energy) vary with RAYON_NUM_THREADS by ~µHa. Collecting *all*
            // per-chunk partials then serial-summing pins the order but holds
            // (b/chunk)·n²·8 bytes live at once — the DF-K scaling hazard
            // (~97 GB at 50-atom/aug-cc-pVTZ). `grouped_deterministic_sum`
            // processes the chunks in byte-budgeted bands, summing each band in
            // chunk order before the next: same ascending-chunk fold order (so
            // still bit-identical across thread counts), but the live set is one
            // band (≤512 MiB), not every chunk.
            let n_chunks = b.div_ceil(chunk);
            crate::reduce::grouped_deterministic_sum(
                k,
                n_chunks,
                n,
                |ci| -> Result<Array2<f64>, FerricError> {
                    let q0 = ci * chunk;
                    let q1 = (q0 + chunk).min(b);
                    let c = q1 - q0;
                    let bchunk = blk.data.slice(ndarray::s![q0..q1, .., ..]);

                    // Zt[μ,P,σ] = Σ_λ B[P,μ,λ] · D[λ,σ] as one (c·n, n)×(n, n)
                    // GEMM on the (μ,P,λ)-repacked chunk, so (P,σ) comes out as
                    // the contiguous trailing axis pair needed below.
                    let mut bswap = Array3::<f64>::zeros((n, c, n));
                    bswap.assign(&bchunk.permuted_axes([1, 0, 2]));
                    let bswap_flat = bswap
                        .into_shape_with_order((n * c, n))
                        .map_err(|e| FerricError::General(format!("B repack reshape: {e}")))?;
                    let mut zt = Array2::<f64>::zeros((n * c, n));
                    general_mat_mul(1.0, &bswap_flat, d, 0.0, &mut zt);
                    let zt_wide = zt
                        .into_shape_with_order((n, c * n))
                        .map_err(|e| FerricError::General(format!("Z reshape: {e}")))?;

                    // K_chunk[μ,ν] = Σ_{P,σ} Zt[μ,(P,σ)] · Bt[(P,σ),ν] with
                    // Bt[P,σ,ν] = B[P,ν,σ]: one wide (n, c·n)×(c·n, n) GEMM
                    // instead of c separate (n,n)×(n,n) ones.
                    let mut bt = Array3::<f64>::zeros((c, n, n));
                    bt.assign(&bchunk.permuted_axes([0, 2, 1]));
                    let bt_flat = bt
                        .into_shape_with_order((c * n, n))
                        .map_err(|e| FerricError::General(format!("Bt reshape: {e}")))?;
                    let mut kc = Array2::<f64>::zeros((n, n));
                    general_mat_mul(1.0, &zt_wide, &bt_flat, 0.0, &mut kc);
                    Ok(kc)
                },
            )?;
            Ok(())
        })?;
        Ok(0)
    }

    fn update_density(&mut self, _d: &Array2<f64>) {}

    fn reset(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::direct_k::DirectK;
    use crate::screening::SchwarzBounds;
    use ferric_core::basis;
    use ferric_core::mol::Molecule;
    use ferric_core::parallel::ParallelContext;

    #[test]
    fn df_k_matches_direct_k_with_jkfit() {
        let mol = Molecule::parse_xyz(
            "3\nH2O\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.26\n",
            0,
            1,
        )
        .unwrap();
        let obs_set = basis::bundled("cc-pvdz").unwrap();
        let obs = PreparedBasis::new(&mol, &obs_set).unwrap();
        let dfbs_set = basis::bundled("def2-universal-jkfit").unwrap();
        let dfbs = PreparedBasis::new(&mol, &dfbs_set).unwrap();

        let op = Operator::coulomb();
        let n = obs.nbasis();

        // Simple diagonal mock density
        let mut d = Array2::<f64>::zeros((n, n));
        for i in 0..n.min(5) {
            d[(i, i)] = 2.0;
        }

        let mut k_direct = Array2::zeros((n, n));
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let ctx = ParallelContext::default();
        let mut dk = DirectK::new(&ctx, &obs, &bounds, 1e-12);
        <DirectK as KBuilder>::build(&mut dk, &d, &mut k_direct).unwrap();

        let mut k_df = Array2::zeros((n, n));
        let mut dfk = DfK::new(op, &obs, &dfbs, usize::MAX).unwrap();
        dfk.build(&d, &mut k_df).unwrap();

        let max_diff: f64 = (&k_df - &k_direct).iter().map(|v| v.abs()).fold(0.0, f64::max);
        // JK-fit basis should give K accurate to ~1e-3 for this small system.
        assert!(max_diff < 5e-3, "DF-K vs direct-K max diff = {} too large", max_diff);
    }

    #[test]
    fn df_k_bit_identical_across_thread_counts() {
        // Regression guard: the aux-chunk accumulation in `build` must be
        // bit-identical regardless of RAYON_NUM_THREADS. A rayon `try_reduce`
        // combined partials in a worker-count-dependent tree, so the K matrix
        // (and hence the SCF energy) drifted ~µHa between thread counts. The
        // collect-then-serial-sum fix pins the order. Uses several heavy atoms so
        // the aux dimension spans multiple chunks (making order actually matter).
        let mol = Molecule::parse_xyz(
            "3\nH2O\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.26\n",
            0,
            1,
        )
        .unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("def2-universal-jkfit").unwrap()).unwrap();
        let op = Operator::coulomb();
        let n = obs.nbasis();

        // Dense symmetric density that couples every (P,σ) pair.
        let mut d = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                d[(i, j)] = 0.01 * ((i * 7 + j * 3) % 11) as f64;
            }
        }
        let d = 0.5 * (&d + &d.t());

        let build_k = |threads: usize| -> Array2<f64> {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                let mut k = Array2::zeros((n, n));
                let mut dfk = DfK::new(op, &obs, &dfbs, usize::MAX).unwrap();
                dfk.build(&d, &mut k).unwrap();
                k
            })
        };

        let k1 = build_k(1);
        let k4 = build_k(4);
        // Bit-identical: exact equality, not a tolerance.
        assert_eq!(
            k1, k4,
            "DfK build must be bit-identical across thread counts (rayon reduction order leak)"
        );
    }

    #[test]
    fn df_k_bit_identical_across_threads_with_narrow_bands() {
        // Same guarantee as above, but with a deliberately tiny reduce-band
        // budget so the two-level grouped_deterministic_sum path actually splits
        // the aux chunks across several bands. The banding must not perturb the
        // ascending-chunk fold order, so K stays bit-identical at 1/2/8 threads.
        let mol = Molecule::parse_xyz("3\nH2O\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.26\n", 0, 1)
            .unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        let dfbs =
            PreparedBasis::new(&mol, &basis::bundled("def2-universal-jkfit").unwrap()).unwrap();
        let op = Operator::coulomb();
        let n = obs.nbasis();

        let mut d = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                d[(i, j)] = 0.01 * ((i * 7 + j * 3) % 11) as f64;
            }
        }
        let d = 0.5 * (&d + &d.t());

        // One nbf² partial for cc-pVDZ water is ~24² · 8 ≈ 4.6 kB; a 4 kB band
        // budget forces band width 1 (each aux chunk its own band).
        std::env::set_var("FERRIC_REDUCE_BAND_BYTES", "4096");
        let build_k = |threads: usize| -> Array2<f64> {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                let mut k = Array2::zeros((n, n));
                let mut dfk = DfK::new(op, &obs, &dfbs, usize::MAX).unwrap();
                dfk.build(&d, &mut k).unwrap();
                k
            })
        };
        let k1 = build_k(1);
        let k2 = build_k(2);
        let k8 = build_k(8);
        std::env::remove_var("FERRIC_REDUCE_BAND_BYTES");
        assert_eq!(k1, k2, "narrow-band DfK must be bit-identical at 1 vs 2 threads");
        assert_eq!(k1, k8, "narrow-band DfK must be bit-identical at 1 vs 8 threads");
    }

    #[test]
    fn df_k_wide_gemm_matches_naive_contraction() {
        // Implementation-equivalence test for the wide-GEMM restructure: the
        // chunked two-wide-GEMM contraction in `build` must reproduce the naive
        // per-P GEMM stack over the SAME dressed B tensor to machine precision.
        // A dense symmetric density exercises every (P,σ) coupling, unlike the
        // diagonal-D accuracy test above (which measures RI fitting error vs
        // direct K and cannot separate algebra bugs from fitting error).
        let mol = Molecule::parse_xyz(
            "3\nH2O\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.26\n",
            0,
            1,
        )
        .unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("def2-universal-jkfit").unwrap()).unwrap();
        let op = Operator::coulomb();
        let n = obs.nbasis();

        let mut d = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                // Symmetric, deterministic, modest magnitude.
                d[(i, j)] = 0.02 * (((i * j + 3) % 11) as f64);
            }
            d[(i, i)] += 1.0;
        }

        let mut dfk = DfK::new(op, &obs, &dfbs, usize::MAX).unwrap();

        // Reference: naive per-P contraction (the pre-restructure algorithm)
        // over the same dressed tensor.
        let mut k_ref = Array2::<f64>::zeros((n, n));
        dfk.dressed
            .for_each_block(|blk| {
                let b = blk.data.shape()[0];
                for p in 0..b {
                    let bp = blk.data.slice(ndarray::s![p, .., ..]);
                    let zp = bp.dot(&d);
                    k_ref += &zp.dot(&bp.t());
                }
                Ok(())
            })
            .unwrap();

        let mut k_df = Array2::zeros((n, n));
        dfk.build(&d, &mut k_df).unwrap();

        let max_diff: f64 = (&k_df - &k_ref).iter().map(|v| v.abs()).fold(0.0, f64::max);
        assert!(
            max_diff < 1e-10,
            "wide-GEMM DF-K vs naive per-P contraction max diff = {} too large",
            max_diff
        );
    }

    /// Determinism demo, run explicitly at several thread counts:
    ///   RAYON_NUM_THREADS=N cargo test -p ferric-scf --lib \
    ///     df_k_scf_energy_determinism_demo -- --ignored --nocapture
    /// Full DF-JK RHF (water/cc-pVDZ, def2-universal-jkfit) on the AMBIENT rayon
    /// pool; prints the total energy to 17 significant digits plus its raw f64
    /// bit pattern. The printed value must be identical at N = 1, 2, 8: DF-J is
    /// serial GEMM and DF-K is the only rayon reduction in this configuration,
    /// so this pins the grouped deterministic accumulation end-to-end through a
    /// real SCF energy.
    #[test]
    #[ignore]
    fn df_k_scf_energy_determinism_demo() {
        use crate::rhf::{solve_rhf, RhfConfig};
        let mol = Molecule::parse_xyz(
            "3\nH2O\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.26\n",
            0,
            1,
        )
        .unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let ctx = ParallelContext::default();
        let config = RhfConfig {
            energy_conv: 1e-12,
            density_conv: 1e-10,
            df_j_aux: Some("def2-universal-jkfit".into()),
            df_k_aux: Some("def2-universal-jkfit".into()),
            ..Default::default()
        };
        let result = solve_rhf(&ctx, &mol, &obs, op, &bounds, &config).unwrap();
        assert!(result.converged);
        println!(
            "DF-JK RHF energy @ {} rayon threads: {:.17e}  bits=0x{:016x}",
            rayon::current_num_threads(),
            result.energy,
            result.energy.to_bits()
        );
    }

    #[test]
    fn df_k_source_backed_matches_incore() {
        let mol = Molecule::parse_xyz(
            "3\nH2O\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.26\n",
            0,
            1,
        )
        .unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("def2-universal-jkfit").unwrap()).unwrap();
        let op = Operator::coulomb();
        let n = obs.nbasis();

        let mut d = Array2::<f64>::zeros((n, n));
        for i in 0..n.min(5) {
            d[(i, i)] = 2.0;
        }

        // Huge budget → in-core path
        let mut k_big = Array2::zeros((n, n));
        let mut dfk_big = DfK::new(op, &obs, &dfbs, usize::MAX).unwrap();
        dfk_big.build(&d, &mut k_big).unwrap();

        // Tiny budget → spill path; must match to ≤1e-10
        let tiny = n * n * 8 * 3;
        let mut k_small = Array2::zeros((n, n));
        let mut dfk_small = DfK::new(op, &obs, &dfbs, tiny).unwrap();
        dfk_small.build(&d, &mut k_small).unwrap();

        let maxdiff = (&k_big - &k_small).iter().map(|v| v.abs()).fold(0.0, f64::max);
        assert!(maxdiff < 1e-10, "spill K != in-core K, maxdiff={maxdiff}");
    }
}
