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
//! The dressed 3-center tensor B[P,μ,ν] = Σ_Q V^{-1/2}_{PQ} (Q|μν) is built once
//! at construction and reused every SCF iteration.
//!
//! Accuracy of DF-K depends critically on the auxiliary basis: use a JK-fit
//! basis (e.g. `def2-universal-jkfit`), not an RI/MP2-fit basis.

use crate::fock::KBuilder;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_integrals::threeindex::{coulomb_metric_2c, eri3_tensor};
use ndarray::linalg::general_mat_mul;
use ndarray::{Array2, Array3};
use ndarray_linalg::{Eigh, UPLO};

/// DF-K exchange builder. Caches the V^{-1/2}-dressed 3-center tensor and a
/// per-call scratch buffer for the Z intermediate so the hot SCF path does
/// zero heap allocation outside of the two GEMMs.
pub struct DfK {
    /// (naux, n, n) dressed 3-center tensor B[P, μ, ν] = Σ_Q V^{-1/2}_{PQ} (Q|μν).
    b: Array3<f64>,
    /// Scratch buffer for Z[P,μ,σ] = Σ_λ B[P,μ,λ] D[λ,σ], shape (naux, n, n).
    z_scratch: Array3<f64>,
}

impl DfK {
    /// Build the DF-K cache from orbital and auxiliary bases.
    ///
    /// Computes V^{-1/2} = U · diag(λ^{-1/2}) · U^T from the symmetric eigendecomp
    /// of the (P|Q) Coulomb metric, then forms B[P,μ,ν] = Σ_Q V^{-1/2}_{PQ} (Q|μν).
    pub fn new(op: Operator, obs: &PreparedBasis, dfbs: &PreparedBasis) -> Result<Self, FerricError> {
        let v = coulomb_metric_2c(op, dfbs)?;
        let eri3 = eri3_tensor(op, obs, dfbs)?;
        let (naux, n, _) = eri3.dim();

        // V^{-1/2} via symmetric eigendecomposition with canonical orthogonalization.
        // The 2-center metric `(P|w(r12)|Q)` is positive-definite analytically, but
        // for range-separated operators (erf, erfc) with JK-fit aux on heavy atoms,
        // some eigenvalues can be near-zero and turn slightly negative under
        // floating-point roundoff. Drop those modes — equivalent to PySCF's
        // `lindep` threshold in `df.aux_e2`.
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
        let v_inv_sqrt = u_scaled.dot(&evecs.t()); // (naux, naux)

        // Dress: B[P,μ,ν] = Σ_Q V^{-1/2}_{PQ} (Q|μν)
        // Reshape eri3 to (naux, n*n), gemm, reshape back.
        let eri_flat = eri3
            .view()
            .into_shape_with_order((naux, n * n))
            .map_err(|e| FerricError::General(format!("eri3 reshape: {e}")))?;
        let b_flat = v_inv_sqrt.dot(&eri_flat); // (naux, n*n)
        let b = b_flat
            .into_shape_with_order((naux, n, n))
            .map_err(|e| FerricError::General(format!("B reshape: {e}")))?;

        let z_scratch = Array3::<f64>::zeros((naux, n, n));
        Ok(DfK { b, z_scratch })
    }
}

impl KBuilder for DfK {
    fn build(&mut self, d: &Array2<f64>, k: &mut Array2<f64>) -> Result<usize, FerricError> {
        let (naux, n, _) = self.b.dim();

        // First GEMM: Z[P,μ,σ] = Σ_λ B[P,μ,λ] · D[λ,σ]
        // View self.b as (naux*n, n), self.z_scratch as (naux*n, n), call DGEMM
        // in-place to avoid the 58 MB owned allocation that .dot() would do.
        {
            let b_flat = self
                .b
                .view()
                .into_shape_with_order((naux * n, n))
                .map_err(|e| FerricError::General(format!("B reshape: {e}")))?;
            let mut z_flat = self
                .z_scratch
                .view_mut()
                .into_shape_with_order((naux * n, n))
                .map_err(|e| FerricError::General(format!("Z reshape: {e}")))?;
            general_mat_mul(1.0, &b_flat, d, 0.0, &mut z_flat);
        }

        // Second pass: K[μ,ν] = Σ_{P,σ} Z[P,μ,σ] · B[P,ν,σ]
        // Per-P slab: K += Z[P] · B[P]^T accumulated with general_mat_mul
        // (in-place, beta=1.0). Each Z[P] and B[P] is an (n,n) C-contiguous view
        // with no allocation; the result goes straight into k. For naux=558 and
        // n=114 the slab is 1.5 MFlops, ~750 µs/iter total for the loop.
        k.fill(0.0);
        for p in 0..naux {
            let zp = self.z_scratch.slice(ndarray::s![p, .., ..]);
            let bp = self.b.slice(ndarray::s![p, .., ..]);
            general_mat_mul(1.0, &zp, &bp.t(), 1.0, k);
        }

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
        let mut dfk = DfK::new(op, &obs, &dfbs).unwrap();
        dfk.build(&d, &mut k_df).unwrap();

        let max_diff: f64 = (&k_df - &k_direct).iter().map(|v| v.abs()).fold(0.0, f64::max);
        // JK-fit basis should give K accurate to ~1e-3 for this small system.
        assert!(max_diff < 5e-3, "DF-K vs direct-K max diff = {} too large", max_diff);
    }
}
