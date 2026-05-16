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
use ndarray::{Array2, Array3};
use ndarray_linalg::{Eigh, UPLO};

/// DF-K exchange builder. Caches the V^{-1/2}-dressed 3-center tensor.
pub struct DfK {
    /// (naux, n, n) dressed 3-center tensor B[P, μ, ν] = Σ_Q V^{-1/2}_{PQ} (Q|μν).
    b: Array3<f64>,
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

        // V^{-1/2} via symmetric eigendecomposition
        let (evals, evecs) = v
            .eigh(UPLO::Upper)
            .map_err(|e| FerricError::Lapack(format!("V eigh in DfK: {e}")))?;
        let mut u_scaled = evecs.clone();
        for k in 0..naux {
            let s = 1.0 / evals[k].sqrt();
            for r in 0..naux {
                u_scaled[(r, k)] *= s;
            }
        }
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

        Ok(DfK { b })
    }
}

impl KBuilder for DfK {
    fn build(&mut self, d: &Array2<f64>, k: &mut Array2<f64>) -> Result<usize, FerricError> {
        let (naux, n, _) = self.b.dim();

        // Z[P,μ,σ] = Σ_λ B[P,μ,λ] · D[λ,σ]
        // Reshape B to (naux*n, n), GEMM with D (n, n), reshape back.
        let b_flat = self
            .b
            .view()
            .into_shape_with_order((naux * n, n))
            .map_err(|e| FerricError::General(format!("B reshape Z: {e}")))?;
        let z_flat = b_flat.dot(d); // (naux*n, n)
        let z = z_flat
            .into_shape_with_order((naux, n, n))
            .map_err(|e| FerricError::General(format!("Z reshape: {e}")))?;

        // K[μ,ν] = Σ_{P,σ} Z[P,μ,σ] · B[P,ν,σ]
        //
        // Reshape both to (n, naux·n) by treating μ (resp. ν) as the row axis and
        // flattening (P, σ) into a single contracted index. Then one big DGEMM
        //   K = Z_flat · B_flat^T
        // replaces naux small (n,n)·(n,n) products that each allocated a new
        // result array.
        //
        // Strides: Z and B are (naux, n, n) row-major with strides (n*n, n, 1).
        // The (μ, P, σ) layout we need has μ first; getting that requires a
        // permute that is not stride-compatible with view-only reshape. So we
        // materialize permuted views as owned arrays of shape (n, naux, n) →
        // reshape to (n, naux*n).
        let z_perm = z.permuted_axes([1, 0, 2]).as_standard_layout().to_owned();
        let b_perm = self
            .b
            .view()
            .permuted_axes([1, 0, 2])
            .as_standard_layout()
            .to_owned();
        let z_flat = z_perm
            .into_shape_with_order((n, naux * n))
            .map_err(|e| FerricError::General(format!("Z_perm reshape: {e}")))?;
        let b_flat = b_perm
            .into_shape_with_order((n, naux * n))
            .map_err(|e| FerricError::General(format!("B_perm reshape: {e}")))?;
        let k_full = z_flat.dot(&b_flat.t());
        k.assign(&k_full);

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
