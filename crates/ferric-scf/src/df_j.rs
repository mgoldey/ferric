//! Density-fitted Coulomb matrix builder (RI-J).
//!
//! Replaces the O(N^4) direct ERI Coulomb build with O(N^2 · naux) GEMMs using
//! a 3-center auxiliary expansion:
//!
//!   J_{μν} = Σ_{P,Q} (μν|P) (P|Q)^{-1} (Q|λσ) D_{λσ}
//!          = Σ_P (μν|P) c_P,    c_P = Σ_Q (P|Q)^{-1} d_Q,    d_Q = Σ_{λσ} (Q|λσ) D_{λσ}
//!
//! The 3-center tensor B_{P,μν} = (P|μν) and the inverse Coulomb metric V^{-1}
//! are precomputed once at construction. Each SCF iteration does two GEMMs.

use crate::fock::JBuilder;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_integrals::threeindex::{coulomb_metric_2c, eri3_tensor};
use ndarray::{Array2, Array3};
use ndarray_linalg::Inverse;

/// DF-J Coulomb builder. Caches the 3-center tensor and inverse metric.
pub struct DfJ {
    /// (naux, n, n) 3-center ERIs B[P, μ, ν] = (P|μν).
    b: Array3<f64>,
    /// (naux, naux) inverse Coulomb metric V^{-1}.
    v_inv: Array2<f64>,
}

impl DfJ {
    /// Build the DF-J cache from orbital and auxiliary bases.
    pub fn new(op: Operator, obs: &PreparedBasis, dfbs: &PreparedBasis) -> Result<Self, FerricError> {
        let b = eri3_tensor(op, obs, dfbs)?;
        let v = coulomb_metric_2c(op, dfbs)?;
        let v_inv = v
            .inv()
            .map_err(|e| FerricError::Lapack(format!("V^-1 in DfJ: {e}")))?;
        Ok(DfJ { b, v_inv })
    }
}

impl JBuilder for DfJ {
    fn build(&mut self, d: &Array2<f64>, j: &mut Array2<f64>) -> Result<usize, FerricError> {
        let (naux, n, _) = self.b.dim();

        // d_P = Σ_{μν} B[P, μ, ν] · D[μ, ν]
        // Reshape B to (naux, n*n), D to (n*n,), contract.
        let b_flat = self
            .b
            .view()
            .into_shape_with_order((naux, n * n))
            .map_err(|e| FerricError::General(format!("B reshape: {e}")))?;
        let d_flat = d
            .view()
            .into_shape_with_order(n * n)
            .map_err(|e| FerricError::General(format!("D reshape: {e}")))?;
        let d_p = b_flat.dot(&d_flat); // (naux,)

        // c_P = Σ_Q V^{-1}[P, Q] · d_Q
        let c_p = self.v_inv.dot(&d_p);

        // J[μ, ν] = Σ_P B[P, μ, ν] · c_P  — reshape B^T (n*n, naux) · c (naux,)
        let j_flat = b_flat.t().dot(&c_p); // (n*n,)
        let j_mat = j_flat
            .into_shape_with_order((n, n))
            .map_err(|e| FerricError::General(format!("J reshape: {e}")))?;
        j.assign(&j_mat);

        Ok(0)
    }

    fn reset(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::direct_j::DirectJ;
    use crate::screening::SchwarzBounds;
    use ferric_core::basis;
    use ferric_core::mol::Molecule;
    use ferric_core::parallel::ParallelContext;

    #[test]
    fn df_j_matches_direct_j_within_fit_error() {
        let mol = Molecule::parse_xyz(
            "3\nH2O\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.26\n",
            0,
            1,
        )
        .unwrap();
        let obs_set = basis::bundled("cc-pvdz").unwrap();
        let obs = PreparedBasis::new(&mol, &obs_set).unwrap();
        let dfbs_set = basis::bundled("cc-pvdz-ri").unwrap();
        let dfbs = PreparedBasis::new(&mol, &dfbs_set).unwrap();

        let op = Operator::coulomb();
        let n = obs.nbasis();

        // Make a representative non-trivial density (approximate; just for J-build comparison).
        let mut d = Array2::<f64>::zeros((n, n));
        for i in 0..n.min(5) {
            d[(i, i)] = 2.0;
        }

        let mut j_direct = Array2::zeros((n, n));
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let ctx = ParallelContext::default();
        let mut dj = DirectJ::new(&ctx, &obs, &bounds, 1e-12);
        <DirectJ as JBuilder>::build(&mut dj, &d, &mut j_direct).unwrap();

        let mut j_df = Array2::zeros((n, n));
        let mut dfj = DfJ::new(op, &obs, &dfbs).unwrap();
        dfj.build(&d, &mut j_df).unwrap();

        let max_diff: f64 = (&j_df - &j_direct).iter().map(|v| v.abs()).fold(0.0, f64::max);
        // RI-fit basis is tuned for correlation, not J — accept ~1e-3 Ha-scale error.
        assert!(max_diff < 5e-3, "DF-J vs direct-J max diff = {} too large", max_diff);
    }
}
