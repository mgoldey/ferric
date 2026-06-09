//! Density-fitted Coulomb matrix builder (RI-J).
//!
//! Replaces the O(N^4) direct ERI Coulomb build with O(N^2 · naux) GEMMs using
//! a 3-center auxiliary expansion:
//!
//!   J_{μν} = Σ_{P,Q} (μν|P) (P|Q)^{-1} (Q|λσ) D_{λσ}
//!          = Σ_P (μν|P) c_P,    c_P = Σ_Q (P|Q)^{-1} d_Q,    d_Q = Σ_{λσ} (Q|λσ) D_{λσ}
//!
//! The 3-center tensor B_{P,μν} = (P|μν) and the inverse Coulomb metric V^{-1}
//! are precomputed once at construction. Each SCF iteration does two GEMM passes
//! over aux-blocks (budget-bounded; spills to disk when naux·nao²·8 > budget_bytes).

use crate::fock::JBuilder;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_integrals::threeindex::coulomb_metric_2c;
use ferric_integrals::three_index_source::ThreeIndexSource;
use ndarray::Array2;
use ndarray_linalg::Inverse;

/// DF-J Coulomb builder. Uses a budget-bounded ThreeIndexSource for raw (P|μν).
pub struct DfJ {
    /// Budget-bounded raw 3-index source (in-core or disk-spill).
    source: ThreeIndexSource,
    /// (naux, naux) inverse Coulomb metric V^{-1}.
    v_inv: Array2<f64>,
}

impl DfJ {
    /// Build the DF-J cache from orbital and auxiliary bases.
    ///
    /// `budget_bytes` is the hard ceiling for the resident raw 3-index footprint.
    /// Pass `usize::MAX` for the old in-core behaviour.
    pub fn new(
        op: Operator,
        obs: &PreparedBasis,
        dfbs: &PreparedBasis,
        budget_bytes: usize,
    ) -> Result<Self, FerricError> {
        let source = ThreeIndexSource::build(op, obs, dfbs, budget_bytes)?;
        let v = coulomb_metric_2c(op, dfbs)?;
        let v_inv = v
            .inv()
            .map_err(|e| FerricError::Lapack(format!("V^-1 in DfJ: {e}")))?;
        Ok(DfJ { source, v_inv })
    }
}

impl JBuilder for DfJ {
    fn build(&mut self, d: &Array2<f64>, j: &mut Array2<f64>) -> Result<usize, FerricError> {
        let naux = self.source.naux();
        let n = self.source.nao();

        let d_flat = d
            .view()
            .into_shape_with_order(n * n)
            .map_err(|e| FerricError::General(format!("D reshape: {e}")))?;

        // Pass 1: d_P = Σ_{μν} B[P,μν] D[μν], accumulated over aux-blocks.
        let mut d_p = ndarray::Array1::<f64>::zeros(naux);
        self.source.for_each_block(|blk| {
            let b = blk.data.shape()[0];
            let flat = blk
                .data
                .into_shape_with_order((b, n * n))
                .map_err(|e| FerricError::General(format!("blk reshape: {e}")))?;
            let part = flat.dot(&d_flat); // (b,)
            d_p.slice_mut(ndarray::s![blk.p0..blk.p0 + b]).assign(&part);
            Ok(())
        })?;

        // c_P = V^{-1} d_P
        let c_p = self.v_inv.dot(&d_p);

        // Pass 2: J[μν] = Σ_P B[P,μν] c_P, accumulated over aux-blocks.
        j.fill(0.0);
        let mut j_flat = j
            .view_mut()
            .into_shape_with_order(n * n)
            .map_err(|e| FerricError::General(format!("J reshape: {e}")))?;
        self.source.for_each_block(|blk| {
            let b = blk.data.shape()[0];
            let flat = blk
                .data
                .into_shape_with_order((b, n * n))
                .map_err(|e| FerricError::General(format!("blk reshape: {e}")))?;
            let c_blk = c_p.slice(ndarray::s![blk.p0..blk.p0 + b]);
            // J += flat^T · c_blk   (n*n,)
            let contrib = flat.t().dot(&c_blk);
            j_flat += &contrib;
            Ok(())
        })?;

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
    use ndarray::Array2;

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
        let mut dfj = DfJ::new(op, &obs, &dfbs, usize::MAX).unwrap();
        dfj.build(&d, &mut j_df).unwrap();

        let max_diff: f64 = (&j_df - &j_direct).iter().map(|v| v.abs()).fold(0.0, f64::max);
        // RI-fit basis is tuned for correlation, not J — accept ~1e-3 Ha-scale error.
        assert!(max_diff < 5e-3, "DF-J vs direct-J max diff = {} too large", max_diff);
    }

    #[test]
    fn df_j_source_backed_matches_incore() {
        let mol = Molecule::parse_xyz(
            "3\nH2O\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.26\n",
            0,
            1,
        )
        .unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        let op = Operator::coulomb();
        let n = obs.nbasis();
        let mut d = Array2::<f64>::zeros((n, n));
        for i in 0..n.min(5) {
            d[(i, i)] = 2.0;
        }

        // huge budget = in-core path
        let mut j_big = Array2::zeros((n, n));
        let mut dfj_big = DfJ::new(op, &obs, &dfbs, usize::MAX).unwrap();
        dfj_big.build(&d, &mut j_big).unwrap();

        // tiny budget = spill path; must be bit-identical
        let tiny = n * n * 8 * 3;
        let mut j_small = Array2::zeros((n, n));
        let mut dfj_small = DfJ::new(op, &obs, &dfbs, tiny).unwrap();
        dfj_small.build(&d, &mut j_small).unwrap();

        let maxdiff = (&j_big - &j_small).iter().map(|v| v.abs()).fold(0.0, f64::max);
        assert!(maxdiff < 1e-10, "spill J != in-core J, maxdiff={maxdiff}");
    }
}
