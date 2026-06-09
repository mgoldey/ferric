//! Memory-budgeted aux-blocked 3-index (P|μν) integral source.
//!
//! Serves RAW (un-dressed) 3-center integrals in aux-blocks under a fixed byte
//! budget. In-core when the full tensor fits the budget; disk-spill otherwise.
//! Consumers apply their own metric (V^{-1} for J, V^{-1/2} for K).

use crate::basis_bridge::PreparedBasis;
use crate::operator::Operator;
use ferric_core::FerricError;
use ndarray::{Array3, ArrayView3};

/// Largest number of aux rows whose (block_naux × nao × nao × 8) bytes fit the
/// budget; at least 1 (a single aux row must always be representable).
fn block_naux_for(budget_bytes: usize, nao: usize) -> usize {
    let row_bytes = nao.saturating_mul(nao).saturating_mul(8).max(1);
    (budget_bytes / row_bytes).max(1)
}

/// One aux-block of raw (P|μν), rows [p0, p0+data.shape()[0]).
pub struct AuxBlock<'a> {
    pub p0: usize,
    pub data: ArrayView3<'a, f64>,
}

enum Backend {
    InCore(Array3<f64>),
    // DiskSpill added in Task A3.
}

pub struct ThreeIndexSource {
    naux: usize,
    nao: usize,
    block_naux: usize,
    backend: Backend,
}

impl ThreeIndexSource {
    /// `budget_bytes` is the hard ceiling for the resident raw 3-index footprint.
    pub fn build(
        op: Operator, obs: &PreparedBasis, dfbs: &PreparedBasis, budget_bytes: usize,
    ) -> Result<Self, FerricError> {
        let naux = dfbs.nbasis();
        let nao = obs.nbasis();
        let needed = naux.saturating_mul(nao).saturating_mul(nao).saturating_mul(8);
        if needed <= budget_bytes {
            let eri = crate::threeindex::eri3_tensor(op, obs, dfbs)?;
            Ok(Self { naux, nao, block_naux: naux, backend: Backend::InCore(eri) })
        } else {
            // Spill path in Task A3; for now error so the test for in-core passes
            // and the spill test (A3) drives the implementation.
            Err(FerricError::General(
                "ThreeIndexSource spill backend not yet implemented (Task A3)".into(),
            ))
        }
    }

    pub fn naux(&self) -> usize { self.naux }
    pub fn nao(&self) -> usize { self.nao }
    pub fn n_blocks(&self) -> usize {
        self.naux.div_ceil(self.block_naux.max(1))
    }

    /// Primary iteration API. Calls `f` once per raw aux-block, in order.
    pub fn for_each_block(
        &mut self,
        mut f: impl FnMut(AuxBlock<'_>) -> Result<(), FerricError>,
    ) -> Result<(), FerricError> {
        match &self.backend {
            Backend::InCore(eri) => {
                let nb = self.n_blocks();
                for i in 0..nb {
                    let p0 = i * self.block_naux;
                    let p1 = (p0 + self.block_naux).min(self.naux);
                    let view = eri.slice(ndarray::s![p0..p1, .., ..]);
                    f(AuxBlock { p0, data: view })?;
                }
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operator::Operator;
    use crate::basis_bridge::PreparedBasis;
    use ferric_core::basis;
    use ferric_core::mol::Molecule;

    fn water() -> (Molecule,) { (Molecule::parse_xyz("3\nH2O\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.26\n", 0, 1).unwrap(),) }

    #[test]
    fn block_naux_respects_budget() {
        // nao=10 → one aux row is 10*10*8 = 800 bytes.
        // budget 4000 bytes → block_naux = 4000/800 = 5.
        assert_eq!(block_naux_for(4000, 10), 5);
        // budget smaller than one row → at least 1.
        assert_eq!(block_naux_for(500, 10), 1);
    }

    #[test]
    fn in_core_block_equals_dense_eri3() {
        let (mol,) = water();
        let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        let op = Operator::coulomb();
        let dense = crate::threeindex::eri3_tensor(op, &obs, &dfbs).unwrap();
        // Huge budget → single in-core block, raw (un-dressed).
        let mut src = ThreeIndexSource::build(op, &obs, &dfbs, usize::MAX).unwrap();
        assert_eq!(src.n_blocks(), 1);
        let mut reassembled = ndarray::Array3::<f64>::zeros(dense.dim());
        src.for_each_block(|blk| {
            reassembled.slice_mut(ndarray::s![blk.p0..blk.p0 + blk.data.shape()[0], .., ..])
                .assign(&blk.data);
            Ok(())
        }).unwrap();
        let maxdiff = (&reassembled - &dense).iter().map(|v| v.abs()).fold(0.0, f64::max);
        assert!(maxdiff == 0.0, "in-core raw block != dense eri3, maxdiff={maxdiff}");
    }
}
