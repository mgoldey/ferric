//! Schwarz integral screening for efficient Fock matrix construction.

use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_integrals::schwarz;
use ndarray::Array2;

/// Precomputed Schwarz screening bounds for shell-pair integrals.
///
/// Used to skip negligible shell quartets during Fock matrix construction:
/// if Q(s1,s2) * Q(s3,s4) * max|D| < threshold, the quartet is skipped.
pub struct SchwarzBounds {
    pub q: Array2<f64>,
    pub q_shell: Vec<f64>,
    pub op: Operator,
    pub nshells: usize,
}

impl SchwarzBounds {
    /// Compute Schwarz screening bounds for all shell pairs.
    pub fn compute(op: Operator, prep: &PreparedBasis) -> Result<Self, FerricError> {
        let q = schwarz::schwarz(op, prep)?;
        let nsh = prep.nshells();
        let mut q_shell = vec![0.0; nsh];
        for i in 0..nsh {
            let mut max_val = 0.0f64;
            for j in 0..nsh {
                max_val = max_val.max(q[(i, j)]);
            }
            q_shell[i] = max_val;
        }
        Ok(SchwarzBounds {
            q,
            q_shell,
            op,
            nshells: nsh,
        })
    }

    /// Upper bound estimate for the shell quartet (sh1 sh2 | sh3 sh4).
    pub fn estimate(&self, sh1: usize, sh2: usize, sh3: usize, sh4: usize) -> f64 {
        self.q[(sh1, sh2)] * self.q[(sh3, sh4)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::basis;
    use ferric_core::mol::Molecule;
    use ferric_integrals::basis_bridge::PreparedBasis;

    #[test]
    fn test_schwarz_bounds_estimate() {
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
        let est = bounds.estimate(0, 1, 2, 3);
        let want = bounds.q[(0, 1)] * bounds.q[(2, 3)];
        assert!((est - want).abs() < 1e-15);
    }
}
