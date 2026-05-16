//! Pair list infrastructure for linear-scaling exchange (LinK).
//!
//! - [`SignificantPairs`]: geometry-dependent pair lists built once per geometry.
//! - [`DensityPairs`]: density-dependent pair lists rebuilt each SCF cycle.
//! - [`intersect_sorted`]: O(n) merge intersection of two sorted slices.

use crate::screening::Bound;
use ferric_integrals::basis_bridge::PreparedBasis;
use ndarray::Array2;

/// Significant shell pairs based on integral bounds.
///
/// `pairs[i]` is a sorted list of shell indices `j` such that
/// `bound.estimate(i, j, i, j) > threshold`. Built once per geometry from
/// any type implementing [`Bound`].
pub struct SignificantPairs {
    /// `pairs[i]` = sorted Vec of shells j significant with shell i.
    pairs: Vec<Vec<usize>>,
    threshold: f64,
}

impl SignificantPairs {
    /// Build significant pairs from a bound and threshold.
    ///
    /// For each shell `i`, collects all shells `j` where the diagonal bound
    /// `estimate(i, j, i, j)` exceeds `threshold`, sorted ascending.
    pub fn build(bound: &dyn Bound, nshells: usize, threshold: f64) -> Self {
        let mut pairs = Vec::with_capacity(nshells);
        for i in 0..nshells {
            let mut row = Vec::new();
            for j in 0..nshells {
                if bound.estimate(i, j, i, j) > threshold {
                    row.push(j);
                }
            }
            // Already in ascending order since j iterates 0..nshells.
            pairs.push(row);
        }
        SignificantPairs { pairs, threshold }
    }

    /// Significant partners of shell `i`, sorted ascending.
    pub fn partners(&self, i: usize) -> &[usize] {
        &self.pairs[i]
    }

    /// The threshold used to build the pair list.
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Total number of shells.
    pub fn nshells(&self) -> usize {
        self.pairs.len()
    }

    /// Total number of significant pairs (sum of all row lengths).
    pub fn total_pairs(&self) -> usize {
        self.pairs.iter().map(|v| v.len()).sum()
    }
}

/// Density-dependent pair lists rebuilt each SCF cycle.
///
/// `pairs[j]` is a sorted list of shell indices `sigma` where the maximum
/// absolute density element in the `(j, sigma)` shell block times the
/// Schwarz-like bound Q(j, sigma) exceeds the threshold.
pub struct DensityPairs {
    pairs: Vec<Vec<usize>>,
}

impl DensityPairs {
    /// Build density-dependent pair lists.
    ///
    /// For each shell `j`, finds shells `sigma` where the maximum absolute density
    /// matrix element in the (j, sigma) block times `bound.estimate(j, sigma, j, sigma).sqrt()`
    /// exceeds `threshold`.
    pub fn build(
        d: &Array2<f64>,
        bound: &dyn Bound,
        prep: &PreparedBasis,
        threshold: f64,
    ) -> Self {
        let nsh = prep.nshells();
        let dims = prep.shell_dims();
        let offs = prep.shell_offsets();
        let mut pairs = Vec::with_capacity(nsh);

        for j in 0..nsh {
            let mut row = Vec::new();
            for sigma in 0..nsh {
                // Find max |D_element| in the (j, sigma) shell block.
                let mut dmax = 0.0f64;
                for mu in offs[j]..offs[j] + dims[j] {
                    for nu in offs[sigma]..offs[sigma] + dims[sigma] {
                        dmax = dmax.max(d[(mu, nu)].abs());
                    }
                }
                // Q(j, sigma) from diagonal bound estimate.
                let q_js = bound.estimate(j, sigma, j, sigma).sqrt();
                if dmax * q_js > threshold {
                    row.push(sigma);
                }
            }
            // Already sorted ascending since sigma iterates 0..nsh.
            pairs.push(row);
        }
        DensityPairs { pairs }
    }

    /// Density-significant partners of shell `j`, sorted ascending.
    pub fn partners(&self, j: usize) -> &[usize] {
        &self.pairs[j]
    }
}

/// O(n) merge intersection of two sorted slices.
///
/// Returns a new `Vec<usize>` containing elements present in both `a` and `b`.
/// Both inputs must be sorted in ascending order.
pub fn intersect_sorted(a: &[usize], b: &[usize]) -> Vec<usize> {
    let mut result = Vec::new();
    let mut i = 0;
    let mut j = 0;
    while i < a.len() && j < b.len() {
        if a[i] == b[j] {
            result.push(a[i]);
            i += 1;
            j += 1;
        } else if a[i] < b[j] {
            i += 1;
        } else {
            j += 1;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::screening::SchwarzBounds;
    use ferric_core::basis;
    use ferric_core::mol::Molecule;
    use ferric_integrals::basis_bridge::PreparedBasis;
    use ferric_integrals::operator::Operator;

    #[test]
    fn test_significant_pairs_contains_all_direct_k_pairs() {
        // Build SP with Schwarz bounds and a tight threshold.
        // Verify that every pair (s1,s2) visited by the DirectK canonical loop
        // with the same threshold is present in SP.
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
        let nsh = prep.nshells();
        let thresh = 1e-12;

        let sp = SignificantPairs::build(&bounds, nsh, thresh);

        // The DirectK canonical loop visits (s1,s2) with s1>=s2 if Q(s1,s2)*Q_max > thresh.
        // SP should include all these pairs.
        let max_q: f64 = bounds.q.iter().cloned().fold(0.0f64, f64::max);
        for s1 in 0..nsh {
            for s2 in 0..=s1 {
                // In the canonical loop, a pair (s1,s2) is visited if there exists
                // any (s3,s4) such that Q(s1,s2)*Q(s3,s4) > thresh.
                // This is equivalent to Q(s1,s2) * max_Q > thresh.
                let q12 = bounds.q[(s1, s2)];
                if q12 * max_q > thresh {
                    assert!(
                        sp.partners(s1).contains(&s2),
                        "SP[{s1}] missing {s2} (diagonal bound = {})",
                        bounds.estimate(s1, s2, s1, s2)
                    );
                    assert!(
                        sp.partners(s2).contains(&s1),
                        "SP[{s2}] missing {s1} (diagonal bound = {})",
                        bounds.estimate(s2, s1, s2, s1)
                    );
                }
            }
        }
    }

    #[test]
    fn test_significant_pairs_sorted() {
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
        let nsh = prep.nshells();

        let sp = SignificantPairs::build(&bounds, nsh, 1e-12);
        for i in 0..nsh {
            let p = sp.partners(i);
            for w in p.windows(2) {
                assert!(w[0] < w[1], "SP[{i}] not sorted: {:?}", p);
            }
        }
    }

    #[test]
    fn test_significant_pairs_symmetric() {
        // If j is in SP[i], then i must be in SP[j] (Schwarz Q is symmetric).
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
        let nsh = prep.nshells();

        let sp = SignificantPairs::build(&bounds, nsh, 1e-12);
        for i in 0..nsh {
            for &j in sp.partners(i) {
                assert!(
                    sp.partners(j).contains(&i),
                    "SP[{i}] has {j} but SP[{j}] missing {i}"
                );
            }
        }
    }

    #[test]
    fn test_density_pairs_nonempty() {
        // Build DensityPairs from a converged RHF density. At least some pairs
        // should be present for a non-trivial density matrix.
        use crate::rhf::{solve_rhf, RhfConfig};

        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();

        let config = RhfConfig {
            energy_conv: 1e-10,
            density_conv: 1e-8,
            ..Default::default()
        };
        let result = solve_rhf(&ferric_core::parallel::ParallelContext::default(), &mol, &prep, op, &bounds, &config).unwrap();
        assert!(result.converged);

        let dp = DensityPairs::build(&result.density_r(), &bounds, &prep, 1e-12);
        let nsh = prep.nshells();

        // Every shell should have at least itself as a partner (diagonal density
        // blocks are non-zero for a real molecule).
        let mut total = 0usize;
        for j in 0..nsh {
            let partners = dp.partners(j);
            total += partners.len();
            assert!(
                !partners.is_empty(),
                "DensityPairs[{j}] is empty for converged water density"
            );
        }
        assert!(
            total > nsh,
            "expected more density pairs than just diagonal: got {total}"
        );
    }

    #[test]
    fn test_intersect_sorted_basic() {
        assert_eq!(intersect_sorted(&[1, 3, 5, 7, 9], &[2, 3, 5, 8, 9]), vec![3, 5, 9]);
        assert_eq!(intersect_sorted(&[0, 1, 2], &[0, 1, 2]), vec![0, 1, 2]);
        assert_eq!(intersect_sorted(&[1], &[1]), vec![1]);
    }

    #[test]
    fn test_intersect_sorted_empty() {
        assert_eq!(intersect_sorted(&[1, 3, 5], &[2, 4, 6]), Vec::<usize>::new());
        assert_eq!(intersect_sorted(&[], &[1, 2, 3]), Vec::<usize>::new());
        assert_eq!(intersect_sorted(&[1, 2, 3], &[]), Vec::<usize>::new());
        assert_eq!(intersect_sorted(&[], &[]), Vec::<usize>::new());
    }
}
