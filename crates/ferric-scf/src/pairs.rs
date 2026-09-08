//! Pair list infrastructure for linear-scaling exchange (LinK).
//!
//! - [`crate::pairs::SignificantPairs`]: geometry-dependent pair lists built once per geometry.
//! - [`crate::pairs::DensityPairs`]: density-dependent pair lists rebuilt each SCF cycle.
//! - [`crate::pairs::intersect_sorted`]: O(n) merge intersection of two sorted slices.

use crate::screening::Bound;
use ferric_integrals::basis_bridge::PreparedBasis;
use ndarray::Array2;

/// Significant shell pairs based on integral bounds.
///
/// `pairs[i]` is a sorted list of shell indices `j` such that
/// `Q(i,j) · Q_max > threshold`, with `Q(i,j) = sqrt(estimate(i,j,i,j))` and
/// `Q_max` the largest `Q` over all shell pairs. Built once per geometry from
/// any type implementing [`Bound`].
#[derive(Debug, Clone)]
pub struct SignificantPairs {
    /// `pairs[i]` = sorted Vec of shells j significant with shell i.
    pairs: Vec<Vec<usize>>,
    threshold: f64,
}

impl SignificantPairs {
    /// Build significant pairs from a bound and threshold.
    ///
    /// A quartet `(ij|kl)` survives the kernel's screen only if
    /// `Q(i,j)·Q(k,l)·|D| ≥ threshold`, so a pair `(i,j)` can matter only if
    /// `Q(i,j)·Q_max ≥ threshold` — the density-free necessary condition (the
    /// same pair prescreen PySCF's `q_cond` applies against `direct_scf_tol`).
    /// The former criterion compared `estimate(i,j,i,j) = Q(i,j)²` against
    /// `threshold`, i.e. cut pairs at `Q > sqrt(threshold)`: at 1e-12 that
    /// dropped every pair with `Q < 1e-6`, each of which can still carry
    /// O(1e-6)-sized quartets, and LinK-vs-direct differed by
    /// `0.37·sqrt(threshold)` (butane/def2-SVP: 3.7e-7 at 1e-12, 3.7e-9 at
    /// 1e-16, 5.6e-11 at 1e-20) instead of at threshold scale.
    ///
    /// Non-positive thresholds keep every pair whose bound is positive
    /// (thresh 0) or every pair outright (negative), as before.
    pub fn build(bound: &dyn Bound, nshells: usize, threshold: f64) -> Self {
        let q = |i: usize, j: usize| bound.estimate(i, j, i, j).sqrt();
        let q_max = (0..nshells)
            .flat_map(|i| (0..nshells).map(move |j| (i, j)))
            .map(|(i, j)| q(i, j))
            .fold(0.0f64, f64::max);
        let mut pairs = Vec::with_capacity(nshells);
        for i in 0..nshells {
            let mut row = Vec::new();
            for j in 0..nshells {
                if q(i, j) * q_max > threshold {
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

/// Below this many shells, run `DensityPairs::build` serially — this is pure
/// scalar arithmetic (no libint engines), so the rayon dispatch/collect
/// overhead only pays off once `nsh` is large enough; keeps free-atom/tiny-
/// basis SCF cycles on the cheap serial path.
const PAR_DENSITY_PAIRS_THRESHOLD: usize = 64;

/// Density-dependent pair lists rebuilt each SCF cycle.
///
/// `pairs[j]` is a sorted list of shell indices `sigma` such that the exchange
/// contribution the density block `D[j, sigma]` can make through ANY quartet
/// LinK will visit exceeds the threshold — see [`DensityPairs::build`] for what
/// that bound is and why the obvious form of it is nearly vacuous.
#[derive(Debug, Clone)]
pub struct DensityPairs {
    pairs: Vec<Vec<usize>>,
}

impl DensityPairs {
    /// Build density-dependent pair lists.
    ///
    /// A density block `D[j,σ]` enters K only through quartets `(i j|σ l)` (up
    /// to the 8-fold permutations), each bounded by `Q(i,j)·Q(σ,l)·|D[j,σ]|`.
    ///
    /// # The bound, and the two maxima that made it vacuous
    ///
    /// The obvious necessary condition takes the worst case over both free
    /// indices independently:
    ///
    /// ```text
    ///   max|D[j,σ]| · qmax(j) · qmax(σ) > threshold,   qmax(x) = max_y Q(x,y)
    /// ```
    ///
    /// That IS valid, and it is what shipped. It is also nearly vacuous, for a
    /// structural reason rather than a coding error. `qmax(x)` maximizes over
    /// ALL partners `y` in the molecule, and for any given `x` that maximum is
    /// realized by a tight core s-shell pair whose value barely depends on `x`.
    /// So `qmax(j)·qmax(σ)` is not a per-pair quantity at all — it is a
    /// molecule-wide constant `≈ Qmax²`, and the criterion degenerates to
    ///
    /// ```text
    ///   max|D[j,σ]| > threshold / Qmax²
    /// ```
    ///
    /// With `Qmax²` of order 1 and a 1e-12 threshold this asks only whether the
    /// block exceeds ~1e-12. Across ~50 Bohr of alkane the density tail is still
    /// above that (the alkane density-matrix decay length is ~30 Bohr), so the
    /// answer is "yes" for EVERY pair. MEASURED at thresh 1e-12 / def2-SVP on a
    /// converged density: 10404/10404 kept on alkane_8 and 39204/39204 on
    /// alkane_16 — the list pruned exactly nothing at either size
    /// (`scripts/queue/out/link_fixed_counts.md`).
    ///
    /// The defect is the two INDEPENDENT maxima: no single quartet realizes
    /// both. Taking the global max over `i` and over `l` separately discards
    /// the locality on both sides at once, which is precisely the information
    /// the list exists to exploit.
    ///
    /// # What is used instead
    ///
    /// The `l` side keeps its maximum — LinK genuinely iterates `lsh` over all
    /// of `sp(ksh)`, so `qmax(σ)` is the honest bound there and tightening it
    /// would make the condition unnecessary (dropping real contributions).
    ///
    /// The `i` side does not: `D[j,σ]` is reached only from a bra pair
    /// CONTAINING `j`, and the largest Schwarz factor such a pair can have is
    /// `qmax(j)` — but that bound is only attained when `j`'s best partner is
    /// also the molecule's best, which is the coincidence that flattened the
    /// product into a constant. Replacing the free-floating global with the
    /// pair's own coupling strength `Q(j,σ)` would restore locality, but
    /// OVER-tightens: `Q(j,σ)` decays like a Gaussian OVERLAP while `D[j,σ]`
    /// decays only exponentially, so it discards genuinely contributing far
    /// pairs — that was the PRE-#50 criterion and it cost
    /// `max|K_LinK - K_direct|` = 1.7e-3 on butane/def2-SVP at 1e-12
    /// (2.3e-7 at the atom-block-diagonal SAD guess, where the far blocks are
    /// exactly zero and the error therefore hides). See
    /// `tests/link_scf_anchor.rs`.
    ///
    /// So the correct quantity is neither of those. What bounds the exchange
    /// contribution of `(j,σ)` is the best bra pair `j` can actually form
    /// TOGETHER WITH the best ket pair `σ` can actually form, and both of those
    /// are already exactly `qmax(·)`. **The shipped criterion is therefore the
    /// tightest bound available at pair-list granularity**, and its weakness is
    /// intrinsic to the granularity, not to the formula.
    ///
    /// The resolution is that the pair list is the wrong place to recover this.
    /// Locality on the `i` side is a property of the (bra pair, ket pair)
    /// COMBINATION, which only exists once both are known — i.e. per quartet.
    /// That is where it now lives: `LinkK::build` screens each quartet on the
    /// pairwise `max(d13,d14,d23,d24)` density key
    /// (`DensityScreen::FourPairK`), which uses the density block belonging to
    /// the specific shells in hand rather than any maximum. This list keeps its
    /// job of bounding the ket LOOP; the per-quartet screen bounds the WORK.
    ///
    /// Threshold semantics: the SAME `threshold` the per-quartet screen
    /// enforces, deliberately. The list is a necessary condition feeding that
    /// screen, so a pair it drops must be one no surviving quartet could need;
    /// equal thresholds are what make it a pure accelerator with no accuracy
    /// cost. Raising it to force pruning would trade correctness for counts.
    ///
    /// Parallelized over the outer shell index `j` once `nsh` clears
    /// `PAR_DENSITY_PAIRS_THRESHOLD`. Each `j` reads only `d`/`bound`/`prep`
    /// (shared, read-only) and *produces* its own `row: Vec<usize>` — there is
    /// no shared mutable state or scatter to reason about, just a per-index
    /// pure function `j ↦ row(j)`. `into_par_iter().map(..).collect()` is
    /// index-order-preserving (rayon's documented guarantee), so the resulting
    /// `Vec<Vec<usize>>` is in ascending-`j` order exactly like the serial
    /// push loop — bit/element-for-element identical, not just equal as sets.
    pub fn build(
        d: &Array2<f64>,
        bound: &dyn Bound,
        prep: &PreparedBasis,
        threshold: f64,
    ) -> Self {
        let nsh = prep.nshells();
        let dims = prep.shell_dims();
        let offs = prep.shell_offsets();

        // qmax[x] = max_y Q(x,y): the largest Schwarz factor of any shell pair
        // containing x (O(nsh²) diagonal estimates, once per build).
        let qmax: Vec<f64> = (0..nsh)
            .map(|x| (0..nsh).map(|y| bound.estimate(x, y, x, y).sqrt()).fold(0.0f64, f64::max))
            .collect();

        let row_for = |j: usize| -> Vec<usize> {
            let mut row = Vec::new();
            for sigma in 0..nsh {
                // Find max |D_element| in the (j, sigma) shell block.
                let mut dmax = 0.0f64;
                for mu in offs[j]..offs[j] + dims[j] {
                    for nu in offs[sigma]..offs[sigma] + dims[sigma] {
                        dmax = dmax.max(d[(mu, nu)].abs());
                    }
                }
                if dmax * qmax[j] * qmax[sigma] > threshold {
                    row.push(sigma);
                }
            }
            // Already sorted ascending since sigma iterates 0..nsh.
            row
        };

        let pairs: Vec<Vec<usize>> = if nsh < PAR_DENSITY_PAIRS_THRESHOLD {
            (0..nsh).map(row_for).collect()
        } else {
            use rayon::prelude::*;
            (0..nsh).into_par_iter().map(row_for).collect()
        };
        DensityPairs { pairs }
    }

    /// Density-significant partners of shell `j`, sorted ascending.
    pub fn partners(&self, j: usize) -> &[usize] {
        &self.pairs[j]
    }

    /// Total number of density-significant pairs (sum of all row lengths).
    pub fn total_pairs(&self) -> usize {
        self.pairs.iter().map(|v| v.len()).sum()
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

        let dp = DensityPairs::build(result.density_r(), &bounds, &prep, 1e-12);
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

    /// Serial reference for `DensityPairs::build` (pre-parallelization
    /// implementation, kept verbatim).
    fn density_pairs_build_serial(
        d: &Array2<f64>,
        bound: &dyn crate::screening::Bound,
        prep: &PreparedBasis,
        threshold: f64,
    ) -> Vec<Vec<usize>> {
        let nsh = prep.nshells();
        let dims = prep.shell_dims();
        let offs = prep.shell_offsets();
        let mut qmax = vec![0.0f64; nsh];
        for x in 0..nsh {
            for y in 0..nsh {
                qmax[x] = qmax[x].max(bound.estimate(x, y, x, y).sqrt());
            }
        }
        let mut pairs = Vec::with_capacity(nsh);
        for j in 0..nsh {
            let mut row = Vec::new();
            for sigma in 0..nsh {
                let mut dmax = 0.0f64;
                for mu in offs[j]..offs[j] + dims[j] {
                    for nu in offs[sigma]..offs[sigma] + dims[sigma] {
                        dmax = dmax.max(d[(mu, nu)].abs());
                    }
                }
                if dmax * qmax[j] * qmax[sigma] > threshold {
                    row.push(sigma);
                }
            }
            pairs.push(row);
        }
        pairs
    }

    #[test]
    fn test_density_pairs_build_exact_match_to_serial() {
        // alkane_6/cc-pVDZ clears PAR_DENSITY_PAIRS_THRESHOLD (64 shells), so
        // this exercises the rayon path. A synthetic density matrix (no SCF
        // needed) keeps the test fast while still exercising real geometry-
        // dependent Schwarz bounds.
        let mol = Molecule::load_xyz("../../testdata/molecules/alkane_6.xyz").unwrap();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        assert!(prep.nshells() >= 64,
            "test basis too small to exercise the parallel path: {} shells", prep.nshells());
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();

        let n = prep.nbasis();
        // Deterministic pseudo-random symmetric density matrix (no RNG dep):
        // a simple LCG-derived pattern gives varied magnitudes across blocks
        // so both the "kept" and "skipped" branches of the threshold fire.
        let mut d = Array2::<f64>::zeros((n, n));
        let mut state: u64 = 0x243F6A8885A308D3;
        for i in 0..n {
            for j in 0..=i {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let v = ((state >> 33) as f64 / u32::MAX as f64) * 2.0 - 1.0;
                d[(i, j)] = v;
                d[(j, i)] = v;
            }
        }

        for threshold in [1e-14, 1e-6, 1e-2] {
            let par = DensityPairs::build(&d, &bounds, &prep, threshold);
            let ser = density_pairs_build_serial(&d, &bounds, &prep, threshold);
            assert_eq!(par.pairs.len(), ser.len());
            for j in 0..ser.len() {
                assert_eq!(
                    par.partners(j), ser[j].as_slice(),
                    "DensityPairs row {j} mismatch at threshold={threshold:.0e}"
                );
            }
        }
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
