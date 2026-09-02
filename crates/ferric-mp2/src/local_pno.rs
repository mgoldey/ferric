//! Pair natural orbitals (PNOs) — the virtual-side truncation for LinLCCD(hh).
//!
//! Second half of the local-correlation layer; [`crate::pair_domains`] is the first.
//! Together they give the "D" and the "PNO" of DLPNO: domains screen the occupied
//! pair list, PNOs compress the virtual space *within* each retained pair.
//!
//! # What a PNO is, and how it differs from the OSVs already in the tree
//!
//! `ferric-rpa`'s [`pno.rs`](../../ferric_rpa/pno/index.html) builds **OSVs**: it
//! diagonalizes the *diagonal* pair density `D^ii` only, giving one virtual set per
//! occupied orbital. True PNOs diagonalize the **off-diagonal** density `D^ij` for
//! each retained pair `(i,j)`, giving a virtual set per *pair*. That is strictly more
//! adaptive — a pair's correlation is described in the virtuals that pair actually
//! uses — and is why PNO thresholds compress far harder than OSV ones at equal error.
//!
//! The pair density is built from the first-order (semicanonical MP2) amplitudes:
//!
//! ```text
//! D^ij = T^ij (T^ij)† + (T^ij)† T^ij        (symmetrized, trace-normalized)
//! ```
//!
//! Diagonalizing it gives occupation numbers `n_a^ij`; virtuals with
//! `n_a^ij < T_CutPNO` are discarded. The retained count is what buys the compression.
//!
//! # What this is honestly worth, on ferric, today
//!
//! ferric has a MEASURED negative result for virtual-space truncation at small sizes
//! (the OSV sweep: 100% retention at accurate thresholds, or 48–76 mHa error at
//! loose ones — a cliff, not a tradeoff). PNOs compress better than OSVs, but the
//! underlying reason for that result is structural: small molecules have compact,
//! non-redundant virtual spaces with nothing to discard. **Expect the payoff at
//! larger N, and measure it rather than assuming it.**
//!
//! There is also a method-specific caveat worth stating plainly, because it decides
//! how much this layer can ever buy for LinLCCD(hh): the dominant term here is the hh
//! ladder at O(n_o⁴n_v²), only *quadratic* in n_v. Halving the virtual space via PNOs
//! is therefore a ≈4× cut on the dominant term, not the ≈16× that the same truncation
//! buys a method carrying an n_v⁴ pp ladder. The occupied-side domains attack the
//! quartic factor and are the larger lever; PNOs stack on top of them.
//!
//! Every routine here is therefore built to be **exact in the limit** (`t_cut_pno = 0`
//! keeps every virtual and the transform is an orthogonal rotation) and to report its
//! own retention, so the accuracy/cost curve is measurable.

use crate::pair_domains::PairDomains;
use ferric_core::FerricError;
use ferric_core::linalg::{eigh_dc, Uplo};
use ndarray::{Array1, Array2};

/// Per-pair virtual transforms produced by [`build_pno_transforms`].
#[derive(Debug, Clone)]
pub struct PnoTransforms {
    /// One entry per pair in the originating [`crate::pair_domains::PairDomains::pairs`], in the same order.
    pub pairs: Vec<PnoPair>,
    /// Full (untruncated) virtual count the transforms map from.
    pub nvir: usize,
    /// Occupation threshold used.
    pub t_cut_pno: f64,
}

/// The PNO basis for a single occupied pair.
#[derive(Debug, Clone)]
pub struct PnoPair {
    /// The occupied pair `(i, j)`, spatial indices.
    pub ij: (usize, usize),
    /// Transform `(nvir × n_pno)` from canonical virtuals into this pair's PNOs.
    /// Columns are orthonormal.
    pub transform: Array2<f64>,
    /// Occupation numbers of the retained PNOs, descending.
    pub occupations: Array1<f64>,
    /// Sum of the occupation numbers that were DISCARDED — the truncation's own
    /// estimate of what it threw away, per pair.
    pub discarded_weight: f64,
}

impl PnoTransforms {
    /// Retained virtuals summed over pairs, divided by `n_pairs · nvir`.
    ///
    /// 1.0 means no compression at all. ferric's OSV history says to expect exactly
    /// that on small systems at accurate thresholds — hence reporting it rather than
    /// assuming a win.
    pub fn virtual_retention(&self) -> f64 {
        if self.pairs.is_empty() || self.nvir == 0 {
            return 1.0;
        }
        let kept: usize = self.pairs.iter().map(|p| p.transform.ncols()).sum();
        kept as f64 / (self.pairs.len() * self.nvir) as f64
    }

    /// Largest per-pair discarded occupation weight — the worst-case truncation error
    /// indicator across pairs.
    pub fn max_discarded_weight(&self) -> f64 {
        self.pairs.iter().map(|p| p.discarded_weight).fold(0.0, f64::max)
    }

    /// True when nothing was truncated: every pair kept all `nvir` virtuals.
    pub fn is_complete(&self) -> bool {
        self.pairs.iter().all(|p| p.transform.ncols() == self.nvir)
    }
}

/// Build per-pair PNO transforms from semicanonical first-order amplitudes.
///
/// `t2_pair(i, j)` must return the `(nvir × nvir)` amplitude block `T^{ij}_{ab}` for
/// the spatial occupied pair `(i, j)`. Taking a closure rather than a packed tensor
/// keeps this independent of how the caller stores amplitudes, and lets a large
/// system build blocks on demand instead of materializing all of `T`.
///
/// `t_cut_pno` is the occupation-number threshold. **`0.0` keeps every virtual**, so
/// the transforms are square orthogonal matrices and any downstream use is exact —
/// the property `pno_zero_threshold_keeps_everything` pins.
///
/// # Errors
///
/// [`FerricError::General`] on a negative threshold or a non-square/mis-sized
/// amplitude block, and propagates any eigensolver failure with the offending pair
/// named (a silent `unwrap` here would surface as a wrong energy much later).
pub fn build_pno_transforms<F>(
    domains: &PairDomains,
    nvir: usize,
    t_cut_pno: f64,
    mut t2_pair: F,
) -> Result<PnoTransforms, FerricError>
where
    F: FnMut(usize, usize) -> Array2<f64>,
{
    if t_cut_pno < 0.0 {
        return Err(FerricError::General(format!(
            "t_cut_pno must be >= 0 (got {t_cut_pno}); use 0.0 to disable truncation"
        )));
    }

    let mut pairs = Vec::with_capacity(domains.pairs.len());
    for &(i, j) in &domains.pairs {
        let t = t2_pair(i, j);
        if t.nrows() != nvir || t.ncols() != nvir {
            return Err(FerricError::General(format!(
                "t2_pair({i},{j}) returned {:?}, expected ({nvir}, {nvir})",
                t.dim()
            )));
        }

        // Pair density D^ij = T Tᵀ + Tᵀ T, explicitly symmetrized. Both terms are
        // needed for an off-diagonal pair: T^ij is not symmetric when i != j, and
        // using T Tᵀ alone would privilege the a index over b.
        let d = t.dot(&t.t()) + t.t().dot(&t);
        let mut d_sym = Array2::<f64>::zeros((nvir, nvir));
        for a in 0..nvir {
            for b in 0..nvir {
                d_sym[(a, b)] = 0.5 * (d[(a, b)] + d[(b, a)]);
            }
        }

        // Ascending eigenvalues, eigenvectors in columns -- same convention as
        // ndarray_linalg's eigh, but returning Result (a panic here would surface
        // much later as a wrong energy).
        let (eigs, vecs) = eigh_dc(&d_sym, Uplo::Upper).map_err(|e| {
            FerricError::General(format!("PNO eigh failed for pair ({i},{j}): {e}"))
        })?;

        // eigh returns ascending; we want the largest occupations first.
        let mut order: Vec<usize> = (0..nvir).collect();
        order.sort_by(|&x, &y| eigs[y].partial_cmp(&eigs[x]).unwrap_or(std::cmp::Ordering::Equal));

        let keep: Vec<usize> =
            order.iter().copied().filter(|&k| eigs[k].abs() >= t_cut_pno).collect();
        // Never empty a pair: an empty virtual space silently zeroes that pair's
        // correlation rather than approximating it.
        let keep = if keep.is_empty() { vec![order[0]] } else { keep };
        let discarded_weight: f64 =
            order.iter().copied().filter(|k| !keep.contains(k)).map(|k| eigs[k].abs()).sum();

        let mut transform = Array2::<f64>::zeros((nvir, keep.len()));
        let mut occupations = Array1::<f64>::zeros(keep.len());
        for (slot, &k) in keep.iter().enumerate() {
            occupations[slot] = eigs[k];
            for a in 0..nvir {
                transform[(a, slot)] = vecs[(a, k)];
            }
        }

        pairs.push(PnoPair { ij: (i, j), transform, occupations, discarded_weight });
    }

    Ok(PnoTransforms { pairs, nvir, t_cut_pno })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pair_domains::{build_pair_domains, complete_pair_domains};
    use ndarray::array;

    fn centers3() -> Array2<f64> {
        array![[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [2.8, 0.0, 0.0]]
    }

    /// Deterministic amplitude block; asymmetric on purpose so a T·Tᵀ-only pair
    /// density (which would be wrong for i != j) is distinguishable.
    fn amp(nvir: usize, i: usize, j: usize) -> Array2<f64> {
        let mut t = Array2::<f64>::zeros((nvir, nvir));
        let mut s = (i as u64 + 1).wrapping_mul(97).wrapping_add((j as u64 + 1).wrapping_mul(31));
        for a in 0..nvir {
            for b in 0..nvir {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                t[(a, b)] = ((s >> 33) as f64 / (1u64 << 31) as f64) - 1.0;
            }
        }
        t
    }

    /// THE EXACTNESS PROPERTY: a zero threshold must truncate nothing.
    ///
    /// With every virtual retained the transform is a square orthogonal matrix, so
    /// transforming into the PNO basis and back is the identity and any downstream
    /// contraction is unchanged. This is what lets the layer be switched on with
    /// `t_cut_pno = 0` as an exact baseline.
    #[test]
    fn pno_zero_threshold_keeps_everything() {
        let d = complete_pair_domains(&centers3()).unwrap();
        let nvir = 5;
        let p = build_pno_transforms(&d, nvir, 0.0, |i, j| amp(nvir, i, j)).unwrap();

        assert!(p.is_complete(), "t_cut_pno = 0 must keep every virtual");
        assert_eq!(p.virtual_retention(), 1.0);
        assert_eq!(p.max_discarded_weight(), 0.0);
        for pair in &p.pairs {
            assert_eq!(pair.transform.ncols(), nvir);
        }
    }

    /// The transform columns must be orthonormal — it is a rotation of the virtual
    /// space, so `Pᵀ P = I`. A non-orthonormal transform would silently rescale the
    /// correlation energy.
    #[test]
    fn pno_transform_is_orthonormal() {
        let d = complete_pair_domains(&centers3()).unwrap();
        let nvir = 6;
        let p = build_pno_transforms(&d, nvir, 0.0, |i, j| amp(nvir, i, j)).unwrap();

        for pair in &p.pairs {
            let gram = pair.transform.t().dot(&pair.transform);
            let n = gram.nrows();
            for a in 0..n {
                for b in 0..n {
                    let want = if a == b { 1.0 } else { 0.0 };
                    assert!(
                        (gram[(a, b)] - want).abs() < 1e-10,
                        "pair {:?}: PᵀP[{a},{b}] = {}, expected {want}",
                        pair.ij,
                        gram[(a, b)]
                    );
                }
            }
        }
    }

    /// A large threshold must actually compress, and must report what it discarded.
    #[test]
    fn large_threshold_compresses_and_reports_loss() {
        let d = complete_pair_domains(&centers3()).unwrap();
        let nvir = 6;
        let loose = build_pno_transforms(&d, nvir, 1.0, |i, j| amp(nvir, i, j)).unwrap();

        assert!(!loose.is_complete(), "a large threshold should truncate something");
        assert!(
            loose.virtual_retention() < 1.0,
            "retention {} should be < 1",
            loose.virtual_retention()
        );
        assert!(
            loose.max_discarded_weight() > 0.0,
            "truncation must report the weight it discarded"
        );
    }

    /// Occupations come back in descending order, so "keep the top n" is meaningful.
    #[test]
    fn occupations_are_descending() {
        let d = complete_pair_domains(&centers3()).unwrap();
        let nvir = 5;
        let p = build_pno_transforms(&d, nvir, 0.0, |i, j| amp(nvir, i, j)).unwrap();
        for pair in &p.pairs {
            for k in 1..pair.occupations.len() {
                assert!(
                    pair.occupations[k - 1] >= pair.occupations[k],
                    "pair {:?} occupations not descending at {k}",
                    pair.ij
                );
            }
        }
    }

    /// No pair may be emptied, however aggressive the threshold: an empty virtual
    /// space zeroes that pair's correlation rather than approximating it.
    #[test]
    fn pairs_are_never_emptied() {
        let d = complete_pair_domains(&centers3()).unwrap();
        let nvir = 4;
        let p = build_pno_transforms(&d, nvir, 1e300, |i, j| amp(nvir, i, j)).unwrap();
        for pair in &p.pairs {
            assert!(pair.transform.ncols() >= 1, "pair {:?} was emptied", pair.ij);
        }
    }

    /// PNOs stack on top of the occupied domains: screening the pair list must
    /// reduce the number of PNO blocks built.
    #[test]
    fn pno_composes_with_pair_domain_screening() {
        let centers = array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [30.0, 0.0, 0.0]];
        let nvir = 4;
        let all = complete_pair_domains(&centers).unwrap();
        let screened = build_pair_domains(&centers, 5.0, f64::INFINITY).unwrap();
        assert!(screened.pairs.len() < all.pairs.len(), "test premise");

        let p_all = build_pno_transforms(&all, nvir, 0.0, |i, j| amp(nvir, i, j)).unwrap();
        let p_scr = build_pno_transforms(&screened, nvir, 0.0, |i, j| amp(nvir, i, j)).unwrap();
        assert!(
            p_scr.pairs.len() < p_all.pairs.len(),
            "domain screening must reduce the PNO block count ({} vs {})",
            p_scr.pairs.len(),
            p_all.pairs.len()
        );
    }

    /// Bad inputs error rather than panicking or silently producing nonsense.
    #[test]
    fn invalid_inputs_are_rejected() {
        let d = complete_pair_domains(&centers3()).unwrap();
        assert!(build_pno_transforms(&d, 4, -1.0, |i, j| amp(4, i, j)).is_err());
        // Amplitude block of the wrong size.
        assert!(build_pno_transforms(&d, 4, 0.0, |i, j| amp(3, i, j)).is_err());
    }
}
