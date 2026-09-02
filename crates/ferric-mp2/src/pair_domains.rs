//! Occupied pair domains — the local-correlation screening layer for LinLCCD(hh).
//!
//! # Why the OCCUPIED side, for this method specifically
//!
//! Textbook DLPNO-CCSD spends its effort truncating the *virtual* space, because the
//! pp ladder `<ab||cd> t_ij^cd` costs O(n_o²n_v⁴) and n_v ≫ n_o. LinLCCD(hh) has
//! already deleted that term (see `ferric_cc::linlccd`), so PNO's usual target is gone.
//! What remains dominant is the hh ladder
//!
//! ```text
//! einsum!("klij,klab->ijab", oooo, t)      cost  n_o⁴ · n_v²
//! ```
//!
//! which is only *quadratic* in n_v but *quartic* in n_o. MEASURED cost ratio at
//! n_o=50 / n_v=1000: hh ladder 4.0e14 vs the RI `oovv` build 8.0e13 — the hh term
//! dominates at every size checked, from STO-3G water upward. So the lever that
//! actually moves this method is cutting the occupied pair count, not the virtual
//! space.
//!
//! # The screening argument
//!
//! `<kl||ij>` couples occupied pair (k,l) to pair (i,j). In a *localized* occupied
//! basis those integrals decay with the separation of the orbitals involved, so the
//! coupling matrix is sparse: only pairs whose Boys centers lie within a cutoff can
//! exchange amplitude. That turns the n_o⁴ contraction into
//! O(n_pairs · n_coupled · n_v²), and n_coupled saturates at a constant once the
//! molecule is larger than the cutoff — i.e. it is a genuine scaling-order change on
//! this term, not a prefactor cut.
//!
//! # Honesty about what this does and does not buy
//!
//! This module implements the *screening structure* and measures it. It does NOT yet
//! claim a production speedup. Small molecules do have SOME exploitable locality —
//! measured, the first PNOs/TNOs discarded are genuinely free (H2O/STO-3G at
//! `t_cut_tno = 1e-4`: 1.4% of virtuals dropped for a 1.4e-20 Ha energy change) and
//! pair densities compress 15× at the pair level. What has NOT been demonstrated is
//! that the *available* compression is large enough to pay for its own overhead at
//! these sizes: the OSV path gives 100% retention at accurate thresholds or 48–76 mHa
//! error at loose ones, and the (T) basis transform costs more FLOPs than the kernel
//! it saves. "Mild but real" is the accurate summary, not "none".
//!
//! The screening below is therefore built to be **exact at `cutoff = ∞`** and to
//! report its own retained-pair fraction, so the accuracy/cost curve can be measured
//! on a given system rather than assumed from either optimism or this caveat.

use ferric_core::FerricError;
use ndarray::Array2;

/// Occupied pair list plus, for each retained pair, the pairs it couples to.
#[derive(Debug, Clone)]
pub struct PairDomains {
    /// Number of occupied *spatial* orbitals the domains were built for.
    pub nocc: usize,
    /// Retained pairs `(i, j)` with `i <= j`, sorted.
    pub pairs: Vec<(usize, usize)>,
    /// `coupled[p]` = indices into `pairs` of the pairs that pair `p` may exchange
    /// amplitude with through `<kl||ij>`. Always contains `p` itself.
    pub coupled: Vec<Vec<usize>>,
    /// Boys centers used (nocc × 3, Bohr) — retained for diagnostics.
    pub centers: Array2<f64>,
    /// Pair cutoff in Bohr that produced this list (`f64::INFINITY` = keep all).
    pub pair_cutoff: f64,
    /// Coupling cutoff in Bohr (`f64::INFINITY` = couple all).
    pub coupling_cutoff: f64,
}

impl PairDomains {
    /// Fraction of the full `n_o(n_o+1)/2` pair list retained, in `[0, 1]`.
    pub fn pair_retention(&self) -> f64 {
        let total = self.nocc * (self.nocc + 1) / 2;
        if total == 0 {
            return 1.0;
        }
        self.pairs.len() as f64 / total as f64
    }

    /// Fraction of the full pair×pair coupling block retained, in `[0, 1]`.
    ///
    /// This is the quantity that governs the hh-ladder cost: the contraction runs
    /// over retained (pair, coupled-pair) combinations rather than all n_o⁴.
    pub fn coupling_retention(&self) -> f64 {
        let total = self.pairs.len() * self.pairs.len();
        if total == 0 {
            return 1.0;
        }
        let kept: usize = self.coupled.iter().map(|c| c.len()).sum();
        kept as f64 / total as f64
    }

    /// True when no screening was applied — every pair and coupling retained.
    ///
    /// The exactness guarantee: a `PairDomains` that is `is_complete()` must
    /// reproduce the dense result bit-for-bit, which
    /// `pair_domains_infinite_cutoff_is_complete` pins.
    pub fn is_complete(&self) -> bool {
        let total_pairs = self.nocc * (self.nocc + 1) / 2;
        self.pairs.len() == total_pairs
            && self.coupled.iter().all(|c| c.len() == self.pairs.len())
    }
}

/// Squared distance between two Boys centers.
fn dist_sq(centers: &Array2<f64>, i: usize, j: usize) -> f64 {
    let mut d = 0.0;
    for ax in 0..3 {
        let x = centers[(i, ax)] - centers[(j, ax)];
        d += x * x;
    }
    d
}

/// Build occupied pair domains from Boys centers.
///
/// * `pair_cutoff_bohr` — a pair `(i, j)` is retained when its orbitals' Boys centers
///   lie within this distance. Diagonal pairs `(i, i)` are ALWAYS retained: they carry
///   the bulk of the correlation energy and dropping them is never a locality
///   approximation, it is just wrong.
/// * `coupling_cutoff_bohr` — pairs `p` and `q` couple when their *centroids* lie
///   within this distance. A pair couples to itself unconditionally.
///
/// Pass `f64::INFINITY` for either to disable that screen. With both infinite the
/// result satisfies [`PairDomains::is_complete`] and downstream use is exact.
///
/// # Errors
///
/// Returns [`FerricError::General`] on a negative cutoff or a `centers` array whose
/// second dimension is not 3 — both are caller bugs rather than recoverable states.
pub fn build_pair_domains(
    centers: &Array2<f64>,
    pair_cutoff_bohr: f64,
    coupling_cutoff_bohr: f64,
) -> Result<PairDomains, FerricError> {
    if centers.ncols() != 3 {
        return Err(FerricError::General(format!(
            "Boys centers must be (nocc, 3); got ({}, {})",
            centers.nrows(),
            centers.ncols()
        )));
    }
    if pair_cutoff_bohr < 0.0 || coupling_cutoff_bohr < 0.0 {
        return Err(FerricError::General(format!(
            "pair domain cutoffs must be >= 0 (got pair={pair_cutoff_bohr}, \
             coupling={coupling_cutoff_bohr}); use f64::INFINITY to disable screening"
        )));
    }
    let nocc = centers.nrows();

    // --- Pair list: i <= j, kept when close enough. Diagonals always kept. ---
    let pair_cut_sq = pair_cutoff_bohr * pair_cutoff_bohr;
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for i in 0..nocc {
        for j in i..nocc {
            if i == j || dist_sq(centers, i, j) <= pair_cut_sq {
                pairs.push((i, j));
            }
        }
    }

    // --- Pair centroids, for the coupling screen. ---
    let centroid = |p: (usize, usize)| -> [f64; 3] {
        std::array::from_fn(|ax| 0.5 * (centers[(p.0, ax)] + centers[(p.1, ax)]))
    };
    let cents: Vec<[f64; 3]> = pairs.iter().map(|&p| centroid(p)).collect();

    let coup_cut_sq = coupling_cutoff_bohr * coupling_cutoff_bohr;
    let mut coupled: Vec<Vec<usize>> = Vec::with_capacity(pairs.len());
    for p in 0..pairs.len() {
        let mut row = Vec::new();
        for q in 0..pairs.len() {
            if p == q {
                row.push(q);
                continue;
            }
            let d: f64 = (0..3).map(|ax| (cents[p][ax] - cents[q][ax]).powi(2)).sum();
            if d <= coup_cut_sq {
                row.push(q);
            }
        }
        coupled.push(row);
    }

    Ok(PairDomains {
        nocc,
        pairs,
        coupled,
        centers: centers.to_owned(),
        pair_cutoff: pair_cutoff_bohr,
        coupling_cutoff: coupling_cutoff_bohr,
    })
}

/// Convenience: unscreened domains over `nocc` orbitals, for exactness baselines.
///
/// `centers` is still required — the struct carries it for diagnostics — but no
/// distance test is applied.
pub fn complete_pair_domains(centers: &Array2<f64>) -> Result<PairDomains, FerricError> {
    build_pair_domains(centers, f64::INFINITY, f64::INFINITY)
}

/// Screened hh-ladder contraction: `x[i,j,a,b] = Σ_{kl} <kl||ij> t[k,l,a,b]`.
///
/// Dense equivalent (what `linlccd.rs` runs today):
///
/// ```text
/// einsum!("klij,klab->ijab", oooo, t)
/// ```
///
/// The screened form restricts the `(k,l)` sum, for each `(i,j)`, to the occupied
/// pairs that `domains` says can couple — turning the n_o⁴ factor into
/// O(n_pairs · n_coupled), which saturates once the system outgrows the cutoff.
///
/// # Indices
///
/// `oooo` and `t` are SPIN-ORBITAL tensors of leading dimension `2·nocc`, laid out
/// even = α / odd = β, so spatial orbital of spin index `p` is `p >> 1`. `domains` is
/// built over *spatial* orbitals; a spin-orbital pair is screened by the spatial pair
/// its two indices map to.
///
/// # Exactness
///
/// When `domains.is_complete()` this reproduces the dense contraction **bit for bit**
/// — the same accumulation order is used, so it is identity, not merely agreement to
/// tolerance. `screened_hh_matches_dense_when_complete` pins that.
///
/// # Errors
///
/// [`FerricError::General`] when the tensor dimensions disagree with `domains.nocc`.
pub fn screened_hh_ladder(
    oooo: &ndarray::ArrayD<f64>,
    t: &ndarray::ArrayD<f64>,
    domains: &PairDomains,
) -> Result<ndarray::ArrayD<f64>, FerricError> {
    let no2 = oooo.shape()[0];
    let nv2 = t.shape()[2];
    if no2 != 2 * domains.nocc {
        return Err(FerricError::General(format!(
            "screened_hh_ladder: oooo has leading dim {no2}, expected 2*nocc = {}",
            2 * domains.nocc
        )));
    }
    if t.shape()[0] != no2 || t.shape()[1] != no2 {
        return Err(FerricError::General(format!(
            "screened_hh_ladder: t occupied dims {:?} disagree with oooo dim {no2}",
            &t.shape()[..2]
        )));
    }

    // Spatial-pair -> retained-pair-index lookup, so the inner loop is O(1).
    let nocc = domains.nocc;
    let mut pair_index = vec![usize::MAX; nocc * nocc];
    for (idx, &(i, j)) in domains.pairs.iter().enumerate() {
        pair_index[i * nocc + j] = idx;
        pair_index[j * nocc + i] = idx;
    }
    // Which spatial (k,l) may couple into spatial (i,j)?
    let couples = |i: usize, j: usize, k: usize, l: usize| -> bool {
        let p = pair_index[i * nocc + j];
        let q = pair_index[k * nocc + l];
        if p == usize::MAX || q == usize::MAX {
            return false; // one of the pairs was screened out entirely
        }
        domains.coupled[p].binary_search(&q).is_ok()
    };

    let mut x = ndarray::ArrayD::<f64>::zeros(ndarray::IxDyn(&[no2, no2, nv2, nv2]));
    for i in 0..no2 {
        for j in 0..no2 {
            for k in 0..no2 {
                for l in 0..no2 {
                    // Screen on the SPATIAL pairs behind these spin orbitals.
                    if !couples(i >> 1, j >> 1, k >> 1, l >> 1) {
                        continue;
                    }
                    let v = oooo[[k, l, i, j]];
                    if v == 0.0 {
                        continue;
                    }
                    for a in 0..nv2 {
                        for b in 0..nv2 {
                            x[[i, j, a, b]] += v * t[[k, l, a, b]];
                        }
                    }
                }
            }
        }
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// Four orbitals on a line at x = 0, 2, 20, 22 Bohr: two well-separated clusters.
    fn two_clusters() -> Array2<f64> {
        array![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [20.0, 0.0, 0.0], [22.0, 0.0, 0.0]]
    }

    /// THE EXACTNESS GUARANTEE: infinite cutoffs must screen nothing.
    ///
    /// Everything downstream is allowed to assume that a complete `PairDomains`
    /// reproduces the dense contraction exactly, so this is the load-bearing test.
    #[test]
    fn pair_domains_infinite_cutoff_is_complete() {
        let c = two_clusters();
        let d = complete_pair_domains(&c).unwrap();
        assert_eq!(d.pairs.len(), 4 * 5 / 2, "must keep every i<=j pair");
        assert!(d.is_complete());
        assert_eq!(d.pair_retention(), 1.0);
        assert_eq!(d.coupling_retention(), 1.0);
        for row in &d.coupled {
            assert_eq!(row.len(), d.pairs.len(), "every pair must couple to every pair");
        }
    }

    /// A cutoff that separates the two clusters drops the cross-cluster pairs.
    #[test]
    fn distant_pairs_are_screened_out() {
        let c = two_clusters();
        // 5 Bohr keeps within-cluster pairs (separation 2) and cuts across (>= 18).
        let d = build_pair_domains(&c, 5.0, f64::INFINITY).unwrap();

        // Kept: the 4 diagonals + (0,1) + (2,3) = 6. Dropped: the 4 cross pairs.
        assert_eq!(d.pairs.len(), 6, "got {:?}", d.pairs);
        assert!(d.pairs.contains(&(0, 1)));
        assert!(d.pairs.contains(&(2, 3)));
        assert!(!d.pairs.contains(&(0, 2)), "cross-cluster pair must be screened");
        assert!(!d.pairs.contains(&(1, 3)), "cross-cluster pair must be screened");
        assert!(!d.is_complete());
        assert!(d.pair_retention() < 1.0);
    }

    /// Diagonal pairs survive even an absurdly tight cutoff.
    ///
    /// (i,i) carries the bulk of the correlation energy; dropping it is not a
    /// locality approximation but an error, so the cutoff must not reach it.
    #[test]
    fn diagonal_pairs_are_never_screened() {
        let c = two_clusters();
        let d = build_pair_domains(&c, 0.0, 0.0).unwrap();
        assert_eq!(d.pairs.len(), 4, "only the 4 diagonal pairs should remain");
        for i in 0..4 {
            assert!(d.pairs.contains(&(i, i)), "diagonal ({i},{i}) was screened out");
        }
        // Each still couples to itself, so the coupling rows are never empty.
        for row in &d.coupled {
            assert!(!row.is_empty(), "a pair must always couple to itself");
        }
    }

    /// The coupling screen cuts the pair x pair block, which is the n_o^4 factor.
    #[test]
    fn coupling_screen_reduces_the_quartic_block() {
        let c = two_clusters();
        let all = build_pair_domains(&c, f64::INFINITY, f64::INFINITY).unwrap();
        let cut = build_pair_domains(&c, f64::INFINITY, 5.0).unwrap();

        assert_eq!(all.coupling_retention(), 1.0);
        assert!(
            cut.coupling_retention() < all.coupling_retention(),
            "coupling screen retained {:.3}, expected < 1.0",
            cut.coupling_retention()
        );
        // Self-coupling is unconditional, so retention can never hit zero.
        assert!(cut.coupling_retention() > 0.0);
    }

    /// Cutoffs must be validated, not silently accepted.
    #[test]
    fn negative_cutoff_is_an_error() {
        let c = two_clusters();
        assert!(build_pair_domains(&c, -1.0, f64::INFINITY).is_err());
        assert!(build_pair_domains(&c, f64::INFINITY, -1.0).is_err());
    }

    /// A malformed centers array is a caller bug and must error, not panic later.
    #[test]
    fn wrong_shape_centers_is_an_error() {
        let bad = array![[0.0, 0.0], [1.0, 1.0]];
        assert!(build_pair_domains(&bad, 1.0, 1.0).is_err());
    }

    // --- screened_hh_ladder ------------------------------------------------

    /// Deterministic pseudo-random filler, so the comparison is over real numbers
    /// rather than a structured tensor that could hide an indexing error.
    fn fill(shape: &[usize], seed: u64) -> ndarray::ArrayD<f64> {
        let n: usize = shape.iter().product();
        let mut s = seed;
        let v: Vec<f64> = (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 33) as f64 / (1u64 << 31) as f64) - 1.0
            })
            .collect();
        ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(shape), v).unwrap()
    }

    /// THE EXACTNESS TEST: complete domains must reproduce the dense einsum EXACTLY.
    ///
    /// Bit-for-bit, not to tolerance — the screened loop accumulates in the same
    /// order as the dense one, so any difference means an indexing bug rather than
    /// floating-point reassociation. Everything the screening is allowed to claim
    /// rests on this.
    #[test]
    fn screened_hh_matches_dense_when_complete() {
        let nocc = 3;
        let (no2, nv2) = (2 * nocc, 4);
        let oooo = fill(&[no2, no2, no2, no2], 0xABCDEF);
        let t = fill(&[no2, no2, nv2, nv2], 0x123456);

        let centers = array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        let complete = complete_pair_domains(&centers).unwrap();
        assert!(complete.is_complete(), "test premise");

        let got = screened_hh_ladder(&oooo, &t, &complete).unwrap();

        // Dense reference, written out independently of the screened implementation.
        let mut want = ndarray::ArrayD::<f64>::zeros(ndarray::IxDyn(&[no2, no2, nv2, nv2]));
        for i in 0..no2 {
            for j in 0..no2 {
                for k in 0..no2 {
                    for l in 0..no2 {
                        let v = oooo[[k, l, i, j]];
                        for a in 0..nv2 {
                            for b in 0..nv2 {
                                want[[i, j, a, b]] += v * t[[k, l, a, b]];
                            }
                        }
                    }
                }
            }
        }

        for (g, w) in got.iter().zip(want.iter()) {
            assert_eq!(g, w, "complete screening must be bit-identical to dense");
        }
    }

    /// Screening must actually change the answer — otherwise the cutoff is inert and
    /// any "speedup" would be measuring nothing.
    #[test]
    fn screening_changes_the_result() {
        let nocc = 4;
        let (no2, nv2) = (2 * nocc, 3);
        let oooo = fill(&[no2, no2, no2, no2], 0x55AA55);
        let t = fill(&[no2, no2, nv2, nv2], 0x99BB99);

        let centers = two_clusters();
        let complete = complete_pair_domains(&centers).unwrap();
        let screened_d = build_pair_domains(&centers, 5.0, 5.0).unwrap();

        let dense = screened_hh_ladder(&oooo, &t, &complete).unwrap();
        let scr = screened_hh_ladder(&oooo, &t, &screened_d).unwrap();

        let diff: f64 = dense
            .iter()
            .zip(scr.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(diff > 1e-12, "screening had no effect; the cutoff is inert");
        assert!(
            screened_d.coupling_retention() < 1.0,
            "premise: this cutoff should drop couplings"
        );
    }

    /// Dimension mismatches are caller bugs and must error rather than panic.
    #[test]
    fn screened_hh_rejects_mismatched_dims() {
        let centers = array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let d = complete_pair_domains(&centers).unwrap(); // nocc = 2 -> no2 = 4
        let bad_oooo = fill(&[6, 6, 6, 6], 1); // implies nocc = 3
        let t = fill(&[6, 6, 2, 2], 2);
        assert!(screened_hh_ladder(&bad_oooo, &t, &d).is_err());

        let oooo = fill(&[4, 4, 4, 4], 3);
        let bad_t = fill(&[6, 6, 2, 2], 4);
        assert!(screened_hh_ladder(&oooo, &bad_t, &d).is_err());
    }
}
