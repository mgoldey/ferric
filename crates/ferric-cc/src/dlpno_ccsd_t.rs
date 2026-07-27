//! DLPNO-CCSD(T) — triple-domain screening for closed-shell perturbative triples.
//!
//! # Scope, stated up front
//!
//! This is the **triple-screening half** of DLPNO-CCSD(T), not the full method.
//! It screens which occupied triples `(i,j,k)` are evaluated by
//! [`crate::ccsd_t_closed_shell`], attacking the `n_o³` triple count exactly as
//! [`crate::dlpno_ccsd`] attacks the `n_o²` pair count and
//! [`ferric_mp2::pair_domains`] attacks LinLCCD(hh)'s `n_o⁴` coupling block.
//!
//! It does **not** build per-triple PNO virtual bases. Real DLPNO-(T) evaluates
//! each retained triple in a `[TNO]³` block whose dimension is far below `nv`,
//! and that is where most of the published speedup lives. Doing that means
//! re-deriving `raw_w_block`'s two GEMMs, the six-permutation `W`, the `r3`
//! pattern and the `V` singles terms in a virtual basis that differs per triple
//! — a much larger change than this. Shipping that behind an untested
//! "DLPNO-CCSD(T)" label would be worse than shipping the honest half.
//!
//! # Why the triple list is the right first half
//!
//! The triple list composes with everything downstream: `ccsd_t_closed_shell`'s
//! kernel is already a *per-triple* streaming loop over
//! [`occ_triples_with_repeats`](crate::ccsd_t_closed_shell)-shaped bands, and
//! each triple's work is independent. Dropping a triple from the list removes
//! its entire `O(nv³)` working set — six `raw_w` GEMMs, the `r3` permutations
//! and the `W̃·V/D` reduction — without touching the triples algebra at all. And
//! as with pairs, per-triple TNOs are only defined *for retained triples*, so
//! the list has to be right before a per-triple virtual basis means anything.
//!
//! # Exactness contract
//!
//! With an infinite cutoff [`build_triple_domains`] retains every `i<=j<=k`
//! triple, [`TripleDomains::is_complete`] is true, and
//! [`screened_triple_energy`] reproduces the unscreened weighted sum **bit for
//! bit** — same accumulation order, so it is identity rather than agreement to
//! tolerance. `triple_domains_infinite_cutoff_is_complete` and
//! `screened_energy_is_identity_when_complete` pin that.
//!
//! # THE DIVISOR / MULTIPLICITY TRAP
//!
//! This is the single most dangerous thing in this module, and it is a trap that
//! does **not** exist for pairs.
//!
//! [`crate::ccsd_t_closed_shell`] bands over occupied triples `i<=j<=k` with an
//! explicit multiplicity weight `m(i,j,k) ∈ {1, 3, 6}` and an overall divisor of
//! **3** — NOT the `i<j<k` + `/6` convention of the spin-orbital sibling
//! [`crate::ccsd_t`]. The reason is physical: `i`, `j`, `k` label *spatial*
//! orbitals, each holding two electrons, so repeated indices `i=j` and `i=j=k`
//! are allowed and numerically large — 58% of the energy on the derivation
//! system quoted in that module's doc.
//!
//! Two consequences bind this module:
//!
//! 1. **The screening predicate must be a function of the unordered multiset
//!    `{i,j,k}` only.** The weight `m` stands in for `m` distinct *orderings* of
//!    the same triple, all of which the unrestricted sum would evaluate. A
//!    predicate that could accept one ordering and reject another would make the
//!    banded weighted sum stop equalling the unrestricted sum, and the `/3`
//!    divisor would silently become wrong. [`triple_is_retained`] is built from
//!    the max pairwise Boys separation, which is manifestly symmetric;
//!    `retention_is_permutation_invariant` pins it over all 6 orderings.
//!
//! 2. **Screening must never touch the repeated-index classes.** `m=1`
//!    (`i=j=k`) and `m=3` (exactly two equal) triples are the diagonal-analogue
//!    of `dlpno_ccsd`'s `(i,i)` pairs: they carry large weight and dropping them
//!    is an error, not an approximation. They are retained unconditionally, at
//!    any cutoff including zero. Note this is *automatic* under the max-pairwise
//!    rule for `m=1` (all separations are 0) but **not** for `m=3`, where the
//!    two distinct orbitals can be arbitrarily far apart — so `m=3` is
//!    force-retained explicitly. `repeated_index_triples_are_never_screened`
//!    pins both classes.
//!
//! Because screening only ever removes whole multiset classes and never
//! reweights a retained one, [`TripleDomains`] carries the same `m(i,j,k)` and
//! the same `/3` divisor as the dense path. `multiplicity_classes_are_preserved`
//! and `complete_domains_reproduce_the_unrestricted_count` pin that the weights
//! coming out of this module are the dense weights, unmodified.
//!
//! # What is NOT claimed
//!
//! No wall-clock measurement, no speedup claim. Screening is reported as
//! *counts* ([`TripleDomains::triple_retention`],
//! [`TripleDomains::weighted_retention`]) and validated on *energies*.
//!
//! Note the two retention numbers are NOT interchangeable, and the plain triple
//! count is the flattering one. Screening can only ever remove the all-distinct
//! `m=6` class, so every dropped triple costs 1 band entry but 6 of the `nocc³`
//! weighted orderings — measured on the two-cluster fixture at nocc=4, a cutoff
//! that retains 0.8 of the band retains only 0.625 of the weighted sum. Quote
//! [`TripleDomains::weighted_retention`] when the question is how much physics
//! was dropped. Whether
//! a retention below 1 is affordable at a target accuracy is an empirical
//! question this module deliberately leaves to measurement — ferric's history
//! with sparsity methods (see `ferric_mp2::pair_domains`' module doc and the
//! OSV finding) is that small molecules have no exploitable locality.

use ferric_core::FerricError;
use ndarray::Array2;

/// Retained occupied triples `(i,j,k)` with `i <= j <= k`, plus their dense
/// multiplicity weights.
///
/// The weights are **not** modified by screening — they are exactly the
/// `m(i,j,k) ∈ {1,3,6}` of [`crate::ccsd_t_closed_shell`], carried here so a
/// consumer never has to re-derive them (re-deriving them is precisely how the
/// `/6` spin-orbital convention leaks in and produces a silent >50% error).
#[derive(Debug, Clone)]
pub struct TripleDomains {
    /// Number of occupied *spatial* orbitals the domains were built for.
    pub nocc: usize,
    /// Retained triples `(i, j, k)` with `i <= j <= k`, ascending lexicographic
    /// — the same order [`crate::ccsd_t_closed_shell`] enumerates, so a screened
    /// reduction folds partials in the same sequence as the dense one.
    pub triples: Vec<(usize, usize, usize)>,
    /// `weights[t]` = `m(triples[t])` ∈ {1, 3, 6}: the number of distinct
    /// orderings the banded triple stands for. Parallel to `triples`.
    pub weights: Vec<f64>,
    /// Boys centers used (nocc × 3, Bohr) — retained for diagnostics.
    pub centers: Array2<f64>,
    /// Triple cutoff in Bohr that produced this list (`f64::INFINITY` = keep
    /// all). A triple is retained when its **maximum pairwise** Boys-center
    /// separation is within this distance.
    pub triple_cutoff: f64,
}

/// Total number of `i <= j <= k` triples over `nocc` orbitals — the size of the
/// unscreened band. `C(nocc + 2, 3)` (multiset coefficient).
fn total_banded_triples(nocc: usize) -> usize {
    nocc * (nocc + 1) * (nocc + 2) / 6
}

/// Number of distinct orderings of the occupied triple `(i,j,k)` given
/// `i <= j <= k`.
///
/// Deliberately a local copy of `ccsd_t_closed_shell::occ_triple_multiplicity`
/// (which is private) rather than an approximation of it. The two are pinned
/// against each other's defining property — the weighted band must sum to
/// `nocc³`, the unrestricted ordered-triple count — by
/// `complete_domains_reproduce_the_unrestricted_count`.
fn occ_triple_multiplicity(i: usize, j: usize, k: usize) -> f64 {
    debug_assert!(i <= j && j <= k);
    if i == k {
        1.0 // i == j == k
    } else if i == j || j == k {
        3.0
    } else {
        6.0
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

/// Is triple `(i,j,k)` retained at `cutoff_bohr`?
///
/// The criterion is the **maximum pairwise** Boys-center separation:
///
/// ```text
/// retained  ⟺  max(|r_i − r_j|, |r_i − r_k|, |r_j − r_k|) <= cutoff
/// ```
///
/// Two properties make this the right predicate, and both are load-bearing for
/// the multiplicity bookkeeping described in the module doc:
///
/// * **Symmetric under any permutation of `(i,j,k)`** — a max over the three
///   unordered pairs cannot distinguish orderings, so the banded weight `m`
///   still stands for `m` orderings that would *all* have been retained.
/// * **Never rejects a repeated-index triple.** `i=j=k` has all separations 0.
///   `i=j≠k` is force-retained here: its max separation is `|r_i − r_k|`, which
///   a tight cutoff would otherwise reject even though the `m=3` class carries
///   large weight (module doc, point 2).
///
/// Max-pairwise (rather than, say, a centroid radius) also makes the triple
/// screen *consistent with* the pair screen: a triple is retained only if all
/// three of its constituent pairs would be, so a triple can never survive on
/// pairs that were themselves screened away.
pub fn triple_is_retained(
    centers: &Array2<f64>,
    i: usize,
    j: usize,
    k: usize,
    cutoff_bohr: f64,
) -> bool {
    // Repeated spatial indices are physical and carry large weight — never
    // screened, at any cutoff. See the module doc's DIVISOR / MULTIPLICITY TRAP.
    if i == j || j == k || i == k {
        return true;
    }
    let cut_sq = cutoff_bohr * cutoff_bohr;
    dist_sq(centers, i, j) <= cut_sq
        && dist_sq(centers, i, k) <= cut_sq
        && dist_sq(centers, j, k) <= cut_sq
}

impl TripleDomains {
    /// Fraction of the full `C(nocc+2, 3)` banded triple list retained, in
    /// `[0, 1]`.
    ///
    /// This counts *banded* triples, which is what the `(T)` loop iterates and
    /// therefore what the screen actually saves.
    pub fn triple_retention(&self) -> f64 {
        let total = total_banded_triples(self.nocc);
        if total == 0 {
            return 1.0;
        }
        self.triples.len() as f64 / total as f64
    }

    /// Fraction of the *weighted* triple count retained, in `[0, 1]`.
    ///
    /// The honest cost proxy, and it is the SMALLER of the two. Screening can
    /// only remove the all-distinct `m=6` class (repeated-index triples are
    /// never screened), which is the class carrying the *most* weight per
    /// triple — so each dropped triple removes 6 of the `nocc³` orderings while
    /// costing only 1 of the `C(nocc+2,3)` band entries.
    ///
    /// Plain [`triple_retention`](Self::triple_retention) therefore
    /// **overstates** how much of the unrestricted sum survives. Measured on the
    /// two-cluster fixture at nocc=4: 16/20 = 0.8 of the band retained, but only
    /// 40/64 = 0.625 of the weighted sum. Report both, and quote this one when
    /// the question is how much physics was dropped.
    pub fn weighted_retention(&self) -> f64 {
        let total = self.nocc * self.nocc * self.nocc;
        if total == 0 {
            return 1.0;
        }
        let kept: f64 = self.weights.iter().sum();
        kept / total as f64
    }

    /// True when no screening was applied — every `i <= j <= k` triple retained.
    ///
    /// The exactness guarantee: a `TripleDomains` that is `is_complete()` must
    /// reproduce the dense weighted sum bit for bit, which
    /// `screened_energy_is_identity_when_complete` pins.
    pub fn is_complete(&self) -> bool {
        self.triples.len() == total_banded_triples(self.nocc)
    }
}

/// Build occupied triple domains from Boys centers.
///
/// * `cutoff_bohr` — a triple `(i,j,k)` is retained when its maximum pairwise
///   Boys-center separation is within this distance (see
///   [`triple_is_retained`]). Triples with any repeated index are ALWAYS
///   retained: they carry `m ∈ {1,3}` weight in the closed-shell `(T)` band and
///   dropping them is not a locality approximation, it is just wrong.
///
/// Pass `f64::INFINITY` to disable the screen; the result then satisfies
/// [`TripleDomains::is_complete`] and downstream use is exact.
///
/// The emitted `weights` are the *dense* multiplicities `m(i,j,k) ∈ {1,3,6}` —
/// screening removes whole triples, it never reweights a retained one.
///
/// # Errors
///
/// Returns [`FerricError::General`] on a negative cutoff or a `centers` array
/// whose second dimension is not 3 — both are caller bugs rather than
/// recoverable states.
pub fn build_triple_domains(
    centers: &Array2<f64>,
    cutoff_bohr: f64,
) -> Result<TripleDomains, FerricError> {
    if centers.ncols() != 3 {
        return Err(FerricError::General(format!(
            "Boys centers must be (nocc, 3); got ({}, {})",
            centers.nrows(),
            centers.ncols()
        )));
    }
    if cutoff_bohr < 0.0 {
        return Err(FerricError::General(format!(
            "triple domain cutoff must be >= 0 (got {cutoff_bohr}); \
             use f64::INFINITY to disable screening"
        )));
    }
    let nocc = centers.nrows();

    // Ascending lexicographic i <= j <= k — the SAME enumeration order as
    // ccsd_t_closed_shell::occ_triples_with_repeats, so a screened reduction
    // folds partials in the dense path's sequence and stays bit-comparable.
    let mut triples: Vec<(usize, usize, usize)> = Vec::new();
    let mut weights: Vec<f64> = Vec::new();
    for i in 0..nocc {
        for j in i..nocc {
            for k in j..nocc {
                if triple_is_retained(centers, i, j, k, cutoff_bohr) {
                    triples.push((i, j, k));
                    weights.push(occ_triple_multiplicity(i, j, k));
                }
            }
        }
    }

    Ok(TripleDomains {
        nocc,
        triples,
        weights,
        centers: centers.to_owned(),
        triple_cutoff: cutoff_bohr,
    })
}

/// Convenience: unscreened triple domains, for exactness baselines.
///
/// `centers` is still required — the struct carries it for diagnostics — but no
/// distance test is applied.
pub fn complete_triple_domains(centers: &Array2<f64>) -> Result<TripleDomains, FerricError> {
    build_triple_domains(centers, f64::INFINITY)
}

/// Screened closed-shell `(T)` energy from a per-triple contribution callback.
///
/// This is the triples analogue of [`crate::dlpno_ccsd::apply_pair_mask`]. It
/// cannot be a *mask on a tensor*, because `ccsd_t_closed_shell` never
/// materializes a triples tensor — the whole point of its streaming formulation
/// is that the `[nv,nv,nv]` block for triple `t` is built, consumed and dropped.
/// So the screening surface is the triple *list*, and this function is the
/// reduction that consumes it.
///
/// `contribution(i, j, k)` must return the **unweighted** per-triple sum
/// `Σ_{a,b,c} W̃[a,b,c]·V[a,b,c] / D[a,b,c]` — i.e. exactly the `partial` local
/// in `ccsd_t_closed_shell`'s parallel map, *before* its `mult * partial`. This
/// function applies the multiplicity weight and the divisor itself, which is the
/// point: a caller cannot accidentally reintroduce the spin-orbital `/6`
/// convention, because it never sees the divisor.
///
/// ```text
/// E_(T) = (1/3) Σ_{retained (i<=j<=k)} m(i,j,k) · contribution(i,j,k)
/// ```
///
/// # Determinism and exactness
///
/// Contributions are accumulated serially in ascending triple order, the same
/// order and the same `mult * partial` then `+=` sequence the dense path uses.
/// With [`TripleDomains::is_complete`] domains the result is therefore
/// **bit-identical** to the dense weighted sum, not merely equal to tolerance.
///
/// # Errors
///
/// Propagates any error from `contribution`, and returns
/// [`FerricError::General`] if `domains` is internally inconsistent (weights not
/// parallel to triples — only reachable by hand-constructing the struct).
pub fn screened_triple_energy<F>(
    domains: &TripleDomains,
    mut contribution: F,
) -> Result<f64, FerricError>
where
    F: FnMut(usize, usize, usize) -> Result<f64, FerricError>,
{
    if domains.weights.len() != domains.triples.len() {
        return Err(FerricError::General(format!(
            "screened_triple_energy: {} weights for {} triples",
            domains.weights.len(),
            domains.triples.len()
        )));
    }
    let mut et = 0.0f64;
    for (&(i, j, k), &mult) in domains.triples.iter().zip(domains.weights.iter()) {
        et += mult * contribution(i, j, k)?;
    }
    // Divisor 3 — NOT 6, NOT 36. See ccsd_t_closed_shell's "Combinatorial
    // factor" section: the weighted i<=j<=k band equals the unrestricted
    // Σ_{i,j,k} Σ_{a,b,c} sum, which is exactly 3·E_(T).
    Ok(et / 3.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// Four orbitals on a line at x = 0, 2, 20, 22 Bohr: two well-separated
    /// clusters. Same fixture as `ferric_mp2::pair_domains`' tests.
    fn two_clusters() -> Array2<f64> {
        array![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [20.0, 0.0, 0.0], [22.0, 0.0, 0.0]]
    }

    fn line_centers(nocc: usize, spacing: f64) -> Array2<f64> {
        Array2::from_shape_fn((nocc, 3), |(i, ax)| {
            if ax == 0 {
                i as f64 * spacing
            } else {
                0.0
            }
        })
    }

    /// Deterministic pseudo-random per-triple contribution, so exactness
    /// comparisons run over real f64s rather than a structured pattern that
    /// could hide an indexing or weighting error.
    fn contrib(i: usize, j: usize, k: usize) -> f64 {
        let mut s = (i as u64)
            .wrapping_mul(0x9E3779B97F4A7C15)
            .wrapping_add((j as u64).wrapping_mul(0xBF58476D1CE4E5B9))
            .wrapping_add((k as u64).wrapping_mul(0x94D049BB133111EB))
            .wrapping_add(1);
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 33) as f64 / (1u64 << 31) as f64) - 1.0
    }

    // --- THE EXACTNESS CONTRACT --------------------------------------------

    /// THE EXACTNESS GUARANTEE, part 1: an infinite cutoff must screen nothing.
    #[test]
    fn triple_domains_infinite_cutoff_is_complete() {
        let c = two_clusters();
        let d = complete_triple_domains(&c).unwrap();

        // C(4+2, 3) = 20 banded triples over 4 occupied orbitals.
        assert_eq!(d.triples.len(), 20, "must keep every i<=j<=k triple");
        assert!(d.is_complete());
        assert_eq!(d.triple_retention(), 1.0);
        assert_eq!(d.weighted_retention(), 1.0);
    }

    /// THE EXACTNESS GUARANTEE, part 2 — the load-bearing test.
    ///
    /// With complete domains the screened reduction must reproduce the dense
    /// weighted band **bit for bit**. Asserted on raw f64 bits, not to a
    /// tolerance: the accumulation order is identical by construction, so any
    /// difference at all means a weighting or ordering bug, and everything this
    /// module is allowed to claim rests on this.
    #[test]
    fn screened_energy_is_identity_when_complete() {
        let nocc = 5;
        let centers = line_centers(nocc, 1.0);
        let d = complete_triple_domains(&centers).unwrap();
        assert!(d.is_complete(), "test premise");

        let got = screened_triple_energy(&d, |i, j, k| Ok(contrib(i, j, k))).unwrap();

        // Dense reference, written out independently of the screened path and
        // mirroring ccsd_t_closed_shell's own band + weight + /3.
        let mut want = 0.0f64;
        for i in 0..nocc {
            for j in i..nocc {
                for k in j..nocc {
                    want += occ_triple_multiplicity(i, j, k) * contrib(i, j, k);
                }
            }
        }
        want /= 3.0;

        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "complete triple screening must be bit-identical to the dense band: \
             {got:.17e} vs {want:.17e}"
        );
    }

    /// Exactness must not be an artifact of a tidy geometry: a cutoff large
    /// enough to reach every orbital is also the identity, on a geometry that
    /// *would* screen at a tighter cutoff.
    #[test]
    fn generous_finite_cutoff_is_also_the_identity() {
        let c = two_clusters(); // max separation 22 Bohr
        let d = build_triple_domains(&c, 1000.0).unwrap();
        assert!(d.is_complete(), "a cutoff past the whole molecule must screen nothing");

        let got = screened_triple_energy(&d, |i, j, k| Ok(contrib(i, j, k))).unwrap();
        let dense = screened_triple_energy(&complete_triple_domains(&c).unwrap(), |i, j, k| {
            Ok(contrib(i, j, k))
        })
        .unwrap();
        assert_eq!(got.to_bits(), dense.to_bits());
    }

    // --- THE DIVISOR / MULTIPLICITY TRAP -----------------------------------

    /// **The divisor trap, pinned.** The weighted `i<=j<=k` band must reproduce
    /// the unrestricted ordered `nocc³` count — that identity is the entire
    /// justification for `ccsd_t_closed_shell`'s `/3` divisor.
    ///
    /// If this module ever emitted the spin-orbital `i<j<k` convention (weight 6
    /// everywhere, divisor 6) this sum would come out as `6·C(nocc,3)` — e.g.
    /// 24 instead of 64 at nocc=4, a 62% error — so this test is what stands
    /// between the screening and a silent >50% energy loss.
    #[test]
    fn complete_domains_reproduce_the_unrestricted_count() {
        for nocc in 1..7usize {
            let d = complete_triple_domains(&line_centers(nocc, 1.0)).unwrap();
            let total: f64 = d.weights.iter().sum();
            assert_eq!(
                total,
                (nocc * nocc * nocc) as f64,
                "weighted i<=j<=k band must cover all nocc^3 ordered triples (nocc={nocc})"
            );
            // And the spin-orbital convention must NOT accidentally agree.
            let so_convention = 6.0 * (nocc * (nocc.saturating_sub(1)) * (nocc.saturating_sub(2))
                / 6) as f64;
            if nocc >= 3 {
                assert!(
                    (total - so_convention).abs() > 0.5,
                    "nocc={nocc}: the closed-shell count must differ from the \
                     spin-orbital i<j<k count ({total} vs {so_convention})"
                );
            }
        }
    }

    /// **Multiplicity classes must survive screening unmodified.** Screening
    /// removes whole triples; it must never reweight a retained one. A retained
    /// triple's weight must equal the dense `m(i,j,k)` exactly, in every class.
    #[test]
    fn multiplicity_classes_are_preserved() {
        let c = two_clusters();
        for &cut in &[0.0, 1.0, 5.0, 25.0, f64::INFINITY] {
            let d = build_triple_domains(&c, cut).unwrap();
            assert_eq!(d.weights.len(), d.triples.len());
            let (mut n1, mut n3, mut n6) = (0usize, 0usize, 0usize);
            for (&(i, j, k), &w) in d.triples.iter().zip(d.weights.iter()) {
                assert!(i <= j && j <= k, "band order violated at ({i},{j},{k})");
                assert_eq!(
                    w,
                    occ_triple_multiplicity(i, j, k),
                    "cutoff {cut}: triple ({i},{j},{k}) was reweighted by screening"
                );
                match w as usize {
                    1 => n1 += 1,
                    3 => n3 += 1,
                    6 => n6 += 1,
                    other => panic!("impossible multiplicity {other}"),
                }
            }
            // The two repeated-index classes are cutoff-independent: 4 triples
            // with i==j==k, and 3*C(4,2)=... explicitly, every (i,i,j) i!=j
            // ordered into the band gives 2 per unordered pair => 12.
            assert_eq!(n1, 4, "cutoff {cut}: all i==j==k triples must survive");
            assert_eq!(n3, 12, "cutoff {cut}: all two-equal triples must survive");
            // Only the all-distinct class may shrink.
            assert!(n6 <= 4, "cutoff {cut}: nocc=4 has C(4,3)=4 distinct triples");
        }
    }

    /// **Repeated-index triples are never screened**, at any cutoff including
    /// zero. Same reasoning as `dlpno_ccsd`'s diagonal pairs: `(i,i,i)` and
    /// `(i,i,j)` carry `m ∈ {1,3}` weight in the closed-shell band and dropping
    /// them is an error, not an approximation.
    ///
    /// Note `(i,i,j)` is the case that needs an explicit guard — its orbitals
    /// can be arbitrarily far apart, so a pure max-pairwise rule *would* reject
    /// it. Here the centers are 10 Bohr apart and the cutoff is 0.
    #[test]
    fn repeated_index_triples_are_never_screened() {
        let nocc = 4;
        let c = line_centers(nocc, 10.0);
        let d = build_triple_domains(&c, 0.0).unwrap();

        for i in 0..nocc {
            assert!(
                d.triples.contains(&(i, i, i)),
                "triple ({i},{i},{i}) was screened away"
            );
            for j in (i + 1)..nocc {
                assert!(
                    d.triples.contains(&(i, i, j)),
                    "triple ({i},{i},{j}) was screened away"
                );
                assert!(
                    d.triples.contains(&(i, j, j)),
                    "triple ({i},{j},{j}) was screened away"
                );
            }
        }
        // At cutoff 0 with 10 Bohr spacing, ONLY the repeated classes survive:
        // 4 of m=1 plus 12 of m=3 = 16, and no m=6 triple.
        assert_eq!(d.triples.len(), 16, "got {:?}", d.triples);
        assert!(
            !d.triples.iter().any(|&(i, j, k)| i != j && j != k),
            "an all-distinct triple survived a zero cutoff"
        );
    }

    /// The screening predicate must depend only on the unordered multiset
    /// `{i,j,k}`. If it could accept one ordering and reject another, the banded
    /// weight `m` would stop standing for `m` retained orderings and the `/3`
    /// divisor would silently become wrong.
    #[test]
    fn retention_is_permutation_invariant() {
        let c = two_clusters();
        let nocc = c.nrows();
        for &cut in &[0.0, 1.0, 3.0, 5.0, 19.0, 21.0, 25.0] {
            for i in 0..nocc {
                for j in 0..nocc {
                    for k in 0..nocc {
                        let want = triple_is_retained(&c, i, j, k, cut);
                        for p in [
                            (i, j, k),
                            (i, k, j),
                            (j, i, k),
                            (j, k, i),
                            (k, i, j),
                            (k, j, i),
                        ] {
                            assert_eq!(
                                triple_is_retained(&c, p.0, p.1, p.2, cut),
                                want,
                                "cutoff {cut}: retention of ({i},{j},{k}) is not \
                                 permutation invariant (differs at {p:?})"
                            );
                        }
                    }
                }
            }
        }
    }

    // --- Screening behaviour ------------------------------------------------

    /// A cutoff that separates the two clusters drops the cross-cluster
    /// all-distinct triples and nothing else.
    #[test]
    fn distant_triples_are_screened_out() {
        let c = two_clusters();
        // 5 Bohr keeps within-cluster separations (2 Bohr) and cuts across
        // (>= 18 Bohr). With 2 orbitals per cluster there is NO all-distinct
        // triple inside a single cluster, so every m=6 triple must go.
        let d = build_triple_domains(&c, 5.0).unwrap();

        assert!(!d.is_complete());
        assert!(d.triple_retention() < 1.0);
        assert!(!d.triples.contains(&(0, 1, 2)), "cross-cluster triple survived");
        assert!(!d.triples.contains(&(1, 2, 3)), "cross-cluster triple survived");
        assert_eq!(d.triples.len(), 16, "16 repeated-index triples, 0 distinct");

        // Weighted retention must be BELOW plain retention. Screening can only
        // remove the m=6 class, which carries the most weight per triple, so
        // dropping 4 of 20 band entries (0.8 retained) removes 24 of 64 weighted
        // orderings (0.625 retained). Quoting the plain count alone would
        // overstate how much physics survived.
        assert_eq!(d.triple_retention(), 16.0 / 20.0);
        assert_eq!(d.weighted_retention(), 40.0 / 64.0);
        assert!(
            d.weighted_retention() < d.triple_retention(),
            "weighted {} vs plain {}",
            d.weighted_retention(),
            d.triple_retention()
        );
    }

    /// Screening must actually change the energy — otherwise the cutoff is inert
    /// and any claim about it would be measuring nothing.
    #[test]
    fn screening_changes_the_energy() {
        let c = two_clusters();
        let dense = complete_triple_domains(&c).unwrap();
        let cut = build_triple_domains(&c, 5.0).unwrap();

        let e_dense = screened_triple_energy(&dense, |i, j, k| Ok(contrib(i, j, k))).unwrap();
        let e_cut = screened_triple_energy(&cut, |i, j, k| Ok(contrib(i, j, k))).unwrap();

        assert!(
            (e_dense - e_cut).abs() > 1e-12,
            "screening had no effect; the cutoff is inert ({e_dense} vs {e_cut})"
        );
        assert!(cut.triple_retention() < 1.0, "premise: this cutoff should drop triples");
    }

    /// The screened energy must equal the dense one restricted to the retained
    /// triples — i.e. screening only ever *omits* terms, it never perturbs the
    /// ones it keeps. Checked bit-for-bit against a hand-built restricted sum.
    #[test]
    fn screened_energy_is_the_dense_sum_over_retained_triples() {
        let c = two_clusters();
        let d = build_triple_domains(&c, 5.0).unwrap();
        let got = screened_triple_energy(&d, |i, j, k| Ok(contrib(i, j, k))).unwrap();

        let mut want = 0.0f64;
        for i in 0..c.nrows() {
            for j in i..c.nrows() {
                for k in j..c.nrows() {
                    if triple_is_retained(&c, i, j, k, 5.0) {
                        want += occ_triple_multiplicity(i, j, k) * contrib(i, j, k);
                    }
                }
            }
        }
        want /= 3.0;
        assert_eq!(got.to_bits(), want.to_bits());
    }

    /// A triple may only survive if all three of its constituent pairs would —
    /// so the triple screen can never be looser than the pair screen it sits
    /// above. (Repeated-index triples are exempt by construction, as are
    /// diagonal pairs in `dlpno_ccsd`.)
    #[test]
    fn triple_screen_is_consistent_with_the_pair_screen() {
        let c = two_clusters();
        let cut = 5.0;
        let d = build_triple_domains(&c, cut).unwrap();
        let pd = ferric_mp2::pair_domains::build_pair_domains(&c, cut, f64::INFINITY).unwrap();

        for &(i, j, k) in &d.triples {
            if i == j || j == k || i == k {
                continue; // exempt class
            }
            for (p, q) in [(i, j), (i, k), (j, k)] {
                let (lo, hi) = if p <= q { (p, q) } else { (q, p) };
                assert!(
                    pd.pairs.contains(&(lo, hi)),
                    "retained triple ({i},{j},{k}) rests on screened pair ({lo},{hi})"
                );
            }
        }
    }

    /// A zero-occupied system has no triples and must not divide by zero.
    #[test]
    fn empty_system_is_handled() {
        let c = Array2::<f64>::zeros((0, 3));
        let d = complete_triple_domains(&c).unwrap();
        assert!(d.triples.is_empty());
        assert!(d.is_complete());
        assert_eq!(d.triple_retention(), 1.0);
        assert_eq!(d.weighted_retention(), 1.0);
        assert_eq!(screened_triple_energy(&d, |_, _, _| Ok(1.0)).unwrap(), 0.0);
    }

    /// One occupied orbital gives exactly one triple `(0,0,0)` with weight 1 —
    /// the degenerate case `ccsd_t_closed_shell` explicitly supports (unlike its
    /// spin-orbital sibling's `no2 < 3` early return).
    #[test]
    fn single_occupied_orbital_gives_one_unit_weight_triple() {
        let d = complete_triple_domains(&line_centers(1, 1.0)).unwrap();
        assert_eq!(d.triples, vec![(0, 0, 0)]);
        assert_eq!(d.weights, vec![1.0]);
        assert_eq!(d.weighted_retention(), 1.0);
    }

    /// An error from the per-triple callback must propagate, not be swallowed
    /// into a plausible-looking energy.
    #[test]
    fn callback_errors_propagate() {
        let d = complete_triple_domains(&line_centers(3, 1.0)).unwrap();
        let r = screened_triple_energy(&d, |i, _, _| {
            if i == 1 {
                Err(FerricError::General("boom".into()))
            } else {
                Ok(1.0)
            }
        });
        assert!(r.is_err());
    }

    // --- Input validation ---------------------------------------------------

    /// Cutoffs must be validated, not silently accepted.
    #[test]
    fn negative_cutoff_is_an_error() {
        assert!(build_triple_domains(&two_clusters(), -1.0).is_err());
    }

    /// A malformed centers array is a caller bug and must error, not panic
    /// later.
    #[test]
    fn wrong_shape_centers_is_an_error() {
        let bad = array![[0.0, 0.0], [1.0, 1.0]];
        assert!(build_triple_domains(&bad, 1.0).is_err());
    }

    /// Inconsistent hand-built domains must error rather than mis-weight.
    #[test]
    fn mismatched_weights_are_rejected() {
        let mut d = complete_triple_domains(&line_centers(3, 1.0)).unwrap();
        d.weights.pop();
        assert!(screened_triple_energy(&d, |_, _, _| Ok(1.0)).is_err());
    }
}
