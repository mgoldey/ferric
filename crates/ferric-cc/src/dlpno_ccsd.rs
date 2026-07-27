//! DLPNO-CCSD — pair screening for closed-shell CCSD.
//!
//! # Scope, stated up front
//!
//! This is the **pair-screening half** of DLPNO-CCSD, not the full method. It
//! screens which occupied pairs `(i,j)` carry amplitudes, which attacks the same
//! n_o-order factor that [`ferric_mp2::pair_domains`] attacks for LinLCCD(hh).
//!
//! It does **not** yet rotate the t1/t2 amplitudes and all ~40 residual
//! contractions into a per-pair PNO virtual basis. That is the other half, and it
//! is a much larger change: `ccsd_closed_shell.rs`'s residual is a tightly coupled
//! set of contractions over shared `oooo/ovov/oovv/ovvo/ovoo/ovvv/vvvv` blocks, so
//! a per-pair virtual basis means re-deriving every one of them in a basis that
//! differs per pair. Doing that behind an untested "DLPNO-CCSD" label would be
//! worse than shipping the honest half — see the field notes in the module tests.
//!
//! # Why pair screening is the right first half
//!
//! The pair mask composes with everything downstream: once `t2[i,j,:,:]` is zero
//! for screened pairs, every contraction that consumes `t2` inherits the sparsity
//! for free, without touching the residual algebra. So this is both the cheaper
//! change and the prerequisite — PNOs are defined *per retained pair*, so the pair
//! list has to be right before per-pair virtual bases mean anything.
//!
//! # Exactness contract
//!
//! With complete domains, [`apply_pair_mask`] is the identity on `t2`, so a CCSD
//! run through it reproduces the unscreened one bit for bit.
//! `pair_mask_is_identity_when_complete` pins that.

use ferric_core::FerricError;
use ferric_mp2::pair_domains::PairDomains;
use ndarray::Array4;

/// Zero out the amplitude blocks of occupied pairs that `domains` screened away.
///
/// `t2` is the closed-shell spatial-orbital amplitude tensor `t2[i,j,a,b]` over
/// `(nocc, nocc, nvir, nvir)`. Pairs are symmetric — screening `(i,j)` screens
/// `(j,i)` — because `t2[i,j,a,b] == t2[j,i,b,a]` and keeping only one half would
/// break that symmetry rather than approximate it.
///
/// Returns the number of `(i,j)` index combinations zeroed, so a caller can report
/// what the screen actually removed rather than assume.
///
/// # Errors
///
/// [`FerricError::General`] when `t2`'s occupied dimensions disagree with
/// `domains.nocc`.
pub fn apply_pair_mask(
    t2: &mut Array4<f64>,
    domains: &PairDomains,
) -> Result<usize, FerricError> {
    let (ni, nj, _, _) = t2.dim();
    if ni != domains.nocc || nj != domains.nocc {
        return Err(FerricError::General(format!(
            "apply_pair_mask: t2 occupied dims ({ni}, {nj}) disagree with domains.nocc = {}",
            domains.nocc
        )));
    }

    // Retained-pair lookup, symmetric in (i, j).
    let nocc = domains.nocc;
    let mut keep = vec![false; nocc * nocc];
    for &(i, j) in &domains.pairs {
        keep[i * nocc + j] = true;
        keep[j * nocc + i] = true;
    }

    let mut zeroed = 0usize;
    for i in 0..nocc {
        for j in 0..nocc {
            if keep[i * nocc + j] {
                continue;
            }
            zeroed += 1;
            t2.slice_mut(ndarray::s![i, j, .., ..]).fill(0.0);
        }
    }
    Ok(zeroed)
}

/// Fraction of `(i,j)` index combinations a mask would retain, in `[0, 1]`.
///
/// Counts the full `nocc²` grid, not the `i <= j` triangle, because that is what
/// the `t2` tensor actually stores and therefore what the screening saves.
pub fn pair_mask_retention(domains: &PairDomains) -> f64 {
    let nocc = domains.nocc;
    if nocc == 0 {
        return 1.0;
    }
    let mut keep = vec![false; nocc * nocc];
    for &(i, j) in &domains.pairs {
        keep[i * nocc + j] = true;
        keep[j * nocc + i] = true;
    }
    keep.iter().filter(|&&k| k).count() as f64 / (nocc * nocc) as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_mp2::pair_domains::{build_pair_domains, complete_pair_domains};
    use ndarray::{array, Array2};

    fn line_centers(nocc: usize, spacing: f64) -> Array2<f64> {
        Array2::from_shape_fn((nocc, 3), |(i, ax)| if ax == 0 { i as f64 * spacing } else { 0.0 })
    }

    fn filled_t2(nocc: usize, nvir: usize) -> Array4<f64> {
        let mut s = 0x9E3779B97F4A7C15u64;
        Array4::from_shape_fn((nocc, nocc, nvir, nvir), |_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f64 / (1u64 << 31) as f64) - 1.0
        })
    }

    /// THE EXACTNESS CONTRACT: complete domains must leave `t2` untouched.
    ///
    /// Everything this module can claim rests on the screened path collapsing to
    /// the dense one when the screens are off, so this is checked bit-for-bit.
    #[test]
    fn pair_mask_is_identity_when_complete() {
        let (nocc, nvir) = (4, 3);
        let t2 = filled_t2(nocc, nvir);
        let mut masked = t2.clone();

        let d = complete_pair_domains(&line_centers(nocc, 1.0)).unwrap();
        let zeroed = apply_pair_mask(&mut masked, &d).unwrap();

        assert_eq!(zeroed, 0, "complete domains must zero nothing");
        assert_eq!(pair_mask_retention(&d), 1.0);
        for (a, b) in masked.iter().zip(t2.iter()) {
            assert_eq!(a, b, "complete pair mask must be the identity on t2");
        }
    }

    /// Screening must zero exactly the screened pairs and nothing else.
    #[test]
    fn mask_zeroes_only_screened_pairs() {
        let (nocc, nvir) = (4, 3);
        let t2 = filled_t2(nocc, nvir);
        let mut masked = t2.clone();

        // Two clusters 30 Bohr apart; a 5 Bohr cutoff separates them.
        let centers = array![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
            [31.0, 0.0, 0.0]
        ];
        let d = build_pair_domains(&centers, 5.0, f64::INFINITY).unwrap();
        apply_pair_mask(&mut masked, &d).unwrap();

        let nocc_ = d.nocc;
        let mut keep = vec![false; nocc_ * nocc_];
        for &(i, j) in &d.pairs {
            keep[i * nocc_ + j] = true;
            keep[j * nocc_ + i] = true;
        }
        for i in 0..nocc {
            for j in 0..nocc {
                let block = masked.slice(ndarray::s![i, j, .., ..]);
                if keep[i * nocc + j] {
                    let orig = t2.slice(ndarray::s![i, j, .., ..]);
                    assert_eq!(block, orig, "retained pair ({i},{j}) was modified");
                } else {
                    assert!(
                        block.iter().all(|&v| v == 0.0),
                        "screened pair ({i},{j}) was not zeroed"
                    );
                }
            }
        }
    }

    /// The mask must stay symmetric: `t2[i,j,a,b] == t2[j,i,b,a]` is a structural
    /// identity of closed-shell CCSD, so screening (i,j) without (j,i) would
    /// corrupt the amplitudes rather than approximate them.
    #[test]
    fn mask_is_symmetric_in_the_pair() {
        let (nocc, nvir) = (4, 3);
        let mut masked = filled_t2(nocc, nvir);
        let centers = array![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
            [31.0, 0.0, 0.0]
        ];
        let d = build_pair_domains(&centers, 5.0, f64::INFINITY).unwrap();
        apply_pair_mask(&mut masked, &d).unwrap();

        for i in 0..nocc {
            for j in 0..nocc {
                let ij_zero = masked.slice(ndarray::s![i, j, .., ..]).iter().all(|&v| v == 0.0);
                let ji_zero = masked.slice(ndarray::s![j, i, .., ..]).iter().all(|&v| v == 0.0);
                assert_eq!(ij_zero, ji_zero, "pair ({i},{j}) masked asymmetrically");
            }
        }
    }

    /// Diagonal pairs must survive any cutoff — they carry the bulk of the
    /// correlation energy, so zeroing them is an error, not an approximation.
    #[test]
    fn diagonal_amplitudes_always_survive() {
        let (nocc, nvir) = (4, 3);
        let t2 = filled_t2(nocc, nvir);
        let mut masked = t2.clone();
        let d = build_pair_domains(&line_centers(nocc, 10.0), 0.0, 0.0).unwrap();
        apply_pair_mask(&mut masked, &d).unwrap();

        for i in 0..nocc {
            let orig = t2.slice(ndarray::s![i, i, .., ..]);
            let got = masked.slice(ndarray::s![i, i, .., ..]);
            assert_eq!(got, orig, "diagonal pair ({i},{i}) was screened away");
        }
    }

    /// Dimension mismatches are caller bugs and must error rather than corrupt.
    #[test]
    fn mismatched_dims_are_rejected() {
        let mut t2 = filled_t2(3, 2);
        let d = complete_pair_domains(&line_centers(4, 1.0)).unwrap();
        assert!(apply_pair_mask(&mut t2, &d).is_err());
    }
}
