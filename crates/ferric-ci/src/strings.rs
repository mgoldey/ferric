//! Determinant-string enumeration for full CI within an active space.
//!
//! A *string* is an occupation pattern of `n_elec` electrons in `n_orb`
//! spatial orbitals, one per spin channel. We store each string as a `u64`
//! bitmask: bit `p` set means active orbital `p` is occupied, so an active
//! space holds at most
//! [`MAX_ACTIVE_ORBITALS`](crate::strings::MAX_ACTIVE_ORBITALS) orbitals (in
//! practice <= ~14).
//!
//! The full FCI determinant list factorizes as the outer product of the
//! α-string list and the β-string list (standard Knowles–Handy / Olsen
//! factorization): a determinant is `(alpha_string, beta_string)` and the CI
//! coefficient array has shape `(n_alpha_strings, n_beta_strings)`.

/// Largest active space the `u64` string representation supports. The CAS-CI
/// and CASSCF entry points reject larger ones before any integral work.
pub const MAX_ACTIVE_ORBITALS: usize = 63;

/// All `u64` bitmasks with exactly `k` of the low `n` bits set, in ascending
/// numeric order. Ascending numeric order is a stable, deterministic
/// enumeration; the actual order does not matter for FCI correctness as long as
/// it is used consistently to index the coefficient array.
///
/// Returns an empty vector if `k > n` (no such strings). `n == 0, k == 0`
/// yields the single empty string `0`.
pub fn enumerate_strings(n_orb: usize, k: usize) -> Vec<u64> {
    let mut out = Vec::new();
    if k > n_orb {
        return out;
    }
    if k == 0 {
        out.push(0u64);
        return out;
    }
    // Gosper's hack: iterate bit patterns with exactly k bits set.
    let mut x: u64 = (1u64 << k) - 1;
    let limit: u64 = if n_orb >= 64 { u64::MAX } else { 1u64 << n_orb };
    while x < limit {
        out.push(x);
        // Next bit permutation with the same popcount.
        //
        // clippy >= 1.98 suggests `x.isolate_lowest_one()` here. Do NOT take it:
        // that method is still UNSTABLE on 1.95 (feature
        // `isolate_most_least_significant_one`), and this crate advertises
        // Rust 1.75+ in README.md/CLAUDE.md, so using it would break every
        // toolchain older than its stabilization. `x & x.wrapping_neg()` is the
        // portable idiom and compiles identically.
        // `unknown_lints` first: 1.95 does not KNOW manual_isolate_lowest_one
        // and errors on the allow itself under -D warnings, so the pair is
        // required for the file to lint clean on both old and new toolchains.
        #[allow(unknown_lints)]
        #[allow(clippy::manual_isolate_lowest_one)]
        let c = x & x.wrapping_neg();
        let r = x + c;
        if c == 0 {
            break;
        }
        x = (((x ^ r) >> 2) / c) | r;
    }
    out
}

/// Number of strings [`enumerate_strings`] returns for `(n_orb, k)`, the
/// binomial `C(n_orb, k)`, computed without enumerating; `None` when it does
/// not fit in `usize`. Zero when `k > n_orb`.
pub fn count_strings(n_orb: usize, k: usize) -> Option<usize> {
    if k > n_orb {
        return Some(0);
    }
    let k = k.min(n_orb - k);
    // C(n, i+1) = C(n, i) (n - i) / (i + 1), exact at every step. The
    // intermediate C(n, i) (n - i) <= C(n, i+1) k, so u128 holds it for any
    // result that fits in u64.
    let mut c: u128 = 1;
    for i in 0..k {
        c = c.checked_mul((n_orb - i) as u128)? / (i as u128 + 1);
    }
    usize::try_from(c).ok()
}

/// Population count: number of set bits (occupied orbitals) in a string.
#[inline]
pub fn popcount(s: u64) -> u32 {
    s.count_ones()
}

/// The list of occupied orbital indices in a string, ascending.
pub fn occupied_orbitals(s: u64) -> Vec<usize> {
    let mut v = Vec::with_capacity(s.count_ones() as usize);
    let mut m = s;
    while m != 0 {
        let p = m.trailing_zeros() as usize;
        v.push(p);
        m &= m - 1;
    }
    v
}

/// Phase (+1 / -1) from the Jordan–Wigner / second-quantization sign rule for
/// annihilating (or creating) an electron in orbital `p` of string `s`: it is
/// `(-1)^(number of occupied orbitals below p)`. Orbital `p` is assumed to be
/// occupied (annihilation) or empty (creation) as appropriate; the caller
/// guarantees that.
#[inline]
pub fn phase_below(s: u64, p: usize) -> f64 {
    // Count set bits strictly below position p.
    let mask = (1u64 << p) - 1;
    if (s & mask).count_ones().is_multiple_of(2) {
        1.0
    } else {
        -1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn count_matches_binomial() {
        // C(4,2) = 6, C(6,3) = 20, C(7,5) = 21.
        assert_eq!(enumerate_strings(4, 2).len(), 6);
        assert_eq!(enumerate_strings(6, 3).len(), 20);
        assert_eq!(enumerate_strings(7, 5).len(), 21);
    }

    #[test]
    fn count_strings_matches_enumeration() {
        for n in 0..=12 {
            for k in 0..=n + 1 {
                assert_eq!(
                    count_strings(n, k),
                    Some(enumerate_strings(n, k).len()),
                    "C({n},{k})"
                );
            }
        }
        // C(30,15) and C(63,31) by the closed form; C(200,100) overflows.
        assert_eq!(count_strings(30, 15), Some(155_117_520));
        assert_eq!(count_strings(63, 31), Some(916_312_070_471_295_267));
        assert_eq!(count_strings(200, 100), None);
    }

    #[test]
    fn edge_cases() {
        assert_eq!(enumerate_strings(0, 0), vec![0u64]);
        assert!(enumerate_strings(3, 5).is_empty());
        // Exactly-full: C(4,4) = 1, all four low bits set.
        assert_eq!(enumerate_strings(4, 4), vec![0b1111u64]);
    }

    #[test]
    fn strings_have_correct_popcount() {
        for s in enumerate_strings(6, 3) {
            assert_eq!(popcount(s), 3);
        }
    }

    #[test]
    fn ascending_and_unique() {
        let v = enumerate_strings(5, 2);
        for w in v.windows(2) {
            assert!(w[0] < w[1], "not strictly ascending");
        }
    }

    #[test]
    fn occupied_orbitals_roundtrip() {
        let s = 0b101101u64;
        assert_eq!(occupied_orbitals(s), vec![0, 2, 3, 5]);
    }

    #[test]
    fn phase_below_rule() {
        // s = orbitals {0,2} occupied. phase below orbital 2 = (-1)^1 = -1
        // (one occupied orbital, #0, sits below 2). Below orbital 0 = +1.
        let s = 0b101u64;
        assert_eq!(phase_below(s, 0), 1.0);
        assert_eq!(phase_below(s, 2), -1.0);
    }
}
