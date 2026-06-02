//! Active orbital-space partition shared across post-HF methods.
//!
//! Correlated methods (MP2, OO-MP2, RPA, GW, …) all index a *correlated*
//! subset of the molecular orbitals — the occupied/virtual orbitals that
//! survive any frozen-core truncation — and repeatedly need to map a
//! correlated index back into the full MO list. [`OrbitalSpace`] bundles the
//! four quantities that describe that partition so they travel together
//! instead of as four loose `usize` arguments.

/// The active occupied/virtual orbital partition for a correlated calculation.
///
/// `nocc`/`nvir` count the correlated occupied/virtual orbitals. `first_occ`
/// is the full-MO index of the first correlated occupied orbital (i.e. the
/// number of frozen-core orbitals), and `nocc_total` is the full-MO index of
/// the first virtual orbital (the total number of occupied orbitals,
/// including frozen core). A correlated occupied index `i` maps to full-MO
/// index `first_occ + i`; a correlated virtual index `a` maps to
/// `nocc_total + a`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OrbitalSpace {
    /// Number of correlated (non-frozen) occupied orbitals.
    pub nocc: usize,
    /// Number of correlated virtual orbitals.
    pub nvir: usize,
    /// Full-MO index of the first virtual orbital (total occupied count).
    pub nocc_total: usize,
    /// Full-MO index of the first correlated occupied orbital (frozen-core count).
    pub first_occ: usize,
}

impl OrbitalSpace {
    /// Construct an orbital-space partition.
    pub fn new(nocc: usize, nvir: usize, nocc_total: usize, first_occ: usize) -> Self {
        Self { nocc, nvir, nocc_total, first_occ }
    }

    /// Number of occupied-virtual pairs (`nocc * nvir`) — the dimension of the
    /// orbital-response / amplitude vectors.
    #[inline]
    pub fn nov(&self) -> usize {
        self.nocc * self.nvir
    }

    /// Full-MO index of correlated occupied orbital `i`.
    #[inline]
    pub fn occ_mo(&self, i: usize) -> usize {
        self.first_occ + i
    }

    /// Full-MO index of correlated virtual orbital `a`.
    #[inline]
    pub fn vir_mo(&self, a: usize) -> usize {
        self.nocc_total + a
    }
}
