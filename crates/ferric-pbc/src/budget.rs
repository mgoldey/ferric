//! Memory gating for the ferric-pbc builders (Stage 1 step 10,
//! `reference/pbc/stage1-design.md` §5).
//!
//! The budget is ferric's unified one: [`ferric_core::memory::resolve_budget_bytes`]
//! (explicit config value > `FERRIC_MEM_BUDGET_GB` > legacy env vars > 0.8 ×
//! available RAM > 2 GiB). Every large buffer is RESERVED on a `Ledger`
//! before it is allocated; the reservation fails with
//! [`ferric_core::memory::check_alloc`]'s error, whose label names the
//! quantity and carries the exact byte counts (the GB figures in
//! `check_alloc`'s own text round tiny test budgets to 0.00).
//!
//! The ledger composes: each reservation is checked against what is LEFT of
//! the budget after the earlier, still-live reservations — not against the
//! whole budget (the "gates do not compose" defect of
//! `ferric_core::memory::plan`'s module doc). It does not read RSS, so a gate
//! decision is a pure function of the shapes and the budget (deterministic
//! in tests).

use ferric_core::memory::{check_alloc, resolve_budget_bytes};
use ferric_core::FerricError;

/// Resolve a ferric-pbc budget: `explicit` (a config field) or ferric's
/// unified default chain ([`resolve_budget_bytes`]).
pub fn resolve(explicit: Option<usize>) -> usize {
    resolve_budget_bytes(explicit)
}

/// `count × size` as bytes, saturating (a saturated value always fails a
/// gate instead of wrapping to a small number).
pub(crate) fn bytes_of(count: u64, size: usize) -> usize {
    usize::try_from(count)
        .unwrap_or(usize::MAX)
        .saturating_mul(size)
}

/// Running account of the live reservations against one budget.
#[derive(Debug, Clone)]
pub(crate) struct Ledger {
    budget: usize,
    resident: usize,
}

impl Ledger {
    pub(crate) fn new(budget: usize) -> Self {
        Self {
            budget,
            resident: 0,
        }
    }

    pub(crate) fn budget(&self) -> usize {
        self.budget
    }

    /// Bytes reserved so far.
    pub(crate) fn resident(&self) -> usize {
        self.resident
    }

    /// Bytes still available.
    pub(crate) fn remaining(&self) -> usize {
        self.budget.saturating_sub(self.resident)
    }

    /// Check `bytes` of `quantity` against the remaining budget WITHOUT
    /// reserving it (for per-chunk buffers that are freed each iteration).
    pub(crate) fn check(&self, quantity: &str, bytes: usize) -> Result<(), FerricError> {
        let left = self.remaining();
        check_alloc(
            &format!(
                "ferric-pbc {quantity} ({bytes} bytes; {left} of the {} byte budget left)",
                self.budget
            ),
            bytes,
            left,
        )
    }

    /// [`Ledger::check`], then count `bytes` as resident.
    pub(crate) fn reserve(&mut self, quantity: &str, bytes: usize) -> Result<(), FerricError> {
        self.check(quantity, bytes)?;
        self.resident = self.resident.saturating_add(bytes);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ledger_composes_and_names_the_numbers() {
        let mut l = Ledger::new(1000);
        l.reserve("first", 600).unwrap();
        // 600 + 500 > 1000 although 500 alone fits the whole budget.
        let msg = l.reserve("second buffer", 500).unwrap_err().to_string();
        assert!(
            msg.contains("second buffer (500 bytes; 400 of the 1000 byte budget left)"),
            "{msg}"
        );
        assert_eq!(l.resident(), 600);
        l.reserve("third", 400).unwrap();
        assert_eq!(l.remaining(), 0);
        assert_eq!(bytes_of(u64::MAX, 24), usize::MAX);
    }
}
