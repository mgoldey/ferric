//! MWE: the `b_vv` block the guard was written for must actually be gated.
//!
//! # The defect: a guard modelling an allocation it never sees
//!
//! `check_mo_side_alloc` takes an `include_b_vv` flag and charges
//! `naux·nvir²·8` when it is set:
//!
//! ```text
//! let b_vv = if include_b_vv {
//!     naux.saturating_mul(nvir).saturating_mul(nvir).saturating_mul(8)
//! } else { 0 };
//! ```
//!
//! `grep -rn check_mo_side_alloc crates/` finds exactly ONE call site in the
//! whole workspace, and it passes `false`:
//!
//! ```text
//! rimp2.rs:866: check_mo_side_alloc("RI-MP2", dfbs.nbasis(), nocc, nvir, false, budget_bytes)?;
//! ```
//!
//! So the arm is dead. Meanwhile `compute_mp2_intermediates_impl` builds the
//! block on a different path with no gate of any kind — its own comment calls
//! it "the 13 GB hog":
//!
//! ```text
//! // ... optionally B^P_{ij} and B^P_{ab} (CPKS only — the gradient
//! // pipeline never reads them, and b_vv is the 13 GB hog).
//! let (b_oo, b_vv) = if with_oo_vv {
//!     (Some(eri3_mo_block_dressed(.., &c_occ, &c_occ)?),
//!      Some(eri3_mo_block_dressed(.., &c_vir, &c_vir)?))
//! } else { (None, None) };
//! ```
//!
//! # Size
//!
//! `b_vv` is `nvir/nocc` times `b_ov`, so it dominates wherever the virtual
//! space is large:
//!
//! ```text
//!   system                  naux  nocc  nvir      b_ov      b_vv
//!   benzene/aug-cc-pVTZ     1512    21   393   0.100 GB  1.868 GB
//!   danuglipron/def2-SVP    2800    90   610   1.230 GB  8.335 GB
//! ```
//!
//! # Scope, and what these contracts do NOT cover
//!
//! They pin the guard's ARITHMETIC only. Mutation-checking makes the split
//! explicit:
//!
//! ```text
//!   drop the b_vv term from mo_side_alloc_bytes   2 of 4 contracts FAIL
//!   revert the call site to pass `false`          ALL 4 PASS
//! ```
//!
//! The second row is the honest limitation: reverting
//! `compute_mp2_intermediates_impl`'s call from `with_oo_vv` to `false` — the
//! exact defect this fixes — leaves this file green, because arithmetic tests
//! cannot see which argument a caller passes.
//!
//! Catching that needs an end-to-end run whose `b_vv` is large enough to cross
//! a budget, i.e. a drug-scale MP2; at any fixture this suite can afford, the
//! block fits either way and the gate is inert. So the call site is verified
//! by reading it, not by test, and that is stated here rather than left for
//! someone to assume.

use ferric_mp2::rimp2::mo_side_alloc_bytes;

const NAUX: usize = 1512;
const NOCC: usize = 21;
const NVIR: usize = 393;
const F64: usize = 8;

/// CONTRACT 1: `include_b_vv` actually changes the charge.
///
/// If the flag were inert the guard would be decorative. It is not inert —
/// the defect was that nothing ever passed `true`.
#[test]
fn the_b_vv_flag_changes_the_charge() {
    let without = mo_side_alloc_bytes(NAUX, NOCC, NVIR, false);
    let with = mo_side_alloc_bytes(NAUX, NOCC, NVIR, true);
    assert!(
        with > without,
        "include_b_vv must add the naux·nvir² block: {without} vs {with}"
    );
}

/// CONTRACT 2: the added amount is exactly `naux·nvir²·8`.
///
/// Pin the arithmetic, not just the direction — a charge that added a token
/// amount would satisfy CONTRACT 1 while still admitting an 8 GB allocation.
#[test]
fn the_b_vv_charge_is_the_full_block() {
    let delta = mo_side_alloc_bytes(NAUX, NOCC, NVIR, true)
        - mo_side_alloc_bytes(NAUX, NOCC, NVIR, false);
    assert_eq!(
        delta,
        NAUX * NVIR * NVIR * F64,
        "the b_vv charge must be the whole (naux, nvir, nvir) block"
    );
}

/// CONTRACT 3: `b_vv` dominates `b_ov` at realistic shapes.
///
/// The reachability guard. If `b_vv` were comparable to the blocks already
/// charged, leaving it out would be a rounding error rather than a defect —
/// and this whole fix would be noise. At `nvir >> nocc`, which is every
/// production MP2 shape, it is the largest MO-side block by far.
#[test]
fn b_vv_dominates_the_occupied_block() {
    let b_ov = NAUX * NOCC * NVIR * F64;
    let b_vv = NAUX * NVIR * NVIR * F64;
    assert!(
        b_vv > 10 * b_ov,
        "at nvir={NVIR} / nocc={NOCC}, b_vv ({b_vv}) should dwarf b_ov ({b_ov}) — if it did \
         not, gating it would not matter"
    );
}

/// CONTRACT 4: the charge grows quadratically in `nvir`.
///
/// `b_vv` is `naux·nvir²`. A linear charge would track the flag correctly at
/// one shape and under-count badly at another, which a single-point test
/// cannot distinguish.
#[test]
fn the_b_vv_charge_is_quadratic_in_nvir() {
    let base = mo_side_alloc_bytes(NAUX, NOCC, 100, true)
        - mo_side_alloc_bytes(NAUX, NOCC, 100, false);
    let doubled = mo_side_alloc_bytes(NAUX, NOCC, 200, true)
        - mo_side_alloc_bytes(NAUX, NOCC, 200, false);
    assert_eq!(
        doubled,
        4 * base,
        "doubling nvir must quadruple the b_vv charge (naux·nvir²)"
    );
}
