//! Weekly validation (every test `#[ignore = "validation: ..."]`) half of
//! `pbc_lr_force_bitwise.rs`: the Gamma RS-GDF force's LR (G ≠ 0) derivative
//! pass and the full force are bit-identical across rayon thread counts
//! (FNV-1a hashes of the force and its LR fit parts vs the 1-thread run, in
//! process). Construction and cases: `pbc_lr_force_bitwise.rs`; harness:
//! `lr_force_bitwise_support/mod.rs`.
//!
//! WHY WEEKLY: the thread-invariance family (bitwise across threads) moved
//! out of per-commit CI; the cc-pVDZ case alone is ~66 s on a CI shard. The
//! per-commit canaries are `lr_force_sto3g_unsplit_is_thread_invariant`
//! (2/6 threads) and `lr_force_sto3g_split_two_threads_canary` (2 threads).

mod common;
#[path = "lr_force_bitwise_support/mod.rs"]
mod support;

use support::*;

#[test]
#[ignore = "validation: PBC Gamma RS-GDF LR force pass bitwise across threads (STO-3G split, 2/6 threads); CI ~16 s"]
fn lr_force_sto3g_split_is_thread_invariant() {
    check(1, &[2, 6]);
}

#[test]
#[ignore = "validation: PBC Gamma RS-GDF LR force pass bitwise across threads (cc-pVDZ split, rotated); CI ~66 s"]
fn lr_force_ccpvdz_split_rotated_is_thread_invariant() {
    check(2, &[6]);
}
