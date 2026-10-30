//! Weekly validation half of `pbc_kuks.rs` (every test
//! `#[ignore = "validation: ..."]`): the tests whose per-commit CI time
//! was >= 120 s (mean of runs 40957025137 and 40953725119), moved so the
//! fast tier stays balanced over 8 shards. The module docs, hypotheses and
//! bars are in `pbc_kuks.rs`; the shared helpers are `pbc_kuks_support/mod.rs`.
//!
//! Fast-tier canaries that keep these invariants per commit:
//! * `uniform_grid_kuks_matches_a_pinned_pyscf_subset` (two of the 15 pins: Hd
//!   PBE0 1x1x3, H2t PBE 1x1x1)

mod common;
#[path = "pbc_kuks_support/mod.rs"]
mod support;

use support::*;

// ------------------------------------------------------------- PySCF pins

#[test]
#[ignore = "validation: PBC k-point UKS LDA/PBE/PBE0 vs all 15 PySCF KUKS pins; CI ~152 s"]
fn uniform_grid_kuks_matches_pinned_pyscf() {
    assert_kuks_pins(PYSCF);
}
