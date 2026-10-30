//! Weekly validation (every test `#[ignore = "validation: ..."]`) half of
//! `pbc_sr_rotation.rs`, section (h): the FORCES AND STRESS of the Gamma SR
//! column rotation. Construction, bars (`D_BAR`, `D_MUTANT_FLOOR`,
//! `FD_S_BAR`) and their derivations: `pbc_sr_rotation.rs`'s module doc;
//! fixtures and bodies: `sr_rotation_support/mod.rs`.
//!
//! WHY WEEKLY: each anchor costs 2-2.7 min on a CI shard (mean of runs
//! 40957025137 and 40953725119), above the 120 s per-commit threshold. The
//! per-commit canaries (same bodies, smaller case sets) are listed in
//! `pbc_sr_rotation.rs`'s module doc.

mod common;
#[path = "sr_rotation_support/mod.rs"]
mod support;

use common::*;
use ferric_pbc::rsgdf::RangeSplit;
use support::*;

/// EXACTNESS ANCHOR (forces): with the rotation requested, the analytic
/// force equals the unrotated force. Two measures: the WALK anchor (one
/// fixed density through both builds: only the derivative walks differ) and
/// the END-TO-END anchor (each build's own SCF). Mutants of the forward
/// transform must miss the walk anchor.
#[test]
#[ignore = "validation: PBC Gamma SR column rotation forces and stress (h); CI ~149 s"]
fn rotated_forces_equal_the_unrotated_forces_and_mutants_miss() {
    rotated_forces_case(force_fixtures());
}

/// EXACTNESS ANCHOR (stress), same structure as the forces'.
#[test]
#[ignore = "validation: PBC Gamma SR column rotation forces and stress (h); CI ~139 s"]
fn rotated_stress_equals_the_unrotated_stress_and_matches_fd() {
    let all: Vec<(usize, usize)> = (0..3).flat_map(|i| (0..3).map(move |j| (i, j))).collect();
    rotated_stress_case(&[None, Some(RangeSplit::default())], &all);
}

/// TRIVIAL LIMIT: a basis with nothing to rotate. Force and stress with the
/// rotation requested (and every forward-transform mutant, which cannot act)
/// are BITWISE the unrotated ones.
///
/// The unmutated check runs on the triclinic cell (non-orthogonal images);
/// the mutants cannot act on a basis with nothing to rotate, so they are
/// checked on the cheap cubic H2 cell (the full triclinic mutant sweep took
/// ~4.5 min and timed out the CI shard).
#[test]
#[ignore = "validation: PBC Gamma SR column rotation forces and stress (h); CI ~162 s"]
fn identity_rotation_forces_and_stress_are_bitwise_unrotated() {
    identity_rotation_forces_and_stress_case(vec![
        (triclinic_cell(), false, vec![None]),
        (
            h2_off_axis(&H2_OFF_AXIS),
            true,
            vec![None, Some(RangeSplit::default())],
        ),
    ]);
}

/// The rotated force and stress are BITWISE identical across thread counts
/// (the ordered-parallel derivative walks and the per-element `Tᵀ Y T`
/// read are deterministic), and the frozen serial hcore oracle
/// (`SerialDerivWalks`) agrees bitwise with the ordered walk on the rotated
/// shells.
#[test]
#[ignore = "validation: PBC Gamma SR column rotation forces and stress (h); CI ~122 s"]
fn rotated_forces_and_stress_are_bitwise_across_thread_counts() {
    rotated_forces_and_stress_threads_case(&[None, Some(RangeSplit::default())], &[2, 6]);
}
