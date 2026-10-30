//! Weekly validation (every test `#[ignore = "validation: ..."]`) half of
//! `pbc_parallel_bitwise.rs`: the pair-parallel and ordered-parallel walks
//! are BIT-IDENTICAL across 1/2/6 rayon threads and to the frozen serial
//! loops, on the full fixtures. Construction under test, what each anchor
//! guards and the mutants: `pbc_parallel_bitwise.rs`'s module doc (and the
//! section comments in `parallel_bitwise_support/mod.rs`, which holds every
//! body).
//!
//! WHY WEEKLY. These anchors cost 15 s to 4.5 min each on a CI shard (mean
//! of runs 40957025137 and 40953725119, given per test); the bitwise-across-
//! threads family as a whole was the largest per-commit binary (~1,780 s).
//! Each family keeps a shrunk per-commit canary in `pbc_parallel_bitwise.rs`
//! (1/2 threads, smaller fixtures, the same body).

mod common;
#[path = "parallel_bitwise_support/mod.rs"]
mod support;

use support::*;

#[test]
#[ignore = "validation: PBC parallel walks bitwise across threads and vs the frozen serial loops; CI ~143 s"]
fn rsgdf_sr_sums_are_bitwise_across_threads_and_vs_serial() {
    rsgdf_sr_sums_are_bitwise_across_threads_and_vs_serial_body();
}

#[test]
#[ignore = "validation: PBC parallel walks bitwise across threads and vs the frozen serial loops; CI ~136 s"]
fn hcore_sr_attraction_is_bitwise_across_threads_and_vs_serial() {
    hcore_sr_attraction_is_bitwise_across_threads_and_vs_serial_body();
}

#[test]
#[ignore = "validation: PBC parallel walks bitwise across threads and vs the frozen serial loops; CI ~141 s"]
fn krsgdf_sr_bins_are_bitwise_across_threads_and_vs_serial() {
    krsgdf_sr_bins_are_bitwise_across_threads_and_vs_serial_body();
}

#[test]
#[ignore = "validation: PBC parallel walks bitwise across threads and vs the frozen serial loops; CI ~31 s"]
fn hcore_kpts_sr_attraction_is_bitwise_across_threads_and_vs_serial() {
    hcore_kpts_sr_attraction_is_bitwise_across_threads_and_vs_serial_body();
}

#[test]
#[ignore = "validation: PBC parallel walks bitwise across threads and vs the frozen serial loops; CI ~195 s"]
fn krsgdf_blocks_and_krhf_energy_are_bitwise_across_threads() {
    krsgdf_blocks_and_krhf_energy_are_bitwise_across_threads_body();
}

#[test]
#[ignore = "validation: PBC parallel walks bitwise across threads and vs the frozen serial loops; CI ~254 s"]
fn gamma_rhf_rsgdf_force_is_bitwise_across_threads_and_vs_serial_walks() {
    gamma_rhf_rsgdf_force_is_bitwise_across_threads_and_vs_serial_walks_body();
}

#[test]
#[ignore = "validation: PBC parallel walks bitwise across threads and vs the frozen serial loops; CI ~75 s"]
fn gamma_uks_rsgdf_force_is_bitwise_across_threads_and_vs_serial_walks() {
    gamma_uks_rsgdf_force_is_bitwise_across_threads_and_vs_serial_walks_body();
}

#[test]
#[ignore = "validation: PBC parallel walks bitwise across threads and vs the frozen serial loops; CI ~261 s"]
fn gamma_rhf_rsgdf_stress_is_bitwise_across_threads_and_vs_serial_walks() {
    gamma_rhf_rsgdf_stress_is_bitwise_across_threads_and_vs_serial_walks_body();
}

#[test]
#[ignore = "validation: PBC parallel walks bitwise across threads and vs the frozen serial loops; CI ~69 s"]
fn kpoint_rhf_rsgdf_force_is_bitwise_across_threads_and_vs_serial_walks() {
    kpoint_rhf_rsgdf_force_is_bitwise_across_threads_and_vs_serial_walks_body();
}

#[test]
#[ignore = "validation: PBC parallel walks bitwise across threads and vs the frozen serial loops; CI ~33 s"]
fn kpoint_rhf_rsgdf_stress_is_bitwise_across_threads() {
    kpoint_rhf_rsgdf_stress_is_bitwise_across_threads_body();
}

#[test]
#[ignore = "validation: PBC parallel walks bitwise across threads and vs the frozen serial loops; CI ~21 s"]
fn kpoint_rhf_split_rsgdf_stress_is_bitwise_across_threads() {
    kpoint_rhf_split_rsgdf_stress_is_bitwise_across_threads_body();
}

#[test]
#[ignore = "validation: PBC parallel walks bitwise across threads and vs the frozen serial loops; CI ~27 s"]
fn ecp_force_term_is_bitwise_across_threads_and_vs_serial_loop() {
    ecp_force_term_is_bitwise_across_threads_and_vs_serial_loop_body();
}

#[test]
#[ignore = "validation: PBC parallel walks bitwise across threads and vs the frozen serial loops; CI ~18 s"]
fn rsgdf_lr_sums_are_invariant_to_the_g_chunking() {
    rsgdf_lr_sums_are_invariant_to_the_g_chunking_body();
}

#[test]
#[ignore = "validation: PBC parallel walks bitwise across threads and vs the frozen serial loops; CI ~42 s"]
fn rsgdf_lr_sums_match_frozen_serial_across_threads() {
    rsgdf_lr_sums_match_frozen_serial_across_threads_body();
}

#[test]
#[ignore = "validation: PBC parallel walks bitwise across threads and vs the frozen serial loops; CI ~85 s"]
fn rsgdf_b_and_energy_are_bitwise_across_threads_with_several_lr_chunks() {
    rsgdf_b_and_energy_are_bitwise_across_threads_with_several_lr_chunks_body();
}

#[test]
#[ignore = "validation: PBC parallel walks bitwise across threads and vs the frozen serial loops; CI ~179 s"]
fn overlap_kinetic_force_end_to_end_is_bitwise_across_threads_and_vs_serial_loop() {
    overlap_kinetic_force_end_to_end_is_bitwise_across_threads_and_vs_serial_loop_body();
}
