//! Weekly validation (every test `#[ignore = "validation: ..."]`) half of
//! `pbc_kpair_symmetry.rs`: the k-point RS-GDF SR residue bins' s2 walk on
//! the full triclinic 4 H fixture. Relations, items 1-7, the artifact
//! hypothesis and the mutants: `pbc_kpair_symmetry.rs`'s module doc;
//! harness: `kpair_symmetry_support/mod.rs`.
//!
//! WHY WEEKLY: ~6.3 and ~3.8 min on a CI shard (mean of runs 40957025137 and
//! 40953725119), above the 120 s per-commit threshold. Per-commit canaries
//! (same bodies on the 2 H twin of the fixture): `krsgdf_s2_bins_canary`,
//! `krsgdf_s2_bin_mutants_canary`.

mod common;
#[path = "kpair_symmetry_support/mod.rs"]
mod support;

use ferric_pbc::kpts::KPointMesh;
use support::*;

#[test]
#[ignore = "validation: PBC k-point RS-GDF SR bins s2 walk vs the frozen ordered walk (4 H, three meshes / mutants); CI ~378 s"]
fn krsgdf_s2_bins_are_the_ordered_walk_under_the_minus_rl_map() {
    let (cell, prep, aux) = tri_sp();
    for (tag, mesh) in [
        ("Gamma 1x1x2 (TRIM)", KPointMesh::gamma_centred(&cell, TRIM)),
        ("Gamma 1x1x3", KPointMesh::gamma_centred(&cell, NON_TRIM)),
        // Pair-image moduli 2N = [1, 4, 4] differ from the aux-image N.
        ("MP 1x2x2", KPointMesh::monkhorst_pack(&cell, [1, 2, 2])),
    ] {
        let mesh = mesh.unwrap();
        check_kbins(&cell, &prep, &aux, &mesh, None, tag, &THREADS);
    }
}

/// Every bins mutant leaves `{x, y}` by far more than round-off on the
/// non-TRIM mesh; on the TRIM mesh `WrongPairBin` is bitwise the production
/// walk and every other mutant is still caught.
#[test]
#[ignore = "validation: PBC k-point RS-GDF SR bins s2 walk vs the frozen ordered walk (4 H, three meshes / mutants); CI ~227 s"]
fn krsgdf_s2_bin_mutants_fail_and_wrong_pair_bin_needs_a_non_trim_mesh() {
    let (cell, prep, aux) = tri_sp();
    kbins_mutants_case(&cell, &prep, &aux, &[(NON_TRIM, false), (TRIM, true)]);
}
