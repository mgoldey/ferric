//! Weekly validation half of `pbc_krks.rs` (every test
//! `#[ignore = "validation: ..."]`): the tests whose per-commit CI time
//! was >= 120 s (mean of runs 40957025137 and 40953725119), moved so the
//! fast tier stays balanced over 8 shards. The module docs, hypotheses and
//! bars are in `pbc_krks.rs`; the shared helpers are `pbc_krks_support/mod.rs`.
//!
//! Fast-tier canaries that keep these invariants per commit:
//! * `tpss_krks_and_gamma_match_pinned_pyscf` (k-point meta-GGA vs PySCF KRKS,
//!   1x1x1 and 1x1x2, same cell)
//! * `gamma_mgga_matches_pinned_pyscf_gamma_rks` (SCAN and r2SCAN pinned
//!   separately on the two-orbital cell: catches the swap)
//! * `mgga_k_mesh_is_the_supercell`, `mgga_one_point_mesh_is_gamma_rks`

mod common;
#[path = "pbc_krks_support/mod.rs"]
mod support;

use support::*;

#[test]
#[ignore = "validation: PBC k-point RKS SCAN/r2SCAN H2 vs PySCF KRKS pins incl. 2x2x2; CI ~185 s"]
fn mgga_uniform_grid_krks_matches_pinned_pyscf() {
    let cell = h2_cell(4.0);
    let bs = pyscf_sto3g_h();
    for &(functional, n, e_ref) in &PYSCF_MGGA {
        let mesh = KPointMesh::gamma_centred(&cell, n).unwrap();
        let r = krks_uniform(&cell, &bs, &mesh, functional, 61);
        eprintln!(
            "H2 {functional} {n:?}: {:.12} PySCF {e_ref:.12} dE {:.2e}",
            r.scf.energy,
            r.scf.energy - e_ref
        );
        assert!(
            (r.scf.energy - e_ref).abs() < 1e-9,
            "{functional} {n:?}: {} vs {e_ref}",
            r.scf.energy
        );
    }
}

// Tolerance 1e-9: measured worst miss 2.7e-12 (H2) / 5.8e-13 (H4) = >300x
// headroom for platform rounding; the other side is the SCAN - r2SCAN gap
// (2.2e-3, 2e6x the tolerance) and the 5e-4 H2 1x1x2 separation, so no
// functional mix-up or tau-term defect above 1e-9 can pass.

/// SCAN and r2SCAN pinned separately on a cell where PySCF's own SCAN - r2SCAN
/// gap is 2.2e-3 Ha (2e4 x the 1e-7 pin). Swapping the two functional
/// mappings moves each energy by that gap, so this fails under the swap
/// mutant, which the H2 1x1x1 pins cannot see.
#[test]
#[ignore = "validation: PBC k-point RKS SCAN vs r2SCAN on the H4 triclinic cell vs PySCF; CI ~144 s"]
fn mgga_scan_and_r2scan_are_distinguished_by_the_pins() {
    let cell = triclinic_cell();
    let bs = pyscf_sto3g_h();
    let mut e = std::collections::HashMap::new();
    for &(functional, n, e_ref) in &PYSCF_MGGA_TRI {
        let mesh = KPointMesh::gamma_centred(&cell, n).unwrap();
        let r = krks_uniform(&cell, &bs, &mesh, functional, 61);
        eprintln!(
            "H4 tri {functional} {n:?}: {:.12} PySCF {e_ref:.12} dE {:.2e}",
            r.scf.energy,
            r.scf.energy - e_ref
        );
        assert!(
            (r.scf.energy - e_ref).abs() < 1e-9,
            "{functional} {n:?}: {} vs {e_ref}",
            r.scf.energy
        );
        e.insert((functional, n), r.scf.energy);
    }
    for n in [[1, 1, 1], [1, 1, 2]] {
        let gap = e[&("SCAN", n)] - e[&("R2SCAN", n)];
        assert!(gap.abs() > 2e-3, "{n:?}: SCAN - r2SCAN = {gap:.3e}");
    }
}
