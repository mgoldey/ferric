//! Weekly half of `pbc_krsh_rsgdf.rs` (range-separated exchange on the
//! k-point RS-GDF J/K): the two anchors whose per-commit CI time was >= 120 s
//! (runs 40957025137 / 40953725119, mean 157 / 172 s), moved here so the
//! per-commit shards stay short. Every test is
//! `#[ignore = "validation: ..."]`. Hypotheses are in the doc of
//! `pbc_krsh_rsgdf.rs`; the helpers are `krsh_rsgdf_support/mod.rs`.
//! Per-commit canaries that remain:
//! `rsgdf_rsh_is_the_dense_rsh_to_the_fit_error_canary`,
//! `rsgdf_rsh_matches_one_pinned_pyscf_energy`, and the kernel-level and
//! supercell anchors with their mutants.

mod common;
#[path = "krsh_rsgdf_support/mod.rs"]
mod support;

use support::*;

/// SCF energies, HSE06, CAM-B3LYP and wB97X, RS-GDF vs dense on the same
/// grid, 1x1x1 and 1x1x3, cc-pVDZ-RI and def2-universal-JKFIT: the difference
/// must be of the order of the plain-Coulomb (PBE0) RS-GDF-vs-dense
/// difference. Per-commit canary: `rsgdf_rsh_is_the_dense_rsh_to_the_fit_error_canary`.
#[test]
#[ignore = "validation: k-RKS RSH on RS-GDF vs dense to the plain-Coulomb fit error, 2 meshes x 2 aux x 3 functionals; CI ~157 s"]
fn rsgdf_rsh_is_the_dense_rsh_to_the_fit_error() {
    fit_error_case(
        &[[1, 1, 1], [1, 1, 3]],
        &["cc-pvdz-ri", "def2-universal-jkfit"],
        &["HSE06", "HYB_GGA_XC_CAM_B3LYP", "HYB_GGA_XC_WB97X"],
    );
}

/// RS-GDF with the bundled cc-pVDZ-RI aux is not PySCF's AFT: E - pin is the
/// fit error of that aux, whose scale is the plain-Coulomb (PBE0) RS-GDF vs
/// dense gap at the same mesh (the dense energy equals the pin to 1e-9). A
/// defect moves E by >= 1.7e-2 against scales <= 3e-4. All six pins;
/// per-commit canary: `rsgdf_rsh_matches_one_pinned_pyscf_energy`.
#[test]
#[ignore = "validation: k-RKS HSE06 / CAM-B3LYP / wB97X on RS-GDF vs six PySCF 2.13 KRKS pins to the fit error, 61^3; CI ~172 s"]
fn rsgdf_rsh_matches_pinned_pyscf_to_the_fit_error() {
    pyscf_pin_case(&PYSCF);
}
