//! Weekly half of `pbc_kxcstress_mgga.rs` (closed-shell k-point meta-GGA
//! stress): the anchors whose per-commit CI time was >= 120 s (runs
//! 40957025137 / 40953725119, mean 125-201 s each), moved here so the
//! per-commit shards stay short. Every test is `#[ignore = "validation: ..."]`.
//! Hypotheses, bars and the measured numbers are in the doc of
//! `pbc_kxcstress_mgga.rs` and at each test; the helpers are
//! `kxcstress_mgga_support/mod.rs`. Per-commit canaries that remain:
//! `tpss_components_xc_stress_matches_fd_of_exc`,
//! `uniform_grid_fixed_density_xc_stress_canary`,
//! `uniform_grid_fixed_density_xc_stress_r2scan_canary`, `mutants_are_caught`,
//! `meta_gga_stress_matches_fd_of_the_scf_energy`,
//! `tpss_stress_matches_fd_of_the_scf_energy`.

mod common;
#[path = "kxcstress_mgga_support/mod.rs"]
mod support;

use support::*;

/// The XC part (`xc_ao + xc_weight`) against central FD of `E_xc[D]` at fixed
/// `D` (the independent construction: no SCF, no HF part). Physics hypothesis:
/// residual at the FD floor. Artifact hypothesis: a missing/doubled tau strain
/// term, lost weight response or wrong prefactor misses by O(|v_tau dtau/de|),
/// far above the floor; this test also asserts each defect misses.
#[test]
#[ignore = "validation: k-RKS SCAN/r2SCAN fixed-D XC stress vs FD of E_xc, atom grid, all nine + XC mutants; CI ~125 s"]
fn fixed_density_xc_stress_matches_fd_of_exc() {
    let grid = atom_grid(20, 50, Some(CUTOFF));
    for f in ["SCAN", "R2SCAN"] {
        let fx = fixture(&H2, [1, 1, 3], f, Jk::Dense, grid.clone());
        let s = stress(&fx, None);
        let d0 = fx.res.scf.densities.clone();
        let fd = fd_xc_all9(&fx.cell, [1, 1, 3], f, &grid, &d0, FD_H_XC);
        let miss = mdiff(&xc_part(&s), &fd);
        eprintln!(
            "{f} atom (20,50): |xc - FD| {miss:.2e}; |xc| max {:.3e}",
            mmax(&xc_part(&s))
        );
        assert!(miss < FD_BAR_XC, "{f}: {miss:e}");
        for m in XC_MUTANTS {
            let ms = stress(&fx, Some(m));
            let mm = mdiff(&xc_part(&ms), &fd);
            eprintln!("  {f} {m:?}: |xc - FD| {mm:.2e}");
            assert!(mm > MUTANT_BAR, "{f} {m:?} survives the FD anchor: {mm:e}");
        }
    }
}

/// Same on a uniform grid (no partition, no home atom).
#[test]
#[ignore = "validation: k-RKS SCAN/r2SCAN fixed-D XC stress vs FD of E_xc, uniform 24^3, all nine + XC mutants; CI ~201 s"]
fn fixed_density_xc_stress_matches_fd_on_the_uniform_grid() {
    uniform_fixed_density_xc_case(&["SCAN", "R2SCAN"], &ALL9, &XC_MUTANTS);
}

#[test]
#[ignore = "validation: k-RKS SCAN stress vs PySCF 2.13.1 krks_stress, 61^3; CI ~149 s"]
fn uniform_grid_stress_matches_pyscf_fftdf_krks_stress_scan() {
    pyscf_mgga_case("SCAN", "SCAN");
}

/// TPSS row of the PySCF anchor (issue #492). PySCF's own TPSS analytic
/// stress agrees with the central FD of its energy to 2.3e-10 (h = 1e-4,
/// xx / yy / zz / xz, run by hand with the generator's cell); its xx and yy
/// differ by 4.1e-8 and its xz is -3.4e-9 in BOTH the analytic tensor and the
/// FD (the uniform-grid egg-box effect, larger for TPSS than for SCAN), so
/// they are part of the reference, not noise. MEASURED (2026-10-22): ferric
/// vs PySCF 7.5e-11 (bar 1e-9); `NoTauStrain` misses by 6.5e-4. The source
/// mutant this pin exists for, `TPSS` mapped to the SCAN pair in the libxc
/// name table (run and reverted), misses by 7.8e-5: FD of ferric's own
/// energy cannot see it.
#[test]
#[ignore = "validation: k-RKS TPSS stress vs PySCF 2.13.1 krks_stress, 61^3; CI ~138 s"]
fn uniform_grid_stress_matches_pyscf_fftdf_krks_stress_tpss() {
    pyscf_mgga_case("TPSS", "TPSS");
}
