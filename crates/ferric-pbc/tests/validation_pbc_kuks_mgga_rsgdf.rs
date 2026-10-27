//! Weekly validation of the open-shell k-point meta-GGA (UKS SCAN / r2SCAN)
//! force and stress on the RS-GDF J/K (issue #492). Every test here is
//! `#[ignore]`d (the SCF-energy FD costs ~5 min per test). Per commit,
//! `pbc_kuks_mgga_rsgdf.rs` runs the exact-aux anchor, the PySCF energy pins
//! with the tau defects, the singlet UKS == RKS anchor with `TauDropBeta`,
//! `TauDoubled` and `PulayJOneSpin` (force and stress), and ONE polarized
//! strain component against FD with `TauNoVxcBeta`. The alpha / beta
//! exchange defects of the derivative (`TauSpinSwap`, `TauTotalBothChannels`
//! on the force) and every remaining component are caught HERE only.
//!
//! System: the H3 doublet (N_alpha = 2, N_beta = 1) in the a = 4 cube,
//! STO-3G H, even-tempered s + p aux, RS-GDF `omega = 0.8`, hcore
//! `omega = 0.8`, gamma-centred 1x1x3 (complex Bloch AOs). Force on the atom
//! grid (20, 50) with neighbour cutoff 14, stress on the uniform 24^3 grid
//! (which strains with the cell). Oracle: the central FD of the converged
//! RS-GDF SCF energy, every component.
//!
//! MEASURED: the eprintln table of each test (numbers in the test docs).

mod common;
#[path = "kuks_mgga_rsgdf_support/mod.rs"]
mod support;

use ferric_pbc::kxcgrad::KXcGrid;
use support::*;

const FORCE_H: f64 = 1e-4;
const STRESS_H: f64 = 1e-4;
const FD_BAR: f64 = 1e-7;
const STRESS_FD_BAR: f64 = 5e-7;
const MUTANT_BAR: f64 = 1e-5;

fn all_atoms(n: usize) -> Vec<(usize, usize)> {
    (0..n).flat_map(|a| (0..3).map(move |x| (a, x))).collect()
}

/// Force: all nine components, SCAN and r2SCAN, with the spin / tau / Pulay
/// defects missing the same FD on every component taken together.
#[test]
#[ignore = "validation: k-UKS SCAN / r2SCAN force on RS-GDF J vs FD of the SCF energy, all components (36 SCFs)"]
fn polarized_rsgdf_force_matches_fd_all_components() {
    let comps = all_atoms(H3.len());
    for f in ["SCAN", "R2SCAN"] {
        let fx = fixture(h_cell(&H3), [1, 1, 3], f, Jk::Gdf, atom_grid());
        assert!(spin_asymmetry(&fx) > 0.1, "{f}: not polarized");
        let fd = fd_scf_force(&fx, &h_cell, &H3, FORCE_H, &comps);
        let g = force(&fx, None, None);
        let ok = force_miss(&g.grad, &fd, &comps);
        let sum: f64 = (0..3)
            .map(|x| g.grad.column(x).sum().abs())
            .fold(0.0, f64::max);
        eprintln!(
            "{f} RS-GDF force, all 9: |analytic - FD| {ok:.2e}, max |FD| {:.3e}, |sum_A F_A| {sum:.1e}",
            fd.iter().fold(0.0_f64, |m, x| m.max(x.abs()))
        );
        assert!(ok < FD_BAR, "{f}: {ok:e}");
        for (name, xm, sm) in force_mutants() {
            let bad = force_miss(&force(&fx, xm, sm).grad, &fd, &comps);
            eprintln!("  {f} force {name}: FD miss {bad:.2e}");
            assert!(bad > MUTANT_BAR, "{f} {name}: {bad:e}");
        }
    }
}

/// Stress: all nine strain components, SCAN and r2SCAN, with the defects.
#[test]
#[ignore = "validation: k-UKS SCAN / r2SCAN stress on RS-GDF J vs FD of the SCF energy, all components (36 SCFs)"]
fn polarized_rsgdf_stress_matches_fd_all_components() {
    for f in ["SCAN", "R2SCAN"] {
        let fx = fixture(
            h_cell(&H3),
            [1, 1, 3],
            f,
            Jk::Gdf,
            KXcGrid::Uniform([24; 3]),
        );
        assert!(spin_asymmetry(&fx) > 0.1, "{f}: not polarized");
        let fd = fd_scf_stress(&fx, STRESS_H, &ALL9);
        let s = stress(&fx, None, None);
        let ok = stress_miss(&s.de_deps, &fd, &ALL9);
        eprintln!(
            "{f} RS-GDF stress, all 9: |analytic - FD| {ok:.2e}, max |FD| {:.3e}",
            fd.iter().fold(0.0_f64, |m, x| m.max(x.abs()))
        );
        assert!(ok < STRESS_FD_BAR, "{f}: {ok:e}");
        for (name, xm, sm) in stress_mutants() {
            let bad = stress_miss(&stress(&fx, xm, sm).de_deps, &fd, &ALL9);
            eprintln!("  {f} stress {name}: FD miss {bad:.2e}");
            assert!(bad > MUTANT_BAR, "{f} {name}: {bad:e}");
        }
    }
}
