//! Open-shell k-point meta-GGA (UKS, SCAN / r2SCAN) on the RS-GDF J/K:
//! energy, nuclear force and stress (issue #492, item "open-shell meta-GGA on
//! RS-GDF"). Until this file the energy (`solve_kuks`) and the derivatives
//! (`kpoint_uks_gradient`, `kpoint_uks_stress`) refused the combination.
//!
//! What composes with what. A pure meta-GGA has no exact exchange, so the
//! J/K route enters only through the Coulomb matrix `J[D_α + D_β]`; the
//! per-spin tau path (`KPeriodicXcPolarized`, the tau force / strain terms of
//! `kuxcgrad` / `kuxcstress`) is the same object on both routes. The RS-GDF
//! open-shell J derivative is the one the UKS PBE / RSH RS-GDF force and
//! stress already use (#514, #555, #557). Nothing new is derived here; the
//! tests check the composition.
//!
//! # Artifact hypothesis next to the physics hypothesis
//!
//! * Physics right: (1) with an aux set that spans every pair density the
//!   RS-GDF fit is EXACT, so the polarized RS-GDF energy equals the dense-AFT
//!   one and PySCF `KUKS` (FFTDF) to SCF noise; (2) a singlet equals the
//!   closed-shell RS-GDF k-RKS (energy, force, stress); (3) the analytic
//!   force and stress equal the FD of the converged RS-GDF SCF energy.
//! * Artifact: the route drops or double-counts the tau term, or runs the
//!   spin-summed evaluator, on RS-GDF only. A tau defect moves the polarized
//!   energy by >= 1e-6 (measured on each test), far outside the 1e-9 / 1e-10
//!   bars; the singlet anchor is BLIND to spin-channel defects (there
//!   `τ_α = τ_β`), so every spin mutant is required to fail a POLARIZED pin.
//!
//! The weekly all-components FD and the r2SCAN derivative rows are
//! `validation_pbc_kuks_mgga_rsgdf.rs`.
//!
//! # Measured (2026-10-27, test profile)
//!
//! * Exact-aux anchor: |E_rsgdf - E_dense| <= 2.1e-13, |dE_xc| <= 4.9e-15;
//!   incomplete aux misses by 5.2e-4..2.1e-3.
//! * PySCF 2.13.1 `KUKS` pins (H3, one s per H, uniform 32^3): |dE| <= 1.5e-13
//!   (1x1x1 and 1x1x3, SCAN and r2SCAN). PySCF's energy is unchanged to 1e-12
//!   at J meshes 32 / 40 / 48.
//! * Singlet UKS == RKS on RS-GDF (H2, 1x1x3): energy 0, force 1.4e-15, stress
//!   2.6e-15.
//! * FD of the RS-GDF SCF energy (H3, et-sp aux, 1x1x3): force 2.0e-9 (SCAN) /
//!   1.9e-9 (r2SCAN) over all 9 components at h = 1e-4; stress 7.8e-8 / 5.8e-8
//!   over all 9 at h = 1e-4 (the h^2 stencil, as the dense-AFT UKS row).
//!
//! # Mutation ledger
//!
//! Test-hook defects run on the RS-GDF route (miss vs the oracle; SCAN, all
//! components, from the validation run):
//!
//! | defect | energy vs PySCF pin | force vs FD | stress vs FD |
//! |---|---|---|---|
//! | `SwapTau` / `TauSpinSwap` | 1.4e-2 | 3.4e-3 | 3.0e-2 |
//! | `DropBetaVtau` / `TauNoVxcBeta` | 3.0e-5 | 2.1e-3 | 1.4e-2 |
//! | `TotalTauBothChannels` | 5.2e-2 | 2.8e-3 | 1.6e-2 |
//! | `UnpolarizedEvaluator` | 4.3e-2 | - | - |
//! | `TauDropBeta` | - | 4.4e-3 | 2.5e-2 |
//! | `TauNoAo` / `NoTauStrain` | - | 6.9e-3 | 8.7e-2 |
//! | `TauDoubled` | - | 6.9e-3 | 8.7e-2 |
//! | `PulayJOneSpin` (RS-GDF J in the Pulay Fock) | - | 1.1e-2 | 7.8e-2 |
//!
//! Source mutants (edited, run, reverted): (1) `tau_beta` built from `D_alpha`
//! in `KPeriodicXcPolarized`: the PySCF pin misses by 4.1e-2 while the
//! exact-aux anchor (both routes share the defect) and the singlet anchor
//! (`D_alpha = D_beta`) stay green, which is why the pin exists; (2) the beta
//! `V^tau` dropped on the RS-GDF route only: the anchor misses by 2.2e-5, the
//! pin by 2.2e-5, and the singlet derivative is refused by the Fock guard.
//!
//! Not used as an oracle: the closed-shell "(RS-GDF - dense) transfers from
//! PBE to SCAN" check. On this polarized cell |SCAN - PBE| is 1.3e-5 (force)
//! and 4.8e-5 (stress) against route differences of 1.1e-4 / 4.1e-4, a 12%
//! density dependence of the fit error, too loose to be a bar.

mod common;
#[path = "kuks_mgga_rsgdf_support/mod.rs"]
mod support;

use common::*;
use ferric_pbc::kuks::KUksMggaMutation;
use ferric_pbc::kxcgrad::KXcGrid;
use support::*;

const FUNCTIONALS: [&str; 2] = ["SCAN", "R2SCAN"];

fn pin_run<'a>(
    bs: &'a ferric_core::basis::BasisSet,
    aux: Option<&'a ferric_integrals::basis_bridge::PreparedBasis>,
    n: [usize; 3],
    f: &'a str,
    grid: &'a KXcGrid,
) -> Run<'a> {
    Run {
        bs,
        aux,
        n,
        f,
        grid,
        mutation: None,
    }
}

// ------------------------------------------------------------ energies

/// EXACTNESS ANCHOR (written and passed before any PySCF number): with the
/// exact pair-midpoint aux the polarized RS-GDF SCAN / r2SCAN energy equals
/// the dense-AFT one. Negative control: the aux cut to the on-site class 0
/// (an incomplete fit) must miss.
///
/// Measured (H3 doublet, one s per H alpha 0.5, uniform 32^3): |E_rsgdf -
/// E_dense| 2.0e-13 / 2.0e-13 (1x1x1 SCAN / r2SCAN), 3.7e-14 / 3.5e-14
/// (1x1x3), |dE_xc| <= 3.1e-15; incomplete aux 2.1e-3 (1x1x1) and 5.2e-4
/// (1x1x3), >= 5e6 x the bar.
#[test]
fn polarized_mgga_rsgdf_equals_dense_in_the_trivial_aux_limit() {
    let cell = cube_cell(&H3, 2);
    let bs = single_s_h(ANCHOR_ALPHA);
    let full = midpoint_aux(&cell, true);
    let partial = midpoint_aux(&cell, false);
    let grid = KXcGrid::Uniform([PIN_GRID; 3]);
    for n in [[1, 1, 1], [1, 1, 3]] {
        for f in FUNCTIONALS {
            let d = kuks(&cell, &pin_run(&bs, None, n, f, &grid));
            let g = kuks(&cell, &pin_run(&bs, Some(&full), n, f, &grid));
            let p = kuks(&cell, &pin_run(&bs, Some(&partial), n, f, &grid));
            eprintln!(
                "anchor {n:?} {f}: E_dense {:.12} E_rsgdf {:.12} dE {:.2e} dE_xc {:.2e}; \
                 incomplete aux dE {:.2e}",
                d.scf.energy,
                g.scf.energy,
                g.scf.energy - d.scf.energy,
                g.e_xc - d.e_xc,
                p.scf.energy - d.scf.energy,
            );
            assert!(
                (g.scf.energy - d.scf.energy).abs() < ANCHOR_BAR,
                "{n:?} {f}"
            );
            assert!((g.e_xc - d.e_xc).abs() < ANCHOR_BAR, "{n:?} {f} E_xc");
            assert!(
                (p.scf.energy - d.scf.energy).abs() > 1e-5,
                "{n:?} {f}: incomplete aux left the anchor green"
            );
        }
    }
}

const ANCHOR_BAR: f64 = 1e-10;

/// The polarized RS-GDF energy at the exact-aux limit against PySCF 2.13.1
/// `KUKS` (FFTDF, the same uniform 32^3 XC grid;
/// `scripts/gen_pyscf_kuks_mgga_rsgdf_refs.py`), and the spin-resolved tau
/// defects of `solve_kuks` run on the RS-GDF route miss the same pin.
///
/// Measured: see `PIN_BAR` and the ledger in the module doc.
#[test]
fn polarized_mgga_rsgdf_matches_pinned_pyscf() {
    let cell = cube_cell(&H3, 2);
    let bs = single_s_h(ANCHOR_ALPHA);
    let aux = midpoint_aux(&cell, true);
    let grid = KXcGrid::Uniform([PIN_GRID; 3]);
    for (n, tag) in [([1, 1, 1], "1x1x1"), ([1, 1, 3], "1x1x3")] {
        for f in FUNCTIONALS {
            let want = pyscf_pin(&format!("{f} {tag}"));
            let e = kuks(&cell, &pin_run(&bs, Some(&aux), n, f, &grid))
                .scf
                .energy;
            eprintln!(
                "PySCF pin {f} {tag}: ferric {e:.12} PySCF {want:.12} dE {:.2e}",
                e - want
            );
            assert!((e - want).abs() < PIN_BAR, "{f} {tag}: {e} vs {want}");
            if tag == "1x1x1" {
                continue;
            }
            for m in [
                KUksMggaMutation::SwapTau,
                KUksMggaMutation::DropBetaVtau,
                KUksMggaMutation::TotalTauBothChannels,
                KUksMggaMutation::UnpolarizedEvaluator,
            ] {
                let mut run = pin_run(&bs, Some(&aux), n, f, &grid);
                run.mutation = Some(m);
                let bad = kuks(&cell, &run).scf.energy;
                eprintln!("  {f} {tag} {m:?}: miss {:.2e}", bad - want);
                assert!(
                    (bad - want).abs() > 10.0 * PIN_BAR,
                    "{f} {m:?} survives the pin"
                );
            }
        }
    }
}

const PIN_BAR: f64 = 1e-9;

// ------------------------------------------------- force and stress

/// Singlet k-UKS == k-RKS on the RS-GDF J/K: energy, force and stress from
/// independent SCF loops (H2 / STO-3G, et-sp aux, 1x1x3, atom grid). Blind to
/// every alpha / beta mix-up by construction (module doc); a dropped or
/// doubled tau term on the open-shell route alone would show.
#[test]
fn singlet_rsgdf_uks_equals_rks() {
    let fx = fixture(h_cell(&H2), [1, 1, 3], "SCAN", Jk::Gdf, atom_grid());
    let g = force(&fx, None, None);
    let s = stress(&fx, None, None);
    let (e, rg, rs) = rks_twin(&fx);
    let rows = [
        ("energy", (fx.res.scf.energy - e).abs()),
        ("force", diff(&g.grad, &rg.grad)),
        ("force xc_ao", diff(&g.xc_ao, &rg.xc_ao)),
        ("stress", mdiff(&s.de_deps, &rs.de_deps)),
        ("stress xc_ao", mdiff(&s.xc_ao, &rs.xc_ao)),
    ];
    assert!(amax(&g.xc_ao) > 1e-3 && mmax(&s.xc_ao) > 1e-3);
    for (name, d) in rows {
        eprintln!("SCAN RS-GDF singlet UKS vs RKS: {name:<13} {d:.2e}");
        assert!(d < SINGLET_BAR, "{name}: {d:e}");
    }
}

const SINGLET_BAR: f64 = 1e-10;

/// FD step, components and bars of the SCF-energy FD (H3 doublet, 1x1x3).
/// One component each per commit (every component, both functionals, in the
/// validation file). Bars: the dense-AFT UKS meta-GGA FD bar (force) and the
/// closed-shell RS-GDF meta-GGA stress bar (`pbc_kxcstress_mgga.rs`).
const FORCE_H: f64 = 1e-4;
const FORCE_COMPS: [(usize, usize); 1] = [(1, 2)];
const STRESS_H: f64 = 1e-4;
const STRESS_COMPS: [(usize, usize); 1] = [(2, 2)];
const FD_BAR: f64 = 1e-7;
const STRESS_FD_BAR: f64 = 5e-7;
const MUTANT_BAR: f64 = 1e-5;

/// Polarized RS-GDF SCAN force against the central FD of the converged RS-GDF
/// SCF energy (H3 doublet / STO-3G, et-sp aux, 1x1x3 complex Bloch AOs, atom
/// grid), and every spin / tau / RS-GDF-J Pulay defect misses the same FD.
#[test]
fn polarized_rsgdf_force_matches_fd_of_the_scf_energy() {
    let fx = fixture(h_cell(&H3), [1, 1, 3], "SCAN", Jk::Gdf, atom_grid());
    assert!(spin_asymmetry(&fx) > 0.1, "not polarized");
    let fd = fd_scf_force(&fx, &h_cell, &H3, FORCE_H, &FORCE_COMPS);
    let ok = force_miss(&force(&fx, None, None).grad, &fd, &FORCE_COMPS);
    eprintln!("SCAN RS-GDF force vs FD of the SCF energy: {ok:.2e} (|FD| {fd:?})");
    assert!(ok < FD_BAR, "{ok:e}");
    for (name, xm, sm) in force_mutants() {
        let bad = force_miss(&force(&fx, xm, sm).grad, &fd, &FORCE_COMPS);
        eprintln!("  force {name}: FD miss {bad:.2e}");
        assert!(bad > MUTANT_BAR, "{name} survives the FD: {bad:e}");
    }
}

/// Polarized RS-GDF SCAN stress against the central FD of the converged
/// RS-GDF SCF energy under cell strain (uniform 24^3 grid, which strains with
/// the cell), and the defects miss the same FD.
#[test]
fn polarized_rsgdf_stress_matches_fd_of_the_scf_energy() {
    let fx = fixture(
        h_cell(&H3),
        [1, 1, 3],
        "SCAN",
        Jk::Gdf,
        KXcGrid::Uniform([24; 3]),
    );
    let fd = fd_scf_stress(&fx, STRESS_H, &STRESS_COMPS);
    let ok = stress_miss(&stress(&fx, None, None).de_deps, &fd, &STRESS_COMPS);
    eprintln!("SCAN RS-GDF stress vs FD of the SCF energy: {ok:.2e} (FD {fd:?})");
    assert!(ok < STRESS_FD_BAR, "{ok:e}");
    for (name, xm, sm) in stress_mutants() {
        let bad = stress_miss(&stress(&fx, xm, sm).de_deps, &fd, &STRESS_COMPS);
        eprintln!("  stress {name}: FD miss {bad:.2e}");
        assert!(bad > MUTANT_BAR, "{name} survives the FD: {bad:e}");
    }
}
