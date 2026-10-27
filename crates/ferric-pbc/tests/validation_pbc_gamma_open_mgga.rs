//! Weekly validation of the Gamma-point open-shell meta-GGA energy of issue
//! #492 (`ferric_pbc::dft::gamma_uks`, `ferric_pbc::rohf::gamma_roks`, SCAN and
//! r2SCAN): the full table of PySCF 2.13.1 Gamma `pbc.dft.UKS` /
//! `pbc.dft.ROKS` pins (AFTDF + UniformGrids 61^3,
//! `scripts/gen_pyscf_gamma_open_mgga_refs.py`,
//! `testdata/reference/pbc_h3_sto3g_gamma_open_mgga.json`) and every tau mutant
//! (`GammaMggaMutation`) against those pins on BOTH routes. The per-commit
//! anchors, hypotheses and the other half of the mutation ledger are in
//! `pbc_gamma_open_mgga.rs`. Each 61^3 SCF takes ~20 s.

mod common;
#[path = "gamma_open_mgga_support/mod.rs"]
mod support;

use ferric_pbc::dft::PeriodicGrid;
use support::*;

/// Every route x functional pin to 1e-9.
#[test]
#[ignore = "validation: four 61^3 Gamma UKS / ROKS SCFs vs PySCF (~1.5 min)"]
fn h3_gamma_open_mgga_matches_every_pyscf_pin() {
    let su = setup(h3_doublet());
    let grid = PeriodicGrid::uniform(&su.cell, [61; 3]).unwrap();
    for route in ["UKS", "ROKS"] {
        for f in FUNCTIONALS {
            let e_ref = pyscf_ref(&format!("{route} {f}"));
            let (e, _) = run_route(route, &su, &mut open_xc(&su, &grid, f, None), f).unwrap();
            eprintln!(
                "H3 {route} {f}: {e:.12} PySCF {e_ref:.12} dE {:.2e}",
                e - e_ref
            );
            assert!((e - e_ref).abs() < 1e-9, "{route} {f}: {e} vs {e_ref}");
        }
    }
}

/// Every tau mutant misses the SCAN pin of both routes by > 1e-6 (the energy
/// is only second order in a dropped beta `V^tau`, hence the low bar; the
/// per-commit FD check is first order in it).
#[test]
#[ignore = "validation: eight mutated 61^3 Gamma UKS / ROKS SCFs vs PySCF (~3 min)"]
fn every_tau_mutant_misses_the_pyscf_pins() {
    let su = setup(h3_doublet());
    let grid = PeriodicGrid::uniform(&su.cell, [61; 3]).unwrap();
    for route in ["UKS", "ROKS"] {
        let e_ref = pyscf_ref(&format!("{route} SCAN"));
        for m in MUTANTS {
            let run = run_route(
                route,
                &su,
                &mut open_xc(&su, &grid, "SCAN", Some(m)),
                "SCAN",
            );
            let miss = mutant_miss(run, e_ref);
            eprintln!("H3 {route} SCAN {m:?}: |E - E_pyscf| {miss:.2e}");
            assert!(miss > 1e-6, "{route} {m:?} not detected: {miss:.2e}");
        }
    }
}
