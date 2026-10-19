//! Closed-shell k-point meta-GGA (SCAN / r2SCAN / TPSS) ENERGY on the RS-GDF
//! J/K (`solve_krks` with `jk = rsgdf`), issue #355.
//!
//! A pure meta-GGA has no exact exchange, so the J/K route enters only
//! through the Coulomb matrix `J(k)`, which is built from the q = 0 fit
//! blocks alone; the XC (tau) path is the same object on both routes. What
//! is independent of what:
//!
//! * `mgga_rsgdf_equals_dense_in_the_trivial_aux_limit` — the exactness
//!   anchor: on H2 / one s per H (alpha 0.5) at 1x1x3 the 24 pair-midpoint
//!   aux s functions of `pbc_krsgdf.rs` span every pair density, so RS-GDF
//!   is EXACT and the SCAN / r2SCAN / TPSS SCF energies on the two routes must agree
//!   to SCF noise. Negative control: the same anchor with the aux reduced to
//!   the on-site classes (an incomplete fit) must miss.
//! * `mgga_rsgdf_gap_is_the_fixed_density_coulomb_fit_error` — with a real
//!   aux set the RS-GDF energy differs from the dense-AFT energy by the
//!   Coulomb fit error. For a pure functional both SCFs are variational and
//!   differ only in J, so the gap equals the fixed-density fit error
//!   `E_J^fit(D) - E_J^dense(D)` at the dense route's own converged density,
//!   to second order in the fit error. PBE runs through the same check as the
//!   plain-functional reference; its gap is quoted but is not the oracle off
//!   1x1x1 (it is the fit error of a different density).
//!
//! Artifact hypothesis vs physics hypothesis: if the meta-GGA composes with
//! RS-GDF correctly, the anchor gap is SCF noise (<= 1e-10) and the real-aux
//! gap equals the fixed-density fit error to second order in that error. If the RS-GDF route dropped or double-counted the
//! tau term, or ran a different functional, the gap would be the size of a
//! tau contribution (the SCAN - r2SCAN gap is 2.2e-3 Ha on the triclinic cell
//! of `pbc_krks.rs`; a dropped V^tau on one route moves the anchor by 6.5e-5),
//! far outside both bars. Measurements are on each test.
//!
//! Mutation ledger (source mutants, run and reverted 2026-10-18):
//! * V^tau zeroed on the RS-GDF route only (`f_tau = 0` in `KPeriodicXc` when
//!   an aux basis is given): the anchor misses by 6.5e-5 Ha (SCAN; bar
//!   1e-10); the fit-error test fails with an unconverged TPSS SCF (the Fock
//!   is no longer the energy's derivative). The 1x1x1 fit-error cases are
//!   BLIND to it (residual 9e-16): the H2 / STO-3G density there is fixed by
//!   symmetry, so no Fock defect can move it; the 1x1x3 cases and the anchor
//!   carry the check.
//! * Incomplete aux (in-test control): the anchor misses by 2.3e-4 Ha.

mod common;

use common::*;
use ferric_core::basis::{self, BasisSet};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::site_basis::SiteBasis;
use ferric_pbc::dense_aft::ExxDiv;
use ferric_pbc::dft::PeriodicGrid;
use ferric_pbc::hcore::PeriodicHcoreConfig;
use ferric_pbc::kdft::{solve_krks_keep_on_grid, solve_krks_on_grid, KRksConfig, KRksResult};
use ferric_pbc::kpts::KPointMesh;
use ferric_pbc::kscf::{KJkKind, KPointJk, KRetained, KRetainedJk};
use ferric_pbc::lattice::Cell;
use ferric_pbc::rsgdf::kpoint::KRsGdfConfig;
use ferric_pbc::rsgdf::RsGdfConfig;
use ndarray::Array2;
use num_complex::Complex64;

const OMEGA_HCORE: f64 = 0.8;
const ANCHOR_ALPHA: f64 = 0.5;
const AMPLE: usize = 1 << 30;
const GRID: usize = 32;

fn krks_cfg(cell: &Cell, functional: &str, rsgdf: bool) -> KRksConfig {
    let mut cfg = KRksConfig::new(cell, ExxDiv::Ewald, functional);
    cfg.krhf.hcore = PeriodicHcoreConfig::with_omega(OMEGA_HCORE);
    cfg.krhf.scf.energy_conv = 1e-12;
    cfg.krhf.scf.grad_conv = 1e-9;
    if rsgdf {
        cfg.krhf.jk = KJkKind::RsGdf;
        cfg.krhf.rsgdf = KRsGdfConfig {
            gdf: RsGdfConfig {
                omega: 0.8,
                exxdiv: ExxDiv::Ewald,
                budget_bytes: Some(AMPLE),
                ..Default::default()
            },
            mutation: None,
        };
    }
    cfg
}

/// SCF energy per cell on the uniform `GRID`^3 grid; `aux = None` is the
/// dense-AFT J/K, `Some` the RS-GDF J/K on that aux basis.
fn energy(
    cell: &Cell,
    bs: &BasisSet,
    aux: Option<&PreparedBasis>,
    mesh: &KPointMesh,
    functional: &str,
) -> f64 {
    let prep = prep_for(cell, bs);
    let grid = PeriodicGrid::uniform(cell, [GRID; 3]).unwrap();
    let cfg = krks_cfg(cell, functional, aux.is_some());
    solve_krks_on_grid(cell, &prep, aux, mesh, &grid, &cfg)
        .unwrap_or_else(|e| panic!("solve_krks {functional}: {e}"))
        .scf
        .energy
}

/// The pair-midpoint aux centres of `pbc_krsgdf.rs` restricted to the
/// half-lattice `classes` (all 8 = the exact anchor).
fn anchor_sites(cell: &Cell, classes: &[usize]) -> Vec<[f64; 4]> {
    let r = cell.positions();
    let a = cell.lattice();
    let mut out = Vec::new();
    for (i, j) in [(0usize, 0usize), (1, 1), (0, 1)] {
        for k in classes {
            let h = [(k >> 2) & 1, (k >> 1) & 1, k & 1];
            let mut c = [0.0; 3];
            for d in 0..3 {
                let ha: f64 = (0..3).map(|x| h[x] as f64 * a[x][d]).sum();
                c[d] = 0.5 * (r[i][d] + r[j][d] + ha);
            }
            out.push([c[0], c[1], c[2], 2.0 * ANCHOR_ALPHA]);
        }
    }
    out
}

// Measured (H2 / one s per H, 1x1x3, uniform 32^3): |E_rsgdf - E_dense|
// 3.4e-14 / 3.2e-14 / 3.3e-14 Ha for SCAN / r2SCAN / TPSS. Bar 1e-10 (the
// k-RHF anchor bar of `pbc_krsgdf.rs`). Negative control (aux reduced to the
// half-lattice class 0): misses by 2.3e-4 / 2.4e-4 Ha, >= 1e6 x the bar.

#[test]
fn mgga_rsgdf_equals_dense_in_the_trivial_aux_limit() {
    let cell = h2_cell(4.0);
    let bs = single_s_h(ANCHOR_ALPHA);
    let mesh = KPointMesh::gamma_centred(&cell, [1, 1, 3]).unwrap();
    let all: Vec<usize> = (0..8).collect();
    let full = SiteBasis::new(&anchor_sites(&cell, &all), 0).unwrap();
    let partial = SiteBasis::new(&anchor_sites(&cell, &[0]), 0).unwrap();
    for f in ["SCAN", "R2SCAN", "TPSS"] {
        let d = energy(&cell, &bs, None, &mesh, f);
        let g = energy(&cell, &bs, Some(&full.prep), &mesh, f);
        let p = energy(&cell, &bs, Some(&partial.prep), &mesh, f);
        eprintln!(
            "anchor {f}: E_dense {d:.12} E_rsgdf {g:.12} dE {:.2e}; incomplete aux dE {:.2e}",
            g - d,
            p - d
        );
        assert!((g - d).abs() < 1e-10, "{f}: {g} vs {d}");
        assert!(
            (p - d).abs() > 1e-5,
            "{f}: incomplete aux left the anchor green"
        );
    }
}

/// `E_J(D) = (1/(2 N_k)) Σ_k Re Tr[D(k) J(k)]` of `jk` at the densities `d`
/// (the k-point Fock convention: `J(k)` carries no `1/N_k`).
fn coulomb_energy(jk: &mut dyn KPointJk, d: &[Array2<Complex64>]) -> f64 {
    let n = d[0].nrows();
    let mut j = vec![Array2::<Complex64>::zeros((n, n)); d.len()];
    let mut k = j.clone();
    jk.build(d, &mut j, &mut k).unwrap();
    let mut e = 0.0;
    for (dk, jk) in d.iter().zip(&j) {
        for a in 0..n {
            for b in 0..n {
                e += (dk[(a, b)] * jk[(b, a)]).re;
            }
        }
    }
    e / (2.0 * d.len() as f64)
}

/// SCF result and retained J/K on the uniform grid (as [`energy`]).
fn keep(
    cell: &Cell,
    bs: &BasisSet,
    aux: Option<&PreparedBasis>,
    mesh: &KPointMesh,
    functional: &str,
) -> (KRksResult, KRetained) {
    let prep = prep_for(cell, bs);
    let grid = PeriodicGrid::uniform(cell, [GRID; 3]).unwrap();
    let cfg = krks_cfg(cell, functional, aux.is_some());
    solve_krks_keep_on_grid(cell, &prep, aux, mesh, &grid, &cfg)
        .unwrap_or_else(|e| panic!("solve_krks {functional}: {e}"))
}

// The oracle. A pure functional's RS-GDF and dense-AFT energies differ only
// through J, and both SCFs are variational, so
//   E_rsgdf - E_dense = [E_J^fit(D) - E_J^dense(D)] + R,   R <= 0,
// at the dense route's converged density D: the fixed-density Coulomb fit
// error of THIS functional's density plus the orbital relaxation R of the
// fitted SCF away from D, which is second order in the fit error and
// non-positive (the fitted SCF minimizes its own energy). At 1x1x1 the one
// occupied orbital of H2 / STO-3G is fixed by symmetry, so R = 0.
//
// Measured (H2 / STO-3G a = 4, uniform 32^3; the eprintln table):
//   1x1x1, both aux, PBE / SCAN / r2SCAN / TPSS: |R| <= 1.7e-15 (the test
//   keeps cc-pvdz-ri only at 1x1x1).
//   1x1x3 cc-pvdz-ri: R = -8.50e-8 / -8.27e-8 / -8.19e-8 / -8.81e-8 against
//   fits of 7.0e-5..8.1e-5, |R| / |fit| <= 1.15e-3 (PBE), 1.09e-3 (TPSS).
//   1x1x3 def2-universal-jkfit: R = -2.1e-10..-3.5e-10 (|R| / |fit| <= 2.6e-5).
// Bar REL_BAR = 1e-2 (~9x the worst measured ratio). The other side: the fit
// error of the WRONG density (the PBE density's) misses the meta-GGA gap by
// 4.1..9.0 % of the fit at 1x1x3 cc-pvdz-ri and 1.4..3.3 % with
// def2-universal-jkfit, and a tau defect on one route is >= 1e-3 Ha (> 10x
// any fit here). So the plain-functional (PBE) gap is not a first-order
// oracle off 1x1x1; it is printed alongside.
const REL_BAR: f64 = 1e-2;

#[test]
fn mgga_rsgdf_gap_is_the_fixed_density_coulomb_fit_error() {
    let cell = h2_cell(4.0);
    let bs = pyscf_sto3g_h();
    // 1x1x1 (symmetric, R = 0 exactly) once; 1x1x3 with both aux sets.
    for (n, aux_name) in [
        ([1, 1, 1], "cc-pvdz-ri"),
        ([1, 1, 3], "cc-pvdz-ri"),
        ([1, 1, 3], "def2-universal-jkfit"),
    ] {
        let mesh = KPointMesh::gamma_centred(&cell, n).unwrap();
        let aux = PreparedBasis::new(cell.mol(), &basis::bundled(aux_name).unwrap()).unwrap();
        let mut pbe_fit = 0.0;
        for f in ["PBE", "SCAN", "R2SCAN", "TPSS"] {
            let (d, dk) = keep(&cell, &bs, None, &mesh, f);
            let (g, gk) = keep(&cell, &bs, Some(&aux), &mesh, f);
            let (KRetainedJk::Dense(eri), KRetainedJk::RsGdf(gdf)) = (&dk.jk, &gk.jk) else {
                panic!("unexpected J/K routes");
            };
            let dm = &d.scf.densities;
            let fit = coulomb_energy(&mut gdf.jk_builder(), dm)
                - coulomb_energy(&mut eri.jk_builder(), dm);
            let gap = g.scf.energy - d.scf.energy;
            if f == "PBE" {
                pbe_fit = fit;
            }
            eprintln!(
                "{n:?} {aux_name} {f}: gap {gap:.6e} fixed-D fit {fit:.6e} residual {:.2e} \
                 (PBE-density fit {pbe_fit:.6e})",
                gap - fit
            );
            assert!(fit.abs() > 1e-6, "fit error is not measurable");
            let r = gap - fit;
            assert!(
                r < 1e-12 && r.abs() < REL_BAR * fit.abs(),
                "{n:?} {aux_name} {f}: gap {gap:e} vs fixed-density fit {fit:e}"
            );
        }
    }
}
