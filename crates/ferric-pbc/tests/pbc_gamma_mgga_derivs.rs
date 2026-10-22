//! Gamma-point closed-shell meta-GGA forces and stress (issue #492),
//! `ferric_pbc::gamma_mgga`: the Gamma SCF (`gamma_rks`, real lattice-summed
//! AOs, the Gamma J and XC) routed into the 1x1x1 k-point force / stress.
//!
//! Because the derivative IS the k-point code, the oracles that test the
//! ROUTE are constructions the k-point code does not share:
//! * `tpss_gamma_force_matches_fd_of_the_gamma_energy` /
//!   `tpss_gamma_stress_matches_fd_of_the_gamma_energy` — central FD of the
//!   GAMMA SCF energy (fresh `gamma_rks` per displacement / strain), TPSS
//!   per commit; SCAN / r2SCAN / TPSS, all three components and the h sweeps
//!   weekly (`validation_pbc_mgga_derivs.rs`). Physics hypothesis: the h^2
//!   floor. Artifact hypothesis: a glue slip (density factor, Fock, k-point,
//!   hcore rebuilt from another config) or a missing tau term misses by
//!   >= 1e-4.
//! * `routed_gga_equals_the_native_gamma_force_and_stress` — for PBE the
//!   routed path and the NATIVE Gamma force / stress (`grad.rs`,
//!   `stress.rs`, real-AO code) are independent implementations of the same
//!   derivative; they must agree to roundoff-level.
//! * `gamma_mgga_force_is_the_one_point_mesh_force` — the routed Gamma force
//!   against `solve_krks` + `kpoint_rks_gradient` at 1x1x1 (separate SCFs on
//!   the same grid): a GLUE check (the derivative code is shared).
//! * refusals and the Fock guard: another grid, an RS-GDF Gamma SCF, the
//!   wrong functional, an open-shell result, a range-separated functional or
//!   kernel, `V^tau` missing from the rebuilt Fock; the native Gamma entries
//!   refuse a meta-GGA with a pointer here; the binding gates.
//!
//! MEASURED (H2 / STO-3G, cubic 4 Bohr, (20, 50) grid, cutoff 14 Bohr,
//! 2026-10-22): routed vs native PBE force 1.3e-15 (max |F| 0.17), dE/deps
//! 2.1e-14; Gamma vs 1x1x1 force 5.6e-17 (SCAN) / 1.4e-16 (TPSS), energy
//! difference <= 4.4e-16, routed Fock guard <= 8.9e-16; an RS-GDF Gamma SCAN
//! SCF misses the guard by 9.5e-6 (bar 1e-8). `GLUE_BAR` 1e-9 is >= 1e5 x the
//! measured glue residuals and >= 1e6 below the smallest mutant miss.
//!
//! MUTATION LEDGER: in the test docs below (each mutant must miss the FD of the
//! Gamma energy by more than `MUTANT_BAR`).
//!
//! NOT covered: Gamma UKS / ROKS meta-GGA (no Gamma tau SCF), RS-GDF J
//! (refused before the SCF by the binding gate and by the Fock guard after
//! it), a caller-supplied grid (`gamma_rks_on_grid`; the Fock guard refuses a
//! mismatched grid), p/d shells, the (75, 302) production grid.

mod common;
#[path = "gamma_mgga_support/mod.rs"]
mod support;

use common::*;
use ferric_pbc::dense_aft::ExxDiv;
use ferric_pbc::dft::{gamma_rks, PeriodicGridConfig};
use ferric_pbc::gamma_mgga::{gamma_rks_mgga_gradient, GammaMggaDerivConfig};
use ferric_pbc::grad::{gamma_rks_gradient_with, GammaGradConfig};
use ferric_pbc::hcore::periodic_hcore;
use ferric_pbc::kdense_aft::KDenseAftEri;
use ferric_pbc::kdft::{solve_krks, KRksConfig};
use ferric_pbc::kgrad::KGradJk;
use ferric_pbc::kpts::KPointMesh;
use ferric_pbc::kxcgrad::{kpoint_rks_gradient, KRksGradConfig, KXcGrid, KXcMutation};
use ferric_pbc::kxcstress::KXcStressMutation;
use ferric_pbc::rsgdf::{RsGdf, RsGdfConfig};
use ferric_pbc::stress::{gamma_rks_stress, GammaStressConfig};
use ferric_pbc::uhf::GammaUhfIntegrals;
use support::*;

/// Per-commit FD rows: the two force components and two strain components
/// on which every ledger mutant misses (the full three-functional matrix and
/// the h sweeps are in `validation_pbc_mgga_derivs.rs`).
const FORCE_ROW: [(usize, usize); 2] = [(0, 2), (1, 1)];
const STRAIN_ROW: [(usize, usize); 2] = [(0, 0), (2, 2)];

// ------------------------------------------------------------------ FD oracles

/// TPSS routed Gamma force against FD of the Gamma SCF energy (two
/// components, h = 1e-4), and the force mutants against the same FD.
///
/// MUTATION LEDGER (production-wired `KXcMutation`; miss = max |F_mut - FD|
/// over the full three-component row of the weekly test, H2 / STO-3G cubic
/// 4 Bohr, (20, 50) grid, measured 2026-10-22): `TauNoAo` 1.71e-2,
/// `TauDoubled` 1.71e-2, `TauNoPoint` 4.07e-3, `TauNoVxcUnguarded` (V^tau
/// zeroed in the Pulay Fock) 1.37e-3, `NoWeightDeriv` (grid response
/// dropped) 1.13e-1, `NoPointMotion` 1.15e-1; correct 2.3e-9. `TauNoVxc`
/// (guard on) is refused (`refusals_and_the_fock_guard`). The TPSS -> SCAN
/// mapping (source mutant, run and reverted 2026-10-22) is invisible to an FD
/// of ferric's own energy (2.3e-9, both sides move); it is caught here by the
/// SCAN-vs-TPSS gap below (measured 1.6e-3 correct, 2.3e-9 under the mutant)
/// and by the PySCF pins (`pbc_krks.rs` energy, `pbc_kxcstress_mgga.rs`
/// stress, 7.8e-5).
#[test]
fn tpss_gamma_force_matches_fd_of_the_gamma_energy() {
    let g = gamma(cell_of(&H2), "TPSS");
    let an = force(&g, None);
    let fd = fd_force("TPSS", H, &FORCE_ROW);
    let w = worst_force(&an.grad, &fd, &FORCE_ROW);
    eprintln!("TPSS: Gamma routed force vs FD of the Gamma energy {w:.2e}");
    assert!(w < FD_BAR, "{w:e}");
    for m in [
        KXcMutation::TauNoAo,
        KXcMutation::TauDoubled,
        KXcMutation::TauNoPoint,
        KXcMutation::TauNoVxcUnguarded,
        KXcMutation::NoWeightDeriv,
        KXcMutation::NoPointMotion,
    ] {
        let mm = worst_force(&force(&g, Some(m)).grad, &fd, &FORCE_ROW);
        eprintln!("  TPSS {m:?}: miss vs FD {mm:.2e}");
        assert!(mm > MUTANT_BAR, "{m:?} survives: {mm:e}");
    }
    // TPSS is not SCAN here: the SCAN force on its own SCF is far off this FD.
    let scan = force(&gamma(cell_of(&H2), "SCAN"), None);
    let gap = worst_force(&scan.grad, &fd, &FORCE_ROW);
    eprintln!("SCAN force vs the TPSS FD {gap:.2e}");
    assert!(gap > MUTANT_BAR, "{gap:e}");
}

/// TPSS routed Gamma stress against FD of the Gamma SCF energy under strain
/// (two components), and the stress mutants.
///
/// MUTATION LEDGER (`KXcStressMutation`, miss = max |dE/deps_mut - FD| over
/// the weekly three-component row, measured 2026-10-22): `NoTauStrain`
/// 2.95e-2, `TauDoubled` 2.95e-2, `TauNoVxc` (V^tau dropped from the Pulay
/// Fock, guard skipped) 2.97e-2, `NoWeightDeriv` 5.42e-1, `NoPointMotion`
/// 1.62e-1; correct 3.0e-8.
#[test]
fn tpss_gamma_stress_matches_fd_of_the_gamma_energy() {
    let g = gamma(cell_of(&H2), "TPSS");
    let s = stress(&g, None);
    let fd = fd_stress("TPSS", H, &STRAIN_ROW);
    let w = worst_stress(&s.de_deps, &fd, &STRAIN_ROW);
    eprintln!("TPSS: Gamma routed dE/deps vs FD of the Gamma energy {w:.2e}");
    assert!(w < STRESS_FD_BAR, "{w:e}");
    for m in [
        KXcStressMutation::NoTauStrain,
        KXcStressMutation::TauDoubled,
        KXcStressMutation::TauNoVxc,
        KXcStressMutation::NoWeightDeriv,
        KXcStressMutation::NoPointMotion,
    ] {
        let mm = worst_stress(&stress(&g, Some(m)).de_deps, &fd, &STRAIN_ROW);
        eprintln!("  TPSS {m:?}: miss vs FD {mm:.2e}");
        assert!(mm > MUTANT_BAR, "{m:?} survives: {mm:e}");
    }
}

// ----------------------------------------------------------- glue anchors

/// PBE through the route equals the NATIVE Gamma force and stress (real-AO
/// `grad.rs` / `stress.rs`, an independent implementation). Artifact: a
/// route slip (density factor 2, hcore rebuilt differently, a wrong k-point
/// or Madelung) moves the force by >= 1e-3.
#[test]
fn routed_gga_equals_the_native_gamma_force_and_stress() {
    let g = gamma(cell_of(&H2), "PBE");
    let routed = force(&g, None);
    let gcfg = GammaGradConfig {
        budget_bytes: Some(AMPLE),
        ..Default::default()
    };
    let native = gamma_rks_gradient_with(
        &g.cell,
        &g.prep,
        &hcfg(),
        &g.hc,
        &g.eri,
        &g.scf,
        &g.cfg,
        &gcfg,
    )
    .unwrap();
    let d = diff(&routed.grad, &native.grad);
    eprintln!(
        "PBE routed vs native Gamma force {d:.2e} (max |F| {:.3e})",
        native.grad.iter().fold(0.0_f64, |m, v| m.max(v.abs()))
    );
    assert!(d < GLUE_BAR, "{d:e}");
    let scfg = GammaStressConfig {
        budget_bytes: Some(AMPLE),
        ..Default::default()
    };
    let native_s = gamma_rks_stress(
        &g.cell,
        &g.prep,
        &hcfg(),
        &g.hc,
        &g.eri,
        &g.scf,
        &g.cfg,
        &scfg,
    )
    .unwrap();
    let ds = mdiff(&stress(&g, None).de_deps, &native_s.de_deps);
    eprintln!("PBE routed vs native Gamma dE/deps {ds:.2e}");
    assert!(ds < GLUE_BAR, "{ds:e}");
}

/// The routed Gamma force equals `solve_krks` + `kpoint_rks_gradient` at
/// 1x1x1 (two SCFs, the Gamma J / XC / hcore vs the k ones). A glue check:
/// the derivative code is shared.
#[test]
fn gamma_mgga_force_is_the_one_point_mesh_force() {
    for f in ["SCAN", "TPSS"] {
        let g = gamma(cell_of(&H2), f);
        let routed = force(&g, None);
        let mesh = KPointMesh::gamma_centred(&g.cell, [1, 1, 1]).unwrap();
        let mut kc = KRksConfig::new(&g.cell, ExxDiv::Ewald, f);
        kc.krhf.hcore = hcfg();
        kc.krhf.scf.energy_conv = 1e-13;
        kc.krhf.scf.grad_conv = 1e-10;
        kc.krhf.scf.max_iter = 300;
        kc.grid = grid();
        let k = solve_krks(&g.cell, &g.prep, None, &mesh, &kc).unwrap();
        let hk = ferric_pbc::hcore::kpoint::periodic_hcore_kpts(&g.cell, &g.prep, &mesh, &hcfg())
            .unwrap();
        let eri = KDenseAftEri::build(
            &g.cell,
            &g.prep,
            &mesh,
            &hk.s,
            ExxDiv::Ewald,
            &Default::default(),
        )
        .unwrap();
        let kg = kpoint_rks_gradient(
            &g.cell,
            &g.prep,
            &mesh,
            &hcfg(),
            &hk,
            KGradJk::Dense(&eri),
            &k.scf,
            f,
            &KRksGradConfig {
                grad: dcfg().grad,
                grid: KXcGrid::Atom(grid()),
                exxdiv: Some(ExxDiv::Ewald),
                ..Default::default()
            },
        )
        .unwrap();
        let d = diff(&routed.grad, &kg.grad);
        eprintln!(
            "{f}: Gamma vs 1x1x1 force {d:.2e}; E {:.2e}; routed Fock guard {:.1e}",
            g.scf.energy - k.scf.energy,
            routed.hf.fock_mismatch
        );
        assert!(d < GLUE_BAR, "{f}: {d:e}");
    }
}

// ------------------------------------------------------ refusals and guard

#[test]
fn refusals_and_the_fock_guard() {
    let g = gamma(cell_of(&H2), "TPSS");
    let ok = force(&g, None);
    eprintln!("TPSS routed Fock guard {:.1e}", ok.hf.fock_mismatch);
    // Another grid in the config: another V_xc, refused by the guard.
    let mut other = g.cfg.clone();
    other.grid = PeriodicGridConfig::with_size(30, 110);
    other.grid.neighbour_cutoff = Some(CUTOFF);
    let e = try_force(&g, &other, &g.scf, &dcfg())
        .expect_err("another grid accepted")
        .to_string();
    assert!(e.contains("differs"), "{e}");
    // The wrong functional (the TPSS -> SCAN mapping seen from the force).
    for wrong in ["SCAN", "PBE"] {
        let e = try_force(&g, &rks_cfg(wrong), &g.scf, &dcfg())
            .expect_err(wrong)
            .to_string();
        assert!(e.contains("differs"), "{wrong}: {e}");
    }
    // V^tau dropped from the rebuilt Fock with the guard on.
    let c = GammaMggaDerivConfig {
        xc_grad_mutation: Some(KXcMutation::TauNoVxc),
        ..dcfg()
    };
    let e = try_force(&g, &g.cfg, &g.scf, &c)
        .expect_err("Fock without V^tau accepted")
        .to_string();
    assert!(e.contains("differs"), "{e}");
    // An open-shell result.
    let mut open = g.scf.clone();
    open.spin = ferric_scf::result::Spin::Unrestricted;
    let e = try_force(&g, &g.cfg, &open, &dcfg())
        .expect_err("open shell accepted")
        .to_string();
    assert!(e.contains("closed-shell (restricted) results only"), "{e}");
    // Range-separated: the functional and a range-separated kernel config.
    let e = try_force(&g, &rks_cfg("HSE06"), &g.scf, &dcfg())
        .expect_err("HSE06 accepted")
        .to_string();
    assert!(e.contains("gamma_rsh"), "{e}");
    let mut rk = dcfg();
    rk.kdense.rsh = Some(ferric_pbc::rsh::RshParams {
        omega: 0.11,
        c_sr: 0.25,
        c_lr: 0.0,
    });
    let e = try_force(&g, &g.cfg, &g.scf, &rk)
        .expect_err("RSH kernels accepted")
        .to_string();
    assert!(e.contains("gamma_rsh"), "{e}");
    // The native Gamma entries refuse a meta-GGA with a pointer here.
    let e = gamma_rks_gradient_with(
        &g.cell,
        &g.prep,
        &hcfg(),
        &g.hc,
        &g.eri,
        &g.scf,
        &g.cfg,
        &GammaGradConfig::default(),
    )
    .expect_err("native Gamma meta-GGA force")
    .to_string();
    assert!(e.contains("gamma_rks_mgga_gradient"), "{e}");
    let e = gamma_rks_stress(
        &g.cell,
        &g.prep,
        &hcfg(),
        &g.hc,
        &g.eri,
        &g.scf,
        &g.cfg,
        &GammaStressConfig::default(),
    )
    .expect_err("native Gamma meta-GGA stress")
    .to_string();
    assert!(e.contains("gamma_rks_mgga_stress"), "{e}");
}

/// An RS-GDF Gamma SCF: the routed derivative rebuilds the dense J, so the
/// Fock guard refuses it (the fit error is far above the guard bar).
#[test]
fn an_rsgdf_gamma_scf_is_refused_by_the_fock_guard() {
    let cell = cell_of(&H2);
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    let hc = periodic_hcore(&cell, &prep, &hcfg()).unwrap();
    let aux = ferric_integrals::basis_bridge::PreparedBasis::new(
        cell.mol(),
        &ferric_core::basis::bundled("cc-pvdz-ri").unwrap(),
    )
    .unwrap();
    let cfg = rks_cfg("SCAN");
    let gcfg = RsGdfConfig {
        omega: 1.0,
        exxdiv: cfg.exxdiv,
        budget_bytes: Some(AMPLE),
        ..Default::default()
    };
    let gdf = RsGdf::build(&cell, &prep, &aux, &hc.s, &gcfg).unwrap();
    let r = gamma_rks(&cell, &prep, &hc, GammaUhfIntegrals::RsGdf(&gdf), &cfg).unwrap();
    let e = gamma_rks_mgga_gradient(&cell, &prep, &hcfg(), &r.scf, &cfg, &dcfg())
        .expect_err("RS-GDF Gamma SCF accepted")
        .to_string();
    eprintln!("RS-GDF Gamma SCAN: {e}");
    assert!(e.contains("differs"), "{e}");
}

/// The binding gates (shared by Python and the CLI): SCAN / r2SCAN / TPSS
/// pass the Gamma derivative gate, raw meta-GGA names do not, and an RS-GDF
/// J is refused for a meta-GGA derivative before any SCF.
#[test]
fn binding_gates() {
    use ferric_pbc::dft::{
        check_gamma_derivs_functional, check_gamma_mgga_deriv_jk, check_gamma_rks_request,
    };
    for f in ["SCAN", "r2SCAN", "TPSS", "PBE"] {
        check_gamma_derivs_functional(f).unwrap_or_else(|e| panic!("{f}: {e}"));
        check_gamma_rks_request(f, true, false).unwrap_or_else(|e| panic!("{f}: {e}"));
        check_gamma_mgga_deriv_jk(f, false).unwrap();
    }
    let e = check_gamma_derivs_functional("MGGA_X_TPSS")
        .unwrap_err()
        .to_string();
    assert!(e.contains("SCAN / r2SCAN / TPSS"), "{e}");
    let e = check_gamma_mgga_deriv_jk("TPSS", true)
        .unwrap_err()
        .to_string();
    assert!(e.contains("dense-AFT J only"), "{e}");
    // The request gate the bindings call carries the same J/K refusal.
    for f in ["SCAN", "r2SCAN", "TPSS"] {
        let e = check_gamma_rks_request(f, true, true).unwrap_err().to_string();
        assert!(e.contains("dense-AFT J only"), "{f}: {e}");
        check_gamma_rks_request(f, false, true).unwrap_or_else(|e| panic!("{f}: {e}"));
    }
    check_gamma_mgga_deriv_jk("PBE", true).unwrap();
}
